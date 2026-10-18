# SPDX-License-Identifier: Apache-2.0
"""Measure target vLLM wake latency with optional co-located model load."""
import argparse
import http.client
import json
import math
import platform
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit


class Client:
    def __init__(self, url, timeout):
        parsed = urlsplit(url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment
                or parsed.username or parsed.password):
            raise ValueError("base URL must be an http(s) origin")
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        self.connection = cls(parsed.hostname, parsed.port, timeout=timeout)

    def request(self, method, path, payload=None):
        body = json.dumps(payload) if payload is not None else None
        self.connection.request(method, path, body, {"Content-Type": "application/json"})
        response = self.connection.getresponse()
        if response.status != 200:
            detail = response.read().decode(errors="replace")
            raise RuntimeError(f"{path}: HTTP {response.status}: {detail[:1000]}")
        return response

    def control(self, path):
        start = time.perf_counter()
        self.request("POST", path).read()
        return (time.perf_counter() - start) * 1000

    def assert_sleeping(self, expected):
        state = json.loads(self.request("GET", "/is_sleeping").read())
        if state.get("is_sleeping") is not expected:
            raise RuntimeError(f"unexpected sleep state: {state}")

    def generate(self, payload, on_first=None):
        start = time.perf_counter()
        response = self.request("POST", "/v1/completions", payload)
        if "text/event-stream" not in response.getheader("Content-Type", ""):
            response.read()
            raise RuntimeError("expected a streaming SSE response")
        first = None
        text = ""
        done = False
        while line := response.readline():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            for choice in event.get("choices", []):
                chunk = choice.get("text", "")
                if chunk:
                    if first is None:
                        first = time.perf_counter()
                        if on_first is not None:
                            on_first(first)
                    text += chunk
        response.read()
        if first is None or not done:
            raise RuntimeError("stream has no output or lacks [DONE]")
        return {"ttft_ms": (first - start) * 1000,
                "request_ms": (time.perf_counter() - start) * 1000,
                "output": text}, first


def summarize(rows):
    successful = [row for row in rows if row["status"] == "ok"]
    result = {"successful": len(successful), "failed": len(rows) - len(successful)}
    for key in ("baseline_ttft_ms", "sleep_ms", "background_load_ms", "wake_api_ms",
                "ttft_ms", "request_ms", "wake_to_first_output_ms", "background_drain_ms"):
        values = sorted(row[key] for row in successful if key in row)
        if values:
            result[key] = {f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)]
                           for p in (50, 95)}
    return result


def memory_snapshot(gpu):
    if gpu is None:
        return None
    result = subprocess.run([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True, timeout=15)
    used, free = (int(value.strip()) for value in result.stdout.strip().split(","))
    return {"used_mib": used, "free_mib": free}


class BackgroundLoad:
    """One independent HTTP connection per load worker; no shared target connection."""

    def __init__(self, args, round_index, emit):
        self.args, self.round_index, self.emit = args, round_index, emit
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.active = [0] * len(args.background)
        self.errors = []
        self.spans = []
        self.clients = []
        self.threads = []

    def start(self):
        for endpoint, (url, model) in enumerate(self.args.background):
            for worker in range(self.args.background_concurrency):
                client = Client(url, self.args.timeout)
                self.clients.append(client)
                thread = threading.Thread(target=self.run, args=(endpoint, model, worker, client), daemon=True)
                self.threads.append(thread)
                thread.start()

    def run(self, endpoint, model, worker, client):
        sequence = 0
        try:
            while not self.stop.is_set():
                record = {"kind": "background_request", "round": self.round_index,
                          "endpoint": endpoint, "model": model, "worker": worker,
                          "sequence": sequence, "status": "failed"}
                first = None

                def started(timestamp):
                    nonlocal first
                    first = timestamp
                    with self.condition:
                        self.active[endpoint] += 1
                        self.condition.notify_all()

                try:
                    record["request_start_monotonic_s"] = time.perf_counter()
                    result, _ = client.generate({
                        "model": model, "stream": True, "temperature": 0,
                        "ignore_eos": True, "max_tokens": self.args.background_max_tokens,
                        # Vary the start to avoid repeatedly reusing one cached prefix.
                        "prompt": f"Round {self.round_index}, worker {worker}, request {sequence}.\n"
                                  + self.args.background_prompt,
                    }, on_first=started)
                    record.update({key: value for key, value in result.items() if key != "output"})
                    record["status"] = "ok"
                except Exception as exc:
                    record["error"] = str(exc)
                    with self.condition:
                        self.errors.append(record.copy())
                        self.condition.notify_all()
                    return
                finally:
                    end = time.perf_counter()
                    record["first_output_monotonic_s"] = first
                    record["completed_monotonic_s"] = end
                    with self.condition:
                        if first is not None:
                            self.active[endpoint] -= 1
                            self.spans.append((endpoint, first, end, record["status"]))
                        self.condition.notify_all()
                    self.emit(record)
                sequence += 1
        finally:
            client.connection.close()

    def wait_active(self):
        deadline = time.perf_counter() + self.args.background_ready_timeout
        with self.condition:
            while self.active and not all(self.active):
                if self.errors:
                    raise RuntimeError(f"background load failed: {self.errors[0]}")
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise RuntimeError("not every background endpoint produced active streaming output")
                self.condition.wait(remaining)
            if self.errors:
                raise RuntimeError(f"background load failed: {self.errors[0]}")

    def finish(self, wake_start=None, first_output=None):
        self.stop.set()
        deadline = time.perf_counter() + self.args.background_drain_timeout
        for thread in self.threads:
            thread.join(max(0, deadline - time.perf_counter()))
        unfinished = sum(thread.is_alive() for thread in self.threads)
        overlaps = [0] * len(self.args.background)
        if wake_start is not None and first_output is not None:
            for endpoint, start, end, status in self.spans:
                if status == "ok" and start < first_output and end > wake_start:
                    overlaps[endpoint] += 1
        result = {"overlapping_requests": overlaps, "failed_requests": len(self.errors),
                  "unfinished_workers": unfinished}
        if unfinished:
            # Interrupt local sockets; the remote server may still finish those requests.
            import socket
            for client in self.clients:
                sock = client.connection.sock
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
            for thread in self.threads:
                thread.join(1)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True, help="e.g. native or kvcached")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--idle-seconds", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--gpu", type=int, help="sample device-wide MiB outside timed wake/inference")
    parser.add_argument("--prompt", default="Explain why the sky is blue in one sentence.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="JSON file with server versions, revision and launch command")
    parser.add_argument("--background", nargs=2, action="append", default=[], metavar=("URL", "MODEL"),
                        help="co-located background endpoint and served model; repeat for more models")
    parser.add_argument("--background-concurrency", type=int, default=2, help="workers per background endpoint")
    parser.add_argument("--background-max-tokens", type=int, default=256)
    parser.add_argument("--background-prompt-file", type=Path, help="UTF-8 prompt used to control background KV demand")
    parser.add_argument("--background-ready-timeout", type=float, default=120)
    parser.add_argument("--background-drain-timeout", type=float, default=30)
    args = parser.parse_args()
    if (args.rounds < 1 or args.warmup_rounds < 1 or args.max_tokens < 1
            or not math.isfinite(args.idle_seconds) or args.idle_seconds < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or args.background_concurrency < 1 or args.background_max_tokens < 1
            or not math.isfinite(args.background_ready_timeout) or args.background_ready_timeout <= 0
            or not math.isfinite(args.background_drain_timeout) or args.background_drain_timeout <= 0
            or (args.gpu is not None and args.gpu < 0)):
        parser.error("rounds, warmup-rounds, max-tokens and timeout must be positive; idle-seconds and gpu nonnegative")
    try:
        server_metadata = json.loads(args.metadata.read_text()) if args.metadata else None
        args.background_prompt = (args.background_prompt_file.read_text(encoding="utf-8")
                                  if args.background_prompt_file else
                                  "Describe the design and tradeoffs of a database cache in detail.")
        client = Client(args.base_url, args.timeout)
        urls = [args.base_url.rstrip("/")] + [url.rstrip("/") for url, _ in args.background]
        if len(set(urls)) != len(urls):
            raise ValueError("target and background endpoints must be distinct")
        for url, _ in args.background:
            Client(url, args.timeout).connection.close()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    payload = {"model": args.model, "prompt": args.prompt, "temperature": 0,
               "max_tokens": args.max_tokens, "stream": True, "ignore_eos": True}
    rows = []
    # Exclusive creation prevents accidentally overwriting experiment evidence.
    with args.output.open("x") as output:
        output_lock = threading.Lock()

        def emit(record):
            with output_lock:
                output.write(json.dumps(record) + "\n")
                output.flush()

        emit({"kind": "metadata", "args": {key: str(value) if isinstance(value, Path) else value
                                           for key, value in vars(args).items()},
              "unix_time": time.time(), "client_platform": platform.platform(),
              "server_metadata": server_metadata})
        try:
            models = json.loads(client.request("GET", "/v1/models").read())
            emit({"kind": "server_models", "models": models})
            if args.model not in {model["id"] for model in models.get("data", [])}:
                raise RuntimeError(f"model {args.model!r} is not served by this endpoint")
            client.assert_sleeping(False)
            for url, model in args.background:
                background_client = Client(url, args.timeout)
                try:
                    listing = json.loads(background_client.request("GET", "/v1/models").read())
                    if model not in {item["id"] for item in listing.get("data", [])}:
                        raise RuntimeError(f"background model {model!r} is not served at {url}")
                    background_client.assert_sleeping(False)
                    emit({"kind": "background_models", "url": url, "models": listing})
                finally:
                    background_client.connection.close()
            # Validate optional monitoring before mutating server state.
            memory_snapshot(args.gpu)
            for _ in range(args.warmup_rounds):
                client.generate(payload)
        except Exception as exc:
            emit({"kind": "preflight", "status": "failed", "error": str(exc)})
            emit({"kind": "summary", "successful": 0, "failed": 0, "preflight_failed": True})
            client.connection.close()
            print(f"Preflight failed: {exc}")
            return 1
        for index in range(args.rounds):
            row = {"kind": "round", "round": index, "status": "failed"}
            sleep_attempted = False
            background = BackgroundLoad(args, index, emit)
            wake_start = first = None
            try:
                row["phase"] = "baseline"
                # Sleep discards native KV. Compare with an equally cold prefix baseline.
                client.control("/reset_prefix_cache")
                baseline, _ = client.generate(payload)
                row["baseline_ttft_ms"] = baseline["ttft_ms"]
                # kvcached may keep KV outside the engine's sleep allocator.
                # Remove baseline prefixes before sleep in both arms.
                client.control("/reset_prefix_cache")
                row["awake_memory"] = memory_snapshot(args.gpu)
                sleep_attempted = True
                row["phase"] = "sleep"
                row["sleep_ms"] = client.control("/sleep?level=1&mode=wait")
                client.assert_sleeping(True)
                row["sleep_memory"] = memory_snapshot(args.gpu)
                row["phase"] = "background_load"
                load_start = time.perf_counter()
                background.start()
                background.wait_active()
                time.sleep(args.idle_seconds)
                background.wait_active()
                row["background_load_ms"] = (time.perf_counter() - load_start) * 1000
                row["before_wake_memory"] = memory_snapshot(args.gpu)
                wake_start = time.perf_counter()
                row["wake_start_monotonic_s"] = wake_start
                row["phase"] = "wake"
                row["wake_api_ms"] = client.control("/wake_up")
                # No health/status polling between wake and inference: it would add delay.
                row["phase"] = "first_inference"
                generation, first = client.generate(payload)
                row.update(generation)
                row["first_output_monotonic_s"] = first
                row["wake_to_first_output_ms"] = (first - wake_start) * 1000
                client.assert_sleeping(False)
                row["restored_memory"] = memory_snapshot(args.gpu)
                if generation["output"] != baseline["output"]:
                    raise RuntimeError("greedy output changed after sleep/wake")
                row["status"] = "ok"
                row["phase"] = "complete"
            except Exception as exc:
                row["error"] = str(exc)
                # A timed-out sleep may still be executing remotely. Stop the run.
                if sleep_attempted:
                    row["recovery"] = "Inspect server state; sleep/wake may still be executing."
            finally:
                drain_start = time.perf_counter()
                row["background"] = background.finish(wake_start, first)
                row["background_drain_ms"] = (time.perf_counter() - drain_start) * 1000
            if (row["background"]["failed_requests"] or row["background"]["unfinished_workers"]
                    or (row["status"] == "ok" and args.background
                        and not all(row["background"]["overlapping_requests"]))):
                row["status"] = "failed"
                row.setdefault("error", "background load failed, did not drain, or did not overlap wake")
            rows.append(row)
            emit(row)
            if row["status"] != "ok":
                break
        summary = summarize(rows)
        emit({"kind": "summary", **summary})
        print(json.dumps(summary, indent=2))
    client.connection.close()
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
