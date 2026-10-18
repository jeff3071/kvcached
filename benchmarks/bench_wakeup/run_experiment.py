# SPDX-License-Identifier: Apache-2.0
"""Start co-located vLLM servers, benchmark wake-up, report timings, clean up."""
import argparse
import importlib.metadata
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BENCH = Path(__file__).resolve().parent


def stop_servers(processes):
    # Each server is a session leader; stop its workers as well as its API process.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        live = False
        for process in processes:
            process.poll()  # Reap exited leaders before checking their groups.
            try:
                os.killpg(process.pid, 0)
                live = True
            except ProcessLookupError:
                pass
        if not live:
            break
        time.sleep(0.1)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def wait_ready(process, port, timeout, log):
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"server exited ({process.returncode}); see {log}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return (time.perf_counter() - start) * 1000
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server readiness timeout; see {log}")


def report(folder, probe=False):
    records = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    summary = next((record for record in reversed(records) if record["kind"] == "summary"), None)
    if summary is None:
        raise RuntimeError("benchmark produced no summary")
    lines = [f"{folder.name}: {summary['successful']} successful, {summary['failed']} failed rounds",
             f"{'Step (ms)':32} {'p50':>12} {'p95':>12}"]
    for key, value in summary.items():
        if key.endswith("_ms"):
            lines.append(f"{key:32} {value['p50']:12.2f} {value['p95']:12.2f}")
    if probe:
        events = [json.loads(line) for path in (folder / "probe").glob("wake-*.jsonl")
                  for line in path.read_text().splitlines()]
        rounds = [record for record in records if record["kind"] == "round" and record["status"] == "ok"]
        metrics = {}
        for row in rounds:
            window = [event for event in events if "start_monotonic_s" in event
                      and row["wake_start_monotonic_s"] <= event["start_monotonic_s"]
                      and event["end_monotonic_s"] <= row["first_output_monotonic_s"]]
            if not any(event["kind"] == "allocator_wake" for event in window):
                raise RuntimeError("requested probe did not capture allocator wake; inspect target.log/probe")
            for kind, field in (("worker_wake", "total_ms"), ("allocator_wake", "total_ms"),
                                ("allocator_wake", "map_ms"), ("allocator_wake", "copy_ms"),
                                ("kvcached_alloc", "total_ms"), ("kvcached_available_size", "total_ms"),
                                ("kvcached_alloc_page", "total_ms"), ("kvcached_map_ipc", "total_ms")):
                selected = [event for event in window if event["kind"] == kind]
                if selected:
                    metrics.setdefault(f"{kind}.{field}", []).append(sum(event[field] for event in selected))
        if rounds and folder.name == "kvcached" and "kvcached_alloc.total_ms" not in metrics:
            raise RuntimeError("requested kvcached probe captured no KV allocations before first output")
        lines.append("Probe: inclusive totals per wake-to-first-output window; nested times must not be added.")
        for name, values in metrics.items():
            values.sort()
            p50 = values[math.ceil(len(values) * 0.5) - 1]
            p95 = values[math.ceil(len(values) * 0.95) - 1]
            lines.append(f"{name:40} {p50:10.2f} {p95:10.2f} (n={len(values)})")
    for record in records:
        if record["kind"] == "round" and record["status"] != "ok":
            lines.append(f"FAILED round {record['round']} during {record.get('phase')}: {record.get('error')}")
        elif record["kind"] == "preflight" and record["status"] == "failed":
            lines.append(f"FAILED preflight: {record['error']}")
    text = "\n".join(lines) + "\n"
    (folder / "summary.txt").write_text(text)
    print(text, flush=True)
    return summary


def run_arm(args, mode, root):
    folder = root / mode
    folder.mkdir()
    processes, logs, launches = [], [], []
    started = time.perf_counter()
    try:
        for index, model in enumerate(args.models):
            port = args.base_port + index
            name = "target" if index == 0 else f"background-{index}"
            command = ([args.vllm_bin] if args.vllm_bin else
                       [sys.executable, "-m", "vllm.entrypoints.cli.main"])
            command += ["serve", model, "--host", "127.0.0.1", "--port", str(port),
                        "--served-model-name", name, "--enable-sleep-mode",
                        "--gpu-memory-utilization", str(args.memory_utilization),
                        "--max-model-len", str(args.max_model_len),
                        "--max-num-seqs", str(args.max_num_seqs)]
            if not args.compiled:
                command.append("--enforce-eager")
            command += args.server_arg
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=args.gpu, ENABLE_KVCACHED=str(mode == "kvcached").lower(),
                       KVCACHED_AUTOPATCH="1" if mode == "kvcached" else "0",
                       KVCACHED_IPC_NAME=f"wake-{root.name}-{mode}-{index}", VLLM_SERVER_DEV_MODE="1")
            # Probe only the target. All server invocations retain their environment in metadata.
            env.pop("WAKEUP_PROBE_DIR", None)
            if args.probe and index == 0:
                env["PYTHONPATH"] = str(BENCH / "probe") + os.pathsep + env.get("PYTHONPATH", "")
                env["WAKEUP_PROBE_DIR"] = str(folder / "probe")
            selected_env = {key: value for key, value in env.items()
                            if key.startswith(("KVCACHED", "ENABLE_KVCACHED", "VLLM", "CUDA_VISIBLE", "WAKEUP", "HF_HUB_OFFLINE"))}
            log_path = folder / f"{name}.log"
            log = log_path.open("w")
            logs.append(log)
            print(f"[{mode}] starting {name}: {model} on GPU {args.gpu}, port {port}", flush=True)
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            processes.append(process)
            launch = {"model": model, "name": name, "command": command, "environment": selected_env}
            launches.append(launch)
            # Persist commands even if startup fails.
            (folder / "launches.json").write_text(json.dumps(launches, indent=2) + "\n")
            launch["startup_ms"] = wait_ready(process, port, args.ready_timeout, log_path)
            print(f"[{mode}] {name} ready: {launch['startup_ms']:.0f} ms", flush=True)
        if mode == "kvcached":
            for launch in launches:
                log_text = (folder / f"{launch['name']}.log").read_text(errors="replace")
                if "Successfully patched vllm" not in log_text:
                    raise RuntimeError(f"missing kvcached patch engagement marker: {launch['name']}.log")
        metadata = {"launches": launches, "gpu": args.gpu,
                    "startup_total_ms": (time.perf_counter() - started) * 1000,
                    "instrumented": args.probe, "packages": {}}
        for package in ("vllm", "torch", "kvcached"):
            try:
                metadata["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                metadata["packages"][package] = None
        metadata["nvidia_smi"] = subprocess.check_output([
            "nvidia-smi", "-i", args.gpu,
            "--query-gpu=uuid,pci.bus_id,name,driver_version,memory.total", "--format=csv,noheader",
        ], text=True, timeout=15).strip()
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BENCH,
                                  capture_output=True, text=True)
        metadata["source_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
        (folder / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        command = [sys.executable, str(BENCH / "profile_wakeup.py"),
                   "--base-url", f"http://127.0.0.1:{args.base_port}", "--model", "target",
                   "--label", mode, "--rounds", str(args.rounds), "--gpu", args.gpu,
                   "--idle-seconds", str(args.idle_seconds), "--timeout", str(args.timeout),
                   "--background-concurrency", str(args.concurrency),
                   "--background-max-tokens", str(args.background_max_tokens),
                   "--output", str(folder / "results.jsonl"),
                   "--metadata", str(folder / "environment.json")]
        if args.background_prompt_file:
            command += ["--background-prompt-file", str(args.background_prompt_file)]
        for index in range(1, len(args.models)):
            command += ["--background", f"http://127.0.0.1:{args.base_port + index}", f"background-{index}"]
        (folder / "benchmark-command.txt").write_text(shlex.join(command) + "\n")
        result = subprocess.run(command)
        report(folder, args.probe)
        if result.returncode:
            raise RuntimeError(f"{mode} benchmark failed; see {folder}")
    finally:
        stop_servers(processes)
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("both", "native", "kvcached"), default="both")
    parser.add_argument("--model", dest="models", action="append", help="repeat; first model is target")
    parser.add_argument("--gpu", default="0", help="physical GPU index")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-port", type=int, default=18765)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4, help="per background model")
    parser.add_argument("--background-max-tokens", type=int, default=256)
    parser.add_argument("--background-prompt-file", type=Path)
    parser.add_argument("--idle-seconds", type=float, default=5)
    parser.add_argument("--memory-utilization", type=float, default=0.2, help="per server")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--compiled", action="store_true", help="use engine default compilation/graphs")
    parser.add_argument("--probe", action="store_true", help="include worker/allocator and kvcached KV timings")
    parser.add_argument("--server-arg", action="append", default=[], help="extra vLLM argument, e.g. --server-arg=--dtype=half")
    parser.add_argument("--vllm-bin", help="vLLM executable; defaults to the current Python's vLLM module")
    args = parser.parse_args()
    args.models = args.models or ["Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct",
                                  "Qwen/Qwen3-0.6B"]
    if not args.gpu.isdecimal() or len(args.models) < 2:
        parser.error("select one numeric GPU index and at least two models")
    if (min(args.rounds, args.concurrency, args.background_max_tokens, args.max_model_len, args.max_num_seqs) < 1
            or not 0 < args.memory_utilization < 1
            or not all(math.isfinite(value) for value in (args.idle_seconds, args.timeout, args.ready_timeout))
            or args.idle_seconds < 0 or min(args.timeout, args.ready_timeout) <= 0
            or args.base_port < 1 or args.base_port + len(args.models) > 65536):
        parser.error("invalid counts, memory utilization, timeout, or port range")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    root = args.output_dir or Path("/tmp") / f"wakeup-{time.time_ns()}"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for port in range(args.base_port, args.base_port + len(args.models)):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))  # Fail before launch if an endpoint is occupied.
    print(f"Results: {root}", flush=True)
    try:
        for mode in (("native", "kvcached") if args.mode == "both" else (args.mode,)):
            run_arm(args, mode, root)
    except (Exception, KeyboardInterrupt) as exc:
        (root / "failure.txt").write_text(str(exc) + "\n")
        print(f"FAILED: {exc}\nEvidence: {root}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
