# SPDX-License-Identifier: Apache-2.0
"""Protocol tests; no GPU or engine dependencies."""
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from profile_wakeup import Client, summarize


class Response(io.BytesIO):
    def getheader(self, name, default=""):
        return "text/event-stream"


class ProtocolTests(unittest.TestCase):
    def generate(self, body):
        client = object.__new__(Client)
        client.request = lambda *args: Response(body)
        return client.generate({})

    def test_first_output_ignores_empty_and_usage_events(self):
        stream = (b'data: {"choices": []}\n\n'
                  b'data: {"choices": [{"text": ""}]}\n\n'
                  b'data: {"choices": [{"text": "hello"}]}\n\n'
                  b'data: [DONE]\n\n')
        with patch("profile_wakeup.time.perf_counter", side_effect=[1, 2, 4]):
            result, first = self.generate(stream)
        self.assertEqual(first, 2)
        self.assertEqual(result, {"ttft_ms": 1000, "request_ms": 3000, "output": "hello"})

    def test_truncated_stream_fails(self):
        with self.assertRaisesRegex(RuntimeError, "lacks"):
            self.generate(b'data: {"choices": [{"text": "hello"}]}\n\n')

    def test_empty_stream_fails(self):
        with self.assertRaisesRegex(RuntimeError, "no output"):
            self.generate(b'data: [DONE]\n\n')

    def test_sse_error_fails(self):
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            self.generate(b'data: {"error": "engine failed"}\n\n')

    def test_failed_rounds_are_counted_not_percentiles(self):
        result = summarize([{"status": "failed", "wake_api_ms": 10000},
                            {"status": "ok", **{key: 2 for key in (
                                "baseline_ttft_ms", "sleep_ms", "wake_api_ms", "ttft_ms",
                                "wake_to_first_output_ms")}}])
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["wake_api_ms"], {"p50": 2, "p95": 2})


class RoundTripTests(unittest.TestCase):
    def run_cli(self, *, fail_wake=False, wrong_model=False, backgrounds=0, fail_background=False):
        calls = []
        target_output = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            sleeping = False

            def log_message(self, *args):
                pass

            def reply(self, body, content_type="application/json", status=200):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                calls.append(self.path)
                data = ({"data": [{"id": "other" if wrong_model else self.server.model}]}
                        if self.path == "/v1/models" else {"is_sleeping": self.sleeping})
                self.reply(json.dumps(data).encode())

            def do_POST(self):
                calls.append(self.path)
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if self.path == "/v1/completions":
                    payload = json.loads(body)
                    if payload.get("model") != self.server.model or not payload.get("stream"):
                        self.reply(b"{}", status=400)
                        return
                    if self.server.model != "test":
                        if fail_background:
                            self.reply(b'{"error": "background failed"}', status=500)
                            return
                        start = b'data: {"choices": [{"text": "background"}]}\n\n'
                        end = b'data: [DONE]\n\n'
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Content-Length", str(len(start) + len(end)))
                        self.end_headers()
                        self.wfile.write(start)
                        self.wfile.flush()
                        target_output.wait(3)
                        time.sleep(0.02)
                        self.wfile.write(end)
                        return
                    if "/wake_up" in calls:
                        target_output.set()
                    self.reply(b'data: {"choices": [{"text": ""}]}\n\n'
                               b'data: {"choices": [{"text": "hello"}]}\n\n'
                               b'data: [DONE]\n\n', "text/event-stream")
                elif self.path.startswith("/sleep?"):
                    self.sleeping = True
                    self.reply(b"{}")
                elif self.path == "/wake_up":
                    if fail_wake:
                        self.reply(b'{"error": "wake failed"}', status=500)
                    else:
                        self.sleeping = False
                        self.reply(b"{}")
                else:
                    self.reply(b"{}")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.model = "test"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        background_servers = []
        for index in range(backgrounds):
            background_server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            background_server.model = f"background-{index}"
            background_thread = threading.Thread(target=background_server.serve_forever, daemon=True)
            background_thread.start()
            background_servers.append((background_server, background_thread))
        try:
            with tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "results.jsonl"
                metadata = Path(folder) / "environment.json"
                metadata.write_text('{"engine": "fake"}')
                command = [
                    sys.executable, str(Path(__file__).with_name("profile_wakeup.py")),
                    "--base-url", f"http://127.0.0.1:{server.server_port}",
                    "--model", "test", "--label", "test", "--rounds", "1" if backgrounds else "2",
                    "--warmup-rounds", "1", "--idle-seconds", "0",
                    "--metadata", str(metadata), "--output", str(output),
                ]
                for background_server, _ in background_servers:
                    command.extend(["--background", f"http://127.0.0.1:{background_server.server_port}",
                                    background_server.model])
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                rows = [json.loads(line) for line in output.read_text().splitlines()]
        finally:
            target_output.set()
            server.shutdown()
            server.server_close()
            thread.join()
            for background_server, background_thread in background_servers:
                background_server.shutdown()
                background_server.server_close()
                background_thread.join()
        return result, rows, calls

    def test_two_complete_cycles_and_metadata(self):
        result, rows, calls = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(rows[0]["server_metadata"], {"engine": "fake"})
        self.assertEqual(rows[-1]["successful"], 2)
        self.assertEqual(rows[-1]["failed"], 0)
        self.assertEqual(calls.count("/reset_prefix_cache"), 4)
        for row in rows:
            if row["kind"] == "round":
                self.assertEqual(row["output"], "hello")
                self.assertGreaterEqual(row["wake_to_first_output_ms"], row["ttft_ms"])

    def test_failed_wake_is_retained_and_stops_run(self):
        result, rows, calls = self.run_cli(fail_wake=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(rows[-1]["failed"], 1)
        failed = [row for row in rows if row["kind"] == "round"]
        self.assertEqual(len(failed), 1)
        self.assertIn("HTTP 500", failed[0]["error"])
        self.assertEqual(calls.count("/wake_up"), 1)

    def test_wrong_model_fails_before_sleep(self):
        result, rows, calls = self.run_cli(wrong_model=True)
        self.assertEqual(result.returncode, 1)
        self.assertTrue(rows[-1]["preflight_failed"])
        self.assertFalse(any(path.startswith("/sleep") for path in calls))

    def test_wake_overlaps_two_other_models(self):
        result, rows, calls = self.run_cli(backgrounds=2)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        round_row = next(row for row in rows if row["kind"] == "round")
        self.assertTrue(all(round_row["background"]["overlapping_requests"]))
        self.assertEqual(round_row["background"]["unfinished_workers"], 0)
        requests = [row for row in rows if row["kind"] == "background_request"]
        self.assertEqual({row["model"] for row in requests}, {"background-0", "background-1"})
        self.assertTrue(all(row["status"] == "ok" for row in requests))
        self.assertEqual(calls.count("/sleep?level=1&mode=wait"), 1)

    def test_background_failure_cannot_produce_green_wake(self):
        result, rows, _ = self.run_cli(backgrounds=1, fail_background=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(rows[-1]["failed"], 1)
        round_row = next(row for row in rows if row["kind"] == "round")
        self.assertGreater(round_row["background"]["failed_requests"], 0)


if __name__ == "__main__":
    unittest.main()
