# SPDX-License-Identifier: Apache-2.0
"""Exercise the one-command lifecycle with fake engine executables; no GPU."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from run_experiment import report

BENCH = Path(__file__).resolve().parent

FAKE_SERVER = '''
import argparse, json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
p = argparse.ArgumentParser()
p.add_argument('--port', type=int)
p.add_argument('--served-model-name')
a, _ = p.parse_known_args()
print('PID=' + str(os.getpid()), flush=True)
if sys.argv[2] == 'fail':
    raise SystemExit(2)
class Handler(BaseHTTPRequestHandler):
    sleeping = False
    def log_message(self, *args): pass
    def reply(self, value, content_type='application/json'):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(value)))
        self.end_headers()
        self.wfile.write(value)
    def do_GET(self):
        if self.path == '/v1/models': value = {'data': [{'id': a.served_model_name}]}
        elif self.path == '/is_sleeping': value = {'is_sleeping': Handler.sleeping}
        else: value = {}
        self.reply(json.dumps(value).encode())
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        if self.path == '/v1/completions':
            start = b'data: {"choices": [{"text": "hello"}]}\\n\\n'
            end = b'data: [DONE]\\n\\n'
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(start) + len(end)))
            self.end_headers()
            self.wfile.write(start)
            self.wfile.flush()
            time.sleep(0.2 if a.served_model_name != 'target' else 0.01)
            self.wfile.write(end)
        else:
            if self.path.startswith('/sleep?'): Handler.sleeping = True
            elif self.path == '/wake_up': Handler.sleeping = False
            self.reply(b'{}')
ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
'''


class LauncherTests(unittest.TestCase):
    def run_script(self, fail=False):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            server = folder / "fake-vllm"
            server.write_text(f"#!{sys.executable}\n" + FAKE_SERVER)
            server.chmod(0o755)
            smi = folder / "nvidia-smi"
            smi.write_text("#!/bin/sh\ncase \"$*\" in\n*memory.used*) echo '100, 200';;\n*) echo 'GPU-fake, 0000:01:00.0, fake, 1, 300';;\nesac\n")
            smi.chmod(0o755)
            # Find two adjacent free localhost ports for this isolated test.
            for _ in range(100):
                with socket.socket() as first, socket.socket() as second:
                    first.bind(("127.0.0.1", 0))
                    port = first.getsockname()[1]
                    try:
                        second.bind(("127.0.0.1", port + 1))
                    except OSError:
                        continue
                    break
            output = folder / "results"
            env = {**os.environ, "PYTHON": sys.executable,
                   "PATH": str(folder) + os.pathsep + os.environ["PATH"]}
            result = subprocess.run([
                "bash", str(BENCH / "run.sh"), "--mode", "native",
                "--model", "fail" if fail else "target-model", "--model", "background-model",
                "--vllm-bin", str(server), "--output-dir", str(output),
                "--base-port", str(port), "--ready-timeout", "5", "--timeout", "3",
                "--rounds", "1", "--idle-seconds", "0", "--concurrency", "1",
            ], env=env, capture_output=True, text=True, timeout=30)
            logs = list(output.glob("native/*.log"))
            for log in logs:
                for line in log.read_text().splitlines():
                    if line.startswith("PID="):
                        with self.assertRaises(ProcessLookupError):
                            os.kill(int(line[4:]), 0)
            if fail:
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue((output / "failure.txt").exists())
            else:
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                summary = (output / "native/summary.txt").read_text()
                self.assertIn("1 successful, 0 failed", summary)
                for field in ("sleep_ms", "background_load_ms", "wake_api_ms", "ttft_ms",
                              "wake_to_first_output_ms", "background_drain_ms"):
                    self.assertIn(field, summary)
                metadata = json.loads((output / "native/environment.json").read_text())
                self.assertEqual(len(metadata["launches"]), 2)
                self.assertTrue(all(launch["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
                                    for launch in metadata["launches"]))

    def test_one_command_runs_and_cleans_up(self):
        self.run_script()

    def test_startup_failure_preserves_evidence_and_cleans_up(self):
        self.run_script(fail=True)


class ProbeReportTests(unittest.TestCase):
    def test_probe_filters_to_wake_window_and_keeps_nested_rows_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "kvcached"
            (folder / "probe").mkdir(parents=True)
            records = [{"kind": "round", "round": 0, "status": "ok",
                        "wake_start_monotonic_s": 10, "first_output_monotonic_s": 12},
                       {"kind": "summary", "successful": 1, "failed": 0,
                        "wake_api_ms": {"p50": 8, "p95": 8}}]
            (folder / "results.jsonl").write_text("\n".join(map(json.dumps, records)))
            events = [
                {"kind": "allocator_wake", "start_monotonic_s": 10, "end_monotonic_s": 10.5,
                 "total_ms": 7, "map_ms": 2, "copy_ms": 5},
                {"kind": "kvcached_alloc", "start_monotonic_s": 11, "end_monotonic_s": 11.5,
                 "total_ms": 4},
                {"kind": "kvcached_alloc", "start_monotonic_s": 1, "end_monotonic_s": 2,
                 "total_ms": 1000},
            ]
            path = folder / "probe/wake-1.jsonl"
            path.write_text("\n".join(map(json.dumps, events)))
            report(folder, probe=True)
            summary = (folder / "summary.txt").read_text()
            self.assertIn("allocator_wake.map_ms", summary)
            self.assertIn("allocator_wake.copy_ms", summary)
            self.assertIn("kvcached_alloc.total_ms", summary)
            self.assertNotIn("1000.00", summary)
            path.write_text("")
            with self.assertRaisesRegex(RuntimeError, "did not capture allocator wake"):
                report(folder, probe=True)


if __name__ == "__main__":
    unittest.main()
