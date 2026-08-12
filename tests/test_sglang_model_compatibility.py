# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Manual SGLang + kvcached layout compatibility smoke test for issue #425.

Run only when explicitly requested:

    RUN_KVCACHED_MODEL_MATRIX=1 pytest -s \
        tests/test_sglang_model_compatibility.py

By default this runs both contiguous and non-contiguous kvcached layouts. To
limit the run:

    KVCACHED_MODEL_MATRIX_LAYOUTS=non-contiguous RUN_KVCACHED_MODEL_MATRIX=1 \
        pytest -s tests/test_sglang_model_compatibility.py

Each case starts SGLang with kvcached, sends one short request, and reports
startup crashes separately from readable-output failures.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

MODELS = [
    ("Standard GQA", "Qwen/Qwen3-8B"),
    ("SWA + dense hybrid", "openai/gpt-oss-20b"),
    ("Hybrid linear", "Qwen/Qwen3.5-9B"),
    ("Cross-layer KV sharing", "google/gemma-4-E2B-it"),
    ("Heterogeneous KV groups", "google/gemma-4-12B-it"),
]

PROMPT = "What city is the capital of France? Answer with the city name only."
LAYOUT_ALIASES = {
    "contiguous": ("contiguous", True),
    "contig": ("contiguous", True),
    "true": ("contiguous", True),
    "1": ("contiguous", True),
    "yes": ("contiguous", True),
    "non-contiguous": ("non-contiguous", False),
    "non_contiguous": ("non-contiguous", False),
    "noncontiguous": ("non-contiguous", False),
    "no-contiguous": ("non-contiguous", False),
    "no_contiguous": ("non-contiguous", False),
    "nocontiguous": ("non-contiguous", False),
    "non-contig": ("non-contiguous", False),
    "false": ("non-contiguous", False),
    "0": ("non-contiguous", False),
    "no": ("non-contiguous", False),
}


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_KVCACHED_MODEL_MATRIX") != "1",
    reason="manual GPU matrix; set RUN_KVCACHED_MODEL_MATRIX=1",
)


def _selected_layouts() -> list[tuple[str, bool]]:
    raw = os.getenv("KVCACHED_MODEL_MATRIX_LAYOUTS", "contiguous,non-contiguous")
    layouts: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        try:
            name, enabled = LAYOUT_ALIASES[key]
        except KeyError:
            valid = ", ".join(("contiguous", "non-contiguous"))
            raise ValueError(
                f"Unknown KVCACHED_MODEL_MATRIX_LAYOUTS entry {item!r}; "
                f"expected one or both of: {valid}"
            ) from None
        if name in seen:
            continue
        seen.add(name)
        layouts.append((name, enabled))
    if not layouts:
        raise ValueError("KVCACHED_MODEL_MATRIX_LAYOUTS did not select any layouts")
    return layouts


def _free_port() -> int:
    for port in range(30000, 65000):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free localhost port in 30000..64999")


def _json(
    method: str,
    url: str,
    payload: dict | None = None,
    timeout: int = 30,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode() or "{}")


def _ready(port: int) -> bool:
    for path in ("/health", "/v1/models"):
        try:
            _json("GET", f"http://127.0.0.1:{port}{path}", timeout=2)
            return True
        except Exception:
            pass
    return False


def _wait_ready(proc: subprocess.Popen, port: int, timeout: int, log_path: Path) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            log_tail = log_path.read_text(errors="replace")[-4000:]
            pytest.fail(
                f"crash-at-startup: exit={proc.returncode}\n"
                f"log={log_path}\n{log_tail}"
            )
        if _ready(port):
            return
        time.sleep(2)
    log_tail = log_path.read_text(errors="replace")[-4000:]
    pytest.fail(
        f"crash-at-startup: timeout waiting for ready\nlog={log_path}\n{log_tail}"
    )


def _generate(port: int, model: str) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_completion_tokens": 64,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        "stream": False,
    }
    res = _json("POST", f"http://127.0.0.1:{port}/v1/chat/completions", body)
    message = res["choices"][0]["message"]
    return message.get("content") or message.get("reasoning_content") or ""


def _has_long_repeated_run(out: str, limit: int = 8) -> bool:
    if not out:
        return False
    run = 1
    prev = out[0]
    for ch in out[1:]:
        if ch == prev:
            run += 1
            if run >= limit:
                return True
        else:
            prev = ch
            run = 1
    return False


def _assert_readable(text: str) -> None:
    out = text.strip()
    out_lower = out.lower()
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in out)
    if (
        not out
        or "paris" not in out_lower
        or "\ufffd" in out
        or "\x00" in out
        or printable / max(len(out), 1) < 0.95
        or _has_long_repeated_run(out)
    ):
        pytest.fail(f"garbled-output: {text!r}")


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=20)


@pytest.mark.parametrize(
    "layout_name,contiguous_layout",
    [pytest.param(name, enabled, id=name) for name, enabled in _selected_layouts()],
)
@pytest.mark.parametrize("arch,model", MODELS)
def test_sglang_model_layout_compatibility(
    layout_name: str,
    contiguous_layout: bool,
    arch: str,
    model: str,
    tmp_path: Path,
):
    port = _free_port()
    log_path = tmp_path / f"{layout_name}_{model.replace('/', '__')}.log"
    env = os.environ | {
        "ENABLE_KVCACHED": "true",
        "KVCACHED_AUTOPATCH": "1",
        "KVCACHED_CONTIGUOUS_LAYOUT": str(contiguous_layout).lower(),
        "KVCACHED_PAGE_SIZE_MB": "4",
    }
    env.pop("SGLANG_GRPC_PORT", None)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model",
        model,
        "--port",
        str(port),
        "--disable-radix-cache",
        "--trust-remote-code",
    ]

    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
    try:
        timeout = int(os.getenv("KVCACHED_MODEL_MATRIX_TIMEOUT", "900"))
        _wait_ready(proc, port, timeout, log_path)
        text = _generate(port, model)
        _assert_readable(text)
        print(f"pass: {layout_name} / {arch} / {model}: {text!r}")
    finally:
        _stop(proc)
