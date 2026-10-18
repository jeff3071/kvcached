# Sleep-to-first-output experiment

This experiment measures vLLM level-1 sleep/wake of a target model while other
models on the same GPU serve concurrent requests. Each model has its own
already running vLLM server. The benchmark controls target sleep/wake and
generates background traffic; it does not measure controller queueing.
Level 2 needs weight reload and is not covered.

## One command on Linux

With vLLM and this checkout installed, run from anywhere:

```bash
bash benchmarks/bench_wakeup/run.sh --probe
```

The entry point selects the repo's `.venv/bin/python` when present, otherwise
`python3`; set `PYTHON=/path/to/venv/bin/python` to choose your engine environment.
It starts three models sequentially on GPU 0, waits for readiness, runs native
then kvcached under background traffic, prints step p50/p95 timings, and stops
its own server process groups, including on errors or interruption. It does
not install packages or change the sampler/attention backend. Model downloads
follow your normal Hugging Face settings.

Override models, device and load as needed (the first model is the wake target):

```bash
bash benchmarks/bench_wakeup/run.sh --probe --gpu 0 --rounds 20 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --model Qwen/Qwen3-0.6B \
  --concurrency 4 --background-max-tokens 256 --idle-seconds 5 \
  --output-dir /tmp/wakeup-experiment
```

The output directory must be new. Each arm retains server logs, exact commands,
startup times, GPU identity/package metadata, per-request `results.jsonl`, and
`summary.txt`. A failed arm stops the run with a nonzero exit and `failure.txt`.
Use `--mode kvcached` or `--mode native` for one arm. Use `--memory-utilization`
to set each server's budget, `--background-prompt-file` to increase KV demand,
`--base-port` to select ports, and `--server-arg=--dtype=half` to pass engine
options. Default execution is eager; `--compiled` enables the engine's usual
compilation/graphs. `bash benchmarks/bench_wakeup/run.sh --help` lists options.

The report includes startup, awake baseline TTFT, sleep, background load setup
(including the configured idle interval), wake API, post-wake TTFT, complete
inference request, wake-to-first-output, and background drain times. Startup
times appear in the console and environment JSON; the per-round table is saved
in `summary.txt`.

`--probe` additionally reports worker wake, allocator map/copy, and kvcached
allocation/capacity/page-allocation/IPC times within each wake-to-first-output
window. Missing requested core probe data fails the run. Page/IPC rows only
appear if those paths execute and their hooks are supported. These are nested,
inclusive wall times; do not add them or call them a measured native/kvcached
delta. Probe hooks and log writes add overhead. Omit `--probe` for final latency
comparisons. Probe compatibility has to be checked against your vLLM version.

## Multiple models on one GPU

Start one target and two background servers in separate terminals on the Linux
GPU host. These small models and conservative budgets are a starting point;
size budgets, context limits and concurrency for your GPU. Use the same physical
GPU for every server, a different port and a unique kvcached IPC name for each.
For example, after installing this checkout and vLLM in your environment:

```bash
# Terminal 1: target (requires sleep mode)
CUDA_VISIBLE_DEVICES=0 ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 \
KVCACHED_IPC_NAME=wakeup-target VLLM_SERVER_DEV_MODE=1 \
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 18765 --served-model-name target --enable-sleep-mode \
  --gpu-memory-utilization 0.2 --max-model-len 4096 --max-num-seqs 8 --enforce-eager

# Terminal 2: background A
CUDA_VISIBLE_DEVICES=0 ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 \
KVCACHED_IPC_NAME=wakeup-background-a VLLM_SERVER_DEV_MODE=1 \
vllm serve HuggingFaceTB/SmolLM2-135M-Instruct \
  --host 127.0.0.1 --port 18766 --served-model-name background-a \
  --gpu-memory-utilization 0.2 --max-model-len 4096 --max-num-seqs 8 --enforce-eager

# Terminal 3: background B
CUDA_VISIBLE_DEVICES=0 ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 \
KVCACHED_IPC_NAME=wakeup-background-b VLLM_SERVER_DEV_MODE=1 \
vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 18767 --served-model-name background-b \
  --gpu-memory-utilization 0.2 --max-model-len 4096 --max-num-seqs 8 --enforce-eager
```

Wait for all three servers to be ready, then run on the GPU host:

```bash
python3 benchmarks/bench_wakeup/profile_wakeup.py \
  --base-url http://127.0.0.1:18765 --model target --label kvcached-three-models \
  --background http://127.0.0.1:18766 background-a \
  --background http://127.0.0.1:18767 background-b \
  --background-concurrency 4 --background-max-tokens 256 \
  --idle-seconds 5 --rounds 20 --gpu 0 \
  --output /tmp/kvcached-three-models.jsonl
```

Each round performs an awake target baseline without benchmark-generated
background traffic, puts the target to sleep, then starts a continuous load
on each background endpoint. It waits for nonempty streaming output from every
endpoint, lets the load run for `--idle-seconds`, and wakes the target while
that load continues through the target's first output and completion. Finally
it stops issuing background requests and drains in-flight requests. Only the
target receives sleep/wake or prefix-reset control calls.

`--background` is repeatable; omitting it retains the single-model control.
Concurrency is **per background model**. Every worker uses its own connection.
Use `--background-prompt-file long-prompt.txt` and increase concurrency to grow
background KV demand; keep input plus output within each server's context limit.
The default short prompt is a functional starting point, not a saturation test.

Background samples are saved as `background_request` records. Each target round
includes counts of successfully completed background requests whose streaming
output intervals overlap the target's wake-to-first-output interval, separately
for every endpoint. Both target and background records include monotonic
timestamps from the benchmark process for auditing that overlap. Background
failures, missing overlap on any endpoint, and failure to drain invalidate the
round and stop the run. `--background-ready-timeout` bounds waiting for active
streams; `--background-drain-timeout` bounds draining after the target finishes.
If draining times out, sockets are interrupted and the run fails; remote work
may still finish on the server.

`before_wake_memory` captures device memory after background load starts,
outside the timed wake interval. Check this together with server KV metrics/logs
to distinguish light co-location from actual memory pressure. HTTP endpoints
do not expose physical placement: the client cannot prove that remote servers
share a GPU. Retain launch commands and GPU UUID/PCI bus ID in `--metadata`.
`--gpu` samples the **client host's** device, so run there for meaningful memory
measurements. This script does not lower another model's quota automatically;
if weight reload cannot fit, the failed wake is retained as a failed result.

For a native comparison, restart **all** servers with `ENABLE_KVCACHED=false`
and `KVCACHED_AUTOPATCH=0`, keep the same workload/budgets, and use a new output
path. The unloaded awake baseline is not a measurement of kvcached overhead;
compare matching multi-model native/kvcached arms and inspect their actual
memory occupancy.

`profile_wakeup.py` uses Python's standard library. It records each round as
flushed JSONL, including failures, output, and a nearest-rank p50/p95 summary:

* `sleep_ms`: the sleep HTTP request, including draining/offload.
* `wake_api_ms`: the complete wake HTTP request.
* `ttft_ms`: subsequent completion request to first **nonempty text** SSE event.
* `wake_to_first_output_ms`: wake request start to that event, measured directly.
* `baseline_ttft_ms`: awake completion, with prefix cache reset before it.

The prefix cache is reset again after the baseline and before sleep. This
prevents retained kvcached prefixes from making the post-wake request a cache
hit while native vLLM must prefill it. This is a cold-prefix restoration test;
prefix retention and long-context workloads need separate experiments.

First text is an observable TTFT proxy; tokenizer buffering means it need not
be exactly one token. Empty SSE events do not count. Every request is drained
through `[DONE]`, so the next sleep cannot race an outstanding generation.
The awake baseline and post-wake greedy text must match. A failed round stops
the experiment and returns a nonzero exit code; a control timeout does not
cancel the remote operation. Inspect the server before resuming.

## Reproduce

From the repository root, start a server in one terminal. Keep all flags and
versions identical between arms. Use separate output paths and unique IPC names.
The example disables compilation/graphs to isolate restoration first; repeat
with the production compilation configuration before drawing deployment conclusions.

```bash
ENABLE_KVCACHED=false KVCACHED_AUTOPATCH=0 \
VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 \
.venv/bin/vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 18765 --served-model-name wakeup \
  --enable-sleep-mode --gpu-memory-utilization 0.35 \
  --max-model-len 1024 --max-num-seqs 4 --enforce-eager \
  --attention-backend TRITON_ATTN
```

After server readiness, run:

```bash
python3 benchmarks/bench_wakeup/profile_wakeup.py \
  --base-url http://127.0.0.1:18765 --model wakeup --label native \
  --rounds 20 --idle-seconds 1 --gpu 0 --output /tmp/native-wakeup.jsonl
```

Stop that server, then repeat with `ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1`
and `KVCACHED_IPC_NAME=wakeup-experiment`. Verify patch engagement in the server
log, and retain the log alongside the JSONL. `--label` only labels results; it
does not enable kvcached or verify server configuration.

The client requires Python >=3.10, no third-party Python packages. The example
server flags and probe were exercised with vLLM 0.24.0. The engine must expose
`/sleep`, `/wake_up`, `/is_sleeping`, `/reset_prefix_cache`, `/v1/models` and
streaming `/v1/completions`. `VLLM_SERVER_DEV_MODE=1` enables the control APIs.
The server must start awake. The script validates the served model and performs
two warm-up requests before measuring. Use `--warmup-rounds`, `--max-tokens`
and `--prompt` to control those inputs; keep them identical across arms.

Pass `--metadata /path/to/environment.json` to embed your server environment
in the result file, for example:

```json
{
  "source_commit": "git rev-parse HEAD",
  "model_revision": "exact Hugging Face snapshot revision",
  "vllm": "installed version",
  "torch": "installed version including CUDA suffix",
  "gpu_driver": "nvidia-smi output",
  "server_command": "full launch command including environment variables"
}
```

Replace the example values with the actual evidence. The script labels its own
platform as `client_platform`; it cannot infer remote server package versions.
The JSONL output path must not already exist. Preflight errors are recorded
before any sleep request, and the process exits nonzero on any failed cycle.
`--timeout` is the socket timeout for each HTTP operation, not a run deadline.

`--gpu 0` samples device-wide used/free MiB before sleep, while sleeping, before
wake under background load, and after inference. These subprocess calls are outside the timed wake/inference
interval but affect the total sleep duration. Desktop usage can affect these
numbers; they are not per-process allocation measurements.

## Breakdown probe

For a separate diagnostic run, prepend the server command with:

```bash
PYTHONPATH="$PWD/benchmarks/bench_wakeup/probe" \
WAKEUP_PROBE_DIR=/tmp/native-wakeup-probe
```

The opt-in `sitecustomize.py` installs import hooks in vLLM worker processes.
It records allocator wake wall time, synchronous create/map and weight-copy
totals/counts/bytes, and worker wake wall time, in one JSONL file per PID.
This probe targets the installed vLLM 0.24 implementation; check hook installation
and nonzero wake records when using another version. It does not change copy
ordering, pinning, or synchronization. Instrumented latency includes wrapper
overhead; collect uninstrumented runs separately for final comparisons.

Read diagnostic records without extra dependencies:

```bash
python3 - /tmp/native-wakeup-probe <<'PY'
import json, statistics, sys
from pathlib import Path
rows = [json.loads(line) for path in Path(sys.argv[1]).glob("wake-*.jsonl")
        for line in path.read_text().splitlines()]
wakes = [r for r in rows if r["kind"] == "allocator_wake" and r["status"] == "ok"]
if not wakes:
    raise SystemExit("No successful allocator wake records: check probe hooks")
for key in ("total_ms", "map_ms", "copy_ms", "map_calls", "copy_calls", "copy_bytes"):
    print(key, statistics.median(r[key] for r in wakes))
PY
```

Probe files append across restarts, so use a new directory for each arm/start.
Durations are nested: allocator total includes map/copy, and worker total
includes allocator total. Do not add these totals together.

Use the breakdown to choose the next controlled experiment: map costs versus
weight copy versus worker post-wake work versus HTTP/engine overhead. Avoid
attributing total wake latency to PCIe without measuring transferred bytes.
Retain the exact model revision, engine/torch/CUDA versions, GPU/driver, server
flags, environment and logs with each result. Alternate native/kvcached starts
and repeat them to expose startup/clock/cache effects.

Protocol tests:

```bash
python3 -m unittest discover -s benchmarks/bench_wakeup -v
```
