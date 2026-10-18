# SPDX-License-Identifier: Apache-2.0
"""Opt-in vLLM 0.24 wake breakdown. Put this directory on server PYTHONPATH."""
import functools
import json
import os
import threading
import time
from pathlib import Path

if os.environ.get("WAKEUP_PROBE_DIR"):
    from wrapt.importer import when_imported

    state = threading.local()

    def emit(record):
        folder = Path(os.environ["WAKEUP_PROBE_DIR"])
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"wake-{os.getpid()}.jsonl").open("a") as output:
            output.write(json.dumps({"pid": os.getpid(), "unix_time": time.time(), **record}) + "\n")

    @when_imported("vllm.device_allocator.cumem")
    def patch_allocator(module):
        def timed_call(original, category, byte_count):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                record = getattr(state, "record", None)
                if record is None:
                    return original(*args, **kwargs)
                start = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    record[category + "_ms"] += (time.perf_counter() - start) * 1000
                    record[category + "_calls"] += 1
                    record[category + "_bytes"] += byte_count(args)
            return wrapped

        module.create_and_map = timed_call(module.create_and_map, "map", lambda args: args[0][1])
        module.libcudart.cudaMemcpy = timed_call(module.libcudart.cudaMemcpy, "copy", lambda args: args[2])
        original = module.CuMemAllocator.wake_up

        @functools.wraps(original)
        def wake(self, *args, **kwargs):
            record = {"kind": "allocator_wake", **{f"{category}_{field}": 0
                      for category in ("map", "copy") for field in ("ms", "calls", "bytes")}}
            state.record = record
            start = time.perf_counter()
            try:
                result = original(self, *args, **kwargs)
                record["status"] = "ok"
                return result
            except Exception:
                record["status"] = "failed"
                raise
            finally:
                end = time.perf_counter()
                record.update(start_monotonic_s=start, end_monotonic_s=end,
                              total_ms=(end - start) * 1000)
                state.record = None
                emit(record)

        module.CuMemAllocator.wake_up = wake
        emit({"kind": "probe_installed", "module": module.__name__,
              "pin_memory": module.PIN_MEMORY})

    @when_imported("vllm.v1.worker.gpu_worker")
    def patch_worker(module):
        original = module.Worker.wake_up

        @functools.wraps(original)
        def wake(self, *args, **kwargs):
            start = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                end = time.perf_counter()
                emit({"kind": "worker_wake", "total_ms": (end - start) * 1000,
                      "start_monotonic_s": start, "end_monotonic_s": end})

        module.Worker.wake_up = wake

    def instrument(owner, name, kind):
        original = getattr(owner, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            start = time.perf_counter()
            status = "failed"
            try:
                result = original(*args, **kwargs)
                status = "ok"
                return result
            finally:
                end = time.perf_counter()
                emit({"kind": kind, "status": status, "start_monotonic_s": start,
                      "end_monotonic_s": end, "total_ms": (end - start) * 1000})

        setattr(owner, name, measured)

    @when_imported("kvcached.kv_cache_manager")
    def patch_manager(module):
        instrument(module.KVCacheManager, "_alloc", "kvcached_alloc")
        instrument(module.KVCacheManager, "available_size", "kvcached_available_size")
        emit({"kind": "probe_installed", "module": module.__name__})

    @when_imported("kvcached.vmm_ops")
    def patch_pages(module):
        try:
            instrument(module.PageAllocator, "alloc_page", "kvcached_alloc_page")
            emit({"kind": "probe_installed", "module": module.__name__})
        except (AttributeError, TypeError) as exc:
            emit({"kind": "probe_unavailable", "module": module.__name__, "error": str(exc)})

    @when_imported("kvcached.tp_ipc_util")
    def patch_ipc(module):
        instrument(module, "broadcast_map_to_kv_tensors", "kvcached_map_ipc")
