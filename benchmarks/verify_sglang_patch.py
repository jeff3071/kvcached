# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Verify that a real SGLang workload used every kvcached VMM pool."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_file", type=Path)
    parser.add_argument(
        "--reset", action="store_true", help="remove an old trace before startup"
    )
    args = parser.parse_args()

    if args.reset:
        args.trace_file.unlink(missing_ok=True)
        print(f"kvcached verification trace: {args.trace_file}")
        return 0

    if not args.trace_file.is_file():
        print("FAIL: SGLang did not create a kvcached VMM trace", file=sys.stderr)
        return 1

    events: dict[tuple[int, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for line_number, line in enumerate(
        args.trace_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            pid_text, pool_name, event, size_text = line.split("\t")
            events[(int(pid_text), pool_name)][event] += int(size_text)
        except ValueError:
            print(
                f"FAIL: malformed trace line {line_number}: {line!r}",
                file=sys.stderr,
            )
            return 1

    registered = {
        key for key, counts in events.items() if counts.get("register", 0) > 0
    }
    failures = []
    for pid, pool_name in sorted(registered):
        counts = events[(pid, pool_name)]
        if counts.get("alloc", 0) == 0:
            failures.append(f"pid={pid} pool={pool_name}: no KVCacheManager.alloc")
        if counts.get("free", 0) == 0:
            failures.append(f"pid={pid} pool={pool_name}: no KVCacheManager.free")

    if not registered:
        failures.append("no kvcached VMM pool was created")

    if failures:
        print("FAIL: SGLang bypassed kvcached for one or more pools", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("PASS: the SGLang workload exercised kvcached VMM")
    for key in sorted(registered):
        counts = events[key]
        print(
            f"  pid={key[0]} pool={key[1]}: "
            f"allocated={counts['alloc']} freed={counts['free']} blocks"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
