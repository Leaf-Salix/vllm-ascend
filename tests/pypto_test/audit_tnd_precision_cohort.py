# SPDX-License-Identifier: Apache-2.0
"""Run a frozen cohort's numerical gate, reporting READY qualification separately."""

import argparse
import importlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("native", "csa"))
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSON file; never overwrite")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # Add after Python startup: putting job in PYTHONPATH would execute its
    # sitecustomize under the CPU auditor's interpreter.
    sys.path.insert(0, str(args.job.resolve()))
    qualify = importlib.import_module("qualify192").qualify
    try:
        report = qualify(args.left, args.right, args.phase)
        strict = {"pass": True, "scope": "Default strict gate including READY reuse"}
    except ValueError as error:
        if args.phase != "csa" or str(error) not in {f"rank{rank}: READY reuse failed" for rank in range(16)}:
            raise
        strict = {"pass": False, "error": str(error), "scope": "Strict READY failure retained"}
        report = qualify(args.left, args.right, args.phase, require_ready_reuse=False)
    if args.phase == "csa":
        stats = []
        for rank in range(16):
            data = json.loads((args.right / f"rank{rank}.mixed.json").read_text())
            counters = data["observation"][0]["pypto_cache_stats"]
            if any(
                counters[name] != 0 for name in ("bypasses", "invalid_entries", "storage_errors", "forced_rebuilds")
            ):
                raise ValueError(f"rank{rank}: cache bypass/error")
            stats.append({"rank": rank, "stats": counters})
        report["cache_error_counters_zero_all16"] = True
        report["cache_stats_from_raw_observations"] = stats
    result = {"strict_qualification": strict, "numerical_qualification": report}
    # Exclusive creation also protects against another auditor choosing this file.
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"numeric_pass": report["pass"], "strict_pass": strict["pass"], "phase": args.phase}))


if __name__ == "__main__":
    main()
