# SPDX-License-Identifier: Apache-2.0
"""仅管理本次自有进程组；首差或720秒时终止，并保留退出状态。"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path


def stop_group(process, grace=5):
    """All ranks in the frozen runner inherit this owned session/group."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=grace)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    return process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("bank", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    source, output = args.source, args.output
    manifest = json.loads((source.parent / "manifest.json").read_text())
    if manifest["arm"] != "native":
        assert args.reference is not None
        for rank in range(16):
            row = json.loads((args.reference / f"rank{rank}.mixed.json").read_text())
            assert len(row["requests"]) == 40 and all(len(r["output_token_ids"]) == 192 for r in row["requests"])
    else:
        assert args.reference is None
    experiment = {
        "native_reference": str(args.reference) if args.reference else None,
        "arm": manifest["arm"],
        "common_framework": manifest["common_framework"],
        "csa_source": manifest["csa_source"],
        "TP": 1,
        "DP": 16,
        "EP": 16,
        "batch_per_rank": 40,
        "prefix": 8192,
        "output_tokens": 192,
        "HCA": "native",
        "gpu_blocks_override": 22000,
        "runtime_ring_heap_bytes": [256 << 20, 512 << 20, 256 << 20, 256 << 20],
        "event_mode": 1,
        "eplb": False,
        "atomic_add": False,
        "capture_sizes": [6, 240],
        "task_device": os.environ["TASK_DEVICE"],
    }
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    command = [
        sys.executable,
        str(source / "tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py"),
        "--shared-toolchain",
        "--native-plugin-root",
        str(source),
        str(source / "tests/pypto_test/offline_pd/run.py"),
        "decode",
        "--bank",
        str(args.bank),
        "--output",
        str(output),
        "--backend",
        manifest["backend"],
        "--pto-attention",
        "csa",
        "--mixed-requests",
        "--batch",
        "40",
        "--decode-dp",
        "16",
        "--decode-tokens",
        "192",
        "--max-num-batched-tokens",
        "400",
        "--capture-sizes",
        "6",
        "240",
        "--graph-mode",
        "full_decode_only",
        "--gpu-memory-utilization",
        "0.95",
        "--weight-nz-mode",
        "2",
        "--deterministic",
        "--event-work-mode",
        "1",
    ]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    start = time.monotonic()
    reason = None
    with (output / "launcher.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if (output / "STOP").exists() or time.monotonic() - start >= 720:
                    reason = "first_difference" if (output / "STOP").exists() else "720s_limit"
                    break
                time.sleep(0.5)
        finally:
            code = stop_group(process)
    status = {"exit_code": code, "reason": reason, "elapsed_s": time.monotonic() - start}
    (output / "exit.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    raise SystemExit(1 if reason or code else 0)


if __name__ == "__main__":
    main()
