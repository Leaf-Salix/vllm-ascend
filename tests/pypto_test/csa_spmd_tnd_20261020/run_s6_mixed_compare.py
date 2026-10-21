# SPDX-License-Identifier: Apache-2.0
"""Run the requested two input shapes on one queue-assigned device.

Each child uses the existing exact-regression and paired full-chain L1 harness.
This driver changes neither operator code nor timing/precision checks. The old
SPMD kernel also consumes the real Q/L in the mixed diagnostic; its measurement
does not establish that the old BSH service entry supports mixed requests.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HISTORY = 131072
BATCH = 16
MIXED_QUERIES = (1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 3, 4, 6)
CASES = (("uniform_s6", (6,) * BATCH), ("mixed_s1_s6", MIXED_QUERIES))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def validate_result(report, queries, windows):
    """Reject a wrong fixture, failed regression, or incomplete timing window."""
    if not all(report.get(key) for key in ("passed", "functional_passed", "device_execution", "performance_measured")):
        raise ValueError("case did not finish functional/performance validation")
    if report.get("stage") != "complete":
        raise ValueError("case is not complete")
    if (report["batch"], report["history"], report["query_tokens"]) != (BATCH, HISTORY, 6):
        raise ValueError("wrong B/history/query upper bound")
    if report["query_lengths"] != list(queries) or report["history_lengths"] != [HISTORY] * BATCH:
        raise ValueError("wrong per-request query/history lengths")
    tokens = sum(queries)
    if (report["actual_tokens"], report["token_capacity"], report["request_capacity"]) != (tokens, tokens, BATCH):
        raise ValueError("unexpected padding or request capacity")
    if not report["original_spmd_attribution"]["candidate_exactly_matches"]:
        raise ValueError("new difference against the same-input original SPMD")
    if not report["bsh_wait_policy_audit"]["passed"] or report["bsh_wait_policy_audit"]["ticks"] != 150:
        raise ValueError("wait policy audit failed")
    if not report["toolchain_post_run_audit"]["passed"]:
        raise ValueError("shared toolchain audit failed")
    l1 = report["l1"]
    if not l1["passed"] or not all(l1["protocol"].values()) or l1.get("shared_workspace"):
        raise ValueError("L1/protocol failed or diagnostic shared workspace enabled")
    stats = {}
    if len(l1["windows"]) != 2 * windows:
        raise ValueError("wrong number of paired windows")
    for side in ("spmd_original", "spmd_tnd"):
        records = [row for row in l1["windows"] if row["side"] == side]
        if len(records) != windows or any(
            (row["cores"], row["task_count"], row["clock_freq_hz"]) != (72, 1, 50000000) for row in records
        ):
            raise ValueError("incomplete single-MIX/full72core L1 measurement")
        samples = [row["incore_span_us"] for row in records]
        if not all(math.isfinite(value) and value > 0 for value in samples):
            raise ValueError("invalid latency")
        median = statistics.median(samples)
        if samples != l1["statistics_us"][side]["samples_us"] or median != l1["statistics_us"][side]["median_us"]:
            raise ValueError("raw windows and reported statistics disagree")
        stats[side] = {
            "samples_us": samples,
            "median_us": median,
            "us_per_actual_token": median / tokens,
            "actual_tokens_per_second": tokens * 1e6 / median,
        }
    return {
        "query_lengths": list(queries),
        "query_histogram": dict(sorted(Counter(queries).items())),
        "actual_tokens": tokens,
        "token_capacity": tokens,
        "mean_query_tokens": tokens / BATCH,
        "policy": report["policy"],
        "statistics_us": stats,
        "tnd_vs_original_same_input_latency_change_pct": (
            stats["spmd_tnd"]["median_us"] / stats["spmd_original"]["median_us"] - 1
        )
        * 100,
        "no_new_regression": True,
        "native_precision_passed": report["precision_exact"],
    }


def summarize(output, windows):
    rows, reports, snapshots = {}, {}, []
    for name, queries in CASES:
        result_path = output / name / "result.json"
        report = json.loads(result_path.read_text())
        reports[name] = report
        rows[name] = validate_result(report, queries, windows)
        rows[name]["raw_result"] = str(result_path)
        snapshot = output / (name + "_source")
        snapshots.append(
            (
                json.loads((snapshot / "source_manifest.json").read_text()),
                json.loads((snapshot / "shared_toolchain_manifest.json").read_text()),
            )
        )
    if snapshots[0] != snapshots[1]:
        raise ValueError("operator/baseline/harness/toolchain changed between shapes")
    if snapshots[0][1].get("status") != "validated":
        raise ValueError("shared toolchain release is not validated")
    first, second = (reports[name] for name, _ in CASES)
    if first["runtime_artifacts"] != second["runtime_artifacts"]:
        raise ValueError("loaded runtime artifacts differ between shapes")
    driver = json.loads((output / "driver.json").read_text())
    uniform, mixed = (rows[name] for name, _ in CASES)
    changes = {}
    for side in ("spmd_original", "spmd_tnd"):
        before = uniform["statistics_us"][side]
        after = mixed["statistics_us"][side]
        changes[side] = {
            "mixed_vs_s6_latency_change_pct": (after["median_us"] / before["median_us"] - 1) * 100,
            "mixed_vs_s6_us_per_token_change_pct": (after["us_per_actual_token"] / before["us_per_actual_token"] - 1)
            * 100,
        }
    result = {
        "passed": True,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "device": driver["device"],
        "scope": "TP1/layer4/W8A8/NZ2; full HC_pre/RMSNorm/CSA/HC_post L1; median of five paired windows",
        "history_per_request": HISTORY,
        "batch": BATCH,
        "fixture_seed": 1024,
        "operator_candidate": driver["candidate"],
        "shared_toolchain": snapshots[0][1],
        "same_source_and_runtime": True,
        "formal_eight_case_gate_eligible": False,
        "rows": rows,
        "input_workload_comparison": changes,
        "actual_token_count_change_pct": (mixed["actual_tokens"] / uniform["actual_tokens"] - 1) * 100,
        "policy_differences": {
            key: {"s6": value, "mixed": mixed["policy"][key]}
            for key, value in uniform["policy"].items()
            if mixed["policy"][key] != value
        },
        "interpretation": [
            "16 cannot divide equally among six lengths: each S appears two or three times; mixed T=56, average S=3.5.",
            "Both cases are packed without dummy/padding tokens, using their actual token capacities 96 and 56.",
            "Mixed-input latency includes fewer tokens and different policies; it is not an equal-work layout speedup.",
            "Original SPMD uses the same real input; mixed success does not establish old BSH service support.",
            "Each shape has five alternating windows. Shapes run in separate processes on one reserved card.",
            "Weights and seed match; shape changes RNG consumption, so payloads are not a byte-identical subset.",
            "Existing Native discrepancies remain separate; strict same-input original-SPMD equality is required.",
        ],
    }
    save_json(output / "comparison.json", result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.summarize_only:
        result = summarize(args.output, 5)
        print(json.dumps(result["input_workload_comparison"], ensure_ascii=False), flush=True)
        return
    if "TASK_DEVICE" not in os.environ or args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("device execution must use the queue-assigned TASK_DEVICE")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.launcher.is_file() or not (args.candidate_source / "entry.py").is_file():
        raise FileNotFoundError("missing frozen launcher or candidate")
    args.output.mkdir(parents=True)
    state = {
        "device": args.device,
        "candidate": str(args.candidate_source.resolve()),
        "launcher": str(args.launcher.resolve()),
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "cases": [],
    }
    save_json(args.output / "driver.json", state)
    for name, queries in CASES:
        command = [
            "bash",
            str(args.launcher.resolve()),
            str((args.output / name).resolve()),
            "--bsh-probe",
            "--native-toolchain",
            "--candidate-source",
            str(args.candidate_source.resolve()),
            "--batch",
            str(BATCH),
            "--history",
            str(HISTORY),
            "--query-tokens",
            "6",
            "--query-lengths",
            ",".join(map(str, queries)),
            "--history-lengths",
            ",".join([str(HISTORY)] * BATCH),
            "--token-capacity",
            str(sum(queries)),
            "--request-capacity",
            str(BATCH),
            "--l1-windows",
            "5",
            "--device",
            str(args.device),
        ]
        log_path = args.output / (name + ".log")
        entry = {"name": name, "query_lengths": list(queries), "command": command, "log": str(log_path)}
        state["cases"].append(entry)
        save_json(args.output / "driver.json", state)
        print(f"Starting {name}, T={sum(queries)}, device={args.device}, log={log_path}", flush=True)
        with log_path.open("x") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        entry["exit_code"] = completed.returncode
        save_json(args.output / "driver.json", state)
        if completed.returncode:
            raise RuntimeError(f"{name} failed (exit={completed.returncode}); see {log_path}")
        report = json.loads((args.output / name / "result.json").read_text())
        row = validate_result(report, queries, 5)
        print(json.dumps({"case": name, "statistics_us": row["statistics_us"]}, ensure_ascii=False), flush=True)
    result = summarize(args.output, 5)
    print(json.dumps(result["input_workload_comparison"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
