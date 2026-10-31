# SPDX-License-Identifier: Apache-2.0
"""离线汇总真实模型三方结果；不把启动耗时、接受事件或空样本当作速度/精度。"""

import argparse
import json
import statistics
from pathlib import Path

from postflight import summarize_tokens, timing_inputs


def summarize(root, native):
    experiment = json.loads((root / "experiment.json").read_text())
    report = {"arm": experiment["arm"], "csa_source": experiment["csa_source"], "root": str(root)}
    difference = root / "FIRST_DIFFERENCE.json"
    if difference.exists():
        report["first_difference"] = json.loads(difference.read_text())
    if not (root / "postflight.json").exists() or not all((root / f"rank{i}.mixed.json").exists() for i in range(8)):
        report.update(complete=False, performance_qualified=False, reason="incomplete generation/observation")
        return report
    ranks = [json.loads((root / f"rank{i}.mixed.json").read_text()) for i in range(8)]
    reference = [json.loads((native / f"rank{i}.mixed.json").read_text()) for i in range(8)]
    tokens = summarize_tokens(root, native, 8)
    report["tokens"] = tokens
    observations = [row["observation"][0]["threeway_timing"] for row in ranks]
    samples = [row["samples"] for row in observations]
    matching = [timing_inputs(a) == timing_inputs(n) for a, n in zip(ranks, reference, strict=True)]
    complete = all(row["complete"] and len(row["samples"]) == 20 for row in observations)
    report.update(complete=True, timing_complete=complete, timing_inputs_equal_by_rank=matching)
    if complete:
        values = [row["forward_us"] for rows in samples for row in rows]
        slowest = [max(samples[rank][i]["forward_us"] for rank in range(8)) for i in range(20)]
        report["target_forward_us"] = {
            "samples": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "ep8_slowest_rank_median": statistics.median(slowest),
            "scope": "20 selected B8 FULL240 target-model forwards/rank, excludes draft; not single CSA",
        }
    waves = [row.get("generation_waves", []) for row in ranks]
    if all(len(rows) == 5 and [row["offset"] for row in rows] == [0, 8, 16, 24, 32] for rows in waves):
        seconds = sum(max(waves[rank][i]["elapsed_s"] for rank in range(8)) for i in range(5))
        token_count = sum(row["output_tokens"] for rows in waves for row in rows)
        report["client_waves"] = {
            "summed_slowest_rank_s": seconds,
            "output_tokens": token_count,
            "tokens_per_s": token_count / seconds,
            "scope": (
                "generate_aligned_batch wall time including enqueue/admission coordination; excludes outer done barrier"
            ),
        }
    strict = json.loads((root / "postflight.json").read_text())
    report["acceptance"] = {
        k: strict.get("acceptance", {}).get(k)
        for k in ("mean_accepted", "accepted_length_histogram", "total_preemptions", "qualified")
    }
    report["performance_qualified"] = (
        complete
        and all(matching)
        and not strict["coverage_errors"]
        and tokens["zero_preemptions"]
        and bool(experiment.get("wave_measurement"))
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps([summarize(root, args.native) for root in args.roots], ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
