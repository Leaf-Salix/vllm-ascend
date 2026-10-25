# SPDX-License-Identifier: Apache-2.0
"""保留完整交替样本与正确性结果，单轮微小收益不自动升级为正式采用。"""

import argparse
import json
import shutil
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt")
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    source = here / "runs" / args.attempt / "result.json"
    report = json.loads(source.read_text())
    frozen = json.loads((here / "manifest.json").read_text())
    assert report["passed"] and report["stage"] == "complete"
    assert report["source_hashes"]["reference"] == frozen["baseline"]["files"]
    assert report["shared_toolchain"]["release"] == frozen["shared_toolchain"]["release"]
    assert all(
        all(check["guards"].values())
        and all(value["bit_equal"] and value["status"] == "PASS" for value in check["numerical"].values())
        for check in report["checks"].values()
    )
    sides = {}
    for side in ("reference", "candidate"):
        windows = [w for w in report["windows"] if w["side"] == side]
        assert len(windows) >= 10
        assert all(w["records"] == 120 and w["task_groups"] == 2 and w["cores"] == 72 for w in windows)
        samples = [w["incore_span_us"] for w in windows]
        assert samples == report["statistics_us"][side]["samples"]
        assert statistics.median(samples) == report["statistics_us"][side]["median"]
        sides[side] = {
            "median_us": statistics.median(samples),
            "samples_us": samples,
            "main_mix_median_us": statistics.median(w["main_mix_us"] for w in windows),
        }
    a, b = [sides[k]["samples_us"] for k in ("reference", "candidate")]
    assert len(a) == len(b)
    ratio = sides["candidate"]["median_us"] / sides["reference"]["median_us"]
    evidence = here / "evidence" / (args.attempt + ".json")
    shutil.copy2(source, evidence)
    result = {
        "task_id": args.task,
        "case": report["case"]["id"],
        "device": report["device_id"],
        "candidate": report["sources"]["candidate"],
        "raw_report": str(evidence.relative_to(here)),
        "full_L1": sides,
        "ratio_to_same_task_frozen_control": ratio,
        "reduction_pct": (1 - ratio) * 100,
        "paired_windows_faster": sum(y < x for x, y in zip(a, b)),
        "paired_difference_us": [x - y for x, y in zip(a, b)],
        "median_paired_difference_us": statistics.median(x - y for x, y in zip(a, b)),
        "exact_eager_graph_post_l1": True,
        "adopted": False,
        "decision": "单轮达到0.5%，需独立复测稳定性" if ratio <= 0.995 else "暂存结果，当前收益未达到0.5%",
        "acceptance_preference": "用户明确：稳定可获得的0.5%以上收益也值得保留，同时推进更大优化",
        "eight_case_goal_achieved": False,
    }
    (here / (args.attempt + "_result.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("case", "device", "reduction_pct", "paired_windows_faster", "decision")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
