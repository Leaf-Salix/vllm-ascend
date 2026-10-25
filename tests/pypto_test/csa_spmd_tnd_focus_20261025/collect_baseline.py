# SPDX-License-Identifier: Apache-2.0
"""按原冻结算法归档当前SDK缺失的基线，合并全部20个窗口取中位数。"""

import argparse
import json
import shutil
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("attempt")
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    source = here / "runs" / args.attempt / "result.json"
    report = json.loads(source.read_text())
    frozen = json.loads((here / "manifest.json").read_text())
    baseline = json.loads((here / "baseline.json").read_text())
    assert report["case"]["id"] == args.case_id and report["passed"] and report["stage"] == "complete"
    assert report["source_hashes"]["reference"] == report["source_hashes"]["candidate"] == frozen["baseline"]["files"]
    assert report["shared_toolchain"]["release"] == frozen["shared_toolchain"]["release"]
    for sdk in ("pypto", "ptoas", "simpler"):
        assert report["shared_toolchain"][sdk]["commit"] == frozen["shared_toolchain"][sdk]["commit"]
    assert all(
        all(c["guards"].values()) and all(v["bit_equal"] and v["status"] == "PASS" for v in c["numerical"].values())
        for c in report["checks"].values()
    )
    windows = report["windows"]
    assert len(windows) == 20
    assert all(w["records"] == 120 and w["task_groups"] == 2 and w["cores"] == 72 for w in windows)
    for side in ("reference", "candidate"):
        samples = [w["incore_span_us"] for w in windows if w["side"] == side]
        assert len(samples) == 10 and samples == report["statistics_us"][side]["samples"]
    target = here / "evidence" / (args.case_id + "_baseline_v1.json")
    if target.exists():
        raise FileExistsError("不得覆盖已冻结基线")
    shutil.copy2(source, target)
    baseline["cases"][args.case_id] = {
        "task_id": args.task,
        "device": report["device_id"],
        "report": str(target.relative_to(here)),
        "samples_us": [w["incore_span_us"] for w in windows],
        "baseline_median_us": statistics.median(w["incore_span_us"] for w in windows),
        "main_mix_median_us": statistics.median(w["main_mix_us"] for w in windows),
        "initialization_median_us": statistics.median(w["initialization_us"] for w in windows),
        "gap_median_us": statistics.median(w["initialization_to_mix_gap_us"] for w in windows),
        "exact_eager_graph_post_l1": True,
    }
    baseline["other_cases_pending"] = [f"C{i:02}" for i in range(1, 9) if f"C{i:02}" not in baseline["cases"]]
    (here / "baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n")
    print(args.case_id, baseline["cases"][args.case_id]["baseline_median_us"])


if __name__ == "__main__":
    main()
