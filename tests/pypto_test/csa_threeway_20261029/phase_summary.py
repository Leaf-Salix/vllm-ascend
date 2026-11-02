# SPDX-License-Identifier: Apache-2.0
"""汇总诊断发布端点：完成里程碑与每核等待，不把跨核时间相加。"""

import argparse
import json
import statistics
from pathlib import Path


def summarize(directory):
    arms = {}
    for arm in ("ours", "nalinaly"):
        trials = []
        for path in sorted(directory.glob(f"phase_{arm}_*/phase_clock.json")):
            rows = json.loads(path.read_text())
            origin = min(r["notify_start_us"] for r in rows if r["phase"] == 0)
            stages = {}
            for name in sorted({r["name"] for r in rows}):
                selected = [r for r in rows if r["name"] == name]
                roles = {}
                for role, subset in (
                    ("AIC", [r for r in selected if r["core"] < 24]),
                    ("AIV", [r for r in selected if r["core"] >= 24]),
                ):
                    intervals = [r for r in subset if "interval_us" in r]
                    if not subset:
                        continue
                    roles[role] = dict(
                        cores=len(subset),
                        finish_us=max(r["notify_end_us"] for r in subset) - origin,
                        median_interval_us=statistics.median(r["interval_us"] for r in intervals)
                        if intervals
                        else None,
                        median_gm_wait_us=statistics.median(r["wait_us"] for r in intervals) if intervals else None,
                        median_other_us=statistics.median(r["other_us"] for r in intervals) if intervals else None,
                    )
                stages[name] = roles
            trials.append(stages)
        assert len(trials) == 5, (arm, len(trials))
        aggregate = {}
        for stage, roles in trials[0].items():
            aggregate[stage] = {}
            for role in roles:
                aggregate[stage][role] = {}
                for key in roles[role]:
                    values = [t[stage][role][key] for t in trials]
                    aggregate[stage][role][key] = (
                        dict(median=statistics.median(values), min=min(values), max=max(values))
                        if all(v is not None for v in values)
                        else None
                    )
        arms[arm] = aggregate
    return dict(
        diagnostic_only=True,
        interpretation="发布完成里程碑；other包含搬运/FFTS等待/探针，跨核阶段不相加。",
        arms=arms,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / "phase-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name in ("PHASE_NORM", "PHASE_QUERY", "PHASE_TOPK", "PHASE_ATTN_CUBE", "PHASE_ATTN_VECTOR", "PHASE_OUT"):
        print(name, {a: result["arms"][a].get(name) for a in result["arms"]})
