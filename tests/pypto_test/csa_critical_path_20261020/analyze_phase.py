# SPDX-License-Identifier: Apache-2.0
"""从已有全核端点生成阶段摘要；不是硬件引擎独占时间或正式性能。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def summarize(rows):
    by_core = defaultdict(list)
    by_phase = defaultdict(list)
    for row in rows:
        by_core[row["core"]].append(row)
    if sorted(by_core) != list(range(72)):
        raise ValueError("阶段数据必须覆盖24个AIC及48个AIV")
    for core, entries in by_core.items():
        previous = None
        for row in sorted(entries, key=lambda value: value["notify_start_us"]):
            interval = row["notify_start_us"] - previous["notify_end_us"] if previous else 0
            wait = row["cumulative_wait_us"] - previous["cumulative_wait_us"] if previous else 0
            by_phase[row["name"]].append({"core": core, "interval": interval,
                "wait": wait, "other": interval - wait, "end": row["notify_end_us"]})
            previous = row
    result = {}
    for name, entries in by_phase.items():
        result[name] = {"roles": len(entries), "end_us": max(row["end"] for row in entries),
            **{key + "_us": round(median(row[key] for row in entries), 2)
               for key in ("interval", "wait", "other")}}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"source": str(args.raw), "performance_eligible": False,
              "stages": summarize(json.loads(args.raw.read_text()))}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
