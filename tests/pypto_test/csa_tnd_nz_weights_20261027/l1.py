# SPDX-License-Identifier: Apache-2.0
"""完整TND调用的L1首尾，分别校验单SPMD和多task结构。"""

import json
import math
from collections import Counter


def summarize(raw, variant):
    meta, rows = raw["metadata"], raw["aicore_tasks"]
    frequency = meta["clock_freq_hz"]
    assert raw["chip_swimlane_level"] == 1 and math.isfinite(frequency) and frequency > 0
    assert len(meta["run_boundaries"]) == 1 and meta["dropped_run_boundaries"] == 0
    assert meta["num_cores"] == 72 and Counter(meta["core_types"]) == {"aic": 24, "aiv": 48}
    assert rows and all(len(r) == 6 and r[3] > 0 and r[4] >= r[3] and 0 <= r[0] < 72 for r in rows)
    assert meta["run_boundaries"][0][1] <= min(r[3] for r in rows)
    tasks = Counter(r[1] for r in rows)
    assert {r[0] for r in rows} == set(range(72))
    if variant == "spmd":
        assert len(tasks) == 1 and len(rows) == 72
    else:
        assert variant == "mpmd" and len(tasks) > 1
    return {
        "incore_span_us": (max(r[4] for r in rows) - min(r[3] for r in rows)) * 1e6 / frequency,
        "records": len(rows),
        "task_groups": len(tasks),
        "cores": 72,
        "task_roles": {str(key): count for key, count in tasks.items()},
    }


def export(source, destination, *, side, variant, kernel_config):
    destination.mkdir()
    raw_path = source / "chip_swimlane_records.json"
    data = json.loads(raw_path.read_text())
    summary = summarize(data, variant)
    (destination / raw_path.name).write_bytes(raw_path.read_bytes())
    (destination / "kernel_config.py").write_bytes(kernel_config.read_bytes())
    return {"side": side, "variant": variant, **summary, "raw": str(destination / raw_path.name)}
