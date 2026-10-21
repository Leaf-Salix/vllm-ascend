# SPDX-License-Identifier: Apache-2.0
"""Prepare B1 idle-publication and independent O_A row-tile prototypes."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_control_candidates import replace_once


def idle_publications(source):
    source = replace_once(
        source,
        "    signal = publish_phase(signal, PHASE_INPUT, core, CORES, generation)\n",
        "    if BALANCED and QUERY_GROUP == 2:\n"
        "        if worker < (tokens + 7) // 8:\n"
        "            signal = publish_phase(signal, PHASE_INPUT, core, CORES, generation)\n"
        "        elif worker == pl.min((tokens + 7) // 8, VECTORS - 1):\n"
        "            # The idle metadata planner still acquires this invocation's Q/L/pages.\n"
        "            # Other idle roles acquire business data at the existing QA wait.\n"
        "            pl.system.fence()\n"
        "            pl.system.cacheinvalid()\n"
        "            pl.system.fence()\n"
        "    else:\n"
        "        signal = publish_phase(signal, PHASE_INPUT, core, CORES, generation)\n",
    )
    for phase in ("QR", "ROPE"):
        source = replace_once(
            source,
            f"    signal = publish_phase(signal, PHASE_{phase}, core, CORES, generation)\n",
            "    if not (BALANCED and QUERY_GROUP == 2) or worker < (tokens + 7) // 8:\n"
            f"        signal = publish_phase(signal, PHASE_{phase}, core, CORES, generation)\n",
        )
    return source


def row_work_model(tokens, rows):
    """Valid-row arithmetic and weight reload counts, not a latency prediction."""
    blocks = (tokens + rows - 1) // rows
    valid = [0] * 24
    rounded = [0] * 24
    jobs = [0] * 24
    for logical in range(8 * blocks * 4):  # The tested N256 domain.
        row = (logical % (blocks * 4)) // 4
        count = min(rows, tokens - row * rows)
        valid[logical % 24] += count
        rounded[logical % 24] += (count + 15) // 16 * 16
        jobs[logical % 24] += 1
    return dict(
        tokens=tokens,
        row_tile=rows,
        row_blocks=blocks,
        valid_rows_per_aic=valid,
        rounded_m16_rows_per_aic=rounded,
        weight_tiles_per_aic=jobs,
        maximum_rounded_rows=max(rounded),
        maximum_weight_tiles=max(jobs),
        total_weight_tiles=sum(jobs),
    )


def main():
    here = Path(__file__).resolve().parent
    reference = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.glob("*.py")} == expected
    sources = {
        name: (reference / name).read_text()
        for name in ("vector_schedule.py", "cube_schedule.py", "output_projection_a_stages.py")
    }
    variants = {"b1_idle_publish": {"vector_schedule.py": idle_publications(sources["vector_schedule.py"])}}
    for rows in (48, 64):
        revised = {
            "output_projection_a_stages.py": replace_once(
                sources["output_projection_a_stages.py"], "PROJ_A_ROW_TILE = 128\n", f"PROJ_A_ROW_TILE = {rows}\n"
            )
        }
        for name, count in (("cube_schedule.py", 1), ("vector_schedule.py", 2)):
            text = sources[name]
            before = "((tokens + 127) // 128) * (1024 // OA_COLS)"
            assert text.count(before) == count
            revised[name] = text.replace(before, f"((tokens + {rows - 1}) // {rows}) * (1024 // OA_COLS)")
        variants[f"oa_rows{rows}"] = revised
    report = {
        "reference_kind": "frozen_tnd_v122",
        "adopted": False,
        "device_execution": False,
        "scope": "Independent prototypes; O_A tile is global here, not yet a service policy",
        "variants": {},
    }
    for name, revised in variants.items():
        target = here / "runs" / name
        shutil.copytree(reference, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for filename, text in revised.items():
            ast.parse(text)
            (target / filename).write_text(text)
        report["variants"][name] = dict(
            source=str(target),
            changed_files=list(revised),
            files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
        )
    report["oa_work_models"] = [
        row_work_model(t, r) for t, sizes in ((144, (128, 48)), (168, (128, 64)), (192, (128, 64))) for r in sizes
    ]
    with (here / "control_tiling_candidates.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"prepared": list(variants), "device_execution": False}))


if __name__ == "__main__":
    main()
