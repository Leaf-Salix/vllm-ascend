# SPDX-License-Identifier: Apache-2.0
"""Compose independently measured candidates without moving the v122 denominator."""

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_b1_sort_pipeline import skip_unused_fills
from prepare_batched_sort import transform
from prepare_control_tiling_candidates import idle_publications
from prepare_mixed_candidates import live_half_source


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1-unused-fills", action="store_true")
    args = parser.parse_args()
    name = "combined_v2" if args.b1_unused_fills else "combined_v1"
    reference = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.glob("*.py")} == expected
    score = (reference / "indexer_score_stages.py").read_text()
    # Each transformation touches a distinct function. This is a source-level
    # composition check, not proof that the generated pipelines' gains add.
    combined = transform(live_half_source(score))
    assert combined == live_half_source(transform(score))
    if args.b1_unused_fills:
        combined = skip_unused_fills(combined)
        assert combined == transform(live_half_source(skip_unused_fills(score)))
    vector = idle_publications((reference / "vector_schedule.py").read_text())
    revised = {"indexer_score_stages.py": combined, "vector_schedule.py": vector}
    target = here / "runs" / name
    shutil.copytree(reference, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for file_name, source_text in revised.items():
        ast.parse(source_text)
        (target / file_name).write_text(source_text)
    report = {
        "source": str(target),
        "reference_kind": "frozen_tnd_v122",
        "adopted": False,
        "device_execution": False,
        "components": ["b1_idle_publish", "live_half", "batched_sort3_v2"],
        "excluded": ["live_half_rotate", "output_balance_off", "short_half_ub96", "oa_rows48", "oa_rows64"],
        "independent_function_transforms_commute": True,
        "changed_files": list(revised),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    }
    if args.b1_unused_fills:
        report["components"].append("b1_sort_unused_fills")
        report["excluded"].append("b1_sort_stream")
    with (here / f"{name}.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
