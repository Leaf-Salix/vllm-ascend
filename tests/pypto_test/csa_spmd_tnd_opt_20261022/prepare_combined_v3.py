# SPDX-License-Identifier: Apache-2.0
"""Compose tested Query/ordering changes with the previously validated v1 set.

Preparation and compilation do not imply adoption or additive performance gains.
The independent Query boundary trials must finish before submitting this set.
"""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_control_tiling_candidates import idle_publications
from prepare_main_vector_late import transform as move_main
from prepare_query_head_guard import transform_vector as forward_query_flag


def hashes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("*.py")}


def main():
    here = Path(__file__).resolve().parent
    parents = {}
    for name in ("frozen_v122", "combined_v1", "query_head_guard", "main_vector_late"):
        path = here / "runs" / name
        expected = json.loads((here / (name + ".json")).read_text())["files"]
        assert hashes(path) == expected, name
        parents[name] = path
    original_vector = (parents["frozen_v122"] / "vector_schedule.py").read_text()
    assert move_main(original_vector) == (parents["main_vector_late"] / "vector_schedule.py").read_text()
    assert idle_publications(original_vector) == (parents["combined_v1"] / "vector_schedule.py").read_text()
    # The B1 publication changes precede the moved main-compressor block. Prove
    # that these source transformations commute before applying the new flag.
    vector = move_main(idle_publications(original_vector))
    assert vector == idle_publications(move_main(original_vector))
    vector = forward_query_flag(vector)
    target = here / "runs/combined_v3"
    shutil.copytree(parents["query_head_guard"], target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "vector_schedule.py").write_text(vector)
    shutil.copyfile(parents["combined_v1"] / "indexer_score_stages.py", target / "indexer_score_stages.py")
    for file in target.glob("*.py"):
        ast.parse(file.read_text())
    report = dict(
        source=str(target),
        reference_kind="frozen_tnd_v122",
        adopted=False,
        device_execution=False,
        components=["b1_idle_publish", "live_half", "batched_sort3_v2", "main_vector_late", "query_head_guard"],
        excluded=[
            "b1_sort_unused_fills (combined_v2 direction mixed)",
            "WS output pool (no full-L1 benefit)",
            "front_partition",
            "wide Score panels",
            "unguarded Query head orientation",
        ],
        sources={key: str(path) for key, path in parents.items()},
        publication_and_main_order_changes_commute=True,
        performance="Not measured; independent improvements cannot be added",
        files=hashes(target),
    )
    with (here / "combined_v3.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
