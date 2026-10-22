# SPDX-License-Identifier: Apache-2.0
"""Combine the validated v1 set, main scheduling and the B1 quantization guard.

No Query head rewrite or previously rejected broad guard is included. Components
retain their measured domain; only a fresh full-chain pair can measure this set.
"""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_control_tiling_candidates import idle_publications
from prepare_main_vector_late import transform as move_main
from prepare_output_quant_guard import transform_vector as forward_quant_flag


def hashes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("*.py")}


def main():
    here = Path(__file__).resolve().parent
    parents = {}
    for name in ("frozen_v122", "combined_v1", "output_quant_guard", "main_vector_late"):
        path = here / "runs" / name
        expected = json.loads((here / (name + ".json")).read_text())["files"]
        assert hashes(path) == expected, name
        parents[name] = path
    original = (parents["frozen_v122"] / "vector_schedule.py").read_text()
    assert move_main(original) == (parents["main_vector_late"] / "vector_schedule.py").read_text()
    assert idle_publications(original) == (parents["combined_v1"] / "vector_schedule.py").read_text()
    vector = move_main(idle_publications(original))
    assert vector == idle_publications(move_main(original))
    vector = forward_quant_flag(vector)
    target = here / "runs/combined_v4"
    shutil.copytree(parents["output_quant_guard"], target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "vector_schedule.py").write_text(vector)
    shutil.copyfile(parents["combined_v1"] / "indexer_score_stages.py", target / "indexer_score_stages.py")
    for file in target.glob("*.py"):
        ast.parse(file.read_text())
    report = dict(
        source=str(target),
        reference_kind="frozen_tnd_v122",
        adopted=False,
        device_execution=False,
        components=["b1_idle_publish", "live_half", "batched_sort3_v2", "main_vector_late", "output_quant_guard"],
        excluded=["all Query head rewrites", "rejected combined_v3", "unused sort fills", "wide Score", "WS pool"],
        sources={key: str(path) for key, path in parents.items()},
        publication_and_main_order_changes_commute=True,
        performance="Not measured; independent improvements cannot be added",
        files=hashes(target),
    )
    with (here / "combined_v4.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
