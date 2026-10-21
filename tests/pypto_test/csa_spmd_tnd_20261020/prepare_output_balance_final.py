# SPDX-License-Identifier: Apache-2.0
"""Name the ownership constants and freeze reviewable source for the full cohort."""

import difflib
import hashlib
import json
import shutil
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/output_work_balance_native_candidate_v3"
    target = here / "runs/output_work_balance_native_candidate_v4"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "cube_schedule.py"
    before = path.read_text()
    anchor = "\n\n@pl.jit.incore\ndef cube_schedule("
    constants = """

# The existing N128 output grid has 64 jobs on 24 physical AICs: the first
# sixteen get three O_A jobs, the last eight get two. Software O_B ranks
# compensate with two/four jobs respectively; they do not add physical cores.
OUTPUT_BALANCE_HEAVY_CORES = 16
OUTPUT_BALANCE_VIRTUAL_WORKERS = 32
OUTPUT_BALANCE_LAST_RANK = OUTPUT_BALANCE_VIRTUAL_WORKERS - 1
OUTPUT_BALANCE_COLUMNS_PER_GROUP = 8
"""
    assert before.count(anchor) == 1
    text = before.replace(anchor, constants + anchor)
    start = text.index("        if OUTPUT_WORK_BALANCE:")
    end = text.index("        else:\n            for ob_group", start)
    old_block = text[start:end]
    block = old_block
    replacements = {
        "worker + 16": "worker + OUTPUT_BALANCE_HEAVY_CORES",
        "worker >= 16": "worker >= OUTPUT_BALANCE_HEAVY_CORES",
        "worker - 16": "worker - OUTPUT_BALANCE_HEAVY_CORES",
        "(ob_group % 2) * 8": "(ob_group % 2) * OUTPUT_BALANCE_COLUMNS_PER_GROUP",
        "ob_group * 8": "ob_group * OUTPUT_BALANCE_COLUMNS_PER_GROUP",
        "ob_first + 31": "ob_first + OUTPUT_BALANCE_LAST_RANK",
        "// 32) * 32": "// OUTPUT_BALANCE_VIRTUAL_WORKERS) * OUTPUT_BALANCE_VIRTUAL_WORKERS",
        "ob_first + 8": "ob_first + OUTPUT_BALANCE_COLUMNS_PER_GROUP",
        "ob_worker, 32, OB_ROWS": "ob_worker, OUTPUT_BALANCE_VIRTUAL_WORKERS, OB_ROWS",
    }
    for old, new in replacements.items():
        assert block.count(old) == 1, old
        block = block.replace(old, new)
    text = text[:start] + block + text[end:]
    path.write_text(text)
    cleanup = dict(constants=constants, anchor=anchor, replacements=replacements,
                   before_block=old_block, after_block=block)
    (here / "output_balance_constant_cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
    (here / "output_balance_constant_cleanup.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), text.splitlines(True), fromfile="a/cube_schedule.py", tofile="b/cube_schedule.py",
    )))
    old_snapshot = here / "runs/output_balance_long_mixed_t65_bcap16_v72_source"
    canonical = here / "runs/output_balance_final_canonical_v4_source"
    shutil.copytree(old_snapshot, canonical, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    entries = json.loads((canonical / "source_manifest.json").read_text())
    for name in (target / "cube_schedule.py",):
        shutil.copyfile(name, canonical / "spmd" / name.name)
    for name in entries:
        entries[name] = hashlib.sha256((canonical / name).read_bytes()).hexdigest()
    assert all((canonical / "spmd" / f.name).read_bytes() == f.read_bytes() for f in target.glob("*.py"))
    (canonical / "source_manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    report = dict(candidate=str(target), parent=str(parent), canonical_snapshot=str(canonical),
                  device_execution=False, adopted=False,
                  scope="Constants only since v3; exact generated-code proof is still required")
    (here / "output_work_balance_v4.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
