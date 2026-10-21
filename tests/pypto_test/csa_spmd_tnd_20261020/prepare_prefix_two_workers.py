# SPDX-License-Identifier: Apache-2.0
"""Use two Hadamard producers and22 Q_B producers for at most eight tokens."""

import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path

from prepare_score_owner_harness import replace_once
from prepare_short_half_ub import append_argument, append_parameter


def parameter(text, name):
    text = append_parameter(text, name, "PREFIX_BRANCH_GROUPS")
    return replace_once(text, "    PREFIX_BRANCH_GROUPS: pl.constexpr = False,\n",
                        "    PREFIX_BRANCH_GROUPS: pl.constexpr = 4,\n")


def main():
    here = Path(__file__).resolve().parent
    source = here / "runs/qb_wide_candidate_v108b"
    target = here / "runs/prefix_two_candidate_v111"
    assert not target.exists()
    originals = {name: (source / name).read_text() for name in ("entry.py", "cube_schedule.py", "policy.py")}
    revised = {}
    text = originals["entry.py"]
    for name in ("_csa_core_group", "decode_csa_spmd"):
        text = parameter(text, name)
    for name in ("_csa_core_group", "cube_schedule"):
        text = append_argument(text, name, "PREFIX_BRANCH_GROUPS")
    revised["entry.py"] = text
    text = parameter(originals["cube_schedule.py"], "cube_schedule")
    text = replace_once(text, "idx_normed, idx_had, wkv, kv_fp32, signal, worker, generation, 4,\n",
                        "idx_normed, idx_had, wkv, kv_fp32, signal, worker, generation, PREFIX_BRANCH_GROUPS,\n")
    revised["cube_schedule.py"] = text
    text = replace_once(originals["policy.py"], "    qb_wide_n: bool = False\n",
                        "    qb_wide_n: bool = False\n    prefix_branch_groups: int = 4\n")
    text = replace_once(text, '            "QB_WIDE_N": self.qb_wide_n,\n',
                        '            "QB_WIDE_N": self.qb_wide_n,\n'
                        '            "PREFIX_BRANCH_GROUPS": self.prefix_branch_groups,\n')
    text = replace_once(text, "SHORT_HALF_SCORE_GROUP_LIMIT = 48\n",
                        "SHORT_HALF_SCORE_GROUP_LIMIT = 48\n"
                        "# Two M256 query-Hadamard tiles cover the existing first M8 token block.\n"
                        "TWO_HADAMARD_MAX_TOKENS = 8\n")
    text = replace_once(text, "        qb_wide_n=tokens < EARLY_GEOMETRY_MIN_TOKENS,\n",
                        "        qb_wide_n=tokens < EARLY_GEOMETRY_MIN_TOKENS,\n"
                        "        prefix_branch_groups=2 if tokens <= TWO_HADAMARD_MAX_TOKENS else 4,\n")
    revised["policy.py"] = text
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patch = []
    for name, text in revised.items():
        ast.parse(text)
        (target / name).write_text(text)
        patch.extend(difflib.unified_diff(originals[name].splitlines(True), text.splitlines(True),
                                         fromfile="a/" + name, tofile="b/" + name))
    for batch, branches in ((1, 2), (8, 4)):
        path = here / "compile_descriptors" / f"qb_wide_h131072_b{batch}_v108b.json"
        data = json.loads(path.read_text())
        data["policy"]["PREFIX_BRANCH_GROUPS"] = branches
        path = here / "compile_descriptors" / f"prefix_two_b{batch}_v111.json"
        path.write_text(json.dumps(data, indent=2) + "\n")
    # Preserve tile identity: only the originally active Hadamard cores0/1 work
    # in this tier. Compact rows cannot exceed the number of new decode tokens.
    checks = []
    for tokens in range(1, 9):
        old_query = [(worker, block) for worker in range(4) for block in range(worker, (tokens + 3) // 4, 4)]
        new_query = [(worker, block) for worker in range(2) for block in range(worker, (tokens + 3) // 4, 2)]
        assert old_query == new_query
        for compact in range(tokens + 1):
            tasks = ((compact + 15) // 16) * 2
            old = [(worker, item) for worker in range(4) for item in range(worker, tasks, 4)]
            new = [(worker, item) for worker in range(2) for item in range(worker, tasks, 2)]
            assert old == new
            checks.append((tokens, compact))
    columns = [block for core in range(2, 24) for block in range(core % 22, 64, 22)]
    assert sorted(columns) == list(range(64))
    assert sorted((core + 24 - 2) % 24 for core in range(24)) == list(range(24))
    report = dict(source=str(target), base=str(source), adopted=False, device_execution=False,
                  changed_files=list(revised), tile_owner_cases=len(checks),
                  hadamard_actual_tiles_and_owners_unchanged=True,
                  qb_columns_exact_once=True, kv_permutation_bijective=True,
                  source_manifest={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
                  next="Full CPU active/off builds; then incremental comparison against validated QB-wide source")
    (here / "prefix_two_v111.json").write_text(json.dumps(report, indent=2) + "\n")
    (here / "prefix_two_v111.patch").write_text("".join(patch))
    print(target)


if __name__ == "__main__":
    main()
