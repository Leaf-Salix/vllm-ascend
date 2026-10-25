# SPDX-License-Identifier: Apache-2.0
"""组合两项独立正向候选，保持原冻结基线，组合收益须重新测量。"""

import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path

from prepare_fused_signal_init import patch_entry, patch_schedule


def main():
    here = Path(__file__).resolve().parent
    parent = here / "candidates/fused_coefficients_current_v2"
    target = here / "candidates/init_coeff_combined_v1"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patches = []
    for filename, transform in (
        ("entry.py", patch_entry),
        ("cube_schedule.py", lambda text: patch_schedule(text, "cube")),
        ("vector_schedule.py", lambda text: patch_schedule(text, "vector")),
    ):
        path = target / filename
        before = path.read_text()
        after = transform(before)
        ast.parse(after)
        path.write_text(after)
        patches.extend(
            difflib.unified_diff(
                before.splitlines(True), after.splitlines(True), fromfile="a/" + filename, tofile="b/" + filename
            )
        )
    for role, marker in (
        ("cube", "    tokens = pl.tensor.dim(x_fp32, 0)\n"),
        ("vector", "    tokens = pl.tensor.dim(x, 0)\n"),
    ):
        before = (parent / f"{role}_schedule.py").read_text()
        after = (target / f"{role}_schedule.py").read_text()
        assert before[before.index(marker) :] == after[after.index(marker) :]
    before_tree = ast.parse((parent / "entry.py").read_text())
    after_tree = ast.parse((target / "entry.py").read_text())
    entries = {}
    for name in ("decode_csa_spmd", "decode_csa_spmd_debug"):
        before = next(n for n in before_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        after = next(n for n in after_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(before.args) == ast.dump(after.args)
        allocations = lambda node: [
            ast.dump(c)
            for c in ast.walk(node)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "create_tensor"
        ]
        assert allocations(before) == allocations(after)
        group_calls = [
            n
            for n in ast.walk(after)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_csa_core_group"
        ]
        assert len(group_calls) == 2
        for call in group_calls:
            keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
            assert keywords["FUSED_INDEXER_COEFF"] == "FUSED_INDEXER_COEFF"
            assert keywords["FUSED_SIGNAL_INIT"] in ("True", "False")
        entries[name] = {
            "signature_same_as_coeff_parent": True,
            "GM_allocations_same_as_coeff_parent": True,
            "both_branches_forward_coefficient_strategy": True,
        }
    for path in parent.glob("*.py"):
        if path.name not in ("entry.py", "cube_schedule.py", "vector_schedule.py"):
            assert path.read_bytes() == (target / path.name).read_bytes()
    (here / "init_coeff_combined_v1.patch").write_text("".join(patches))
    report = {
        "source": str(target),
        "parents": ["fused_coefficients_current_v2", "fused_signal_init_v3"],
        "reference_algorithm": "frozen_current_internal_workspace_20261025",
        "score_cross_step_included": False,
        "source_contracts": entries,
        "business_math_and_sync_suffixes_exact_to_coeff_parent": True,
        "coefficient_helper_and_policy_exact_to_coeff_parent": True,
        "init_transform": "原fused_signal_init_v3同一变换函数；相同24组guard、2D alias、clear/barrier",
        "status": "combined_CPU_gate_pending",
        "adopted": False,
        "device_execution": False,
        "performance_claim": "不能将两项独立收益相加，组合与原冻结基线重新测量",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.glob("*.py"))},
    }
    (here / "init_coeff_combined_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
