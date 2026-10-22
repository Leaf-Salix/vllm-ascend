# SPDX-License-Identifier: Apache-2.0
"""Audit the M192-only policy, unchanged enabled inputs, off code and short-case resources."""

import ast
import importlib.util
import json
import runpy
from pathlib import Path

from audit_score_group_guard import content, fingerprint


def main():
    here = Path(__file__).resolve().parent
    source = here / "runs/critical_prefix_v2"
    previous = here / "runs/critical_prefix_v1"
    production = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    changed = [p.name for p in source.glob("*.py") if p.read_bytes() != (previous / p.name).read_bytes()]
    assert changed == ["policy.py"]
    select = runpy.run_path(str(source / "policy.py"))["select_policy"]
    old = runpy.run_path(str(production / "policy.py"))["select_policy"]
    checks = 0
    for tokens in range(1, 258):
        for width in (1, 5, 6):
            for history in (8192, 131072, 262144):
                args = dict(
                    tokens=tokens,
                    requests=(tokens + width - 1) // width,
                    max_query_tokens=width,
                    max_seq_len=history + width,
                )
                actual = select(**args).constexprs()
                assert actual.pop("JOINT_BF16_PREFIX") == (128 < tokens <= 192)
                assert actual.pop("PREFIX_PROJECTION_ROWS") == (192 if 128 < tokens <= 192 else 128)
                assert actual == old(**args).constexprs()
                checks += 1
    c05_new = json.loads((here / "descriptors/C05_critical_prefix_v2.json").read_text())
    c05_old = json.loads((here / "descriptors/C05_critical_prefix_v1.json").read_text())
    assert c05_new == c05_old
    c06 = json.loads((here / "descriptors/C06_critical_prefix_v2.json").read_text())
    assert c06["policy"] == c05_new["policy"]
    # C06 actual device code still must be checked against this enabled class.
    report = dict(
        candidate="critical_prefix_v2",
        device_execution=False,
        policy_checks=checks,
        changed_files=changed,
        C05_source_and_compile_inputs_unchanged=True,
        C06_constexprs_equal_C05=True,
        cases={},
    )
    helper = here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"
    spec = importlib.util.spec_from_file_location("memory_audit", helper)
    memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory)
    for case, compiled, reference in (
        ("C04", "compile_critical_prefix_C04_off_v2", "compile_leaf_guard_production_C04"),
        ("C07", "compile_critical_prefix_C07_v2", "compile_leaf_scale_guard_C07_v1"),
    ):
        root = here / "runs" / compiled
        kernel = root / "kernel"
        original = here / "runs" / reference / "kernel"
        assert json.loads((root / "result.json").read_text())["full_codegen_passed"]
        a, b = (json.loads((p / "compiled_meta.json").read_text()) for p in (kernel, original))
        assert all(a[k] == b[k] for k in ("kernel_abi", "params", "num_return_types"))
        (ir,) = kernel.glob("passes_dump/*after_AllocateMemoryAddr.py")
        tree = ast.parse(ir.read_text())
        limits = dict(mat=524288, vec=188416, left=65536, right=65536, acc=131072)
        resources = {}
        for fn in (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in ("cube_schedule", "vector_schedule")
        ):
            maximum = {}
            for n in ast.walk(fn):
                if isinstance(n, ast.AnnAssign) and (m := memory.memory(n.annotation)):
                    kind = m[0].strip("'").removeprefix("mem_").split("_")[0]
                    if kind == "ddr":
                        continue
                    assert kind in limits
                    maximum[kind] = max(maximum.get(kind, 0), m[1] + m[2])
            assert maximum and all(v <= limits[k] for k, v in maximum.items())
            resources[fn.name] = maximum
        row = dict(
            full_codegen_passed=True, root_and_return_abi_unchanged=True, kernel=str(kernel), resources=resources
        )
        if case == "C04":
            assert content(kernel) == content(original)
            row.update(off_cpp_equal_retained=True, compiled_contract_sha256=fingerprint(content(kernel)))
        report["cases"][case] = row
    with (here / "critical_prefix_v2_codegen.json").open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
