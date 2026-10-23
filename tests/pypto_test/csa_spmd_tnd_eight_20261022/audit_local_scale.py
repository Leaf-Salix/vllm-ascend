# SPDX-License-Identifier: Apache-2.0
"""Audit the actual strided load and one-valid-row proof in generated C04 code."""

import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    root = here / "runs/compile_local_scale_C04_v1/kernel"
    assert json.loads((root.parent / "result.json").read_text())["full_codegen_passed"]
    spec = importlib.util.spec_from_file_location(
        "memory_audit", here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (ir,) = (root / "passes_dump").glob("*after_AllocateMemoryAddr.py")
    tree = ast.parse(ir.read_text())
    limits = dict(vec=196608, mat=524288, left=65536, right=65536, acc=131072)
    resources, packed, maximum = {}, [], []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name not in ("cube_schedule", "vector_schedule"):
            continue
        maxima = {}
        for node in ast.walk(fn):
            if not isinstance(node, ast.AnnAssign):
                continue
            if memory := module.memory(node.annotation):
                kind = memory[0].removeprefix("mem_").split("_")[0]
                if kind in limits:
                    maxima[kind] = max(maxima.get(kind, 0), memory[1] + memory[2])
            if node.target.id.startswith("packed_inline"):
                assert ast.unparse(node.value.func) == "pl.tile.load"
                assert ast.literal_eval(node.value.args[2]) == [32, 64]
                assert ast.literal_eval(node.value.args[3]) == [32, 64]
                packed.append(ast.unparse(node))
            if node.target.id.startswith("maximum_inline"):
                assert "valid_shape=[1, 1]" in ast.unparse(node.annotation)
                assert ast.unparse(node.value.func) == "pl.tile.row_max"
                maximum.append(ast.unparse(node))
        assert all(maxima[k] <= limits[k] for k in maxima)
        resources[fn.name] = maxima
    assert len(packed) == len(maximum) == 1
    baseline = json.loads((here / "baseline.json").read_text())
    original = Path(baseline["cases"]["C04"]["source_report"]).parent / "build_output"
    (metadata,) = original.glob("decode_csa_spmd_*/compiled_meta.json")
    assert (
        json.loads((root / "compiled_meta.json").read_text())["kernel_abi"]
        == json.loads(metadata.read_text())["kernel_abi"]
    )
    cpp = root / "ptoas/dynamic_csa_full.cpp"
    code = cpp.read_text()
    (destination,) = re.findall(r"// pto: %packed_inline[^\n]*\n\s+(v\d+) = Tile<TileType::Vec, int8_t, 32, 64,", code)
    assert code.count(f"TLOAD({destination},") == 1
    assert "GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 64>" in code
    report = dict(
        passed=True,
        device_execution=False,
        packed_abi_unchanged=True,
        resources=resources,
        local_proof=maximum,
        strided_load=packed,
        generated_tload_destination=destination,
        cpp=str(cpp),
        cpp_sha256=hashlib.sha256(code.encode()).hexdigest(),
        limitations=(
            "Static lowering only; ISA TLOAD preserves dynamic row stride, "
            "but device behavior and benefit remain unverified"
        ),
    )
    with (here / "local_scale_v1_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=True, resources=resources, strided_loads=1)))


if __name__ == "__main__":
    main()
