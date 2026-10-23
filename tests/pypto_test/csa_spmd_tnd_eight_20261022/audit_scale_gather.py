# SPDX-License-Identifier: Apache-2.0
"""Check the generated A3 gather variants, memory bounds and unchanged root ABI."""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", choices=("TSEL", "TSELS"), default="TSEL")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = args.compile_dir.resolve()
    assert json.loads((root / "result.json").read_text())["full_codegen_passed"]
    helper = here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"
    spec = importlib.util.spec_from_file_location("existing_memory_audit", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (ir_path,) = (root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
    tree = ast.parse(ir_path.read_text())
    resources, gathers = {}, []
    limits = {"vec": 196608, "mat": 524288, "left": 65536, "right": 65536, "acc": 131072}
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        if fn.name not in ("cube_schedule", "vector_schedule"):
            continue
        maxima = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.AnnAssign) and (memory := module.memory(node.annotation)):
                maxima[memory[0]] = max(maxima.get(memory[0], 0), memory[1] + memory[2])
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.tile.mgather":
                gathers.append(dict(function=fn.name, source=ast.unparse(node.args[0]), expression=ast.unparse(node)))
        for name, maximum in maxima.items():
            kind = name.removeprefix("mem_").split("_")[0]
            if kind in limits:
                assert maximum <= limits[kind], (fn.name, name, maximum)
        resources[fn.name] = maxima
    selected = [g for g in gathers if "idx_scale_rows" in g["source"]]
    assert len(selected) == 1 and "coalesce=0, gather_oob=3" in selected[0]["expression"]
    assert any("indexer_table" in g["source"] and "coalesce=1, gather_oob=3" in g["expression"] for g in gathers)
    cpp = root / "kernel/ptoas/dynamic_csa_full.cpp"
    code = cpp.read_text()
    assert "MGATHER<pto::Coalesce::Elem, pto::GatherOOB::Zero>" in code
    assert "MGATHER<pto::Coalesce::Row, pto::GatherOOB::Zero>" in code
    assert code.count(args.selection + "(") >= 3
    baseline = json.loads((here / "baseline.json").read_text())
    original = Path(baseline["cases"]["C04"]["source_report"]).parent / "build_output"
    (metadata,) = original.glob("decode_csa_spmd_*/compiled_meta.json")
    actual = json.loads((root / "kernel/compiled_meta.json").read_text())["kernel_abi"]
    expected = json.loads(metadata.read_text())["kernel_abi"]
    assert actual == expected
    report = dict(
        passed=True,
        device_execution=False,
        packed_abi_unchanged=True,
        selection=args.selection,
        baseline_meta=str(metadata),
        resources=resources,
        gathers=gathers,
        ir=str(ir_path),
        cpp=str(cpp),
        cpp_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
        limitations="Allocation and emitted instructions only; device precision and performance remain unverified",
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(passed=True, packed_abi_unchanged=True, resources=resources), indent=2))


if __name__ == "__main__":
    main()
