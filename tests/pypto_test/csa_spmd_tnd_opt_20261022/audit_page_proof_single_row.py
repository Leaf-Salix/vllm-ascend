# SPDX-License-Identifier: Apache-2.0
"""Check real eight-row physical reductions with one valid row in IR and C++."""

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "csa_spmd_tnd_20261020"))
from audit_score_wide_k128 import memory  # noqa: E402


def valid_shape(node):
    views = [n for n in ast.walk(node.annotation) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.TileView"]
    assert len(views) == 1
    return next(k.value.elts for k in views[0].keywords if k.arg == "valid_shape")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.compile_dir.resolve()
    assert json.loads((root / "result.json").read_text())["full_codegen_passed"]
    (path,) = (root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
    fn = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "vector_schedule"
    )
    nodes = {n.target.id: n for n in ast.walk(fn) if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    cpp_path = root / "kernel/ptoas/dynamic_csa_full.cpp"
    cpp = cpp_path.read_text()
    cpp = cpp[cpp.index("AICORE void vector_schedule(") :]
    rows = []
    for node in nodes.values():
        if not node.target.id.startswith(("block_ok_inline", "max_difference_inline")):
            continue
        assert ast.literal_eval(node.annotation.slice.elts[0]) == [8, 1]
        assert memory(node.annotation)[2] == 32
        assert [ast.literal_eval(n) for n in valid_shape(node)] == [1, 1]
        source = nodes[node.value.args[0].id]
        assert ast.literal_eval(source.annotation.slice.elts[0]) == [8, 1024]
        assert ast.literal_eval(valid_shape(source)[0]) == 1
        assert ast.unparse(valid_shape(source)[1]).startswith(("live_", "valid_pages_"))
        reads = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and ast.unparse(n.func) == "pl.tile.read"
            and isinstance(n.args[0], ast.Name)
            and n.args[0].id == node.target.id
        ]
        assert len(reads) == 1 and ast.literal_eval(reads[0].args[1]) == [0, 0]

        def emitted_tile(n, shape):
            marker = cpp.index("// pto: %" + n.target.id + "\n")
            expression = r"\b(v\d+) = Tile<TileType::Vec, float, " + shape + r", [^\n]+>\((v\d+), (v\d+)\);"
            match = re.search(expression, cpp[marker : marker + 700])
            assert match
            assert re.search(r"\b" + match.group(2) + r"\s*=\s*1;", cpp)
            return match.group(1), match.group(3)

        result_var, result_cols = emitted_tile(node, "8, 1")
        assert re.search(r"\b" + result_cols + r"\s*=\s*1;", cpp)
        # PTOAS emits set_validshape on the preceding tile rather than creating
        # another C++ object with the alias's IR name. Follow only this verified
        # metadata alias, preserving physical address and effective valid rows.
        emitted_source = source
        if ast.unparse(source.value.func) == "pl.tile.set_validshape":
            emitted_source = nodes[source.value.args[0].id]
            assert memory(emitted_source.annotation) == memory(source.annotation)
            assert ast.literal_eval(valid_shape(emitted_source)[0]) == 1
        input_var, _ = emitted_tile(emitted_source, "8, 1024")
        operation = {"pl.tile.row_min": "TROWMIN", "pl.tile.row_max": "TROWMAX"}[ast.unparse(node.value.func)]
        assert re.search(r"\b" + operation + r"\(" + result_var + ", " + input_var + r", v\d+\);", cpp)
        rows.append(
            dict(
                name=node.target.id,
                operation=operation,
                physical_result=[8, 1],
                valid_result=[1, 1],
                result_region=memory(node.annotation),
                physical_input=[8, 1024],
                valid_input_rows=1,
                valid_columns=ast.unparse(valid_shape(source)[1]),
                only_row_zero_consumed=True,
            )
        )
    assert {r["operation"] for r in rows} == {"TROWMIN", "TROWMAX"}
    maximum = max(r[1] + r[2] for n in nodes.values() if (r := memory(n.annotation)) and r[0].startswith("mem_vec_"))
    assert maximum <= 196608
    previous = root.parent / "compile_output_quant_reuse_b16/kernel/compiled_meta.json"
    assert json.loads((root / "kernel/compiled_meta.json").read_text()) == json.loads(previous.read_text())
    report = dict(
        passed=True,
        device_execution=False,
        ir=str(path),
        ir_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cpp_sha256=hashlib.sha256(cpp_path.read_bytes()).hexdigest(),
        reductions=rows,
        vector_address_max=maximum,
        packed_abi_unchanged=True,
        limitations="Physical alignment and valid-row codegen proof only; page/state device correctness and L1 pending",
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "reductions": len(rows), "vector_address_max": maximum}))


if __name__ == "__main__":
    main()
