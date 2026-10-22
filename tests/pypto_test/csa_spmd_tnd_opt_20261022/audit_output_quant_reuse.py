# SPDX-License-Identifier: Apache-2.0
"""Verify original O_A values survive abs/amax and scratch until quantization."""

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "csa_spmd_tnd_20261020"))
from audit_score_wide_k128 import memory  # noqa: E402


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
    nodes = sorted((n for n in ast.walk(fn) if isinstance(n, ast.AnnAssign)), key=lambda n: n.lineno)
    loads = [
        n
        for n in nodes
        if isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.tile.load"
        and ast.unparse(n.value.args[0]).startswith("o_r_")
    ]
    assert len(loads) == 3 and all(n.target.id.startswith("oc_amax_") for n in loads)
    lifetimes = []
    for load in loads:
        region = memory(load.annotation)
        aliases = {load.target.id}
        for node in nodes:
            if isinstance(node.value, ast.Name) and node.value.id in aliases:
                assert memory(node.annotation) == region
                aliases.add(node.target.id)
        uses = [
            n
            for n in nodes
            if isinstance(n.value, ast.Call)
            and any(isinstance(v, ast.Name) and v.id in aliases for arg in n.value.args for v in ast.walk(arg))
        ]
        assert [ast.unparse(n.value.func) for n in uses] == ["pl.tile.abs", "pl.tile.row_expand_mul"]
        absolute, consumer = uses
        overlapping = []
        for node in nodes:
            address = memory(node.annotation)
            if (
                load.lineno < node.lineno < consumer.lineno
                and address
                and address[0].startswith("mem_vec_")
                and address[1] < region[1] + region[2]
                and region[1] < address[1] + address[2]
            ):
                overlapping.append(node)
        assert len(overlapping) == 1
        assert isinstance(overlapping[0].value, ast.Name) and overlapping[0].value.id == load.target.id
        lifetimes.append(
            dict(
                load_line=load.lineno,
                input_region=region,
                absolute_region=memory(absolute.annotation),
                final_consumer_line=consumer.lineno,
                final_consumer_region=memory(consumer.annotation),
                readonly_until_final_multiply=True,
                only_overlap_is_read_alias=True,
            )
        )
    cpp_path = root / "kernel/ptoas/dynamic_csa_full.cpp"
    cpp = cpp_path.read_text()
    marker = cpp.index("// pto: %" + loads[0].target.id + "\n")
    match = re.search(r"\b(v\d+) = Tile<TileType::Vec, float, 8, 1024", cpp[marker : marker + 700])
    assert match
    variable = match.group(1)
    alias = next(n for n in nodes if isinstance(n.value, ast.Name) and n.value.id == loads[0].target.id)
    alias_marker = cpp.index("// pto: %" + alias.target.id + "\n")
    alias_match = re.search(r"\b(v\d+) = Tile<TileType::Vec, float, 8, 1024", cpp[alias_marker : alias_marker + 700])
    assert alias_match
    alias_variable = alias_match.group(1)

    def assigned_address(start, name):
        address = re.search(r"(v\d+) = \(uint64_t\) (v\d+);\s*TASSIGN\(" + name + r", \1\);", cpp[start : start + 700])
        assert address
        return address.group(2)

    assert assigned_address(marker, variable) == assigned_address(alias_marker, alias_variable)
    transfer = re.search(r"\bTLOAD\(" + variable + r", [^;]+;", cpp[marker:])
    absolute = re.search(r"\bTABS\((v\d+), " + variable + r"\);", cpp[marker:])
    multiply = re.search(r"\bTROWEXPANDMUL\(v\d+, " + alias_variable + r", v\d+\);", cpp[marker:])
    assert transfer and absolute and multiply and transfer.end() < absolute.start() < multiply.start()
    assert absolute.group(1) != variable
    maximum = max(r[1] + r[2] for n in nodes if (r := memory(n.annotation)) and r[0].startswith("mem_vec_"))
    assert maximum <= 196608
    reference = root.parent / "compile_query_resident_swap_b16/kernel/compiled_meta.json"
    assert json.loads((root / "kernel/compiled_meta.json").read_text()) == json.loads(reference.read_text())
    report = dict(
        passed=True,
        device_execution=False,
        ir=str(path),
        ir_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cpp_sha256=hashlib.sha256(cpp_path.read_bytes()).hexdigest(),
        output_quant_input_load_sites=3,
        previous_load_sites=6,
        lifetimes=lifetimes,
        physical_overlap_ignores_allocation_names=True,
        first_emitted_load_abs_multiply_checked=True,
        vector_address_max=maximum,
        packed_abi_unchanged=True,
        limitations="Static lifetime evidence; exact NPU result and complete L1 remain required",
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "output_quant_input_load_sites", "vector_address_max")}))


if __name__ == "__main__":
    main()
