# SPDX-License-Identifier: Apache-2.0
"""Audit the physical lifetime and emitted consumers of the resident Query index."""

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
    active = next(
        n
        for n in fn.body
        if isinstance(n, ast.If)
        and any(
            isinstance(a, ast.AnnAssign)
            and isinstance(a.target, ast.Name)
            and a.target.id.startswith("resident_head_scale_")
            for a in ast.walk(n)
        )
    )
    loop = next(n for n in active.body if isinstance(n, ast.For))
    guard = next(n for n in active.body if isinstance(n, ast.If) and ast.unparse(n.test).startswith("0 < tokens_"))
    assert active.body.index(guard) < active.body.index(loop)
    load = next(
        n
        for n in guard.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.tile.load"
    )
    assert ast.unparse(load.value.args[0]).startswith("rope_swap_")
    assert ast.literal_eval(load.value.args[1]) == [0, 0]
    assert not any(isinstance(n, ast.For) for n in ast.walk(guard))
    owner = next(
        n
        for n in guard.body
        if isinstance(n, ast.AnnAssign)
        and n.target.id.startswith("q_abs_swap_")
        and ast.unparse(n.value.func) == "pl.yield_"
    )
    region = memory(owner.annotation)
    assert region[2] == 2048
    computation = next(n for n in guard.body if isinstance(n, ast.AnnAssign) and n.target.id == owner.value.args[0].id)
    assert ast.unparse(computation.value.func) == "pl.tile.cast"
    assert memory(computation.annotation) == region
    remainder = ast.Module(body=active.body[active.body.index(loop) :], type_ignores=[])
    for node in ast.walk(remainder):
        if isinstance(node, ast.AnnAssign):
            address = memory(node.annotation)
            assert not (
                address
                and address[0].startswith("mem_vec_")
                and address[1] < region[1] + region[2]
                and region[1] < address[1] + address[2]
            ), f"Resident index physical UB region reused: {ast.unparse(node)}"
    gathers = [n for n in ast.walk(remainder) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.tile.gather"]
    assert len(gathers) == 3  # Two repeated work items plus one final work item.
    assert all(n.args[1].id == owner.target.id for n in gathers)
    assert len({n.args[2].id for n in gathers}) == 3  # Scratch remains per work.
    assert not any(
        isinstance(n, ast.Call)
        and ast.unparse(n.func) == "pl.tile.load"
        and ast.unparse(n.args[0]).startswith("rope_swap_")
        for n in ast.walk(remainder)
    )
    cpp_path = root / "kernel/ptoas/dynamic_csa_full.cpp"
    cpp = cpp_path.read_text()

    def cpp_tile(name):
        marker = cpp.index("// pto: %" + name + "\n")
        match = re.search(r"\b(v\d+) = Tile<TileType::Vec", cpp[marker : marker + 700])
        assert match
        return match.group(1)

    owner_var, load_var = cpp_tile(owner.target.id), cpp_tile(load.target.id)
    transfers = list(re.finditer(r"\bTLOAD\(" + load_var + r", [^;]+;", cpp))
    consumers = list(re.finditer(r"\bTGATHER\(v\d+, v\d+, " + owner_var + r", v\d+\);", cpp))
    assert len(transfers) == 1 and len(consumers) == 3
    assert transfers[0].end() < consumers[0].start()
    convert_var = cpp_tile(computation.target.id)
    convert = re.search(r"\bTCVT\(" + convert_var + r", [^;]+;", cpp)
    assert convert and transfers[0].end() < convert.start() < consumers[0].start()
    prefix = cpp[transfers[0].end() : convert.start()]
    sets = set(re.findall(r"set_flag\(PIPE_MTE2, PIPE_V, (EVENT_ID\d+)\)", prefix))
    waits = set(re.findall(r"wait_flag\(PIPE_MTE2, PIPE_V, (EVENT_ID\d+)\)", prefix))
    assert sets & waits
    all_vec = [memory(n.annotation) for n in ast.walk(fn) if isinstance(n, ast.AnnAssign)]
    maximum = max(r[1] + r[2] for r in all_vec if r and r[0].startswith("mem_vec_"))
    assert maximum <= 196608
    parent = root.parent / "compile_query_resident_weights_b16/kernel/compiled_meta.json"
    assert json.loads((root / "kernel/compiled_meta.json").read_text()) == json.loads(parent.read_text())
    report = dict(
        passed=True,
        device_execution=False,
        ir=str(path),
        ir_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cpp_sha256=hashlib.sha256(cpp_path.read_bytes()).hexdigest(),
        resident_index_region=region,
        static_load_sites=1,
        source_row=0,
        new_load_guarded_for_zero_tokens=True,
        repeated_and_final_work_region_readonly=True,
        physical_overlap_ignores_allocation_names=True,
        gather_sites=len(consumers),
        writable_gather_scratch_remains_per_work=True,
        generated_mte2_to_vector_events=sorted(sets & waits),
        vector_address_max=maximum,
        packed_abi_unchanged=True,
        isa_readonly_evidence=(
            "Pinned a2a3/TGather.hpp float/int32 path vmuls writes dstPtr from src1Ptr, "
            "then vgather writes dstPtr; the source index is not modified"
        ),
        limitations="Static source/IR/C++ evidence; device correctness and complete paired L1 required",
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("passed", "resident_index_region", "static_load_sites", "vector_address_max")}
        )
    )


if __name__ == "__main__":
    main()
