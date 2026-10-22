# SPDX-License-Identifier: Apache-2.0
"""Audit the resident weight load, loop lifetime, tail retirement and ABI."""

import argparse
import ast
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
    body = ast.Module(body=active.body, type_ignores=[])
    loads = [
        n
        for n in ast.walk(body)
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.tile.load"
        and ast.unparse(n.value.args[0]).startswith("q_head_weights_")
    ]
    assert len(loads) == 1 and loads[0] in active.body
    owner = loads[0]
    region = memory(owner.annotation)
    assert region[2] == 16384
    loops = [n for n in active.body if isinstance(n, ast.For)]
    assert len(loops) == 1 and active.body.index(owner) < active.body.index(loops[0])
    main_loop = loops[0]
    after = active.body[active.body.index(main_loop) + 1 :]
    tails = [n for n in after if isinstance(n, ast.If)]
    assert len(tails) == 1 and ast.unparse(tails[0].test) == "unroll_rem == 1"
    tail = tails[0]
    assert not any(isinstance(n, ast.For) for n in ast.walk(tail))
    metadata_ops = {
        "pl.tile.create",
        "pl.tile.slice",
        "pl.tile.reshape",
        "pl.tile.reinterpret_view",
        "pl.tile.set_validshape",
    }

    def writes(node):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.value, ast.Call):
            return False
        r = memory(node.annotation)
        # After AllocateMemoryAddr, Vec offsets are physical UB addresses.
        # Different MemRef allocation names do not create independent UB banks.
        physical_overlap = bool(
            r and r[0].startswith("mem_vec_") and r[1] < region[1] + region[2] and region[1] < r[1] + r[2]
        )
        return physical_overlap and ast.unparse(node.value.func) not in metadata_ops

    assert not any(writes(n) for n in ast.walk(main_loop)), "Resident weight overwritten in the repeated loop"
    loop_aliases = []
    for node in ast.walk(main_loop):
        if isinstance(node, ast.AnnAssign):
            r = memory(node.annotation)
            if r and r[0].startswith("mem_vec_") and r[1] < region[1] + region[2] and region[1] < r[1] + r[2]:
                loop_aliases.append(node)
    # Also exclude scratch allocations whose writes occur through a tmp operand.
    assert len(loop_aliases) == 2 and all(
        isinstance(n.value, ast.Name) and n.value.id == owner.target.id for n in loop_aliases
    )
    aliases = {owner.target.id}
    for n in ast.walk(body):
        if isinstance(n, ast.AnnAssign) and isinstance(n.value, ast.Name) and n.value.id in aliases:
            assert memory(n.annotation) == region
            aliases.add(n.target.id)
    clobbers = sorted((n for n in ast.walk(tail) if writes(n)), key=lambda n: n.lineno)
    assert clobbers
    first = clobbers[0]
    assert ast.unparse(first.value.func) == "pl.tile.mul" and first.value.args[1].id in aliases
    readers = [
        n
        for n in ast.walk(tail)
        if isinstance(n, ast.Call)
        and any(isinstance(v, ast.Name) and v.id in aliases for arg in n.args for v in ast.walk(arg))
    ]
    assert readers and max(n.lineno for n in readers) == first.lineno
    # Only the final single-work epilogue may overwrite the read-only scale,
    # in-place with its final elementwise multiply; there is no next iteration.
    cpp = (root / "kernel/ptoas/dynamic_csa_full.cpp").read_text()
    marker = cpp.index("// pto: %" + owner.target.id)
    match = re.search(r"\b(v\d+) = Tile<TileType::Vec", cpp[marker : marker + 1500])
    assert match
    variable = match.group(1)
    transfers = list(re.finditer(r"\bTLOAD\(" + variable + r", [^;]+;", cpp))
    assert len(transfers) == 1
    load = transfers[0]
    query_loads = [
        n
        for n in ast.walk(body)
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.tile.load"
        and ast.unparse(n.value.args[0]).startswith("q_head_input_")
    ]
    assert len(query_loads) == 3  # Two repeated lanes and the single-work epilogue.
    # PTOAS coalesces some unrolled SSA name hints. Check the enclosing C++
    # block and its INT32 8x512 query-input tile type, not missing _1/_2 comments.
    # Keep character positions when removing comments and string literals.
    sanitized = re.sub(
        r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
        lambda m: "".join("\n" if c == "\n" else " " for c in m.group()),
        cpp,
    )
    stack = []
    for index, char in enumerate(sanitized[: load.start()]):
        if char == "{":
            stack.append(index)
        elif char == "}":
            stack.pop()
    block_start = stack[-1]
    depth, block_end = 1, None
    for index in range(block_start + 1, len(sanitized)):
        depth += (sanitized[index] == "{") - (sanitized[index] == "}")
        if depth == 0:
            block_end = index
            break
    assert block_end is not None
    block = cpp[block_start:block_end]
    query_vars = set(re.findall(r"\b(v\d+) = Tile<TileType::Vec, int32_t, 8, 512\b", block))
    assert query_vars
    operations = [op for op in re.finditer(r"\bTLOAD\((v\d+), [^;]+;", block) if op.group(1) in query_vars]
    assert len(operations) >= 3 and all(block_start + op.start() > load.end() for op in operations)
    cpp_query_loads = len(operations)
    # The owner address previously held the row-index preparation. Its final
    # Vector read is retired before the MTE2 overwrite.
    prefix = cpp[marker - 2500 : load.start()]
    sets = set(re.findall(r"set_flag\(PIPE_V, PIPE_MTE2, (EVENT_ID\d+)\)", prefix))
    waits = set(re.findall(r"wait_flag\(PIPE_V, PIPE_MTE2, (EVENT_ID\d+)\)", prefix))
    assert sets & waits
    all_vec = [memory(n.annotation) for n in ast.walk(fn) if isinstance(n, ast.AnnAssign)]
    maximum = max(r[1] + r[2] for r in all_vec if r and r[0].startswith("mem_vec_"))
    assert maximum <= 196608
    parent = root.parent / "compile_query_contiguous_heads_b16_v4/kernel/compiled_meta.json"
    assert json.loads((root / "kernel/compiled_meta.json").read_text()) == json.loads(parent.read_text())
    report = dict(
        passed=True,
        device_execution=False,
        ir=str(path),
        resident_region=region,
        static_weight_load_count=1,
        load_outside_repeated_loop=True,
        loop_readonly=True,
        physical_overlap_ignores_allocation_names=True,
        loop_overlap_is_only_two_read_aliases=True,
        aliases=sorted(aliases),
        final_tail_reuse_line=first.lineno,
        tail_reuse="Only after repeated loop; final MUL reads and overwrites the scale, with no later weight reader",
        pre_load_reuse_event_ids=sorted(sets & waits),
        generated_query_load_sites_after_weight=cpp_query_loads,
        vector_address_max=maximum,
        packed_abi_unchanged=True,
        readiness=(
            "Weight TLOAD precedes the Query input loads; "
            "the existing MTE2-to-Vector acquisitions cover prior MTE2 work"
        ),
        limitations="Static generated-code/lifetime evidence; device exactness and complete L1 still required",
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("passed", "resident_region", "static_weight_load_count", "vector_address_max")}
        )
    )


if __name__ == "__main__":
    main()
