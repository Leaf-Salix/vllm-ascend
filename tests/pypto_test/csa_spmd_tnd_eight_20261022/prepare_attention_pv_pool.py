# SPDX-License-Identifier: Apache-2.0
"""Keep PV N128/K128 math and Right buffering; collect four Acc columns for FIX."""

import ast
import hashlib
import json
import shutil
import textwrap
from pathlib import Path


def transform(source):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "attention_qk_pv_cube_stage")
    start = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "pv_first_right"
    )
    end = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.store"
        and ast.unparse(n.value.args[1]) == "[pv_transfer_row, HEAD_DIM - PV_N_TILE]"
    )
    body = """\
pv_acc_pool = pl.tile.create([H, HEAD_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
pv_first_right = pl.tile.extract(
    qk_l1, pv_l1_row, 0, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right
)
pv_first_slot = pl.tile.slice(pv_acc_pool, [H, PV_N_TILE], [0, 0])
pv_first_result = pl.tile.matmul_acc(pv_first_slot, pv_probability_left, pv_first_right, init_cond=True)
pv_acc_pool = pl.tile.assemble(pv_acc_pool, pv_first_result, [0, 0])
for pv_n in pl.unroll(1, HEAD_DIM // PV_N_TILE):
    pv_next_right = pl.tile.extract(
        qk_l1, pv_l1_row, pv_n * PV_N_TILE, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right
    )
    pv_slot_acc = pl.tile.slice(pv_acc_pool, [H, PV_N_TILE], [0, pv_n * PV_N_TILE])
    pv_result = pl.tile.matmul_acc(pv_slot_acc, pv_probability_left, pv_next_right, init_cond=True)
    pv_acc_pool = pl.tile.assemble(pv_acc_pool, pv_result, [0, pv_n * PV_N_TILE])
pl.store(pv_acc_pool, [pv_transfer_row, 0], pv_transfer)
"""
    lines = source.splitlines(keepends=True)
    lines[start.lineno - 1 : end.end_lineno] = [textwrap.indent(body, " " * start.col_offset)]
    result = "".join(lines)
    modified = ast.parse(result)
    other = lambda t: [
        ast.dump(n, include_attributes=False) for n in t.body if not isinstance(n, ast.FunctionDef) or n.name != fn.name
    ]
    assert other(tree) == other(modified), "Only the Cube helper may change"
    calls = lambda t: [
        ast.dump(n, include_attributes=False)
        for n in ast.walk(t)
        if isinstance(n, ast.Call) and ast.unparse(n.func).startswith("pl.system.sync_")
    ]
    assert calls(tree) == calls(modified), "Cross-core protocol must remain exact"
    return result


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "output_rectangle_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    source["attention_stream_stages.py"] = transform(source["attention_stream_stages.py"])
    target = here / "runs/attention_pv_pool_v1"
    target.mkdir(exist_ok=False)
    for name, value in source.items():
        (target / name).write_text(value)
    shutil.copy2(here / "descriptors/C04_output_rectangle_v1.json", here / "descriptors/C04_attention_pv_pool_v1.json")
    report = dict(
        source=str(target),
        parent="output_rectangle_production_v1.json",
        adopted=False,
        device_execution=False,
        mechanism="Four original M64/N128/K128 PV products occupy M64/N512 Acc column windows; "
        "one contiguous FIX store",
        unchanged="All operands and K order, QK, AIV, KV L1 reuse, 3-slot events, GM payload/allocations and ABI",
        scope="C04 pilot only; full eight-case objective unchanged",
        tradeoff="128KiB Acc may overlap the earlier QK allocation and constrain FIX/M overlap; "
        "actual codegen required",
        historical_distinction="N256 widened Right to 64KiB; 32-head/N512 used K64. "
        "This keeps original N128/K128 and its actual single 32KiB Right buffer",
        initial_right_buffer_assumption_corrected=True,
        reference="ops-transformer@28f4035406a1aea65a8d65c6eb527a85ab84e624: "
        "attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_csa_block_cube.h",
        files={name: hashlib.sha256(value.encode()).hexdigest() for name, value in source.items()},
    )
    with (here / "attention_pv_pool_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
