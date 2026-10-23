# SPDX-License-Identifier: Apache-2.0
"""Prefetch old PV payload after KV gathering, before current softmax."""

import argparse
import ast
import hashlib
import json
import shutil
import textwrap
from pathlib import Path

from attention_pv_prefetch_protocol import verify_all

PREFETCH = """\
pv_left_hold = pl.tile.create([H // 2, HEAD_DIM // 2], dtype=pl.FP32)
pv_right_hold = pl.tile.create([H // 2, HEAD_DIM // 2], dtype=pl.FP32)
if qk_tick >= QK_PRE_LAUNCH + 1:
    pf_pv_work = qk_tick - QK_PRE_LAUNCH - 1
    pf_pv_sb = pf_pv_work % SPARSE_BLOCKS
    pf_pv_t = qk_core + (pf_pv_work // SPARSE_BLOCKS) * CORE_GROUPS
    if pl.read(valid_block_mask, [pf_pv_t, pf_pv_sb]) > 0:
        pf_pv_slot = qk_core * QK_TRANSFER_SLOTS + pf_pv_work % QK_TRANSFER_SLOTS
        pf_pv_row = pf_pv_slot * H
        pl.system.sync_wait(EVENT_BASE + QK_PV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
        pf_pv_left = pl.load(pv_transfer, [pf_pv_row + qk_lane_head, 0], [H // 2, HEAD_DIM // 2])
        pf_pv_right = pl.load(
            pv_transfer, [pf_pv_row + qk_lane_head, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2]
        )
        pv_left_valid, pv_right_valid = pl.yield_(pf_pv_left, pf_pv_right)
    else:
        pv_left_valid, pv_right_valid = pl.yield_(pv_left_hold, pv_right_hold)
    prefetched_pv_left, prefetched_pv_right = pl.yield_(pv_left_valid, pv_right_valid)
else:
    prefetched_pv_left, prefetched_pv_right = pl.yield_(pv_left_hold, pv_right_hold)
"""


def transform(source, halves=2):
    assert halves in (1, 2)
    prefetch = PREFETCH
    if halves == 1:
        prefetch = prefetch.replace("pv_right_hold = pl.tile.create([H // 2, HEAD_DIM // 2], dtype=pl.FP32)\n", "")
        prefetch = prefetch.replace(
            "        pf_pv_right = pl.load(\n"
            "            pv_transfer, [pf_pv_row + qk_lane_head, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2]\n"
            "        )\n",
            "",
        )
        for old, new in (
            ("pv_left_valid, pv_right_valid", "pv_left_valid"),
            ("prefetched_pv_left, prefetched_pv_right", "prefetched_pv_left"),
            ("pl.yield_(pf_pv_left, pf_pv_right)", "pl.yield_(pf_pv_left)"),
            ("pl.yield_(pv_left_hold, pv_right_hold)", "pl.yield_(pv_left_hold)"),
            ("pl.yield_(pv_left_valid, pv_right_valid)", "pl.yield_(pv_left_valid)"),
        ):
            prefetch = prefetch.replace(old, new)
    tree = ast.parse(source)
    name = "attention_qk_pv_vector_lane_stage"
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    gather = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == "qk_tick < qk_block_count")
    loads = {
        n.targets[0].id: n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("pv_left", "pv_right")
    }
    assert set(loads) == {"pv_left", "pv_right"}
    if halves == 1:
        del loads["pv_right"]
    waits = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.system.sync_wait"
        and ast.unparse(n.value.args[0]) == "EVENT_BASE + QK_PV_READY_EVENT"
    ]
    assert len(waits) == 1
    wait = waits[0]
    changes = [(gather.end_lineno, gather.end_lineno, textwrap.indent(prefetch, " " * gather.col_offset))]
    changes.append((wait.lineno - 1, wait.end_lineno, ""))
    for key, node in loads.items():
        changes.append((node.lineno - 1, node.end_lineno, " " * node.col_offset + f"{key} = prefetched_{key}\n"))
    lines = source.splitlines(keepends=True)
    for start, end, text in sorted(changes, reverse=True):
        lines[start:end] = [text] if text else []
    result = "".join(lines)
    after = ast.parse(result)
    changed = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == name)
    changed_loop = next(n for n in changed.body if isinstance(n, ast.For))
    inserted = loop.body.index(gather) + 1
    del changed_loop.body[inserted : inserted + len(ast.parse(prefetch).body)]
    for node in ast.walk(changed):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in loads:
            node.value = loads[node.targets[0].id].value
    old_valid = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and wait in n.body)
    new_valid = next(
        n for n in ast.walk(changed) if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(old_valid.test)
    )
    new_valid.body.insert(old_valid.body.index(wait), wait)
    assert ast.dump(tree) == ast.dump(after), "Restoring only the two loads and wait must recover all original code"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--halves", type=int, choices=(1, 2), default=2)
    parser.add_argument("--revision", type=int, default=1)
    args = parser.parse_args()
    assert args.revision > 0
    here = Path(__file__).resolve().parent
    protocol = verify_all()
    parent = json.loads((here / "output_rectangle_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    source["attention_stream_stages.py"] = transform(source["attention_stream_stages.py"], args.halves)
    name = f"attention_pv_prefetch_v{args.revision}"
    target = here / "runs" / name
    target.mkdir(exist_ok=False)
    for filename, text in source.items():
        (target / filename).write_text(text)
    shutil.copy2(here / "descriptors/C04_output_rectangle_v1.json", here / f"descriptors/C04_{name}.json")
    report = dict(
        source=str(target),
        parent="output_rectangle_production_v1.json",
        adopted=False,
        device_execution=False,
        mechanism="Acquire old PV after current KV stores, prefetch payload, then execute unchanged softmax and merge",
        unchanged="Cube, all FP32 math/order, original KV/Score/Probability notifications, three GM slots, root ABI",
        halves=args.halves,
        added_live_payload_bytes=32768 * args.halves,
        protocol=protocol,
        source_AST_recovery_passed=True,
        reference="CANN recipes buffer ownership and A3 SFA PV lifetimes; early-PV order is this experiment",
        pending="CPU memory planning and actual DMA/Vector overlap audit; no performance assumption",
        files={filename: hashlib.sha256(text.encode()).hexdigest() for filename, text in source.items()},
    )
    with (here / f"{name}.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"source": str(target), "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
