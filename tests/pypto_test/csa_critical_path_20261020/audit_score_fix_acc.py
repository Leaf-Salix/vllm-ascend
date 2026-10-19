# SPDX-License-Identifier: Apache-2.0
"""核对提前FIX是否保留独立Acc及没有被同一完成事件重新串行；不替代真机。"""
import argparse
import ast
import json
from pathlib import Path
import re


def audit(kernel):
    ir_path = next((kernel / "passes_dump").glob("*AllocateMemoryAddr*"))
    tree = ast.parse(ir_path.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cube_schedule")
    qk, ws = [], []
    for node in ast.walk(function):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name) or not isinstance(node.value, ast.Call):
            continue
        name = node.target.id
        op = ast.unparse(node.value.func)
        if op not in ("pl.tile.matmul", "pl.tile.matmul_acc"):
            continue
        family = qk if name.startswith("buf_scores_inline") else ws if name.startswith(("buf_reduced_pair", "buf_last_scores")) else None
        if family is None:
            continue
        match = re.findall(r"MemRef\(([^,]+), pl.const\((\d+), pl.INT64\), (\d+)\)", ast.unparse(node.annotation))
        assert len(match) == 1
        base, offset, size = match[0]
        family.append(dict(name=name, line=node.lineno, base=base, offset=int(offset), bytes=int(size), op=op))
    assert len(qk) == 16 and len(ws) == 16, (len(qk), len(ws))
    overlap = any(a["offset"] < b["offset"] + b["bytes"] and b["offset"] < a["offset"] + a["bytes"] for a in qk for b in ws)
    assert all(a["offset"] + a["bytes"] <= 131072 for a in qk + ws)
    cpp_path = kernel / "kernels/aic/cube_schedule.cpp"
    lines = cpp_path.read_text().splitlines()
    qk_names, ws_names = set(), set()
    for line in lines:
        match = re.search(r"Tile<TileType::Acc, (int32_t|float), (384|16), 64,.*> (v\d+);", line)
        if match:
            dtype, rows, name = match.groups()
            if dtype == "int32_t" and rows == "384":
                qk_names.add(name)
            if dtype == "float" and rows == "16":
                ws_names.add(name)
    qk_calls, ws_calls = [], []
    for i, line in enumerate(lines):
        match = re.search(r"TMATMUL(?:_ACC)?\((v\d+),", line)
        if match and match[1] in qk_names:
            qk_calls.append(i)
        if match and match[1] in ws_names:
            ws_calls.append(i)
    assert len(qk_calls) == 16, len(qk_calls)
    # 同尺寸的其它前置投影可能也是16×64；仅取Score第一份QK之后的WS。
    ws_calls = [i for i in ws_calls if i > qk_calls[0]]
    assert len(ws_calls) == 16, len(ws_calls)
    checks = []
    for panel in range(1, 16):
        start, end = qk_calls[panel], ws_calls[panel - 1]
        assert start < end
        insert = [i for i in range(start, end) if "TINSERT<" in lines[i]]
        assert len(insert) == 1
        gap = "\n".join(lines[insert[0] + 1:end])
        sets = re.findall(r"set_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)", gap)
        waits = re.findall(r"wait_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)", gap)
        checks.append(dict(panel=panel, qk_line=start + 1, fix_line=insert[0] + 1,
                           ws_line=end + 1, current_fix_release_events=sets,
                           ws_acquire_events=waits, same_fix_event_blocks_ws=bool(set(sets) & set(waits))))
    return dict(ir=str(ir_path), cpp=str(cpp_path), qk=qk, ws=ws,
                acc_ranges_disjoint=not overlap, panel_events=checks,
                current_qk_fix_blocks_ws_panels=sum(x["same_fix_event_blocks_ws"] for x in checks),
                mechanism_supported=not overlap and not any(x["same_fix_event_blocks_ws"] for x in checks),
                device_validated=False, performance_eligible=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.kernel)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print({key: result[key] for key in ("acc_ranges_disjoint", "current_qk_fix_blocks_ws_panels", "mechanism_supported")})
