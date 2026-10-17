# SPDX-License-Identifier: Apache-2.0
"""N64固定L1环的实际分配/指令审计，不将编译成功当作设备性能。"""

import argparse
import json
import re
from pathlib import Path

from qk_phase_clock import close_brace


def stage(source):
    source = source.split("AICORE void csa_static_aiv(", 1)[0]
    end = source.index("// pto: %done_16__tile\n")
    opening = list(re.finditer(r"^  if \([^\n]+\) \{", source[:end], re.M))[-1]
    return source[opening.start():close_brace(source, opening.end() - 1) + 1]


def audit(directory):
    directory = Path(directory)
    ir = (directory / "csa_static_group.pto").read_text().split("func.func @csa_static_aiv")[0]
    cpp = stage((directory / "csa_static_group.cpp").read_text())
    constants = {n: int(v) for n, v in re.findall(r"(%\w+) = arith.constant (-?\d+) : i64", ir)}
    tiles = {}
    for line in ir.splitlines():
        match = re.search(r"(%stage_16_\w+) = pto.alloc_tile addr = (%\w+).*"
                          r"!pto.tile_buf<loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", line)
        if match:
            name, addr, memory, dtype, rows, cols = match.groups()
            tiles[name] = {"address": constants[addr], "memory": memory,
                           "dtype": dtype, "shape": [int(rows), int(cols)]}

    def select(memory, dtype, shape):
        return {name: v for name, v in tiles.items()
                if (v["memory"], v["dtype"], v["shape"]) == (memory, dtype, shape)}

    bad_moves = [line.strip() for line in ir.splitlines() if "pto.tmov " in line
                 and "%stage_16_" in line
                 and (line.count("loc=mat") == 2 or line.count("loc=right") == 2)]
    if bad_moves:
        raise ValueError("生成了同层Mat/Right搬运")
    key_pool, score_pool = select("mat", "i8", [192, 128]), select("mat", "f16", [384, 128])
    if not key_pool or not score_pool:
        raise ValueError("未发现固定Key/score L1池")
    key_addresses = {v["address"] for v in key_pool.values()}
    score_addresses = {v["address"] for v in score_pool.values()}
    if len(key_addresses) != 1 or len(score_addresses) != 1:
        raise ValueError("L1池在循环中未原位复用")
    ka, sa = next(iter(key_addresses)), next(iter(score_addresses))
    if not (ka + 24576 <= sa or sa + 98304 <= ka):
        raise ValueError("两种L1池重叠")
    qk, ws = select("acc", "i32", [384, 64]), select("acc", "f32", [16, 64])
    qk_outputs, ws_outputs = [], []
    for line in ir.splitlines():
        if "pto.tmatmul ins(" in line:
            name = re.search(r"outs\((%\w+)", line)[1]
            if name in qk:
                qk_outputs.append(name)
            if name in ws:
                ws_outputs.append(name)
    if len(qk_outputs) != 3 or len(ws_outputs) != 3:
        raise ValueError("应为首块/循环/末块各三个QK与WS静态位点")
    for q, w in ((qk_outputs[1], ws_outputs[0]), (qk_outputs[2], ws_outputs[1])):
        qa, wa = qk[q]["address"], ws[w]["address"]
        if not (qa + 98304 <= wa or wa + 4096 <= qa):
            raise ValueError("同轮QK与前轮WS累加区重叠")
    inserts = [line for line in ir.splitlines() if "pto.tinsert ins(" in line and "%stage_16_" in line]
    if len(inserts) != 3 or any("reluPreMode" not in line or "pre_quant" not in line for line in inserts):
        raise ValueError("FIX量化/ReLU位点不完整")
    quant = [re.search(r"pre_quant (%\w+)", line)[1] for line in inserts]
    if any(constants[q] != 981467136 for q in quant):
        raise ValueError("FIX缩放不是FP32的2^-10")
    loops = []
    for match in re.finditer(r"^\s*for \([^\n]+\) \{", cpp, re.M):
        segment = cpp[match.start():close_brace(cpp, match.end() - 1) + 1]
        if (segment.count("TMATMUL(") == 2 and segment.count("TINSERT<") == 1
                and segment.count("TSTORE(") == 1):
            loops.append((match.group().strip(), segment))
    if len(loops) != 1:
        raise ValueError("没有唯一的QK(current)/WS(previous)循环")
    header, loop = loops[0]
    counts = {"matmul": cpp.count("TMATMUL("), "fix_to_l1": cpp.count("TINSERT<"),
              "gm_store": cpp.count("TSTORE("), "inner_pipe_all": loop.count("pipe_barrier(PIPE_ALL)"),
              "inner_mte1_to_mte2_wait": loop.count("wait_flag(PIPE_MTE1, PIPE_MTE2"),
              "inner_mte2_to_mte1_wait": loop.count("wait_flag(PIPE_MTE2, PIPE_MTE1")}
    if (counts["matmul"], counts["fix_to_l1"], counts["gm_store"], counts["inner_pipe_all"]) != (6, 3, 3, 0):
        raise ValueError(f"出现预期外的循环/指令：{counts}")
    return {"artifact": str(directory.resolve()), "key_pool": key_pool, "score_pool": score_pool,
            "key_right": select("right", "i8", [128, 64]),
            "ws_right": select("right", "f16", [384, 64]),
            "qk_acc": qk, "ws_acc": ws, "unsupported_moves": bad_moves,
            "qk_ws_acc_disjoint": True, "l1_pools_disjoint_and_fixed": True,
            "fix_scale_and_relu_preserved": True, "inner_loop": header,
            "stage_cpp_bytes": len(cpp.encode()), "loop_cpp_bytes": len(loop.encode()),
            "counts": counts, "device_validated": False,
            "scope": "编译布局和指令，不是设备异步或性能证据"}, cpp, loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, cpp, loop = audit(args.artifact)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".cpp.txt").write_text(cpp + "\n")
    args.output.with_suffix(".loop.cpp.txt").write_text(loop + "\n")
    print(json.dumps({"counts": result["counts"], "stage_cpp_bytes": result["stage_cpp_bytes"]}, ensure_ascii=False))
