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


def audit(directory, *, key_pairs=False):
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
    key_pool, score_pool = select("mat", "i8", [64, 128]), select("mat", "f16", [384, 128])
    if not key_pool or not score_pool:
        raise ValueError("未发现固定Key/score L1池")
    # 首面板与循环内Key池是不同局部对象，各自的分支phi必须保持同一地址。
    key_families = {}
    for name, tile in key_pool.items():
        family = name.split("__", 1)[0]
        key_families.setdefault(family, set()).add(tile["address"])
    key_addresses = {v["address"] for v in key_pool.values()}
    score_addresses = {v["address"] for v in score_pool.values()}
    if any(len(addresses) != 1 for addresses in key_families.values()) or len(score_addresses) != 1:
        raise ValueError("L1池在循环中未原位复用")
    sa = next(iter(score_addresses))
    if any(not (ka + 8192 <= sa or sa + 98304 <= ka) for ka in key_addresses):
        raise ValueError("两种L1池重叠")
    qk, ws = select("acc", "i32", [384, 64]), select("acc", "f32", [16, 64])
    qk_outputs, ws_outputs, right_inputs = [], [], {}
    for line in ir.splitlines():
        if "pto.tmatmul ins(" in line:
            name = re.search(r"outs\((%\w+)", line)[1]
            right_inputs[name] = re.search(r"ins\(%\w+, (%\w+)", line)[1]
            if name in qk:
                qk_outputs.append(name)
            if name in ws:
                ws_outputs.append(name)
    expected_sites = 4 if key_pairs else 2
    if len(qk_outputs) != expected_sites or len(ws_outputs) != expected_sites:
        raise ValueError(f"QK/WS静态位点数不符，预期各{expected_sites}")
    for q, w in zip(qk_outputs[1:], ws_outputs[:-1]):
        qa, wa = qk[q]["address"], ws[w]["address"]
        if not (qa + 98304 <= wa or wa + 4096 <= qa):
            raise ValueError("同轮QK与前轮WS累加区重叠")
        qr, wr = tiles[right_inputs[q]], tiles[right_inputs[w]]
        if not (qr["memory"] == wr["memory"] == "right"
                and qr["shape"] == [128, 64] and wr["shape"] == [384, 64]
                and (qr["address"] + 8192 <= wr["address"]
                     or wr["address"] + 49152 <= qr["address"])):
            raise ValueError("同轮Key/score的Right缓冲必须分开")
    inserts = [line for line in ir.splitlines() if "pto.tinsert ins(" in line and "%stage_16_" in line]
    if len(inserts) != expected_sites or any("reluPreMode" not in line or "pre_quant" not in line for line in inserts):
        raise ValueError("FIX量化/ReLU位点不完整")
    quant = [re.search(r"pre_quant (%\w+)", line)[1] for line in inserts]
    if any(constants[q] != 981467136 for q in quant):
        raise ValueError("FIX缩放不是FP32的2^-10")
    loops = []
    for match in re.finditer(r"^\s*for \([^\n]+\) \{", cpp, re.M):
        segment = cpp[match.start():close_brace(cpp, match.end() - 1) + 1]
        inner_panels = 2 if key_pairs else 1
        if (segment.count("TMATMUL(") == 2 * inner_panels
                and segment.count("TINSERT<") == inner_panels
                and segment.count("TSTORE(") == inner_panels):
            loops.append((match.group().strip(), segment))
    if len(loops) != 1:
        raise ValueError("没有唯一的QK(current)/WS(previous)循环")
    header, loop = loops[0]
    counts = {"matmul": cpp.count("TMATMUL("), "fix_to_l1": cpp.count("TINSERT<"),
              "gm_store": cpp.count("TSTORE("), "inner_pipe_all": loop.count("pipe_barrier(PIPE_ALL)"),
              "inner_mte1_to_mte2_wait": loop.count("wait_flag(PIPE_MTE1, PIPE_MTE2"),
              "inner_mte2_to_mte1_wait": loop.count("wait_flag(PIPE_MTE2, PIPE_MTE1")}
    if (counts["matmul"], counts["fix_to_l1"], counts["gm_store"], counts["inner_pipe_all"]) != (
            2 * expected_sites, expected_sites, expected_sites, 0):
        raise ValueError(f"出现预期外的循环/指令：{counts}")
    prefetch = None
    if key_pairs:
        lines = loop.splitlines()
        matmuls = [i for i, line in enumerate(lines) if "TMATMUL(" in line]
        loads = [i for i, line in enumerate(lines) if "TLOAD(" in line]
        # 每面板两页、每页两种地址分支，共四个静态TLOAD位点。
        if len(loads) != 8 or not all(i < matmuls[0] for i in loads[4:]):
            raise ValueError("第二块Key没有在第一块QK前提交")
        pending = [(i, re.search(r"EVENT_ID\d+", line)[0]) for i, line in enumerate(lines)
                   if loads[-1] < i < matmuls[0] and "set_flag(PIPE_MTE2, PIPE_MTE1," in line]
        if len(pending) != 1:
            raise ValueError("不能唯一确定下一块Key的搬运完成事件")
        issued, event = pending[0]
        consumers = [i for i, line in enumerate(lines) if i > issued
                     and f"wait_flag(PIPE_MTE2, PIPE_MTE1, {event})" in line]
        if not consumers or not matmuls[1] < consumers[0] < matmuls[2]:
            raise ValueError("下一块Key的完成等待没有推迟到第一块QK/WS提交之后")
        prefetch = {"next_key_static_load_sites": [i + 1 for i in loads[4:]],
                    "transfer_event": event, "set_line": issued + 1,
                    "first_qk_line": matmuls[0] + 1, "first_ws_line": matmuls[1] + 1,
                    "wait_line": consumers[0] + 1, "second_qk_line": matmuls[2] + 1,
                    "scope": "C++提交/等待顺序，不是设备实测重叠时长"}
    return {"artifact": str(directory.resolve()), "key_pool": key_pool, "score_pool": score_pool,
            "revision": "v1_two_panels_key_prefetch" if key_pairs else "v4_score_ring_key_local",
            "key_pairs": key_pairs,
            "next_key_prefetch_order": prefetch,
            "key_families": {k: sorted(v) for k, v in key_families.items()},
            "key_right": select("right", "i8", [128, 64]),
            "ws_right": select("right", "f16", [384, 64]),
            "qk_acc": qk, "ws_acc": ws, "unsupported_moves": bad_moves,
            "qk_ws_acc_disjoint": True, "l1_pools_disjoint_and_fixed": True,
            "current_key_and_previous_score_right_disjoint": True,
            "fix_scale_and_relu_preserved": True, "inner_loop": header,
            "stage_cpp_bytes": len(cpp.encode()), "loop_cpp_bytes": len(loop.encode()),
            "counts": counts, "device_validated": False,
            "scope": "编译布局和指令，不是设备异步或性能证据"}, cpp, loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-pairs", action="store_true")
    args = parser.parse_args()
    result, cpp, loop = audit(args.artifact, key_pairs=args.key_pairs)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".cpp.txt").write_text(cpp + "\n")
    args.output.with_suffix(".loop.cpp.txt").write_text(loop + "\n")
    print(json.dumps({"counts": result["counts"], "stage_cpp_bytes": result["stage_cpp_bytes"]}, ensure_ascii=False))
