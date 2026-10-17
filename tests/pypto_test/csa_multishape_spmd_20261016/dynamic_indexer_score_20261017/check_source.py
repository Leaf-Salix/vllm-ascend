# SPDX-License-Identifier: Apache-2.0
"""比较score数学AST并执行有界请求/尾组/双槽协议模型；不是NPU精度或缓存一致性测试。"""

import ast
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_score_stages.py"
STRATEGIES = [(2, 128, 768, False), (6, 64, 768, False), (2, 128, 1024, False), (6, 64, 1024, True)]


def math_nodes(node):
    ops = {
        "pl.tile.matmul",
        "pl.tile.assemble",
        "pl.mul",
        "pl.sort32",
        "pl.mrgsort",
        "pl.tile.mrgsort",
        "pl.maximum",
        "pl.tile.fillpad",
    }
    result = collections.defaultdict(set)
    for item in ast.walk(node):
        if (
            isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
            and isinstance(item.value, ast.Call)
            and ast.unparse(item.value.func) in ops
        ):
            result[item.targets[0].id].add(ast.dump(item.value, include_attributes=False))
    return result


def plan(history, groups, balanced):
    leaves = max((history + 8191) // 8192, 1)
    tile, extra = 8, 0
    multiple = 1
    if groups in (8, 16):
        multiple = 3
    elif groups == 4:
        multiple = 6
    if balanced and multiple > 1:
        count = (leaves + multiple - 1) // multiple * multiple
        if count <= 32:
            leaves = count
            total = (history + 1023) // 1024
            tile, extra = max(total // count, 1), total % count
    return leaves, tile, extra


def ownership(qlens, history, group, tile, balanced, invalidate=False):
    # 真实Q是累积边界；长度和position体现每请求不等长，不固定S。
    rows = []
    first = 0
    for request, count in enumerate(qlens):
        length = max(history * 4 - request * 7, count)
        visible = [max(min(length // 4, (length - count + i + 1) // 4, 262144), 0) for i in range(count)]
        if invalidate and visible:
            visible[len(visible) // 2] = 0  # causal不匹配时正式helper返回0。
        for start in range(0, count, group):
            live = min(group, count - start)
            rows.append((first + start, live, visible[start : start + live], request))
        first += count
    groups = len(rows)
    max_history = max([max(history * 4 - r * 7, n) // 4 for r, n in enumerate(qlens) if n] or [0])
    leaves, leaf_tiles, extra = plan(min(max_history, 262144), groups, balanced)
    produced = set()
    written = set()
    counted = 0
    for worker in range(24):
        for item in range(worker, groups * leaves, 24):
            gi, leaf = (item % groups, item // groups) if groups < 24 else (item // leaves, item % leaves)
            assert (gi, leaf) not in produced
            produced.add((gi, leaf))
            query, live, vis, _ = rows[gi]
            assert 0 < live <= group and query + live <= first
            begin = (leaf * leaf_tiles + min(leaf, extra)) * 1024
            capacity = (leaf_tiles + (leaf < extra)) * 1024
            valid = max(min(max(vis) - begin, capacity), 0)
            if not valid:
                continue
            lane_span = min(((valid + tile - 1) // tile) * (tile // 2), 4096)
            cube_steps = (lane_span + tile // 2 - 1) // (tile // 2)
            vector_steps = (valid + tile - 1) // tile if balanced else cube_steps
            assert cube_steps == vector_steps
            assert capacity <= 8192  # 固定score/TopK数学tile的适用边界。
            for lane in range(2):
                for qlane in range(group):
                    if qlane >= live:
                        continue
                    if balanced:
                        if qlane // (group // 2) != lane:
                            continue
                        slot = (query + qlane) * 64 + leaf
                        valid_query = max(min(vis[qlane] - begin, capacity), 0)
                        # 每个实际query都写一个根；可见候选为0时根保持NEG_INF。
                        segments = [
                            max(min(valid_query - s, min(2048, capacity - s)), 0)
                            for s in range(0, vector_steps * tile, 2048)
                        ]
                        assert sum(segments) == valid_query
                    else:
                        slot = (query + qlane) * 64 + leaf * 2 + lane
                        half_begin = begin + lane * lane_span
                        half_valid = max(min(vis[qlane] - half_begin, lane_span), 0)
                        # 排序只读取[0,half_valid)，所有读取字节必须已由本轮score store覆盖。
                        writes = [
                            max(min(vis[qlane] - half_begin - step * (tile // 2), tile // 2), 0)
                            for step in range(vector_steps)
                        ]
                        assert sum(writes) >= half_valid
                        assert half_valid <= 4096
                        assert vector_steps * (tile // 2) <= 8192
                    assert 0 <= slot < first * 64 and slot not in written
                    written.add(slot)
            counted += 1
    assert len(produced) == groups * leaves
    # 发布者需要的每个query/leaf根必须已写；容量多余的叶不必初始化或读取。
    for query, live, vis, _ in rows:
        for qlane in range(live):
            for leaf in range(leaves):
                begin = (leaf * leaf_tiles + min(leaf, extra)) * 1024
                if vis[qlane] > begin:
                    slots = (
                        [((query + qlane) * 64 + leaf)]
                        if balanced
                        else [((query + qlane) * 64 + leaf * 2 + lane) for lane in range(2)]
                    )
                    assert all(slot in written for slot in slots)
    return {
        "queries": first,
        "groups": groups,
        "work_items": len(produced),
        "active_work_items": counted,
        "written_roots": len(written),
    }


def fifo_states(steps, skip_empty_lane=False):
    # 抽象FFTS模式2：一次AIC消费ACK必须收到两个AIV各一次；非硬件实现模拟。
    # 状态=已发布数、lane0消费数、lane1消费数。最多两个未被双方消费的槽。
    start = (0, 0, 0)
    pending = [start]
    seen = {start}
    dead = []
    while pending:
        produced, c0, c1 = pending.pop()
        if (produced, c0, c1) == (steps, steps, steps):
            continue
        nexts = []
        if produced < steps and produced - min(c0, c1) < 2:
            assert produced < 2 or min(c0, c1) >= produced - 1
            nexts.append((produced + 1, c0, c1))
        if c0 < produced:
            nexts.append((produced, c0 + 1, c1))
        if c1 < produced and not skip_empty_lane:
            nexts.append((produced, c0, c1 + 1))
        if not nexts:
            dead.append([produced, c0, c1])
        for state in nexts:
            assert state[0] - min(state[1:]) <= 2
            if state not in seen:
                seen.add(state)
                pending.append(state)
    return len(seen), dead


def main():
    original = next(
        item
        for item in ast.parse((HERE / "reference_decode_indexer.py").read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "indexer_score_topk_native_cube"
    )
    current = ast.parse(SOURCE.read_text())
    old, new = math_nodes(original), math_nodes(current)
    math = {name: values <= old[name] for name, values in new.items()}
    assert all(math.values()) and len(math) == 16
    assert not any(
        isinstance(node, ast.Call) and ast.unparse(node.func) in ("pl.spmd", "pl.submit") for node in ast.walk(current)
    )
    qsets = [
        [0],
        [1],
        [2, 2, 2],
        [4, 2],
        [5, 6],
        [1, 0, 7],
        [6] * 4,
        [6] * 8,
        [6] * 16,
        [6] * 24,
        [6] * 28,
        [6] * 32,
        [5, 1, 7, 0, 3],
        [25, 2, 1],
    ]
    histories = [0, 1, 31, 32, 511, 512, 767, 768, 8191, 8192, 8193, 32768, 262144]
    rows = []
    for group, panel, tile, balanced in STRATEGIES:
        for qlens in qsets:
            for history in histories:
                for invalid in (False, True):
                    outcome = ownership(qlens, history, group, tile, balanced, invalid)
                    rows.append(
                        {
                            "G": group,
                            "panel": panel,
                            "score_tile": tile,
                            "balanced": balanced,
                            "qlens": qlens,
                            "history": history,
                            "causal_gap": invalid,
                            **outcome,
                        }
                    )
    protocol = []
    for steps in range(1, 12):
        count, dead = fifo_states(steps)
        assert not dead
        protocol.append({"steps": steps, "reachable_states": count, "deadlocks": dead})
    _, negative = fifo_states(3, True)
    assert negative
    result = {
        "status": "PASS",
        "device_execution": False,
        "precision_test": False,
        "math_ast": math,
        "math_scope": (
            "16类实际矩阵乘、FIX assemble、乘scale和sort/merge数学表达式与当前canonical逐AST相同；不声称整个函数AST相同"
        ),
        "explicit_changes": [
            "Q/L请求内分组替换固定S",
            "真实live*64限制QK有效M与WS有效K",
            "pair每query固定64行",
            "Native页表/物理页边界guard",
            "A3不支持INT8 full，改INT16零按位reinterpret；只初始化scale未gather字节",
            "删除外层nonbalanced分支中恒假的balanced旧备用段",
        ],
        "geometry_cases": len(rows),
        "cases": rows,
        "protocol_models": protocol,
        "negative_skip_empty_aiv_deadlocks": negative,
        "limits": [
            "模型不证明硬件FFTS/缓存可见性",
            "非法visible页metadata不具有定义的正确TopK数值",
            "尚不覆盖short Vector score producer",
            "完整组数值顺序保持，新的不等长尾组待设备确认",
        ],
    }
    (HERE / "source_geometry_protocol_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "protocol_models")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
