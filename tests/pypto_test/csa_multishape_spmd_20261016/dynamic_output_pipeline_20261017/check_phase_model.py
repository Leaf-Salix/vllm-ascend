# SPDX-License-Identifier: Apache-2.0
"""对已编译快照做轻量 CPU 所有权/阶段图核对，不模拟矩阵运算或设备缓存。

直接执行去掉注解后的 cube/vector 编排函数及 wait_row_owners；数学调用用事件
桩代替。owner 和尾零范围取数学 helper 的同源 AST 表达式，避免另写七档等待表。
拓扑检查覆盖零 T、有效尾块和七档 T。无环仅是协议必要条件，不代替设备执行。
"""

import argparse
import ast
import copy
import json
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def plain(node):
    node = copy.deepcopy(node)
    node.decorator_list = []
    node.returns = None
    for argument in node.args.args:
        argument.annotation = None
    return node


def evaluate(node, env):
    return eval(compile(ast.Expression(node), "<同源几何表达式>", "eval"), env)


def assign(fn, names, env):
    for name in names:
        statement = next(
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        )
        env[name] = evaluate(statement.value, env)


def own_work(snapshot, tokens):
    a = ast.parse((snapshot / "output_projection_a_stages.py").read_text())
    b = ast.parse((snapshot / "output_projection_b_stages.py").read_text())
    oa, quant = function(a, "o_projection_a_stage_nz"), function(a, "o_projection_group_quant_stage")
    ob = function(b, "_o_b_nz_stage")
    # 常数是本次组合的数学/硬件参数；token 仍由每个 case 提供。
    pl = SimpleNamespace(range=range, min=min, max=max)
    common = dict(
        pl=pl,
        tokens=tokens,
        O_GROUPS=8,
        O_LORA=1024,
        PROJ_A_ROW_TILE=128,
        A_COL_TILE=256,
        QUANT_TASK_T_TILE=32,
        QUANT_TOKEN_TILE=8,
        ROW_TILE=96,
        D=4096,
        PROJ_B_D_TILE=512,
    )
    oa_cells, quant_cells, ob_cells, pad_owners = {}, {}, {}, set()
    for worker in range(24):
        env = dict(common, worker_id=worker, WORKERS=24)
        assign(oa, ["row_blocks", "column_blocks"], env)
        outer = next(node for node in oa.body if isinstance(node, ast.For))
        for logical in evaluate(outer.iter, env):
            env["logical"] = logical
            assign(oa, ["g", "pa_unit", "pa_rb", "nf", "pa_r0", "pa_rows"], env)
            for row in range(env["pa_r0"], env["pa_r0"] + env["pa_rows"]):
                key = (env["g"], row, env["nf"])
                assert key not in oa_cells
                oa_cells[key] = worker
    for worker in range(48):
        env = dict(common, worker_id=worker, WORKERS=48)
        assign(quant, ["task_rows", "padded_rows"], env)
        outer = next(node for node in quant.body if isinstance(node, ast.For))
        for logical in evaluate(outer.iter, env):
            env["logical"] = logical
            assign(quant, ["g", "block", "quant_start"], env)
            for row in range(env["quant_start"], min(env["quant_start"] + 32, tokens)):
                key = (env["g"], row)
                assert key not in quant_cells
                quant_cells[key] = 24 + worker
            tail = next(node for node in outer.body if isinstance(node, ast.If))
            if evaluate(tail.test, env):
                for row in range(tokens, env["padded_rows"]):
                    key = (env["g"], row)
                    assert key not in quant_cells
                    quant_cells[key] = 24 + worker
                    pad_owners.add(24 + worker)
    for worker in range(24):
        env = dict(common, worker_id=worker, WORKERS=24)
        assign(ob, ["proj_b_t_rows", "worker"], env)
        outer = next(node for node in ob.body if isinstance(node, ast.For))
        for round_id in evaluate(outer.iter, env):
            env["round_id"] = round_id
            assign(ob, ["work", "g", "d0"], env)
            for row in range(env["proj_b_t_rows"] * 96):
                key = (env["g"], row, env["d0"] // 512)
                assert key not in ob_cells
                ob_cells[key] = worker
    padded = (tokens + 95) // 96 * 96
    assert len(oa_cells) == 8 * tokens * 4
    assert len(quant_cells) == 8 * padded
    assert len(ob_cells) == 8 * padded * 8
    return {1: set(oa_cells.values()), 2: set(quant_cells.values()), 3: set(ob_cells.values())}, pad_owners


def run_case(snapshot, tokens):
    probe = ast.parse((snapshot / "probe.py").read_text())
    phase = ast.parse((snapshot / "phase_sync.py").read_text())
    trace = []
    context = {"core": 0}
    pl = SimpleNamespace(
        range=range,
        min=min,
        max=max,
        func_attr=lambda _: None,
        tensor=SimpleNamespace(dim=lambda tensor, axis: tokens),
        tile=SimpleNamespace(get_block_idx=lambda: context["block"], get_subblock_idx=lambda: context["lane"]),
        system=SimpleNamespace(fence=lambda: None, cacheinvalid=lambda: None),
    )

    def begin(signal, core):
        trace.append(("begin", core))
        assert core == context["core"]
        return signal, 1

    def publish(signal, phase_id, core, count, generation):
        assert count == 72 and generation == 1 and core == context["core"]
        trace.append(("publish", phase_id, core))
        return signal

    def wait(signal, offset, generation):
        assert offset % 128 == 0 and generation == 1
        phase_id, producer = divmod(offset // 128, 72)
        trace.append(("wait", phase_id, producer))
        return signal

    env = dict(
        pl=pl,
        GROUPS=24,
        VECTORS=48,
        CORES=72,
        RB=96,
        SIGNAL_SLOT_WORDS=128,
        PHASE_OA=1,
        PHASE_QUANT=2,
        PHASE_OB=3,
        PHASE_OUT=4,
        begin_invocation=begin,
        publish_phase=publish,
        csa_signal_wait=wait,
        o_projection_a_stage_nz=lambda *args: args[2],
        o_projection_group_quant_stage=lambda *args: (args[1], args[2]),
        o_b_stage=lambda *args: args[2],
        o_dequant_hc_post_stage=lambda *args: args[6],
    )
    nodes = [
        plain(function(phase, "wait_row_owners")),
        plain(function(probe, "cube")),
        plain(function(probe, "vector")),
    ]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<同源阶段编排>", "exec"), env)
    owners, pad_owners = own_work(snapshot, tokens)
    edges, begins, waits_by_consumer = set(), set(), {}
    for core in range(72):
        context.update(
            core=core, block=core if core < 24 else (core - 24) // 2, lane=0 if core < 24 else (core - 24) % 2
        )
        trace.clear()
        env["cube" if core < 24 else "vector"](*([None] * 13))
        assert sum(event[0] == "begin" for event in trace) == 1 and trace[0] == ("begin", core)
        begins.add(core)
        previous = (core, 0)
        pending = set()
        for event in trace[1:]:
            if event[0] == "wait":
                pending.add((event[2], event[1]))
            else:
                assert event[0] == "publish"
                node = (core, event[1])
                edges.add((previous, node))
                edges.update((producer, node) for producer in pending)
                waits_by_consumer[node] = set(pending)
                previous, pending = node, set()
        assert not pending
    assert begins == set(range(72))
    for core in range(24):
        waited = {producer for producer, phase_id in waits_by_consumer[(core, 3)] if phase_id == 2}
        assert waited == owners[2] and pad_owners <= waited
    for core in range(24, 72):
        assert {producer for producer, phase_id in waits_by_consumer[(core, 2)] if phase_id == 1} == owners[1]
        waited = {producer for producer, phase_id in waits_by_consumer[(core, 4)] if phase_id == 3}
        # T=0 时 O_B 没有矩阵任务，但源码仍发布 24 个完成信号；等待保守且可满足。
        assert waited == (owners[3] if tokens else set(range(24)))
    adjacency, degree = defaultdict(set), defaultdict(int)
    for start, end in edges:
        adjacency[start].add(end)
        degree[start] += 0
        degree[end] += 1
    ready = deque(node for node, count in degree.items() if count == 0)
    visited = []
    while ready:
        node = ready.popleft()
        visited.append(node)
        for successor in adjacency[node]:
            degree[successor] -= 1
            if degree[successor] == 0:
                ready.append(successor)
    assert len(visited) == len(degree) == 216
    return {
        "tokens": tokens,
        "padded_rows": (tokens + 95) // 96 * 96,
        "owner_counts": {str(k): len(v) for k, v in owners.items()},
        "pad_owner_count": len(pad_owners),
        "begin_once_cores": 72,
        "phase_nodes": len(degree),
        "dependency_edges": len(edges),
        "acyclic": True,
        "actual_owners_waited": True,
        "padding_owner_waited": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(__file__).parent / "cpu_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "不覆盖历史证据"
    tokens = [0, 1, 7, 8, 24, 31, 32, 33, 48, 64, 95, 96, 97, 127, 128, 129, 144, 168, 169, 192, 193, 257]
    rows = [run_case(args.run, value) for value in tokens]
    result = {
        "passed": True,
        "scope": "同源 AST 的几何所有权和阶段有向图；不执行浮点数学或设备缓存模型",
        "cases": rows,
        "case_count": len(rows),
        "device_execution": False,
        "hardware_cache_visibility_proven": False,
        "device_precision_validated": False,
        "full_csa_validated": False,
        "performance_measured": False,
        "limits": [
            "仅 NZ/RB96/N256 组合",
            "信号单流独占、初次清零、72核每轮各推进一次是调用合同",
            "无环不证明设备调度公平性、访存完成或实际吞吐",
            "T=0仍有24个O_B空阶段完成通知",
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "case_count": len(rows), "phase_nodes_per_case": 216}))


if __name__ == "__main__":
    main()
