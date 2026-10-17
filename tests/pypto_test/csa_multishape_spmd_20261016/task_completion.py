# SPDX-License-Identifier: Apache-2.0
"""测试用任务完成计数：保留原数学 DAG，减少逐核进度轮询。

每个原逻辑 task 的成员分别 AtomicAdd(1)，消费者等待该 task 的全部成员。
它不引入全卡 barrier，也没有指定负责汇聚的 leader。每轮不清零计数器；
目标是 generation * member_count。业务写回、DCCI/DSB 与本核流水排空保持。
临时 signal API 尚无 AtomicAdd 参数，因此只在本测试编译进程把 -1 哨兵
转为已有 PTO-ISA TNOTIFY(AtomicAdd, 1)，不修改共享 PyPTO 或 Simpler。
"""

import ast
import collections
import contextlib
import copy
import json
from pathlib import Path


def use_task_counters(source, layout, schedule):
    from build_schedule_manifest import ancestor_sets

    tasks = {row["task_id"]: row for row in schedule["tasks"]}
    ids = sorted(tasks, key=lambda tid: schedule["call_bindings"][tid]["ordinal"])
    if len(ids) > 72:
        raise ValueError("当前预留的独占 counter 行最多 72 个逻辑 task")
    nodes = [row for sequence in schedule["per_core_sequences"].values() for row in sequence]
    indices = {(str(row["core"]), row["dispatch_seq"]): i for i, row in enumerate(nodes)}
    members = collections.defaultdict(set)
    for i, row in enumerate(nodes):
        members[row["task"]].add(i)
    counts = {tid: len(members[tid]) for tid in ids}
    if not all(0 < count < 1024 for count in counts.values()):
        raise ValueError("counter 成员数必须在当前 INT32 代际范围内")
    # 每个 512 B 行的第 256 字节；与首个 cacheline 的单写者代际标记分开。
    offsets = {tid: ordinal * 128 + 64 for ordinal, tid in enumerate(ids)}
    before, after = [set() for _ in nodes], [set() for _ in nodes]
    for i, row in enumerate(nodes):
        if row["dispatch_seq"] > 1:
            parent = indices[str(row["core"]), row["dispatch_seq"] - 1]
            before[i].add(parent)
            after[i].add(parent)
        before[i].update(indices[str(core), step]
                         for core, step in row["wait_for_core_progress"].items())
        for tid in tasks[row["task"]]["predecessors"]:
            after[i].update(members[tid])
    _, original_ancestors = ancestor_sets(before)
    _, counter_ancestors = ancestor_sets(after)
    if original_ancestors != counter_ancestors:
        raise ValueError("任务计数替换改变了数学/同核依赖偏序")

    module = ast.parse(source)
    phases = 0
    for role, batches in layout["batches"].items():
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                        and node.name == "csa_static_" + role)
        stages = [node for node in function.body if isinstance(node, ast.If)
                  and ast.unparse(node.test).startswith("mapped_block_")]
        if len(stages) != len(batches):
            raise ValueError("原核体阶段与分核清单数量不符")
        for stage, batch in zip(stages, batches):
            tid = batch["task"]
            if {row["task"] for row in batch["members"]} != {tid}:
                raise ValueError("一个控制阶段不能包含多个逻辑 task")
            if not isinstance(stage.body[5], ast.For) or not isinstance(stage.body[-1], ast.If):
                raise ValueError("不支持已改写的等待/发布形状")
            math = copy.deepcopy(stage.body[6:-1])
            waits = []
            for parent in tasks[tid]["predecessors"]:
                waits += ast.parse(
                    f"progress = csa_signal_wait(progress, {offsets[parent]}, "
                    f"pl.cast((generation_base // 1024) * {counts[parent]}, pl.INT32))").body
            arrive = ast.parse(
                f"progress = csa_signal_notify(progress, {offsets[tid]}, pl.cast(-1, pl.INT32))").body
            # 这些字段只用于已删除的逐核等待/发布；subslot 保持原数学可用。
            stage.body = stage.body[:1] + waits + stage.body[6:-1] + arrive
            assert ast.dump(ast.Module(body=math, type_ignores=[])) == ast.dump(
                ast.Module(body=stage.body[1 + len(waits):-1], type_ignores=[]))
            for row in batch["members"]:
                row["publish_completion"] = False
                row["task_counter_offset"] = offsets[tid]
                row["task_counter_expected_per_generation"] = counts[tid]
            phases += 1
    audit = {
        "scope": "仅测试候选，尚无设备验证；不修改数学或业务缓存维护",
        "strategy": "task_atomic_completion", "phases": phases,
        "original_member_waits": sum(len(row["wait_for_core_progress"]) for row in nodes),
        "task_waits": sum(len(tasks[row["task"]]["predecessors"]) for row in nodes),
        "arrival_publications": len(nodes), "identical_dependency_partial_order": True,
        "new_global_barriers": 0, "signal_dtype": "INT32", "counter_cacheline_offset_bytes": 256,
        "counter_stride_bytes": 512, "counter_initial_value": 0,
        "generation_stride": 1024, "math_unchanged": True,
        "tasks": [{"task": tid, "offset_words": offsets[tid], "members": counts[tid],
                   "predecessors": tasks[tid]["predecessors"]} for tid in ids],
    }
    layout["task_completion"] = audit
    layout["protocol"]["signal_api"] = "temporary_task_atomic_add_and_ge"
    ast.fix_missing_locations(module)
    return "# 任务完成计数实验：编译必须同时使用 task_counter_codegen。\n" + ast.unparse(module) + "\n", audit


HELPER = r'''
template <typename Signal>
static AICORE inline void CSA_TASK_NOTIFY(Signal& signal, int32_t value, pto::comm::NotifyOp op) {
  PTO_ASSERT(op == pto::comm::NotifyOp::Set, "temporary input op must be Set");
  if (value == -1) {
    // 已有 A3 路径：set_st_atomic_cfg, DCCI, st_atomic, DCCI, DSB。
    // atomic 仅解决多写者丢更新；发布和业务数据完成仍由缓存维护/屏障保证。
    pto::comm::TNOTIFY(signal, int32_t(1), pto::comm::NotifyOp::AtomicAdd);
  } else {
    PTO_ASSERT(value >= 0, "unexpected negative progress");
    pto::comm::TNOTIFY(signal, value, op);
  }
}
'''


@contextlib.contextmanager
def task_counter_codegen(output, expected_sites):
    from pypto.backend import pto_backend

    original = pto_backend._compile_pto_module
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)

    def compile_counter(pto_code, name, output_dir, *args, **kwargs):
        source = original(pto_code, name, output_dir, *args, **kwargs)
        if name != "csa_static_group":
            return source
        needle, marker = "pto::comm::TNOTIFY(", "using namespace pto;"
        sites = source.count(needle)
        if sites != expected_sites + 2 or source.count(marker) != 1:
            raise ValueError(f"任务计数发布位置不匹配：{sites}/{expected_sites + 2}")
        (directory / "original.cpp").write_text(source)
        source = source.replace(needle, "CSA_TASK_NOTIFY(").replace(marker, marker + "\n" + HELPER)
        (directory / "task_counters.cpp").write_text(source)
        (directory / "change.json").write_text(json.dumps(
            {"notify_sites": sites, "counter_sites": expected_sites,
             "atomic_implementation": "PTO-ISA A3 TNOTIFY(AtomicAdd)",
             "payload_maintenance_unchanged": True, "device_validated": False},
            ensure_ascii=False, indent=2) + "\n")
        return source

    pto_backend._compile_pto_module = compile_counter
    try:
        yield
    finally:
        pto_backend._compile_pto_module = original
