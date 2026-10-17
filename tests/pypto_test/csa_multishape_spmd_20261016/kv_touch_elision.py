# SPDX-License-Identifier: Apache-2.0
"""把原TMR的KV原样回写标记折叠为显式依赖，不删除真正的写回等待。

仅用于本实验的独占GM单SPMD：生产者fence/缓存发布和消费者读前维护保持。
不修改原动态对照，也不推断普通GM信号具有缓存一致性。
"""

import ast
import copy
import json
from pathlib import Path


def ancestors(tasks):
    result = {}
    for tid in tasks:
        pending = list(tasks[tid]["predecessors"])
        seen = set()
        while pending:
            parent = pending.pop()
            if parent == tid:
                raise ValueError("任务图存在环")
            if parent not in seen:
                seen.add(parent)
                pending.extend(tasks[parent]["predecessors"])
        result[tid] = seen
    return result


def elide(schedule, anchor, catalog, output, cost_profile):
    functions = {n.name: n for n in ast.walk(ast.parse(Path(catalog["body_bundle"]).read_text()))
                 if isinstance(n, ast.FunctionDef)}
    fn = functions["kv_touch"]
    # 精确限定为一次BF16 load/store，防止将来marker添加业务操作后仍被删除。
    if len(fn.args.args) != 1 or len(fn.body) != 4:
        raise ValueError("kv_touch已不是单参数原样回写")
    root = fn.args.args[0].arg
    load, store, ret = fn.body[1:]
    if (not isinstance(load, ast.AnnAssign) or not isinstance(store, ast.AnnAssign)
            or ast.unparse(load.value.func) != "pl.tile.load"
            or ast.unparse(store.value.func) != "pl.tile.store"
            or [ast.unparse(x) for x in load.value.args] != [root, "[0, 0]", "[1, 512]", "[1, 512]"]
            or [ast.unparse(x) for x in store.value.args] != [load.target.id, "[0, 0]", root]
            or not isinstance(ret, ast.Return) or ast.unparse(ret.value) != root
            or "pl.BF16" not in ast.unparse(load.annotation)):
        raise ValueError("kv_touch的原样读写合同改变")
    calls = schedule["call_bindings"]
    ids = [tid for tid, call in calls.items() if call["logical_function"] == "kv_touch"]
    if len(ids) != 1:
        raise ValueError("预期唯一KV标记")
    removed = ids[0]
    tasks = {t["task_id"]: t for t in schedule["tasks"]}
    task, call = tasks[removed], calls[removed]
    if task["block_num"] != 1 or task["member_count"] != 1 or task["kernel_names"] != [None, "kv_touch", None]:
        raise ValueError("KV标记的成员合同改变")
    binding = call["arguments"][0]["value"]
    if (binding["allocation"] != "argument:kv_cache" or binding["dtype"] != "BF16"
            or binding["offset"] != 0 or binding["strides"] != [512, 1]):
        raise ValueError("KV标记不再绑定Native原始KV起点")
    parents = task["predecessors"]
    if len(parents) != 1 or calls[parents[0]]["logical_function"] != "csa_cache_writeback":
        raise ValueError("KV标记的写回前驱改变")
    parent_args = calls[parents[0]]["arguments"]
    parent_annotations = {p.arg: ast.unparse(p.annotation)
                          for p in functions["csa_cache_writeback"].args.args}
    # 调用清单保存的是实参Tensor类型，可能已去掉Out；写属性以实际核体形参为准。
    if not any(isinstance(a["value"], dict) and a["value"].get("allocation") == binding["allocation"]
               and any(kind in parent_annotations[a["parameter"]] for kind in ("pl.Out[", "pl.InOut["))
               for a in parent_args):
        raise ValueError("写回前驱没有同一Native KV写参数")
    before = ancestors(tasks)
    redirects = []
    for tid, node in tasks.items():
        if removed in node["predecessors"]:
            if calls[tid]["logical_function"] != "qk_pv":
                raise ValueError("存在未审计的KV标记消费者")
            old = list(node["predecessors"])
            node["predecessors"] = sorted((set(old) - {removed}) | set(parents))
            redirects.append({"task": tid, "before": old, "after": node["predecessors"]})
    if len(redirects) != 1:
        raise ValueError("预期唯一QK/PV消费者")
    del tasks[removed]
    after = ancestors(tasks)
    if any(after[tid] != before[tid] - {removed} for tid in tasks):
        raise ValueError("折叠标记改变了其他计算任务的先后关系")
    schedule["tasks"] = [t for t in schedule["tasks"] if t["task_id"] != removed]
    del calls[removed]
    new_anchor = copy.deepcopy(anchor)
    for queues in (schedule["per_core_sequences"], new_anchor["per_core_sequences"]):
        for core, rows in queues.items():
            queues[core] = [r for r in rows if r["task"] != removed]
    # rebalance将完整重建物理顺序与等待，不复用被删除成员之后的旧进度编号。
    costs = json.loads(Path(cost_profile).read_text())
    if costs["kind"] != "same_shape_dynamic_l1_cost":
        raise ValueError("只接受原动态同规格成本")
    removed_cost = [r for r in costs["task_slots"] if r["task"] == removed]
    if len(removed_cost) != 1:
        raise ValueError("缺少标记的单成员成本")
    costs["task_slots"] = [r for r in costs["task_slots"] if r["task"] != removed]
    costs["note"] = "仅删除已证明原样回写的kv_touch成本；其余保留原动态实测，非候选实测"
    schedule["binding_checks"].update(same_logical_calls=False,
                                      retained_calls_unchanged=True,
                                      identity_marker_elided=removed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cost_path = output / "retained_cost.json"
    cost_path.write_text(json.dumps(costs, ensure_ascii=False, indent=2) + "\n")
    audit = {"removed_task": removed, "removed_kernel": "kv_touch",
             "identity_bytes": 1024, "native_allocation": binding["allocation"],
             "required_writeback_parents": parents, "redirected_consumers": redirects,
             "retained_task_ancestor_sets_identical": True,
             "other_math_and_root_and_temporaries_unchanged": True,
             "writeback_publication_and_qk_read_cache_maintenance_kept": True,
             "removed_dynamic_cost": removed_cost[0], "device_validated": False}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return new_anchor, cost_path, audit
