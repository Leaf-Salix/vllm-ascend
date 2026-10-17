# SPDX-License-Identifier: Apache-2.0
"""由当前 CSA 编排和生产核合同构造 DAG，不读取历史调度或性能数据。"""

import ast
import collections
import copy
from pathlib import Path

from .ir import functions_from_ir
from .schedule import SLOTS, assign_members

WORKER_PARAMETERS = frozenset(
    {
        "original_block_idx",
        "original_block_num",
        "original_subblock_idx",
        "original_subblock_num",
    }
)


def _access(annotation):
    if "pl.Tensor[" not in annotation:
        return set()
    if annotation.startswith("pl.InOut["):
        return {"R", "W"}
    return {"W"} if annotation.startswith("pl.Out[") else {"R"}


class SourceContracts:
    """核对生产数学函数与逻辑接口，合并早期和后期读写效果。

    物理函数曾把 scalar 排在 Tensor 前，因此按名字核对方向/类型，不能
    把重排后的列表当成逻辑调用顺序。仅四个明确的 INDEX worker 参数由
    SPMD 代码生成器补入；其他参数必须在逻辑调用中绑定。
    """

    def __init__(self, catalog):
        self.catalog = catalog
        self.logical = functions_from_ir(Path(catalog["logical_source_ir"]))[1]
        physical = functions_from_ir(Path(catalog["body_bundle"]))[1]
        self.by_logical = collections.defaultdict(dict)
        self.access = collections.defaultdict(lambda: collections.defaultdict(set))
        seen = set()
        for kernel in catalog["kernels"]:
            name = kernel["name"]
            if name in seen or name not in physical:
                raise ValueError("生产核 catalog 重复或缺少数学函数：" + name)
            seen.add(name)
            function = physical[name]
            actual = {}
            for parameter in function.args.args:
                annotation = ast.unparse(parameter.annotation)
                if parameter.arg in WORKER_PARAMETERS:
                    if annotation != "pl.Scalar[pl.INDEX]":
                        raise ValueError("worker 参数必须是 INDEX：" + parameter.arg)
                    continue
                actual[parameter.arg] = annotation
            expected = {parameter["name"]: parameter["annotation"] for parameter in kernel["parameters"]}
            if len(expected) != len(kernel["parameters"]) or actual != expected:
                raise ValueError("生产核参数与 catalog 不一致：" + name)
            if [parameter.arg for parameter in function.args.args] != kernel["inline_parameter_order"]:
                raise ValueError("生产核最终传参顺序与 catalog 不一致：" + name)
            logical, role = kernel["logical_task"], kernel["core_type"]
            if logical not in self.logical or role not in {"aic", "aiv"}:
                raise ValueError("核的逻辑来源或物理角色未知：" + name)
            if role in self.by_logical[logical]:
                raise ValueError("同一逻辑任务有多个同角色核，必须显式建模：" + logical)
            self.by_logical[logical][role] = kernel
            for parameter, annotation in actual.items():
                self.access[logical][parameter] |= _access(annotation)
        for logical, parameters in self.access.items():
            for parameter in self.logical[logical].args.args:
                parameters[parameter.arg] |= _access(ast.unparse(parameter.annotation))
            # 已有 allocation 的写仍保守包含读取，覆盖原状态及 WAW 顺序。
            for modes in parameters.values():
                if "W" in modes:
                    modes.add("R")

    def check_bindings(self, bindings):
        scopes = {scope["id"]: scope for scope in bindings["scopes"]}
        if len(scopes) != len(bindings["scopes"]):
            raise ValueError("作用域编号重复")
        for scope in scopes.values():
            if set(scope["options"]) - {"mode"} or scope["options"].get("mode", "auto") not in {"auto", "manual"}:
                raise ValueError("未知作用域选项")
        for ordinal, task in enumerate(bindings["tasks"]):
            if task["ordinal"] != ordinal or set(task["scope"]) - set(scopes):
                raise ValueError("编排序号或作用域不完整")
            if task["kind"] == "dependency_only":
                if task["block_num"] != 0 or task["logical_function"] is not None or set(task["options"]) - {"deps"}:
                    raise ValueError("只支持纯依赖 dummy")
                continue
            if task["kind"] != "compute" or type(task["block_num"]) is not int or task["block_num"] <= 0:
                raise ValueError("未知任务种类或非法分片数")
            if set(task["options"]) - {"deps", "name_hint", "allow_early_resolve", "sync_start"}:
                raise ValueError("存在未建模的任务选项")
            logical = task["logical_function"]
            if logical not in self.by_logical:
                raise ValueError("逻辑调用没有生产数学体：" + logical)
            expected = [
                (parameter.arg, ast.unparse(parameter.annotation)) for parameter in self.logical[logical].args.args
            ]
            actual = [(argument["parameter"], argument["annotation"]) for argument in task["arguments"]]
            if expected != actual:
                raise ValueError("逻辑调用的参数顺序或注解改变：" + logical)
            known = {argument["parameter"] for argument in task["arguments"]}
            for kernel in self.by_logical[logical].values():
                if {parameter["name"] for parameter in kernel["parameters"]} - known:
                    raise ValueError("物理核存在未绑定的参数：" + kernel["name"])
        return {
            "logical_call_contracts": sum(task["kind"] == "compute" for task in bindings["tasks"]),
            "physical_parameter_contracts": len(self.catalog["kernels"]),
            "physical_parameters_checked_by_name": True,
            "effects": "逻辑接口与生产物理核的保守并集；写入保守包含读取",
            "original_manual_scopes_retained": True,
            "manual_dep_modifier_tracking_skipped": True,
            "dummy_nodes_expanded_without_kernel": True,
        }


def _task_refs(value):
    if isinstance(value, dict):
        if "task_ordinal" in value:
            ordinal = value["task_ordinal"]
            if type(ordinal) is not int or ordinal < -1:
                raise ValueError("非法 TaskRef")
            yield ordinal
        else:
            for child in value.values():
                yield from _task_refs(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _task_refs(child)


def _interval(ref):
    shape, strides = ref["shape"], ref["strides"]
    if len(shape) != len(strides) or ref["offset"] < 0 or any(n < 0 for n in shape) or any(s < 0 for s in strides):
        raise ValueError("不支持负偏移、负 stride 或不完整的 Tensor view")
    begin = ref["offset"]
    end = begin if any(n == 0 for n in shape) else begin + sum((n - 1) * s for n, s in zip(shape, strides)) + 1
    return begin, end


def _overlap(left, right):
    if left["allocation"] != right["allocation"]:
        return False
    if left["dtype"] != right["dtype"]:
        raise ValueError("同一 allocation 的混合 dtype 不能按元素区间推断重叠")
    left_begin, left_end = _interval(left)
    right_begin, right_end = _interval(right)
    return left_begin < left_end and right_begin < right_end and left_begin < right_end and right_begin < left_end


def closure(parents):
    """编排 DAG 的祖先集合；调用序号只能依赖之前的调用。"""
    result = {}
    for task in sorted(parents):
        ancestors = set(parents[task])
        for parent in parents[task]:
            if parent >= task or parent not in result:
                raise ValueError("编排包含未来或未知 TaskRef")
            ancestors.update(result[parent])
        result[task] = ancestors
    return result


def infer_dependencies(bindings, access):
    """显式 TaskRef 与自动作用域的 RAW/WAW/WAR 取并集。

    手动作用域及 manual_dep 的并行写集合由源码显式 deps 负责，不能用
    allocation 包围区间把它们强行串行化。静态 workspace 整次调用存活，
    本模块不推断 allocation 复用，也不发现不同根参数的运行时 storage 别名。
    """
    tasks = bindings["tasks"]
    manual = {scope["id"] for scope in bindings["scopes"] if scope["options"].get("mode") == "manual"}
    graph = {task["ordinal"]: set() for task in tasks}
    explicit = {task["ordinal"]: set() for task in tasks}
    writes, reads, war_edges = [], [], set()
    for task in tasks:
        ordinal = task["ordinal"]
        refs = list(_task_refs(task["options"]))
        refs.extend(_task_refs([argument["value"] for argument in task.get("arguments", [])]))
        for parent in refs:
            if parent == -1:
                continue
            if parent >= ordinal or parent not in graph:
                raise ValueError("TaskRef 必须引用前序调用")
            graph[ordinal].add(parent)
            explicit[ordinal].add(parent)
        if task["kind"] != "compute" or manual.intersection(task["scope"]):
            continue
        new_writes, new_reads = [], []
        for argument in task["arguments"]:
            ref = argument["value"]
            if not isinstance(ref, dict) or "allocation" not in ref:
                continue
            modes = access[task["logical_function"]][argument["parameter"]]
            if not modes:
                raise ValueError("Tensor 参数缺少读写效果")
            if ref.get("manual_dep", False):
                continue
            _interval(ref)
            if "R" in modes:
                graph[ordinal].update(producer for producer, old_ref in writes if _overlap(old_ref, ref))
                new_reads.append((ordinal, ref))
            if "W" in modes:
                for reader, old_ref in reads:
                    if _overlap(old_ref, ref):
                        graph[ordinal].add(reader)
                        war_edges.add((reader, ordinal))
                new_writes.append((ordinal, ref))
        writes.extend(new_writes)
        reads.extend(new_reads)

    def expand_dummy(source):
        expanded = {}
        for task in tasks:
            if task["kind"] != "compute":
                continue
            pending, parents = list(source[task["ordinal"]]), set()
            while pending:
                parent = pending.pop()
                if tasks[parent]["kind"] == "compute":
                    parents.add(parent)
                else:
                    pending.extend(source[parent])
            expanded[task["ordinal"]] = parents
        return expanded

    return expand_dummy(graph), expand_dummy(explicit), war_edges


def build_source_schedule(bindings, catalog, *, policy="earliest_ready", member_costs=None, contracts=None):
    """返回完整 schedule 与 audit，供 layout/codegen 使用。

    全部形状统一从本轮编排建图。member_costs 是可选模型权重映射：支持
    kernel_name、(kernel_name, slot)、(call_ordinal, slot)，后者优先。
    默认10.0；不读取 anchor/profile，也不执行数学核或读取 device 数值。
    """
    contracts = contracts or SourceContracts(catalog)
    if contracts.catalog != catalog:
        raise ValueError("来源合同不属于当前 catalog")
    checks = contracts.check_bindings(bindings)
    parents, explicit, war_edges = infer_dependencies(bindings, contracts.access)
    before = closure(parents)
    task_ids = {ordinal: str(ordinal + 1) for ordinal in parents}
    tasks, calls, templates = [], {}, {}
    manual = {scope["id"] for scope in bindings["scopes"] if scope["options"].get("mode") == "manual"}
    for call in bindings["tasks"]:
        if call["kind"] != "compute":
            continue
        ordinal, logical = call["ordinal"], call["logical_function"]
        task_id, kernels = task_ids[ordinal], contracts.by_logical[logical]
        if set(kernels) == {"aic", "aiv"}:
            roles = [("AIC", kernels["aic"]), ("AIV0", kernels["aiv"]), ("AIV1", kernels["aiv"])]
        elif set(kernels) == {"aic"}:
            roles = [("AIC", kernels["aic"])]
        elif set(kernels) == {"aiv"}:
            roles = [("AIV0", kernels["aiv"])]
        else:
            raise ValueError("未支持的物理角色组合：" + logical)
        kernel_names, kernel_ids = [None] * len(SLOTS), [-1] * len(SLOTS)
        for slot, kernel in roles:
            index = SLOTS.index(slot)
            kernel_names[index], kernel_ids[index] = kernel["name"], kernel["func_id"]
            templates[task_id, slot] = {
                "task": task_id,
                "core_kind": kernel["core_type"],
                "original_subslot": slot,
                "possible_kernel_names": [kernel["name"]],
                "kernel_id": kernel["func_id"],
                "kernel_name": kernel["name"],
                "call_ordinal": ordinal,
                "original_sync_start": call["options"].get("sync_start", False),
                "body_source_line": kernel["generated_line"],
                "body_parameter_names": kernel["inline_parameter_order"],
            }
        tasks.append(
            {
                "task_id": task_id,
                "task_id_kind": "source_call_ordinal_plus_one",
                "scope": "manual" if manual.intersection(call["scope"]) else "auto",
                "early_dispatch": call["options"].get("allow_early_resolve", False),
                "original_sync_start": call["options"].get("sync_start", False),
                "kernel_ids": kernel_ids,
                "kernel_names": kernel_names,
                "block_num": call["block_num"],
                "predecessors": [task_ids[parent] for parent in sorted(parents[ordinal])],
                "member_count": len(roles) * call["block_num"],
            }
        )
        calls[task_id] = copy.deepcopy(call)
    schedule = {
        "schema_version": 1,
        "source": "生产编排、数学效果与模型成本",
        "scope": "源码功能调度；无设备性能结论",
        "source_kind": "source_derived_shape_balanced_schedule",
        "summary": {
            "compute_submissions": len(tasks),
            "compute_dependency_pairs": sum(map(len, parents.values())),
            "data_and_per_core_graph_acyclic": True,
        },
        "tasks": tasks,
        "call_bindings": calls,
        "non_compute_tasks": [copy.deepcopy(task) for task in bindings["tasks"] if task["kind"] != "compute"],
        "temporary_allocations": copy.deepcopy(bindings["allocations"]),
        "root_arguments": copy.deepcopy(bindings["root_arguments"]),
        "selected_control_path": copy.deepcopy(bindings["selected_control_path"]),
        "body_bundle": catalog["body_bundle"],
        "all_call_bindings_verified": True,
        "binding_checks": checks,
        "logical_block_mapping_complete": True,
    }
    costs = assign_members(schedule, templates, policy=policy, member_costs=member_costs)
    after = {
        calls[task["task_id"]]["ordinal"]: {calls[parent]["ordinal"] for parent in task["predecessors"]}
        for task in tasks
    }
    if closure(after) != before:
        raise ValueError("分核改变了数学任务偏序")
    schedule["summary"]["logical_compute_members"] = sum(task["member_count"] for task in tasks)
    schedule["mixed_logical_groups_checked"] = sum(task["block_num"] for task in tasks if all(task["kernel_names"]))
    audit = {
        "source_contract": checks,
        "task_graph_reachable_pairs": sum(map(len, before.values())),
        "task_graph_direct_edges": sum(map(len, parents.values())),
        "explicit_dependency_direct_edges": sum(map(len, explicit.values())),
        "dependency_only_nodes": len(schedule["non_compute_tasks"]),
        "math_task_dag_preserved_by_scheduler": True,
        "global_barriers_added": 0,
        "automatic_read_write_hazards": "RAW/WAW/WAR；manual scope/manual_dep 保留显式合同",
        "war_direct_edges": sorted(war_edges),
        "serial_task_chain_fallback": False,
        "cost_sources": costs,
        "hardware_trace_required_for_functional_template": False,
        "device_correctness_verified": False,
        "performance_measured": False,
        "wait_plan": schedule["wait_plan"],
    }
    schedule["source_dependency_audit"] = audit
    return schedule, audit
