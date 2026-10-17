# SPDX-License-Identifier: Apache-2.0
"""由实际描述符生成 CSA 单 SPMD 变体；不读取 device Tensor 的数值。

编译策略按数学分支和形状选择，不要求不同 batch 共用二进制。已捕获的七档
只提供核顺序参考；实际调用参数、分片数量、视图和等待表均重新计算。
默认保留同构参考路径；区间入口可由当前 IR 读写效应及显式依赖构造新分支。
"""

import ast
import collections
import copy
import json
import sys
from pathlib import Path

STATIC_ROOT = Path(__file__).resolve().parent.parent / "csa_static_spmd_feasibility_20261015"
sys.path.insert(0, str(STATIC_ROOT))

from bind_orchestration import DTYPES, bind
from build_schedule_manifest import ancestor_sets
from generate_static_entry import generate, jit_source
from prepare_static_layout import prepare
from validate_static_protocol import validate

from signal_lowering import use_local_signals
from historical_sources import KERNEL_SOURCE


def descriptors_from_tensors(arguments):
    """只访问 CPU 已有的 Tensor 描述符；不调用 cpu/item/tolist。"""
    return {name: {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                   "dtype": str(tensor.dtype)} for name, tensor in arguments.items()}


def descriptors_from_roots(roots):
    names = {value: key for key, value in DTYPES.items()}
    return {name: {"shape": ref["shape"], "stride": ref["strides"], "dtype": names[ref["dtype"]]}
            for name, ref in roots.items()}


def catalog():
    result = json.loads((STATIC_ROOT / "generated/core_body_catalog.json").read_text())
    # 引用归档内的符号 IR，不依赖旧会话的绝对 build_output 路径。
    result.update(body_bundle=str(KERNEL_SOURCE),
                  source_ir=str(STATIC_ROOT / "references/symbolic_ir/25_after_ExpandMixedKernel.py"),
                  logical_source_ir=str(STATIC_ROOT / "references/symbolic_ir/10_after_OutlineIncoreScopes.py"))
    return result


def calls_in_order(bindings):
    calls = bindings.get("call_bindings")
    if calls is not None:
        calls = calls.values()
    else:
        calls = (call for call in bindings["tasks"] if call["kind"] == "compute")
    return sorted(calls, key=lambda call: call["ordinal"])


def call_signature(bindings):
    # 相同计算名仍不能忽略 sync_start、显式依赖参数和 scope 的改变。
    return [(call["ordinal"], call["logical_function"], call["options"], call["scope"],
             [(arg["parameter"], arg["annotation"]) for arg in call["arguments"]])
            for call in calls_in_order(bindings)]


class MissingScheduleAnchor(ValueError):
    """没有整图同构参考，需使用源码依赖构造器。"""


def choose_anchor(bindings):
    compatible = []
    wanted = call_signature(bindings)
    tokens = bindings["root_arguments"]["x_hc"]["shape"][0]
    for path in sorted((STATIC_ROOT / "bindings").glob("*_prepared.json")):
        candidate = json.loads(path.read_text())
        if call_signature(candidate) == wanted:
            distance = abs(candidate["root_arguments"]["x_hc"]["shape"][0] - tokens)
            compatible.append((distance, path.name, candidate))
    if not compatible:
        raise MissingScheduleAnchor("当前数学分支尚无同构调度参考；需增加该分支，不能套用 B16 等待表")
    _, filename, result = min(compatible, key=lambda value: value[:2])
    return filename.removesuffix("_prepared.json"), result


def check_alias_pattern(anchor, bindings):
    """核对重绑定没有引入参考任务图未涵盖的视图重叠。

    这里比较的是各调用 Tensor 视图的地址包围区间，保守地包含 stride 空洞。
    同一个区间模式才能复用参考的 task 依赖；不会以历史 task_id 代替实际视图。
    """
    old_refs, new_refs = [], []
    for before, after in zip(calls_in_order(anchor), calls_in_order(bindings)):
        for old, new in zip(before["arguments"], after["arguments"]):
            left, right = old["value"], new["value"]
            if isinstance(left, dict) and "allocation" in left:
                for field in ("allocation", "dtype", "layout", "manual_dep"):
                    if left[field] != right[field]:
                        raise ValueError(f"调用视图属性改变：{after['logical_function']}/{field}")
                old_refs.append(left)
                new_refs.append(right)
            elif isinstance(left, dict) and left != right:
                raise ValueError("显式 task 依赖发生变化，不能复用历史任务图")

    def overlap(left, right):
        if left["allocation"] != right["allocation"]:
            return False
        def end(ref):
            return ref["offset"] + sum((n - 1) * s for n, s in zip(ref["shape"], ref["strides"])) + 1
        return left["offset"] < end(right) and right["offset"] < end(left)

    pairs = 0
    for index, old in enumerate(old_refs):
        for other in range(index):
            if overlap(old, old_refs[other]) != overlap(new_refs[index], new_refs[other]):
                raise ValueError(f"重绑定改变了视图重叠关系：参数 {index}/{other}")
            pairs += 1
    return pairs


def rebuild_waits(per_core, tasks):
    """根据新的工作量重新裁剪依赖；不增加阶段级全核屏障。

    算法沿用静态基线 build_schedule_manifest.py 的 DAG 位集传递约简，
    但输入是重新分配后的成员，旧 dispatch_seq/等待表不再有效。
    """
    nodes = [member for sequence in per_core.values() for member in sequence]
    ids = {(str(row["core"]), row["dispatch_seq"]): i for i, row in enumerate(nodes)}
    members = collections.defaultdict(list)
    for i, row in enumerate(nodes):
        members[row["task"]].append(i)
    parents = {task["task_id"]: task["predecessors"] for task in tasks}
    original = [set() for _ in nodes]
    for i, row in enumerate(nodes):
        if row["dispatch_seq"] > 1:
            original[i].add(ids[(str(row["core"]), row["dispatch_seq"] - 1)])
        for task in parents[row["task"]]:
            original[i].update(members[task])
    order, ancestors = ancestor_sets(original)
    position = {node: index for index, node in enumerate(order)}
    reduced = [set() for _ in nodes]
    for node, predecessors in enumerate(original):
        covered = 0
        for parent in sorted(predecessors, key=position.get, reverse=True):
            if not covered & (1 << parent):
                reduced[node].add(parent)
                covered |= ancestors[parent] | (1 << parent)

    published = collections.defaultdict(set)
    for core, sequence in per_core.items():
        observed = collections.defaultdict(int)
        for row in sequence:
            required = collections.defaultdict(int)
            for parent in reduced[ids[(core, row["dispatch_seq"])]]:
                before = nodes[parent]
                source = str(before["core"])
                if source != core:
                    required[source] = max(required[source], before["dispatch_seq"])
            row["wait_for_core_progress"] = {
                source: step for source, step in sorted(required.items(), key=lambda pair: int(pair[0]))
                if step > observed[source]}
            for source, step in row["wait_for_core_progress"].items():
                observed[source] = step
                published[source].add(step)
    planned = [set() for _ in nodes]
    for i, row in enumerate(nodes):
        core = str(row["core"])
        row["publish_completion"] = row["dispatch_seq"] in published[core]
        if row["dispatch_seq"] > 1:
            planned[i].add(ids[(core, row["dispatch_seq"] - 1)])
        planned[i].update(ids[(source, step)] for source, step in row["wait_for_core_progress"].items())
    _, after = ancestor_sets(planned)
    if after != ancestors:
        raise ValueError("生成等待表后依赖偏序改变")
    return {"remote_core_progress_requirements": sum(len(row["wait_for_core_progress"]) for row in nodes),
            "completion_publications": sum(row["publish_completion"] for row in nodes),
            "original_member_edges": sum(map(len, original)), "planned_member_edges": sum(map(len, planned)),
            "identical_dependency_partial_order": True, "global_barriers_added": 0}


def rebind_schedule(anchor_name, anchor, bindings):
    pairs = check_alias_pattern(anchor, bindings)
    new_calls = {call["ordinal"]: call for call in calls_in_order(bindings)}
    result = copy.deepcopy(anchor)
    tasks = {task["task_id"]: task for task in result["tasks"]}
    for task_id, old_call in anchor["call_bindings"].items():
        new_call = new_calls[old_call["ordinal"]]
        result["call_bindings"][task_id] = new_call
        tasks[task_id]["block_num"] = new_call["block_num"]
        # 旧 trace 的 Tensor ID/shape 不是新调用证据；不留在新清单中误导读者。
        tasks[task_id].pop("args", None)

    per_core = {}
    for core, sequence in anchor["per_core_sequences"].items():
        members = []
        for old in sequence:
            call = result["call_bindings"][old["task"]]
            # 一个参考位置承接 b、b+原块数、...，但不复制核体或进度记录。
            # 在源码里生成核内循环，所有分片完成后只发布一次。参考位置
            # 只属于同一个逻辑任务，下游本来就等待它的全部分片，故可合并发布。
            # 新规格更小时允许零次循环，保留控制位置，避免改变其他核的顺序。
            row = {key: copy.deepcopy(value) for key, value in old.items()
                   if key not in {"end", "dispatch_evidence", "wait_for_core_progress", "publish_completion"}}
            row.update(logical_block_num=call["block_num"],
                       logical_block_stride=old["logical_block_num"], dispatch_seq=len(members) + 1,
                       reference_end=old["end"])
            members.append(row)
        per_core[core] = members

    actual = collections.defaultdict(collections.Counter)
    mixed = collections.defaultdict(dict)
    for sequence in per_core.values():
        for row in sequence:
            task = tasks[row["task"]]
            for block in range(row["logical_block"], row["logical_block_num"], row["logical_block_stride"]):
                actual[row["task"]][row["original_subslot"], block] += 1
                if task["kernel_names"][0] and any(task["kernel_names"][1:]):
                    mixed[row["task"], block][row["original_subslot"]] = row["core"]
    for task_id, task in tasks.items():
        expected = collections.Counter((slot, block) for slot, name in
                                       zip(("AIC", "AIV0", "AIV1"), task["kernel_names"])
                                       if name for block in range(task["block_num"]))
        if actual[task_id] != expected:
            raise ValueError(f"新形状分片遗漏或重复：{task_id}")
        task["member_count"] = sum(actual[task_id].values())
    for key, group in mixed.items():
        for slot, lane in (("AIV0", 0), ("AIV1", 1)):
            if slot in group and group[slot] != 24 + 2 * group["AIC"] + lane:
                raise ValueError(f"新形状破坏 MIX 配对：{key}")
    result.update(per_core_sequences=per_core, wait_plan=rebuild_waits(per_core, list(tasks.values())),
                  source=f"参数化分配，参考 {anchor_name}", source_kind="shape_specialization",
                  temporary_allocations=bindings["allocations"], root_arguments=bindings["root_arguments"],
                  selected_control_path=bindings["selected_control_path"],
                  mixed_logical_groups_checked=len(mixed), all_call_bindings_verified=True,
                  binding_checks={"same_logical_calls": True, "alias_pairs_checked": pairs,
                                  "logical_block_coverage": True, "mixed_pairing": True},
                  scope="参数化后的 CPU 准备结果；尚未验证此规格的设备执行或性能")
    result["summary"].update(physical_members=sum(map(len, per_core.values())),
                             logical_compute_members=sum(sum(values.values()) for values in actual.values()))
    for obsolete in ("dispatch_epochs_per_core", "tensors", "original_edges", "remaining_checks"):
        result.pop(obsolete, None)
    return result


def loop_rebound_blocks(source, layout):
    """保持等待/发布位置，把一个控制位置承接的逻辑分片放进核内循环。"""
    module = ast.parse(source)
    rewritten = 0
    for role in ("aic", "aiv"):
        fn = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                  and node.name == "csa_static_" + role)
        for stmt in fn.body:
            if not isinstance(stmt, ast.If) or not ast.unparse(stmt.test).startswith("mapped_block_"):
                continue
            index = int(stmt.test.left.id.removeprefix("mapped_block_"))
            members = layout["batches"][role][index]["members"]
            strides = {row["logical_block_stride"] for row in members}
            counts = {row["logical_block_num"] for row in members}
            assert len(strides) == len(counts) == 1
            stride, count = next(iter(strides)), next(iter(counts))
            if stride == count:
                continue
            # signal_lowering 生成 6 项控制语句、3 项读侧维护、原数学体、
            # 3 项写侧维护与最后的条件发布。仅对数学体加循环。
            math_body = stmt.body[9:-4]
            old_name, new_name = f"block_{index}", f"logical_block_{index}"

            class RenameBlock(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == old_name:
                        return ast.copy_location(ast.Name(id=new_name, ctx=node.ctx), node)
                    return node

            # PTOAS 当前对“外层 INDEX 循环变量直接作内层起点”的 lowering
            # 会产生 emitc.for 类型不一致。改为固定波数 + 动态 block 偏移，
            # 数学上的 block 集合与执行顺序不变；尾波使用显式范围保护。
            loop = ast.parse(
                f"for block_wave_{index} in pl.range({(count + stride - 1) // stride}):\n"
                f"    {new_name} = {old_name} + block_wave_{index} * {stride}\n"
                f"    if {new_name} < {count}:\n        pass").body[0]
            loop.body[1].body = [RenameBlock().visit(node) for node in math_body]
            stmt.body = stmt.body[:9] + [loop] + stmt.body[-4:]
            rewritten += 1
    ast.fix_missing_locations(module)
    return "# 按形状特化的 CSA 单 SPMD；新增分片在核内循环，复用核体。\n" + ast.unparse(module) + "\n", rewritten


def build_variant(descriptors, *, max_seq_len, output_root, case_name, aggregate_completion=False,
                  balanced_schedule=False, compile_schedule=False, precise_payload_cache=False,
                  schedule_policy="critical_path", task_completion=False, cost_profile=None,
                  rotate_wait_order=False, priority_weight=0.0, oproj_n_tile=128,
                  reuse_gm=False, qproj_workers=20, qk_vector_indices=False,
                  indexer_n128=False, qk_full_blocks=False, qk_cache_masks=False,
                  qk_mgather=False, hc_rows=8, indexer_page_views=False,
                  qk_scoped_zero=False, qk_valid_gather=False, oproj_k512=False,
                  qk_chunked_gather=False, indexer_leaf16384=False, indexer_score_pingpong=False,
                  qk_merge_heads=False, qk_direct_l1=False, row_dependencies=False, indexer_n80=False,
                  hc_mix_resident=False, qproj_n128=False, qproj_l0_prefetch=False,
                  hc_front_fusion=False, hc_front_tiles=False, indexer_scale_owners=False,
                  indexer_input_dependencies=False, hc_mix_chunk_buffer=False,
                  restore_weight_cache_policy=False, qk_query_metadata=False,
                  indexer_n80_ws128=False, qk_all_valid=False, indexer_short_tail=False,
                  indexer_rolled_panels=False, indexer_panel_group=0,
                  indexer_transfer_2048=False, qproj_l1_columns=False, indexer_n32_acc=False,
                  drop_kv_touch=False, kv_touch_signal_only=False, indexer_l1_ring=False,
                  indexer_ws_load_order=False, indexer_key_pairs=False, qk_early_kv=False,
                  qk_role_split=False, qk_role_wide_heads=False, rope_flat_gather=False,
                  allow_source_schedule=False):
    """在初始化/图捕获前生成变体。max_seq_len 必须来自已有 Host metadata。

    只用它选择原编排中的 Indexer 分支；绝不把该值写进实际 kv_seq_lens。
    编译后的核体继续读取调用者的原 device 长度、位置、slot 与页表。
    """
    inputs = copy.deepcopy(descriptors)
    if qk_role_wide_heads and not qk_role_split:
        raise ValueError("32head softmax/16head PV/8head发布候选必须启用双AIV分工")
    if qk_role_split and (inputs["x_hc"]["shape"][0] != 96
            or not all((qk_vector_indices, row_dependencies, balanced_schedule))
            or any((qk_early_kv, qk_query_metadata, qk_all_valid, qk_direct_l1, qk_merge_heads,
                    qk_chunked_gather, qk_valid_gather, qk_scoped_zero, qk_mgather,
                    qk_full_blocks, qk_cache_masks, indexer_l1_ring, indexer_key_pairs,
                    indexer_ws_load_order, hc_front_fusion))):
        raise ValueError("双AIV分工先独立验证T96向量索引与新输出所有权依赖")
    if qk_early_kv and (inputs["x_hc"]["shape"][0] != 96
            or not all((qk_vector_indices, row_dependencies, balanced_schedule))
            or any((qk_query_metadata, qk_all_valid, qk_direct_l1, qk_merge_heads,
                    qk_chunked_gather, qk_valid_gather, qk_scoped_zero, qk_mgather,
                    qk_full_blocks, qk_cache_masks, indexer_l1_ring, indexer_key_pairs,
                    indexer_ws_load_order))):
        raise ValueError("提前KV通知先独立验证T96现选向量索引与行依赖策略")
    if indexer_key_pairs and any((indexer_l1_ring, indexer_ws_load_order,
            qk_merge_heads, hc_front_fusion, hc_mix_resident, hc_mix_chunk_buffer)):
        raise ValueError("N64两面板Key预取只与已选策略组合")
    if indexer_ws_load_order and (inputs["x_hc"]["shape"][0] != 96
            or not balanced_schedule or not row_dependencies or any((
            indexer_l1_ring, indexer_key_pairs, indexer_n32_acc, indexer_transfer_2048, indexer_panel_group,
            indexer_rolled_panels, indexer_n80, indexer_n80_ws128, indexer_n128,
            indexer_page_views, indexer_leaf16384, indexer_score_pingpong, indexer_short_tail,
            indexer_scale_owners, indexer_input_dependencies, qproj_l1_columns,
            hc_front_fusion, hc_mix_resident, hc_mix_chunk_buffer,
            qk_merge_heads, drop_kv_touch, kv_touch_signal_only))):
        raise ValueError("WS加载顺序先独立验证T96现选控制，不组合其他新候选")
    if (indexer_l1_ring or indexer_key_pairs) and (inputs["x_hc"]["shape"][0] != 96 or any((
            indexer_n80, indexer_n80_ws128, indexer_n128, indexer_page_views,
            indexer_leaf16384, indexer_score_pingpong, indexer_short_tail,
            indexer_rolled_panels, indexer_panel_group, indexer_transfer_2048,
            qproj_l1_columns, indexer_scale_owners, indexer_input_dependencies,
            indexer_n32_acc, drop_kv_touch, kv_touch_signal_only))):
        raise ValueError("N64固定L1环先独立验证T96原叶计划，不组合其他新候选")
    if kv_touch_signal_only and (not balanced_schedule or not row_dependencies
            or inputs["x_hc"]["shape"][0] != 96 or any((drop_kv_touch,
            hc_front_fusion, indexer_n32_acc, indexer_transfer_2048, qproj_l1_columns,
            indexer_panel_group, indexer_short_tail, indexer_input_dependencies,
            indexer_scale_owners, precise_payload_cache))):
        raise ValueError("KV纯信号节点先保留T96现选编排，不组合其他新候选")
    if drop_kv_touch and (not balanced_schedule or not cost_profile or not row_dependencies
            or inputs["x_hc"]["shape"][0] != 96 or any((hc_front_fusion,
            indexer_n32_acc, indexer_transfer_2048, qproj_l1_columns, indexer_panel_group,
            indexer_short_tail, indexer_input_dependencies, indexer_scale_owners))):
        raise ValueError("删除KV原样回写标记先只组合T96已选行依赖/分核，不叠加新数学候选")
    if indexer_n32_acc and (inputs["x_hc"]["shape"][0] != 96 or any((
            indexer_n80, indexer_n80_ws128, indexer_n128, indexer_page_views,
            indexer_leaf16384, indexer_score_pingpong, indexer_short_tail,
            indexer_rolled_panels, indexer_panel_group, indexer_transfer_2048,
            qproj_l1_columns, indexer_scale_owners, indexer_input_dependencies))):
        raise ValueError("N32双Acc先独立检查T96原叶计划，不组合其他Indexer数学变体")
    if qproj_l1_columns and (qproj_workers != 20 or inputs["x_hc"]["shape"][0] != 96
            or any((qproj_n128, qproj_l0_prefetch, oproj_k512, oproj_n_tile != 128,
                    hc_front_fusion, hc_mix_resident, hc_mix_chunk_buffer,
                    indexer_transfer_2048, indexer_panel_group, indexer_short_tail,
                    indexer_scale_owners, indexer_input_dependencies))):
        raise ValueError("主Q双列驻L1先保持T96原分工，不组合其他数学候选")
    if indexer_transfer_2048 and any((indexer_n80, indexer_n80_ws128, indexer_n128,
            indexer_page_views, indexer_leaf16384, indexer_score_pingpong, indexer_short_tail,
            indexer_rolled_panels, indexer_panel_group, reuse_gm)):
        raise ValueError("2048分数交接只组合原N64/叶分片与独占工作区")
    if indexer_panel_group and (indexer_panel_group not in (2, 4, 8) or any((indexer_n80,
            indexer_n80_ws128, indexer_n128, indexer_page_views, indexer_leaf16384,
            indexer_score_pingpong, indexer_short_tail, indexer_rolled_panels))):
        raise ValueError("N64分组循环只允许2/4/8，不组合其他Indexer数学变体")
    if indexer_rolled_panels and any((indexer_n80, indexer_n80_ws128, indexer_n128,
            indexer_page_views, indexer_leaf16384, indexer_score_pingpong, indexer_short_tail)):
        raise ValueError("循环面板先保持原N64/叶分片，不组合其他Indexer数学变体")
    if indexer_short_tail and any((indexer_n80, indexer_n80_ws128, indexer_n128,
            indexer_page_views, indexer_leaf16384, indexer_score_pingpong)):
        raise ValueError("短尾先保持原N64/叶分片，不组合其他Indexer数学变体")
    if qk_all_valid and (not qk_vector_indices or any((qk_query_metadata, qk_direct_l1,
            qk_merge_heads, qk_chunked_gather, qk_valid_gather, qk_scoped_zero, qk_mgather,
            qk_full_blocks, qk_cache_masks, indexer_n80_ws128))):
        raise ValueError("QK全有效分支先保持现选数学、行搬运与控制")
    if indexer_n80_ws128 and (inputs["kv_seq_lens"]["shape"] != [16] or any((
            indexer_n80, indexer_n128, indexer_page_views, indexer_leaf16384,
            indexer_score_pingpong, qk_query_metadata, restore_weight_cache_policy))):
        raise ValueError("N80 WS K128先独立验证长历史B16，不叠加其他Indexer或新缓存候选")
    if qk_query_metadata and (not qk_vector_indices or any((qk_direct_l1, qk_merge_heads,
            qk_chunked_gather, qk_valid_gather, qk_scoped_zero, qk_mgather, qk_full_blocks,
            qk_cache_masks))):
        raise ValueError("每query地址缓存先只组合原QK地址缓存、原数学/搬运/同步")
    if hc_mix_chunk_buffer and (not row_dependencies or any((hc_front_fusion,
            hc_mix_resident, qproj_n128, qproj_l0_prefetch, indexer_n80,
            indexer_scale_owners, indexer_input_dependencies))):
        raise ValueError("HC分块UB缓冲先只组合已有行依赖/QK地址缓存")
    if indexer_input_dependencies and (not row_dependencies or any((hc_front_fusion,
            hc_mix_resident, qproj_n128, qproj_l0_prefetch, indexer_n80, indexer_scale_owners))):
        raise ValueError("Indexer输入行依赖先只组合已有行依赖和QK地址缓存")
    if indexer_scale_owners and (not balanced_schedule or hc_front_fusion or any((
            hc_mix_resident, indexer_n128, indexer_n80, indexer_leaf16384, indexer_score_pingpong,
            indexer_page_views, qproj_n128, qproj_l0_prefetch))):
        raise ValueError("scale页分工先单独组合原数学、分核及行依赖/QK地址缓存")
    if hc_front_tiles and not hc_front_fusion:
        raise ValueError("HC Tile直传要求先启用三阶段融合")
    if hc_front_fusion and (not balanced_schedule or hc_rows != 8 or hc_mix_resident
                            or aggregate_completion or task_completion or compile_schedule
                            or precise_payload_cache or reuse_gm or qproj_n128 or qproj_l0_prefetch):
        raise ValueError("HC前端融合只组合原8行数学、独立GM及已有行依赖/QK地址缓存")
    if qproj_l0_prefetch and (qproj_workers != 20 or qproj_n128 or oproj_k512 or oproj_n_tile != 128):
        raise ValueError("主Q L0预取不叠加其他逻辑IR重分片")
    if qproj_n128 and (qproj_workers != 20 or oproj_k512 or oproj_n_tile != 128):
        raise ValueError("主Q N128先不组合其他逻辑IR重分片")
    if hc_mix_resident and (not balanced_schedule or hc_rows != 8):
        raise ValueError("整行HC mix只修改自身分片，须重新排程且不叠加旧HC两阶段重分片")
    if indexer_n80 and (inputs["kv_seq_lens"]["shape"] != [16] or any((
            indexer_n128, indexer_page_views, indexer_leaf16384, indexer_score_pingpong))):
        raise ValueError("N80先只验证长历史B16，不叠加其他Indexer数学变换")
    if row_dependencies and (not balanced_schedule or any((aggregate_completion, task_completion,
            compile_schedule, precise_payload_cache, rotate_wait_order, reuse_gm, oproj_n_tile != 128,
            qproj_workers != 20, indexer_n128, qk_full_blocks, qk_cache_masks, qk_mgather,
            hc_rows != 8, indexer_page_views, qk_scoped_zero, qk_valid_gather, oproj_k512,
            qk_chunked_gather, indexer_leaf16384, indexer_score_pingpong, qk_direct_l1))):
        raise ValueError("行依赖只组合原数学/独占工作区与QK地址缓存，不能叠加其他变换")
    if task_completion and (aggregate_completion or compile_schedule or precise_payload_cache):
        raise ValueError("任务计数候选先单独验证，不与其他通知/边界变换组合")
    if rotate_wait_order and (aggregate_completion or compile_schedule or task_completion):
        raise ValueError("等待次序旋转只适用于原始逐核等待表")
    if oproj_n_tile not in (128, 256) or (oproj_n_tile == 256 and not balanced_schedule):
        raise ValueError("N256输出投影候选必须重新分配逻辑分片")
    if oproj_k512 and oproj_n_tile != 128:
        raise ValueError("K512候选保持N128，不与N256重分片混用")
    if reuse_gm and (aggregate_completion or task_completion):
        raise ValueError("临时 GM 复用先使用原逐核通知，保持控制区布局独立")
    if qproj_workers not in (20, 24, 32, 64, 128) or (qproj_workers != 20 and not balanced_schedule):
        raise ValueError("主Q投影重分片必须使用明确候选数并重新分核")
    if qk_full_blocks and not qk_vector_indices:
        raise ValueError("全有效块搬运需要已有QK地址缓存")
    if qk_mgather and (not qk_vector_indices or qk_full_blocks):
        raise ValueError("MGATHER要求已有QK地址缓存，且不能同时替换为全有效块循环")
    if qk_scoped_zero and (qk_mgather or qk_full_blocks):
        raise ValueError("初始化作用域候选先保留原逐行DMA")
    if qk_valid_gather and (not qk_vector_indices or qk_mgather or qk_full_blocks or qk_scoped_zero):
        raise ValueError("全有效块无清零候选需要QK地址缓存，并独立选择搬运版本")
    if qk_chunked_gather and (not qk_vector_indices or qk_mgather or qk_full_blocks
                              or qk_scoped_zero or qk_valid_gather):
        raise ValueError("32行分块搬运要求原QK地址缓存，不与其他搬运变换叠加")
    if hc_rows not in (2, 4, 8) or (hc_rows != 8 and not balanced_schedule):
        raise ValueError("HC行分片只支持2/4/8行，变更后须重新分核")
    if indexer_page_views and indexer_n128:
        raise ValueError("Key页视图候选先保持原N64分块")
    if indexer_leaf16384 and (indexer_page_views or indexer_n128):
        raise ValueError("Indexer叶容量候选先保持原N64及页读取数学体")
    if indexer_score_pingpong and (indexer_leaf16384 or indexer_page_views or indexer_n128):
        raise ValueError("Indexer双Acc候选先保持原叶计划，不叠加其他Indexer变换")
    batch = inputs["kv_seq_lens"]["shape"][0]
    tokens = inputs["x_hc"]["shape"][0]
    if tokens != batch * 6 or not 1 <= tokens <= 384:
        raise ValueError("当前数学接口要求 T=B×6，且 T<=384")
    if not isinstance(max_seq_len, int) or not 6 <= max_seq_len <= 131200:
        raise ValueError("max_seq_len 必须为有效的已有 Host 长度上界")
    # 这是 CPU 编排模拟器的策略输入，不是从 NPU 读取的样本值。只允许该
    # 一个根输入带合成 metadata；其余输入禁止参与 CPU 数值求值。
    for descriptor in inputs.values():
        descriptor.pop("metadata_values", None)
    inputs["kv_seq_lens"]["metadata_values"] = [max_seq_len] * batch
    bodies = catalog()
    rope_flat_gather_audit = None
    if rope_flat_gather:
        from rope_flat_gather import use_flat_gather

        bodies, rope_flat_gather_audit = use_flat_gather(
            bodies, Path(output_root) / "math_variants" / case_name / "rope_flat_gather")
    indexer_ws_order_audit = None
    if indexer_ws_load_order:
        from indexer_ws_load_order import reorder

        bodies, indexer_ws_order_audit = reorder(
            bodies, Path(output_root) / "math_variants" / case_name / "indexer_ws_load_order")
    bindings = bind(Path(bodies["logical_source_ir"]), inputs)
    source_schedule_audit = None
    try:
        anchor_name, anchor = choose_anchor(bindings)
    except MissingScheduleAnchor:
        if not allow_source_schedule:
            raise
        if not balanced_schedule or cost_profile is not None:
            raise ValueError("源码任务图需要重新分核；旧 task ID 的成本不能直接套用")
        from anchor_gap_design_v1.source_schedule import build_source_schedule

        references = {path.stem.removesuffix("_prepared"): json.loads(path.read_text())
                      for path in sorted((STATIC_ROOT / "bindings").glob("*_prepared.json"))}
        schedule, source_schedule_audit = build_source_schedule(
            bindings, bodies, references, policy=schedule_policy)
        anchor_name = "source_derived_task_dag"
        anchor = source_schedule_audit["cost_seed_anchor"]
    else:
        schedule = rebind_schedule(anchor_name, anchor, bindings)
    qk_role_split_audit = None
    if qk_role_split:
        from qk_role_split import use_roles

        bodies, qk_role_split_audit = use_roles(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "qk_roles",
            head_tile=16 if qk_role_wide_heads else 8,
            softmax_tile=32 if qk_role_wide_heads else 8, publication_tile=8)
    qk_early_kv_audit = None
    if qk_early_kv:
        from qk_early_kv import use_early_kv

        bodies, qk_early_kv_audit = use_early_kv(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "qk_early_kv")
    qk_index_cache = None
    if qk_vector_indices and not qk_role_split:
        from qk_index_cache import cache_qk_indices

        bodies, qk_index_cache = cache_qk_indices(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "qk_indices")
    qk_full_block_audit = None
    if qk_full_blocks:
        from qk_full_block import use_full_block

        bodies, qk_full_block_audit = use_full_block(
            bodies, Path(output_root) / "math_variants" / case_name / "qk_full_block")
    qk_mask_audit = None
    qk_mgather_audit = None
    qk_zero_audit = None
    qk_valid_audit = None
    qk_chunked_audit = None
    if qk_chunked_gather:
        from qk_chunked_gather import use_chunked_gather

        bodies, qk_chunked_audit = use_chunked_gather(
            bodies, Path(output_root) / "math_variants" / case_name / "qk_chunked_gather")
    if qk_valid_gather:
        from qk_valid_gather import use_valid_gather

        bodies, qk_valid_audit = use_valid_gather(
            bodies, Path(output_root) / "math_variants" / case_name / "qk_valid_gather")
    if qk_scoped_zero:
        from qk_scoped_zero import scope_zero

        bodies, qk_zero_audit = scope_zero(
            bodies, Path(output_root) / "math_variants" / case_name / "qk_scoped_zero")
    if qk_mgather:
        from qk_mgather import use_mgather

        bodies, qk_mgather_audit = use_mgather(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "qk_mgather")
    if qk_cache_masks:
        from qk_mask_cache import cache_qk_masks

        bodies, qk_mask_audit = cache_qk_masks(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "qk_mask_cache")
    qk_query_metadata_audit = None
    if qk_query_metadata:
        from qk_query_metadata import use_query_metadata

        bodies, qk_query_metadata_audit = use_query_metadata(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "qk_query_metadata")
    indexer_n128_audit = None
    indexer_n80_audit = None
    indexer_n80_ws128_audit = None
    qproj_n128_audit = None
    qproj_l0_prefetch_audit = None
    qproj_l1_columns_audit = None
    if qproj_l1_columns:
        from qproj_l1_columns import use_l1_columns

        bodies, qproj_l1_columns_audit = use_l1_columns(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "qproj_l1_columns")
    if qproj_l0_prefetch:
        from qproj_l0_prefetch import use_prefetch

        bodies, qproj_l0_prefetch_audit = use_prefetch(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "qproj_l0_prefetch")
    if qproj_n128:
        from qproj_n128 import use_n128

        bodies, qproj_n128_audit = use_n128(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "qproj_n128")
    hc_mix_chunk_buffer_audit = None
    if hc_mix_chunk_buffer:
        from hc_mix_chunk_buffer import use_chunk_buffer

        bodies, hc_mix_chunk_buffer_audit = use_chunk_buffer(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "hc_mix_chunk_buffer")
    hc_mix_resident_audit = None
    if hc_mix_resident:
        from hc_mix_resident import use_resident_mix

        bodies, hc_mix_resident_audit = use_resident_mix(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "hc_mix_resident")
    qk_merge_heads_audit = None
    qk_direct_l1_audit = None
    if qk_direct_l1:
        from qk_direct_l1 import use_direct_l1

        if any((qk_merge_heads, qk_chunked_gather, qk_valid_gather, qk_scoped_zero,
                qk_mgather, qk_full_blocks, qk_cache_masks)):
            raise ValueError("直接L1候选不与其他QK变换叠加")
        bodies, qk_direct_l1_audit = use_direct_l1(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "qk_direct_l1")
    if qk_merge_heads:
        from qk_merge_heads import split_pv_merge

        bodies, qk_merge_heads_audit = split_pv_merge(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "qk_merge_heads")
    indexer_leaf_audit = None
    indexer_pingpong_audit = None
    indexer_page_view_audit = None
    if indexer_page_views:
        from indexer_page_views import use_page_views

        bodies, indexer_page_view_audit = use_page_views(
            bodies, Path(output_root) / "math_variants" / case_name / "indexer_page_views")
    if indexer_n128:
        from indexer_n128 import use_indexer_n128

        bodies, indexer_n128_audit = use_indexer_n128(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "indexer_n128")
    if indexer_n80:
        from indexer_n80 import use_indexer_n80

        bodies, indexer_n80_audit = use_indexer_n80(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "indexer_n80")
    if indexer_n80_ws128:
        from indexer_n80_ws128 import use_ws128

        bodies, indexer_n80_ws128_audit = use_ws128(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "indexer_n80_ws128")
    if indexer_leaf16384:
        from indexer_leaf_capacity import use_larger_leaves

        bodies, indexer_leaf_audit = use_larger_leaves(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "indexer_leaf16384")
    if indexer_score_pingpong:
        from indexer_score_pingpong import use_score_pingpong

        bodies, indexer_pingpong_audit = use_score_pingpong(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "indexer_score_pingpong")
    retile_audit = None
    oproj_k_audit = None
    duration_scales = None
    if oproj_k512:
        from oproj_k512 import use_k512

        bodies, oproj_k_audit = use_k512(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "oproj_k512")
    if oproj_n_tile == 256:
        from oproj_retile import use_n256

        bodies, retile_audit = use_n256(
            schedule, bodies, Path(output_root) / "math_variants" / case_name)
        # 新列块尚无实测成本；初次分核按列数比例放大原成本，不把模型估计当收益。
        duration_scales = {item["kernel"]: 2.0 for item in retile_audit}
    qproj_partition = None
    if qproj_workers != 20:
        from qproj_partition import repartition

        bodies, qproj_partition = repartition(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "qproj", qproj_workers)
        duration_scales = {**(duration_scales or {}), "qproj_matmul": 20 / qproj_workers}
    hc_row_audit = None
    if hc_rows != 8:
        from hc_row_partition import repartition_rows

        bodies, hc_row_audit = repartition_rows(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "hc_rows", hc_rows)
        # Scalar循环长度不变；先保留至少一半原估计成本，不把行数缩减当实测加速。
        duration_scales = {**(duration_scales or {}),
                           "hc_widen_rms": max(0.5, hc_rows / 8),
                           "mix_x_rms_norm": max(0.5, hc_rows / 8)}
    scale_owner_audit = None
    if indexer_scale_owners:
        from indexer_scale_page_owner import use_page_owners

        bodies, scale_owner_audit = use_page_owners(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "scale_page_owners")
        duration_scales = {**(duration_scales or {}), "idx_kv_scale_commit": 0.5}
    hc_front_audit = None
    if hc_front_fusion:
        from hc_front_fusion import fuse_front

        bodies, anchor, cost_profile, hc_front_audit = fuse_front(
            schedule, anchor, bodies, Path(output_root) / "math_variants" / case_name / "hc_front_fusion",
            cost_profile)
        if hc_front_tiles:
            from hc_front_forward_tiles import forward_tiles

            bodies, tile_audit = forward_tiles(
                bodies, Path(output_root) / "math_variants" / case_name / "hc_front_tiles")
            hc_front_audit["tile_forwarding"] = tile_audit
    kv_touch_audit = None
    if drop_kv_touch:
        from kv_touch_elision import elide

        anchor, cost_profile, kv_touch_audit = elide(
            schedule, anchor, bodies, Path(output_root) / "control_variants" / case_name / "kv_touch_elision",
            cost_profile)
    if balanced_schedule:
        from balanced_schedule import rebalance

        rebalance(schedule, anchor, policy=schedule_policy, cost_profile=cost_profile,
                  priority_weight=priority_weight, duration_scales=duration_scales)
        if hc_front_fusion:
            schedule["balanced_schedule"]["math_task_dag_unchanged"] = False
            schedule["balanced_schedule"]["hc_three_stages_fused"] = True
            schedule["balanced_schedule"]["estimator"] = "原成员实测成本，融合HC用三个成本之和估计；不是设备预测"
        if drop_kv_touch:
            schedule["balanced_schedule"]["math_task_dag_unchanged"] = False
            schedule["balanced_schedule"]["identity_marker_elided"] = True
            schedule["balanced_schedule"]["retained_task_dependency_order_unchanged"] = True
    row_dependency_audit = None
    if row_dependencies:
        from row_dependencies import refine

        row_dependency_audit = refine(
            schedule, bodies, Path(output_root) / "control_variants" / case_name / "row_dependencies",
            indexer_inputs=indexer_input_dependencies, qk_roles=qk_role_split)
    qk_all_valid_audit = None
    indexer_short_tail_audit = None
    indexer_rolled_panels_audit = None
    indexer_panel_group_audit = None
    indexer_transfer_audit = None
    indexer_n32_acc_audit = None
    indexer_l1_ring_audit = None
    indexer_key_pairs_audit = None
    if indexer_key_pairs:
        from indexer_l1_key_pairs import use_key_pairs

        bodies, indexer_key_pairs_audit = use_key_pairs(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "indexer_key_pairs")
    if indexer_l1_ring:
        from indexer_l1_ring import use_l1_ring

        bodies, indexer_l1_ring_audit = use_l1_ring(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "indexer_l1_ring")
    if indexer_n32_acc:
        from indexer_n32_acc import use_n32_acc

        bodies, indexer_n32_acc_audit = use_n32_acc(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "indexer_n32_acc")
    if indexer_transfer_2048:
        from indexer_transfer_2048 import use_transfer_2048

        bodies, indexer_transfer_audit = use_transfer_2048(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "indexer_transfer_2048")
    if indexer_panel_group:
        from indexer_panel_groups import use_panel_groups

        bodies, indexer_panel_group_audit = use_panel_groups(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "indexer_panel_group",
            indexer_panel_group)
    if indexer_rolled_panels:
        from indexer_rolled_panels import use_rolled_panels

        bodies, indexer_rolled_panels_audit = use_rolled_panels(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "indexer_rolled_panels")
    if indexer_short_tail:
        from indexer_short_tail import use_short_tail

        bodies, indexer_short_tail_audit = use_short_tail(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "indexer_short_tail")
    if qk_all_valid:
        from qk_all_valid import use_all_valid

        # 先对原访问体核对行依赖。快路径只专门化已核对的块条件，回退体保留。
        bodies, qk_all_valid_audit = use_all_valid(
            schedule, bodies, Path(output_root) / "math_variants" / case_name / "qk_all_valid")
    if aggregate_completion:
        from completion_aggregation import aggregate, extend_layout, add_post_waits

        aggregate(schedule)
    layout = prepare(schedule, bodies)
    rotation = None
    if rotate_wait_order:
        from rotated_waits import rotate_waits

        rotation = rotate_waits(layout)
    protocol = extend_layout(schedule, layout) if aggregate_completion else validate(schedule, layout)
    gm_reuse = None
    if reuse_gm:
        from gm_reuse import reuse_equal_spans

        gm_reuse = reuse_equal_spans(schedule, layout)
        # 原校验器证明的是复用前布局，不能把它冒充复用后区域互不重叠。
        protocol["baseline_workspace_regions_disjoint"] = protocol.pop("workspace_regions_disjoint")
        protocol["workspace_lifetime_safe_reuse"] = gm_reuse["workspace_lifetime_safe_reuse"]
    program, specialized = generate(schedule, layout, bodies, workspace_binding="external",
                                     native_subviews_in_entry=True)
    source, layout, specialized, signal_audit = use_local_signals(jit_source(program), layout, specialized)
    source, looped_batches = loop_rebound_blocks(source, layout)
    # 归档辅助器描述的是当时 32 Tensor 的主线 ABI；本实验明确使用此前
    # 已验证的独立 256 Tensor runtime。容量描述不等于新候选已通过真机。
    leaf_contract = specialized["_entry_contract"]
    leaf_contract.update(legacy_core_limit=leaf_contract["current_core_limit"],
                         current_core_limit=256,
                         runtime_supported=leaf_contract["leaf_tensor_count"] <= 256,
                         runtime_scope="独立 Simpler 256 Tensor ABI；不代表共享安装能力")
    layout["abi"].update(legacy_runtime_core_tensor_limit=32, runtime_core_tensor_limit=256,
                         fits_current_runtime=leaf_contract["runtime_supported"],
                         source="独立 simpler-csa-core-args-256-20261016",
                         runtime_scope=leaf_contract["runtime_scope"])
    protocol["current_runtime_abi_supported"] = leaf_contract["runtime_supported"]
    task_counter_audit = None
    if task_completion:
        from task_completion import use_task_counters

        source, task_counter_audit = use_task_counters(source, layout, schedule)
    payload_cache = None
    if precise_payload_cache:
        from payload_cache import specialize_payload_cache

        source, payload_cache = specialize_payload_cache(source)
    if aggregate_completion:
        source = add_post_waits(source, layout)
    compiled_control = None
    if compile_schedule:
        from compiled_control import compile_controls

        source, compiled_control = compile_controls(source, layout)
    weight_cache_policy_audit = None
    if restore_weight_cache_policy:
        from weight_cache_policy import restore_policies

        source, weight_cache_policy_audit = restore_policies(source, schedule, bodies)
    kv_touch_signal_audit = None
    if kv_touch_signal_only:
        from kv_touch_signal_only import signal_only

        source, kv_touch_signal_audit = signal_only(source, layout)
    output_root = Path(output_root)
    case = output_root / "variants" / case_name
    case.mkdir(parents=True, exist_ok=False)
    (output_root / "bindings").mkdir(exist_ok=True)

    def save(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    save(output_root / "bindings" / f"{case_name}_prepared.json", schedule)
    save(case / "layout.json", layout)
    save(case / "entry.bindings.json", specialized)
    (case / "entry_jit.py").write_text(source)
    contract = {"batch": batch, "tokens": tokens, "query_tokens_per_request": 6,
                "anchor": anchor_name, "host_max_seq_len_for_strategy": max_seq_len,
                "source_schedule_audit": source_schedule_audit,
                "descriptors": descriptors, "selected_control_path": bindings["selected_control_path"],
                "per_step_device_metadata_preserved": True, "one_binary_for_all_batch_required": False,
                "wait_plan": schedule["wait_plan"], "cpu_protocol": protocol,
                "completion_aggregation": schedule.get("completion_aggregation"),
                "balanced_schedule": schedule.get("balanced_schedule"),
                "row_dependencies": schedule.get("row_dependencies"),
                "kv_touch_elision": kv_touch_audit,
                "kv_touch_signal_only": kv_touch_signal_audit,
                "compiled_control": compiled_control,
                "payload_cache": payload_cache,
                "task_completion": task_counter_audit,
                "wait_order_rotation": rotation,
                "oproj_retile": retile_audit,
                "gm_reuse": gm_reuse,
                "qproj_partition": qproj_partition,
                "qk_index_cache": qk_index_cache,
                "indexer_n128": indexer_n128_audit,
                "indexer_n80": indexer_n80_audit,
                "indexer_n80_ws128": indexer_n80_ws128_audit,
                "qproj_n128": qproj_n128_audit,
                "qproj_l0_prefetch": qproj_l0_prefetch_audit,
                "qproj_l1_columns": qproj_l1_columns_audit,
                "hc_mix_resident": hc_mix_resident_audit,
                "hc_front_fusion": hc_front_audit,
                "hc_mix_chunk_buffer": hc_mix_chunk_buffer_audit,
                "restored_weight_cache_policy": weight_cache_policy_audit,
                "indexer_scale_owners": scale_owner_audit,
                "indexer_leaf16384": indexer_leaf_audit,
                "indexer_score_pingpong": indexer_pingpong_audit,
                "qk_merge_heads": qk_merge_heads_audit,
                "qk_early_kv": qk_early_kv_audit,
                "qk_role_split": qk_role_split_audit,
                "rope_flat_gather": rope_flat_gather_audit,
                "qk_direct_l1": qk_direct_l1_audit,
                "qk_full_blocks": qk_full_block_audit,
                "qk_mgather": qk_mgather_audit,
                "qk_scoped_zero": qk_zero_audit,
                "oproj_k512": oproj_k_audit,
                "qk_valid_gather": qk_valid_audit,
                "qk_chunked_gather": qk_chunked_audit,
                "hc_row_partition": hc_row_audit,
                "indexer_page_views": indexer_page_view_audit,
                "qk_mask_cache": qk_mask_audit,
                "qk_query_metadata": qk_query_metadata_audit,
                "qk_all_valid": qk_all_valid_audit,
                "indexer_short_tail": indexer_short_tail_audit,
                "indexer_rolled_panels": indexer_rolled_panels_audit,
                "indexer_panel_group": indexer_panel_group_audit,
                "indexer_transfer_2048": indexer_transfer_audit,
                "indexer_n32_acc": indexer_n32_acc_audit,
                "indexer_l1_ring": indexer_l1_ring_audit,
                "indexer_key_pairs": indexer_key_pairs_audit,
                "indexer_ws_load_order": indexer_ws_order_audit,
                "signal_audit": signal_audit, "looped_batches": looped_batches, "device_validated": False,
                "performance_validated": False}
    save(case / "preparation.json", contract)
    return case, contract
