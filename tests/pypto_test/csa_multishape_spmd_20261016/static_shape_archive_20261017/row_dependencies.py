# SPDX-License-Identifier: Apache-2.0
"""按已核对的query/head读写范围收窄单SPMD成员依赖，暂限T96长Indexer。

这会有意放松原TMR整任务deps，不宣称与原任务偏序相同。数学体、核分配、
各核顺序、工作区与缓存维护不改；其他依赖保留。每个执行实例独占临时区。
Indexer叶数仍由device长度决定，依赖覆盖全部32个可能叶，不固定样本长度。
"""

import ast
import collections
import copy
from pathlib import Path

from .wait_plan import ancestor_sets


def _expressions(function):
    return {ast.unparse(n) for n in ast.walk(function)}


def source_contracts(catalog):
    """遇到数学体分片改变立即拒绝，不能静默套用旧的行所有权公式。"""
    tree = ast.parse(Path(catalog["body_bundle"]).read_text())
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    checks = {
        "indexer_score_topk_native_pair_aiv": [
            "pl.range(buf_worker_inline8783__ssa_v0, buf_query_count_inline8792__ssa_v0 // 6 * buf_max_leaves_inline8794__ssa_v1, 24)",  # noqa: E501 保留原数学表达式合同。
            "single_item_inline8866__idx_v0 % (buf_query_count_inline8792__ssa_v0 // 6) * 6",
            "single_item_inline8866__idx_v0 // (buf_query_count_inline8792__ssa_v0 // 6)",
            "buf_query_count_inline8792__ssa_v0 < 144",
            "(single_query_inline8669__phi_v2 + single_query_lane_inline8644__ssa_v3) * 64 + single_leaf_inline8684__phi_v2",  # noqa: E501 保留原数学表达式合同。
            "buf_score_lane_inline8716__ssa_v0 * 3",
            "buf_score_lane_inline8716__ssa_v0 * 3 + 2",
        ],
        "indexer_topk_query_merge__2": [
            "pl.range(worker__ssa_v0, query_count__ssa_v0, 48)",
            "query__idx_v0 * 64",
            "pl.tile.load(pair_arena__ssa_v0, [arena_base_inline735__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)",  # noqa: E501 保留原数学表达式合同。
        ],
        "csa_slots_build_valid_qk_plan": [
            "pl.range(plan_worker_inline4432__ssa_v0 * 8, t_dim_inline4335__ssa_v0, 128, init_values=(cmp_sparse_indices_inline4329__ssa_v0, sparse_bias_inline4301__ssa_v0))",  # noqa: E501 保留原数学表达式合同。
            "pl.min(t_dim_inline4335__ssa_v0 - bias_t0_inline4319__idx_v0, 8)",
            "bias_t0_inline4319__idx_v0 + bias_dt_inline4299__idx_v0",
            "pl.tile.load(index_indices__ssa_v0, [bias_t0_inline4319__idx_v0, 0], [8, 512], [bias_rows_inline4314__ssa_v0, 512], target_memory=pl.Mem.Vec)",  # noqa: E501 保留原数学表达式合同。
        ],
        "qk_pv_aic": [
            "(t_dim_inline4335__ssa_v0 - qk_core_inline4347__ssa_v0 + 23) // 24",
            "qk_core_inline4347__ssa_v0 + qk_tick_inline4364__idx_v0 // 5 * 24",
        ],
        "qk_pv_aiv": [
            "(t_dim_inline4335__ssa_v0 - qk_core_inline4347__ssa_v0 + 23) // 24",
            "qk_core_inline4347__ssa_v0 + pv_work_inline4417__ssa_v1 // 5 * 24",
            "qk_aiv_inline4434__ssa_v0 * 32",
            "m_h0_inline4238__ssa_v0 // 8 * 384 + pv_t_inline4420__ssa_v1",
            "n_pack_row_inline4226__ssa_v0 + 384",
            "m_h0_inline4238__ssa_v1 // 8 * 384 + pv_t_inline4420__ssa_v1",
            "n_pack_row_inline4226__ssa_v1 + 384",
        ],
        "proj_a_mm_0": [
            "pa_unit_inline616_inline4620__ssa_v0 // 8",
            "pa_rb_inline603_inline4636__ssa_v0 * 128",
            "pl.min(t_dim_inline4627__ssa_v0 - pa_r0_inline609_inline4640__ssa_v0, 128)",
            "row_base_o_inline4605__ssa_v0 + pa_r0_inline609_inline4640__ssa_v0",
        ],
    }
    for name, expected in checks.items():
        available = _expressions(functions[name])
        missing = set(expected) - available
        if missing:
            raise ValueError(f"行依赖分片合同改变：{name}: {missing}")
    cube = functions["indexer_score_topk_native_pair_aic"]
    cube_body = ast.Module(body=cube.body, type_ignores=[])
    if any(isinstance(n, ast.Name) and "pair_arena" in n.id for n in ast.walk(cube_body)):
        raise ValueError("Indexer Cube新增pair arena访问")
    return checks


def indexer_groups(queries, batch=16):
    return {(leaf * batch + q // 6) % 24 for q in queries for leaf in range(32)}


def prove_rows(tokens=96):
    """独立枚举实际写行，检查消费范围覆盖及不能等待错误的邻居核。"""
    indexer = collections.defaultdict(set)
    for leaf in range(32):
        for request in range(tokens // 6):
            worker = (leaf * (tokens // 6) + request) % 24
            for lane in range(2):
                for local in range(3):
                    indexer[worker, lane].add((request * 6 + lane * 3 + local) * 64 + leaf)
    checked = 0
    for worker in range(48):
        queries = set(range(worker, tokens, 48))
        parents = indexer_groups(queries)
        writes = set().union(*(indexer[p, lane] for p in parents for lane in range(2)))
        reads = {q * 64 + leaf for q in queries for leaf in range(32)}
        assert reads <= writes and len(parents) == 3
        for leaf_count in range(1, 33):
            assert {q * 64 + leaf for q in queries for leaf in range(leaf_count)} <= writes
            checked += 1
    topk = {w: set(range(w, tokens, 48)) for w in range(48)}
    plan = {w: {q for base in range(w * 8, tokens, 128) for q in range(base, min(base + 8, tokens))} for w in range(16)}
    qk = {w: set(range(w, tokens, 24)) for w in range(24)}
    for w, reads in plan.items():
        producers = {q % 48 for q in reads}
        assert reads <= set().union(*(topk[p] for p in producers))
    for w, reads in qk.items():
        producers = {q // 8 % 16 for q in reads}
        assert reads <= set().union(*(plan[p] for p in producers)) and len(producers) == 4
    packed = {}
    for w, queries in qk.items():
        for lane in range(2):
            groups = range(lane * 4, lane * 4 + 4)
            packed[w, lane] = {group * 384 + q for group in groups for q in queries}
    all_writes = [row for rows in packed.values() for row in rows]
    assert len(all_writes) == len(set(all_writes)) == 8 * tokens
    for group in range(8):
        parents = [(q % 24, group // 4) for q in range(tokens)]
        writes = set().union(*(packed[p] for p in parents))
        assert {group * 384 + q for q in range(tokens)} <= writes
    return {
        "indexer_query_leaf_count_cases": checked,
        "indexer_max_leaves_covered": 32,
        "query_rows": tokens,
        "topk_workers": 48,
        "plan_workers": 16,
        "qk_groups": 24,
        "packed_unique_rows": len(all_writes),
        "packed_owner": "AIV0_first32_AIV1_last32",
        "device_lengths_read_on_host": False,
        "numeric_or_hardware_proof": False,
    }


def refine(schedule, catalog):
    roots = schedule["root_arguments"]
    if roots["x_hc"]["shape"][0] != 96 or roots["kv_seq_lens"]["shape"] != [16]:
        raise ValueError("行依赖候选暂限B16/S6；其他档必须重核分片公式")
    contracts = source_contracts(catalog)
    coverage = prove_rows()
    tasks = {t["task_id"]: t for t in schedule["tasks"]}
    calls = schedule["call_bindings"]
    by_name = collections.defaultdict(list)
    for tid, call in calls.items():
        by_name[call["logical_function"]].append(tid)
    identifiers = {}
    for label, name in (
        ("indexer", "indexer_score_topk_native_pair"),
        ("merge", "indexer_topk_query_merge__2"),
        ("plan", "csa_slots_build_valid_qk_plan"),
        ("qk", "qk_pv"),
    ):
        if len(by_name[name]) != 1:
            raise ValueError("不是T96长Indexer的唯一调用：" + name)
        identifiers[label] = by_name[name][0]
    (idx, merge, plan, qk) = (identifiers[k] for k in ("indexer", "merge", "plan", "qk"))
    projections = by_name["proj_a_mm_0"]
    if len(projections) != 8:
        raise ValueError("O-A分组数量改变")
    bindings = {tid: {a["parameter"]: a["value"] for a in call["arguments"]} for (tid, call) in calls.items()}
    links = [
        (idx, "pair_arena_inline3226_inline4202__ssa_v0", merge, "pair_arena__ssa_v0"),
        (merge, "topk_indices__ssa_v0", plan, "index_indices__ssa_v0"),
        (plan, "cmp_sparse_indices_inline4329__ssa_v0", qk, "cmp_sparse_indices_inline4329__rv_v2"),
        (plan, "sparse_bias_inline4301__ssa_v0", qk, "sparse_bias_inline4301__rv_v2"),
        (plan, "valid_block_mask_inline4413__ssa_v0", qk, "valid_block_mask_inline4413__ssa_v0"),
    ] + [(qk, "o_packed_heads__ssa_v0", tid, "o_packed_heads__ssa_v1") for tid in projections]
    for producer, pname, consumer, cname in links:
        (left, right) = (bindings[producer][pname], bindings[consumer][cname])
        for key in ("allocation", "offset", "shape", "strides", "dtype", "layout"):
            if left[key] != right[key]:
                raise ValueError(f"行依赖的视图不是同一地址映射：{pname}/{cname}/{key}")
    allowed = {
        (idx, merge): {bindings[idx]["pair_arena_inline3226_inline4202__ssa_v0"]["allocation"]},
        (merge, plan): {bindings[merge]["topk_indices__ssa_v0"]["allocation"]},
        (plan, qk): {
            bindings[plan][n]["allocation"]
            for n in (
                "cmp_sparse_indices_inline4329__ssa_v0",
                "sparse_bias_inline4301__ssa_v0",
                "valid_block_mask_inline4413__ssa_v0",
            )
        },
    }
    for tid in projections:
        allowed[qk, tid] = {bindings[qk]["o_packed_heads__ssa_v0"]["allocation"]}
    shared_refs = {}
    for edge in allowed:
        (a, b) = edge
        refs = [
            {v["allocation"] for v in bindings[tid].values() if isinstance(v, dict) and "allocation" in v}
            for tid in edge
        ]
        shared = refs[0] & refs[1]
        extra = shared - allowed[edge]
        if any(not name.startswith("argument:") for name in extra):
            raise ValueError("收窄边还共享其他临时区：" + str(extra))
        shared_refs[f"{a}->{b}"] = {"products": sorted(allowed[edge]), "shared_root_inputs": sorted(extra)}
    nodes = [r for rows in schedule["per_core_sequences"].values() for r in rows]
    ids = {(str(r["core"]), r["dispatch_seq"]): i for (i, r) in enumerate(nodes)}
    members = collections.defaultdict(list)
    for i, row in enumerate(nodes):
        members[row["task"]].append(i)
    original = [set() for _ in nodes]
    for i, row in enumerate(nodes):
        if row["dispatch_seq"] > 1:
            original[i].add(ids[str(row["core"]), row["dispatch_seq"] - 1])
        for parent in tasks[row["task"]]["predecessors"]:
            original[i].update(members[parent])
    (_, original_ancestors) = ancestor_sets(original)
    required = copy.deepcopy(original)
    changes = []
    for i, row in enumerate(nodes):
        (tid, block) = (row["task"], row["logical_block"])
        if tid == merge:
            parent = idx
            workers = indexer_groups(range(block, 96, 48))
            selected = {p for p in members[parent] if nodes[p]["logical_block"] in workers}
            reason = "两query的全部32可能叶；保留每个相关Indexer的完整MIX核组"
        elif tid == plan:
            parent = merge
            rows = [q for base in range(block * 8, 96, 128) for q in range(base, min(base + 8, 96))]
            workers = {q % 48 for q in rows}
            selected = {p for p in members[parent] if nodes[p]["logical_block"] in workers}
            reason = "本plan分片的最多8个query；空分片不读TopK"
        elif tid == qk:
            parent = plan
            workers = {q // 8 % 16 for q in range(block, 96, 24)}
            selected = {p for p in members[parent] if nodes[p]["logical_block"] in workers}
            reason = "本QK组四个query的plan写入者；其他KV/Q/RoPE父任务保持"
        elif tid in projections:
            parent = qk
            group = bindings[tid]["g_inline4648__idx_v0"]
            if (
                not isinstance(group, int)
                or not 0 <= group < 8
                or bindings[tid]["row_base_o_inline4605__ssa_v0"] != group * 384
            ):
                raise ValueError("O-A head组和packed行映射改变")
            first = block // 8 * 128
            workers = {q % 24 for q in range(first, min(first + 128, 96))}
            lane = group // 4
            selected = {
                p
                for p in members[parent]
                if nodes[p]["logical_block"] in workers and nodes[p]["original_subslot"] == f"AIV{lane}"
            }
            reason = "仅对应head组的packed行生产者；每核自身Cube先后仍保留"
        else:
            continue
        if parent not in tasks[tid]["predecessors"]:
            raise ValueError("待收窄的原task边不存在")
        own_previous = ids.get((str(row["core"]), row["dispatch_seq"] - 1))
        required[i].difference_update(members[parent])
        required[i].update(selected)
        if own_previous is not None:
            required[i].add(own_previous)
        changes.append(
            {
                "node": i,
                "core": row["core"],
                "sequence": row["dispatch_seq"],
                "task": tid,
                "parent_task": parent,
                "logical_block": block,
                "old_parent_members": len(members[parent]),
                "new_parent_members": len(selected),
                "selected_nodes": sorted(selected),
                "reason": reason,
            }
        )
    # 同组 FFTS ID 被复用，QK 前保留本组 Indexer 三成员完成边。
    resource_edges = 0
    for i in members[qk]:
        core = nodes[i]["core"]
        group = core if core < 24 else (core - 24) // 2
        group_cores = {group, 24 + group * 2, 25 + group * 2}
        previous = {p for p in members[idx] if nodes[p]["core"] in group_cores}
        if len(previous) != 3:
            raise ValueError("本物理组的Indexer MIX成员不完整")
        required[i].update(previous)
        resource_edges += len(previous)
    (_, ancestors) = ancestor_sets(required)
    for i, bits in enumerate(ancestors):
        if bits & ~original_ancestors[i]:
            raise ValueError("收窄反而新增了原图不保证的计算先后")
    for i, row in enumerate(nodes):
        for parent in tasks[row["task"]]["predecessors"]:
            if (parent, row["task"]) not in allowed:
                if any(not ancestors[i] >> p & 1 for p in members[parent]):
                    raise ValueError("误删未获证明的原task边")
    (order, _) = ancestor_sets(required)
    positions = {n: k for (k, n) in enumerate(order)}
    reduced = [set() for _ in nodes]
    for i, parents in enumerate(required):
        covered = 0
        for parent in sorted(parents, key=positions.get, reverse=True):
            if not covered >> parent & 1:
                reduced[i].add(parent)
                covered |= ancestors[parent] | 1 << parent
    published = collections.defaultdict(set)
    for core, sequence in schedule["per_core_sequences"].items():
        observed = collections.defaultdict(int)
        for row in sequence:
            needed = collections.defaultdict(int)
            for parent in reduced[ids[core, row["dispatch_seq"]]]:
                before = nodes[parent]
                other = str(before["core"])
                if other != core:
                    needed[other] = max(needed[other], before["dispatch_seq"])
            row["wait_for_core_progress"] = {
                k: v for (k, v) in sorted(needed.items(), key=lambda p: int(p[0])) if v > observed[k]
            }
            for other, step in row["wait_for_core_progress"].items():
                observed[other] = step
                published[other].add(step)
    translated = [set() for _ in nodes]
    for i, row in enumerate(nodes):
        (core, seq) = (str(row["core"]), row["dispatch_seq"])
        row["publish_completion"] = seq in published[core]
        if seq > 1:
            translated[i].add(ids[core, seq - 1])
        translated[i].update(ids[p, step] for (p, step) in row["wait_for_core_progress"].items())
    (_, translated_ancestors) = ancestor_sets(translated)
    if translated_ancestors != ancestors:
        raise ValueError("进度表翻译不保持已证明的成员图")
    prior = copy.deepcopy(schedule["wait_plan"])
    wait_plan = {
        "remote_core_progress_requirements": sum(len(r["wait_for_core_progress"]) for r in nodes),
        "completion_publications": sum(r["publish_completion"] for r in nodes),
        "original_member_edges": sum(map(len, original)),
        "required_member_edges": sum(map(len, required)),
        "planned_member_edges": sum(map(len, translated)),
        "identical_dependency_partial_order": False,
        "refined_partial_order_preserved": True,
        "full_task_deps_replaced_only_under_explicit_row_contracts": True,
        "global_barriers_added": 0,
    }
    schedule["wait_plan"] = wait_plan
    if "balanced_schedule" in schedule:
        schedule["balanced_schedule"].update(
            math_task_dag_unchanged=False,
            member_dependencies_refined_after_placement=True,
            original_task_descriptors_retained_for_provenance=True,
        )
    audit = {
        "prior_wait_plan": prior,
        "new_wait_plan": wait_plan,
        "source_contracts": contracts,
        "indexer_input_dependencies": None,
        "row_coverage": coverage,
        "changed_members": changes,
        "shared_views": shared_refs,
        "ffts_group_resource_edges": resource_edges,
        "original_core_order_and_mapping_preserved": True,
        "all_unmodified_task_edges_preserved": True,
        "math_bodies_changed": False,
        "cache_maintenance_changed": False,
        "gm_allocations_reused": False,
        "all_legal_dynamic_leaf_counts_covered": True,
        "device_validated": False,
    }
    schedule["row_dependencies"] = {
        k: v for (k, v) in audit.items() if k not in ("source_contracts", "changed_members")
    }
    return audit
