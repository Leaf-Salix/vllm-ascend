#!/usr/bin/env python3
"""保持 leaf/root 数学，仅把短尾叶分给少一个重叶的核；有限计数模型。

精确解释原 Scalar 几何。工作分配不假定 NPU 延迟、FFTS 免费或相同历史。
"""

import argparse
import json
import math
from pathlib import Path

from model import evaluate, get_geometry, make_case, policy_for, scalar_environment


def choose_owner_mode(groups, plan, score_tile):
    count, _, leaves, leaf_tiles, extra = plan[:5]
    if not plan[6] or plan[7] != 1 or score_tile != 1024:
        return {"enabled": False, "reason": "only_existing_balanced_single_root"}
    if count < 24 or leaves < 2 or leaf_tiles != 8 or extra != 0:
        return {"enabled": False, "reason": "preserve_existing_small_group_or_rebalanced_plan"}
    heavy_total = count * (leaves - 1)
    remainder = heavy_total % 24
    if not remainder:
        return {"enabled": False, "reason": "no_core_has_one_fewer_heavy_leaf"}
    tail_begin = (leaves - 1) * 8192
    min_visible = min(x["visible"] for x in groups)
    max_visible = max(x["visible"] for x in groups)
    # 这是关键动态 gate：所有组的前面叶必须真的满，不能用 max_history 冒充。
    if min_visible < tail_begin:
        return {"enabled": False, "reason": "mixed_history_or_stale_group_has_incomplete_heavy_leaf",
                "min_group_visible": min_visible, "last_leaf_begin": tail_begin}
    tail_steps = math.ceil(max(max_visible - tail_begin, 0) / score_tile)
    light_cores = 24 - remainder
    max_tail_per_core = math.ceil(count / light_cores)
    if not tail_steps or max_tail_per_core * tail_steps > 8:
        return {"enabled": False, "reason": "tail_does_not_fit_one_full_leaf_gap",
                "tail_steps_upper": tail_steps, "tail_leaves_per_light_core_upper": max_tail_per_core}
    return dict(enabled=True, reason="all_heavy_leaves_full_and_tail_fits_gap", heavy_total=heavy_total,
                remainder=remainder, light_cores=light_cores, min_group_visible=min_visible,
                max_group_visible=max_visible, tail_steps_upper=tail_steps,
                tail_leaves_per_light_core_upper=max_tail_per_core,
                tail_step_slack=8-max_tail_per_core*tail_steps)


def owner_map(env, groups, plan, mode):
    count, _, leaves = plan[:3]
    if not mode["enabled"]:
        return [(item % 24, *env["indexer_work_item"](item, count, leaves)) for item in range(count*leaves)]
    heavy_total, rem = mode["heavy_total"], mode["remainder"]
    light_cores = 24-rem
    placements = []
    for core in range(24):
        heavy_count = max((heavy_total-core+23)//24, 0)
        tail_count = max((count-(core-rem)+light_cores-1)//light_cores, 0) if core >= rem else 0
        for local in range(heavy_count+tail_count):
            if local < heavy_count:
                flat = core+local*24
                group, leaf = divmod(flat, leaves-1)
            else:
                group = core-rem+(local-heavy_count)*light_cores
                leaf = leaves-1
            assert 0 <= group < count
            placements.append((core, group, leaf))
    assert sorted((g, l) for _, g, l in placements) == [(g, l) for g in range(count) for l in range(leaves)]
    return placements


def run(env, case, force_group=None):
    policy = policy_for(env, case)
    group = force_group or policy.query_group
    balanced = policy.score_tile == 1024 and group == 6
    roots = 1 if balanced else 2
    plan, groups = get_geometry(env, case, group, balanced, roots)
    panel = 128 if group == 2 else 64
    mode = choose_owner_mode(groups, plan, policy.score_tile)
    old = evaluate(env, groups, plan, group, panel, policy.score_tile)
    placements = owner_map(env, groups, plan, mode)
    new = evaluate(env, groups, plan, group, panel, policy.score_tile, placements=placements)
    # 变更仅为 owner；所有真正工作的 leaf、根数、算术量与消息总数保持。
    equal_fields = ("plan", "total_steps", "total_qk_mac", "total_ws_mac", "valid_leaves",
                    "total_work_items", "qk_calls", "ws_calls", "key_dma_requested_bytes",
                    "query_and_coeff_load_requested_bytes", "roots_written", "roots_consumed",
                    "final_merge_calls", "leaf_drain_waits", "total_ack_waits", "descriptor_scalar_reads_all_roles")
    assert all(old[k] == new[k] for k in equal_fields)
    if mode["enabled"]:
        assert new["max_steps"] <= old["max_steps"]
    return dict(case=case, mode=mode, baseline=old, candidate=new,
                mapping=placements, work_and_root_counts_identical=True)


def concise(record):
    return dict(name=record["case"]["name"], mode=record["mode"],
                old_max_steps=record["baseline"]["max_steps"],
                new_max_steps=record["candidate"]["max_steps"],
                old_max_work_items=max(x["work_items"] for x in record["baseline"]["cores"]),
                new_max_work_items=max(x["work_items"] for x in record["candidate"]["cores"]),
                old_max_leaf_drain=max(x["leaf_drain"] for x in record["baseline"]["cores"]),
                new_max_leaf_drain=max(x["leaf_drain"] for x in record["candidate"]["cores"]),
                old_max_roots_written=max(x["roots_written"] for x in record["baseline"]["cores"]),
                new_max_roots_written=max(x["roots_written"] for x in record["candidate"]["cores"]),
                total_steps=record["baseline"]["total_steps"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env, audit = scalar_environment(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    # 固定 history 端点：短尾 1/2/4/8 step、刚跨 leaf、接近容量上限。
    histories = (32768, 36864, 45056, 61440, 65536, 98304, 131072, 135168,
                 143360, 159744, 196608, 262144, 524288, 1048568)
    for groups in range(1, 97):
        for h in histories:
            cases.append((make_case(f"uniform_g{groups}_h{h}", [6]*groups, [h]*groups), 6))
    # 实际七档/S1/S5策略，以及 Host 上界6但请求长度不等的情形。
    for h, b in [(131072, b) for b in (4,8,16,24,28)]+[(8192,b) for b in (16,32)]:
        for q in (1,5,6):
            cases.append((make_case(f"official_h{h}_b{b}_s{q}", [q]*b, [h]*b), None))
        for pattern in ((1,5),(1,2,3,4,5,6),(6,0),(1,1,1,6)):
            qs=[pattern[k%len(pattern)] for k in range(b)]
            cases.append((make_case(f"mixed_q_h{h}_b{b}_p{pattern}", qs, [h]*b, upper_query=6), None))
    for b in (12,16,20,24,28,30,32,36,40,48,56,64):
        for pattern in ((131072,8192),(131072,65536,8192),(131072,0,8192),
                        (131072,131072,131072,8192),(131072,131068)):
            for shift in range(min(b,8)):
                hs=[pattern[(k+shift)%len(pattern)] for k in range(b)]
                qs=[6 if h else 0 for h in hs]
                cases.append((make_case(f"mixed_h_b{b}_p{pattern}_r{shift}", qs, hs, upper_query=6), None))
        for owner in range(b):
            hs=[8192]*b
            hs[owner]=131072
            cases.append((make_case(f"one_long_b{b}_r{owner}",[6]*b,hs),None))
            cases.append((make_case(f"one_stale_b{b}_r{owner}",[6]*b,[131072]*b,stale=[owner]),None))
    summaries, representatives = [], []
    total_enabled=0
    for case, group in cases:
        result=run(env,case,group)
        summaries.append(concise(result))
        total_enabled += result["mode"]["enabled"]
        if case["name"] in ("official_h131072_b28_s6", "official_h131072_b16_s6", "one_long_b28_r0", "one_stale_b28_r0"):
            representatives.append(result)
    summary = dict(cases=len(cases), enabled=total_enabled,
                   max_steps_improves=sum(x["new_max_steps"]<x["old_max_steps"] for x in summaries),
                   max_steps_equal=sum(x["new_max_steps"]==x["old_max_steps"] for x in summaries),
                   max_steps_regresses=sum(x["new_max_steps"]>x["old_max_steps"] for x in summaries),
                   complete_leaf_coverage_unique=True, unchanged_root_and_arithmetic_counts=True,
                   limitation="Finite integer work/placement model; not a timing, ISA, floating-point or device proof.")
    result=dict(summary=summary,scalar_ast_audit=audit,cases=summaries,representatives=representatives,
                fixed_item_count_lower_bound_b28=dict(
                    heavy_items=112,heavy_steps=8,tail_items=28,tail_steps=1,
                    six_item_cores=20,five_item_cores=4,
                    heavy_capacity_if_max40=20*4+4*5,
                    required_heavy=112,lower_bound_max_steps=41,
                    unrestricted_owner_lower_bound=40))
    (args.output/"owner_model_result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    main()
