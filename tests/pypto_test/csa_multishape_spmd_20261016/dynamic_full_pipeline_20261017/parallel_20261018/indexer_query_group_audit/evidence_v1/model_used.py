#!/usr/bin/env python3
"""只读有限模型：直接解释正式 Scalar 几何，计工作量，不拟合延迟。

不导入 PyPTO/Torch、不编译 kernel、不访问设备。G4 只核算理论资源，
历史已否决的方向不生成新内核。leaf 候选仅改变计划与逻辑 leaf 置换。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace


WORKERS = 24
MAX_CANDIDATES = 262144
SOURCE_NAMES = (
    "policy.py", "request_geometry.py", "indexer_group_stages.py",
    "indexer_score_stages.py", "indexer_publish_stages.py",
)
FUNCTIONS = {
    "policy.py": ("select_policy",),
    "request_geometry.py": ("request_bounds",),
    "indexer_group_stages.py": (
        "indexer_long_leaf_plan", "indexer_group_plan_stage", "indexer_work_item",
        "indexer_leaf_range", "indexer_query_visibility",
    ),
    "indexer_publish_stages.py": ("indexer_visible_root_count",),
}


class Buffer:
    def __init__(self, data):
        self.data = list(data)


def read(tensor, indices):
    assert len(indices) == 1
    return tensor.data[indices[0]]


def write(tensor, indices, value):
    assert len(indices) == 1
    tensor.data[indices[0]] = value


def scalar_environment(source):
    """仅擦除注解/decorator，函数体 AST 原样交给普通 Python 解释器。"""
    pl = SimpleNamespace(
        INDEX="index", INT32="int32", cast=lambda x, dtype: x,
        min=min, max=max, range=range, read=read, write=write,
        tensor=SimpleNamespace(read=read, dim=lambda t, axis: len(t.data)),
    )
    env = dict(
        pl=pl, CSASPMDPolicy=lambda **kw: SimpleNamespace(**kw),
        SCORE_TILE=1024, CANDIDATES_PER_LEAF=8192, SCORE_WORKERS=24,
        MAX_CANDIDATES=MAX_CANDIDATES, MAX_LEAVES=32, COMPRESS_RATIO=4,
        GROUP_DESCRIPTOR_ALIGNMENT=16, GROUP_DESCRIPTOR_WORDS=4,
        TOPK_CANDIDATES_PER_LEAF=8192,
    )
    audit = []
    for name, wanted in FUNCTIONS.items():
        tree = ast.parse((source / name).read_text())
        found = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for fn_name in wanted:
            fn = found[fn_name]
            original = ast.dump(ast.Module(body=fn.body, type_ignores=[]), include_attributes=False)
            fn.decorator_list = []
            fn.returns = None
            for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
                arg.annotation = None
            module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
            assert original == ast.dump(ast.Module(body=fn.body, type_ignores=[]), include_attributes=False)
            exec(compile(module, str(source / name), "exec"), env)
            audit.append({"source": name, "function": fn_name, "line": fn.lineno, "body_ast_unchanged": True})
    return env, audit


def make_case(name, queries, histories, upper_query=None, stale=None):
    assert len(queries) == len(histories)
    bounds = [0]
    lengths, positions = [], []
    stale = set(stale or ())
    for req, (q, history) in enumerate(zip(queries, histories, strict=True)):
        bounds.append(bounds[-1] + q)
        lengths.append(history + q if q else 0)
        positions.extend(history + lane + (7 if req in stale else 0) for lane in range(q))
    return dict(
        name=name, queries=list(queries), histories=list(histories),
        upper_query=upper_query or max(queries, default=1),
        bounds=bounds, lengths=lengths, positions=positions, stale_requests=sorted(stale),
    )


def get_geometry(env, case, group, balanced, roots):
    tokens = len(case["positions"])
    b = Buffer(case["bounds"])
    lengths = Buffer(case["lengths"])
    positions = Buffer(case["positions"])
    descriptor_base = math.ceil(len(case["bounds"]) / 16) * 16
    offsets = Buffer([-991] * (descriptor_base + 4 * max(tokens, 1)))
    plan = Buffer([0] * 8)
    env["indexer_group_plan_stage"](b, lengths, positions, offsets, plan, tokens, 0, group, balanced, roots)
    groups = []
    for g in range(plan.data[0]):
        req, first, live, visible = offsets.data[descriptor_base + 4*g:descriptor_base + 4*g+4]
        qs = [env["indexer_query_visibility"](b, lengths, positions, req, first+k)[0] for k in range(live)]
        groups.append(dict(request=req, first=first, live=live, visible=visible, query_visible=qs))
    assert sum(x["live"] for x in groups) == tokens
    assert len(groups) == sum(math.ceil(q / group) for q in case["queries"] if q)
    return plan.data, groups


def propose_plan(plan):
    """受限 gcd 规则：延续现有至多 6 的倍数；不按 B28 白名单。

    只在整除关系适度时向上取整 leaf_count，不扩叶容量、不减少 leaf。
    此候选不宣称适用于 mixed 历史；模型会保留反例。
    """
    result = list(plan)
    groups, max_history = plan[:2]
    if not groups or not plan[6]:
        return result
    multiple = WORKERS // math.gcd(WORKERS, groups)
    tiles = math.ceil(max_history / 1024)
    base = max(math.ceil(max_history / 8192), 1)
    leaves = math.ceil(base / multiple) * multiple
    if 1 < multiple <= 6 and leaves <= min(32, tiles):
        result[2:5] = [leaves, max(tiles // leaves, 1), tiles % leaves]
    return result


def evaluate(env, groups, plan, group, panel, score_tile, rotated=False):
    count, _, leaves, leaf_tiles, extra = plan[:5]
    cores = [dict(steps=0, work_items=0, valid_leaves=0, leaf_drain=0,
                  query_rows_loaded=0, roots_written=0, score_columns=0) for _ in range(24)]
    seen = set()
    consumed_roots = 0
    merge_calls = 0
    for desc in groups:
        for visible in desc["query_visible"]:
            n = env["indexer_visible_root_count"](visible, Buffer(plan), plan[7], plan[6])
            consumed_roots += n
            merge_calls += math.ceil(max(n - 1, 0) / 3) if plan[6] else max(n - 1, 0)
    for item in range(count * leaves):
        owner = item % 24
        g, leaf = env["indexer_work_item"](item, count, leaves)
        if rotated and count >= 24:
            # 同 core 再次遇到同一 physical leaf 的 group 周期。
            group_period = 24 // math.gcd(24, leaves)
            leaf = (leaf + g // group_period) % leaves
        assert (g, leaf) not in seen
        seen.add((g, leaf))
        core, desc = cores[owner], groups[g]
        core["work_items"] += 1
        begin, capacity, valid = env["indexer_leaf_range"](leaf, leaf_tiles, extra, desc["visible"])
        if valid <= 0:
            continue
        assert capacity <= 8192
        steps = math.ceil(valid / score_tile)
        core["steps"] += steps
        core["valid_leaves"] += 1
        core["leaf_drain"] += min(steps, 2)
        # 尾组仍完整物理 M；padding 从本组首条有效 query 重读。
        core["query_rows_loaded"] += group
        core["roots_written"] += desc["live"] * plan[7]
        core["score_columns"] += steps * score_tile
    assert len(seen) == count * leaves
    total_steps = sum(x["steps"] for x in cores)
    total_columns = sum(x["score_columns"] for x in cores)
    valid_leaves = sum(x["valid_leaves"] for x in cores)
    qk_mac = total_columns * group * 64 * 128
    ws_mac = total_columns * 16 * group * 64
    max_steps = max(x["steps"] for x in cores)
    roots_written = sum(x["roots_written"] for x in cores)
    useful_query_columns = sum(sum(x["query_visible"]) for x in groups)
    histogram = {str(n): sum(x["steps"] == n for x in cores) for n in sorted({x["steps"] for x in cores})}
    return dict(
        plan=plan, rotated=rotated, group=group, panel=panel, score_tile=score_tile,
        active_cores=sum(x["steps"] > 0 for x in cores), total_steps=total_steps,
        max_steps=max_steps, mean_steps=total_steps / 24, steps_histogram=histogram,
        max_step_owners=[k for k, x in enumerate(cores) if x["steps"] == max_steps],
        total_work_items=count * leaves, valid_leaves=valid_leaves,
        empty_work_items=count * leaves - valid_leaves,
        total_qk_mac=qk_mac, total_ws_mac=ws_mac,
        max_core_qk_mac=max(x["score_columns"] for x in cores) * group * 64 * 128,
        qk_padding_ratio=(total_columns * group / useful_query_columns) if useful_query_columns else None,
        key_dma_page_submissions=total_columns // 32,
        key_dma_requested_bytes=total_columns * 128,
        qk_calls=total_columns // panel, ws_calls=total_columns // panel,
        coeff_prepare_bytes=count * 2048 * group,
        query_and_coeff_load_requested_bytes=valid_leaves * group * (8192 + 2048),
        roots_written=roots_written, root_store_bytes=roots_written * 4096,
        roots_consumed=consumed_roots, final_merge_calls=merge_calls,
        leaf_drain_waits=sum(x["leaf_drain"] for x in cores),
        steady_ack_waits=total_steps-sum(x["leaf_drain"] for x in cores),
        total_ack_waits=total_steps, ready_events_per_lane=total_steps,
        descriptor_scalar_reads_all_roles=12 * count * leaves,
        cores=cores,
    )


def policy_for(env, case):
    return env["select_policy"](
        tokens=max(len(case["positions"]), 1), requests=len(case["queries"]),
        max_query_tokens=case["upper_query"], max_seq_len=max(max(case["lengths"]), 8192),
    )


def run_case(env, case, force_group=None, balanced_override=None):
    p = policy_for(env, case)
    group = force_group or p.query_group
    balanced = (p.score_tile == 1024 and group == 6) if balanced_override is None else balanced_override
    roots = 1 if balanced else 2
    panel = 128 if group == 2 else 64
    plan, groups = get_geometry(env, case, group, balanced, roots)
    base = evaluate(env, groups, plan, group, panel, p.score_tile)
    candidate_plan = propose_plan(plan)
    unrotated = evaluate(env, groups, candidate_plan, group, panel, p.score_tile)
    candidate = evaluate(env, groups, candidate_plan, group, panel, p.score_tile, rotated=True)
    return dict(case=case, selected_policy=vars(p), group_descriptors=groups,
                baseline=base, new_leaves_no_rotation=unrotated, candidate=candidate)


def summary(record):
    old, new = record["baseline"], record["candidate"]
    keys = ("group", "panel", "plan", "active_cores", "total_steps", "max_steps",
            "mean_steps", "max_step_owners", "steps_histogram", "valid_leaves", "total_work_items",
            "total_qk_mac", "max_core_qk_mac", "qk_padding_ratio", "key_dma_requested_bytes",
            "query_and_coeff_load_requested_bytes", "roots_written", "roots_consumed",
            "final_merge_calls", "leaf_drain_waits", "total_ack_waits", "descriptor_scalar_reads_all_roles")
    return dict(name=record["case"]["name"],
                baseline={k: old[k] for k in keys}, candidate={k: new[k] for k in keys},
                no_rotation_max_steps=record["new_leaves_no_rotation"]["max_steps"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in SOURCE_NAMES:
        (snapshot / name).write_text((args.source / name).read_text())
    env, ast_audit = scalar_environment(snapshot)
    cases = []
    seven = [(131072, b) for b in (4, 8, 16, 24, 28)] + [(8192, b) for b in (16, 32)]
    for h, b in seven:
        for q in (1, 5, 6):
            cases.append(make_case(f"h{h}_b{b}_s{q}", [q]*b, [h]*b))
    for h in (8192, 32768, 131072):
        for b in (16, 28, 32):
            for pattern in ((1, 5), (1, 2, 3, 4, 5, 6), (6, 0), (1, 1, 1, 6)):
                q = [pattern[k % len(pattern)] for k in range(b)]
                cases.append(make_case(f"mixed_q_h{h}_b{b}_p{'-'.join(map(str,pattern))}", q, [h]*b, upper_query=6))
    records = [run_case(env, case) for case in cases]
    comparisons = []
    for case in cases:
        for group in (2, 4, 6):
            comparisons.append(summary(run_case(env, case, force_group=group)))
    # 固定有限矩阵：涵盖实际 group=1..96、多个 history 端点；不是性能样本。
    uniform = []
    for groups in range(1, 97):
        for history in (32768, 36864, 61440, 65536, 98304, 131072, 196608, 262144, 524288, 1048568):
            case = make_case(f"uniform_g{groups}_h{history}", [6]*groups, [history]*groups)
            uniform.append(summary(run_case(env, case, force_group=6)))
    # 不等长真实 visible，轮转长请求所在位置；不以 max_history 代替各组工作。
    mixed = []
    for b in (12, 16, 20, 24, 28, 30, 32, 36, 40, 48, 56, 64):
        for pattern in ((131072, 8192), (131072, 65536, 8192), (131072, 0, 8192), (131072, 131072, 131072, 8192)):
            for shift in range(min(b, 8)):
                histories = [pattern[(k+shift) % len(pattern)] for k in range(b)]
                queries = [6 if h else 0 for h in histories]
                case = make_case(f"mixed_h_b{b}_p{'-'.join(map(str,pattern))}_r{shift}", queries, histories, upper_query=6)
                mixed.append(summary(run_case(env, case, force_group=6)))
        for long_owner in range(b):
            histories = [8192]*b
            histories[long_owner] = 131072
            mixed.append(summary(run_case(env, make_case(f"one_long_b{b}_r{long_owner}", [6]*b, histories), force_group=6)))
    def classify(rows):
        improves = [x for x in rows if x["candidate"]["max_steps"] < x["baseline"]["max_steps"]]
        regresses = [x for x in rows if x["candidate"]["max_steps"] > x["baseline"]["max_steps"]]
        changed = [x for x in rows if x["candidate"]["plan"] != x["baseline"]["plan"]]
        return dict(cases=len(rows), plan_changed=len(changed), max_steps_improves=len(improves),
                    max_steps_regresses=len(regresses), max_steps_equal=len(rows)-len(improves)-len(regresses),
                    regression_examples=regresses[:8], improvement_examples=improves[:4])
    result = dict(
        scope="finite scalar/work model only; no floating point, synchronization latency, compiler or device proof",
        ast_interpretation=ast_audit, cases=[summary(x) for x in records],
        query_group_comparisons=comparisons, uniform_leaf_matrix=uniform,
        mixed_history_leaf_matrix=mixed, uniform_summary=classify(uniform), mixed_summary=classify(mixed),
        representative_full=[x for x in records if x["case"]["name"] in ("h131072_b28_s6", "h131072_b16_s5", "h8192_b32_s6")],
        source_inventory=[dict(path=str(args.source/name), size_bytes=(args.source/name).stat().st_size) for name in SOURCE_NAMES],
    )
    (args.output / "model_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    concise = dict(uniform_summary=result["uniform_summary"], mixed_summary=result["mixed_summary"],
                   seven_s6=[summary(x) for x in records if x["case"]["name"].endswith("_s6")])
    (args.output / "summary.json").write_text(json.dumps(concise, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({k: {f: v for f, v in value.items() if not f.endswith("examples")}
                      for k, value in (("uniform", result["uniform_summary"]),("mixed", result["mixed_summary"]))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
