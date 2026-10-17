# SPDX-License-Identifier: Apache-2.0
"""直接解释正式编排的phase/角色依赖；验证等待环、唯一planner和启动重叠。

数学helper替换为保留输出别名的完成事件，Score/Attention增加同组72角色握手。
它验证源代码的组合顺序，不模拟硬件cache、FFTS指令或数值；不能替代NPU验证。
不根据历史泳道图重建固定DAG，不导入PyPTO/torch，不编译第二份数学。
"""

import ast
import json
import random
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
INTERFACE = json.loads((HERE / "source_interface.json").read_text())


class CallsToEvents(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        node.returns = None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            return ast.YieldFrom(
                ast.Call(ast.Name("event", ast.Load()), [ast.Constant(node.func.id), *node.args], node.keywords)
            )
        return node


def owners(start, count, tile, workers, base, stride):
    if count <= 0:
        return []
    return [
        base + (n % workers) * stride
        for n in range(start // tile, min((start + count + tile - 1) // tile, start // tile + workers))
    ]


def run_case(
    tokens, q_lengths, histories, seed, query_group, output_group_pipeline=True, oa_cols=256,
    hc_row_dependencies=False,
):
    rng = random.Random(seed)
    phases = {"PHASE_" + k: v for k, v in INTERFACE["phases"].items()}
    tensor_names = {**INTERFACE["runtime_tensors"],**INTERFACE.get("internal_view_tensors",{})}
    tensors = {name: name for name in tensor_names}
    B = len(q_lengths)
    groups = sum((n + query_group - 1) // query_group for n in q_lengths)
    compact = sum((h + n) // 4 - h // 4 for h, n in zip(histories, q_lengths))
    metadata = {"idx_count": [compact], "main_count": [compact], "group_plan": [groups, 0, 0, 0, 0, 0, 0, 0]}
    completed = set()
    begun = set()
    entries = {}
    trace = []
    planner_writers = []
    tail_written = {"oa": set(), "quant": set(), "ob": set()}
    oa_per_group = ((tokens + 127) // 128) * (1024 // oa_cols)
    quant_per_group = (tokens + 31) // 32

    def dim(name, axis):
        if name in ("x", "x_fp32", "positions"):
            return tokens
        if name in ("idx_lengths", "main_lengths"):
            return B
        if name == "idx_scales":
            return tokens
        if name in ("cache_rows", "main_cache"):
            return 32768
        raise AssertionError((name, axis))

    output_args = {
        "hc_widen_rms_range": [1, 2],
        "hc_project_range": [2],
        "hc_reduce_range": [1],
        "hc_gates_range": [4, 5],
        "hc_comb_range": [4],
        "hc_mix_norm_range": [3, 4],
        "q_a_stage": [2],
        "kv_project_stage": [2],
        "q_b_stage": [2],
        "q_r_stage": [2, 3, 4, 5],
        "q_rope_prepare_stage": [1, 2],
        "attention_inverse_rope_sign_stage": [1],
        "kv_norm_rope_stage": [5],
        "original_kv_writeback_stage": [2],
        "q_dequant_rope_stage": [4],
        "compact_offsets_and_count_stage": [2, 3],
        "indexer_group_plan_stage": [3, 4],
        "indexer_query_project_stage": [2],
        "indexer_weights_project_stage": [2],
        "compressor_project_stage": [3, 4],
        "indexer_compressor_project_stage": [3, 4],
        "indexer_query_hadamard_stage": [2],
        "indexer_compressor_hadamard_stage": [2],
        "indexer_query_dequant_rope_stage": [5],
        "indexer_weights_reduce_stage": [1],
        "indexer_query_quant_stage": [1, 2],
        "indexer_head_coefficients_stage": [6],
        "compressor_pool_stage": [8],
        "compressor_state_commit_stage": [7],
        "compressor_norm_rope_cache_stage": [9, 10, 11],
        "indexer_compressor_pool_stage": [8],
        "indexer_compressor_state_commit_stage": [7],
        "indexer_compressor_norm_rope_stage": [8],
        "indexer_compact_quant_cache_stage": [6, 7, 8],
        "indexer_scale_page_commit_stage": [2],
        "indexer_scale_page_plan_stage": [6],
        "indexer_score_cube_stage": [11],
        "indexer_score_vector_lane_stage": [8, 9],
        "indexer_pair_publish_stage": [5, 6],
        "attention_slot_plan_stage": [6, 7, 8],
        "attention_qk_pv_cube_stage": [3, 5],
        "attention_qk_pv_vector_lane_stage": [13, 15, 17, 18, 19],
        "o_projection_a_stage_nz": [2],
        "o_projection_a_groups_stage_nz": [2],
        "o_projection_group_quant_stage": [1, 2],
        "o_projection_groups_quant_stage": [1, 2],
        "o_b_stage": [2],
        "o_b_groups_stage": [2],
        "o_dequant_hc_post_stage": [6],
    }

    def make_event(core):
        def event(name, *args):
            if name == "begin_invocation":
                assert args[1] == core and core not in begun
                begun.add(core)
                yield ("step", name)
                return args[0], 1
            if name == "publish_phase":
                phase, writer = args[1], args[2]
                assert writer == core and (phase, core) not in completed
                completed.add((phase, core))
                trace.append((core, name, phase))
                yield ("step", name)
                return args[0]
            if name == "wait_producer":
                yield ("wait", [(args[1], args[2])])
                return args[0]
            if name == "wait_row_owners":
                yield ("wait", [(args[1], o) for o in owners(*args[4:])])
                return args[0]
            if name in ("compact_offsets_and_count_stage", "indexer_group_plan_stage", "indexer_scale_page_plan_stage"):
                planner_writers.append(core)
                assert (phases["PHASE_INPUT"], core) in completed
                assert core == 24 + min((tokens + 7) // 8, 47)
            if name.startswith("indexer_score_") or name.startswith("attention_qk_pv_"):
                kind = "score" if name.startswith("indexer_score_") else "attention"
                group = core if core < 24 else (core - 24) // 2
                entries.setdefault((kind, group), set()).add(core)
                yield ("pair", (kind, group))
            # 按数学真实访问枚举HC数据依赖，不复用新等待公式。即使信号图无环，
            # 漏等某个M8输入、四份K片或对应RAW行，仍在数学消费入口直接失败。
            needed = set()
            if name == "hc_project_range":
                for task in range(core, ((tokens + 15) // 16) * 4, 24):
                    start = task // 4 * 16
                    for first in range(start, min(start + 16, tokens), 8):
                        needed.add((phases["PHASE_INPUT"], 24 + (first // 8) % 48))
            elif name == "hc_reduce_range":
                for rb in range(core - 24, (tokens + 15) // 16, 48):
                    for k in range(4):
                        needed.add((phases["PHASE_PARTIAL"], (rb * 4 + k) % 24))
            elif name in ("hc_gates_range", "hc_comb_range"):
                for rb in range(core - 24, (tokens + 7) // 8, 48):
                    needed.add((phases["PHASE_RAW"], 24 + (rb // 2) % 48))
                    needed.add((phases["PHASE_INPUT"], core))
            elif name == "hc_mix_norm_range" and core - 24 < (tokens + 7) // 8:
                needed.update(((phases["PHASE_GATES"], core), (phases["PHASE_INPUT"], core)))
            assert needed <= completed, (core, name, needed - completed)
            # 在源编排完成事件处核对数据依赖，而不只检查信号图没有环。
            # 每个输出组的O_A列块全部完成才可量化；全部量化块(含尾零owner)
            # 完成才可做该组O_B；最后HCpost必须读到八组全部partial。
            tail = {
                "o_projection_a_stage_nz": ("oa", oa_per_group, False),
                "o_projection_a_groups_stage_nz": ("oa", oa_per_group, True),
                "o_projection_group_quant_stage": ("quant", quant_per_group, False),
                "o_projection_groups_quant_stage": ("quant", quant_per_group, True),
                "o_b_stage": ("ob", 8, False),
                "o_b_groups_stage": ("ob", 8, True),
            }
            if name in tail:
                kind, per_group, grouped = tail[name]
                worker, workers = args[4:6]
                first_group, end_group = args[7:9] if grouped else (0, 8)
                for g in range(first_group, end_group):
                    start, end = g * per_group, (g + 1) * per_group
                    owned = [i for i in range(start, end) if i % workers == worker]
                    if owned:
                        if kind == "quant":
                            needed = set(range(g * oa_per_group, (g + 1) * oa_per_group))
                            assert needed <= tail_written["oa"], (core, name, needed - tail_written["oa"])
                        elif kind == "ob":
                            needed = set(range(g * quant_per_group, (g + 1) * quant_per_group))
                            assert needed <= tail_written["quant"], (core, name, needed - tail_written["quant"])
                    assert not set(owned) & tail_written[kind], (core, name, owned)
                    tail_written[kind].update(owned)
            if name == "o_dequant_hc_post_stage":
                assert tail_written["ob"] == set(range(64))
            trace.append((core, name, None))
            yield ("step", name)
            outputs = tuple(args[i] for i in output_args[name])
            return outputs[0] if len(outputs) == 1 else outputs

        return event

    controls = dict(
        QUERY_GROUP=query_group,
        QK_PANEL=64 if query_group == 6 else 128,
        SCORE_TILE=1024,
        PREFETCH=1,
        TO_L0=True,
        BALANCED=True,
        ROOTS=1,
        OA_COLS=oa_cols,
        OB_ROWS=32 if tokens <= 32 else 96 if tokens <= 96 else 128,
        OUTPUT_GROUP_PIPELINE=output_group_pipeline,
        HC_ROW_DEPENDENCIES=hc_row_dependencies,
        ATTENTION_QUERY_METADATA=True,
    )
    generators = {}
    for role in ("cube", "vector"):
        source = ast.parse((PKG / (role + "_schedule.py")).read_text())
        fn = next(n for n in source.body if isinstance(n, ast.FunctionDef))
        tree = ast.fix_missing_locations(ast.Module([CallsToEvents().visit(fn)], type_ignores=[]))
        for core in range(24) if role == "cube" else range(24, 72):
            group, lane = (core, 0) if core < 24 else ((core - 24) // 2, (core - 24) % 2)
            pl = SimpleNamespace(
                tile=SimpleNamespace(get_block_idx=lambda g=group: g, get_subblock_idx=lambda lane_id=lane: lane_id),
                tensor=SimpleNamespace(dim=dim),
                system=SimpleNamespace(set_ffts=lambda _x: None),
                func_attr=lambda _x: None,
                cast=lambda x, _t: int(x),
                read=lambda name, index: metadata[name][index[0]],
                min=min,
                max=max,
                range=range,
                INDEX="index",
                INT32="int32",
            )
            namespace = dict(pl=pl, GROUPS=24, VECTORS=48, CORES=72, event=make_event(core), **phases)
            exec(compile(tree, str(PKG / (role + "_schedule.py")), "exec"), namespace)
            function = namespace[role + "_schedule"]
            names = INTERFACE["roles"][role]["arguments"]
            # query几何复用只影响AIV，不为AIC引入无用的constexpr形参。
            parameters = {arg.arg for arg in fn.args.args}
            generators[core] = function(*(tensors[n] for n in names), **{k: v for k, v in controls.items() if k in parameters})

    waiting = {}
    finished = set()
    rounds = 0
    while len(finished) < 72:
        changed = False
        order = list(generators)
        rng.shuffle(order)
        for core in order:
            if core in finished:
                continue
            pending = waiting.get(core)
            if pending:
                kind, value = pending
                if kind == "wait" and not set(value) <= completed:
                    continue
                if kind == "pair" and len(entries[value]) != 3:
                    continue
            try:
                waiting[core] = next(generators[core])
            except StopIteration:
                finished.add(core)
                waiting.pop(core, None)
            changed = True
        rounds += 1
        if not changed:
            raise AssertionError({"T": tokens, "Q": q_lengths, "seed": seed, "blocked": waiting})
        assert rounds < 10000
    assert len(begun) == 72 and len(planner_writers) == 4 and len(set(planner_writers)) == 1
    assert sum(1 for _, name, _ in trace if name == "attention_slot_plan_stage") == 48
    assert tail_written["oa"] == set(range(8 * oa_per_group))
    assert tail_written["quant"] == set(range(8 * quant_per_group))
    assert tail_written["ob"] == set(range(64))
    # 源码有唯一metadata角色，但前段AIC只等widen；依赖集合不能含META。
    planner = 24 + min((tokens + 7) // 8, 47)
    if (tokens + 7) // 8 < 48:
        assert planner not in owners(0, tokens, 8, 48, 24, 1)
    return {
        "T": tokens,
        "B": B,
        "query_group": query_group,
        "output_group_pipeline": output_group_pipeline,
        "hc_row_dependencies": hc_row_dependencies,
        "oa_cols": oa_cols,
        "tail_unique_coverage": {name: len(writes) for name, writes in tail_written.items()},
        "planner": planner,
        "planner_idle_during_widen": (tokens + 7) // 8 < 48,
        "events": len(trace),
        "rounds": rounds,
    }


def main():
    cases = [(0, [], [])]
    for t in (1, 7, 8, 9, 23, 24, 25, 48, 96, 144, 168, 169, 192, 376, 377, 384, 385, 511, 512, 513):
        q = [6] * (t // 6) + ([t % 6] if t % 6 else [])
        cases.append((t, q, [131072] * len(q)))
    cases.extend(
        [
            (16, [2, 0, 7, 1], [8191, 0, 131071, 7]),
            (24, [0, 0, 0, 0], [0, 0, 0, 0]),
            (24, [1, 4, 3, 2], [131073, 32767, 8191, 0]),
        ]
    )
    rows = [
        run_case(t, q, h, seed, g, grouped, cols)
        for t, q, h in cases
        for g in (2, 6)
        for seed in range(3)
        for grouped in (False, True)
        for cols in (128, 256)
    ]
    result = {
        "passed": True,
        "source": "正式cube/vector_schedule.py AST",
        "cases": len(rows),
        "events": sum(r["events"] for r in rows),
        "rows": rows,
        "device_cache_validated": False,
        "numerical_accuracy_validated": False,
        "phase_dag_only": True,
        "hardware_ring_events_simulated": False,
    }
    (HERE / "schedule_model.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
