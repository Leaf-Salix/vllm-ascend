# SPDX-License-Identifier: Apache-2.0
"""从实际编排AST验证Attention入口获取的传递闭包，不模拟设备缓存或数学。

每核有自己的已获取phase集合。发布时保存该核此前获取的完整闭包，等待成功
后把生产者闭包传给消费者。这样检查的是happens-before，而不只是某次随机
交错恰好看见所有生产者已完成。数学helper只产生完成事件，FFTS以同组角色
配对建模；输出数值、硬件可见性、环槽事件和性能仍须真机完整验证。
"""

import ast
import json
import random
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
PHASE_CALLS = {"begin_invocation", "publish_phase", "wait_producer", "wait_row_owners"}
REQUIRED = ("PHASE_QUERY", "PHASE_ORIGINAL_CACHE", "PHASE_MAIN_OUTPUT", "PHASE_ROPE")


class ScheduleEvents(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list = []
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        # 数学helper仅写原Tensor；保留其事件，不需要构造另一份数学或真实Tensor。
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id not in PHASE_CALLS:
            return ast.Expr(self.visit_Call(value))
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            return ast.YieldFrom(ast.Call(ast.Name("event", ast.Load()), [ast.Constant(node.func.id), *node.args], []))
        return node


def owners(first, count, tile, workers, base, stride):
    return (
        [
            base + (index % workers) * stride
            for index in range(first // tile, min((first + count + tile - 1) // tile, first // tile + workers))
        ]
        if count > 0
        else []
    )


def run_case(tokens, query_lengths, histories, seed, enabled):
    constants = {}
    for statement in ast.parse((PKG / "runtime_abi.py").read_text()).body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name.startswith("PHASE_"):
                constants[name] = ast.literal_eval(statement.value)
    knowledge = {core: set() for core in range(72)}
    published = {}
    entered = {}
    begun = set()
    attention_reads = 0
    plan_closures = 0
    wait_slots = 0
    group = 6
    metadata = {
        "idx_count": sum((h + q) // 4 - h // 4 for h, q in zip(histories, query_lengths)),
        "group_plan": sum((q + group - 1) // group for q in query_lengths),
    }
    all_payloads = {(constants[phase], core) for phase in REQUIRED for core in range(24, 72)}

    def dim(name, _axis):
        if name in ("x", "x_fp32", "positions", "idx_scales"):
            return tokens
        if name in ("main_lengths", "idx_lengths"):
            return len(query_lengths)
        if name in ("cache_rows", "main_cache"):
            return 32768
        raise AssertionError(name)

    def make_event(core):
        def event(name, *args):
            nonlocal attention_reads, plan_closures, wait_slots
            if name == "begin_invocation":
                assert args[1] == core and core not in begun
                begun.add(core)
                yield "step", None
                return args[0], 1
            if name == "publish_phase":
                phase, writer = args[1:3]
                assert writer == core and (phase, core) not in published
                knowledge[core].add((phase, core))
                published[phase, core] = frozenset(knowledge[core])
                if phase == constants["PHASE_ATTN_PLAN"]:
                    # 包括无真实请求/无工作量的AIV，不能只检查活跃plan写者。
                    assert all_payloads <= knowledge[core], (core, all_payloads - knowledge[core])
                    plan_closures += 1
                yield "step", None
                return args[0]
            if name in ("wait_producer", "wait_row_owners"):
                dependencies = (
                    [(args[1], args[2])]
                    if name == "wait_producer"
                    else [(args[1], owner) for owner in owners(*args[4:])]
                )
                wait_slots += len(dependencies)
                yield "wait", dependencies
                for dependency in dependencies:
                    assert dependency in published
                    knowledge[core].update(published[dependency])
                return args[0]
            if name.startswith("attention_qk_pv_"):
                if tokens:
                    if core < 24:
                        required = {(constants["PHASE_QUERY"], producer) for producer in range(24, 72)}
                    else:
                        required = {
                            (constants[phase], producer) for phase in REQUIRED[1:] for producer in range(24, 72)
                        }
                    assert required <= knowledge[core], (core, required - knowledge[core])
                attention_reads += 1
            if name.startswith("indexer_score_") or name.startswith("attention_qk_pv_"):
                kind = "score" if name.startswith("indexer_score_") else "attention"
                pair = (kind, core if core < 24 else (core - 24) // 2)
                entered.setdefault(pair, set()).add(core)
                yield "pair", pair
            yield "step", None

        return event

    controls = dict(
        QUERY_GROUP=group,
        QK_PANEL=64,
        SCORE_TILE=1024,
        PREFETCH=1,
        TO_L0=True,
        BALANCED=True,
        ROOTS=1,
        OA_COLS=128 if tokens <= 96 else 256,
        OB_ROWS=32 if tokens <= 32 else 96 if tokens <= 96 else 128,
        OUTPUT_GROUP_PIPELINE=True,
        HC_ROW_DEPENDENCIES=True,
        HC_CONSUMER_REDUCE=False,
        ATTENTION_QUERY_METADATA=False,
        ATTENTION_TRANSITIVE_DEPENDENCIES=enabled,
    )
    generators = {}
    for role in ("cube", "vector"):
        function = next(
            node
            for node in ast.parse((PKG / f"{role}_schedule.py").read_text()).body
            if isinstance(node, ast.FunctionDef)
        )
        names = [arg.arg for arg in function.args.args]
        tree = ast.fix_missing_locations(ast.Module([ScheduleEvents().visit(function)], type_ignores=[]))
        for core in range(24) if role == "cube" else range(24, 72):
            block, lane = (core, 0) if core < 24 else ((core - 24) // 2, (core - 24) % 2)
            pl = SimpleNamespace(
                tile=SimpleNamespace(get_block_idx=lambda b=block: b, get_subblock_idx=lambda lane_id=lane: lane_id),
                tensor=SimpleNamespace(dim=dim),
                system=SimpleNamespace(set_ffts=lambda _: None),
                func_attr=lambda _: None,
                cast=lambda x, _: int(x),
                read=lambda name, _: metadata[name],
                min=min,
                max=max,
                range=range,
                INDEX="index",
                INT32="int32",
            )
            namespace = dict(pl=pl, event=make_event(core), GROUPS=24, VECTORS=48, CORES=72, **constants)
            exec(compile(tree, str(PKG / f"{role}_schedule.py"), "exec"), namespace)
            generators[core] = namespace[f"{role}_schedule"](*(controls.get(name, name) for name in names))
    rng = random.Random(seed)
    waiting = {}
    done = set()
    for round_index in range(10000):
        progress = False
        order = list(generators)
        rng.shuffle(order)
        for core in order:
            if core in done:
                continue
            kind, payload = waiting.get(core, ("step", None))
            if kind == "wait" and not all(item in published for item in payload):
                continue
            if kind == "pair" and len(entered[payload]) != 3:
                continue
            try:
                waiting[core] = next(generators[core])
            except StopIteration:
                done.add(core)
            progress = True
        assert progress or len(done) == 72, (tokens, query_lengths, waiting)
        if len(done) == 72:
            break
    assert len(done) == len(begun) == attention_reads == 72 and plan_closures == 48
    return dict(
        tokens=tokens,
        requests=len(query_lengths),
        queries=query_lengths,
        enabled=enabled,
        seed=seed,
        rounds=round_index + 1,
        wait_slots=wait_slots,
        plan_closures=plan_closures,
    )


def main():
    cases = [
        (0, [], []),
        (24, [0, 0, 0, 0], [0, 0, 0, 0]),
        (16, [2, 0, 7, 1], [8191, 0, 131071, 7]),
        (24, [1, 4, 3, 2], [131073, 32767, 8191, 0]),
    ]
    for tokens in (1, 7, 8, 9, 23, 24, 25, 48, 96, 144, 168, 169, 192, 376, 377, 384, 385, 511, 512, 513):
        queries = [6] * (tokens // 6) + ([tokens % 6] if tokens % 6 else [])
        cases.append((tokens, queries, [131072] * len(queries)))
    rows = [run_case(t, q, h, seed, enabled) for t, q, h in cases for seed in range(3) for enabled in (False, True)]
    report = dict(
        passed=True,
        cases=len(rows),
        source="实际cube/vector_schedule.py AST",
        guarantee="Attention读取前已获取的发布闭包包含全部必需业务生产者",
        device_cache_validated=False,
        numerical_accuracy_validated=False,
        performance_measured=False,
        rows=rows,
    )
    (HERE / "dependency_proof.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
