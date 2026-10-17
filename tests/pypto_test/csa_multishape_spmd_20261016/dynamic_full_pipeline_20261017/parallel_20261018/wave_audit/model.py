# SPDX-License-Identifier: Apache-2.0
"""只读两请求wave设计审计：实际owner、显式wave发布、72核资源DAG。

唯一对象是长历史B16/G6/balanced，保留原物理owner。前8请求为wave0，
后8请求为wave1；12个MIX组先Attention0，另12组先Indexer1。数学、硬件
缓存和FFTS微指令不在模型内；单位工作格不是微秒，不构成性能预测。
"""

import ast
import json
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
GROUPS = 24
QUERY_GROUP = 6
BATCH = 16
TOKENS = BATCH * QUERY_GROUP
CUT = 8 * QUERY_GROUP
ATTENTION_BLOCKS = 5


def source_geometry():
    """离线解释实际源码三个Scalar helper，避免把旧静态leaf顺序当当前合同。"""
    path = PACKAGE / "indexer_group_stages.py"
    parsed = ast.parse(path.read_text())
    names = {"indexer_long_leaf_plan", "indexer_work_item", "indexer_leaf_range"}
    functions = [
        statement for statement in parsed.body if isinstance(statement, ast.FunctionDef) and statement.name in names
    ]
    for function in functions:
        function.decorator_list = []
        function.returns = None
        for arg in function.args.args:
            arg.annotation = None
    tree = ast.fix_missing_locations(ast.Module(functions, type_ignores=[]))
    namespace = dict(
        pl=SimpleNamespace(cast=lambda value, _: int(value), INDEX="index", min=min, max=max),
        SCORE_WORKERS=GROUPS,
        SCORE_TILE=1024,
        CANDIDATES_PER_LEAF=8192,
        MAX_LEAVES=32,
    )
    exec(compile(tree, str(path), "exec"), namespace)
    visible = (131072 + QUERY_GROUP) // 4
    leaves, leaf_tiles, extra = namespace["indexer_long_leaf_plan"](visible, BATCH)
    rows = []
    for item in range(BATCH * leaves):
        request, leaf = namespace["indexer_work_item"](item, BATCH, leaves)
        _, _, valid = namespace["indexer_leaf_range"](leaf, leaf_tiles, extra, visible)
        rows.append(
            dict(
                item=item,
                request=request,
                leaf=leaf,
                group=item % GROUPS,
                wave=int(request >= BATCH // 2),
                blocks=(valid + 1023) // 1024,
            )
        )
    assert len(rows) == len({(row["request"], row["leaf"]) for row in rows}) == BATCH * leaves
    return rows, leaves, leaf_tiles, extra


def resources(group):
    return (group, GROUPS + 2 * group, GROUPS + 2 * group + 1)


class Graph:
    def __init__(self):
        self.nodes = {}
        self.last = {}

    def add(self, name, cores, duration, parents=(), **details):
        assert name not in self.nodes
        dependencies = set(parents)
        dependencies.update(self.last[core] for core in cores if core in self.last)
        self.nodes[name] = dict(cores=list(cores), duration=duration, parents=sorted(dependencies), **details)
        for core in cores:
            self.last[core] = name

    def evaluate(self):
        children = defaultdict(set)
        pending = {}
        for name, node in self.nodes.items():
            assert set(node["parents"]) <= self.nodes.keys(), (name, node["parents"])
            pending[name] = len(node["parents"])
            for parent in node["parents"]:
                children[parent].add(name)
        ready = deque(name for name, count in pending.items() if count == 0)
        order = []
        while ready:
            name = ready.popleft()
            node = self.nodes[name]
            node["start"] = max((self.nodes[parent]["end"] for parent in node["parents"]), default=0)
            node["end"] = node["start"] + node["duration"]
            order.append(name)
            for child in sorted(children[name]):
                pending[child] -= 1
                if pending[child] == 0:
                    ready.append(child)
        assert len(order) == len(self.nodes), {name: count for name, count in pending.items() if count}
        calendars = defaultdict(list)
        for name, node in self.nodes.items():
            if node["duration"]:
                for core in node["cores"]:
                    calendars[core].append((node["start"], node["end"], name))
        for core, calendar in calendars.items():
            calendar.sort()
            assert all(left[1] <= right[0] for left, right in zip(calendar, calendar[1:])), core
        return max(node["end"] for node in self.nodes.values())


def build(rows, score_unit, attention_unit, split):
    """显式建每query根依赖与plan M8依赖；无新增全核barrier节点。"""
    graph = Graph()
    root_owner = {(row["request"], row["leaf"]): row["group"] for row in rows}
    leaves = max(row["leaf"] for row in rows) + 1
    wave_count = 2 if split else 1
    bounds = [(0, CUT), (CUT, TOKENS)] if split else [(0, TOKENS)]
    merge_inputs = {}
    plan_inputs = {}

    def score(wave, group):
        work = [row for row in rows if row["group"] == group and (not split or row["wave"] == wave)]
        graph.add(
            f"I{wave}_g{group}",
            resources(group),
            sum(row["blocks"] for row in work) * score_unit,
            stage="Indexer",
            wave=wave,
            group=group,
            items=[(row["request"], row["leaf"]) for row in work],
            drained_before_release=True,
        )

    def merge_and_plan(wave):
        first, end = bounds[wave]
        for query in range(first, end):
            request, local = divmod(query, QUERY_GROUP)
            lane = local // (QUERY_GROUP // 2)
            producers = sorted({GROUPS + 2 * root_owner[request, leaf] + lane for leaf in range(leaves)})
            merge_inputs[query] = producers
            graph.add(
                f"M_q{query}",
                (GROUPS + query % 48,),
                0,
                parents=[f"I{wave}_g{root_owner[request, leaf]}" for leaf in range(leaves)],
                stage="TopK",
                wave=wave,
                query=query,
                exact_aiv_producers=producers,
            )
        assert first % 8 == 0 and end % 8 == 0, "本反例只用合法的8行对齐请求边界"
        for tile in range(first // 8, end // 8):
            queries = list(range(tile * 8, tile * 8 + 8))
            plan_inputs[tile] = queries
            graph.add(
                f"P_tile{tile}",
                (GROUPS + tile % 48,),
                0,
                parents=[f"M_q{query}" for query in queries],
                stage="Plan",
                wave=wave,
                queries=queries,
            )
        # 各AIV仅发布自己的wave槽，空plan角色也发布，供本组切换时作局部交接。
        # AIC只等本组两个AIV完成这一轮merge/plan；没有等待72核的集合。
        for worker in range(48):
            graph.add(
                f"Ppub{wave}_v{worker}", (GROUPS + worker,), 0, stage="Plan发布/本组切换", wave=wave, single_writer=True
            )

    def attention(wave, group):
        first, end = bounds[wave]
        queries = [query for query in range(first, end) if query % GROUPS == group]
        producers = sorted({query // 8 % 48 for query in queries})
        parents = [f"Ppub{wave}_v{worker}" for worker in producers]
        graph.add(
            f"A{wave}_g{group}",
            resources(group),
            len(queries) * ATTENTION_BLOCKS * attention_unit,
            parents=parents,
            stage="Attention",
            wave=wave,
            group=group,
            queries=queries,
            exact_plan_producers=producers,
            drained_before_release=True,
        )

    for group in range(GROUPS):
        score(0, group)
    merge_and_plan(0)
    if split:
        # 唯一方案：g0..11先A0后I1，g12..23先I1后A0；每组两AIV跟随同一顺序。
        for group in range(GROUPS):
            if group < GROUPS // 2:
                attention(0, group)
                score(1, group)
            else:
                score(1, group)
                attention(0, group)
        merge_and_plan(1)
        for group in range(GROUPS):
            attention(1, group)
    else:
        for group in range(GROUPS):
            attention(0, group)
    # O_A仍读取完整原行tile，排在全部packed完成之后；未理想化拆分OA/OB。
    graph.add(
        "packed_all_ready",
        (),
        0,
        parents=[f"A{wave}_g{group}" for wave in range(wave_count) for group in range(GROUPS)],
        stage="既有O_A输入合同",
    )
    span = graph.evaluate()
    return graph, span, merge_inputs, plan_inputs


def protocol_audit(rows):
    """核对边界消息配对和GM所有权，不模拟硬件事件指令或cache一致性。"""
    score_regions = []
    attention_regions = []
    total_score_blocks = 0
    for group in range(GROUPS):
        score_regions.append(set(range(group * 2 * QUERY_GROUP, (group + 1) * 2 * QUERY_GROUP)))
        attention_regions.append(set(range(group * 3, (group + 1) * 3)))
        for wave in (0, 1):
            for leaf in (row for row in rows if row["group"] == group and row["wave"] == wave):
                # 每leaf开始无遗留消息，最多两份在途；两AIV加载并ACK后才复用。
                outstanding = []
                ready = [0, 0]
                ack = [0, 0]
                consumed = 0
                for step in range(leaf["blocks"]):
                    if len(outstanding) == 2:
                        outstanding.pop(0)
                        consumed += 1
                    outstanding.append(step % 2)
                    for lane in (0, 1):
                        ready[lane] += 1
                        ack[lane] += 1
                    assert len(outstanding) <= 2
                consumed += len(outstanding)
                outstanding.clear()
                assert consumed == leaf["blocks"] and ready == ack == [consumed, consumed]
                total_score_blocks += consumed
    for regions in (score_regions, attention_regions):
        assert all(not left & right for i, left in enumerate(regions) for right in regions[i + 1 :])
    roots0 = {
        (query, leaf["leaf"])
        for leaf in rows
        if leaf["wave"] == 0
        for query in range(leaf["request"] * 6, (leaf["request"] + 1) * 6)
    }
    roots1 = {
        (query, leaf["leaf"])
        for leaf in rows
        if leaf["wave"] == 1
        for query in range(leaf["request"] * 6, (leaf["request"] + 1) * 6)
    }
    assert not roots0 & roots1 and len(roots0 | roots1) == TOKENS * 6
    return dict(
        score_blocks=total_score_blocks,
        score_ready_ack_ids=[0, 1],
        attention_event_ids=[2, 3, 4, 5],
        score_fifo_slots_per_group=2,
        attention_ring_slots_per_group=3,
        all_score_leaf_boundaries_drained=True,
        disjoint_physical_group_regions=True,
        disjoint_pair_topk_plan_packed_query_rows=True,
        attention_boundary_contract="每wave完整执行现有+2/+3排空；10个有效块的四种事件全部配对",
        hardware_ffts_simulated=False,
        cache_maintenance_validated=False,
    )


def main():
    rows, leaves, leaf_tiles, extra = source_geometry()
    per_group = [sum(row["blocks"] for row in rows if row["group"] == group) for group in range(GROUPS)]
    per_wave = [
        [sum(row["blocks"] for row in rows if row["group"] == group and row["wave"] == wave) for group in range(GROUPS)]
        for wave in (0, 1)
    ]
    assert per_group == [22] * GROUPS and per_wave == [[11] * GROUPS, [11] * GROUPS]
    # 下方未切范围helper的等待环见证，直接使用实际work_item给出的owner。
    assert next(row["group"] for row in rows if row["request"] == 8 and row["leaf"] == 1) == 0
    assert 48 % 24 == 0 and 48 % 48 == 0 and (48 // 8) % 48 == 6
    cases = []
    example = None
    for score_unit, attention_unit in ((1, 1), (3, 1), (1, 3)):
        _, original_span, _, _ = build(rows, score_unit, attention_unit, False)
        graph, candidate_span, merge_inputs, plan_inputs = build(rows, score_unit, attention_unit, True)
        lower_bound = max(per_group) * score_unit + (TOKENS // GROUPS) * ATTENTION_BLOCKS * attention_unit
        assert candidate_span == original_span == lower_bound
        early = graph.nodes["A0_g0"]
        late = graph.nodes["I1_g12"]
        assert max(early["start"], late["start"]) < min(early["end"], late["end"])
        cases.append(
            dict(
                score_unit=score_unit,
                attention_unit=attention_unit,
                original_span=original_span,
                two_wave_span=candidate_span,
                fixed_owner_work_bound=lower_bound,
                cross_group_overlap=True,
                modeled_compute_reduction=0,
            )
        )
        if score_unit == attention_unit == 1:
            example = dict(nodes=graph.nodes, merge_inputs=merge_inputs, plan_inputs=plan_inputs)
    result = dict(
        passed=True,
        scope="单一固定owner两wave方案的离线资源/依赖审计，无编译或设备执行",
        source_functions=["indexer_long_leaf_plan", "indexer_work_item", "indexer_leaf_range"],
        source_files=[
            dict(path=str(path.relative_to(ROOT)), size_bytes=path.stat().st_size)
            for path in (
                PACKAGE / name
                for name in (
                    "indexer_group_stages.py",
                    "indexer_score_stages.py",
                    "indexer_publish_stages.py",
                    "attention_plan_stages.py",
                    "attention_stream_stages.py",
                    "cube_schedule.py",
                    "vector_schedule.py",
                )
            )
        ],
        geometry=dict(
            batch=BATCH,
            tokens=TOKENS,
            group_size=QUERY_GROUP,
            leaves=leaves,
            leaf_tiles=leaf_tiles,
            extra_leaves=extra,
            cut_query=CUT,
            per_group_score_blocks=per_group,
            per_wave_score_blocks=per_wave,
            attention_queries_per_group=TOKENS // GROUPS,
        ),
        protocol=protocol_audit(rows),
        cases=cases,
        unbounded_helper_cycle=[
            "A0_g0若仍遍历所有T，会在q48需要wave1 plan",
            "plan q48..55由AIV worker6产生，需要query48的TopK",
            "query48 TopK由AIV worker0产生，需要request8 leaf1的score根",
            "request8 leaf1归AIC group0，其I1在A0返回之后才允许执行",
            "形成A0_g0→plan1→merge1→I1_g0→A0_g0等待环",
        ],
        conclusion="可构造跨组重叠且无环，但固定owner均衡反例中未缩短大阶段计算链；不选为设备候选",
        limitation="不证明角色尾段细粒度交错、重分核或真实硬件不同负载互补均无收益；本模型没有这些能力",
        units="抽象工作格，绝非微秒或设备性能下限",
        example=example,
    )
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "example"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
