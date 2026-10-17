#!/usr/bin/env python3
"""只读枚举 compact Had owner 提前的工作量、偏序和已有阶段时钟。

输入为正式动态源码和已存在的 2026-10-18 长 B16 时钟。输出 JSON 不包含
预测性能：模型只判断既定源级先后/两槽 READY-ACK 是否成环，记录每核数学
块数。没有导入 PyPTO、torch 或设备模块，没有生成/编译算子。

模型将前段已发布的 IDX_PROJECT/QUERY_PROJECT/WEIGHTS/KV 抽成 E 节点，
将 query Had 完成抽成 R 节点；NORM 不依赖 QB。保留每个 AIV 的 compact
量化、全部 48 scale 发布、全 QB 获取、主压缩 pool 集合和两个 ACK。
不模拟真实缓存、指令流水、内存竞争、资源分配或事件硬件实现。
"""

import argparse
import ast
import json
from collections import defaultdict, deque
from pathlib import Path


CORE_GROUPS = 24
VECTORS = 48
SEVEN = [(131072, b) for b in (4, 8, 16, 24, 28)] + [(8192, b) for b in (16, 32)]


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def function_lines(path, names):
    tree = ast.parse(path.read_text())
    return {n.name: [n.lineno, n.end_lineno] for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}


def geometry(history, queries):
    """直接对应 Q/L 关闭数公式；不把 C 写成 T/4，也不靠 position 猜行容量。"""
    tokens, requests = sum(queries), len(queries)
    compact = sum((h + q) // 4 - h // 4 for h, q in zip(history, queries) if q > 0)
    max_history = max(((h + q) // 4 for h, q in zip(history, queries) if q > 0), default=0)
    long = max_history > 8192
    group = 6 if long and tokens >= 24 and max(queries, default=0) >= 6 else 2
    if not long and max(queries, default=0) >= 6 and requests >= 24:
        if 4 * 6 * ceildiv(requests, 24) <= 5 * 2 * ceildiv(ceildiv(tokens, 2), 24):
            group = 6
    balanced = long and group == 6
    groups = []
    for request, (h, q) in enumerate(zip(history, queries)):
        for offset in range(0, q, group):
            live = min(group, q - offset)
            visible = min((h + q) // 4, (h + offset + live) // 4)
            groups.append(dict(request=request, offset=offset, live=live, visible=visible))
    leaves, tiles, extra = max(ceildiv(max_history, 8192), 1), 8, 0
    if balanced:
        count = len(groups)
        multiple = 3 if 3 * count in (24, 48) else 6 if 6 * count == 24 else 1
        if multiple > 1:
            leaves = ceildiv(leaves, multiple) * multiple
            tiles, extra = max(ceildiv(max_history, 1024) // leaves, 1), ceildiv(max_history, 1024) % leaves
    had = [[] for _ in range(24)]
    row_blocks = ceildiv(compact, 16)
    for logical in range(2 * row_blocks):
        had[logical % 24].append(dict(row=logical % row_blocks * 16, cols=(logical // row_blocks) * 64,
                                      live_rows=min(16, compact - logical % row_blocks * 16)))
    qb = [list(range(core, 128, 24)) for core in range(24)]
    main = [[] for _ in range(24)]
    for logical in range(ceildiv(tokens, 64) * 16):
        row = logical // 16 * 64
        main[logical % 24].append(dict(row=row, col=logical % 16 * 64, live_rows=min(64, tokens - row)))
    score = [[] for _ in range(24)]
    for item in range(len(groups) * leaves):
        g, leaf = (item % len(groups), item // len(groups)) if len(groups) < 24 else divmod(item, leaves)
        begin = (leaf * tiles + min(leaf, extra)) * 1024
        capacity = (tiles + (leaf < extra)) * 1024
        valid = max(min(groups[g]['visible'] - begin, capacity), 0)
        tile = 1024 if long else 768
        tile_count = ceildiv(valid, tile)
        # 与原非 balanced 双 half 路径相同，score_iters 由候选 half 上界计算。
        iterations = tile_count if balanced else ceildiv(min(tile_count * (tile // 2), 4096), tile // 2)
        score[item % 24].append(dict(item=item, group=g, leaf=leaf, valid=valid, steps=iterations))
    cores = []
    for core in range(24):
        cores.append(dict(core=core, had_tiles=had[core], qb_columns=qb[core],
                          qb_m128_tiles=len(qb[core]) * ceildiv(tokens, 128),
                          main_tiles=main[core], main_live_row_sum=sum(x['live_rows'] for x in main[core]),
                          score_items=score[core], score_steps=sum(x['steps'] for x in score[core])))
    return dict(tokens=tokens, requests=requests, histories=history, queries=queries,
                compact_count=compact, had_owners=[i for i in range(24) if had[i]], group=group,
                groups=len(groups), leaves=leaves, leaf_tiles=tiles, extra=extra, balanced=balanced, cores=cores)


class Graph:
    def __init__(self):
        self.parents = defaultdict(set)

    def add(self, name, *parents):
        self.parents[name].update(parents)
        for parent in parents:
            self.parents[parent]
        return name

    def summary(self):
        indegree = {n: len(p) for n, p in self.parents.items()}
        children = defaultdict(list)
        for n, parents in self.parents.items():
            for p in parents:
                children[p].append(n)
        queue = deque(n for n, degree in indegree.items() if not degree)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        return dict(acyclic=visited == len(indegree), nodes=len(indegree), edges=sum(map(len, self.parents.values())),
                    blocked_example=[n for n, degree in indegree.items() if degree][:12])


def schedule_graph(case, owner_first, unsafe_wait_cube=False):
    """同一个有限偏序模型的两种排序；逐 leaf drain 和两条 lane ACK 都建边。"""
    graph = Graph()
    graph.add('META')
    early = [graph.add(f'E{i}') for i in range(24)]
    rope = graph.add('IDX_QUERY_ROPE', *early)
    for vector in range(48):
        graph.add(f'N{vector}', *early)
    norm_owners = [f'N{i}' for i in range(min(case['requests'], 48))]
    for core in range(24):
        graph.add(f'R{core}', f'E{core}', rope)
        if owner_first and core in case['had_owners']:
            graph.add(f'H{core}', f'R{core}', 'META', *norm_owners)
            graph.add(f'Q{core}', f'H{core}')
        else:
            graph.add(f'Q{core}', f'R{core}')
            # 无 Had 数学的核也保留原等待/通知块，本分析不偷偷删通知或换 owner。
            graph.add(f'H{core}', f'Q{core}', 'META', *norm_owners)
        graph.add(f'M{core}', f'H{core}', f'Q{core}')
    had_owners = [f'H{i}' for i in case['had_owners']]
    for vector in range(48):
        graph.add(f'COEFF{vector}', f'N{vector}', *[f'R{i}' for i in range(24)])
        graph.add(f'QUANT{vector}', f'COEFF{vector}', *had_owners)
    quant = [f'QUANT{i}' for i in range(min(case['requests'], 48))]
    for vector in range(48):
        graph.add(f'SCALE{vector}', f'QUANT{vector}', *quant)
        graph.add(f'QUERY{vector}', f'SCALE{vector}', *[f'Q{i}' for i in range(24)])
        graph.add(f'POOL{vector}', f'QUERY{vector}', *[f'M{i}' for i in range(24) if case['cores'][i]['main_tiles']])
    pools = [f'POOL{i}' for i in range(min(case['requests'], 48))]
    for vector in range(48):
        graph.add(f'VECTOR_ENTRY{vector}', f'POOL{vector}', *pools, *[f'SCALE{i}' for i in range(48)])
    for core in range(24):
        cube_previous = graph.add(f'CUBE_ENTRY{core}', f'M{core}', *[f'SCALE{i}' for i in range(48)])
        lane_previous = [f'VECTOR_ENTRY{2 * core + lane}' for lane in range(2)]
        for item in case['cores'][core]['score_items']:
            acknowledgements = []
            for step in range(item['steps']):
                # 第三个生产动作才复用 slot0；第一/第二槽不需要等 AIV 到达。
                parents = [cube_previous]
                if step >= 2:
                    parents += acknowledgements[step - 2]
                produced = graph.add(f'P{core}.{item["item"]}.{step}', *parents)
                acks = []
                for lane in range(2):
                    ack = graph.add(f'A{core}.{item["item"]}.{step}.{lane}', produced, lane_previous[lane])
                    lane_previous[lane] = graph.add(f'V{core}.{item["item"]}.{step}.{lane}', ack)
                    acks.append(ack)
                acknowledgements.append(acks)
                cube_previous = produced
            if acknowledgements:
                cube_previous = graph.add(f'DRAIN{core}.{item["item"]}', cube_previous,
                                          *[n for pair in acknowledgements[-2:] for n in pair])
        graph.add(f'CUBE_DONE{core}', cube_previous)
        for lane in range(2):
            graph.add(f'VECTOR_DONE{2 * core + lane}', lane_previous[lane])
            if unsafe_wait_cube:
                graph.add(f'VECTOR_ENTRY{2 * core + lane}', f'CUBE_DONE{core}')
    return graph.summary()


def old_clock(root):
    rounds = []
    for number in range(3):
        path = root / f'clock_round_{number}_spmd_reference/phase_clock.json'
        records = json.loads(path.read_text())
        by = {(r['name'], r['core']): r for r in records}
        norm = max(by['IDX_NORM', 24 + i]['notify_end_us'] for i in range(16))
        coeff = max(by['COEFF', 24 + i]['notify_end_us'] for i in range(48))
        qb = max(by['QB', i]['notify_end_us'] for i in range(24))
        owners = []
        for core in (0, 1):
            r, q, m, h = [by[name, core]['notify_end_us'] for name in ('IDX_QUERY_HAD', 'QB', 'MAIN_PROJECT', 'IDX_HADAMARD')]
            owners.append(dict(core=core, before_qb_endpoint_us=r, qb_endpoint_us=q,
                               main_endpoint_us=m, had_endpoint_us=h,
                               old_interval_qb_us=q-r, old_interval_had_us=h-m,
                               norm_not_ready_gap_us=max(norm-r, 0)))
        rounds.append(dict(round=number, source=str(path), norm_actual_owner_max_us=norm,
                           coeff_all_publish_max_us=coeff, qb_all_publish_max_us=qb, owners=owners,
                           existing_instrumented_endpoints_only=True))
    return rounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('evidence.json'))
    args = parser.parse_args()
    p = args.repo / 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
    n = args.repo / 'tests/pypto_test/csa_multishape_spmd_20261016'
    seven = [geometry([h] * b, [6] * b) for h, b in SEVEN]
    # 边界组只验证这个偏序抽象；不宣称它们通过真实完整入口或数值校验。
    extras = [geometry([131072, 131073, 131074, 131075], [1, 5, 0, 3]),
              geometry([131073] * 32, [6] * 32), geometry([131074] * 32, [6] * 32),
              geometry([131072] * 32, [24] * 32), geometry([131075] * 32, [25] * 32),
              geometry([131072] * 4, [1] * 4)]
    checks = []
    for i, case in enumerate(seven + extras):
        current = schedule_graph(case, False)
        candidate = schedule_graph(case, True)
        assert current['acyclic'] and candidate['acyclic']
        checks.append(dict(case=i, current=current, owner_first=candidate))
    negative = schedule_graph(seven[2], True, unsafe_wait_cube=True)
    assert not negative['acyclic']
    sources = {
        'cube_schedule.py': ['cube_schedule'], 'vector_schedule.py': ['vector_schedule'],
        'indexer_compressor_output_stages.py': ['indexer_compressor_hadamard_stage'],
        'query_projection_stages.py': ['_q_b_nz_stage'],
        'compressor_projection_stages.py': ['compressor_project_stage'],
        'metadata_stages.py': ['_scan_compact_offsets'],
        'indexer_group_stages.py': ['indexer_long_leaf_plan', 'indexer_group_plan_stage', 'indexer_work_item', 'indexer_leaf_range'],
        'indexer_score_stages.py': ['indexer_score_cube_stage', 'indexer_score_vector_lane_stage'],
    }
    result = dict(scope='源级只读有限偏序/真实工作量/旧时钟端点，不是性能仿真',
                  compiled=False, npu_executed=False, performance_measured=False,
                  model_count=1, seven_shapes=seven, extra_geometry_cases=extras, graph_checks=checks,
                  deliberately_invalid_wait_cube=negative,
                  source_locations={str(p / name): function_lines(p / name, names) for name, names in sources.items()},
                  prior_deferred_main_q=json.loads((n / 'deferred_main_q_schedule_analysis.json').read_text()),
                  old_clock=old_clock(n / 'dynamic_full_pipeline_20261017/parallel_20261018/current_clock_h131072_b16_v1'))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), geometry_cases=len(checks), graph_variants=len(checks)*2,
                          negative_control_detected=True, no_compile_no_device=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
