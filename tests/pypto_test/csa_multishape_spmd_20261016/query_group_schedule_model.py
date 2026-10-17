# SPDX-License-Identifier: Apache-2.0
"""两组query的完整成员图/排程原型；尚不生成或执行设备代码。

使用现选B16的已证明行依赖，拆Indexer→merge→plan→QK→O-A→quant→O-B→HCpost。
两个MIX阶段保持原logical block到物理组映射；组间切换等待三成员一起结束，
保护FFTS复用以及原logical block私有的GM交接区。矩阵缩行成本仅作敏感性估计。
"""

import argparse
import collections
import copy
import json
import math
from pathlib import Path

from balanced_schedule import earliest_fit
from member_schedule import _costs, _estimate, _problem, _resource_parents
from build_schedule_manifest import ancestor_sets


SPLIT = {
    'indexer_score_topk_native_pair', 'indexer_topk_query_merge__2',
    'csa_slots_build_valid_qk_plan', 'qk_pv', 'proj_a_mm_0', 'quant_0',
    '_proj_b_mm_nz_kernel__2', 'proj_b_act_hc_post_0',
}
MIX = {'indexer_score_topk_native_pair', 'qk_pv'}


def build_problem(schedule, row_audit, cost_profile, projection_fraction):
    old, parents, _, _, _, _, _, mix_names = _problem(schedule, row_audit)
    costs = _costs(schedule, old, cost_profile)
    baseline = _estimate(_resource_parents(old, parents, mix_names), costs)
    names = {tid: call['logical_function'] for tid, call in schedule['call_bindings'].items()}
    nodes, origin = [], collections.defaultdict(list)
    for i, row in enumerate(old):
        name, block = names[row['task']], row['logical_block']
        phases = (0, 1) if name in SPLIT else (None,)
        if name == 'csa_slots_build_valid_qk_plan':
            phases = (block // 6,) if block < 12 else ()
        if name == 'proj_b_act_hc_post_0':
            phases = (block // 24,)
        for phase in phases:
            r = copy.deepcopy(row)
            if name in ('csa_slots_build_valid_qk_plan', 'proj_b_act_hc_post_0'):
                r['logical_block'] %= 6 if name == 'csa_slots_build_valid_qk_plan' else 24
            fraction = 1.0
            if name in SPLIT - {'csa_slots_build_valid_qk_plan', 'proj_b_act_hc_post_0'}:
                fraction = projection_fraction if name in ('proj_a_mm_0', '_proj_b_mm_nz_kernel__2') else 0.5
            r.update(origin=i, phase=phase, name=name,
                     duration_us=(costs[i] - 1.0) * fraction + 1.0,
                     task=row['task'] if phase is None else row['task'] + f':q{phase}')
            origin[i].append(len(nodes))
            nodes.append(r)
    math_parents = []
    for r in nodes:
        chosen = set()
        for p in parents[r['origin']]:
            chosen.update(j for j in origin[p]
                          if r['phase'] is None or nodes[j]['phase'] in (None, r['phase']))
        math_parents.append(chosen)
    ancestor_sets(math_parents)

    # 明确核对被改变的生产/消费边，而非认为复制DAG就自然保持数据流。
    for i, r in enumerate(nodes):
        if r['phase'] is None:
            continue
        first, end = r['phase'] * 48, (r['phase'] + 1) * 48
        relevant = collections.defaultdict(list)
        for p in math_parents[i]:
            relevant[nodes[p]['name']].append(nodes[p])
        block = r['logical_block']
        if r['name'] == 'indexer_topk_query_merge__2':
            request = (first + block) // 6
            expected = {(leaf * 16 + request) % 24 for leaf in range(32)}
            actual = relevant['indexer_score_topk_native_pair']
            assert {p['logical_block'] for p in actual} == expected and len(actual) == 9
        elif r['name'] == 'csa_slots_build_valid_qk_plan':
            expected = set(range(block * 8, block * 8 + 8))
            assert {p['logical_block'] for p in relevant['indexer_topk_query_merge__2']} == expected
        elif r['name'] == 'qk_pv':
            expected = {(q - first) // 8 for q in range(first + block, end, 24)}
            assert {p['logical_block'] for p in relevant['csa_slots_build_valid_qk_plan']} == expected
        elif r['name'] == 'proj_a_mm_0':
            actual = relevant['qk_pv']
            assert {p['logical_block'] for p in actual} == set(range(24)) and len(actual) == 24
        elif r['name'] == 'quant_0':
            assert len(relevant['proj_a_mm_0']) == 8
        elif r['name'] == '_proj_b_mm_nz_kernel__2':
            assert len(relevant['quant_0']) == 3
        elif r['name'] == 'proj_b_act_hc_post_0':
            assert len(relevant['_proj_b_mm_nz_kernel__2']) == 64

    grouped = collections.defaultdict(list)
    for i, r in enumerate(nodes):
        grouped[r['task'], r['logical_block'], None if r['name'] in MIX else r['original_subslot']].append(i)
    units, unit_of = [], {}
    for ids in grouped.values():
        ids.sort(key=lambda i: ('AIC', 'AIV0', 'AIV1').index(nodes[i]['original_subslot']))
        if nodes[ids[0]]['name'] in MIX:
            assert [nodes[i]['original_subslot'] for i in ids] == ['AIC', 'AIV0', 'AIV1']
            g = nodes[ids[0]]['core']
            assert [nodes[i]['core'] for i in ids] == [g, 24 + 2*g, 25 + 2*g]
        else:
            assert len(ids) == 1
        for i in ids:
            unit_of[i] = len(units)
        units.append(ids)
    unit_parents = [{unit_of[p] for i in ids for p in math_parents[i]} for ids in units]
    order, _ = ancestor_sets(unit_parents)
    return nodes, math_parents, units, unit_of, unit_parents, order, baseline


def schedule_model(problem, policy, flexible_mixed=False):
    nodes, math_parents, units, unit_of, unit_parents, order, baseline = problem
    children = collections.defaultdict(set)
    for u, ps in enumerate(unit_parents):
        for p in ps:
            children[p].add(u)
    rank = {}
    for u in reversed(order):
        rank[u] = max(nodes[i]['duration_us'] for i in units[u]) + max((rank[c] for c in children[u]), default=0)
    calendars = {c: [] for c in range(72)}
    used_mixed = collections.defaultdict(set)
    remaining = [len(ps) for ps in unit_parents]
    ready = {u for u, n in enumerate(remaining) if n == 0}
    ends, placements = {}, {}
    while ready:
        choices = []
        for u in ready:
            ids = units[u]
            row = nodes[ids[0]]
            release = max((ends[p] for p in unit_parents[u]), default=0)
            elapsed = [nodes[i]['duration_us'] for i in ids]
            gm_resource = None
            if row['name'] in MIX:
                if flexible_mixed:
                    candidates = [(g, 24+2*g, 25+2*g) for g in range(24)
                                  if g not in used_mixed[row['task']]]
                else:
                    candidates = [tuple(nodes[i]['core'] for i in ids)]
                # MIX切换必须等三个成员完成；这是资源约束，不把FFTS换成全卡barrier。
                occupied = [max(elapsed)] * 3
                gm_resource = ('gm', row['name'], row['logical_block'])
                calendars.setdefault(gm_resource, [])
            elif row['core_kind'] == 'aic':
                candidates, occupied = [(c,) for c in range(24)], elapsed
            else:
                candidates, occupied = [(c,) for c in range(24, 72)], elapsed
            start, cores = min((earliest_fit(
                calendars, cores + ((gm_resource,) if gm_resource else ()), release,
                occupied + ([max(occupied)] if gm_resource else [])), cores) for cores in candidates)
            tie = (row['call_ordinal'], row['phase'] or 0, row['logical_block'])
            if policy == 'earliest':
                priority = (start, -rank[u], tie)
            elif policy == 'critical':
                priority = (-rank[u], start, tie)
            elif policy == 'phase0_first':
                priority = (row['phase'] == 1, -rank[u], start, tie)
            elif policy == 'prefix_then_phase0':
                # 先为未拆分的主Q等前缀保留排程位置，再推进前半链路。
                # 这只是list-scheduling优先级，earliest_fit仍可填前面的空隙，
                # 不新增全卡barrier，也不假设前缀数据早于真实生产者完成。
                priority = (0 if row['phase'] is None else 1 + row['phase'],
                            -rank[u], start, tie)
            elif policy == 'main_q_then_phase0':
                # 前半Indexer曾把主Q挤到后面；只保护主Q两阶段，避免
                # prefix_then_phase0连其他无关前缀都提前而推迟Indexer。
                main_q = row['name'] in ('qproj_matmul', 'qproj_dequant_rms_nope_rope')
                priority = (0 if main_q else 2 if row['phase'] == 1 else 1,
                            -rank[u], start, tie)
            else:
                raise ValueError(policy)
            choices.append((priority, u, start, cores, occupied, gm_resource))
        _, u, start, cores, occupied, gm_resource = min(choices)
        ids = units[u]
        for i, core, span in zip(ids, cores, occupied):
            calendars[core].append((start, start + span))
            calendars[core].sort()
            placements[i] = {'core': core, 'start': start, 'end': start + nodes[i]['duration_us']}
        ends[u] = start + max(occupied)
        if gm_resource is not None:
            calendars[gm_resource].append((start, ends[u]))
            calendars[gm_resource].sort()
            used_mixed[nodes[ids[0]]['task']].add(cores[0])
        ready.remove(u)
        for c in children[u]:
            remaining[c] -= 1
            if remaining[c] == 0:
                ready.add(c)
    assert len(ends) == len(units)
    for calendar in calendars.values():
        assert all(a[1] <= b[0] + 1e-8 for a, b in zip(calendar, calendar[1:]))
    for i, ps in enumerate(math_parents):
        assert all(placements[p]['end'] <= placements[i]['start'] + 1e-8 for p in ps)
    # 检查同一物理MIX组及同一logical worker的GM交接区跨两半不重叠。
    mixed = collections.defaultdict(list)
    transfers = collections.defaultdict(list)
    for u, ids in enumerate(units):
        r = nodes[ids[0]]
        if r['name'] not in MIX:
            continue
        interval = (placements[ids[0]]['start'], max(placements[i]['end'] for i in ids))
        mixed[placements[ids[0]]['core']].append(interval)
        transfers[r['name'], r['logical_block']].append(interval)
    for intervals in [*mixed.values(), *transfers.values()]:
        intervals.sort()
        assert all(a[1] <= b[0] + 1e-8 for a, b in zip(intervals, intervals[1:]))
    assert all(len(groups) == 24 for groups in used_mixed.values())
    phase_rows = collections.defaultdict(list)
    for i, r in enumerate(nodes):
        if r['name'] in SPLIT:
            phase_rows[r['name'], r['phase']].append(placements[i])
    spans = [{'name': k[0], 'phase': k[1], 'members': len(v),
              'start': min(x['start'] for x in v), 'end': max(x['end'] for x in v)}
             for k, v in sorted(phase_rows.items())]
    return {'policy': policy, 'estimated_span_us': max(ends.values()), 'original_estimated_span_us': baseline,
            'flexible_mixed_placement': flexible_mixed,
            'mixed_group_bijection_and_logical_gm_intervals_checked': True,
            'members': len(nodes), 'units': len(units), 'math_edges': sum(map(len, math_parents)),
            'all_modeled_dependencies_and_resource_intervals_checked': True,
            'stage_spans': spans, 'device_prediction': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projection-fraction', type=float, default=0.5)
    parser.add_argument('--policy', choices=('earliest', 'critical', 'phase0_first',
                                            'prefix_then_phase0', 'main_q_then_phase0'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--flexible-mixed', action='store_true',
                        help='允许两半重新分配物理组，但显式保护每个logical worker的GM交接区')
    args = parser.parse_args()
    if not math.isfinite(args.projection_fraction) or not 0.5 <= args.projection_fraction <= 1:
        raise ValueError('半批投影成本比例只用于0.5..1的敏感性检查')
    root = Path(__file__).resolve().parent
    s = json.loads((root/'b16_cached_row_deps_v1/specialization/bindings/h131072_b16_prepared.json').read_text())
    proof = json.loads((root/'b16_cached_row_deps_v1/specialization/control_variants/h131072_b16/row_dependencies/audit.json').read_text())
    problem = build_problem(s, proof, root/'b16_dynamic_l1_cost.json', args.projection_fraction)
    result = schedule_model(problem, args.policy, args.flexible_mixed)
    result.update(scope='CPU成员图原型，无设备代码/精度/性能结论', projection_fraction=args.projection_fraction,
                  assumed_split_extra_per_member_us=1.0, original_math_k_reduction_order_required=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'stage_spans'}, ensure_ascii=False))
