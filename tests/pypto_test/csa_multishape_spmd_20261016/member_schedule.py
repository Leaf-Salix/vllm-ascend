# SPDX-License-Identifier: Apache-2.0
"""利用已证明的成员依赖重新排程，而不是先按整task分核再删等待。

只支持当前B16/S6的原数学体。MIX三成员保持同一物理核组，同一MIX调用
的24组映射为双射；Indexer→QK的本组FFTS复用边按新物理位置重新建立。
成本模型只用于筛选候选，不是设备性能承诺。实际长度仍由原数学核读取。
"""

import argparse
import collections
import copy
import json
import math
import statistics
import sys
from pathlib import Path

STATIC_ROOT = Path(__file__).resolve().parent.parent/'csa_static_spmd_feasibility_20261015'
sys.path.insert(0,str(STATIC_ROOT))
from build_schedule_manifest import ancestor_sets
from balanced_schedule import earliest_fit


def _problem(schedule, row_audit):
    if schedule['root_arguments']['x_hc']['shape'][0] != 96:
        raise ValueError('当前成员排程只支持已证明的T96原核体')
    nodes = [copy.deepcopy(r) for rows in schedule['per_core_sequences'].values() for r in rows]
    keys = [(r['task'],r['original_subslot'],r['logical_block']) for r in nodes]
    if len(set(keys)) != len(keys):
        raise ValueError('逻辑成员不唯一')
    members = collections.defaultdict(list)
    for i,r in enumerate(nodes):
        members[r['task']].append(i)
    tasks = {t['task_id']:t for t in schedule['tasks']}
    parents = [set(p for tid in tasks[r['task']]['predecessors'] for p in members[tid]) for r in nodes]
    changed = set()
    for change in row_audit['changed_members']:
        i = change['node']
        r = nodes[i]
        if (r['core'],r['dispatch_seq'],r['task'],r['logical_block']) != (
                change['core'],change['sequence'],change['task'],change['logical_block']):
            raise ValueError('成员证明与传入的实际调度不是同一版本')
        parent = change['parent_task']
        selected = set(change['selected_nodes'])
        if parent not in tasks[r['task']]['predecessors'] or not selected <= set(members[parent]):
            raise ValueError('已证明边不属于原父task成员')
        parents[i].difference_update(members[parent])
        parents[i].update(selected)
        changed.add((i,parent))
    # 只使用上游refine刚完成的显式合同，不能从原required_parents里减去旧物理边：
    # 同一条边可能同时有数学与同核顺序两种原因，按集合盲删会丢数学约束。
    for i,r in enumerate(nodes):
        for tid in tasks[r['task']]['predecessors']:
            if (i,tid) not in changed and not set(members[tid]) <= parents[i]:
                raise ValueError('未获证明的整task边被修改')
    ancestor_sets(parents)
    mixed = {tid for tid,t in tasks.items() if t['kernel_names'][0] and t['kernel_names'][1]}
    names = {tid:schedule['call_bindings'][tid]['logical_function'] for tid in mixed}
    if set(names.values()) != {'indexer_score_topk_native_pair','qk_pv'}:
        raise ValueError('当前只证明Indexer与QK两种MIX资源复用')
    for tid in mixed:
        if tasks[tid]['block_num'] != 24 or tasks[tid]['member_count'] != 72:
            raise ValueError('MIX成员数改变')
    bundles, bundle_of = [], {}
    grouped = collections.defaultdict(list)
    for i,r in enumerate(nodes):
        key = (r['task'],r['logical_block'],None if r['task'] in mixed else r['original_subslot'])
        grouped[key].append(i)
    for ids in grouped.values():
        ids.sort(key=lambda i:('AIC','AIV0','AIV1').index(nodes[i]['original_subslot']))
        if nodes[ids[0]]['task'] in mixed:
            if [nodes[i]['original_subslot'] for i in ids] != ['AIC','AIV0','AIV1']:
                raise ValueError('逻辑MIX组不完整')
        elif len(ids) != 1:
            raise ValueError('纯AIC/AIV分片不是单成员')
        for i in ids:
            bundle_of[i] = len(bundles)
        bundles.append(ids)
    unit_parents = [{bundle_of[p] for i in ids for p in parents[i]} for ids in bundles]
    order,_ = ancestor_sets(unit_parents)
    return nodes,parents,bundles,bundle_of,unit_parents,order,mixed,names


def _costs(schedule, nodes, profile):
    path = Path(profile)
    data = json.loads(path.read_text())
    if not isinstance(data,dict) or data.get('kind') != 'same_shape_dynamic_l1_cost':
        raise ValueError('本轮先固定同规格原动态L1成本，避免成本与排程同时改变')
    if data['root_shapes'] != {k:v['shape'] for k,v in schedule['root_arguments'].items()}:
        raise ValueError('成本规格不匹配')
    table = {(r['task'],r['slot']):r for r in data['task_slots']}
    if set(table) != {(r['task'],r['original_subslot']) for r in nodes}:
        raise ValueError('成本未精确覆盖全部task/slot')
    durations = []
    for row in nodes:
        cost = table[row['task'],row['original_subslot']]
        if cost['kernel'] != row['kernel_name'] or cost['samples'] <= 0:
            raise ValueError('成本核名或样本无效')
        value = cost['median_us']+1.0
        if not math.isfinite(value) or value <= 0:
            raise ValueError('成本非法')
        durations.append(value)
    return durations


def _resource_parents(nodes, math_parents, mixed_names):
    parents = copy.deepcopy(math_parents)
    queues = collections.defaultdict(list)
    for i,r in enumerate(nodes):
        queues[r['core']].append(i)
    for indices in queues.values():
        indices.sort(key=lambda i:nodes[i]['dispatch_seq'])
        for a,b in zip(indices,indices[1:]):
            parents[b].add(a)
    idx = next(t for t,n in mixed_names.items() if n=='indexer_score_topk_native_pair')
    qk = next(t for t,n in mixed_names.items() if n=='qk_pv')
    groups = collections.defaultdict(list)
    for i,r in enumerate(nodes):
        if r['task']==idx:
            group = r['core'] if r['core']<24 else (r['core']-24)//2
            groups[group].append(i)
    if set(groups) != set(range(24)) or any(len(v)!=3 for v in groups.values()):
        raise ValueError('新分核的Indexer不是每个物理组一份完整MIX')
    count = 0
    for i,r in enumerate(nodes):
        if r['task']==qk:
            group = r['core'] if r['core']<24 else (r['core']-24)//2
            parents[i].update(groups[group])
            count += len(groups[group])
    if count != 216:
        raise ValueError('FFTS复用边数量不符')
    return parents


def _estimate(parents, durations):
    order,_ = ancestor_sets(parents)
    ends = [0.0]*len(parents)
    for i in order:
        ends[i] = max((ends[p] for p in parents[i]),default=0.0)+durations[i]
    return max(ends)


def _waits(nodes, parents):
    order,ancestors = ancestor_sets(parents)
    positions = {n:k for k,n in enumerate(order)}
    reduced = [set() for _ in nodes]
    for i,ps in enumerate(parents):
        covered = 0
        for p in sorted(ps,key=positions.get,reverse=True):
            if not covered>>p&1:
                reduced[i].add(p)
                covered |= ancestors[p]|(1<<p)
    queues = collections.defaultdict(list)
    for i,row in enumerate(nodes):
        queues[row['core']].append(i)
    published = collections.defaultdict(set)
    for core,indices in queues.items():
        indices.sort(key=lambda i:nodes[i]['dispatch_seq'])
        observed = collections.defaultdict(int)
        for i in indices:
            needed = collections.defaultdict(int)
            for p in reduced[i]:
                other = nodes[p]
                if other['core'] != core:
                    needed[other['core']] = max(needed[other['core']],other['dispatch_seq'])
            nodes[i]['wait_for_core_progress'] = {str(c):s for c,s in sorted(needed.items()) if s>observed[c]}
            for c,s in nodes[i]['wait_for_core_progress'].items():
                observed[int(c)] = s
                published[int(c)].add(s)
    ids = {(r['core'],r['dispatch_seq']):i for i,r in enumerate(nodes)}
    translated = [set() for _ in nodes]
    for i,r in enumerate(nodes):
        r['publish_completion'] = r['dispatch_seq'] in published[r['core']]
        if r['dispatch_seq']>1:
            translated[i].add(ids[r['core'],r['dispatch_seq']-1])
        translated[i].update(ids[int(c),s] for c,s in r['wait_for_core_progress'].items())
    _,actual = ancestor_sets(translated)
    if actual != ancestors:
        raise ValueError('进度表不保持新成员图的偏序')
    return {'remote_core_progress_requirements':sum(len(r['wait_for_core_progress']) for r in nodes),
            'completion_publications':sum(r['publish_completion'] for r in nodes),
            'planned_member_edges':sum(map(len,translated)),
            'identical_dependency_partial_order':False,'refined_partial_order_preserved':True,
            'global_barriers_added':0},translated


def reschedule(schedule, row_audit, profile, *, policy='critical', output=None):
    if policy not in ('critical','earliest','finish'):
        raise ValueError('未知成员排程策略')
    nodes,parents,bundles,bundle_of,unit_parents,order,mixed,names = _problem(schedule,row_audit)
    durations = _costs(schedule,nodes,profile)
    baseline = _estimate(_resource_parents(nodes,parents,names),durations)
    children = collections.defaultdict(set)
    for u,ps in enumerate(unit_parents):
        for p in ps:
            children[p].add(u)
    ranks = {}
    for u in reversed(order):
        ranks[u] = max(durations[i] for i in bundles[u])+max((ranks[c] for c in children[u]),default=0.0)
    idx = next(t for t,n in names.items() if n=='indexer_score_topk_native_pair')
    qk = next(t for t,n in names.items() if n=='qk_pv')
    idx_units = {u for u,ids in enumerate(bundles) if nodes[ids[0]]['task']==idx}
    calendars = {c:[] for c in range(72)}
    used_mixed = collections.defaultdict(set)
    idx_group_end = {}
    completed,ends,placements = set(),{},{}
    remaining = [len(ps) for ps in unit_parents]
    ready = {u for u,n in enumerate(remaining) if not n}
    while ready:
        choices = []
        for u in ready:
            ids = bundles[u]
            task = nodes[ids[0]]['task']
            if task==qk and not idx_units<=completed:
                continue
            release = max((ends[p] for i in ids for p in parents[i]),default=0.0)
            elapsed = [durations[i] for i in ids]
            if task in mixed:
                candidates = [(g,24+2*g,25+2*g) for g in range(24) if g not in used_mixed[task]]
            elif nodes[ids[0]]['core_kind']=='aic':
                candidates = [(c,) for c in range(24)]
            else:
                candidates = [(c,) for c in range(24,72)]
            possible = []
            for cores in candidates:
                start = max(release,idx_group_end.get(cores[0],0.0)) if task==qk else release
                start = earliest_fit(calendars,cores,start,elapsed)
                possible.append((start,cores))
            if not possible:
                raise ValueError('MIX双射没有可用物理组')
            start,cores = min(possible)
            tie = (schedule['call_bindings'][task]['ordinal'],nodes[ids[0]]['logical_block'])
            priority = {'critical':(-ranks[u],start,tie),
                        'earliest':(start,-ranks[u],tie),
                        'finish':(start+max(elapsed),-ranks[u],tie)}[policy]
            choices.append((priority,u,start,cores))
        if not choices:
            raise ValueError('成员图无可放置单元')
        _,u,start,cores = min(choices)
        ids = bundles[u]
        task = nodes[ids[0]]['task']
        for i,c in zip(ids,cores):
            ends[i] = start+durations[i]
            placements[i] = (c,start,ends[i])
            calendars[c].append((start,ends[i]))
            calendars[c].sort()
        if task in mixed:
            used_mixed[task].add(cores[0])
            if task==idx:
                idx_group_end[cores[0]] = max(ends[i] for i in ids)
        ready.remove(u)
        completed.add(u)
        for child in children[u]:
            remaining[child]-=1
            if not remaining[child]:
                ready.add(child)
    if len(completed)!=len(bundles):
        raise ValueError('成员没有全部排入')
    queues = {str(c):[] for c in range(72)}
    for i,row in enumerate(nodes):
        c,start,end = placements[i]
        row.update(core=c,start=start*50,reference_end=end*50,
                   estimated_start_us=start,estimated_end_us=end)
        queues[str(c)].append(row)
    for sequence in queues.values():
        sequence.sort(key=lambda r:(r['estimated_start_us'],r['estimated_end_us']))
        for seq,row in enumerate(sequence,1):
            row['dispatch_seq']=seq
    required = _resource_parents(nodes,parents,names)
    final_estimate = _estimate(required,durations)
    if abs(final_estimate-max(ends.values()))>1e-5:
        raise ValueError('排程模型与最终依赖图的完成时间不符')
    wait_plan,translated = _waits(nodes,required)
    schedule['per_core_sequences']=queues
    schedule['wait_plan']=wait_plan
    audit={'policy':policy,'members':len(nodes),'scheduling_units':len(bundles),
           'cost_profile':str(profile),'baseline_same_cost_estimate_us':baseline,
           'candidate_same_cost_estimate_us':final_estimate,'estimated_change_us':final_estimate-baseline,
           'cost_is_device_prediction':False,'math_bodies_changed':False,
           'row_dependency_contracts_retained':True,'original_core_order_preserved':False,
           'mixed_group_bijection':True,'ffts_group_resource_edges':216,'wait_plan':wait_plan,
           'device_validated':False}
    schedule['member_schedule']=audit
    schedule['source_kind']='shape_member_dependency_schedule'
    schedule['balanced_schedule'].update(member_schedule=audit,
                                         original_placement_superseded=True)
    schedule['row_dependencies'].update(
        original_core_order_and_mapping_preserved=False,
        placement_and_wait_counts_superseded_by='member_schedule')
    if output is not None:
        output=Path(output)
        output.mkdir(parents=True,exist_ok=False)
        (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
        (output/'schedule.json').write_text(json.dumps(schedule,ensure_ascii=False,indent=2)+'\n')
        (output/'logical_and_physical_graph.json').write_text(json.dumps({
            'nodes':[{k:r[k] for k in ('task','logical_block','original_subslot','core','dispatch_seq')} for r in nodes],
            'math_parents':[sorted(p) for p in parents],
            'required_parents':[sorted(p) for p in required],
            'translated_parents':[sorted(p) for p in translated]},ensure_ascii=False)+'\n')
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule',type=Path,required=True)
    parser.add_argument('--row-audit',type=Path,required=True)
    parser.add_argument('--cost-profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy',choices=['critical','earliest','finish'],required=True)
    args=parser.parse_args()
    print(json.dumps(reschedule(json.loads(args.schedule.read_text()),json.loads(args.row_audit.read_text()),
                               args.cost_profile,policy=args.policy,output=args.output),ensure_ascii=False,indent=2))
