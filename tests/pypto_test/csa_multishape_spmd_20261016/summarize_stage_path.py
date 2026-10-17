# SPDX-License-Identifier: Apache-2.0
"""汇总已通过用例的阶段探针，沿最晚显式前驱回溯实际完成链。

不是硬件忙时或理论下限分析：数学区间还包含核内FFTS等待，前驱发布时戳
略晚于实际notify，after_ready也含未记录控制和缓存维护。不能替代正式L1性能。
"""

import argparse
import collections
import json
import statistics
from pathlib import Path


def summarize(case, output):
    case, output = Path(case), Path(output)
    result = json.loads((case/'result.json').read_text())
    if not result.get('passed') or not result.get('stage_clock_enabled'):
        raise ValueError('只分析已完整通过的阶段探针用例')
    if result.get('eligible_for_700us_target'):
        raise ValueError('插桩用例必须排除在正式性能验收之外')
    rows = json.loads((case/'stage_clock/stages.json').read_text())
    sources = list((case/'specialization/bindings').glob('*_prepared.json'))
    if len(sources) != 1:
        raise ValueError('用例的实际调度表不唯一')
    schedule = json.loads(sources[0].read_text())
    observed = {(r['core'],r['sequence']):r for r in rows}
    scheduled = {(r['core'],r['dispatch_seq']):r
                 for sequence in schedule['per_core_sequences'].values() for r in sequence}
    if len(observed) != len(rows) or observed.keys() != scheduled.keys():
        raise ValueError('阶段记录与实际分核不是一一对应')
    origin = min(r['times_us'][0] for r in rows)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['kernel']].append(row)
    kernels = [{
        'kernel':name, 'members':len(items),
        'first_math_us':min(r['times_us'][1] for r in items)-origin,
        'last_math_us':max(r['times_us'][2] for r in items)-origin,
        'math_and_fence_median_us':statistics.median(r['math_and_fence_us'] for r in items),
        'math_and_fence_max_us':max(r['math_and_fence_us'] for r in items),
    } for name,items in groups.items()]
    kernels.sort(key=lambda r:r['first_math_us'])
    key = max(observed,key=lambda k:observed[k]['times_us'][3])
    chain, visited = [], set()
    while True:
        if key in visited:
            raise ValueError('显式前驱图出现环')
        visited.add(key)
        row, node = observed[key], scheduled[key]
        parents = [(int(c),seq) for c,seq in node['wait_for_core_progress'].items()]
        if key[1]>1:
            parents.append((key[0],key[1]-1))
        parent = max(parents,key=lambda p:observed[p]['times_us'][3]) if parents else None
        ready = observed[parent]['times_us'][3] if parent else origin
        chain.append({
            'core':key[0], 'sequence':key[1], 'kernel':row['kernel'],
            'math_start_us':row['times_us'][1]-origin, 'finish_us':row['times_us'][3]-origin,
            'math_and_notify_us':row['times_us'][3]-row['times_us'][1],
            'after_latest_parent_us':row['times_us'][1]-ready,
            'from_own_sequence':parent is not None and parent[0]==key[0],
        })
        if parent is None:
            break
        key = parent
    chain.reverse()
    output.mkdir(parents=True,exist_ok=True)
    common = {'case':str(case), 'eligible_for_700us_target':False,
              'scope':'已通过的单次插桩观测；非硬件下限，after_ready不等于软同步耗时'}
    (output/'kernels.json').write_text(json.dumps({**common,'kernels':kernels},ensure_ascii=False,indent=2)+'\n')
    report = {**common, 'math_and_notify_sum_us':sum(r['math_and_notify_us'] for r in chain),
              'after_latest_parent_sum_us':sum(r['after_latest_parent_us'] for r in chain),
              'instrumented_span_us':max(r['times_us'][3] for r in rows)-origin, 'chain':chain}
    (output/'latest_parent_chain.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return {k:v for k,v in report.items() if k!='chain'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.case,args.output),ensure_ascii=False,indent=2))
