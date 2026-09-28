"""只读已有模型图和单层泳道，保留两次独立采集的边界。"""

import collections
import importlib.util
import json
import math
import statistics
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT.parents[1]
BUNDLE = TESTS / 'downloads/DSV4_7cases_20260928_30f2b228'


def read(path):
    return json.loads(path.read_text())


def stats(values):
    ordered = sorted(values)
    return {'samples': len(values), 'min_us': ordered[0], 'mean_us': statistics.mean(values),
            'p50_us': statistics.median(values), 'p95_us': ordered[math.ceil(.95 * len(values)) - 1],
            'max_us': ordered[-1]}


def main():
    model_path = BUNDLE / '04_8K_B16_02_PTO_PyTorch_FullModel_Rank00.json'
    swimlane_path = BUNDLE / '04_8K_B16_03_PTO_Swimlane_SingleCSA_SyntheticHistory.json'
    gap = read(ROOT.parent / 'csa_qa_matrix_20260928/model/model_gap_rank0.json')
    intervals = next(c['pto']['intervals'] for c in gap['cases'] if c['history'] == 8192 and c['batch'] == 16)
    data = read(model_path)
    events = data['traceEvents'] if isinstance(data, dict) else data
    kernels = sorted((e for e in events if e.get('ph') == 'X' and
                      e.get('name') == 'aicore_kernel_mode_0_mix_aic'), key=lambda e: Decimal(e['ts']))
    if len(kernels) != 63 or len(intervals) != 63:
        raise ValueError('模型3步×21个CSA样本不完整')
    rows = []
    for event, interval in zip(kernels, intervals):
        start_ns = Decimal(event['ts']) * 1000
        end_ns = start_ns + Decimal(str(event['dur'])) * 1000
        if not interval['start_ns'] <= start_ns < end_ns <= interval['end_ns']:
            raise ValueError('CSA层号映射与实际设备时间不一致')
        rows.append({'step': interval['step'] + 1, 'layer': interval['layer'],
                     'aic_us': event['dur'], 'aicpu_us': interval['body_us'],
                     'aic_ts_us': event['ts'], 'aic_task_id': event['args']['Task Id']})
    selected = [r for r in rows if r['step'] == 3 and r['layer'] in (12, 14)]
    if [r['aic_us'] for r in selected] != [765.84, 807.84]:
        raise ValueError('用户指出的相邻CSA样本不匹配')
    overlaps = []
    for row in selected:
        start = Decimal(row['aic_ts_us'])
        end = start + Decimal(str(row['aic_us']))
        concurrent = [e for e in events if e.get('ph') == 'X' and e.get('args', {}).get('Task Type')
                      and Decimal(str(e.get('ts', 0))) < end
                      and Decimal(str(e.get('ts', 0))) + Decimal(str(e.get('dur', 0))) > start]
        overlaps.append({'step': row['step'], 'layer': row['layer'],
                         'task_names': dict(collections.Counter(e['name'] for e in concurrent))})

    spec = importlib.util.spec_from_file_location(
        'worker', ROOT.parent / 'csa_scheduling_20260927/upstream_725/compare.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    swimlane = read(swimlane_path)
    summary = worker.summarize(swimlane_path)
    pids = {e['pid'] for e in swimlane['traceEvents'] if e.get('name') == 'process_name'
            and e.get('args', {}).get('name') == 'Worker View'}
    score = {}
    for core in ('aic', 'aiv'):
        name = 'indexer_score_topk_native_pair_' + core
        blocks = [e for e in swimlane['traceEvents'] if e.get('pid') in pids and e.get('ph') == 'X'
                  and worker.canonical(e.get('name', '')) == name]
        assignment = collections.Counter(e['tid'] for e in blocks)
        score[core] = {**summary['tasks'][name], 'used_cores': len(assignment),
                       'max_blocks_per_core': max(assignment.values()), 'per_core_blocks': dict(assignment)}
    # 同一e58算子、同输入的两个既有窗口，只作另一份调度证据，不映射到模型的两个层。
    previous = read(ROOT.parent / 'csa_qr_k512_20260928/report.json')
    current = next(c for c in previous['cases'] if c['history'] == 8192)
    prior = []
    for window in current['measurements']['baseline']['worker_windows']:
        prior.append({k: window[k] for k in ('path', 'worker_span_us', 'phases_us')})
        prior[-1]['score_aic'] = window['tasks']['indexer_score_topk_native_pair_aic']
    result = {
        'model_source': str(model_path), 'swimlane_source': str(swimlane_path), 'operator': '30f2b228',
        'scope': '模型为EP16 rank0独立3步；泳道为单卡layer4合成历史的一次调用，二者非同一运行。',
        'selected_adjacent_csas': selected, 'selected_overlaps': overlaps,
        'aic_all_63': stats([r['aic_us'] for r in rows]),
        'aicpu_body_all_63': stats([r['aicpu_us'] for r in rows]), 'model_intervals': rows,
        'swimlane': {'launches': swimlane.get('metadata', {}).get('run_boundaries'),
                     'worker_span_us': summary['worker_span_us'], 'score': score},
        'separate_e58_two_windows': prior,
        'limits': 'PyTorch图不含内部task；rank0未见其他计算/HCCL任务重叠不排除其他rank争用。'
                  '单窗口启动分散与相邻层42us变化尚无因果映射；sync_start收益待单变量对照。',
    }
    (ROOT / 'existing_trace.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('selected_adjacent_csas', 'aic_all_63', 'aicpu_body_all_63')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
