"""定向对照CSA候选：固定规约状态、原始计时及独立窗口关键链。"""

import argparse
import collections
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402

STATE_NAMES = ('x_out', 'idx_topk', 'swa.0', 'compressed.0', 'state.0',
               'indexer.0', 'indexer.1', 'indexer_state.0')


def read(path):
    return json.loads(path.read_text())


def stats(samples):
    ordered = sorted(samples)
    p50 = statistics.median(samples)
    return {'samples_us': samples, 'mean_us': statistics.mean(samples), 'p50_us': p50,
            'p95_us': ordered[math.ceil(.95 * len(ordered)) - 1], 'max_us': ordered[-1],
            'over_p50_2pct': sum(v > p50 * 1.02 for v in samples),
            'over_p50_5pct': sum(v > p50 * 1.05 for v in samples)}


def scheduling(path, canonical):
    events = read(path)['traceEvents']
    pids = {e['args']['name']: e['pid'] for e in events if e.get('name') == 'process_name'}
    workers = [e for e in events if e.get('pid') == pids['Worker View'] and e.get('ph') == 'X'
               and 'kernel-duration-us' in e.get('args', {})]
    origin = min(e['ts'] for e in workers)
    track_names = {e['tid']: e['args']['name'] for e in events
                   if e.get('name') == 'thread_name' and e.get('pid') == pids['Worker View']}

    def assignment_for(task):
        result = {}
        for core in ('aic', 'aiv'):
            blocks = [e for e in workers if canonical(e['name']) == task + '_' + core]
            assignment = collections.Counter(e['tid'] for e in blocks)
            result[core] = {
                'blocks': len(blocks), 'used_cores': len(assignment),
                'max_blocks_per_core': max(assignment.values(), default=0),
                'blocks_per_track': {track_names[t]: n for t, n in sorted(assignment.items())},
                'unused_tracks': [name for tid, name in sorted(track_names.items())
                                  if name.startswith(core.upper() + '_') and tid not in assignment],
                'reused_track_events': [
                    {'track': track_names[e['tid']], 'task_id': e['args']['taskId'],
                     'receive_us': e['ts'] - origin, 'end_us': e['ts'] + e['dur'] - origin,
                     'kernel_us': e['args']['kernel-duration-us']}
                    for e in blocks if assignment[e['tid']] > 1],
            }
        return result

    drains = [{'thread': e['tid'], 'start_us': e['ts'] - origin, 'duration_us': e['dur'],
               'staged_blocks': e['args']['tasks_processed']}
              for e in events if e.get('pid') == pids['AICPU Scheduler'] and e.get('ph') == 'X'
              and e.get('args', {}).get('phase') == 'drain']
    # 三个调度线程的外层区间会重叠；只计时间并集，不累加线程耗时。
    intervals = sorted((e['start_us'], e['start_us'] + e['duration_us']) for e in drains)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return {'score_assignment': assignment_for('indexer_score_topk_native_pair'),
            'sparse_assignment': assignment_for('qk_pv'), 'drain_markers': drains,
            'drain_marker_union_us': sum(end - start for start, end in merged),
            'drain_limit': '仅实际stage的drain才有DFX标记，未包含所有无进展重试；不能当作全部暂停派发时间。'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--history', type=int, default=8192)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--windows', type=int, default=4)
    parser.add_argument('--task', default='task_20260928_112109_314698014843')
    parser.add_argument('--operator', default='e58ddc94 + short Score sync_start=True; early_resolve remains True')
    args = parser.parse_args()
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location(
        'worker', ROOT.parent / 'csa_scheduling_20260927/upstream_725/compare.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    folder = args.root / f'h{args.history}_b{args.batch}'
    reports = {s: read(folder / s / 'report.json') for s in ('baseline', 'candidate')}
    before, after = (reports[s] for s in ('baseline', 'candidate'))
    errors = []
    if before['history'] != args.history or before['batch'] != args.batch:
        errors.append('输入档位与报告不一致')
    for field in ('checkpoint', 'seed', 'batch', 'history', 'layer_index', 'weight_nz_mode', 'variant',
                  'deterministic_level', 'hccl_deterministic', 'pto_reduction', 'accuracy_fixture'):
        if before[field] != after[field]:
            errors.append(f'配置不同: {field}')
    if before['timing']['device'] != after['timing']['device']:
        errors.append('两侧非同卡')
    states = {s: torch.load(folder / s / 'states.pt', map_location='cpu', weights_only=True)['pto']
              for s in reports}
    if any(set(s) != set(STATE_NAMES) for s in states.values()):
        raise ValueError('保存的PTO状态项不完整')
    checks = {name: compare_tensor(states['candidate'][name], states['baseline'][name], 0, 0)
              for name in STATE_NAMES}
    if any(c['status'] != 'PASS' for c in checks.values()):
        errors.append('PTO候选跨版本状态不一致')
    graph = after.get('graph', after.get('graph_replay', {}))
    if graph.get('status') != 'PASS':
        errors.append('候选A→B→A图重放失败')
    measurements = {}
    for side, report in reports.items():
        if report['timing']['iters'] != args.iters:
            errors.append(f'{side}: 计时样本数不等于{args.iters}')
        for values in (report['pto_self'], report['timing']['pto']['eager_comparison']):
            if any(c['status'] != 'PASS' for c in values.values()):
                errors.append(f'{side}: 固定规约自重放不一致')
        guards = [*report['native_guards'], *report['pto_guards'], report['timing']['pto']['guards']]
        if any(c['status'] != 'PASS' for g in guards for c in g.values()):
            errors.append(f'{side}: metadata/保护区失败')
        if report['topk_selection']['structural_errors']:
            errors.append(f'{side}: Top-K结构失败')
        swimlane_path = folder / 'swimlane' / side / 'report.json'
        windows = read(swimlane_path)['swimlane_windows']
        if len(windows) != args.windows:
            errors.append(f'{side}: DFX窗口数不等于{args.windows}')
        measurements[side] = {'timing_source': str(folder / side / 'report.json'),
                              'timing': {s: stats(report['timing'][s]['samples_us']) for s in ('pto', 'native')},
                              'worker_windows': [worker.summarize(Path(w['merged_swimlane'])) for w in windows],
                              'scheduler_windows': [scheduling(Path(w['merged_swimlane']), worker.canonical)
                                                    for w in windows]}
    result = {'operator': args.operator,
              'task': args.task,
              'scope': f'H{args.history}/B{args.batch}/S6，layer4真实权重/合成历史，'
                       'mode2/atomic0/det0；单卡图重放。',
              'limits': 'PTO八类状态精确比较不含idx_topk_scores；det0 Native浮点仅作控制，'
                        '不要求Native跨进程逐bit一致。独立DFX不与无profiler计时逐个对应，非真实EP16验收。',
              'checks': checks, 'graph_status': graph.get('status'), 'errors': errors,
              'timing_iters': args.iters, 'dfx_windows': args.windows,
              'status': 'FAIL' if errors else 'PASS', 'measurements': measurements}
    (args.output or args.root / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(result['status'], errors)
    for side, value in measurements.items():
        print(side, {k: v for k, v in value['timing']['pto'].items() if k != 'samples_us'})
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
