"""定向对照短档 Score sync_start：固定规约状态、100次计时及四窗口关键链。"""

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


def main():
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location(
        'worker', ROOT.parent / 'csa_scheduling_20260927/upstream_725/compare.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    folder = ROOT / 'h8192_b16'
    reports = {s: read(folder / s / 'report.json') for s in ('baseline', 'candidate')}
    before, after = (reports[s] for s in ('baseline', 'candidate'))
    errors = []
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
        errors.append('PTO调度候选跨版本状态不一致')
    graph = after.get('graph', after.get('graph_replay', {}))
    if graph.get('status') != 'PASS':
        errors.append('候选A→B→A图重放失败')
    measurements = {}
    for side, report in reports.items():
        if report['timing']['iters'] != 100:
            errors.append(f'{side}: 计时样本不足100')
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
        if len(windows) != 4:
            errors.append(f'{side}: DFX窗口不足4')
        measurements[side] = {'timing_source': str(folder / side / 'report.json'),
                              'timing': {s: stats(report['timing'][s]['samples_us']) for s in ('pto', 'native')},
                              'worker_windows': [worker.summarize(Path(w['merged_swimlane'])) for w in windows]}
    result = {'operator': 'e58ddc94 + short Score sync_start=True; early_resolve remains True',
              'task': 'task_20260928_112109_314698014843',
              'scope': 'H8192/B16/S6，layer4真实权重/合成历史，mode2/atomic0/det0；单卡图重放。',
              'limits': 'PTO八类状态精确比较不含idx_topk_scores；det0 Native浮点仅作控制，'
                        '不要求Native跨进程逐bit一致。独立DFX不与100次计时逐个对应，非真实EP16验收。',
              'checks': checks, 'graph_status': graph.get('status'), 'errors': errors,
              'status': 'FAIL' if errors else 'PASS', 'measurements': measurements}
    (ROOT / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(result['status'], errors)
    for side, value in measurements.items():
        print(side, {k: v for k, v in value['timing']['pto'].items() if k != 'samples_us'})
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
