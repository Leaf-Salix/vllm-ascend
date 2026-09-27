"""Read the long-only Score admission candidate; never launch device work."""
import collections
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def tail(path):
    timing = json.loads(path.read_text())['timing']
    return {side: {'p95_over_p50_pct': (timing[side]['us_p95'] / timing[side]['us_p50'] - 1) * 100,
                   'p99_us': sorted(timing[side]['samples_us'])[math.ceil(0.99 * len(timing[side]['samples_us'])) - 1],
                   'max_us': max(timing[side]['samples_us']), 'samples_us': timing[side]['samples_us']}
            for side in ('native', 'pto')}


def window(path, worker):
    summary = worker.summarize(path)
    events = json.loads(path.read_text())['traceEvents']
    pids = {e['pid'] for e in events if e.get('ph') == 'M' and e.get('name') == 'process_name'
            and e.get('args', {}).get('name') == 'Worker View'}
    counts = collections.Counter(e['tid'] for e in events if e.get('pid') in pids and e.get('ph') == 'X'
                                 and worker.canonical(e.get('name', '')) == 'indexer_score_topk_native_pair_aic')
    return {'path': str(path), 'worker_span_us': summary['worker_span_us'],
            'score_aic_cores': len(counts), 'score_max_blocks_per_aic': max(counts.values()),
            'score_aic_blocks_per_core': dict(counts),
            'tasks': {name: task for name, task in summary['tasks'].items()
                      if name in ('indexer_score_topk_native_pair_aic', 'indexer_score_topk_native_pair_aiv',
                                  'indexer_topk_query_merge', 'qk_pv_aic')}}


def main():
    common = load('common', RESULTS / 'csa_native_cube_matrix_20260927/summarize.py')
    worker = load('worker', RESULTS / 'csa_scheduling_20260927/upstream_725/compare.py')
    old = {c['history']: c for c in json.loads((RESULTS / 'csa_cache_panel_read_20260927/report.json').read_text())['cases']}
    rows = []
    for history in (8192, 131072):
        folder = RESULTS / 'csa_split_optimization_20260927/cache_tail_guard' / f'h{history}_b16'
        path = folder / 'timing/report.json'
        if not path.exists():
            continue
        measured = common.summarize_timing(path)
        previous = old[history]['after']
        row = {'history': history, 'batch': 16, 'baseline_revision': '05cb2758',
               'before': previous, 'after': measured, 'tail': tail(path),
               'before_tail': tail(Path(previous['report'])),
               'full_change_pct': (measured['pto_full']['mean_us'] / previous['pto_full']['mean_us'] - 1) * 100,
               'p95_change_pct': (measured['pto_full']['p95_us'] / previous['pto_full']['p95_us'] - 1) * 100,
               'full_vs_native_pct': (measured['pto_full']['mean_us'] / measured['native']['mean_us'] - 1) * 100,
               'windows': [],
               'before_windows': [window(Path(w['path']), worker) for w in old[history]['windows']]}
        if (folder / 'swimlane/report.json').exists():
            raw = json.loads((folder / 'swimlane/report.json').read_text())
            row['swimlane_guard_failures'] = common.guard_failures(raw['pto_guards'])
            row['windows'] = [window(Path(w['merged_swimlane']), worker) for w in raw['swimlane_windows']]
        rows.append(row)
    confirmation = {}
    for variant in ('baseline', 'candidate'):
        path = ROOT / 'confirmation' / variant / 'report.json'
        if path.exists():
            confirmation[variant] = {'timing': common.summarize_timing(path), 'tail': tail(path)}
    result = {'task_ids': ['task_20260927_214806_30823321511', 'task_20260927_220159_43960632294'], 'retained': True,
              'complete': len(rows) == 2 and all(len(r['windows']) == 4 for r in rows), 'cases': rows,
              'confirmation': confirmation,
              'confirmation_complete': len(confirmation) == 2 and all(
                  c['timing']['pto_full']['samples'] == 100 for c in confirmation.values()),
              'scope': 'B16 layer4 formal weights plus synthetic history; metadata reuse, mode2, atomic1, no EPLB; five warmups and 20 unprofiled samples; four independent DFX windows.',
              'limits': 'Only long-history (>8192 compressed rows) Score metadata changes: sync_start True and early_resolve False. Short branch retains original flags. No new numerical/padding sweep for metadata-only change. Separate 100-sample before/after long-history confirmation does not prove EP16 tail acceptance.'}
    (ROOT / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for r in rows:
        print(r['history'], 'before', r['before']['pto_full'], 'after', r['after']['pto_full'],
              'tail%', r['tail']['pto']['p95_over_p50_pct'], 'guards', r['after']['guard_failures'])
    print('complete', result['complete'])
    for variant, measured in confirmation.items():
        print('confirmation', variant, measured['timing']['pto_full'],
              'tail%', measured['tail']['pto']['p95_over_p50_pct'],
              'p99', measured['tail']['pto']['p99_us'], 'max', measured['tail']['pto']['max_us'],
              'guards', measured['timing']['guard_failures'])


if __name__ == '__main__':
    main()
