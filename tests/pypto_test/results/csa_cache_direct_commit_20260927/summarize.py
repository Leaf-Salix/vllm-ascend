"""Read-only comparison against the completed 3d1f0f65 matrix."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main():
    common = module('common', RESULTS / 'csa_native_cube_matrix_20260927/summarize.py')
    worker = module('worker', RESULTS / 'csa_scheduling_20260927/upstream_725/compare.py')
    baseline = json.loads((RESULTS / 'csa_incore_20260927/final_3d1f0f65/cases.json').read_text())
    old = {(c['history'], c['batch']): c for c in baseline['cases']}
    rows = []
    for history in (8192, 131072):
        folder = RESULTS / 'csa_split_optimization_20260927/cache_direct_commit' / f'h{history}_b16'
        if not (folder / 'timing/report.json').exists():
            continue
        measured = common.summarize_timing(folder / 'timing/report.json')
        raw = json.loads((folder / 'timing/report.json').read_text())
        phases = raw['timing']['split_cache_phases']
        previous = old[(history, 16)]
        row = {'history': history, 'batch': 16, 'baseline_revision': '3d1f0f65',
               'before': previous['timing'], 'after': measured,
               'native_commit_inside_csa': phases['native_commit_inside_csa'],
               'native_slot_updates_equal': phases['native_slot_updates_equal'],
               'phase_guard_failures': common.guard_failures(phases['guards']),
               'full_change_pct': (measured['pto_full']['mean_us'] / previous['timing']['pto_full']['mean_us'] - 1) * 100,
               'full_vs_native_pct': (measured['pto_full']['mean_us'] / measured['native']['mean_us'] - 1) * 100,
               'windows': []}
        if (folder / 'swimlane/report.json').exists():
            swimlane = json.loads((folder / 'swimlane/report.json').read_text())
            row['swimlane_guard_failures'] = common.guard_failures(swimlane['pto_guards'])
            for item in swimlane['swimlane_windows']:
                window = worker.summarize(Path(item['merged_swimlane']))
                row['windows'].append({'path': window['path'], 'worker_span_us': window['worker_span_us'],
                                       'tasks': {name: task for name, task in window['tasks'].items()
                                                 if name in ('kv_and_cache_write', 'kv_and_cache_write_0',
                                                             'idx_kv_scale_commit', 'indexer_score_topk_native_pair_aic',
                                                             'indexer_score_topk_native_pair_aiv')}})
        rows.append(row)
    padding_path = ROOT / 'padding/report.json'
    padding = None
    if padding_path.exists():
        raw = json.loads(padding_path.read_text())
        check = raw.get('padding_graph', {})
        padding = {'report': str(padding_path), 'status': check.get('status'),
                   'active_batches': [r['active_batch'] for r in check.get('replays', [])],
                   'failures': common.guard_failures(check)}
    result = {'task_ids': ['task_20260927_205352_412770127178', 'task_20260927_205353_412802212678'],
              'complete': len(rows) == 2 and all(len(r['windows']) == 4 for r in rows) and padding is not None,
              'cases': rows, 'padding': padding,
              'scope': 'B16 layer4 formal weights plus synthetic history; metadata reuse, mode2, atomic1, no EPLB; five warmups and 20 unprofiled samples; four independent DFX windows. Padding check B4/H4095 uses atomic0 and deterministic1.',
              'limits': 'External writeback is removed, so its zero is structural, not an empty-graph measurement. Body includes direct Native slot updates. Entrance still copies history. Not seven-case or 16-card acceptance.'}
    (ROOT / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for row in rows:
        print(row['history'], 'full', row['before']['pto_full']['mean_us'], '->', row['after']['pto_full']['mean_us'],
              'native', row['after']['native']['mean_us'], 'body', row['after']['body']['mean_us'],
              'guards', row['after']['guard_failures'], 'slots_equal', row['native_slot_updates_equal'])
    print('complete', result['complete'], 'padding', padding)


if __name__ == '__main__':
    main()
