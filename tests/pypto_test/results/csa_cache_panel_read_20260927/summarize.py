"""Read-only comparison against bad985a9; never launches device work."""
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
    baseline = json.loads((RESULTS / 'csa_cache_direct_commit_20260927/report.json').read_text())
    old = {c['history']: c for c in baseline['cases']}
    direct = json.loads((RESULTS / 'csa_cache_direct_read_20260927/report.json').read_text())
    direct_old = {c['history']: c for c in direct['cases']}
    rows = []
    for history in (8192, 131072):
        folder = RESULTS / 'csa_split_optimization_20260927/cache_panel_read' / f'h{history}_b16'
        path = folder / 'timing/report.json'
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        measured = common.summarize_timing(path)
        binding = raw['indexer_cache_binding']
        zero_copy = (binding['history_copy_before_csa'] is False
                     and set(binding['root_views']) == {'idx_native_kv_cache'}
                     and all(v['storage_ptr'] == binding['native_storage_ptr']
                             for v in binding['root_views'].values()))
        previous = old[history]['after']
        row = {'history': history, 'batch': 16, 'baseline_revision': 'bad985a9',
               'before': previous, 'after': measured, 'binding': binding,
               'direct_before': direct_old[history]['after'], 'direct_before_windows': direct_old[history]['windows'],
               'single_native_root_without_history_copy': zero_copy,
               'full_change_pct': (measured['pto_full']['mean_us'] / previous['pto_full']['mean_us'] - 1) * 100,
               'full_vs_native_pct': (measured['pto_full']['mean_us'] / measured['native']['mean_us'] - 1) * 100,
               'windows': [], 'previous_windows': old[history]['windows']}
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
        check = json.loads(padding_path.read_text()).get('padding_graph', {})
        padding = {'report': str(padding_path), 'status': check.get('status'),
                   'active_batches': [r['active_batch'] for r in check.get('replays', [])],
                   'failures': common.guard_failures(check)}
    result = {'retained': True,
              'decision': 'Both complete paths improve against bad985a9; N128 panel loading improves incore versus the prior paged candidate. Score incore remains slower than the old contiguous-cache body; record this tradeoff.',
              'task_ids': ['task_20260927_213048_1480693352', 'task_20260927_213355_17426121650'],
              'complete': len(rows) == 2 and all(len(r['windows']) == 4 for r in rows) and padding is not None and padding['status'] == 'PASS',
              'cases': rows, 'padding': padding,
              'scope': 'B16 layer4 formal weights plus synthetic history; metadata reuse, mode2, atomic1, no EPLB; five warmups and 20 unprofiled samples; four independent DFX windows.',
              'limits': 'Only AIC N128 panel loading changed relative to the direct-read candidate; new B4/H32767 atomic0/deterministic1 padding run targets the Cube path; its unchanged Vector fallback reuses the prior candidate evidence. History copy and external commit absent. Native zero-tolerance differences independent of guards. Not seven-case/16-card acceptance.'}
    (ROOT / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for row in rows:
        print(row['history'], 'full', row['before']['pto_full']['mean_us'], '->', row['after']['pto_full']['mean_us'],
              'native', row['after']['native']['mean_us'], 'guards', row['after']['guard_failures'],
              'single_root', row['single_native_root_without_history_copy'])
    print('complete', result['complete'], 'padding', padding)


if __name__ == '__main__':
    main()
