"""Compare an incore candidate with the retained ec12e0e9 PV baseline."""
import argparse
import importlib.util
import json
import statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('output', type=Path)
    parser.add_argument('--history', type=int, nargs='+', default=[8192, 131072])
    args = parser.parse_args()
    timing = load('timing', RESULTS / 'csa_native_cube_matrix_20260927/summarize.py')
    worker = load('worker', RESULTS / 'csa_scheduling_20260927/upstream_725/compare.py')
    selected = ('csa_slots_build_valid_qk_plan', 'qk_pv_aic', 'qk_pv_aiv', 'merge_norm',
                'indexer_score_topk_native_pair_aic', 'indexer_score_topk_native_pair_aiv', 'indexer_topk_query_merge')
    cases = []
    for history in args.history:
        for label in ('incore_sparse_pv_n128_pair', args.label):
            directory = RESULTS / 'csa_split_optimization_20260927' / label / f'h{history}_b16'
            raw = json.loads((directory / 'swimlane/report.json').read_text())
            windows = [worker.summarize(Path(w['merged_swimlane'])) for w in raw['swimlane_windows']]
            windows = [{key: w[key] for key in ('path', 'worker_span_us', 'phases_us')} |
                       {'tasks': {name: w['tasks'][name] for name in selected if name in w['tasks']}}
                       for w in windows]
            tasks = {}
            for name in windows[0]['tasks']:
                values = [w['tasks'][name]['kernel_mean_us'] for w in windows]
                tasks[name] = {'kernel_mean_us': statistics.mean(values), 'window_mean_min_us': min(values), 'window_mean_max_us': max(values),
                               'blocks': [w['tasks'][name]['blocks'] for w in windows]}
            cases.append({'label': label, 'history': history, 'batch': 16,
                          'timing': timing.summarize_timing(directory / 'timing/report.json'),
                          'tasks': tasks, 'windows': windows})
    report = {'baseline_commit': 'ec12e0e9', 'baseline_label': 'incore_sparse_pv_n128_pair', 'candidate_label': args.label,
              'scope': 'Layer4 real weights + synthetic input/history; second CSA metadata reuse; TP1/S6/mode2/atomic1/deterministic0, no EPLB; 5/20 unprofiled and four independent DFX windows.',
              'limits': 'Worker block duration includes in-task waits. Native complete kernels are not per-block equivalents. Not whole-model token/DSpark acceptance.', 'cases': cases}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for c in cases:
        print(c['history'], c['label'], 'timing', json.dumps(c['timing']))
        print(json.dumps({name: c['tasks'][name] for name in selected if name in c['tasks']}))
if __name__ == '__main__':
    main()
