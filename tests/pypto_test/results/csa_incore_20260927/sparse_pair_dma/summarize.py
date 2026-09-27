"""Read the existing paired-DMA diagnostic counters; never execute an NPU job."""

import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def counters(build):
    assignment = next(node for node in ast.parse((build / 'kernel_config.py').read_text()).body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'KERNELS' for t in node.targets))
    names = {}
    for item in assignment.value.elts:
        fields = {ast.literal_eval(k): v for k, v in zip(item.keys, item.values)}
        names[ast.literal_eval(fields['func_id'])] = ast.literal_eval(fields['name'])
    with (build / 'dfx_outputs/pmu.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    result = {'source': str(build / 'dfx_outputs/pmu.csv'), 'names': names}
    for name, expected in (('qk_pv_aic', 24), ('qk_pv_aiv', 48)):
        group = [r for r in rows if names[int(r['func_id'])] == name]
        if len(group) != expected or any(int(r['pmu_total_cycles']) <= 0 for r in group):
            raise ValueError(f'Incomplete counters: {name}, {build}')
        total = sum(int(r['pmu_total_cycles']) for r in group)
        result[name] = {'blocks': len(group), 'cycles_mean': total / len(group),
                        'busy_pct': {k: 100 * sum(int(r[k]) for r in group) / total
                                     for k in group[0] if k.endswith('_busy_cycles')}}
    return result


def main():
    result = {'base_revision': 'cd910e2c', 'main_inspected': 'b046b15c',
              'scope': 'One standalone program PMU sample per variant; not full CSA steady timing',
              'task_ids': ['task_20260927_162723_20380203069', 'task_20260927_163121_205364024952']}
    builds = {'current': ROOT.parent / 'sparse_kv_early/gated_standalone/build',
              'pair': ROOT / 'build', 'pair_chunked': ROOT / 'chunked/build'}
    for label, build in builds.items():
        result[label] = counters(build)
        if label != 'current':
            result[label]['aic_change_pct'] = 100 * (
                result[label]['qk_pv_aic']['cycles_mean'] / result['current']['qk_pv_aic']['cycles_mean'] - 1)
            report = ROOT / ('chunked/report.json' if label == 'pair_chunked' else 'report.json')
            result[label]['comparison'] = json.loads(report.read_text())['comparison']
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for label in builds:
        print(label, result[label]['qk_pv_aic']['cycles_mean'], result[label]['qk_pv_aiv']['cycles_mean'])


if __name__ == '__main__':
    main()
