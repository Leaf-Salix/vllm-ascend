"""Summarize the paired layout experiment without discarding samples."""

import json
import statistics
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    candidate = 'coalesced' if root.name == 'coalesced' else 'split'
    result = {'base': '2a740c1f', 'candidate': candidate,
              'page_order': 'consecutive diagnostic' if candidate == 'coalesced' else 'nonconsecutive/reverse',
              'cases': [], 'errors': []}
    lines = ['| 档位/状态 | 原布局CSA μs | 候选CSA μs | 变化 | 原/新P95 μs | 原/新max μs |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for history in (8192, 131072):
        accuracy = json.loads((root / f'accuracy/h{history}_b4/comparison.json').read_text())
        if accuracy['status'] != 'PASS':
            result['errors'].append(f'{history}: accuracy')
        sides = {}
        for layout in ('original', candidate):
            record = json.loads((root / f'timing/h{history}_b16/{layout}/cache_pressure.json').read_text())
            if record['status'] != 'MEASURED':
                result['errors'].append(f'{history}/{layout}: incomplete')
            sides[layout] = record
            for side in ('native', 'pto'):
                for state in ('normal', 'pressure'):
                    data = record['measurements'][side][state]
                    if len(data['samples_us']) != 20 or any(c['status'] != 'PASS' for c in data['guards'].values()):
                        result['errors'].append(f'{history}/{layout}/{side}/{state}: samples/guards')
        for state in ('normal', 'pressure'):
            values = {layout: record['measurements']['pto'][state] for layout, record in sides.items()}
            means = {k: statistics.mean(v['samples_us']) for k, v in values.items()}
            delta = (means[candidate] / means['original'] - 1) * 100
            before, after = values['original'], values[candidate]
            row = {'history': history, 'batch': 16, 'state': state, 'means_us': means,
                   'change_percent': delta, 'measurements': values,
                   'native_means_us': {layout: record['measurements']['native'][f'{state}_mean_us']
                                       for layout, record in sides.items()},
                   'accuracy_status': accuracy['status']}
            result['cases'].append(row)
            lines.append(f'| {history}/B16 {state} | {means["original"]:.2f} | {means[candidate]:.2f} | '
                         f'{delta:+.2f}% | {before["us_p95"]:.2f}/{after["us_p95"]:.2f} | '
                         f'{before["us_max"]:.2f}/{after["us_max"]:.2f} |')
    result['status'] = 'FAIL' if result['errors'] else 'PASS'
    (root / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (root / 'TABLE.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(result['status'], result['errors'])
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
