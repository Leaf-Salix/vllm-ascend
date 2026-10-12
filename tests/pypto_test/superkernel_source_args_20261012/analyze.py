"""Summarize the fixed original/candidate/original sequence without selecting samples."""
import json
from pathlib import Path
from statistics import mean, median

root = Path(__file__).resolve().parent
runs = {}
for name in ('control_before', 'candidate', 'control_after'):
    record = json.loads((root / name / 'results.json').read_text())
    assert record['status'] == 'FUNCTIONAL_AND_TIMING_COMPLETE'
    assert record['variants'] == ['plain_score', 'early_score']
    assert len(record['samples']) == 60
    by_mode = [{sample['repeat']: sample['us'] for sample in record['samples'] if sample['mode'] == mode}
               for mode in (0, 1)]
    assert set(by_mode[0]) == set(by_mode[1]) == set(range(30))
    pairs = [by_mode[1][i] - by_mode[0][i] for i in range(30)]
    runs[name] = {
        'stats': {row['path']: row for row in record['summary']},
        'early_minus_plain_paired_us': {'mean': mean(pairs), 'median': median(pairs),
                                       'early_faster_pairs': sum(v < 0 for v in pairs), 'count': 30},
        'checks': {'eager_exact_states': len(record['eager_exact']),
                   'changed_input_exact_states': len(record['changed_input_exact']),
                   'guards': {path: {'count': len(checks), 'all_pass': all(v['status'] == 'PASS' for v in checks.values())}
                              for path, checks in record['guards'].items()}},
    }
comparison = {'runs': runs, 'candidate_minus_original_us': {}}
for control in ('control_before', 'control_after'):
    comparison['candidate_minus_original_us'][control] = {
        path: {stat: runs['candidate']['stats'][path][stat] - runs[control]['stats'][path][stat]
               for stat in ('min_us', 'mean_us', 'max_us')}
        for path in ('plain_score', 'early_score')}
comparison['early_mean_vs_control_midpoint_us'] = runs['candidate']['stats']['early_score']['mean_us'] - mean(
    runs[control]['stats']['early_score']['mean_us'] for control in ('control_before', 'control_after'))
(root / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
print(json.dumps(comparison, indent=2))
