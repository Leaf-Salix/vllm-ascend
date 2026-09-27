"""Pair real replay routing snapshots; do not interpret diagnostic time as performance."""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from offline_pd.compare import _spec_stats  # noqa: E402


def counts(record):
    values = record['group_list']
    if record['group_list_type'] == 0:
        values = [a - b for a, b in zip(values, [0] + values[:-1])]
    elif record['group_list_type'] != 1:
        raise ValueError(f'Unknown group list type: {record["group_list_type"]}')
    if not values or any(x < 0 for x in values):
        raise ValueError('Invalid expert counts')
    return values


def main():
    report = {'operator_revision': 'f76b3ad4', 'history': 8192, 'batch': 40,
              'scope': 'EP16, 3 paired post-warmup steps; routing integers only, no performance claim',
              'errors': [], 'token_mismatches': 0, 'dspark_mismatched_ranks': [],
              'rank_configs': {}, 'layers': {}, 'paired_rank_steps': 0}
    for rank in range(16):
        pair = {side: json.loads((ROOT / 'retry' / side / f'rank{rank}.moe-routing.json').read_text())
                for side in ('native', 'pto')}
        native, pto = pair['native'], pair['pto']
        report['rank_configs'][str(rank)] = {k: v.get('worker_runtime_config') for k, v in pair.items()}
        if len(native['output_token_ids']) != len(pto['output_token_ids']):
            raise ValueError('Different output request count')
        for left, right in zip(native['output_token_ids'], pto['output_token_ids']):
            if len(left) != len(right):
                raise ValueError('Different output token count')
            report['token_mismatches'] += sum(a != b for a, b in zip(left, right))
        if _spec_stats(native['spec_decode'], 5) != _spec_stats(pto['spec_decode'], 5):
            report['dspark_mismatched_ranks'].append(rank)
        windows = {side: item['window'][0] for side, item in pair.items()}
        for side, window in windows.items():
            if not window['sufficient'] or window['dp_rank'] != rank or len(window['samples']) != 3:
                raise ValueError(f'Incomplete rank window: {side}/{rank}')
        for before, after in zip(windows['native']['samples'], windows['pto']['samples']):
            for key in ('steady_step_index', 'input_ids', 'positions', 'forward_calls'):
                if before[key] != after[key]:
                    raise ValueError(f'Unpaired model input: rank={rank}, key={key}')
            step = before['steady_step_index']
            report['paired_rank_steps'] += 1
            for layer in range(43):
                old, new = (sample['layers'][str(layer)] for sample in (before, after))
                if old['active_mask'] != new['active_mask']:
                    raise ValueError('Different active-token masks')
                if len(old['topk_ids']) != 240 or len(new['topk_ids']) != 240:
                    raise ValueError('Unexpected captured bucket')
                row = report['layers'].setdefault(str(layer), {
                    'compared_tokens': 0, 'changed_ordered_tokens': 0, 'changed_expert_set_tokens': 0,
                    'changed_id_entries': 0, 'changed_group_rank_steps': 0,
                    'native_received_by_step': {}, 'pto_received_by_step': {},
                })
                for token, (a, b) in enumerate(zip(old['topk_ids'], new['topk_ids'])):
                    if old['active_mask'] is not None and not old['active_mask'][token]:
                        continue
                    if len(a) != len(b):
                        raise ValueError('Different experts per token')
                    row['compared_tokens'] += 1
                    row['changed_ordered_tokens'] += a != b
                    row['changed_expert_set_tokens'] += set(a) != set(b)
                    row['changed_id_entries'] += sum(x != y for x, y in zip(a, b))
                groups = {side: counts(value) for side, value in (('native', old), ('pto', new))}
                row['changed_group_rank_steps'] += groups['native'] != groups['pto']
                for side, values in groups.items():
                    row[f'{side}_received_by_step'].setdefault(str(step), {})[str(rank)] = sum(values)
    for key, row in report['layers'].items():
        row['load_by_step'] = {}
        for step in row['native_received_by_step']:
            row['load_by_step'][step] = {}
            for side in ('native', 'pto'):
                values = list(row[f'{side}_received_by_step'][step].values())
                if len(values) != 16:
                    raise ValueError('Missing EP rank load')
                mean = statistics.mean(values)
                row['load_by_step'][step][side] = {
                    'total': sum(values), 'max': max(values), 'min': min(values),
                    'max_over_mean': max(values) / mean if mean else None,
                }
    report['changed_expert_set_tokens'] = sum(v['changed_expert_set_tokens'] for v in report['layers'].values())
    report['changed_group_rank_steps'] = sum(v['changed_group_rank_steps'] for v in report['layers'].values())
    report['status'] = 'MEASURED'
    (ROOT / 'routing_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print({k: v for k, v in report.items() if k not in ('layers', 'rank_configs')})


if __name__ == '__main__':
    main()
