"""Pair real replay routing snapshots; do not interpret diagnostic time as performance."""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from offline_pd.compare import _spec_stats  # noqa: E402
from offline_pd.performance import compare_worker_configs  # noqa: E402


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
              'scope': 'EP16, 3 post-warmup steps; exact request-input pairs selected explicitly, no timing claim',
              'errors': [], 'token_mismatches': 0, 'dspark_mismatched_ranks': [],
              'rank_configs': {}, 'layers': {}, 'paired_rank_steps': 0,
              'input_mismatches': [], 'matched_request_steps': 0, 'compared_request_steps': 0,
              'limits': '仅相同rank/step且整条S6输入token与position一致的请求比较专家选择；'
                        '整rank接收负载含未配对请求，不能据其变化归因于PTO数值。'}
    histograms, observed_groups = {}, {}
    for rank in range(16):
        pair = {side: json.loads((ROOT / 'retry' / side / f'rank{rank}.moe-routing.json').read_text())
                for side in ('native', 'pto')}
        native, pto = pair['native'], pair['pto']
        report['rank_configs'][str(rank)] = {k: v.get('worker_runtime_config') for k, v in pair.items()}
        compare_worker_configs(native['worker_runtime_config'][0], pto['worker_runtime_config'][0])
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
            for key in ('steady_step_index', 'forward_calls'):
                if before[key] != after[key]:
                    raise ValueError(f'Unpaired model input: rank={rank}, key={key}')
            step = before['steady_step_index']
            matched = []
            for request in range(40):
                region = slice(request * 6, (request + 1) * 6)
                matched.append(all(before[k][region] == after[k][region] for k in ('input_ids', 'positions')))
            report['compared_request_steps'] += 40
            report['matched_request_steps'] += sum(matched)
            report['paired_rank_steps'] += all(matched)
            if not all(matched):
                report['input_mismatches'].append({
                    'rank': rank, 'step': step, 'requests': [i for i, ok in enumerate(matched) if not ok],
                    'native_request_positions': before['positions'][::6],
                    'pto_request_positions': after['positions'][::6],
                })
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
                hist = histograms.setdefault((layer, step), {
                    side: {'full': [0] * 256, 'matched': [0] * 256} for side in ('native', 'pto')})
                for side, record in (('native', old), ('pto', new)):
                    for token, ids in enumerate(record['topk_ids']):
                        if record['active_mask'] is not None and not record['active_mask'][token]:
                            continue
                        for expert in ids:
                            if not 0 <= expert < 256:
                                raise ValueError('Unexpected global expert id')
                            hist[side]['full'][expert] += 1
                            if matched[token // 6]:
                                hist[side]['matched'][expert] += 1
                for token, (a, b) in enumerate(zip(old['topk_ids'], new['topk_ids'])):
                    if not matched[token // 6]:
                        continue
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
                    observed_groups[layer, step, side, rank] = values
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
    summary = {side: {'active_experts': [], 'max_rank_tokens': [], 'max_rank_active_experts': []}
               for side in ('native', 'pto')}
    mapping_checks = 0
    for (layer, step), hist in histograms.items():
        paired = report['layers'][str(layer)].setdefault('matched_input_load_by_step', {}).setdefault(str(step), {})
        for side in ('native', 'pto'):
            for rank in range(16):
                if hist[side]['full'][rank * 16:rank * 16 + 16] != observed_groups[layer, step, side, rank]:
                    raise ValueError('Global expert ids do not reconstruct actual MC2 group counts')
                mapping_checks += 1
            experts = hist[side]['matched']
            received = [sum(experts[i:i + 16]) for i in range(0, 256, 16)]
            active = [sum(x > 0 for x in experts[i:i + 16]) for i in range(0, 256, 16)]
            paired[side] = {'received_by_rank': received, 'active_experts_by_rank': active,
                            'active_experts': sum(active)}
            summary[side]['active_experts'].append(sum(active))
            summary[side]['max_rank_tokens'].append(max(received))
            summary[side]['max_rank_active_experts'].append(max(active))
    report['expert_mapping'] = {'experts_per_rank': 16, 'count_checks': mapping_checks,
                                'status': 'PASS', 'scope': '所有实际MC2逐专家计数与全量topk_ids重建一致'}
    report['matched_input_load_mean'] = {
        side: {key: statistics.mean(values) for key, values in data.items()} for side, data in summary.items()}
    report['changed_expert_set_tokens'] = sum(v['changed_expert_set_tokens'] for v in report['layers'].values())
    report['changed_group_rank_steps'] = sum(v['changed_group_rank_steps'] for v in report['layers'].values())
    report['status'] = 'INPUTS_DIFFER' if report['input_mismatches'] else 'PAIRED'
    (ROOT / 'routing_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print({k: v for k, v in report.items() if k not in ('layers', 'rank_configs', 'input_mismatches')})


if __name__ == '__main__':
    main()
