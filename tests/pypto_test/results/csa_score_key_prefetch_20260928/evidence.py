"""从既有单卡汇总提取定向证据和长短七三变化率；不运行设备。"""

import argparse
import json
import statistics
from pathlib import Path

TASKS = (
    'indexer_score_topk_native_pair_aic', 'indexer_score_topk_native_pair_aiv',
    'indexer_topk_query_merge', 'qk_pv_aic', 'qk_pv_aiv', 'merge_norm',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--long-batch', type=int, default=16)
    parser.add_argument('--short-batch', type=int, default=16)
    parser.add_argument('--baseline', default='2d2f9ca0')
    args = parser.parse_args()
    cases = []
    for history, batch in ((131072, args.long_batch), (8192, args.short_batch)):
        folder = args.root / f'h{history}_b{batch}'
        source = json.loads((folder / 'summary.json').read_text())
        evidence = {key: value for key, value in source.items() if key != 'measurements'}
        measurements = {}
        for side, value in source['measurements'].items():
            measurements[side] = {
                **value,
                'worker_windows': [
                    {**window, 'tasks': {name: window['tasks'][name] for name in TASKS}}
                    for window in value['worker_windows']
                ],
            }
        evidence['measurements'] = measurements
        (folder / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')

        metrics = {}
        for metric in ('csa', 'score_aic', 'score_aiv'):
            values = {}
            for side, data in measurements.items():
                if metric == 'csa':
                    values[side] = data['timing']['pto']['mean_us']
                else:
                    name = 'indexer_score_topk_native_pair_' + metric.removeprefix('score_')
                    values[side] = statistics.mean(
                        window['tasks'][name]['kernel_mean_us'] for window in data['worker_windows']
                    )
            metrics[metric] = {
                **values,
                'change_pct': (values['candidate'] / values['baseline'] - 1) * 100,
            }
        cases.append({'history': history, 'batch': batch, 'status': source['status'], 'metrics': metrics})

    result = {
        'scope': f'同轮候选对{args.baseline}；无profiler完整CSA与独立四窗口DFX分别计算。',
        'limits': 'AIV核内包含等候AIC；未更改任务的波动不能归为候选收益。保留P95及所有原始样本。',
        'cases': cases,
        'weighted_change_pct': {
            metric: .7 * cases[0]['metrics'][metric]['change_pct']
            + .3 * cases[1]['metrics'][metric]['change_pct']
            for metric in ('csa', 'score_aic', 'score_aiv')
        },
    }
    (args.root / 'weighted.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
