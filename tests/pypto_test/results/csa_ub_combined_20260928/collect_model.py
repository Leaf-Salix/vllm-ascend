"""收集组合源码的EP16证据，按128K/8K七三权重记录forward变化。"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_module(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    collector = load_module(
        'ordered_forward_collector', ROOT.parent / 'csa_source_split_ab_20260927/ordered/collect_model.py'
    )
    formatter = load_module(
        'prior_result_formatter', ROOT.parent / 'csa_ascendc_topk_hc_ep16_20260928/collect_model.py'
    )
    collector.ROOT = ROOT / 'model'
    collector.CASES = ((131072, 16), (8192, 16))
    collector.OPERATOR_REVISION = ('2d2f9ca0: Top-K UB + fixed QR input/gamma residency over d1f170ff; '
                                   'prewarmed events; atomic0/mode2/det0')
    collector.main()
    path = collector.ROOT / 'forward.json'
    data = json.loads(path.read_text())
    measured = [row for row in data['cases'] if 'forward' in row]
    observations = {'cases_with_comparable_timing': len(measured)}
    for label, field, metric in (
        ('lower_forward_mean_cases', 'forward', 'mean_us'),
        ('lower_forward_p95_cases', 'forward', 'p95_us'),
        ('lower_forward_max_cases', 'forward', 'max_us'),
        ('lower_slowest_rank_mean_cases', 'slowest_rank_forward', 'mean_us'),
    ):
        observations[label] = sum(row[field]['pto'][metric] < row[field]['native'][metric] for row in measured)
    if len(measured) == 2 and all(not row['errors'] for row in measured):
        by_history = {row['history']: row for row in measured}
        observations['weighted_forward_change_pct'] = (
            0.7 * by_history[131072]['change_pct'] + 0.3 * by_history[8192]['change_pct']
        )
    observations['scope'] = ('同轮Native为基线，128K/8K变化率按7:3；不代表对旧PTO的单因素收益。'
                             'P95/最慢rank与token/DSpark单列，不以加权均值覆盖异常。')
    data['performance_observations'] = observations
    path.write_text(formatter.format_json(data) + '\n')
    if 'weighted_forward_change_pct' in observations:
        with (collector.ROOT / 'RESULTS.md').open('a') as output:
            output.write(
                '\n128K/8K的正式forward均值变化率按7:3加权为'
                f"{observations['weighted_forward_change_pct']:+.3f}%。\n"
                '同轮Native为基线，不代表对旧PTO的单因素收益。各档P95、逐步最慢rank及'
                'token/DSpark单独列示，不能由均值权重抵消异常。\n'
            )
    print(observations)


if __name__ == '__main__':
    main()
