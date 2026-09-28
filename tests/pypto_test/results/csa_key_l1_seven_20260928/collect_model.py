"""同一源码七档正式forward，历史长度内部等权后按七三汇总。"""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = '554b3bca'
CASES = ((131072, 4), (131072, 8), (131072, 16),
         (8192, 16), (8192, 24), (8192, 32), (8192, 40))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weighted_changes(rows):
    if len(rows) != len(CASES) or {(r['history'], r['batch']) for r in rows} != set(CASES):
        raise ValueError('七档必须齐全且不重复，才能计算阶段七三指标')
    means = {history: statistics.mean(r['change_pct'] for r in rows if r['history'] == history)
             for history in (131072, 8192)}
    # The compact JSON formatter expects object keys to be strings.
    return {'mean_change_pct_by_history': {str(history): value for history, value in means.items()},
            'weighted_change_pct': .7 * means[131072] + .3 * means[8192],
            'scope': '同轮Native为基线，各history内batch等权后按128K/8K七三合成；'
                     '不替代P95、最慢rank、token及DSpark检查，不作跨轮单因素归因。'}


def main():
    collector = load_module('seven_forward', ROOT.parent / 'csa_source_split_ab_20260927/ordered/collect_model.py')
    formatter = load_module('forward_formatter', ROOT.parent / 'csa_ascendc_topk_hc_ep16_20260928/collect_model.py')
    collector.ROOT = ROOT / 'model'
    collector.CASES = CASES
    collector.OPERATOR_REVISION = REVISION + '; Key L1 + UB/QR; mode2/atomic0/det0; EPLB off'
    collector.main()
    path = collector.ROOT / 'forward.json'
    data = json.loads(path.read_text())
    if data['all_cases_pass']:
        data['performance_observations'] = weighted_changes(data['cases'])
        with (collector.ROOT / 'RESULTS.md').open('a') as output:
            output.write('\n各history内部等权后，128K/8K七三forward变化为'
                         f"{data['performance_observations']['weighted_change_pct']:+.3f}%。\n")
    path.write_text(formatter.format_json(data) + '\n')
    print(data.get('performance_observations', '尚无完整且通过的七档，不计算综合收益'))


if __name__ == '__main__':
    main()
