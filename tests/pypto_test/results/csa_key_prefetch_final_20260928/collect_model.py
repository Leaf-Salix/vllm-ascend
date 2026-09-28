"""只验收新增长B4/B8预取及短B16控制，不重复完整七档。"""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = '8e176285'
CASES = ((131072, 4), (131072, 8), (8192, 16))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weighted_changes(rows):
    if len(rows) != len(CASES) or {(r['history'], r['batch']) for r in rows} != set(CASES):
        raise ValueError('三个定向档位必须齐全且不重复')
    means = {str(history): statistics.mean(r['change_pct'] for r in rows if r['history'] == history)
             for history in (131072, 8192)}
    return {'mean_change_pct_by_history': means,
            'weighted_change_pct': .7 * means['131072'] + .3 * means['8192'],
            'scope': '本轮同Native对照；长B4/B8内部等权后，与短B16按七三合成。'
                     '是新增策略的定向验收，不是完整七档指标，也不能作跨轮单因素归因。'}


def main():
    collector = load_module('targeted_forward', ROOT.parent / 'csa_source_split_ab_20260927/ordered/collect_model.py')
    formatter = load_module('forward_formatter', ROOT.parent / 'csa_ascendc_topk_hc_ep16_20260928/collect_model.py')
    collector.ROOT = ROOT / 'model'
    collector.CASES = CASES
    collector.OPERATOR_REVISION = REVISION + '; combined B4/B8 Key prefetch; mode2/atomic0/det0; EPLB off'
    collector.main()
    path = collector.ROOT / 'forward.json'
    data = json.loads(path.read_text())
    if data['all_cases_pass']:
        data['performance_observations'] = weighted_changes(data['cases'])
        with (collector.ROOT / 'RESULTS.md').open('a') as output:
            output.write('\n定向长B4/B8内部等权、再与短B16按七三合成：'
                         f"{data['performance_observations']['weighted_change_pct']:+.3f}%，不是完整七档指标。\n")
    path.write_text(formatter.format_json(data) + '\n')
    print(data.get('performance_observations', '定向三档尚未齐全且通过，不计算综合收益'))


if __name__ == '__main__':
    main()
