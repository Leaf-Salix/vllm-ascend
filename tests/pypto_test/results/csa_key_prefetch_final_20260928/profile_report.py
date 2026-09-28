"""定向三档模型区间及六份rank0 JSON，不混入旧七档最优值。"""

import json

from collect_model import REVISION, ROOT, load_module, weighted_changes


def main():
    reporter = load_module('targeted_profile', ROOT.parent / 'csa_ascendc_topk_hc_ep16_20260928/profile_report.py')
    reporter.main(root=ROOT, revision=REVISION, matrix_label='定向长B4/B8及短B16')
    data = json.loads((ROOT / 'model/model_gap_rank0.json').read_text())
    changes = [{'history': row['history'], 'batch': row['batch'],
                'change_pct': (row['pto']['csa']['mean_us'] / row['native']['csa']['mean_us'] - 1) * 100}
               for row in data['cases']]
    result = {'operator': REVISION, 'cases': changes, **weighted_changes(changes),
              'measurement': 'rank0独立三步profile，完整CSA含首次metadata；不替代正式十步forward。'}
    (ROOT / 'model/weighted_csa.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (ROOT / 'model/MODEL_GAP.md').open('a') as output:
        output.write(f"\n定向三档七三CSA变化{result['weighted_change_pct']:+.3f}%，不是完整七档指标。\n")
    print(result)


if __name__ == '__main__':
    main()
