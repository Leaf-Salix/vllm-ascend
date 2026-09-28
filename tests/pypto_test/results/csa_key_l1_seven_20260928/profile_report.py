"""七档真实模型profile与七档PTO单CSA泳道汇集，并分别标清采集口径。"""

import json
import os
from pathlib import Path

from collect_model import CASES, REVISION, ROOT, load_module, weighted_changes


def main():
    data = json.loads((ROOT / 'model/model_gap_rank0.json').read_text())
    changes = [{'history': row['history'], 'batch': row['batch'],
                'change_pct': (row['pto']['csa']['mean_us'] / row['native']['csa']['mean_us'] - 1) * 100}
               for row in data['cases']]
    result = {'operator': REVISION, 'cases': changes, **weighted_changes(changes),
              'scope': 'rank0独立三步profile，完整CSA含首次metadata；'
                       'history内部等权后七三合成，不用profile替代正式十步forward。'}
    reporter = load_module('csa_profile_report', ROOT.parent / 'csa_ascendc_topk_hc_ep16_20260928/profile_report.py')
    reporter.main(root=ROOT, revision=REVISION, matrix_label='统一七档')
    (ROOT / 'model/weighted_csa.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (ROOT / 'model/MODEL_GAP.md').open('a') as output:
        output.write(f"\n各history内部等权后，128K/8K七三CSA变化为{result['weighted_change_pct']:+.3f}%。\n"
                     'P95及最慢rank、正式forward单列。[加权项](weighted_csa.json)。\n')
    layer = json.loads((ROOT / 'layer.json').read_text())
    if not layer['all_cases_pass'] or {(r['history'], r['batch']) for r in layer['cases']} != set(CASES):
        raise ValueError('单卡七档证据不齐全')
    download = ROOT / 'download'
    manifest = json.loads((download / 'manifest.json').read_text())
    if len(manifest) != 14:
        raise ValueError('七档两侧rank0模型profile必须为14份')
    for row in layer['cases']:
        source = Path(row['worker_windows'][0]['path'])
        label = f"{row['history']//1024}K_B{row['batch']}"
        destination = download / f'{len(manifest)+1:02}_{label}_PTO_SingleCSA_SyntheticHistory.json'
        if destination.exists():
            if not os.path.samefile(destination, source):
                raise ValueError(f'{destination}: 已有来源不同')
        else:
            os.link(source, destination)
        manifest.append({'file': destination.name, 'source': str(source), 'operator': REVISION,
                         'scope': '单卡layer4真实权重/合成历史，独立单根DFX；固定取窗口0，不挑最快样本',
                         'all_windows': [w['path'] for w in row['worker_windows']],
                         'tasks': row['sources']['tasks']})
    (download / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (download / 'README.md').write_text(
        f'# {REVISION}七档profile及泳道\n\n'
        '01–14为真实EP16 rank0独立三步PyTorch JSON；15–21为对应七档单卡layer4合成历史PTO泳道。\n'
        '二者采集独立，单卡DFX不等于真实模型某一步某层。DFX固定保留窗口0供下载，'
        '全部窗口见manifest与单卡报告，未按耗时筛选。\n\n' +
        '\n'.join(f"- [{item['file']}]({item['file']})" for item in manifest) +
        '\n\n[来源及全部窗口](manifest.json)、[单卡CSA](../LAYER.md)、'
        '[模型forward](../model/RESULTS.md)、[模型CSA及分项](../model/MODEL_GAP.md)、'
        '[相邻CSA差异](../model/ADJACENT_CSA.md)。\n'
    )
    print({'weighted_csa_change_pct': result['weighted_change_pct'], 'download_files': len(manifest)})


if __name__ == '__main__':
    main()
