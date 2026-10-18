# SPDX-License-Identifier: Apache-2.0
"""汇总已结束的前置几何首版，保留五窗与静态历史来源，不重新测试。"""

import json
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    history_root = here.parents[2]
    historical = {(r['history'], r['batch']): r for r in json.loads(
        (history_root / 'selected_shape_policies.json').read_text())['rows']}
    rows = []
    jobs = json.loads((here / 'SUBMISSIONS_V1.json').read_text())
    for job in sorted(jobs, key=lambda j: (-j['history'], j['batch'])):
        source = Path(job['output']) / 'result.json'
        result = json.loads(source.read_text())
        assert result['passed'] and result['full_csa_validated'] and result['graph_replay_validated']
        matrix = result['source_matrix']
        assert matrix['passed'] and matrix['same_process'] and matrix['same_root_addresses']
        stats = matrix['statistics_us']
        assert all(len(v['samples_us']) == 5 for v in stats.values())
        multi, old, new = (stats[k]['median_us'] for k in ('dynamic_pto', 'spmd_reference', 'spmd_candidate'))
        static = historical.get((job['history'], job['batch']))
        rows.append(dict(history=job['history'], batch=job['batch'], query_tokens=6,
                         task_id=job['task_id'], source=str(source), statistics_us=stats,
                         multi_task_us=multi, previous_dynamic_us=old, candidate_us=new,
                         vs_previous_reduction_pct=(old-new)/old*100,
                         vs_multi_reduction_pct=(multi-new)/multi*100,
                         target_us=multi*0.7, target_met=new <= multi*0.7,
                         paired_faster_windows=sum(a < b for a,b in zip(
                             stats['spmd_candidate']['samples_us'],stats['spmd_reference']['samples_us'])),
                         historical_static=None if static is None else dict(
                             l1_us=static['l1_us'], same_round=False,
                             source=str(history_root / static['source']))))
    assert len(rows) == 7
    report = dict(scope='正式W8A8第4层/S6/合成历史，三方同进程五窗完整L1 incore首尾',
                  target_met_count=sum(r['target_met'] for r in rows), rows=rows,
                  static_note='只列已有已验证静态记录，与本轮不同轮；B28无有效静态数据')
    (here/'SEVEN_RESULTS_V1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# 前置几何首版：七档完整L1', '',
           '全部精确输出、Top-K/score、cache/state、A→B→A、图、保护区及代际检查通过。',
           '前三列为每档同进程三方五窗中位，单位μs。静态列仅为旧轮次参考，不计算混轮收益。', '',
           '| 历史/B | 多task | 旧动态 | 前置几何 | 较旧动态降低 | 较多task降低 | 历史静态 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        old_static = f"{r['historical_static']['l1_us']:.2f}" if r['historical_static'] else '无有效数据'
        lines.append(f"| {r['history']//1024}K/B{r['batch']} | {r['multi_task_us']:.2f} | "
                     f"{r['previous_dynamic_us']:.2f} | {r['candidate_us']:.2f} | "
                     f"{r['vs_previous_reduction_pct']:.2f}% | {r['vs_multi_reduction_pct']:.2f}% | {old_static} |")
    lines += ['', '**相对原多task降低30%的目标仍0/7。** 六档改善、128K/B8回退1.03%，不整体采用，也没有一档达标。',
              'S1/S5、六次metadata与四次padding同图回放另行通过；不代表全部mixed-query、',
              '请求生命周期、连续decode轨迹、Native精度或整网token验收。原始五窗及来源均在JSON。']
    (here/'SEVEN_RESULTS_V1.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'cases':len(rows),'target_met':report['target_met_count'],
                      'reductions_pct':[round(r['vs_previous_reduction_pct'],3) for r in rows]}))


if __name__ == '__main__':
    main()
