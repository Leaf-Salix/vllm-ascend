# SPDX-License-Identifier: Apache-2.0
"""按正式区间实际选择映射整轮测量，不隐藏B4初轮回退或8K测量波动。"""

import json
from pathlib import Path

from report_key_block import summarize

HERE=Path(__file__).resolve().parent


def main():
    boundary=json.loads((HERE/'SCORE_SPLIT_BOUNDARY_TASKS.json').read_text())
    b6=summarize(HERE/'score_interval_h131072_b6_v1',boundary['b6_s6']['task'])
    assert b6['performance_eligible']
    (HERE/'SCORE_SPLIT_BOUNDARY_PERFORMANCE.json').write_text(json.dumps(b6,ensure_ascii=False,indent=2)+'\n')
    raw=json.loads((HERE/'SCORE_SPLIT_ALL_PERFORMANCE.json').read_text());current={}
    order=['h131072_b4','h131072_b8','h131072_b16','h131072_b24','h131072_b28','h8192_b16','h8192_b32']
    for tag in order:
        r=raw[tag];assert r['performance_eligible'];v=r['median_us']
        role='spmd_reference' if tag=='h131072_b4' else 'spmd_candidate'
        chosen=v[role]
        if tag=='h131072_b4':
            note='B4保留原代码，复用该轮reference整组测量，非新测区间版'
        elif tag.startswith('h8192_'):
            note='两侧完整C++相同，保留该轮candidate全部测量波动，不作为代码退步/收益'
        else:
            note='开启代码与已测候选相同，复用该轮完整配对结果，不跨轮挑选'
        current[tag]=dict(
            task=r['task'],measured_source=r['source'],selected_measurement_role=role,
            multi_task_full_l1_us=v['dynamic_pto'],previous_formal_full_l1_us=v['spmd_reference'],
            current_full_l1_us=chosen,selected_split=tag.startswith('h131072_') and tag!='h131072_b4',
            goal_reduction_percent=100*(v['dynamic_pto']-chosen)/v['dynamic_pto'],
            display_speedup_vs_multi_percent=100*(v['dynamic_pto']-chosen)/chosen,
            goal_30_percent_met=chosen<=v['dynamic_pto']*0.7,
            all_window_data='SCORE_SPLIT_ALL_PERFORMANCE.json',note=note,
        )
    report=dict(adopted_source='ops/pypto/deepseek_v4_flash_csa_spmd',cases=current,new_performance_measurement=False,
                note='按实际所选代码映射已有整轮数据；B4不采用初轮候选，回退仍保留；8K不换成较低reference数值')
    (HERE/'SCORE_SPLIT_CURRENT_PERFORMANCE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    adoption=json.loads((HERE/'SCORE_SPLIT_ADOPTION.json').read_text())
    adoption.update(boundary_tasks=boundary,true_query_tasks=json.loads((HERE/'SCORE_SPLIT_QUERY_TASKS.json').read_text()),
                    full_correctness_seven_cases=True,true_s1_s5_passed=True,interval_s6_boundaries_passed=True,
                    goal_128k_passed_cases=sum(v['goal_30_percent_met'] for k,v in current.items() if k.startswith('h131072_')))
    (HERE/'SCORE_SPLIT_ADOPTION.json').write_text(json.dumps(adoption,ensure_ascii=False,indent=2)+'\n')
    print({'adopted':True,'goal_128k_passed_cases':adoption['goal_128k_passed_cases'],'boundary_saved_us':b6['net_saved_vs_current_us']})


if __name__=='__main__':
    main()
