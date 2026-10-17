#!/usr/bin/env python3
"""只读审查当前投影源码、历史现成 PTO 与重复量化加载；不编译、不导入设备库。"""

import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
R = Path('/data/pyptouser/qinchuanyu/pto-eager')
C = R / 'vllm-ascend-dsv4-pto-0251rc1'
P = C / 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
N = C / 'tests/pypto_test/csa_multishape_spmd_20261016'
PAR = N / 'dynamic_full_pipeline_20261017/parallel_20261018'
FUSED = PAR / 'compact_priority_h131072_b16_v1/build_output/decode_csa_spmd_ad5etrbf/ptoas/dynamic_csa_full.pto'
ORIGINAL = PAR / 'current_clock_h131072_b16_v1/build_output/_decode_csa_tp1_layer_0leidxnz/ptoas'
INPUTS = set()


def read(p):
    INPUTS.add(p)
    return p.read_text()


files = ['qkv_stages.py', 'kv_stages.py', 'compressor_projection_stages.py',
         'output_projection_a_stages.py', 'output_projection_b_stages.py']
same = {name: read(P/name) == read(PAR/'adopted_source_d56c9eb6'/name) for name in files}
assert all(same.values())
names = ['qr_proj_matmul', 'qr_proj_matmul_0', 'kv_proj_matmul', 'kv_proj_matmul_0',
         'kv_score_proj', 'kv_score_proj_0', 'proj_a_mm_0', 'proj_a_mm_2',
         '_proj_b_mm_nz_kernel__2', '_proj_b_mm_nz_kernel__3']
tiles = {}
for name in names:
    text = read(ORIGINAL/(name+'.pto'))
    shapes = sorted(set(re.findall(r'!pto.tile_buf<loc=(left|right|acc), dtype=(\w+), rows=(\d+), cols=(\d+)', text)))
    tiles[name] = [dict(space=s, dtype=d, rows=int(m), cols=int(n)) for s,d,m,n in shapes]

fused_lines = read(FUSED).splitlines()
fused_quant = [(i+1, line) for i,line in enumerate(fused_lines)
               if re.search(r'output_projection_a_stages.py":(?:21[2-9]|22[0-9]|23[0-5]):',line)]
assert any('pto.tload ' in line and 'oc_amax' in line for _,line in fused_quant)
assert any('pto.tload ' in line and 'oc_q' in line for _,line in fused_quant)
for label, rows in [('fused_quant',fused_quant),
                    ('original_quant',list(enumerate(read(ORIGINAL/'quant.pto').splitlines(),1)))]:
    # 保留原行号/内容；完整权重、二进制和大日志不纳入这个小文本审计。
    (OUT/(label+'.pto.txt')).write_text('\n'.join(f'{no}: {line}' for no,line in rows)+'\n')

rows = []
for history,batch in [(131072,4),(131072,8),(131072,16),(131072,24),(131072,28),(8192,16),(8192,32)]:
    t = batch*6
    saved = t*8*1024*4
    rows.append(dict(history=history,batch=batch,query_tokens=6,tokens=t,
                     removed_logical_loads=8*((t+7)//8),saved_logical_bytes=saved,
                     saved_logical_mib=saved/(1024*1024)))

result = dict(scope='d253 投影只读审查；无源码候选、无编译、无设备任务',
              source_files_identical_to_d56=same,
              original_task_actual_l0_tiles=tiles,
              selected_candidate=dict(helper='o_projection_groups_quant_stage',
                  source=str(P/'output_projection_a_stages.py'),load_lines=[212,230],
                  minimal_change='删除第二次pl.load，row_expand_mul直接读第一次oc_amax的原值',
                  math_changed=False,shape_or_owner_changed=False,phase_or_workspace_changed=False,
                  original_task_has_same_duplicate=True,
                  fused_ir_load_lines=[13448,13453,13459,13465],
                  fused_stage2_input_addresses=[65536,98304,0,32768],
                  original_abs_reuses_input=True,required_nonoverlap='保留原值时abs结果必须独立于原值槽',
                  existing_five_large_slots_bytes=5*32768,
                  proposed_feasible_liveness_bytes=4*32768,
                  proposed_liveness_explanation='两份流水输入64KiB + 一份abs32KiB + 一份row_max临时32KiB；其后乘法/转换复用已释放槽。不是新后端结果。',
                  logical_traffic_reduction=rows,
                  limitations=['未证明新表达的实际分配/流水深度；不据理论活跃量预记CPU通过。',
                               '逻辑GM load减少不是等量HBM节省，也不能换算完整CSA收益。',
                               '原分task也有相同冗余；不是新融合独有回退的证据。']),
              history_exclusions=['seven_stage_opportunities_v1：OA双输出tile已做资源探针，未完成真实跨组合同。',
                                  'OA K512/N256/共享8组循环已有完整测试；不重提。',
                                  'HCstream/紧凑交接及compact Right、主QB Left已有结果；不重开。'])
result['inputs']=[dict(path=str(p),bytes=p.stat().st_size) for p in sorted(INPUTS)]
(OUT/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(output=str(OUT),same_source=same,candidate='OA量化复用第一次FP32加载'),ensure_ascii=False))
