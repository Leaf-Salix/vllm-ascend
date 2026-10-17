"""只读提取66882634的Attention所有权与既有tick/历史证据。

无PyPTO导入、编译或设备执行；只比较既有源文本、保存精确行号、计算固定
矩形的字节数，不能把字节数或tick区间当可节省的端到端时间。
"""
from pathlib import Path
import json
import subprocess

R=Path('/data/pyptouser/qinchuanyu/pto-eager')
C=R/'vllm-ascend-dsv4-pto-0251rc1'
P=C/'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
N=C/'tests/pypto_test/csa_multishape_spmd_20261016'
PAR=N/'dynamic_full_pipeline_20261017/parallel_20261018'
OUT=Path(__file__).resolve().parent
specs=[
(P/'attention_stream_stages.py',121,203,'AIC完整64头、query owner及QK/PV三槽L1复用'),
(P/'attention_stream_stages.py',233,323,'双lane候选分工及压缩KV逐行gather'),
(P/'attention_stream_stages.py',323,407,'SCORE/PROB/PV事件、tick偏移及槽生命周期'),
(P/'vector_schedule.py',884,926,'相同MIX物理组/lane调用'),
(P/'cube_schedule.py',377,401,'AIC调用和packed消费者'),
(P/'attention_plan_stages.py',279,304,'原TopK顺序与mask写出'),
(PAR/'attention_tick_clock_evidence/ANALYSIS.md',1,90,'既有d253 tick诊断，不是668的新测量'),
(N/'QK_DIRECT_L1_EXPERIMENT.md',1,90,'历史直接L1失败及原有ACK变化'),
]
records=[]
for path,lo,hi,why in specs:
 lines=path.read_text().splitlines()
 records.append({'path':str(path),'bytes':path.stat().st_size,'line_start':lo,'line_end':min(hi,len(lines)),'reason':why,'text':'\n'.join(f'{j+1}: {lines[j]}' for j in range(lo-1,min(hi,len(lines))))})
source_checks=[]
for name in ('attention_stream_stages.py','attention_plan_stages.py','cube_schedule.py','vector_schedule.py'):
 rel=str((P/name).relative_to(C))
 historical=subprocess.check_output(['git','-C',str(C),'show','66882634:'+rel])
 source_checks.append({'file':name,'current_matches_66882634':historical==(P/name).read_bytes()})
assert all(x['current_matches_66882634'] for x in source_checks)
tick_src=PAR/'attention_tick_clock_evidence/source_snapshot/attention_stream_stages.py'
assert tick_src.exists()
obj={
 'baseline_commit':'66882634','source_checks':source_checks,
 'attention_stream_equals_old_d253_tick_snapshot':tick_src.read_bytes()==(P/'attention_stream_stages.py').read_bytes(),
 'scope':'只读源/历史；无编译、无NPU、无新性能数据、无地址重叠测量',
 'ownership':{'query_group':'t % 24','query_ordinal':'floor(t/24)','work':'5*floor(t/24)+block','block':'0=SWA，1..4=压缩','slot':'3*(t%24)+work%3','aic_heads':'0..63全体','aiv_lane0_candidates':'block内0..63','aiv_lane1_candidates':'block内64..127','aiv_lane0_heads':'0..31','aiv_lane1_heads':'32..63','physical_aiv':'24+2*group+lane'},
 'bytes_model':{'one_KV_row_BF16_512':512*2,'one_lane_half_block':64*512*2,'one_block':128*512*2,'one_query_compressed_four_blocks':512*512*2,'one_query_all_five_blocks':640*512*2,'one_AIC_L1_KV_three_slots':3*128*512*2,'one_AIC_Q':64*512*2,'one_AIC_probability':64*128*2,'these_AIC_Mat_total':(3*128+64)*512*2+64*128*2,'24_groups_KV_GM_ring':24*3*128*512*2},
 'no_head_or_indexer_partition_redundant_gather':True,
 'cross_query_physical_row_overlap_measured':False,
 'decision':'否决以当前同query重复head/score分片采集为前提的一组AIV多AIC复用候选；该重复不存在。跨query复用需新证据/协议，不在本轮实现。',
 'evidence':records,
}
(OUT/'evidence.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in obj.items() if k!='evidence'},ensure_ascii=False,indent=2))
