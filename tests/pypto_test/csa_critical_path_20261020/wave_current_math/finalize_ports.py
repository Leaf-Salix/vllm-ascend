# SPDX-License-Identifier: Apache-2.0
"""整理候选中文概览并核对关键调用接口；不把源码接线当编译或设备通过。"""

import ast
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
TARGET=HERE/'source_candidate'
REF=HERE.parent/'source_after_score_split'


def append_overview(name,text):
    path=TARGET/name;s=path.read_text();tree=ast.parse(s)
    doc=tree.body[0];assert isinstance(doc,ast.Expr) and isinstance(doc.value,ast.Constant)
    start=s.index('"""');end=s.index('"""',start+3)
    s=s[:end]+text+s[end:];path.write_text(s)


def main():
    p=TARGET/'vector_schedule.py';s=p.read_text()
    begin=s.index('    本轮交错候选');end=s.index('完整动态CSA的双AIV编排',begin)
    s=s[:begin]+'''    本轮候选在已有SPLIT_SCORE_ROWS/G6资源域和device多wave条件下交错。
    实际wave的plan/Top-K采用当前前置几何及物理行表示，tick真正执行MGATHER
    与区间local mi/li，Score step真正执行连续scale；并保留首wave原KV处理。
    两角色条件一致，未进入交错时沿当前完整fallback；仍须完整联合编译与
    真机验证，不能把源码接线或开关存在当成精度、资源和性能已通过。
'''+s[end:];p.write_text(s)
    p=TARGET/'interleave_vector_wave.py';s=p.read_text()
    s=s.replace('跨分支只显式\n        # yield四个Attention Tile','跨分支只显式\n        # yield五个Attention Tile（含统计环）')
    s=s.replace('不能拿这些未使用Scalar另造一套与AIC不同的leaf长度。',
                '本版本另携带统一control返回的valid，用于连续scale尾页，不另算长度。')
    p.write_text(s)
    p=TARGET/'attention_tick_stages.py';s=p.read_text()
    s=s.replace('本模块直接抽取已冻结的两L1候选，没有实现Score交错、wave控制或新调度。\n    typed-Tile参数/返回依赖尚未验证的JIT能力；源码AST相同不等于可lowering、',
                'AIC沿已验证两L1槽候选；AIV一拍从当前正式函数抽出，包含物理plan/\n    MGATHER及本地统计环，已接入本轮wave候选。已有typed-Tile能力不等于本组合的')
    s=s.replace('返回T/core/block_count/lane_head/lane_kv/swap/m/l/left/right。',
                '返回T/core/block_count/lane_head/lane_kv/swap/m/l/left/right；统计环由wave创建。')
    p.write_text(s)
    p=TARGET/'score_vector_steps_g6.py';s=p.read_text().replace('源码准备，尚未接入调度','当前数学已接入wave候选，待联合验证');p.write_text(s)
    append_overview('cube_schedule.py','''
本轮跨请求流水候选
    已采用SPLIT_SCORE_ROWS且device确有多个wave时，当前Score与前波Attention
    在同一MIX内部交错。Score step使用当前N64双Acc早FIX；Coeff/Query跨步
    驻Mat、每步恢复Left。wave ROOT/Top-K/plan通知与原KV首波处理保留。
    非交错计算仍为当前完整函数；本候选尚未经过联合编译或设备验收。
''')
    append_overview('indexer_score_stages.py','''
本轮跨请求组合
    追加indexer_score_cube_step_g6，数学正文直接来自本文件当前split helper。
    唯一区别是Query/Coeff先由外层驻Mat保存，每step恢复Left，以容纳前波
    Attention的Left使用；叶范围、READY/ACK和输出布局均沿原合同。
''')
    append_overview('interleave_vector_wave.py','''
当前新增的状态与真实路径
    第五份Attention Tile为[6,32] FP32的本lane统计环，softmax写与PV消费
    之间随外层循环携带。physical_mode每wave按相同plan容量谓词计算一次。
    Score另携带next_score_leaf给出的实际valid；连续scale使用同代布局证明，
    每步只重新读取两个不可变Scalar，不重新扫描页表证明，也不把steps当有效长度。
''')
    checked_names={
        'vector_schedule','interleave_cube_wave_g6','interleave_vector_wave_g6',
        'indexer_score_cube_step_g6','score_vector_step_g6','attention_vector_tick',
        'indexer_pair_publish_range','attention_slot_plan_range',
    }
    trees={p.name:ast.parse(p.read_text()) for p in TARGET.glob('*.py')}
    definitions={n.name:n for t in trees.values() for n in t.body if isinstance(n,ast.FunctionDef) and n.name in checked_names}
    assert set(definitions)==checked_names
    calls=[]
    for file,t in trees.items():
        for n in ast.walk(t):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in definitions:
                d=definitions[n.func.id];count=len(n.args)+len(n.keywords);minimum=len(d.args.args)-len(d.args.defaults)
                assert minimum<=count<=len(d.args.args),(file,n.func.id,count,minimum,len(d.args.args))
                calls.append(dict(file=file,function=n.func.id,arguments=count,required=minimum,maximum=len(d.args.args)))
    assert 'SPLIT_SCORE_ROWS and QUERY_GROUP == 6' in (TARGET/'cube_schedule.py').read_text()
    assert 'SPLIT_SCORE_ROWS and QUERY_GROUP == 6' in (TARGET/'vector_schedule.py').read_text()
    spec=importlib.util.spec_from_file_location('wave_current_policy',TARGET/'policy.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    for b in (16,28):
        controls=module.select_policy(tokens=b*6,requests=b,max_query_tokens=6,max_seq_len=131078).constexprs()
        assert controls['SPLIT_SCORE_ROWS']
        (HERE/f'controls_b{b}.json').write_text(json.dumps(controls,indent=2)+'\n')
    report=dict(call_contracts=calls,source_files=len(trees),actual_score_panel=64,
        physical_plan_and_mgather_in_active_path=True,local_stats_in_loop_state=True,
        scale_coalescing_in_active_step=True,early_fix_in_active_step=True,
        formal_modified=False,compiled=False,device_executed=False,
        note='接线与源级结构证据，完整资源/同步/精度/性能均尚待验证')
    (HERE/'SOURCE_CONTRACT_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print({'files':len(trees),'checked_calls':len(calls),'compiled':False})


if __name__=='__main__':main()
