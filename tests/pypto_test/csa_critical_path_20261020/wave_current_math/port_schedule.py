# SPDX-License-Identifier: Apache-2.0
"""迁入当前前后段，并将wave的根发布/plan数学更新为当前已采用路径。

wave外层已经等待本波真实ROOT/TOPK owner，所以range入口不能再次等待
整阶段SCORE_VECTOR/TOPK完成；否则将与后续wave形成环。这里只替换获取
位置，数学、页保护、early geometry和physical plan均来自当前正式函数。
"""

import ast
import json
from pathlib import Path
import sys
import textwrap

from source_utils import block,body,function,header,remove_statements,replace_function,rewrite_calls,signature

HERE=Path(__file__).resolve().parent
REF=HERE.parent/'source_after_score_split'
TARGET=HERE/'source_candidate'
OLD=HERE.parents[3]/'tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_interleave_current_20261019/source_candidate_v2'


def main():
    # Cube冲突按明确段边界解决：当前前段/输出段和完整fallback逐字保留。
    current=(REF/'cube_schedule.py').read_text();old=(OLD/'cube_schedule.py').read_text()
    first=current.index('    indexer_transfer = indexer_score_cube_stage(')
    last=current.index('    # 每个query归q%24组',first)
    old_first=old.index('    wave_groups = ');old_last=old.index('    # 每个query归q%24组',old_first)
    region=old[old_first:old_last];fallback=region.index('    else:\n        indexer_transfer =')
    region=region[:fallback]+'    else:\n'+textwrap.indent(current[first:last],'    ')
    merged=current[:first]+region+current[last:]
    imports='''from .wave_boundaries import next_wave_boundary
from .runtime_abi import WAVE_PHASE_BASE, WAVE_PHASE_STRIDE, WAVE_PLAN
from .wave_dependencies import wait_wave_plan_for_attention
from .interleave_cube_wave import interleave_cube_wave_g6
'''
    merged=merged.replace('import pypto.language as pl\n','import pypto.language as pl\n\n'+imports,1)
    (TARGET/'cube_schedule.py').write_text(merged)
    # 另一冲突文件先保留当前完整Score；新的step将由port_score单独抽取。
    (TARGET/'indexer_score_stages.py').write_bytes((REF/'indexer_score_stages.py').read_bytes())

    # 根range：只移除已经由外层wave ROOT acquire覆盖的整阶段获取。
    current=(REF/'indexer_publish_stages.py').read_text()
    args=signature(current,'indexer_pair_publish_stage')
    args=[x for x in args if x[0] not in ('offsets','signals','generation','QUERY_GROUP')]
    offset=next(i for i,x in enumerate(args) if x[0]=='worker')
    args[offset:offset]=[('query_begin','pl.Scalar[pl.INDEX]',None),('query_end','pl.Scalar[pl.INDEX]',None)]
    code=body(current,'indexer_pair_publish_stage')
    def root_wait(n):
        return isinstance(n,ast.Assign) and (
            any(isinstance(t,ast.Name) and t.id in ('local_query','group') for t in n.targets)
            or isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='wait_indexer_query_roots')
    code,removed=remove_statements(code,root_wait);assert removed==3
    code=code.replace('for query in pl.range(worker, tokens, WORKERS):',
        'first_query = query_begin + (worker - query_begin % WORKERS + WORKERS) % WORKERS\nfor query in pl.range(first_query, query_end, WORKERS):')
    code=code.replace('return topk_scores, topk_indices, signals','return topk_scores, topk_indices')
    code=code.replace('复用本query已算出的唯一request及begin，不额外扫描Q/L，也不把',
                      '外层已按本波真实ROOT owner完成获取；此处不等整阶段完成。')
    code=code.replace('request_begin替换成固定S的除法。根数与下方归并完全相同。',
                      '保留当前前置几何、根数、归并树与无效query处理。')
    new=header('indexer_pair_publish_range',args)+textwrap.indent(code,'    ')
    path=TARGET/'indexer_publish_stages.py';path.write_text(replace_function(path.read_text(),'indexer_pair_publish_range',new))

    # Plan range：复用当前完整物理plan数学，只改变range和已经满足的获取点。
    current=(REF/'attention_plan_stages.py').read_text();args=signature(current,'attention_slot_plan_stage')
    args=[x for x in args if x[0] not in ('signals','generation')]
    offset=next(i for i,x in enumerate(args) if x[0]=='worker_id')
    args[offset:offset]=[('query_begin','pl.Scalar[pl.INDEX]',None),('query_end','pl.Scalar[pl.INDEX]',None)]
    code=body(current,'attention_slot_plan_stage')
    code,removed=remove_statements(code,lambda n:isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='wait_row_owners')
    assert removed==1
    code=code.replace('for first in pl.range(worker_id * ROWS, tokens, WORKERS * ROWS):',
        'begin_tile = query_begin // ROWS\nfirst_tile = begin_tile + (worker_id - begin_tile % WORKERS + WORKERS) % WORKERS\nfor first in pl.range(first_tile * ROWS, query_end, WORKERS * ROWS):')
    code=code.replace('return indices, bias, valid_blocks, signals','return indices, bias, valid_blocks')
    code=code.replace('# 最多八个owner；尾部仅rows行。wait内仍保留业务缓存的获取语义。',
        '# 外层wave TOPK获取覆盖这段真实query范围，保留同样的业务缓存维护。')
    new=header('attention_slot_plan_range',args)+textwrap.indent(code,'    ')
    path=TARGET/'attention_plan_stages.py';path.write_text(replace_function(path.read_text(),'attention_slot_plan_range',new))

    # 两角色使用同一个SPLIT开关，避免小T或手动关闭时一侧交错、另一侧原路径。
    sys.path.insert(0,str(HERE.parent));import prepare_qb_stream_reuse as wiring
    old_flag=wiring.FLAG
    try:
        wiring.FLAG='SPLIT_SCORE_ROWS'
        path=TARGET/'vector_schedule.py';path.write_text(wiring.wire(path.read_text(),('vector_schedule',),()))
        path=TARGET/'entry.py';path.write_text(wiring.wire(path.read_text(),(),('vector_schedule',)))
    finally:wiring.FLAG=old_flag
    condition='if QUERY_GROUP == 6 and QK_PANEL == 64 and SCORE_TILE == 1024 and PREFETCH == 1 and TO_L0 and BALANCED and ROOTS == 1:'
    for name in ('cube_schedule.py','vector_schedule.py'):
        path=TARGET/name;s=path.read_text();assert s.count(condition)==1
        s=s.replace(condition,condition.replace('if ','if SPLIT_SCORE_ROWS and ',1));path.write_text(s)
    path=TARGET/'vector_schedule.py';s=path.read_text()
    s,nroots=rewrite_calls(s,'indexer_pair_publish_range',lambda a:
        a[:4]+['attention_indices']+a[4:]+['EARLY_ATTENTION_GEOMETRY'])
    s,nplans=rewrite_calls(s,'attention_slot_plan_range',lambda a:
        a+['EARLY_ATTENTION_GEOMETRY','PLAN_PHYSICAL_ROWS'])
    path.write_text(s)
    for p in TARGET.glob('*.py'):ast.parse(p.read_text())
    report=dict(cube_current_prefix_and_output_retained=True,cube_current_fallback_retained=True,
        root_range_current_math=True,plan_range_current_math=True,root_acquire_replaced_by_wave=True,
        topk_acquire_replaced_by_wave=True,root_call_sites=nroots,plan_call_sites=nplans,
        split_flag_both_roles=True,formal_modified=False,compiled=False)
    (HERE/'PORT_SCHEDULE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(report)


if __name__=='__main__':main()
