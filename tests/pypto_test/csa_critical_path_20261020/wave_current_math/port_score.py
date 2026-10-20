# SPDX-License-Identifier: Apache-2.0
"""从当前已采用的N64 Score抽取完整step，保留早FIX、合并写回和scale路径。

Cube的Query/Coeff驻Mat跨Attention保留，每个Score step重建两半Left及独立
Acc；不把Left私自跨两个阶段复用。Vector的roots/segments仍跨step保存，
新增实际leaf valid Scalar控制连续scale尾页，不能用steps容量冒充有效长度。
"""

import ast
import json
from pathlib import Path
import re
import sys
import textwrap

from source_utils import block,body,function,header,remove_statements,replace_function,rewrite_calls,signature

HERE=Path(__file__).resolve().parent
REF=HERE.parent/'source_after_score_split'
TARGET=HERE/'source_candidate'
OLD=HERE.parents[3]/'tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_interleave_current_20261019/source_candidate_v2'


def rename(source,mapping):
    for before,after in mapping.items():source=re.sub(r'\b'+re.escape(before)+r'\b',after,source)
    return source


def main():
    current=(REF/'indexer_score_stages.py').read_text()
    fn=function(current,'_indexer_score_cube_split_store')
    step,=[n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='buf_score_step']
    first=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='split_query0')
    lines=current.splitlines(keepends=True)
    setup=textwrap.dedent(''.join(lines[first.lineno-1:step.lineno-1]))
    code=setup+block(current,step.body)+'\nreturn buf_score_transfer\n'
    mapping=dict(query_group_size='G6_QUERY_GROUP',qk_panel_cols='G6_QK_PANEL',score_tile='G6_SCORE_TILE',
                 key_prefetch_panels='G6_PREFETCH',key_prefetch_to_l0='G6_TO_L0',balance_leaves='G6_BALANCED')
    code=rename(code,mapping)
    old=(OLD/'indexer_score_stages.py').read_text();args=signature(old,'indexer_score_cube_step_g6')
    args[0]=('buf_query_vector','pl.Tile[[384, 128], pl.INT8, pl.MemorySpace.Mat]',None)
    args[1]=('buf_coefficients_l1','pl.Tile[[16, 384], pl.FP16, pl.MemorySpace.Mat]',None)
    ret=ast.unparse(function(old,'indexer_score_cube_step_g6').returns)
    doc='''"""当前N64双QK早FIX的一个完整1024列step，AIV FIFO合同保持。

    Query/Coeff为本叶已补齐、不可变的Mat数据；Attention可能使用Left，所以
    每次切回重新提取两份24KiB Query和两份6KiB系数。两48KiB QK Acc与4KiB
    WS Acc在本step创建并按原init_cond使用，保持FIX0早于QK1及六行一次STORE。
    原step>=2 ACK与末尾READY保留；leaf末drain仍由外层执行器负责。
    """
'''
    new=header('indexer_score_cube_step_g6',args,ret)+textwrap.indent(doc+code,'    ')
    globals_text='''
G6_QUERY_GROUP = 6
G6_QK_PANEL = 64
G6_SCORE_TILE = 1024
G6_PREFETCH = 1
G6_TO_L0 = True
G6_BALANCED = True

'''
    path=TARGET/'indexer_score_stages.py';source=path.read_text()
    assert 'def indexer_score_cube_step_g6(' not in source
    path.write_text(source+'\n'+globals_text+'@pl.jit.inline(auto_scope=False)\n'+new)

    # 原leaf loader的数据数学不变，系数改为Mat状态，由step内部恢复Left。
    path=TARGET/'interleave_cube_wave.py';source=path.read_text()
    source=source.replace('pl.Tile[[16, 384], pl.FP16, pl.MemorySpace.Left]',
                          'pl.Tile[[16, 384], pl.FP16, pl.MemorySpace.Mat]')
    old_line='    buf_coefficient_pair = pl.tile.move(buf_coefficients_l1, target_memory=pl.MemorySpace.Left)\n'
    assert source.count(old_line)==1;source=source.replace(old_line,'')
    source=source.replace('return buf_query_vector, buf_coefficient_pair','return buf_query_vector, buf_coefficients_l1')
    old_line='score_coeff = pl.create_tile([16, 384], dtype=pl.FP16, target_memory=pl.MemorySpace.Left)'
    assert source.count(old_line)==1;source=source.replace(old_line,old_line.replace('MemorySpace.Left','MemorySpace.Mat'))
    old_line='                query_left = pl.tile.move(query_ready, target_memory=pl.MemorySpace.Left)\n'
    assert source.count(old_line)==1;source=source.replace(old_line,'')
    source,ncube=rewrite_calls(source,'indexer_score_cube_step_g6',lambda a:['query_ready','coeff_ready']+a[2:])
    source=source.replace('Coeff FP16\n    [16,384] Left跨step保留','Coeff FP16\n    [16,384] Mat跨step保留，每步与两半Query一起恢复Left')
    source=source.replace('Coeff是跨步只读Left，不能与Attention或Query的Left存储重叠。',
                          'Coeff同样驻Mat；本步内恢复两份Left，不跨Attention保留Left。')
    source=source.replace('返回驻Mat Query和完整Coeff Left。Query的TMOV移到每个step之前，方便\n    Attention使用Left后恢复Score输入；Coeff仍必须由调用者完整保留。',
                          '返回驻Mat Query和完整Coeff Mat；step内部恢复当前N64两半Left，\n    Attention切换时不依赖旧Left内容，仍保留完整六query系数及尾query零项。')
    path.write_text(source)

    # Vector step直接使用当前连续scale读取及原sort/merge正文。
    fn=function(current,'indexer_score_vector_lane_stage')
    loop,=[n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='single_step']
    layout_start=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='single_page_begin')
    roots_start=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='single_roots')
    layout=textwrap.dedent(''.join(lines[layout_start.lineno-1:roots_start.lineno-1]))
    vector_code=layout+block(current,loop.body)
    vector_code=rename(vector_code,dict(query_group_size='G6_QUERY_GROUP',score_tile='G6_SCORE_TILE',
        single_roots='single_roots_local',single_segments='single_segments_local'))
    vector_code='single_roots_local = single_roots\nsingle_segments_local = single_segments\n'+vector_code
    vector_code+='\nreturn single_roots_local, single_segments_local\n'
    path=TARGET/'score_vector_steps_g6.py';source=path.read_text();args=signature(source,'score_vector_step_g6')
    at=next(i for i,x in enumerate(args) if x[0]=='buf_score_transfer')
    args.insert(at,('offsets','pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]',None))
    at=next(i for i,x in enumerate(args) if x[0]=='SCORE_READY_EVENT')
    args.insert(at,('single_valid','pl.Scalar[pl.INDEX]',None))
    args.append(('scale_page_coalescing','pl.constexpr','False'))
    doc='''"""当前连续scale/排序数学的完整step，真实leaf valid控制尾页。

    metadata布局证明来自本轮已经获取的offsets；在本step重新读取两个
    不可变Scalar，不改变方向0回退、逆序页、尾页补零或FP32乘法/归并次序。
    roots/segments使用局部Tile别名，显式返回给外层作为跨拍状态。
    """
'''
    source=replace_function(source,'score_vector_step_g6',header('score_vector_step_g6',args)+textwrap.indent(doc+vector_code,'    '))
    anchor='import pypto.language as pl\n';source=source.replace(anchor,anchor+'''\nfrom .indexer_group_stages import OFFSET_ROWS_DYN
from .indexer_scale_layout_stages import LAYOUT_WORDS_PER_REQUEST, indexer_scale_layout_base
from .indexer_score_stages import TOPK_SCORE_WORKERS
''',1)
    source=source.replace('复用正式668的原始数学','复用当前13259f92已采用的数学和连续scale路径')
    path.write_text(source)

    # 把新增真实valid从叶描述符一直携带到step，不用round后的steps代替尾部。
    sys.path.insert(0,str(HERE.parent));import prepare_qb_stream_reuse as wiring
    old_flag=wiring.FLAG
    try:
        wiring.FLAG='SCALE_PAGE_COALESCING'
        path=TARGET/'interleave_vector_wave.py';path.write_text(wiring.wire(path.read_text(),('interleave_vector_wave_g6',),()))
        path=TARGET/'vector_schedule.py';path.write_text(wiring.wire(path.read_text(),(),('interleave_vector_wave_g6',)))
    finally:wiring.FLAG=old_flag
    path=TARGET/'interleave_vector_wave.py';source=path.read_text()
    replacements=(
        ('begin_iter, steps_iter, step_iter, wrote_iter,','begin_iter, valid_iter, steps_iter, step_iter, wrote_iter,'),
        ('initial_cache_len, initial_begin, initial_steps, initial_step, initial_root_wrote,','initial_cache_len, initial_begin, initial_valid, initial_steps, initial_step, initial_root_wrote,'),
        ('        score_begin_value = begin_iter\n','        score_begin_value = begin_iter\n        score_valid_value = valid_iter\n'),
        ('cache_len_after, begin_after, steps_after, step_after, wrote_after,','cache_len_after, begin_after, valid_after, steps_after, step_after, wrote_after,'),
        ('following_live, following_cache_len, following_begin, following_steps,','following_live, following_cache_len, following_begin, following_valid, following_steps,'),
        ('score_cache_len_value, score_begin_value, score_steps_value,','score_cache_len_value, score_begin_value, score_valid_value, score_steps_value,'),
        ('current_cache_after, current_begin_after, current_steps_after,','current_cache_after, current_begin_after, current_valid_after, current_steps_after,'),
        ('query_after, live_after, cache_len_after, begin_after, steps_after,','query_after, live_after, cache_len_after, begin_after, valid_after, steps_after,'),
        ('final_cache, final_begin, final_steps, final_step, root_wrote,','final_cache, final_begin, final_valid, final_steps, final_step, root_wrote,'),
    )
    counts={}
    for before,after in replacements:
        count=source.count(before);assert count>0,(before,count)
        counts[before]=count;source=source.replace(before,after)
    source,nvector=rewrite_calls(source,'score_vector_step_g6',lambda a:
        a[:5]+['group_offsets']+a[5:-2]+['score_valid_value']+a[-2:]+['SCALE_PAGE_COALESCING'])
    path.write_text(source)
    tree=ast.parse(source);loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For)
        and isinstance(n.target,ast.Tuple) and isinstance(n.target.elts[0],ast.Name) and n.target.elts[0].id=='round_id')
    state_count=len(loop.target.elts[1].elts)
    init=next(k.value for k in loop.iter.keywords if k.arg=='init_values')
    assert len(init.elts)==state_count==18,(len(init.elts),state_count)
    assert len(loop.body[-1].value.args)==state_count
    for p in TARGET.glob('*.py'):ast.parse(p.read_text())
    report=dict(cube_step_from_current=True,vector_step_from_current=True,actual_cube_panel=64,
                score_coeff_carry_memory='Mat',left_restored_each_step=True,valid_carried=True,
                vector_loop_state_count=state_count,cube_call_sites=ncube,vector_call_sites=nvector,
                replacement_counts=counts,formal_modified=False,compiled=False,device_executed=False)
    (HERE/'PORT_SCORE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='replacement_counts'})


if __name__=='__main__':main()
