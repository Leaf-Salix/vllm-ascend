"""按完整tick抽取typed-Tile源码，并做纯标准库AST还原检查；不导入PyPTO。"""
import ast
import copy
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'source_reference/attention_stream_stages.py'
OUTPUT = HERE/'source_candidate/attention_tick_stages.py'


def find_fn(tree,name):
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


def doc_free(body):
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):return body[1:]
    return body


def dump(body):
    return ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False)


def format_args(args):
    return '\n'.join('    '+name+': '+annotation+(',' if default is None else ' = '+default+',') for name,annotation,default in args)


def function_text(name,args,returns,doc,math,ret):
    return '@pl.jit.inline(auto_scope=False)\ndef '+name+'(\n'+format_args(args)+'\n) -> tuple[\n'+''.join('    '+r+',\n' for r in returns)+']:\n'+textwrap.indent('"""'+doc+'"""\n','    ')+math+'\n    return '+', '.join(ret)+'\n\n\n'


def main():
    original=SOURCE.read_text();tree=ast.parse(original);lines=original.splitlines(keepends=True)
    cube=find_fn(tree,'attention_qk_pv_cube_stage');vec=find_fn(tree,'attention_qk_pv_vector_lane_stage')
    c_loop=next(n for n in cube.body if isinstance(n,ast.For));v_loop=next(n for n in vec.body if isinstance(n,ast.For))
    c_prefix=doc_free(cube.body[:cube.body.index(c_loop)]);v_prefix=doc_free(vec.body[:vec.body.index(v_loop)])
    c_init=''.join(lines[c_prefix[0].lineno-1:c_loop.lineno-1])
    v_init=''.join(lines[v_prefix[0].lineno-1:v_loop.lineno-1])
    c_body=''.join(line[4:] if line.startswith('    ') else line for line in lines[c_loop.lineno:c_loop.end_lineno])
    # 最后一个yield只是原外层loop的回边；函数边界改显式return，其余if yield保留。
    backedge=v_loop.body[-1]
    assert isinstance(backedge,ast.Assign) and isinstance(backedge.value,ast.Call) and ast.unparse(backedge.value.func)=='pl.yield_'
    v_body=''.join(line[4:] if line.startswith('    ') else line for line in lines[v_loop.body[0].lineno-1:backedge.lineno-1])
    c_orig={a.arg:ast.unparse(a.annotation) for a in cube.args.args}
    v_orig={a.arg:ast.unparse(a.annotation) for a in vec.args.args}
    scalar='pl.Scalar[pl.INDEX]'
    c_q='pl.Tile[[H, HEAD_DIM], pl.BF16, pl.MemorySpace.Mat]'
    c_kv='pl.Tile[[QK_L1_SLOTS * ATTN_K_TILE, HEAD_DIM], pl.BF16, pl.MemorySpace.Mat]'
    v_ml='pl.Tile[[H // 2, 1], pl.FP32, pl.MemorySpace.Vec]'
    v_oi='pl.Tile[[H // 2, HEAD_DIM // 2], pl.FP32, pl.MemorySpace.Vec]'
    v_swap='pl.Tile[[1, FINAL_HEAD_TILE * ROPE_DIM], pl.INT32, pl.MemorySpace.Vec]'
    header='"""动态CSA Attention完整tick的typed-Tile接口：仅源码准备，尚未编译验证。\n\n整体思路、阅读顺序与当前边界\n    attention_cube_init / attention_vector_init保留原初始化数学与set_ffts，\n    返回真实T、当前核的全局query基址/块数及本核跨tick状态；每次调用生命周期\n    只初始化一次。随后调用attention_cube_tick / attention_vector_tick推进一拍。\n    本模块直接抽取已冻结的两L1候选，没有实现Score交错、wave控制或新调度。\n    typed-Tile参数/返回依赖尚未验证的JIT能力；源码AST相同不等于可lowering、\n    设备精度/协议/资源已通过。不得作为已接入完整CSA的生产入口。\n\n输入、输出与动态范围\n    GM布局沿原动态CSA：64头/512维，原KV与压缩KV按Native device页表读取，\n    Q/L/positions/mask仍是device Tensor；业务容量轴保持pl.dynamic。\n    每组query公式仍为qk_core+(work//5)*CORE_GROUPS；没有wave内重编号。\n    tick、block_count、qk_core及两AIV lane偏移显式为Scalar；AIV的t_dim必须\n    是真实全局position_ids.shape[0]，不可替换为wave长度或pad容量。\n    EARLY_KV的T>=144分支因此保持原义。固定Tile尺寸来自模型/数学块，不冻结T/B。\n    当前只传递full-valid数学Tile，不把valid_shape=min(T,...)的Tile跨步传入。\n    full-valid仅指形状metadata；KV未填充槽的数据仍不可读取，必须沿原mask/槽时序。\n    GM参数沿原Tensor/InOut注解，并显式返回原输出Tensor，保留原数据流；\n    Tile参数显式返回更新句柄，不把原地缓冲语义误当成Python赋值已自动回传。\n\n跨步状态与两个角色\n    AIC持Q BF16[64,512] Mat和KV BF16[256,512] Mat，后者内部两槽；GM所有\n    环仍三槽。本拍先PV(tick-2)后QK(tick)，旧KV槽实际MTE1读取完成才可覆盖。\n    AIV持FP32 m/l[32,1]、left/right[32,256]四Tile；只读swap索引INT32\n    [1,1024]由init一次生成。每拍gather本块、softmax上一块、merge三拍前PV，\n    最后块归一化/逆RoPE/分组BF16输出并按原数学重置四Tile；swap不重置。\n    本核临时Tile仍留在完整tick内，不扩成长期驻留状态；不引入新的GM工作区。\n\n阶段/事件与尾部调用合同\n    AIC调用tick范围[0,block_count+2)，AIV为[0,block_count+3)，正好保留填充\n    和排空。调用方不能多推进一拍、跳过有效拍或省掉尾拍；空组仍遵守原边界。\n    同组AIC与双AIV必须并发推进，FFTS基址、EVENT_BASE与原mask一致；每块\n    READY/wait条件和pipe不改，普通phase通知不能代替槽协议。首query/新query\n    的Q加载、无效块跳过、末块状态重置均在原位置，没有新增无条件清零。\n    GM环%3与L1环%2独立；两helper源码局部顺序不是异步完成证据。后端仍须\n    证明MTE1→MTE2覆盖依赖、Mat回边/布局与UB峰值；现有编译产物不能代替本模块。\n"""\n\n'
    # 导入和全局模型常量直接复用两L1源；不保留旧整体stage函数。
    first_import=next(n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)))
    globals_text=''.join(lines[first_import.lineno-1:min(d.lineno for d in cube.decorator_list)-1])
    copyright=''.join(lines[:8])+'\n'
    c_init_args=[(n,c_orig[n],None) for n in ('q_flat','ffts_workspace','core_id','CORE_GROUPS')]
    c_init_text=function_text('attention_cube_init',c_init_args,[scalar,scalar,scalar,c_q,c_kv],
        '初始化AIC跨拍状态；T与block_count沿原全局query公式，FFTS仅此时声明。\n    返回顺序为T/core/block_count/Q Tile/KV双槽Tile；调用者保存后逐拍传回。',c_init,
        ['t_dim','qk_core','qk_block_count','qk_q','qk_l1'])
    c_gm=['q_flat','valid_block_mask','kv_transfer','score_transfer','probability_transfer','pv_transfer']
    c_tick_args=[(n,c_orig[n],None) for n in c_gm]+[(n,scalar,None) for n in ['qk_tick','qk_block_count','qk_core']]+[('qk_q',c_q,None),('qk_l1',c_kv,None),('CORE_GROUPS','pl.constexpr',None),('EVENT_BASE','pl.constexpr','0')]
    c_tick_returns=['pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32]','pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32]',c_q,c_kv]
    c_tick_text=function_text('attention_cube_tick',c_tick_args,c_tick_returns,
        '执行完整AIC一拍；PV旧槽读取在前，QK新槽写入在后。\n    Q/双槽Tile必须来自同一次init/上一tick；返回后不能借Scalar完成推断DMA完成。\n    tick边界由调用者保证，真实mask控制两端事件，不增加入口裁剪改变原语义。',c_body,
        ['score_transfer','pv_transfer','qk_q','qk_l1'])
    v_init_args=[(n,v_orig[n],None) for n in ('position_ids','attn_sink_col','ffts_workspace','core_id','lane_id','CORE_GROUPS')]
    v_init_returns=[scalar]*5+[v_swap,v_ml,v_ml,v_oi,v_oi]
    v_init_text=function_text('attention_vector_init',v_init_args,v_init_returns,
        '初始化本AIV的几何Scalar、只读swap与四份在线归约Tile。\n    m从sink加载、l从零开始，sink只在原最终分母处加入；初始化数学不可替换。\n    返回T/core/block_count/lane_head/lane_kv/swap/m/l/left/right。',v_init,
        ['t_dim','qk_core','qk_block_count','qk_lane_head','qk_lane_kv','m_swap_idx','running_m','running_l','running_left','running_right'])
    v_gm=[a.arg for a in vec.args.args if a.arg not in {'ffts_workspace','core_id','lane_id','CORE_GROUPS','EVENT_BASE','CACHE_QUERY_METADATA'}]
    v_tick_args=[(n,v_orig[n],None) for n in v_gm]+[(n,scalar,None) for n in ['qk_tick','qk_block_count','qk_core','qk_lane_head','qk_lane_kv','t_dim']]+[('m_swap_idx',v_swap,None),('m_iter',v_ml,None),('l_iter',v_ml,None),('left_iter',v_oi,None),('right_iter',v_oi,None),('CORE_GROUPS','pl.constexpr',None),('EVENT_BASE','pl.constexpr','0'),('CACHE_QUERY_METADATA','pl.constexpr','False')]
    v_gm_returns=['pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16]','pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.BF16]','pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]','pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]','pl.Tensor[[PACK_ROWS_DYN, O_GROUP_IN], pl.BF16]']
    v_tick_text=function_text('attention_vector_tick',v_tick_args,v_gm_returns+[v_ml,v_ml,v_oi,v_oi],
        '执行完整AIV一拍，并显式返回m/l/left/right四个下一拍状态。\n    swap只读；原if中的pl.yield保留，只有最外loop回边改为函数返回。\n    实际全局T决定早KV分支；末块即使无效仍按原路径发布/重置，不能跳过。\n    原gather/softmax/merge顺序与Scalar地址guard全部保留。',v_body,
        ['kv_transfer','probability_transfer','mi_transfer','li_transfer','o_packed_heads','m_after','l_after','left_after','right_after'])
    output=copyright+header+globals_text+c_init_text+c_tick_text+v_init_text+v_tick_text
    parsed=ast.parse(output)
    c_i=find_fn(parsed,'attention_cube_init');c_t=find_fn(parsed,'attention_cube_tick');v_i=find_fn(parsed,'attention_vector_init');v_t=find_fn(parsed,'attention_vector_tick')
    assert dump(doc_free(c_i.body)[:-1])==dump(c_prefix)
    assert dump(doc_free(v_i.body)[:-1])==dump(v_prefix)
    assert dump(doc_free(c_t.body)[:-1])==dump(c_loop.body)
    assert dump(doc_free(v_t.body)[:-1])==dump(v_loop.body[:-1])
    assert [ast.unparse(v) for v in backedge.value.args]==['m_after','l_after','left_after','right_after']
    assert [ast.unparse(v) for v in v_t.body[-1].value.elts][-4:]==[ast.unparse(v) for v in backedge.value.args]
    assert [ast.unparse(v) for v in c_t.body[-1].value.elts][:2]==[ast.unparse(v) for v in cube.body[-1].value.elts]
    assert [ast.unparse(v) for v in v_t.body[-1].value.elts][:5]==[ast.unparse(v) for v in vec.body[-1].value.elts]
    original_globals=[n for n in tree.body if not isinstance(n,ast.FunctionDef) and n is not tree.body[0]]
    new_globals=[n for n in parsed.body if not isinstance(n,ast.FunctionDef) and n is not parsed.body[0]]
    assert dump(original_globals)==dump(new_globals)
    # 参数覆盖检查：外层range的索引/状态都必须变成显式输入，不留悬空闭包。
    global_names={'pl','M'}
    for n in parsed.body:
        if isinstance(n,ast.Assign):global_names.update(t.id for t in n.targets if isinstance(t,ast.Name))
        elif isinstance(n,ast.ImportFrom):global_names.update(a.asname or a.name for a in n.names)
    interface=[]
    for fn in [c_i,c_t,v_i,v_t]:
        body=ast.Module(body=doc_free(fn.body),type_ignores=[])
        reads={n.id for n in ast.walk(body) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
        writes={n.id for n in ast.walk(body) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        args={a.arg for a in fn.args.args}
        missing=sorted(reads-writes-args-global_names)
        assert not missing,(fn.name,missing)
        assert ast.dump(fn.decorator_list[0],include_attributes=False)==ast.dump(cube.decorator_list[0],include_attributes=False)
        interface.append({'name':fn.name,'parameters':[{'name':a.arg,'annotation':ast.unparse(a.annotation)} for a in fn.args.args],'return_annotation':ast.unparse(fn.returns),'returns':[ast.unparse(v) for v in fn.body[-1].value.elts],'free_variables':missing,'lines':[fn.lineno,fn.end_lineno]})
    assert not any(isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='m_swap_idx' for t in n.targets for x in ast.walk(t)) for n in ast.walk(v_t))
    compile(output,str(OUTPUT),'exec')
    if OUTPUT.exists():assert OUTPUT.read_text()==output,'拒绝覆盖不同的冻结tick源码'
    else:OUTPUT.write_text(output)
    report={'passed':True,'source':str(SOURCE),'source_bytes':SOURCE.stat().st_size,'output':str(OUTPUT),'output_bytes':OUTPUT.stat().st_size,'checks':{'module_imports_constants_equal':True,'cube_init_math_ast_equal':True,'vector_init_math_ast_equal':True,'cube_tick_body_ast_equal':True,'vector_tick_body_ast_equal_except_outer_backedge':True,'original_gm_returns_preserved':True,'four_vector_carries_explicit':True,'swap_not_reassigned_in_tick':True,'decorators_unchanged':True,'no_free_local_variables':True,'syntax_passed':True},'structural_changes':['两个整体stage拆为各自init+tick','AIC Q/KV Tile新增显式参数/返回','AIV四Tile与swap新增typed参数，四Tile显式返回','外层range不在tick helper内；调用者负责原tick界及尾拍','原AIV最外pl.yield改为return；内部if yields不变'],'interface':interface,'pypto_imported':False,'pypto_parse_or_lowering_run':False,'device_execution':False,'typed_tile_jit_patch_validated':False,'performance_claim':False}
    (HERE/'source_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='interface'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
