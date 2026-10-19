# SPDX-License-Identifier: Apache-2.0
"""审计独立KV与四半段oi状态的实际分配及原地清零；不替代真机正确性。"""

import argparse
import ast
import json
from pathlib import Path
import re


def memory(annotation):
    refs = [x for x in ast.walk(annotation) if isinstance(x,ast.Call) and ast.unparse(x.func)=='pl.MemRef']
    if len(refs)!=1:
        return None
    root,off,size=refs[0].args
    offset=ast.literal_eval(off.args[0]) if isinstance(off,ast.Call) else ast.literal_eval(off)
    return ast.unparse(root),offset,ast.literal_eval(size)


def overlaps(a,b):
    return a[0]==b[0] and a[1]<b[1]+b[2] and b[1]<a[1]+a[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compile_dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.compile_dir.resolve();r=json.loads((root/'result.json').read_text())
    assert r['passed'] and r['single_mix'] and len(r['symbolic_dimensions'])==58
    ir,=(root/'kernel/passes_dump').glob('*after_AllocateMemoryAddr.py')
    tree=ast.parse(ir.read_text());f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='vector_schedule')
    records={}
    for n in ast.walk(f):
        if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name):
            m=memory(n.annotation)
            if m and m[0].startswith('mem_vec_'):
                records[n.target.id]=dict(memory=m,line=n.lineno,operation=ast.unparse(n.value))
    maximum=max(x['memory'][1]+x['memory'][2] for x in records.values())
    assert maximum<=188416
    def one(prefix):
        selected=[(k,v) for k,v in records.items() if k.startswith(prefix) and k.endswith('__ssa_v0')]
        assert len(selected)==1,(prefix,len(selected))
        return selected[0]
    kv_name,kv=one('kv_dma_buffer_');_,fallback=one('qk_kv_half_');_,empty=one('kv_empty_')
    assert kv['memory']==fallback['memory']==empty['memory'] and kv['memory'][2]==65536
    assert 'pl.tile.set_validshape' in empty['operation'] and ', 0, 512)' in empty['operation']
    assert 'fillpad_inplace' in fallback['operation']
    states={}
    for side in ('left','right'):
        for half in (0,1):
            name,initial=one(f'running_{side}{half}_')
            next_name,nxt=one(f'next_{side}{half}_')
            assert initial['memory']==nxt['memory'] and initial['memory'][2]==16384
            assert not overlaps(kv['memory'],initial['memory'])
            states[f'{side}{half}']=dict(initial=name,updated=next_name,memory=initial['memory'])
    regions=[s['memory'] for s in states.values()]
    assert not any(overlaps(a,b) for i,a in enumerate(regions) for b in regions[i+1:])
    loop=next(n for n in ast.walk(f) if isinstance(n,ast.For) and 'qk_tick' in ast.unparse(n.target))
    conflicts=[];full_state_moves=[]
    for name,row in records.items():
        if not loop.lineno<=row['line']<=loop.end_lineno:
            continue
        if 'pl.tile.move(' in row['operation'] and row['memory'][2]>=32768:
            full_state_moves.append(name)
        if any(op in row['operation'] for op in ['pl.tile.row_expand_mul(', 'pl.tile.row_expand_div(', 'pl.tile.row_max(', 'pl.tile.row_sum(', 'pl.tile.exp(']):
            if overlaps(kv['memory'],row['memory']):
                conflicts.append(name)
    assert not conflicts and not full_state_moves,(conflicts,full_state_moves)
    reductions=[n for n in ast.walk(loop) if isinstance(n,ast.Call) and ast.unparse(n.func) in ['pl.tile.row_max','pl.tile.row_sum']]
    assert len(reductions)==2
    cpp=root/'kernel/ptoas/dynamic_csa_full.cpp'
    text=cpp.read_text()
    # 源码中的零有效行必须确实成为原地硬件填充，而不是另申请64KiB结果。
    marker=text.index('// pto: %qk_kv_half_', text.index('// pto: %kv_dma_buffer_'))
    following=text[marker:marker+2200]
    assert 'TFILLPAD<pto::TFillPadMode::InPlace>(' in following
    assert 'wait_flag(PIPE_MTE3, PIPE_V,' in following
    report=dict(passed=True,compile_dir=str(root),ir=str(ir),cpp=str(cpp),
                vec_bytes=maximum,vec_limit=188416,kv=dict(name=kv_name,**kv),
                fallback=fallback,zero_view=empty,states=states,
                full_32head_state_copies=full_state_moves,math_kv_overlap=conflicts,
                softmax_reduction_count=2,hardware_fillpad_inplace=True,
                single_mix=True,dynamic_axes=58,device_validated=False,
                note='地址/代码机制审计；真实页保护、舍入、同步与完整L1须设备验证')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print({'passed':True,'Vec_bytes':maximum,'KV_bytes':65536,'state_bytes_each':16384})


if __name__=='__main__':
    main()
