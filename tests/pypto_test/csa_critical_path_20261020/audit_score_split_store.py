# SPDX-License-Identifier: Apache-2.0
"""审计双QK早FIX和单次WS写回的真实地址/事件，不以容量估算替代生成证据。"""

import argparse
import ast
import json
from pathlib import Path
import re


def mem(annotation):
    refs=[n for n in ast.walk(annotation) if isinstance(n,ast.Call) and ast.unparse(n.func)=='pl.MemRef']
    if len(refs)!=1:
        return None
    root,offset,size=refs[0].args
    off=ast.literal_eval(offset.args[0]) if isinstance(offset,ast.Call) else ast.literal_eval(offset)
    return [ast.unparse(root),off,ast.literal_eval(size)]


def overlap(a,b):
    return a[0]==b[0] and a[1]<b[1]+b[2] and b[1]<a[1]+a[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compile_dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.compile_dir.resolve()
    result=json.loads((root/'result.json').read_text())
    assert result['passed'] and result['single_mix'] and len(result['symbolic_dimensions'])==58
    path,=(root/'kernel/passes_dump').glob('*after_AllocateMemoryAddr.py')
    tree=ast.parse(path.read_text());cube=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='cube_schedule')
    records={}
    for n in ast.walk(cube):
        if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name):
            region=mem(n.annotation)
            if region:records[n.target.id]=dict(memory=region,operation=ast.unparse(n.value))
    def one(prefix):
        vals=[v for n,v in records.items() if n.startswith(prefix) and n.endswith('__ssa_v0')]
        assert len(vals)==1,(prefix,len(vals))
        return vals[0]
    q0=one('split_qk_acc0_')['memory'];q1=one('split_qk_acc1_')['memory'];ws=one('split_ws_acc_')['memory']
    assert q0[2]==q1[2]==49152 and ws[2]==4096
    assert not any(overlap(a,b) for a,b in [(q0,q1),(q0,ws),(q1,ws)])
    left=[one(f'split_{kind}{half}_')['memory'] for kind in ['query','coefficient'] for half in (0,1)]
    assert sorted(x[2] for x in left)==[6144,6144,24576,24576]
    assert not any(overlap(a,b) for i,a in enumerate(left) for b in left[i+1:])
    right={n:v['memory'] for n,v in records.items() if re.match(r'split_right[01](?:_last)?_',n)}
    assert len(right)==32 and all(m[2]==24576 for m in right.values())
    right_regions=sorted({tuple(v) for v in right.values()})
    assert len(right_regions)==2 and not overlap(*right_regions)
    step,=[n for n in ast.walk(cube) if isinstance(n,ast.For) and 'buf_score_step' in ast.unparse(n.target)]
    calls=[n for n in ast.walk(step) if isinstance(n,ast.Call)]
    stores=[n for n in calls if ast.unparse(n.func)=='pl.tile.store' and ast.unparse(n.args[2]).startswith('indexer_transfer')]
    assert len(stores)==16
    qk_ops=[n for n in calls if ast.unparse(n.func)=='pl.tile.matmul_acc' and ast.unparse(n.args[0]).startswith('split_qk_acc')]
    ws_first=[n for n in calls if ast.unparse(n.func)=='pl.tile.matmul_acc' and ast.unparse(n.args[0]).startswith('split_ws_acc')]
    ws_second=[n for n in calls if ast.unparse(n.func)=='pl.tile.matmul_acc' and ast.unparse(n.args[0]).startswith('split_ws_first')]
    assert len(qk_ops)==32 and len(ws_first)==len(ws_second)==16
    cpp_path=root/'kernel/ptoas/dynamic_csa_full.cpp';cpp=cpp_path.read_text()
    def tile(prefix):
        start=cpp.index('// pto: %'+prefix)
        return re.search(r'\b(v\d+) = Tile<TileType::Acc',cpp[start:]).group(1)
    q0var=tile('split_qk_acc0_');q1var=tile('split_qk_acc1_');wsvar=tile('split_ws_acc_')
    def mma(var,suffix=''):
        return list(re.finditer(r'\bTMATMUL'+suffix+r'\('+var+r', [^;]+;',cpp))
    q0m=mma(q0var);q1m=mma(q1var);w0m=mma(wsvar);w1m=mma(wsvar,'_ACC')
    def fix(var):
        return list(re.finditer(r'\bTINSERT[^\n]+\(v\d+, '+var+r', [^;]+;',cpp))
    q0f=fix(q0var);q1f=fix(q1var)
    assert all(len(x)==16 for x in [q0m,q1m,q0f,q1f,w0m,w1m])
    events=[]
    for i in range(16):
        assert q0m[i].start()<q0f[i].start()<q1m[i].start()<q1f[i].start()
        fragment=cpp[q0f[i].end():q1m[i].start()]
        publish=re.findall(r'set_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)',fragment)
        waits=re.findall(r'wait_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)',fragment)
        # 最后panel在本轮不再复用QK Acc，可没有FIX→M释放事件；这不应
        # 被误判为未形成早FIX。此前panel仍要求明确释放并核对事件不交叉。
        assert (publish or i==15) and not set(publish)&set(waits)
        item=dict(panel=i,qk0_fix_before_qk1=True,qk0_release=publish,qk1_wait=waits,current_fix_does_not_gate_qk1=True)
        if i>0:
            fragment=cpp[q1f[i].end():w0m[i-1].start()]
            publish=re.findall(r'set_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)',fragment)
            waits=re.findall(r'wait_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)',fragment)
            assert (publish or i==15) and not set(publish)&set(waits)
            item.update(qk1_release=publish,previous_ws_wait=waits,current_fix_does_not_gate_previous_ws=True)
        events.append(item)
    assert len(re.findall(r'\bTSTORE\(v\d+, '+wsvar+r'\);',cpp))==16
    report=dict(passed=True,compile_dir=str(root),ir=str(path),cpp=str(cpp_path),dynamic_axes=58,single_mix=True,
                qk_acc=[q0,q1],ws_acc=ws,left=left,right_regions=right_regions,
                qk_mma_count=32,qk_fix_count=32,ws_first_count=16,ws_accumulate_count=16,ws_store_count=16,
                cpp_acc_variables=dict(qk0=q0var,qk1=q1var,ws=wsvar),panel_events=events,
                note='地址和事件提交机制通过；不证明硬件实际重叠、舍入或完整L1收益')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print({'passed':True,'qk_acc_bytes':[49152,49152],'ws_acc_bytes':4096,'ws_stores':16,'early_fix_panels':16})


if __name__=='__main__':
    main()
