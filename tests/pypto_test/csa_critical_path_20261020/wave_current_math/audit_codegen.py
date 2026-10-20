# SPDX-License-Identifier: Apache-2.0
"""审计完整联合分配和活跃wave中的当前数学路径，不计作设备通过。"""

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from audit_score_split_store import mem,overlap

LIMITS={'pl.Mem.Vec':188416,'pl.Mem.Left':65536,'pl.Mem.Right':65536,
        'pl.Mem.Acc':131072,'pl.Mem.Mat':524288}


def annotated(fn):
    records=[]
    def walk(node,loops):
        if isinstance(node,(ast.For,ast.While)):
            label=ast.unparse(node.target) if isinstance(node,ast.For) else ast.unparse(node.test)
            loops=loops+[label]
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
            try:region=mem(node.annotation)
            except (TypeError,ValueError):region=None
            records.append(dict(name=node.target.id,line=node.lineno,memory=region,
                operation=ast.unparse(node.value),loops=loops,
                in_wave=any('round_id' in s or 'local_round' in s for s in loops)))
        for child in ast.iter_child_nodes(node):walk(child,loops)
    walk(fn,[])
    return sorted(records,key=lambda r:r['line'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compile_dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.compile_dir.resolve()
    result=json.loads((root/'result.json').read_text())
    assert result['passed'] and result['single_mix'] and len(result['symbolic_dimensions'])==58
    assert len(result['compiled_abi'])==130 and not result['sample_shapes_supplied']
    ir,=(root/'kernel/passes_dump').glob('*after_AllocateMemoryAddr.py')
    tree=ast.parse(ir.read_text());functions={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    allocations={};records={}
    for name in ('cube_schedule','vector_schedule'):
        totals=defaultdict(int);pools=[]
        for n in functions[name].body:
            if isinstance(n,ast.AnnAssign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='pl.tile.alloc':
                space=ast.unparse(n.value.args[0]);size=ast.literal_eval(n.value.args[1])
                totals[space]+=size;pools.append(dict(root=n.target.id,space=space,bytes=size))
        assert all(v<=LIMITS[k] for k,v in totals.items()),(name,totals)
        allocations[name]=dict(totals=dict(totals),pools=pools)
        records[name]=annotated(functions[name])
    vector=records['vector_schedule'];cube=records['cube_schedule']
    gather=[r for r in vector if r['operation'].startswith('pl.tile.mgather(')]
    stats=[r for r in vector if r['name'].startswith('stats_cache_mi_')]
    runs=[r for r in vector if r['name'].startswith('run_bytes_') and r['operation'].startswith('pl.tile.load(')]
    assert len(gather)==3 and sum(r['in_wave'] for r in gather)==2
    expect_stats=result['controls']['LOCAL_ATTENTION_STATS']
    expect_scale=result['controls']['SCALE_PAGE_COALESCING']
    assert len(stats)==(3 if expect_stats else 0)
    assert sum(r['in_wave'] for r in stats)==(2 if expect_stats else 0)
    assert len(runs)==(3 if expect_scale else 0)
    assert sum(r['in_wave'] for r in runs)==(2 if expect_scale else 0)
    instances=[]
    q0=[r for r in cube if r['name'].startswith('split_qk_acc0_') and r['name'].endswith('__ssa_v0')]
    q1=[r for r in cube if r['name'].startswith('split_qk_acc1_') and r['name'].endswith('__ssa_v0')]
    ws=[r for r in cube if r['name'].startswith('split_ws_acc_') and r['name'].endswith('__ssa_v0')]
    assert len(q0)==len(q1)==len(ws)==3
    assert sum(r['in_wave'] for r in q0)==2
    cpp_path=root/'kernel/ptoas/dynamic_csa_full.cpp';cpp=cpp_path.read_text()
    start=cpp.index('AICORE void cube_schedule(')
    end=cpp.index('AICORE void vector_schedule(',start)
    cube_cpp=cpp[start:end]
    def tile(name):
        at=cube_cpp.index('// pto: %'+name)
        return re.search(r'\b(v\d+) = Tile<TileType::Acc',cube_cpp[at:]).group(1)
    def mma(var,suffix=''):
        return list(re.finditer(r'\bTMATMUL'+suffix+r'\('+var+r', [^;]+;',cube_cpp))
    def fix(var):
        return list(re.finditer(r'\bTINSERT[^\n]+\(v\d+, '+var+r', [^;]+;',cube_cpp))
    def dependencies(a,b):
        segment=cube_cpp[a:b]
        sets=re.findall(r'set_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)',segment)
        waits=re.findall(r'wait_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)',segment)
        return dict(set=sets,wait=waits)
    for left,right,weight in zip(q0,q1,ws):
        a,b,c=left['memory'],right['memory'],weight['memory']
        assert a[2]==b[2]==49152 and c[2]==4096
        assert not any(overlap(x,y) for x,y in ((a,b),(a,c),(b,c)))
        av,bv,cv=tile(left['name']),tile(right['name']),tile(weight['name'])
        am,bm,af,bf,wm,wa=mma(av),mma(bv),fix(av),fix(bv),mma(cv),mma(cv,'_ACC')
        assert all(len(xs)==16 for xs in (am,bm,af,bf,wm,wa))
        assert len(re.findall(r'\bTSTORE\(v\d+, '+cv+r'\);',cube_cpp))==16
        events=[]
        for i in range(16):
            assert am[i].start()<af[i].start()<bm[i].start()<bf[i].start()
            dep=dependencies(af[i].end(),bm[i].start())
            assert not set(dep['set'])&set(dep['wait']),(left['name'],i,dep)
            item=dict(panel=i,qk0_to_qk1=dep)
            if i:
                dep=dependencies(bf[i].end(),wm[i-1].start())
                assert not set(dep['set'])&set(dep['wait']),(left['name'],i,dep)
                item['qk1_to_previous_ws']=dep
            events.append(item)
        instances.append(dict(qk0=left,qk1=right,ws=weight,cpp_vars=[av,bv,cv],
                              counts=dict(qk=32,fix=32,ws_init=16,ws_acc=16,store=16),events=events))
    report=dict(passed=True,ir=str(ir),cpp=str(cpp_path),dynamic_axes=58,root_tensors=130,
        single_mix=True,allocation_totals=allocations,active_mgather=gather,
        active_local_stats=stats,active_scale_coalescing=runs,score_instances=instances,
        device_executed=False,performance_eligible=False,
        note='完整分配总量及真实wave代码路径/事件证据；不替代设备协议、精度或性能')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(dict(passed=True,mat=allocations['cube_schedule']['totals']['pl.Mem.Mat'],
        vec=allocations['vector_schedule']['totals']['pl.Mem.Vec'],wave_mgather=2,
        wave_local_stats=expect_stats,wave_scale_coalescing=expect_scale,wave_score=2))


if __name__=='__main__':main()
