"""只读取既有PTO规划产物；未过PTOAS时明确不报告实际C++/机器事件。

核对完整单SPMD的Tile物理区间、DQ内部索引操作和不相关阶段分配。
保留源码定位与全部分配条目，容量上界是allocator地址包络，不是实测驻留峰值。
"""
from pathlib import Path
import ast
import collections
import json
import re

OUT=Path(__file__).resolve().parent
BASE=Path('/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_indexer_20261018/query_dequant_group_20261018')

def collect(kernel,source):
    text=(kernel/'ptoas/dynamic_csa_full.pto').read_text()
    functions={}
    for p in source.glob('*.py'):
        functions[p.name]=[(n.lineno,n.end_lineno,n.name) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)]
    role=''
    rows=[]
    maxima={}
    for lineno,line in enumerate(text.splitlines(),1):
        f=re.search(r'func.func @(\w+)',line)
        if f: role=f[1]
        if 'pto.alloc_tile' not in line: continue
        m=re.search(r'(%\w+) = pto.alloc_tile addr = %c(\d+)_i64.*?loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)',line)
        assert m,line
        name,addr,space,dtype,rs,cs=m.groups()
        size=int(rs)*int(cs)*int(re.search(r'\d+',dtype)[0])//8
        loc=re.search(r'loc\("[^\"]*/([^/]+\.py)":(\d+):',line)
        function=None
        if loc:
            function=next((n for lo,hi,n in functions.get(loc[1],[]) if lo<=int(loc[2])<=hi),None)
        row=dict(name=name,role=role,address=int(addr),bytes=size,end=int(addr)+size,space=space,dtype=dtype,rows=int(rs),cols=int(cs),source_file=loc[1] if loc else None,source_function=function,pto_line=lineno)
        rows.append(row)
        key=role+'/'+space
        maxima[key]=max(maxima.get(key,0),row['end'])
    dq=[r for r in rows if r['source_function']=='q_dequant_rope_stage' and r['role']=='vector_schedule']
    stage_ops=[]
    for lineno,line in enumerate(text.splitlines(),1):
        if re.search(r'pto\.(?:tload|tcvt|tmuls|tadd|tci|txors)\b',line):
            loc=re.search(r'query_projection_stages.py":(\d+):',line)
            if loc and any(lo<=int(loc[1])<=hi and n=='q_dequant_rope_stage' for lo,hi,n in functions['query_projection_stages.py']):
                stage_ops.append({'line':lineno,'op':re.search(r'pto\.(\w+)',line)[1],'text':line})
    return {'path':str(kernel),'allocation_upper_bounds':maxima,'allocations':rows,'dq_allocations':dq,'dq_operation_counts':dict(collections.Counter(x['op'] for x in stage_ops)),'dq_operations':stage_ops},text

def main():
    import sys
    version=sys.argv[1] if len(sys.argv)>1 else 'v1'
    source=OUT/('source_candidate' if version=='v1' else 'source_v2')
    cpu=OUT/('cpu_'+version)
    old,oldpto=collect(BASE/'cpu_v1/kernel',BASE/'source_candidate')
    new,newpto=collect(cpu/'kernel',source)
    def key(r):
        return tuple(r[k] for k in ('role','source_file','source_function','space','dtype','rows','cols','address','bytes'))
    other=lambda x:[key(r) for r in x['allocations'] if r['source_function']!='q_dequant_rope_stage']
    named={name:next((r for r in new['dq_allocations'] if r['name'].startswith('%'+name+'_') and 'ci_tmp' not in r['name']),None) for name in ('q_flat_i','q_swap_xor_tmp','q_abs_swap','q_swap_i16')}
    live=[named[k] for k in ('q_flat_i','q_swap_xor_tmp', 'q_abs_swap' if version=='v1' else 'q_swap_i16')]
    independent=all(a['end']<=b['address'] or b['end']<=a['address'] for n,a in enumerate(live) for b in live[n+1:])
    result=json.loads((cpu/'result.json').read_text())
    obj={'cpu_passed':result['passed'],'symbolic_dimensions':len(result['symbolic_dimensions']),'device_execution':False,'baseline':old,'candidate':new,'checks':{'txors_live_three_disjoint_in_ir':independent,'all_other_stage_allocation_sequence_equal':other(old)==other(new),'whole_allocator_upper_bounds_equal':old['allocation_upper_bounds']==new['allocation_upper_bounds']},'txors_tiles':named,'difference_dq_op_counts':{k:new['dq_operation_counts'].get(k,0)-old['dq_operation_counts'].get(k,0) for k in old['dq_operation_counts'].keys()|new['dq_operation_counts'].keys()},'scope':'PTO规划产物；未过PTOAS时不是可执行C++证据','cpp_generated':(cpu/'kernel/ptoas/dynamic_csa_full.cpp').exists(),'runtime_or_machine_effect_validated':False}
    (OUT/('ir_audit_'+version+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in obj.items() if k not in ('baseline','candidate','txors_tiles')},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
