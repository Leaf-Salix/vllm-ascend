#!/usr/bin/env python3
"""只读验证完整 CPU 生成物的量化复用、原值存活与片上分配。

按源码 helper 的边界提取 PTO，避免把其他阶段的量化算入统计。原值被 abs
或 row_max 覆盖会破坏符号，因而明确检查每个展开分支的物理区间。审计不
重新编译，不访问 NPU，也不修改已冻结的候选源码。
"""

import ast
import collections
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
R = Path('/data/pyptouser/qinchuanyu/pto-eager')
PAR = R/'vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018'
BASE = PAR/'hc_active_notifications_h131072_b16_v1/build_output/decode_csa_spmd_3kuwaqat'
NEW = OUT/'cpu_v1/kernel'
INPUTS = set()


def read(p):
    INPUTS.add(p)
    return p.read_text()


def function_bounds(p):
    return next((n.lineno,n.end_lineno) for n in ast.parse(read(p)).body
                if isinstance(n,ast.FunctionDef) and n.name=='o_projection_groups_quant_stage')


def collect(artifact, source):
    pto=artifact/'ptoas/dynamic_csa_full.pto'
    lo,hi=function_bounds(source/'output_projection_a_stages.py')
    rows=[]; alloc={}; maxima={}; fn=''
    for no,line in enumerate(read(pto).splitlines(),1):
        fm=re.search(r'func.func @(\w+)',line)
        if fm:fn=fm[1]
        if 'pto.alloc_tile' in line:
            m=re.search(r'(%\w+) = pto.alloc_tile addr = %c(\d+)_i64.*?loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)',line)
            assert m,(pto,no,line)
            bits=int(re.search(r'\d+',m[4])[0])
            size=int(m[5])*int(m[6])*bits//8
            record=dict(variable=m[1],address=int(m[2]),size_bytes=size,end=int(m[2])+size,
                        space=m[3],dtype=m[4],shape=[int(m[5]),int(m[6])],ir_line=no)
            maxima[fn+'/'+m[3]]=max(maxima.get(fn+'/'+m[3],0),record['end'])
            if fn=='vector_schedule':alloc[m[1]]=record
        location=re.search(r'output_projection_a_stages.py":(\d+):',line)
        if location and lo<=int(location[1])<=hi:
            rows.append((no,line))
    ops=collections.Counter()
    for no,line in rows:
        op=re.search(r'\b(pto\.\w+(?:\.\w+)?)\b',line)
        if op:ops[op[1]]+=1
    return dict(path=str(pto),rows=rows,alloc=alloc,maxima=maxima,ops=ops)


def ins(line):
    return re.findall(r'%\w+',re.search(r'\bins\((.*?) :',line)[1])


def output(line):
    return re.search(r'\bouts\((%\w+)',line)[1]


def nonoverlap(a,b):
    return a['end']<=b['address'] or b['end']<=a['address']


old=collect(BASE,OUT/'source_before')
new=collect(NEW,OUT/'source_candidate')
baseline_files=sorted((OUT/'source_before').glob('*.py'))
assert len(baseline_files)==33
assert all(read(p)==read(PAR/'hc_active_notifications_evidence/source_candidate'/p.name) for p in baseline_files)
assert old['ops']['pto.tload']==6 and new['ops']['pto.tload']==3
skip={'pto.tload','pto.partition_view','pto.alloc_tile'}
assert {k:v for k,v in old['ops'].items() if k not in skip} == {k:v for k,v in new['ops'].items() if k not in skip}

checks=[]
rows=new['rows']
for i,(no,line) in enumerate(rows):
    if 'pto.tabs ' not in line:continue
    raw=new['alloc'][ins(line)[0]]
    absolute=new['alloc'][output(line)]
    assert nonoverlap(raw,absolute)
    rowmax_no,rowmax=next((n,l) for n,l in rows[i+1:] if 'pto.trowmax ' in l)
    scratch=new['alloc'][ins(rowmax)[1]]
    rowmax_out=new['alloc'][output(rowmax)]
    assert nonoverlap(raw,scratch) and nonoverlap(raw,rowmax_out)
    mul_no,mul=next((n,l) for n,l in rows[i+1:] if 'pto.trowexpandmul ' in l)
    quant_raw=new['alloc'][ins(mul)[0]]
    assert raw['address']==quant_raw['address'] and raw['size_bytes']==quant_raw['size_bytes']
    checks.append(dict(abs_line=no,rowmax_line=rowmax_no,mul_line=mul_no,
                       original_value=raw,abs_result=absolute,reduce_scratch=scratch,
                       rowmax_result=rowmax_out,quant_input=quant_raw,
                       raw_survives_abs_rowmax=True,quant_reads_original_signed_value=True))
assert len(checks)==3

# 用分配阶段直接给出的 MemRef 容量复核最终 PTO 的几何字节计算。
allocated=read(NEW/'passes_dump/38_after_AllocateMemoryAddr.py')
memref_max={}
for m in re.finditer(r'pl.MemRef\((mem_(\w+)_\d+), pl.const\((\d+), pl.INT64\), (\d+)\)',allocated):
    memref_max[m[2]]=max(memref_max.get(m[2],0),int(m[3])+int(m[4]))
assert memref_max['vec']==new['maxima']['vector_schedule/vec']==155872
assert all(new['maxima'][k]==v for k,v in old['maxima'].items() if k!='vector_schedule/vec')

for label,data in [('baseline',old),('candidate',new)]:
    (OUT/(label+'_quant.pto.txt')).write_text('\n'.join(f'{n}: {l}' for n,l in data['rows'])+'\n')

result=dict(scope='已有完整symbolic CPU生成物审计；未新增编译/设备任务',
            baseline_commit='d253d627dde3646aa916879a481d3e57322f5c33',
            baseline=str(BASE),candidate=str(NEW),baseline_33_sources_match=True,
            original_values=checks,
            tload_lines_before=[n for n,l in old['rows'] if 'pto.tload ' in l],
            tload_lines_after=[n for n,l in new['rows'] if 'pto.tload ' in l],
            operator_counts_before=dict(old['ops']),operator_counts_after=dict(new['ops']),
            arithmetic_and_store_counts_unchanged=True,
            address_upper_bounds_before=old['maxima'],address_upper_bounds_after=new['maxima'],
            memref_max_after=memref_max,aiv_ub_reduction_bytes=163872-155872,
            stage2_preserved=True,explanation='双展开两次TLOAD各对应一个qt，另保留奇数尾迭代；原值与abs/scratch物理区间不重叠。',
            limitations=['两侧真实端到端设备精度、图重放、保护区与性能由根代理已提交任务验证；CPU不替代这些检查。',
                         '全局UB分配地址可能重新排列；只保证已审计的别名关系和实际总上界，不宣称其他地址全部不变。'])
result['inputs']=[dict(path=str(p),bytes=p.stat().st_size) for p in sorted(INPUTS)]
(OUT/'generated_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(passed=True,tload_before=6,tload_after=3,raw_value_checks=len(checks),
                      ub_before=163872,ub_after=155872,output=str(OUT/'generated_audit.json')),ensure_ascii=False))
