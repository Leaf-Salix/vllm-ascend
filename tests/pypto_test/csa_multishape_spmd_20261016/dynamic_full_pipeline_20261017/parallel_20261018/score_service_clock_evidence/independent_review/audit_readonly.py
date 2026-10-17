#!/usr/bin/env python3
"""只用标准库对冻结 hook/runner 和既有 C++ 做独立审阅；不编译或上卡。"""
import ast
import json
from pathlib import Path
import re

ROOT=Path('/data/pyptouser/qinchuanyu/pto-eager')
MAIN=ROOT/'vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017'
PAR=MAIN/'parallel_20261018'
H=ROOT/'.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/score_service_clock_20261018'
F=ROOT/'.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/score_service_clock_runner_20261018'
REF=PAR/'oa_pair_manual_h131072_b16_v1/build_output/decode_csa_spmd_jp7o4g2r'
HERE=Path(__file__).resolve().parent

def module_text(path):
    # 这些脚本的模块级代码只定义函数/常量及导入标准库；main和后端上下文不执行。
    ns={'__name__':'readonly_'+path.stem,'__file__':str(path)}
    exec(compile(path.read_text(),str(path),'exec'),ns)
    return ns

hook=module_text(H/'score_service_clock.py')
runner=module_text(F/'run_score_diagnostic.py')
phase=module_text(MAIN/'compile_diagnostics.py')
wait=module_text(F/'source_snapshot/wait_codegen.py')
prep=module_text(F/'prepare_runner.py')

def scope_pairs(marked):
    # 等长遮蔽注释/字符串，再按实际词法大括号建立祖先栈；不是C++语义编译。
    lexical=re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',re.S)
    clean=lexical.sub(lambda m:' '*len(m[0]),marked)
    starts=list(re.finditer(r'const uint64_t (csa_score_start_\d+) = get_sys_cnt\(\);',clean))
    ends=list(re.finditer(r'get_sys_cnt\(\) - (csa_score_start_\d+)',clean))
    assert len(starts)==len(ends)==len({m[1] for m in starts})==20
    points=sorted([(m.start(),'start',m[1]) for m in starts]+[(m.start(),'end',m[1]) for m in ends])
    scopes={};stack=[];previous=0
    for pos,kind,name in points:
        for index in range(previous,pos):
            if clean[index]=='{':stack.append(index)
            elif clean[index]=='}':stack.pop()
        scopes[(kind,name)]=tuple(stack);previous=pos
    result=[]
    for start in starts:
        end=[m for m in ends if m[1]==start[1]]
        assert len(end)==1
        name=start[1];a=scopes[('start',name)];b=scopes[('end',name)]
        assert a==b and start.start()<end[0].start()
        result.append({'name':name,'same_scope':True,'start_precedes_unique_end':True,
                       'start_line':marked[:start.start()].count('\n')+1,
                       'end_line':marked[:end[0].start()].count('\n')+1,
                       'brace_ancestor_depth':len(a)})
    return result

result={'ready':True,'scope':'标准库和既有CPP只读；不等于CCEC或设备验证','inputs':[]}
for label,path in [('raw',REF/'ptoas/dynamic_csa_full.cpp'),('aic_wrapper',REF/'kernels/aic/cube_schedule.cpp'),('aiv_wrapper',REF/'kernels/aiv/vector_schedule.cpp')]:
    original=path.read_text(); marked,audit=hook['instrument_cpp'](original)
    assert hook['TAG'].sub('',marked)==original
    assert hook['SYNC'].findall(marked)==hook['SYNC'].findall(original)
    assert hook['_function'](marked,'vector_schedule')[2]==hook['_function'](original,'vector_schedule')[2]
    assert re.findall(r'csa_score_dst\[(\d+)\] =',marked)[-1]=='1'
    stores=[int(x) for x in re.findall(r'csa_score_dst\[(\d+)\] =',marked)]
    assert len(stores)==len(set(stores))==32 and set(stores)==set(range(32))
    assert marked.count('csa_score_dst = ')==1
    assert f'.data()) + {34*72*128+32};' in marked
    checks=scope_pairs(marked)
    result['inputs'].append({'label':label,'path':str(path),'bytes':path.stat().st_size,
                            'strip_exact':True,'sync_sequence_exact':True,'vector_function_exact':True,
                            'scope_checks':checks,'padding_words_once':32,'epoch_store_last':True})
    if label=='raw':raw=original;raw_marked=marked;raw_audit=audit

# 以实际组合次序做纯文本变换；Score之外的Vector变化只能来自已有phase_clock。
combined,_=wait['rewrite_wait_source'](raw_marked)
combined,_=phase['instrument_clock'](combined,wait_backoff=True)
phase_only,_=wait['rewrite_wait_source'](raw)
phase_only,_=phase['instrument_clock'](phase_only,wait_backoff=True)
assert hook['_function'](combined,'vector_schedule')[2]==hook['_function'](phase_only,'vector_schedule')[2]
class TextPath:
    def __init__(self,role):self.parent=type('Parent',(),{'name':role})();self.role=role
    def read_text(self):return combined
    def __str__(self):return 'in_memory_combined/'+self.role+'/schedule.cpp'
result['runner_active_guards']=runner['check_active_probe']([TextPath('aic'),TextPath('aiv')])
result['combined_vector_equals_phase_only']=True

# 检查插桩中不同区间的源级边界；共享边界的计时/累计本身有交叠开销。
result['segments']=raw_audit['segments']
result['instrumentation_boundary_caveat']='qk begin与前WS end、FIX begin与WS math end共用源位置，登记顺序使少量读钟/累计开销交叠；不得把区间当互斥纯硬件分解。'

generation=7
slots=[[[0]*128 for _ in range(72)] for _ in range(60)]
for core in range(72):slots[0][core][0]=generation
for core in range(24):
    values=slots[34][core];values[0]=generation
    row=[0]*32;row[0]=hook['SCHEMA'];row[1]=generation;row[21]=10
    if core<12:
        row[2]=row[3]=row[18]=row[19]=1;row[20]=10
        ticks=[150,100,120,20,20,20,10] # leaf尾ACK可超过step，仍小于body。
    else:ticks=[8,0,0,0,0,0,0]
    for i,tick in enumerate(ticks):row[4+2*i]=tick
    values[32:64]=row
assert hook['check_probe_padding'](slots,generation)
negative=[('schema',34,0,32),('epoch',34,0,33),('stage_epoch',34,0,0),('begin_epoch',0,71,0),
          ('AIV_padding',34,24,32),('other_phase_padding',33,0,32),('FFTS20_forbidden',34,0,20),
          ('probe_reserved54',34,0,54),('other_slot_tail',59,71,127),('ack_count',34,0,50),('sites',34,0,53)]
for name,phase_id,core,col in negative:
    old=slots[phase_id][core][col];slots[phase_id][core][col]=old+1
    assert not hook['check_probe_padding'](slots,generation),name
    slots[phase_id][core][col]=old
addresses={34*72*128+core*128+col for core in range(24) for col in range(32,64)}
assert len(addresses)==768
result['slot_audit']={'unique_words':len(addresses),'min_word':min(addresses),'max_word':max(addresses),
                      'writers':list(range(24)),'phase':34,'columns':[32,63],
                      'empty_owner_and_tail_ack_case':True,'negative_cases':[x[0] for x in negative]}

source={p.name:p.read_bytes() for p in (F/'source_snapshot').glob('*.py')}
reference={p.name:p.read_bytes() for p in (PAR/'qb_global_n128_evidence/source_reference').glob('*.py')}
assert len(source)==33 and source==reference
assert (H/'score_service_clock.py').read_bytes()==(F/'score_service_clock.py').read_bytes()
runner_text=(F/'run_score_diagnostic.py').read_text()
inverse=runner_text.replace(prep['HELPERS'],'',1)
for old,new in reversed(prep['REPLACEMENTS']):
    assert inverse.count(new)==1
    inverse=inverse.replace(new,old)
origin=(PAR/'attention_tick_clock_evidence/run_tick_diagnostic.py').read_text()
assert inverse==origin and ast.dump(ast.parse(inverse))==ast.dump(ast.parse(origin))
result.update(runner_inverse_text_and_ast_exact=True,frozen_sources_33_direct_equal=True,hook_copy_direct_equal=True,
              new_pypto_compilation=False,device_execution=False,ccec_clock_order_or_spill_validated=False)
(HERE/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','segments','runner_active_guards')},ensure_ascii=False,indent=2))
