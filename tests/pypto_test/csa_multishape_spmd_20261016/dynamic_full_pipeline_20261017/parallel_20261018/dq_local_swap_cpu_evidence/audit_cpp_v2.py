"""只读对照同controls既有9d产物与v2实际C++，不再编译。

以DQ首个输入Tile到其后的原phase发布分段，逐序比较段外同步；单独列
CPP直接barrier与ISA内TXORS展开，避免把模板调用数量当硬件指令数。
"""
from pathlib import Path
import collections
import difflib
import json
import re
import audit_ir

OUT=Path(__file__).resolve().parent
BASE=audit_ir.BASE/'cpu_v1/kernel'
NEW=OUT/'cpu_v2/kernel'

def function(s,name):
    b=s.index('AICORE void '+name+'(')
    e=s.find('AICORE void ',b+20)
    return s[b:e if e>=0 else len(s)]

def normal(s):
    s=re.sub(r'//[^\n]*','',s)
    names={}
    s=re.sub(r'\bv\d+\b|\blabel\d+\b',lambda m:names.setdefault(m[0],'ssa_'+str(len(names))),s)
    return re.sub(r'\s+','',s)

def sync(s):
    return re.findall(r'(?m)^\s*((?:pipe_barrier|wait_flag|set_flag|wait_flag_dev|ffts_cross_core_sync|dsb|dcci|pto::comm::TNOTIFY|pto::comm::TWAIT)(?:<[^;]+?>)?\([^;]+\));',s)

def calls(s):
    return collections.Counter(re.findall(r'(?m)^\s*(?:pto::comm::)?(T[A-Z]\w*|pipe_barrier|wait_flag|set_flag|wait_flag_dev|ffts_cross_core_sync|dsb|dcci)(?:<[^\n]+>)?\(',s))

def slices(s):
    a=s.index('// pto: %qr_scale_dq_t_')
    # 原DQ全部写回后才发布；切至发布前，包含原pipeline drain。
    b=s.index('pto::comm::TNOTIFY(',a)
    i=s.index('// pto: %q_flat_gather_tmp_',a)
    return s[:a],s[a:b],s[b:],s[a:i],s[i:b]

def load(k):
    path=k/'ptoas/dynamic_csa_full.cpp'
    s=path.read_text()
    return path,function(s,'cube_schedule'),function(s,'vector_schedule')

old_path,oldcube,oldvec=load(BASE)
new_path,newcube,newvec=load(NEW)
o=slices(oldvec); n=slices(newvec)
checks={
 'aic_cpp_normalized_equal':normal(oldcube)==normal(newcube),
 'aiv_sync_before_dq_order_equal':normal('\n'.join(sync(o[0])))==normal('\n'.join(sync(n[0]))),
 'aiv_sync_after_dq_order_equal':normal('\n'.join(sync(o[2])))==normal('\n'.join(sync(n[2]))),
 'dq_head_tail_and_publish_prefix_cpp_normalized_equal':normal(o[4])==normal(n[4]),
}
# 参数SSA不用于判断物理地址；地址及生存期核对已由ir_audit_v2覆盖。
for label,old,new in [('all_cpp_calls',calls(oldvec),calls(newvec)),('dq_cpp_calls',calls(o[1]),calls(n[1])),('prep_cpp_calls',calls(o[3]),calls(n[3]))]:
    diff={k:new[k]-old[k] for k in old.keys()|new.keys() if new[k]!=old[k]}
    checks[label+'_delta']=diff
sync_old=collections.Counter(sync(o[3])); sync_new=collections.Counter(sync(n[3]))
sync_delta={k:sync_new[k]-sync_old[k] for k in sync_old.keys()|sync_new.keys() if sync_old[k]!=sync_new[k]}
# 事件分配变化必须如实输出，不能要求不成立的“只改局部事件”。
def remove_sync(s):
    return re.sub(r'(?m)^\s*(?:pipe_barrier|wait_flag|set_flag|wait_flag_dev|ffts_cross_core_sync|dsb|dcci|pto::comm::TNOTIFY|pto::comm::TWAIT)(?:<[^;]+?>)?\([^;]+\);','',s)
checks['dq_math_after_removing_sync_normalized_equal']=normal(remove_sync(o[4]))==normal(remove_sync(n[4]))
checks['all_stages_after_dq_math_without_sync_normalized_equal']=normal(remove_sync(o[2]))==normal(remove_sync(n[2]))
assert checks['all_cpp_calls_delta']['TLOAD']==-1
assert checks['all_cpp_calls_delta']['TCVT']==-3
assert checks['all_cpp_calls_delta']['TXORS']==1
assert checks['aic_cpp_normalized_equal']
assert checks['dq_math_after_removing_sync_normalized_equal']
assert checks['all_stages_after_dq_math_without_sync_normalized_equal']
events={}
for label,index in [('before_dq',0),('dq_math_tail',4),('after_dq',2)]:
    old_events=[re.sub(r'\bv\d+\b','SSA',x) for x in sync(o[index])]
    new_events=[re.sub(r'\bv\d+\b','SSA',x) for x in sync(n[index])]
    diffs=[]
    for tag,i,j,k,l in difflib.SequenceMatcher(a=old_events,b=new_events,autojunk=False).get_opcodes():
        if tag!='equal':
            diffs.append({'kind':tag,'baseline_ordinal':[i,j],'candidate_ordinal':[k,l],'baseline':old_events[i:j],'candidate':new_events[k:l]})
    events[label]={'baseline_count':len(old_events),'candidate_count':len(new_events),'differences':diffs}
(OUT/'event_changes_v2.json').write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n')
old_barriers=collections.Counter(re.findall(r'pipe_barrier\((PIPE_\w+)\)',oldvec))
new_barriers=collections.Counter(re.findall(r'pipe_barrier\((PIPE_\w+)\)',newvec))
barrier_delta={k:new_barriers[k]-old_barriers[k] for k in old_barriers.keys()|new_barriers.keys()}
(OUT/'baseline_prep.cpp.txt').write_text(o[3])
(OUT/'candidate_v2_prep.cpp.txt').write_text(n[3])
(OUT/'prep_generated.diff').write_text(''.join(difflib.unified_diff(o[3].splitlines(True),n[3].splitlines(True),fromfile=str(old_path),tofile=str(new_path))))
obj={'audit_completed':True,'backend_passed':True,'candidate_adopted':False,'baseline':str(old_path),'candidate':str(new_path),'checks':checks,'prep_direct_sync_delta':sync_delta,'whole_direct_barrier_delta':barrier_delta,'event_changes':'event_changes_v2.json','txors_isa_vector_stages':6,'txors_isa_internal_pipe_v_barriers':5,'integer_casts_v2':2,'TCI_b32_512_source_intrinsic_calls':21,'TCI_b16_512_source_intrinsic_calls':34,'TCI_b32_internal_PIPE_V':8,'TCI_b16_internal_PIPE_V':10,'TCI_scratch_baseline_bytes':768,'TCI_scratch_candidate_bytes':1792,'index_TLOAD_saved_per_full_chunk_bytes':2048,'scope':'已生成C++静态调用/顺序和ISA实现；未执行机器码，无周期估计','limitation':'TCI从INT32改为INT16，scratch和内部路径不同；不能仅CPP的barrier差推算全部硬件barrier或性能。'}
(OUT/'cpp_audit_v2.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(obj,ensure_ascii=False,indent=2))
