#!/usr/bin/env python3
"""只读现成 Q_B global N128 / 同控668生成物；不调用编译器或设备。"""
import ast
import difflib
import json
from pathlib import Path
import re

ROOT = Path('/data/pyptouser/qinchuanyu/pto-eager')
PAR = ROOT/'vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018'
G = PAR/'qb_global_n128_evidence'
C = G/'cpu_v1/kernel'
R = PAR/'oa_pair_manual_h131072_b16_v1/build_output/decode_csa_spmd_jp7o4g2r'
HERE = Path(__file__).resolve().parent

# 仅复用四个无副作用文本函数；不 import 主目录脚本，避免生成 __pycache__。
helper = PAR/'idx_had_owner_evidence/generated_review/audit_generated.py'
nodes = [n for n in ast.parse(helper.read_text()).body if isinstance(n, ast.FunctionDef)
         and n.name in ('pto_function', 'normalize_pto', 'allocations', 'cpp_function')]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(helper), 'exec'))

def strip_names(items):
    return [{k:v for k,v in item.items() if k != 'name'} for item in items]

def resource(items):
    return {space:max(a['address']+a['size'] for a in items if a['space']==space)
            for space in sorted({a['space'] for a in items})}

def qb_bounds(text):
    lines = text.splitlines(keepends=True)
    start = next(i for i,l in enumerate(lines) if 'scf.for %qproj_round' in l)-6
    end = next(i for i in range(start+7,len(lines)) if 'pto.fence.barrier_all' in lines[i])
    return ''.join(lines[:start]), ''.join(lines[start:end]), ''.join(lines[end:])

def excerpts(path, ranges):
    lines=path.read_text().splitlines()
    result=[]
    for a,b in ranges:
        result.append(f'原物：{path}，行 {a}..{b}')
        result.extend(f'{i+1}: '+re.sub(r' loc\(.*$', '', lines[i]) for i in range(a-1,b))
    return '\n'.join(result)+'\n'

result=json.loads((G/'cpu_v1/result.json').read_text())
assert result['full_codegen_passed'] and result['passed']
reference_controls=json.loads((PAR/'oa_pair_manual_h131072_b16_v1/source_matrix/result.json').read_text())['policies']['spmd_reference']
assert result['controls']==reference_controls
texts={role:(base/'ptoas/dynamic_csa_full.pto').read_text() for role,base in [('candidate',C),('reference',R)]}
functions={role:{fn:pto_function(text,fn) for fn in ('cube_schedule','vector_schedule')} for role,text in texts.items()}
qb={role:qb_bounds(funcs['cube_schedule'])[1] for role,funcs in functions.items()}
alloc={role:{fn:allocations(text) for fn,text in funcs.items()} for role,funcs in functions.items()}
summary={
    'scope':'既有完整动态CPU生成物的只读文本/地址/局部事件审阅；不等于CCEC机器码或设备精度/性能验证',
    'controls':result['controls'], 'controls_match_reference':True,
    'cpu':{k:result[k] for k in ('passed','full_codegen_passed','single_mix','sync_start','sample_shapes_supplied','device_execution')},
    'compiled_abi_roots':len(result['compiled_abi']),
    'symbolic_axes':len(result['symbolic_dimensions']),
    'pto_aiv_normalized_exact':normalize_pto(functions['candidate']['vector_schedule'])==normalize_pto(functions['reference']['vector_schedule']),
    'global_allocation':{}, 'qb_allocation':{}, 'local_events':{},
}
for role in functions:
    summary['global_allocation'][role]={fn:{'count':len(items),'end_bytes_by_space':resource(items)} for fn,items in alloc[role].items()}
    summary['qb_allocation'][role]={'records':allocations(qb[role]),'end_bytes_by_space':resource(allocations(qb[role]))}
summary['aiv_allocations_exact']=strip_names(alloc['candidate']['vector_schedule'])==strip_names(alloc['reference']['vector_schedule'])
outside={role:qb_bounds(funcs['cube_schedule'])[0]+'\n// QB body omitted\n'+qb_bounds(funcs['cube_schedule'])[2] for role,funcs in functions.items()}
summary['aic_outside_qb_normalized_exact']=normalize_pto(outside['candidate'])==normalize_pto(outside['reference'])
summary['aic_outside_qb_allocations_exact']=strip_names(allocations(outside['candidate']))==strip_names(allocations(outside['reference']))
def erase_alloc_addresses(text):
    return re.sub(r'(pto\.alloc_tile addr = )\[[^]]+\]', r'\1[ADDRESS]', normalize_pto(text))
summary['aic_outside_qb_only_alloc_addresses_differ']=erase_alloc_addresses(outside['candidate'])==erase_alloc_addresses(outside['reference'])
summary['non_qb_allocation_differences']=[]
for a,b in zip(allocations(outside['reference']),allocations(outside['candidate'])):
    if strip_names([a])!=strip_names([b]):
        summary['non_qb_allocation_differences'].append({'reference':a,'candidate':b})
for fn in ('cube_schedule','vector_schedule'):
    events={}
    for role,base in [('candidate',C),('reference',R)]:
        raw_cpp=(base/'ptoas/dynamic_csa_full.cpp').read_text()
        text=cpp_function(raw_cpp.replace('AICORE void '+fn+'(', 'static __aicore__ void '+fn+'('),fn)
        events[role]=re.findall(r'(?:set_flag|wait_flag|pipe_barrier)\([^;\n]*\);',text)
    summary['local_events'][fn]={'candidate_count':len(events['candidate']),'reference_count':len(events['reference']), 'flat_sequence_exact':events['candidate']==events['reference']}
    if events['candidate']!=events['reference']:
        (HERE/(fn+'_events.diff')).write_text('\n'.join(difflib.unified_diff(events['reference'],events['candidate'],fromfile='reference',tofile='candidate'))+'\n')
summary['qb_pto_math_operations']={role:re.findall(r'pto\.tmatmul(?:\.acc)?',text) for role,text in qb.items()}
summary['qb_right_addresses']={role:sorted({a['address'] for a in allocations(text) if a['space']=='right'}) for role,text in qb.items()}
summary['qb_left_addresses']={role:sorted({a['address'] for a in allocations(text) if a['space']=='left'}) for role,text in qb.items()}
summary['qb_acc_addresses']={role:sorted({a['address'] for a in allocations(text) if a['space']=='acc'}) for role,text in qb.items()}
summary['physical_double_right']=len(summary['qb_right_addresses']['candidate'])>1
summary['manual_cpp_review']={
    'file':str(C/'kernels/aic/cube_schedule.cpp'),
    'qb_lines':[10437,10894],
    'single_right_reuse_wait_lines':[10539,10694,10718,10804],
    'mma_lines':[10562,10709,10732,10819],
    'store_then_drain_publish_lines':[10858,10894],
    'non_qb_new_pipe_mte1_barriers_raw_cpp_lines':[11045,11395,11517],
    'non_qb_new_barrier_stages':['compact Hadamard Right TMOV','main compressor first K512 Left extract','main compressor second K512 Left extract'],
    'caveat':'扁平事件对照统计是C++静态语句，不是运行次数或耗时；局部wait_flag不用于推断Scalar真实等待时长。',
}
summary['tail_examples']={str(t):[min(128,t-t0) for t0 in range(0,(t+127)//128*128,128)] for t in (1,96,128,129,192,255,256,257)}
for role,text in qb.items():
    (HERE/(role+'_qb_pto_excerpt.txt')).write_text(re.sub(r' loc\("[^"\n]+"(?::\d+:\d+)?\)','',text))
cpp_path=C/'kernels/aic/cube_schedule.cpp'
lines=cpp_path.read_text().splitlines()
interesting=re.compile(r'\b(for|TASSIGN|TLOAD|TMOV|TMATMUL|TMATMUL_ACC|TSTORE|set_flag|wait_flag|pipe_barrier|dcci|dsb|TNOTIFY)\b')
(HERE/'candidate_qb_cpp_operations.txt').write_text(str(cpp_path)+'\n'+ '\n'.join(f'{i+1}: {lines[i].strip()}' for i in range(10436,10894) if interesting.search(lines[i]))+'\n')
(HERE/'candidate_qb_acquire_pto_excerpt.txt').write_text(excerpts(C/'ptoas/dynamic_csa_full.pto',[(9439,9451)]))
(HERE/'candidate_non_qb_cpp_barriers_excerpt.txt').write_text(excerpts(C/'ptoas/dynamic_csa_full.cpp',[(11032,11049),(11387,11401),(11511,11523)]))
if not summary['aic_outside_qb_normalized_exact']:
    diff=list(difflib.unified_diff(normalize_pto(outside['reference']).splitlines(),normalize_pto(outside['candidate']).splitlines(),fromfile='reference',tofile='candidate'))
    (HERE/'aic_outside_qb.diff').write_text('\n'.join(diff)+'\n')
summary['artifacts']=[{'path':str(p),'size_bytes':p.stat().st_size} for p in [G/'cpu_v1/result.json',C/'compiled_meta.json',C/'ptoas/dynamic_csa_full.pto',C/'ptoas/dynamic_csa_full.cpp',C/'kernels/aic/cube_schedule.cpp',C/'kernels/aiv/vector_schedule.cpp',R/'ptoas/dynamic_csa_full.pto',R/'ptoas/dynamic_csa_full.cpp']]
(HERE/'review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('qb_allocation','artifacts')},ensure_ascii=False,indent=2))
