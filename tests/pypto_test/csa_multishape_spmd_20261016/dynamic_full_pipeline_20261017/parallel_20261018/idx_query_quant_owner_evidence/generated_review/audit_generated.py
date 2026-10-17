#!/usr/bin/env python3
"""单次完整CPU后的只读产物审阅；不触发编译、NPU或源码改写。

输入为本目录父级cpu_v1和已完成同控668设备reference。按函数作用域
比较PTO，只剔除唯一HAD→quant获取块；量化数学及后继全QUANT获取保留。
输出只写本审阅目录，小摘录附真实路径/行号，大文件只登记路径和大小。
"""
import difflib
import json
from pathlib import Path
import re

from text_audit_tools import pto_function, normalize_pto, allocations, cpp_function, excerpt

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
CANDIDATE = EVIDENCE/'cpu_v1/kernel'
REFERENCE = EVIDENCE.parent/'oa_pair_manual_h131072_b16_v1/build_output/decode_csa_spmd_jp7o4g2r'


def wait_bounds(text):
    first = text.index(' = pto.alloc_tile',text.index('%qh_full_f32_inline'))
    start = text.index('\n',text.rfind('pto.comm.tnotify(',0,first))+1
    end = text.rfind('\n  }\n',start,first)+len('\n  }\n')
    assert start<end<first
    assert text[start:end].count('pto.comm.twait(')==1
    assert 'arith.muli %arg132, %c64_index' in text[end:first]
    return start,end


def main():
    cpu_path = EVIDENCE/'cpu_v1/result.json'
    result = json.loads(cpu_path.read_text())
    matrix_path = REFERENCE.parents[1]/'source_matrix/result.json'
    matrix = json.loads(matrix_path.read_text())
    assert result['passed'] and result['full_codegen_passed'] and result['single_mix'] and result['sync_start']
    assert not result['device_execution'] and not result['sample_shapes_supplied']
    assert len(result['compiled_abi'])==130 and len(result['symbolic_dimensions'])==58
    assert result['controls']==matrix['policies']['spmd_reference']
    assert matrix['kernel_configs']['spmd_reference']==str(REFERENCE/'kernel_config.py')
    ref_source = Path(matrix['reference_source'])
    source_comparisons = {f.name:f.read_bytes()==(ref_source/f.name).read_bytes()
                          for f in (EVIDENCE/'source_reference').glob('*.py')}
    assert len(source_comparisons)==33 and all(source_comparisons.values())
    assert (EVIDENCE/'source_reference/cube_schedule.py').read_bytes()==(EVIDENCE/'source_candidate/cube_schedule.py').read_bytes()
    pto_paths = [d/'ptoas/dynamic_csa_full.pto' for d in (REFERENCE,CANDIDATE)]
    pto = [p.read_text() for p in pto_paths]
    reports = {}
    for role,name in [('aic','cube_schedule'),('aiv','vector_schedule')]:
        functions = [pto_function(t,name) for t in pto]
        allocs = [allocations(t) for t in functions]
        fields = ['address','space','dtype','rows','cols','size']
        assert len(allocs[0])==len(allocs[1])
        assert all([a[k] for k in fields]==[b[k] for k in fields] for a,b in zip(*allocs))
        if role=='aiv':
            spans = [wait_bounds(t) for t in functions]
            functions = [t[:a]+t[b:] for t,(a,b) in zip(functions,spans)]
        normalized = [normalize_pto(t) for t in functions]
        (HERE/f'{role}_normalized_pto.diff').write_text(''.join(difflib.unified_diff(
            normalized[0].splitlines(True),normalized[1].splitlines(True),fromfile='668_reference',tofile='single_owner_cpu')))
        assert normalized[0]==normalized[1]
        cpp = [cpp_function((d/f'kernels/{role}/{name}.cpp').read_text(),name) for d in (REFERENCE,CANDIDATE)]
        events = [[x.strip() for x in re.findall(r'^\s*(?:set_flag|wait_flag|pipe_barrier)\([^;]+;',t,re.M)] for t in cpp]
        (HERE/f'{role}_local_events.diff').write_text(''.join(difflib.unified_diff(
            [x+'\n' for x in events[0]],[x+'\n' for x in events[1]],fromfile='668_reference',tofile='single_owner_cpu')))
        assert events[0]==events[1]
        reports[role] = dict(normalized_pto_equal=True,omitted='only HAD→quant wait block' if role=='aiv' else 'nothing',
                             allocation_count=len(allocs[1]),allocation_address_shape_dtype_equal=True,
                             upper_bounds={s:max(a['address']+a['size'] for a in allocs[1] if a['space']==s)
                                           for s in sorted({a['space'] for a in allocs[1]})},
                             local_event_count=len(events[1]),local_event_sequence_and_ids_equal=True)
    # 原始PTO中保留完整量化读/算/写以及其后的发布、weights及全QUANT获取。
    a,b = wait_bounds(pto[1])
    wait = pto[1][a:b]
    assert 'arith.cmpi slt, %452, %arg132' in wait
    assert 'arith.remsi %452, %c24_index' in wait
    assert '%c165888_index' in wait and '%c128_index' in wait
    assert '%c24_index' in wait and wait.count('arith.addi')==1  # 仅加phase基址，不加AIV偏移。
    assert wait.count('pto.comm.twait')==1 and 'scf.for' not in wait
    assert wait.index('pto.comm.twait')<wait.index('pto.fence.barrier_all')<wait.index('pto.cmo.cacheinvalid')
    assert wait.count('pto.fence.barrier_all')==2 and wait.count('pto.cmo.cacheinvalid')==1
    publish = pto[1].index('pto.comm.tnotify(',b)
    publish_end = pto[1].index('\n',publish)+1
    quant = pto[1][b:publish_end]
    assert quant.index('pto.tload')<quant.index('pto.tstore')<quant.index('pto.comm.tnotify')
    assert quant.count('pto.tstore')==2 and '%c175104_index' in quant
    assert quant.count('pto.fence.barrier_all')==2 and quant.count('pto.cmo.cacheinvalid')==1
    # 下一个weights获取之后仍为min(T,48)个AIV的全量quant获取。
    successor_start = pto[1].index('pto.comm.twait(',publish_end)
    successor_wait = pto[1].index('pto.comm.twait(',successor_start+1)
    successor_end = pto[1].index('\n  }\n',successor_wait)+len('\n  }\n')
    successor = pto[1][publish_end:successor_end]
    assert successor.count('pto.comm.twait')==2 and '%c175104_index' in successor
    assert 'arith.subi %arg132, %c0_index' in successor and 'arith.minsi %1861, %c48_index' in successor
    assert 'arith.addi %1864, %c24_index' in successor
    pto_excerpt = excerpt(pto_paths[1],a,successor_end,'wait_quant_successor.pto.txt')
    # C++带label的真实CFG：true进入472等待，false直接473进入后续量化，非inline猜测。
    cpp_path = CANDIDATE/'kernels/aiv/vector_schedule.cpp'
    full_cpp = cpp_path.read_text()
    cpp = cpp_function(full_cpp,'vector_schedule')
    first_quant = cpp.index('// pto: %qh_full_f32_inline')
    cpp_start = cpp.index('\n',cpp.rfind('pto::comm::TNOTIFY(',0,first_quant))+1
    cpp_publish = cpp.index('pto::comm::TNOTIFY(',first_quant)
    cpp_publish_end = cpp.index('\n',cpp_publish)+1
    body = cpp[cpp_start:cpp_publish_end]
    constants = dict(re.findall(r'const int64_t (v\d+) = (-?\d+);',cpp))
    expected = {'v338':24,'v349':128,'v357':0,'v421':165888}
    assert all(int(constants[k])==v for k,v in expected.items())
    assert 'v5667 = v1515 < v133' in body and 'goto label472' in body and 'goto label473' in body
    address = next(l for l in body.splitlines() if l.strip().startswith('v5668 ='))
    assert 'v1515 % v338' in address and '* (uint64_t) v349' in address and '+ (uint64_t) v421' in address
    assert address.count('+')==1
    first_load = body.index('TLOAD(')
    acquire = body[:first_load]
    assert acquire.index('pto::comm::TWAIT')<acquire.index('dsb(DSB_DDR)')<acquire.index('dcci(')
    assert acquire.count('dsb(DSB_DDR)')==2 and acquire.count('dcci(')==1
    after_store = body[body.rindex('TSTORE('):]
    assert after_store.index('pipe_barrier(PIPE_ALL)')<after_store.index('dcci(')<after_store.index('pto::comm::TNOTIFY')
    signature = cpp[:cpp.index('{')]
    params = re.findall(r'\b(v\d+)\b',signature)
    call = next(l.strip() for l in full_cpp.splitlines() if l.strip().startswith('vector_schedule('))
    args = [x.strip() for x in call[len('vector_schedule('):-2].split(',')]
    assert len(params)==len(args)
    assert dict(zip(params,args))['v133']=='CSA_FULL_T'
    assert 'CSA_FULL_T = static_cast<int64_t>(x_fp32__ssa_v0_tensor->shapes[0])' in full_cpp
    offset = full_cpp.index(cpp)
    cpp_excerpt = excerpt(cpp_path,offset+cpp_start,offset+cpp_publish_end,'wait_quant.cpp.txt')
    (HERE/'cpp_constants.txt').write_text('\n'.join(l for l in cpp.splitlines()
        if l.strip().startswith('const int64_t') and any(re.search(r'\b'+k+r'\b',l) for k in expected))+'\n')
    report = dict(status='PASS',cpu_compilations=1,device_execution=False,source_frozen_unchanged=True,
                  cpu_result=str(cpu_path),reference_kernel=str(REFERENCE),candidate_kernel=str(CANDIDATE),
                  reference_role_controls_evidence=str(matrix_path),same_33_reference_source=True,
                  roots=130,dynamic_axes=58,single_mix=True,sync_start=True,controls=result['controls'],
                  roles=reports,owner='logical AIV worker % 24; physical AIC core has no +24',
                  actual_T='x_fp32.shape[0] → CSA_FULL_T → v133',
                  signal_word_offset='18*72*128 + (worker%24)*128',
                  acquiring_worker_condition='worker<T',
                  business_maintenance_after_wait_before_quant_TLOAD=True,
                  two_quant_TSTORE_and_original_publish_retained=True,
                  successor_full_quant_wait_retained='min(T,48) AIV owners at 24+owner, phase19',
                  excerpts=[pto_excerpt,cpp_excerpt],
                  normalization='函数作用域内仅去loc/SSA名、内联同值同类型常量。AIC全文；AIV只排除唯一wait块，保留T*64与其后完整数学/获取。',
                  limitations=['CPU默认TWAIT，参考设备wait-backoff辅助头不同；不称全文机器码等价。',
                               '局部事件顺序相同不是运行时次数相同；新条件与单owner正是实验变量。',
                               '未NPU；不得将后端通过替代精度、graph、保护区或性能结论。'])
    (HERE/'generated_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    paths = [cpu_path,matrix_path]+pto_paths+[d/f'kernels/{r}/{n}.cpp' for d in (REFERENCE,CANDIDATE)
                                           for r,n in [('aic','cube_schedule'),('aiv','vector_schedule')]]
    (HERE/'INPUT_ARTIFACTS.json').write_text(json.dumps([dict(path=str(p),bytes=p.stat().st_size,
        disposition='只读原件；大生成物不复制') for p in paths],ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',roles=reports),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
