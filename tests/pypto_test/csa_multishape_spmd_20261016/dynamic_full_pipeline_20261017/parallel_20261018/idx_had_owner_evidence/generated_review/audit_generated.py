#!/usr/bin/env python3
"""只读主会话已经生成的 HAD owner 候选与同控制正式参考。

按函数作用域归一化SSA/常量名，避免不同函数同名SSA污染比较。只允许剔除
KV发布后到HAD之前的一个等待块，其后的T*64计算与全部数学/控制必须保留。
不启动CPU后端或NPU；大PTO/C++只登记路径/字节数，摘录限于同步邻域。
"""
import difflib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/pyptouser/qinchuanyu/pto-eager')
PAR = ROOT/'vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018'
CANDIDATE = PAR/'idx_had_owner_evidence/cpu_v1/kernel'
REFERENCE = PAR/'oa_pair_manual_h131072_b16_v1/build_output/decode_csa_spmd_jp7o4g2r'


def pto_function(text, name):
    start = text.index('func.func @'+name+'(')
    next_function = re.search(r'func.func @', text[start+1:])
    end = start+1+next_function.start() if next_function else len(text)
    return text[start:end]


def normalize_pto(text):
    # 只能传入单函数，常量名/数字SSA在不同函数可重复，不能跨函数收集。
    assert len(re.findall(r'func.func @', text)) == 1
    text = re.sub(r' loc\("[^"\n]+"(?::\d+:\d+)?\)', '', text)
    constants, lines, names = {}, [], {}
    for line in text.splitlines():
        match = re.fullmatch(r'\s*(%[\w.]+) = arith.constant (.+)', line)
        if match:
            constants[match[1]] = '['+match[2]+']'
        else:
            lines.append(line)
    def rename(match):
        name = match.group()
        if name in constants:
            return constants[name]
        if name not in names:
            names[name] = '%v'+str(len(names))
        return names[name]
    return re.sub(r'%[\w.]+', rename, '\n'.join(lines))


def wait_bounds(text):
    had_alloc = text.rfind('\n', 0, text.index(' = pto.alloc_tile', text.index('%qh_hadamard_inline')))+1
    start = text.index('\n', text.rfind('pto.comm.tnotify(', 0, had_alloc))+1
    # 最后一个两空格闭括号就是原/新等待if；保留其后的bs_heads=T*64计算。
    end = text.rfind('\n  }\n', start, had_alloc)+len('\n  }\n')
    assert start < end < had_alloc
    assert text[start:end].count('pto.comm.twait(') == 1
    assert 'arith.muli %arg132, %c64_index' in text[end:had_alloc]
    return start, end


def allocations(text):
    pattern = re.compile(r'%([\w]+) = pto.alloc_tile addr = %c(\d+)_i64.*?!pto.tile_buf<'
                         r'loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)')
    widths = {'i8':1,'i16':2,'i32':4,'ui8':1,'ui16':2,'ui32':4,'f16':2,'bf16':2,'f32':4}
    result = []
    for match in pattern.finditer(text):
        name, address, space, dtype, rows, cols = match.groups()
        result.append(dict(name=name,address=int(address),space=space,dtype=dtype,
                           rows=int(rows),cols=int(cols),size=int(rows)*int(cols)*widths[dtype]))
    return result


def cpp_function(text, name):
    start = text.index('static __aicore__ void '+name+'(')
    brace = text.index('{',start)
    level = 1
    for i in range(brace+1,len(text)):
        if text[i] == '{': level += 1
        elif text[i] == '}': level -= 1
        if level == 0:
            return text[start:i+1]
    raise ValueError('C++函数括号不完整')


def excerpt(path, begin, end, name):
    text = path.read_text()
    first, last = text[:begin].count('\n')+1, text[:end].count('\n')+1
    lines = text.splitlines()
    (HERE/name).write_text(f'原始路径：{path}\n原始行：{first}–{last}\n\n'+
                          '\n'.join(f'{i}: {lines[i-1]}' for i in range(first,last+1))+'\n')
    return dict(path=str(path),first_line=first,last_line=last,excerpt=name)


def main():
    result_path = CANDIDATE.parent/'result.json'
    result = json.loads(result_path.read_text())
    matrix_path = REFERENCE.parents[1]/'source_matrix/result.json'
    matrix = json.loads(matrix_path.read_text())
    assert result['passed'] and result['single_mix'] and result['full_codegen_passed']
    assert not result['device_execution'] and not result['sample_shapes_supplied']
    assert result['controls'] == matrix['policies']['spmd_reference']
    assert matrix['kernel_configs']['spmd_reference'] == str(REFERENCE/'kernel_config.py')
    assert len(result['compiled_abi']) == 130 and len(result['symbolic_dimensions']) == 58
    # 基线role和控制相同还不够：设备参考的33份源必须与本候选668参考逐字一致。
    source_reference = Path(matrix['reference_source'])
    source_equal = {p.name:p.read_bytes()==(source_reference/p.name).read_bytes()
                    for p in (HERE.parent/'source_reference').glob('*.py')}
    assert len(source_equal)==33 and all(source_equal.values())
    pto_paths = [d/'ptoas/dynamic_csa_full.pto' for d in (REFERENCE,CANDIDATE)]
    pto_text = [p.read_text() for p in pto_paths]
    functions, allocation_reports, event_reports = {}, {}, {}
    fields = ('address','space','dtype','rows','cols','size')
    for role,name in [('aic','cube_schedule'),('aiv','vector_schedule')]:
        bodies = [pto_function(t,name) for t in pto_text]
        alloc = [allocations(t) for t in bodies]
        assert len(alloc[0])==len(alloc[1])
        assert all(tuple(a[k] for k in fields)==tuple(b[k] for k in fields) for a,b in zip(*alloc))
        allocation_reports[role] = dict(count=len(alloc[1]),same_address_shape_dtype=True,
                                        upper_bounds={space:max(a['address']+a['size'] for a in alloc[1] if a['space']==space)
                                                      for space in sorted({a['space'] for a in alloc[1]})})
        if role=='aic':
            spans = [wait_bounds(t) for t in bodies]
            bodies = [t[:a]+t[b:] for t,(a,b) in zip(bodies,spans)]
        norm = [normalize_pto(t) for t in bodies]
        equal = norm[0]==norm[1]
        (HERE/f'{role}_normalized_pto.diff').write_text(''.join(difflib.unified_diff(
            norm[0].splitlines(True),norm[1].splitlines(True),fromfile='668_device_reference',tofile='candidate_cpu')))
        assert equal
        functions[role] = dict(pto_equal=equal,omitted='only RoPE→HAD wait block' if role=='aic' else 'nothing')
        cpp = [cpp_function((d/f'kernels/{role}/{name}.cpp').read_text(),name) for d in (REFERENCE,CANDIDATE)]
        event_pattern = r'^\s*(?:set_flag|wait_flag|pipe_barrier)\([^;]+;'
        events = [[x.strip() for x in re.findall(event_pattern,t,re.M)] for t in cpp]
        event_reports[role] = dict(reference_count=len(events[0]),candidate_count=len(events[1]),
                                  identical_flat_local_sequence_and_ids=events[0]==events[1])
        (HERE/f'{role}_local_events.diff').write_text(''.join(difflib.unified_diff(
            [x+'\n' for x in events[0]],[x+'\n' for x in events[1]],fromfile='668_reference',tofile='candidate')))
        assert events[0]==events[1]
    # 直接核对PTO owner表达式及业务维护相对HAD读写的顺序。
    pto = pto_text[1]
    a,b = wait_bounds(pto)
    wait = pto[a:b]
    required = ['arith.cmpi slt, %worker__ssa_v0, %arg132',
                'arith.divsi %worker__ssa_v0, %c8_index','%c16_index',
                'arith.remsi','%c48_index','%c24_index','%c156672_index']
    assert all(x in wait for x in required)
    assert wait.index('pto.comm.twait') < wait.index('pto.fence.barrier_all') < wait.index('pto.cmo.cacheinvalid')
    assert wait.count('pto.fence.barrier_all')==2 and wait.count('pto.cmo.cacheinvalid')==1
    had_end = pto.index('\n',pto.index('pto.comm.tnotify(',b))+1
    had = pto[b:had_end]
    assert had.index('pto.tload') < had.index('pto.tmatmul') < had.index('pto.tstore') < had.index('pto.comm.tnotify')
    assert '%c165888_index' in had and had.count('pto.comm.tnotify')==1
    pto_excerpt = excerpt(pto_paths[1],a,had_end,'wait_had.pto.txt')
    # 真实C++常量已经折叠16次循环，保持worker/T和物理核偏移；不拿PTO代替C++。
    cpp_path = CANDIDATE/'kernels/aic/cube_schedule.cpp'
    full_cpp = cpp_path.read_text()
    cpp = cpp_function(full_cpp,'cube_schedule')
    signature = cpp[:cpp.index('{')]
    parameter_names = re.findall(r'\b(v\d+)\b',signature)
    wrapper_call = next(line.strip() for line in full_cpp.splitlines() if line.strip().startswith('cube_schedule('))
    argument_names = [x.strip() for x in wrapper_call[len('cube_schedule('):-2].split(',')]
    assert len(parameter_names)==len(argument_names)
    wrapper_binding = dict(zip(parameter_names,argument_names))
    assert wrapper_binding['v133']=='CSA_FULL_T'
    assert wrapper_binding['v189']=='__pypto_spmd_block_idx'
    assert 'CSA_FULL_T = static_cast<int64_t>(x_fp32__ssa_v0_tensor->shapes[0])' in full_cpp
    constants = dict(re.findall(r'const int64_t (v\d+) = (-?\d+);',cpp))
    expected = {'v245':24,'v247':1,'v249':64,'v250':16,'v258':128,'v260':8,'v267':0,
                'v272':48,'v285':156672,'v286':165888}
    assert all(int(constants[k])==v for k,v in expected.items())
    first_had = cpp.index('// pto: %qh_hadamard_inline')
    begin_cpp = cpp.index('\n',cpp.rfind('pto::comm::TNOTIFY(',0,first_had))+1
    end_cpp = cpp.index('\n',cpp.index('pto::comm::TNOTIFY(',first_had))+1
    target = cpp[begin_cpp:end_cpp]
    assert 'if (v926 < v133)' in target
    assert 'v4666 = v267; v4666 < v250; v4666 += v247' in target
    assert 'v926 / v260' in target and '% v272' in target
    pre_load = target[:target.index('TLOAD(')]
    assert pre_load.index('pto::comm::TWAIT(') < pre_load.index('dsb(DSB_DDR)') < pre_load.index('dcci(')
    assert pre_load.count('dsb(DSB_DDR)')==2 and pre_load.count('dcci(')==1
    after_store = target[target.rindex('TSTORE('):]
    assert after_store.index('wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0)') < after_store.index('dcci(') < after_store.index('pipe_barrier(PIPE_ALL)') < after_store.index('pto::comm::TNOTIFY(')
    offset = full_cpp.index(cpp)
    cpp_excerpt = excerpt(cpp_path,offset+begin_cpp,offset+end_cpp,'wait_had.cpp.txt')
    constant_lines = [line for line in cpp.splitlines() if any(re.search(r'\b'+key+r'\b',line) for key in expected)
                      and line.strip().startswith('const int64_t')]
    (HERE/'cpp_constants.txt').write_text('\n'.join(constant_lines)+'\n')
    assert 'CSA_FULL_T' in full_cpp and 'v926 = (int64_t) v189' in cpp
    report = dict(status='PASS',scope='只读主会话一次CPU产物；本审计未编译/未NPU',
                  reference_kernel=str(REFERENCE),candidate_kernel=str(CANDIDATE),
                  cpu_result=str(result_path),reference_role_controls_evidence=str(matrix_path),
                  controls=result['controls'],source_reference_33_byte_equal=True,
                  roots=130,dynamic_axes=58,single_mix=True,sync_start=True,
                  normalized_pto=functions,allocations=allocation_reports,cpp_local_events=event_reports,
                  generated_wait=dict(worker_less_than_actual_T=True,owner_count=16,
                                      wrapper_T='x_fp32.shape[0] → CSA_FULL_T → v133',
                                      wrapper_worker='__pypto_spmd_block_idx → v189 → v926',
                                      owner='((worker//8)*16+ordinal)%48',core='24+owner',
                                      signal_words='17*72*128+core*128',had_publish_words='18*72*128+worker*128',
                                      maintenance_after_wait_before_HAD_TLOAD=True,
                                      original_HAD_TSTORE_FIX_completion_before_publish=True),
                  excerpts=[pto_excerpt,cpp_excerpt],
                  normalization='按函数单独处理，去loc/SSA名、内联同值同类型常量；保留数据类型/地址/运算/控制。AIC只排除一个wait块，保留其后T*64。',
                  audit_correction='初查跨module的SSA常量表使不同函数同名SSA串扰，已按函数归一化。脚本首次运行缺ui32宽度而KeyError，补为4字节后只重读原件，未编译/改产物。',
                  limitations=['参考设备辅助头启用wait-backoff，CPU默认TWAIT；不宣称辅助头/机器码全文等价。',
                               '局部事件列表相同不表示新条件分支执行次数相同；worker<T及owner集合改变是本候选目的。',
                               '后端通过与静态顺序证据不代替A3设备正确性、graph/保护区或完整性能验证。'])
    (HERE/'generated_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    files = [result_path,matrix_path]+pto_paths
    files += [d/f'kernels/{role}/{name}.cpp' for d in (REFERENCE,CANDIDATE)
              for role,name in [('aic','cube_schedule'),('aiv','vector_schedule')]]
    files += [d/'kernel_config.py' for d in (REFERENCE,CANDIDATE)]
    inventory = [dict(path=str(p),bytes=p.stat().st_size,disposition='只读原件；不复制大文件') for p in files]
    (HERE/'INPUT_ARTIFACTS.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',normalized_pto=functions,allocations=allocation_reports,events=event_reports),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
