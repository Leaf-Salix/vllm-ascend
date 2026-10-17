#!/usr/bin/env python3
"""只读已有完整产物；核对 PV 分配、提交顺序及同步，不调用编译器。"""

import collections
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
REFERENCE = json.loads((OUT / 'baseline_artifact_reference.json').read_text())
BASE = Path(REFERENCE['baseline_kernel'])
NEW = OUT / 'cpu_v1/kernel'


def difference(a, b):
    return {'removed': dict(collections.Counter(a) - collections.Counter(b)),
            'added': dict(collections.Counter(b) - collections.Counter(a))}


def cpp_function(text, name):
    begin = text.index('AICORE void ' + name + '(')
    end = text.find('AICORE void ', begin + 20)
    return text[begin:end if end >= 0 else len(text)]


def normalize_cpp(text):
    text = re.sub(r'//[^\n]*', '', text)
    names = {}
    def replace(match):
        name = match[0]
        return names.setdefault(name, 'ssa_' + str(len(names)))
    text = re.sub(r'\bv\d+\b', replace, text)
    return re.sub(r'\s+', '', text)


def calls(text):
    return dict(collections.Counter(re.findall(
        r'(?m)^\s*(T[A-Z]\w*|wait_flag_dev|ffts_cross_core_sync|wait_flag|set_flag|pipe_barrier|dsb|dcci)\(', text)))


def sync_calls(text):
    # 保留 PIPE/事件类型，跨核参数 SSA 不用于猜测硬件事件号。
    return dict(collections.Counter(re.findall(
        r'(?m)^\s*((?:wait_flag|set_flag|pipe_barrier)\([^;]+\));', text)))


def collect(root, label):
    pto_path = root / 'ptoas/dynamic_csa_full.pto'
    cpp_path = root / 'ptoas/dynamic_csa_full.cpp'
    pto = pto_path.read_text()
    cpp = cpp_path.read_text()
    upper = {}
    pv_alloc = []
    pto_pv = []
    role = ''
    for number, line in enumerate(pto.splitlines(), 1):
        match = re.search(r'func.func @(\w+)', line)
        if match:
            role = match[1]
        if 'pto.alloc_tile' in line:
            match = re.search(r'(%\w+) = pto.alloc_tile addr = %c(\d+)_i64.*?loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)', line)
            assert match, line
            name, offset, space, dtype, rows, cols = match.groups()
            size = int(rows) * int(cols) * int(re.search(r'\d+', dtype)[0]) // 8
            end = int(offset) + size
            key = role + '/' + space
            upper[key] = max(upper.get(key, 0), end)
            if role == 'cube_schedule' and name.startswith('%pv_'):
                pv_alloc.append(dict(name=name, space=space, offset=int(offset), bytes=size,
                                     end=end, rows=int(rows), cols=int(cols), pto_line=number))
        loc = re.search(r'attention_stream_stages.py":(\d+):', line)
        if role == 'cube_schedule' and loc and 168 <= int(loc[1]) <= 198:
            pto_pv.append((number, line))
    cube = cpp_function(cpp, 'cube_schedule')
    vector = cpp_function(cpp, 'vector_schedule')
    probability = cpp.index('// pto: %pv_probability_inline')
    begin = cpp.rfind('\n', 0, cpp.rfind('wait_flag_dev(', 0, probability)) + 1
    finish = cpp.index('\n', cpp.index('ffts_cross_core_sync(', probability))
    fragment = cpp[begin:finish]
    line0 = cpp[:begin].count('\n') + 1
    numbered = '\n'.join(f'{line0 + i}: {s}' for i, s in enumerate(fragment.splitlines())) + '\n'
    (OUT / (label + '_pv.cpp.txt')).write_text(numbered)
    (OUT / (label + '_pv.pto.txt')).write_text('\n'.join(f'{n}: {s}' for n, s in pto_pv) + '\n')
    pv_operations = collections.Counter()
    store_ranges = []
    for number, line in pto_pv:
        op = re.search(r'\b(pto\.\w+(?:\.\w+)?)\b', line)
        if op:
            pv_operations[op[1]] += 1
        if 'pto.partition_view %pv_ring' in line:
            m = re.search(r'offsets = \[%[^,]+, %c(\d+)_index\], sizes = \[%c(\d+)_index, %c(\d+)_index\]', line)
            assert m, line
            col, rows, width = map(int, m.groups())
            store_ranges.append({'column': col, 'rows': rows, 'width': width, 'pto_line': number})
    return dict(pto=str(pto_path), cpp=str(cpp_path), upper_bounds=upper,
                pv_allocations=pv_alloc, pv_pto_calls=dict(pv_operations),
                pv_cpp_calls=calls(fragment), pv_sync_calls=sync_calls(fragment),
                pv_cpp_begin_line=line0, pv_cpp_end_line=cpp[:finish].count('\n')+1,
                pv_store_ranges=store_ranges,
                full_cpp_calls={'cube': calls(cube), 'vector': calls(vector)},
                full_sync_calls={'cube': sync_calls(cube), 'vector': sync_calls(vector)}), vector


def main():
    cpu = json.loads((OUT / 'cpu_v1/result.json').read_text())
    assert cpu['passed'] and cpu['full_codegen_passed'] and cpu['single_mix']
    old, old_vector = collect(BASE, 'baseline')
    new, new_vector = collect(NEW, 'candidate')
    assert old['upper_bounds'] == new['upper_bounds']
    acc = [x for x in new['pv_allocations'] if x['space'] == 'acc']
    right = [x for x in new['pv_allocations'] if x['space'] == 'right']
    assert {(x['offset'], x['bytes']) for x in acc} == {(0, 65536), (65536, 65536)}
    assert {(x['offset'], x['bytes']) for x in right} == {(0, 65536)}
    assert [(x['column'], x['rows'], x['width']) for x in new['pv_store_ranges']] == [(0,64,256),(256,64,256)]
    assert [(x['column'], x['rows'], x['width']) for x in old['pv_store_ranges']] == [(0,64,128),(128,64,128),(256,64,128),(384,64,128)]
    vector_equal = normalize_cpp(old_vector) == normalize_cpp(new_vector)
    assert vector_equal
    assert new['pv_cpp_calls']['TMATMUL'] == new['pv_cpp_calls']['TEXTRACT'] == new['pv_cpp_calls']['TSTORE'] == 2
    assert old['pv_cpp_calls']['TMATMUL'] == old['pv_cpp_calls']['TEXTRACT'] == old['pv_cpp_calls']['TSTORE'] == 4
    assert new['pv_cpp_calls'].get('pipe_barrier', 0) <= old['pv_cpp_calls'].get('pipe_barrier', 0)
    allocated = (NEW / 'passes_dump/38_after_AllocateMemoryAddr.py').read_text()
    memref = {}
    for m in re.finditer(r'pl.MemRef\((mem_(\w+)_\d+), pl.const\((\d+), pl.INT64\), (\d+)\)', allocated):
        memref[m[2]] = max(memref.get(m[2], 0), int(m[3]) + int(m[4]))
    result = dict(passed=True, scope='唯一一次完整CPU lowering产物的有限只读审计',
                  baseline_source_33_files_equal=True, baseline=old, candidate=new,
                  cpp_delta={r: difference(old['full_cpp_calls'][r],new['full_cpp_calls'][r]) for r in ('cube','vector')},
                  local_sync_delta={r: difference(old['full_sync_calls'][r],new['full_sync_calls'][r]) for r in ('cube','vector')},
                  pv_cpp_delta=difference(old['pv_cpp_calls'],new['pv_cpp_calls']),
                  pv_sync_delta=difference(old['pv_sync_calls'],new['pv_sync_calls']),
                  vector_cpp_equal_ignoring_comments_and_ssa_ids=vector_equal,
                  candidate_memref_upper_bounds=memref,
                  logical_pv_mac_per_valid_block=64*512*128,
                  logical_pv_gm_bytes_per_valid_block=64*512*4,
                  device_execution=False, performance_measured=False,
                  limitation='PTO/C++地址和事件次序不等于实际设备逐bit结果或硬件重叠；未做新ISA反汇编。')
    (OUT / 'generated_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','cpp_delta','local_sync_delta','pv_cpp_delta','pv_sync_delta','vector_cpp_equal_ignoring_comments_and_ssa_ids','candidate_memref_upper_bounds')},ensure_ascii=False))


if __name__ == '__main__':
    main()
