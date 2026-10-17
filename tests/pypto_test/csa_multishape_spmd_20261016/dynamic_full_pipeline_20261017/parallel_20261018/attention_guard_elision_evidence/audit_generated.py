#!/usr/bin/env python3
"""只读核对一次完整 CPU 产物，不编译、不执行设备、不修改已冻结源码。

基线使用已有 d253 真机任务保留的 hook 前 PTO/C++；先确认其 33 源与 e581
冻结源相同。比较完整函数的数学/搬运/同步与片上分配上界，并单独统计此次
Attention 源行对应的 Scalar 操作，避免把声明减少解释为硬件性能收益。
"""

import ast
import collections
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
PAR = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/'
           'csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018')
BASE = PAR / 'hc_active_notifications_h131072_b16_v1/build_output/decode_csa_spmd_3kuwaqat'
BASE_SOURCE = PAR / 'hc_active_notifications_evidence/source_candidate'
NEW = OUT / 'cpu_v1/kernel'
INPUTS = set()


def read(path):
    INPUTS.add(path)
    return path.read_text()


def collect(path, source):
    pto = path / 'ptoas/dynamic_csa_full.pto'
    cpp = path / 'ptoas/dynamic_csa_full.cpp'
    tree = ast.parse(read(source / 'attention_stream_stages.py'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'attention_qk_pv_vector_lane_stage')
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'qk_row')
    whole, attention, upper, gather_rows = {}, {}, {}, []
    function = ''
    for no, line in enumerate(read(pto).splitlines(), 1):
        fm = re.search(r'func.func @(\w+)', line)
        if fm:
            function = fm[1]
        operation = re.search(r'\b(pto\.\w+(?:\.\w+)?)\b', line)
        if operation:
            whole.setdefault(function, collections.Counter())[operation[1]] += 1
        location = re.search(r'attention_stream_stages.py":(\d+):', line)
        if location:
            scalar_or_tile = re.search(r'\b((?:pto|arith|scf)\.\w+(?:\.\w+)?)\b', line)
            if scalar_or_tile:
                attention.setdefault(function, collections.Counter())[scalar_or_tile[1]] += 1
            if loop.lineno <= int(location[1]) <= loop.end_lineno:
                gather_rows.append((no, line))
        if 'pto.alloc_tile' in line:
            allocation = re.search(
                r'addr = %c(\d+)_i64.*?loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)', line
            )
            assert allocation, (pto, no, line)
            address, space, dtype, rows, columns = allocation.groups()
            bits = int(re.search(r'\d+', dtype)[0])
            end = int(address) + int(rows) * int(columns) * bits // 8
            key = function + '/' + space
            upper[key] = max(upper.get(key, 0), end)
    cpp_text = read(cpp)
    cpp_ops = {}
    for role in ('cube_schedule', 'vector_schedule'):
        begin = cpp_text.index('AICORE void ' + role + '(')
        end = cpp_text.find('AICORE void ', begin + 20)
        body = cpp_text[begin:end if end >= 0 else len(cpp_text)]
        names = re.findall(
            r'(?m)^\s*(T[A-Z]\w*|wait_flag_dev|ffts_cross_core_sync|wait_flag|set_flag|pipe_barrier|dsb|dcci)\(', body
        )
        cpp_ops[role] = dict(
            calls=dict(collections.Counter(names)), if_sites=len(re.findall(r'\bif \(', body))
        )
    return dict(pto=str(pto), cpp=str(cpp), whole=whole, attention=attention,
                address_upper_bounds=upper, cpp_ops=cpp_ops, gather_rows=gather_rows)


def difference(old, new):
    return dict(removed=dict(collections.Counter(old) - collections.Counter(new)),
                added=dict(collections.Counter(new) - collections.Counter(old)))


def main():
    files = sorted((OUT / 'source_before').glob('*.py'))
    assert len(files) == len(list(BASE_SOURCE.glob('*.py'))) == 33
    assert all(read(p) == read(BASE_SOURCE / p.name) for p in files)
    old = collect(BASE, OUT / 'source_before')
    new = collect(NEW, OUT / 'source_candidate')
    assert old['address_upper_bounds'] == new['address_upper_bounds']
    pto_delta = {name:difference(old['whole'][name], new['whole'][name]) for name in old['whole']}
    assert pto_delta['cube_schedule'] == dict(removed={}, added={})
    assert pto_delta['vector_schedule'] == dict(removed={'pto.alloc_tile':2}, added={})
    attention_delta = {
        name:difference(old['attention'][name], new['attention'][name]) for name in old['attention']
    }
    assert attention_delta['cube_schedule'] == dict(removed={}, added={})
    assert attention_delta['vector_schedule'] == dict(
        removed={'arith.divsi':1, 'arith.cmpi':3, 'arith.andi':1}, added={}
    )
    cpp_delta = {
        name:difference(old['cpp_ops'][name]['calls'], new['cpp_ops'][name]['calls'])
        for name in old['cpp_ops']
    }
    assert cpp_delta['cube_schedule'] == dict(removed={}, added={})
    assert cpp_delta['vector_schedule'] == dict(removed={'TASSIGN':2}, added={})
    assert old['cpp_ops']['vector_schedule']['if_sites'] - new['cpp_ops']['vector_schedule']['if_sites'] == 2
    # 声明层少两份别名 Tile，不能解释为新增或减少了两次业务 copy。
    # 整个 C++ 的硬件 Tile 运算、读写和本地/跨核同步调用没有增加。
    allocated = read(NEW / 'passes_dump/38_after_AllocateMemoryAddr.py')
    memref = {}
    for m in re.finditer(r'pl.MemRef\((mem_(\w+)_\d+), pl.const\((\d+), pl.INT64\), (\d+)\)', allocated):
        memref[m[2]] = max(memref.get(m[2], 0), int(m[3]) + int(m[4]))
    assert memref['vec'] == new['address_upper_bounds']['vector_schedule/vec'] == 163872
    for label, data in (('baseline', old), ('candidate', new)):
        (OUT / (label + '_compressed_gather.pto.txt')).write_text(
            '\n'.join(f'{no}: {line}' for no, line in data.pop('gather_rows')) + '\n'
        )
    result = dict(
        passed=True, scope='已有完整CPU生成物有限审计；未追加编译或设备任务',
        baseline_source_33_files_equal=True, baseline=old, candidate=new,
        pto_differences=pto_delta, attention_operation_differences=attention_delta,
        cpp_call_differences=cpp_delta, candidate_memref_upper_bounds=memref,
        mathematical_tile_ops_and_load_store_counts_equal=True,
        sync_call_counts_equal=True,
        source_sync_and_math_control_unchanged='由source_audit的完整AST归一化相等支撑；生成调用计数另行交叉核对。',
        original_metadata_plan_and_index_semantics_unchanged=True,
        device_execution=False, performance_measured=False,
        limitations=[
            'C++/PTO条件减少不等于CCEC最终动态机器指令数或完整L1收益；本轮没有反汇编新二进制。',
            '片上地址上界相同不宣称每个临时量的物理地址逐一相同。',
            '真实地址/页表交换、图重放、信号代际、保护区和性能由根代理两例设备任务验证。',
        ],
        inputs=[dict(path=str(p), bytes=p.stat().st_size) for p in sorted(INPUTS)],
    )
    (OUT / 'generated_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(passed=True, address_upper_bounds=new['address_upper_bounds'],
                         attention_delta=attention_delta, cpp_delta=cpp_delta), ensure_ascii=False))


if __name__ == '__main__':
    main()
