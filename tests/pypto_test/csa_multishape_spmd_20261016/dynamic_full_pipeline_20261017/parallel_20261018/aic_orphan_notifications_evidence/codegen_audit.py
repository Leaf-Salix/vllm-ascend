"""只读对照本轮已生成产物；不编译、不运行设备、不改冻结源码。"""

import collections
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = Path('/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/ultra_compact_priority_20261018/cpu_v1')
REF = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/adopted_source_d56c9eb6')
CAND = HERE / 'cpu_v1'


def normalize_locations(text):
    return re.sub(r'loc\("([^\"]+)":', lambda m: 'loc("' + Path(m[1]).name + '":', text)


def split_cpp(text):
    cube = text.index('AICORE void cube_schedule')
    vector = text.index('AICORE void vector_schedule')
    return {'cube_schedule': text[cube:vector], 'vector_schedule': text[vector:]}


def read_pto(path):
    allocations = []
    operations = collections.Counter()
    syncs = []
    role = ''
    lines = path.read_text().splitlines()
    for line in lines:
        m = re.search(r'func.func @(\w+)', line)
        if m:
            role = m[1]
        if 'pto.alloc_tile' in line:
            m = re.search(r'pto.alloc_tile addr = %c(\d+)_i64.*?!pto.tile_buf<([^>]+)>', line)
            loc = re.search(r'loc\("([^\"]+)":(\d+):(\d+)\)', line)
            assert m and loc, line
            allocations.append((role, int(m[1]), m[2], Path(loc[1]).name, int(loc[2]), int(loc[3])))
        operations.update((role, op) for op in re.findall(r'(?<![!#])\bpto\.([a-zA-Z_0-9.]+)', line))
        if re.search(r'\bpto\.sync\.', line):
            syncs.append((role, normalize_locations(line.strip())))
    return allocations, operations, syncs


def cpp_counts(text):
    patterns = {
        'TNOTIFY': r'pto::comm::TNOTIFY\(', 'TWAIT': r'pto::comm::TWAIT\(',
        'wait_flag_dev': r'\bwait_flag_dev\(', 'ffts_cross_core_sync': r'\bffts_cross_core_sync\(',
        'PIPE_ALL': r'pipe_barrier\(PIPE_ALL\)', 'DSB_DDR': r'dsb\(DSB_DDR\)',
        'whole_DCCI': r'dcci\(\(__gm__ void\*\)0, cache_line_t::ENTIRE_DATA_CACHE\)',
        'set_flag': r'\bset_flag\(', 'wait_flag': r'\bwait_flag\(',
    }
    return {key: len(re.findall(pattern, text)) for key, pattern in patterns.items()}


def local_events(text):
    return re.findall(r'\b(?:set_flag|wait_flag)\([^\n]+\);', text)


base_result = json.loads((BASE / 'result.json').read_text())
cand_result = json.loads((CAND / 'result.json').read_text())
files = sorted(REF.glob('*.py'))
assert len(files) == 33
base_source_matches = all(f.read_text() == (BASE / 'source' / f.name).read_text() for f in files)
frozen_source_matches = all((HERE / 'source_candidate' / f.name).read_text() == (CAND / 'source' / f.name).read_text() for f in files)
ba, bo, bs = read_pto(BASE / 'kernel/ptoas/dynamic_csa_full.pto')
ca, co, cs = read_pto(CAND / 'kernel/ptoas/dynamic_csa_full.pto')
bct = (BASE / 'kernel/ptoas/dynamic_csa_full.cpp').read_text()
cct = (CAND / 'kernel/ptoas/dynamic_csa_full.cpp').read_text()
bc, cc = split_cpp(bct), split_cpp(cct)
role_counts = {role: {'baseline': cpp_counts(bc[role]), 'candidate': cpp_counts(cc[role])} for role in bc}
allocation_counts = collections.Counter((a[0], re.search(r'loc=(\w+)', a[2])[1]) for a in ba)
op_differences = [{'role': k[0], 'operation': k[1], 'baseline': bo[k], 'candidate': co[k]} for k in sorted(set(bo) | set(co)) if bo[k] != co[k]]
normalize_hint = lambda text: re.sub(r' at [^\n]*/([^/\n]+\.py:\d+:\d+)', r' at \1', text)
base_hints = (BASE / 'kernel/report/perf_hints.log').read_text()
cand_hints = (CAND / 'kernel/report/perf_hints.log').read_text()
checks = {
    'cpu_full_dynamic_passed': cand_result['passed'],
    'baseline_33_source_files_direct_text_match_d56': base_source_matches,
    'frozen_33_candidate_files_direct_text_match_compiled_copy': frozen_source_matches,
    'controls_equal': base_result['controls'] == cand_result['controls'],
    'compiled_abi_equal': base_result['compiled_abi'] == cand_result['compiled_abi'],
    'all_allocation_records_same_order': ba == ca,
    'all_allocation_records_same_multiset': collections.Counter(ba) == collections.Counter(ca),
    'all_FFTS_PTO_sync_records_equal_except_source_prefix': bs == cs,
    'cube_local_hardware_event_sequence_equal': local_events(bc['cube_schedule']) == local_events(cc['cube_schedule']),
    'vector_local_hardware_event_sequence_equal': local_events(bc['vector_schedule']) == local_events(cc['vector_schedule']),
    'vector_cpp_identical_after_removing_comments_only': re.sub(r'//[^\n]*', '', bc['vector_schedule']) == re.sub(r'//[^\n]*', '', cc['vector_schedule']),
    'SCORE_CUBE_and_ATTN_CUBE_slot_constants_removed': all(str(offset) in bc['cube_schedule'] and str(offset) not in cc['cube_schedule'] for offset in (313344, 350208)),
    'performance_hints_identical_except_source_prefix': normalize_hint(base_hints) == normalize_hint(cand_hints),
}
result = {
    'passed': all(checks.values()), 'checks': checks,
    'baseline_cpu': str(BASE), 'candidate_cpu': str(CAND),
    'controls': cand_result['controls'], 'symbolic_dimension_count': len(cand_result['symbolic_dimensions']),
    'required_root_dimension_count': len(cand_result['required_root_dimensions']),
    'compiled_abi_argument_count': len(cand_result['compiled_abi']),
    'single_mix': cand_result['single_mix'], 'sample_shapes_supplied': cand_result['sample_shapes_supplied'],
    'allocation_record_count': len(ba),
    'allocation_method': '按出现顺序直接比较(role, physical byte address, complete tile type including shape/layout, source basename/line/column)。记录包括重复/复用视图，非独立物理缓冲区数量；不比较SSA临时名。',
    'allocation_records_by_space': [{'role': k[0], 'space': k[1], 'count': v} for k, v in sorted(allocation_counts.items())],
    'cpp_static_occurrence_counts': role_counts,
    'pto_operation_count_differences': op_differences,
    'ffts_records': cs,
    'trace_boundary': '本审计比较全部物理allocation记录、FFTS指令和本核set_flag/wait_flag顺序；AIV仅去注释后全文相同。没有声称删除发布后的AIC全指令时序相同，也未执行最终机器码或测量耗时。数学与原wait的参数/控制流由source_audit.json的AST对照约束。',
    'device_precision_validated': False, 'performance_measured': False,
    'local_artifacts': [{'path': str(root / rel), 'bytes': (root / rel).stat().st_size} for root in (BASE, CAND) for rel in ('result.json', 'kernel/ptoas/dynamic_csa_full.pto', 'kernel/ptoas/dynamic_csa_full.cpp', 'kernel/report/perf_hints.log')],
}
(HERE / 'codegen_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
excerpt = []
for label, text, ranges in (
    ('baseline', bct, [(17459, 17526), (18063, 18132)]),
    ('candidate', cct, [(17435, 17518), (18010, 18071)]),
):
    lines = text.splitlines()
    for start, end in ranges:
        excerpt.append(f'[{label}: dynamic_csa_full.cpp:{start}-{end}]')
        excerpt.extend(f'{i + 1}: {lines[i]}' for i in range(start - 1, end))
(HERE / 'codegen_excerpt.txt').write_text('\n'.join(excerpt) + '\n')
print(json.dumps({'passed': result['passed'], 'checks': checks, 'allocation_record_count': len(ba), 'cpp_static_occurrence_counts': role_counts}, ensure_ascii=False, indent=2))
assert result['passed']
