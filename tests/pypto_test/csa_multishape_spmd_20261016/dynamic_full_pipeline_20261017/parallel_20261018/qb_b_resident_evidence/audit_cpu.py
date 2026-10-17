"""只读既有CPU产物，保留v1失败与v2资源/事件证据；不触发任何编译。"""

import ast
import collections
import difflib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path('/data/pyptouser/qinchuanyu/pto-eager')
PAR = ROOT / 'vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018'
BASE = PAR / 'query_linear_h131072_b24_v1/build_output/decode_csa_spmd_5e_gzb5p'
OLD_SOURCE = HERE.parents[1] / 'ultra_indexer_20261018/query_dequant_linear_20261018/source_reference'
PY = ROOT / '.worktrees/pypto-csa-static-fixes-20261016'
ALLOC = re.compile(
    r'%([\w]+) = pto.alloc_tile addr = %c(\d+)_i64.*?!pto.tile_buf<'
    r'loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)'
)


def numbered(path, first, last, target):
    lines = path.read_text().splitlines()
    text = f'原始路径：{path}\n原始行：{first}–{last}\n\n'
    text += '\n'.join(f'{i}: {lines[i-1]}' for i in range(first, last + 1)) + '\n'
    (HERE / target).write_text(text)


def cube_pto(path):
    return path.read_text().split('func.func @vector_schedule')[0]


def allocs(text, skip_qb=False):
    values = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if skip_qb and 'query_projection_stages.py' in line:
            continue
        match = ALLOC.search(line)
        if match:
            name, addr, space, dtype, rows, cols = match.groups()
            width = {'i8': 1, 'i16': 2, 'i32': 4, 'f16': 2, 'bf16': 2, 'f32': 4}[dtype]
            values.append(dict(name=name, address=int(addr), space=space, dtype=dtype,
                               rows=int(rows), cols=int(cols), size=int(rows)*int(cols)*width,
                               line=line_no))
    return values


def without_qb(text):
    lines = text.splitlines()
    matched = [i for i, line in enumerate(lines) if 'query_projection_stages.py' in line]
    assert matched
    begin, end = matched[0], matched[-1] + 1
    while end < len(lines) and lines[end].strip() == '}':
        end += 1
    return '\n'.join(lines[:begin] + ['  // Q_B omitted for comparison'] + lines[end:])


def normalize_pto(text):
    # 保留所有运算、常量值/类型、布局、地址、控制流；只去位置、SSA名和常量声明顺序。
    text = re.sub(r' loc\("[^"\n]+"(?::\d+:\d+)?\)', '', text)
    constants = {}
    lines = []
    for line in text.splitlines():
        match = re.fullmatch(r'\s*(%[\w.]+) = arith.constant (.+)', line)
        if match:
            constants[match[1]] = '[' + match[2] + ']'
        else:
            lines.append(line)
    names = {}
    def rename(match):
        value = match.group()
        if value in constants:
            return constants[value]
        if value not in names:
            names[value] = '%v' + str(len(names))
        return names[value]
    return re.sub(r'%[\w.]+', rename, '\n'.join(lines))


def cpp_qb_span(text, token):
    at = text.index(token)
    begin = text.rfind('  pto::comm::TNOTIFY(', 0, at)
    begin = text.index('\n', begin) + 1
    end = text.index('  pto::comm::TNOTIFY(', at)
    end = text.index('\n', end) + 1
    return begin, end


class FoldQA(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == '_q_a_m16_parallel_worker':
            return None  # QA=false不调用新增helper。
        defaults = dict(zip([a.arg for a in node.args.args[-len(node.args.defaults):]], node.args.defaults)) if node.args.defaults else {}
        node.args.args = [a for a in node.args.args if a.arg != 'QA_M16_PARALLEL']
        node.args.defaults = [defaults[a.arg] for a in node.args.args if a.arg in defaults]
        return self.generic_visit(node)

    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == 'QA_M16_PARALLEL':
            return [self.visit(n) for n in node.orelse]
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.args = [arg for arg in node.args if not (isinstance(arg, ast.Name) and arg.id == 'QA_M16_PARALLEL')]
        return self.generic_visit(node)


def main():
    v1 = HERE / 'cpu_v1'
    v2 = HERE / 'cpu_v2'
    first = json.loads((v1 / 'result.json').read_text())
    second = json.loads((v2 / 'result.json').read_text())
    assert not first['passed'] and '131072' in first['error'] and second['passed']
    assert second['full_codegen_passed'] and second['single_mix'] and not second['sample_shapes_supplied']
    paths = {str(n): v1 / 'kernel/passes_dump' / filename for n, filename in (
        (32, '32_after_LowerPipelineLoops.py'), (33, '33_after_CanonicalizeIOOrder.py'),
        (37, '37_after_MemoryReuse.py'))}
    order = {}
    for n, path in paths.items():
        lines = path.read_text().splitlines()
        pattern = ('resident_w_inline1977__ssa_v0_textract:',
                   'pl.tile.matmul_acc(resident_acc_inline1983__iter_v1',
                   'resident_w_inline1977__ssa_v0_textract_1:')
        order[n] = {key: next(i for i, line in enumerate(lines, 1) if item in line)
                    for key, item in zip(('extract_k256', 'mma_k256', 'extract_k512'), pattern)}
        begin = next(i for i, line in enumerate(lines, 1) if 'for resident_round_' in line)
        end = next(i for i, line in enumerate(lines[begin:], begin + 1) if 'q_proj_v1__ssa_v0:' in line)
        numbered(path, begin, end, f'v1_pass{n}_qb.txt')
    assert order['32']['extract_k256'] < order['32']['mma_k256'] < order['32']['extract_k512']
    assert order['33']['extract_k256'] < order['33']['extract_k512'] < order['33']['mma_k256']
    reused = paths['37'].read_text()
    assert 'resident_w_first_inline1974__ssa_v0_textract: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_380,' in reused
    assert 'resident_w_inline1977__ssa_v0_textract_1: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_386,' in reused
    numbered(PY / 'src/ir/transforms/canonicalize_io_order_pass.cpp', 68, 191, 'io_category_rule.txt')
    numbered(PY / 'src/ir/transforms/canonicalize_io_order_pass.cpp', 223, 280, 'io_pipeline_scope_rule.txt')

    pto_path = v2 / 'kernel/ptoas/dynamic_csa_full.pto'
    cpp_path = v2 / 'kernel/kernels/aic/cube_schedule.cpp'
    pto = cube_pto(pto_path)
    tiles = allocs(pto)
    qb = [item for item in tiles if item['name'].startswith('resident_')]
    numbered(pto_path, 1475, 1543, 'v2_qb.pto.txt')
    numbered(cpp_path, 10394, 10770, 'v2_qb.cpp.txt')
    right = [item for item in qb if item['space'] == 'right']
    qr = [item for item in qb if item['space'] == 'mat' and item['name'].startswith('resident_qr')]
    assert len(right) == 4 and {(x['address'], x['size']) for x in right} == {(0, 65536)}
    assert [x['address'] for x in qr] == [327680, 360448, 327680, 360448]
    assert all(x['size'] == 32768 for x in qr)
    maximum = {}
    for item in tiles:
        maximum[item['space']] = max(maximum.get(item['space'], 0), item['address'] + item['size'])
    assert maximum == dict(mat=475136, left=65536, right=65536, acc=131072, vec=16384)
    body_start = pto.index('  scf.for %resident_round_')
    body_end = pto.index('  pto.fence.barrier_all', body_start)
    body = pto[body_start:body_end]
    assert body.count('pto.tmatmul ins') == 1 and body.count('pto.tmatmul.acc ins') == 3
    assert body.count('pto.textract ins') == 4 and body.count('pto.tstore ins') == 1
    assert body.count('pto.tload ins') == 5
    assert 'valid_row = %721' in body and 'compact=1' in body
    assert 'sizes = [%721, %c256_index]' in body and '%721 = arith.minsi %720, %c128_index' in body
    cpp = cpp_path.read_text()
    begin, end = cpp_qb_span(cpp, '// pto: %resident_weight_')
    cpp_body = cpp[begin:end]
    # 区域包含QB阶段发布本来需要的末尾PIPE_ALL，N/M计算体内部没有。
    first_load = cpp_body.index('    TLOAD(')
    store = cpp_body.index('      TSTORE(')
    assert 'pipe_barrier(PIPE_ALL)' not in cpp_body[first_load:store]
    assert cpp_body.count('TEXTRACT(') == 4
    assert cpp_body.count('TMATMUL(') == 1 and cpp_body.count('TMATMUL_ACC(') == 3
    required = ('wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);',
                'wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);',
                'wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);',
                'wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);')
    assert all(value in cpp_body for value in required)

    # 既有baseline是9d正式候选对照；先证明668的QA=false可达源AST与之相同。
    source_equal = {}
    for path in sorted((HERE / 'source_reference').glob('*.py')):
        if path.name == 'policy.py':
            continue  # Host QA分档后来改变，但本轮controls固定QA=false。
        a, b = [FoldQA().visit(ast.parse(p.read_text())) for p in (OLD_SOURCE / path.name, path)]
        source_equal[path.name] = ast.dump(a) == ast.dump(b)
    assert all(source_equal.values()), [k for k, v in source_equal.items() if not v]
    base_pto = cube_pto(BASE / 'ptoas/dynamic_csa_full.pto')
    matrix_path = BASE.parents[1] / 'source_matrix/result.json'
    matrix = json.loads(matrix_path.read_text())
    assert matrix['kernel_configs']['spmd_reference'] == str(BASE / 'kernel_config.py')
    expected_controls = dict(second['controls'])
    assert expected_controls.pop('QB_B_RESIDENT') is True
    assert expected_controls.pop('QA_M16_PARALLEL') is False
    assert matrix['policies']['spmd_reference'] == expected_controls
    before, after = allocs(base_pto, True), allocs(pto, True)
    fields = ('address', 'space', 'dtype', 'rows', 'cols', 'size')
    assert len(before) == len(after) == 535
    allocation_equal = all(tuple(a[k] for k in fields) == tuple(b[k] for k in fields) for a, b in zip(before, after))
    assert allocation_equal
    a, b = [normalize_pto(without_qb(t)) for t in (base_pto, pto)]
    pto_diff = ''.join(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                         fromfile='existing_baseline_non_qb', tofile='v2_non_qb'))
    (HERE / 'non_qb_normalized_pto.diff').write_text(pto_diff)
    old_aiv, new_aiv = [p.read_text().split('  func.func @vector_schedule', 1)[1]
                        for p in (BASE / 'ptoas/dynamic_csa_full.pto', pto_path)]
    aiv_equal = normalize_pto(old_aiv) == normalize_pto(new_aiv)
    assert aiv_equal
    base_cpp = (BASE / 'kernels/aic/cube_schedule.cpp').read_text()
    bb, be = cpp_qb_span(base_cpp, '// pto: %qr_first_')
    event_pattern = re.compile(r'^\s*(?:set_flag|wait_flag|pipe_barrier)\([^;]+;', re.M)
    old_start = base_cpp.index('static __aicore__ void cube_schedule(')
    new_start = cpp.index('static __aicore__ void cube_schedule(')
    old_events = [line.strip() for line in event_pattern.findall(base_cpp[old_start:bb] + base_cpp[be:])]
    new_events = [line.strip() for line in event_pattern.findall(cpp[new_start:begin] + cpp[end:])]
    event_diff = ''.join(difflib.unified_diff([x+'\n' for x in old_events], [x+'\n' for x in new_events],
                                           fromfile='existing_baseline_non_qb_events', tofile='v2_non_qb_events'))
    (HERE / 'non_qb_events.diff').write_text(event_diff)
    summary = dict(
        scope='静态生成物审阅，不能代替设备精度/性能；只读既有产物，未新增编译',
        cpu_v1=dict(passed=False, error=first['error'], order=order,
                    first_right_reuses_pipeline0_and_tail=True,
                    cause='Pipeline IO重排把K512 Right提前至K256 MMA前；两份64KiB真实重叠，不是first存活或K增大'),
        cpu_v2=dict(passed=second['passed'], full_codegen_passed=second['full_codegen_passed'],
                    symbolic_dimensions=len(second['symbolic_dimensions']), single_mix=second['single_mix'],
                    device_execution=False, sample_shapes_supplied=False,
                    qb_tiles=qb, full_aic_max_end_bytes=maximum, qb_live_mat_bytes=327680,
                    four_k_order=[0, 256, 512, 768], interior_pipe_all=0,
                    qr_slot_reuse='QR0/2同址、QR1/3同址；MTE1→MTE2消费完毕后覆盖',
                    right_reuse='四Right同址；每次M→MTE1后才提取下一段，MTE1→M后MMA',
                    acc_and_store='首段/下一M首段等FIX→M；末段M→FIX后有效行TSTORE',
                    valid='min(T-t0,128)同时约束QR GM view、Left、compact Acc和store GM view'),
        non_qb=dict(
            baseline_artifact=str(BASE), baseline_source=str(OLD_SOURCE),
            baseline_scope='已有9d对照，QA=false可达32文件AST与668相同；不是新编668二进制',
            baseline_role_verified_by=str(matrix_path), controls_equal_after_new_flags=True,
            baseline_controls=matrix['policies']['spmd_reference'],
            reachable_source_ast_equal=source_equal, allocation_entries=len(before),
            allocations_equal=allocation_equal, normalized_pto_equal=a == b,
            full_aiv_normalized_pto_equal=aiv_equal,
            normalization='只去loc与SSA名、展开常量声明；保留数学/控制流/类型/布局/所有地址',
            cpp_local_event_entries=[len(old_events), len(new_events)],
            cpp_local_event_sequence_equal=old_events == new_events,
            cpp_scope='只比较cube_schedule函数体非Q_B的set_flag/wait_flag/pipe_barrier。旧设备注入wait-backoff helper，本CPU没有；不声称整C++或GM等待helper相同',
        ),
        artifact_paths=dict(reference_source=str(HERE / 'source_reference'),
                            candidate_v1=str(HERE / 'source_candidate'),
                            candidate_v2=str(HERE / 'source_candidate_v2'),
                            cpu_v1_result=str(v1 / 'result.json'), cpu_v2_result=str(v2 / 'result.json')),
    )
    (HERE / 'cpu_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(v1_preserved=True, v2_passed=True, full_aic_max_end_bytes=maximum,
                         non_qb_allocations_equal=allocation_equal, non_qb_pto_equal=a == b,
                         non_qb_events_equal=old_events == new_events,
                         event_counts=[len(old_events), len(new_events)]), ensure_ascii=False))


if __name__ == '__main__':
    main()
