"""只读两轮已完成B4候选的源码/生成物；不导入设备库、不编译、不测性能。"""

import ast
import difflib
import importlib.util
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path('/data/pyptouser/qinchuanyu/pto-eager')
PAR = ROOT / 'vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018'
TASKS = ('oa_pair_manual_h131072_b4_v1', 'oa_pair_small_t_h131072_b4_v1')
HELPER = ROOT / '.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_qb_weight_reuse_design_20261018/b_resident_cpu/audit_cpu.py'
OLD_TEST = ast.parse('OA_COLS == 128 and tokens > 0 and tokens <= 96', mode='eval').body
NEW_TEST = ast.parse('OA_COLS == 128 and OB_ROWS == 32 and tokens > 0 and tokens <= 32', mode='eval').body


class SourceNormalizer(ast.NodeTransformer):
    def __init__(self, restore=False):
        self.restore = restore
        self.changed = 0

    def visit(self, node):
        node = super().visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if self.restore and ast.dump(node.test) == ast.dump(NEW_TEST):
            node.test = OLD_TEST
            self.changed += 1
        return node


def function_body(text, name):
    start = text.index('static __aicore__ void ' + name + '(')
    brace = text.index('{', start)
    depth = 1
    for i in range(brace + 1, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        if depth == 0:
            return text[start:i + 1]
    raise ValueError(name)


def normalize_cpp(text):
    # 每函数单独处理：只消除注释/空白、常量声明顺序与临时变量编号。
    # 常量的类型和值均原样保留；不改表达式、layout、事件、分支或数学顺序。
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    constants, lines, names = {}, [], {}
    for line in text.splitlines():
        match = re.fullmatch(r'\s*const ([\w:]+) (v\d+) = (.+);\s*', line)
        if match:
            constants[match[2]] = '[' + match[1] + ':' + match[3] + ']'
        elif line.strip():
            lines.append(line.strip())

    def replace(match):
        key = match[0]
        if key in constants:
            return constants[key]
        if key not in names:
            names[key] = 'v' + str(len(names))
        return names[key]

    return re.sub(r'\bv\d+\b', replace, '\n'.join(lines))


def main():
    spec = importlib.util.spec_from_file_location('readonly_pto_normalizer', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)  # 仅标准库和函数定义；不执行其main。
    matrices = [json.loads((PAR / t / 'source_matrix/result.json').read_text()) for t in TASKS]
    builds = [Path(m['kernel_configs']['spmd_candidate']).parent for m in matrices]
    assert matrices[0]['policies']['spmd_candidate'] == matrices[1]['policies']['spmd_candidate']
    report = {'scope': '只读现成生成物；无新编译/NPU，无校验摘要', 'role': '两侧均source_matrix指定的spmd_candidate',
              'builds': list(map(str, builds)), 'policy': matrices[0]['policies']['spmd_candidate'],
              'existing_results': [{r: m['statistics_us'][r] for r in ('spmd_reference', 'spmd_candidate')} for m in matrices]}
    source_checks = []
    for old_file in sorted((PAR / TASKS[0] / 'source').glob('*.py')):
        new_file = PAR / TASKS[1] / 'source' / old_file.name
        left = SourceNormalizer().visit(ast.parse(old_file.read_text()))
        norm = SourceNormalizer(restore=True)
        right = norm.visit(ast.parse(new_file.read_text()))
        equal = ast.dump(left) == ast.dump(right)
        assert equal, old_file.name
        source_checks.append({'file': old_file.name, 'text_equal': old_file.read_bytes() == new_file.read_bytes(),
                              'ast_equal_restoring_predicate': equal, 'restored_predicates': norm.changed})
    assert len(source_checks) == 33 and sum(x['restored_predicates'] for x in source_checks) == 2
    report['actual_source_snapshot_checks'] = source_checks
    ptos = [(b / 'ptoas/dynamic_csa_full.pto').read_text() for b in builds]
    normalized = [helper.normalize_pto(t) for t in ptos]
    (HERE / 'whole_pto.diff').write_text(''.join(difflib.unified_diff(normalized[0].splitlines(True), normalized[1].splitlines(True), fromfile='T96', tofile='T32')))
    replacements = normalized[0].count('arith.cmpi sle, %v132, [96 : index]')
    assert replacements == 2
    assert normalized[0].replace('arith.cmpi sle, %v132, [96 : index]', 'arith.cmpi sle, %v132, [32 : index]') == normalized[1]
    report['whole_pto_only_two_threshold_changes'] = True
    report['functions'] = {}
    for core, name, kid in [('aic', 'cube_schedule', 0), ('aiv', 'vector_schedule', 1)]:
        paths = [b / f'kernels/{core}/{name}.cpp' for b in builds]
        bodies = [function_body(p.read_text(), name) for p in paths]
        cpp_normal = [normalize_cpp(t) for t in bodies]
        assert cpp_normal[0].count('<= [int64_t:96]') == 1
        assert cpp_normal[0].replace('<= [int64_t:96]', '<= [int64_t:32]') == cpp_normal[1]
        (HERE / f'{core}_cpp.diff').write_text(''.join(difflib.unified_diff(cpp_normal[0].splitlines(True), cpp_normal[1].splitlines(True), fromfile='T96', tofile='T32')))
        events = [[v.strip() for v in re.findall(r'^\s*(?:set_flag|wait_flag|pipe_barrier)\([^;]+;', b, re.M)] for b in bodies]
        assert events[0] == events[1]
        tiles = []
        for text in ptos:
            text = text.split('func.func @vector_schedule')[0] if core == 'aic' else text.split('func.func @vector_schedule', 1)[1]
            current = []
            for match in helper.ALLOC.finditer(text):
                _, addr, space, dtype, rows, cols = match.groups()
                bits = int(re.search(r'(\d+)$', dtype)[1])
                current.append((int(addr), space, dtype, int(rows), int(cols), int(rows) * int(cols) * bits // 8))
            tiles.append(current)
        assert tiles[0] == tiles[1]
        maximum = {space: max(a + size for a, sp, _, _, _, size in tiles[0] if sp == space) for space in {x[1] for x in tiles[0]}}
        bin_paths = [b / f'prebuilt/kernel_{kid}.bin' for b in builds]
        blobs = [p.read_bytes() for p in bin_paths]
        assert len(blobs[0]) == len(blobs[1])
        offsets = [i for i, (a, b) in enumerate(zip(*blobs)) if a != b]
        assert len(offsets) == 1
        report['functions'][name] = {'core': core, 'cpp_paths': list(map(str, paths)), 'cpp_only_one_threshold_change': True,
            'local_events_count_each': len(events[0]), 'local_events_and_ids_equal': True,
            'pipe_all_count_each': sum('PIPE_ALL' in v for v in events[0]),
            'allocation_count_each': len(tiles[0]), 'allocation_address_type_shape_size_equal': True,
            'physical_address_upper_bounds_each': maximum, 'raw_image_paths': list(map(str, bin_paths)),
            'raw_image_bytes_each': len(blobs[0]), 'raw_images_equal': False, 'raw_different_byte_count': len(offsets),
            'raw_different_byte_offsets_zero_based': offsets,
            'isa_limit': '没有重新反汇编；不以单字节差异独立认定具体ISA操作码或立即数字段'}
    report['performance_limit'] = '不同轮绝对值和改善比例不能归因为编译变化；小T候选不采用、不补测。'
    (HERE / 'evidence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['functions'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
