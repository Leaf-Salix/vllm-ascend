"""只读本次生成PTO/C++，记录真实地址上界、QB有效行与事件；不编译。"""
import ast
import json
from pathlib import Path
import re

E = Path(__file__).resolve().parent
F = Path('/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_qb_weight_reuse_design_20261018/b_resident_cpu')
pto = E / 'cpu_v1/kernel/ptoas/dynamic_csa_full.pto'
cpp = E / 'cpu_v1/kernel/kernels/aic/cube_schedule.cpp'
result = json.loads((E / 'cpu_v1/result.json').read_text())
assert result['passed'] and result['full_codegen_passed']
fn, constants, upper, qb, unresolved = '', {}, {}, [], []
for lineno, line in enumerate(pto.read_text().splitlines(), 1):
    match = re.search(r'func.func @(\w+)', line)
    if match:
        fn, constants = match[1], {}
    match = re.search(r'(%[\w]+) = arith.constant (\d+) : (?:i64|index)', line)
    if match:
        constants[match[1]] = int(match[2])
    if 'pto.alloc_tile' not in line:
        continue
    match = re.search(r'addr = (%\w+).*loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)', line)
    if not match or match[1] not in constants:
        unresolved.append({'function': fn, 'line': lineno})
        continue
    addr, space, dtype, rows, cols = constants[match[1]], match[2], match[3], int(match[4]), int(match[5])
    bits = 16 if dtype == 'bf16' else int(re.search(r'\d+', dtype)[0])
    size = rows * cols * bits // 8
    key = fn + ':' + space
    upper[key] = max(upper.get(key, 0), addr + size)
    if '%resident_' in line:
        name = line.strip().split(' = ')[0]
        qb.append({'name': name, 'line': lineno, 'space': space, 'dtype': dtype,
                   'rows': rows, 'cols': cols, 'address': addr, 'bytes': size})
assert not unresolved
mat = [r for r in qb if r['space'] == 'mat']
assert {(r['address'], r['bytes']) for r in mat} == {(0, 262144), (262144, 262144)}
assert {r['address'] for r in qb if r['space'] in ('left', 'right', 'acc')} == {0}
assert upper['cube_schedule:mat'] == 524288
lines = cpp.read_text().splitlines()
start = next(i for i, line in enumerate(lines) if '// pto: %resident_qr_full_' in line)
stop = next(i for i in range(start, len(lines)) if 'pto::comm::TNOTIFY(' in lines[i])
events = [f'{i+1}: {lines[i].strip()}' for i in range(start, stop + 1)
          if re.search(r'\b(?:TLOAD|TEXTRACT|TMATMUL|TMATMUL_ACC|TSTORE|TNOTIFY|wait_flag|set_flag|pipe_barrier|dsb|dcci)\(', lines[i])
          or 'for (int64_t ' in lines[i]]
(E / 'qb_events.txt').write_text('\n'.join(events) + '\n')

def function_body(path):
    fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_q_b_nz_stage')
    return [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]

formal_body = function_body(F / 'source_reference/query_projection_stages.py')
candidate_body = function_body(E / 'source_candidate/query_projection_stages.py')
flattened = []
for node in candidate_body:
    if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'QB_B_RESIDENT':
        flattened.extend(node.orelse)
    else:
        flattened.append(node)
false_matches_formal = ast.dump(ast.Module(body=formal_body, type_ignores=[])) == ast.dump(ast.Module(body=flattened, type_ignores=[]))
assert false_matches_formal
audit = {
    'passed': True,
    'cpu_result': str(E / 'cpu_v1/result.json'),
    'pto': {'path': str(pto), 'bytes': pto.stat().st_size},
    'cpp': {'path': str(cpp), 'bytes': cpp.stat().st_size},
    'formal_reference_source': str(F / 'source_reference'),
    'b_only_source': str(F / 'source_candidate'),
    'candidate_source': str(E / 'source_candidate'),
    'symbolic_dimensions': len(result['symbolic_dimensions']),
    'root_tensors': len(result['compiled_abi']),
    'false_body_ast_equals_formal_668': false_matches_formal,
    'allocation_global_end_bytes': upper,
    'qb_allocations': qb,
    'allocation_unresolved': unresolved,
    'validity': {'pto_lines': [1476,1492,1493,1494,1495,1502,1508,1514,1515,1516,1519,1520],
                 'rule': 'QR physical256 rows, validT; Left/Acc/store min(max(T-t0,0),128); source STRIDE stays256',
                 'tail_valid_rows': {'129':[128,1], '255':[128,127], '256':[128,128]}},
    'events_file': str(E / 'qb_events.txt'),
    'scope': '一次完整CPU；无NPU。地址最大末端不是同时存活量。未声称非QB事件或所有分配与formal逐字相同。',
}
(E / 'generated_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k:audit[k] for k in ['passed','root_tensors','symbolic_dimensions','allocation_global_end_bytes','false_body_ast_equals_formal_668']},ensure_ascii=False))
