"""只读检查动态M候选的AIC生成代码：MMA输入、Acc与写回的行距契约。"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANDIDATES = ('aligned_stage1_rows_v4', 'aligned_rows_token_major_v1')
FUNCTIONS = ('hc_medium96', 'hc_large144', 'hc_large')


def declared_tile(prefix, name):
    matches = list(re.finditer(r'Tile<([^>]+)> ' + name + r' = Tile<[^>]+>\(([^;]*)\);', prefix))
    if not matches:
        raise ValueError(f'无法找到tile声明: {name}')
    match = matches[-1]
    return ([item.strip() for item in match.group(1).split(',')],
            [item.strip() for item in match.group(2).split(',')])


def inspect(candidate, function):
    file = ROOT / f'build_{candidate}/build/ptoas/{function}.cpp'
    cpp = file.read_text().split('#if defined(__DAV_VEC__)')[0]
    stores = list(re.finditer(r'TSTORE\((\w+), (\w+)\);', cpp))
    if len(stores) != 1:
        raise ValueError(f'{candidate}/{function}: AIC投影写回数量不是1')
    store = stores[0]
    dst, src = store.groups()
    fields, args = declared_tile(cpp[:store.start()], src)
    owned = args[0]
    physical_m = int(fields[2])
    static_int = {name: int(number) for name, number in
                  re.findall(r'const int(?:32|64)_t (v\d+) = (\d+);', cpp)}
    calls = []
    for call in re.finditer(r'(TMATMUL(?:_ACC)?)\(([^;]+)\);', cpp):
        values = [x.strip() for x in call.group(2).split(',')]
        acc = values[0]
        left = values[-2]
        prefix = cpp[:call.start()]
        acc_type, _ = declared_tile(prefix, acc)
        left_type, left_args = declared_tile(prefix, left)
        row = left_args[0]
        # PyPTO将M输入表示成min(owned_rows,物理M)；域证明在后面的14场景中。
        clamp = re.search(r'int64_t ' + row + r' = (v\d+) < (v\d+) \? \1 : \2;', prefix)
        row_matches = row == owned or bool(clamp and clamp.group(1) == owned and static_int.get(clamp.group(2)) == physical_m)
        set_shapes = list(re.finditer(re.escape(acc) + r'\.SetValidShape\(([^,]+), ([^)]+)\);', prefix))
        last_shape = set_shapes[-1].groups() if set_shapes else None
        calls.append(dict(operation=call.group(1), acc=acc, left=left,
                          owned_rows_symbol=owned, left_valid_M_symbol=row,
                          left_M_is_owned_rows=row_matches,
                          acc_M_is_owned_rows=bool(last_shape and last_shape[0] == owned),
                          acc_N_is_24=bool(last_shape and static_int.get(last_shape[1]) == 24),
                          compact_acc=acc_type[-1] == 'CompactMode::Normal',
                          compact_left=left_type[-1] == 'CompactMode::Normal'))
    passed = bool(calls) and all(
        c['left_M_is_owned_rows'] and c['acc_M_is_owned_rows'] and c['acc_N_is_24']
        and c['compact_acc'] and c['compact_left'] for c in calls)
    return dict(cpp=str(file.relative_to(ROOT)), physical_M=physical_m,
                source_store_valid_N=fields[6], store_uses_compact_acc=fields[-1] == 'CompactMode::Normal',
                all_MMA_contracts_hold=passed, calls=calls)


results = {candidate: {name: inspect(candidate, name) for name in FUNCTIONS} for candidate in CANDIDATES}
rows = json.loads((ROOT / 'aligned_stage1_rows_static_audit.json').read_text())['cases']
domain = []
for case in rows:
    if not case['balanced']:
        continue
    physical = {'hc_medium96':32, 'hc_large144':48, 'hc_large':64}[case['variant']]
    owned = [x['owned_rows'] for x in case['after']]
    domain.append(dict(B=case['B'],S=case['S'],physical_M=physical,owned_rows=owned,
                       positive_and_within_physical_M=all(0 < m <= physical for m in owned),
                       compact_pitch_rows=[(m+15)//16*16 for m in owned],
                       compact_storage_within_allocation=all((m+15)//16*16 <= physical for m in owned)))
report = dict(
    status='STATIC_CODEGEN_CONTRACT_AUDIT', device_execution=False,
    candidates=results, M_domain=domain,
    all_contracts_hold=all(v['all_MMA_contracts_hold'] and v['store_uses_compact_acc'] and v['source_store_valid_N']=='24'
                           for c in results.values() for v in c.values())
                       and all(x['positive_and_within_physical_M'] and x['compact_storage_within_allocation'] for x in domain),
    source_contracts=[
        'pto-isa-src/include/pto/npu/a2a3/TMatmul.hpp: mad的输出列块行距为ceil16(m)',
        'pto-isa-src/include/pto/common/arch/memory/tstore_common.hpp: CompactMode::Normal写回使用ceil16(validRow)',
    ],
    limitation='核对的是生成代码和地址契约；不能证明设备执行数值、同步或性能。',
)
output=ROOT/'compact_acc_contract_audit.json'
output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(output)
print('全部动态M/紧凑Acc契约成立:',report['all_contracts_hold'])
for candidate, functions in results.items():
    print(candidate,{k:v['all_MMA_contracts_hold'] for k,v in functions.items()})
