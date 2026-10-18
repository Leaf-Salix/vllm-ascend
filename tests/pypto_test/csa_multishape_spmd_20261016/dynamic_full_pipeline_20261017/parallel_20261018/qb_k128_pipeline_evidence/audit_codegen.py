# SPDX-License-Identifier: Apache-2.0
"""审阅真实生成物的双L0地址和当前/下一片就绪事件，不以stage2字样代替证据。"""

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
path = HERE / "cpu_v2/kernel/ptoas/dynamic_csa_full.cpp"
source = path.read_text()
cube = source[:source.index("AICORE void vector_schedule(")]
constants = {k: int(v) for k, v in re.findall(r"const int64_t (v\d+) = (-?\d+);", cube)}
addresses = {}
for variable, value in re.findall(r"(v\d+) = \(uint64_t\) (v\d+);", cube):
    if value in constants:
        addresses[variable] = constants[value]
tile_addresses = {tile: addresses[address] for tile, address in
                  re.findall(r"TASSIGN\((v\d+), (v\d+)\);", cube) if address in addresses}
tiles = {}
pattern = re.compile(r"// pto: %(qr(?:0|_current|_next)_inline\w+|b(?:0|_current|_next)_inline\w+)\n"
                     r"\s*;\n\s*(v\d+) = Tile<TileType::(Left|Right), int8_t, (\d+), (\d+),")
for name, var, space, rows, cols in pattern.findall(cube):
    tiles[var] = dict(name=name, space=space, shape=[int(rows),int(cols)], address=tile_addresses[var])
left = sorted({v['address'] for v in tiles.values() if v['space']=='Left'})
right = sorted({v['address'] for v in tiles.values() if v['space']=='Right'})
assert left == [0,16384] and right == [0,32768], (left,right)
calls = []
for match in re.finditer(r"TMATMUL(?:_ACC)?\(([^\n]+)\);", cube):
    args = [v.strip() for v in match[1].split(',')]
    if args[-1] not in tiles:
        continue
    current = tiles[args[-1]]
    assert current['space'] == 'Right'
    before = cube[:match.start()]
    ready = re.findall(r"wait_flag\(PIPE_MTE1, PIPE_M, EVENT_ID(\d+)\);",before)[-1]
    expected = current['address']//32768
    assert int(ready)==expected
    calls.append(dict(line=cube[:match.start()].count('\n')+1,call=match[0],right_address=current['address'],ready_event=int(ready)))
assert len(calls)==8
assert [x['right_address'] for x in calls]==[0,32768]*4
accs=re.findall(r"Tile<TileType::Acc, int32_t, 128, 256,[^\n]*CompactMode::Normal>\(([^\n]+)\)",cube)
assert accs, '动态尾行累加器必须保留compact语义'
report=dict(passed=True,source=str(path.resolve()),left_addresses=left,right_addresses=right,
            actual_two_l0_buffers=True,current_mma_waits_current_slot=True,matmul_calls=calls,
            dynamic_acc_valid_shape_arguments=sorted(set(accs)),tiles=tiles,
            limitations=['本审阅只证明生成地址和发射依赖，不证明真实重叠收益',
                         'L1 Mat仍有复用；不声称GM搬运拥有相同双槽',
                         'K128增加矩阵指令次数，必须用完整CSA L1判断取舍'])
(HERE/'CODEGEN_REVIEW.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('passed','left_addresses','right_addresses','actual_two_l0_buffers')},ensure_ascii=False))
