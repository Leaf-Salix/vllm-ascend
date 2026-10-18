# SPDX-License-Identifier: Apache-2.0
"""执行真实前置几何控制代码，对照独立请求/页合法性oracle；不模拟设备数学。"""

import ast
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent


class Tensor:
    def __init__(self, shape, fill=37, values=None):
        self.shape, self.fill, self.values, self.writes = shape, fill, values or {}, []
        self.valid = shape

    def read(self, idx):
        assert len(idx) == len(self.shape) and all(0 <= i < s for i, s in zip(idx, self.shape))
        return self.values.get(tuple(idx), self.fill)

    def write(self, idx, value):
        self.read(idx)
        assert -(1 << 31) <= value < (1 << 31)
        self.values[tuple(idx)] = value
        self.writes.append(tuple(idx))


def vector(values):
    return Tensor((len(values),), values={(i,): v for i, v in enumerate(values)})


def valid_pages(table, request, length, pages):
    needed = (length // 4 + 31) // 32
    return int(length <= 16777216 and 0 < pages <= 16777216 and needed <= table.shape[1]
               and all(0 <= table.read((request, col)) < pages for col in range(needed)))


def load_control():
    nodes = []
    for file, names in [('request_geometry.py', ['request_bounds', 'query_request']),
                        ('attention_plan_stages.py', ['attention_geometry_prepare_stage'])]:
        for n in ast.parse((HERE/'source_candidate'/file).read_text()).body:
            if isinstance(n, ast.FunctionDef) and n.name in names:
                n = copy.deepcopy(n)
                n.decorator_list, n.returns = [], None
                for a in n.args.args:
                    a.annotation = None
                nodes.append(n)
    def store(tile, start, dst):
        for i in range(tile.valid[0]):
            for j in range(tile.valid[1]):
                dst.write((start[0]+i, start[1]+j), tile.read((i,j)))
        return dst
    def validshape(tile, rows, cols):
        tile.valid = (rows, cols)
        return tile
    pl = SimpleNamespace(INDEX=None, INT32=None, cast=lambda v,t:int(v), min=min, max=max,
                         range=range, read=lambda t,i:t.read(i), store=store, set_validshape=validshape,
                         tensor=SimpleNamespace(dim=lambda t,i:t.shape[i], read=lambda t,i:t.read(i)),
                         tile=SimpleNamespace(full=lambda s,dtype,value:Tensor(tuple(s),fill=value),
                                              write=lambda t,i,v:t.write(i,v)))
    env = dict(pl=pl, ROWS=8, MAX_CANDIDATES=262144,
               EARLY_QUERY_OWNER_COL=0, EARLY_QUERY_BEGIN_COL=1, EARLY_QUERY_POSITION_COL=2,
               EARLY_QUERY_PAGES_COL=3, EARLY_QUERY_VISIBLE_COL=4,
               compressed_pages_all_valid=valid_pages)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                 str(HERE/'source_candidate/attention_plan_stages.py'),'exec'),env)
    return env['attention_geometry_prepare_stage']


def oracle(q, bounds, lengths, positions, tokens, table, pages):
    owners = []
    for r, length in enumerate(lengths):
        if r+1 < len(bounds) and length > 0:
            begin, end = bounds[r:r+2]
            if 0 <= begin < end <= tokens and end-begin <= length and begin <= q < end:
                owners.append((r,begin,end))
    if len(owners) == 1:
        r,begin,end = owners[0]
        position = positions[q]
        if position == lengths[r]-(end-begin)+q-begin:
            return [r,begin,position,valid_pages(table,r,lengths[r],pages),
                    max(min(lengths[r]//4,(position+1)//4,262144),0)]
    return [-1,0,0,0,0]


def main():
    control = load_control()
    rng = random.Random(20261018)
    queries = 0
    for case in range(320):
        sizes = [rng.choice([1,2,5,6,7,0]) for _ in range(rng.randrange(1,10))]
        bounds = [0]
        for s in sizes:
            bounds.append(bounds[-1]+s)
        tokens = bounds[-1] + rng.randrange(0,8)
        lengths = [rng.choice([0,1,128,8192,131072,2147483647]) for _ in sizes]
        if case % 7 == 0:
            bounds[rng.randrange(len(bounds))] = rng.choice([-1,tokens+2,0])
        if case % 11 == 0:
            bounds = bounds[:-1]
        columns, pages = rng.choice([0,1,64,1025]), rng.choice([0,16,2048,16777217])
        table = Tensor((len(sizes),columns),fill=1)
        if columns and case % 3 == 0:
            table.values[(rng.randrange(len(sizes)),rng.randrange(columns))] = rng.choice([-1,pages])
        positions = [rng.randrange(-2,132000) for _ in range(tokens)]
        for r in range(len(sizes)):
            if r+1 < len(bounds):
                for q in range(max(0,bounds[r]),min(tokens,bounds[r+1])):
                    positions[q] = lengths[r]-(bounds[r+1]-bounds[r])+q-bounds[r]
        if tokens and case % 5 == 0:
            positions[rng.randrange(tokens)] += 1
        dst = Tensor((tokens+8,512),fill=1234567)
        for worker in range(48):
            control(vector(positions),vector(bounds),vector(lengths),table,dst,pages,worker,48)
        assert len(dst.writes) == tokens*8
        assert set(dst.writes) == {(q,c) for q in range(tokens) for c in range(8)}
        for q in range(tokens):
            expected = oracle(q,bounds,lengths,positions,tokens,table,pages)+[0,0,0]
            assert [dst.read((q,c)) for c in range(8)] == expected, (case,q)
            queries += 1
    # 页检查函数未改变；late plan的向量数学调用与原表达逐项相同。
    functions = []
    for side in ['source_reference','source_candidate']:
        functions.append({n.name:n for n in ast.parse((HERE/side/'attention_plan_stages.py').read_text()).body
                          if isinstance(n,ast.FunctionDef)})
    assert ast.dump(functions[0]['compressed_pages_all_valid']) == ast.dump(functions[1]['compressed_pages_all_valid'])
    ops = {'pl.mul','pl.add','pl.sub','pl.minimum','pl.maximum','pl.row_expand_mul','pl.col_expand_mul',
           'pl.row_max','pl.store','pl.tile.assemble'}
    math = [[ast.dump(n) for n in ast.walk(fs['attention_slot_plan_stage'])
             if isinstance(n,ast.Call) and ast.unparse(n.func) in ops] for fs in functions]
    assert math[0] == math[1]
    report = dict(cases=320,queries=queries,page_function_ast_unchanged=True,
                  late_plan_vector_math_ast_unchanged=True,only_actual_rows_first_eight_columns_written=True,
                  coverage=['S1/S5/S6及混合长度','dummy/空T','非法/缺失/重叠Q','陈旧位置',
                            '短页表/页空洞/物理范围边界','INT32位置上界','尾行和未使用列保护'],
                  hardware_sync_validated=False,performance_evidence=False)
    (HERE/'PREPLAN_CONTROL.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
