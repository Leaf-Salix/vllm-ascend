"""四 Scalar 几何缓存的纯 Python 同源模型与可逆 AST 审计。

只从冻结源提取原 geometry 函数及候选缓存控制块，移除类型注解后解释执行。
不导入 PyPTO、不进行 lowering、不接触 NPU。它证明指定循环语义及源码变化范围，
不能证明后端寄存器分配、设备缓存/事件语义或性能。
"""

import ast
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
NAME = 'attention_qk_pv_vector_lane_stage'
STATES = ['query', 'owner', 'position', 'active']


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def tick_loop(fn):
    return next(n for n in fn.body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
                and ast.unparse(n.target.elts[0]) == 'qk_tick')


def assign_name(node):
    if isinstance(node, ast.Assign):
        return ast.unparse(node.targets[0])
    return ''


def drop_docs(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body.pop(0)
    return tree


def source_audit():
    refs = HERE / 'source_reference'
    cands = HERE / 'source_candidate'
    files = sorted(refs.glob('*.py'))
    assert len(files) == len(list(cands.glob('*.py'))) == 33
    changed = [p.name for p in files if p.read_bytes() != (cands / p.name).read_bytes()]
    assert changed == ['attention_stream_stages.py']
    old = ast.parse((refs / changed[0]).read_text())
    new = ast.parse((cands / changed[0]).read_text())
    restored = copy.deepcopy(new)
    old_fn, fn = function(old, NAME), function(restored, NAME)
    loop, old_loop = tick_loop(fn), tick_loop(old_fn)
    pos = fn.body.index(loop)
    assert [assign_name(n) for n in fn.body[pos-4:pos]] == [f'geometry_{s}' for s in STATES]
    del fn.body[pos-4:pos]
    assert [ast.unparse(n) for n in loop.target.elts[1].elts[4:]] == [f'geometry_{s}_iter' for s in STATES]
    loop.target.elts[1].elts = loop.target.elts[1].elts[:4]
    init = next(k.value for k in loop.iter.keywords if k.arg == 'init_values')
    assert [ast.unparse(n) for n in init.elts[4:]] == [f'geometry_{s}' for s in STATES]
    init.elts = init.elts[:4]
    real = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'qk_tick < qk_block_count')
    old_real = next(n for n in old_loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'qk_tick < qk_block_count')
    valid, old_valid = real.body[0], old_real.body[0]
    cache = next(n for n in valid.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'qk_t != geometry_query_iter')
    index = valid.body.index(cache)
    assert [assign_name(n) for n in valid.body[index+1:index+4]] == ['qk_b', 'qk_pos', 'q_active']
    old_geometry = next(n for n in old_valid.body if assign_name(n) == '(qk_b, qk_pos, q_active)')
    valid.body[index:index+4] = [copy.deepcopy(old_geometry)]
    assert assign_name(valid.body[-1]).startswith('(geometry_valid_query,')
    valid.body.pop()
    assert len(valid.orelse) == 1 and assign_name(valid.orelse[0]).startswith('(geometry_valid_query,')
    valid.orelse.clear()
    assert len(real.body) == 2 and assign_name(real.body[-1]).startswith('(geometry_after_query,')
    real.body.pop()
    assert len(real.orelse) == 1 and assign_name(real.orelse[0]).startswith('(geometry_after_query,')
    real.orelse.clear()
    end = loop.body[-1]
    assert [ast.unparse(n) for n in end.targets[0].elts[4:]] == [f'geometry_{s}' for s in STATES]
    assert [ast.unparse(n) for n in end.value.args[4:]] == [f'geometry_after_{s}' for s in STATES]
    end.targets[0].elts = end.targets[0].elts[:4]
    end.value.args = end.value.args[:4]
    assert ast.dump(drop_docs(restored)) == ast.dump(drop_docs(copy.deepcopy(old)))
    return new, {'changed_files': changed, 'all_33_sources_from_668': True,
                 'inverse_ast_equals_original_except_docstrings': True,
                 'guard_math_store_ffts_publish_unchanged': True,
                 'additional_loop_scalars': 4, 'new_tile_gm_root_phase': 0}


class PL:
    INDEX = 'INDEX'
    INT32 = 'INT32'

    def __init__(self):
        self.reads = 0
        self.tensor = SimpleNamespace(dim=lambda a, axis: len(a) if axis == 0 else len(a[0]), read=self.read)

    def read(self, value, coordinates):
        self.reads += 1
        for coordinate in coordinates:
            assert coordinate >= 0
            value = value[coordinate]
        return value

    @staticmethod
    def cast(value, dtype):
        return int(value)

    range = staticmethod(range)
    yield_ = staticmethod(lambda *values: values)


def strip_types(fn):
    fn = copy.deepcopy(fn)
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    return fn


def build_model(module):
    refs = HERE / 'source_reference'
    geometry = ast.parse((refs / 'request_geometry.py').read_text())
    plan = ast.parse((refs / 'attention_plan_stages.py').read_text())
    functions = [strip_types(function(geometry, n)) for n in ['request_bounds', 'query_request']]
    functions.append(strip_types(function(plan, 'attention_query_geometry')))
    pl = PL()
    env = {'pl': pl, 'QUERY_OWNER_COL': 5, 'QUERY_POSITION_COL': 6}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), '<same-source-geometry>', 'exec'), env)
    actual_geometry = env['attention_query_geometry']
    calls = []

    def record_geometry(*args):
        calls.append(args[4])
        return actual_geometry(*args)

    env['attention_query_geometry'] = record_geometry
    fn = function(module, NAME)
    loop = tick_loop(fn)
    real = copy.deepcopy(next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'qk_tick < qk_block_count'))
    valid = real.body[0]
    cache = next(n for n in valid.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'qk_t != geometry_query_iter')
    at = valid.body.index(cache)
    # 只执行实际候选的几何控制和yield；不模拟KV/向量数学/硬件事件。
    valid.body = valid.body[at:at+4] + ast.parse('observed = (qk_b, qk_pos, q_active)').body + [valid.body[-1]]
    stub = ast.parse('''
def step(state, qk_tick, qk_block_count, qk_t, qk_sb, position_ids, bounds, lengths, valid_block_mask):
    geometry_query_iter, geometry_owner_iter, geometry_position_iter, geometry_active_iter = state
    observed = None
    CACHE_QUERY_METADATA = False
    return (), observed
''').body[0]
    stub.body[-1:-1] = [real]
    stub.body[-1] = ast.parse('return (geometry_after_query, geometry_after_owner, geometry_after_position, geometry_after_active), observed').body[0]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), '<candidate-cache-block>', 'exec'), env)
    pos = fn.body.index(loop)
    initial = ast.Module(body=copy.deepcopy(fn.body[pos-4:pos]), type_ignores=[])

    def init():
        # 每个模拟调用实际执行候选的四条初始化，而非延续上次state。
        init_env = {'pl': pl}
        exec(compile(ast.fix_missing_locations(initial), '<candidate-initializers>', 'exec'), init_env)
        return tuple(init_env[f'geometry_{s}'] for s in STATES)

    return pl, env, calls, actual_geometry, init


def main():
    source, audit = source_audit()
    pl, env, calls, geometry, initialize = build_model(source)
    checked_invocations = checked_ticks = checked_observations = 0
    aggregate_original_calls = aggregate_candidate_calls = 0

    def invoke(tokens, core, mask, bounds, lengths, positions):
        nonlocal checked_invocations, checked_ticks, checked_observations
        nonlocal aggregate_original_calls, aggregate_candidate_calls
        state = initialize()
        qcount = (tokens + 24 - 1 - core) // 24
        count = qcount * 5
        seen, observed_trace = set(), []
        calls.clear()
        original_calls = 0
        for tick in range(count + 3):
            query, block = core + (tick // 5)*24, tick % 5
            expected = None
            if tick < count and mask[query][block] > 0:
                expected = geometry(positions, bounds, lengths, mask, query, False)
                original_calls += 1
                seen.add(query)
            before_calls = len(calls)
            state, observed = env['step'](state, tick, count, query, block, positions, bounds, lengths, mask)
            assert observed == expected
            if tick >= count or (tick < count and mask[query][block] <= 0):
                assert len(calls) == before_calls
            if observed is not None:
                assert state == (query, *expected)
                checked_observations += 1
            observed_trace.append(observed)
            checked_ticks += 1
        assert set(calls) == seen and len(calls) == len(seen)
        aggregate_original_calls += original_calls
        aggregate_candidate_calls += len(calls)
        checked_invocations += 1
        return observed_trace, original_calls, len(calls)

    # 两个同核query（0、24）的全部32×32 mask组合，包含空首块/仅末块/全空。
    tokens = 25
    base_bounds, base_lengths = [0, 1, 9, 25], [100, 200, 300]
    base_positions = [99] + list(range(192, 200)) + list(range(284, 300))
    cases = [
        ('mixed_active', base_bounds, base_lengths, base_positions),
        ('dummy_stale_position', base_bounds, [0, 200, 300], [999999] + base_positions[1:]),
        ('causal_mismatch', base_bounds, base_lengths, base_positions[:-1]+[999999]),
        ('overlap_rejected', [0, 25, 1, 25], [100, 200, 300], base_positions),
    ]
    for _, bounds, lengths, positions in cases:
        for first, second in itertools.product(range(32), repeat=2):
            mask = [[0]*16 for _ in range(tokens)]
            mask[0][:5] = [(first >> bit) & 1 for bit in range(5)]
            mask[24][:5] = [(second >> bit) & 1 for bit in range(5)]
            invoke(tokens, 0, mask, bounds, lengths, positions)

    # T与24核波次/非整波边界；没有按B或S决定缓存。T0仅为几何模型，非Host支持声明。
    token_cases = [0,1,2,7,8,9,16,17,23,24,25,31,32,33,47,48,49,95,96,97,143,144,145,192,193,384,385]
    for tokens in token_cases:
        bounds, lengths = [0, tokens], [tokens+100]
        positions = list(range(100, tokens+100))
        for pattern in [0,1,2,16,21,31]:
            mask = [[(pattern >> (bit % 5)) & 1 for bit in range(16)] for _ in range(tokens)]
            for core in range(24):
                invoke(tokens, core, mask, bounds, lengths, positions)

    # 同T和地址概念下A→B→A：Q改变，查询24的owner/position随之改变，每次重新初始化。
    tokens = 25
    mask = [[1]*16 for _ in range(tokens)]
    first = invoke(tokens, 0, mask, base_bounds, base_lengths, base_positions)
    middle = invoke(tokens, 0, mask, [0,25], [50], list(range(25,50)))
    last = invoke(tokens, 0, mask, base_bounds, base_lengths, base_positions)
    assert first == last and first[0] != middle[0]
    result = {
        'status':'PASS', 'source_audit':audit, 'pypto_compile':False, 'device_execution':False,
        'model':'从真实候选提取缓存分支和初始化；原geometry/request helper同源Python解释',
        'mask_pairs':1024, 'geometry_variants':[c[0] for c in cases],
        'token_boundaries':token_cases, 'invocations':checked_invocations,
        'ticks':checked_ticks, 'valid_gather_observations':checked_observations,
        'original_geometry_calls':aggregate_original_calls, 'candidate_geometry_calls':aggregate_candidate_calls,
        'all_empty_and_inactive_and_drain_safe':True, 'A_B_A_query_metadata_reset':True,
        'limitations':['未执行PyPTO/后端/设备','不模拟FFTS/DMA/Scalar缓存或寄存器分配',
                       '不推断性能；T0只为模型边界，Host仍拒绝非正T'],
    }
    (HERE/'source_model_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
