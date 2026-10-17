"""标准库有限模型/完整AST审计；不导入PyPTO、Torch或设备库。"""

import ast
import difflib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
MAIN = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1')
REFERENCE = HERE / 'source_reference'
CANDIDATE = HERE / 'source_candidate'
OLD = ast.parse('signal = wait_row_owners(signal, PHASE_IDX_QUERY_HAD, CORES, generation, 0, tokens, 1, GROUPS, 0, 1)').body[0]
NEW = ast.parse('if worker < tokens:\n    signal = wait_producer(signal, PHASE_IDX_QUERY_HAD, worker % GROUPS, CORES, generation)').body[0]


class RestoreOnlyWait(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_If(self, node):
        if ast.dump(node) == ast.dump(NEW):
            self.count += 1
            return OLD
        return self.generic_visit(node)


def constant_ints(path):
    values = {}
    for stmt in ast.parse(path.read_text()).body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            if isinstance(stmt.value, ast.Constant) and type(stmt.value.value) is int:
                values[stmt.targets[0].id] = stmt.value.value
    return values


def main():
    source_rows = []
    for ref in sorted(REFERENCE.glob('*.py')):
        cand = CANDIDATE / ref.name
        official = subprocess.check_output(['git', 'show', '66882634:vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/' + ref.name], cwd=MAIN)
        assert ref.read_bytes() == official
        restore = RestoreOnlyWait()
        old_ast = ast.parse(ref.read_text())
        new_ast = restore.visit(ast.parse(cand.read_text()))
        equal = ast.dump(old_ast, include_attributes=False) == ast.dump(new_ast, include_attributes=False)
        assert equal
        source_rows.append({'file': ref.name, 'reference_exact_668_text': True,
                            'candidate_text_equal': ref.read_bytes() == cand.read_bytes(),
                            'full_ast_equal_restoring_only_wait': equal, 'restored_wait_nodes': restore.count})
    assert len(source_rows) == 33
    assert [x['file'] for x in source_rows if not x['candidate_text_equal']] == ['vector_schedule.py']
    assert sum(x['restored_wait_nodes'] for x in source_rows) == 1
    (HERE / 'candidate.diff').write_text(''.join(difflib.unified_diff(
        (REFERENCE / 'vector_schedule.py').read_text().splitlines(True),
        (CANDIDATE / 'vector_schedule.py').read_text().splitlines(True),
        fromfile='668/vector_schedule.py', tofile='candidate/vector_schedule.py')))

    constants = constant_ints(REFERENCE / 'runtime_abi.py')
    tiles = constant_ints(REFERENCE / 'indexer_query_stages.py')
    groups, vectors = constants['GROUPS'], constants['VECTORS']
    assert (groups, vectors, constants['CORES']) == (24, 48, 72)
    assert tiles['QH_MM_TILE'] == tiles['QH_QUANT_TILE'] == 64
    assert tiles['QH_HEAD_DIM_TILE'] == 64
    config = ast.parse((MAIN / 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa/config.py').read_text())
    flash = next(n.value for n in config.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FLASH' for t in n.targets))
    dims = {k.arg: k.value.value for k in flash.keywords if isinstance(k.value, ast.Constant)}
    heads, width = dims['index_n_heads'], dims['index_head_dim']
    assert (heads, width) == (64, 128)
    functions = {n.name: n for n in ast.parse((REFERENCE / 'indexer_query_stages.py').read_text()).body if isinstance(n, ast.FunctionDef)}
    had, quant = functions['indexer_query_hadamard_stage'], functions['indexer_query_quant_stage']
    had_loop = next(n for n in ast.walk(had) if isinstance(n, ast.For))
    quant_loop = next(n for n in ast.walk(quant) if isinstance(n, ast.For))
    assert ast.unparse(had_loop.iter) == 'pl.range(qh_worker, bs_heads // QH_MM_TILE, WORKERS)'
    assert ast.unparse(quant_loop.iter) == 'pl.range(qh_quant_worker, bs_heads // QH_QUANT_TILE, WORKERS)'
    loads = [ast.unparse(n) for n in ast.walk(quant) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == 'qh_acc_gm']
    assert loads == ['qh_acc_gm[o0:o0 + QH_QUANT_TILE, 0:IDX_HEAD_DIM]']
    # 唯一GM读取跨度=一token的全部64×128；第二维两个64列归约仅访问已载入tile。
    observed = []
    token_visits = 0
    cases = 0
    for tokens in range(4097):
        producer_of = [-1] * tokens
        for producer in range(groups):
            for token in range(producer, tokens * heads // tiles['QH_MM_TILE'], groups):
                assert producer_of[token] == -1
                producer_of[token] = producer
        assert all(p >= 0 for p in producer_of)
        consumers = [-1] * tokens
        union = set()
        longest_wave = 0
        for worker in range(vectors):
            consumed = list(range(worker, tokens * heads // tiles['QH_QUANT_TILE'], vectors))
            actual = {producer_of[token] for token in consumed}
            expected = {worker % groups} if worker < tokens else set()
            assert actual == expected
            union.update(actual)
            longest_wave = max(longest_wave, len(consumed))
            for token in consumed:
                assert consumers[token] == -1
                consumers[token] = worker
                start, end = token * heads, (token + 1) * heads
                assert 0 <= start < end <= tokens * heads
                assert (end - start) * width * 4 == 32768
                assert token == start // heads == (end - 1) // heads
                assert producer_of[token] == worker % groups
                token_visits += 1
            cases += 1
        assert all(w >= 0 for w in consumers)
        # 后继仍等待全部active QUANT owner；其前驱Had并集恢复全部active AIC。
        assert union == set(range(min(tokens, groups)))
        observed.append({'T': tokens, 'active_had_aic': min(tokens, groups),
                         'active_quant_aiv': min(tokens, vectors), 'max_quant_waves': longest_wave,
                         'old_signal_wait_invocations': vectors * min(tokens, groups),
                         'new_signal_wait_invocations': min(tokens, vectors),
                         'old_payload_acquires': vectors if tokens else 0,
                         'new_payload_acquires': min(tokens, vectors), 'owner_exact': True,
                         'unique_token_writes': True, 'successor_full_had_union_restored': True})
    (HERE / 'owner_model.json').write_text(json.dumps({
        'range_inclusive': [0, 4096], 'aiv_T_cases': cases, 'token_read_write_blocks': token_visits,
        'contract': {'groups': groups, 'vectors': vectors, 'heads': heads, 'head_dim': width},
        'zero_note': 'T0仅验证空循环/空集合；正式Host select_policy拒绝非正tokens，不声称整层T0支持',
        'method': '显式枚举原Had/quant的全部token波；不枚举Tensor数值，不模拟硬件缓存或耗时',
        'cases': observed,
    }, ensure_ascii=False, indent=2) + '\n')
    (HERE / 'SOURCE_AUDIT.json').write_text(json.dumps({
        'status': 'PASS', 'scope': '完整33源AST及owner有限模型；未CPU lowering/上卡',
        'files': source_rows, 'actual_had_loop': ast.unparse(had_loop.iter),
        'actual_quant_loop': ast.unparse(quant_loop.iter), 'actual_quant_external_load': loads,
        'groups': groups, 'vectors': vectors, 'no_cube_candidate_combination': True,
        'selected_T': [observed[t] for t in (0, 1, 16, 24, 25, 47, 48, 49, 96, 144, 168, 192, 4096)],
    }, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'source_files': len(source_rows), 'changed_files': ['vector_schedule.py'],
                      'aiv_T_cases': cases, 'token_read_write_blocks': token_visits, 'passed': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
