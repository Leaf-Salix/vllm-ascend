#!/usr/bin/env python3
"""从正式 668 的 33 份源码冻结一个只改 HAD 前等待的独立候选。

本脚本只读 Git，写自己的实验目录；拒绝覆盖已有源码。数学、参数、信号
发布与动态范围不变，只让有 HAD token 的 AIC 等待它真正依赖的 16 个 AIV。
这里不导入 PyPTO，不编译或运行设备。
"""
import ast
import copy
import difflib
import json
from pathlib import Path
import subprocess
import textwrap

C = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1')
P = 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
HERE = Path(__file__).resolve().parent
COMMIT = '66882634'
OLD = '''    # query的8行×4head反量化块共有ceil(T/8)*16个，不把T误当作其owner数量。
    query_rope_tasks = ((tokens + 7) // 8) * 16
    signal = wait_row_owners(
        signal, PHASE_IDX_QUERY_ROPE, CORES, generation, 0, query_rope_tasks, 1, VECTORS, GROUPS, 1
    )
'''
NEW = '''    # Indexer query Hadamard的64行正好是一个token的64个head；AIC worker
    # 处理t=worker+24*wave。RoPE生产unit=floor(t/8)*16+head_group，
    # 每unit写8行×4head，owner为unit%min(ceil(T/8)*16,48)。
    # T>=17时已有48个生产者；24-token步长令unit每wave恰加48，故本核
    # 全部token始终只依赖[16*floor(worker/8), +16)这16个AIV。
    # T<=16为单wave，实际只16/32个生产者，但活跃worker的上述区间完全
    # 位于实际生产者内；wait中的48模数不会改变owner。非8倍数尾行同样
    # 由原unit写回，不能缩减为某一个head或只等第一wave的假完成标记。
    # first_row/count在这里以RoPE unit为单位，rows_per_tile=1；AIV物理
    # core为GROUPS+owner。仍等这些核完成本阶段全部wave，再统一执行原
    # fence/cacheinvalid/fence，既不靠经过时间保证可见性，也不新增ACK。
    # worker>=T没有任何HAD token，因此不读idx_query_rope_heads；仅跳过
    # 此无关等待，原空HAD发布保留。后续QB只读前面已等待的QR，不能把本
    # 次优化理解成删除QB/QR的真实依赖。24AIC/48AIV及64head/8行/4head
    # 是当前调度合同；若将来改变这些常量，必须重新推导本等待集合。
    if worker < tokens:
        signal = wait_row_owners(
            signal, PHASE_IDX_QUERY_ROPE, CORES, generation,
            (worker // 8) * 16, 16, 1, VECTORS, GROUPS, 1
        )
'''


def dump(x):
    return ast.dump(x, include_attributes=False)


def main():
    reference, candidate = HERE/'source_reference', HERE/'source_candidate'
    if reference.exists() or candidate.exists():
        raise SystemExit('冻结源码已存在；拒绝覆盖，请另建实验目录。')
    paths = subprocess.check_output(['git', 'ls-tree', '--name-only', COMMIT, P+'/'], cwd=C, text=True).splitlines()
    assert len(paths) == 33 and all(x.endswith('.py') for x in paths)
    reference.mkdir()
    candidate.mkdir()
    sources = {}
    for path in paths:
        body = subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'], cwd=C)
        name = Path(path).name
        (reference/name).write_bytes(body)
        (candidate/name).write_bytes(body)
        sources[name] = body.decode()
    text = sources['cube_schedule.py']
    assert text.count(OLD) == 1
    changed = text.replace(OLD, NEW)
    (candidate/'cube_schedule.py').write_text(changed)
    (HERE/'candidate.patch').write_text(''.join(difflib.unified_diff(text.splitlines(True), changed.splitlines(True),
                                                                  fromfile='source_reference/cube_schedule.py',
                                                                  tofile='source_candidate/cube_schedule.py')))
    old, new = ast.parse(text), ast.parse(changed)
    function_old = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'cube_schedule')
    function_new = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'cube_schedule')
    assert dump(function_old.args) == dump(function_new.args)
    # 使用dedent还原函数体片段；if后的返回/发布和所有其他节点必须逐AST相同。
    old_fragment = ast.parse(textwrap.dedent(OLD)).body
    replacement = ast.parse(textwrap.dedent(NEW)).body
    candidate_tree = copy.deepcopy(new)
    fn = next(n for n in candidate_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cube_schedule')
    indices = [i for i, stmt in enumerate(fn.body) if dump(stmt) == dump(replacement[0])]
    assert len(indices) == 1
    at = indices[0]
    fn.body[at:at+1] = old_fragment
    assert dump(old) == dump(candidate_tree)
    changes = [p.name for p in reference.glob('*.py') if p.read_bytes() != (candidate/p.name).read_bytes()]
    assert changes == ['cube_schedule.py']
    assert 'GROUPS = 24' in sources['runtime_abi.py'] and 'VECTORS = 48' in sources['runtime_abi.py']
    assert 'QH_MM_TILE = 64' in sources['indexer_query_stages.py']
    facts = {
        'base_commit': subprocess.check_output(['git', 'rev-parse', COMMIT], cwd=C, text=True).strip(),
        'status': 'source_ast_pass_only', 'compile_performed': False, 'device_performed': False,
        'files': 33, 'changed_files': changes, 'other_files_byte_identical': 32,
        'restore_only_wait_block_equals_original_module_ast': True,
        'cube_function_signature_unchanged': True, 'helper_math_and_all_publish_calls_unchanged': True,
        'host_policy_entry_binding_runtime_abi_byte_identical': True,
        'source_reference': str(reference), 'source_candidate': str(candidate),
        'contract': {'AIC': 24, 'AIV': 48, 'heads': 64, 'rope_token_tile': 8, 'rope_head_tile': 4,
                     'HAD_rows': 64, 'first_row_unit': 'RoPE logical unit', 'rows_per_tile': 1,
                     'producer_workers': 48, 'producer_base': 24, 'producer_stride': 1},
    }
    (HERE/'source_audit.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
