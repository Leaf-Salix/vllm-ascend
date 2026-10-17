#!/usr/bin/env python3
"""冻结 e581 正式源，依据同次 plan 合同删除消费者两个冗余地址 guard。

只操作本目录 source_before/source_candidate；拒绝覆盖，不导入 PyPTO 或设备。
保留最终 INT32 逻辑索引、页表读取、逐行 gather、零 KV、数学和所有同步。
AST 证据将唯一两处 If 展平后，要求其余执行语义完全相同。
"""

import ast
import collections
import copy
import difflib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
C = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1')
PACKAGE = 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
BASE = 'e58169ec'


def dump(node):
    return ast.dump(node, include_attributes=False)


class StripDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        body = getattr(node, 'body', None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            if isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                node.body = body[1:]
        return node


class RemoveOnlyProvedGuards(ast.NodeTransformer):
    def __init__(self):
        self.removed = []

    def visit_If(self, node):
        test = ast.unparse(node.test)
        self.generic_visit(node)
        if test in (
            'qk_column < pl.tensor.dim(cmp_block_table, 1)',
            'qk_page >= 0 and qk_page < pl.tensor.dim(cmp_kv_flat, 0) // BLOCK_SIZE',
        ):
            assert not node.orelse
            self.removed.append(test)
            return node.body
        return node


def calls(node):
    return collections.Counter(
        ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)
    )


def main():
    commit = subprocess.check_output(['git', 'rev-parse', BASE], cwd=C, text=True).strip()
    names = subprocess.check_output(
        ['git', 'ls-tree', '--name-only', commit + ':' + PACKAGE], cwd=C, text=True
    ).splitlines()
    names = sorted(name for name in names if name.endswith('.py'))
    assert len(names) == 33
    before, after = OUT / 'source_before', OUT / 'source_candidate'
    before.mkdir(exist_ok=False)
    after.mkdir(exist_ok=False)
    for name in names:
        content = subprocess.check_output(['git', 'show', commit + ':' + PACKAGE + '/' + name], cwd=C)
        assert content == (C / PACKAGE / name).read_bytes(), '正式工作区与冻结版本不同：' + name
        (before / name).write_bytes(content)
        (after / name).write_bytes(content)
    name = 'attention_stream_stages.py'
    old = (before / name).read_text()
    target = '''                                    if qk_column < pl.tensor.dim(cmp_block_table, 1):
                                        qk_page = pl.cast(pl.read(cmp_block_table, [qk_b, qk_column]), pl.INDEX)
                                        if qk_page >= 0 and qk_page < pl.tensor.dim(cmp_kv_flat, 0) // BLOCK_SIZE:
                                            qk_src = qk_page * BLOCK_SIZE + qk_ridx % BLOCK_SIZE
                                            qk_kv_half = pl.gather_row(
                                                qk_kv_half, cmp_kv_flat, [qk_row, 0], [qk_src, 0], [1, HEAD_DIM]
                                            )'''
    replacement = '''                                    # 同次plan已检查这个最终INT32逻辑索引对应的列和物理页。
                                    # 快路径仅在FP32精确区间检查整个页前缀，且仍验证
                                    # final_index<upper；空洞/短表/大值回退逐个检查最终
                                    # c_page_indices。两条路径均保证非负输出的页地址合法。
                                    # 依赖外层等PHASE_ATTN_PLAN并刷新缓存，且本次
                                    # Q/L/positions/页表及cache容量保持不变；不能将
                                    # 任意外部Top-K或上一代indices直接接入本helper。
                                    qk_page = pl.cast(pl.read(cmp_block_table, [qk_b, qk_column]), pl.INDEX)
                                    qk_src = qk_page * BLOCK_SIZE + qk_ridx % BLOCK_SIZE
                                    qk_kv_half = pl.gather_row(
                                        qk_kv_half, cmp_kv_flat, [qk_row, 0], [qk_src, 0], [1, HEAD_DIM]
                                    )'''
    assert old.count(target) == 1
    new = old.replace(target, replacement)
    old_note = '''    gather再次检查Q/L活动、causal position、页表列和物理页上下界；inactive
    不访问cache。plan与metadata必须保持一致，不能一侧私自改mask而破坏事件对数。'''
    new_note = '''    gather仍检查Q/L活动和causal position；SWA仍检查页表列与物理页上下界。
    压缩候选只在同次plan已验证的INT32 indices>=0时查页并采样，复用plan的
    列容量/物理页安全结论，不重复两个guard。索引含义仍是逻辑行号，未增加
    物理行plan或地址缓存。plan与消费者必须使用同一批只读Q/L/positions/页表，
    compressed_pages必须等于dim(cmp_kv_flat,0)//32，且消费前取得本代plan的
    阶段发布并完成本核缓存维护。这个helper不能直接消费外部或过期indices。
    inactive不访问cache，非法索引保留零KV；不改mask或任一事件配对。'''
    assert new.count(old_note) == 1
    new = new.replace(old_note, new_note)
    old_comment = '''                        # indices已经包含plan的causal/页mask，保留顺序并再做
                        # 地址上界保护；非法行留在本轮初始化的零KV中。'''
    new_comment = '''                        # indices来自同次已发布plan，已包含causal/列/物理页mask。
                        # 保留逻辑顺序、非负判断与原页读取；负索引保持本轮零KV。'''
    assert new.count(old_comment) == 1
    new = new.replace(old_comment, new_comment)
    old_ast, new_ast = ast.parse(old), ast.parse(new)
    transform = RemoveOnlyProvedGuards()
    expected = transform.visit(copy.deepcopy(old_ast))
    assert len(transform.removed) == 2
    assert dump(StripDocstrings().visit(expected)) == dump(StripDocstrings().visit(copy.deepcopy(new_ast)))
    delta = calls(old_ast) - calls(new_ast)
    assert delta == collections.Counter({'pl.tensor.dim': 2}), delta
    assert not (calls(new_ast) - calls(old_ast))
    (after / name).write_text(new)
    unchanged = [n for n in names if n != name and (before / n).read_bytes() == (after / n).read_bytes()]
    assert len(unchanged) == 32
    (OUT / 'candidate.patch').write_text(''.join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True), fromfile='e581/' + name, tofile='candidate/' + name
    )))
    (OUT / 'controls.json').write_text(json.dumps({'OA_COLS':128, 'OB_ROWS':96}, indent=2) + '\n')
    result = dict(
        baseline_commit=commit, frozen_utc=datetime.now(timezone.utc).isoformat(),
        source_before=str(before), source_candidate=str(after),
        formal_33_files_match_commit=True, changed_files=[name], unchanged_files=unchanged,
        removed_conditions=transform.removed, normalized_ast_equal=True,
        removed_calls=dict(delta), new_calls={},
        page_reads_gather_zero_initialization_math_events_unchanged=True,
        plan_source_unchanged=True, root_abi_and_dynamic_axes_unchanged=True,
        cpu_compiled=False, device_executed=False,
    )
    (OUT / 'source_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
