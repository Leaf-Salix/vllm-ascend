#!/usr/bin/env python3
"""直接执行两份源码的分组标量 AST，复算 T1..512 owner/准备次数；无 PyPTO/NPU。"""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
TARGET = "query_projection_stages.py"
WORKERS = 48
HEADS = 64
ROWS = 8


def module(side):
    return ast.parse((ROOT / side / TARGET).read_text())


def stage(tree):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "q_dequant_rope_stage")


def work_loop(fn):
    return next(n for n in fn.body if isinstance(n, ast.For))


def selector(fn):
    prefix = []
    for node in fn.body:
        if isinstance(node, ast.For):
            break
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        prefix.append(copy.deepcopy(node))
    code = compile(ast.fix_missing_locations(ast.Module(body=prefix, type_ignores=[])), "<frozen-head-selection>", "exec")

    def run(tokens):
        scope = dict(pl=SimpleNamespace(tensor=SimpleNamespace(dim=lambda tensor, axis: tensor)),
                     q_flat=tokens, H=HEADS, Q_DEQUANT_WORKERS=WORKERS, Q_ROPE_T_TILE=ROWS, Q_ROPE_H_TILE=4)
        exec(code, scope)
        return scope["dq_head_tile"]
    return run


def ownership(tokens, group):
    blocks = (tokens + ROWS - 1) // ROWS
    groups = HEADS // group
    tasks = blocks * groups
    touched = bytearray(tokens * HEADS)
    prepares = [0] * WORKERS
    head_blocks = [0] * WORKERS
    token_heads = [0] * WORKERS
    for owner in range(WORKERS):
        for work in range(owner, tasks, WORKERS):
            row0 = work // groups * ROWS
            head0 = work % groups * group
            live_rows = min(ROWS, tokens - row0)
            inner_heads = group if live_rows == ROWS else 4
            assert inner_heads == group  # 尾循环仍固定4，不能漏写8/16/32/64分组。
            prepares[owner] += 1
            head_blocks[owner] += inner_heads
            for head in range(head0, head0 + inner_heads):
                assert 0 <= head < HEADS
                for row in range(row0, row0 + live_rows):
                    offset = row * HEADS + head
                    assert touched[offset] == 0
                    touched[offset] = 1
                    token_heads[owner] += 1
    assert all(touched)
    assert sum(token_heads) == tokens * HEADS
    return dict(group=group, task_preparations=tasks, active_aiv=sum(p > 0 for p in prepares),
                head_block_iterations=blocks * HEADS,
                prepares_per_aiv=prepares, head_blocks_per_aiv=head_blocks, token_heads_per_aiv=token_heads)


def main():
    reference = module("source_reference")
    candidate = module("source_candidate")
    ref_fn, new_fn = stage(reference), stage(candidate)
    # 从外层work循环开始（包括完整/尾head数学、GM地址及store）逐节点相同。
    assert ast.dump(work_loop(ref_fn), include_attributes=False) == ast.dump(work_loop(new_fn), include_attributes=False)
    ref_rest, new_rest = copy.deepcopy(reference), copy.deepcopy(candidate)
    for tree in (ref_rest, new_rest):
        tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == ref_fn.name)]
        if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
            tree.body = tree.body[1:]  # 只忽略已同步更新的模块中文总览。
    assert ast.dump(ref_rest, include_attributes=False) == ast.dump(new_rest, include_attributes=False)
    paths = sorted((ROOT / "source_reference").glob("*.py"))
    differences = [p.name for p in paths if p.read_text() != (ROOT / "source_candidate" / p.name).read_text()]
    assert len(paths) == 33 and differences == [TARGET]
    choose_old, choose_new = selector(ref_fn), selector(new_fn)
    rows = []
    changed = []
    for tokens in range(1, 513):
        old = ownership(tokens, choose_old(tokens))
        new = ownership(tokens, choose_new(tokens))
        assert all(b <= a for a, b in zip(old["head_blocks_per_aiv"], new["head_blocks_per_aiv"]))
        assert all(b <= a for a, b in zip(old["token_heads_per_aiv"], new["token_heads_per_aiv"]))
        assert new["head_block_iterations"] == old["head_block_iterations"]
        assert new["task_preparations"] <= old["task_preparations"]
        if tokens % ROWS:
            assert new["group"] == old["group"] == 4
        if new["group"] > 4:
            assert tokens % ROWS == 0 and new["task_preparations"] % WORKERS == 0
            assert new["active_aiv"] == WORKERS
        row = dict(tokens=tokens, reference=old, candidate=new)
        rows.append(row)
        if old["group"] != new["group"]:
            changed.append(dict(tokens=tokens, old_group=old["group"], new_group=new["group"],
                                old_tasks=old["task_preparations"], new_tasks=new["task_preparations"],
                                old_max_head_blocks=max(old["head_blocks_per_aiv"]),
                                new_max_head_blocks=max(new["head_blocks_per_aiv"]),
                                active_aiv=new["active_aiv"]))
    result = dict(passed=True, tokens_enumerated=[1,512], cases=len(rows),
                  rule="完整八行时取最大g∈{8,16,32,64}使ceil(T/8)*(64/g)是正48倍数；否则g=4",
                  full_and_tail_math_ast_equal=True, outside_target_function_ast_equal=True,
                  source_changed_files=differences, unique_token_head_writer=True,
                  every_aiv_head_load_nonincreasing=True, every_aiv_token_head_load_nonincreasing=True,
                  tail_group_stays_four=True, device_execution=False,
                  changed_cases=changed,
                  evidence_limit="工作量及owner证明，不是设备精度/性能结果；未假定48个AIV到达时刻相同")
    (ROOT / "ownership_cases.json").write_text(json.dumps(rows, ensure_ascii=False, separators=(",",":")) + "\n")
    (ROOT / "source_and_owner_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

