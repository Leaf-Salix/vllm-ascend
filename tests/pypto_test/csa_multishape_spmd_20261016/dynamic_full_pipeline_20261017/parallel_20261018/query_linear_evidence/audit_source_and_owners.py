#!/usr/bin/env python3
"""执行冻结源的 Scalar 调度 AST，枚举 T1..512；不加载 PyPTO，不编译。"""

import ast
import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
TARGET = "query_projection_stages.py"
WORKERS, HEADS, ROWS = 48, 64, 8


def read(side):
    tree = ast.parse((ROOT / side / TARGET).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "q_dequant_rope_stage")
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    math_if = next(n for n in loop.body if isinstance(n, ast.If) and "tg + Q_ROPE_T_TILE" in ast.unparse(n.test))
    return tree, fn, loop, math_if


def math_checks(old, new):
    old_math, new_math = copy.deepcopy(old[3]), copy.deepcopy(new[3])
    a = next(n for n in old_math.body if isinstance(n, ast.For))
    b = next(n for n in new_math.body if isinstance(n, ast.For))
    assert ast.unparse(a.iter) == "pl.pipeline(dq_head_tile, stage=2)"
    assert ast.unparse(b.iter) == "pl.pipeline(dq_heads_in_row, stage=2)"
    b.iter = copy.deepcopy(a.iter)
    assert ast.dump(old_math, include_attributes=False) == ast.dump(new_math, include_attributes=False)
    old_other, new_other = copy.deepcopy(old[0]), copy.deepcopy(new[0])
    for tree in (old_other, new_other):
        tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == old[1].name)]
        tree.body = tree.body[1:]  # 仅忽略更新后的模块中文总览。
    assert ast.dump(old_other, include_attributes=False) == ast.dump(new_other, include_attributes=False)
    assert ast.dump(old[1].args, include_attributes=False) == ast.dump(new[1].args, include_attributes=False)


def schedule(parts, candidate):
    fn, loop, math_if = parts[1:]
    body = []
    for node in fn.body:
        if node is loop:
            break
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            body.append(copy.deepcopy(node))
    work = copy.deepcopy(loop)
    group = "dq_heads_in_row" if candidate else "dq_head_tile"
    record = ast.parse(f"record(tg, hg, {group}, tg + Q_ROPE_T_TILE <= tokens)").body[0]
    work.body = [record if n.lineno == math_if.lineno else n for n in work.body]
    body.append(work)
    code = compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "<frozen-scalar-schedule>", "exec")

    def run(tokens, owner):
        records = []
        scope = dict(pl=SimpleNamespace(tensor=SimpleNamespace(dim=lambda tensor, axis: tensor),
                                        range=range, min=min, max=max),
                     q_flat=tokens, H=HEADS, Q_DEQUANT_WORKERS=WORKERS, Q_ROPE_T_TILE=ROWS,
                     Q_ROPE_H_TILE=4, WORKERS=WORKERS, worker_id=owner,
                     record=lambda *args: records.append(args))
        exec(code, scope)
        return records
    return run


def ownership(tokens, run):
    writes = bytearray(tokens * HEADS)
    per_owner, chunk_lengths, one_head = [], Counter(), []
    records_all = []
    for owner in range(WORKERS):
        records = run(tokens, owner)
        records_all.append(records)
        heads = token_heads = 0
        for row0, head0, length, full in records:
            assert 0 <= row0 < tokens and row0 % ROWS == 0
            assert 0 <= head0 < HEADS and 1 <= length <= HEADS - head0
            assert full == (row0 + ROWS <= tokens)
            assert full or length == 4
            live = min(ROWS, tokens - row0)
            heads += length
            token_heads += live * length
            chunk_lengths[length] += 1
            if length == 1:
                one_head.append(dict(owner=owner, row0=row0, head0=head0))
            for row in range(row0, row0 + live):
                for head in range(head0, head0 + length):
                    offset = row * HEADS + head
                    assert writes[offset] == 0
                    writes[offset] = 1
        per_owner.append(dict(prepares=len(records), head_blocks=heads, token_heads=token_heads))
    assert all(writes) and sum(x["token_heads"] for x in per_owner) == tokens * HEADS
    return dict(prepares=sum(x["prepares"] for x in per_owner),
                max_prepares=max(x["prepares"] for x in per_owner),
                max_head_blocks=max(x["head_blocks"] for x in per_owner),
                min_head_blocks=min(x["head_blocks"] for x in per_owner),
                max_token_heads=max(x["token_heads"] for x in per_owner),
                active_aiv=sum(x["prepares"] > 0 for x in per_owner),
                per_owner=per_owner, chunk_lengths=dict(sorted(chunk_lengths.items())),
                one_head_chunks=one_head), records_all


def main():
    old, new = read("source_reference"), read("source_candidate")
    math_checks(old, new)
    paths = sorted((ROOT / "source_reference").glob("*.py"))
    changed = [p.name for p in paths if p.read_text() != (ROOT / "source_candidate" / p.name).read_text()]
    assert len(paths) == 33 and changed == [TARGET]
    reference, candidate = schedule(old, False), schedule(new, True)
    cases, changed_counts, one_head_cases = [], [], []
    for tokens in range(1, 513):
        before, old_records = ownership(tokens, reference)
        after, new_records = ownership(tokens, candidate)
        assert after["max_head_blocks"] <= before["max_head_blocks"]
        assert after["max_token_heads"] <= before["max_token_heads"]
        assert after["prepares"] <= before["prepares"]
        assert after["max_prepares"] <= before["max_prepares"]
        if tokens % ROWS or tokens < 24:
            assert new_records == old_records
        else:
            assert after["active_aiv"] == WORKERS
            assert after["max_head_blocks"] - after["min_head_blocks"] <= 1
        row = dict(tokens=tokens, reference=before, candidate=after)
        cases.append(row)
        if new_records != old_records:
            changed_counts.append(tokens)
        if after["one_head_chunks"]:
            one_head_cases.append(dict(tokens=tokens, chunks=after["one_head_chunks"]))
    # T0也必须是空工作；不把它算入512个正T枚举。
    assert all(candidate(0, owner) == [] for owner in range(WORKERS))
    summary = dict(passed=True, base_commit="9d5327e5", tokens_enumerated=[1, 512], cases=512,
                   changed_files=changed, math_ast_equal_except_pipeline_trip_variable=True,
                   preparation_and_tail_math_ast_equal=True, outside_target_ast_equal=True,
                   unique_token_head_writer=True, maximum_worker_head_load_nonincreasing=True,
                   total_and_maximum_worker_prepares_nonincreasing=True,
                   tail_and_t8_t16_owner_identical=True, zero_tokens_empty=True,
                   linear_rule="T%8==0 且 ceil(T/8)*(64/4)>=48；区间[floor(N*w/48),floor(N*(w+1)/48))，N=(T/8)*64",
                   changed_tokens=changed_counts, one_head_chunk_cases=one_head_cases,
                   selected=[{k:v for k,v in x.items()} for x in cases if x["tokens"] in (8,16,24,32,40,48,96,144,168,192,384,512)],
                   limits="纯调度/数学AST证据；每个固定owner的负载可以迁移，不声称设备exact或收益")
    (ROOT / "ownership_cases.json").write_text(json.dumps(cases, ensure_ascii=False, separators=(",", ":")) + "\n")
    (ROOT / "source_and_owner_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(passed=True, changed_tokens=changed_counts,
                          selected=[dict(tokens=x["tokens"], old_prepares=x["reference"]["prepares"],
                                         new_prepares=x["candidate"]["prepares"],
                                         old_max_prepares=x["reference"]["max_prepares"],
                                         new_max_prepares=x["candidate"]["max_prepares"],
                                         old_max_heads=x["reference"]["max_head_blocks"],
                                         new_max_heads=x["candidate"]["max_head_blocks"])
                                    for x in summary["selected"]],
                          one_head_chunk_case_count=len(one_head_cases)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
