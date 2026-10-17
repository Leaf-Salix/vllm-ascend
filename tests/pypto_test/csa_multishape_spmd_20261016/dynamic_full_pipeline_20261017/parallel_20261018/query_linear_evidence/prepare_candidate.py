#!/usr/bin/env python3
"""冻结已采用 DQ 整波的正式 33 源，只替换反量化工作的 Scalar 分配。"""

import ast
import difflib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORMAL = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd")
ADOPTED = ROOT.parent / "query_dequant_group_20261018/source_candidate"
TARGET = "query_projection_stages.py"

PREFIX = '''    tokens = pl.tensor.dim(q_flat, 0)
    dq_token_tiles = (tokens + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE
    # 原g4至少有WORKERS个任务才扩大连续分工；正式48 AIV对应完整T>=24。
    # T8/T16及任何尾行保留g4条带，不为小工作量增加scale/cos/sin/swap准备。
    dq_contiguous = (
        tokens % Q_ROPE_T_TILE == 0
        and dq_token_tiles * (H // Q_ROPE_H_TILE) >= WORKERS
    )
    dq_work_start = worker_id
    dq_work_end = dq_token_tiles * (H // Q_ROPE_H_TILE)
    dq_work_stride = WORKERS
    dq_first_head = 0
    dq_last_head = 0
    if dq_contiguous:
        # 线性编号=row_tile*64+head。floor分界覆盖全部head且负载至多差1；
        # 不固定T，不改变一个head内的8行/512列规约和RoPE算术顺序。
        dq_total_heads = dq_token_tiles * H
        dq_first_head = dq_total_heads * worker_id // WORKERS
        dq_last_head = dq_total_heads * (worker_id + 1) // WORKERS
        dq_work_start = dq_first_head // H
        dq_work_end = (dq_last_head + H - 1) // H
        dq_work_stride = 1
    # 同一个数学体：完整区间逐行拆chunk，其余仍按旧g4任务条带。
    # 入口保证连续区间非空；每个chunk在一个64-head行内，head数为1..64。
    for dq_work in pl.range(dq_work_start, dq_work_end, dq_work_stride):
        hg = dq_work % (H // Q_ROPE_H_TILE) * Q_ROPE_H_TILE
        tg = dq_work // (H // Q_ROPE_H_TILE) * Q_ROPE_T_TILE
        dq_heads_in_row = Q_ROPE_H_TILE
        if dq_contiguous:
            dq_row_head = dq_work * H
            hg = pl.max(dq_first_head, dq_row_head) - dq_row_head
            tg = dq_work * Q_ROPE_T_TILE
            dq_heads_in_row = pl.min(dq_last_head, dq_row_head + H) - pl.max(dq_first_head, dq_row_head)
'''


def main():
    paths = sorted(FORMAL.glob("*.py"))
    assert len(paths) == 33
    assert all(p.read_text() == (ADOPTED / p.name).read_text() for p in paths), "正式源已变化，停止物化"
    for name in ("source_reference", "source_candidate"):
        (ROOT / name).mkdir()
        for p in paths:
            (ROOT / name / p.name).write_text(p.read_text())
    old = (ROOT / "source_reference" / TARGET).read_text()
    tree = ast.parse(old)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "q_dequant_rope_stage")
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    math_if = next(n for n in loop.body if isinstance(n, ast.If))
    lines = old.splitlines(keepends=True)
    tokens_line = next(n.lineno for n in fn.body if isinstance(n, ast.Assign))
    new = "".join(lines[:tokens_line - 1]) + PREFIX + "".join(lines[math_if.lineno - 1:])
    assert new.count("pl.pipeline(dq_head_tile, stage=2)") == 1
    new = new.replace("pl.pipeline(dq_head_tile, stage=2)", "pl.pipeline(dq_heads_in_row, stage=2)")
    new = new.replace(
        "    反量化用八行、4/8/16/32/64 个独立 head 的分组策略；完整八行时选\n"
        "    最大且能形成48-AIV整波的组，尾行固定4。WORKERS只决定条带owner，\n"
        "    每个head的512列数学不变；RoPE准备仍按八行条带。",
        "    反量化仍用八行×单head数学。完整八行且原g4任务足够覆盖WORKERS时，\n"
        "    将(row_tile,head)线性空间按floor首尾分给AIV，再沿64-head行边界\n"
        "    拆chunk复用准备；小T和尾行保留g4条带。RoPE准备仍按八行条带。")
    new = new.replace(
        "    Q_B 列块 owner=(column//N_TILE)%WORKERS。反量化的八行/head 组编号\n"
        "    work=(row//8)*(64/head_tile)+head//head_tile，其 owner=work%WORKERS；\n"
        "    每个 head 完整包含两个 NZ N256 列块。RoPE owner=(row//8)%active，",
        "    Q_B 列块 owner=(column//N_TILE)%WORKERS。反量化的连续区间为\n"
        "    [total_heads*worker//WORKERS,total_heads*(worker+1)//WORKERS)；\n"
        "    回退时g4任务owner=work%WORKERS。每个head完整包含两个NZ N256列块。\n"
        "    RoPE owner=(row//8)%active，")
    new = new.replace(
        "    完整八行与尾块沿用不同的既有 RMS 算术；每次工作只写八行中的一个 head 组。",
        "    完整八行连续分配独立head，沿行边界分chunk；尾块保留原g4及既有RMS算术。")
    new = new.replace(
        "            # 扩组分支显式要求T整除8；有尾行必回退4。此处继续固定4，",
        "            # 连续区间显式要求T整除8；有尾行必回退4。此处继续固定4，")
    (ROOT / "source_candidate" / TARGET).write_text(new)
    changed = [p.name for p in paths if p.read_text() != (ROOT / "source_candidate" / p.name).read_text()]
    assert changed == [TARGET]
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile="source_reference/" + TARGET, tofile="source_candidate/" + TARGET)))
    controls = json.loads((ROOT.parent / "query_dequant_group_20261018/controls.json").read_text())
    (ROOT / "controls.json").write_text(json.dumps(controls, indent=2) + "\n")
    record = dict(formal=str(FORMAL), base_commit="9d5327e5", source_count=33,
                  formal_equals_adopted_dq_33_by_direct_text=True, changed_files=changed,
                  candidate=str(ROOT / "source_candidate"), reference=str(ROOT / "source_reference"),
                  device_execution=False, controls=controls,
                  files=[dict(name=p.name, reference_bytes=p.stat().st_size,
                              candidate_bytes=(ROOT / "source_candidate" / p.name).stat().st_size) for p in paths])
    (ROOT / "source_snapshot.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(source_count=33, changed_files=changed, candidate=record["candidate"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
