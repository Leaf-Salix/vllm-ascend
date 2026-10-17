#!/usr/bin/env python3
"""冻结正式 33 源文件，仅调整 query 反量化的独立 head 分组；不导入 PyPTO。"""

import difflib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORMAL = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd")
REFERENCE = ROOT / "source_reference"
CANDIDATE = ROOT / "source_candidate"
TARGET = "query_projection_stages.py"

OLD = """    dq_head_tile = Q_ROPE_H_TILE
    # head_tile 只影响独立 head 的工作分组；每个 head 仍单独做 512 列 RMS。
    if tokens >= Q_DEQUANT_WORKERS * Q_ROPE_T_TILE and tokens % (Q_DEQUANT_WORKERS * Q_ROPE_T_TILE) == 0:
        dq_head_tile = H
    elif (
        tokens >= Q_DEQUANT_WORKERS * Q_ROPE_T_TILE * 16 // H
        and tokens % (Q_DEQUANT_WORKERS * Q_ROPE_T_TILE * 16 // H) == 0
    ):
        dq_head_tile = 16
"""
NEW = """    dq_token_tiles = (tokens + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE
    dq_head_tile = Q_ROPE_H_TILE
    # 只合并完整八行块的独立 head：选择最大的 64/32/16/8，使
    # dq_token_tiles*(H/head_tile) 恰为正的 48-AIV 整波；不满足则保留4。
    # 扩组后每个AIV的head迭代数不增，只减少scale/cos/sin/swap的重复准备。
    # 此条件取实际设备T，不用B/S或测试点；WORKERS仍只决定任务条带。
    # 有尾行必须留4，保持下面固定4-head尾循环及原recip(sqrt)数学。
    if tokens % Q_ROPE_T_TILE == 0:
        if dq_token_tiles >= Q_DEQUANT_WORKERS and dq_token_tiles % Q_DEQUANT_WORKERS == 0:
            dq_head_tile = H
        elif (
            dq_token_tiles >= Q_DEQUANT_WORKERS * 32 // H
            and dq_token_tiles % (Q_DEQUANT_WORKERS * 32 // H) == 0
        ):
            dq_head_tile = 32
        elif (
            dq_token_tiles >= Q_DEQUANT_WORKERS * 16 // H
            and dq_token_tiles % (Q_DEQUANT_WORKERS * 16 // H) == 0
        ):
            dq_head_tile = 16
        elif (
            dq_token_tiles >= Q_DEQUANT_WORKERS * 8 // H
            and dq_token_tiles % (Q_DEQUANT_WORKERS * 8 // H) == 0
        ):
            dq_head_tile = 8
"""

def main():
    paths = sorted(FORMAL.glob("*.py"))
    assert len(paths) == 33
    REFERENCE.mkdir()
    CANDIDATE.mkdir()
    for path in paths:
        text = path.read_text()
        (REFERENCE / path.name).write_text(text)
        (CANDIDATE / path.name).write_text(text)
    old = (CANDIDATE / TARGET).read_text()
    assert old.count(OLD) == 1
    new = old.replace(OLD, NEW)
    new = new.replace(
        "    反量化仍用八行、4/16/64 个 head 的原分组策略；原阈值以 48 AIV 为\n"
        "    基准，WORKERS 只决定谁执行，不擅自改变数学分组。RoPE 准备按八行条带。",
        "    反量化用八行、4/8/16/32/64 个独立 head 的分组策略；完整八行时选\n"
        "    最大且能形成48-AIV整波的组，尾行固定4。WORKERS只决定条带owner，\n"
        "    每个head的512列数学不变；RoPE准备仍按八行条带。")
    new = new.replace(
        "            # 64/16-head策略分别要求T整除384/96，因而必为8倍数；一旦有尾行，\n"
        "            # head_tile只能是4。这里固定4安全，未来改阈值时必须保留这条蕴含。",
        "            # 扩组分支显式要求T整除8；有尾行必回退4。此处继续固定4，\n"
        "            # 不改变原尾块的有效写回和recip(sqrt)路径。")
    (CANDIDATE / TARGET).write_text(new)
    changed = [p.name for p in paths if p.read_text() != (CANDIDATE / p.name).read_text()]
    assert changed == [TARGET]
    assert all(p.read_text() == (REFERENCE / p.name).read_text() for p in paths)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile="source_reference/" + TARGET, tofile="source_candidate/" + TARGET)))
    controls = dict(
        QUERY_GROUP=6, QK_PANEL=64, SCORE_TILE=1024, PREFETCH=1,
        TO_L0=True, BALANCED=True, ROOTS=1, OA_COLS=128, OB_ROWS=96,
        OUTPUT_GROUP_PIPELINE=True, HC_ROW_DEPENDENCIES=True,
        ATTENTION_QUERY_METADATA=False, HC_CONSUMER_REDUCE=False)
    (ROOT / "controls.json").write_text(json.dumps(controls, indent=2) + "\n")
    record = dict(formal=str(FORMAL), reference=str(REFERENCE), candidate=str(CANDIDATE),
                  source_count=len(paths), changed_files=changed,
                  direct_text_snapshot_matches_formal=True,
                  controls=controls, device_run=False,
                  files=[dict(name=p.name, reference_bytes=(REFERENCE/p.name).stat().st_size,
                              candidate_bytes=(CANDIDATE/p.name).stat().st_size) for p in paths])
    (ROOT / "source_snapshot.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:record[k] for k in ("source_count", "changed_files", "candidate")}, ensure_ascii=False))

if __name__ == "__main__":
    main()

