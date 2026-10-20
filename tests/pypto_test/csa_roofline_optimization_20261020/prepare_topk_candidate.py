# SPDX-License-Identifier: Apache-2.0
"""冻结当前动态算子，并仅在长历史 balanced 路径插入精确阈值筛选。"""
import ast
import json
import shutil
import subprocess
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    formal = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    reference = here / "source_reference"
    candidate = here / "source_topk_candidate"
    assert not reference.exists() and not candidate.exists(), "不覆盖已冻结源"
    for dst in (reference, candidate):
        dst.mkdir()
        for src in formal.glob("*.py"):
            shutil.copy2(src, dst / src.name)
    shutil.copy2(here / "topk_threshold_stages.py", candidate / "topk_threshold_stages.py")
    path = candidate / "indexer_score_stages.py"
    text = path.read_text()
    anchor = "from .indexer_topk_stages import PAIR_ROWS_DYN, SCORE_ROWS_DYN, indexer_topk_half_leaf"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + "\nfrom .topk_threshold_stages import threshold_merge_top512")
    begin = text.index("                                    logical_i32 = pl.cast(single_segment_begin, pl.INT32)")
    end = text.index("                for single_local in pl.unroll(query_group_size // 2):", begin)
    original = text[begin:end]
    prefix = """                                    # 首段没有可信阈值，保留原排序；已有根且本段超过
                                    # 512 项时，才尝试按上轮第512项进行保序筛选。
                                    # 本分支只位于长历史 balanced 策略，不枚举 batch。
                                    if single_step >= 2 and single_segment_valid > 512:
                                        threshold_old_root = pl.tile.extract(
                                            single_roots, single_local, 0, [1, TOPK_PAIR_WIDTH],
                                            target_memory=pl.MemorySpace.Vec,
                                        )
                                        threshold_next = threshold_merge_top512(
                                            single_segment_data, threshold_old_root,
                                            pl.cast(single_segment_begin, pl.INT32),
                                            pl.cast(single_segment_valid, pl.INT32),
                                        )
                                        single_roots = pl.tile.assemble(
                                            single_roots, threshold_next, [single_local, 0],
                                        )
                                    else:
"""
    text = text[:begin] + prefix + "".join("    " + ln if ln.strip() else ln
                                         for ln in original.splitlines(keepends=True)) + text[end:]
    ast.parse(text)
    path.write_text(text)
    record = dict(
        repo_head=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        reference_files=len(list(reference.glob("*.py"))),
        source_changes=["indexer_score_stages.py", "topk_threshold_stages.py"],
        formal_changed=False, compiled=False, device_validated=False,
        numerical_contract="score/Top-K 同分顺序及原冻结检查不放宽",
        route="既有长历史 balanced 区间；首段及不超过512项的段沿原路径",
    )
    (here / "TOPK_PREPARATION.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
