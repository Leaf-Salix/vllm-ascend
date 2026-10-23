# SPDX-License-Identifier: Apache-2.0
"""只把压缩KV的逐元素Scalar mask写入改为逐页Vector生成。

地址、页合法性、KV读取和计算顺序不变，输出仍是相同的FP32零/一掩码。
使用[页数,32]整行拼接避免已知的Vec列subview限制；最终reshape回[1,128]。
候选只在tests冻结验证，不因数学等价就全局替换正式策略。
"""

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("正式gather源码已变化，须重新审查变换")
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-work-items", type=int, default=0,
                        help="每核聚合项数达到门槛才用Vector；0保留历史全量Vector候选")
    parser.add_argument("--output", type=Path, default=HERE / "candidates/attention_gather_vector_mask.py")
    args = parser.parse_args()
    if args.min_work_items < 0:
        raise ValueError("分档门槛不能为负")
    source = HERE.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_hca_tnd/attention.py"
    text = source.read_text()
    text = replace_once(text, "            for gather_item in pl.range(gather_begin, gather_end):",
                        "            # 每页最多32行，FP32可精确表示所有行号和有效长度；各item复用列号。\n"
                        "            gather_mask_columns = pl.cast(pl.tile.arange(0, [1, CMP_STORAGE_BLOCK_SIZE], dtype=pl.INT32), pl.FP32)\n"
                        "            for gather_item in pl.range(gather_begin, gather_end):")
    text = replace_once(text,
                        "                gather_mask = pl.tile.full([1, CMP_ATTN_K_TILE], dtype=pl.FP32, value=0.0)",
                        "                # 按页整行拼接，不使用A3受限的列subview TMOV。\n"
                        "                gather_mask_grid = pl.tile.full([CMP_PAGES_PER_WORK, CMP_STORAGE_BLOCK_SIZE], dtype=pl.FP32, value=0.0)")
    old = """                                for gather_mask_col in pl.unroll(CMP_STORAGE_BLOCK_SIZE):
                                    if gather_mask_col < gather_valid_rows:
                                        pl.tile.write(gather_mask, [0, gather_col + gather_mask_col], 1.0)"""
    new = """                                # 仅合法物理页写入0/1：无效页与未初始化尾行仍保持0。
                                # 取代每页32次Scalar UB写，避免在Vector搬运任务内串行逐列更新。
                                gather_mask_count = pl.cast(pl.cast(gather_valid_rows, pl.INT32), pl.FP32)
                                gather_page_mask = pl.minimum(pl.maximum(pl.neg(pl.sub(
                                    gather_mask_columns, gather_mask_count)), 0.0), 1.0)
                                gather_mask_grid = pl.tile.assemble(gather_mask_grid, gather_page_mask, [gather_page, 0])"""
    if args.min_work_items:
        text = replace_once(text, "ROPE_TILE = 16",
                            f"GATHER_VECTOR_MIN_WORK_ITEMS = {args.min_work_items}\nROPE_TILE = 16")
        new = ("                                # 分档来自每核聚合项数，随请求数量和页表容量动态变化，非固定B点。\n"
               "                                if gather_work_tile >= GATHER_VECTOR_MIN_WORK_ITEMS:\n"
               + "\n".join("    " + line for line in new.splitlines())
               + "\n                                else:\n"
               "                                    # 小工作量保留原Scalar写入；网格视图只是同一128项的逐页寻址。\n"
               "                                    for gather_mask_col in pl.unroll(CMP_STORAGE_BLOCK_SIZE):\n"
               "                                        if gather_mask_col < gather_valid_rows:\n"
               "                                            pl.tile.write(gather_mask_grid, [gather_page, gather_mask_col], 1.0)")
    text = replace_once(text, old, new)
    text = replace_once(text, "                pl.store(gather_mask, [gather_item, 0], cmp_work_valid)",
                        "                pl.store(pl.reshape(gather_mask_grid, [1, CMP_ATTN_K_TILE]), [gather_item, 0], cmp_work_valid)")
    target = args.output.resolve()
    target.write_text(text)
    compile(text, str(target), "exec")
    print(target)


if __name__ == "__main__":
    main()
