# SPDX-License-Identifier: Apache-2.0
"""核外选择独立gather策略：小档完整保留原Scalar布局与写入代码。

同一TND入口根据shape推得每核工作项，在TMR里只提交一种gather任务；
两个任务分别以编译期常量选Scalar/Vector，避免核内if、共享新布局或
两分支共同的活跃区影响旧小档。Q/L和页表内容仍在设备读取，无Host取值。
"""

import argparse
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("正式gather代码已变化，须重新核对变换")
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-work-items", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = HERE.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_hca_tnd/attention.py"
    text = source.read_text()
    start = text.index('        with pl.spmd(cmp_gather_blocks, name_hint="hca_cmp_work_gather"')
    end = text.index("\n\n\n        # 全部历史共用一组 MIX 任务", start)
    scalar_scope = text[start:end]
    vector_scope = once(scalar_scope, "            for gather_item in pl.range(gather_begin, gather_end):",
                        "            # 仅Vector核持有列号；Scalar核源码中完全没有此初始化。\n"
                        "            gather_mask_columns = pl.cast(pl.tile.arange(0, [1, CMP_STORAGE_BLOCK_SIZE], dtype=pl.INT32), pl.FP32)\n"
                        "            for gather_item in pl.range(gather_begin, gather_end):")
    old_mask = "                gather_mask = pl.tile.full([1, CMP_ATTN_K_TILE], dtype=pl.FP32, value=0.0)"
    vector_scope = once(vector_scope, old_mask,
                        "                gather_mask_grid = pl.tile.full([CMP_PAGES_PER_WORK, CMP_STORAGE_BLOCK_SIZE], dtype=pl.FP32, value=0.0)")
    old_write = """                                for gather_mask_col in pl.unroll(CMP_STORAGE_BLOCK_SIZE):
                                    if gather_mask_col < gather_valid_rows:
                                        pl.tile.write(gather_mask, [0, gather_col + gather_mask_col], 1.0)"""
    new_write = """                                # 页内0..31及有效长度都可被FP32精确表示；负页和未初始化尾行不进入本分支。
                                gather_mask_count = pl.cast(pl.cast(gather_valid_rows, pl.INT32), pl.FP32)
                                gather_page_mask = pl.minimum(pl.maximum(pl.neg(pl.sub(
                                    gather_mask_columns, gather_mask_count)), 0.0), 1.0)
                                gather_mask_grid = pl.tile.assemble(gather_mask_grid, gather_page_mask, [gather_page, 0])"""
    vector_scope = once(vector_scope, old_write, new_write)
    old_store = "                pl.store(gather_mask, [gather_item, 0], cmp_work_valid)"
    vector_scope = once(vector_scope, old_store,
                        "                pl.store(pl.reshape(gather_mask_grid, [1, CMP_ATTN_K_TILE]), [gather_item, 0], cmp_work_valid)")
    # 源码中分开两个核体，避免把未消费的Tile初始化寄希望于后续DCE。
    # 重复的是同一份生成模板；最终文件可以直接运行，不依赖tests生成器。
    helper = ""
    for kind, scope in (("scalar", scalar_scope), ("vector", vector_scope)):
        scope = once(scope, 'name_hint="hca_cmp_work_gather"', f'name_hint="hca_cmp_work_gather_{kind}"')
        helper += f'''
@pl.jit.inline(auto_scope=False)
def _gather_compressed_{kind}(
    cmp_kv_flat: pl.Tensor, cmp_block_table: pl.Tensor, kv_seq_lens: pl.Tensor,
    cmp_work_kv: pl.InOut[pl.Tensor], cmp_work_valid: pl.InOut[pl.Tensor],
    cmp_cache_ready_dep: pl.Scalar[pl.TASK_ID],
):
    """独立{kind}路径；原页表与有效行保护相同，无另一种mask的初始化。"""
    request_count = pl.tensor.dim(kv_seq_lens, 0)
    cmp_table_blocks = pl.tensor.dim(cmp_block_table, 1)
    cmp_block_num = pl.tensor.dim(cmp_kv_flat, 0) // CMP_STORAGE_BLOCK_SIZE
    cmp_work_count = (cmp_table_blocks + CMP_PAGES_PER_WORK - 1) // CMP_PAGES_PER_WORK
    cmp_gather_count = request_count * cmp_work_count
    gather_work_tile = (cmp_gather_count + CMP_GATHER_AIV_WAVE - 1) // CMP_GATHER_AIV_WAVE
    cmp_gather_blocks = (cmp_gather_count + gather_work_tile - 1) // gather_work_tile
''' + textwrap.indent(textwrap.dedent(scope), "    ") + '''
    return cmp_gather_tid


'''
    selection = '''        # 核外TMR仅用已绑定的shape计算选择，不读取设备Q/L到Host，不增加metadata任务。
        # 每次只提交一种gather；被选核内部不含策略判断，小档仍是原来的[1,128]Scalar布局。
        if gather_work_tile >= GATHER_VECTOR_MIN_WORK_ITEMS:
            vector_tid = _gather_compressed_vector(
                cmp_kv_flat, cmp_block_table, kv_seq_lens, cmp_work_kv, cmp_work_valid, cmp_cache_ready_dep,
            )
            cmp_gather_tid = pl.yield_(vector_tid)
        else:
            scalar_tid = _gather_compressed_scalar(
                cmp_kv_flat, cmp_block_table, kv_seq_lens, cmp_work_kv, cmp_work_valid, cmp_cache_ready_dep,
            )
            cmp_gather_tid = pl.yield_(scalar_tid)
'''
    text = text[:start] + selection + text[end:]
    text = once(text, "        cmp_gather_blocks = (cmp_gather_count + gather_work_tile - 1) // gather_work_tile\n", "")
    pos = text.index("def _make_sparse_attention(MACRO_K):")
    text = text[:pos] + helper + text[pos:]
    text = once(text, "ROPE_TILE = 16", f"GATHER_VECTOR_MIN_WORK_ITEMS = {args.min_work_items}\nROPE_TILE = 16")
    text = text.replace("两档共用请求gather、device Q/L、FFTS及输出/逆RoPE，仅编译期绑定不同计算块。",
                        "两档共用device Q/L、FFTS及输出/逆RoPE；压缩gather在TMR按工作量选择独立Scalar/Vector核。")
    args.output.write_text(text)
    compile(text, str(args.output), "exec")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
