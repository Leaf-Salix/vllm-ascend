# SPDX-License-Identifier: Apache-2.0
"""HCA长历史直接读取Native压缩KV的候选生成器，不改正式算子。

保留当前K512/D256、两步QK预发、三槽、跨query流水、FFTS和全部浮点顺序。
把GM中的完整压缩KV聚合窗口替换为每128行一条64B页计划和一个64KiB零模板。
metadata可与QKV/C128生产重叠，attention另外显式等待真实cache写完。
连续四页直接一笔GM→L1；非连续、负页或部分尾页按页填入预清零L1，不能读NaN尾区。

参考本仓Native arch32的Raw DataCopyPA分页加载。其compressed路径实际使用
kvMergeGm环缓冲；本候选不是宣称Native已经直接读compressed，而是HCA dense场景的独立验证。
"""

from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("attention计算体已变化，必须重新核对直接KV变换")
    return text.replace(old, new)


LOADER = '''
@pl.jit.inline(auto_scope=False)
def _load_native_cmp_half(
    native_kv: pl.Tensor, page_plan: pl.Tensor, zero_half: pl.Tensor,
    item: pl.Scalar[pl.INDEX], dim0: pl.Scalar[pl.INDEX],
):
    """读取128×256 BF16到L1；计划由本次device Q/L/页表生成，不使用Host地址列表。"""
    # 两个分支写同一份预分配L1，避免if合流被实现成A3不支持的Mat→Mat TMOV。
    result = pl.create_tile([ATTN_K_TILE, 256], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
    contiguous = pl.read(page_plan, [item, 8])
    if contiguous == 1:
        # 四页均完整有效且物理连续。与原聚合窗口相同，只改变GM源地址。
        first = pl.cast(pl.read(page_plan, [item, 0]), pl.INDEX)
        result = pl.gather_row(result, native_kv, [0, 0], [first, dim0], [ATTN_K_TILE, 256])
    else:
        # Mat不直接做Vector清零；加载共享零模板，再只覆盖合法页的有效行。
        # 即使概率为0，PV也不能读到未初始化KV中的NaN。
        result = pl.gather_row(result, zero_half, [0, 0], [0, 0], [ATTN_K_TILE, 256])
        for page in pl.range(CMP_PAGES_PER_WORK):
            rows = pl.cast(pl.read(page_plan, [item, 4 + page]), pl.INDEX)
            if rows > 0:
                source = pl.cast(pl.read(page_plan, [item, page]), pl.INDEX)
                result = pl.gather_row(result, native_kv, [page * CMP_STORAGE_BLOCK_SIZE, 0],
                    [source, dim0], [CMP_STORAGE_BLOCK_SIZE, 256], valid_shape=[rows, 256])
    return result


'''

METADATA = '''        # 只生成页计划及原0/1 mask，不搬运整份KV。计划不依赖cache内容，允许提前执行。
        # attention另加cmp_cache_ready_dep，保证真正MTE2读KV时压缩写回已经完成。
        gather_work_tile = (cmp_gather_count + CMP_GATHER_AIV_WAVE - 1) // CMP_GATHER_AIV_WAVE
        cmp_gather_blocks = (cmp_gather_count + gather_work_tile - 1) // gather_work_tile
        with pl.spmd(cmp_gather_blocks, name_hint="hca_cmp_page_plan", allow_early_resolve=True) as cmp_gather_tid:
            gather_block = pl.tile.get_block_idx()
            if gather_block == 0:
                # 仅一个writer初始化零模板；共用该task完成依赖，没有额外初始化task。
                pl.store(pl.tile.full([ATTN_K_TILE, 256], dtype=pl.BF16, value=0.0), [0, 0], cmp_zero_half)
            gather_begin = gather_block * gather_work_tile
            gather_end = pl.min(gather_begin + gather_work_tile, cmp_gather_count)
            for gather_item in pl.range(gather_begin, gather_end):
                gather_request = gather_item // cmp_work_count
                gather_work = gather_item - gather_request * cmp_work_count
                gather_first_col = gather_work * CMP_PAGES_PER_WORK
                # 0..3物理行起点，4..7有效行数，8表示四个完整页物理连续，其余保留为零。
                # 一行64B由单个AIV通过MTE3一次写回，避免Scalar GM短写及跨核伪共享。
                page_record = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
                gather_mask = pl.tile.full([1, CMP_ATTN_K_TILE], dtype=pl.FP32, value=0.0)
                gather_rows = pl.min(HCA_MAX_COMPRESSED_ROWS, pl.max(
                    pl.read(kv_seq_lens, [gather_request]), 0,
                ) // COMPRESS_RATIO)
                for gather_page in pl.range(CMP_PAGES_PER_WORK):
                    page_col = gather_first_col + gather_page
                    valid_rows = pl.min(CMP_STORAGE_BLOCK_SIZE, pl.max(
                        gather_rows - page_col * CMP_STORAGE_BLOCK_SIZE, 0,
                    ))
                    if page_col < cmp_table_blocks and valid_rows > 0:
                        physical_page = pl.read(cmp_block_table, [gather_request, page_col])
                        if physical_page >= 0:
                            if physical_page < cmp_block_num:
                                row_begin = pl.cast(physical_page, pl.INDEX) * CMP_STORAGE_BLOCK_SIZE
                                pl.tile.write(page_record, [0, gather_page], pl.cast(row_begin, pl.INT32))
                                pl.tile.write(page_record, [0, 4 + gather_page], pl.cast(valid_rows, pl.INT32))
                                # 首版保留已验证Scalar mask，避免与另一项Vector mask变化混合。
                                for col in pl.unroll(CMP_STORAGE_BLOCK_SIZE):
                                    if col < valid_rows:
                                        pl.tile.write(gather_mask, [0, gather_page * CMP_STORAGE_BLOCK_SIZE + col], 1.0)
                full0 = pl.tile.read(page_record, [0, 4])
                full1 = pl.tile.read(page_record, [0, 5])
                full2 = pl.tile.read(page_record, [0, 6])
                full3 = pl.tile.read(page_record, [0, 7])
                row0 = pl.tile.read(page_record, [0, 0])
                row1 = pl.tile.read(page_record, [0, 1])
                row2 = pl.tile.read(page_record, [0, 2])
                row3 = pl.tile.read(page_record, [0, 3])
                if (full0 == CMP_STORAGE_BLOCK_SIZE and full1 == CMP_STORAGE_BLOCK_SIZE
                    and full2 == CMP_STORAGE_BLOCK_SIZE and full3 == CMP_STORAGE_BLOCK_SIZE
                    and row1 == row0 + CMP_STORAGE_BLOCK_SIZE
                    and row2 == row1 + CMP_STORAGE_BLOCK_SIZE
                    and row3 == row2 + CMP_STORAGE_BLOCK_SIZE):
                    pl.tile.write(page_record, [0, 8], pl.cast(1, pl.INT32))
                pl.store(page_record, [gather_item, 0], cmp_page_plan)
                pl.store(gather_mask, [gather_item, 0], cmp_work_valid)
'''


def main():
    source = (HERE.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_hca_tnd/attention.py").read_text()
    start = source.index("def _make_sparse_attention(MACRO_K):")
    end = source.index("\nsparse_attn_hca_tp1 = _make_sparse_attention", start)
    direct = source[start:end]
    direct = once(direct, "def _make_sparse_attention(MACRO_K):", "def _make_direct_sparse_attention(MACRO_K):")
    # 本候选只绑定K512。去掉另一分支的源码，不能保留对已删除大GM窗口的悬空引用。
    a = direct.index("            if MACRO_K == ATTN_K_TILE:")
    b = direct.index("            else:\n                # 两级Scalar FIFO", a)
    c = direct.index("\n            for lane in pl.split_aiv", b)
    direct = direct[:a] + textwrap.indent(textwrap.dedent(direct[b + len("            else:\n"):c]), "            ") + direct[c:]
    direct = once(direct,
        "        cmp_work_kv = pl.create_tensor([cmp_gather_count * CMP_ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16)",
        "        cmp_page_plan = pl.create_tensor([cmp_gather_count, 16], dtype=pl.INT32)\n"
        "        cmp_zero_half = pl.create_tensor([ATTN_K_TILE, 256], dtype=pl.BF16)")
    a = direct.index("        # 每块承担的搬运项数")
    b = direct.index("        # 全部历史共用一组 MIX 任务", a)
    direct = direct[:a] + METADATA + "\n" + direct[b:]
    direct = once(direct, "deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, rope_cs_tid, q_ready]",
                  "deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, cmp_cache_ready_dep, rope_cs_tid, q_ready]")
    direct = once(direct, "            worker = pl.tile.get_block_idx()\n            pl.system.set_ffts(ffts)",
        "            worker = pl.tile.get_block_idx()\n"
        "            # 页计划每次写入，Scalar读取前显式维护本核缓存；不是跨核同步的替代。\n"
        "            pl.system.cacheinvalid()\n            pl.system.fence()\n            pl.system.set_ffts(ffts)")
    old = """cmp_piece = pl.load(cmp_work_kv, [(request * cmp_work_count + tick * MACRO_PARTS + part) * ATTN_K_TILE, dim0],
                                                        [ATTN_K_TILE, 256], target_memory=pl.MemorySpace.Mat)"""
    # 源码缩进随移除K128外层而变化，逐位置取整段到下一piece yield，避免依赖空格凑数。
    a = direct.index("cmp_piece = pl.load(cmp_work_kv,")
    b = direct.index("\n", direct.index("target_memory=pl.MemorySpace.Mat)", a))
    direct = direct[:a] + "cmp_piece = _load_native_cmp_half(cmp_kv_flat, cmp_page_plan, cmp_zero_half,\n" \
        + "                                    request * cmp_work_count + tick * MACRO_PARTS + part, dim0)" + direct[b:]
    for half, dim0 in ((0, 0), (1, 256)):
        a = direct.index(f"cmp{half} = pl.load(cmp_work_kv,")
        b = direct.index("\n", direct.index("target_memory=pl.MemorySpace.Mat)", a))
        direct = direct[:a] + f"cmp{half} = _load_native_cmp_half(cmp_kv_flat, cmp_page_plan, cmp_zero_half,\n" \
            + f"                                old_src // ATTN_K_TILE + part, {dim0})" + direct[b:]
    if "cmp_work_kv" in direct:
        raise ValueError("直接路径仍引用完整压缩KV中间窗口")
    direct = direct.replace("K128保留3份完整KV的L1环以供QK/PV复用；K512按D256供数、PV重载KV，",
                            "长历史直接读取Native压缩KV；每128行提供64B页计划，QK/PV保留D256供数，")
    target = HERE / "candidates/attention_direct_cmp_kv.py"
    text = source + LOADER + direct + "\nsparse_attn_hca_tp1_long = _make_direct_sparse_attention(512)\n"
    compile(text, str(target), "exec")
    target.write_text(text)
    print(target.resolve())


if __name__ == "__main__":
    main()
