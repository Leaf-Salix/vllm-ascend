# SPDX-License-Identifier: Apache-2.0
"""长历史 CSA 的精确 Top-512 阈值候选：只减少不可能胜出的排序工作。

整体思路
    输入是当前 2048 容量的 FP32 score 段和已经排好的旧 Top-512。
    旧根第512个 score 是可靠的下界：严格低于它的新增项不能进最终根；
    相等项全部保留，继续使用 incoming 优先的原归并方向。
    筛选在 AIV 完成：新后端以真实 Scalar 阈值做 GE 比较并保序压缩
    位置，直接保留相等项，不创建中间 0/1 flag，也不改变原始 score。
    GE 是本轮 A3 后端的明确扩展，须验证实际生成与设备语义。

执行范围与回退
    只由既有 balanced 长历史路线调用，不增加 batch 单点策略。
    首段及 <=512 项由调用方保留原函数。没有512个真实旧项时不剪枝。
    保序压缩后 <=512 或 <=1024 才缩短排序；更多项继续原排序，所有
    筛选、计数、分支成本都由完整 CSA 测量承担。零项直接保留旧根。
    无效尾部不参与筛选；索引使用 INT32 位表示，不能经 FP32 数值转换。

实现状态
    依赖本轮独立 PyPTO 计数对齐修复及 PTOAS A3 Scalar compare lowering。
    46组 A3 排序 probe 支持保序压缩的同分顺序，尚不代替本函数/完整
    CSA 的设备验证。本文件是测试候选，未采用到正式 ops。
"""
import pypto.language as pl

NEGATIVE_SENTINEL = -3.4028234663852886e38
MAX_SEGMENT = 2048
PAIR_WIDTH = 1024
COMPARE_SCRATCH_BYTES = 18432


@pl.jit.inline(auto_scope=False)
def sort_top512(
    values: pl.Tile[[1, MAX_SEGMENT], pl.FP32],
    indices: pl.Tile[[1, MAX_SEGMENT], pl.INT32],
    valid: pl.Scalar[pl.INT32],
):
    """严格沿原 512/1024/2048 排序树，保留同分先后关系。"""
    valid_index = pl.cast(valid, pl.INDEX)
    if valid <= 512:
        tiny_v = pl.tile.extract(values, 0, 0, [1, 512], target_memory=pl.MemorySpace.Vec)
        tiny_v = pl.set_validshape(tiny_v, 1, valid_index)
        tiny_v = pl.maximum(pl.fillpad(tiny_v, pad_value=pl.PadValue.min), NEGATIVE_SENTINEL)
        tiny_i = pl.tile.slice(indices, [1, 512], [0, 0])
        tiny_pairs = pl.sort32(tiny_v, pl.reinterpret_view(tiny_i, pl.UINT32))
        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=64)
        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=256)
        root = pl.tile.extract(tiny_pairs, 0, 0, [1, PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    elif valid <= 1024:
        small_v = pl.tile.extract(values, 0, 0, [1, 1024], target_memory=pl.MemorySpace.Vec)
        small_v = pl.set_validshape(small_v, 1, valid_index)
        small_v = pl.maximum(pl.fillpad(small_v, pad_value=pl.PadValue.min), NEGATIVE_SENTINEL)
        small_i = pl.tile.slice(indices, [1, 1024], [0, 0])
        small_pairs = pl.sort32(small_v, pl.reinterpret_view(small_i, pl.UINT32))
        small_pairs = pl.mrgsort(small_pairs, block_len=64)
        small_pairs = pl.mrgsort(small_pairs, block_len=256)
        small_left = pl.tile.slice(small_pairs, [1, PAIR_WIDTH], [0, 0])
        small_right = pl.tile.slice(small_pairs, [1, PAIR_WIDTH], [0, 1024])
        small_tmp = pl.tile.create([1, 2 * PAIR_WIDTH], dtype=pl.FP32)
        small_merged = pl.tile.mrgsort(small_left, small_right, tmp=small_tmp)
        root = pl.tile.extract(small_merged, 0, 0, [1, PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    else:
        large_v = pl.set_validshape(values, 1, pl.cast(valid, pl.INDEX))
        large_v = pl.maximum(pl.fillpad(large_v, pad_value=pl.PadValue.min), NEGATIVE_SENTINEL)
        large_pairs = pl.sort32(large_v, pl.reinterpret_view(indices, pl.UINT32))
        large_pairs = pl.mrgsort(large_pairs, block_len=64)
        large_pairs = pl.mrgsort(large_pairs, block_len=256)
        large_pairs = pl.mrgsort(large_pairs, block_len=1024)
        root = pl.tile.extract(large_pairs, 0, 0, [1, PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    return root


@pl.jit.inline(auto_scope=False)
def threshold_merge_top512(
    values: pl.Tile[[1, MAX_SEGMENT], pl.FP32],
    old_root: pl.Tile[[1, PAIR_WIDTH], pl.FP32],
    logical_begin: pl.Scalar[pl.INT32],
    valid: pl.Scalar[pl.INT32],
):
    threshold = pl.tile.read(old_root, [0, PAIR_WIDTH - 2])
    original_indices = pl.add(pl.tile.arange(0, [1, MAX_SEGMENT], dtype=pl.INT32), logical_begin)
    kept = valid
    # 旧根尾部仍为 pad 时，不存在512项完整阈值，不进行裁剪。
    if threshold > NEGATIVE_SENTINEL:
        valid_values = pl.set_validshape(values, 1, pl.cast(valid, pl.INDEX))
        compare_tmp = pl.tile.create([1, COMPARE_SCRATCH_BYTES], dtype=pl.UINT8)
        selected, count = pl.tile.gather_compare(
            valid_values, threshold, compare_tmp, cmp_mode="ge", out_cols=MAX_SEGMENT,
        )
        kept = pl.tile.read(count, [0, 0])
    else:
        # 分支输出需要有统一的 Tile 类型；此处的占位不会被回退分支读取。
        selected = pl.tile.full([1, MAX_SEGMENT], dtype=pl.INT32, value=0)
    if kept == 0:
        next_root = old_root
    else:
        if kept < valid and kept <= 1024 and (kept <= 512 or valid > 1024) and threshold > NEGATIVE_SENTINEL:
            # 只读 gather_compare 写入的有效前缀；先补零索引，使下面的
            # 全容量 gather 不会访问未初始化或越界的下标。尾 score 在排序
            # 入口变为最小值，旧根有512个真实项，伪尾项不能进入最终根。
            valid_indices = pl.set_validshape(selected, 1, pl.cast(kept, pl.INDEX))
            clean_indices = pl.fillpad(valid_indices, pad_value=pl.PadValue.zero)
            clean_indices = pl.set_validshape(clean_indices, 1, MAX_SEGMENT)
            gather_tmp = pl.tile.create([1, MAX_SEGMENT], dtype=pl.INT32)
            compressed_values = pl.tile.gather(values, clean_indices, gather_tmp)
            compressed_indices = pl.add(clean_indices, logical_begin)
            new_root = sort_top512(compressed_values, compressed_indices, kept)
        else:
            new_root = sort_top512(values, original_indices, valid)
        merge_tmp = pl.tile.create([1, 2 * PAIR_WIDTH], dtype=pl.FP32)
        merged = pl.tile.mrgsort(new_root, old_root, tmp=merge_tmp)
        next_root = pl.tile.extract(merged, 0, 0, [1, PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    return next_root
