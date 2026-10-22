# SPDX-License-Identifier: Apache-2.0
"""来自已采用多task的精确Top-K短前缀；保留相等项和完整回退。

保留旧根阈值、GE相等项、稳定压缩、原归并方向及完整回退。
仅复制helper；真实query/causal/页表与同步仍由当前TND调用方管理。
算法与6ba55b45/c49ee5b2一致；其旧S6记录不代替本轮设备/混长/同分验收。
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
def gather_sorted_prefix(
    values: pl.Tile[[1, MAX_SEGMENT], pl.FP32],
    selected: pl.Tile[[1, MAX_SEGMENT], pl.INT32],
    logical_begin: pl.Scalar[pl.INT32],
    kept: pl.Scalar[pl.INT32],
):
    """只搬运将被实际排序的前缀容量，避免先gather2048项再丢弃尾部。"""
    # A3的INT32窄TEXTRACT不支持；窄view直接fillpad也有后端容量约束。
    # 因此保留原2048索引补零，只缩短随后真正耗时的score gather和全局索引偏移。
    # 短slice保留母块物理跨度，整数加零将其物化为512/1024容量的独立Tile；
    # 索引不经过浮点转换，额外Vector复制的成本也必须计入完整性能。
    selected_valid = pl.set_validshape(selected, 1, pl.cast(kept, pl.INDEX))
    selected_clean = pl.fillpad(selected_valid, pad_value=pl.PadValue.zero)
    selected_clean = pl.set_validshape(selected_clean, 1, MAX_SEGMENT)
    if kept <= 512:
        # 只消费压缩结果的有效前缀；补零的索引安全指向原段第0项。
        # 排序前再把无效score置最小值，旧根已满512项，伪尾不能进入结果。
        tiny_view = pl.tile.slice(selected_clean, [1, 512], [0, 0])
        tiny_clean = pl.add(tiny_view, pl.const(0, pl.INT32))
        tiny_tmp = pl.tile.create([1, 512], dtype=pl.INT32)
        tiny_values = pl.tile.gather(values, tiny_clean, tiny_tmp)
        tiny_global = pl.add(tiny_clean, logical_begin)
        tiny_values = pl.set_validshape(tiny_values, 1, pl.cast(kept, pl.INDEX))
        tiny_values = pl.maximum(pl.fillpad(tiny_values, pad_value=pl.PadValue.min), NEGATIVE_SENTINEL)
        tiny_pairs = pl.sort32(tiny_values, pl.reinterpret_view(tiny_global, pl.UINT32))
        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=64)
        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=256)
        root = pl.tile.extract(tiny_pairs, 0, 0, [1, PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    else:
        # 513～1024项沿用原1024排序树及左段优先的归并顺序。
        small_view = pl.tile.slice(selected_clean, [1, 1024], [0, 0])
        small_clean = pl.add(small_view, pl.const(0, pl.INT32))
        small_tmp = pl.tile.create([1, 1024], dtype=pl.INT32)
        small_values = pl.tile.gather(values, small_clean, small_tmp)
        small_global = pl.add(small_clean, logical_begin)
        small_values = pl.set_validshape(small_values, 1, pl.cast(kept, pl.INDEX))
        small_values = pl.maximum(pl.fillpad(small_values, pad_value=pl.PadValue.min), NEGATIVE_SENTINEL)
        small_pairs = pl.sort32(small_values, pl.reinterpret_view(small_global, pl.UINT32))
        small_pairs = pl.mrgsort(small_pairs, block_len=64)
        small_pairs = pl.mrgsort(small_pairs, block_len=256)
        small_left = pl.tile.slice(small_pairs, [1, PAIR_WIDTH], [0, 0])
        small_right = pl.tile.slice(small_pairs, [1, PAIR_WIDTH], [0, PAIR_WIDTH])
        small_merge_tmp = pl.tile.create([1, 2 * PAIR_WIDTH], dtype=pl.FP32)
        small_merged = pl.tile.mrgsort(small_left, small_right, tmp=small_merge_tmp)
        root = pl.tile.extract(small_merged, 0, 0, [1, PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    return root


@pl.jit.inline(auto_scope=False)
def threshold_merge_top512(
    values: pl.Tile[[1, MAX_SEGMENT], pl.FP32],
    old_root: pl.Tile[[1, PAIR_WIDTH], pl.FP32],
    logical_begin: pl.Scalar[pl.INT32],
    valid: pl.Scalar[pl.INT32],
):
    threshold = pl.tile.read(old_root, [0, PAIR_WIDTH - 2])
    kept = valid
    # 旧根尾部仍为 pad 时，不存在512项完整阈值，不进行裁剪。
    if threshold > NEGATIVE_SENTINEL:
        valid_values = pl.set_validshape(values, 1, pl.cast(valid, pl.INDEX))
        compare_tmp = pl.tile.create([1, COMPARE_SCRATCH_BYTES], dtype=pl.UINT8)
        selected, count = pl.tile.gather_compare(
            valid_values,
            threshold,
            compare_tmp,
            cmp_mode="ge",
            out_cols=MAX_SEGMENT,
        )
        kept = pl.tile.read(count, [0, 0])
    else:
        # 分支输出需要有统一的 Tile 类型；此处的占位不会被回退分支读取。
        selected = pl.tile.full([1, MAX_SEGMENT], dtype=pl.INT32, value=0)
    if kept == 0:
        next_root = old_root
    else:
        if kept < valid and kept <= 1024 and (kept <= 512 or valid > 1024) and threshold > NEGATIVE_SENTINEL:
            new_root = gather_sorted_prefix(values, selected, logical_begin, kept)
        else:
            # 只在原排序回退时建立完整索引，快速路径复用压缩得到的位置。
            original_indices = pl.add(pl.tile.arange(0, [1, MAX_SEGMENT], dtype=pl.INT32), logical_begin)
            new_root = sort_top512(values, original_indices, valid)
        merge_tmp = pl.tile.create([1, 2 * PAIR_WIDTH], dtype=pl.FP32)
        merged = pl.tile.mrgsort(new_root, old_root, tmp=merge_tmp)
        next_root = pl.tile.extract(merged, 0, 0, [1, PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    return next_root
