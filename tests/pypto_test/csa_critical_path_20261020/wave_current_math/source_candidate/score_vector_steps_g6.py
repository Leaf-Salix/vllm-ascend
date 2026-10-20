# SPDX-License-Identifier: Apache-2.0
"""动态 CSA 长档 G6 Score 的 AIV 可暂停一步：当前数学已接入wave候选，待联合验证。

整体思路与阅读顺序
    leaf_state 创建本叶的 roots/segments；vector_step 完整消费一个1024候选
    块；leaf_publish 在全部step之后写回真实query的根。三个helper均不创建
    task，不决定wave，不缓存上次调用metadata，也不执行跨wave发布。

动态输入与固定数学tile
    G6=6是现有Indexer矩阵分组策略，单AIV负责三条query，score tile=1024。
    这两个常量不代表B、T、S、历史长度白名单。Tensor行数/页数/列数保留
    原pl.dynamic，实际query、请求、leaf、step及causal/页边界均为设备Scalar
    或device Tensor。只有原G6/balanced策略可以调用本模块；其它策略沿原路。

跨步状态与核间同步
    roots FP32[3,1024]保存已合并的512对(score,index)，segments FP32[6,1024]
    保存两轮各1024候选。偶数step留下的半段必须保持到下一步；调用者必须接住
    两个返回Tile，并在外层循环作为显式状态携带，不得每step重新初始化。
    两个状态均由原tile.full创建，要求完整静态valid_shape。真实query和候选
    有效范围仍由Scalar/mask控制；当前不把动态valid_shape Tile作为此接口合同。
    同组AIC与双AIV必须处于同一FFTS域。READY后MTE2读取双槽，随后ACK仅表示
    FIFO已读取，仍有scale、排序、归并未完成。暂停点是整个vector_step返回；
    根写回及业务发布在leaf_publish之后，不能用ACK代替根就绪。

数值与生命周期
    复用当前13259f92已采用的数学和连续scale路径：FP16 scale按位解释、FP32乘法、causal有效行、
    sort32/mrgsort顺序、末步512/1024/2048分支和incoming优先平局顺序均不变。
    有工作但空query lane仍load并ACK，只跳过不存在query的数学/根写回。
    roots/segments的物理分配与异步消费者完成尚待联合lowering核对；显式Tile
    入参/返回需要当前正在独立overlay验证的JIT注解修复。源码AST相同不代表
    编译、真机精度或性能通过。此文件不是独立性能候选，也不替代完整CSA。
"""

import pypto.language as pl

from .indexer_group_stages import OFFSET_ROWS_DYN
from .indexer_scale_layout_stages import LAYOUT_WORDS_PER_REQUEST, indexer_scale_layout_base
from .indexer_score_stages import TOPK_SCORE_WORKERS

from ..deepseek_v4_flash_csa.config import FP32_NEG_INF
from .indexer_group_stages import TOKENS_DYN, indexer_query_visibility
from .indexer_score_stages import (
    BLOCK_SIZE, BUFFERED_LONG_SCORE_TILE, CACHE_PAGES_DYN, INDEXER_KEY_BYTES,
    PAGE_BYTES_DYN, TABLE_COLUMNS_DYN, TOPK_PAIR_WIDTH, TOPK_ROWS_PER_QUERY,
    TRANSFER_COLS_DYN, TRANSFER_ROWS_DYN,
)
from .indexer_topk_stages import PAIR_ROWS_DYN
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN

G6_QUERY_GROUP = 6
G6_SCORE_TILE = 1024


@pl.jit.inline(auto_scope=False)
def score_vector_leaf_state_g6():
    """只在进入有效叶时初始化一次；不在每个step或Attention切回时重新创建。"""
    single_roots = pl.tile.full([G6_QUERY_GROUP // 2, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF)
    single_segments = pl.tile.full(
        [G6_QUERY_GROUP, BUFFERED_LONG_SCORE_TILE], dtype=pl.FP32, value=FP32_NEG_INF
    )
    return single_roots, single_segments


@pl.jit.inline(auto_scope=False)
def score_vector_step_g6(
    idx_native_kv_cache: pl.Tensor[[CACHE_PAGES_DYN, PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    position_ids: pl.Tensor[[TOKENS_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    buf_score_transfer: pl.Tensor[[TRANSFER_ROWS_DYN, TRANSFER_COLS_DYN], pl.FP32],
    single_roots: pl.Tile[[3, 1024], pl.FP32, pl.MemorySpace.Vec],
    single_segments: pl.Tile[[6, 1024], pl.FP32, pl.MemorySpace.Vec],
    single_begin: pl.Scalar[pl.INDEX],
    single_step: pl.Scalar[pl.INDEX],
    single_steps: pl.Scalar[pl.INDEX],
    single_cache_len: pl.Scalar[pl.INDEX],
    single_batch: pl.Scalar[pl.INDEX],
    single_query: pl.Scalar[pl.INDEX],
    buf_worker: pl.Scalar[pl.INDEX],
    buf_score_lane: pl.Scalar[pl.INDEX],
    buf_live: pl.Scalar[pl.INDEX],
    single_valid: pl.Scalar[pl.INDEX],
    SCORE_READY_EVENT: pl.constexpr,
    SCORE_CONSUMED_EVENT: pl.constexpr,
    scale_page_coalescing: pl.constexpr = False,
):
    """当前连续scale/排序数学的完整step，真实leaf valid控制尾页。

        metadata布局证明来自本轮已经获取的offsets；在本step重新读取两个
        不可变Scalar，不改变方向0回退、逆序页、尾页补零或FP32乘法/归并次序。
        roots/segments使用局部Tile别名，显式返回给外层作为跨拍状态。
        """
    single_roots_local = single_roots
    single_segments_local = single_segments
    single_page_begin = single_begin // BLOCK_SIZE
    single_page_count = (single_valid + BLOCK_SIZE - 1) // BLOCK_SIZE
    single_contiguous = pl.cast(0, pl.INT32)
    single_first_page = pl.cast(0, pl.INDEX)
    if scale_page_coalescing:
        layout_base = indexer_scale_layout_base(
            pl.cast(pl.tensor.dim(kv_seq_lens, 0), pl.INDEX),
            pl.tensor.dim(position_ids, 0), TOPK_SCORE_WORKERS * 2,
        )
        layout_row = layout_base + LAYOUT_WORDS_PER_REQUEST * single_batch
        single_contiguous = pl.read(offsets, [layout_row + 1])
        single_first_page = pl.cast(pl.read(offsets, [layout_row]), pl.INDEX)
        single_first_page = single_first_page + pl.cast(single_contiguous, pl.INDEX) * single_page_begin
    single_score_begin = single_begin + single_step * G6_SCORE_TILE
    single_transfer_row = (
        buf_worker * 2 * G6_QUERY_GROUP
        + single_step % 2 * G6_QUERY_GROUP
        + buf_score_lane * (G6_QUERY_GROUP // 2)
    )
    # A3没有INT8常量填充。FP16零的所有字节同样为零；显式
    # 标记zero padding，使快慢两条路径的Tile类型一致。
    # 此处全区域有效，fillpad不改变数值；尾页路径才实际补零。
    single_scale_bytes = pl.tile.reinterpret_view(
        pl.fillpad(pl.tile.full([1, G6_SCORE_TILE], dtype=pl.FP16, value=0.0),
                   pad_value=pl.PadValue.zero), pl.INT8
    )
    if scale_page_coalescing and single_contiguous != 0:
        # Native每页的key后仍是原64B scale；源行距由Tensor描述符
        # 的page_bytes提供。一次二维MTE2搬入多页，不碰key区域。
        run_pages = pl.min(G6_SCORE_TILE // BLOCK_SIZE,
                           single_page_count - single_step * (G6_SCORE_TILE // BLOCK_SIZE))
        run_first = single_first_page + single_step * (G6_SCORE_TILE // BLOCK_SIZE) * pl.cast(single_contiguous, pl.INDEX)
        run_lowest = run_first
        if single_contiguous < 0:
            run_lowest = run_first - run_pages + 1
        run_bytes = pl.load(
            idx_native_kv_cache,
            [run_lowest, INDEXER_KEY_BYTES],
            [G6_SCORE_TILE // BLOCK_SIZE, BLOCK_SIZE * 2], valid_shape=[run_pages, BLOCK_SIZE * 2],
        )
        # 尾块只加载真实页。按FP16填零未加载的行，再展平，避免
        # INT8常量填充的不支持及将UB未初始化数据带入后续数学。
        run_half = pl.tile.reinterpret_view(run_bytes, pl.FP16)
        run_clean = pl.fillpad(run_half, pad_value=pl.PadValue.zero)
        run_full = pl.set_validshape(run_clean, G6_SCORE_TILE // BLOCK_SIZE, BLOCK_SIZE)
        run_flat = pl.reshape(run_full, [1, G6_SCORE_TILE])
        if single_contiguous < 0:
            # MTE2始终按物理地址递增读。逆序页在UB按64B页倒序，
            # 每页内部32个FP16保持原顺序。TGATHERB每索引搬32B，
            # 两个索引组成一页，仅需64个UINT32字节偏移。
            run_blocks = pl.tile.ci(0, [1, 2 * (G6_SCORE_TILE // BLOCK_SIZE)], dtype=pl.INT32)
            run_page_ids = pl.tile.shrs(run_blocks, 1)
            run_halves = pl.sub(run_blocks, pl.mul(run_page_ids, 2))
            run_reverse_pages = pl.maximum(
                pl.add(pl.neg(run_page_ids), pl.cast(run_pages - 1, pl.INT32)), 0
            )
            run_offsets = pl.add(pl.mul(run_reverse_pages, BLOCK_SIZE * 2), pl.mul(run_halves, 32))
            run_reversed = pl.tile.gatherb(run_flat, pl.tile.reinterpret_view(run_offsets, pl.UINT32))
            # 尾页后的索引钳制到已加载第0页，不访问未加载或负地址；
            # 随后按真实页数补零，保证无效区不会带入陈旧UB数据。
            run_used = pl.set_validshape(run_reversed, 1, run_pages * BLOCK_SIZE)
            run_flat = pl.set_validshape(pl.fillpad(run_used, pad_value=pl.PadValue.zero), 1, G6_SCORE_TILE)
        single_scale_bytes = pl.tile.reinterpret_view(run_flat, pl.INT8)
    else:
        # 两AIV读取同一候选区间的scale；页表与slab容量检查先于gather。
        for single_scale_page in pl.range(G6_SCORE_TILE // BLOCK_SIZE):
            single_scale_logical = pl.min(
                single_score_begin // BLOCK_SIZE + single_scale_page,
                pl.max((single_cache_len - 1) // BLOCK_SIZE, 0),
            )
            if single_scale_logical >= 0 and single_scale_logical < pl.tensor.dim(idx_block_table, 1):
                single_scale_physical = pl.cast(
                    pl.read(idx_block_table, [single_batch, single_scale_logical]), pl.INDEX
                )
                if single_scale_physical >= 0 and single_scale_physical < pl.tensor.dim(
                    idx_native_kv_cache, 0
                ):
                    single_scale_bytes = pl.gather_row(
                        single_scale_bytes,
                        idx_native_kv_cache,
                        [0, single_scale_page * BLOCK_SIZE * 2],
                        [single_scale_physical, INDEXER_KEY_BYTES],
                        [1, BLOCK_SIZE * 2],
                    )
    single_scale_half = pl.tile.reinterpret_view(single_scale_bytes, pl.FP16)
    single_kv_scale = pl.cast(single_scale_half, pl.FP32)
    pl.system.sync_wait(SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
    single_scores = pl.load(
        buf_score_transfer, [single_transfer_row, 0], [G6_QUERY_GROUP // 2, G6_SCORE_TILE]
    )
    pl.system.sync_set(
        SCORE_CONSUMED_EVENT, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV
    )
    for single_local in pl.unroll(G6_QUERY_GROUP // 2):
        single_score_row = pl.tile.extract(
            single_scores, single_local, 0, [1, G6_SCORE_TILE], target_memory=pl.MemorySpace.Vec
        )
        single_scaled = pl.mul(single_score_row, single_kv_scale)
        single_segments_local = pl.tile.assemble(
            single_segments_local, single_scaled, [single_local * 2 + single_step % 2, 0]
        )
    if single_step % 2 == 1 or single_step + 1 == single_steps:
        single_segment_begin = single_begin + single_step // 2 * (2 * G6_SCORE_TILE)
        single_segment_span = (single_step % 2 + 1) * G6_SCORE_TILE
        single_segment_rows = pl.tile.reshape(
            single_segments_local, [G6_QUERY_GROUP // 2, 2 * BUFFERED_LONG_SCORE_TILE]
        )
        for single_local in pl.unroll(G6_QUERY_GROUP // 2):
            single_query_lane = buf_score_lane * (G6_QUERY_GROUP // 2) + single_local
            if single_query_lane < buf_live:
                (single_query_visible, _causal) = indexer_query_visibility(
                    bounds, kv_seq_lens, position_ids, single_batch, single_query + single_query_lane
                )
                single_segment_valid = pl.max(
                    pl.min(single_query_visible - single_segment_begin, single_segment_span), 0
                )
                if single_segment_valid > 0:
                    single_segment_data = pl.tile.extract(
                        single_segment_rows,
                        single_local,
                        0,
                        [1, 2 * BUFFERED_LONG_SCORE_TILE],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    logical_i32 = pl.cast(single_segment_begin, pl.INT32)
                    if single_segment_valid <= 512:
                        tiny_indices = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), logical_i32)
                        tiny_tile = pl.tile.extract(
                            single_segment_data, 0, 0, [1, 512], target_memory=pl.MemorySpace.Vec
                        )
                        tiny_raw = pl.set_validshape(tiny_tile, 1, single_segment_valid)
                        tiny_scores = pl.maximum(
                            pl.tile.fillpad(tiny_raw, pad_value=pl.PadValue.min), FP32_NEG_INF
                        )
                        tiny_pairs = pl.sort32(
                            tiny_scores, pl.reinterpret_view(tiny_indices, pl.UINT32)
                        )
                        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=64)
                        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=256)
                        # 提取512对(score,index)，限制循环root存储寿命；保留原sort32→merge输入顺序。
                        single_new_root = pl.tile.extract(
                            tiny_pairs, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
                        )
                    elif single_segment_valid <= 1024:
                        small_indices = pl.add(
                            pl.tile.arange(0, [1, 1024], dtype=pl.INT32), logical_i32
                        )
                        small_tile = pl.tile.extract(
                            single_segment_data, 0, 0, [1, 1024], target_memory=pl.MemorySpace.Vec
                        )
                        small_raw = pl.set_validshape(small_tile, 1, single_segment_valid)
                        small_scores = pl.maximum(
                            pl.tile.fillpad(small_raw, pad_value=pl.PadValue.min), FP32_NEG_INF
                        )
                        small_pairs = pl.sort32(
                            small_scores, pl.reinterpret_view(small_indices, pl.UINT32)
                        )
                        small_pairs = pl.mrgsort(small_pairs, block_len=64)
                        small_pairs = pl.mrgsort(small_pairs, block_len=256)
                        small_left = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
                        small_right = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 1024])
                        small_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
                        small_merged = pl.tile.mrgsort(small_left, small_right, tmp=small_tmp)
                        single_new_root = pl.tile.extract(
                            small_merged, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
                        )
                    else:
                        large_indices = pl.add(
                            pl.tile.arange(0, [1, 2048], dtype=pl.INT32), logical_i32
                        )
                        large_tile = pl.tile.extract(
                            single_segment_data, 0, 0, [1, 2048], target_memory=pl.MemorySpace.Vec
                        )
                        large_raw = pl.set_validshape(large_tile, 1, single_segment_valid)
                        large_scores = pl.maximum(
                            pl.tile.fillpad(large_raw, pad_value=pl.PadValue.min), FP32_NEG_INF
                        )
                        large_pairs = pl.sort32(
                            large_scores, pl.reinterpret_view(large_indices, pl.UINT32)
                        )
                        large_pairs = pl.mrgsort(large_pairs, block_len=64)
                        large_pairs = pl.mrgsort(large_pairs, block_len=256)
                        large_pairs = pl.mrgsort(large_pairs, block_len=1024)
                        single_new_root = pl.tile.extract(
                            large_pairs, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
                        )
                    if single_step < 2:
                        single_roots_local = pl.tile.assemble(
                            single_roots_local, single_new_root, [single_local, 0]
                        )
                    else:
                        single_old_root = pl.tile.extract(
                            single_roots_local,
                            single_local,
                            0,
                            [1, TOPK_PAIR_WIDTH],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        single_merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
                        # 新2048段在旧根之前，保留同分时后到段优先，不交换两输入。
                        single_merged = pl.tile.mrgsort(
                            single_new_root, single_old_root, tmp=single_merge_tmp
                        )
                        single_next_root = pl.tile.extract(
                            single_merged, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
                        )
                        single_roots_local = pl.tile.assemble(
                            single_roots_local, single_next_root, [single_local, 0]
                        )

    return single_roots_local, single_segments_local


@pl.jit.inline(auto_scope=False)
def score_vector_leaf_publish_g6(
    single_roots: pl.Tile[[3, 1024], pl.FP32, pl.MemorySpace.Vec],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
    buf_score_lane: pl.Scalar[pl.INDEX],
    buf_live: pl.Scalar[pl.INDEX],
    single_query: pl.Scalar[pl.INDEX],
    single_leaf: pl.Scalar[pl.INDEX],
):
    """所有step数学结束后写根；外层必须在此后再发布本波根完成。"""
    for single_local in pl.unroll(G6_QUERY_GROUP // 2):
        single_query_lane = buf_score_lane * (G6_QUERY_GROUP // 2) + single_local
        if single_query_lane < buf_live:
            single_root = pl.tile.extract(
                single_roots, single_local, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
            )
            pl.store(
                single_root,
                [(single_query + single_query_lane) * TOPK_ROWS_PER_QUERY + single_leaf, 0],
                pair_arena,
            )
    return pair_arena
