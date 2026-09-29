for single_item in pl.range(
    buf_worker, buf_query_count // query_group_size * buf_max_leaves, TOPK_SCORE_WORKERS
):
    if buf_query_count < query_group_size * TOPK_SCORE_WORKERS:
        single_query = single_item % (buf_query_count // query_group_size) * query_group_size
        single_leaf = single_item // (buf_query_count // query_group_size)
    else:
        single_query = single_item // buf_max_leaves * query_group_size
        single_leaf = single_item % buf_max_leaves
    single_batch = single_query // S
    single_cache_len = pl.read(kv_seq_lens, [single_batch]) // COMPRESS_RATIO
    single_last_position = pl.read(position_ids, [single_query + query_group_size - 1])
    single_visible = pl.max(
        pl.min(pl.min(single_cache_len, (single_last_position + 1) // COMPRESS_RATIO),
               TOPK_MAX_CANDIDATES), 0
    )
    single_begin = (single_leaf * buf_leaf_tiles + pl.min(single_leaf, buf_extra_leaves)) * score_tile
    single_capacity = (buf_leaf_tiles + pl.cast(single_leaf < buf_extra_leaves, pl.INDEX)) * score_tile
    if single_begin < single_visible:
        single_valid = pl.min(single_capacity, single_visible - single_begin)
        single_steps = (single_valid + score_tile - 1) // score_tile
        # Native QLI ProcessVec1 assigns complete query rows to each AIV.
        # Keep one root per query and a two-step contiguous 2048-score segment.
        single_roots = pl.tile.full(
            [query_group_size // 2, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF
        )
        single_segments = pl.tile.full(
            [query_group_size, BUFFERED_LONG_SCORE_TILE], dtype=pl.FP32, value=FP32_NEG_INF
        )
        for single_step in pl.range(single_steps):
            single_score_begin = single_begin + single_step * score_tile
            single_transfer_row = (
                buf_worker * 2 * query_group_size + single_step % 2 * query_group_size
                + buf_score_lane * (query_group_size // 2)
            )
            # Scales are shared across the three queries on one AIV. Reading
            # the complete candidate tile duplicates scale traffic across AIVs.
            single_scale_bytes = pl.tile.create([1, score_tile * 2], dtype=pl.INT8)
            for single_scale_page in pl.range(score_tile // BLOCK_SIZE):
                single_scale_logical = pl.min(
                    single_score_begin // BLOCK_SIZE + single_scale_page,
                    pl.max((single_cache_len - 1) // BLOCK_SIZE, 0),
                )
                single_scale_physical = pl.max(
                    pl.cast(pl.read(idx_block_table, [single_batch, single_scale_logical]), pl.INDEX), 0
                )
                single_scale_bytes = pl.gather_row(
                    single_scale_bytes, idx_native_kv_cache,
                    [0, single_scale_page * BLOCK_SIZE * 2],
                    [single_scale_physical, INDEXER_KEY_BYTES], [1, BLOCK_SIZE * 2],
                )
            single_scale_half = pl.tile.reinterpret_view(single_scale_bytes, pl.FP16)
            single_kv_scale = pl.cast(single_scale_half, pl.FP32)
            pl.system.sync_wait(SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
            single_scores = pl.load(
                buf_score_transfer, [single_transfer_row, 0], [query_group_size // 2, score_tile]
            )
            pl.system.sync_set(
                SCORE_CONSUMED_EVENT, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV
            )
            for single_local in pl.unroll(query_group_size // 2):
                single_score_row = pl.tile.extract(
                    single_scores, single_local, 0, [1, score_tile], target_memory=pl.MemorySpace.Vec
                )
                single_scaled = pl.mul(single_score_row, single_kv_scale)
                # Full-width row assemble avoids a partial-column UB move.
                single_segments = pl.tile.assemble(
                    single_segments, single_scaled, [single_local * 2 + single_step % 2, 0]
                )
            if single_step % 2 == 1 or single_step + 1 == single_steps:
                single_segment_begin = single_begin + (single_step // 2) * (2 * score_tile)
                single_segment_span = (single_step % 2 + 1) * score_tile
                single_segment_rows = pl.tile.reshape(
                    single_segments, [query_group_size // 2, 2 * BUFFERED_LONG_SCORE_TILE]
                )
                for single_local in pl.unroll(query_group_size // 2):
                    single_query_lane = buf_score_lane * (query_group_size // 2) + single_local
                    single_position = pl.read(position_ids, [single_query + single_query_lane])
                    single_query_visible = pl.max(
                        pl.min(pl.min(single_cache_len, (single_position + 1) // COMPRESS_RATIO),
                               TOPK_MAX_CANDIDATES), 0
                    )
                    single_segment_valid = pl.max(
                        pl.min(single_query_visible - single_segment_begin, single_segment_span), 0
                    )
                    if single_segment_valid > 0:
                        single_segment_data = pl.tile.extract(
                            single_segment_rows, single_local, 0, [1, 2 * BUFFERED_LONG_SCORE_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        single_new_root = indexer_topk_tile_pairs(
                            single_segment_data, single_segment_begin, single_segment_valid
                        )
                        if single_step < 2:
                            single_roots = pl.tile.assemble(
                                single_roots, single_new_root, [single_local, 0]
                            )
                        else:
                            single_old_root = pl.tile.extract(
                                single_roots, single_local, 0, [1, TOPK_PAIR_WIDTH],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            single_merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
                            # Native merges each new contiguous 2048 segment
                            # before the accumulated root when scores tie.
                            single_merged = pl.tile.mrgsort(
                                single_new_root, single_old_root, tmp=single_merge_tmp
                            )
                            single_next_root = pl.tile.extract(
                                single_merged, 0, 0, [1, TOPK_PAIR_WIDTH],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            single_roots = pl.tile.assemble(
                                single_roots, single_next_root, [single_local, 0]
                            )
        for single_local in pl.unroll(query_group_size // 2):
            single_query_lane = buf_score_lane * (query_group_size // 2) + single_local
            single_root = pl.tile.extract(
                single_roots, single_local, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
            )
            pl.store(
                single_root,
                [(single_query + single_query_lane) * TOPK_ROWS_PER_QUERY + single_leaf, 0], pair_arena,
            )
