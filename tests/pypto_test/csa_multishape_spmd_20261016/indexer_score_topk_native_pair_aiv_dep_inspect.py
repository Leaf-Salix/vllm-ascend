@pl.function(type=pl.FunctionType.Inline, auto_scope=False)
def indexer_score_topk_native_pair_aiv(native_page_bytes_inline8726__ssa_v0: pl.Scalar[pl.INDEX], original_block_idx: pl.Scalar[pl.INDEX], original_subblock_idx: pl.Scalar[pl.INDEX], position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], buf_score_ffts_inline8757__ssa_v0: pl.Tensor[[256], pl.INT64], qr_hadamard_i8_inline583__rv_v2: pl.Tensor[[24576, 128], pl.INT8], coefficients_inline8733__ssa_v0: pl.Tensor[[1024, 384], pl.FP16], idx_block_table__ssa_v0: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32], idx_kv_cache_inline8854__ssa_v0: pl.Tensor[[key_rows_inline8656__ssa_v0, 128], pl.INT8], idx_kv_cache_shift64_inline8790__ssa_v0: pl.Tensor[[shifted_rows_inline8748__ssa_v0, 128], pl.INT8], buf_score_transfer_inline8845__ssa_v0: pl.InOut[pl.Tensor[[288, 1024], pl.FP32]], idx_native_kv_cache__rv_v2: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], pair_arena_inline3226_inline4202__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32]]) -> tuple[pl.Tensor[[288, 1024], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
    pl.func_attr({'mx_tensor_views_blocked': True, 'split_aiv': True, 'dual_aiv_dispatch': True})
    buf_worker_inline8783__ssa_v0: pl.Scalar[pl.INDEX] = original_block_idx
    buf_query_count_inline8792__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
    buf_max_cache_len_inline8842__ssa_v0: pl.Scalar[pl.INDEX] = 0
    for (buf_batch_inline8692__idx_v0, (buf_max_cache_len_inline8842__iter_v1,)) in pl.range(buf_query_count_inline8792__ssa_v0 // 6, init_values=(buf_max_cache_len_inline8842__ssa_v0,)):
        t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [buf_batch_inline8692__idx_v0])
        buf_max_cache_len_inline8842__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(buf_max_cache_len_inline8842__iter_v1, pl.cast(t__tile, pl.INDEX) // 4)
        buf_max_cache_len_inline8842__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(buf_max_cache_len_inline8842__ssa_v3)
    buf_capped_history_inline8793__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(buf_max_cache_len_inline8842__rv_v2, 262144)
    tile_count_inline1302_inline8734__ssa_v0: pl.Scalar[pl.INDEX] = (buf_capped_history_inline8793__ssa_v0 + 1023) // 1024
    leaf_count_inline1301_inline8775__ssa_v0: pl.Scalar[pl.INDEX] = pl.max((buf_capped_history_inline8793__ssa_v0 + 8191) // 8192, 1)
    leaf_tiles_inline1300_inline8753__ssa_v0: pl.Scalar[pl.INDEX] = 8
    extra_leaves_inline1299_inline8670__ssa_v0: pl.Scalar[pl.INDEX] = 0
    leaf_multiple_inline1298_inline8785__ssa_v0: pl.Scalar[pl.INDEX] = 1
    if buf_query_count_inline8792__ssa_v0 // 6 * 3 == 48:
        leaf_multiple_inline1298_inline8785__ssa_v1: pl.Scalar[pl.INDEX] = 3
        leaf_multiple_inline1298_inline8785__phi_v6: pl.Scalar[pl.INDEX] = pl.yield_(leaf_multiple_inline1298_inline8785__ssa_v1)
    else:
        if buf_query_count_inline8792__ssa_v0 // 6 * 3 == 24:
            leaf_multiple_inline1298_inline8785__ssa_v2: pl.Scalar[pl.INDEX] = 3
            leaf_multiple_inline1298_inline8785__phi_v5: pl.Scalar[pl.INDEX] = pl.yield_(leaf_multiple_inline1298_inline8785__ssa_v2)
        else:
            if buf_query_count_inline8792__ssa_v0 // 6 * 6 == 24:
                leaf_multiple_inline1298_inline8785__ssa_v3: pl.Scalar[pl.INDEX] = 6
                leaf_multiple_inline1298_inline8785__phi_v4: pl.Scalar[pl.INDEX] = pl.yield_(leaf_multiple_inline1298_inline8785__ssa_v3)
            else:
                leaf_multiple_inline1298_inline8785__phi_v4: pl.Scalar[pl.INDEX] = pl.yield_(leaf_multiple_inline1298_inline8785__ssa_v0)
            leaf_multiple_inline1298_inline8785__phi_v5: pl.Scalar[pl.INDEX] = pl.yield_(leaf_multiple_inline1298_inline8785__phi_v4)
        leaf_multiple_inline1298_inline8785__phi_v6: pl.Scalar[pl.INDEX] = pl.yield_(leaf_multiple_inline1298_inline8785__phi_v5)
    if 1 < leaf_multiple_inline1298_inline8785__phi_v6:
        balanced_count_inline1297_inline8798__ssa_v0: pl.Scalar[pl.INDEX] = (leaf_count_inline1301_inline8775__ssa_v0 + leaf_multiple_inline1298_inline8785__phi_v6 - 1) // leaf_multiple_inline1298_inline8785__phi_v6 * leaf_multiple_inline1298_inline8785__phi_v6
        if balanced_count_inline1297_inline8798__ssa_v0 <= 32:
            leaf_count_inline1301_inline8775__ssa_v1: pl.Scalar[pl.INDEX] = balanced_count_inline1297_inline8798__ssa_v0
            leaf_tiles_inline1300_inline8753__ssa_v1: pl.Scalar[pl.INDEX] = pl.max(tile_count_inline1302_inline8734__ssa_v0 // balanced_count_inline1297_inline8798__ssa_v0, 1)
            extra_leaves_inline1299_inline8670__ssa_v1: pl.Scalar[pl.INDEX] = tile_count_inline1302_inline8734__ssa_v0 % balanced_count_inline1297_inline8798__ssa_v0
            (extra_leaves_inline1299_inline8670__phi_v2, leaf_count_inline1301_inline8775__phi_v2, leaf_tiles_inline1300_inline8753__phi_v2) = pl.yield_(extra_leaves_inline1299_inline8670__ssa_v1, leaf_count_inline1301_inline8775__ssa_v1, leaf_tiles_inline1300_inline8753__ssa_v1)
        else:
            (extra_leaves_inline1299_inline8670__phi_v2, leaf_count_inline1301_inline8775__phi_v2, leaf_tiles_inline1300_inline8753__phi_v2) = pl.yield_(extra_leaves_inline1299_inline8670__ssa_v0, leaf_count_inline1301_inline8775__ssa_v0, leaf_tiles_inline1300_inline8753__ssa_v0)
        (extra_leaves_inline1299_inline8670__phi_v3, leaf_count_inline1301_inline8775__phi_v3, leaf_tiles_inline1300_inline8753__phi_v3) = pl.yield_(extra_leaves_inline1299_inline8670__phi_v2, leaf_count_inline1301_inline8775__phi_v2, leaf_tiles_inline1300_inline8753__phi_v2)
    else:
        (extra_leaves_inline1299_inline8670__phi_v3, leaf_count_inline1301_inline8775__phi_v3, leaf_tiles_inline1300_inline8753__phi_v3) = pl.yield_(extra_leaves_inline1299_inline8670__ssa_v0, leaf_count_inline1301_inline8775__ssa_v0, leaf_tiles_inline1300_inline8753__ssa_v0)
    buf_max_leaves_inline8794__ssa_v1: pl.Scalar[pl.INDEX] = leaf_count_inline1301_inline8775__phi_v3
    buf_leaf_tiles_inline8752__ssa_v1: pl.Scalar[pl.INDEX] = leaf_tiles_inline1300_inline8753__phi_v3
    buf_extra_leaves_inline8749__ssa_v1: pl.Scalar[pl.INDEX] = extra_leaves_inline1299_inline8670__phi_v3
    pl.system.set_ffts(buf_score_ffts_inline8757__ssa_v0)
    buf_score_lane_inline8716__ssa_v0: pl.Scalar[pl.INDEX] = original_subblock_idx
    for single_item_inline8866__idx_v0 in pl.range(buf_worker_inline8783__ssa_v0, buf_query_count_inline8792__ssa_v0 // 6 * buf_max_leaves_inline8794__ssa_v1, 24):
        if buf_query_count_inline8792__ssa_v0 < 144:
            single_query_inline8669__ssa_v0: pl.Scalar[pl.INDEX] = single_item_inline8866__idx_v0 % (buf_query_count_inline8792__ssa_v0 // 6) * 6
            single_leaf_inline8684__ssa_v0: pl.Scalar[pl.INDEX] = single_item_inline8866__idx_v0 // (buf_query_count_inline8792__ssa_v0 // 6)
            (single_leaf_inline8684__phi_v2, single_query_inline8669__phi_v2) = pl.yield_(single_leaf_inline8684__ssa_v0, single_query_inline8669__ssa_v0)
        else:
            single_query_inline8669__ssa_v1: pl.Scalar[pl.INDEX] = single_item_inline8866__idx_v0 // buf_max_leaves_inline8794__ssa_v1 * 6
            single_leaf_inline8684__ssa_v1: pl.Scalar[pl.INDEX] = single_item_inline8866__idx_v0 % buf_max_leaves_inline8794__ssa_v1
            (single_leaf_inline8684__phi_v2, single_query_inline8669__phi_v2) = pl.yield_(single_leaf_inline8684__ssa_v1, single_query_inline8669__ssa_v1)
        single_batch_inline8814__ssa_v0: pl.Scalar[pl.INDEX] = single_query_inline8669__phi_v2 // 6
        t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [single_batch_inline8814__ssa_v0])
        single_cache_len_inline8773__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX) // 4
        single_last_position_inline8861__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [single_query_inline8669__phi_v2 + 5])
        single_visible_inline8668__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(single_cache_len_inline8773__ssa_v0, (single_last_position_inline8861__tile + 1) // 4), 262144), 0)
        single_begin_inline8697__ssa_v0: pl.Scalar[pl.INDEX] = (single_leaf_inline8684__phi_v2 * buf_leaf_tiles_inline8752__ssa_v1 + pl.min(single_leaf_inline8684__phi_v2, buf_extra_leaves_inline8749__ssa_v1)) * 1024
        single_capacity_inline8666__ssa_v0: pl.Scalar[pl.INDEX] = (buf_leaf_tiles_inline8752__ssa_v1 + pl.cast(single_leaf_inline8684__phi_v2 < buf_extra_leaves_inline8749__ssa_v1, pl.INDEX)) * 1024
        if single_begin_inline8697__ssa_v0 < single_visible_inline8668__ssa_v0:
            single_valid_inline8808__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(single_capacity_inline8666__ssa_v0, single_visible_inline8668__ssa_v0 - single_begin_inline8697__ssa_v0)
            single_steps_inline8664__ssa_v0: pl.Scalar[pl.INDEX] = (single_valid_inline8808__ssa_v0 + 1023) // 1024
            single_roots_inline8822__ssa_v0: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([3, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
            single_segments_inline8768__ssa_v0: pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([6, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
            for (single_step_inline8751__idx_v0, (single_roots_inline8822__iter_v1, single_segments_inline8768__iter_v1)) in pl.range(single_steps_inline8664__ssa_v0, init_values=(single_roots_inline8822__ssa_v0, single_segments_inline8768__ssa_v0)):
                single_score_begin_inline8724__ssa_v0: pl.Scalar[pl.INDEX] = single_begin_inline8697__ssa_v0 + single_step_inline8751__idx_v0 * 1024
                single_transfer_row_inline8662__ssa_v0: pl.Scalar[pl.INDEX] = buf_worker_inline8783__ssa_v0 * 12 + single_step_inline8751__idx_v0 % 2 * 6 + buf_score_lane_inline8716__ssa_v0 * 3
                single_scale_bytes_inline8711__ssa_v0: pl.Tile[[1, 2048], pl.INT8, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.INT8, target_memory=pl.Mem.Vec)
                for (single_scale_page_inline8661__idx_v0, (single_scale_bytes_inline8711__iter_v1,)) in pl.range(32, init_values=(single_scale_bytes_inline8711__ssa_v0,)):
                    single_scale_logical_inline8802__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(single_score_begin_inline8724__ssa_v0 // 32 + single_scale_page_inline8661__idx_v0, pl.max((single_cache_len_inline8773__ssa_v0 - 1) // 32, 0))
                    t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(idx_block_table__ssa_v0, [single_batch_inline8814__ssa_v0, single_scale_logical_inline8802__ssa_v0])
                    single_scale_physical_inline8795__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(t__tile_2, pl.INDEX), 0)
                    single_scale_bytes_inline8711__ssa_v3: pl.Tile[[1, 2048], pl.INT8, pl.Mem.Vec] = pl.tile.gather_row(single_scale_bytes_inline8711__iter_v1, idx_native_kv_cache__rv_v2, [0, single_scale_page_inline8661__idx_v0 * 64], [single_scale_physical_inline8795__ssa_v0, 4096], [1, 64], transpose=False)
                    single_scale_bytes_inline8711__rv_v2: pl.Tile[[1, 2048], pl.INT8, pl.Mem.Vec] = pl.yield_(single_scale_bytes_inline8711__ssa_v3)
                single_scale_half_inline8704__ssa_v0: pl.Tile[[1, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.reinterpret_view(single_scale_bytes_inline8711__rv_v2, dtype=pl.FP16)
                single_kv_scale_inline8654__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.cast(single_scale_half_inline8704__ssa_v0, target_type=pl.FP32, mode='round')
                pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                single_scores_inline8652__ssa_v0: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(buf_score_transfer_inline8845__ssa_v0, [single_transfer_row_inline8662__ssa_v0, 0], [3, 1024], [3, 1024], target_memory=pl.Mem.Vec)
                pl.system.sync_set(1, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV)
                single_score_row_inline8766__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_scores_inline8652__ssa_v0, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                single_scaled_inline8747__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.mul(single_score_row_inline8766__ssa_v0, single_kv_scale_inline8654__ssa_v0)
                single_segments_inline8768__ssa_v3: pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(single_segments_inline8768__iter_v1, single_scaled_inline8747__ssa_v0, [single_step_inline8751__idx_v0 % 2, 0])
                single_score_row_inline8766__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_scores_inline8652__ssa_v0, 1, 0, [1, 1024], target_memory=pl.Mem.Vec)
                single_scaled_inline8747__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.mul(single_score_row_inline8766__ssa_v1, single_kv_scale_inline8654__ssa_v0)
                single_segments_inline8768__ssa_v4: pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(single_segments_inline8768__ssa_v3, single_scaled_inline8747__ssa_v1, [single_step_inline8751__idx_v0 % 2 + 2, 0])
                single_score_row_inline8766__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_scores_inline8652__ssa_v0, 2, 0, [1, 1024], target_memory=pl.Mem.Vec)
                single_scaled_inline8747__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.mul(single_score_row_inline8766__ssa_v2, single_kv_scale_inline8654__ssa_v0)
                single_segments_inline8768__ssa_v5: pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(single_segments_inline8768__ssa_v4, single_scaled_inline8747__ssa_v2, [single_step_inline8751__idx_v0 % 2 + 4, 0])
                if single_step_inline8751__idx_v0 % 2 == 1 or single_step_inline8751__idx_v0 + 1 == single_steps_inline8664__ssa_v0:
                    single_segment_begin_inline8760__ssa_v0: pl.Scalar[pl.INDEX] = single_begin_inline8697__ssa_v0 + single_step_inline8751__idx_v0 // 2 * 2048
                    single_segment_span_inline8784__ssa_v0: pl.Scalar[pl.INDEX] = single_step_inline8751__idx_v0 % 2 * 1024 + 1024
                    single_segment_rows_inline8736__ssa_v0: pl.Tile[[3, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(single_segments_inline8768__ssa_v5, [3, 2048])
                    single_query_lane_inline8644__ssa_v0: pl.Scalar[pl.INDEX] = buf_score_lane_inline8716__ssa_v0 * 3
                    single_position_inline8643__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [single_query_inline8669__phi_v2 + single_query_lane_inline8644__ssa_v0])
                    single_query_visible_inline8891__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(single_cache_len_inline8773__ssa_v0, (single_position_inline8643__tile + 1) // 4), 262144), 0)
                    single_segment_valid_inline8659__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(single_query_visible_inline8891__ssa_v0 - single_segment_begin_inline8760__ssa_v0, single_segment_span_inline8784__ssa_v0), 0)
                    if 0 < single_segment_valid_inline8659__ssa_v0:
                        single_segment_data_inline8728__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_rows_inline8736__ssa_v0, 0, 0, [1, 2048], target_memory=pl.Mem.Vec)
                        logical_i32_inline8817__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(single_segment_begin_inline8760__ssa_v0, pl.INT32)
                        if single_segment_valid_inline8659__ssa_v0 <= 512:
                            t__tmp_v353: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                            tiny_indices_inline8818__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(t__tmp_v353, logical_i32_inline8817__ssa_v0)
                            tiny_tile_inline8821__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_data_inline8728__ssa_v0, 0, 0, [1, 512], target_memory=pl.Mem.Vec)
                            tiny_raw_inline8703__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, single_segment_valid_inline8659__ssa_v0])] = pl.tile.set_validshape(tiny_tile_inline8821__ssa_v0, 1, single_segment_valid_inline8659__ssa_v0)
                            t__tmp_v354: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(tiny_raw_inline8703__ssa_v0, pad_value=pl.PadValue.min)
                            tiny_scores_inline8759__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(t__tmp_v354, -3.4028234663852886e+38)
                            t__tmp_v355: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(tiny_indices_inline8818__ssa_v0, dtype=pl.UINT32)
                            tiny_pairs_inline8750__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(tiny_scores_inline8759__ssa_v0, t__tmp_v355)
                            tiny_pairs_v1_inline8825__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(tiny_pairs_inline8750__ssa_v0, pl.const(64, pl.INT32))
                            tiny_pairs_v2_inline8827__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(tiny_pairs_v1_inline8825__ssa_v0, pl.const(256, pl.INT32))
                            single_new_root_inline8829__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(tiny_pairs_v2_inline8827__ssa_v0, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                            single_new_root_inline8829__phi_v4: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_new_root_inline8829__ssa_v0)
                        else:
                            if single_segment_valid_inline8659__ssa_v0 <= 1024:
                                t__tmp_v356: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                                small_indices_inline8831__ssa_v0: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(t__tmp_v356, logical_i32_inline8817__ssa_v0)
                                small_tile_inline8658__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_data_inline8728__ssa_v0, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                small_raw_inline8678__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, single_segment_valid_inline8659__ssa_v0])] = pl.tile.set_validshape(small_tile_inline8658__ssa_v0, 1, single_segment_valid_inline8659__ssa_v0)
                                t__tmp_v357: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(small_raw_inline8678__ssa_v0, pad_value=pl.PadValue.min)
                                small_scores_inline8712__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(t__tmp_v357, -3.4028234663852886e+38)
                                t__tmp_v358: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(small_indices_inline8831__ssa_v0, dtype=pl.UINT32)
                                small_pairs_inline8838__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(small_scores_inline8712__ssa_v0, t__tmp_v358)
                                small_pairs_v1_inline8830__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(small_pairs_inline8838__ssa_v0, pl.const(64, pl.INT32))
                                small_pairs_v2_inline8834__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(small_pairs_v1_inline8830__ssa_v0, pl.const(256, pl.INT32))
                                small_left_inline8815__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(small_pairs_v2_inline8834__ssa_v0, [1, 1024], [0, 0])
                                small_right_inline8836__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(small_pairs_v2_inline8834__ssa_v0, [1, 1024], [0, 1024])
                                small_tmp_inline8906__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                small_merged_inline8843__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(small_left_inline8815__ssa_v0, small_right_inline8836__ssa_v0, small_tmp_inline8906__ssa_v0, exhausted=False)
                                single_new_root_inline8829__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(small_merged_inline8843__ssa_v0, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                single_new_root_inline8829__phi_v3: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_new_root_inline8829__ssa_v1)
                            else:
                                t__tmp_v359: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                                large_indices_inline8847__ssa_v0: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(t__tmp_v359, logical_i32_inline8817__ssa_v0)
                                large_tile_inline8849__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_data_inline8728__ssa_v0, 0, 0, [1, 2048], target_memory=pl.Mem.Vec)
                                large_raw_inline8851__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, single_segment_valid_inline8659__ssa_v0])] = pl.tile.set_validshape(large_tile_inline8849__ssa_v0, 1, single_segment_valid_inline8659__ssa_v0)
                                t__tmp_v360: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(large_raw_inline8851__ssa_v0, pad_value=pl.PadValue.min)
                                large_scores_inline8857__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(t__tmp_v360, -3.4028234663852886e+38)
                                t__tmp_v361: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(large_indices_inline8847__ssa_v0, dtype=pl.UINT32)
                                large_pairs_inline8819__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(large_scores_inline8857__ssa_v0, t__tmp_v361)
                                large_pairs_v1_inline8663__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(large_pairs_inline8819__ssa_v0, pl.const(64, pl.INT32))
                                large_pairs_v2_inline8667__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(large_pairs_v1_inline8663__ssa_v0, pl.const(256, pl.INT32))
                                large_pairs_v3_inline8715__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(large_pairs_v2_inline8667__ssa_v0, pl.const(1024, pl.INT32))
                                single_new_root_inline8829__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(large_pairs_v3_inline8715__ssa_v0, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                single_new_root_inline8829__phi_v3: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_new_root_inline8829__ssa_v2)
                            single_new_root_inline8829__phi_v4: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_new_root_inline8829__phi_v3)
                        if single_step_inline8751__idx_v0 < 2:
                            single_roots_inline8822__ssa_v3: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(single_roots_inline8822__iter_v1, single_new_root_inline8829__phi_v4, [0, 0])
                            single_roots_inline8822__phi_v5: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__ssa_v3)
                        else:
                            single_old_root_inline8864__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_roots_inline8822__iter_v1, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                            single_merge_tmp_inline8769__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                            single_merged_inline8898__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(single_new_root_inline8829__phi_v4, single_old_root_inline8864__ssa_v0, single_merge_tmp_inline8769__ssa_v0, exhausted=False)
                            single_next_root_inline8859__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_merged_inline8898__ssa_v0, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                            single_roots_inline8822__ssa_v4: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(single_roots_inline8822__iter_v1, single_next_root_inline8859__ssa_v0, [0, 0])
                            single_roots_inline8822__phi_v5: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__ssa_v4)
                        single_roots_inline8822__phi_v6: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__phi_v5)
                    else:
                        single_roots_inline8822__phi_v6: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__iter_v1)
                    single_query_lane_inline8644__ssa_v1: pl.Scalar[pl.INDEX] = buf_score_lane_inline8716__ssa_v0 * 3 + 1
                    single_position_inline8643__tile_1: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [single_query_inline8669__phi_v2 + single_query_lane_inline8644__ssa_v1])
                    single_query_visible_inline8891__ssa_v1: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(single_cache_len_inline8773__ssa_v0, (single_position_inline8643__tile_1 + 1) // 4), 262144), 0)
                    single_segment_valid_inline8659__ssa_v1: pl.Scalar[pl.INDEX] = pl.max(pl.min(single_query_visible_inline8891__ssa_v1 - single_segment_begin_inline8760__ssa_v0, single_segment_span_inline8784__ssa_v0), 0)
                    if 0 < single_segment_valid_inline8659__ssa_v1:
                        single_segment_data_inline8728__ssa_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_rows_inline8736__ssa_v0, 1, 0, [1, 2048], target_memory=pl.Mem.Vec)
                        logical_i32_inline8817__ssa_v1: pl.Scalar[pl.INT32] = pl.cast(single_segment_begin_inline8760__ssa_v0, pl.INT32)
                        if single_segment_valid_inline8659__ssa_v1 <= 512:
                            t__tmp_v362: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                            tiny_indices_inline8818__ssa_v1: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(t__tmp_v362, logical_i32_inline8817__ssa_v1)
                            tiny_tile_inline8821__ssa_v1: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_data_inline8728__ssa_v1, 0, 0, [1, 512], target_memory=pl.Mem.Vec)
                            tiny_raw_inline8703__ssa_v1: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, single_segment_valid_inline8659__ssa_v1])] = pl.tile.set_validshape(tiny_tile_inline8821__ssa_v1, 1, single_segment_valid_inline8659__ssa_v1)
                            t__tmp_v363: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(tiny_raw_inline8703__ssa_v1, pad_value=pl.PadValue.min)
                            tiny_scores_inline8759__ssa_v1: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(t__tmp_v363, -3.4028234663852886e+38)
                            t__tmp_v364: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(tiny_indices_inline8818__ssa_v1, dtype=pl.UINT32)
                            tiny_pairs_inline8750__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(tiny_scores_inline8759__ssa_v1, t__tmp_v364)
                            tiny_pairs_v1_inline8825__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(tiny_pairs_inline8750__ssa_v1, pl.const(64, pl.INT32))
                            tiny_pairs_v2_inline8827__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(tiny_pairs_v1_inline8825__ssa_v1, pl.const(256, pl.INT32))
                            single_new_root_inline8829__ssa_v5: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(tiny_pairs_v2_inline8827__ssa_v1, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                            single_new_root_inline8829__phi_v9: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_new_root_inline8829__ssa_v5)
                        else:
                            if single_segment_valid_inline8659__ssa_v1 <= 1024:
                                t__tmp_v365: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                                small_indices_inline8831__ssa_v1: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(t__tmp_v365, logical_i32_inline8817__ssa_v1)
                                small_tile_inline8658__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_data_inline8728__ssa_v1, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                small_raw_inline8678__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, single_segment_valid_inline8659__ssa_v1])] = pl.tile.set_validshape(small_tile_inline8658__ssa_v1, 1, single_segment_valid_inline8659__ssa_v1)
                                t__tmp_v366: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(small_raw_inline8678__ssa_v1, pad_value=pl.PadValue.min)
                                small_scores_inline8712__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(t__tmp_v366, -3.4028234663852886e+38)
                                t__tmp_v367: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(small_indices_inline8831__ssa_v1, dtype=pl.UINT32)
                                small_pairs_inline8838__ssa_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(small_scores_inline8712__ssa_v1, t__tmp_v367)
                                small_pairs_v1_inline8830__ssa_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(small_pairs_inline8838__ssa_v1, pl.const(64, pl.INT32))
                                small_pairs_v2_inline8834__ssa_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(small_pairs_v1_inline8830__ssa_v1, pl.const(256, pl.INT32))
                                small_left_inline8815__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(small_pairs_v2_inline8834__ssa_v1, [1, 1024], [0, 0])
                                small_right_inline8836__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(small_pairs_v2_inline8834__ssa_v1, [1, 1024], [0, 1024])
                                small_tmp_inline8906__ssa_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                small_merged_inline8843__ssa_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(small_left_inline8815__ssa_v1, small_right_inline8836__ssa_v1, small_tmp_inline8906__ssa_v1, exhausted=False)
                                single_new_root_inline8829__ssa_v6: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(small_merged_inline8843__ssa_v1, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                single_new_root_inline8829__phi_v8: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_new_root_inline8829__ssa_v6)
                            else:
                                t__tmp_v368: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                                large_indices_inline8847__ssa_v1: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(t__tmp_v368, logical_i32_inline8817__ssa_v1)
                                large_tile_inline8849__ssa_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_data_inline8728__ssa_v1, 0, 0, [1, 2048], target_memory=pl.Mem.Vec)
                                large_raw_inline8851__ssa_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, single_segment_valid_inline8659__ssa_v1])] = pl.tile.set_validshape(large_tile_inline8849__ssa_v1, 1, single_segment_valid_inline8659__ssa_v1)
                                t__tmp_v369: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(large_raw_inline8851__ssa_v1, pad_value=pl.PadValue.min)
                                large_scores_inline8857__ssa_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(t__tmp_v369, -3.4028234663852886e+38)
                                t__tmp_v370: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(large_indices_inline8847__ssa_v1, dtype=pl.UINT32)
                                large_pairs_inline8819__ssa_v1: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(large_scores_inline8857__ssa_v1, t__tmp_v370)
                                large_pairs_v1_inline8663__ssa_v1: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(large_pairs_inline8819__ssa_v1, pl.const(64, pl.INT32))
                                large_pairs_v2_inline8667__ssa_v1: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(large_pairs_v1_inline8663__ssa_v1, pl.const(256, pl.INT32))
                                large_pairs_v3_inline8715__ssa_v1: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(large_pairs_v2_inline8667__ssa_v1, pl.const(1024, pl.INT32))
                                single_new_root_inline8829__ssa_v7: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(large_pairs_v3_inline8715__ssa_v1, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                single_new_root_inline8829__phi_v8: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_new_root_inline8829__ssa_v7)
                            single_new_root_inline8829__phi_v9: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_new_root_inline8829__phi_v8)
                        if single_step_inline8751__idx_v0 < 2:
                            single_roots_inline8822__ssa_v7: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(single_roots_inline8822__phi_v6, single_new_root_inline8829__phi_v9, [1, 0])
                            single_roots_inline8822__phi_v9: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__ssa_v7)
                        else:
                            single_old_root_inline8864__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_roots_inline8822__phi_v6, 1, 0, [1, 1024], target_memory=pl.Mem.Vec)
                            single_merge_tmp_inline8769__ssa_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                            single_merged_inline8898__ssa_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(single_new_root_inline8829__phi_v9, single_old_root_inline8864__ssa_v1, single_merge_tmp_inline8769__ssa_v1, exhausted=False)
                            single_next_root_inline8859__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_merged_inline8898__ssa_v1, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                            single_roots_inline8822__ssa_v8: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(single_roots_inline8822__phi_v6, single_next_root_inline8859__ssa_v1, [1, 0])
                            single_roots_inline8822__phi_v9: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__ssa_v8)
                        single_roots_inline8822__phi_v10: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__phi_v9)
                    else:
                        single_roots_inline8822__phi_v10: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__phi_v6)
                    single_query_lane_inline8644__ssa_v2: pl.Scalar[pl.INDEX] = buf_score_lane_inline8716__ssa_v0 * 3 + 2
                    single_position_inline8643__tile_2: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [single_query_inline8669__phi_v2 + single_query_lane_inline8644__ssa_v2])
                    single_query_visible_inline8891__ssa_v2: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(single_cache_len_inline8773__ssa_v0, (single_position_inline8643__tile_2 + 1) // 4), 262144), 0)
                    single_segment_valid_inline8659__ssa_v2: pl.Scalar[pl.INDEX] = pl.max(pl.min(single_query_visible_inline8891__ssa_v2 - single_segment_begin_inline8760__ssa_v0, single_segment_span_inline8784__ssa_v0), 0)
                    if 0 < single_segment_valid_inline8659__ssa_v2:
                        single_segment_data_inline8728__ssa_v2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_rows_inline8736__ssa_v0, 2, 0, [1, 2048], target_memory=pl.Mem.Vec)
                        logical_i32_inline8817__ssa_v2: pl.Scalar[pl.INT32] = pl.cast(single_segment_begin_inline8760__ssa_v0, pl.INT32)
                        if single_segment_valid_inline8659__ssa_v2 <= 512:
                            t__tmp_v371: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                            tiny_indices_inline8818__ssa_v2: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(t__tmp_v371, logical_i32_inline8817__ssa_v2)
                            tiny_tile_inline8821__ssa_v2: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_data_inline8728__ssa_v2, 0, 0, [1, 512], target_memory=pl.Mem.Vec)
                            tiny_raw_inline8703__ssa_v2: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, single_segment_valid_inline8659__ssa_v2])] = pl.tile.set_validshape(tiny_tile_inline8821__ssa_v2, 1, single_segment_valid_inline8659__ssa_v2)
                            t__tmp_v372: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(tiny_raw_inline8703__ssa_v2, pad_value=pl.PadValue.min)
                            tiny_scores_inline8759__ssa_v2: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(t__tmp_v372, -3.4028234663852886e+38)
                            t__tmp_v373: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(tiny_indices_inline8818__ssa_v2, dtype=pl.UINT32)
                            tiny_pairs_inline8750__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(tiny_scores_inline8759__ssa_v2, t__tmp_v373)
                            tiny_pairs_v1_inline8825__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(tiny_pairs_inline8750__ssa_v2, pl.const(64, pl.INT32))
                            tiny_pairs_v2_inline8827__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(tiny_pairs_v1_inline8825__ssa_v2, pl.const(256, pl.INT32))
                            single_new_root_inline8829__ssa_v10: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(tiny_pairs_v2_inline8827__ssa_v2, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                            single_new_root_inline8829__phi_v14: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_new_root_inline8829__ssa_v10)
                        else:
                            if single_segment_valid_inline8659__ssa_v2 <= 1024:
                                t__tmp_v374: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                                small_indices_inline8831__ssa_v2: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(t__tmp_v374, logical_i32_inline8817__ssa_v2)
                                small_tile_inline8658__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_data_inline8728__ssa_v2, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                small_raw_inline8678__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, single_segment_valid_inline8659__ssa_v2])] = pl.tile.set_validshape(small_tile_inline8658__ssa_v2, 1, single_segment_valid_inline8659__ssa_v2)
                                t__tmp_v375: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(small_raw_inline8678__ssa_v2, pad_value=pl.PadValue.min)
                                small_scores_inline8712__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(t__tmp_v375, -3.4028234663852886e+38)
                                t__tmp_v376: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(small_indices_inline8831__ssa_v2, dtype=pl.UINT32)
                                small_pairs_inline8838__ssa_v2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(small_scores_inline8712__ssa_v2, t__tmp_v376)
                                small_pairs_v1_inline8830__ssa_v2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(small_pairs_inline8838__ssa_v2, pl.const(64, pl.INT32))
                                small_pairs_v2_inline8834__ssa_v2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(small_pairs_v1_inline8830__ssa_v2, pl.const(256, pl.INT32))
                                small_left_inline8815__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(small_pairs_v2_inline8834__ssa_v2, [1, 1024], [0, 0])
                                small_right_inline8836__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(small_pairs_v2_inline8834__ssa_v2, [1, 1024], [0, 1024])
                                small_tmp_inline8906__ssa_v2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                small_merged_inline8843__ssa_v2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(small_left_inline8815__ssa_v2, small_right_inline8836__ssa_v2, small_tmp_inline8906__ssa_v2, exhausted=False)
                                single_new_root_inline8829__ssa_v11: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(small_merged_inline8843__ssa_v2, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                single_new_root_inline8829__phi_v13: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_new_root_inline8829__ssa_v11)
                            else:
                                t__tmp_v377: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                                large_indices_inline8847__ssa_v2: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(t__tmp_v377, logical_i32_inline8817__ssa_v2)
                                large_tile_inline8849__ssa_v2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_data_inline8728__ssa_v2, 0, 0, [1, 2048], target_memory=pl.Mem.Vec)
                                large_raw_inline8851__ssa_v2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, single_segment_valid_inline8659__ssa_v2])] = pl.tile.set_validshape(large_tile_inline8849__ssa_v2, 1, single_segment_valid_inline8659__ssa_v2)
                                t__tmp_v378: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(large_raw_inline8851__ssa_v2, pad_value=pl.PadValue.min)
                                large_scores_inline8857__ssa_v2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(t__tmp_v378, -3.4028234663852886e+38)
                                t__tmp_v379: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(large_indices_inline8847__ssa_v2, dtype=pl.UINT32)
                                large_pairs_inline8819__ssa_v2: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(large_scores_inline8857__ssa_v2, t__tmp_v379)
                                large_pairs_v1_inline8663__ssa_v2: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(large_pairs_inline8819__ssa_v2, pl.const(64, pl.INT32))
                                large_pairs_v2_inline8667__ssa_v2: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(large_pairs_v1_inline8663__ssa_v2, pl.const(256, pl.INT32))
                                large_pairs_v3_inline8715__ssa_v2: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(large_pairs_v2_inline8667__ssa_v2, pl.const(1024, pl.INT32))
                                single_new_root_inline8829__ssa_v12: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(large_pairs_v3_inline8715__ssa_v2, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                single_new_root_inline8829__phi_v13: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_new_root_inline8829__ssa_v12)
                            single_new_root_inline8829__phi_v14: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_new_root_inline8829__phi_v13)
                        if single_step_inline8751__idx_v0 < 2:
                            single_roots_inline8822__ssa_v11: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(single_roots_inline8822__phi_v10, single_new_root_inline8829__phi_v14, [2, 0])
                            single_roots_inline8822__phi_v13: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__ssa_v11)
                        else:
                            single_old_root_inline8864__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_roots_inline8822__phi_v10, 2, 0, [1, 1024], target_memory=pl.Mem.Vec)
                            single_merge_tmp_inline8769__ssa_v2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                            single_merged_inline8898__ssa_v2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(single_new_root_inline8829__phi_v14, single_old_root_inline8864__ssa_v2, single_merge_tmp_inline8769__ssa_v2, exhausted=False)
                            single_next_root_inline8859__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_merged_inline8898__ssa_v2, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                            single_roots_inline8822__ssa_v12: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(single_roots_inline8822__phi_v10, single_next_root_inline8859__ssa_v2, [2, 0])
                            single_roots_inline8822__phi_v13: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__ssa_v12)
                        single_roots_inline8822__phi_v14: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__phi_v13)
                    else:
                        single_roots_inline8822__phi_v14: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__phi_v10)
                    single_roots_inline8822__phi_v15: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__phi_v14)
                else:
                    single_roots_inline8822__phi_v15: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(single_roots_inline8822__iter_v1)
                (single_roots_inline8822__rv_v2, single_segments_inline8768__rv_v2) = pl.yield_(single_roots_inline8822__phi_v15, single_segments_inline8768__ssa_v5)
            single_query_lane_inline8644__ssa_v3: pl.Scalar[pl.INDEX] = buf_score_lane_inline8716__ssa_v0 * 3
            single_root_inline8698__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_roots_inline8822__rv_v2, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
            pair_arena_inline3226_inline4202__store: pl.Tensor[[24576, 1024], pl.FP32] = pl.tile.store(single_root_inline8698__ssa_v0, [(single_query_inline8669__phi_v2 + single_query_lane_inline8644__ssa_v3) * 64 + single_leaf_inline8684__phi_v2, 0], pair_arena_inline3226_inline4202__ssa_v0)
            single_query_lane_inline8644__ssa_v4: pl.Scalar[pl.INDEX] = buf_score_lane_inline8716__ssa_v0 * 3 + 1
            single_root_inline8698__ssa_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_roots_inline8822__rv_v2, 1, 0, [1, 1024], target_memory=pl.Mem.Vec)
            pair_arena_inline3226_inline4202__store_v0: pl.Tensor[[24576, 1024], pl.FP32] = pl.tile.store(single_root_inline8698__ssa_v1, [(single_query_inline8669__phi_v2 + single_query_lane_inline8644__ssa_v4) * 64 + single_leaf_inline8684__phi_v2, 0], pair_arena_inline3226_inline4202__ssa_v0)
            single_query_lane_inline8644__ssa_v5: pl.Scalar[pl.INDEX] = buf_score_lane_inline8716__ssa_v0 * 3 + 2
            single_root_inline8698__ssa_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_roots_inline8822__rv_v2, 2, 0, [1, 1024], target_memory=pl.Mem.Vec)
            pair_arena_inline3226_inline4202__store_v1: pl.Tensor[[24576, 1024], pl.FP32] = pl.tile.store(single_root_inline8698__ssa_v2, [(single_query_inline8669__phi_v2 + single_query_lane_inline8644__ssa_v5) * 64 + single_leaf_inline8684__phi_v2, 0], pair_arena_inline3226_inline4202__ssa_v0)
    return (buf_score_transfer_inline8845__ssa_v0, pair_arena_inline3226_inline4202__ssa_v0)
