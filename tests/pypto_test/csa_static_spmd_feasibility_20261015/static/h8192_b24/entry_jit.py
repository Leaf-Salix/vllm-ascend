# 同源 kernel-mode 入口；必须通过 CPU 编译与 ABI 闸门后才可在 task-submit 内调用。
import pypto.language as pl
B_DYN = pl.dynamic('B_DYN')
CMP_BLOCK_NUM_DYN = pl.dynamic('CMP_BLOCK_NUM_DYN')
COMPRESSED_ROWS_DYN = pl.dynamic('COMPRESSED_ROWS_DYN')
COMPRESSED_TABLE_COLUMNS_DYN = pl.dynamic('COMPRESSED_TABLE_COLUMNS_DYN')
IDX_NATIVE_CACHE_BLOCK_NUM_DYN = pl.dynamic('IDX_NATIVE_CACHE_BLOCK_NUM_DYN')
INDEXER_PAGE_BYTES_DYN = pl.dynamic('INDEXER_PAGE_BYTES_DYN')
INDEXER_ROWS_DYN = pl.dynamic('INDEXER_ROWS_DYN')
INDEXER_TABLE_COLUMNS_DYN = pl.dynamic('INDEXER_TABLE_COLUMNS_DYN')
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic('INNER_STATE_BLOCK_NUM_DYN')
INNER_STATE_PAGE_ELEMENTS_DYN = pl.dynamic('INNER_STATE_PAGE_ELEMENTS_DYN')
INNER_STATE_TABLE_COLUMNS_DYN = pl.dynamic('INNER_STATE_TABLE_COLUMNS_DYN')
MAIN_STATE_BLOCK_NUM_DYN = pl.dynamic('MAIN_STATE_BLOCK_NUM_DYN')
ORIGINAL_TABLE_COLUMNS_DYN = pl.dynamic('ORIGINAL_TABLE_COLUMNS_DYN')
ORI_BLOCK_NUM_DYN = pl.dynamic('ORI_BLOCK_NUM_DYN')
QUERY_BOUNDS_DYN = pl.dynamic('QUERY_BOUNDS_DYN')
STATE_PAGE_ELEMENTS_DYN = pl.dynamic('STATE_PAGE_ELEMENTS_DYN')
STATE_TABLE_COLUMNS_DYN = pl.dynamic('STATE_TABLE_COLUMNS_DYN')
T_DYN = pl.dynamic('T_DYN')
t_dim_inline4335__ssa_v0 = pl.dynamic('t_dim_inline4335__ssa_v0')
bs_heads_inline2066_inline4048__ssa_v0 = pl.dynamic('bs_heads_inline2066_inline4048__ssa_v0')
cmp_block_num_inline3819__ssa_v0 = pl.dynamic('cmp_block_num_inline3819__ssa_v0')
cmp_block_num_inline4371__ssa_v0 = pl.dynamic('cmp_block_num_inline4371__ssa_v0')
hc_padded_rows__ssa_v0 = pl.dynamic('hc_padded_rows__ssa_v0')
key_rows_inline3222_inline4169__ssa_v0 = pl.dynamic('key_rows_inline3222_inline4169__ssa_v0')
key_rows_inline8656__ssa_v0 = pl.dynamic('key_rows_inline8656__ssa_v0')
key_rows_inline9041__ssa_v0 = pl.dynamic('key_rows_inline9041__ssa_v0')
key_rows_inline9391__ssa_v0 = pl.dynamic('key_rows_inline9391__ssa_v0')
key_rows_inline9688__ssa_v0 = pl.dynamic('key_rows_inline9688__ssa_v0')
ori_block_num_inline4369__ssa_v0 = pl.dynamic('ori_block_num_inline4369__ssa_v0')
qa_tokens_inline48_inline8448__ssa_v0 = pl.dynamic('qa_tokens_inline48_inline8448__ssa_v0')
qa_tokens_inline79_inline8471__ssa_v0 = pl.dynamic('qa_tokens_inline79_inline8471__ssa_v0')
qproj_t_matmul_inline10_inline3586__ssa_v0 = pl.dynamic('qproj_t_matmul_inline10_inline3586__ssa_v0')
qr_t_matmul_inline2_inline3597__ssa_v0 = pl.dynamic('qr_t_matmul_inline2_inline3597__ssa_v0')
shifted_rows_inline3223_inline4178__ssa_v0 = pl.dynamic('shifted_rows_inline3223_inline4178__ssa_v0')
shifted_rows_inline8748__ssa_v0 = pl.dynamic('shifted_rows_inline8748__ssa_v0')
shifted_rows_inline9108__ssa_v0 = pl.dynamic('shifted_rows_inline9108__ssa_v0')
shifted_rows_inline9361__ssa_v0 = pl.dynamic('shifted_rows_inline9361__ssa_v0')
shifted_rows_inline9696__ssa_v0 = pl.dynamic('shifted_rows_inline9696__ssa_v0')
t_dim_inline1450_inline3644__ssa_v0 = pl.dynamic('t_dim_inline1450_inline3644__ssa_v0')
t_dim_inline1474_inline3632__ssa_v0 = pl.dynamic('t_dim_inline1474_inline3632__ssa_v0')
t_dim_inline3567__ssa_v0 = pl.dynamic('t_dim_inline3567__ssa_v0')
t_dim_inline3668__ssa_v0 = pl.dynamic('t_dim_inline3668__ssa_v0')
t_dim_inline440__ssa_v0 = pl.dynamic('t_dim_inline440__ssa_v0')
t_dim_inline4518__ssa_v0 = pl.dynamic('t_dim_inline4518__ssa_v0')
t_dim_inline4627__ssa_v0 = pl.dynamic('t_dim_inline4627__ssa_v0')
t_dim_inline4745__ssa_v0 = pl.dynamic('t_dim_inline4745__ssa_v0')
t_dim_inline4891__ssa_v0 = pl.dynamic('t_dim_inline4891__ssa_v0')
t_dim_inline529__ssa_v0 = pl.dynamic('t_dim_inline529__ssa_v0')
t_dim_inline560__ssa_v0 = pl.dynamic('t_dim_inline560__ssa_v0')
t_dim_inline8493__ssa_v0 = pl.dynamic('t_dim_inline8493__ssa_v0')
t_dim_inline8556__ssa_v0 = pl.dynamic('t_dim_inline8556__ssa_v0')
t_heads_inline4306__ssa_v0 = pl.dynamic('t_heads_inline4306__ssa_v0')
t_linear_inline461__ssa_v0 = pl.dynamic('t_linear_inline461__ssa_v0')
t_matmul_inline3663__ssa_v0 = pl.dynamic('t_matmul_inline3663__ssa_v0')
t_pad_inline523__ssa_v0 = pl.dynamic('t_pad_inline523__ssa_v0')
HC_PAD_ROWS_DYN = pl.dynamic('HC_PAD_ROWS_DYN')
QKV_KV_MM_T_DYN = pl.dynamic('QKV_KV_MM_T_DYN')
QKV_QPROJ_MM_T_DYN = pl.dynamic('QKV_QPROJ_MM_T_DYN')
RMS_NORM_CHUNK_D_DYN = pl.dynamic('RMS_NORM_CHUNK_D_DYN')
_jitdyn_hc_mix_norm_x_mixed_d0 = pl.dynamic('_jitdyn_hc_mix_norm_x_mixed_d0')

@pl.jit.incore
def csa_static_aic(temporary_4: pl.InOut[pl.Tensor[[144, 1], pl.FP32]], temporary_5: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]], view_2: pl.InOut[pl.Tensor[[144, 16384], pl.FP32]], view_3: pl.Tensor[[144, 16384], pl.BF16], temporary_7: pl.InOut[pl.Tensor[[576, 32], pl.FP32]], arg_hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], temporary_8: pl.InOut[pl.Tensor[[144, 32], pl.FP32]], arg_hc_attn_scale: pl.Tensor[[3], pl.FP32], temporary_0: pl.InOut[pl.Tensor[[144, 4], pl.FP32]], temporary_9: pl.InOut[pl.Tensor[[8, 8], pl.FP32]], temporary_6: pl.InOut[pl.Tensor[[144, 8], pl.FP32]], arg_hc_attn_base: pl.Tensor[[24], pl.FP32], view_12: pl.Tensor[[1, 24], pl.FP32], temporary_1: pl.InOut[pl.Tensor[[144, 16], pl.FP32]], temporary_10: pl.InOut[pl.Tensor[[8, 32], pl.FP32]], temporary_11: pl.InOut[pl.Tensor[[144, 4096], pl.BF16]], temporary_2: pl.InOut[pl.Tensor[[144, 4096], pl.BF16]], temporary_12: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]], arg_attn_norm_w: pl.Tensor[[4096], pl.BF16], arg_cmp_seq_lens: pl.Tensor[[24], pl.INT32], arg_cmp_query_start_loc: pl.Tensor[[25], pl.INT32], temporary_14: pl.InOut[pl.Tensor[[24], pl.INT32]], arg_kv_seq_lens: pl.Tensor[[24], pl.INT32], arg_idx_query_start_loc: pl.Tensor[[25], pl.INT32], temporary_15: pl.InOut[pl.Tensor[[24], pl.INT32]], arg_freqs_sin: pl.Tensor[[144, 64], pl.FP32], temporary_13: pl.InOut[pl.Tensor[[144, 64], pl.FP32]], temporary_20: pl.InOut[pl.Tensor[[144, 64], pl.FP32]], temporary_21: pl.InOut[pl.Tensor[[144, 64], pl.INT32]], temporary_24: pl.InOut[pl.Tensor[[144, 1024], pl.FP32]], arg_wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ], view_31: pl.Tensor[[1, 1024], pl.BF16], temporary_23: pl.InOut[pl.Tensor[[512, 1], pl.FP32]], temporary_19: pl.InOut[pl.Tensor[[144, 1], pl.FP32]], temporary_22: pl.InOut[pl.Tensor[[512, 1024], pl.INT8]], temporary_18: pl.InOut[pl.Tensor[[144, 1024], pl.INT8]], temporary_25: pl.InOut[pl.Tensor[[256, 32768], pl.INT32]], arg_wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ], arg_freqs_cos: pl.Tensor[[144, 64], pl.FP32], arg_wq_b_scale: pl.Tensor[[32768], pl.FP32], view_40: pl.InOut[pl.Tensor[[144, 32768], pl.BF16]], temporary_26: pl.InOut[pl.Tensor[[144, 512], pl.FP32]], arg_wkv: pl.Tensor[[4096, 512], pl.BF16], temporary_17: pl.InOut[pl.Tensor[[144, 512], pl.BF16]], arg_gamma_ckv: pl.Tensor[[512], pl.BF16], view_45: pl.InOut[pl.Tensor[[6944, 512], pl.BF16]], arg_ori_slot_mapping: pl.Tensor[[144, 2], pl.INT32], temporary_29: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], temporary_30: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], arg_cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], arg_cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], temporary_28: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], arg_position_ids: pl.Tensor[[144], pl.INT64], arg_cmp_ape: pl.Tensor[[4, 1024], pl.FP32], arg_state_block_table: pl.Tensor[[24, 4100], pl.INT32], arg_compress_state: pl.InOut[pl.Tensor[[217, 8192], pl.FP32]], arg_state_slot_mapping: pl.Tensor[[144, 2], pl.INT32], arg_cmp_freqs_cos: pl.Tensor[[60, 64], pl.FP32], arg_cmp_freqs_sin: pl.Tensor[[60, 64], pl.FP32], temporary_31: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], view_60: pl.Tensor[[1, 512], pl.BF16], view_61: pl.InOut[pl.Tensor[[50720, 512], pl.BF16]], temporary_27: pl.InOut[pl.Tensor[[144, 512], pl.FP32]], arg_cmp_slot_mapping: pl.Tensor[[60, 2], pl.INT32], temporary_34: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], temporary_35: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], arg_inner_wkv: pl.Tensor[[256, 4096], pl.BF16], arg_inner_wgate: pl.Tensor[[256, 4096], pl.BF16], temporary_36: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_inner_ape: pl.Tensor[[4, 256], pl.FP32], arg_inner_state_block_table: pl.Tensor[[24, 4100], pl.INT32], arg_inner_compress_state: pl.InOut[pl.Tensor[[217, 1040], pl.FP32]], arg_inner_state_slot_mapping: pl.Tensor[[144, 2], pl.INT32], temporary_33: pl.InOut[pl.Tensor[[384, 128], pl.BF16]], arg_inner_freqs_cos: pl.Tensor[[60, 64], pl.FP32], arg_inner_freqs_sin: pl.Tensor[[60, 64], pl.FP32], view_76: pl.Tensor[[1, 128], pl.BF16], temporary_38: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_hadamard_idx: pl.Tensor[[128, 128], pl.BF16], temporary_37: pl.InOut[pl.Tensor[[384, 1], pl.FP32]], arg_idx_native_kv_cache: pl.InOut[pl.Tensor[[1585, 4160], pl.INT8]], temporary_32: pl.InOut[pl.Tensor[[144, 128], pl.FP32]], arg_idx_slot_mapping: pl.Tensor[[60, 2], pl.INT32], temporary_42: pl.InOut[pl.Tensor[[384, 8192], pl.INT32]], arg_idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], view_85: pl.InOut[pl.Tensor[[384, 8192], pl.BF16]], arg_idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], temporary_43: pl.InOut[pl.Tensor[[9216, 128], pl.FP32]], temporary_41: pl.InOut[pl.Tensor[[24576, 128], pl.BF16]], temporary_39: pl.InOut[pl.Tensor[[24576, 128], pl.INT8]], temporary_40: pl.InOut[pl.Tensor[[24576, 1], pl.FP32]], temporary_45: pl.InOut[pl.Tensor[[1536, 64], pl.FP32]], arg_weights_proj: pl.Tensor[[4096, 64], pl.BF16], temporary_44: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], view_94: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_48: pl.InOut[pl.Tensor[[1024, 384], pl.FP16]], temporary_50: pl.InOut[pl.Tensor[[256], pl.INT64]], arg_idx_block_table: pl.Tensor[[24, 66], pl.INT32], view_98: pl.InOut[pl.Tensor[[51512, 128], pl.INT8]], view_99: pl.InOut[pl.Tensor[[51512, 128], pl.INT8]], temporary_49: pl.InOut[pl.Tensor[[288, 768], pl.FP32]], temporary_47: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_46: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32]], arg_idx_topk_scores: pl.InOut[pl.Tensor[[144, 512], pl.FP32]], arg_idx_topk: pl.InOut[pl.Tensor[[144, 512], pl.INT32]], temporary_53: pl.InOut[pl.Tensor[[144, 512], pl.INT32]], temporary_52: pl.InOut[pl.Tensor[[144, 640], pl.FP32]], view_107: pl.Tensor[[144, 1], pl.INT64], temporary_54: pl.InOut[pl.Tensor[[144, 16], pl.INT32]], arg_ori_block_table: pl.Tensor[[24, 258], pl.INT32], temporary_55: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_62: pl.InOut[pl.Tensor[[256], pl.INT64]], view_112: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_56: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_57: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], temporary_58: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], temporary_59: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], view_117: pl.Tensor[[64, 1], pl.FP32], arg_cmp_block_table: pl.Tensor[[24, 66], pl.INT32], temporary_60: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_61: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], arg_wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ], temporary_63: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_65: pl.InOut[pl.Tensor[[8, 384], pl.FP32]], temporary_64: pl.InOut[pl.Tensor[[384, 8192], pl.INT8]], arg_wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ], temporary_66: pl.InOut[pl.Tensor[[384, 32768], pl.INT32]], arg_wo_b_scale: pl.Tensor[[4096], pl.FP32], view_129: pl.InOut[pl.Tensor[[144, 16384], pl.BF16]], progress: pl.InOut[pl.Tensor[[4608], pl.INT64]], records: pl.Tensor[[34128], pl.INT64], waits: pl.Tensor[[38978], pl.INT64]) -> tuple[pl.Tensor[[144, 1], pl.FP32], pl.Tensor[[8, 16384], pl.FP32], pl.Tensor[[144, 16384], pl.FP32], pl.Tensor[[576, 32], pl.FP32], pl.Tensor[[144, 32], pl.FP32], pl.Tensor[[144, 4], pl.FP32], pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[144, 8], pl.FP32], pl.Tensor[[144, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32], pl.Tensor[[144, 4096], pl.BF16], pl.Tensor[[144, 4096], pl.BF16], pl.Tensor[[8, 4096], pl.BF16], pl.Tensor[[24], pl.INT32], pl.Tensor[[24], pl.INT32], pl.Tensor[[144, 64], pl.FP32], pl.Tensor[[144, 64], pl.FP32], pl.Tensor[[144, 64], pl.INT32], pl.Tensor[[144, 1024], pl.FP32], pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[144, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[144, 1024], pl.INT8], pl.Tensor[[256, 32768], pl.INT32], pl.Tensor[[144, 32768], pl.BF16], pl.Tensor[[144, 512], pl.FP32], pl.Tensor[[144, 512], pl.BF16], pl.Tensor[[6944, 512], pl.BF16], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[217, 8192], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[50720, 512], pl.BF16], pl.Tensor[[144, 512], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[217, 1040], pl.FP32], pl.Tensor[[384, 128], pl.BF16], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[1585, 4160], pl.INT8], pl.Tensor[[144, 128], pl.FP32], pl.Tensor[[384, 8192], pl.INT32], pl.Tensor[[384, 8192], pl.BF16], pl.Tensor[[9216, 128], pl.FP32], pl.Tensor[[24576, 128], pl.BF16], pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Tensor[[1536, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[1024, 384], pl.FP16], pl.Tensor[[256], pl.INT64], pl.Tensor[[51512, 128], pl.INT8], pl.Tensor[[51512, 128], pl.INT8], pl.Tensor[[288, 768], pl.FP32], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Tensor[[144, 512], pl.FP32], pl.Tensor[[144, 512], pl.INT32], pl.Tensor[[144, 512], pl.INT32], pl.Tensor[[144, 640], pl.FP32], pl.Tensor[[144, 16], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[256], pl.INT64], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8], pl.Tensor[[384, 32768], pl.INT32], pl.Tensor[[144, 16384], pl.BF16], pl.Tensor[[4608], pl.INT64]]:
    pl.func_attr({'mx_tensor_views_blocked': True})
    group_id = pl.tile.get_block_idx()
    core_id = group_id
    own_flag = core_id * 64
    pl.system.cacheinvalid(progress, [1], [own_flag])
    prior_progress = pl.tensor.read(progress, [own_flag])
    generation_base = (prior_progress // 1024 + 1) * 1024
    pl.tensor.write(progress, [own_flag], generation_base)
    pl.system.cacheinvalid(progress, [1], [own_flag])
    pl.system.fence()
    record_0 = 0 + core_id * 6
    mapped_block_0 = pl.cast(pl.tensor.read(records, [record_0]), pl.INDEX)
    block_0 = pl.max(mapped_block_0, 0)
    if mapped_block_0 >= 0:
        subslot_0 = pl.cast(pl.tensor.read(records, [record_0 + 1]), pl.INDEX)
        done_0 = pl.tensor.read(records, [record_0 + 2])
        wait_base_0 = pl.cast(pl.tensor.read(records, [record_0 + 3]), pl.INDEX)
        wait_count_0 = pl.cast(pl.tensor.read(records, [record_0 + 4]), pl.INDEX)
        publish_0 = pl.tensor.read(records, [record_0 + 5])
        for dependency_0 in pl.range(wait_count_0):
            wait_index_0 = (wait_base_0 + dependency_0) * 2
            other_0 = pl.cast(pl.tensor.read(waits, [wait_index_0]), pl.INDEX)
            required_0 = pl.tensor.read(waits, [wait_index_0 + 1])
            target_0 = generation_base + required_0
            flag_0 = other_0 * 64
            pl.system.cacheinvalid(progress, [1], [flag_0])
            initial_0 = pl.tensor.read(progress, [flag_0])
            for (ready_0,) in pl.while_(init_values=(initial_0,)):
                pl.cond(ready_0 < target_0)
                pl.system.cacheinvalid(progress, [1], [flag_0])
                after_0 = pl.yield_(pl.tensor.read(progress, [flag_0]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_0_v4: pl.Scalar[pl.INDEX] = block_0
        for (stage_0_v5, (stage_0_v14,)) in pl.range(stage_0_v4, 36, 24, init_values=(temporary_7,)):
            stage_0_v6: pl.Scalar[pl.INDEX] = stage_0_v5 // 4 * 16
            stage_0_v7: pl.Scalar[pl.INDEX] = stage_0_v5 % 4
            stage_0_v8: pl.Scalar[pl.INDEX] = stage_0_v7 * 4096
            stage_0_v9: pl.Scalar[pl.INDEX] = pl.min(144 - stage_0_v6, 16)
            stage_0_v10: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_0_v15, (stage_0_v24,)) in pl.pipeline(16, stage=2, init_values=(stage_0_v10,)):
                stage_0_v16: pl.Scalar[pl.INDEX] = stage_0_v8 + stage_0_v15 * 256
                stage_0_v17: pl.Tile[[16, 256], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[stage_0_v9, 256])] = pl.tile.load(view_2, [stage_0_v6, stage_0_v16], [16, 256], [stage_0_v9, 256], target_memory=pl.Mem.Mat)
                stage_0_v18: pl.Tile[[32, 256], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[24, 256])] = pl.tile.load(arg_hc_attn_fn, [0, stage_0_v16], [32, 256], [24, 256], target_memory=pl.Mem.Mat)
                stage_0_v19: pl.Tile[[256, 32], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[256, 24], blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_0_v18)
                stage_0_v20: pl.Tile[[16, 256], pl.FP32, pl.Mem.Left, pl.TileView(valid_shape=[stage_0_v9, 256])] = pl.tile.move(stage_0_v17, target_memory=pl.Mem.Left)
                stage_0_v21: pl.Tile[[256, 32], pl.FP32, pl.Mem.Right, pl.TileView(valid_shape=[256, 24])] = pl.tile.move(stage_0_v19, target_memory=pl.Mem.Right)
                stage_0_v22: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_0_v24, stage_0_v20, stage_0_v21, stage_0_v15 == 0)
                stage_0_v23: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_0_v22)
            stage_0_v11: pl.Scalar[pl.INDEX] = stage_0_v7 * 144 + stage_0_v6
            stage_0_v12 = pl.tile.store(stage_0_v23, [stage_0_v11, 0], stage_0_v14)
            stage_0_v13 = pl.yield_(stage_0_v12)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_0 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_0)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_1 = 432 + core_id * 6
    mapped_block_1 = pl.cast(pl.tensor.read(records, [record_1]), pl.INDEX)
    block_1 = pl.max(mapped_block_1, 0)
    if mapped_block_1 >= 0:
        subslot_1 = pl.cast(pl.tensor.read(records, [record_1 + 1]), pl.INDEX)
        done_1 = pl.tensor.read(records, [record_1 + 2])
        wait_base_1 = pl.cast(pl.tensor.read(records, [record_1 + 3]), pl.INDEX)
        wait_count_1 = pl.cast(pl.tensor.read(records, [record_1 + 4]), pl.INDEX)
        publish_1 = pl.tensor.read(records, [record_1 + 5])
        for dependency_1 in pl.range(wait_count_1):
            wait_index_1 = (wait_base_1 + dependency_1) * 2
            other_1 = pl.cast(pl.tensor.read(waits, [wait_index_1]), pl.INDEX)
            required_1 = pl.tensor.read(waits, [wait_index_1 + 1])
            target_1 = generation_base + required_1
            flag_1 = other_1 * 64
            pl.system.cacheinvalid(progress, [1], [flag_1])
            initial_1 = pl.tensor.read(progress, [flag_1])
            for (ready_1,) in pl.while_(init_values=(initial_1,)):
                pl.cond(ready_1 < target_1)
                pl.system.cacheinvalid(progress, [1], [flag_1])
                after_1 = pl.yield_(pl.tensor.read(progress, [flag_1]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_1_v4: pl.Scalar[pl.INDEX] = block_1
        stage_1_v5: pl.Scalar[pl.INDEX] = 0
        stage_1_v6: pl.Scalar[pl.INDEX] = stage_1_v4 % 8 * 128
        stage_1_v7: pl.Scalar[pl.INDEX] = stage_1_v4 // 8
        for (stage_1_v8, (stage_1_v17,)) in pl.range(stage_1_v7 * 64, 128, 2 * 64, init_values=(temporary_24,)):
            stage_1_v9: pl.Scalar[pl.INDEX] = 0 + stage_1_v8
            stage_1_v10 = pl.tensor.create([64, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for (stage_1_v18, (stage_1_v33,)) in pl.pipeline(16, stage=2, init_values=(stage_1_v10,)):
                stage_1_v19: pl.Scalar[pl.INDEX] = stage_1_v5 + stage_1_v18 * 256
                stage_1_v20 = pl.tensor.slice(temporary_2, [64, 256], [stage_1_v9, stage_1_v19])
                stage_1_v21 = pl.tensor.slice(arg_wq_a, [128, 256], [stage_1_v6, stage_1_v19])
                stage_1_v22 = pl.tensor.matmul_acc(stage_1_v33, stage_1_v20, stage_1_v21, init_cond=stage_1_v18 == 0, a_trans=False, b_trans=True)
                stage_1_v23 = pl.yield_(stage_1_v22)
            stage_1_v11 = pl.tensor.assemble(stage_1_v17, stage_1_v23, [stage_1_v8, stage_1_v6])
            stage_1_v12 = pl.yield_(stage_1_v11)
        for (stage_1_v13, (stage_1_v24,)) in pl.range(128 + stage_1_v7 * 16, 144, 2 * 16, init_values=(stage_1_v12,)):
            stage_1_v14 = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for (stage_1_v25, (stage_1_v34,)) in pl.pipeline(16, stage=2, init_values=(stage_1_v14,)):
                stage_1_v26: pl.Scalar[pl.INDEX] = stage_1_v5 + stage_1_v25 * 256
                stage_1_v27: pl.Scalar[pl.INDEX] = pl.min(144 - stage_1_v13, 16)
                stage_1_v28: pl.Scalar[pl.INDEX] = 0 + stage_1_v13
                stage_1_v29 = pl.tensor.slice(temporary_2, [16, 256], [stage_1_v28, stage_1_v26], [stage_1_v27, 256])
                stage_1_v30 = pl.tensor.slice(arg_wq_a, [128, 256], [stage_1_v6, stage_1_v26])
                stage_1_v31 = pl.tensor.matmul_acc(stage_1_v34, stage_1_v29, stage_1_v30, init_cond=stage_1_v25 == 0, a_trans=False, b_trans=True)
                stage_1_v32 = pl.yield_(stage_1_v31)
            stage_1_v15 = pl.tensor.assemble(stage_1_v24, stage_1_v32, [stage_1_v13, stage_1_v6])
            stage_1_v16 = pl.yield_(stage_1_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_1 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_1)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_2 = 864 + core_id * 6
    mapped_block_2 = pl.cast(pl.tensor.read(records, [record_2]), pl.INDEX)
    block_2 = pl.max(mapped_block_2, 0)
    if mapped_block_2 >= 0:
        subslot_2 = pl.cast(pl.tensor.read(records, [record_2 + 1]), pl.INDEX)
        done_2 = pl.tensor.read(records, [record_2 + 2])
        wait_base_2 = pl.cast(pl.tensor.read(records, [record_2 + 3]), pl.INDEX)
        wait_count_2 = pl.cast(pl.tensor.read(records, [record_2 + 4]), pl.INDEX)
        publish_2 = pl.tensor.read(records, [record_2 + 5])
        for dependency_2 in pl.range(wait_count_2):
            wait_index_2 = (wait_base_2 + dependency_2) * 2
            other_2 = pl.cast(pl.tensor.read(waits, [wait_index_2]), pl.INDEX)
            required_2 = pl.tensor.read(waits, [wait_index_2 + 1])
            target_2 = generation_base + required_2
            flag_2 = other_2 * 64
            pl.system.cacheinvalid(progress, [1], [flag_2])
            initial_2 = pl.tensor.read(progress, [flag_2])
            for (ready_2,) in pl.while_(init_values=(initial_2,)):
                pl.cond(ready_2 < target_2)
                pl.system.cacheinvalid(progress, [1], [flag_2])
                after_2 = pl.yield_(pl.tensor.read(progress, [flag_2]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_2_v4: pl.Scalar[pl.INDEX] = block_2
        for (stage_2_v5, (stage_2_v14,)) in pl.range(stage_2_v4, 9 * 4, 8, init_values=(temporary_45,)):
            stage_2_v6: pl.Scalar[pl.INDEX] = stage_2_v5 // 4
            stage_2_v7: pl.Scalar[pl.INDEX] = stage_2_v5 - stage_2_v6 * 4
            stage_2_v8: pl.Scalar[pl.INDEX] = stage_2_v6 * 16
            stage_2_v9: pl.Scalar[pl.INDEX] = pl.min(144 - stage_2_v8, 16)
            stage_2_v10: pl.Scalar[pl.INDEX] = stage_2_v7 * 1024
            stage_2_v11: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_2_v15, (stage_2_v20,)) in pl.range(2, init_values=(stage_2_v11,)):
                stage_2_v16: pl.Scalar[pl.INDEX] = stage_2_v10 + stage_2_v15 * 512
                stage_2_v17: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_2_v9, 512])] = pl.tile.load(temporary_2, [stage_2_v8, stage_2_v16], [16, 512], [stage_2_v9, 512], target_memory=pl.Mem.Mat)
                stage_2_v18: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_weights_proj, [stage_2_v16, 0], [512, 64], [512, 64], target_memory=pl.Mem.Mat)
                for (stage_2_v21, (stage_2_v26,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_2_v20,)):
                    stage_2_v22: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_2_v9, pl.const(16, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_2_v17, 0, stage_2_v21, [16, 256], target_memory=pl.Mem.Left)
                    stage_2_v23: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_2_v18, stage_2_v21, 0, [256, 64], target_memory=pl.Mem.Right)
                    stage_2_v24: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_2_v26, stage_2_v22, stage_2_v23, stage_2_v15 == 0 and stage_2_v21 == 0)
                    stage_2_v25: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_2_v24)
                stage_2_v19: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_2_v25)
            stage_2_v12 = pl.tile.store(stage_2_v19, [stage_2_v7 * 384 + stage_2_v8, 0], stage_2_v14)
            stage_2_v13 = pl.yield_(stage_2_v12)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_2 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_2)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_3 = 1296 + core_id * 6
    mapped_block_3 = pl.cast(pl.tensor.read(records, [record_3]), pl.INDEX)
    block_3 = pl.max(mapped_block_3, 0)
    if mapped_block_3 >= 0:
        subslot_3 = pl.cast(pl.tensor.read(records, [record_3 + 1]), pl.INDEX)
        done_3 = pl.tensor.read(records, [record_3 + 2])
        wait_base_3 = pl.cast(pl.tensor.read(records, [record_3 + 3]), pl.INDEX)
        wait_count_3 = pl.cast(pl.tensor.read(records, [record_3 + 4]), pl.INDEX)
        publish_3 = pl.tensor.read(records, [record_3 + 5])
        for dependency_3 in pl.range(wait_count_3):
            wait_index_3 = (wait_base_3 + dependency_3) * 2
            other_3 = pl.cast(pl.tensor.read(waits, [wait_index_3]), pl.INDEX)
            required_3 = pl.tensor.read(waits, [wait_index_3 + 1])
            target_3 = generation_base + required_3
            flag_3 = other_3 * 64
            pl.system.cacheinvalid(progress, [1], [flag_3])
            initial_3 = pl.tensor.read(progress, [flag_3])
            for (ready_3,) in pl.while_(init_values=(initial_3,)):
                pl.cond(ready_3 < target_3)
                pl.system.cacheinvalid(progress, [1], [flag_3])
                after_3 = pl.yield_(pl.tensor.read(progress, [flag_3]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_3_v6: pl.Scalar[pl.INDEX] = block_3
        for (stage_3_v7, (stage_3_v14, stage_3_v15)) in pl.range(stage_3_v6, 144 // 2, 24, init_values=(temporary_34, temporary_35)):
            stage_3_v8: pl.Scalar[pl.INDEX] = stage_3_v7 // 8 * 16
            stage_3_v9: pl.Scalar[pl.INDEX] = stage_3_v7 % 8 * 32
            stage_3_v10: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            stage_3_v11: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_3_v16, (stage_3_v31, stage_3_v32)) in pl.pipeline(8, stage=2, init_values=(stage_3_v10, stage_3_v11)):
                stage_3_v17: pl.Scalar[pl.INDEX] = stage_3_v16 * 512
                stage_3_v18: pl.Scalar[pl.INDEX] = pl.min(144 - stage_3_v8, 16)
                stage_3_v19: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_3_v18, 512])] = pl.tile.load(temporary_2, [stage_3_v8, stage_3_v17], [16, 512], [stage_3_v18, 512], target_memory=pl.Mem.Mat)
                stage_3_v20: pl.Tile[[32, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_inner_wkv, [stage_3_v9, stage_3_v17], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
                stage_3_v21: pl.Tile[[32, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_inner_wgate, [stage_3_v9, stage_3_v17], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
                stage_3_v22: pl.Tile[[512, 32], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_3_v20)
                stage_3_v23: pl.Tile[[16, 512], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[stage_3_v18, 512])] = pl.tile.move(stage_3_v19, target_memory=pl.Mem.Left)
                stage_3_v24: pl.Tile[[512, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_3_v22, target_memory=pl.Mem.Right)
                stage_3_v25: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_3_v31, stage_3_v23, stage_3_v24, stage_3_v16 == 0)
                stage_3_v26: pl.Tile[[512, 32], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_3_v21)
                stage_3_v27: pl.Tile[[512, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_3_v26, target_memory=pl.Mem.Right)
                stage_3_v28: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_3_v32, stage_3_v23, stage_3_v27, stage_3_v16 == 0)
                (stage_3_v33, stage_3_v34) = pl.yield_(stage_3_v25, stage_3_v28)
            stage_3_v12 = pl.tile.store(stage_3_v33, [stage_3_v8, stage_3_v9], stage_3_v14)
            stage_3_v13 = pl.tile.store(stage_3_v34, [stage_3_v8, stage_3_v9], stage_3_v15)
            (stage_3_v29, stage_3_v30) = pl.yield_(stage_3_v12, stage_3_v13)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_3 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_3)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_4 = 1728 + core_id * 6
    mapped_block_4 = pl.cast(pl.tensor.read(records, [record_4]), pl.INDEX)
    block_4 = pl.max(mapped_block_4, 0)
    if mapped_block_4 >= 0:
        subslot_4 = pl.cast(pl.tensor.read(records, [record_4 + 1]), pl.INDEX)
        done_4 = pl.tensor.read(records, [record_4 + 2])
        wait_base_4 = pl.cast(pl.tensor.read(records, [record_4 + 3]), pl.INDEX)
        wait_count_4 = pl.cast(pl.tensor.read(records, [record_4 + 4]), pl.INDEX)
        publish_4 = pl.tensor.read(records, [record_4 + 5])
        for dependency_4 in pl.range(wait_count_4):
            wait_index_4 = (wait_base_4 + dependency_4) * 2
            other_4 = pl.cast(pl.tensor.read(waits, [wait_index_4]), pl.INDEX)
            required_4 = pl.tensor.read(waits, [wait_index_4 + 1])
            target_4 = generation_base + required_4
            flag_4 = other_4 * 64
            pl.system.cacheinvalid(progress, [1], [flag_4])
            initial_4 = pl.tensor.read(progress, [flag_4])
            for (ready_4,) in pl.while_(init_values=(initial_4,)):
                pl.cond(ready_4 < target_4)
                pl.system.cacheinvalid(progress, [1], [flag_4])
                after_4 = pl.yield_(pl.tensor.read(progress, [flag_4]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_4_v6: pl.Scalar[pl.INDEX] = block_4
        for (stage_4_v7, (stage_4_v14, stage_4_v15)) in pl.range(stage_4_v6, 144 // 2, 24, init_values=(temporary_34, temporary_35)):
            stage_4_v8: pl.Scalar[pl.INDEX] = stage_4_v7 // 8 * 16
            stage_4_v9: pl.Scalar[pl.INDEX] = stage_4_v7 % 8 * 32
            stage_4_v10: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            stage_4_v11: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_4_v16, (stage_4_v31, stage_4_v32)) in pl.pipeline(8, stage=2, init_values=(stage_4_v10, stage_4_v11)):
                stage_4_v17: pl.Scalar[pl.INDEX] = stage_4_v16 * 512
                stage_4_v18: pl.Scalar[pl.INDEX] = pl.min(144 - stage_4_v8, 16)
                stage_4_v19: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_4_v18, 512])] = pl.tile.load(temporary_2, [stage_4_v8, stage_4_v17], [16, 512], [stage_4_v18, 512], target_memory=pl.Mem.Mat)
                stage_4_v20: pl.Tile[[32, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_inner_wkv, [stage_4_v9, stage_4_v17], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
                stage_4_v21: pl.Tile[[32, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_inner_wgate, [stage_4_v9, stage_4_v17], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
                stage_4_v22: pl.Tile[[512, 32], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_4_v20)
                stage_4_v23: pl.Tile[[16, 512], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[stage_4_v18, 512])] = pl.tile.move(stage_4_v19, target_memory=pl.Mem.Left)
                stage_4_v24: pl.Tile[[512, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_4_v22, target_memory=pl.Mem.Right)
                stage_4_v25: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_4_v31, stage_4_v23, stage_4_v24, stage_4_v16 == 0)
                stage_4_v26: pl.Tile[[512, 32], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_4_v21)
                stage_4_v27: pl.Tile[[512, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_4_v26, target_memory=pl.Mem.Right)
                stage_4_v28: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_4_v32, stage_4_v23, stage_4_v27, stage_4_v16 == 0)
                (stage_4_v33, stage_4_v34) = pl.yield_(stage_4_v25, stage_4_v28)
            stage_4_v12 = pl.tile.store(stage_4_v33, [stage_4_v8, stage_4_v9], stage_4_v14)
            stage_4_v13 = pl.tile.store(stage_4_v34, [stage_4_v8, stage_4_v9], stage_4_v15)
            (stage_4_v29, stage_4_v30) = pl.yield_(stage_4_v12, stage_4_v13)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_4 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_4)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_5 = 2160 + core_id * 6
    mapped_block_5 = pl.cast(pl.tensor.read(records, [record_5]), pl.INDEX)
    block_5 = pl.max(mapped_block_5, 0)
    if mapped_block_5 >= 0:
        subslot_5 = pl.cast(pl.tensor.read(records, [record_5 + 1]), pl.INDEX)
        done_5 = pl.tensor.read(records, [record_5 + 2])
        wait_base_5 = pl.cast(pl.tensor.read(records, [record_5 + 3]), pl.INDEX)
        wait_count_5 = pl.cast(pl.tensor.read(records, [record_5 + 4]), pl.INDEX)
        publish_5 = pl.tensor.read(records, [record_5 + 5])
        for dependency_5 in pl.range(wait_count_5):
            wait_index_5 = (wait_base_5 + dependency_5) * 2
            other_5 = pl.cast(pl.tensor.read(waits, [wait_index_5]), pl.INDEX)
            required_5 = pl.tensor.read(waits, [wait_index_5 + 1])
            target_5 = generation_base + required_5
            flag_5 = other_5 * 64
            pl.system.cacheinvalid(progress, [1], [flag_5])
            initial_5 = pl.tensor.read(progress, [flag_5])
            for (ready_5,) in pl.while_(init_values=(initial_5,)):
                pl.cond(ready_5 < target_5)
                pl.system.cacheinvalid(progress, [1], [flag_5])
                after_5 = pl.yield_(pl.tensor.read(progress, [flag_5]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_5_v6: pl.Scalar[pl.INDEX] = block_5
        for (stage_5_v7, (stage_5_v19, stage_5_v20)) in pl.range(stage_5_v6, 192 // 4, 24, init_values=(temporary_29, temporary_30)):
            stage_5_v8: pl.Scalar[pl.INDEX] = stage_5_v7 // 16 * 64
            stage_5_v9: pl.Scalar[pl.INDEX] = stage_5_v7 % 16 * 64
            stage_5_v10: pl.Scalar[pl.INDEX] = pl.min(144 - stage_5_v8, 64)
            stage_5_v11: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_5_v12: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_5_v13: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            stage_5_v14: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_5_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_5_v13, stage_5_v10, 64)
            stage_5_v15: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            stage_5_v16: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_5_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_5_v15, stage_5_v10, 64)
            for (stage_5_v21, (stage_5_v28, stage_5_v29)) in pl.pipeline(8, stage=2, init_values=(stage_5_v14, stage_5_v16)):
                stage_5_v22: pl.Scalar[pl.INDEX] = stage_5_v21 * 512
                stage_5_v23: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_5_v10, 512])] = pl.tile.load(temporary_2, [stage_5_v8, stage_5_v22], [64, 512], [stage_5_v10, 512], target_memory=pl.Mem.Mat)
                stage_5_v24: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_cmp_wkv, [stage_5_v9, stage_5_v22], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
                stage_5_v25: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_cmp_wgate, [stage_5_v9, stage_5_v22], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
                if stage_5_v21 == 0:
                    stage_5_v30: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_5_v24)
                    stage_5_v31: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    stage_5_v32: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_5_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_5_v31, stage_5_v10, 64)
                    for (stage_5_v40, (stage_5_v62,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_5_v32,)):
                        stage_5_v41: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_5_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_5_v23, 0, stage_5_v40, [64, 256], target_memory=pl.Mem.Left)
                        stage_5_v42: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_5_v30, stage_5_v40, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_5_v43: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_5_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_5_v62, stage_5_v41, stage_5_v42, stage_5_v40 == 0)
                        stage_5_v44: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_5_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_5_v43)
                    stage_5_v33: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_5_v25)
                    stage_5_v34: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    stage_5_v35: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_5_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_5_v34, stage_5_v10, 64)
                    for (stage_5_v45, (stage_5_v63,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_5_v35,)):
                        stage_5_v46: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_5_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_5_v23, 0, stage_5_v45, [64, 256], target_memory=pl.Mem.Left)
                        stage_5_v47: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_5_v33, stage_5_v45, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_5_v48: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_5_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_5_v63, stage_5_v46, stage_5_v47, stage_5_v45 == 0)
                        stage_5_v49: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_5_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_5_v48)
                    (stage_5_v50, stage_5_v51) = pl.yield_(stage_5_v44, stage_5_v49)
                else:
                    stage_5_v36: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_5_v24)
                    for (stage_5_v52, (stage_5_v64,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_5_v28,)):
                        stage_5_v53: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_5_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_5_v23, 0, stage_5_v52, [64, 256], target_memory=pl.Mem.Left)
                        stage_5_v54: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_5_v36, stage_5_v52, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_5_v55: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_5_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_5_v64, stage_5_v53, stage_5_v54)
                        stage_5_v56: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_5_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_5_v55)
                    stage_5_v37: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_5_v25)
                    for (stage_5_v57, (stage_5_v65,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_5_v29,)):
                        stage_5_v58: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_5_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_5_v23, 0, stage_5_v57, [64, 256], target_memory=pl.Mem.Left)
                        stage_5_v59: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_5_v37, stage_5_v57, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_5_v60: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_5_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_5_v65, stage_5_v58, stage_5_v59)
                        stage_5_v61: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_5_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_5_v60)
                    (stage_5_v50, stage_5_v51) = pl.yield_(stage_5_v56, stage_5_v61)
                (stage_5_v38, stage_5_v39) = pl.yield_(stage_5_v50, stage_5_v51)
            stage_5_v17 = pl.tile.store(stage_5_v38, [stage_5_v8, stage_5_v9], stage_5_v19)
            stage_5_v18 = pl.tile.store(stage_5_v39, [stage_5_v8, stage_5_v9], stage_5_v20)
            (stage_5_v26, stage_5_v27) = pl.yield_(stage_5_v17, stage_5_v18)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_5 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_5)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_6 = 2592 + core_id * 6
    mapped_block_6 = pl.cast(pl.tensor.read(records, [record_6]), pl.INDEX)
    block_6 = pl.max(mapped_block_6, 0)
    if mapped_block_6 >= 0:
        subslot_6 = pl.cast(pl.tensor.read(records, [record_6 + 1]), pl.INDEX)
        done_6 = pl.tensor.read(records, [record_6 + 2])
        wait_base_6 = pl.cast(pl.tensor.read(records, [record_6 + 3]), pl.INDEX)
        wait_count_6 = pl.cast(pl.tensor.read(records, [record_6 + 4]), pl.INDEX)
        publish_6 = pl.tensor.read(records, [record_6 + 5])
        for dependency_6 in pl.range(wait_count_6):
            wait_index_6 = (wait_base_6 + dependency_6) * 2
            other_6 = pl.cast(pl.tensor.read(waits, [wait_index_6]), pl.INDEX)
            required_6 = pl.tensor.read(waits, [wait_index_6 + 1])
            target_6 = generation_base + required_6
            flag_6 = other_6 * 64
            pl.system.cacheinvalid(progress, [1], [flag_6])
            initial_6 = pl.tensor.read(progress, [flag_6])
            for (ready_6,) in pl.while_(init_values=(initial_6,)):
                pl.cond(ready_6 < target_6)
                pl.system.cacheinvalid(progress, [1], [flag_6])
                after_6 = pl.yield_(pl.tensor.read(progress, [flag_6]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_6_v6: pl.Scalar[pl.INDEX] = block_6
        for (stage_6_v7, (stage_6_v19, stage_6_v20)) in pl.range(stage_6_v6, 192 // 4, 24, init_values=(temporary_29, temporary_30)):
            stage_6_v8: pl.Scalar[pl.INDEX] = stage_6_v7 // 16 * 64
            stage_6_v9: pl.Scalar[pl.INDEX] = stage_6_v7 % 16 * 64
            stage_6_v10: pl.Scalar[pl.INDEX] = pl.min(144 - stage_6_v8, 64)
            stage_6_v11: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_6_v12: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_6_v13: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            stage_6_v14: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_6_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_6_v13, stage_6_v10, 64)
            stage_6_v15: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            stage_6_v16: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_6_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_6_v15, stage_6_v10, 64)
            for (stage_6_v21, (stage_6_v28, stage_6_v29)) in pl.pipeline(8, stage=2, init_values=(stage_6_v14, stage_6_v16)):
                stage_6_v22: pl.Scalar[pl.INDEX] = stage_6_v21 * 512
                stage_6_v23: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_6_v10, 512])] = pl.tile.load(temporary_2, [stage_6_v8, stage_6_v22], [64, 512], [stage_6_v10, 512], target_memory=pl.Mem.Mat)
                stage_6_v24: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_cmp_wkv, [stage_6_v9, stage_6_v22], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
                stage_6_v25: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_cmp_wgate, [stage_6_v9, stage_6_v22], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
                if stage_6_v21 == 0:
                    stage_6_v30: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_6_v24)
                    stage_6_v31: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    stage_6_v32: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_6_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_6_v31, stage_6_v10, 64)
                    for (stage_6_v40, (stage_6_v62,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_6_v32,)):
                        stage_6_v41: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_6_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_6_v23, 0, stage_6_v40, [64, 256], target_memory=pl.Mem.Left)
                        stage_6_v42: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_6_v30, stage_6_v40, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_6_v43: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_6_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_6_v62, stage_6_v41, stage_6_v42, stage_6_v40 == 0)
                        stage_6_v44: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_6_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_6_v43)
                    stage_6_v33: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_6_v25)
                    stage_6_v34: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    stage_6_v35: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_6_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_6_v34, stage_6_v10, 64)
                    for (stage_6_v45, (stage_6_v63,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_6_v35,)):
                        stage_6_v46: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_6_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_6_v23, 0, stage_6_v45, [64, 256], target_memory=pl.Mem.Left)
                        stage_6_v47: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_6_v33, stage_6_v45, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_6_v48: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_6_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_6_v63, stage_6_v46, stage_6_v47, stage_6_v45 == 0)
                        stage_6_v49: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_6_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_6_v48)
                    (stage_6_v50, stage_6_v51) = pl.yield_(stage_6_v44, stage_6_v49)
                else:
                    stage_6_v36: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_6_v24)
                    for (stage_6_v52, (stage_6_v64,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_6_v28,)):
                        stage_6_v53: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_6_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_6_v23, 0, stage_6_v52, [64, 256], target_memory=pl.Mem.Left)
                        stage_6_v54: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_6_v36, stage_6_v52, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_6_v55: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_6_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_6_v64, stage_6_v53, stage_6_v54)
                        stage_6_v56: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_6_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_6_v55)
                    stage_6_v37: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_6_v25)
                    for (stage_6_v57, (stage_6_v65,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_6_v29,)):
                        stage_6_v58: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_6_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_6_v23, 0, stage_6_v57, [64, 256], target_memory=pl.Mem.Left)
                        stage_6_v59: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_6_v37, stage_6_v57, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_6_v60: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_6_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_6_v65, stage_6_v58, stage_6_v59)
                        stage_6_v61: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_6_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_6_v60)
                    (stage_6_v50, stage_6_v51) = pl.yield_(stage_6_v56, stage_6_v61)
                (stage_6_v38, stage_6_v39) = pl.yield_(stage_6_v50, stage_6_v51)
            stage_6_v17 = pl.tile.store(stage_6_v38, [stage_6_v8, stage_6_v9], stage_6_v19)
            stage_6_v18 = pl.tile.store(stage_6_v39, [stage_6_v8, stage_6_v9], stage_6_v20)
            (stage_6_v26, stage_6_v27) = pl.yield_(stage_6_v17, stage_6_v18)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_6 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_6)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_7 = 3024 + core_id * 6
    mapped_block_7 = pl.cast(pl.tensor.read(records, [record_7]), pl.INDEX)
    block_7 = pl.max(mapped_block_7, 0)
    if mapped_block_7 >= 0:
        subslot_7 = pl.cast(pl.tensor.read(records, [record_7 + 1]), pl.INDEX)
        done_7 = pl.tensor.read(records, [record_7 + 2])
        wait_base_7 = pl.cast(pl.tensor.read(records, [record_7 + 3]), pl.INDEX)
        wait_count_7 = pl.cast(pl.tensor.read(records, [record_7 + 4]), pl.INDEX)
        publish_7 = pl.tensor.read(records, [record_7 + 5])
        for dependency_7 in pl.range(wait_count_7):
            wait_index_7 = (wait_base_7 + dependency_7) * 2
            other_7 = pl.cast(pl.tensor.read(waits, [wait_index_7]), pl.INDEX)
            required_7 = pl.tensor.read(waits, [wait_index_7 + 1])
            target_7 = generation_base + required_7
            flag_7 = other_7 * 64
            pl.system.cacheinvalid(progress, [1], [flag_7])
            initial_7 = pl.tensor.read(progress, [flag_7])
            for (ready_7,) in pl.while_(init_values=(initial_7,)):
                pl.cond(ready_7 < target_7)
                pl.system.cacheinvalid(progress, [1], [flag_7])
                after_7 = pl.yield_(pl.tensor.read(progress, [flag_7]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_7_v4: pl.Scalar[pl.INDEX] = block_7
        for (stage_7_v5, (stage_7_v9,)) in pl.range((19 - stage_7_v4 + 128) // 20, init_values=(temporary_25,)):
            stage_7_v6: pl.Scalar[pl.INDEX] = stage_7_v4 + stage_7_v5 * 20
            stage_7_v7: pl.Scalar[pl.INDEX] = stage_7_v6 * 256
            for (stage_7_v10, (stage_7_v17,)) in pl.range(0, 256, 128, init_values=(stage_7_v9,)):
                stage_7_v11: pl.Scalar[pl.INDEX] = pl.min(144 - stage_7_v10, 128)
                stage_7_v12 = pl.tensor.slice(temporary_22, [128, 256], [stage_7_v10, 0], [stage_7_v11, 256])
                stage_7_v13 = pl.tensor.slice(arg_wq_b, [256, 256], [0, stage_7_v7])
                stage_7_v14 = pl.tensor.matmul(stage_7_v12, stage_7_v13, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                for (stage_7_v18, (stage_7_v23,)) in pl.pipeline(256, 1024, 256, stage=2, init_values=(stage_7_v14,)):
                    stage_7_v19 = pl.tensor.slice(temporary_22, [128, 256], [stage_7_v10, stage_7_v18], [stage_7_v11, 256])
                    stage_7_v20 = pl.tensor.slice(arg_wq_b, [256, 256], [stage_7_v18, stage_7_v7])
                    stage_7_v21 = pl.tensor.matmul_acc(stage_7_v23, stage_7_v19, stage_7_v20, a_trans=False, b_trans=False)
                    stage_7_v22 = pl.yield_(stage_7_v21)
                stage_7_v15 = pl.tensor.assemble(stage_7_v17, stage_7_v22, [stage_7_v10, stage_7_v7])
                stage_7_v16 = pl.yield_(stage_7_v15)
            stage_7_v8 = pl.yield_(stage_7_v16)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_7 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_7)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_8 = 3456 + core_id * 6
    mapped_block_8 = pl.cast(pl.tensor.read(records, [record_8]), pl.INDEX)
    block_8 = pl.max(mapped_block_8, 0)
    if mapped_block_8 >= 0:
        subslot_8 = pl.cast(pl.tensor.read(records, [record_8 + 1]), pl.INDEX)
        done_8 = pl.tensor.read(records, [record_8 + 2])
        wait_base_8 = pl.cast(pl.tensor.read(records, [record_8 + 3]), pl.INDEX)
        wait_count_8 = pl.cast(pl.tensor.read(records, [record_8 + 4]), pl.INDEX)
        publish_8 = pl.tensor.read(records, [record_8 + 5])
        for dependency_8 in pl.range(wait_count_8):
            wait_index_8 = (wait_base_8 + dependency_8) * 2
            other_8 = pl.cast(pl.tensor.read(waits, [wait_index_8]), pl.INDEX)
            required_8 = pl.tensor.read(waits, [wait_index_8 + 1])
            target_8 = generation_base + required_8
            flag_8 = other_8 * 64
            pl.system.cacheinvalid(progress, [1], [flag_8])
            initial_8 = pl.tensor.read(progress, [flag_8])
            for (ready_8,) in pl.while_(init_values=(initial_8,)):
                pl.cond(ready_8 < target_8)
                pl.system.cacheinvalid(progress, [1], [flag_8])
                after_8 = pl.yield_(pl.tensor.read(progress, [flag_8]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_8_v4: pl.Scalar[pl.INDEX] = block_8
        for (stage_8_v5, (stage_8_v9,)) in pl.range((19 - stage_8_v4 + 128) // 20, init_values=(temporary_25,)):
            stage_8_v6: pl.Scalar[pl.INDEX] = stage_8_v4 + stage_8_v5 * 20
            stage_8_v7: pl.Scalar[pl.INDEX] = stage_8_v6 * 256
            for (stage_8_v10, (stage_8_v17,)) in pl.range(0, 256, 128, init_values=(stage_8_v9,)):
                stage_8_v11: pl.Scalar[pl.INDEX] = pl.min(144 - stage_8_v10, 128)
                stage_8_v12 = pl.tensor.slice(temporary_22, [128, 256], [stage_8_v10, 0], [stage_8_v11, 256])
                stage_8_v13 = pl.tensor.slice(arg_wq_b, [256, 256], [0, stage_8_v7])
                stage_8_v14 = pl.tensor.matmul(stage_8_v12, stage_8_v13, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                for (stage_8_v18, (stage_8_v23,)) in pl.pipeline(256, 1024, 256, stage=2, init_values=(stage_8_v14,)):
                    stage_8_v19 = pl.tensor.slice(temporary_22, [128, 256], [stage_8_v10, stage_8_v18], [stage_8_v11, 256])
                    stage_8_v20 = pl.tensor.slice(arg_wq_b, [256, 256], [stage_8_v18, stage_8_v7])
                    stage_8_v21 = pl.tensor.matmul_acc(stage_8_v23, stage_8_v19, stage_8_v20, a_trans=False, b_trans=False)
                    stage_8_v22 = pl.yield_(stage_8_v21)
                stage_8_v15 = pl.tensor.assemble(stage_8_v17, stage_8_v22, [stage_8_v10, stage_8_v7])
                stage_8_v16 = pl.yield_(stage_8_v15)
            stage_8_v8 = pl.yield_(stage_8_v16)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_8 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_8)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_9 = 3888 + core_id * 6
    mapped_block_9 = pl.cast(pl.tensor.read(records, [record_9]), pl.INDEX)
    block_9 = pl.max(mapped_block_9, 0)
    if mapped_block_9 >= 0:
        subslot_9 = pl.cast(pl.tensor.read(records, [record_9 + 1]), pl.INDEX)
        done_9 = pl.tensor.read(records, [record_9 + 2])
        wait_base_9 = pl.cast(pl.tensor.read(records, [record_9 + 3]), pl.INDEX)
        wait_count_9 = pl.cast(pl.tensor.read(records, [record_9 + 4]), pl.INDEX)
        publish_9 = pl.tensor.read(records, [record_9 + 5])
        for dependency_9 in pl.range(wait_count_9):
            wait_index_9 = (wait_base_9 + dependency_9) * 2
            other_9 = pl.cast(pl.tensor.read(waits, [wait_index_9]), pl.INDEX)
            required_9 = pl.tensor.read(waits, [wait_index_9 + 1])
            target_9 = generation_base + required_9
            flag_9 = other_9 * 64
            pl.system.cacheinvalid(progress, [1], [flag_9])
            initial_9 = pl.tensor.read(progress, [flag_9])
            for (ready_9,) in pl.while_(init_values=(initial_9,)):
                pl.cond(ready_9 < target_9)
                pl.system.cacheinvalid(progress, [1], [flag_9])
                after_9 = pl.yield_(pl.tensor.read(progress, [flag_9]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_9_v4: pl.Scalar[pl.INDEX] = block_9
        stage_9_v5: pl.Scalar[pl.INDEX] = stage_9_v4 // 2 * 128
        stage_9_v6: pl.Scalar[pl.INDEX] = 0
        stage_9_v7: pl.Scalar[pl.INDEX] = stage_9_v4 % 2
        for (stage_9_v8, (stage_9_v17,)) in pl.range(stage_9_v7 * 64, 128, 2 * 64, init_values=(temporary_26,)):
            stage_9_v9: pl.Scalar[pl.INDEX] = 0 + stage_9_v8
            stage_9_v10: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_9_v18, (stage_9_v31,)) in pl.pipeline(8, stage=2, init_values=(stage_9_v10,)):
                stage_9_v19: pl.Scalar[pl.INDEX] = stage_9_v6 + stage_9_v18 * 512
                stage_9_v20: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(temporary_2, [stage_9_v9, stage_9_v19], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
                stage_9_v21: pl.Tile[[512, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_wkv, [stage_9_v19, stage_9_v5], [512, 128], [512, 128], target_memory=pl.Mem.Mat, cache=1)
                for (stage_9_v32, (stage_9_v43,)) in pl.pipeline(0, 512, 128, stage=2, init_values=(stage_9_v31,)):
                    stage_9_v33: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(stage_9_v20, 0, stage_9_v32, [64, 128], target_memory=pl.Mem.Left)
                    stage_9_v34: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_9_v21, stage_9_v32, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_9_v35: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_9_v43, stage_9_v33, stage_9_v34, stage_9_v18 == 0 and stage_9_v32 == 0)
                    stage_9_v36: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_9_v35)
                stage_9_v22: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_9_v36)
            stage_9_v11 = pl.tile.store(stage_9_v22, [stage_9_v8, stage_9_v5], stage_9_v17)
            stage_9_v12 = pl.yield_(stage_9_v11)
        for (stage_9_v13, (stage_9_v23,)) in pl.range(128 + stage_9_v7 * 16, 144, 2 * 16, init_values=(stage_9_v12,)):
            stage_9_v14: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_9_v24, (stage_9_v37,)) in pl.pipeline(8, stage=2, init_values=(stage_9_v14,)):
                stage_9_v25: pl.Scalar[pl.INDEX] = stage_9_v6 + stage_9_v24 * 512
                stage_9_v26: pl.Scalar[pl.INDEX] = pl.min(144 - stage_9_v13, 16)
                stage_9_v27: pl.Scalar[pl.INDEX] = 0 + stage_9_v13
                stage_9_v28: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_9_v26, 512])] = pl.tile.load(temporary_2, [stage_9_v27, stage_9_v25], [16, 512], [stage_9_v26, 512], target_memory=pl.Mem.Mat)
                stage_9_v29: pl.Tile[[512, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_wkv, [stage_9_v25, stage_9_v5], [512, 128], [512, 128], target_memory=pl.Mem.Mat, cache=1)
                for (stage_9_v38, (stage_9_v44,)) in pl.pipeline(0, 512, 128, stage=2, init_values=(stage_9_v37,)):
                    stage_9_v39: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_9_v26, pl.const(16, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_9_v28, 0, stage_9_v38, [16, 128], target_memory=pl.Mem.Left)
                    stage_9_v40: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_9_v29, stage_9_v38, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_9_v41: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_9_v44, stage_9_v39, stage_9_v40, stage_9_v24 == 0 and stage_9_v38 == 0)
                    stage_9_v42: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_9_v41)
                stage_9_v30: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_9_v42)
            stage_9_v15 = pl.tile.store(stage_9_v30, [stage_9_v13, stage_9_v5], stage_9_v23)
            stage_9_v16 = pl.yield_(stage_9_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_9 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_9)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_10 = 4320 + core_id * 6
    mapped_block_10 = pl.cast(pl.tensor.read(records, [record_10]), pl.INDEX)
    block_10 = pl.max(mapped_block_10, 0)
    if mapped_block_10 >= 0:
        subslot_10 = pl.cast(pl.tensor.read(records, [record_10 + 1]), pl.INDEX)
        done_10 = pl.tensor.read(records, [record_10 + 2])
        wait_base_10 = pl.cast(pl.tensor.read(records, [record_10 + 3]), pl.INDEX)
        wait_count_10 = pl.cast(pl.tensor.read(records, [record_10 + 4]), pl.INDEX)
        publish_10 = pl.tensor.read(records, [record_10 + 5])
        for dependency_10 in pl.range(wait_count_10):
            wait_index_10 = (wait_base_10 + dependency_10) * 2
            other_10 = pl.cast(pl.tensor.read(waits, [wait_index_10]), pl.INDEX)
            required_10 = pl.tensor.read(waits, [wait_index_10 + 1])
            target_10 = generation_base + required_10
            flag_10 = other_10 * 64
            pl.system.cacheinvalid(progress, [1], [flag_10])
            initial_10 = pl.tensor.read(progress, [flag_10])
            for (ready_10,) in pl.while_(init_values=(initial_10,)):
                pl.cond(ready_10 < target_10)
                pl.system.cacheinvalid(progress, [1], [flag_10])
                after_10 = pl.yield_(pl.tensor.read(progress, [flag_10]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_10_v4: pl.Scalar[pl.INDEX] = block_10
        for (stage_10_v5, (stage_10_v9,)) in pl.range(stage_10_v4, 32, 24, init_values=(temporary_42,)):
            stage_10_v6: pl.Scalar[pl.INDEX] = stage_10_v5 * 256
            stage_10_v7: pl.Tile[[1024, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_idx_wq_b, [0, stage_10_v6], [1024, 256], [1024, 256], target_memory=pl.Mem.Mat)
            for (stage_10_v10, (stage_10_v18,)) in pl.range(3, init_values=(stage_10_v9,)):
                stage_10_v11: pl.Scalar[pl.INDEX] = stage_10_v10 * 64
                stage_10_v12: pl.Scalar[pl.INDEX] = pl.min(144 - stage_10_v11, 64)
                stage_10_v13: pl.Tile[[64, 1024], pl.INT8, pl.Mem.Mat, pl.TileView(valid_shape=[stage_10_v12, 1024])] = pl.tile.load(temporary_18, [stage_10_v11, 0], [64, 1024], [stage_10_v12, 1024], target_memory=pl.Mem.Mat)
                stage_10_v14: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc, compact=True)
                stage_10_v15: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v12, 256], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_10_v14, stage_10_v12, 256)
                for (stage_10_v19, (stage_10_v24,)) in pl.pipeline(0, 1024, 128, stage=2, init_values=(stage_10_v15,)):
                    stage_10_v20: pl.Tile[[64, 128], pl.INT8, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_10_v12, pl.const(64, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_10_v13, 0, stage_10_v19, [64, 128], target_memory=pl.Mem.Left)
                    stage_10_v21: pl.Tile[[128, 256], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_10_v7, stage_10_v19, 0, [128, 256], target_memory=pl.Mem.Right)
                    stage_10_v22: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v12, 256], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_10_v24, stage_10_v20, stage_10_v21, stage_10_v19 == 0)
                    stage_10_v23: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v12, 256], compact=pl.CompactMode.normal)] = pl.yield_(stage_10_v22)
                stage_10_v16 = pl.tile.store(stage_10_v23, [stage_10_v11, stage_10_v6], stage_10_v18)
                stage_10_v17 = pl.yield_(stage_10_v16)
            stage_10_v8 = pl.yield_(stage_10_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_10 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_10)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_11 = 4752 + core_id * 6
    mapped_block_11 = pl.cast(pl.tensor.read(records, [record_11]), pl.INDEX)
    block_11 = pl.max(mapped_block_11, 0)
    if mapped_block_11 >= 0:
        subslot_11 = pl.cast(pl.tensor.read(records, [record_11 + 1]), pl.INDEX)
        done_11 = pl.tensor.read(records, [record_11 + 2])
        wait_base_11 = pl.cast(pl.tensor.read(records, [record_11 + 3]), pl.INDEX)
        wait_count_11 = pl.cast(pl.tensor.read(records, [record_11 + 4]), pl.INDEX)
        publish_11 = pl.tensor.read(records, [record_11 + 5])
        for dependency_11 in pl.range(wait_count_11):
            wait_index_11 = (wait_base_11 + dependency_11) * 2
            other_11 = pl.cast(pl.tensor.read(waits, [wait_index_11]), pl.INDEX)
            required_11 = pl.tensor.read(waits, [wait_index_11 + 1])
            target_11 = generation_base + required_11
            flag_11 = other_11 * 64
            pl.system.cacheinvalid(progress, [1], [flag_11])
            initial_11 = pl.tensor.read(progress, [flag_11])
            for (ready_11,) in pl.while_(init_values=(initial_11,)):
                pl.cond(ready_11 < target_11)
                pl.system.cacheinvalid(progress, [1], [flag_11])
                after_11 = pl.yield_(pl.tensor.read(progress, [flag_11]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_11_v4: pl.Scalar[pl.INDEX] = block_11
        for (stage_11_v5, (stage_11_v9,)) in pl.range(stage_11_v4, 32, 24, init_values=(temporary_42,)):
            stage_11_v6: pl.Scalar[pl.INDEX] = stage_11_v5 * 256
            stage_11_v7: pl.Tile[[1024, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_idx_wq_b, [0, stage_11_v6], [1024, 256], [1024, 256], target_memory=pl.Mem.Mat)
            for (stage_11_v10, (stage_11_v18,)) in pl.range(3, init_values=(stage_11_v9,)):
                stage_11_v11: pl.Scalar[pl.INDEX] = stage_11_v10 * 64
                stage_11_v12: pl.Scalar[pl.INDEX] = pl.min(144 - stage_11_v11, 64)
                stage_11_v13: pl.Tile[[64, 1024], pl.INT8, pl.Mem.Mat, pl.TileView(valid_shape=[stage_11_v12, 1024])] = pl.tile.load(temporary_18, [stage_11_v11, 0], [64, 1024], [stage_11_v12, 1024], target_memory=pl.Mem.Mat)
                stage_11_v14: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc, compact=True)
                stage_11_v15: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_11_v12, 256], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_11_v14, stage_11_v12, 256)
                for (stage_11_v19, (stage_11_v24,)) in pl.pipeline(0, 1024, 128, stage=2, init_values=(stage_11_v15,)):
                    stage_11_v20: pl.Tile[[64, 128], pl.INT8, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_11_v12, pl.const(64, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_11_v13, 0, stage_11_v19, [64, 128], target_memory=pl.Mem.Left)
                    stage_11_v21: pl.Tile[[128, 256], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_11_v7, stage_11_v19, 0, [128, 256], target_memory=pl.Mem.Right)
                    stage_11_v22: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_11_v12, 256], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_11_v24, stage_11_v20, stage_11_v21, stage_11_v19 == 0)
                    stage_11_v23: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_11_v12, 256], compact=pl.CompactMode.normal)] = pl.yield_(stage_11_v22)
                stage_11_v16 = pl.tile.store(stage_11_v23, [stage_11_v11, stage_11_v6], stage_11_v18)
                stage_11_v17 = pl.yield_(stage_11_v16)
            stage_11_v8 = pl.yield_(stage_11_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_11 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_11)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_12 = 5184 + core_id * 6
    mapped_block_12 = pl.cast(pl.tensor.read(records, [record_12]), pl.INDEX)
    block_12 = pl.max(mapped_block_12, 0)
    if mapped_block_12 >= 0:
        subslot_12 = pl.cast(pl.tensor.read(records, [record_12 + 1]), pl.INDEX)
        done_12 = pl.tensor.read(records, [record_12 + 2])
        wait_base_12 = pl.cast(pl.tensor.read(records, [record_12 + 3]), pl.INDEX)
        wait_count_12 = pl.cast(pl.tensor.read(records, [record_12 + 4]), pl.INDEX)
        publish_12 = pl.tensor.read(records, [record_12 + 5])
        for dependency_12 in pl.range(wait_count_12):
            wait_index_12 = (wait_base_12 + dependency_12) * 2
            other_12 = pl.cast(pl.tensor.read(waits, [wait_index_12]), pl.INDEX)
            required_12 = pl.tensor.read(waits, [wait_index_12 + 1])
            target_12 = generation_base + required_12
            flag_12 = other_12 * 64
            pl.system.cacheinvalid(progress, [1], [flag_12])
            initial_12 = pl.tensor.read(progress, [flag_12])
            for (ready_12,) in pl.while_(init_values=(initial_12,)):
                pl.cond(ready_12 < target_12)
                pl.system.cacheinvalid(progress, [1], [flag_12])
                after_12 = pl.yield_(pl.tensor.read(progress, [flag_12]))
        pl.system.fence()
        pl.system.cacheinvalid()
        for (stage_12_v3, (stage_12_v6,)) in pl.range(0, 128, 64, init_values=(temporary_38,)):
            stage_12_v4: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_hadamard_idx, [0, stage_12_v3], [128, 64], [128, 64], target_memory=pl.Mem.Mat)
            for (stage_12_v7, (stage_12_v16,)) in pl.range(3, init_values=(stage_12_v6,)):
                stage_12_v8: pl.Scalar[pl.INDEX] = stage_12_v7 * 16
                stage_12_v9: pl.Scalar[pl.INDEX] = pl.min(48 - stage_12_v8, 16)
                stage_12_v10: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_12_v9, 128])] = pl.tile.load(temporary_33, [stage_12_v8, 0], [16, 128], [stage_12_v9, 128], target_memory=pl.Mem.Mat)
                stage_12_v11: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[stage_12_v9, 128])] = pl.tile.move(stage_12_v10, target_memory=pl.Mem.Left)
                stage_12_v12: pl.Tile[[128, 64], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_12_v4, target_memory=pl.Mem.Right)
                stage_12_v13: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_12_v9, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul(stage_12_v11, stage_12_v12)
                stage_12_v14 = pl.tile.store(stage_12_v13, [stage_12_v8, stage_12_v3], stage_12_v16)
                stage_12_v15 = pl.yield_(stage_12_v14)
            stage_12_v5 = pl.yield_(stage_12_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_12 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_12)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_13 = 5616 + core_id * 6
    mapped_block_13 = pl.cast(pl.tensor.read(records, [record_13]), pl.INDEX)
    block_13 = pl.max(mapped_block_13, 0)
    if mapped_block_13 >= 0:
        subslot_13 = pl.cast(pl.tensor.read(records, [record_13 + 1]), pl.INDEX)
        done_13 = pl.tensor.read(records, [record_13 + 2])
        wait_base_13 = pl.cast(pl.tensor.read(records, [record_13 + 3]), pl.INDEX)
        wait_count_13 = pl.cast(pl.tensor.read(records, [record_13 + 4]), pl.INDEX)
        publish_13 = pl.tensor.read(records, [record_13 + 5])
        for dependency_13 in pl.range(wait_count_13):
            wait_index_13 = (wait_base_13 + dependency_13) * 2
            other_13 = pl.cast(pl.tensor.read(waits, [wait_index_13]), pl.INDEX)
            required_13 = pl.tensor.read(waits, [wait_index_13 + 1])
            target_13 = generation_base + required_13
            flag_13 = other_13 * 64
            pl.system.cacheinvalid(progress, [1], [flag_13])
            initial_13 = pl.tensor.read(progress, [flag_13])
            for (ready_13,) in pl.while_(init_values=(initial_13,)):
                pl.cond(ready_13 < target_13)
                pl.system.cacheinvalid(progress, [1], [flag_13])
                after_13 = pl.yield_(pl.tensor.read(progress, [flag_13]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_13_v4: pl.Scalar[pl.INDEX] = block_13
        stage_13_v5: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_hadamard_idx, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat)
        for (stage_13_v6, (stage_13_v14,)) in pl.range(stage_13_v4, 9216 // 64, 24, init_values=(temporary_43,)):
            stage_13_v7: pl.Scalar[pl.INDEX] = stage_13_v6 * 64
            stage_13_v8: pl.Tile[[64, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(temporary_41, [stage_13_v7, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat)
            stage_13_v9: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(stage_13_v8, target_memory=pl.Mem.Left)
            stage_13_v10: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_13_v5, target_memory=pl.Mem.Right)
            stage_13_v11: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v9, stage_13_v10)
            stage_13_v12 = pl.tile.store(stage_13_v11, [stage_13_v7, 0], stage_13_v14)
            stage_13_v13 = pl.yield_(stage_13_v12)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_13 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_13)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_14 = 6048 + core_id * 6
    mapped_block_14 = pl.cast(pl.tensor.read(records, [record_14]), pl.INDEX)
    block_14 = pl.max(mapped_block_14, 0)
    if mapped_block_14 >= 0:
        subslot_14 = pl.cast(pl.tensor.read(records, [record_14 + 1]), pl.INDEX)
        done_14 = pl.tensor.read(records, [record_14 + 2])
        wait_base_14 = pl.cast(pl.tensor.read(records, [record_14 + 3]), pl.INDEX)
        wait_count_14 = pl.cast(pl.tensor.read(records, [record_14 + 4]), pl.INDEX)
        publish_14 = pl.tensor.read(records, [record_14 + 5])
        for dependency_14 in pl.range(wait_count_14):
            wait_index_14 = (wait_base_14 + dependency_14) * 2
            other_14 = pl.cast(pl.tensor.read(waits, [wait_index_14]), pl.INDEX)
            required_14 = pl.tensor.read(waits, [wait_index_14 + 1])
            target_14 = generation_base + required_14
            flag_14 = other_14 * 64
            pl.system.cacheinvalid(progress, [1], [flag_14])
            initial_14 = pl.tensor.read(progress, [flag_14])
            for (ready_14,) in pl.while_(init_values=(initial_14,)):
                pl.cond(ready_14 < target_14)
                pl.system.cacheinvalid(progress, [1], [flag_14])
                after_14 = pl.yield_(pl.tensor.read(progress, [flag_14]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_14_v4: pl.Scalar[pl.INDEX] = block_14
        stage_14_v5: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_hadamard_idx, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat)
        for (stage_14_v6, (stage_14_v14,)) in pl.range(stage_14_v4, 9216 // 64, 24, init_values=(temporary_43,)):
            stage_14_v7: pl.Scalar[pl.INDEX] = stage_14_v6 * 64
            stage_14_v8: pl.Tile[[64, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(temporary_41, [stage_14_v7, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat)
            stage_14_v9: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(stage_14_v8, target_memory=pl.Mem.Left)
            stage_14_v10: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_14_v5, target_memory=pl.Mem.Right)
            stage_14_v11: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_14_v9, stage_14_v10)
            stage_14_v12 = pl.tile.store(stage_14_v11, [stage_14_v7, 0], stage_14_v14)
            stage_14_v13 = pl.yield_(stage_14_v12)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_14 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_14)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_15 = 6480 + core_id * 6
    mapped_block_15 = pl.cast(pl.tensor.read(records, [record_15]), pl.INDEX)
    block_15 = pl.max(mapped_block_15, 0)
    if mapped_block_15 >= 0:
        subslot_15 = pl.cast(pl.tensor.read(records, [record_15 + 1]), pl.INDEX)
        done_15 = pl.tensor.read(records, [record_15 + 2])
        wait_base_15 = pl.cast(pl.tensor.read(records, [record_15 + 3]), pl.INDEX)
        wait_count_15 = pl.cast(pl.tensor.read(records, [record_15 + 4]), pl.INDEX)
        publish_15 = pl.tensor.read(records, [record_15 + 5])
        for dependency_15 in pl.range(wait_count_15):
            wait_index_15 = (wait_base_15 + dependency_15) * 2
            other_15 = pl.cast(pl.tensor.read(waits, [wait_index_15]), pl.INDEX)
            required_15 = pl.tensor.read(waits, [wait_index_15 + 1])
            target_15 = generation_base + required_15
            flag_15 = other_15 * 64
            pl.system.cacheinvalid(progress, [1], [flag_15])
            initial_15 = pl.tensor.read(progress, [flag_15])
            for (ready_15,) in pl.while_(init_values=(initial_15,)):
                pl.cond(ready_15 < target_15)
                pl.system.cacheinvalid(progress, [1], [flag_15])
                after_15 = pl.yield_(pl.tensor.read(progress, [flag_15]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_15_v13: pl.Scalar[pl.INDEX] = block_15
        stage_15_v14: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        stage_15_v15: pl.Scalar[pl.INDEX] = 0
        for (stage_15_v19, (stage_15_v30,)) in pl.range(stage_15_v14 // 6, init_values=(stage_15_v15,)):
            stage_15_v20: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_15_v19])
            stage_15_v21: pl.Scalar[pl.INDEX] = pl.max(stage_15_v30, pl.cast(stage_15_v20, pl.INDEX) // 4)
            stage_15_v22: pl.Scalar[pl.INDEX] = pl.yield_(stage_15_v21)
        stage_15_v16: pl.Scalar[pl.INDEX] = pl.min(stage_15_v22, 262144)
        stage_15_v17: pl.Scalar[pl.INDEX] = pl.max((stage_15_v16 + 8191) // 8192, 1)
        pl.system.set_ffts(temporary_50)
        for stage_15_v18 in pl.range(stage_15_v13, stage_15_v14 // 6 * stage_15_v17, 24):
            if stage_15_v14 < 144:
                stage_15_v31: pl.Scalar[pl.INDEX] = stage_15_v18 % (stage_15_v14 // 6) * 6
                stage_15_v32: pl.Scalar[pl.INDEX] = stage_15_v18 // (stage_15_v14 // 6)
                (stage_15_v45, stage_15_v46) = pl.yield_(stage_15_v32, stage_15_v31)
            else:
                stage_15_v33: pl.Scalar[pl.INDEX] = stage_15_v18 // stage_15_v17 * 6
                stage_15_v34: pl.Scalar[pl.INDEX] = stage_15_v18 % stage_15_v17
                (stage_15_v45, stage_15_v46) = pl.yield_(stage_15_v34, stage_15_v33)
            stage_15_v23: pl.Scalar[pl.INDEX] = stage_15_v46 // 6
            stage_15_v24: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_15_v46 + 5])
            stage_15_v25: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_15_v23])
            stage_15_v26: pl.Scalar[pl.INDEX] = pl.cast(stage_15_v25, pl.INDEX) // 4
            stage_15_v27: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_15_v26, (stage_15_v24 + 1) // 4), 262144), 0)
            stage_15_v28: pl.Scalar[pl.INDEX] = stage_15_v45 * 8192
            stage_15_v29: pl.Scalar[pl.INDEX] = 8192
            if stage_15_v28 < stage_15_v27:
                stage_15_v35: pl.Scalar[pl.INDEX] = pl.min(stage_15_v29, stage_15_v27 - stage_15_v28)
                stage_15_v36: pl.Scalar[pl.INDEX] = (stage_15_v35 + 767) // 768
                stage_15_v37: pl.Scalar[pl.INDEX] = pl.min(stage_15_v36 * 384, 4096)
                stage_15_v38: pl.Scalar[pl.INDEX] = (stage_15_v37 + 383) // 384
                stage_15_v39: pl.Tile[[384, 128], pl.INT8, pl.Mem.Mat] = pl.tile.load(temporary_39, [stage_15_v46 * 64, 0], [384, 128], [384, 128], target_memory=pl.Mem.Mat)
                stage_15_v40: pl.Tile[[16, 384], pl.FP16, pl.Mem.Mat] = pl.tile.load(temporary_48, [stage_15_v46 // 6 * 16, 0], [16, 384], [16, 384], target_memory=pl.Mem.Mat)
                stage_15_v41: pl.Tile[[384, 128], pl.INT8, pl.Mem.Left] = pl.tile.move(stage_15_v39, target_memory=pl.Mem.Left)
                stage_15_v42: pl.Tile[[16, 384], pl.FP16, pl.Mem.Left] = pl.tile.move(stage_15_v40, target_memory=pl.Mem.Left)
                for stage_15_v43 in pl.range(stage_15_v38):
                    if 2 <= stage_15_v43:
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    stage_15_v47: pl.Scalar[pl.INDEX] = stage_15_v43 * 384
                    stage_15_v48: pl.Scalar[pl.INDEX] = stage_15_v13 * 12 + stage_15_v43 % 2 * 6
                    stage_15_v49: pl.Scalar[pl.INDEX] = 0
                    stage_15_v50: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_15_v51: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v49
                    stage_15_v52: pl.Scalar[pl.INDEX] = pl.min(stage_15_v51 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v53: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v52])
                    stage_15_v54: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v53, pl.INDEX), 0)
                    stage_15_v55: pl.Scalar[pl.INDEX] = stage_15_v54 * 4160
                    if stage_15_v55 % 128 == 0:
                        stage_15_v336: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v50, view_98, [0, 0], [stage_15_v55 // 128, 0], [32, 128], transpose=False)
                        stage_15_v337: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v336)
                    else:
                        stage_15_v338: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v50, view_99, [0, 0], [stage_15_v55 // 128, 0], [32, 128], transpose=False)
                        stage_15_v337: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v338)
                    stage_15_v56: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v49 + 32
                    stage_15_v57: pl.Scalar[pl.INDEX] = pl.min(stage_15_v56 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v58: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v57])
                    stage_15_v59: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v58, pl.INDEX), 0)
                    stage_15_v60: pl.Scalar[pl.INDEX] = stage_15_v59 * 4160
                    if stage_15_v60 % 128 == 0:
                        stage_15_v339: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v337, view_98, [32, 0], [stage_15_v60 // 128, 0], [32, 128], transpose=False)
                        stage_15_v340: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v339)
                    else:
                        stage_15_v341: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v337, view_99, [32, 0], [stage_15_v60 // 128, 0], [32, 128], transpose=False)
                        stage_15_v340: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v341)
                    stage_15_v61: pl.Tile[[128, 64], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_15_v340)
                    stage_15_v62: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_15_v61, target_memory=pl.Mem.Right)
                    stage_15_v63: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_15_v62
                    stage_15_v64: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_15_v65: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v41, stage_15_v63)
                    stage_15_v66: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_15_v64, stage_15_v65, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_15_v67: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_15_v66
                    stage_15_v68: pl.Scalar[pl.INDEX] = 64
                    stage_15_v69: pl.Scalar[pl.INDEX] = 64
                    stage_15_v70: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_15_v71: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v69
                    stage_15_v72: pl.Scalar[pl.INDEX] = pl.min(stage_15_v71 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v73: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v72])
                    stage_15_v74: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v73, pl.INDEX), 0)
                    stage_15_v75: pl.Scalar[pl.INDEX] = stage_15_v74 * 4160
                    if stage_15_v75 % 128 == 0:
                        stage_15_v342: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v70, view_98, [0, 0], [stage_15_v75 // 128, 0], [32, 128], transpose=False)
                        stage_15_v343: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v342)
                    else:
                        stage_15_v344: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v70, view_99, [0, 0], [stage_15_v75 // 128, 0], [32, 128], transpose=False)
                        stage_15_v343: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v344)
                    stage_15_v76: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v69 + 32
                    stage_15_v77: pl.Scalar[pl.INDEX] = pl.min(stage_15_v76 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v78: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v77])
                    stage_15_v79: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v78, pl.INDEX), 0)
                    stage_15_v80: pl.Scalar[pl.INDEX] = stage_15_v79 * 4160
                    if stage_15_v80 % 128 == 0:
                        stage_15_v345: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v343, view_98, [32, 0], [stage_15_v80 // 128, 0], [32, 128], transpose=False)
                        stage_15_v346: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v345)
                    else:
                        stage_15_v347: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v343, view_99, [32, 0], [stage_15_v80 // 128, 0], [32, 128], transpose=False)
                        stage_15_v346: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v347)
                    stage_15_v81: pl.Tile[[128, 64], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_15_v346)
                    stage_15_v82: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_15_v81, target_memory=pl.Mem.Right)
                    stage_15_v83: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_15_v82
                    stage_15_v84: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_15_v85: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_15_v67, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_15_v86: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v41, stage_15_v83)
                    stage_15_v87: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v42, stage_15_v85)
                    stage_15_v88: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_15_v84, stage_15_v86, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_15_v89: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_15_v87, 6, 64)
                    stage_15_v90 = pl.tile.store(stage_15_v89, [stage_15_v48, stage_15_v68 - 64], temporary_49)
                    stage_15_v91: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_15_v88
                    stage_15_v92: pl.Scalar[pl.INDEX] = 128
                    stage_15_v93: pl.Scalar[pl.INDEX] = 128
                    stage_15_v94: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_15_v95: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v93
                    stage_15_v96: pl.Scalar[pl.INDEX] = pl.min(stage_15_v95 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v97: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v96])
                    stage_15_v98: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v97, pl.INDEX), 0)
                    stage_15_v99: pl.Scalar[pl.INDEX] = stage_15_v98 * 4160
                    if stage_15_v99 % 128 == 0:
                        stage_15_v348: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v94, view_98, [0, 0], [stage_15_v99 // 128, 0], [32, 128], transpose=False)
                        stage_15_v349: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v348)
                    else:
                        stage_15_v350: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v94, view_99, [0, 0], [stage_15_v99 // 128, 0], [32, 128], transpose=False)
                        stage_15_v349: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v350)
                    stage_15_v100: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v93 + 32
                    stage_15_v101: pl.Scalar[pl.INDEX] = pl.min(stage_15_v100 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v102: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v101])
                    stage_15_v103: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v102, pl.INDEX), 0)
                    stage_15_v104: pl.Scalar[pl.INDEX] = stage_15_v103 * 4160
                    if stage_15_v104 % 128 == 0:
                        stage_15_v351: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v349, view_98, [32, 0], [stage_15_v104 // 128, 0], [32, 128], transpose=False)
                        stage_15_v352: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v351)
                    else:
                        stage_15_v353: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v349, view_99, [32, 0], [stage_15_v104 // 128, 0], [32, 128], transpose=False)
                        stage_15_v352: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v353)
                    stage_15_v105: pl.Tile[[128, 64], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_15_v352)
                    stage_15_v106: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_15_v105, target_memory=pl.Mem.Right)
                    stage_15_v107: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_15_v106
                    stage_15_v108: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_15_v109: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_15_v91, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_15_v110: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v41, stage_15_v107)
                    stage_15_v111: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v42, stage_15_v109)
                    stage_15_v112: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_15_v108, stage_15_v110, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_15_v113: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_15_v111, 6, 64)
                    stage_15_v114 = pl.tile.store(stage_15_v113, [stage_15_v48, stage_15_v92 - 64], temporary_49)
                    stage_15_v115: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_15_v112
                    stage_15_v116: pl.Scalar[pl.INDEX] = 192
                    stage_15_v117: pl.Scalar[pl.INDEX] = 192
                    stage_15_v118: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_15_v119: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v117
                    stage_15_v120: pl.Scalar[pl.INDEX] = pl.min(stage_15_v119 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v121: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v120])
                    stage_15_v122: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v121, pl.INDEX), 0)
                    stage_15_v123: pl.Scalar[pl.INDEX] = stage_15_v122 * 4160
                    if stage_15_v123 % 128 == 0:
                        stage_15_v354: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v118, view_98, [0, 0], [stage_15_v123 // 128, 0], [32, 128], transpose=False)
                        stage_15_v355: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v354)
                    else:
                        stage_15_v356: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v118, view_99, [0, 0], [stage_15_v123 // 128, 0], [32, 128], transpose=False)
                        stage_15_v355: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v356)
                    stage_15_v124: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v117 + 32
                    stage_15_v125: pl.Scalar[pl.INDEX] = pl.min(stage_15_v124 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v126: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v125])
                    stage_15_v127: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v126, pl.INDEX), 0)
                    stage_15_v128: pl.Scalar[pl.INDEX] = stage_15_v127 * 4160
                    if stage_15_v128 % 128 == 0:
                        stage_15_v357: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v355, view_98, [32, 0], [stage_15_v128 // 128, 0], [32, 128], transpose=False)
                        stage_15_v358: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v357)
                    else:
                        stage_15_v359: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v355, view_99, [32, 0], [stage_15_v128 // 128, 0], [32, 128], transpose=False)
                        stage_15_v358: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v359)
                    stage_15_v129: pl.Tile[[128, 64], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_15_v358)
                    stage_15_v130: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_15_v129, target_memory=pl.Mem.Right)
                    stage_15_v131: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_15_v130
                    stage_15_v132: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_15_v133: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_15_v115, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_15_v134: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v41, stage_15_v131)
                    stage_15_v135: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v42, stage_15_v133)
                    stage_15_v136: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_15_v132, stage_15_v134, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_15_v137: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_15_v135, 6, 64)
                    stage_15_v138 = pl.tile.store(stage_15_v137, [stage_15_v48, stage_15_v116 - 64], temporary_49)
                    stage_15_v139: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_15_v136
                    stage_15_v140: pl.Scalar[pl.INDEX] = 256
                    stage_15_v141: pl.Scalar[pl.INDEX] = 256
                    stage_15_v142: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_15_v143: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v141
                    stage_15_v144: pl.Scalar[pl.INDEX] = pl.min(stage_15_v143 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v145: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v144])
                    stage_15_v146: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v145, pl.INDEX), 0)
                    stage_15_v147: pl.Scalar[pl.INDEX] = stage_15_v146 * 4160
                    if stage_15_v147 % 128 == 0:
                        stage_15_v360: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v142, view_98, [0, 0], [stage_15_v147 // 128, 0], [32, 128], transpose=False)
                        stage_15_v361: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v360)
                    else:
                        stage_15_v362: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v142, view_99, [0, 0], [stage_15_v147 // 128, 0], [32, 128], transpose=False)
                        stage_15_v361: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v362)
                    stage_15_v148: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v141 + 32
                    stage_15_v149: pl.Scalar[pl.INDEX] = pl.min(stage_15_v148 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v150: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v149])
                    stage_15_v151: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v150, pl.INDEX), 0)
                    stage_15_v152: pl.Scalar[pl.INDEX] = stage_15_v151 * 4160
                    if stage_15_v152 % 128 == 0:
                        stage_15_v363: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v361, view_98, [32, 0], [stage_15_v152 // 128, 0], [32, 128], transpose=False)
                        stage_15_v364: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v363)
                    else:
                        stage_15_v365: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v361, view_99, [32, 0], [stage_15_v152 // 128, 0], [32, 128], transpose=False)
                        stage_15_v364: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v365)
                    stage_15_v153: pl.Tile[[128, 64], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_15_v364)
                    stage_15_v154: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_15_v153, target_memory=pl.Mem.Right)
                    stage_15_v155: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_15_v154
                    stage_15_v156: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_15_v157: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_15_v139, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_15_v158: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v41, stage_15_v155)
                    stage_15_v159: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v42, stage_15_v157)
                    stage_15_v160: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_15_v156, stage_15_v158, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_15_v161: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_15_v159, 6, 64)
                    stage_15_v162 = pl.tile.store(stage_15_v161, [stage_15_v48, stage_15_v140 - 64], temporary_49)
                    stage_15_v163: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_15_v160
                    stage_15_v164: pl.Scalar[pl.INDEX] = 320
                    stage_15_v165: pl.Scalar[pl.INDEX] = 320
                    stage_15_v166: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_15_v167: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v165
                    stage_15_v168: pl.Scalar[pl.INDEX] = pl.min(stage_15_v167 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v169: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v168])
                    stage_15_v170: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v169, pl.INDEX), 0)
                    stage_15_v171: pl.Scalar[pl.INDEX] = stage_15_v170 * 4160
                    if stage_15_v171 % 128 == 0:
                        stage_15_v366: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v166, view_98, [0, 0], [stage_15_v171 // 128, 0], [32, 128], transpose=False)
                        stage_15_v367: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v366)
                    else:
                        stage_15_v368: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v166, view_99, [0, 0], [stage_15_v171 // 128, 0], [32, 128], transpose=False)
                        stage_15_v367: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v368)
                    stage_15_v172: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v165 + 32
                    stage_15_v173: pl.Scalar[pl.INDEX] = pl.min(stage_15_v172 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v174: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v173])
                    stage_15_v175: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v174, pl.INDEX), 0)
                    stage_15_v176: pl.Scalar[pl.INDEX] = stage_15_v175 * 4160
                    if stage_15_v176 % 128 == 0:
                        stage_15_v369: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v367, view_98, [32, 0], [stage_15_v176 // 128, 0], [32, 128], transpose=False)
                        stage_15_v370: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v369)
                    else:
                        stage_15_v371: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v367, view_99, [32, 0], [stage_15_v176 // 128, 0], [32, 128], transpose=False)
                        stage_15_v370: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v371)
                    stage_15_v177: pl.Tile[[128, 64], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_15_v370)
                    stage_15_v178: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_15_v177, target_memory=pl.Mem.Right)
                    stage_15_v179: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_15_v178
                    stage_15_v180: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_15_v181: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_15_v163, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_15_v182: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v41, stage_15_v179)
                    stage_15_v183: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v42, stage_15_v181)
                    stage_15_v184: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_15_v180, stage_15_v182, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_15_v185: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_15_v183, 6, 64)
                    stage_15_v186 = pl.tile.store(stage_15_v185, [stage_15_v48, stage_15_v164 - 64], temporary_49)
                    stage_15_v187: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_15_v184
                    stage_15_v188: pl.Scalar[pl.INDEX] = 384
                    stage_15_v189: pl.Scalar[pl.INDEX] = 384
                    stage_15_v190: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_15_v191: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v189 // 384 * stage_15_v37 + stage_15_v189 % 384
                    stage_15_v192: pl.Scalar[pl.INDEX] = pl.min(stage_15_v191 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v193: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v192])
                    stage_15_v194: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v193, pl.INDEX), 0)
                    stage_15_v195: pl.Scalar[pl.INDEX] = stage_15_v194 * 4160
                    if stage_15_v195 % 128 == 0:
                        stage_15_v372: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v190, view_98, [0, 0], [stage_15_v195 // 128, 0], [32, 128], transpose=False)
                        stage_15_v373: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v372)
                    else:
                        stage_15_v374: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v190, view_99, [0, 0], [stage_15_v195 // 128, 0], [32, 128], transpose=False)
                        stage_15_v373: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v374)
                    stage_15_v196: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v189 // 384 * stage_15_v37 + stage_15_v189 % 384 + 32
                    stage_15_v197: pl.Scalar[pl.INDEX] = pl.min(stage_15_v196 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v198: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v197])
                    stage_15_v199: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v198, pl.INDEX), 0)
                    stage_15_v200: pl.Scalar[pl.INDEX] = stage_15_v199 * 4160
                    if stage_15_v200 % 128 == 0:
                        stage_15_v375: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v373, view_98, [32, 0], [stage_15_v200 // 128, 0], [32, 128], transpose=False)
                        stage_15_v376: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v375)
                    else:
                        stage_15_v377: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v373, view_99, [32, 0], [stage_15_v200 // 128, 0], [32, 128], transpose=False)
                        stage_15_v376: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v377)
                    stage_15_v201: pl.Tile[[128, 64], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_15_v376)
                    stage_15_v202: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_15_v201, target_memory=pl.Mem.Right)
                    stage_15_v203: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_15_v202
                    stage_15_v204: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_15_v205: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_15_v187, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_15_v206: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v41, stage_15_v203)
                    stage_15_v207: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v42, stage_15_v205)
                    stage_15_v208: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_15_v204, stage_15_v206, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_15_v209: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_15_v207, 6, 64)
                    stage_15_v210 = pl.tile.store(stage_15_v209, [stage_15_v48, stage_15_v188 - 64], temporary_49)
                    stage_15_v211: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_15_v208
                    stage_15_v212: pl.Scalar[pl.INDEX] = 448
                    stage_15_v213: pl.Scalar[pl.INDEX] = 448
                    stage_15_v214: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_15_v215: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v213 // 384 * stage_15_v37 + stage_15_v213 % 384
                    stage_15_v216: pl.Scalar[pl.INDEX] = pl.min(stage_15_v215 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v217: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v216])
                    stage_15_v218: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v217, pl.INDEX), 0)
                    stage_15_v219: pl.Scalar[pl.INDEX] = stage_15_v218 * 4160
                    if stage_15_v219 % 128 == 0:
                        stage_15_v378: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v214, view_98, [0, 0], [stage_15_v219 // 128, 0], [32, 128], transpose=False)
                        stage_15_v379: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v378)
                    else:
                        stage_15_v380: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v214, view_99, [0, 0], [stage_15_v219 // 128, 0], [32, 128], transpose=False)
                        stage_15_v379: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v380)
                    stage_15_v220: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v213 // 384 * stage_15_v37 + stage_15_v213 % 384 + 32
                    stage_15_v221: pl.Scalar[pl.INDEX] = pl.min(stage_15_v220 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v222: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v221])
                    stage_15_v223: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v222, pl.INDEX), 0)
                    stage_15_v224: pl.Scalar[pl.INDEX] = stage_15_v223 * 4160
                    if stage_15_v224 % 128 == 0:
                        stage_15_v381: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v379, view_98, [32, 0], [stage_15_v224 // 128, 0], [32, 128], transpose=False)
                        stage_15_v382: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v381)
                    else:
                        stage_15_v383: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v379, view_99, [32, 0], [stage_15_v224 // 128, 0], [32, 128], transpose=False)
                        stage_15_v382: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v383)
                    stage_15_v225: pl.Tile[[128, 64], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_15_v382)
                    stage_15_v226: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_15_v225, target_memory=pl.Mem.Right)
                    stage_15_v227: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_15_v226
                    stage_15_v228: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_15_v229: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_15_v211, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_15_v230: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v41, stage_15_v227)
                    stage_15_v231: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v42, stage_15_v229)
                    stage_15_v232: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_15_v228, stage_15_v230, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_15_v233: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_15_v231, 6, 64)
                    stage_15_v234 = pl.tile.store(stage_15_v233, [stage_15_v48, stage_15_v212 - 64], temporary_49)
                    stage_15_v235: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_15_v232
                    stage_15_v236: pl.Scalar[pl.INDEX] = 512
                    stage_15_v237: pl.Scalar[pl.INDEX] = 512
                    stage_15_v238: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_15_v239: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v237 // 384 * stage_15_v37 + stage_15_v237 % 384
                    stage_15_v240: pl.Scalar[pl.INDEX] = pl.min(stage_15_v239 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v241: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v240])
                    stage_15_v242: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v241, pl.INDEX), 0)
                    stage_15_v243: pl.Scalar[pl.INDEX] = stage_15_v242 * 4160
                    if stage_15_v243 % 128 == 0:
                        stage_15_v384: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v238, view_98, [0, 0], [stage_15_v243 // 128, 0], [32, 128], transpose=False)
                        stage_15_v385: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v384)
                    else:
                        stage_15_v386: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v238, view_99, [0, 0], [stage_15_v243 // 128, 0], [32, 128], transpose=False)
                        stage_15_v385: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v386)
                    stage_15_v244: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v237 // 384 * stage_15_v37 + stage_15_v237 % 384 + 32
                    stage_15_v245: pl.Scalar[pl.INDEX] = pl.min(stage_15_v244 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v246: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v245])
                    stage_15_v247: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v246, pl.INDEX), 0)
                    stage_15_v248: pl.Scalar[pl.INDEX] = stage_15_v247 * 4160
                    if stage_15_v248 % 128 == 0:
                        stage_15_v387: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v385, view_98, [32, 0], [stage_15_v248 // 128, 0], [32, 128], transpose=False)
                        stage_15_v388: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v387)
                    else:
                        stage_15_v389: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v385, view_99, [32, 0], [stage_15_v248 // 128, 0], [32, 128], transpose=False)
                        stage_15_v388: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v389)
                    stage_15_v249: pl.Tile[[128, 64], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_15_v388)
                    stage_15_v250: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_15_v249, target_memory=pl.Mem.Right)
                    stage_15_v251: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_15_v250
                    stage_15_v252: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_15_v253: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_15_v235, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_15_v254: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v41, stage_15_v251)
                    stage_15_v255: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v42, stage_15_v253)
                    stage_15_v256: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_15_v252, stage_15_v254, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_15_v257: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_15_v255, 6, 64)
                    stage_15_v258 = pl.tile.store(stage_15_v257, [stage_15_v48, stage_15_v236 - 64], temporary_49)
                    stage_15_v259: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_15_v256
                    stage_15_v260: pl.Scalar[pl.INDEX] = 576
                    stage_15_v261: pl.Scalar[pl.INDEX] = 576
                    stage_15_v262: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_15_v263: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v261 // 384 * stage_15_v37 + stage_15_v261 % 384
                    stage_15_v264: pl.Scalar[pl.INDEX] = pl.min(stage_15_v263 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v265: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v264])
                    stage_15_v266: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v265, pl.INDEX), 0)
                    stage_15_v267: pl.Scalar[pl.INDEX] = stage_15_v266 * 4160
                    if stage_15_v267 % 128 == 0:
                        stage_15_v390: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v262, view_98, [0, 0], [stage_15_v267 // 128, 0], [32, 128], transpose=False)
                        stage_15_v391: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v390)
                    else:
                        stage_15_v392: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v262, view_99, [0, 0], [stage_15_v267 // 128, 0], [32, 128], transpose=False)
                        stage_15_v391: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v392)
                    stage_15_v268: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v261 // 384 * stage_15_v37 + stage_15_v261 % 384 + 32
                    stage_15_v269: pl.Scalar[pl.INDEX] = pl.min(stage_15_v268 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v270: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v269])
                    stage_15_v271: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v270, pl.INDEX), 0)
                    stage_15_v272: pl.Scalar[pl.INDEX] = stage_15_v271 * 4160
                    if stage_15_v272 % 128 == 0:
                        stage_15_v393: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v391, view_98, [32, 0], [stage_15_v272 // 128, 0], [32, 128], transpose=False)
                        stage_15_v394: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v393)
                    else:
                        stage_15_v395: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v391, view_99, [32, 0], [stage_15_v272 // 128, 0], [32, 128], transpose=False)
                        stage_15_v394: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v395)
                    stage_15_v273: pl.Tile[[128, 64], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_15_v394)
                    stage_15_v274: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_15_v273, target_memory=pl.Mem.Right)
                    stage_15_v275: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_15_v274
                    stage_15_v276: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_15_v277: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_15_v259, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_15_v278: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v41, stage_15_v275)
                    stage_15_v279: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v42, stage_15_v277)
                    stage_15_v280: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_15_v276, stage_15_v278, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_15_v281: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_15_v279, 6, 64)
                    stage_15_v282 = pl.tile.store(stage_15_v281, [stage_15_v48, stage_15_v260 - 64], temporary_49)
                    stage_15_v283: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_15_v280
                    stage_15_v284: pl.Scalar[pl.INDEX] = 640
                    stage_15_v285: pl.Scalar[pl.INDEX] = 640
                    stage_15_v286: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_15_v287: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v285 // 384 * stage_15_v37 + stage_15_v285 % 384
                    stage_15_v288: pl.Scalar[pl.INDEX] = pl.min(stage_15_v287 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v289: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v288])
                    stage_15_v290: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v289, pl.INDEX), 0)
                    stage_15_v291: pl.Scalar[pl.INDEX] = stage_15_v290 * 4160
                    if stage_15_v291 % 128 == 0:
                        stage_15_v396: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v286, view_98, [0, 0], [stage_15_v291 // 128, 0], [32, 128], transpose=False)
                        stage_15_v397: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v396)
                    else:
                        stage_15_v398: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v286, view_99, [0, 0], [stage_15_v291 // 128, 0], [32, 128], transpose=False)
                        stage_15_v397: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v398)
                    stage_15_v292: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v285 // 384 * stage_15_v37 + stage_15_v285 % 384 + 32
                    stage_15_v293: pl.Scalar[pl.INDEX] = pl.min(stage_15_v292 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v294: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v293])
                    stage_15_v295: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v294, pl.INDEX), 0)
                    stage_15_v296: pl.Scalar[pl.INDEX] = stage_15_v295 * 4160
                    if stage_15_v296 % 128 == 0:
                        stage_15_v399: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v397, view_98, [32, 0], [stage_15_v296 // 128, 0], [32, 128], transpose=False)
                        stage_15_v400: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v399)
                    else:
                        stage_15_v401: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v397, view_99, [32, 0], [stage_15_v296 // 128, 0], [32, 128], transpose=False)
                        stage_15_v400: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v401)
                    stage_15_v297: pl.Tile[[128, 64], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_15_v400)
                    stage_15_v298: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_15_v297, target_memory=pl.Mem.Right)
                    stage_15_v299: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_15_v298
                    stage_15_v300: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_15_v301: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_15_v283, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_15_v302: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v41, stage_15_v299)
                    stage_15_v303: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v42, stage_15_v301)
                    stage_15_v304: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_15_v300, stage_15_v302, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_15_v305: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_15_v303, 6, 64)
                    stage_15_v306 = pl.tile.store(stage_15_v305, [stage_15_v48, stage_15_v284 - 64], temporary_49)
                    stage_15_v307: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_15_v304
                    stage_15_v308: pl.Scalar[pl.INDEX] = 704
                    stage_15_v309: pl.Scalar[pl.INDEX] = 704
                    stage_15_v310: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_15_v311: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v309 // 384 * stage_15_v37 + stage_15_v309 % 384
                    stage_15_v312: pl.Scalar[pl.INDEX] = pl.min(stage_15_v311 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v313: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v312])
                    stage_15_v314: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v313, pl.INDEX), 0)
                    stage_15_v315: pl.Scalar[pl.INDEX] = stage_15_v314 * 4160
                    if stage_15_v315 % 128 == 0:
                        stage_15_v402: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v310, view_98, [0, 0], [stage_15_v315 // 128, 0], [32, 128], transpose=False)
                        stage_15_v403: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v402)
                    else:
                        stage_15_v404: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v310, view_99, [0, 0], [stage_15_v315 // 128, 0], [32, 128], transpose=False)
                        stage_15_v403: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v404)
                    stage_15_v316: pl.Scalar[pl.INDEX] = stage_15_v28 + stage_15_v47 + stage_15_v309 // 384 * stage_15_v37 + stage_15_v309 % 384 + 32
                    stage_15_v317: pl.Scalar[pl.INDEX] = pl.min(stage_15_v316 // 32, pl.max((stage_15_v26 - 1) // 32, 0))
                    stage_15_v318: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_15_v23, stage_15_v317])
                    stage_15_v319: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_15_v318, pl.INDEX), 0)
                    stage_15_v320: pl.Scalar[pl.INDEX] = stage_15_v319 * 4160
                    if stage_15_v320 % 128 == 0:
                        stage_15_v405: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v403, view_98, [32, 0], [stage_15_v320 // 128, 0], [32, 128], transpose=False)
                        stage_15_v406: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v405)
                    else:
                        stage_15_v407: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_15_v403, view_99, [32, 0], [stage_15_v320 // 128, 0], [32, 128], transpose=False)
                        stage_15_v406: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_15_v407)
                    stage_15_v321: pl.Tile[[128, 64], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_15_v406)
                    stage_15_v322: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_15_v321, target_memory=pl.Mem.Right)
                    stage_15_v323: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_15_v322
                    stage_15_v324: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_15_v325: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_15_v307, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_15_v326: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v41, stage_15_v323)
                    stage_15_v327: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v42, stage_15_v325)
                    stage_15_v328: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_15_v324, stage_15_v326, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_15_v329: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_15_v327, 6, 64)
                    stage_15_v330 = pl.tile.store(stage_15_v329, [stage_15_v48, stage_15_v308 - 64], temporary_49)
                    stage_15_v331: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_15_v328
                    stage_15_v332: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_15_v331, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_15_v333: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v42, stage_15_v332)
                    stage_15_v334: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_15_v333, 6, 64)
                    stage_15_v335 = pl.tile.store(stage_15_v334, [stage_15_v48, 704], temporary_49)
                    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                for stage_15_v44 in pl.range(pl.min(stage_15_v38, 2)):
                    pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_15 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_15)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_16 = 6912 + core_id * 6
    mapped_block_16 = pl.cast(pl.tensor.read(records, [record_16]), pl.INDEX)
    block_16 = pl.max(mapped_block_16, 0)
    if mapped_block_16 >= 0:
        subslot_16 = pl.cast(pl.tensor.read(records, [record_16 + 1]), pl.INDEX)
        done_16 = pl.tensor.read(records, [record_16 + 2])
        wait_base_16 = pl.cast(pl.tensor.read(records, [record_16 + 3]), pl.INDEX)
        wait_count_16 = pl.cast(pl.tensor.read(records, [record_16 + 4]), pl.INDEX)
        publish_16 = pl.tensor.read(records, [record_16 + 5])
        for dependency_16 in pl.range(wait_count_16):
            wait_index_16 = (wait_base_16 + dependency_16) * 2
            other_16 = pl.cast(pl.tensor.read(waits, [wait_index_16]), pl.INDEX)
            required_16 = pl.tensor.read(waits, [wait_index_16 + 1])
            target_16 = generation_base + required_16
            flag_16 = other_16 * 64
            pl.system.cacheinvalid(progress, [1], [flag_16])
            initial_16 = pl.tensor.read(progress, [flag_16])
            for (ready_16,) in pl.while_(init_values=(initial_16,)):
                pl.cond(ready_16 < target_16)
                pl.system.cacheinvalid(progress, [1], [flag_16])
                after_16 = pl.yield_(pl.tensor.read(progress, [flag_16]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_16_v21: pl.Scalar[pl.INDEX] = block_16
        pl.system.set_ffts(temporary_62)
        stage_16_v22: pl.Scalar[pl.INDEX] = (144 - stage_16_v21 + 23) // 24
        stage_16_v23: pl.Scalar[pl.INDEX] = stage_16_v22 * 5
        stage_16_v24: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.create([64, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat)
        stage_16_v25: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.tile.create([384, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat)
        for (stage_16_v26, (stage_16_v27, stage_16_v28)) in pl.range(stage_16_v23 + 2, init_values=(stage_16_v25, stage_16_v24)):
            if stage_16_v26 < stage_16_v23:
                stage_16_v29: pl.Scalar[pl.INDEX] = stage_16_v26 % 5
                stage_16_v30: pl.Scalar[pl.INDEX] = stage_16_v21 + stage_16_v26 // 5 * 24
                if stage_16_v29 == 0:
                    stage_16_v38: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.gather_row(stage_16_v28, view_112, [0, 0], [stage_16_v30 * 64, 0], [64, 512], transpose=False)
                    stage_16_v39: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.yield_(stage_16_v38)
                else:
                    stage_16_v39: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.yield_(stage_16_v28)
                stage_16_v31: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_16_v30, stage_16_v29])
                if 0 < pl.cast(stage_16_v31, pl.INDEX):
                    stage_16_v40: pl.Scalar[pl.INDEX] = stage_16_v21 * 3 + stage_16_v26 % 3
                    stage_16_v41: pl.Scalar[pl.INDEX] = stage_16_v40 * 128
                    stage_16_v42: pl.Scalar[pl.INDEX] = stage_16_v40 * 64
                    pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    stage_16_v43: pl.Scalar[pl.INDEX] = stage_16_v26 % 3 * 128
                    stage_16_v44: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.tile.gather_row(stage_16_v27, temporary_56, [stage_16_v43, 0], [stage_16_v41, 0], [128, 512], transpose=False)
                    stage_16_v45: pl.Tile[[512, 384], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_16_v44)
                    stage_16_v46: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                    for (stage_16_v71, (stage_16_v76,)) in pl.pipeline(0, 512, 128, stage=2, init_values=(stage_16_v46,)):
                        stage_16_v72: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(stage_16_v39, 0, stage_16_v71, [64, 128], target_memory=pl.Mem.Left)
                        stage_16_v73: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_16_v45, stage_16_v71, stage_16_v43, [128, 128], target_memory=pl.Mem.Right)
                        stage_16_v74: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_16_v76, stage_16_v72, stage_16_v73, stage_16_v71 == 0)
                        stage_16_v75: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_16_v74)
                    stage_16_v47 = pl.tile.store(stage_16_v75, [stage_16_v42, 0], temporary_57)
                    pl.system.sync_set(1, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                    stage_16_v48: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.yield_(stage_16_v44)
                else:
                    stage_16_v48: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.yield_(stage_16_v27)
                (stage_16_v49, stage_16_v50) = pl.yield_(stage_16_v48, stage_16_v39)
            else:
                (stage_16_v49, stage_16_v50) = pl.yield_(stage_16_v27, stage_16_v28)
            if 2 <= stage_16_v26:
                stage_16_v32: pl.Scalar[pl.INDEX] = stage_16_v26 - 2
                stage_16_v33: pl.Scalar[pl.INDEX] = stage_16_v32 % 5
                stage_16_v34: pl.Scalar[pl.INDEX] = stage_16_v21 + stage_16_v32 // 5 * 24
                stage_16_v35: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_16_v34, stage_16_v33])
                if 0 < pl.cast(stage_16_v35, pl.INDEX):
                    stage_16_v51: pl.Scalar[pl.INDEX] = stage_16_v21 * 3 + stage_16_v32 % 3
                    stage_16_v52: pl.Scalar[pl.INDEX] = stage_16_v51 * 64
                    pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    stage_16_v53: pl.Tile[[64, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(temporary_58, [stage_16_v52, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat)
                    stage_16_v54: pl.Scalar[pl.INDEX] = stage_16_v32 % 3 * 128
                    stage_16_v55: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(stage_16_v53, target_memory=pl.Mem.Left)
                    stage_16_v56: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_16_v49, stage_16_v54, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_16_v57: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_16_v55, stage_16_v56)
                    stage_16_v58: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_16_v49, stage_16_v54, 128, [128, 128], target_memory=pl.Mem.Right)
                    stage_16_v59: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_16_v55, stage_16_v58)
                    stage_16_v60 = pl.tile.store(stage_16_v57, [stage_16_v52, 0], temporary_59)
                    stage_16_v61: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = stage_16_v59
                    stage_16_v62: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_16_v49, stage_16_v54, 256, [128, 128], target_memory=pl.Mem.Right)
                    stage_16_v63: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_16_v55, stage_16_v62)
                    stage_16_v64 = pl.tile.store(stage_16_v61, [stage_16_v52, 128], temporary_59)
                    stage_16_v65: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = stage_16_v63
                    stage_16_v66: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_16_v49, stage_16_v54, 384, [128, 128], target_memory=pl.Mem.Right)
                    stage_16_v67: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_16_v55, stage_16_v66)
                    stage_16_v68 = pl.tile.store(stage_16_v65, [stage_16_v52, 256], temporary_59)
                    stage_16_v69: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = stage_16_v67
                    stage_16_v70 = pl.tile.store(stage_16_v69, [stage_16_v52, 384], temporary_59)
                    pl.system.sync_set(3, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            (stage_16_v36, stage_16_v37) = pl.yield_(stage_16_v49, stage_16_v50)
        pl.system.set_ffts(temporary_62)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_16 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_16)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_17 = 7344 + core_id * 6
    mapped_block_17 = pl.cast(pl.tensor.read(records, [record_17]), pl.INDEX)
    block_17 = pl.max(mapped_block_17, 0)
    if mapped_block_17 >= 0:
        subslot_17 = pl.cast(pl.tensor.read(records, [record_17 + 1]), pl.INDEX)
        done_17 = pl.tensor.read(records, [record_17 + 2])
        wait_base_17 = pl.cast(pl.tensor.read(records, [record_17 + 3]), pl.INDEX)
        wait_count_17 = pl.cast(pl.tensor.read(records, [record_17 + 4]), pl.INDEX)
        publish_17 = pl.tensor.read(records, [record_17 + 5])
        for dependency_17 in pl.range(wait_count_17):
            wait_index_17 = (wait_base_17 + dependency_17) * 2
            other_17 = pl.cast(pl.tensor.read(waits, [wait_index_17]), pl.INDEX)
            required_17 = pl.tensor.read(waits, [wait_index_17 + 1])
            target_17 = generation_base + required_17
            flag_17 = other_17 * 64
            pl.system.cacheinvalid(progress, [1], [flag_17])
            initial_17 = pl.tensor.read(progress, [flag_17])
            for (ready_17,) in pl.while_(init_values=(initial_17,)):
                pl.cond(ready_17 < target_17)
                pl.system.cacheinvalid(progress, [1], [flag_17])
                after_17 = pl.yield_(pl.tensor.read(progress, [flag_17]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_17_v4: pl.Scalar[pl.INDEX] = block_17
        stage_17_v5: pl.Scalar[pl.INDEX] = stage_17_v4 // 4
        stage_17_v6: pl.Scalar[pl.INDEX] = stage_17_v4 % 4
        stage_17_v7: pl.Scalar[pl.INDEX] = stage_17_v5 * 128
        stage_17_v8: pl.Scalar[pl.INDEX] = pl.min(144 - stage_17_v7, 128)
        stage_17_v9: pl.Scalar[pl.INDEX] = 2688 + stage_17_v7
        stage_17_v10: pl.Scalar[pl.INDEX] = stage_17_v6 * 256
        stage_17_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_17_v9, 0], [stage_17_v8, 256])
        stage_17_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [7, 0, stage_17_v10])
        stage_17_v13 = pl.tensor.matmul(stage_17_v11, stage_17_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_17_v15, (stage_17_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_17_v13,)):
            stage_17_v16: pl.Scalar[pl.INDEX] = stage_17_v15 * 256
            stage_17_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_17_v9, stage_17_v16], [stage_17_v8, 256])
            stage_17_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [7, stage_17_v16, stage_17_v10])
            stage_17_v19 = pl.tensor.matmul_acc(stage_17_v21, stage_17_v17, stage_17_v18, a_trans=False, b_trans=False)
            stage_17_v20 = pl.yield_(stage_17_v19)
        stage_17_v14 = pl.tensor.assemble(temporary_63, stage_17_v20, [stage_17_v7, 7168 + stage_17_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_17 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_17)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_18 = 7776 + core_id * 6
    mapped_block_18 = pl.cast(pl.tensor.read(records, [record_18]), pl.INDEX)
    block_18 = pl.max(mapped_block_18, 0)
    if mapped_block_18 >= 0:
        subslot_18 = pl.cast(pl.tensor.read(records, [record_18 + 1]), pl.INDEX)
        done_18 = pl.tensor.read(records, [record_18 + 2])
        wait_base_18 = pl.cast(pl.tensor.read(records, [record_18 + 3]), pl.INDEX)
        wait_count_18 = pl.cast(pl.tensor.read(records, [record_18 + 4]), pl.INDEX)
        publish_18 = pl.tensor.read(records, [record_18 + 5])
        for dependency_18 in pl.range(wait_count_18):
            wait_index_18 = (wait_base_18 + dependency_18) * 2
            other_18 = pl.cast(pl.tensor.read(waits, [wait_index_18]), pl.INDEX)
            required_18 = pl.tensor.read(waits, [wait_index_18 + 1])
            target_18 = generation_base + required_18
            flag_18 = other_18 * 64
            pl.system.cacheinvalid(progress, [1], [flag_18])
            initial_18 = pl.tensor.read(progress, [flag_18])
            for (ready_18,) in pl.while_(init_values=(initial_18,)):
                pl.cond(ready_18 < target_18)
                pl.system.cacheinvalid(progress, [1], [flag_18])
                after_18 = pl.yield_(pl.tensor.read(progress, [flag_18]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_18_v4: pl.Scalar[pl.INDEX] = block_18
        stage_18_v5: pl.Scalar[pl.INDEX] = stage_18_v4 // 4
        stage_18_v6: pl.Scalar[pl.INDEX] = stage_18_v4 % 4
        stage_18_v7: pl.Scalar[pl.INDEX] = stage_18_v5 * 128
        stage_18_v8: pl.Scalar[pl.INDEX] = pl.min(144 - stage_18_v7, 128)
        stage_18_v9: pl.Scalar[pl.INDEX] = 768 + stage_18_v7
        stage_18_v10: pl.Scalar[pl.INDEX] = stage_18_v6 * 256
        stage_18_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_18_v9, 0], [stage_18_v8, 256])
        stage_18_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [2, 0, stage_18_v10])
        stage_18_v13 = pl.tensor.matmul(stage_18_v11, stage_18_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_18_v15, (stage_18_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_18_v13,)):
            stage_18_v16: pl.Scalar[pl.INDEX] = stage_18_v15 * 256
            stage_18_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_18_v9, stage_18_v16], [stage_18_v8, 256])
            stage_18_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [2, stage_18_v16, stage_18_v10])
            stage_18_v19 = pl.tensor.matmul_acc(stage_18_v21, stage_18_v17, stage_18_v18, a_trans=False, b_trans=False)
            stage_18_v20 = pl.yield_(stage_18_v19)
        stage_18_v14 = pl.tensor.assemble(temporary_63, stage_18_v20, [stage_18_v7, 2048 + stage_18_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_18 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_18)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_19 = 8208 + core_id * 6
    mapped_block_19 = pl.cast(pl.tensor.read(records, [record_19]), pl.INDEX)
    block_19 = pl.max(mapped_block_19, 0)
    if mapped_block_19 >= 0:
        subslot_19 = pl.cast(pl.tensor.read(records, [record_19 + 1]), pl.INDEX)
        done_19 = pl.tensor.read(records, [record_19 + 2])
        wait_base_19 = pl.cast(pl.tensor.read(records, [record_19 + 3]), pl.INDEX)
        wait_count_19 = pl.cast(pl.tensor.read(records, [record_19 + 4]), pl.INDEX)
        publish_19 = pl.tensor.read(records, [record_19 + 5])
        for dependency_19 in pl.range(wait_count_19):
            wait_index_19 = (wait_base_19 + dependency_19) * 2
            other_19 = pl.cast(pl.tensor.read(waits, [wait_index_19]), pl.INDEX)
            required_19 = pl.tensor.read(waits, [wait_index_19 + 1])
            target_19 = generation_base + required_19
            flag_19 = other_19 * 64
            pl.system.cacheinvalid(progress, [1], [flag_19])
            initial_19 = pl.tensor.read(progress, [flag_19])
            for (ready_19,) in pl.while_(init_values=(initial_19,)):
                pl.cond(ready_19 < target_19)
                pl.system.cacheinvalid(progress, [1], [flag_19])
                after_19 = pl.yield_(pl.tensor.read(progress, [flag_19]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_19_v4: pl.Scalar[pl.INDEX] = block_19
        stage_19_v5: pl.Scalar[pl.INDEX] = stage_19_v4 // 4
        stage_19_v6: pl.Scalar[pl.INDEX] = stage_19_v4 % 4
        stage_19_v7: pl.Scalar[pl.INDEX] = stage_19_v5 * 128
        stage_19_v8: pl.Scalar[pl.INDEX] = pl.min(144 - stage_19_v7, 128)
        stage_19_v9: pl.Scalar[pl.INDEX] = 384 + stage_19_v7
        stage_19_v10: pl.Scalar[pl.INDEX] = stage_19_v6 * 256
        stage_19_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_19_v9, 0], [stage_19_v8, 256])
        stage_19_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [1, 0, stage_19_v10])
        stage_19_v13 = pl.tensor.matmul(stage_19_v11, stage_19_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_19_v15, (stage_19_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_19_v13,)):
            stage_19_v16: pl.Scalar[pl.INDEX] = stage_19_v15 * 256
            stage_19_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_19_v9, stage_19_v16], [stage_19_v8, 256])
            stage_19_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [1, stage_19_v16, stage_19_v10])
            stage_19_v19 = pl.tensor.matmul_acc(stage_19_v21, stage_19_v17, stage_19_v18, a_trans=False, b_trans=False)
            stage_19_v20 = pl.yield_(stage_19_v19)
        stage_19_v14 = pl.tensor.assemble(temporary_63, stage_19_v20, [stage_19_v7, 1024 + stage_19_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_19 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_19)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_20 = 8640 + core_id * 6
    mapped_block_20 = pl.cast(pl.tensor.read(records, [record_20]), pl.INDEX)
    block_20 = pl.max(mapped_block_20, 0)
    if mapped_block_20 >= 0:
        subslot_20 = pl.cast(pl.tensor.read(records, [record_20 + 1]), pl.INDEX)
        done_20 = pl.tensor.read(records, [record_20 + 2])
        wait_base_20 = pl.cast(pl.tensor.read(records, [record_20 + 3]), pl.INDEX)
        wait_count_20 = pl.cast(pl.tensor.read(records, [record_20 + 4]), pl.INDEX)
        publish_20 = pl.tensor.read(records, [record_20 + 5])
        for dependency_20 in pl.range(wait_count_20):
            wait_index_20 = (wait_base_20 + dependency_20) * 2
            other_20 = pl.cast(pl.tensor.read(waits, [wait_index_20]), pl.INDEX)
            required_20 = pl.tensor.read(waits, [wait_index_20 + 1])
            target_20 = generation_base + required_20
            flag_20 = other_20 * 64
            pl.system.cacheinvalid(progress, [1], [flag_20])
            initial_20 = pl.tensor.read(progress, [flag_20])
            for (ready_20,) in pl.while_(init_values=(initial_20,)):
                pl.cond(ready_20 < target_20)
                pl.system.cacheinvalid(progress, [1], [flag_20])
                after_20 = pl.yield_(pl.tensor.read(progress, [flag_20]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_20_v4: pl.Scalar[pl.INDEX] = block_20
        stage_20_v5: pl.Scalar[pl.INDEX] = stage_20_v4 // 4
        stage_20_v6: pl.Scalar[pl.INDEX] = stage_20_v4 % 4
        stage_20_v7: pl.Scalar[pl.INDEX] = stage_20_v5 * 128
        stage_20_v8: pl.Scalar[pl.INDEX] = pl.min(144 - stage_20_v7, 128)
        stage_20_v9: pl.Scalar[pl.INDEX] = 1920 + stage_20_v7
        stage_20_v10: pl.Scalar[pl.INDEX] = stage_20_v6 * 256
        stage_20_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_20_v9, 0], [stage_20_v8, 256])
        stage_20_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [5, 0, stage_20_v10])
        stage_20_v13 = pl.tensor.matmul(stage_20_v11, stage_20_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_20_v15, (stage_20_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_20_v13,)):
            stage_20_v16: pl.Scalar[pl.INDEX] = stage_20_v15 * 256
            stage_20_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_20_v9, stage_20_v16], [stage_20_v8, 256])
            stage_20_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [5, stage_20_v16, stage_20_v10])
            stage_20_v19 = pl.tensor.matmul_acc(stage_20_v21, stage_20_v17, stage_20_v18, a_trans=False, b_trans=False)
            stage_20_v20 = pl.yield_(stage_20_v19)
        stage_20_v14 = pl.tensor.assemble(temporary_63, stage_20_v20, [stage_20_v7, 5120 + stage_20_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_20 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_20)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_21 = 9072 + core_id * 6
    mapped_block_21 = pl.cast(pl.tensor.read(records, [record_21]), pl.INDEX)
    block_21 = pl.max(mapped_block_21, 0)
    if mapped_block_21 >= 0:
        subslot_21 = pl.cast(pl.tensor.read(records, [record_21 + 1]), pl.INDEX)
        done_21 = pl.tensor.read(records, [record_21 + 2])
        wait_base_21 = pl.cast(pl.tensor.read(records, [record_21 + 3]), pl.INDEX)
        wait_count_21 = pl.cast(pl.tensor.read(records, [record_21 + 4]), pl.INDEX)
        publish_21 = pl.tensor.read(records, [record_21 + 5])
        for dependency_21 in pl.range(wait_count_21):
            wait_index_21 = (wait_base_21 + dependency_21) * 2
            other_21 = pl.cast(pl.tensor.read(waits, [wait_index_21]), pl.INDEX)
            required_21 = pl.tensor.read(waits, [wait_index_21 + 1])
            target_21 = generation_base + required_21
            flag_21 = other_21 * 64
            pl.system.cacheinvalid(progress, [1], [flag_21])
            initial_21 = pl.tensor.read(progress, [flag_21])
            for (ready_21,) in pl.while_(init_values=(initial_21,)):
                pl.cond(ready_21 < target_21)
                pl.system.cacheinvalid(progress, [1], [flag_21])
                after_21 = pl.yield_(pl.tensor.read(progress, [flag_21]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_21_v4: pl.Scalar[pl.INDEX] = block_21
        stage_21_v5: pl.Scalar[pl.INDEX] = stage_21_v4 // 4
        stage_21_v6: pl.Scalar[pl.INDEX] = stage_21_v4 % 4
        stage_21_v7: pl.Scalar[pl.INDEX] = stage_21_v5 * 128
        stage_21_v8: pl.Scalar[pl.INDEX] = pl.min(144 - stage_21_v7, 128)
        stage_21_v9: pl.Scalar[pl.INDEX] = 2304 + stage_21_v7
        stage_21_v10: pl.Scalar[pl.INDEX] = stage_21_v6 * 256
        stage_21_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_21_v9, 0], [stage_21_v8, 256])
        stage_21_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [6, 0, stage_21_v10])
        stage_21_v13 = pl.tensor.matmul(stage_21_v11, stage_21_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_21_v15, (stage_21_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_21_v13,)):
            stage_21_v16: pl.Scalar[pl.INDEX] = stage_21_v15 * 256
            stage_21_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_21_v9, stage_21_v16], [stage_21_v8, 256])
            stage_21_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [6, stage_21_v16, stage_21_v10])
            stage_21_v19 = pl.tensor.matmul_acc(stage_21_v21, stage_21_v17, stage_21_v18, a_trans=False, b_trans=False)
            stage_21_v20 = pl.yield_(stage_21_v19)
        stage_21_v14 = pl.tensor.assemble(temporary_63, stage_21_v20, [stage_21_v7, 6144 + stage_21_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_21 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_21)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_22 = 9504 + core_id * 6
    mapped_block_22 = pl.cast(pl.tensor.read(records, [record_22]), pl.INDEX)
    block_22 = pl.max(mapped_block_22, 0)
    if mapped_block_22 >= 0:
        subslot_22 = pl.cast(pl.tensor.read(records, [record_22 + 1]), pl.INDEX)
        done_22 = pl.tensor.read(records, [record_22 + 2])
        wait_base_22 = pl.cast(pl.tensor.read(records, [record_22 + 3]), pl.INDEX)
        wait_count_22 = pl.cast(pl.tensor.read(records, [record_22 + 4]), pl.INDEX)
        publish_22 = pl.tensor.read(records, [record_22 + 5])
        for dependency_22 in pl.range(wait_count_22):
            wait_index_22 = (wait_base_22 + dependency_22) * 2
            other_22 = pl.cast(pl.tensor.read(waits, [wait_index_22]), pl.INDEX)
            required_22 = pl.tensor.read(waits, [wait_index_22 + 1])
            target_22 = generation_base + required_22
            flag_22 = other_22 * 64
            pl.system.cacheinvalid(progress, [1], [flag_22])
            initial_22 = pl.tensor.read(progress, [flag_22])
            for (ready_22,) in pl.while_(init_values=(initial_22,)):
                pl.cond(ready_22 < target_22)
                pl.system.cacheinvalid(progress, [1], [flag_22])
                after_22 = pl.yield_(pl.tensor.read(progress, [flag_22]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_22_v4: pl.Scalar[pl.INDEX] = block_22
        stage_22_v5: pl.Scalar[pl.INDEX] = stage_22_v4 // 4
        stage_22_v6: pl.Scalar[pl.INDEX] = stage_22_v4 % 4
        stage_22_v7: pl.Scalar[pl.INDEX] = stage_22_v5 * 128
        stage_22_v8: pl.Scalar[pl.INDEX] = pl.min(144 - stage_22_v7, 128)
        stage_22_v9: pl.Scalar[pl.INDEX] = 1536 + stage_22_v7
        stage_22_v10: pl.Scalar[pl.INDEX] = stage_22_v6 * 256
        stage_22_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_22_v9, 0], [stage_22_v8, 256])
        stage_22_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [4, 0, stage_22_v10])
        stage_22_v13 = pl.tensor.matmul(stage_22_v11, stage_22_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_22_v15, (stage_22_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_22_v13,)):
            stage_22_v16: pl.Scalar[pl.INDEX] = stage_22_v15 * 256
            stage_22_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_22_v9, stage_22_v16], [stage_22_v8, 256])
            stage_22_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [4, stage_22_v16, stage_22_v10])
            stage_22_v19 = pl.tensor.matmul_acc(stage_22_v21, stage_22_v17, stage_22_v18, a_trans=False, b_trans=False)
            stage_22_v20 = pl.yield_(stage_22_v19)
        stage_22_v14 = pl.tensor.assemble(temporary_63, stage_22_v20, [stage_22_v7, 4096 + stage_22_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_22 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_22)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_23 = 9936 + core_id * 6
    mapped_block_23 = pl.cast(pl.tensor.read(records, [record_23]), pl.INDEX)
    block_23 = pl.max(mapped_block_23, 0)
    if mapped_block_23 >= 0:
        subslot_23 = pl.cast(pl.tensor.read(records, [record_23 + 1]), pl.INDEX)
        done_23 = pl.tensor.read(records, [record_23 + 2])
        wait_base_23 = pl.cast(pl.tensor.read(records, [record_23 + 3]), pl.INDEX)
        wait_count_23 = pl.cast(pl.tensor.read(records, [record_23 + 4]), pl.INDEX)
        publish_23 = pl.tensor.read(records, [record_23 + 5])
        for dependency_23 in pl.range(wait_count_23):
            wait_index_23 = (wait_base_23 + dependency_23) * 2
            other_23 = pl.cast(pl.tensor.read(waits, [wait_index_23]), pl.INDEX)
            required_23 = pl.tensor.read(waits, [wait_index_23 + 1])
            target_23 = generation_base + required_23
            flag_23 = other_23 * 64
            pl.system.cacheinvalid(progress, [1], [flag_23])
            initial_23 = pl.tensor.read(progress, [flag_23])
            for (ready_23,) in pl.while_(init_values=(initial_23,)):
                pl.cond(ready_23 < target_23)
                pl.system.cacheinvalid(progress, [1], [flag_23])
                after_23 = pl.yield_(pl.tensor.read(progress, [flag_23]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_23_v4: pl.Scalar[pl.INDEX] = block_23
        stage_23_v5: pl.Scalar[pl.INDEX] = stage_23_v4 // 4
        stage_23_v6: pl.Scalar[pl.INDEX] = stage_23_v4 % 4
        stage_23_v7: pl.Scalar[pl.INDEX] = stage_23_v5 * 128
        stage_23_v8: pl.Scalar[pl.INDEX] = pl.min(144 - stage_23_v7, 128)
        stage_23_v9: pl.Scalar[pl.INDEX] = 384 + stage_23_v7
        stage_23_v10: pl.Scalar[pl.INDEX] = stage_23_v6 * 256
        stage_23_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_23_v9, 0], [stage_23_v8, 256])
        stage_23_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [1, 0, stage_23_v10])
        stage_23_v13 = pl.tensor.matmul(stage_23_v11, stage_23_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_23_v15, (stage_23_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_23_v13,)):
            stage_23_v16: pl.Scalar[pl.INDEX] = stage_23_v15 * 256
            stage_23_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_23_v9, stage_23_v16], [stage_23_v8, 256])
            stage_23_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [1, stage_23_v16, stage_23_v10])
            stage_23_v19 = pl.tensor.matmul_acc(stage_23_v21, stage_23_v17, stage_23_v18, a_trans=False, b_trans=False)
            stage_23_v20 = pl.yield_(stage_23_v19)
        stage_23_v14 = pl.tensor.assemble(temporary_63, stage_23_v20, [stage_23_v7, 1024 + stage_23_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_23 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_23)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_24 = 10368 + core_id * 6
    mapped_block_24 = pl.cast(pl.tensor.read(records, [record_24]), pl.INDEX)
    block_24 = pl.max(mapped_block_24, 0)
    if mapped_block_24 >= 0:
        subslot_24 = pl.cast(pl.tensor.read(records, [record_24 + 1]), pl.INDEX)
        done_24 = pl.tensor.read(records, [record_24 + 2])
        wait_base_24 = pl.cast(pl.tensor.read(records, [record_24 + 3]), pl.INDEX)
        wait_count_24 = pl.cast(pl.tensor.read(records, [record_24 + 4]), pl.INDEX)
        publish_24 = pl.tensor.read(records, [record_24 + 5])
        for dependency_24 in pl.range(wait_count_24):
            wait_index_24 = (wait_base_24 + dependency_24) * 2
            other_24 = pl.cast(pl.tensor.read(waits, [wait_index_24]), pl.INDEX)
            required_24 = pl.tensor.read(waits, [wait_index_24 + 1])
            target_24 = generation_base + required_24
            flag_24 = other_24 * 64
            pl.system.cacheinvalid(progress, [1], [flag_24])
            initial_24 = pl.tensor.read(progress, [flag_24])
            for (ready_24,) in pl.while_(init_values=(initial_24,)):
                pl.cond(ready_24 < target_24)
                pl.system.cacheinvalid(progress, [1], [flag_24])
                after_24 = pl.yield_(pl.tensor.read(progress, [flag_24]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_24_v4: pl.Scalar[pl.INDEX] = block_24
        stage_24_v5: pl.Scalar[pl.INDEX] = stage_24_v4 // 4
        stage_24_v6: pl.Scalar[pl.INDEX] = stage_24_v4 % 4
        stage_24_v7: pl.Scalar[pl.INDEX] = stage_24_v5 * 128
        stage_24_v8: pl.Scalar[pl.INDEX] = pl.min(144 - stage_24_v7, 128)
        stage_24_v9: pl.Scalar[pl.INDEX] = 0 + stage_24_v7
        stage_24_v10: pl.Scalar[pl.INDEX] = stage_24_v6 * 256
        stage_24_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_24_v9, 0], [stage_24_v8, 256])
        stage_24_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [0, 0, stage_24_v10])
        stage_24_v13 = pl.tensor.matmul(stage_24_v11, stage_24_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_24_v15, (stage_24_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_24_v13,)):
            stage_24_v16: pl.Scalar[pl.INDEX] = stage_24_v15 * 256
            stage_24_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_24_v9, stage_24_v16], [stage_24_v8, 256])
            stage_24_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [0, stage_24_v16, stage_24_v10])
            stage_24_v19 = pl.tensor.matmul_acc(stage_24_v21, stage_24_v17, stage_24_v18, a_trans=False, b_trans=False)
            stage_24_v20 = pl.yield_(stage_24_v19)
        stage_24_v14 = pl.tensor.assemble(temporary_63, stage_24_v20, [stage_24_v7, 0 + stage_24_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_24 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_24)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_25 = 10800 + core_id * 6
    mapped_block_25 = pl.cast(pl.tensor.read(records, [record_25]), pl.INDEX)
    block_25 = pl.max(mapped_block_25, 0)
    if mapped_block_25 >= 0:
        subslot_25 = pl.cast(pl.tensor.read(records, [record_25 + 1]), pl.INDEX)
        done_25 = pl.tensor.read(records, [record_25 + 2])
        wait_base_25 = pl.cast(pl.tensor.read(records, [record_25 + 3]), pl.INDEX)
        wait_count_25 = pl.cast(pl.tensor.read(records, [record_25 + 4]), pl.INDEX)
        publish_25 = pl.tensor.read(records, [record_25 + 5])
        for dependency_25 in pl.range(wait_count_25):
            wait_index_25 = (wait_base_25 + dependency_25) * 2
            other_25 = pl.cast(pl.tensor.read(waits, [wait_index_25]), pl.INDEX)
            required_25 = pl.tensor.read(waits, [wait_index_25 + 1])
            target_25 = generation_base + required_25
            flag_25 = other_25 * 64
            pl.system.cacheinvalid(progress, [1], [flag_25])
            initial_25 = pl.tensor.read(progress, [flag_25])
            for (ready_25,) in pl.while_(init_values=(initial_25,)):
                pl.cond(ready_25 < target_25)
                pl.system.cacheinvalid(progress, [1], [flag_25])
                after_25 = pl.yield_(pl.tensor.read(progress, [flag_25]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_25_v4: pl.Scalar[pl.INDEX] = block_25
        stage_25_v5: pl.Scalar[pl.INDEX] = stage_25_v4 * 512
        for (stage_25_v6, (stage_25_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_25_v7: pl.Scalar[pl.INDEX] = stage_25_v5 + stage_25_v6 * 256
            stage_25_v8: pl.Scalar[pl.INDEX] = pl.max(2, 0) * 1024
            stage_25_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_25_v8, stage_25_v7])
            for (stage_25_v12, (stage_25_v18,)) in pl.range(2, init_values=(stage_25_v11,)):
                stage_25_v13: pl.Scalar[pl.INDEX] = stage_25_v12 * 128
                stage_25_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_25_v13, 2048])
                stage_25_v15 = pl.tensor.matmul(stage_25_v14, stage_25_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_25_v16 = pl.tensor.assemble(stage_25_v18, stage_25_v15, [stage_25_v13, 2 * 4096 + stage_25_v7])
                stage_25_v17 = pl.yield_(stage_25_v16)
            stage_25_v10 = pl.yield_(stage_25_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_25 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_25)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_26 = 11232 + core_id * 6
    mapped_block_26 = pl.cast(pl.tensor.read(records, [record_26]), pl.INDEX)
    block_26 = pl.max(mapped_block_26, 0)
    if mapped_block_26 >= 0:
        subslot_26 = pl.cast(pl.tensor.read(records, [record_26 + 1]), pl.INDEX)
        done_26 = pl.tensor.read(records, [record_26 + 2])
        wait_base_26 = pl.cast(pl.tensor.read(records, [record_26 + 3]), pl.INDEX)
        wait_count_26 = pl.cast(pl.tensor.read(records, [record_26 + 4]), pl.INDEX)
        publish_26 = pl.tensor.read(records, [record_26 + 5])
        for dependency_26 in pl.range(wait_count_26):
            wait_index_26 = (wait_base_26 + dependency_26) * 2
            other_26 = pl.cast(pl.tensor.read(waits, [wait_index_26]), pl.INDEX)
            required_26 = pl.tensor.read(waits, [wait_index_26 + 1])
            target_26 = generation_base + required_26
            flag_26 = other_26 * 64
            pl.system.cacheinvalid(progress, [1], [flag_26])
            initial_26 = pl.tensor.read(progress, [flag_26])
            for (ready_26,) in pl.while_(init_values=(initial_26,)):
                pl.cond(ready_26 < target_26)
                pl.system.cacheinvalid(progress, [1], [flag_26])
                after_26 = pl.yield_(pl.tensor.read(progress, [flag_26]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_26_v4: pl.Scalar[pl.INDEX] = block_26
        stage_26_v5: pl.Scalar[pl.INDEX] = stage_26_v4 // 4
        stage_26_v6: pl.Scalar[pl.INDEX] = stage_26_v4 % 4
        stage_26_v7: pl.Scalar[pl.INDEX] = stage_26_v5 * 128
        stage_26_v8: pl.Scalar[pl.INDEX] = pl.min(144 - stage_26_v7, 128)
        stage_26_v9: pl.Scalar[pl.INDEX] = 1152 + stage_26_v7
        stage_26_v10: pl.Scalar[pl.INDEX] = stage_26_v6 * 256
        stage_26_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_26_v9, 0], [stage_26_v8, 256])
        stage_26_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [3, 0, stage_26_v10])
        stage_26_v13 = pl.tensor.matmul(stage_26_v11, stage_26_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_26_v15, (stage_26_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_26_v13,)):
            stage_26_v16: pl.Scalar[pl.INDEX] = stage_26_v15 * 256
            stage_26_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_26_v9, stage_26_v16], [stage_26_v8, 256])
            stage_26_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [3, stage_26_v16, stage_26_v10])
            stage_26_v19 = pl.tensor.matmul_acc(stage_26_v21, stage_26_v17, stage_26_v18, a_trans=False, b_trans=False)
            stage_26_v20 = pl.yield_(stage_26_v19)
        stage_26_v14 = pl.tensor.assemble(temporary_63, stage_26_v20, [stage_26_v7, 3072 + stage_26_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_26 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_26)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_27 = 11664 + core_id * 6
    mapped_block_27 = pl.cast(pl.tensor.read(records, [record_27]), pl.INDEX)
    block_27 = pl.max(mapped_block_27, 0)
    if mapped_block_27 >= 0:
        subslot_27 = pl.cast(pl.tensor.read(records, [record_27 + 1]), pl.INDEX)
        done_27 = pl.tensor.read(records, [record_27 + 2])
        wait_base_27 = pl.cast(pl.tensor.read(records, [record_27 + 3]), pl.INDEX)
        wait_count_27 = pl.cast(pl.tensor.read(records, [record_27 + 4]), pl.INDEX)
        publish_27 = pl.tensor.read(records, [record_27 + 5])
        for dependency_27 in pl.range(wait_count_27):
            wait_index_27 = (wait_base_27 + dependency_27) * 2
            other_27 = pl.cast(pl.tensor.read(waits, [wait_index_27]), pl.INDEX)
            required_27 = pl.tensor.read(waits, [wait_index_27 + 1])
            target_27 = generation_base + required_27
            flag_27 = other_27 * 64
            pl.system.cacheinvalid(progress, [1], [flag_27])
            initial_27 = pl.tensor.read(progress, [flag_27])
            for (ready_27,) in pl.while_(init_values=(initial_27,)):
                pl.cond(ready_27 < target_27)
                pl.system.cacheinvalid(progress, [1], [flag_27])
                after_27 = pl.yield_(pl.tensor.read(progress, [flag_27]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_27_v4: pl.Scalar[pl.INDEX] = block_27
        stage_27_v5: pl.Scalar[pl.INDEX] = stage_27_v4 // 4
        stage_27_v6: pl.Scalar[pl.INDEX] = stage_27_v4 % 4
        stage_27_v7: pl.Scalar[pl.INDEX] = stage_27_v5 * 128
        stage_27_v8: pl.Scalar[pl.INDEX] = pl.min(144 - stage_27_v7, 128)
        stage_27_v9: pl.Scalar[pl.INDEX] = 384 + stage_27_v7
        stage_27_v10: pl.Scalar[pl.INDEX] = stage_27_v6 * 256
        stage_27_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_27_v9, 0], [stage_27_v8, 256])
        stage_27_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [1, 0, stage_27_v10])
        stage_27_v13 = pl.tensor.matmul(stage_27_v11, stage_27_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_27_v15, (stage_27_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_27_v13,)):
            stage_27_v16: pl.Scalar[pl.INDEX] = stage_27_v15 * 256
            stage_27_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_27_v9, stage_27_v16], [stage_27_v8, 256])
            stage_27_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [1, stage_27_v16, stage_27_v10])
            stage_27_v19 = pl.tensor.matmul_acc(stage_27_v21, stage_27_v17, stage_27_v18, a_trans=False, b_trans=False)
            stage_27_v20 = pl.yield_(stage_27_v19)
        stage_27_v14 = pl.tensor.assemble(temporary_63, stage_27_v20, [stage_27_v7, 1024 + stage_27_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_27 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_27)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_28 = 12096 + core_id * 6
    mapped_block_28 = pl.cast(pl.tensor.read(records, [record_28]), pl.INDEX)
    block_28 = pl.max(mapped_block_28, 0)
    if mapped_block_28 >= 0:
        subslot_28 = pl.cast(pl.tensor.read(records, [record_28 + 1]), pl.INDEX)
        done_28 = pl.tensor.read(records, [record_28 + 2])
        wait_base_28 = pl.cast(pl.tensor.read(records, [record_28 + 3]), pl.INDEX)
        wait_count_28 = pl.cast(pl.tensor.read(records, [record_28 + 4]), pl.INDEX)
        publish_28 = pl.tensor.read(records, [record_28 + 5])
        for dependency_28 in pl.range(wait_count_28):
            wait_index_28 = (wait_base_28 + dependency_28) * 2
            other_28 = pl.cast(pl.tensor.read(waits, [wait_index_28]), pl.INDEX)
            required_28 = pl.tensor.read(waits, [wait_index_28 + 1])
            target_28 = generation_base + required_28
            flag_28 = other_28 * 64
            pl.system.cacheinvalid(progress, [1], [flag_28])
            initial_28 = pl.tensor.read(progress, [flag_28])
            for (ready_28,) in pl.while_(init_values=(initial_28,)):
                pl.cond(ready_28 < target_28)
                pl.system.cacheinvalid(progress, [1], [flag_28])
                after_28 = pl.yield_(pl.tensor.read(progress, [flag_28]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_28_v4: pl.Scalar[pl.INDEX] = block_28
        stage_28_v5: pl.Scalar[pl.INDEX] = stage_28_v4 * 512
        for (stage_28_v6, (stage_28_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_28_v7: pl.Scalar[pl.INDEX] = stage_28_v5 + stage_28_v6 * 256
            stage_28_v8: pl.Scalar[pl.INDEX] = pl.max(7, 0) * 1024
            stage_28_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_28_v8, stage_28_v7])
            for (stage_28_v12, (stage_28_v18,)) in pl.range(2, init_values=(stage_28_v11,)):
                stage_28_v13: pl.Scalar[pl.INDEX] = stage_28_v12 * 128
                stage_28_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_28_v13, 7168])
                stage_28_v15 = pl.tensor.matmul(stage_28_v14, stage_28_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_28_v16 = pl.tensor.assemble(stage_28_v18, stage_28_v15, [stage_28_v13, 7 * 4096 + stage_28_v7])
                stage_28_v17 = pl.yield_(stage_28_v16)
            stage_28_v10 = pl.yield_(stage_28_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_28 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_28)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_29 = 12528 + core_id * 6
    mapped_block_29 = pl.cast(pl.tensor.read(records, [record_29]), pl.INDEX)
    block_29 = pl.max(mapped_block_29, 0)
    if mapped_block_29 >= 0:
        subslot_29 = pl.cast(pl.tensor.read(records, [record_29 + 1]), pl.INDEX)
        done_29 = pl.tensor.read(records, [record_29 + 2])
        wait_base_29 = pl.cast(pl.tensor.read(records, [record_29 + 3]), pl.INDEX)
        wait_count_29 = pl.cast(pl.tensor.read(records, [record_29 + 4]), pl.INDEX)
        publish_29 = pl.tensor.read(records, [record_29 + 5])
        for dependency_29 in pl.range(wait_count_29):
            wait_index_29 = (wait_base_29 + dependency_29) * 2
            other_29 = pl.cast(pl.tensor.read(waits, [wait_index_29]), pl.INDEX)
            required_29 = pl.tensor.read(waits, [wait_index_29 + 1])
            target_29 = generation_base + required_29
            flag_29 = other_29 * 64
            pl.system.cacheinvalid(progress, [1], [flag_29])
            initial_29 = pl.tensor.read(progress, [flag_29])
            for (ready_29,) in pl.while_(init_values=(initial_29,)):
                pl.cond(ready_29 < target_29)
                pl.system.cacheinvalid(progress, [1], [flag_29])
                after_29 = pl.yield_(pl.tensor.read(progress, [flag_29]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_29_v4: pl.Scalar[pl.INDEX] = block_29
        stage_29_v5: pl.Scalar[pl.INDEX] = stage_29_v4 * 512
        for (stage_29_v6, (stage_29_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_29_v7: pl.Scalar[pl.INDEX] = stage_29_v5 + stage_29_v6 * 256
            stage_29_v8: pl.Scalar[pl.INDEX] = pl.max(4, 0) * 1024
            stage_29_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_29_v8, stage_29_v7])
            for (stage_29_v12, (stage_29_v18,)) in pl.range(2, init_values=(stage_29_v11,)):
                stage_29_v13: pl.Scalar[pl.INDEX] = stage_29_v12 * 128
                stage_29_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_29_v13, 4096])
                stage_29_v15 = pl.tensor.matmul(stage_29_v14, stage_29_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_29_v16 = pl.tensor.assemble(stage_29_v18, stage_29_v15, [stage_29_v13, 4 * 4096 + stage_29_v7])
                stage_29_v17 = pl.yield_(stage_29_v16)
            stage_29_v10 = pl.yield_(stage_29_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_29 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_29)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_30 = 12960 + core_id * 6
    mapped_block_30 = pl.cast(pl.tensor.read(records, [record_30]), pl.INDEX)
    block_30 = pl.max(mapped_block_30, 0)
    if mapped_block_30 >= 0:
        subslot_30 = pl.cast(pl.tensor.read(records, [record_30 + 1]), pl.INDEX)
        done_30 = pl.tensor.read(records, [record_30 + 2])
        wait_base_30 = pl.cast(pl.tensor.read(records, [record_30 + 3]), pl.INDEX)
        wait_count_30 = pl.cast(pl.tensor.read(records, [record_30 + 4]), pl.INDEX)
        publish_30 = pl.tensor.read(records, [record_30 + 5])
        for dependency_30 in pl.range(wait_count_30):
            wait_index_30 = (wait_base_30 + dependency_30) * 2
            other_30 = pl.cast(pl.tensor.read(waits, [wait_index_30]), pl.INDEX)
            required_30 = pl.tensor.read(waits, [wait_index_30 + 1])
            target_30 = generation_base + required_30
            flag_30 = other_30 * 64
            pl.system.cacheinvalid(progress, [1], [flag_30])
            initial_30 = pl.tensor.read(progress, [flag_30])
            for (ready_30,) in pl.while_(init_values=(initial_30,)):
                pl.cond(ready_30 < target_30)
                pl.system.cacheinvalid(progress, [1], [flag_30])
                after_30 = pl.yield_(pl.tensor.read(progress, [flag_30]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_30_v4: pl.Scalar[pl.INDEX] = block_30
        stage_30_v5: pl.Scalar[pl.INDEX] = stage_30_v4 * 512
        for (stage_30_v6, (stage_30_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_30_v7: pl.Scalar[pl.INDEX] = stage_30_v5 + stage_30_v6 * 256
            stage_30_v8: pl.Scalar[pl.INDEX] = pl.max(6, 0) * 1024
            stage_30_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_30_v8, stage_30_v7])
            for (stage_30_v12, (stage_30_v18,)) in pl.range(2, init_values=(stage_30_v11,)):
                stage_30_v13: pl.Scalar[pl.INDEX] = stage_30_v12 * 128
                stage_30_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_30_v13, 6144])
                stage_30_v15 = pl.tensor.matmul(stage_30_v14, stage_30_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_30_v16 = pl.tensor.assemble(stage_30_v18, stage_30_v15, [stage_30_v13, 6 * 4096 + stage_30_v7])
                stage_30_v17 = pl.yield_(stage_30_v16)
            stage_30_v10 = pl.yield_(stage_30_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_30 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_30)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_31 = 13392 + core_id * 6
    mapped_block_31 = pl.cast(pl.tensor.read(records, [record_31]), pl.INDEX)
    block_31 = pl.max(mapped_block_31, 0)
    if mapped_block_31 >= 0:
        subslot_31 = pl.cast(pl.tensor.read(records, [record_31 + 1]), pl.INDEX)
        done_31 = pl.tensor.read(records, [record_31 + 2])
        wait_base_31 = pl.cast(pl.tensor.read(records, [record_31 + 3]), pl.INDEX)
        wait_count_31 = pl.cast(pl.tensor.read(records, [record_31 + 4]), pl.INDEX)
        publish_31 = pl.tensor.read(records, [record_31 + 5])
        for dependency_31 in pl.range(wait_count_31):
            wait_index_31 = (wait_base_31 + dependency_31) * 2
            other_31 = pl.cast(pl.tensor.read(waits, [wait_index_31]), pl.INDEX)
            required_31 = pl.tensor.read(waits, [wait_index_31 + 1])
            target_31 = generation_base + required_31
            flag_31 = other_31 * 64
            pl.system.cacheinvalid(progress, [1], [flag_31])
            initial_31 = pl.tensor.read(progress, [flag_31])
            for (ready_31,) in pl.while_(init_values=(initial_31,)):
                pl.cond(ready_31 < target_31)
                pl.system.cacheinvalid(progress, [1], [flag_31])
                after_31 = pl.yield_(pl.tensor.read(progress, [flag_31]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_31_v4: pl.Scalar[pl.INDEX] = block_31
        stage_31_v5: pl.Scalar[pl.INDEX] = stage_31_v4 * 512
        for (stage_31_v6, (stage_31_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_31_v7: pl.Scalar[pl.INDEX] = stage_31_v5 + stage_31_v6 * 256
            stage_31_v8: pl.Scalar[pl.INDEX] = pl.max(5, 0) * 1024
            stage_31_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_31_v8, stage_31_v7])
            for (stage_31_v12, (stage_31_v18,)) in pl.range(2, init_values=(stage_31_v11,)):
                stage_31_v13: pl.Scalar[pl.INDEX] = stage_31_v12 * 128
                stage_31_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_31_v13, 5120])
                stage_31_v15 = pl.tensor.matmul(stage_31_v14, stage_31_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_31_v16 = pl.tensor.assemble(stage_31_v18, stage_31_v15, [stage_31_v13, 5 * 4096 + stage_31_v7])
                stage_31_v17 = pl.yield_(stage_31_v16)
            stage_31_v10 = pl.yield_(stage_31_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_31 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_31)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_32 = 13824 + core_id * 6
    mapped_block_32 = pl.cast(pl.tensor.read(records, [record_32]), pl.INDEX)
    block_32 = pl.max(mapped_block_32, 0)
    if mapped_block_32 >= 0:
        subslot_32 = pl.cast(pl.tensor.read(records, [record_32 + 1]), pl.INDEX)
        done_32 = pl.tensor.read(records, [record_32 + 2])
        wait_base_32 = pl.cast(pl.tensor.read(records, [record_32 + 3]), pl.INDEX)
        wait_count_32 = pl.cast(pl.tensor.read(records, [record_32 + 4]), pl.INDEX)
        publish_32 = pl.tensor.read(records, [record_32 + 5])
        for dependency_32 in pl.range(wait_count_32):
            wait_index_32 = (wait_base_32 + dependency_32) * 2
            other_32 = pl.cast(pl.tensor.read(waits, [wait_index_32]), pl.INDEX)
            required_32 = pl.tensor.read(waits, [wait_index_32 + 1])
            target_32 = generation_base + required_32
            flag_32 = other_32 * 64
            pl.system.cacheinvalid(progress, [1], [flag_32])
            initial_32 = pl.tensor.read(progress, [flag_32])
            for (ready_32,) in pl.while_(init_values=(initial_32,)):
                pl.cond(ready_32 < target_32)
                pl.system.cacheinvalid(progress, [1], [flag_32])
                after_32 = pl.yield_(pl.tensor.read(progress, [flag_32]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_32_v4: pl.Scalar[pl.INDEX] = block_32
        stage_32_v5: pl.Scalar[pl.INDEX] = stage_32_v4 * 512
        for (stage_32_v6, (stage_32_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_32_v7: pl.Scalar[pl.INDEX] = stage_32_v5 + stage_32_v6 * 256
            stage_32_v8: pl.Scalar[pl.INDEX] = pl.max(7, 0) * 1024
            stage_32_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_32_v8, stage_32_v7])
            for (stage_32_v12, (stage_32_v18,)) in pl.range(2, init_values=(stage_32_v11,)):
                stage_32_v13: pl.Scalar[pl.INDEX] = stage_32_v12 * 128
                stage_32_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_32_v13, 7168])
                stage_32_v15 = pl.tensor.matmul(stage_32_v14, stage_32_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_32_v16 = pl.tensor.assemble(stage_32_v18, stage_32_v15, [stage_32_v13, 7 * 4096 + stage_32_v7])
                stage_32_v17 = pl.yield_(stage_32_v16)
            stage_32_v10 = pl.yield_(stage_32_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_32 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_32)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_33 = 14256 + core_id * 6
    mapped_block_33 = pl.cast(pl.tensor.read(records, [record_33]), pl.INDEX)
    block_33 = pl.max(mapped_block_33, 0)
    if mapped_block_33 >= 0:
        subslot_33 = pl.cast(pl.tensor.read(records, [record_33 + 1]), pl.INDEX)
        done_33 = pl.tensor.read(records, [record_33 + 2])
        wait_base_33 = pl.cast(pl.tensor.read(records, [record_33 + 3]), pl.INDEX)
        wait_count_33 = pl.cast(pl.tensor.read(records, [record_33 + 4]), pl.INDEX)
        publish_33 = pl.tensor.read(records, [record_33 + 5])
        for dependency_33 in pl.range(wait_count_33):
            wait_index_33 = (wait_base_33 + dependency_33) * 2
            other_33 = pl.cast(pl.tensor.read(waits, [wait_index_33]), pl.INDEX)
            required_33 = pl.tensor.read(waits, [wait_index_33 + 1])
            target_33 = generation_base + required_33
            flag_33 = other_33 * 64
            pl.system.cacheinvalid(progress, [1], [flag_33])
            initial_33 = pl.tensor.read(progress, [flag_33])
            for (ready_33,) in pl.while_(init_values=(initial_33,)):
                pl.cond(ready_33 < target_33)
                pl.system.cacheinvalid(progress, [1], [flag_33])
                after_33 = pl.yield_(pl.tensor.read(progress, [flag_33]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_33_v4: pl.Scalar[pl.INDEX] = block_33
        stage_33_v5: pl.Scalar[pl.INDEX] = stage_33_v4 * 512
        for (stage_33_v6, (stage_33_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_33_v7: pl.Scalar[pl.INDEX] = stage_33_v5 + stage_33_v6 * 256
            stage_33_v8: pl.Scalar[pl.INDEX] = pl.max(0, 0) * 1024
            stage_33_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_33_v8, stage_33_v7])
            for (stage_33_v12, (stage_33_v18,)) in pl.range(2, init_values=(stage_33_v11,)):
                stage_33_v13: pl.Scalar[pl.INDEX] = stage_33_v12 * 128
                stage_33_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_33_v13, 0])
                stage_33_v15 = pl.tensor.matmul(stage_33_v14, stage_33_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_33_v16 = pl.tensor.assemble(stage_33_v18, stage_33_v15, [stage_33_v13, 0 * 4096 + stage_33_v7])
                stage_33_v17 = pl.yield_(stage_33_v16)
            stage_33_v10 = pl.yield_(stage_33_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_33 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_33)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_34 = 14688 + core_id * 6
    mapped_block_34 = pl.cast(pl.tensor.read(records, [record_34]), pl.INDEX)
    block_34 = pl.max(mapped_block_34, 0)
    if mapped_block_34 >= 0:
        subslot_34 = pl.cast(pl.tensor.read(records, [record_34 + 1]), pl.INDEX)
        done_34 = pl.tensor.read(records, [record_34 + 2])
        wait_base_34 = pl.cast(pl.tensor.read(records, [record_34 + 3]), pl.INDEX)
        wait_count_34 = pl.cast(pl.tensor.read(records, [record_34 + 4]), pl.INDEX)
        publish_34 = pl.tensor.read(records, [record_34 + 5])
        for dependency_34 in pl.range(wait_count_34):
            wait_index_34 = (wait_base_34 + dependency_34) * 2
            other_34 = pl.cast(pl.tensor.read(waits, [wait_index_34]), pl.INDEX)
            required_34 = pl.tensor.read(waits, [wait_index_34 + 1])
            target_34 = generation_base + required_34
            flag_34 = other_34 * 64
            pl.system.cacheinvalid(progress, [1], [flag_34])
            initial_34 = pl.tensor.read(progress, [flag_34])
            for (ready_34,) in pl.while_(init_values=(initial_34,)):
                pl.cond(ready_34 < target_34)
                pl.system.cacheinvalid(progress, [1], [flag_34])
                after_34 = pl.yield_(pl.tensor.read(progress, [flag_34]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_34_v4: pl.Scalar[pl.INDEX] = block_34
        stage_34_v5: pl.Scalar[pl.INDEX] = stage_34_v4 * 512
        for (stage_34_v6, (stage_34_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_34_v7: pl.Scalar[pl.INDEX] = stage_34_v5 + stage_34_v6 * 256
            stage_34_v8: pl.Scalar[pl.INDEX] = pl.max(3, 0) * 1024
            stage_34_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_34_v8, stage_34_v7])
            for (stage_34_v12, (stage_34_v18,)) in pl.range(2, init_values=(stage_34_v11,)):
                stage_34_v13: pl.Scalar[pl.INDEX] = stage_34_v12 * 128
                stage_34_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_34_v13, 3072])
                stage_34_v15 = pl.tensor.matmul(stage_34_v14, stage_34_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_34_v16 = pl.tensor.assemble(stage_34_v18, stage_34_v15, [stage_34_v13, 3 * 4096 + stage_34_v7])
                stage_34_v17 = pl.yield_(stage_34_v16)
            stage_34_v10 = pl.yield_(stage_34_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_34 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_34)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_35 = 15120 + core_id * 6
    mapped_block_35 = pl.cast(pl.tensor.read(records, [record_35]), pl.INDEX)
    block_35 = pl.max(mapped_block_35, 0)
    if mapped_block_35 >= 0:
        subslot_35 = pl.cast(pl.tensor.read(records, [record_35 + 1]), pl.INDEX)
        done_35 = pl.tensor.read(records, [record_35 + 2])
        wait_base_35 = pl.cast(pl.tensor.read(records, [record_35 + 3]), pl.INDEX)
        wait_count_35 = pl.cast(pl.tensor.read(records, [record_35 + 4]), pl.INDEX)
        publish_35 = pl.tensor.read(records, [record_35 + 5])
        for dependency_35 in pl.range(wait_count_35):
            wait_index_35 = (wait_base_35 + dependency_35) * 2
            other_35 = pl.cast(pl.tensor.read(waits, [wait_index_35]), pl.INDEX)
            required_35 = pl.tensor.read(waits, [wait_index_35 + 1])
            target_35 = generation_base + required_35
            flag_35 = other_35 * 64
            pl.system.cacheinvalid(progress, [1], [flag_35])
            initial_35 = pl.tensor.read(progress, [flag_35])
            for (ready_35,) in pl.while_(init_values=(initial_35,)):
                pl.cond(ready_35 < target_35)
                pl.system.cacheinvalid(progress, [1], [flag_35])
                after_35 = pl.yield_(pl.tensor.read(progress, [flag_35]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_35_v4: pl.Scalar[pl.INDEX] = block_35
        stage_35_v5: pl.Scalar[pl.INDEX] = stage_35_v4 * 512
        for (stage_35_v6, (stage_35_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_35_v7: pl.Scalar[pl.INDEX] = stage_35_v5 + stage_35_v6 * 256
            stage_35_v8: pl.Scalar[pl.INDEX] = pl.max(1, 0) * 1024
            stage_35_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_35_v8, stage_35_v7])
            for (stage_35_v12, (stage_35_v18,)) in pl.range(2, init_values=(stage_35_v11,)):
                stage_35_v13: pl.Scalar[pl.INDEX] = stage_35_v12 * 128
                stage_35_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_35_v13, 1024])
                stage_35_v15 = pl.tensor.matmul(stage_35_v14, stage_35_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_35_v16 = pl.tensor.assemble(stage_35_v18, stage_35_v15, [stage_35_v13, 1 * 4096 + stage_35_v7])
                stage_35_v17 = pl.yield_(stage_35_v16)
            stage_35_v10 = pl.yield_(stage_35_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_35 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_35)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_36 = 15552 + core_id * 6
    mapped_block_36 = pl.cast(pl.tensor.read(records, [record_36]), pl.INDEX)
    block_36 = pl.max(mapped_block_36, 0)
    if mapped_block_36 >= 0:
        subslot_36 = pl.cast(pl.tensor.read(records, [record_36 + 1]), pl.INDEX)
        done_36 = pl.tensor.read(records, [record_36 + 2])
        wait_base_36 = pl.cast(pl.tensor.read(records, [record_36 + 3]), pl.INDEX)
        wait_count_36 = pl.cast(pl.tensor.read(records, [record_36 + 4]), pl.INDEX)
        publish_36 = pl.tensor.read(records, [record_36 + 5])
        for dependency_36 in pl.range(wait_count_36):
            wait_index_36 = (wait_base_36 + dependency_36) * 2
            other_36 = pl.cast(pl.tensor.read(waits, [wait_index_36]), pl.INDEX)
            required_36 = pl.tensor.read(waits, [wait_index_36 + 1])
            target_36 = generation_base + required_36
            flag_36 = other_36 * 64
            pl.system.cacheinvalid(progress, [1], [flag_36])
            initial_36 = pl.tensor.read(progress, [flag_36])
            for (ready_36,) in pl.while_(init_values=(initial_36,)):
                pl.cond(ready_36 < target_36)
                pl.system.cacheinvalid(progress, [1], [flag_36])
                after_36 = pl.yield_(pl.tensor.read(progress, [flag_36]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_36_v4: pl.Scalar[pl.INDEX] = block_36
        stage_36_v5: pl.Scalar[pl.INDEX] = stage_36_v4 * 512
        for (stage_36_v6, (stage_36_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_36_v7: pl.Scalar[pl.INDEX] = stage_36_v5 + stage_36_v6 * 256
            stage_36_v8: pl.Scalar[pl.INDEX] = pl.max(3, 0) * 1024
            stage_36_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_36_v8, stage_36_v7])
            for (stage_36_v12, (stage_36_v18,)) in pl.range(2, init_values=(stage_36_v11,)):
                stage_36_v13: pl.Scalar[pl.INDEX] = stage_36_v12 * 128
                stage_36_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_36_v13, 3072])
                stage_36_v15 = pl.tensor.matmul(stage_36_v14, stage_36_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_36_v16 = pl.tensor.assemble(stage_36_v18, stage_36_v15, [stage_36_v13, 3 * 4096 + stage_36_v7])
                stage_36_v17 = pl.yield_(stage_36_v16)
            stage_36_v10 = pl.yield_(stage_36_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_36 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_36)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    return (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_129, progress)

@pl.jit.incore
def csa_static_aiv(temporary_4: pl.InOut[pl.Tensor[[144, 1], pl.FP32]], temporary_5: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]], view_2: pl.InOut[pl.Tensor[[144, 16384], pl.FP32]], view_3: pl.Tensor[[144, 16384], pl.BF16], temporary_7: pl.InOut[pl.Tensor[[576, 32], pl.FP32]], arg_hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], temporary_8: pl.InOut[pl.Tensor[[144, 32], pl.FP32]], arg_hc_attn_scale: pl.Tensor[[3], pl.FP32], temporary_0: pl.InOut[pl.Tensor[[144, 4], pl.FP32]], temporary_9: pl.InOut[pl.Tensor[[8, 8], pl.FP32]], temporary_6: pl.InOut[pl.Tensor[[144, 8], pl.FP32]], arg_hc_attn_base: pl.Tensor[[24], pl.FP32], view_12: pl.Tensor[[1, 24], pl.FP32], temporary_1: pl.InOut[pl.Tensor[[144, 16], pl.FP32]], temporary_10: pl.InOut[pl.Tensor[[8, 32], pl.FP32]], temporary_11: pl.InOut[pl.Tensor[[144, 4096], pl.BF16]], temporary_2: pl.InOut[pl.Tensor[[144, 4096], pl.BF16]], temporary_12: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]], arg_attn_norm_w: pl.Tensor[[4096], pl.BF16], arg_cmp_seq_lens: pl.Tensor[[24], pl.INT32], arg_cmp_query_start_loc: pl.Tensor[[25], pl.INT32], temporary_14: pl.InOut[pl.Tensor[[24], pl.INT32]], arg_kv_seq_lens: pl.Tensor[[24], pl.INT32], arg_idx_query_start_loc: pl.Tensor[[25], pl.INT32], temporary_15: pl.InOut[pl.Tensor[[24], pl.INT32]], arg_freqs_sin: pl.Tensor[[144, 64], pl.FP32], temporary_13: pl.InOut[pl.Tensor[[144, 64], pl.FP32]], temporary_20: pl.InOut[pl.Tensor[[144, 64], pl.FP32]], temporary_21: pl.InOut[pl.Tensor[[144, 64], pl.INT32]], temporary_24: pl.InOut[pl.Tensor[[144, 1024], pl.FP32]], arg_wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ], view_31: pl.Tensor[[1, 1024], pl.BF16], temporary_23: pl.InOut[pl.Tensor[[512, 1], pl.FP32]], temporary_19: pl.InOut[pl.Tensor[[144, 1], pl.FP32]], temporary_22: pl.InOut[pl.Tensor[[512, 1024], pl.INT8]], temporary_18: pl.InOut[pl.Tensor[[144, 1024], pl.INT8]], temporary_25: pl.InOut[pl.Tensor[[256, 32768], pl.INT32]], arg_wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ], arg_freqs_cos: pl.Tensor[[144, 64], pl.FP32], arg_wq_b_scale: pl.Tensor[[32768], pl.FP32], view_40: pl.InOut[pl.Tensor[[144, 32768], pl.BF16]], temporary_26: pl.InOut[pl.Tensor[[144, 512], pl.FP32]], arg_wkv: pl.Tensor[[4096, 512], pl.BF16], temporary_17: pl.InOut[pl.Tensor[[144, 512], pl.BF16]], arg_gamma_ckv: pl.Tensor[[512], pl.BF16], view_45: pl.InOut[pl.Tensor[[6944, 512], pl.BF16]], arg_ori_slot_mapping: pl.Tensor[[144, 2], pl.INT32], temporary_29: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], temporary_30: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], arg_cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], arg_cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], temporary_28: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], arg_position_ids: pl.Tensor[[144], pl.INT64], arg_cmp_ape: pl.Tensor[[4, 1024], pl.FP32], arg_state_block_table: pl.Tensor[[24, 4100], pl.INT32], arg_compress_state: pl.InOut[pl.Tensor[[217, 8192], pl.FP32]], arg_state_slot_mapping: pl.Tensor[[144, 2], pl.INT32], arg_cmp_freqs_cos: pl.Tensor[[60, 64], pl.FP32], arg_cmp_freqs_sin: pl.Tensor[[60, 64], pl.FP32], temporary_31: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], view_60: pl.Tensor[[1, 512], pl.BF16], view_61: pl.InOut[pl.Tensor[[50720, 512], pl.BF16]], temporary_27: pl.InOut[pl.Tensor[[144, 512], pl.FP32]], arg_cmp_slot_mapping: pl.Tensor[[60, 2], pl.INT32], temporary_34: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], temporary_35: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], arg_inner_wkv: pl.Tensor[[256, 4096], pl.BF16], arg_inner_wgate: pl.Tensor[[256, 4096], pl.BF16], temporary_36: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_inner_ape: pl.Tensor[[4, 256], pl.FP32], arg_inner_state_block_table: pl.Tensor[[24, 4100], pl.INT32], arg_inner_compress_state: pl.InOut[pl.Tensor[[217, 1040], pl.FP32]], arg_inner_state_slot_mapping: pl.Tensor[[144, 2], pl.INT32], temporary_33: pl.InOut[pl.Tensor[[384, 128], pl.BF16]], arg_inner_freqs_cos: pl.Tensor[[60, 64], pl.FP32], arg_inner_freqs_sin: pl.Tensor[[60, 64], pl.FP32], view_76: pl.Tensor[[1, 128], pl.BF16], temporary_38: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_hadamard_idx: pl.Tensor[[128, 128], pl.BF16], temporary_37: pl.InOut[pl.Tensor[[384, 1], pl.FP32]], arg_idx_native_kv_cache: pl.InOut[pl.Tensor[[1585, 4160], pl.INT8]], temporary_32: pl.InOut[pl.Tensor[[144, 128], pl.FP32]], arg_idx_slot_mapping: pl.Tensor[[60, 2], pl.INT32], temporary_42: pl.InOut[pl.Tensor[[384, 8192], pl.INT32]], arg_idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], view_85: pl.InOut[pl.Tensor[[384, 8192], pl.BF16]], arg_idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], temporary_43: pl.InOut[pl.Tensor[[9216, 128], pl.FP32]], temporary_41: pl.InOut[pl.Tensor[[24576, 128], pl.BF16]], temporary_39: pl.InOut[pl.Tensor[[24576, 128], pl.INT8]], temporary_40: pl.InOut[pl.Tensor[[24576, 1], pl.FP32]], temporary_45: pl.InOut[pl.Tensor[[1536, 64], pl.FP32]], arg_weights_proj: pl.Tensor[[4096, 64], pl.BF16], temporary_44: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], view_94: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_48: pl.InOut[pl.Tensor[[1024, 384], pl.FP16]], temporary_50: pl.InOut[pl.Tensor[[256], pl.INT64]], arg_idx_block_table: pl.Tensor[[24, 66], pl.INT32], view_98: pl.InOut[pl.Tensor[[51512, 128], pl.INT8]], view_99: pl.InOut[pl.Tensor[[51512, 128], pl.INT8]], temporary_49: pl.InOut[pl.Tensor[[288, 768], pl.FP32]], temporary_47: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_46: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32]], arg_idx_topk_scores: pl.InOut[pl.Tensor[[144, 512], pl.FP32]], arg_idx_topk: pl.InOut[pl.Tensor[[144, 512], pl.INT32]], temporary_53: pl.InOut[pl.Tensor[[144, 512], pl.INT32]], temporary_52: pl.InOut[pl.Tensor[[144, 640], pl.FP32]], view_107: pl.Tensor[[144, 1], pl.INT64], temporary_54: pl.InOut[pl.Tensor[[144, 16], pl.INT32]], arg_ori_block_table: pl.Tensor[[24, 258], pl.INT32], temporary_55: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_62: pl.InOut[pl.Tensor[[256], pl.INT64]], view_112: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_56: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_57: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], temporary_58: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], temporary_59: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], view_117: pl.Tensor[[64, 1], pl.FP32], arg_cmp_block_table: pl.Tensor[[24, 66], pl.INT32], temporary_60: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_61: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], arg_wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ], temporary_63: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_65: pl.InOut[pl.Tensor[[8, 384], pl.FP32]], temporary_64: pl.InOut[pl.Tensor[[384, 8192], pl.INT8]], arg_wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ], temporary_66: pl.InOut[pl.Tensor[[384, 32768], pl.INT32]], arg_wo_b_scale: pl.Tensor[[4096], pl.FP32], view_129: pl.InOut[pl.Tensor[[144, 16384], pl.BF16]], progress: pl.InOut[pl.Tensor[[4608], pl.INT64]], records: pl.Tensor[[34128], pl.INT64], waits: pl.Tensor[[38978], pl.INT64]) -> tuple[pl.Tensor[[144, 1], pl.FP32], pl.Tensor[[8, 16384], pl.FP32], pl.Tensor[[144, 16384], pl.FP32], pl.Tensor[[576, 32], pl.FP32], pl.Tensor[[144, 32], pl.FP32], pl.Tensor[[144, 4], pl.FP32], pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[144, 8], pl.FP32], pl.Tensor[[144, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32], pl.Tensor[[144, 4096], pl.BF16], pl.Tensor[[144, 4096], pl.BF16], pl.Tensor[[8, 4096], pl.BF16], pl.Tensor[[24], pl.INT32], pl.Tensor[[24], pl.INT32], pl.Tensor[[144, 64], pl.FP32], pl.Tensor[[144, 64], pl.FP32], pl.Tensor[[144, 64], pl.INT32], pl.Tensor[[144, 1024], pl.FP32], pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[144, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[144, 1024], pl.INT8], pl.Tensor[[256, 32768], pl.INT32], pl.Tensor[[144, 32768], pl.BF16], pl.Tensor[[144, 512], pl.FP32], pl.Tensor[[144, 512], pl.BF16], pl.Tensor[[6944, 512], pl.BF16], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[217, 8192], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[50720, 512], pl.BF16], pl.Tensor[[144, 512], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[217, 1040], pl.FP32], pl.Tensor[[384, 128], pl.BF16], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[1585, 4160], pl.INT8], pl.Tensor[[144, 128], pl.FP32], pl.Tensor[[384, 8192], pl.INT32], pl.Tensor[[384, 8192], pl.BF16], pl.Tensor[[9216, 128], pl.FP32], pl.Tensor[[24576, 128], pl.BF16], pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Tensor[[1536, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[1024, 384], pl.FP16], pl.Tensor[[256], pl.INT64], pl.Tensor[[51512, 128], pl.INT8], pl.Tensor[[51512, 128], pl.INT8], pl.Tensor[[288, 768], pl.FP32], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Tensor[[144, 512], pl.FP32], pl.Tensor[[144, 512], pl.INT32], pl.Tensor[[144, 512], pl.INT32], pl.Tensor[[144, 640], pl.FP32], pl.Tensor[[144, 16], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[256], pl.INT64], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8], pl.Tensor[[384, 32768], pl.INT32], pl.Tensor[[144, 16384], pl.BF16], pl.Tensor[[4608], pl.INT64]]:
    pl.func_attr({'mx_tensor_views_blocked': True, 'split_aiv': True})
    group_id = pl.tile.get_block_idx()
    lane_id = pl.tile.get_subblock_idx()
    core_id = 24 + group_id * 2 + lane_id
    own_flag = core_id * 64
    pl.system.cacheinvalid(progress, [1], [own_flag])
    prior_progress = pl.tensor.read(progress, [own_flag])
    generation_base = (prior_progress // 1024 + 1) * 1024
    pl.tensor.write(progress, [own_flag], generation_base)
    pl.system.cacheinvalid(progress, [1], [own_flag])
    pl.system.fence()
    record_0 = 15984 + core_id * 6
    mapped_block_0 = pl.cast(pl.tensor.read(records, [record_0]), pl.INDEX)
    block_0 = pl.max(mapped_block_0, 0)
    if mapped_block_0 >= 0:
        subslot_0 = pl.cast(pl.tensor.read(records, [record_0 + 1]), pl.INDEX)
        done_0 = pl.tensor.read(records, [record_0 + 2])
        wait_base_0 = pl.cast(pl.tensor.read(records, [record_0 + 3]), pl.INDEX)
        wait_count_0 = pl.cast(pl.tensor.read(records, [record_0 + 4]), pl.INDEX)
        publish_0 = pl.tensor.read(records, [record_0 + 5])
        for dependency_0 in pl.range(wait_count_0):
            wait_index_0 = (wait_base_0 + dependency_0) * 2
            other_0 = pl.cast(pl.tensor.read(waits, [wait_index_0]), pl.INDEX)
            required_0 = pl.tensor.read(waits, [wait_index_0 + 1])
            target_0 = generation_base + required_0
            flag_0 = other_0 * 64
            pl.system.cacheinvalid(progress, [1], [flag_0])
            initial_0 = pl.tensor.read(progress, [flag_0])
            for (ready_0,) in pl.while_(init_values=(initial_0,)):
                pl.cond(ready_0 < target_0)
                pl.system.cacheinvalid(progress, [1], [flag_0])
                after_0 = pl.yield_(pl.tensor.read(progress, [flag_0]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_0_v5: pl.Scalar[pl.INDEX] = block_0
        for (stage_0_v6, (stage_0_v16, stage_0_v17, stage_0_v18)) in pl.range(stage_0_v5, 18, pl.min(18, 48), init_values=(temporary_4, temporary_5, view_2)):
            stage_0_v7: pl.Scalar[pl.INDEX] = stage_0_v6 * 8
            stage_0_v8: pl.Scalar[pl.INDEX] = pl.min(144 - stage_0_v7, 8)
            stage_0_v9: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
            for (stage_0_v19, (stage_0_v26, stage_0_v27, stage_0_v28)) in pl.pipeline(32, stage=4, init_values=(stage_0_v9, stage_0_v17, stage_0_v18)):
                stage_0_v20: pl.Scalar[pl.INDEX] = stage_0_v19 * 512
                stage_0_v21: pl.Tile[[8, 512], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_0_v8, 512])] = pl.tile.load(view_3, [stage_0_v7, stage_0_v20], [8, 512], [stage_0_v8, 512], target_memory=pl.Mem.Vec)
                stage_0_v22: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_0_v8, 512])] = pl.tile.cast(stage_0_v21, target_type=pl.FP32, mode='round')
                if stage_0_v8 == 8:
                    stage_0_v29 = pl.tile.store(stage_0_v22, [stage_0_v7, stage_0_v20], stage_0_v28)
                    stage_0_v30: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_0_v8, 512])] = pl.tile.mul(stage_0_v22, stage_0_v22)
                    stage_0_v31: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_0_v32: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_0_v8, 1])] = pl.tile.row_sum(stage_0_v30, stage_0_v31)
                    stage_0_v33: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_0_v8])] = pl.tile.reshape(stage_0_v32, [1, 8])
                    stage_0_v34: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_0_v26, stage_0_v33)
                    (stage_0_v47, stage_0_v48, stage_0_v49) = pl.yield_(stage_0_v34, stage_0_v27, stage_0_v29)
                else:
                    stage_0_v35: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_0_v8, 512])] = pl.tile.set_validshape(stage_0_v22, stage_0_v8, 512)
                    stage_0_v36: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_0_v35, pad_value=pl.PadValue.zero)
                    stage_0_v37: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.mul(stage_0_v36, stage_0_v36)
                    stage_0_v38: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_0_v39: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_0_v37, stage_0_v38)
                    stage_0_v40: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_0_v39, [1, 8])
                    stage_0_v41: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_0_v26, stage_0_v40)
                    stage_0_v42 = pl.tile.store(stage_0_v22, [0, stage_0_v20], stage_0_v27)
                    stage_0_v43: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_0_v8, 512])] = pl.tile.load(stage_0_v42, [0, stage_0_v20], [8, 512], [stage_0_v8, 512], target_memory=pl.Mem.Vec)
                    pl.tile.store(stage_0_v43, [stage_0_v7, stage_0_v20], stage_0_v28)
                    (stage_0_v47, stage_0_v48, stage_0_v49) = pl.yield_(stage_0_v41, stage_0_v42, stage_0_v28)
                (stage_0_v44, stage_0_v45, stage_0_v46) = pl.yield_(stage_0_v47, stage_0_v48, stage_0_v49)
            stage_0_v10: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_0_v44, 6.103515625e-05)
            stage_0_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_0_v10, 1e-06)
            stage_0_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_0_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(stage_0_v11, stage_0_v12)
            stage_0_v14: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_0_v13, [8, 1])
            stage_0_v15 = pl.tile.store(stage_0_v14, [stage_0_v7, 0], stage_0_v16)
            (stage_0_v23, stage_0_v24, stage_0_v25) = pl.yield_(stage_0_v15, stage_0_v45, stage_0_v46)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_0 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_0)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_1 = 16416 + core_id * 6
    mapped_block_1 = pl.cast(pl.tensor.read(records, [record_1]), pl.INDEX)
    block_1 = pl.max(mapped_block_1, 0)
    if mapped_block_1 >= 0:
        subslot_1 = pl.cast(pl.tensor.read(records, [record_1 + 1]), pl.INDEX)
        done_1 = pl.tensor.read(records, [record_1 + 2])
        wait_base_1 = pl.cast(pl.tensor.read(records, [record_1 + 3]), pl.INDEX)
        wait_count_1 = pl.cast(pl.tensor.read(records, [record_1 + 4]), pl.INDEX)
        publish_1 = pl.tensor.read(records, [record_1 + 5])
        for dependency_1 in pl.range(wait_count_1):
            wait_index_1 = (wait_base_1 + dependency_1) * 2
            other_1 = pl.cast(pl.tensor.read(waits, [wait_index_1]), pl.INDEX)
            required_1 = pl.tensor.read(waits, [wait_index_1 + 1])
            target_1 = generation_base + required_1
            flag_1 = other_1 * 64
            pl.system.cacheinvalid(progress, [1], [flag_1])
            initial_1 = pl.tensor.read(progress, [flag_1])
            for (ready_1,) in pl.while_(init_values=(initial_1,)):
                pl.cond(ready_1 < target_1)
                pl.system.cacheinvalid(progress, [1], [flag_1])
                after_1 = pl.yield_(pl.tensor.read(progress, [flag_1]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_1_v6: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        stage_1_v7: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_cmp_seq_lens, 0)
        for (stage_1_v10, (stage_1_v24,)) in pl.range(stage_1_v7, init_values=(stage_1_v6,)):
            stage_1_v11: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_query_start_loc, [stage_1_v10])
            stage_1_v12: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_query_start_loc, [stage_1_v10 + 1])
            stage_1_v13: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_seq_lens, [stage_1_v10])
            stage_1_v14: pl.Scalar[pl.INT32] = stage_1_v13 + stage_1_v11 - stage_1_v12
            pl.tensor.write(temporary_14, [stage_1_v10], pl.cast(stage_1_v24 - pl.cast(stage_1_v14, pl.INDEX) // 4 - 1, pl.INT32))
            stage_1_v15: pl.Scalar[pl.INDEX] = stage_1_v24 + pl.cast(stage_1_v13, pl.INDEX) // 4 - pl.cast(stage_1_v14, pl.INDEX) // 4
            stage_1_v16: pl.Scalar[pl.INDEX] = pl.yield_(stage_1_v15)
        stage_1_v8: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        stage_1_v9: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_kv_seq_lens, 0)
        for (stage_1_v17, (stage_1_v25,)) in pl.range(stage_1_v9, init_values=(stage_1_v8,)):
            stage_1_v18: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_query_start_loc, [stage_1_v17])
            stage_1_v19: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_query_start_loc, [stage_1_v17 + 1])
            stage_1_v20: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_1_v17])
            stage_1_v21: pl.Scalar[pl.INT32] = stage_1_v20 + stage_1_v18 - stage_1_v19
            pl.tensor.write(temporary_15, [stage_1_v17], pl.cast(stage_1_v25 - pl.cast(stage_1_v21, pl.INDEX) // 4 - 1, pl.INT32))
            stage_1_v22: pl.Scalar[pl.INDEX] = stage_1_v25 + pl.cast(stage_1_v20, pl.INDEX) // 4 - pl.cast(stage_1_v21, pl.INDEX) // 4
            stage_1_v23: pl.Scalar[pl.INDEX] = pl.yield_(stage_1_v22)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_1 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_1)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_2 = 16848 + core_id * 6
    mapped_block_2 = pl.cast(pl.tensor.read(records, [record_2]), pl.INDEX)
    block_2 = pl.max(mapped_block_2, 0)
    if mapped_block_2 >= 0:
        subslot_2 = pl.cast(pl.tensor.read(records, [record_2 + 1]), pl.INDEX)
        done_2 = pl.tensor.read(records, [record_2 + 2])
        wait_base_2 = pl.cast(pl.tensor.read(records, [record_2 + 3]), pl.INDEX)
        wait_count_2 = pl.cast(pl.tensor.read(records, [record_2 + 4]), pl.INDEX)
        publish_2 = pl.tensor.read(records, [record_2 + 5])
        for dependency_2 in pl.range(wait_count_2):
            wait_index_2 = (wait_base_2 + dependency_2) * 2
            other_2 = pl.cast(pl.tensor.read(waits, [wait_index_2]), pl.INDEX)
            required_2 = pl.tensor.read(waits, [wait_index_2 + 1])
            target_2 = generation_base + required_2
            flag_2 = other_2 * 64
            pl.system.cacheinvalid(progress, [1], [flag_2])
            initial_2 = pl.tensor.read(progress, [flag_2])
            for (ready_2,) in pl.while_(init_values=(initial_2,)):
                pl.cond(ready_2 < target_2)
                pl.system.cacheinvalid(progress, [1], [flag_2])
                after_2 = pl.yield_(pl.tensor.read(progress, [flag_2]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_2_v4: pl.Scalar[pl.INDEX] = block_2
        stage_2_v5: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        stage_2_v6: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_2_v7: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_2_v6, target_type=pl.FP32, mode='round')
        stage_2_v8: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_2_v5, stage_2_v7)
        stage_2_v9: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_2_v8, 0.5)
        stage_2_v10: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_2_v9, target_type=pl.INT32, mode='trunc')
        stage_2_v11: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_2_v10, target_type=pl.FP32, mode='round')
        stage_2_v12: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_2_v11, 2.0)
        stage_2_v13: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_2_v8, stage_2_v12)
        stage_2_v14: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_2_v8, 1.0)
        stage_2_v15: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_2_v13, 2.0)
        stage_2_v16: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_2_v14, stage_2_v15)
        stage_2_v17: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_2_v16, target_type=pl.INT32, mode='round')
        stage_2_v18: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_2_v13, 2.0)
        stage_2_v19: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_2_v18, 1.0)
        for (stage_2_v20, (stage_2_v23, stage_2_v24)) in pl.range(stage_2_v4, 18, pl.min(18, 48), init_values=(temporary_20, temporary_21)):
            stage_2_v21: pl.Scalar[pl.INDEX] = stage_2_v20 * 8
            stage_2_v22: pl.Scalar[pl.INDEX] = pl.min(144 - stage_2_v21, 8)
            if stage_2_v22 == 8:
                stage_2_v25: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_sin, [stage_2_v21, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_2_v26: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_2_v25, stage_2_v19)
                stage_2_v27 = pl.tile.store(stage_2_v26, [stage_2_v21, 0], stage_2_v23)
                stage_2_v28 = pl.tile.store(stage_2_v17, [stage_2_v21, 0], stage_2_v24)
                (stage_2_v51, stage_2_v52) = pl.yield_(stage_2_v27, stage_2_v28)
            else:
                stage_2_v29: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_2_v22, 64])] = pl.tile.load(arg_freqs_sin, [stage_2_v21, 0], [8, 64], [stage_2_v22, 64], target_memory=pl.Mem.Vec)
                stage_2_v30: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                stage_2_v31: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                stage_2_v32: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_2_v31, target_type=pl.FP32, mode='round')
                stage_2_v33: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_2_v30, stage_2_v32)
                stage_2_v34: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_2_v33, 0.5)
                stage_2_v35: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_2_v34, target_type=pl.INT32, mode='trunc')
                stage_2_v36: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_2_v35, target_type=pl.FP32, mode='round')
                stage_2_v37: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_2_v36, 2.0)
                stage_2_v38: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_2_v33, stage_2_v37)
                stage_2_v39: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_2_v33, 1.0)
                stage_2_v40: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_2_v38, 2.0)
                stage_2_v41: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_2_v39, stage_2_v40)
                stage_2_v42: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_2_v38, 2.0)
                stage_2_v43: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_2_v42, 1.0)
                stage_2_v44: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_2_v22, 64])] = pl.tile.set_validshape(stage_2_v43, stage_2_v22, 64)
                stage_2_v45: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_2_v22, 64])] = pl.tile.mul(stage_2_v29, stage_2_v44)
                stage_2_v46: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_2_v22, 64])] = pl.tile.set_validshape(stage_2_v45, stage_2_v22, 64)
                pl.tile.store(stage_2_v46, [stage_2_v21, 0], stage_2_v23)
                stage_2_v47: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_2_v41, target_type=pl.INT32, mode='round')
                stage_2_v48: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_2_v22, 64])] = pl.tile.set_validshape(stage_2_v47, stage_2_v22, 64)
                pl.tile.store(stage_2_v48, [stage_2_v21, 0], stage_2_v24)
                (stage_2_v51, stage_2_v52) = pl.yield_(stage_2_v23, stage_2_v24)
            (stage_2_v49, stage_2_v50) = pl.yield_(stage_2_v51, stage_2_v52)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_2 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_2)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_3 = 17280 + core_id * 6
    mapped_block_3 = pl.cast(pl.tensor.read(records, [record_3]), pl.INDEX)
    block_3 = pl.max(mapped_block_3, 0)
    if mapped_block_3 >= 0:
        subslot_3 = pl.cast(pl.tensor.read(records, [record_3 + 1]), pl.INDEX)
        done_3 = pl.tensor.read(records, [record_3 + 2])
        wait_base_3 = pl.cast(pl.tensor.read(records, [record_3 + 3]), pl.INDEX)
        wait_count_3 = pl.cast(pl.tensor.read(records, [record_3 + 4]), pl.INDEX)
        publish_3 = pl.tensor.read(records, [record_3 + 5])
        for dependency_3 in pl.range(wait_count_3):
            wait_index_3 = (wait_base_3 + dependency_3) * 2
            other_3 = pl.cast(pl.tensor.read(waits, [wait_index_3]), pl.INDEX)
            required_3 = pl.tensor.read(waits, [wait_index_3 + 1])
            target_3 = generation_base + required_3
            flag_3 = other_3 * 64
            pl.system.cacheinvalid(progress, [1], [flag_3])
            initial_3 = pl.tensor.read(progress, [flag_3])
            for (ready_3,) in pl.while_(init_values=(initial_3,)):
                pl.cond(ready_3 < target_3)
                pl.system.cacheinvalid(progress, [1], [flag_3])
                after_3 = pl.yield_(pl.tensor.read(progress, [flag_3]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_3_v3: pl.Scalar[pl.INDEX] = block_3
        for stage_3_v4 in pl.range(stage_3_v3, 24, pl.min(24, 16)):
            stage_3_v5: pl.Scalar[pl.INDEX] = stage_3_v4 * 6
            stage_3_v6: pl.Scalar[pl.INDEX] = pl.min(144 - stage_3_v5, 6)
            stage_3_v7: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([6, 64], dtype=pl.FP32, value=1.0)
            stage_3_v8: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            stage_3_v9: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_3_v8, target_type=pl.FP32, mode='round')
            stage_3_v10: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_3_v7, stage_3_v9)
            stage_3_v11: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_3_v10, 0.5)
            stage_3_v12: pl.Tile[[6, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_3_v11, target_type=pl.INT32, mode='trunc')
            stage_3_v13: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_3_v12, target_type=pl.FP32, mode='round')
            stage_3_v14: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_3_v13, 2.0)
            stage_3_v15: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_3_v10, stage_3_v14)
            stage_3_v16: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_3_v15, 2.0)
            stage_3_v17: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_3_v16, 1.0)
            stage_3_v18: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_3_v17)
            stage_3_v19: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_3_v6, 64])] = pl.tile.load(arg_freqs_sin, [stage_3_v5, 0], [6, 64], [stage_3_v6, 64], target_memory=pl.Mem.Vec)
            stage_3_v20: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_3_v6, 64])] = pl.tile.set_validshape(stage_3_v18, stage_3_v6, 64)
            stage_3_v21: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_3_v6, 64])] = pl.tile.mul(stage_3_v19, stage_3_v20)
            stage_3_v22 = pl.tile.store(stage_3_v21, [stage_3_v5, 0], temporary_55)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_3 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_3)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_4 = 17712 + core_id * 6
    mapped_block_4 = pl.cast(pl.tensor.read(records, [record_4]), pl.INDEX)
    block_4 = pl.max(mapped_block_4, 0)
    if mapped_block_4 >= 0:
        subslot_4 = pl.cast(pl.tensor.read(records, [record_4 + 1]), pl.INDEX)
        done_4 = pl.tensor.read(records, [record_4 + 2])
        wait_base_4 = pl.cast(pl.tensor.read(records, [record_4 + 3]), pl.INDEX)
        wait_count_4 = pl.cast(pl.tensor.read(records, [record_4 + 4]), pl.INDEX)
        publish_4 = pl.tensor.read(records, [record_4 + 5])
        for dependency_4 in pl.range(wait_count_4):
            wait_index_4 = (wait_base_4 + dependency_4) * 2
            other_4 = pl.cast(pl.tensor.read(waits, [wait_index_4]), pl.INDEX)
            required_4 = pl.tensor.read(waits, [wait_index_4 + 1])
            target_4 = generation_base + required_4
            flag_4 = other_4 * 64
            pl.system.cacheinvalid(progress, [1], [flag_4])
            initial_4 = pl.tensor.read(progress, [flag_4])
            for (ready_4,) in pl.while_(init_values=(initial_4,)):
                pl.cond(ready_4 < target_4)
                pl.system.cacheinvalid(progress, [1], [flag_4])
                after_4 = pl.yield_(pl.tensor.read(progress, [flag_4]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_4_v3: pl.Scalar[pl.INDEX] = block_4
        for stage_4_v4 in pl.range(stage_4_v3, 36, pl.min(36, 16)):
            stage_4_v5: pl.Scalar[pl.INDEX] = stage_4_v4 * 4
            stage_4_v6: pl.Scalar[pl.INDEX] = pl.min(144 - stage_4_v5, 4)
            stage_4_v7: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([4, 64], dtype=pl.FP32, value=1.0)
            stage_4_v8: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            stage_4_v9: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_4_v8, target_type=pl.FP32, mode='round')
            stage_4_v10: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_4_v7, stage_4_v9)
            stage_4_v11: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_4_v10, 0.5)
            stage_4_v12: pl.Tile[[4, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_4_v11, target_type=pl.INT32, mode='trunc')
            stage_4_v13: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_4_v12, target_type=pl.FP32, mode='round')
            stage_4_v14: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_4_v13, 2.0)
            stage_4_v15: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_4_v10, stage_4_v14)
            stage_4_v16: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_4_v15, 2.0)
            stage_4_v17: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_4_v16, 1.0)
            stage_4_v18: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_4_v6, 64])] = pl.tile.load(arg_freqs_sin, [stage_4_v5, 0], [4, 64], [stage_4_v6, 64], target_memory=pl.Mem.Vec)
            stage_4_v19: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_4_v6, 64])] = pl.tile.set_validshape(stage_4_v17, stage_4_v6, 64)
            stage_4_v20: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_4_v6, 64])] = pl.tile.mul(stage_4_v18, stage_4_v19)
            stage_4_v21 = pl.tile.store(stage_4_v20, [stage_4_v5, 0], temporary_13)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_4 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_4)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_5 = 18144 + core_id * 6
    mapped_block_5 = pl.cast(pl.tensor.read(records, [record_5]), pl.INDEX)
    block_5 = pl.max(mapped_block_5, 0)
    if mapped_block_5 >= 0:
        subslot_5 = pl.cast(pl.tensor.read(records, [record_5 + 1]), pl.INDEX)
        done_5 = pl.tensor.read(records, [record_5 + 2])
        wait_base_5 = pl.cast(pl.tensor.read(records, [record_5 + 3]), pl.INDEX)
        wait_count_5 = pl.cast(pl.tensor.read(records, [record_5 + 4]), pl.INDEX)
        publish_5 = pl.tensor.read(records, [record_5 + 5])
        for dependency_5 in pl.range(wait_count_5):
            wait_index_5 = (wait_base_5 + dependency_5) * 2
            other_5 = pl.cast(pl.tensor.read(waits, [wait_index_5]), pl.INDEX)
            required_5 = pl.tensor.read(waits, [wait_index_5 + 1])
            target_5 = generation_base + required_5
            flag_5 = other_5 * 64
            pl.system.cacheinvalid(progress, [1], [flag_5])
            initial_5 = pl.tensor.read(progress, [flag_5])
            for (ready_5,) in pl.while_(init_values=(initial_5,)):
                pl.cond(ready_5 < target_5)
                pl.system.cacheinvalid(progress, [1], [flag_5])
                after_5 = pl.yield_(pl.tensor.read(progress, [flag_5]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_5_v3: pl.Scalar[pl.INDEX] = block_5
        stage_5_v4: pl.Scalar[pl.INDEX] = stage_5_v3 * 16
        stage_5_v5: pl.Tile[[16, 32], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_7, [stage_5_v4, 0], [16, 32], [16, 32], target_memory=pl.Mem.Vec)
        for (stage_5_v7, (stage_5_v12,)) in pl.range(1, 4, init_values=(stage_5_v5,)):
            stage_5_v8: pl.Scalar[pl.INDEX] = stage_5_v7 * 144 + stage_5_v4
            stage_5_v9: pl.Tile[[16, 32], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_7, [stage_5_v8, 0], [16, 32], [16, 32], target_memory=pl.Mem.Vec)
            stage_5_v10: pl.Tile[[16, 32], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_5_v12, stage_5_v9)
            stage_5_v11: pl.Tile[[16, 32], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_5_v10)
        stage_5_v6 = pl.tile.store(stage_5_v11, [stage_5_v4, 0], temporary_8)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_5 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_5)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_6 = 18576 + core_id * 6
    mapped_block_6 = pl.cast(pl.tensor.read(records, [record_6]), pl.INDEX)
    block_6 = pl.max(mapped_block_6, 0)
    if mapped_block_6 >= 0:
        subslot_6 = pl.cast(pl.tensor.read(records, [record_6 + 1]), pl.INDEX)
        done_6 = pl.tensor.read(records, [record_6 + 2])
        wait_base_6 = pl.cast(pl.tensor.read(records, [record_6 + 3]), pl.INDEX)
        wait_count_6 = pl.cast(pl.tensor.read(records, [record_6 + 4]), pl.INDEX)
        publish_6 = pl.tensor.read(records, [record_6 + 5])
        for dependency_6 in pl.range(wait_count_6):
            wait_index_6 = (wait_base_6 + dependency_6) * 2
            other_6 = pl.cast(pl.tensor.read(waits, [wait_index_6]), pl.INDEX)
            required_6 = pl.tensor.read(waits, [wait_index_6 + 1])
            target_6 = generation_base + required_6
            flag_6 = other_6 * 64
            pl.system.cacheinvalid(progress, [1], [flag_6])
            initial_6 = pl.tensor.read(progress, [flag_6])
            for (ready_6,) in pl.while_(init_values=(initial_6,)):
                pl.cond(ready_6 < target_6)
                pl.system.cacheinvalid(progress, [1], [flag_6])
                after_6 = pl.yield_(pl.tensor.read(progress, [flag_6]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_6_v7: pl.Scalar[pl.INDEX] = block_6
        stage_6_v8: pl.Scalar[pl.FP32] = pl.tensor.read(arg_hc_attn_scale, [2])
        stage_6_v9: pl.Scalar[pl.INDEX] = stage_6_v7 * 8
        stage_6_v10: pl.Scalar[pl.INDEX] = pl.min(144 - stage_6_v9, 8)
        stage_6_v11: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 1])] = pl.tile.load(temporary_4, [stage_6_v9, 0], [8, 1], [stage_6_v10, 1], target_memory=pl.Mem.Vec)
        stage_6_v12: pl.Scalar[pl.INDEX] = 8
        stage_6_v13: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.load(temporary_8, [stage_6_v9, stage_6_v12], [8, 8], [stage_6_v10, 4], target_memory=pl.Mem.Vec)
        stage_6_v14: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.load(temporary_8, [stage_6_v9, stage_6_v12 + 4], [8, 8], [stage_6_v10, 4], target_memory=pl.Mem.Vec)
        stage_6_v15: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.load(temporary_8, [stage_6_v9, stage_6_v12 + 8], [8, 8], [stage_6_v10, 4], target_memory=pl.Mem.Vec)
        stage_6_v16: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.load(temporary_8, [stage_6_v9, stage_6_v12 + 12], [8, 8], [stage_6_v10, 4], target_memory=pl.Mem.Vec)
        stage_6_v17: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(view_12, [0, stage_6_v12], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        stage_6_v18: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(view_12, [0, stage_6_v12 + 4], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        stage_6_v19: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(view_12, [0, stage_6_v12 + 8], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        stage_6_v20: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(view_12, [0, stage_6_v12 + 12], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        stage_6_v21: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.row_expand_mul(stage_6_v13, stage_6_v11)
        stage_6_v22: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.muls(stage_6_v21, stage_6_v8)
        stage_6_v23: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.col_expand(stage_6_v13, stage_6_v17)
        stage_6_v24: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.add(stage_6_v22, stage_6_v23)
        stage_6_v25: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.row_expand_mul(stage_6_v14, stage_6_v11)
        stage_6_v26: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.muls(stage_6_v25, stage_6_v8)
        stage_6_v27: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.col_expand(stage_6_v14, stage_6_v18)
        stage_6_v28: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.add(stage_6_v26, stage_6_v27)
        stage_6_v29: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.row_expand_mul(stage_6_v15, stage_6_v11)
        stage_6_v30: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.muls(stage_6_v29, stage_6_v8)
        stage_6_v31: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.col_expand(stage_6_v15, stage_6_v19)
        stage_6_v32: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.add(stage_6_v30, stage_6_v31)
        stage_6_v33: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.row_expand_mul(stage_6_v16, stage_6_v11)
        stage_6_v34: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.muls(stage_6_v33, stage_6_v8)
        stage_6_v35: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.col_expand(stage_6_v16, stage_6_v20)
        stage_6_v36: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.add(stage_6_v34, stage_6_v35)
        stage_6_v37: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_6_v24, pad_value=pl.PadValue.min)
        stage_6_v38: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_6_v28, pad_value=pl.PadValue.min)
        stage_6_v39: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_6_v32, pad_value=pl.PadValue.min)
        stage_6_v40: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_6_v36, pad_value=pl.PadValue.min)
        stage_6_v41: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_6_v42: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_6_v43: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_6_v37, stage_6_v41)
        stage_6_v44: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_6_v38, stage_6_v41)
        stage_6_v45: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_6_v39, stage_6_v41)
        stage_6_v46: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_6_v40, stage_6_v41)
        stage_6_v47: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(stage_6_v37, stage_6_v43)
        stage_6_v48: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(stage_6_v47)
        stage_6_v49: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(stage_6_v38, stage_6_v44)
        stage_6_v50: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(stage_6_v49)
        stage_6_v51: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(stage_6_v39, stage_6_v45)
        stage_6_v52: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(stage_6_v51)
        stage_6_v53: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(stage_6_v40, stage_6_v46)
        stage_6_v54: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(stage_6_v53)
        stage_6_v55: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_6_v48, stage_6_v42)
        stage_6_v56: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_6_v50, stage_6_v42)
        stage_6_v57: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_6_v52, stage_6_v42)
        stage_6_v58: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_6_v54, stage_6_v42)
        stage_6_v59: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(stage_6_v48, stage_6_v55)
        stage_6_v60: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(stage_6_v59, 1e-06)
        stage_6_v61: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(stage_6_v50, stage_6_v56)
        stage_6_v62: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(stage_6_v61, 1e-06)
        stage_6_v63: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(stage_6_v52, stage_6_v57)
        stage_6_v64: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(stage_6_v63, 1e-06)
        stage_6_v65: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(stage_6_v54, stage_6_v58)
        stage_6_v66: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(stage_6_v65, 1e-06)
        stage_6_v67: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_6_v60, 8, 4)
        stage_6_v68: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_6_v62, 8, 4)
        stage_6_v69: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_6_v64, 8, 4)
        stage_6_v70: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_6_v66, 8, 4)
        stage_6_v71: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_6_v67, pad_value=pl.PadValue.zero)
        stage_6_v72: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_6_v68, pad_value=pl.PadValue.zero)
        stage_6_v73: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_6_v69, pad_value=pl.PadValue.zero)
        stage_6_v74: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_6_v70, pad_value=pl.PadValue.zero)
        stage_6_v75: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_6_v76: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_6_v71, stage_6_v72)
        stage_6_v77: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_6_v73, stage_6_v74)
        stage_6_v78: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_6_v76, stage_6_v77)
        stage_6_v79: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(stage_6_v78, 1e-06)
        stage_6_v80: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_6_v71, stage_6_v79)
        stage_6_v81: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_6_v72, stage_6_v79)
        stage_6_v82: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_6_v73, stage_6_v79)
        stage_6_v83: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_6_v74, stage_6_v79)
        for (stage_6_v84, (stage_6_v133, stage_6_v134, stage_6_v135, stage_6_v136)) in pl.pipeline(19, stage=2, init_values=(stage_6_v80, stage_6_v81, stage_6_v82, stage_6_v83)):
            stage_6_v85: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_6_v133, stage_6_v75)
            stage_6_v86: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_6_v85, [1, 8])
            stage_6_v87: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_6_v86, 1e-06)
            stage_6_v88: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_6_v87, [8, 1])
            stage_6_v89: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_6_v134, stage_6_v75)
            stage_6_v90: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_6_v89, [1, 8])
            stage_6_v91: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_6_v90, 1e-06)
            stage_6_v92: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_6_v91, [8, 1])
            stage_6_v93: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_6_v135, stage_6_v75)
            stage_6_v94: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_6_v93, [1, 8])
            stage_6_v95: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_6_v94, 1e-06)
            stage_6_v96: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_6_v95, [8, 1])
            stage_6_v97: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_6_v136, stage_6_v75)
            stage_6_v98: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_6_v97, [1, 8])
            stage_6_v99: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_6_v98, 1e-06)
            stage_6_v100: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_6_v99, [8, 1])
            stage_6_v101: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(stage_6_v133, stage_6_v88)
            stage_6_v102: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(stage_6_v134, stage_6_v92)
            stage_6_v103: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(stage_6_v135, stage_6_v96)
            stage_6_v104: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(stage_6_v136, stage_6_v100)
            stage_6_v105: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_6_v101, stage_6_v102)
            stage_6_v106: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_6_v103, stage_6_v104)
            stage_6_v107: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_6_v105, stage_6_v106)
            stage_6_v108: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(stage_6_v107, 1e-06)
            stage_6_v109: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_6_v101, stage_6_v108)
            stage_6_v110: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_6_v102, stage_6_v108)
            stage_6_v111: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_6_v103, stage_6_v108)
            stage_6_v112: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_6_v104, stage_6_v108)
            (stage_6_v137, stage_6_v138, stage_6_v139, stage_6_v140) = pl.yield_(stage_6_v109, stage_6_v110, stage_6_v111, stage_6_v112)
        if stage_6_v10 == 8:
            stage_6_v113: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(stage_6_v137, 8, 4)
            stage_6_v114: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(stage_6_v138, 8, 4)
            stage_6_v115: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(stage_6_v139, 8, 4)
            stage_6_v116: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(stage_6_v140, 8, 4)
            stage_6_v117 = pl.tile.store(stage_6_v113, [stage_6_v9, 0], temporary_1)
            stage_6_v118 = pl.tile.store(stage_6_v114, [stage_6_v9, 4], temporary_1)
            stage_6_v119 = pl.tile.store(stage_6_v115, [stage_6_v9, 8], temporary_1)
            stage_6_v120 = pl.tile.store(stage_6_v116, [stage_6_v9, 12], temporary_1)
        else:
            stage_6_v121 = pl.tile.store(stage_6_v137, [0, 0], temporary_10)
            stage_6_v122 = pl.tile.store(stage_6_v138, [0, 8], temporary_10)
            stage_6_v123 = pl.tile.store(stage_6_v139, [0, 16], temporary_10)
            stage_6_v124 = pl.tile.store(stage_6_v140, [0, 24], temporary_10)
            stage_6_v125: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.load(temporary_10, [0, 0], [8, 8], [stage_6_v10, 4], target_memory=pl.Mem.Vec)
            stage_6_v126: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.load(temporary_10, [0, 8], [8, 8], [stage_6_v10, 4], target_memory=pl.Mem.Vec)
            stage_6_v127: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.load(temporary_10, [0, 16], [8, 8], [stage_6_v10, 4], target_memory=pl.Mem.Vec)
            stage_6_v128: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v10, 4])] = pl.tile.load(temporary_10, [0, 24], [8, 8], [stage_6_v10, 4], target_memory=pl.Mem.Vec)
            stage_6_v129 = pl.tile.store(stage_6_v125, [stage_6_v9, 0], temporary_1)
            stage_6_v130 = pl.tile.store(stage_6_v126, [stage_6_v9, 4], temporary_1)
            stage_6_v131 = pl.tile.store(stage_6_v127, [stage_6_v9, 8], temporary_1)
            stage_6_v132 = pl.tile.store(stage_6_v128, [stage_6_v9, 12], temporary_1)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_6 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_6)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_7 = 19008 + core_id * 6
    mapped_block_7 = pl.cast(pl.tensor.read(records, [record_7]), pl.INDEX)
    block_7 = pl.max(mapped_block_7, 0)
    if mapped_block_7 >= 0:
        subslot_7 = pl.cast(pl.tensor.read(records, [record_7 + 1]), pl.INDEX)
        done_7 = pl.tensor.read(records, [record_7 + 2])
        wait_base_7 = pl.cast(pl.tensor.read(records, [record_7 + 3]), pl.INDEX)
        wait_count_7 = pl.cast(pl.tensor.read(records, [record_7 + 4]), pl.INDEX)
        publish_7 = pl.tensor.read(records, [record_7 + 5])
        for dependency_7 in pl.range(wait_count_7):
            wait_index_7 = (wait_base_7 + dependency_7) * 2
            other_7 = pl.cast(pl.tensor.read(waits, [wait_index_7]), pl.INDEX)
            required_7 = pl.tensor.read(waits, [wait_index_7 + 1])
            target_7 = generation_base + required_7
            flag_7 = other_7 * 64
            pl.system.cacheinvalid(progress, [1], [flag_7])
            initial_7 = pl.tensor.read(progress, [flag_7])
            for (ready_7,) in pl.while_(init_values=(initial_7,)):
                pl.cond(ready_7 < target_7)
                pl.system.cacheinvalid(progress, [1], [flag_7])
                after_7 = pl.yield_(pl.tensor.read(progress, [flag_7]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_7_v8: pl.Scalar[pl.INDEX] = block_7
        stage_7_v9: pl.Scalar[pl.FP32] = pl.tensor.read(arg_hc_attn_scale, [0])
        stage_7_v10: pl.Scalar[pl.FP32] = pl.tensor.read(arg_hc_attn_scale, [1])
        for (stage_7_v11, (stage_7_v40, stage_7_v41, stage_7_v42)) in pl.range(stage_7_v8, 18, pl.min(18, 16), init_values=(temporary_0, temporary_9, temporary_6)):
            stage_7_v12: pl.Scalar[pl.INDEX] = stage_7_v11 * 8
            stage_7_v13: pl.Scalar[pl.INDEX] = pl.min(144 - stage_7_v12, 8)
            stage_7_v14: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_4, [stage_7_v12, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
            stage_7_v15: pl.Tile[[8], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_hc_attn_base, [0], [8], [8], target_memory=pl.Mem.Vec)
            stage_7_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_7_v15, [1, 8])
            stage_7_v17: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_8, [stage_7_v12, 0], [8, 8], [8, 8], target_memory=pl.Mem.Vec)
            stage_7_v18: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_7_v17, stage_7_v14)
            stage_7_v19: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_7_v18, stage_7_v9)
            stage_7_v20: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand(stage_7_v19, stage_7_v16)
            stage_7_v21: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_7_v19, stage_7_v20)
            stage_7_v22: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_7_v21)
            stage_7_v23: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_7_v22)
            stage_7_v24: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_7_v23, 1.0)
            stage_7_v25: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_7_v24)
            stage_7_v26: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_7_v25, 1e-06)
            stage_7_v27 = pl.tile.store(stage_7_v26, [stage_7_v12, 0], stage_7_v42)
            stage_7_v28: pl.Tile[[8], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_hc_attn_base, [4], [8], [8], target_memory=pl.Mem.Vec)
            stage_7_v29: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_7_v28, [1, 8])
            stage_7_v30: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_8, [stage_7_v12, 4], [8, 8], [8, 8], target_memory=pl.Mem.Vec)
            stage_7_v31: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_7_v30, stage_7_v14)
            stage_7_v32: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_7_v31, stage_7_v10)
            stage_7_v33: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand(stage_7_v32, stage_7_v29)
            stage_7_v34: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_7_v32, stage_7_v33)
            stage_7_v35: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_7_v34)
            stage_7_v36: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_7_v35)
            stage_7_v37: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_7_v36, 1.0)
            stage_7_v38: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_7_v37)
            stage_7_v39: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_7_v38, 2.0)
            if stage_7_v13 == 8:
                stage_7_v43: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4])] = pl.tile.slice(stage_7_v39, [8, 8], [0, 0], [8, 4])
                stage_7_v44 = pl.tile.store(stage_7_v43, [stage_7_v12, 0], stage_7_v40)
                (stage_7_v50, stage_7_v51) = pl.yield_(stage_7_v44, stage_7_v41)
            else:
                stage_7_v45 = pl.tile.store(stage_7_v39, [0, 0], stage_7_v41)
                stage_7_v46: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v13, 4])] = pl.tile.load(stage_7_v45, [0, 0], [8, 8], [stage_7_v13, 4], target_memory=pl.Mem.Vec)
                pl.tile.store(stage_7_v46, [stage_7_v12, 0], stage_7_v40)
                (stage_7_v50, stage_7_v51) = pl.yield_(stage_7_v40, stage_7_v45)
            (stage_7_v47, stage_7_v48, stage_7_v49) = pl.yield_(stage_7_v50, stage_7_v51, stage_7_v27)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_7 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_7)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_8 = 19440 + core_id * 6
    mapped_block_8 = pl.cast(pl.tensor.read(records, [record_8]), pl.INDEX)
    block_8 = pl.max(mapped_block_8, 0)
    if mapped_block_8 >= 0:
        subslot_8 = pl.cast(pl.tensor.read(records, [record_8 + 1]), pl.INDEX)
        done_8 = pl.tensor.read(records, [record_8 + 2])
        wait_base_8 = pl.cast(pl.tensor.read(records, [record_8 + 3]), pl.INDEX)
        wait_count_8 = pl.cast(pl.tensor.read(records, [record_8 + 4]), pl.INDEX)
        publish_8 = pl.tensor.read(records, [record_8 + 5])
        for dependency_8 in pl.range(wait_count_8):
            wait_index_8 = (wait_base_8 + dependency_8) * 2
            other_8 = pl.cast(pl.tensor.read(waits, [wait_index_8]), pl.INDEX)
            required_8 = pl.tensor.read(waits, [wait_index_8 + 1])
            target_8 = generation_base + required_8
            flag_8 = other_8 * 64
            pl.system.cacheinvalid(progress, [1], [flag_8])
            initial_8 = pl.tensor.read(progress, [flag_8])
            for (ready_8,) in pl.while_(init_values=(initial_8,)):
                pl.cond(ready_8 < target_8)
                pl.system.cacheinvalid(progress, [1], [flag_8])
                after_8 = pl.yield_(pl.tensor.read(progress, [flag_8]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_8_v7: pl.Scalar[pl.INDEX] = block_8
        stage_8_v8: pl.Scalar[pl.INDEX] = stage_8_v7 * 8
        stage_8_v9: pl.Scalar[pl.INDEX] = pl.min(144 - stage_8_v8, 8)
        stage_8_v10: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_6, [stage_8_v8, 0], [8, 8], [8, 8], target_memory=pl.Mem.Vec)
        stage_8_v11: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_8_v12: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(stage_8_v10, 0, 1, stage_8_v11)
        stage_8_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_8_v12, [1, 8], [0, 0])
        stage_8_v14: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_8_v13, [8, 1])
        stage_8_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_8_v12, [1, 8], [1, 0])
        stage_8_v16: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_8_v15, [8, 1])
        stage_8_v17: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_8_v12, [1, 8], [2, 0])
        stage_8_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_8_v17, [8, 1])
        stage_8_v19: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_8_v12, [1, 8], [3, 0])
        stage_8_v20: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_8_v19, [8, 1])
        stage_8_v21: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for (stage_8_v31, (stage_8_v64, stage_8_v65)) in pl.pipeline(16, stage=2, init_values=(stage_8_v21, temporary_11)):
            stage_8_v32: pl.Scalar[pl.INDEX] = stage_8_v31 * 256
            stage_8_v33: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.load(view_2, [stage_8_v8, stage_8_v32], [8, 256], [stage_8_v9, 256], target_memory=pl.Mem.Vec)
            stage_8_v34: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.load(view_2, [stage_8_v8, stage_8_v32 + 4096], [8, 256], [stage_8_v9, 256], target_memory=pl.Mem.Vec)
            stage_8_v35: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.load(view_2, [stage_8_v8, stage_8_v32 + 8192], [8, 256], [stage_8_v9, 256], target_memory=pl.Mem.Vec)
            stage_8_v36: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.load(view_2, [stage_8_v8, stage_8_v32 + 12288], [8, 256], [stage_8_v9, 256], target_memory=pl.Mem.Vec)
            stage_8_v37: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.row_expand_mul(stage_8_v33, stage_8_v14)
            stage_8_v38: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.row_expand_mul(stage_8_v34, stage_8_v16)
            stage_8_v39: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.row_expand_mul(stage_8_v35, stage_8_v18)
            stage_8_v40: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.row_expand_mul(stage_8_v36, stage_8_v20)
            stage_8_v41: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.add(stage_8_v37, stage_8_v38)
            stage_8_v42: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.add(stage_8_v39, stage_8_v40)
            stage_8_v43: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.add(stage_8_v41, stage_8_v42)
            stage_8_v44: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.cast(stage_8_v43, target_type=pl.BF16, mode='rint')
            stage_8_v45 = pl.tile.store(stage_8_v44, [stage_8_v8, stage_8_v32], stage_8_v65)
            stage_8_v46: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.cast(stage_8_v44, target_type=pl.FP32, mode='round')
            stage_8_v47: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.mul(stage_8_v46, stage_8_v46)
            stage_8_v48: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_8_v49: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 1])] = pl.tile.row_sum(stage_8_v47, stage_8_v48)
            stage_8_v50: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_8_v9])] = pl.tile.reshape(stage_8_v49, [1, 8])
            stage_8_v51: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_8_v64, stage_8_v50)
            (stage_8_v66, stage_8_v67) = pl.yield_(stage_8_v51, stage_8_v45)
        stage_8_v22: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_8_v23: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_8_v22, stage_8_v66, [0, 0])
        stage_8_v24: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_8_v25: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_8_v23, 0.000244140625)
        stage_8_v26: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_8_v25, 1e-06)
        stage_8_v27: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_8_v28: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(stage_8_v26, stage_8_v27)
        stage_8_v29: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = stage_8_v28
        stage_8_v30: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_8_v24, stage_8_v29, [0, 0])
        for (stage_8_v52, (stage_8_v68, stage_8_v69)) in pl.pipeline(16, stage=2, init_values=(temporary_2, temporary_12)):
            stage_8_v53: pl.Scalar[pl.INDEX] = stage_8_v52 * 256
            stage_8_v54: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec] = pl.tile.load(stage_8_v67, [stage_8_v8, stage_8_v53], [8, 256], [8, 256], target_memory=pl.Mem.Vec)
            stage_8_v55: pl.Tile[[256], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_attn_norm_w, [stage_8_v53], [256], [256], target_memory=pl.Mem.Vec)
            stage_8_v56: pl.Tile[[1, 256], pl.BF16, pl.Mem.Vec] = pl.tile.reshape(stage_8_v55, [1, 256])
            stage_8_v57: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_8_v54, target_type=pl.FP32, mode='round')
            stage_8_v58: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_8_v30, [8, 1])
            stage_8_v59: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_8_v56, target_type=pl.FP32, mode='round')
            stage_8_v60: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_8_v57, stage_8_v58)
            stage_8_v61: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_8_v60, stage_8_v59)
            stage_8_v62: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_8_v61, target_type=pl.BF16, mode='rint')
            stage_8_v63: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec] = stage_8_v62
            if stage_8_v9 == 8:
                stage_8_v70 = pl.tile.store(stage_8_v63, [stage_8_v8, stage_8_v53], stage_8_v68)
                (stage_8_v75, stage_8_v76) = pl.yield_(stage_8_v70, stage_8_v69)
            else:
                stage_8_v71 = pl.tile.store(stage_8_v63, [0, stage_8_v53], stage_8_v69)
                stage_8_v72: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v9, 256])] = pl.tile.load(stage_8_v71, [0, stage_8_v53], [8, 256], [stage_8_v9, 256], target_memory=pl.Mem.Vec)
                pl.tile.store(stage_8_v72, [stage_8_v8, stage_8_v53], stage_8_v68)
                (stage_8_v75, stage_8_v76) = pl.yield_(stage_8_v68, stage_8_v71)
            (stage_8_v73, stage_8_v74) = pl.yield_(stage_8_v75, stage_8_v76)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_8 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_8)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_9 = 19872 + core_id * 6
    mapped_block_9 = pl.cast(pl.tensor.read(records, [record_9]), pl.INDEX)
    block_9 = pl.max(mapped_block_9, 0)
    if mapped_block_9 >= 0:
        subslot_9 = pl.cast(pl.tensor.read(records, [record_9 + 1]), pl.INDEX)
        done_9 = pl.tensor.read(records, [record_9 + 2])
        wait_base_9 = pl.cast(pl.tensor.read(records, [record_9 + 3]), pl.INDEX)
        wait_count_9 = pl.cast(pl.tensor.read(records, [record_9 + 4]), pl.INDEX)
        publish_9 = pl.tensor.read(records, [record_9 + 5])
        for dependency_9 in pl.range(wait_count_9):
            wait_index_9 = (wait_base_9 + dependency_9) * 2
            other_9 = pl.cast(pl.tensor.read(waits, [wait_index_9]), pl.INDEX)
            required_9 = pl.tensor.read(waits, [wait_index_9 + 1])
            target_9 = generation_base + required_9
            flag_9 = other_9 * 64
            pl.system.cacheinvalid(progress, [1], [flag_9])
            initial_9 = pl.tensor.read(progress, [flag_9])
            for (ready_9,) in pl.while_(init_values=(initial_9,)):
                pl.cond(ready_9 < target_9)
                pl.system.cacheinvalid(progress, [1], [flag_9])
                after_9 = pl.yield_(pl.tensor.read(progress, [flag_9]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_9_v3: pl.Scalar[pl.INDEX] = block_9
        stage_9_v4: pl.Scalar[pl.INDEX] = stage_9_v3 * 16
        stage_9_v5: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_45, [stage_9_v4, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_9_v6: pl.Scalar[pl.INDEX] = stage_9_v4 + 384
        stage_9_v7: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_45, [stage_9_v6, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_9_v8: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_9_v5, stage_9_v7)
        stage_9_v9: pl.Scalar[pl.INDEX] = stage_9_v4 + 768
        stage_9_v10: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_45, [stage_9_v9, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_9_v11: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_9_v8, stage_9_v10)
        stage_9_v12: pl.Scalar[pl.INDEX] = stage_9_v4 + 1152
        stage_9_v13: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_45, [stage_9_v12, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_9_v14: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_9_v11, stage_9_v13)
        stage_9_v15: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_9_v14, 0.011048543456039806)
        stage_9_v16 = pl.tile.store(stage_9_v15, [stage_9_v4, 0], temporary_44)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_9 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_9)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_10 = 20304 + core_id * 6
    mapped_block_10 = pl.cast(pl.tensor.read(records, [record_10]), pl.INDEX)
    block_10 = pl.max(mapped_block_10, 0)
    if mapped_block_10 >= 0:
        subslot_10 = pl.cast(pl.tensor.read(records, [record_10 + 1]), pl.INDEX)
        done_10 = pl.tensor.read(records, [record_10 + 2])
        wait_base_10 = pl.cast(pl.tensor.read(records, [record_10 + 3]), pl.INDEX)
        wait_count_10 = pl.cast(pl.tensor.read(records, [record_10 + 4]), pl.INDEX)
        publish_10 = pl.tensor.read(records, [record_10 + 5])
        for dependency_10 in pl.range(wait_count_10):
            wait_index_10 = (wait_base_10 + dependency_10) * 2
            other_10 = pl.cast(pl.tensor.read(waits, [wait_index_10]), pl.INDEX)
            required_10 = pl.tensor.read(waits, [wait_index_10 + 1])
            target_10 = generation_base + required_10
            flag_10 = other_10 * 64
            pl.system.cacheinvalid(progress, [1], [flag_10])
            initial_10 = pl.tensor.read(progress, [flag_10])
            for (ready_10,) in pl.while_(init_values=(initial_10,)):
                pl.cond(ready_10 < target_10)
                pl.system.cacheinvalid(progress, [1], [flag_10])
                after_10 = pl.yield_(pl.tensor.read(progress, [flag_10]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_10_v7: pl.Scalar[pl.INDEX] = block_10
        stage_10_v8: pl.Tile[[1, 1024], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_31, [0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
        stage_10_v9: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_10_v8, target_type=pl.FP32, mode='round')
        for stage_10_v10 in pl.range(stage_10_v7, 18, 16):
            stage_10_v11: pl.Scalar[pl.INDEX] = stage_10_v10 * 8
            stage_10_v12: pl.Scalar[pl.INDEX] = pl.min(144 - stage_10_v11, 8)
            stage_10_v13: pl.Scalar[pl.INDEX] = 0 + stage_10_v11
            stage_10_v14: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_24, [stage_10_v11, 0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_10_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
            stage_10_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
            for (stage_10_v32, (stage_10_v58, stage_10_v59)) in pl.pipeline(0, 1024, 256, stage=2, init_values=(stage_10_v16, stage_10_v15)):
                stage_10_v33: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_10_v14, 0, stage_10_v32, [8, 256], target_memory=pl.Mem.Vec)
                stage_10_v34: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_10_v33, stage_10_v33)
                stage_10_v35: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_10_v36: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_10_v34, stage_10_v35)
                stage_10_v37: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_10_v36, [1, 8])
                stage_10_v38: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_10_v59, stage_10_v37)
                stage_10_v39: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_10_v9, 0, stage_10_v32, [1, 256], target_memory=pl.Mem.Vec)
                stage_10_v40: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_10_v33, stage_10_v39)
                stage_10_v41: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_10_v40)
                stage_10_v42: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_10_v43: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_10_v41, stage_10_v42)
                stage_10_v44: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_10_v43, [1, 8])
                stage_10_v45: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_10_v58, stage_10_v44)
                (stage_10_v60, stage_10_v61) = pl.yield_(stage_10_v45, stage_10_v38)
            stage_10_v17: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_10_v61, 0.0009765625)
            stage_10_v18: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_10_v17, 1e-06)
            stage_10_v19: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_10_v20: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(stage_10_v18, stage_10_v19)
            stage_10_v21: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_10_v20, [8, 1])
            stage_10_v22: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_10_v23: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_10_v20, stage_10_v60)
            stage_10_v24: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_10_v22, stage_10_v23)
            stage_10_v25: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_10_v26: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_10_v25, stage_10_v24)
            stage_10_v27: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_10_v26, [8, 1])
            stage_10_v28: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_10_v26)
            stage_10_v29: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_10_v28, [8, 1])
            stage_10_v30 = pl.tile.store(stage_10_v29, [stage_10_v11, 0], temporary_23)
            if stage_10_v12 == 8:
                stage_10_v46 = pl.tile.store(stage_10_v29, [stage_10_v13, 0], temporary_19)
            else:
                stage_10_v47: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v12, 1])] = pl.tile.slice(stage_10_v29, [8, 1], [0, 0], valid_shape=[stage_10_v12, 1])
                stage_10_v48 = pl.tile.store(stage_10_v47, [stage_10_v13, 0], temporary_19)
            for stage_10_v31 in pl.pipeline(0, 1024, 256, stage=2):
                stage_10_v49: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_10_v14, 0, stage_10_v31, [8, 256], target_memory=pl.Mem.Vec)
                stage_10_v50: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_10_v9, 0, stage_10_v31, [1, 256], target_memory=pl.Mem.Vec)
                stage_10_v51: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_10_v49, stage_10_v21)
                stage_10_v52: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_10_v51, stage_10_v50)
                stage_10_v53: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_10_v52, stage_10_v27)
                stage_10_v54: pl.Tile[[8, 256], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_10_v53, target_type=pl.INT32, mode='rint')
                stage_10_v55: pl.Tile[[8, 256], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_10_v54, target_type=pl.FP16, mode='round')
                stage_10_v56: pl.Tile[[8, 256], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_10_v55, target_type=pl.INT8, mode='trunc')
                stage_10_v57 = pl.tile.store(stage_10_v56, [stage_10_v11, stage_10_v31], temporary_22)
                if stage_10_v12 == 8:
                    stage_10_v62 = pl.tile.store(stage_10_v56, [stage_10_v13, stage_10_v31], temporary_18)
                else:
                    stage_10_v63: pl.Tile[[8, 256], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v12, 256])] = pl.tile.set_validshape(stage_10_v56, stage_10_v12, 256)
                    stage_10_v64 = pl.tile.store(stage_10_v63, [stage_10_v13, stage_10_v31], temporary_18)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_10 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_10)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_11 = 20736 + core_id * 6
    mapped_block_11 = pl.cast(pl.tensor.read(records, [record_11]), pl.INDEX)
    block_11 = pl.max(mapped_block_11, 0)
    if mapped_block_11 >= 0:
        subslot_11 = pl.cast(pl.tensor.read(records, [record_11 + 1]), pl.INDEX)
        done_11 = pl.tensor.read(records, [record_11 + 2])
        wait_base_11 = pl.cast(pl.tensor.read(records, [record_11 + 3]), pl.INDEX)
        wait_count_11 = pl.cast(pl.tensor.read(records, [record_11 + 4]), pl.INDEX)
        publish_11 = pl.tensor.read(records, [record_11 + 5])
        for dependency_11 in pl.range(wait_count_11):
            wait_index_11 = (wait_base_11 + dependency_11) * 2
            other_11 = pl.cast(pl.tensor.read(waits, [wait_index_11]), pl.INDEX)
            required_11 = pl.tensor.read(waits, [wait_index_11 + 1])
            target_11 = generation_base + required_11
            flag_11 = other_11 * 64
            pl.system.cacheinvalid(progress, [1], [flag_11])
            initial_11 = pl.tensor.read(progress, [flag_11])
            for (ready_11,) in pl.while_(init_values=(initial_11,)):
                pl.cond(ready_11 < target_11)
                pl.system.cacheinvalid(progress, [1], [flag_11])
                after_11 = pl.yield_(pl.tensor.read(progress, [flag_11]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_11_v8: pl.Scalar[pl.INDEX] = block_11
        for (stage_11_v9, (stage_11_v12,)) in pl.range(stage_11_v8, 24, 24, init_values=(temporary_36,)):
            stage_11_v10: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_11_v9 * 6])
            for (stage_11_v13, (stage_11_v19,)) in pl.range(6, init_values=(stage_11_v12,)):
                stage_11_v14: pl.Scalar[pl.INDEX] = stage_11_v9 * 6 + stage_11_v13
                stage_11_v15: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_11_v14])
                stage_11_v16: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
                stage_11_v17 = pl.tile.store(stage_11_v16, [stage_11_v14, 0], stage_11_v19)
                if (stage_11_v15 + 1) % 4 == 0:
                    stage_11_v20: pl.Scalar[pl.INT64] = stage_11_v15 - 8 + 1
                    for (stage_11_v22, (stage_11_v33,)) in pl.range(0, 128, 64, init_values=(stage_11_v17,)):
                        stage_11_v23: pl.Scalar[pl.INDEX] = pl.cast(stage_11_v15 % 4, pl.INDEX)
                        stage_11_v24: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_35, [stage_11_v14, stage_11_v22 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                        stage_11_v25: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_ape, [stage_11_v23, stage_11_v22 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                        stage_11_v26: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_11_v24, stage_11_v25)
                        stage_11_v27: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_11_v26, stage_11_v26)
                        stage_11_v28: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_11_v27)
                        stage_11_v29: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_34, [stage_11_v14, stage_11_v22 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                        for (stage_11_v34, (stage_11_v50, stage_11_v51, stage_11_v52)) in pl.range(7, init_values=(stage_11_v28, stage_11_v26, stage_11_v29)):
                            stage_11_v35: pl.Scalar[pl.INT64] = stage_11_v20 + stage_11_v34
                            stage_11_v36: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
                            stage_11_v37: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=-3.4028234663852886e+38)
                            stage_11_v38: pl.Scalar[pl.INDEX] = 0
                            if 4 <= stage_11_v34:
                                stage_11_v53: pl.Scalar[pl.INDEX] = 128
                                stage_11_v54: pl.Scalar[pl.INDEX] = pl.yield_(stage_11_v53)
                            else:
                                stage_11_v54: pl.Scalar[pl.INDEX] = pl.yield_(stage_11_v38)
                            if 0 <= stage_11_v35 and stage_11_v35 < stage_11_v10:
                                stage_11_v55: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
                                stage_11_v56: pl.Scalar[pl.INT32] = pl.tensor.read(arg_inner_state_block_table, [stage_11_v9, stage_11_v35 // 2])
                                stage_11_v57: pl.Scalar[pl.INDEX] = pl.cast(stage_11_v56, pl.INDEX)
                                if 0 <= stage_11_v57:
                                    stage_11_v61: pl.Scalar[pl.INDEX] = pl.cast(stage_11_v35 % 2, pl.INDEX) * 512 + stage_11_v54 + stage_11_v22
                                    stage_11_v62: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_compress_state, [stage_11_v57, stage_11_v61], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    stage_11_v63: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_compress_state, [stage_11_v57, stage_11_v61 + 256], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    (stage_11_v74, stage_11_v75) = pl.yield_(stage_11_v63, stage_11_v62)
                                else:
                                    (stage_11_v74, stage_11_v75) = pl.yield_(stage_11_v55, stage_11_v36)
                                (stage_11_v64, stage_11_v65) = pl.yield_(stage_11_v74, stage_11_v75)
                            else:
                                (stage_11_v64, stage_11_v65) = pl.yield_(stage_11_v37, stage_11_v36)
                            if stage_11_v10 <= stage_11_v35:
                                if stage_11_v35 <= stage_11_v15:
                                    stage_11_v66: pl.Scalar[pl.INDEX] = stage_11_v9 * 6 + stage_11_v35 - stage_11_v10
                                    stage_11_v67: pl.Scalar[pl.INDEX] = pl.cast(stage_11_v35 % 4, pl.INDEX)
                                    stage_11_v68: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_34, [stage_11_v66, stage_11_v54 + stage_11_v22], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    stage_11_v69: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_35, [stage_11_v66, stage_11_v54 + stage_11_v22], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    stage_11_v70: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_ape, [stage_11_v67, stage_11_v54 + stage_11_v22], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    stage_11_v71: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_11_v69, stage_11_v70)
                                    (stage_11_v76, stage_11_v77) = pl.yield_(stage_11_v71, stage_11_v68)
                                else:
                                    (stage_11_v76, stage_11_v77) = pl.yield_(stage_11_v64, stage_11_v65)
                                (stage_11_v72, stage_11_v73) = pl.yield_(stage_11_v76, stage_11_v77)
                            else:
                                (stage_11_v72, stage_11_v73) = pl.yield_(stage_11_v64, stage_11_v65)
                            stage_11_v39: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_11_v51, stage_11_v72)
                            stage_11_v40: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_11_v51, stage_11_v39)
                            stage_11_v41: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_11_v40)
                            stage_11_v42: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_11_v72, stage_11_v39)
                            stage_11_v43: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_11_v42)
                            stage_11_v44: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_11_v41, stage_11_v50)
                            stage_11_v45: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_11_v44, stage_11_v43)
                            stage_11_v46: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_11_v52, stage_11_v41)
                            stage_11_v47: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_11_v73, stage_11_v43)
                            stage_11_v48: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_11_v46, stage_11_v47)
                            stage_11_v49: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = stage_11_v39
                            (stage_11_v58, stage_11_v59, stage_11_v60) = pl.yield_(stage_11_v45, stage_11_v49, stage_11_v48)
                        stage_11_v30: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_11_v60, stage_11_v58)
                        stage_11_v31 = pl.tile.store(stage_11_v30, [stage_11_v14, stage_11_v22], stage_11_v33)
                        stage_11_v32 = pl.yield_(stage_11_v31)
                    stage_11_v21 = pl.yield_(stage_11_v32)
                else:
                    stage_11_v21 = pl.yield_(stage_11_v17)
                stage_11_v18 = pl.yield_(stage_11_v21)
            stage_11_v11 = pl.yield_(stage_11_v18)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_11 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_11)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_12 = 21168 + core_id * 6
    mapped_block_12 = pl.cast(pl.tensor.read(records, [record_12]), pl.INDEX)
    block_12 = pl.max(mapped_block_12, 0)
    if mapped_block_12 >= 0:
        subslot_12 = pl.cast(pl.tensor.read(records, [record_12 + 1]), pl.INDEX)
        done_12 = pl.tensor.read(records, [record_12 + 2])
        wait_base_12 = pl.cast(pl.tensor.read(records, [record_12 + 3]), pl.INDEX)
        wait_count_12 = pl.cast(pl.tensor.read(records, [record_12 + 4]), pl.INDEX)
        publish_12 = pl.tensor.read(records, [record_12 + 5])
        for dependency_12 in pl.range(wait_count_12):
            wait_index_12 = (wait_base_12 + dependency_12) * 2
            other_12 = pl.cast(pl.tensor.read(waits, [wait_index_12]), pl.INDEX)
            required_12 = pl.tensor.read(waits, [wait_index_12 + 1])
            target_12 = generation_base + required_12
            flag_12 = other_12 * 64
            pl.system.cacheinvalid(progress, [1], [flag_12])
            initial_12 = pl.tensor.read(progress, [flag_12])
            for (ready_12,) in pl.while_(init_values=(initial_12,)):
                pl.cond(ready_12 < target_12)
                pl.system.cacheinvalid(progress, [1], [flag_12])
                after_12 = pl.yield_(pl.tensor.read(progress, [flag_12]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_12_v9: pl.Scalar[pl.INDEX] = block_12
        for (stage_12_v10, (stage_12_v14,)) in pl.range(stage_12_v9, 24, 24, init_values=(temporary_28,)):
            stage_12_v11: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_12_v10 * 6])
            stage_12_v12: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_seq_lens, [stage_12_v10])
            for (stage_12_v15, (stage_12_v21,)) in pl.range(6, init_values=(stage_12_v14,)):
                stage_12_v16: pl.Scalar[pl.INDEX] = stage_12_v10 * 6 + stage_12_v15
                stage_12_v17: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_12_v16])
                stage_12_v18: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                stage_12_v19 = pl.tile.store(stage_12_v18, [stage_12_v16, 0], stage_12_v21)
                if 0 < pl.cast(stage_12_v12, pl.INDEX) and (stage_12_v17 + 1) % 4 == 0:
                    stage_12_v22: pl.Scalar[pl.INT64] = stage_12_v17 - 8 + 1
                    stage_12_v23: pl.Scalar[pl.INDEX] = pl.cast(stage_12_v17 % 4, pl.INDEX)
                    stage_12_v24: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_30, [stage_12_v16, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    stage_12_v25: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_cmp_ape, [stage_12_v23, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    stage_12_v26: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_12_v24, stage_12_v25)
                    stage_12_v27: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_12_v26, stage_12_v26)
                    stage_12_v28: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_12_v27)
                    stage_12_v29: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_29, [stage_12_v16, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    for (stage_12_v31, (stage_12_v47, stage_12_v48, stage_12_v49)) in pl.range(7, init_values=(stage_12_v28, stage_12_v26, stage_12_v29)):
                        stage_12_v32: pl.Scalar[pl.INT64] = stage_12_v22 + stage_12_v31
                        stage_12_v33: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                        stage_12_v34: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e+38)
                        stage_12_v35: pl.Scalar[pl.INDEX] = 0
                        if 4 <= stage_12_v31:
                            stage_12_v50: pl.Scalar[pl.INDEX] = 512
                            stage_12_v51: pl.Scalar[pl.INDEX] = pl.yield_(stage_12_v50)
                        else:
                            stage_12_v51: pl.Scalar[pl.INDEX] = pl.yield_(stage_12_v35)
                        if 0 <= stage_12_v32 and stage_12_v32 < stage_12_v11:
                            stage_12_v52: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                            stage_12_v53: pl.Scalar[pl.INT32] = pl.tensor.read(arg_state_block_table, [stage_12_v10, stage_12_v32 // 2])
                            stage_12_v54: pl.Scalar[pl.INDEX] = pl.cast(stage_12_v53, pl.INDEX)
                            if 0 <= stage_12_v54:
                                stage_12_v58: pl.Scalar[pl.INDEX] = pl.cast(stage_12_v32 % 2, pl.INDEX) * 2048 + stage_12_v51
                                stage_12_v59: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_compress_state, [stage_12_v54, stage_12_v58], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                stage_12_v60: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_compress_state, [stage_12_v54, stage_12_v58 + 1024], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                (stage_12_v71, stage_12_v72) = pl.yield_(stage_12_v60, stage_12_v59)
                            else:
                                (stage_12_v71, stage_12_v72) = pl.yield_(stage_12_v52, stage_12_v33)
                            (stage_12_v61, stage_12_v62) = pl.yield_(stage_12_v71, stage_12_v72)
                        else:
                            (stage_12_v61, stage_12_v62) = pl.yield_(stage_12_v34, stage_12_v33)
                        if stage_12_v11 <= stage_12_v32:
                            if stage_12_v32 <= stage_12_v17:
                                stage_12_v63: pl.Scalar[pl.INDEX] = stage_12_v10 * 6 + stage_12_v32 - stage_12_v11
                                stage_12_v64: pl.Scalar[pl.INDEX] = pl.cast(stage_12_v32 % 4, pl.INDEX)
                                stage_12_v65: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_29, [stage_12_v63, stage_12_v51], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                stage_12_v66: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_30, [stage_12_v63, stage_12_v51], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                stage_12_v67: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_cmp_ape, [stage_12_v64, stage_12_v51], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                stage_12_v68: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_12_v66, stage_12_v67)
                                (stage_12_v73, stage_12_v74) = pl.yield_(stage_12_v68, stage_12_v65)
                            else:
                                (stage_12_v73, stage_12_v74) = pl.yield_(stage_12_v61, stage_12_v62)
                            (stage_12_v69, stage_12_v70) = pl.yield_(stage_12_v73, stage_12_v74)
                        else:
                            (stage_12_v69, stage_12_v70) = pl.yield_(stage_12_v61, stage_12_v62)
                        stage_12_v36: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_12_v48, stage_12_v69)
                        stage_12_v37: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_12_v48, stage_12_v36)
                        stage_12_v38: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_12_v37)
                        stage_12_v39: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_12_v69, stage_12_v36)
                        stage_12_v40: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_12_v39)
                        stage_12_v41: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_12_v38, stage_12_v47)
                        stage_12_v42: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_12_v41, stage_12_v40)
                        stage_12_v43: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_12_v49, stage_12_v38)
                        stage_12_v44: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_12_v70, stage_12_v40)
                        stage_12_v45: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_12_v43, stage_12_v44)
                        stage_12_v46: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = stage_12_v36
                        (stage_12_v55, stage_12_v56, stage_12_v57) = pl.yield_(stage_12_v42, stage_12_v46, stage_12_v45)
                    stage_12_v30: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_12_v57, stage_12_v55)
                    pl.tile.store(stage_12_v30, [stage_12_v16, 0], stage_12_v19)
                stage_12_v20 = pl.yield_(stage_12_v19)
            stage_12_v13 = pl.yield_(stage_12_v20)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_12 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_12)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_13 = 21600 + core_id * 6
    mapped_block_13 = pl.cast(pl.tensor.read(records, [record_13]), pl.INDEX)
    block_13 = pl.max(mapped_block_13, 0)
    if mapped_block_13 >= 0:
        subslot_13 = pl.cast(pl.tensor.read(records, [record_13 + 1]), pl.INDEX)
        done_13 = pl.tensor.read(records, [record_13 + 2])
        wait_base_13 = pl.cast(pl.tensor.read(records, [record_13 + 3]), pl.INDEX)
        wait_count_13 = pl.cast(pl.tensor.read(records, [record_13 + 4]), pl.INDEX)
        publish_13 = pl.tensor.read(records, [record_13 + 5])
        for dependency_13 in pl.range(wait_count_13):
            wait_index_13 = (wait_base_13 + dependency_13) * 2
            other_13 = pl.cast(pl.tensor.read(waits, [wait_index_13]), pl.INDEX)
            required_13 = pl.tensor.read(waits, [wait_index_13 + 1])
            target_13 = generation_base + required_13
            flag_13 = other_13 * 64
            pl.system.cacheinvalid(progress, [1], [flag_13])
            initial_13 = pl.tensor.read(progress, [flag_13])
            for (ready_13,) in pl.while_(init_values=(initial_13,)):
                pl.cond(ready_13 < target_13)
                pl.system.cacheinvalid(progress, [1], [flag_13])
                after_13 = pl.yield_(pl.tensor.read(progress, [flag_13]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_13_v2: pl.Scalar[pl.INDEX] = block_13
        stage_13_v3: pl.Scalar[pl.INDEX] = stage_13_v2 * 2
        stage_13_v4: pl.Tile[[2, 128], pl.BF16, pl.Mem.Vec] = pl.tile.full([2, 128], dtype=pl.BF16, value=0.0)
        stage_13_v5 = pl.tile.store(stage_13_v4, [stage_13_v3, 0], temporary_33)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_13 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_13)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_14 = 22032 + core_id * 6
    mapped_block_14 = pl.cast(pl.tensor.read(records, [record_14]), pl.INDEX)
    block_14 = pl.max(mapped_block_14, 0)
    if mapped_block_14 >= 0:
        subslot_14 = pl.cast(pl.tensor.read(records, [record_14 + 1]), pl.INDEX)
        done_14 = pl.tensor.read(records, [record_14 + 2])
        wait_base_14 = pl.cast(pl.tensor.read(records, [record_14 + 3]), pl.INDEX)
        wait_count_14 = pl.cast(pl.tensor.read(records, [record_14 + 4]), pl.INDEX)
        publish_14 = pl.tensor.read(records, [record_14 + 5])
        for dependency_14 in pl.range(wait_count_14):
            wait_index_14 = (wait_base_14 + dependency_14) * 2
            other_14 = pl.cast(pl.tensor.read(waits, [wait_index_14]), pl.INDEX)
            required_14 = pl.tensor.read(waits, [wait_index_14 + 1])
            target_14 = generation_base + required_14
            flag_14 = other_14 * 64
            pl.system.cacheinvalid(progress, [1], [flag_14])
            initial_14 = pl.tensor.read(progress, [flag_14])
            for (ready_14,) in pl.while_(init_values=(initial_14,)):
                pl.cond(ready_14 < target_14)
                pl.system.cacheinvalid(progress, [1], [flag_14])
                after_14 = pl.yield_(pl.tensor.read(progress, [flag_14]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_14_v8: pl.Scalar[pl.INDEX] = block_14
        stage_14_v9: pl.Scalar[pl.INDEX] = 4
        if 384 <= 144 and 144 % 384 == 0:
            stage_14_v11: pl.Scalar[pl.INDEX] = 64
            stage_14_v12: pl.Scalar[pl.INDEX] = pl.yield_(stage_14_v11)
        else:
            if 96 <= 144 and 144 % 96 == 0:
                stage_14_v16: pl.Scalar[pl.INDEX] = 16
                stage_14_v17: pl.Scalar[pl.INDEX] = pl.yield_(stage_14_v16)
            else:
                stage_14_v17: pl.Scalar[pl.INDEX] = pl.yield_(stage_14_v9)
            stage_14_v12: pl.Scalar[pl.INDEX] = pl.yield_(stage_14_v17)
        for stage_14_v10 in pl.range(stage_14_v8, (144 + 7) // 8 * (64 // stage_14_v12), 48):
            stage_14_v13: pl.Scalar[pl.INDEX] = stage_14_v10 % (64 // stage_14_v12) * stage_14_v12
            stage_14_v14: pl.Scalar[pl.INDEX] = stage_14_v10 // (64 // stage_14_v12) * 8
            stage_14_v15: pl.Scalar[pl.INDEX] = 0 + stage_14_v14
            if stage_14_v14 + 8 <= 144:
                stage_14_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_23, [stage_14_v14, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_14_v19: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_14_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_14_v20: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_20, [stage_14_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_14_v21: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_21, [stage_14_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_14_v22: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                stage_14_v23: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_14_v22, target_type=pl.FP32, mode='round')
                stage_14_v24: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_14_v23, 0.015625)
                stage_14_v25: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_14_v24, target_type=pl.INT32, mode='trunc')
                stage_14_v26: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_14_v25, target_type=pl.FP32, mode='round')
                stage_14_v27: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_14_v26, 64.0)
                stage_14_v28: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v21, [1, 512])
                stage_14_v29: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_14_v28, target_type=pl.FP32, mode='round')
                stage_14_v30: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_14_v29, stage_14_v27)
                stage_14_v31: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_14_v30, target_type=pl.INT32, mode='round')
                stage_14_v32: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 512], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_14_v33 in pl.pipeline(stage_14_v12, stage=2):
                    stage_14_v62: pl.Scalar[pl.INDEX] = stage_14_v13 + stage_14_v33
                    stage_14_v63: pl.Scalar[pl.INDEX] = stage_14_v62 * 512
                    stage_14_v64: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_25, [stage_14_v14, stage_14_v63], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    stage_14_v65: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wq_b_scale, [stage_14_v63], [512], [512], target_memory=pl.Mem.Vec)
                    stage_14_v66: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v65, [1, 512])
                    stage_14_v67: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_14_v64, target_type=pl.FP32, mode='none')
                    stage_14_v68: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_14_v67, stage_14_v18)
                    stage_14_v69: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_14_v68, stage_14_v66)
                    stage_14_v70: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_14_v69, stage_14_v69)
                    stage_14_v71: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_14_v72: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_14_v70, stage_14_v71)
                    stage_14_v73: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v72, [1, 8])
                    stage_14_v74: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_14_v73, 0.001953125)
                    stage_14_v75: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_14_v74, 1e-06)
                    stage_14_v76: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_14_v77: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(stage_14_v75, stage_14_v76)
                    stage_14_v78: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v77, [8, 1])
                    stage_14_v79: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_14_v69, [8, 448], [0, 0])
                    stage_14_v80: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_14_v79, stage_14_v78)
                    stage_14_v81: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_14_v80, target_type=pl.BF16, mode='rint')
                    stage_14_v82 = pl.tile.store(stage_14_v81, [stage_14_v15, stage_14_v63], view_40)
                    stage_14_v83: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_14_v69, [8, 64], [0, 448])
                    stage_14_v84: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_14_v83, stage_14_v78)
                    stage_14_v85: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v84, [1, 512])
                    stage_14_v86: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_14_v85, stage_14_v31, stage_14_v32)
                    stage_14_v87: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v86, [8, 64])
                    stage_14_v88: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_14_v84, stage_14_v19)
                    stage_14_v89: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_14_v87, stage_14_v20)
                    stage_14_v90: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_14_v88, stage_14_v89)
                    stage_14_v91: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_14_v90, target_type=pl.BF16, mode='rint')
                    stage_14_v92 = pl.tile.store(stage_14_v91, [stage_14_v15, stage_14_v63 + 448], view_40)
            else:
                stage_14_v34: pl.Scalar[pl.INDEX] = 144 - stage_14_v14
                stage_14_v35: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_23, [stage_14_v14, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_14_v36: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_14_v34, 64])] = pl.tile.load(arg_freqs_cos, [stage_14_v15, 0], [8, 64], [stage_14_v34, 64], target_memory=pl.Mem.Vec)
                stage_14_v37: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_14_v34, 64])] = pl.tile.load(temporary_20, [stage_14_v15, 0], [8, 64], [stage_14_v34, 64], target_memory=pl.Mem.Vec)
                stage_14_v38: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                stage_14_v39: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                stage_14_v40: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_14_v39, target_type=pl.FP32, mode='round')
                stage_14_v41: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_14_v38, stage_14_v40)
                stage_14_v42: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_14_v41, 0.5)
                stage_14_v43: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_14_v42, target_type=pl.INT32, mode='trunc')
                stage_14_v44: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_14_v43, target_type=pl.FP32, mode='round')
                stage_14_v45: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_14_v44, 2.0)
                stage_14_v46: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_14_v41, stage_14_v45)
                stage_14_v47: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_14_v41, 1.0)
                stage_14_v48: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_14_v46, 2.0)
                stage_14_v49: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_14_v47, stage_14_v48)
                stage_14_v50: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False)
                stage_14_v51: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_14_v50, target_type=pl.FP32, mode='round')
                stage_14_v52: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_14_v51, 64.0)
                stage_14_v53: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                stage_14_v54: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_14_v53, stage_14_v52)
                stage_14_v55: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_14_v56: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(stage_14_v54, 0, 1, stage_14_v55)
                stage_14_v57: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_14_v49, stage_14_v56)
                stage_14_v58: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_14_v57, target_type=pl.INT32, mode='round')
                stage_14_v59: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_14_v60: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_14_v61 in pl.range(4):
                    stage_14_v93: pl.Scalar[pl.INDEX] = stage_14_v13 + stage_14_v61
                    stage_14_v94: pl.Scalar[pl.INDEX] = stage_14_v93 * 512
                    stage_14_v95: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_25, [stage_14_v14, stage_14_v94], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    stage_14_v96: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wq_b_scale, [stage_14_v94], [512], [512], target_memory=pl.Mem.Vec)
                    stage_14_v97: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v96, [1, 512])
                    stage_14_v98: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_14_v95, target_type=pl.FP32, mode='none')
                    stage_14_v99: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_14_v98, stage_14_v35)
                    stage_14_v100: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_14_v99, stage_14_v97)
                    stage_14_v101: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_14_v100, stage_14_v100)
                    stage_14_v102: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_14_v101, stage_14_v59)
                    stage_14_v103: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v102, [1, 8])
                    stage_14_v104: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_14_v103, 0.001953125)
                    stage_14_v105: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v104, [8, 1])
                    stage_14_v106: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v105, [1, 8])
                    stage_14_v107: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_14_v106, 1e-06)
                    stage_14_v108: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v107, [8, 1])
                    stage_14_v109: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v108, [1, 8])
                    stage_14_v110: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_14_v109)
                    stage_14_v111: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v110, [8, 1])
                    stage_14_v112: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v111, [1, 8])
                    stage_14_v113: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_14_v112)
                    stage_14_v114: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_14_v113, [8, 1])
                    stage_14_v115: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_14_v100, [8, 448], [0, 0])
                    stage_14_v116: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_14_v115, stage_14_v114)
                    stage_14_v117: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_14_v116, target_type=pl.BF16, mode='rint')
                    stage_14_v118: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_14_v34, 448])] = pl.tile.set_validshape(stage_14_v117, stage_14_v34, 448)
                    stage_14_v119 = pl.tile.store(stage_14_v118, [stage_14_v15, stage_14_v94], view_40)
                    stage_14_v120: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_14_v100, [8, 64], [0, 448])
                    stage_14_v121: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_14_v120, stage_14_v114)
                    stage_14_v122: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_14_v121, stage_14_v58, stage_14_v60)
                    stage_14_v123: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_14_v34, 64])] = pl.tile.set_validshape(stage_14_v121, stage_14_v34, 64)
                    stage_14_v124: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_14_v34, 64])] = pl.tile.set_validshape(stage_14_v122, stage_14_v34, 64)
                    stage_14_v125: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_14_v34, 64])] = pl.tile.mul(stage_14_v123, stage_14_v36)
                    stage_14_v126: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_14_v34, 64])] = pl.tile.mul(stage_14_v124, stage_14_v37)
                    stage_14_v127: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_14_v34, 64])] = pl.tile.add(stage_14_v125, stage_14_v126)
                    stage_14_v128: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_14_v34, 64])] = pl.tile.cast(stage_14_v127, target_type=pl.BF16, mode='rint')
                    stage_14_v129: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_14_v34, 64])] = pl.tile.set_validshape(stage_14_v128, stage_14_v34, 64)
                    stage_14_v130 = pl.tile.store(stage_14_v129, [stage_14_v15, stage_14_v94 + 448], view_40)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_14 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_14)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_15 = 22464 + core_id * 6
    mapped_block_15 = pl.cast(pl.tensor.read(records, [record_15]), pl.INDEX)
    block_15 = pl.max(mapped_block_15, 0)
    if mapped_block_15 >= 0:
        subslot_15 = pl.cast(pl.tensor.read(records, [record_15 + 1]), pl.INDEX)
        done_15 = pl.tensor.read(records, [record_15 + 2])
        wait_base_15 = pl.cast(pl.tensor.read(records, [record_15 + 3]), pl.INDEX)
        wait_count_15 = pl.cast(pl.tensor.read(records, [record_15 + 4]), pl.INDEX)
        publish_15 = pl.tensor.read(records, [record_15 + 5])
        for dependency_15 in pl.range(wait_count_15):
            wait_index_15 = (wait_base_15 + dependency_15) * 2
            other_15 = pl.cast(pl.tensor.read(waits, [wait_index_15]), pl.INDEX)
            required_15 = pl.tensor.read(waits, [wait_index_15 + 1])
            target_15 = generation_base + required_15
            flag_15 = other_15 * 64
            pl.system.cacheinvalid(progress, [1], [flag_15])
            initial_15 = pl.tensor.read(progress, [flag_15])
            for (ready_15,) in pl.while_(init_values=(initial_15,)):
                pl.cond(ready_15 < target_15)
                pl.system.cacheinvalid(progress, [1], [flag_15])
                after_15 = pl.yield_(pl.tensor.read(progress, [flag_15]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_15_v7: pl.Scalar[pl.INDEX] = block_15
        stage_15_v8: pl.Scalar[pl.INDEX] = stage_15_v7 * 32
        stage_15_v9: pl.Scalar[pl.INDEX] = pl.min(144 - stage_15_v8, 32)
        stage_15_v10: pl.Scalar[pl.INDEX] = 0 + stage_15_v8
        if stage_15_v9 == 32:
            stage_15_v11: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=0.0)
            for (stage_15_v85, (stage_15_v121,)) in pl.pipeline(0, 512, 64, stage=2, init_values=(stage_15_v11,)):
                stage_15_v86: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_26, [stage_15_v8, stage_15_v85], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
                stage_15_v87: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_15_v86, stage_15_v86)
                stage_15_v88: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_15_v121, stage_15_v87)
                stage_15_v89: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_15_v88)
            stage_15_v12: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_15_v13: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_15_v89, stage_15_v12)
            stage_15_v14: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v13, [1, 32])
            stage_15_v15: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v14, 0.001953125)
            stage_15_v16: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_15_v15, 1e-06)
            stage_15_v17: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_15_v16)
            stage_15_v18: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=1.0)
            stage_15_v19: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_15_v18, stage_15_v17)
            stage_15_v20: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v19, [32, 1])
            for (stage_15_v90, (stage_15_v122,)) in pl.pipeline(0, 448, 64, stage=2, init_values=(temporary_17,)):
                stage_15_v91: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_26, [stage_15_v8, stage_15_v90], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
                stage_15_v92: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_gamma_ckv, [stage_15_v90], [64], [64], target_memory=pl.Mem.Vec)
                stage_15_v93: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v92, target_type=pl.FP32, mode='round')
                stage_15_v94: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v93, [1, 64])
                stage_15_v95: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_15_v91, stage_15_v20)
                stage_15_v96: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_15_v95, stage_15_v94)
                stage_15_v97: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_15_v96, target_type=pl.BF16, mode='rint')
                stage_15_v98 = pl.tile.store(stage_15_v97, [stage_15_v10, stage_15_v90], stage_15_v122)
                stage_15_v99 = pl.yield_(stage_15_v98)
            stage_15_v21: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_gamma_ckv, [448], [64], [64], target_memory=pl.Mem.Vec)
            stage_15_v22: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v21, target_type=pl.FP32, mode='round')
            stage_15_v23: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v22, [1, 64])
            stage_15_v24: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_26, [stage_15_v8, 448], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
            stage_15_v25: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_15_v24, stage_15_v20)
            stage_15_v26: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_15_v25, stage_15_v23)
            stage_15_v27: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_15_v10, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
            stage_15_v28: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_20, [stage_15_v10, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
            stage_15_v29: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_21, [stage_15_v10, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
            stage_15_v30: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for (stage_15_v100, (stage_15_v123,)) in pl.range(32, init_values=(stage_15_v30,)):
                stage_15_v101: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_15_v26, [1, 64], [stage_15_v100, 0], [1, 64])
                stage_15_v102: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_15_v29, [1, 64], [stage_15_v100, 0], [1, 64])
                stage_15_v103: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                stage_15_v104: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_15_v101, stage_15_v102, stage_15_v103)
                stage_15_v105: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_15_v123, stage_15_v104, [stage_15_v100, 0])
                stage_15_v106: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_15_v105)
            stage_15_v31: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = stage_15_v106
            stage_15_v32: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_15_v26, stage_15_v27)
            stage_15_v33: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_15_v31, stage_15_v28)
            stage_15_v34: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_15_v32, stage_15_v33)
            stage_15_v35: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_15_v34, target_type=pl.BF16, mode='rint')
            stage_15_v36 = pl.tile.store(stage_15_v35, [stage_15_v10, 448], stage_15_v99)
        else:
            stage_15_v37: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_15_v38: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=0.0)
            for (stage_15_v107, (stage_15_v124,)) in pl.pipeline(0, 512, 64, stage=2, init_values=(stage_15_v38,)):
                stage_15_v108: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.load(temporary_26, [stage_15_v8, stage_15_v107], [32, 64], [stage_15_v9, 64], target_memory=pl.Mem.Vec)
                stage_15_v109: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.mul(stage_15_v108, stage_15_v108)
                stage_15_v110: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_15_v124, stage_15_v109)
                stage_15_v111: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_15_v110)
            stage_15_v39: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_15_v111, stage_15_v37)
            stage_15_v40: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v39, [1, 32])
            stage_15_v41: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v40, 0.001953125)
            stage_15_v42: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_15_v41, 1e-06)
            stage_15_v43: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_15_v42)
            stage_15_v44: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=1.0)
            stage_15_v45: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_15_v44, stage_15_v43)
            stage_15_v46: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v45, [32, 1])
            for stage_15_v47 in pl.pipeline(0, 448, 64, stage=2):
                stage_15_v112: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.load(temporary_26, [stage_15_v8, stage_15_v47], [32, 64], [stage_15_v9, 64], target_memory=pl.Mem.Vec)
                stage_15_v113: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_gamma_ckv, [stage_15_v47], [64], [64], target_memory=pl.Mem.Vec)
                stage_15_v114: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v113, target_type=pl.FP32, mode='round')
                stage_15_v115: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v114, [1, 64])
                stage_15_v116: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.row_expand_mul(stage_15_v112, stage_15_v46)
                stage_15_v117: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.col_expand_mul(stage_15_v116, stage_15_v115)
                stage_15_v118: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.cast(stage_15_v117, target_type=pl.BF16, mode='rint')
                stage_15_v119: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.set_validshape(stage_15_v118, stage_15_v9, 64)
                stage_15_v120 = pl.tile.store(stage_15_v119, [stage_15_v10, stage_15_v47], temporary_17)
            stage_15_v48: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_gamma_ckv, [448], [64], [64], target_memory=pl.Mem.Vec)
            stage_15_v49: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v48, target_type=pl.FP32, mode='round')
            stage_15_v50: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v49, [1, 64])
            stage_15_v51: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.load(temporary_26, [stage_15_v8, 448], [32, 64], [stage_15_v9, 64], target_memory=pl.Mem.Vec)
            stage_15_v52: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.row_expand_mul(stage_15_v51, stage_15_v46)
            stage_15_v53: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.col_expand_mul(stage_15_v52, stage_15_v50)
            stage_15_v54: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.load(arg_freqs_cos, [stage_15_v10, 0], [32, 64], [stage_15_v9, 64], target_memory=pl.Mem.Vec)
            stage_15_v55: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.load(temporary_20, [stage_15_v10, 0], [32, 64], [stage_15_v9, 64], target_memory=pl.Mem.Vec)
            stage_15_v56: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=1.0)
            stage_15_v57: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            stage_15_v58: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v57, target_type=pl.FP32, mode='round')
            stage_15_v59: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_15_v56, stage_15_v58)
            stage_15_v60: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v59, 0.5)
            stage_15_v61: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_15_v60, target_type=pl.INT32, mode='trunc')
            stage_15_v62: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v61, target_type=pl.FP32, mode='round')
            stage_15_v63: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v62, 2.0)
            stage_15_v64: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_15_v59, stage_15_v63)
            stage_15_v65: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_15_v59, 1.0)
            stage_15_v66: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v64, 2.0)
            stage_15_v67: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_15_v65, stage_15_v66)
            stage_15_v68: pl.Tile[[1, 32], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 32], dtype=pl.INT32, descending=False)
            stage_15_v69: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v68, target_type=pl.FP32, mode='round')
            stage_15_v70: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v69, 64.0)
            stage_15_v71: pl.Tile[[64, 32], pl.FP32, pl.Mem.Vec] = pl.tile.full([64, 32], dtype=pl.FP32, value=1.0)
            stage_15_v72: pl.Tile[[64, 32], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_15_v71, stage_15_v70)
            stage_15_v73: pl.Tile[[64, 32], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_15_v74: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(stage_15_v72, 0, 1, stage_15_v73)
            stage_15_v75: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_15_v67, stage_15_v74)
            stage_15_v76: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_15_v75, target_type=pl.INT32, mode='round')
            stage_15_v77: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            stage_15_v78: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_15_v53, stage_15_v76, stage_15_v77)
            stage_15_v79: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.mul(stage_15_v53, stage_15_v54)
            stage_15_v80: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_15_v78, stage_15_v55)
            stage_15_v81: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.add(stage_15_v79, stage_15_v80)
            stage_15_v82: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.cast(stage_15_v81, target_type=pl.BF16, mode='rint')
            stage_15_v83: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v9, 64])] = pl.tile.set_validshape(stage_15_v82, stage_15_v9, 64)
            stage_15_v84 = pl.tile.store(stage_15_v83, [stage_15_v10, 448], temporary_17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_15 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_15)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_16 = 22896 + core_id * 6
    mapped_block_16 = pl.cast(pl.tensor.read(records, [record_16]), pl.INDEX)
    block_16 = pl.max(mapped_block_16, 0)
    if mapped_block_16 >= 0:
        subslot_16 = pl.cast(pl.tensor.read(records, [record_16 + 1]), pl.INDEX)
        done_16 = pl.tensor.read(records, [record_16 + 2])
        wait_base_16 = pl.cast(pl.tensor.read(records, [record_16 + 3]), pl.INDEX)
        wait_count_16 = pl.cast(pl.tensor.read(records, [record_16 + 4]), pl.INDEX)
        publish_16 = pl.tensor.read(records, [record_16 + 5])
        for dependency_16 in pl.range(wait_count_16):
            wait_index_16 = (wait_base_16 + dependency_16) * 2
            other_16 = pl.cast(pl.tensor.read(waits, [wait_index_16]), pl.INDEX)
            required_16 = pl.tensor.read(waits, [wait_index_16 + 1])
            target_16 = generation_base + required_16
            flag_16 = other_16 * 64
            pl.system.cacheinvalid(progress, [1], [flag_16])
            initial_16 = pl.tensor.read(progress, [flag_16])
            for (ready_16,) in pl.while_(init_values=(initial_16,)):
                pl.cond(ready_16 < target_16)
                pl.system.cacheinvalid(progress, [1], [flag_16])
                after_16 = pl.yield_(pl.tensor.read(progress, [flag_16]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_16_v8: pl.Scalar[pl.INDEX] = block_16
        stage_16_v9: pl.Scalar[pl.INDEX] = 4
        if 384 <= 144 and 144 % 384 == 0:
            stage_16_v11: pl.Scalar[pl.INDEX] = 64
            stage_16_v12: pl.Scalar[pl.INDEX] = pl.yield_(stage_16_v11)
        else:
            if 96 <= 144 and 144 % 96 == 0:
                stage_16_v16: pl.Scalar[pl.INDEX] = 16
                stage_16_v17: pl.Scalar[pl.INDEX] = pl.yield_(stage_16_v16)
            else:
                stage_16_v17: pl.Scalar[pl.INDEX] = pl.yield_(stage_16_v9)
            stage_16_v12: pl.Scalar[pl.INDEX] = pl.yield_(stage_16_v17)
        for stage_16_v10 in pl.range(stage_16_v8, (144 + 7) // 8 * (64 // stage_16_v12), 48):
            stage_16_v13: pl.Scalar[pl.INDEX] = stage_16_v10 % (64 // stage_16_v12) * stage_16_v12
            stage_16_v14: pl.Scalar[pl.INDEX] = stage_16_v10 // (64 // stage_16_v12) * 8
            stage_16_v15: pl.Scalar[pl.INDEX] = 0 + stage_16_v14
            if stage_16_v14 + 8 <= 144:
                stage_16_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_23, [stage_16_v14, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_16_v19: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_16_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_16_v20: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_20, [stage_16_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_16_v21: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_21, [stage_16_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_16_v22: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                stage_16_v23: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_16_v22, target_type=pl.FP32, mode='round')
                stage_16_v24: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_16_v23, 0.015625)
                stage_16_v25: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_16_v24, target_type=pl.INT32, mode='trunc')
                stage_16_v26: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_16_v25, target_type=pl.FP32, mode='round')
                stage_16_v27: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_16_v26, 64.0)
                stage_16_v28: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v21, [1, 512])
                stage_16_v29: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_16_v28, target_type=pl.FP32, mode='round')
                stage_16_v30: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_16_v29, stage_16_v27)
                stage_16_v31: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_16_v30, target_type=pl.INT32, mode='round')
                stage_16_v32: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 512], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_16_v33 in pl.pipeline(stage_16_v12, stage=2):
                    stage_16_v62: pl.Scalar[pl.INDEX] = stage_16_v13 + stage_16_v33
                    stage_16_v63: pl.Scalar[pl.INDEX] = stage_16_v62 * 512
                    stage_16_v64: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_25, [stage_16_v14, stage_16_v63], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    stage_16_v65: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wq_b_scale, [stage_16_v63], [512], [512], target_memory=pl.Mem.Vec)
                    stage_16_v66: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v65, [1, 512])
                    stage_16_v67: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_16_v64, target_type=pl.FP32, mode='none')
                    stage_16_v68: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_16_v67, stage_16_v18)
                    stage_16_v69: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_16_v68, stage_16_v66)
                    stage_16_v70: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_16_v69, stage_16_v69)
                    stage_16_v71: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_16_v72: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_16_v70, stage_16_v71)
                    stage_16_v73: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v72, [1, 8])
                    stage_16_v74: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_16_v73, 0.001953125)
                    stage_16_v75: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_16_v74, 1e-06)
                    stage_16_v76: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_16_v77: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(stage_16_v75, stage_16_v76)
                    stage_16_v78: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v77, [8, 1])
                    stage_16_v79: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_16_v69, [8, 448], [0, 0])
                    stage_16_v80: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_16_v79, stage_16_v78)
                    stage_16_v81: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_16_v80, target_type=pl.BF16, mode='rint')
                    stage_16_v82 = pl.tile.store(stage_16_v81, [stage_16_v15, stage_16_v63], view_40)
                    stage_16_v83: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_16_v69, [8, 64], [0, 448])
                    stage_16_v84: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_16_v83, stage_16_v78)
                    stage_16_v85: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v84, [1, 512])
                    stage_16_v86: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_16_v85, stage_16_v31, stage_16_v32)
                    stage_16_v87: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v86, [8, 64])
                    stage_16_v88: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_16_v84, stage_16_v19)
                    stage_16_v89: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_16_v87, stage_16_v20)
                    stage_16_v90: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_16_v88, stage_16_v89)
                    stage_16_v91: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_16_v90, target_type=pl.BF16, mode='rint')
                    stage_16_v92 = pl.tile.store(stage_16_v91, [stage_16_v15, stage_16_v63 + 448], view_40)
            else:
                stage_16_v34: pl.Scalar[pl.INDEX] = 144 - stage_16_v14
                stage_16_v35: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_23, [stage_16_v14, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_16_v36: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_16_v34, 64])] = pl.tile.load(arg_freqs_cos, [stage_16_v15, 0], [8, 64], [stage_16_v34, 64], target_memory=pl.Mem.Vec)
                stage_16_v37: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_16_v34, 64])] = pl.tile.load(temporary_20, [stage_16_v15, 0], [8, 64], [stage_16_v34, 64], target_memory=pl.Mem.Vec)
                stage_16_v38: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                stage_16_v39: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                stage_16_v40: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_16_v39, target_type=pl.FP32, mode='round')
                stage_16_v41: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_16_v38, stage_16_v40)
                stage_16_v42: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_16_v41, 0.5)
                stage_16_v43: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_16_v42, target_type=pl.INT32, mode='trunc')
                stage_16_v44: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_16_v43, target_type=pl.FP32, mode='round')
                stage_16_v45: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_16_v44, 2.0)
                stage_16_v46: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_16_v41, stage_16_v45)
                stage_16_v47: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_16_v41, 1.0)
                stage_16_v48: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_16_v46, 2.0)
                stage_16_v49: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_16_v47, stage_16_v48)
                stage_16_v50: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False)
                stage_16_v51: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_16_v50, target_type=pl.FP32, mode='round')
                stage_16_v52: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_16_v51, 64.0)
                stage_16_v53: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                stage_16_v54: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_16_v53, stage_16_v52)
                stage_16_v55: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_16_v56: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(stage_16_v54, 0, 1, stage_16_v55)
                stage_16_v57: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_16_v49, stage_16_v56)
                stage_16_v58: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_16_v57, target_type=pl.INT32, mode='round')
                stage_16_v59: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_16_v60: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_16_v61 in pl.range(4):
                    stage_16_v93: pl.Scalar[pl.INDEX] = stage_16_v13 + stage_16_v61
                    stage_16_v94: pl.Scalar[pl.INDEX] = stage_16_v93 * 512
                    stage_16_v95: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_25, [stage_16_v14, stage_16_v94], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    stage_16_v96: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wq_b_scale, [stage_16_v94], [512], [512], target_memory=pl.Mem.Vec)
                    stage_16_v97: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v96, [1, 512])
                    stage_16_v98: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_16_v95, target_type=pl.FP32, mode='none')
                    stage_16_v99: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_16_v98, stage_16_v35)
                    stage_16_v100: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_16_v99, stage_16_v97)
                    stage_16_v101: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_16_v100, stage_16_v100)
                    stage_16_v102: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_16_v101, stage_16_v59)
                    stage_16_v103: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v102, [1, 8])
                    stage_16_v104: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_16_v103, 0.001953125)
                    stage_16_v105: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v104, [8, 1])
                    stage_16_v106: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v105, [1, 8])
                    stage_16_v107: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_16_v106, 1e-06)
                    stage_16_v108: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v107, [8, 1])
                    stage_16_v109: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v108, [1, 8])
                    stage_16_v110: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_16_v109)
                    stage_16_v111: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v110, [8, 1])
                    stage_16_v112: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v111, [1, 8])
                    stage_16_v113: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_16_v112)
                    stage_16_v114: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_16_v113, [8, 1])
                    stage_16_v115: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_16_v100, [8, 448], [0, 0])
                    stage_16_v116: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_16_v115, stage_16_v114)
                    stage_16_v117: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_16_v116, target_type=pl.BF16, mode='rint')
                    stage_16_v118: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_16_v34, 448])] = pl.tile.set_validshape(stage_16_v117, stage_16_v34, 448)
                    stage_16_v119 = pl.tile.store(stage_16_v118, [stage_16_v15, stage_16_v94], view_40)
                    stage_16_v120: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_16_v100, [8, 64], [0, 448])
                    stage_16_v121: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_16_v120, stage_16_v114)
                    stage_16_v122: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_16_v121, stage_16_v58, stage_16_v60)
                    stage_16_v123: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_16_v34, 64])] = pl.tile.set_validshape(stage_16_v121, stage_16_v34, 64)
                    stage_16_v124: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_16_v34, 64])] = pl.tile.set_validshape(stage_16_v122, stage_16_v34, 64)
                    stage_16_v125: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_16_v34, 64])] = pl.tile.mul(stage_16_v123, stage_16_v36)
                    stage_16_v126: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_16_v34, 64])] = pl.tile.mul(stage_16_v124, stage_16_v37)
                    stage_16_v127: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_16_v34, 64])] = pl.tile.add(stage_16_v125, stage_16_v126)
                    stage_16_v128: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_16_v34, 64])] = pl.tile.cast(stage_16_v127, target_type=pl.BF16, mode='rint')
                    stage_16_v129: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_16_v34, 64])] = pl.tile.set_validshape(stage_16_v128, stage_16_v34, 64)
                    stage_16_v130 = pl.tile.store(stage_16_v129, [stage_16_v15, stage_16_v94 + 448], view_40)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_16 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_16)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_17 = 23328 + core_id * 6
    mapped_block_17 = pl.cast(pl.tensor.read(records, [record_17]), pl.INDEX)
    block_17 = pl.max(mapped_block_17, 0)
    if mapped_block_17 >= 0:
        subslot_17 = pl.cast(pl.tensor.read(records, [record_17 + 1]), pl.INDEX)
        done_17 = pl.tensor.read(records, [record_17 + 2])
        wait_base_17 = pl.cast(pl.tensor.read(records, [record_17 + 3]), pl.INDEX)
        wait_count_17 = pl.cast(pl.tensor.read(records, [record_17 + 4]), pl.INDEX)
        publish_17 = pl.tensor.read(records, [record_17 + 5])
        for dependency_17 in pl.range(wait_count_17):
            wait_index_17 = (wait_base_17 + dependency_17) * 2
            other_17 = pl.cast(pl.tensor.read(waits, [wait_index_17]), pl.INDEX)
            required_17 = pl.tensor.read(waits, [wait_index_17 + 1])
            target_17 = generation_base + required_17
            flag_17 = other_17 * 64
            pl.system.cacheinvalid(progress, [1], [flag_17])
            initial_17 = pl.tensor.read(progress, [flag_17])
            for (ready_17,) in pl.while_(init_values=(initial_17,)):
                pl.cond(ready_17 < target_17)
                pl.system.cacheinvalid(progress, [1], [flag_17])
                after_17 = pl.yield_(pl.tensor.read(progress, [flag_17]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_17_v8: pl.Scalar[pl.INDEX] = block_17
        stage_17_v9: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        stage_17_v10: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_17_v11: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_17_v10, target_type=pl.FP32, mode='round')
        stage_17_v12: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_17_v9, stage_17_v11)
        stage_17_v13: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_17_v12, 0.5)
        stage_17_v14: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_17_v13, target_type=pl.INT32, mode='trunc')
        stage_17_v15: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_17_v14, target_type=pl.FP32, mode='round')
        stage_17_v16: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_17_v15, 2.0)
        stage_17_v17: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_17_v12, stage_17_v16)
        stage_17_v18: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_17_v12, 1.0)
        stage_17_v19: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_17_v17, 2.0)
        stage_17_v20: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_17_v18, stage_17_v19)
        stage_17_v21: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_17_v20, target_type=pl.INT32, mode='round')
        for (stage_17_v22, (stage_17_v69,)) in pl.range(stage_17_v8, 9, 2, init_values=(temporary_33,)):
            stage_17_v23: pl.Scalar[pl.INDEX] = stage_17_v22 * 16
            stage_17_v24: pl.Scalar[pl.INDEX] = pl.min(144 - stage_17_v23, 16)
            stage_17_v25: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
            stage_17_v26: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
            for (stage_17_v70, (stage_17_v84, stage_17_v85)) in pl.range(stage_17_v24, init_values=(stage_17_v25, stage_17_v26)):
                stage_17_v71: pl.Scalar[pl.INDEX] = stage_17_v23 + stage_17_v70
                stage_17_v72: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_17_v71])
                if (stage_17_v72 + 1) % 4 == 0:
                    stage_17_v86: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_15, [stage_17_v71 // 6])
                    stage_17_v87: pl.Scalar[pl.INDEX] = pl.cast(stage_17_v86, pl.INDEX) + pl.cast((stage_17_v72 + 1) // 4, pl.INDEX)
                    stage_17_v88: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_inner_freqs_cos, 0)
                    if stage_17_v87 < stage_17_v88:
                        stage_17_v102: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather_row(stage_17_v84, arg_inner_freqs_cos, [stage_17_v70, 0], [stage_17_v87, 0], [1, 64], transpose=False)
                        stage_17_v103: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather_row(stage_17_v85, arg_inner_freqs_sin, [stage_17_v70, 0], [stage_17_v87, 0], [1, 64], transpose=False)
                        (stage_17_v106, stage_17_v107) = pl.yield_(stage_17_v102, stage_17_v103)
                    else:
                        (stage_17_v106, stage_17_v107) = pl.yield_(stage_17_v84, stage_17_v85)
                    (stage_17_v104, stage_17_v105) = pl.yield_(stage_17_v106, stage_17_v107)
                else:
                    (stage_17_v104, stage_17_v105) = pl.yield_(stage_17_v84, stage_17_v85)
                (stage_17_v89, stage_17_v90) = pl.yield_(stage_17_v104, stage_17_v105)
            stage_17_v27: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_17_v89
            stage_17_v28: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_17_v90
            stage_17_v29: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_36, [stage_17_v23, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            stage_17_v30: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_36, [stage_17_v23, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            stage_17_v31: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_17_v29, stage_17_v29)
            stage_17_v32: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_17_v30, stage_17_v30)
            stage_17_v33: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_17_v31, stage_17_v32)
            stage_17_v34: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_17_v35: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_17_v33, stage_17_v34)
            stage_17_v36: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v35, [1, 16])
            stage_17_v37: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_17_v36, 0.0078125)
            stage_17_v38: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v37, [16, 1])
            stage_17_v39: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v38, [1, 16])
            stage_17_v40: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_17_v39, 1e-06)
            stage_17_v41: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v40, [16, 1])
            stage_17_v42: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v41, [1, 16])
            stage_17_v43: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_17_v42)
            stage_17_v44: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v43, [16, 1])
            stage_17_v45: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v44, [1, 16])
            stage_17_v46: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_17_v45)
            stage_17_v47: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v46, [16, 1])
            stage_17_v48: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_36, [stage_17_v23, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            stage_17_v49: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_76, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
            stage_17_v50: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_17_v49, target_type=pl.FP32, mode='round')
            stage_17_v51: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_17_v48, stage_17_v47)
            stage_17_v52: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_17_v51, stage_17_v50)
            stage_17_v53: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_17_v52, target_type=pl.BF16, mode='rint')
            stage_17_v54: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_36, [stage_17_v23, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            stage_17_v55: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_76, [0, 64], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
            stage_17_v56: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_17_v55, target_type=pl.FP32, mode='round')
            stage_17_v57: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_17_v54, stage_17_v47)
            stage_17_v58: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_17_v57, stage_17_v56)
            stage_17_v59: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for (stage_17_v73, (stage_17_v91,)) in pl.range(16, init_values=(stage_17_v59,)):
                stage_17_v74: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_17_v58, [1, 64], [stage_17_v73, 0], [1, 64])
                stage_17_v75: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_17_v21, [1, 64], [stage_17_v73, 0], [1, 64])
                stage_17_v76: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                stage_17_v77: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_17_v74, stage_17_v75, stage_17_v76)
                stage_17_v78: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_17_v91, stage_17_v77, [stage_17_v73, 0])
                stage_17_v79: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_17_v78)
            stage_17_v60: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_17_v79
            stage_17_v61: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_17_v17, 2.0)
            stage_17_v62: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_17_v61, 1.0)
            stage_17_v63: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_17_v28, stage_17_v62)
            stage_17_v64: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_17_v58, stage_17_v27)
            stage_17_v65: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_17_v60, stage_17_v63)
            stage_17_v66: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_17_v64, stage_17_v65)
            stage_17_v67: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_17_v66, target_type=pl.BF16, mode='rint')
            for (stage_17_v80, (stage_17_v92,)) in pl.range(stage_17_v24, init_values=(stage_17_v69,)):
                stage_17_v81: pl.Scalar[pl.INDEX] = stage_17_v23 + stage_17_v80
                stage_17_v82: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_17_v81])
                if (stage_17_v82 + 1) % 4 == 0:
                    stage_17_v93: pl.Scalar[pl.INDEX] = stage_17_v81 // 6
                    stage_17_v94: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_17_v93 * 6])
                    stage_17_v95: pl.Scalar[pl.INDEX] = 3 - stage_17_v94 % 4
                    stage_17_v96: pl.Scalar[pl.INDEX] = stage_17_v93 * 2 + (stage_17_v81 % 6 - stage_17_v95) // 4
                    stage_17_v97: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_17_v53, [1, 64], [stage_17_v80, 0])
                    stage_17_v98 = pl.tile.store(stage_17_v97, [stage_17_v96, 0], stage_17_v92)
                    stage_17_v99: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_17_v67, [1, 64], [stage_17_v80, 0])
                    stage_17_v100 = pl.tile.store(stage_17_v99, [stage_17_v96, 64], stage_17_v98)
                    stage_17_v101 = pl.yield_(stage_17_v100)
                else:
                    stage_17_v101 = pl.yield_(stage_17_v92)
                stage_17_v83 = pl.yield_(stage_17_v101)
            stage_17_v68 = pl.yield_(stage_17_v83)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_17 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_17)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_18 = 23760 + core_id * 6
    mapped_block_18 = pl.cast(pl.tensor.read(records, [record_18]), pl.INDEX)
    block_18 = pl.max(mapped_block_18, 0)
    if mapped_block_18 >= 0:
        subslot_18 = pl.cast(pl.tensor.read(records, [record_18 + 1]), pl.INDEX)
        done_18 = pl.tensor.read(records, [record_18 + 2])
        wait_base_18 = pl.cast(pl.tensor.read(records, [record_18 + 3]), pl.INDEX)
        wait_count_18 = pl.cast(pl.tensor.read(records, [record_18 + 4]), pl.INDEX)
        publish_18 = pl.tensor.read(records, [record_18 + 5])
        for dependency_18 in pl.range(wait_count_18):
            wait_index_18 = (wait_base_18 + dependency_18) * 2
            other_18 = pl.cast(pl.tensor.read(waits, [wait_index_18]), pl.INDEX)
            required_18 = pl.tensor.read(waits, [wait_index_18 + 1])
            target_18 = generation_base + required_18
            flag_18 = other_18 * 64
            pl.system.cacheinvalid(progress, [1], [flag_18])
            initial_18 = pl.tensor.read(progress, [flag_18])
            for (ready_18,) in pl.while_(init_values=(initial_18,)):
                pl.cond(ready_18 < target_18)
                pl.system.cacheinvalid(progress, [1], [flag_18])
                after_18 = pl.yield_(pl.tensor.read(progress, [flag_18]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_18_v12: pl.Scalar[pl.INDEX] = block_18
        stage_18_v13: pl.Scalar[pl.INDEX] = stage_18_v12 * 16
        stage_18_v14: pl.Scalar[pl.INDEX] = pl.min(144 - stage_18_v13, 16)
        stage_18_v15: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        stage_18_v16: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
        for (stage_18_v71, (stage_18_v93, stage_18_v94)) in pl.range(stage_18_v14, init_values=(stage_18_v15, stage_18_v16)):
            stage_18_v72: pl.Scalar[pl.INDEX] = stage_18_v13 + stage_18_v71
            stage_18_v73: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_18_v72])
            if (stage_18_v73 + 1) % 4 == 0:
                stage_18_v95: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_14, [stage_18_v72 // 6])
                stage_18_v96: pl.Scalar[pl.INDEX] = pl.cast(stage_18_v95, pl.INDEX) + pl.cast((stage_18_v73 + 1) // 4, pl.INDEX)
                stage_18_v97: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_cmp_freqs_cos, 0)
                if stage_18_v96 < stage_18_v97:
                    stage_18_v112: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather_row(stage_18_v93, arg_cmp_freqs_cos, [stage_18_v71, 0], [stage_18_v96, 0], [1, 64], transpose=False)
                    stage_18_v113: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather_row(stage_18_v94, arg_cmp_freqs_sin, [stage_18_v71, 0], [stage_18_v96, 0], [1, 64], transpose=False)
                    (stage_18_v123, stage_18_v124) = pl.yield_(stage_18_v112, stage_18_v113)
                else:
                    (stage_18_v123, stage_18_v124) = pl.yield_(stage_18_v93, stage_18_v94)
                (stage_18_v114, stage_18_v115) = pl.yield_(stage_18_v123, stage_18_v124)
            else:
                (stage_18_v114, stage_18_v115) = pl.yield_(stage_18_v93, stage_18_v94)
            (stage_18_v98, stage_18_v99) = pl.yield_(stage_18_v114, stage_18_v115)
        stage_18_v17: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_18_v98
        stage_18_v18: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_18_v99
        stage_18_v19: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_28, [stage_18_v13, 0], [16, 256], [16, 256], target_memory=pl.Mem.Vec)
        stage_18_v20: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_28, [stage_18_v13, 256], [16, 256], [16, 256], target_memory=pl.Mem.Vec)
        stage_18_v21: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_18_v19, stage_18_v19)
        stage_18_v22: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_18_v20, stage_18_v20)
        stage_18_v23: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_18_v21, stage_18_v22)
        stage_18_v24: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_18_v23, [16, 128], [0, 0])
        stage_18_v25: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_18_v23, [16, 128], [0, 128])
        stage_18_v26: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_18_v24, stage_18_v25)
        stage_18_v27: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_18_v26, [16, 64], [0, 0])
        stage_18_v28: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_18_v26, [16, 64], [0, 64])
        stage_18_v29: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_18_v27, stage_18_v28)
        stage_18_v30: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_18_v31: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_18_v29, stage_18_v30)
        stage_18_v32: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v31, [1, 16])
        stage_18_v33: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_18_v32, 0.001953125)
        stage_18_v34: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v33, [16, 1])
        stage_18_v35: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v34, [1, 16])
        stage_18_v36: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_18_v35, 1e-06)
        stage_18_v37: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v36, [16, 1])
        stage_18_v38: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v37, [1, 16])
        stage_18_v39: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_18_v38)
        stage_18_v40: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v39, [16, 1])
        stage_18_v41: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v40, [1, 16])
        stage_18_v42: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_18_v41)
        stage_18_v43: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v42, [16, 1])
        for (stage_18_v74, (stage_18_v100,)) in pl.range(0, 448, 64, init_values=(temporary_31,)):
            stage_18_v75: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_28, [stage_18_v13, stage_18_v74], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            stage_18_v76: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_60, [0, stage_18_v74], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
            stage_18_v77: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_18_v76, target_type=pl.FP32, mode='round')
            stage_18_v78: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_18_v75, stage_18_v43)
            stage_18_v79: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_18_v78, stage_18_v77)
            stage_18_v80 = pl.tile.store(stage_18_v79, [stage_18_v13, stage_18_v74], stage_18_v100)
            stage_18_v81 = pl.yield_(stage_18_v80)
        stage_18_v44: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_28, [stage_18_v13, 448], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_18_v45: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_60, [0, 448], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
        stage_18_v46: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_18_v45, target_type=pl.FP32, mode='round')
        stage_18_v47: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_18_v44, stage_18_v43)
        stage_18_v48: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_18_v47, stage_18_v46)
        stage_18_v49: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        stage_18_v50: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_18_v51: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_18_v50, target_type=pl.FP32, mode='round')
        stage_18_v52: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_18_v49, stage_18_v51)
        stage_18_v53: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_18_v52, 0.5)
        stage_18_v54: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_18_v53, target_type=pl.INT32, mode='trunc')
        stage_18_v55: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_18_v54, target_type=pl.FP32, mode='round')
        stage_18_v56: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_18_v55, 2.0)
        stage_18_v57: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_18_v52, stage_18_v56)
        stage_18_v58: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_18_v52, 1.0)
        stage_18_v59: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_18_v57, 2.0)
        stage_18_v60: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_18_v58, stage_18_v59)
        stage_18_v61: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_18_v60, target_type=pl.INT32, mode='round')
        stage_18_v62: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        for (stage_18_v82, (stage_18_v101,)) in pl.range(16, init_values=(stage_18_v62,)):
            stage_18_v83: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_18_v48, [1, 64], [stage_18_v82, 0], [1, 64])
            stage_18_v84: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_18_v61, [1, 64], [stage_18_v82, 0], [1, 64])
            stage_18_v85: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            stage_18_v86: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_18_v83, stage_18_v84, stage_18_v85)
            stage_18_v87: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_18_v101, stage_18_v86, [stage_18_v82, 0])
            stage_18_v88: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_18_v87)
        stage_18_v63: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_18_v88
        stage_18_v64: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_18_v57, 2.0)
        stage_18_v65: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_18_v64, 1.0)
        stage_18_v66: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_18_v18, stage_18_v65)
        stage_18_v67: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_18_v48, stage_18_v17)
        stage_18_v68: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_18_v63, stage_18_v66)
        stage_18_v69: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_18_v67, stage_18_v68)
        stage_18_v70 = pl.tile.store(stage_18_v69, [stage_18_v13, 448], stage_18_v81)
        for (stage_18_v89, (stage_18_v102, stage_18_v103)) in pl.range(stage_18_v14, init_values=(view_61, temporary_27)):
            stage_18_v90: pl.Scalar[pl.INDEX] = stage_18_v13 + stage_18_v89
            stage_18_v91: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_18_v90])
            stage_18_v92: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_seq_lens, [stage_18_v90 // 6])
            if 0 < pl.cast(stage_18_v92, pl.INDEX) and (stage_18_v91 + 1) % 4 == 0:
                stage_18_v104: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_14, [stage_18_v90 // 6])
                stage_18_v105: pl.Scalar[pl.INDEX] = pl.cast(stage_18_v104, pl.INDEX) + pl.cast((stage_18_v91 + 1) // 4, pl.INDEX)
                stage_18_v106: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_cmp_slot_mapping, 0)
                stage_18_v107: pl.Scalar[pl.INDEX] = pl.min(stage_18_v105, stage_18_v106 - 1)
                stage_18_v108: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_slot_mapping, [stage_18_v107, 0])
                stage_18_v109: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_slot_mapping, [stage_18_v107, 1])
                if stage_18_v105 < stage_18_v106 and 0 <= pl.cast(stage_18_v108, pl.INDEX) and (0 <= pl.cast(stage_18_v109, pl.INDEX)):
                    stage_18_v116: pl.Scalar[pl.INDEX] = pl.cast(stage_18_v108, pl.INDEX) * 32 + pl.cast(stage_18_v109, pl.INDEX)
                    stage_18_v117: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(stage_18_v70, [stage_18_v90, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    stage_18_v118 = pl.tile.store(stage_18_v117, [stage_18_v90, 0], stage_18_v103)
                    stage_18_v119: pl.Tile[[1, 512], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_18_v117, target_type=pl.BF16, mode='rint')
                    stage_18_v120 = pl.tile.store(stage_18_v119, [stage_18_v116, 0], stage_18_v102)
                    (stage_18_v125, stage_18_v126) = pl.yield_(stage_18_v120, stage_18_v118)
                else:
                    (stage_18_v125, stage_18_v126) = pl.yield_(stage_18_v102, stage_18_v103)
                (stage_18_v121, stage_18_v122) = pl.yield_(stage_18_v125, stage_18_v126)
            else:
                (stage_18_v121, stage_18_v122) = pl.yield_(stage_18_v102, stage_18_v103)
            (stage_18_v110, stage_18_v111) = pl.yield_(stage_18_v121, stage_18_v122)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_18 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_18)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_19 = 24192 + core_id * 6
    mapped_block_19 = pl.cast(pl.tensor.read(records, [record_19]), pl.INDEX)
    block_19 = pl.max(mapped_block_19, 0)
    if mapped_block_19 >= 0:
        subslot_19 = pl.cast(pl.tensor.read(records, [record_19 + 1]), pl.INDEX)
        done_19 = pl.tensor.read(records, [record_19 + 2])
        wait_base_19 = pl.cast(pl.tensor.read(records, [record_19 + 3]), pl.INDEX)
        wait_count_19 = pl.cast(pl.tensor.read(records, [record_19 + 4]), pl.INDEX)
        publish_19 = pl.tensor.read(records, [record_19 + 5])
        for dependency_19 in pl.range(wait_count_19):
            wait_index_19 = (wait_base_19 + dependency_19) * 2
            other_19 = pl.cast(pl.tensor.read(waits, [wait_index_19]), pl.INDEX)
            required_19 = pl.tensor.read(waits, [wait_index_19 + 1])
            target_19 = generation_base + required_19
            flag_19 = other_19 * 64
            pl.system.cacheinvalid(progress, [1], [flag_19])
            initial_19 = pl.tensor.read(progress, [flag_19])
            for (ready_19,) in pl.while_(init_values=(initial_19,)):
                pl.cond(ready_19 < target_19)
                pl.system.cacheinvalid(progress, [1], [flag_19])
                after_19 = pl.yield_(pl.tensor.read(progress, [flag_19]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_19_v9: pl.Scalar[pl.INDEX] = block_19
        stage_19_v10: pl.Scalar[pl.INDEX] = stage_19_v9 * 16
        stage_19_v11: pl.Scalar[pl.INDEX] = pl.min(48 - stage_19_v10, 16)
        stage_19_v12: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_38, [stage_19_v10, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec)
        stage_19_v13: pl.Tile[[16, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_19_v12, target_type=pl.BF16, mode='rint')
        stage_19_v14: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_19_v13, target_type=pl.FP32, mode='round')
        stage_19_v15: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_19_v14, 0.08838834764831845)
        stage_19_v16: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_19_v15)
        stage_19_v17: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_19_v18: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_19_v16, stage_19_v17)
        stage_19_v19: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_19_v18, [1, 16])
        stage_19_v20: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0001)
        stage_19_v21: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_19_v19, stage_19_v20)
        stage_19_v22: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=127.0)
        stage_19_v23: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_19_v22, stage_19_v21)
        stage_19_v24: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_19_v23)
        stage_19_v25: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_19_v24, [16, 1])
        stage_19_v26: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_19_v23, [16, 1])
        stage_19_v27 = pl.tile.store(stage_19_v25, [stage_19_v10, 0], temporary_37)
        stage_19_v28: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_19_v15, stage_19_v26)
        stage_19_v29: pl.Tile[[16, 128], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_19_v28, target_type=pl.INT32, mode='rint')
        stage_19_v30: pl.Tile[[16, 128], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_19_v29, target_type=pl.FP16, mode='round')
        stage_19_v31: pl.Tile[[16, 128], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_19_v30, target_type=pl.INT8, mode='trunc')
        for (stage_19_v32, (stage_19_v38, stage_19_v39)) in pl.range(stage_19_v11, init_values=(arg_idx_native_kv_cache, temporary_32)):
            stage_19_v33: pl.Scalar[pl.INDEX] = stage_19_v10 + stage_19_v32
            stage_19_v34: pl.Scalar[pl.INDEX] = stage_19_v33 // 2
            stage_19_v35: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_19_v34 * 6])
            stage_19_v36: pl.Scalar[pl.INDEX] = stage_19_v33 % 2 * 4 - stage_19_v35 % 4 + 3
            stage_19_v37: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_19_v34])
            if 0 < pl.cast(stage_19_v37, pl.INDEX) and stage_19_v36 < 6:
                stage_19_v40: pl.Scalar[pl.INDEX] = stage_19_v34 * 6 + stage_19_v36
                stage_19_v41: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_19_v40])
                stage_19_v42: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_15, [stage_19_v34])
                stage_19_v43: pl.Scalar[pl.INDEX] = pl.cast(stage_19_v42, pl.INDEX) + pl.cast((stage_19_v41 + 1) // 4, pl.INDEX)
                stage_19_v44: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_idx_slot_mapping, 0)
                stage_19_v45: pl.Scalar[pl.INDEX] = pl.min(stage_19_v43, stage_19_v44 - 1)
                stage_19_v46: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_slot_mapping, [stage_19_v45, 0])
                stage_19_v47: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_slot_mapping, [stage_19_v45, 1])
                if stage_19_v43 < stage_19_v44 and 0 <= pl.cast(stage_19_v46, pl.INDEX) and (0 <= pl.cast(stage_19_v47, pl.INDEX)):
                    stage_19_v50: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_19_v15, [1, 128], [stage_19_v32, 0])
                    stage_19_v51 = pl.tile.store(stage_19_v50, [stage_19_v40, 0], stage_19_v39)
                    stage_19_v52: pl.Scalar[pl.INDEX] = pl.cast(stage_19_v46, pl.INDEX)
                    stage_19_v53: pl.Scalar[pl.INDEX] = pl.cast(stage_19_v47, pl.INDEX) * 128
                    stage_19_v54: pl.Tile[[1, 128], pl.INT8, pl.Mem.Vec] = pl.tile.slice(stage_19_v31, [1, 128], [stage_19_v32, 0])
                    stage_19_v55 = pl.tile.store(stage_19_v54, [stage_19_v52, stage_19_v53], stage_19_v38)
                    (stage_19_v58, stage_19_v59) = pl.yield_(stage_19_v55, stage_19_v51)
                else:
                    (stage_19_v58, stage_19_v59) = pl.yield_(stage_19_v38, stage_19_v39)
                (stage_19_v56, stage_19_v57) = pl.yield_(stage_19_v58, stage_19_v59)
            else:
                (stage_19_v56, stage_19_v57) = pl.yield_(stage_19_v38, stage_19_v39)
            (stage_19_v48, stage_19_v49) = pl.yield_(stage_19_v56, stage_19_v57)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_19 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_19)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_20 = 24624 + core_id * 6
    mapped_block_20 = pl.cast(pl.tensor.read(records, [record_20]), pl.INDEX)
    block_20 = pl.max(mapped_block_20, 0)
    if mapped_block_20 >= 0:
        subslot_20 = pl.cast(pl.tensor.read(records, [record_20 + 1]), pl.INDEX)
        done_20 = pl.tensor.read(records, [record_20 + 2])
        wait_base_20 = pl.cast(pl.tensor.read(records, [record_20 + 3]), pl.INDEX)
        wait_count_20 = pl.cast(pl.tensor.read(records, [record_20 + 4]), pl.INDEX)
        publish_20 = pl.tensor.read(records, [record_20 + 5])
        for dependency_20 in pl.range(wait_count_20):
            wait_index_20 = (wait_base_20 + dependency_20) * 2
            other_20 = pl.cast(pl.tensor.read(waits, [wait_index_20]), pl.INDEX)
            required_20 = pl.tensor.read(waits, [wait_index_20 + 1])
            target_20 = generation_base + required_20
            flag_20 = other_20 * 64
            pl.system.cacheinvalid(progress, [1], [flag_20])
            initial_20 = pl.tensor.read(progress, [flag_20])
            for (ready_20,) in pl.while_(init_values=(initial_20,)):
                pl.cond(ready_20 < target_20)
                pl.system.cacheinvalid(progress, [1], [flag_20])
                after_20 = pl.yield_(pl.tensor.read(progress, [flag_20]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_20_v7: pl.Scalar[pl.INDEX] = block_20
        stage_20_v8: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        stage_20_v9: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_20_v10: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_20_v9, target_type=pl.FP32, mode='round')
        stage_20_v11: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_20_v8, stage_20_v10)
        stage_20_v12: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_20_v11, 0.5)
        stage_20_v13: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_20_v12, target_type=pl.INT32, mode='trunc')
        stage_20_v14: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_20_v13, target_type=pl.FP32, mode='round')
        stage_20_v15: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_20_v14, 2.0)
        stage_20_v16: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_20_v11, stage_20_v15)
        stage_20_v17: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_20_v11, 1.0)
        stage_20_v18: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_20_v16, 2.0)
        stage_20_v19: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_20_v17, stage_20_v18)
        stage_20_v20: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_20_v19, target_type=pl.INT32, mode='round')
        for (stage_20_v21, (stage_20_v25,)) in pl.range(stage_20_v7, 288, 48, init_values=(view_85,)):
            stage_20_v22: pl.Scalar[pl.INDEX] = stage_20_v21 % 16 * 4
            stage_20_v23: pl.Scalar[pl.INDEX] = stage_20_v21 // 16 * 8
            if stage_20_v23 + 8 <= 144:
                stage_20_v26: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_19, [stage_20_v23, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_20_v27: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_20_v23, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_20_v28: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_13, [stage_20_v23, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_20_v29: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                stage_20_v30: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                stage_20_v31: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.rems(stage_20_v29, pl.const(2, pl.INT32), stage_20_v30)
                stage_20_v32: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_20_v31, pl.const(2, pl.INT32))
                stage_20_v33: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.sub(stage_20_v29, stage_20_v32)
                stage_20_v34: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_20_v33, pl.const(65, pl.INT32))
                stage_20_v35: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                stage_20_v36: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.col_expand(stage_20_v35, stage_20_v34)
                stage_20_v37: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False)
                stage_20_v38: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_20_v37, pl.const(128, pl.INT32))
                stage_20_v39: pl.Tile[[8, 1], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_20_v38, [8, 1])
                stage_20_v40: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.row_expand_add(stage_20_v36, stage_20_v39)
                stage_20_v41: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_20_v40, [1, 512])
                stage_20_v42: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 512], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_20_v43 in pl.pipeline(4, stage=2):
                    stage_20_v46: pl.Scalar[pl.INDEX] = (stage_20_v22 + stage_20_v43) * 128
                    stage_20_v47: pl.Tile[[128], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_idx_wq_b_scale, [stage_20_v46], [128], [128], target_memory=pl.Mem.Vec)
                    stage_20_v48: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_20_v47, [1, 128])
                    stage_20_v49: pl.Tile[[8, 128], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_42, [stage_20_v23, stage_20_v46], [8, 128], [8, 128], target_memory=pl.Mem.Vec)
                    stage_20_v50: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_20_v49, target_type=pl.FP32, mode='none')
                    stage_20_v51: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_20_v50, stage_20_v26)
                    stage_20_v52: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_20_v51, stage_20_v48)
                    stage_20_v53: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_20_v52, [8, 64], [0, 0])
                    stage_20_v54: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_20_v53, target_type=pl.BF16, mode='rint')
                    stage_20_v55: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_20_v52, [8, 64], [0, 64])
                    stage_20_v56: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_20_v52, [1, 1024])
                    stage_20_v57: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_20_v56, stage_20_v41, stage_20_v42)
                    stage_20_v58: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_20_v57, [8, 64])
                    stage_20_v59: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_20_v55, stage_20_v27)
                    stage_20_v60: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_20_v58, stage_20_v28)
                    stage_20_v61: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_20_v59, stage_20_v60)
                    stage_20_v62: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_20_v61, target_type=pl.BF16, mode='rint')
                    pl.tile.store(stage_20_v54, [stage_20_v23, stage_20_v46], stage_20_v25)
                    pl.tile.store(stage_20_v62, [stage_20_v23, stage_20_v46 + 64], stage_20_v25)
                stage_20_v44 = pl.yield_(stage_20_v25)
            else:
                stage_20_v45: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_20_v20, [1, 64], [0, 0])
                for (stage_20_v63, (stage_20_v68,)) in pl.range(stage_20_v23, 144, init_values=(stage_20_v25,)):
                    stage_20_v64: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_19, [stage_20_v63, 0])
                    stage_20_v65: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_20_v63, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_20_v66: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_13, [stage_20_v63, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    for (stage_20_v69, (stage_20_v89,)) in pl.pipeline(4, stage=2, init_values=(stage_20_v68,)):
                        stage_20_v70: pl.Scalar[pl.INDEX] = (stage_20_v22 + stage_20_v69) * 128
                        stage_20_v71: pl.Tile[[128], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_idx_wq_b_scale, [stage_20_v70], [128], [128], target_memory=pl.Mem.Vec)
                        stage_20_v72: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_20_v71, [1, 128])
                        stage_20_v73: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_42, [stage_20_v63, stage_20_v70], [1, 128], [1, 128], target_memory=pl.Mem.Vec)
                        stage_20_v74: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_20_v73, target_type=pl.FP32, mode='none')
                        stage_20_v75: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_20_v74, stage_20_v64)
                        stage_20_v76: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_20_v75, stage_20_v72)
                        stage_20_v77: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_20_v76, [1, 64], [0, 0])
                        stage_20_v78: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_20_v77, target_type=pl.BF16, mode='rint')
                        stage_20_v79: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_20_v76, [1, 64], [0, 64])
                        stage_20_v80: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        for (stage_20_v90, (stage_20_v97,)) in pl.range(1, init_values=(stage_20_v80,)):
                            stage_20_v91: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_20_v79, [1, 64], [stage_20_v90, 0], [1, 64])
                            stage_20_v92: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_20_v45, [1, 64], [stage_20_v90, 0], [1, 64])
                            stage_20_v93: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                            stage_20_v94: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_20_v91, stage_20_v92, stage_20_v93)
                            stage_20_v95: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_20_v97, stage_20_v94, [stage_20_v90, 0])
                            stage_20_v96: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_20_v95)
                        stage_20_v81: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = stage_20_v96
                        stage_20_v82: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_20_v79, stage_20_v65)
                        stage_20_v83: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_20_v81, stage_20_v66)
                        stage_20_v84: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_20_v82, stage_20_v83)
                        stage_20_v85: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_20_v84, target_type=pl.BF16, mode='rint')
                        stage_20_v86 = pl.tile.store(stage_20_v78, [stage_20_v63, stage_20_v70], stage_20_v89)
                        stage_20_v87 = pl.tile.store(stage_20_v85, [stage_20_v63, stage_20_v70 + 64], stage_20_v86)
                        stage_20_v88 = pl.yield_(stage_20_v87)
                    stage_20_v67 = pl.yield_(stage_20_v88)
                stage_20_v44 = pl.yield_(stage_20_v67)
            stage_20_v24 = pl.yield_(stage_20_v44)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_20 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_20)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_21 = 25056 + core_id * 6
    mapped_block_21 = pl.cast(pl.tensor.read(records, [record_21]), pl.INDEX)
    block_21 = pl.max(mapped_block_21, 0)
    if mapped_block_21 >= 0:
        subslot_21 = pl.cast(pl.tensor.read(records, [record_21 + 1]), pl.INDEX)
        done_21 = pl.tensor.read(records, [record_21 + 2])
        wait_base_21 = pl.cast(pl.tensor.read(records, [record_21 + 3]), pl.INDEX)
        wait_count_21 = pl.cast(pl.tensor.read(records, [record_21 + 4]), pl.INDEX)
        publish_21 = pl.tensor.read(records, [record_21 + 5])
        for dependency_21 in pl.range(wait_count_21):
            wait_index_21 = (wait_base_21 + dependency_21) * 2
            other_21 = pl.cast(pl.tensor.read(waits, [wait_index_21]), pl.INDEX)
            required_21 = pl.tensor.read(waits, [wait_index_21 + 1])
            target_21 = generation_base + required_21
            flag_21 = other_21 * 64
            pl.system.cacheinvalid(progress, [1], [flag_21])
            initial_21 = pl.tensor.read(progress, [flag_21])
            for (ready_21,) in pl.while_(init_values=(initial_21,)):
                pl.cond(ready_21 < target_21)
                pl.system.cacheinvalid(progress, [1], [flag_21])
                after_21 = pl.yield_(pl.tensor.read(progress, [flag_21]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_21_v7: pl.Scalar[pl.INDEX] = block_21
        for (stage_21_v8, (stage_21_v10,)) in pl.range(stage_21_v7, 24, 24, init_values=(arg_compress_state,)):
            for (stage_21_v11, (stage_21_v16,)) in pl.range(6, init_values=(stage_21_v10,)):
                stage_21_v12: pl.Scalar[pl.INDEX] = stage_21_v8 * 6 + stage_21_v11
                stage_21_v13: pl.Scalar[pl.INT32] = pl.tensor.read(arg_state_slot_mapping, [stage_21_v12, 0])
                stage_21_v14: pl.Scalar[pl.INT32] = pl.tensor.read(arg_state_slot_mapping, [stage_21_v12, 1])
                if 0 <= pl.cast(stage_21_v13, pl.INDEX) and 0 <= pl.cast(stage_21_v14, pl.INDEX):
                    stage_21_v17: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_21_v12])
                    stage_21_v18: pl.Scalar[pl.INDEX] = pl.cast(stage_21_v13, pl.INDEX)
                    stage_21_v19: pl.Scalar[pl.INDEX] = pl.cast(stage_21_v14, pl.INDEX) * 2048
                    stage_21_v20: pl.Scalar[pl.INDEX] = pl.cast(stage_21_v17 % 4, pl.INDEX)
                    stage_21_v21: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_29, [stage_21_v12, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_21_v22 = pl.tile.store(stage_21_v21, [stage_21_v18, stage_21_v19], stage_21_v16)
                    stage_21_v23: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_30, [stage_21_v12, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_21_v24: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_cmp_ape, [stage_21_v20, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_21_v25: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_21_v23, stage_21_v24)
                    stage_21_v26 = pl.tile.store(stage_21_v25, [stage_21_v18, stage_21_v19 + 1024], stage_21_v22)
                    stage_21_v27 = pl.yield_(stage_21_v26)
                else:
                    stage_21_v27 = pl.yield_(stage_21_v16)
                stage_21_v15 = pl.yield_(stage_21_v27)
            stage_21_v9 = pl.yield_(stage_21_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_21 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_21)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_22 = 25488 + core_id * 6
    mapped_block_22 = pl.cast(pl.tensor.read(records, [record_22]), pl.INDEX)
    block_22 = pl.max(mapped_block_22, 0)
    if mapped_block_22 >= 0:
        subslot_22 = pl.cast(pl.tensor.read(records, [record_22 + 1]), pl.INDEX)
        done_22 = pl.tensor.read(records, [record_22 + 2])
        wait_base_22 = pl.cast(pl.tensor.read(records, [record_22 + 3]), pl.INDEX)
        wait_count_22 = pl.cast(pl.tensor.read(records, [record_22 + 4]), pl.INDEX)
        publish_22 = pl.tensor.read(records, [record_22 + 5])
        for dependency_22 in pl.range(wait_count_22):
            wait_index_22 = (wait_base_22 + dependency_22) * 2
            other_22 = pl.cast(pl.tensor.read(waits, [wait_index_22]), pl.INDEX)
            required_22 = pl.tensor.read(waits, [wait_index_22 + 1])
            target_22 = generation_base + required_22
            flag_22 = other_22 * 64
            pl.system.cacheinvalid(progress, [1], [flag_22])
            initial_22 = pl.tensor.read(progress, [flag_22])
            for (ready_22,) in pl.while_(init_values=(initial_22,)):
                pl.cond(ready_22 < target_22)
                pl.system.cacheinvalid(progress, [1], [flag_22])
                after_22 = pl.yield_(pl.tensor.read(progress, [flag_22]))
        pl.system.fence()
        pl.system.cacheinvalid()
        for stage_22_v6 in pl.range(48):
            stage_22_v7: pl.Scalar[pl.INDEX] = stage_22_v6 // 2
            stage_22_v8: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_22_v7 * 6])
            stage_22_v9: pl.Scalar[pl.INDEX] = stage_22_v6 % 2 * 4 - stage_22_v8 % 4 + 3
            stage_22_v10: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_22_v7])
            if 0 < pl.cast(stage_22_v10, pl.INDEX) and stage_22_v9 < 6:
                stage_22_v11: pl.Scalar[pl.INDEX] = stage_22_v7 * 6 + stage_22_v9
                stage_22_v12: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_22_v11])
                stage_22_v13: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_15, [stage_22_v7])
                stage_22_v14: pl.Scalar[pl.INDEX] = pl.cast(stage_22_v13, pl.INDEX) + pl.cast((stage_22_v12 + 1) // 4, pl.INDEX)
                stage_22_v15: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_idx_slot_mapping, 0)
                stage_22_v16: pl.Scalar[pl.INDEX] = pl.min(stage_22_v14, stage_22_v15 - 1)
                stage_22_v17: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_slot_mapping, [stage_22_v16, 0])
                stage_22_v18: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_slot_mapping, [stage_22_v16, 1])
                if stage_22_v14 < stage_22_v15 and 0 <= pl.cast(stage_22_v17, pl.INDEX) and (0 <= pl.cast(stage_22_v18, pl.INDEX)):
                    stage_22_v19: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_37, [stage_22_v6, 0])
                    stage_22_v20: pl.Scalar[pl.FP16] = pl.cast(stage_22_v19, pl.FP16)
                    stage_22_v21: pl.Scalar[pl.INDEX] = pl.cast(stage_22_v17, pl.INDEX)
                    stage_22_v22: pl.Tile[[1, 64], pl.INT8, pl.Mem.Vec] = pl.tile.load(arg_idx_native_kv_cache, [stage_22_v21, 4096], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_22_v23: pl.Tile[[1, 32], pl.FP16, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_22_v22, dtype=pl.FP16)
                    pl.tile.write(stage_22_v23, [0, stage_22_v18], stage_22_v20)
                    stage_22_v24: pl.Tile[[1, 64], pl.INT8, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_22_v23, dtype=pl.INT8)
                    stage_22_v25 = pl.tile.store(stage_22_v24, [stage_22_v21, 4096], arg_idx_native_kv_cache)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_22 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_22)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_23 = 25920 + core_id * 6
    mapped_block_23 = pl.cast(pl.tensor.read(records, [record_23]), pl.INDEX)
    block_23 = pl.max(mapped_block_23, 0)
    if mapped_block_23 >= 0:
        subslot_23 = pl.cast(pl.tensor.read(records, [record_23 + 1]), pl.INDEX)
        done_23 = pl.tensor.read(records, [record_23 + 2])
        wait_base_23 = pl.cast(pl.tensor.read(records, [record_23 + 3]), pl.INDEX)
        wait_count_23 = pl.cast(pl.tensor.read(records, [record_23 + 4]), pl.INDEX)
        publish_23 = pl.tensor.read(records, [record_23 + 5])
        for dependency_23 in pl.range(wait_count_23):
            wait_index_23 = (wait_base_23 + dependency_23) * 2
            other_23 = pl.cast(pl.tensor.read(waits, [wait_index_23]), pl.INDEX)
            required_23 = pl.tensor.read(waits, [wait_index_23 + 1])
            target_23 = generation_base + required_23
            flag_23 = other_23 * 64
            pl.system.cacheinvalid(progress, [1], [flag_23])
            initial_23 = pl.tensor.read(progress, [flag_23])
            for (ready_23,) in pl.while_(init_values=(initial_23,)):
                pl.cond(ready_23 < target_23)
                pl.system.cacheinvalid(progress, [1], [flag_23])
                after_23 = pl.yield_(pl.tensor.read(progress, [flag_23]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_23_v7: pl.Scalar[pl.INDEX] = block_23
        for (stage_23_v8, (stage_23_v10,)) in pl.range(stage_23_v7, 24, 24, init_values=(arg_inner_compress_state,)):
            for (stage_23_v11, (stage_23_v16,)) in pl.range(6, init_values=(stage_23_v10,)):
                stage_23_v12: pl.Scalar[pl.INDEX] = stage_23_v8 * 6 + stage_23_v11
                stage_23_v13: pl.Scalar[pl.INT32] = pl.tensor.read(arg_inner_state_slot_mapping, [stage_23_v12, 0])
                stage_23_v14: pl.Scalar[pl.INT32] = pl.tensor.read(arg_inner_state_slot_mapping, [stage_23_v12, 1])
                if 0 <= pl.cast(stage_23_v13, pl.INDEX) and 0 <= pl.cast(stage_23_v14, pl.INDEX):
                    stage_23_v17: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_23_v12])
                    stage_23_v18: pl.Scalar[pl.INDEX] = pl.cast(stage_23_v13, pl.INDEX)
                    stage_23_v19: pl.Scalar[pl.INDEX] = pl.cast(stage_23_v14, pl.INDEX) * 512
                    stage_23_v20: pl.Scalar[pl.INDEX] = pl.cast(stage_23_v17 % 4, pl.INDEX)
                    stage_23_v21: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_34, [stage_23_v12, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                    stage_23_v22 = pl.tile.store(stage_23_v21, [stage_23_v18, stage_23_v19], stage_23_v16)
                    stage_23_v23: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_35, [stage_23_v12, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                    stage_23_v24: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_ape, [stage_23_v20, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                    stage_23_v25: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_23_v23, stage_23_v24)
                    stage_23_v26 = pl.tile.store(stage_23_v25, [stage_23_v18, stage_23_v19 + 256], stage_23_v22)
                    stage_23_v27 = pl.yield_(stage_23_v26)
                else:
                    stage_23_v27 = pl.yield_(stage_23_v16)
                stage_23_v15 = pl.yield_(stage_23_v27)
            stage_23_v9 = pl.yield_(stage_23_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_23 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_23)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_24 = 26352 + core_id * 6
    mapped_block_24 = pl.cast(pl.tensor.read(records, [record_24]), pl.INDEX)
    block_24 = pl.max(mapped_block_24, 0)
    if mapped_block_24 >= 0:
        subslot_24 = pl.cast(pl.tensor.read(records, [record_24 + 1]), pl.INDEX)
        done_24 = pl.tensor.read(records, [record_24 + 2])
        wait_base_24 = pl.cast(pl.tensor.read(records, [record_24 + 3]), pl.INDEX)
        wait_count_24 = pl.cast(pl.tensor.read(records, [record_24 + 4]), pl.INDEX)
        publish_24 = pl.tensor.read(records, [record_24 + 5])
        for dependency_24 in pl.range(wait_count_24):
            wait_index_24 = (wait_base_24 + dependency_24) * 2
            other_24 = pl.cast(pl.tensor.read(waits, [wait_index_24]), pl.INDEX)
            required_24 = pl.tensor.read(waits, [wait_index_24 + 1])
            target_24 = generation_base + required_24
            flag_24 = other_24 * 64
            pl.system.cacheinvalid(progress, [1], [flag_24])
            initial_24 = pl.tensor.read(progress, [flag_24])
            for (ready_24,) in pl.while_(init_values=(initial_24,)):
                pl.cond(ready_24 < target_24)
                pl.system.cacheinvalid(progress, [1], [flag_24])
                after_24 = pl.yield_(pl.tensor.read(progress, [flag_24]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_24_v4: pl.Scalar[pl.INDEX] = block_24
        for (stage_24_v5, (stage_24_v8,)) in pl.range(stage_24_v4, 18, 8, init_values=(view_45,)):
            stage_24_v6: pl.Scalar[pl.INDEX] = stage_24_v5 * 8
            for (stage_24_v9, (stage_24_v14,)) in pl.range(pl.min(144 - stage_24_v6, 8), init_values=(stage_24_v8,)):
                stage_24_v10: pl.Scalar[pl.INDEX] = stage_24_v6 + stage_24_v9
                stage_24_v11: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_slot_mapping, [stage_24_v10, 0])
                stage_24_v12: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_slot_mapping, [stage_24_v10, 1])
                if 0 <= pl.cast(stage_24_v11, pl.INDEX) and 0 <= pl.cast(stage_24_v12, pl.INDEX):
                    stage_24_v15: pl.Scalar[pl.INDEX] = pl.cast(stage_24_v11, pl.INDEX) * 32 + pl.cast(stage_24_v12, pl.INDEX)
                    stage_24_v16: pl.Tile[[1, 512], pl.BF16, pl.Mem.Vec] = pl.tile.load(temporary_17, [stage_24_v10, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    stage_24_v17 = pl.tile.store(stage_24_v16, [stage_24_v15, 0], stage_24_v14)
                    stage_24_v18 = pl.yield_(stage_24_v17)
                else:
                    stage_24_v18 = pl.yield_(stage_24_v14)
                stage_24_v13 = pl.yield_(stage_24_v18)
            stage_24_v7 = pl.yield_(stage_24_v13)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_24 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_24)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_25 = 26784 + core_id * 6
    mapped_block_25 = pl.cast(pl.tensor.read(records, [record_25]), pl.INDEX)
    block_25 = pl.max(mapped_block_25, 0)
    if mapped_block_25 >= 0:
        subslot_25 = pl.cast(pl.tensor.read(records, [record_25 + 1]), pl.INDEX)
        done_25 = pl.tensor.read(records, [record_25 + 2])
        wait_base_25 = pl.cast(pl.tensor.read(records, [record_25 + 3]), pl.INDEX)
        wait_count_25 = pl.cast(pl.tensor.read(records, [record_25 + 4]), pl.INDEX)
        publish_25 = pl.tensor.read(records, [record_25 + 5])
        for dependency_25 in pl.range(wait_count_25):
            wait_index_25 = (wait_base_25 + dependency_25) * 2
            other_25 = pl.cast(pl.tensor.read(waits, [wait_index_25]), pl.INDEX)
            required_25 = pl.tensor.read(waits, [wait_index_25 + 1])
            target_25 = generation_base + required_25
            flag_25 = other_25 * 64
            pl.system.cacheinvalid(progress, [1], [flag_25])
            initial_25 = pl.tensor.read(progress, [flag_25])
            for (ready_25,) in pl.while_(init_values=(initial_25,)):
                pl.cond(ready_25 < target_25)
                pl.system.cacheinvalid(progress, [1], [flag_25])
                after_25 = pl.yield_(pl.tensor.read(progress, [flag_25]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_25_v4: pl.Scalar[pl.INDEX] = block_25
        for (stage_25_v5, (stage_25_v21, stage_25_v22)) in pl.range(stage_25_v4, 9216 // 64, 48, init_values=(temporary_39, temporary_40)):
            stage_25_v6: pl.Scalar[pl.INDEX] = stage_25_v5 * 64
            stage_25_v7: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_43, [stage_25_v6, 0], [64, 128], [64, 128], target_memory=pl.Mem.Vec)
            stage_25_v8: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_25_v7, 0.08838834764831845)
            stage_25_v9: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0001)
            for (stage_25_v23, (stage_25_v35,)) in pl.range(0, 128, 64, init_values=(stage_25_v9,)):
                stage_25_v24: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_25_v8, 0, stage_25_v23, [64, 64], target_memory=pl.Mem.Vec)
                stage_25_v25: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_25_v24)
                stage_25_v26: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_25_v8, 0, stage_25_v23, [64, 64], target_memory=pl.Mem.Vec)
                stage_25_v27: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_25_v26, stage_25_v25)
                stage_25_v28: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_25_v29: pl.Tile[[64, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_25_v27, stage_25_v28)
                stage_25_v30: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_25_v29, [1, 64])
                stage_25_v31: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_25_v35, stage_25_v30)
                stage_25_v32: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_25_v31)
            stage_25_v10: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=127.0)
            stage_25_v11: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_25_v10, stage_25_v32)
            stage_25_v12: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_25_v11)
            stage_25_v13: pl.Tile[[64, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_25_v12, [64, 1])
            stage_25_v14 = pl.tile.store(stage_25_v13, [stage_25_v6, 0], stage_25_v22)
            stage_25_v15: pl.Tile[[64, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_25_v11, [64, 1])
            stage_25_v16: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_25_v8, stage_25_v15)
            stage_25_v17: pl.Tile[[64, 128], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_25_v16, target_type=pl.INT32, mode='rint')
            stage_25_v18: pl.Tile[[64, 128], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_25_v17, target_type=pl.FP16, mode='round')
            stage_25_v19: pl.Tile[[64, 128], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_25_v18, target_type=pl.INT8, mode='trunc')
            stage_25_v20 = pl.tile.store(stage_25_v19, [stage_25_v6, 0], stage_25_v21)
            (stage_25_v33, stage_25_v34) = pl.yield_(stage_25_v20, stage_25_v14)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_25 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_25)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_26 = 27216 + core_id * 6
    mapped_block_26 = pl.cast(pl.tensor.read(records, [record_26]), pl.INDEX)
    block_26 = pl.max(mapped_block_26, 0)
    if mapped_block_26 >= 0:
        subslot_26 = pl.cast(pl.tensor.read(records, [record_26 + 1]), pl.INDEX)
        done_26 = pl.tensor.read(records, [record_26 + 2])
        wait_base_26 = pl.cast(pl.tensor.read(records, [record_26 + 3]), pl.INDEX)
        wait_count_26 = pl.cast(pl.tensor.read(records, [record_26 + 4]), pl.INDEX)
        publish_26 = pl.tensor.read(records, [record_26 + 5])
        for dependency_26 in pl.range(wait_count_26):
            wait_index_26 = (wait_base_26 + dependency_26) * 2
            other_26 = pl.cast(pl.tensor.read(waits, [wait_index_26]), pl.INDEX)
            required_26 = pl.tensor.read(waits, [wait_index_26 + 1])
            target_26 = generation_base + required_26
            flag_26 = other_26 * 64
            pl.system.cacheinvalid(progress, [1], [flag_26])
            initial_26 = pl.tensor.read(progress, [flag_26])
            for (ready_26,) in pl.while_(init_values=(initial_26,)):
                pl.cond(ready_26 < target_26)
                pl.system.cacheinvalid(progress, [1], [flag_26])
                after_26 = pl.yield_(pl.tensor.read(progress, [flag_26]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_26_v4: pl.Scalar[pl.INDEX] = block_26
        for (stage_26_v5, (stage_26_v21, stage_26_v22)) in pl.range(stage_26_v4, 9216 // 64, 48, init_values=(temporary_39, temporary_40)):
            stage_26_v6: pl.Scalar[pl.INDEX] = stage_26_v5 * 64
            stage_26_v7: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_43, [stage_26_v6, 0], [64, 128], [64, 128], target_memory=pl.Mem.Vec)
            stage_26_v8: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_26_v7, 0.08838834764831845)
            stage_26_v9: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0001)
            for (stage_26_v23, (stage_26_v35,)) in pl.range(0, 128, 64, init_values=(stage_26_v9,)):
                stage_26_v24: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_26_v8, 0, stage_26_v23, [64, 64], target_memory=pl.Mem.Vec)
                stage_26_v25: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_26_v24)
                stage_26_v26: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_26_v8, 0, stage_26_v23, [64, 64], target_memory=pl.Mem.Vec)
                stage_26_v27: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_26_v26, stage_26_v25)
                stage_26_v28: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_26_v29: pl.Tile[[64, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_26_v27, stage_26_v28)
                stage_26_v30: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_26_v29, [1, 64])
                stage_26_v31: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_26_v35, stage_26_v30)
                stage_26_v32: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_26_v31)
            stage_26_v10: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=127.0)
            stage_26_v11: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_26_v10, stage_26_v32)
            stage_26_v12: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_26_v11)
            stage_26_v13: pl.Tile[[64, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_26_v12, [64, 1])
            stage_26_v14 = pl.tile.store(stage_26_v13, [stage_26_v6, 0], stage_26_v22)
            stage_26_v15: pl.Tile[[64, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_26_v11, [64, 1])
            stage_26_v16: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_26_v8, stage_26_v15)
            stage_26_v17: pl.Tile[[64, 128], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_26_v16, target_type=pl.INT32, mode='rint')
            stage_26_v18: pl.Tile[[64, 128], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_26_v17, target_type=pl.FP16, mode='round')
            stage_26_v19: pl.Tile[[64, 128], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_26_v18, target_type=pl.INT8, mode='trunc')
            stage_26_v20 = pl.tile.store(stage_26_v19, [stage_26_v6, 0], stage_26_v21)
            (stage_26_v33, stage_26_v34) = pl.yield_(stage_26_v20, stage_26_v14)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_26 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_26)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_27 = 27648 + core_id * 6
    mapped_block_27 = pl.cast(pl.tensor.read(records, [record_27]), pl.INDEX)
    block_27 = pl.max(mapped_block_27, 0)
    if mapped_block_27 >= 0:
        subslot_27 = pl.cast(pl.tensor.read(records, [record_27 + 1]), pl.INDEX)
        done_27 = pl.tensor.read(records, [record_27 + 2])
        wait_base_27 = pl.cast(pl.tensor.read(records, [record_27 + 3]), pl.INDEX)
        wait_count_27 = pl.cast(pl.tensor.read(records, [record_27 + 4]), pl.INDEX)
        publish_27 = pl.tensor.read(records, [record_27 + 5])
        for dependency_27 in pl.range(wait_count_27):
            wait_index_27 = (wait_base_27 + dependency_27) * 2
            other_27 = pl.cast(pl.tensor.read(waits, [wait_index_27]), pl.INDEX)
            required_27 = pl.tensor.read(waits, [wait_index_27 + 1])
            target_27 = generation_base + required_27
            flag_27 = other_27 * 64
            pl.system.cacheinvalid(progress, [1], [flag_27])
            initial_27 = pl.tensor.read(progress, [flag_27])
            for (ready_27,) in pl.while_(init_values=(initial_27,)):
                pl.cond(ready_27 < target_27)
                pl.system.cacheinvalid(progress, [1], [flag_27])
                after_27 = pl.yield_(pl.tensor.read(progress, [flag_27]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_27_v1: pl.Tile[[1, 512], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_45, [0, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
        stage_27_v2 = pl.tile.store(stage_27_v1, [0, 0], view_45)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_27 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_27)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_28 = 28080 + core_id * 6
    mapped_block_28 = pl.cast(pl.tensor.read(records, [record_28]), pl.INDEX)
    block_28 = pl.max(mapped_block_28, 0)
    if mapped_block_28 >= 0:
        subslot_28 = pl.cast(pl.tensor.read(records, [record_28 + 1]), pl.INDEX)
        done_28 = pl.tensor.read(records, [record_28 + 2])
        wait_base_28 = pl.cast(pl.tensor.read(records, [record_28 + 3]), pl.INDEX)
        wait_count_28 = pl.cast(pl.tensor.read(records, [record_28 + 4]), pl.INDEX)
        publish_28 = pl.tensor.read(records, [record_28 + 5])
        for dependency_28 in pl.range(wait_count_28):
            wait_index_28 = (wait_base_28 + dependency_28) * 2
            other_28 = pl.cast(pl.tensor.read(waits, [wait_index_28]), pl.INDEX)
            required_28 = pl.tensor.read(waits, [wait_index_28 + 1])
            target_28 = generation_base + required_28
            flag_28 = other_28 * 64
            pl.system.cacheinvalid(progress, [1], [flag_28])
            initial_28 = pl.tensor.read(progress, [flag_28])
            for (ready_28,) in pl.while_(init_values=(initial_28,)):
                pl.cond(ready_28 < target_28)
                pl.system.cacheinvalid(progress, [1], [flag_28])
                after_28 = pl.yield_(pl.tensor.read(progress, [flag_28]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_28_v5: pl.Scalar[pl.INDEX] = block_28
        stage_28_v6: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        for stage_28_v7 in pl.range(stage_28_v5, stage_28_v6 // 6, 48):
            stage_28_v8: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.full([96, 64], dtype=pl.FP16, value=0.0)
            stage_28_v9: pl.Scalar[pl.INDEX] = stage_28_v7 * 6
            stage_28_v10: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_94, [stage_28_v9, 0], [6, 64], [6, 64], target_memory=pl.Mem.Vec)
            stage_28_v11: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_44, [stage_28_v9, 0], [6, 64], [6, 64], target_memory=pl.Mem.Vec)
            stage_28_v12: pl.Tile[[6, 64], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_28_v10, target_type=pl.FP16, mode='rint')
            stage_28_v13: pl.Tile[[6, 64], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_28_v11, target_type=pl.FP16, mode='rint')
            stage_28_v14: pl.Tile[[6, 64], pl.FP16, pl.Mem.Vec] = pl.tile.mul(stage_28_v12, stage_28_v13)
            stage_28_v15: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_28_v14, 0, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_28_v16: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_28_v8, stage_28_v15, [0, 0])
            stage_28_v17: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_28_v14, 1, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_28_v18: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_28_v16, stage_28_v17, [7, 0])
            stage_28_v19: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_28_v14, 2, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_28_v20: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_28_v18, stage_28_v19, [14, 0])
            stage_28_v21: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_28_v14, 3, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_28_v22: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_28_v20, stage_28_v21, [21, 0])
            stage_28_v23: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_28_v14, 4, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_28_v24: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_28_v22, stage_28_v23, [28, 0])
            stage_28_v25: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_28_v14, 5, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_28_v26: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_28_v24, stage_28_v25, [35, 0])
            stage_28_v27: pl.Tile[[16, 384], pl.FP16, pl.Mem.Vec] = pl.tile.reshape(stage_28_v26, [16, 384])
            stage_28_v28 = pl.tile.store(stage_28_v27, [stage_28_v7 * 16, 0], temporary_48)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_28 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_28)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_29 = 28512 + core_id * 6
    mapped_block_29 = pl.cast(pl.tensor.read(records, [record_29]), pl.INDEX)
    block_29 = pl.max(mapped_block_29, 0)
    if mapped_block_29 >= 0:
        subslot_29 = pl.cast(pl.tensor.read(records, [record_29 + 1]), pl.INDEX)
        done_29 = pl.tensor.read(records, [record_29 + 2])
        wait_base_29 = pl.cast(pl.tensor.read(records, [record_29 + 3]), pl.INDEX)
        wait_count_29 = pl.cast(pl.tensor.read(records, [record_29 + 4]), pl.INDEX)
        publish_29 = pl.tensor.read(records, [record_29 + 5])
        for dependency_29 in pl.range(wait_count_29):
            wait_index_29 = (wait_base_29 + dependency_29) * 2
            other_29 = pl.cast(pl.tensor.read(waits, [wait_index_29]), pl.INDEX)
            required_29 = pl.tensor.read(waits, [wait_index_29 + 1])
            target_29 = generation_base + required_29
            flag_29 = other_29 * 64
            pl.system.cacheinvalid(progress, [1], [flag_29])
            initial_29 = pl.tensor.read(progress, [flag_29])
            for (ready_29,) in pl.while_(init_values=(initial_29,)):
                pl.cond(ready_29 < target_29)
                pl.system.cacheinvalid(progress, [1], [flag_29])
                after_29 = pl.yield_(pl.tensor.read(progress, [flag_29]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_29_v14: pl.Scalar[pl.INDEX] = block_29
        stage_29_v15: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        stage_29_v16: pl.Scalar[pl.INDEX] = 0
        for (stage_29_v21, (stage_29_v32,)) in pl.range(stage_29_v15 // 6, init_values=(stage_29_v16,)):
            stage_29_v22: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_29_v21])
            stage_29_v23: pl.Scalar[pl.INDEX] = pl.max(stage_29_v32, pl.cast(stage_29_v22, pl.INDEX) // 4)
            stage_29_v24: pl.Scalar[pl.INDEX] = pl.yield_(stage_29_v23)
        stage_29_v17: pl.Scalar[pl.INDEX] = pl.min(stage_29_v24, 262144)
        stage_29_v18: pl.Scalar[pl.INDEX] = pl.max((stage_29_v17 + 8191) // 8192, 1)
        pl.system.set_ffts(temporary_50)
        stage_29_v19: pl.Scalar[pl.INDEX] = subslot_29
        for stage_29_v20 in pl.range(stage_29_v14, stage_29_v15 // 6 * stage_29_v18, 24):
            if stage_29_v15 < 144:
                stage_29_v33: pl.Scalar[pl.INDEX] = stage_29_v20 % (stage_29_v15 // 6) * 6
                stage_29_v34: pl.Scalar[pl.INDEX] = stage_29_v20 // (stage_29_v15 // 6)
                (stage_29_v73, stage_29_v74) = pl.yield_(stage_29_v34, stage_29_v33)
            else:
                stage_29_v35: pl.Scalar[pl.INDEX] = stage_29_v20 // stage_29_v18 * 6
                stage_29_v36: pl.Scalar[pl.INDEX] = stage_29_v20 % stage_29_v18
                (stage_29_v73, stage_29_v74) = pl.yield_(stage_29_v36, stage_29_v35)
            stage_29_v25: pl.Scalar[pl.INDEX] = stage_29_v74 // 6
            stage_29_v26: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v74 + 5])
            stage_29_v27: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_29_v25])
            stage_29_v28: pl.Scalar[pl.INDEX] = pl.cast(stage_29_v27, pl.INDEX) // 4
            stage_29_v29: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v26 + 1) // 4), 262144), 0)
            stage_29_v30: pl.Scalar[pl.INDEX] = stage_29_v73 * 8192
            stage_29_v31: pl.Scalar[pl.INDEX] = 8192
            if stage_29_v30 < stage_29_v29:
                stage_29_v37: pl.Scalar[pl.INDEX] = pl.min(stage_29_v31, stage_29_v29 - stage_29_v30)
                stage_29_v38: pl.Scalar[pl.INDEX] = (stage_29_v37 + 767) // 768
                stage_29_v39: pl.Scalar[pl.INDEX] = pl.min(stage_29_v38 * 384, 4096)
                stage_29_v40: pl.Scalar[pl.INDEX] = (stage_29_v39 + 383) // 384
                stage_29_v41: pl.Scalar[pl.INDEX] = stage_29_v19 * stage_29_v39
                for stage_29_v42 in pl.range(stage_29_v40):
                    stage_29_v75: pl.Scalar[pl.INDEX] = stage_29_v42 * 384
                    stage_29_v76: pl.Scalar[pl.INDEX] = stage_29_v14 * 12 + stage_29_v42 % 2 * 6
                    stage_29_v77: pl.Tile[[1, 768], pl.INT8, pl.Mem.Vec] = pl.tile.create([1, 768], dtype=pl.INT8, target_memory=pl.Mem.Vec)
                    for (stage_29_v129, (stage_29_v208,)) in pl.range(12, init_values=(stage_29_v77,)):
                        stage_29_v130: pl.Scalar[pl.INDEX] = stage_29_v30 + stage_29_v75 + stage_29_v41 + stage_29_v129 * 32
                        stage_29_v131: pl.Scalar[pl.INDEX] = pl.min(stage_29_v130 // 32, pl.max((stage_29_v28 - 1) // 32, 0))
                        stage_29_v132: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_29_v25, stage_29_v131])
                        stage_29_v133: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_29_v132, pl.INDEX), 0)
                        stage_29_v134: pl.Tile[[1, 768], pl.INT8, pl.Mem.Vec] = pl.tile.gather_row(stage_29_v208, arg_idx_native_kv_cache, [0, stage_29_v129 * 64], [stage_29_v133, 4096], [1, 64], transpose=False)
                        stage_29_v135: pl.Tile[[1, 768], pl.INT8, pl.Mem.Vec] = pl.yield_(stage_29_v134)
                    stage_29_v78: pl.Tile[[1, 384], pl.FP16, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v135, dtype=pl.FP16)
                    stage_29_v79: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_29_v78, target_type=pl.FP32, mode='round')
                    pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    stage_29_v80: pl.Tile[[6, 384], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_49, [stage_29_v76, stage_29_v19 * 384], [6, 384], [6, 384], target_memory=pl.Mem.Vec)
                    pl.system.sync_set(1, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV)
                    stage_29_v81: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v74])
                    stage_29_v82: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v81 + 1) // 4), 262144), 0)
                    stage_29_v83: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v82 - stage_29_v30 - stage_29_v75 - stage_29_v41, 384), 0)
                    stage_29_v84: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_29_v80, [1, 384], [0, 0])
                    stage_29_v85: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_29_v84, stage_29_v79)
                    if 0 < stage_29_v83:
                        stage_29_v136: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v83])] = pl.tile.set_validshape(stage_29_v85, 1, stage_29_v83)
                        stage_29_v137 = pl.tile.store(stage_29_v136, [stage_29_v14 * 12 + stage_29_v19, stage_29_v75], temporary_47)
                    stage_29_v86: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v74 + 1])
                    stage_29_v87: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v86 + 1) // 4), 262144), 0)
                    stage_29_v88: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v87 - stage_29_v30 - stage_29_v75 - stage_29_v41, 384), 0)
                    stage_29_v89: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_29_v80, [1, 384], [1, 0])
                    stage_29_v90: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_29_v89, stage_29_v79)
                    if 0 < stage_29_v88:
                        stage_29_v138: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v88])] = pl.tile.set_validshape(stage_29_v90, 1, stage_29_v88)
                        stage_29_v139 = pl.tile.store(stage_29_v138, [stage_29_v14 * 12 + stage_29_v19 + 2, stage_29_v75], temporary_47)
                    stage_29_v91: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v74 + 2])
                    stage_29_v92: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v91 + 1) // 4), 262144), 0)
                    stage_29_v93: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v92 - stage_29_v30 - stage_29_v75 - stage_29_v41, 384), 0)
                    stage_29_v94: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_29_v80, [1, 384], [2, 0])
                    stage_29_v95: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_29_v94, stage_29_v79)
                    if 0 < stage_29_v93:
                        stage_29_v140: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v93])] = pl.tile.set_validshape(stage_29_v95, 1, stage_29_v93)
                        stage_29_v141 = pl.tile.store(stage_29_v140, [stage_29_v14 * 12 + stage_29_v19 + 4, stage_29_v75], temporary_47)
                    stage_29_v96: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v74 + 3])
                    stage_29_v97: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v96 + 1) // 4), 262144), 0)
                    stage_29_v98: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v97 - stage_29_v30 - stage_29_v75 - stage_29_v41, 384), 0)
                    stage_29_v99: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_29_v80, [1, 384], [3, 0])
                    stage_29_v100: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_29_v99, stage_29_v79)
                    if 0 < stage_29_v98:
                        stage_29_v142: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v98])] = pl.tile.set_validshape(stage_29_v100, 1, stage_29_v98)
                        stage_29_v143 = pl.tile.store(stage_29_v142, [stage_29_v14 * 12 + stage_29_v19 + 6, stage_29_v75], temporary_47)
                    stage_29_v101: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v74 + 4])
                    stage_29_v102: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v101 + 1) // 4), 262144), 0)
                    stage_29_v103: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v102 - stage_29_v30 - stage_29_v75 - stage_29_v41, 384), 0)
                    stage_29_v104: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_29_v80, [1, 384], [4, 0])
                    stage_29_v105: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_29_v104, stage_29_v79)
                    if 0 < stage_29_v103:
                        stage_29_v144: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v103])] = pl.tile.set_validshape(stage_29_v105, 1, stage_29_v103)
                        stage_29_v145 = pl.tile.store(stage_29_v144, [stage_29_v14 * 12 + stage_29_v19 + 8, stage_29_v75], temporary_47)
                    stage_29_v106: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v74 + 5])
                    stage_29_v107: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v106 + 1) // 4), 262144), 0)
                    stage_29_v108: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v107 - stage_29_v30 - stage_29_v75 - stage_29_v41, 384), 0)
                    stage_29_v109: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_29_v80, [1, 384], [5, 0])
                    stage_29_v110: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_29_v109, stage_29_v79)
                    if 0 < stage_29_v108:
                        stage_29_v146: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v108])] = pl.tile.set_validshape(stage_29_v110, 1, stage_29_v108)
                        stage_29_v147 = pl.tile.store(stage_29_v146, [stage_29_v14 * 12 + stage_29_v19 + 10, stage_29_v75], temporary_47)
                stage_29_v43: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v74])
                stage_29_v44: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v43 + 1) // 4), 262144), 0)
                stage_29_v45: pl.Scalar[pl.INDEX] = stage_29_v30 + stage_29_v41
                stage_29_v46: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v44 - stage_29_v45, stage_29_v39), 0)
                stage_29_v47: pl.Scalar[pl.INDEX] = stage_29_v74 * 64 + stage_29_v73 * 2 + stage_29_v19
                if 0 < stage_29_v46:
                    stage_29_v111: pl.Scalar[pl.INT32] = pl.cast(stage_29_v45, pl.INT32)
                    if stage_29_v46 <= 512:
                        stage_29_v148: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                        stage_29_v149: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v148, stage_29_v111)
                        stage_29_v150: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v46])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19, 0], [1, 512], [1, stage_29_v46], target_memory=pl.Mem.Vec)
                        stage_29_v151: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v150, pad_value=pl.PadValue.min)
                        stage_29_v152: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v151, -3.4028234663852886e+38)
                        stage_29_v153: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v149, dtype=pl.UINT32)
                        stage_29_v154: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v152, stage_29_v153)
                        stage_29_v155: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v154, pl.const(64, pl.INT32))
                        stage_29_v156: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v155, pl.const(256, pl.INT32))
                        stage_29_v157 = pl.tile.store(stage_29_v156, [stage_29_v47, 0], temporary_46)
                    elif stage_29_v46 <= 1024:
                        stage_29_v209: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                        stage_29_v210: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v209, stage_29_v111)
                        stage_29_v211: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v46])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19, 0], [1, 1024], [1, stage_29_v46], target_memory=pl.Mem.Vec)
                        stage_29_v212: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v211, pad_value=pl.PadValue.min)
                        stage_29_v213: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v212, -3.4028234663852886e+38)
                        stage_29_v214: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v210, dtype=pl.UINT32)
                        stage_29_v215: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v213, stage_29_v214)
                        stage_29_v216: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v215, pl.const(64, pl.INT32))
                        stage_29_v217: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v216, pl.const(256, pl.INT32))
                        stage_29_v218: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v217, [1, 1024], [0, 0])
                        stage_29_v219: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v217, [1, 1024], [0, 1024])
                        stage_29_v220: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v221: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v218, stage_29_v219, stage_29_v220, exhausted=False)
                        stage_29_v222: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v221, [1, 1024], [0, 0])
                        stage_29_v223 = pl.tile.store(stage_29_v222, [stage_29_v47, 0], temporary_46)
                    elif stage_29_v46 <= 2048:
                        stage_29_v299: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                        stage_29_v300: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v299, stage_29_v111)
                        stage_29_v301: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v46])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19, 0], [1, 2048], [1, stage_29_v46], target_memory=pl.Mem.Vec)
                        stage_29_v302: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v301, pad_value=pl.PadValue.min)
                        stage_29_v303: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v302, -3.4028234663852886e+38)
                        stage_29_v304: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v300, dtype=pl.UINT32)
                        stage_29_v305: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v303, stage_29_v304)
                        stage_29_v306: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v305, pl.const(64, pl.INT32))
                        stage_29_v307: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v306, pl.const(256, pl.INT32))
                        stage_29_v308: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v307, pl.const(1024, pl.INT32))
                        stage_29_v309: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v308, [1, 1024], [0, 0])
                        stage_29_v310 = pl.tile.store(stage_29_v309, [stage_29_v47, 0], temporary_46)
                    elif stage_29_v46 <= 2560:
                        stage_29_v371: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2560], dtype=pl.INT32, descending=False)
                        stage_29_v372: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v371, stage_29_v111)
                        stage_29_v373: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v46])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19, 0], [1, 2560], [1, stage_29_v46], target_memory=pl.Mem.Vec)
                        stage_29_v374: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v373, pad_value=pl.PadValue.min)
                        stage_29_v375: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v374, -3.4028234663852886e+38)
                        stage_29_v376: pl.Tile[[1, 2560], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v372, dtype=pl.UINT32)
                        stage_29_v377: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v375, stage_29_v376)
                        stage_29_v378: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v377, pl.const(64, pl.INT32))
                        stage_29_v379: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v378, pl.const(256, pl.INT32))
                        stage_29_v380: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v379, 1, 4096)
                        stage_29_v381: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v380, pl.const(1024, pl.INT32))
                        stage_29_v382: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v381, [1, 1024], [0, 0])
                        stage_29_v383: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v379, [1, 1024], [0, 4096])
                        stage_29_v384: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v385: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v383, stage_29_v382, stage_29_v384, exhausted=False)
                        stage_29_v386: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v385, [1, 1024], [0, 0])
                        stage_29_v387 = pl.tile.store(stage_29_v386, [stage_29_v47, 0], temporary_46)
                    elif stage_29_v46 <= 3072:
                        stage_29_v473: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 3072], dtype=pl.INT32, descending=False)
                        stage_29_v474: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v473, stage_29_v111)
                        stage_29_v475: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v46])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19, 0], [1, 3072], [1, stage_29_v46], target_memory=pl.Mem.Vec)
                        stage_29_v476: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v475, pad_value=pl.PadValue.min)
                        stage_29_v477: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v476, -3.4028234663852886e+38)
                        stage_29_v478: pl.Tile[[1, 3072], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v474, dtype=pl.UINT32)
                        stage_29_v479: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v477, stage_29_v478)
                        stage_29_v480: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v479, pl.const(64, pl.INT32))
                        stage_29_v481: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v480, pl.const(256, pl.INT32))
                        stage_29_v482: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v481, 1, 4096)
                        stage_29_v483: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v482, pl.const(1024, pl.INT32))
                        stage_29_v484: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v483, [1, 1024], [0, 0])
                        stage_29_v485: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v481, [1, 1024], [0, 4096])
                        stage_29_v486: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v481, [1, 1024], [0, 5120])
                        stage_29_v487: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 3072], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v488: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v485, stage_29_v486, stage_29_v484, stage_29_v487, exhausted=False)
                        stage_29_v489: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v488, [1, 1024], [0, 0])
                        stage_29_v490 = pl.tile.store(stage_29_v489, [stage_29_v47, 0], temporary_46)
                    else:
                        stage_29_v491: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 4096], dtype=pl.INT32, descending=False)
                        stage_29_v492: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v491, stage_29_v111)
                        stage_29_v493: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v46])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19, 0], [1, 4096], [1, stage_29_v46], target_memory=pl.Mem.Vec)
                        stage_29_v494: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v493, pad_value=pl.PadValue.min)
                        stage_29_v495: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v494, -3.4028234663852886e+38)
                        stage_29_v496: pl.Tile[[1, 4096], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v492, dtype=pl.UINT32)
                        stage_29_v497: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v495, stage_29_v496)
                        stage_29_v498: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v497, pl.const(64, pl.INT32))
                        stage_29_v499: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v498, pl.const(256, pl.INT32))
                        stage_29_v500: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v499, pl.const(1024, pl.INT32))
                        stage_29_v501: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v500, [1, 1024], [0, 0])
                        stage_29_v502: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v500, [1, 1024], [0, 4096])
                        stage_29_v503: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v504: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v502, stage_29_v501, stage_29_v503, exhausted=False)
                        stage_29_v505: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v504, [1, 1024], [0, 0])
                        stage_29_v506 = pl.tile.store(stage_29_v505, [stage_29_v47, 0], temporary_46)
                else:
                    stage_29_v112: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
                    stage_29_v113 = pl.tile.store(stage_29_v112, [stage_29_v47, 0], temporary_46)
                stage_29_v48: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v74 + 1])
                stage_29_v49: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v48 + 1) // 4), 262144), 0)
                stage_29_v50: pl.Scalar[pl.INDEX] = stage_29_v30 + stage_29_v41
                stage_29_v51: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v49 - stage_29_v50, stage_29_v39), 0)
                stage_29_v52: pl.Scalar[pl.INDEX] = stage_29_v74 * 64 + stage_29_v73 * 2 + stage_29_v19 + 64
                if 0 < stage_29_v51:
                    stage_29_v114: pl.Scalar[pl.INT32] = pl.cast(stage_29_v50, pl.INT32)
                    if stage_29_v51 <= 512:
                        stage_29_v158: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                        stage_29_v159: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v158, stage_29_v114)
                        stage_29_v160: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v51])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 2, 0], [1, 512], [1, stage_29_v51], target_memory=pl.Mem.Vec)
                        stage_29_v161: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v160, pad_value=pl.PadValue.min)
                        stage_29_v162: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v161, -3.4028234663852886e+38)
                        stage_29_v163: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v159, dtype=pl.UINT32)
                        stage_29_v164: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v162, stage_29_v163)
                        stage_29_v165: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v164, pl.const(64, pl.INT32))
                        stage_29_v166: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v165, pl.const(256, pl.INT32))
                        stage_29_v167 = pl.tile.store(stage_29_v166, [stage_29_v52, 0], temporary_46)
                    elif stage_29_v51 <= 1024:
                        stage_29_v224: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                        stage_29_v225: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v224, stage_29_v114)
                        stage_29_v226: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v51])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 2, 0], [1, 1024], [1, stage_29_v51], target_memory=pl.Mem.Vec)
                        stage_29_v227: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v226, pad_value=pl.PadValue.min)
                        stage_29_v228: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v227, -3.4028234663852886e+38)
                        stage_29_v229: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v225, dtype=pl.UINT32)
                        stage_29_v230: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v228, stage_29_v229)
                        stage_29_v231: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v230, pl.const(64, pl.INT32))
                        stage_29_v232: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v231, pl.const(256, pl.INT32))
                        stage_29_v233: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v232, [1, 1024], [0, 0])
                        stage_29_v234: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v232, [1, 1024], [0, 1024])
                        stage_29_v235: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v236: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v233, stage_29_v234, stage_29_v235, exhausted=False)
                        stage_29_v237: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v236, [1, 1024], [0, 0])
                        stage_29_v238 = pl.tile.store(stage_29_v237, [stage_29_v52, 0], temporary_46)
                    elif stage_29_v51 <= 2048:
                        stage_29_v311: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                        stage_29_v312: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v311, stage_29_v114)
                        stage_29_v313: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v51])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 2, 0], [1, 2048], [1, stage_29_v51], target_memory=pl.Mem.Vec)
                        stage_29_v314: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v313, pad_value=pl.PadValue.min)
                        stage_29_v315: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v314, -3.4028234663852886e+38)
                        stage_29_v316: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v312, dtype=pl.UINT32)
                        stage_29_v317: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v315, stage_29_v316)
                        stage_29_v318: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v317, pl.const(64, pl.INT32))
                        stage_29_v319: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v318, pl.const(256, pl.INT32))
                        stage_29_v320: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v319, pl.const(1024, pl.INT32))
                        stage_29_v321: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v320, [1, 1024], [0, 0])
                        stage_29_v322 = pl.tile.store(stage_29_v321, [stage_29_v52, 0], temporary_46)
                    elif stage_29_v51 <= 2560:
                        stage_29_v388: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2560], dtype=pl.INT32, descending=False)
                        stage_29_v389: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v388, stage_29_v114)
                        stage_29_v390: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v51])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 2, 0], [1, 2560], [1, stage_29_v51], target_memory=pl.Mem.Vec)
                        stage_29_v391: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v390, pad_value=pl.PadValue.min)
                        stage_29_v392: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v391, -3.4028234663852886e+38)
                        stage_29_v393: pl.Tile[[1, 2560], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v389, dtype=pl.UINT32)
                        stage_29_v394: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v392, stage_29_v393)
                        stage_29_v395: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v394, pl.const(64, pl.INT32))
                        stage_29_v396: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v395, pl.const(256, pl.INT32))
                        stage_29_v397: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v396, 1, 4096)
                        stage_29_v398: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v397, pl.const(1024, pl.INT32))
                        stage_29_v399: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v398, [1, 1024], [0, 0])
                        stage_29_v400: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v396, [1, 1024], [0, 4096])
                        stage_29_v401: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v402: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v400, stage_29_v399, stage_29_v401, exhausted=False)
                        stage_29_v403: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v402, [1, 1024], [0, 0])
                        stage_29_v404 = pl.tile.store(stage_29_v403, [stage_29_v52, 0], temporary_46)
                    elif stage_29_v51 <= 3072:
                        stage_29_v507: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 3072], dtype=pl.INT32, descending=False)
                        stage_29_v508: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v507, stage_29_v114)
                        stage_29_v509: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v51])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 2, 0], [1, 3072], [1, stage_29_v51], target_memory=pl.Mem.Vec)
                        stage_29_v510: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v509, pad_value=pl.PadValue.min)
                        stage_29_v511: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v510, -3.4028234663852886e+38)
                        stage_29_v512: pl.Tile[[1, 3072], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v508, dtype=pl.UINT32)
                        stage_29_v513: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v511, stage_29_v512)
                        stage_29_v514: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v513, pl.const(64, pl.INT32))
                        stage_29_v515: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v514, pl.const(256, pl.INT32))
                        stage_29_v516: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v515, 1, 4096)
                        stage_29_v517: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v516, pl.const(1024, pl.INT32))
                        stage_29_v518: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v517, [1, 1024], [0, 0])
                        stage_29_v519: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v515, [1, 1024], [0, 4096])
                        stage_29_v520: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v515, [1, 1024], [0, 5120])
                        stage_29_v521: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 3072], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v522: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v519, stage_29_v520, stage_29_v518, stage_29_v521, exhausted=False)
                        stage_29_v523: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v522, [1, 1024], [0, 0])
                        stage_29_v524 = pl.tile.store(stage_29_v523, [stage_29_v52, 0], temporary_46)
                    else:
                        stage_29_v525: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 4096], dtype=pl.INT32, descending=False)
                        stage_29_v526: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v525, stage_29_v114)
                        stage_29_v527: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v51])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 2, 0], [1, 4096], [1, stage_29_v51], target_memory=pl.Mem.Vec)
                        stage_29_v528: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v527, pad_value=pl.PadValue.min)
                        stage_29_v529: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v528, -3.4028234663852886e+38)
                        stage_29_v530: pl.Tile[[1, 4096], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v526, dtype=pl.UINT32)
                        stage_29_v531: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v529, stage_29_v530)
                        stage_29_v532: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v531, pl.const(64, pl.INT32))
                        stage_29_v533: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v532, pl.const(256, pl.INT32))
                        stage_29_v534: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v533, pl.const(1024, pl.INT32))
                        stage_29_v535: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v534, [1, 1024], [0, 0])
                        stage_29_v536: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v534, [1, 1024], [0, 4096])
                        stage_29_v537: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v538: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v536, stage_29_v535, stage_29_v537, exhausted=False)
                        stage_29_v539: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v538, [1, 1024], [0, 0])
                        stage_29_v540 = pl.tile.store(stage_29_v539, [stage_29_v52, 0], temporary_46)
                else:
                    stage_29_v115: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
                    stage_29_v116 = pl.tile.store(stage_29_v115, [stage_29_v52, 0], temporary_46)
                stage_29_v53: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v74 + 2])
                stage_29_v54: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v53 + 1) // 4), 262144), 0)
                stage_29_v55: pl.Scalar[pl.INDEX] = stage_29_v30 + stage_29_v41
                stage_29_v56: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v54 - stage_29_v55, stage_29_v39), 0)
                stage_29_v57: pl.Scalar[pl.INDEX] = stage_29_v74 * 64 + stage_29_v73 * 2 + stage_29_v19 + 128
                if 0 < stage_29_v56:
                    stage_29_v117: pl.Scalar[pl.INT32] = pl.cast(stage_29_v55, pl.INT32)
                    if stage_29_v56 <= 512:
                        stage_29_v168: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                        stage_29_v169: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v168, stage_29_v117)
                        stage_29_v170: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v56])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 4, 0], [1, 512], [1, stage_29_v56], target_memory=pl.Mem.Vec)
                        stage_29_v171: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v170, pad_value=pl.PadValue.min)
                        stage_29_v172: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v171, -3.4028234663852886e+38)
                        stage_29_v173: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v169, dtype=pl.UINT32)
                        stage_29_v174: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v172, stage_29_v173)
                        stage_29_v175: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v174, pl.const(64, pl.INT32))
                        stage_29_v176: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v175, pl.const(256, pl.INT32))
                        stage_29_v177 = pl.tile.store(stage_29_v176, [stage_29_v57, 0], temporary_46)
                    elif stage_29_v56 <= 1024:
                        stage_29_v239: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                        stage_29_v240: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v239, stage_29_v117)
                        stage_29_v241: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v56])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 4, 0], [1, 1024], [1, stage_29_v56], target_memory=pl.Mem.Vec)
                        stage_29_v242: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v241, pad_value=pl.PadValue.min)
                        stage_29_v243: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v242, -3.4028234663852886e+38)
                        stage_29_v244: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v240, dtype=pl.UINT32)
                        stage_29_v245: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v243, stage_29_v244)
                        stage_29_v246: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v245, pl.const(64, pl.INT32))
                        stage_29_v247: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v246, pl.const(256, pl.INT32))
                        stage_29_v248: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v247, [1, 1024], [0, 0])
                        stage_29_v249: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v247, [1, 1024], [0, 1024])
                        stage_29_v250: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v251: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v248, stage_29_v249, stage_29_v250, exhausted=False)
                        stage_29_v252: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v251, [1, 1024], [0, 0])
                        stage_29_v253 = pl.tile.store(stage_29_v252, [stage_29_v57, 0], temporary_46)
                    elif stage_29_v56 <= 2048:
                        stage_29_v323: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                        stage_29_v324: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v323, stage_29_v117)
                        stage_29_v325: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v56])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 4, 0], [1, 2048], [1, stage_29_v56], target_memory=pl.Mem.Vec)
                        stage_29_v326: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v325, pad_value=pl.PadValue.min)
                        stage_29_v327: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v326, -3.4028234663852886e+38)
                        stage_29_v328: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v324, dtype=pl.UINT32)
                        stage_29_v329: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v327, stage_29_v328)
                        stage_29_v330: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v329, pl.const(64, pl.INT32))
                        stage_29_v331: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v330, pl.const(256, pl.INT32))
                        stage_29_v332: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v331, pl.const(1024, pl.INT32))
                        stage_29_v333: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v332, [1, 1024], [0, 0])
                        stage_29_v334 = pl.tile.store(stage_29_v333, [stage_29_v57, 0], temporary_46)
                    elif stage_29_v56 <= 2560:
                        stage_29_v405: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2560], dtype=pl.INT32, descending=False)
                        stage_29_v406: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v405, stage_29_v117)
                        stage_29_v407: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v56])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 4, 0], [1, 2560], [1, stage_29_v56], target_memory=pl.Mem.Vec)
                        stage_29_v408: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v407, pad_value=pl.PadValue.min)
                        stage_29_v409: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v408, -3.4028234663852886e+38)
                        stage_29_v410: pl.Tile[[1, 2560], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v406, dtype=pl.UINT32)
                        stage_29_v411: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v409, stage_29_v410)
                        stage_29_v412: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v411, pl.const(64, pl.INT32))
                        stage_29_v413: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v412, pl.const(256, pl.INT32))
                        stage_29_v414: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v413, 1, 4096)
                        stage_29_v415: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v414, pl.const(1024, pl.INT32))
                        stage_29_v416: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v415, [1, 1024], [0, 0])
                        stage_29_v417: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v413, [1, 1024], [0, 4096])
                        stage_29_v418: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v419: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v417, stage_29_v416, stage_29_v418, exhausted=False)
                        stage_29_v420: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v419, [1, 1024], [0, 0])
                        stage_29_v421 = pl.tile.store(stage_29_v420, [stage_29_v57, 0], temporary_46)
                    elif stage_29_v56 <= 3072:
                        stage_29_v541: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 3072], dtype=pl.INT32, descending=False)
                        stage_29_v542: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v541, stage_29_v117)
                        stage_29_v543: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v56])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 4, 0], [1, 3072], [1, stage_29_v56], target_memory=pl.Mem.Vec)
                        stage_29_v544: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v543, pad_value=pl.PadValue.min)
                        stage_29_v545: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v544, -3.4028234663852886e+38)
                        stage_29_v546: pl.Tile[[1, 3072], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v542, dtype=pl.UINT32)
                        stage_29_v547: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v545, stage_29_v546)
                        stage_29_v548: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v547, pl.const(64, pl.INT32))
                        stage_29_v549: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v548, pl.const(256, pl.INT32))
                        stage_29_v550: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v549, 1, 4096)
                        stage_29_v551: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v550, pl.const(1024, pl.INT32))
                        stage_29_v552: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v551, [1, 1024], [0, 0])
                        stage_29_v553: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v549, [1, 1024], [0, 4096])
                        stage_29_v554: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v549, [1, 1024], [0, 5120])
                        stage_29_v555: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 3072], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v556: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v553, stage_29_v554, stage_29_v552, stage_29_v555, exhausted=False)
                        stage_29_v557: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v556, [1, 1024], [0, 0])
                        stage_29_v558 = pl.tile.store(stage_29_v557, [stage_29_v57, 0], temporary_46)
                    else:
                        stage_29_v559: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 4096], dtype=pl.INT32, descending=False)
                        stage_29_v560: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v559, stage_29_v117)
                        stage_29_v561: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v56])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 4, 0], [1, 4096], [1, stage_29_v56], target_memory=pl.Mem.Vec)
                        stage_29_v562: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v561, pad_value=pl.PadValue.min)
                        stage_29_v563: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v562, -3.4028234663852886e+38)
                        stage_29_v564: pl.Tile[[1, 4096], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v560, dtype=pl.UINT32)
                        stage_29_v565: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v563, stage_29_v564)
                        stage_29_v566: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v565, pl.const(64, pl.INT32))
                        stage_29_v567: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v566, pl.const(256, pl.INT32))
                        stage_29_v568: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v567, pl.const(1024, pl.INT32))
                        stage_29_v569: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v568, [1, 1024], [0, 0])
                        stage_29_v570: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v568, [1, 1024], [0, 4096])
                        stage_29_v571: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v572: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v570, stage_29_v569, stage_29_v571, exhausted=False)
                        stage_29_v573: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v572, [1, 1024], [0, 0])
                        stage_29_v574 = pl.tile.store(stage_29_v573, [stage_29_v57, 0], temporary_46)
                else:
                    stage_29_v118: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
                    stage_29_v119 = pl.tile.store(stage_29_v118, [stage_29_v57, 0], temporary_46)
                stage_29_v58: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v74 + 3])
                stage_29_v59: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v58 + 1) // 4), 262144), 0)
                stage_29_v60: pl.Scalar[pl.INDEX] = stage_29_v30 + stage_29_v41
                stage_29_v61: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v59 - stage_29_v60, stage_29_v39), 0)
                stage_29_v62: pl.Scalar[pl.INDEX] = stage_29_v74 * 64 + stage_29_v73 * 2 + stage_29_v19 + 192
                if 0 < stage_29_v61:
                    stage_29_v120: pl.Scalar[pl.INT32] = pl.cast(stage_29_v60, pl.INT32)
                    if stage_29_v61 <= 512:
                        stage_29_v178: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                        stage_29_v179: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v178, stage_29_v120)
                        stage_29_v180: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v61])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 6, 0], [1, 512], [1, stage_29_v61], target_memory=pl.Mem.Vec)
                        stage_29_v181: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v180, pad_value=pl.PadValue.min)
                        stage_29_v182: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v181, -3.4028234663852886e+38)
                        stage_29_v183: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v179, dtype=pl.UINT32)
                        stage_29_v184: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v182, stage_29_v183)
                        stage_29_v185: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v184, pl.const(64, pl.INT32))
                        stage_29_v186: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v185, pl.const(256, pl.INT32))
                        stage_29_v187 = pl.tile.store(stage_29_v186, [stage_29_v62, 0], temporary_46)
                    elif stage_29_v61 <= 1024:
                        stage_29_v254: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                        stage_29_v255: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v254, stage_29_v120)
                        stage_29_v256: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v61])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 6, 0], [1, 1024], [1, stage_29_v61], target_memory=pl.Mem.Vec)
                        stage_29_v257: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v256, pad_value=pl.PadValue.min)
                        stage_29_v258: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v257, -3.4028234663852886e+38)
                        stage_29_v259: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v255, dtype=pl.UINT32)
                        stage_29_v260: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v258, stage_29_v259)
                        stage_29_v261: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v260, pl.const(64, pl.INT32))
                        stage_29_v262: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v261, pl.const(256, pl.INT32))
                        stage_29_v263: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v262, [1, 1024], [0, 0])
                        stage_29_v264: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v262, [1, 1024], [0, 1024])
                        stage_29_v265: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v266: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v263, stage_29_v264, stage_29_v265, exhausted=False)
                        stage_29_v267: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v266, [1, 1024], [0, 0])
                        stage_29_v268 = pl.tile.store(stage_29_v267, [stage_29_v62, 0], temporary_46)
                    elif stage_29_v61 <= 2048:
                        stage_29_v335: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                        stage_29_v336: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v335, stage_29_v120)
                        stage_29_v337: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v61])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 6, 0], [1, 2048], [1, stage_29_v61], target_memory=pl.Mem.Vec)
                        stage_29_v338: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v337, pad_value=pl.PadValue.min)
                        stage_29_v339: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v338, -3.4028234663852886e+38)
                        stage_29_v340: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v336, dtype=pl.UINT32)
                        stage_29_v341: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v339, stage_29_v340)
                        stage_29_v342: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v341, pl.const(64, pl.INT32))
                        stage_29_v343: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v342, pl.const(256, pl.INT32))
                        stage_29_v344: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v343, pl.const(1024, pl.INT32))
                        stage_29_v345: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v344, [1, 1024], [0, 0])
                        stage_29_v346 = pl.tile.store(stage_29_v345, [stage_29_v62, 0], temporary_46)
                    elif stage_29_v61 <= 2560:
                        stage_29_v422: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2560], dtype=pl.INT32, descending=False)
                        stage_29_v423: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v422, stage_29_v120)
                        stage_29_v424: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v61])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 6, 0], [1, 2560], [1, stage_29_v61], target_memory=pl.Mem.Vec)
                        stage_29_v425: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v424, pad_value=pl.PadValue.min)
                        stage_29_v426: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v425, -3.4028234663852886e+38)
                        stage_29_v427: pl.Tile[[1, 2560], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v423, dtype=pl.UINT32)
                        stage_29_v428: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v426, stage_29_v427)
                        stage_29_v429: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v428, pl.const(64, pl.INT32))
                        stage_29_v430: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v429, pl.const(256, pl.INT32))
                        stage_29_v431: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v430, 1, 4096)
                        stage_29_v432: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v431, pl.const(1024, pl.INT32))
                        stage_29_v433: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v432, [1, 1024], [0, 0])
                        stage_29_v434: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v430, [1, 1024], [0, 4096])
                        stage_29_v435: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v436: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v434, stage_29_v433, stage_29_v435, exhausted=False)
                        stage_29_v437: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v436, [1, 1024], [0, 0])
                        stage_29_v438 = pl.tile.store(stage_29_v437, [stage_29_v62, 0], temporary_46)
                    elif stage_29_v61 <= 3072:
                        stage_29_v575: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 3072], dtype=pl.INT32, descending=False)
                        stage_29_v576: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v575, stage_29_v120)
                        stage_29_v577: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v61])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 6, 0], [1, 3072], [1, stage_29_v61], target_memory=pl.Mem.Vec)
                        stage_29_v578: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v577, pad_value=pl.PadValue.min)
                        stage_29_v579: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v578, -3.4028234663852886e+38)
                        stage_29_v580: pl.Tile[[1, 3072], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v576, dtype=pl.UINT32)
                        stage_29_v581: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v579, stage_29_v580)
                        stage_29_v582: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v581, pl.const(64, pl.INT32))
                        stage_29_v583: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v582, pl.const(256, pl.INT32))
                        stage_29_v584: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v583, 1, 4096)
                        stage_29_v585: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v584, pl.const(1024, pl.INT32))
                        stage_29_v586: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v585, [1, 1024], [0, 0])
                        stage_29_v587: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v583, [1, 1024], [0, 4096])
                        stage_29_v588: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v583, [1, 1024], [0, 5120])
                        stage_29_v589: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 3072], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v590: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v587, stage_29_v588, stage_29_v586, stage_29_v589, exhausted=False)
                        stage_29_v591: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v590, [1, 1024], [0, 0])
                        stage_29_v592 = pl.tile.store(stage_29_v591, [stage_29_v62, 0], temporary_46)
                    else:
                        stage_29_v593: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 4096], dtype=pl.INT32, descending=False)
                        stage_29_v594: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v593, stage_29_v120)
                        stage_29_v595: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v61])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 6, 0], [1, 4096], [1, stage_29_v61], target_memory=pl.Mem.Vec)
                        stage_29_v596: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v595, pad_value=pl.PadValue.min)
                        stage_29_v597: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v596, -3.4028234663852886e+38)
                        stage_29_v598: pl.Tile[[1, 4096], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v594, dtype=pl.UINT32)
                        stage_29_v599: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v597, stage_29_v598)
                        stage_29_v600: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v599, pl.const(64, pl.INT32))
                        stage_29_v601: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v600, pl.const(256, pl.INT32))
                        stage_29_v602: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v601, pl.const(1024, pl.INT32))
                        stage_29_v603: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v602, [1, 1024], [0, 0])
                        stage_29_v604: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v602, [1, 1024], [0, 4096])
                        stage_29_v605: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v606: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v604, stage_29_v603, stage_29_v605, exhausted=False)
                        stage_29_v607: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v606, [1, 1024], [0, 0])
                        stage_29_v608 = pl.tile.store(stage_29_v607, [stage_29_v62, 0], temporary_46)
                else:
                    stage_29_v121: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
                    stage_29_v122 = pl.tile.store(stage_29_v121, [stage_29_v62, 0], temporary_46)
                stage_29_v63: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v74 + 4])
                stage_29_v64: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v63 + 1) // 4), 262144), 0)
                stage_29_v65: pl.Scalar[pl.INDEX] = stage_29_v30 + stage_29_v41
                stage_29_v66: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v64 - stage_29_v65, stage_29_v39), 0)
                stage_29_v67: pl.Scalar[pl.INDEX] = stage_29_v74 * 64 + stage_29_v73 * 2 + stage_29_v19 + 256
                if 0 < stage_29_v66:
                    stage_29_v123: pl.Scalar[pl.INT32] = pl.cast(stage_29_v65, pl.INT32)
                    if stage_29_v66 <= 512:
                        stage_29_v188: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                        stage_29_v189: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v188, stage_29_v123)
                        stage_29_v190: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v66])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 8, 0], [1, 512], [1, stage_29_v66], target_memory=pl.Mem.Vec)
                        stage_29_v191: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v190, pad_value=pl.PadValue.min)
                        stage_29_v192: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v191, -3.4028234663852886e+38)
                        stage_29_v193: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v189, dtype=pl.UINT32)
                        stage_29_v194: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v192, stage_29_v193)
                        stage_29_v195: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v194, pl.const(64, pl.INT32))
                        stage_29_v196: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v195, pl.const(256, pl.INT32))
                        stage_29_v197 = pl.tile.store(stage_29_v196, [stage_29_v67, 0], temporary_46)
                    elif stage_29_v66 <= 1024:
                        stage_29_v269: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                        stage_29_v270: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v269, stage_29_v123)
                        stage_29_v271: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v66])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 8, 0], [1, 1024], [1, stage_29_v66], target_memory=pl.Mem.Vec)
                        stage_29_v272: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v271, pad_value=pl.PadValue.min)
                        stage_29_v273: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v272, -3.4028234663852886e+38)
                        stage_29_v274: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v270, dtype=pl.UINT32)
                        stage_29_v275: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v273, stage_29_v274)
                        stage_29_v276: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v275, pl.const(64, pl.INT32))
                        stage_29_v277: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v276, pl.const(256, pl.INT32))
                        stage_29_v278: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v277, [1, 1024], [0, 0])
                        stage_29_v279: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v277, [1, 1024], [0, 1024])
                        stage_29_v280: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v281: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v278, stage_29_v279, stage_29_v280, exhausted=False)
                        stage_29_v282: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v281, [1, 1024], [0, 0])
                        stage_29_v283 = pl.tile.store(stage_29_v282, [stage_29_v67, 0], temporary_46)
                    elif stage_29_v66 <= 2048:
                        stage_29_v347: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                        stage_29_v348: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v347, stage_29_v123)
                        stage_29_v349: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v66])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 8, 0], [1, 2048], [1, stage_29_v66], target_memory=pl.Mem.Vec)
                        stage_29_v350: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v349, pad_value=pl.PadValue.min)
                        stage_29_v351: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v350, -3.4028234663852886e+38)
                        stage_29_v352: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v348, dtype=pl.UINT32)
                        stage_29_v353: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v351, stage_29_v352)
                        stage_29_v354: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v353, pl.const(64, pl.INT32))
                        stage_29_v355: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v354, pl.const(256, pl.INT32))
                        stage_29_v356: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v355, pl.const(1024, pl.INT32))
                        stage_29_v357: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v356, [1, 1024], [0, 0])
                        stage_29_v358 = pl.tile.store(stage_29_v357, [stage_29_v67, 0], temporary_46)
                    elif stage_29_v66 <= 2560:
                        stage_29_v439: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2560], dtype=pl.INT32, descending=False)
                        stage_29_v440: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v439, stage_29_v123)
                        stage_29_v441: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v66])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 8, 0], [1, 2560], [1, stage_29_v66], target_memory=pl.Mem.Vec)
                        stage_29_v442: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v441, pad_value=pl.PadValue.min)
                        stage_29_v443: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v442, -3.4028234663852886e+38)
                        stage_29_v444: pl.Tile[[1, 2560], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v440, dtype=pl.UINT32)
                        stage_29_v445: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v443, stage_29_v444)
                        stage_29_v446: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v445, pl.const(64, pl.INT32))
                        stage_29_v447: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v446, pl.const(256, pl.INT32))
                        stage_29_v448: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v447, 1, 4096)
                        stage_29_v449: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v448, pl.const(1024, pl.INT32))
                        stage_29_v450: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v449, [1, 1024], [0, 0])
                        stage_29_v451: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v447, [1, 1024], [0, 4096])
                        stage_29_v452: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v453: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v451, stage_29_v450, stage_29_v452, exhausted=False)
                        stage_29_v454: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v453, [1, 1024], [0, 0])
                        stage_29_v455 = pl.tile.store(stage_29_v454, [stage_29_v67, 0], temporary_46)
                    elif stage_29_v66 <= 3072:
                        stage_29_v609: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 3072], dtype=pl.INT32, descending=False)
                        stage_29_v610: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v609, stage_29_v123)
                        stage_29_v611: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v66])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 8, 0], [1, 3072], [1, stage_29_v66], target_memory=pl.Mem.Vec)
                        stage_29_v612: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v611, pad_value=pl.PadValue.min)
                        stage_29_v613: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v612, -3.4028234663852886e+38)
                        stage_29_v614: pl.Tile[[1, 3072], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v610, dtype=pl.UINT32)
                        stage_29_v615: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v613, stage_29_v614)
                        stage_29_v616: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v615, pl.const(64, pl.INT32))
                        stage_29_v617: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v616, pl.const(256, pl.INT32))
                        stage_29_v618: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v617, 1, 4096)
                        stage_29_v619: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v618, pl.const(1024, pl.INT32))
                        stage_29_v620: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v619, [1, 1024], [0, 0])
                        stage_29_v621: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v617, [1, 1024], [0, 4096])
                        stage_29_v622: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v617, [1, 1024], [0, 5120])
                        stage_29_v623: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 3072], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v624: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v621, stage_29_v622, stage_29_v620, stage_29_v623, exhausted=False)
                        stage_29_v625: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v624, [1, 1024], [0, 0])
                        stage_29_v626 = pl.tile.store(stage_29_v625, [stage_29_v67, 0], temporary_46)
                    else:
                        stage_29_v627: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 4096], dtype=pl.INT32, descending=False)
                        stage_29_v628: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v627, stage_29_v123)
                        stage_29_v629: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v66])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 8, 0], [1, 4096], [1, stage_29_v66], target_memory=pl.Mem.Vec)
                        stage_29_v630: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v629, pad_value=pl.PadValue.min)
                        stage_29_v631: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v630, -3.4028234663852886e+38)
                        stage_29_v632: pl.Tile[[1, 4096], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v628, dtype=pl.UINT32)
                        stage_29_v633: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v631, stage_29_v632)
                        stage_29_v634: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v633, pl.const(64, pl.INT32))
                        stage_29_v635: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v634, pl.const(256, pl.INT32))
                        stage_29_v636: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v635, pl.const(1024, pl.INT32))
                        stage_29_v637: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v636, [1, 1024], [0, 0])
                        stage_29_v638: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v636, [1, 1024], [0, 4096])
                        stage_29_v639: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v640: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v638, stage_29_v637, stage_29_v639, exhausted=False)
                        stage_29_v641: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v640, [1, 1024], [0, 0])
                        stage_29_v642 = pl.tile.store(stage_29_v641, [stage_29_v67, 0], temporary_46)
                else:
                    stage_29_v124: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
                    stage_29_v125 = pl.tile.store(stage_29_v124, [stage_29_v67, 0], temporary_46)
                stage_29_v68: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v74 + 5])
                stage_29_v69: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v68 + 1) // 4), 262144), 0)
                stage_29_v70: pl.Scalar[pl.INDEX] = stage_29_v30 + stage_29_v41
                stage_29_v71: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v69 - stage_29_v70, stage_29_v39), 0)
                stage_29_v72: pl.Scalar[pl.INDEX] = stage_29_v74 * 64 + stage_29_v73 * 2 + stage_29_v19 + 320
                if 0 < stage_29_v71:
                    stage_29_v126: pl.Scalar[pl.INT32] = pl.cast(stage_29_v70, pl.INT32)
                    if stage_29_v71 <= 512:
                        stage_29_v198: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                        stage_29_v199: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v198, stage_29_v126)
                        stage_29_v200: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v71])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 10, 0], [1, 512], [1, stage_29_v71], target_memory=pl.Mem.Vec)
                        stage_29_v201: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v200, pad_value=pl.PadValue.min)
                        stage_29_v202: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v201, -3.4028234663852886e+38)
                        stage_29_v203: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v199, dtype=pl.UINT32)
                        stage_29_v204: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v202, stage_29_v203)
                        stage_29_v205: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v204, pl.const(64, pl.INT32))
                        stage_29_v206: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v205, pl.const(256, pl.INT32))
                        stage_29_v207 = pl.tile.store(stage_29_v206, [stage_29_v72, 0], temporary_46)
                    elif stage_29_v71 <= 1024:
                        stage_29_v284: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                        stage_29_v285: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v284, stage_29_v126)
                        stage_29_v286: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v71])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 10, 0], [1, 1024], [1, stage_29_v71], target_memory=pl.Mem.Vec)
                        stage_29_v287: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v286, pad_value=pl.PadValue.min)
                        stage_29_v288: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v287, -3.4028234663852886e+38)
                        stage_29_v289: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v285, dtype=pl.UINT32)
                        stage_29_v290: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v288, stage_29_v289)
                        stage_29_v291: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v290, pl.const(64, pl.INT32))
                        stage_29_v292: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v291, pl.const(256, pl.INT32))
                        stage_29_v293: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v292, [1, 1024], [0, 0])
                        stage_29_v294: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v292, [1, 1024], [0, 1024])
                        stage_29_v295: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v296: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v293, stage_29_v294, stage_29_v295, exhausted=False)
                        stage_29_v297: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v296, [1, 1024], [0, 0])
                        stage_29_v298 = pl.tile.store(stage_29_v297, [stage_29_v72, 0], temporary_46)
                    elif stage_29_v71 <= 2048:
                        stage_29_v359: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                        stage_29_v360: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v359, stage_29_v126)
                        stage_29_v361: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v71])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 10, 0], [1, 2048], [1, stage_29_v71], target_memory=pl.Mem.Vec)
                        stage_29_v362: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v361, pad_value=pl.PadValue.min)
                        stage_29_v363: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v362, -3.4028234663852886e+38)
                        stage_29_v364: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v360, dtype=pl.UINT32)
                        stage_29_v365: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v363, stage_29_v364)
                        stage_29_v366: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v365, pl.const(64, pl.INT32))
                        stage_29_v367: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v366, pl.const(256, pl.INT32))
                        stage_29_v368: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v367, pl.const(1024, pl.INT32))
                        stage_29_v369: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v368, [1, 1024], [0, 0])
                        stage_29_v370 = pl.tile.store(stage_29_v369, [stage_29_v72, 0], temporary_46)
                    elif stage_29_v71 <= 2560:
                        stage_29_v456: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2560], dtype=pl.INT32, descending=False)
                        stage_29_v457: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v456, stage_29_v126)
                        stage_29_v458: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v71])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 10, 0], [1, 2560], [1, stage_29_v71], target_memory=pl.Mem.Vec)
                        stage_29_v459: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v458, pad_value=pl.PadValue.min)
                        stage_29_v460: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v459, -3.4028234663852886e+38)
                        stage_29_v461: pl.Tile[[1, 2560], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v457, dtype=pl.UINT32)
                        stage_29_v462: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v460, stage_29_v461)
                        stage_29_v463: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v462, pl.const(64, pl.INT32))
                        stage_29_v464: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v463, pl.const(256, pl.INT32))
                        stage_29_v465: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v464, 1, 4096)
                        stage_29_v466: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v465, pl.const(1024, pl.INT32))
                        stage_29_v467: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v466, [1, 1024], [0, 0])
                        stage_29_v468: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v464, [1, 1024], [0, 4096])
                        stage_29_v469: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v470: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v468, stage_29_v467, stage_29_v469, exhausted=False)
                        stage_29_v471: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v470, [1, 1024], [0, 0])
                        stage_29_v472 = pl.tile.store(stage_29_v471, [stage_29_v72, 0], temporary_46)
                    elif stage_29_v71 <= 3072:
                        stage_29_v643: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 3072], dtype=pl.INT32, descending=False)
                        stage_29_v644: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v643, stage_29_v126)
                        stage_29_v645: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v71])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 10, 0], [1, 3072], [1, stage_29_v71], target_memory=pl.Mem.Vec)
                        stage_29_v646: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v645, pad_value=pl.PadValue.min)
                        stage_29_v647: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v646, -3.4028234663852886e+38)
                        stage_29_v648: pl.Tile[[1, 3072], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v644, dtype=pl.UINT32)
                        stage_29_v649: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v647, stage_29_v648)
                        stage_29_v650: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v649, pl.const(64, pl.INT32))
                        stage_29_v651: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v650, pl.const(256, pl.INT32))
                        stage_29_v652: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v651, 1, 4096)
                        stage_29_v653: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v652, pl.const(1024, pl.INT32))
                        stage_29_v654: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v653, [1, 1024], [0, 0])
                        stage_29_v655: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v651, [1, 1024], [0, 4096])
                        stage_29_v656: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v651, [1, 1024], [0, 5120])
                        stage_29_v657: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 3072], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v658: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v655, stage_29_v656, stage_29_v654, stage_29_v657, exhausted=False)
                        stage_29_v659: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v658, [1, 1024], [0, 0])
                        stage_29_v660 = pl.tile.store(stage_29_v659, [stage_29_v72, 0], temporary_46)
                    else:
                        stage_29_v661: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 4096], dtype=pl.INT32, descending=False)
                        stage_29_v662: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v661, stage_29_v126)
                        stage_29_v663: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v71])] = pl.tile.load(temporary_47, [stage_29_v14 * 12 + stage_29_v19 + 10, 0], [1, 4096], [1, stage_29_v71], target_memory=pl.Mem.Vec)
                        stage_29_v664: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v663, pad_value=pl.PadValue.min)
                        stage_29_v665: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v664, -3.4028234663852886e+38)
                        stage_29_v666: pl.Tile[[1, 4096], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v662, dtype=pl.UINT32)
                        stage_29_v667: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v665, stage_29_v666)
                        stage_29_v668: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v667, pl.const(64, pl.INT32))
                        stage_29_v669: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v668, pl.const(256, pl.INT32))
                        stage_29_v670: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v669, pl.const(1024, pl.INT32))
                        stage_29_v671: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v670, [1, 1024], [0, 0])
                        stage_29_v672: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v670, [1, 1024], [0, 4096])
                        stage_29_v673: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v674: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v672, stage_29_v671, stage_29_v673, exhausted=False)
                        stage_29_v675: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v674, [1, 1024], [0, 0])
                        stage_29_v676 = pl.tile.store(stage_29_v675, [stage_29_v72, 0], temporary_46)
                else:
                    stage_29_v127: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
                    stage_29_v128 = pl.tile.store(stage_29_v127, [stage_29_v72, 0], temporary_46)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_29 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_29)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_30 = 28944 + core_id * 6
    mapped_block_30 = pl.cast(pl.tensor.read(records, [record_30]), pl.INDEX)
    block_30 = pl.max(mapped_block_30, 0)
    if mapped_block_30 >= 0:
        subslot_30 = pl.cast(pl.tensor.read(records, [record_30 + 1]), pl.INDEX)
        done_30 = pl.tensor.read(records, [record_30 + 2])
        wait_base_30 = pl.cast(pl.tensor.read(records, [record_30 + 3]), pl.INDEX)
        wait_count_30 = pl.cast(pl.tensor.read(records, [record_30 + 4]), pl.INDEX)
        publish_30 = pl.tensor.read(records, [record_30 + 5])
        for dependency_30 in pl.range(wait_count_30):
            wait_index_30 = (wait_base_30 + dependency_30) * 2
            other_30 = pl.cast(pl.tensor.read(waits, [wait_index_30]), pl.INDEX)
            required_30 = pl.tensor.read(waits, [wait_index_30 + 1])
            target_30 = generation_base + required_30
            flag_30 = other_30 * 64
            pl.system.cacheinvalid(progress, [1], [flag_30])
            initial_30 = pl.tensor.read(progress, [flag_30])
            for (ready_30,) in pl.while_(init_values=(initial_30,)):
                pl.cond(ready_30 < target_30)
                pl.system.cacheinvalid(progress, [1], [flag_30])
                after_30 = pl.yield_(pl.tensor.read(progress, [flag_30]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_30_v6: pl.Scalar[pl.INDEX] = block_30
        stage_30_v7: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        for stage_30_v8 in pl.range(stage_30_v6, stage_30_v7, 48):
            stage_30_v9: pl.Scalar[pl.INDEX] = stage_30_v8 // 6
            stage_30_v10: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_30_v8])
            stage_30_v11: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_30_v9])
            stage_30_v12: pl.Scalar[pl.INDEX] = pl.cast(stage_30_v11, pl.INDEX) // 4
            stage_30_v13: pl.Scalar[pl.INDEX] = pl.min(stage_30_v12, (stage_30_v10 + 1) // 4)
            stage_30_v14: pl.Scalar[pl.INDEX] = pl.min(stage_30_v13, 262144)
            if 0 < stage_30_v14:
                stage_30_v15: pl.Scalar[pl.INDEX] = (stage_30_v14 + 8191) // 8192
                stage_30_v16: pl.Scalar[pl.INDEX] = stage_30_v15 * 2
                stage_30_v17: pl.Scalar[pl.INDEX] = stage_30_v8 * 64
                for stage_30_v18 in pl.range(1, stage_30_v16):
                    stage_30_v25: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_30_v17, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_30_v26: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_30_v17 + stage_30_v18, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_30_v27: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_30_v28: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(stage_30_v26, stage_30_v25, stage_30_v27, exhausted=False)
                    stage_30_v29: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_30_v28, [1, 1024], [0, 0])
                    pl.tile.store(stage_30_v29, [stage_30_v17, 0], temporary_46)
                stage_30_v19: pl.Scalar[pl.INDEX] = stage_30_v17
                stage_30_v20: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_30_v19, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                stage_30_v21: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather_mask(stage_30_v20, mask_pattern=1, output_dtype=pl.FP32)
                pl.tile.store(stage_30_v21, [stage_30_v8, 0], arg_idx_topk_scores)
                stage_30_v22: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.gather_mask(stage_30_v20, mask_pattern=2, output_dtype=pl.INT32)
                if 512 <= stage_30_v14:
                    pl.tile.store(stage_30_v22, [stage_30_v8, 0], arg_idx_topk)
                else:
                    stage_30_v30: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                    for stage_30_v31 in pl.range(stage_30_v14):
                        stage_30_v32: pl.Scalar[pl.INT32] = pl.tile.read(stage_30_v22, [0, stage_30_v31])
                        pl.tile.write(stage_30_v30, [0, stage_30_v31], stage_30_v32)
                    pl.tile.store(stage_30_v30, [stage_30_v8, 0], arg_idx_topk)
            else:
                stage_30_v23: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e+38)
                pl.tile.store(stage_30_v23, [stage_30_v8, 0], arg_idx_topk_scores)
                stage_30_v24: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                pl.tile.store(stage_30_v24, [stage_30_v8, 0], arg_idx_topk)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_30 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_30)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_31 = 29376 + core_id * 6
    mapped_block_31 = pl.cast(pl.tensor.read(records, [record_31]), pl.INDEX)
    block_31 = pl.max(mapped_block_31, 0)
    if mapped_block_31 >= 0:
        subslot_31 = pl.cast(pl.tensor.read(records, [record_31 + 1]), pl.INDEX)
        done_31 = pl.tensor.read(records, [record_31 + 2])
        wait_base_31 = pl.cast(pl.tensor.read(records, [record_31 + 3]), pl.INDEX)
        wait_count_31 = pl.cast(pl.tensor.read(records, [record_31 + 4]), pl.INDEX)
        publish_31 = pl.tensor.read(records, [record_31 + 5])
        for dependency_31 in pl.range(wait_count_31):
            wait_index_31 = (wait_base_31 + dependency_31) * 2
            other_31 = pl.cast(pl.tensor.read(waits, [wait_index_31]), pl.INDEX)
            required_31 = pl.tensor.read(waits, [wait_index_31 + 1])
            target_31 = generation_base + required_31
            flag_31 = other_31 * 64
            pl.system.cacheinvalid(progress, [1], [flag_31])
            initial_31 = pl.tensor.read(progress, [flag_31])
            for (ready_31,) in pl.while_(init_values=(initial_31,)):
                pl.cond(ready_31 < target_31)
                pl.system.cacheinvalid(progress, [1], [flag_31])
                after_31 = pl.yield_(pl.tensor.read(progress, [flag_31]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_31_v8: pl.Scalar[pl.INDEX] = block_31
        for (stage_31_v9, (stage_31_v63, stage_31_v64)) in pl.range(stage_31_v8 * 8, 144, 128, init_values=(temporary_53, temporary_52)):
            stage_31_v10: pl.Scalar[pl.INDEX] = pl.min(144 - stage_31_v9, 8)
            stage_31_v11: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.load(arg_idx_topk, [stage_31_v9, 0], [8, 512], [stage_31_v10, 512], target_memory=pl.Mem.Vec)
            stage_31_v12: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.cast(stage_31_v11, target_type=pl.FP32, mode='round')
            stage_31_v13: pl.Tile[[8, 1], pl.INT64, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 1])] = pl.tile.load(view_107, [stage_31_v9, 0], [8, 1], [stage_31_v10, 1], target_memory=pl.Mem.Vec)
            stage_31_v14: pl.Tile[[1, 8], pl.INT64, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v10])] = pl.tile.reshape(stage_31_v13, [1, 8])
            stage_31_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v10])] = pl.tile.cast(stage_31_v14, target_type=pl.FP32, mode='round')
            stage_31_v16: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 1])] = pl.tile.reshape(stage_31_v15, [8, 1])
            stage_31_v17: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v10])] = pl.tile.reshape(stage_31_v16, [1, 8])
            stage_31_v18: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v10])] = pl.tile.adds(stage_31_v17, 1.0)
            stage_31_v19: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 1])] = pl.tile.reshape(stage_31_v18, [8, 1])
            stage_31_v20: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v10])] = pl.tile.reshape(stage_31_v19, [1, 8])
            stage_31_v21: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v10])] = pl.tile.muls(stage_31_v20, 0.25)
            stage_31_v22: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 1])] = pl.tile.reshape(stage_31_v21, [8, 1])
            stage_31_v23: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v10])] = pl.tile.reshape(stage_31_v22, [1, 8])
            stage_31_v24: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v10])] = pl.tile.cast(stage_31_v23, target_type=pl.INT32, mode='trunc')
            stage_31_v25: pl.Tile[[8, 1], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 1])] = pl.tile.reshape(stage_31_v24, [8, 1])
            stage_31_v26: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v10])] = pl.tile.reshape(stage_31_v25, [1, 8])
            stage_31_v27: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v10])] = pl.tile.cast(stage_31_v26, target_type=pl.FP32, mode='round')
            stage_31_v28: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 1])] = pl.tile.reshape(stage_31_v27, [8, 1])
            stage_31_v29: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=1.0)
            stage_31_v30: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_31_v29, stage_31_v28)
            stage_31_v31: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.adds(stage_31_v12, 1.0)
            stage_31_v32: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.maximums(stage_31_v31, 0.0)
            stage_31_v33: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.minimums(stage_31_v32, 1.0)
            stage_31_v34: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_31_v30, stage_31_v12)
            stage_31_v35: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.maximums(stage_31_v34, 0.0)
            stage_31_v36: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(stage_31_v35, 1.0)
            stage_31_v37: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.mul(stage_31_v33, stage_31_v36)
            stage_31_v38: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.adds(stage_31_v12, 1.0)
            stage_31_v39: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.mul(stage_31_v37, stage_31_v38)
            stage_31_v40: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.subs(stage_31_v39, 1.0)
            stage_31_v41: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.cast(stage_31_v40, target_type=pl.INT32, mode='round')
            stage_31_v42: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.set_validshape(stage_31_v41, stage_31_v10, 512)
            stage_31_v43 = pl.tile.store(stage_31_v42, [stage_31_v9, 0], stage_31_v63)
            stage_31_v44: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.minimums(stage_31_v40, 0.0)
            stage_31_v45: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.muls(stage_31_v44, 1e+20)
            stage_31_v46: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 512])] = pl.tile.set_validshape(stage_31_v45, stage_31_v10, 512)
            stage_31_v47 = pl.tile.store(stage_31_v46, [stage_31_v9, 128], stage_31_v64)
            for stage_31_v48 in pl.range(1, 5):
                stage_31_v65: pl.Scalar[pl.INDEX] = (stage_31_v48 - 1) * 128
                stage_31_v66: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(stage_31_v10, pl.const(8, pl.INDEX)), 128])] = pl.tile.slice(stage_31_v37, [8, 128], [0, stage_31_v65], [stage_31_v10, 128])
                stage_31_v67: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_31_v68: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(stage_31_v10, pl.const(8, pl.INDEX)), 1])] = pl.tile.row_max(stage_31_v66, stage_31_v67)
                for stage_31_v69 in pl.range(stage_31_v10):
                    stage_31_v90: pl.Scalar[pl.FP32] = pl.tile.read(stage_31_v68, [stage_31_v69, 0])
                    stage_31_v91: pl.Scalar[pl.INT32] = pl.cast(stage_31_v90, pl.INT32)
                    pl.tensor.write(temporary_54, [stage_31_v9 + stage_31_v69, stage_31_v48], stage_31_v91)
            stage_31_v49: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 128], dtype=pl.INT32, descending=False)
            stage_31_v50: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_31_v49, target_type=pl.FP32, mode='round')
            stage_31_v51: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for stage_31_v52 in pl.range(8):
                stage_31_v70: pl.Scalar[pl.INDEX] = stage_31_v9 + pl.min(stage_31_v52, stage_31_v10 - 1)
                stage_31_v71: pl.Scalar[pl.INDEX] = stage_31_v70 // 6
                stage_31_v72: pl.Scalar[pl.INT64] = pl.tensor.read(view_107, [stage_31_v70, 0])
                stage_31_v73: pl.Scalar[pl.INDEX] = pl.cast(stage_31_v72, pl.INDEX)
                stage_31_v74: pl.Scalar[pl.INDEX] = pl.min(stage_31_v73 + 1, 128)
                stage_31_v75: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_31_v71])
                if pl.cast(stage_31_v75, pl.INDEX) <= 0:
                    stage_31_v92: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
                    stage_31_v93: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v92)
                else:
                    stage_31_v93: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v74)
                pl.tile.write(stage_31_v51, [stage_31_v52, 0], pl.cast(pl.cast(stage_31_v93, pl.INT32), pl.FP32))
            stage_31_v53: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 128], dtype=pl.FP32, value=1.0)
            stage_31_v54: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_31_v53, stage_31_v50)
            stage_31_v55: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_sub(stage_31_v54, stage_31_v51)
            stage_31_v56: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_31_v55)
            stage_31_v57: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.maximums(stage_31_v56, 0.0)
            stage_31_v58: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(stage_31_v57, 1.0)
            stage_31_v59: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_31_v58, 1.0)
            stage_31_v60: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_31_v59, 1e+20)
            stage_31_v61: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v10, 128])] = pl.tile.set_validshape(stage_31_v60, stage_31_v10, 128)
            stage_31_v62 = pl.tile.store(stage_31_v61, [stage_31_v9, 0], stage_31_v47)
            for (stage_31_v76, (stage_31_v94,)) in pl.range(stage_31_v10, init_values=(stage_31_v62,)):
                stage_31_v77: pl.Scalar[pl.INDEX] = stage_31_v9 + stage_31_v76
                stage_31_v78: pl.Scalar[pl.INDEX] = stage_31_v77 // 6
                stage_31_v79: pl.Scalar[pl.INT64] = pl.tensor.read(view_107, [stage_31_v77, 0])
                stage_31_v80: pl.Scalar[pl.INDEX] = pl.cast(stage_31_v79, pl.INDEX)
                stage_31_v81: pl.Scalar[pl.INDEX] = pl.min(stage_31_v80 + 1, 128)
                stage_31_v82: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_31_v78])
                if pl.cast(stage_31_v82, pl.INDEX) <= 0:
                    stage_31_v95: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
                    stage_31_v96: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v95)
                else:
                    stage_31_v96: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v81)
                stage_31_v83: pl.Scalar[pl.INDEX] = stage_31_v80 - stage_31_v96 + 1
                stage_31_v84: pl.Scalar[pl.INDEX] = stage_31_v83 % 32
                stage_31_v85: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
                stage_31_v86: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
                for (stage_31_v97, (stage_31_v107, stage_31_v108)) in pl.range(5, init_values=(stage_31_v85, stage_31_v86)):
                    stage_31_v98: pl.Scalar[pl.INDEX] = pl.max(stage_31_v97 * 32 - stage_31_v84, 0)
                    stage_31_v99: pl.Scalar[pl.INDEX] = pl.min(stage_31_v97 * 32 - stage_31_v84 + 32, stage_31_v96)
                    if stage_31_v98 < stage_31_v99:
                        stage_31_v109: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_31_v78, (stage_31_v83 + stage_31_v98) // 32])
                        if 0 <= pl.cast(stage_31_v109, pl.INDEX):
                            stage_31_v116: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
                            (stage_31_v123, stage_31_v124) = pl.yield_(stage_31_v116, stage_31_v108)
                        else:
                            stage_31_v117: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
                            (stage_31_v123, stage_31_v124) = pl.yield_(stage_31_v107, stage_31_v117)
                        (stage_31_v118, stage_31_v119) = pl.yield_(stage_31_v123, stage_31_v124)
                    else:
                        (stage_31_v118, stage_31_v119) = pl.yield_(stage_31_v107, stage_31_v108)
                    (stage_31_v110, stage_31_v111) = pl.yield_(stage_31_v118, stage_31_v119)
                pl.tensor.write(temporary_54, [stage_31_v77, 0], stage_31_v110)
                if pl.cast(stage_31_v111, pl.INDEX) == 1:
                    stage_31_v100: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 128], dtype=pl.INT32, descending=False)
                    stage_31_v101: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_31_v100, target_type=pl.FP32, mode='round')
                    stage_31_v102: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
                    for (stage_31_v112, (stage_31_v120,)) in pl.range(5, init_values=(stage_31_v102,)):
                        stage_31_v113: pl.Scalar[pl.INDEX] = pl.max(stage_31_v112 * 32 - stage_31_v84, 0)
                        stage_31_v114: pl.Scalar[pl.INDEX] = pl.min(stage_31_v112 * 32 - stage_31_v84 + 32, stage_31_v96)
                        if stage_31_v113 < stage_31_v114:
                            stage_31_v121: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_31_v78, (stage_31_v83 + stage_31_v113) // 32])
                            if 0 <= pl.cast(stage_31_v121, pl.INDEX):
                                stage_31_v125: pl.Scalar[pl.FP32] = pl.cast(pl.cast(stage_31_v113, pl.INT32), pl.FP32)
                                stage_31_v126: pl.Scalar[pl.FP32] = pl.cast(pl.cast(stage_31_v114, pl.INT32), pl.FP32)
                                stage_31_v127: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_31_v101, stage_31_v125)
                                stage_31_v128: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_31_v127, 1.0)
                                stage_31_v129: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.maximums(stage_31_v128, 0.0)
                                stage_31_v130: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(stage_31_v129, 1.0)
                                stage_31_v131: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_31_v101)
                                stage_31_v132: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_31_v131, stage_31_v126)
                                stage_31_v133: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.maximums(stage_31_v132, 0.0)
                                stage_31_v134: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(stage_31_v133, 1.0)
                                stage_31_v135: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_31_v130, stage_31_v134)
                                stage_31_v136: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_31_v120, stage_31_v135)
                                stage_31_v137: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v136)
                            else:
                                stage_31_v137: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v120)
                            stage_31_v122: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v137)
                        else:
                            stage_31_v122: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v120)
                        stage_31_v115: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v122)
                    stage_31_v103: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_31_v115, 1.0)
                    stage_31_v104: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_31_v103, 1e+20)
                    stage_31_v105 = pl.tile.store(stage_31_v104, [stage_31_v77, 0], stage_31_v94)
                    stage_31_v106 = pl.yield_(stage_31_v105)
                else:
                    stage_31_v106 = pl.yield_(stage_31_v94)
                stage_31_v87 = pl.yield_(stage_31_v106)
            (stage_31_v88, stage_31_v89) = pl.yield_(stage_31_v43, stage_31_v87)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_31 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_31)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_32 = 29808 + core_id * 6
    mapped_block_32 = pl.cast(pl.tensor.read(records, [record_32]), pl.INDEX)
    block_32 = pl.max(mapped_block_32, 0)
    if mapped_block_32 >= 0:
        subslot_32 = pl.cast(pl.tensor.read(records, [record_32 + 1]), pl.INDEX)
        done_32 = pl.tensor.read(records, [record_32 + 2])
        wait_base_32 = pl.cast(pl.tensor.read(records, [record_32 + 3]), pl.INDEX)
        wait_count_32 = pl.cast(pl.tensor.read(records, [record_32 + 4]), pl.INDEX)
        publish_32 = pl.tensor.read(records, [record_32 + 5])
        for dependency_32 in pl.range(wait_count_32):
            wait_index_32 = (wait_base_32 + dependency_32) * 2
            other_32 = pl.cast(pl.tensor.read(waits, [wait_index_32]), pl.INDEX)
            required_32 = pl.tensor.read(waits, [wait_index_32 + 1])
            target_32 = generation_base + required_32
            flag_32 = other_32 * 64
            pl.system.cacheinvalid(progress, [1], [flag_32])
            initial_32 = pl.tensor.read(progress, [flag_32])
            for (ready_32,) in pl.while_(init_values=(initial_32,)):
                pl.cond(ready_32 < target_32)
                pl.system.cacheinvalid(progress, [1], [flag_32])
                after_32 = pl.yield_(pl.tensor.read(progress, [flag_32]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_32_v22: pl.Scalar[pl.INDEX] = block_32
        pl.system.set_ffts(temporary_62)
        stage_32_v23: pl.Scalar[pl.INDEX] = (144 - stage_32_v22 + 23) // 24
        stage_32_v24: pl.Scalar[pl.INDEX] = stage_32_v23 * 5
        stage_32_v25: pl.Scalar[pl.INDEX] = subslot_32
        pl.system.set_ffts(temporary_62)
        stage_32_v26: pl.Scalar[pl.INDEX] = stage_32_v25 * 32
        stage_32_v27: pl.Scalar[pl.INDEX] = stage_32_v25 * 64
        stage_32_v28: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_32_v29: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        stage_32_v30: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.rems(stage_32_v28, pl.const(2, pl.INT32), stage_32_v29)
        stage_32_v31: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_32_v30, pl.const(2, pl.INT32))
        stage_32_v32: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.sub(stage_32_v28, stage_32_v31)
        stage_32_v33: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_32_v32, pl.const(449, pl.INT32))
        stage_32_v34: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        stage_32_v35: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.col_expand(stage_32_v34, stage_32_v33)
        stage_32_v36: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 16], dtype=pl.INT32, descending=False)
        stage_32_v37: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_32_v36, pl.const(512, pl.INT32))
        stage_32_v38: pl.Tile[[16, 1], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v37, [16, 1])
        stage_32_v39: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.row_expand_add(stage_32_v35, stage_32_v38)
        stage_32_v40: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v39, [1, 1024])
        stage_32_v41: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_117, [stage_32_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
        stage_32_v42: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v41, [1, 32])
        stage_32_v43: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_32_v42, 0.0)
        stage_32_v44: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v43, [32, 1])
        stage_32_v45: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        stage_32_v46: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        for (stage_32_v47, (stage_32_v51, stage_32_v52, stage_32_v53, stage_32_v54)) in pl.range(stage_32_v24 + 3, init_values=(stage_32_v41, stage_32_v44, stage_32_v45, stage_32_v46)):
            stage_32_v48: pl.Scalar[pl.INDEX] = stage_32_v22 + stage_32_v47 // 5 * 24
            stage_32_v49: pl.Scalar[pl.INDEX] = stage_32_v48 // 6
            stage_32_v50: pl.Scalar[pl.INDEX] = stage_32_v47 % 5
            if stage_32_v47 < stage_32_v24:
                stage_32_v55: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_32_v48, stage_32_v50])
                if 0 < pl.cast(stage_32_v55, pl.INDEX):
                    stage_32_v72: pl.Scalar[pl.INDEX] = stage_32_v22 * 3 + stage_32_v47 % 3
                    stage_32_v73: pl.Scalar[pl.INDEX] = stage_32_v72 * 128
                    stage_32_v74: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
                    if stage_32_v50 == 0:
                        stage_32_v256: pl.Scalar[pl.INT64] = pl.tensor.read(view_107, [stage_32_v48, 0])
                        stage_32_v257: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v256, pl.INDEX)
                        stage_32_v258: pl.Scalar[pl.INDEX] = pl.min(stage_32_v257 + 1, 128)
                        stage_32_v259: pl.Scalar[pl.INDEX] = stage_32_v257 - stage_32_v258 + 1
                        stage_32_v260: pl.Scalar[pl.INDEX] = stage_32_v259 % 32
                        stage_32_v261: pl.Scalar[pl.INDEX] = pl.min(pl.max(stage_32_v258 - stage_32_v27, 0), 64)
                        stage_32_v262: pl.Scalar[pl.INDEX] = 0
                        stage_32_v263: pl.Scalar[pl.INDEX] = pl.min(32 - stage_32_v260 - stage_32_v27, stage_32_v261)
                        if stage_32_v262 < stage_32_v263:
                            stage_32_v273: pl.Scalar[pl.INDEX] = stage_32_v259 + stage_32_v27 + stage_32_v262
                            stage_32_v274: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_32_v49, stage_32_v273 // 32])
                            if 0 <= pl.cast(stage_32_v274, pl.INDEX):
                                stage_32_v291: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v274, pl.INDEX) * 32 + stage_32_v273 % 32
                                stage_32_v292: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_32_v74, view_45, [stage_32_v262, 0], [stage_32_v291, 0], [64, 512], valid_shape=[stage_32_v263 - stage_32_v262, 512], transpose=False)
                                stage_32_v293: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v292
                            else:
                                stage_32_v293: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v74
                            stage_32_v275: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v293
                        else:
                            stage_32_v275: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v74
                        stage_32_v264: pl.Scalar[pl.INDEX] = pl.max(32 - stage_32_v260 - stage_32_v27, 0)
                        stage_32_v265: pl.Scalar[pl.INDEX] = pl.min(64 - stage_32_v260 - stage_32_v27, stage_32_v261)
                        if stage_32_v264 < stage_32_v265:
                            stage_32_v276: pl.Scalar[pl.INDEX] = stage_32_v259 + stage_32_v27 + stage_32_v264
                            stage_32_v277: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_32_v49, stage_32_v276 // 32])
                            if 0 <= pl.cast(stage_32_v277, pl.INDEX):
                                stage_32_v294: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v277, pl.INDEX) * 32 + stage_32_v276 % 32
                                stage_32_v295: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_32_v275, view_45, [stage_32_v264, 0], [stage_32_v294, 0], [64, 512], valid_shape=[stage_32_v265 - stage_32_v264, 512], transpose=False)
                                stage_32_v296: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v295
                            else:
                                stage_32_v296: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v275
                            stage_32_v278: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v296
                        else:
                            stage_32_v278: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v275
                        stage_32_v266: pl.Scalar[pl.INDEX] = pl.max(64 - stage_32_v260 - stage_32_v27, 0)
                        stage_32_v267: pl.Scalar[pl.INDEX] = pl.min(96 - stage_32_v260 - stage_32_v27, stage_32_v261)
                        if stage_32_v266 < stage_32_v267:
                            stage_32_v279: pl.Scalar[pl.INDEX] = stage_32_v259 + stage_32_v27 + stage_32_v266
                            stage_32_v280: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_32_v49, stage_32_v279 // 32])
                            if 0 <= pl.cast(stage_32_v280, pl.INDEX):
                                stage_32_v297: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v280, pl.INDEX) * 32 + stage_32_v279 % 32
                                stage_32_v298: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_32_v278, view_45, [stage_32_v266, 0], [stage_32_v297, 0], [64, 512], valid_shape=[stage_32_v267 - stage_32_v266, 512], transpose=False)
                                stage_32_v299: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v298
                            else:
                                stage_32_v299: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v278
                            stage_32_v281: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v299
                        else:
                            stage_32_v281: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v278
                        stage_32_v268: pl.Scalar[pl.INDEX] = pl.max(96 - stage_32_v260 - stage_32_v27, 0)
                        stage_32_v269: pl.Scalar[pl.INDEX] = pl.min(128 - stage_32_v260 - stage_32_v27, stage_32_v261)
                        if stage_32_v268 < stage_32_v269:
                            stage_32_v282: pl.Scalar[pl.INDEX] = stage_32_v259 + stage_32_v27 + stage_32_v268
                            stage_32_v283: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_32_v49, stage_32_v282 // 32])
                            if 0 <= pl.cast(stage_32_v283, pl.INDEX):
                                stage_32_v300: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v283, pl.INDEX) * 32 + stage_32_v282 % 32
                                stage_32_v301: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_32_v281, view_45, [stage_32_v268, 0], [stage_32_v300, 0], [64, 512], valid_shape=[stage_32_v269 - stage_32_v268, 512], transpose=False)
                                stage_32_v302: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v301
                            else:
                                stage_32_v302: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v281
                            stage_32_v284: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v302
                        else:
                            stage_32_v284: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v281
                        stage_32_v270: pl.Scalar[pl.INDEX] = pl.max(128 - stage_32_v260 - stage_32_v27, 0)
                        stage_32_v271: pl.Scalar[pl.INDEX] = pl.min(160 - stage_32_v260 - stage_32_v27, stage_32_v261)
                        if stage_32_v270 < stage_32_v271:
                            stage_32_v285: pl.Scalar[pl.INDEX] = stage_32_v259 + stage_32_v27 + stage_32_v270
                            stage_32_v286: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_32_v49, stage_32_v285 // 32])
                            if 0 <= pl.cast(stage_32_v286, pl.INDEX):
                                stage_32_v303: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v286, pl.INDEX) * 32 + stage_32_v285 % 32
                                stage_32_v304: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_32_v284, view_45, [stage_32_v270, 0], [stage_32_v303, 0], [64, 512], valid_shape=[stage_32_v271 - stage_32_v270, 512], transpose=False)
                                stage_32_v305: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v304
                            else:
                                stage_32_v305: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v284
                            stage_32_v287: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v305
                        else:
                            stage_32_v287: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v284
                        stage_32_v272: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v287
                    else:
                        stage_32_v74: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
                        for (stage_32_v288, (stage_32_v306,)) in pl.range(64, init_values=(stage_32_v74,)):
                            stage_32_v289: pl.Scalar[pl.INDEX] = (stage_32_v50 - 1) * 128 + stage_32_v27 + stage_32_v288
                            if stage_32_v289 < 512:
                                stage_32_v307: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_53, [stage_32_v48, stage_32_v289])
                                if 0 <= pl.cast(stage_32_v307, pl.INDEX):
                                    stage_32_v309: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_block_table, [stage_32_v49, pl.cast(stage_32_v307, pl.INDEX) // 32])
                                    stage_32_v310: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v309, pl.INDEX)
                                    stage_32_v311: pl.Scalar[pl.INDEX] = stage_32_v310 * 32 + pl.cast(stage_32_v307, pl.INDEX) % 32
                                    stage_32_v312: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_32_v306, view_61, [stage_32_v288, 0], [stage_32_v311, 0], [1, 512], transpose=False)
                                    stage_32_v313: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v312
                                else:
                                    stage_32_v313: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v306
                                stage_32_v308: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v313
                            else:
                                stage_32_v308: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v306
                            stage_32_v290: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.yield_(stage_32_v308)
                        stage_32_v272: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_32_v290
                    stage_32_v75 = pl.tile.store(stage_32_v272, [stage_32_v73 + stage_32_v27, 0], temporary_56)
            if 144 <= 144:
                if 0 < stage_32_v47 and stage_32_v47 <= stage_32_v24:
                    stage_32_v76: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_32_v22 + (stage_32_v47 - 1) // 5 * 24, (stage_32_v47 - 1) % 5])
                    if 0 < pl.cast(stage_32_v76, pl.INDEX):
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                if stage_32_v47 < stage_32_v24:
                    stage_32_v77: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_32_v48, stage_32_v50])
                    if 0 < pl.cast(stage_32_v77, pl.INDEX):
                        pl.system.sync_set(0, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if 0 < stage_32_v47 and stage_32_v47 <= stage_32_v24:
                stage_32_v56: pl.Scalar[pl.INDEX] = stage_32_v47 - 1
                stage_32_v57: pl.Scalar[pl.INDEX] = stage_32_v56 % 5
                stage_32_v58: pl.Scalar[pl.INDEX] = stage_32_v22 + stage_32_v56 // 5 * 24
                stage_32_v59: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_32_v58, stage_32_v57])
                if 0 < pl.cast(stage_32_v59, pl.INDEX):
                    stage_32_v78: pl.Scalar[pl.INDEX] = stage_32_v22 * 3 + stage_32_v56 % 3
                    stage_32_v79: pl.Scalar[pl.INDEX] = stage_32_v78 * 64
                    stage_32_v80: pl.Scalar[pl.INDEX] = stage_32_v57 * 128
                    if 144 < 144:
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    stage_32_v81: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_32_v82: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_57, [stage_32_v79 + stage_32_v26, 0], [32, 128], [32, 128], target_memory=pl.Mem.Vec)
                    stage_32_v83: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_52, [stage_32_v58, stage_32_v80], [1, 128], [1, 128], target_memory=pl.Mem.Vec)
                    stage_32_v84: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_32_v82, 0.04419417382415922)
                    stage_32_v85: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_add(stage_32_v84, stage_32_v83)
                    stage_32_v86: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_32_v85, stage_32_v81)
                    stage_32_v87: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_sub(stage_32_v85, stage_32_v86)
                    stage_32_v88: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_32_v87)
                    stage_32_v89: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_32_v88, stage_32_v81)
                    stage_32_v90: pl.Tile[[32, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_32_v88, target_type=pl.BF16, mode='rint')
                    stage_32_v91 = pl.tile.store(stage_32_v90, [stage_32_v79 + stage_32_v26, 0], temporary_58)
                    stage_32_v92 = pl.tile.store(stage_32_v86, [stage_32_v79 + stage_32_v26, 0], temporary_60)
                    stage_32_v93 = pl.tile.store(stage_32_v89, [stage_32_v79 + stage_32_v26, 0], temporary_61)
                    pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if 144 < 144:
                if stage_32_v47 < stage_32_v24:
                    stage_32_v94: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_32_v48, stage_32_v50])
                    if 0 < pl.cast(stage_32_v94, pl.INDEX):
                        pl.system.sync_set(0, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if 3 <= stage_32_v47:
                stage_32_v60: pl.Scalar[pl.INDEX] = stage_32_v47 - 2 - 1
                stage_32_v61: pl.Scalar[pl.INDEX] = stage_32_v60 % 5
                stage_32_v62: pl.Scalar[pl.INDEX] = stage_32_v22 + stage_32_v60 // 5 * 24
                stage_32_v63: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_32_v62, stage_32_v61])
                if 0 < pl.cast(stage_32_v63, pl.INDEX):
                    stage_32_v95: pl.Scalar[pl.INDEX] = stage_32_v22 * 3 + stage_32_v60 % 3
                    stage_32_v96: pl.Scalar[pl.INDEX] = stage_32_v95 * 64
                    pl.system.sync_wait(3, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    stage_32_v97: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_60, [stage_32_v96 + stage_32_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
                    stage_32_v98: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_61, [stage_32_v96 + stage_32_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
                    stage_32_v99: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v51, [1, 32])
                    stage_32_v100: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v97, [1, 32])
                    stage_32_v101: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_32_v99, stage_32_v100)
                    stage_32_v102: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v101, [32, 1])
                    stage_32_v103: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v51, [1, 32])
                    stage_32_v104: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v102, [1, 32])
                    stage_32_v105: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_32_v103, stage_32_v104)
                    stage_32_v106: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v105, [32, 1])
                    stage_32_v107: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v106, [1, 32])
                    stage_32_v108: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_32_v107)
                    stage_32_v109: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v108, [32, 1])
                    stage_32_v110: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v97, [1, 32])
                    stage_32_v111: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v102, [1, 32])
                    stage_32_v112: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_32_v110, stage_32_v111)
                    stage_32_v113: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v112, [32, 1])
                    stage_32_v114: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v113, [1, 32])
                    stage_32_v115: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_32_v114)
                    stage_32_v116: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v115, [32, 1])
                    stage_32_v117: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v109, [1, 32])
                    stage_32_v118: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v52, [1, 32])
                    stage_32_v119: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_32_v117, stage_32_v118)
                    stage_32_v120: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v119, [32, 1])
                    stage_32_v121: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v116, [1, 32])
                    stage_32_v122: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v98, [1, 32])
                    stage_32_v123: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_32_v121, stage_32_v122)
                    stage_32_v124: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v123, [32, 1])
                    stage_32_v125: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v120, [1, 32])
                    stage_32_v126: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v124, [1, 32])
                    stage_32_v127: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_32_v125, stage_32_v126)
                    stage_32_v128: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v127, [32, 1])
                    stage_32_v129: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_59, [stage_32_v96 + stage_32_v26, 0], [32, 256], [32, 256], target_memory=pl.Mem.Vec)
                    stage_32_v130: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_32_v53, stage_32_v109)
                    stage_32_v131: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_32_v129, stage_32_v116)
                    stage_32_v132: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_32_v130, stage_32_v131)
                    stage_32_v133: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_59, [stage_32_v96 + stage_32_v26, 256], [32, 256], [32, 256], target_memory=pl.Mem.Vec)
                    stage_32_v134: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_32_v54, stage_32_v109)
                    stage_32_v135: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_32_v133, stage_32_v116)
                    stage_32_v136: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_32_v134, stage_32_v135)
                    stage_32_v137 = stage_32_v102
                    stage_32_v138 = stage_32_v128
                    stage_32_v139 = stage_32_v132
                    stage_32_v140 = stage_32_v136
                else:
                    stage_32_v137 = stage_32_v51
                    stage_32_v138 = stage_32_v52
                    stage_32_v139 = stage_32_v53
                    stage_32_v140 = stage_32_v54
                if stage_32_v61 == 4:
                    stage_32_v141: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 1024], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    stage_32_v142: pl.Scalar[pl.INDEX] = 0
                    stage_32_v143: pl.Scalar[pl.INDEX] = stage_32_v26 + stage_32_v142
                    stage_32_v144: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v137, [1, 32])
                    stage_32_v145: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_32_v144, 0, stage_32_v142, [1, 16], target_memory=pl.Mem.Vec)
                    stage_32_v146: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v145, [16, 1])
                    stage_32_v147: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v138, [1, 32])
                    stage_32_v148: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_32_v147, 0, stage_32_v142, [1, 16], target_memory=pl.Mem.Vec)
                    stage_32_v149: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v148, [16, 1])
                    stage_32_v150: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_32_v139, [16, 256], [stage_32_v142, 0])
                    stage_32_v151: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_32_v140, [16, 256], [stage_32_v142, 0])
                    stage_32_v152: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.concat(stage_32_v150, stage_32_v151)
                    stage_32_v153: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_117, [stage_32_v143, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    stage_32_v154: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v146, [1, 16])
                    stage_32_v155: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v146, [1, 16])
                    stage_32_v156: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_32_v154, stage_32_v155)
                    stage_32_v157: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v156, [16, 1])
                    stage_32_v158: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v157, [1, 16])
                    stage_32_v159: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v153, [1, 16])
                    stage_32_v160: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_32_v158, stage_32_v159)
                    stage_32_v161: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v160, [16, 1])
                    stage_32_v162: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v161, [1, 16])
                    stage_32_v163: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v146, [1, 16])
                    stage_32_v164: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_32_v162, stage_32_v163)
                    stage_32_v165: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v164, [16, 1])
                    stage_32_v166: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v165, [1, 16])
                    stage_32_v167: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_32_v166)
                    stage_32_v168: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v167, [16, 1])
                    stage_32_v169: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v149, [1, 16])
                    stage_32_v170: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v168, [1, 16])
                    stage_32_v171: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_32_v169, stage_32_v170)
                    stage_32_v172: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v171, [16, 1])
                    stage_32_v173: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_div(stage_32_v152, stage_32_v172)
                    stage_32_v174: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_32_v173, target_type=pl.BF16, mode='rint')
                    stage_32_v175: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_32_v62, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_32_v176: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_55, [stage_32_v62, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_32_v177: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v173, [1, 8192])
                    stage_32_v178: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_32_v177, stage_32_v40, stage_32_v141)
                    stage_32_v179: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v178, [16, 64])
                    stage_32_v180: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_32_v173, 0, 448, [16, 64], target_memory=pl.Mem.Vec)
                    stage_32_v181: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_32_v180, stage_32_v175)
                    stage_32_v182: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_32_v179, stage_32_v176)
                    stage_32_v183: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_32_v181, stage_32_v182)
                    stage_32_v184: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_32_v183, target_type=pl.BF16, mode='rint')
                    stage_32_v185: pl.Tile[[16, 448], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_32_v174, [16, 448], [0, 0])
                    stage_32_v186: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.concat(stage_32_v185, stage_32_v184)
                    stage_32_v187: pl.Tile[[2, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.reshape(stage_32_v186, [2, 4096])
                    stage_32_v188: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_32_v187, [1, 4096], [0, 0])
                    stage_32_v189: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_32_v187, [1, 4096], [1, 0])
                    stage_32_v190: pl.Scalar[pl.INDEX] = stage_32_v143 // 8 * 384 + stage_32_v62
                    stage_32_v191: pl.Scalar[pl.INDEX] = stage_32_v190 + 384
                    stage_32_v192 = pl.tile.store(stage_32_v188, [stage_32_v190, 0], temporary_51)
                    stage_32_v193 = pl.tile.store(stage_32_v189, [stage_32_v191, 0], temporary_51)
                    stage_32_v194: pl.Scalar[pl.INDEX] = 16
                    stage_32_v195: pl.Scalar[pl.INDEX] = stage_32_v26 + stage_32_v194
                    stage_32_v196: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v137, [1, 32])
                    stage_32_v197: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_32_v196, 0, stage_32_v194, [1, 16], target_memory=pl.Mem.Vec)
                    stage_32_v198: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v197, [16, 1])
                    stage_32_v199: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v138, [1, 32])
                    stage_32_v200: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_32_v199, 0, stage_32_v194, [1, 16], target_memory=pl.Mem.Vec)
                    stage_32_v201: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v200, [16, 1])
                    stage_32_v202: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_32_v139, [16, 256], [stage_32_v194, 0])
                    stage_32_v203: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_32_v140, [16, 256], [stage_32_v194, 0])
                    stage_32_v204: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.concat(stage_32_v202, stage_32_v203)
                    stage_32_v205: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_117, [stage_32_v195, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    stage_32_v206: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v198, [1, 16])
                    stage_32_v207: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v198, [1, 16])
                    stage_32_v208: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_32_v206, stage_32_v207)
                    stage_32_v209: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v208, [16, 1])
                    stage_32_v210: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v209, [1, 16])
                    stage_32_v211: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v205, [1, 16])
                    stage_32_v212: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_32_v210, stage_32_v211)
                    stage_32_v213: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v212, [16, 1])
                    stage_32_v214: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v213, [1, 16])
                    stage_32_v215: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v198, [1, 16])
                    stage_32_v216: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_32_v214, stage_32_v215)
                    stage_32_v217: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v216, [16, 1])
                    stage_32_v218: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v217, [1, 16])
                    stage_32_v219: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_32_v218)
                    stage_32_v220: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v219, [16, 1])
                    stage_32_v221: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v201, [1, 16])
                    stage_32_v222: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v220, [1, 16])
                    stage_32_v223: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_32_v221, stage_32_v222)
                    stage_32_v224: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v223, [16, 1])
                    stage_32_v225: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_div(stage_32_v204, stage_32_v224)
                    stage_32_v226: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_32_v225, target_type=pl.BF16, mode='rint')
                    stage_32_v227: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_32_v62, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_32_v228: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_55, [stage_32_v62, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_32_v229: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v225, [1, 8192])
                    stage_32_v230: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_32_v229, stage_32_v40, stage_32_v141)
                    stage_32_v231: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v230, [16, 64])
                    stage_32_v232: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_32_v225, 0, 448, [16, 64], target_memory=pl.Mem.Vec)
                    stage_32_v233: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_32_v232, stage_32_v227)
                    stage_32_v234: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_32_v231, stage_32_v228)
                    stage_32_v235: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_32_v233, stage_32_v234)
                    stage_32_v236: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_32_v235, target_type=pl.BF16, mode='rint')
                    stage_32_v237: pl.Tile[[16, 448], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_32_v226, [16, 448], [0, 0])
                    stage_32_v238: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.concat(stage_32_v237, stage_32_v236)
                    stage_32_v239: pl.Tile[[2, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.reshape(stage_32_v238, [2, 4096])
                    stage_32_v240: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_32_v239, [1, 4096], [0, 0])
                    stage_32_v241: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_32_v239, [1, 4096], [1, 0])
                    stage_32_v242: pl.Scalar[pl.INDEX] = stage_32_v195 // 8 * 384 + stage_32_v62
                    stage_32_v243: pl.Scalar[pl.INDEX] = stage_32_v242 + 384
                    stage_32_v244 = pl.tile.store(stage_32_v240, [stage_32_v242, 0], temporary_51)
                    stage_32_v245 = pl.tile.store(stage_32_v241, [stage_32_v243, 0], temporary_51)
                    stage_32_v246: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_117, [stage_32_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
                    stage_32_v247: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v246, [1, 32])
                    stage_32_v248: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_32_v247, 0.0)
                    stage_32_v249: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_32_v248, [32, 1])
                    stage_32_v250: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
                    stage_32_v251: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
                    stage_32_v252 = stage_32_v246
                    stage_32_v253 = stage_32_v249
                    stage_32_v254 = stage_32_v250
                    stage_32_v255 = stage_32_v251
                else:
                    stage_32_v252 = stage_32_v137
                    stage_32_v253 = stage_32_v138
                    stage_32_v254 = stage_32_v139
                    stage_32_v255 = stage_32_v140
                stage_32_v64 = stage_32_v252
                stage_32_v65 = stage_32_v253
                stage_32_v66 = stage_32_v254
                stage_32_v67 = stage_32_v255
            else:
                stage_32_v64 = stage_32_v51
                stage_32_v65 = stage_32_v52
                stage_32_v66 = stage_32_v53
                stage_32_v67 = stage_32_v54
            (stage_32_v68, stage_32_v69, stage_32_v70, stage_32_v71) = pl.yield_(stage_32_v64, stage_32_v65, stage_32_v66, stage_32_v67)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_32 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_32)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_33 = 30240 + core_id * 6
    mapped_block_33 = pl.cast(pl.tensor.read(records, [record_33]), pl.INDEX)
    block_33 = pl.max(mapped_block_33, 0)
    if mapped_block_33 >= 0:
        subslot_33 = pl.cast(pl.tensor.read(records, [record_33 + 1]), pl.INDEX)
        done_33 = pl.tensor.read(records, [record_33 + 2])
        wait_base_33 = pl.cast(pl.tensor.read(records, [record_33 + 3]), pl.INDEX)
        wait_count_33 = pl.cast(pl.tensor.read(records, [record_33 + 4]), pl.INDEX)
        publish_33 = pl.tensor.read(records, [record_33 + 5])
        for dependency_33 in pl.range(wait_count_33):
            wait_index_33 = (wait_base_33 + dependency_33) * 2
            other_33 = pl.cast(pl.tensor.read(waits, [wait_index_33]), pl.INDEX)
            required_33 = pl.tensor.read(waits, [wait_index_33 + 1])
            target_33 = generation_base + required_33
            flag_33 = other_33 * 64
            pl.system.cacheinvalid(progress, [1], [flag_33])
            initial_33 = pl.tensor.read(progress, [flag_33])
            for (ready_33,) in pl.while_(init_values=(initial_33,)):
                pl.cond(ready_33 < target_33)
                pl.system.cacheinvalid(progress, [1], [flag_33])
                after_33 = pl.yield_(pl.tensor.read(progress, [flag_33]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_33_v4: pl.Scalar[pl.INDEX] = block_33
        stage_33_v5: pl.Scalar[pl.INDEX] = stage_33_v4 * 32
        for (stage_33_v6, (stage_33_v26, stage_33_v27)) in pl.pipeline(stage_33_v5, pl.min(stage_33_v5 + 32, 144), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_33_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_33_v6, 2048], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_33_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_33_v7)
            stage_33_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_33_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_33_v8, stage_33_v9)
            stage_33_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_33_v10, [1, 8])
            stage_33_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_33_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_33_v12, stage_33_v11)
            stage_33_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_33_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_33_v14, stage_33_v13)
            stage_33_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_33_v13, 0.007874015748031496)
            stage_33_v17 = pl.tile.store(stage_33_v16, [2, stage_33_v6], stage_33_v26)
            stage_33_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_33_v15, [8, 1])
            stage_33_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_33_v6, 2048], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_33_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_33_v19, stage_33_v18)
            stage_33_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_33_v20, target_type=pl.INT32, mode='rint')
            stage_33_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_33_v21, target_type=pl.FP16, mode='round')
            stage_33_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_33_v22, target_type=pl.INT8, mode='trunc')
            stage_33_v24 = pl.tile.store(stage_33_v23, [stage_33_v6, 2048], stage_33_v27)
            (stage_33_v28, stage_33_v29) = pl.yield_(stage_33_v17, stage_33_v24)
        if 144 <= stage_33_v5 + 32:
            for (stage_33_v30, (stage_33_v37,)) in pl.range(144, 256, 8, init_values=(stage_33_v29,)):
                stage_33_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_33_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_33_v31, target_type=pl.INT8, mode='trunc')
                stage_33_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_33_v30, 8)
                stage_33_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v33, 1024])] = pl.tile.set_validshape(stage_33_v32, stage_33_v33, 1024)
                stage_33_v35 = pl.tile.store(stage_33_v34, [stage_33_v30, 2048], stage_33_v37)
                stage_33_v36 = pl.yield_(stage_33_v35)
            stage_33_v25 = pl.yield_(stage_33_v36)
        else:
            stage_33_v25 = pl.yield_(stage_33_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_33 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_33)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_34 = 30672 + core_id * 6
    mapped_block_34 = pl.cast(pl.tensor.read(records, [record_34]), pl.INDEX)
    block_34 = pl.max(mapped_block_34, 0)
    if mapped_block_34 >= 0:
        subslot_34 = pl.cast(pl.tensor.read(records, [record_34 + 1]), pl.INDEX)
        done_34 = pl.tensor.read(records, [record_34 + 2])
        wait_base_34 = pl.cast(pl.tensor.read(records, [record_34 + 3]), pl.INDEX)
        wait_count_34 = pl.cast(pl.tensor.read(records, [record_34 + 4]), pl.INDEX)
        publish_34 = pl.tensor.read(records, [record_34 + 5])
        for dependency_34 in pl.range(wait_count_34):
            wait_index_34 = (wait_base_34 + dependency_34) * 2
            other_34 = pl.cast(pl.tensor.read(waits, [wait_index_34]), pl.INDEX)
            required_34 = pl.tensor.read(waits, [wait_index_34 + 1])
            target_34 = generation_base + required_34
            flag_34 = other_34 * 64
            pl.system.cacheinvalid(progress, [1], [flag_34])
            initial_34 = pl.tensor.read(progress, [flag_34])
            for (ready_34,) in pl.while_(init_values=(initial_34,)):
                pl.cond(ready_34 < target_34)
                pl.system.cacheinvalid(progress, [1], [flag_34])
                after_34 = pl.yield_(pl.tensor.read(progress, [flag_34]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_34_v4: pl.Scalar[pl.INDEX] = block_34
        stage_34_v5: pl.Scalar[pl.INDEX] = stage_34_v4 * 32
        for (stage_34_v6, (stage_34_v26, stage_34_v27)) in pl.pipeline(stage_34_v5, pl.min(stage_34_v5 + 32, 144), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_34_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_34_v6, 7168], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_34_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_34_v7)
            stage_34_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_34_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_34_v8, stage_34_v9)
            stage_34_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v10, [1, 8])
            stage_34_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_34_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_34_v12, stage_34_v11)
            stage_34_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_34_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_34_v14, stage_34_v13)
            stage_34_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_34_v13, 0.007874015748031496)
            stage_34_v17 = pl.tile.store(stage_34_v16, [7, stage_34_v6], stage_34_v26)
            stage_34_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v15, [8, 1])
            stage_34_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_34_v6, 7168], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_34_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_34_v19, stage_34_v18)
            stage_34_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_34_v20, target_type=pl.INT32, mode='rint')
            stage_34_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_34_v21, target_type=pl.FP16, mode='round')
            stage_34_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_34_v22, target_type=pl.INT8, mode='trunc')
            stage_34_v24 = pl.tile.store(stage_34_v23, [stage_34_v6, 7168], stage_34_v27)
            (stage_34_v28, stage_34_v29) = pl.yield_(stage_34_v17, stage_34_v24)
        if 144 <= stage_34_v5 + 32:
            for (stage_34_v30, (stage_34_v37,)) in pl.range(144, 256, 8, init_values=(stage_34_v29,)):
                stage_34_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_34_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_34_v31, target_type=pl.INT8, mode='trunc')
                stage_34_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_34_v30, 8)
                stage_34_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_34_v33, 1024])] = pl.tile.set_validshape(stage_34_v32, stage_34_v33, 1024)
                stage_34_v35 = pl.tile.store(stage_34_v34, [stage_34_v30, 7168], stage_34_v37)
                stage_34_v36 = pl.yield_(stage_34_v35)
            stage_34_v25 = pl.yield_(stage_34_v36)
        else:
            stage_34_v25 = pl.yield_(stage_34_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_34 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_34)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_35 = 31104 + core_id * 6
    mapped_block_35 = pl.cast(pl.tensor.read(records, [record_35]), pl.INDEX)
    block_35 = pl.max(mapped_block_35, 0)
    if mapped_block_35 >= 0:
        subslot_35 = pl.cast(pl.tensor.read(records, [record_35 + 1]), pl.INDEX)
        done_35 = pl.tensor.read(records, [record_35 + 2])
        wait_base_35 = pl.cast(pl.tensor.read(records, [record_35 + 3]), pl.INDEX)
        wait_count_35 = pl.cast(pl.tensor.read(records, [record_35 + 4]), pl.INDEX)
        publish_35 = pl.tensor.read(records, [record_35 + 5])
        for dependency_35 in pl.range(wait_count_35):
            wait_index_35 = (wait_base_35 + dependency_35) * 2
            other_35 = pl.cast(pl.tensor.read(waits, [wait_index_35]), pl.INDEX)
            required_35 = pl.tensor.read(waits, [wait_index_35 + 1])
            target_35 = generation_base + required_35
            flag_35 = other_35 * 64
            pl.system.cacheinvalid(progress, [1], [flag_35])
            initial_35 = pl.tensor.read(progress, [flag_35])
            for (ready_35,) in pl.while_(init_values=(initial_35,)):
                pl.cond(ready_35 < target_35)
                pl.system.cacheinvalid(progress, [1], [flag_35])
                after_35 = pl.yield_(pl.tensor.read(progress, [flag_35]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_35_v4: pl.Scalar[pl.INDEX] = block_35
        stage_35_v5: pl.Scalar[pl.INDEX] = stage_35_v4 * 32
        for (stage_35_v6, (stage_35_v26, stage_35_v27)) in pl.pipeline(stage_35_v5, pl.min(stage_35_v5 + 32, 144), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_35_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_35_v6, 6144], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_35_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_35_v7)
            stage_35_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_35_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_35_v8, stage_35_v9)
            stage_35_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_35_v10, [1, 8])
            stage_35_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_35_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_35_v12, stage_35_v11)
            stage_35_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_35_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_35_v14, stage_35_v13)
            stage_35_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_35_v13, 0.007874015748031496)
            stage_35_v17 = pl.tile.store(stage_35_v16, [6, stage_35_v6], stage_35_v26)
            stage_35_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_35_v15, [8, 1])
            stage_35_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_35_v6, 6144], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_35_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_35_v19, stage_35_v18)
            stage_35_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_35_v20, target_type=pl.INT32, mode='rint')
            stage_35_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_35_v21, target_type=pl.FP16, mode='round')
            stage_35_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_35_v22, target_type=pl.INT8, mode='trunc')
            stage_35_v24 = pl.tile.store(stage_35_v23, [stage_35_v6, 6144], stage_35_v27)
            (stage_35_v28, stage_35_v29) = pl.yield_(stage_35_v17, stage_35_v24)
        if 144 <= stage_35_v5 + 32:
            for (stage_35_v30, (stage_35_v37,)) in pl.range(144, 256, 8, init_values=(stage_35_v29,)):
                stage_35_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_35_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_35_v31, target_type=pl.INT8, mode='trunc')
                stage_35_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_35_v30, 8)
                stage_35_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_35_v33, 1024])] = pl.tile.set_validshape(stage_35_v32, stage_35_v33, 1024)
                stage_35_v35 = pl.tile.store(stage_35_v34, [stage_35_v30, 6144], stage_35_v37)
                stage_35_v36 = pl.yield_(stage_35_v35)
            stage_35_v25 = pl.yield_(stage_35_v36)
        else:
            stage_35_v25 = pl.yield_(stage_35_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_35 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_35)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_36 = 31536 + core_id * 6
    mapped_block_36 = pl.cast(pl.tensor.read(records, [record_36]), pl.INDEX)
    block_36 = pl.max(mapped_block_36, 0)
    if mapped_block_36 >= 0:
        subslot_36 = pl.cast(pl.tensor.read(records, [record_36 + 1]), pl.INDEX)
        done_36 = pl.tensor.read(records, [record_36 + 2])
        wait_base_36 = pl.cast(pl.tensor.read(records, [record_36 + 3]), pl.INDEX)
        wait_count_36 = pl.cast(pl.tensor.read(records, [record_36 + 4]), pl.INDEX)
        publish_36 = pl.tensor.read(records, [record_36 + 5])
        for dependency_36 in pl.range(wait_count_36):
            wait_index_36 = (wait_base_36 + dependency_36) * 2
            other_36 = pl.cast(pl.tensor.read(waits, [wait_index_36]), pl.INDEX)
            required_36 = pl.tensor.read(waits, [wait_index_36 + 1])
            target_36 = generation_base + required_36
            flag_36 = other_36 * 64
            pl.system.cacheinvalid(progress, [1], [flag_36])
            initial_36 = pl.tensor.read(progress, [flag_36])
            for (ready_36,) in pl.while_(init_values=(initial_36,)):
                pl.cond(ready_36 < target_36)
                pl.system.cacheinvalid(progress, [1], [flag_36])
                after_36 = pl.yield_(pl.tensor.read(progress, [flag_36]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_36_v4: pl.Scalar[pl.INDEX] = block_36
        stage_36_v5: pl.Scalar[pl.INDEX] = stage_36_v4 * 32
        for (stage_36_v6, (stage_36_v26, stage_36_v27)) in pl.pipeline(stage_36_v5, pl.min(stage_36_v5 + 32, 144), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_36_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_36_v6, 5120], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_36_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_36_v7)
            stage_36_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_36_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_36_v8, stage_36_v9)
            stage_36_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_36_v10, [1, 8])
            stage_36_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_36_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_36_v12, stage_36_v11)
            stage_36_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_36_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_36_v14, stage_36_v13)
            stage_36_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_36_v13, 0.007874015748031496)
            stage_36_v17 = pl.tile.store(stage_36_v16, [5, stage_36_v6], stage_36_v26)
            stage_36_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_36_v15, [8, 1])
            stage_36_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_36_v6, 5120], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_36_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_36_v19, stage_36_v18)
            stage_36_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_36_v20, target_type=pl.INT32, mode='rint')
            stage_36_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_36_v21, target_type=pl.FP16, mode='round')
            stage_36_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_36_v22, target_type=pl.INT8, mode='trunc')
            stage_36_v24 = pl.tile.store(stage_36_v23, [stage_36_v6, 5120], stage_36_v27)
            (stage_36_v28, stage_36_v29) = pl.yield_(stage_36_v17, stage_36_v24)
        if 144 <= stage_36_v5 + 32:
            for (stage_36_v30, (stage_36_v37,)) in pl.range(144, 256, 8, init_values=(stage_36_v29,)):
                stage_36_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_36_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_36_v31, target_type=pl.INT8, mode='trunc')
                stage_36_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_36_v30, 8)
                stage_36_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v33, 1024])] = pl.tile.set_validshape(stage_36_v32, stage_36_v33, 1024)
                stage_36_v35 = pl.tile.store(stage_36_v34, [stage_36_v30, 5120], stage_36_v37)
                stage_36_v36 = pl.yield_(stage_36_v35)
            stage_36_v25 = pl.yield_(stage_36_v36)
        else:
            stage_36_v25 = pl.yield_(stage_36_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_36 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_36)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_37 = 31968 + core_id * 6
    mapped_block_37 = pl.cast(pl.tensor.read(records, [record_37]), pl.INDEX)
    block_37 = pl.max(mapped_block_37, 0)
    if mapped_block_37 >= 0:
        subslot_37 = pl.cast(pl.tensor.read(records, [record_37 + 1]), pl.INDEX)
        done_37 = pl.tensor.read(records, [record_37 + 2])
        wait_base_37 = pl.cast(pl.tensor.read(records, [record_37 + 3]), pl.INDEX)
        wait_count_37 = pl.cast(pl.tensor.read(records, [record_37 + 4]), pl.INDEX)
        publish_37 = pl.tensor.read(records, [record_37 + 5])
        for dependency_37 in pl.range(wait_count_37):
            wait_index_37 = (wait_base_37 + dependency_37) * 2
            other_37 = pl.cast(pl.tensor.read(waits, [wait_index_37]), pl.INDEX)
            required_37 = pl.tensor.read(waits, [wait_index_37 + 1])
            target_37 = generation_base + required_37
            flag_37 = other_37 * 64
            pl.system.cacheinvalid(progress, [1], [flag_37])
            initial_37 = pl.tensor.read(progress, [flag_37])
            for (ready_37,) in pl.while_(init_values=(initial_37,)):
                pl.cond(ready_37 < target_37)
                pl.system.cacheinvalid(progress, [1], [flag_37])
                after_37 = pl.yield_(pl.tensor.read(progress, [flag_37]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_37_v4: pl.Scalar[pl.INDEX] = block_37
        stage_37_v5: pl.Scalar[pl.INDEX] = stage_37_v4 * 32
        for (stage_37_v6, (stage_37_v26, stage_37_v27)) in pl.pipeline(stage_37_v5, pl.min(stage_37_v5 + 32, 144), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_37_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_37_v6, 4096], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_37_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_37_v7)
            stage_37_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_37_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_37_v8, stage_37_v9)
            stage_37_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_37_v10, [1, 8])
            stage_37_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_37_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_37_v12, stage_37_v11)
            stage_37_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_37_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_37_v14, stage_37_v13)
            stage_37_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_37_v13, 0.007874015748031496)
            stage_37_v17 = pl.tile.store(stage_37_v16, [4, stage_37_v6], stage_37_v26)
            stage_37_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_37_v15, [8, 1])
            stage_37_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_37_v6, 4096], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_37_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_37_v19, stage_37_v18)
            stage_37_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_37_v20, target_type=pl.INT32, mode='rint')
            stage_37_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_37_v21, target_type=pl.FP16, mode='round')
            stage_37_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_37_v22, target_type=pl.INT8, mode='trunc')
            stage_37_v24 = pl.tile.store(stage_37_v23, [stage_37_v6, 4096], stage_37_v27)
            (stage_37_v28, stage_37_v29) = pl.yield_(stage_37_v17, stage_37_v24)
        if 144 <= stage_37_v5 + 32:
            for (stage_37_v30, (stage_37_v37,)) in pl.range(144, 256, 8, init_values=(stage_37_v29,)):
                stage_37_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_37_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_37_v31, target_type=pl.INT8, mode='trunc')
                stage_37_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_37_v30, 8)
                stage_37_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_37_v33, 1024])] = pl.tile.set_validshape(stage_37_v32, stage_37_v33, 1024)
                stage_37_v35 = pl.tile.store(stage_37_v34, [stage_37_v30, 4096], stage_37_v37)
                stage_37_v36 = pl.yield_(stage_37_v35)
            stage_37_v25 = pl.yield_(stage_37_v36)
        else:
            stage_37_v25 = pl.yield_(stage_37_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_37 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_37)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_38 = 32400 + core_id * 6
    mapped_block_38 = pl.cast(pl.tensor.read(records, [record_38]), pl.INDEX)
    block_38 = pl.max(mapped_block_38, 0)
    if mapped_block_38 >= 0:
        subslot_38 = pl.cast(pl.tensor.read(records, [record_38 + 1]), pl.INDEX)
        done_38 = pl.tensor.read(records, [record_38 + 2])
        wait_base_38 = pl.cast(pl.tensor.read(records, [record_38 + 3]), pl.INDEX)
        wait_count_38 = pl.cast(pl.tensor.read(records, [record_38 + 4]), pl.INDEX)
        publish_38 = pl.tensor.read(records, [record_38 + 5])
        for dependency_38 in pl.range(wait_count_38):
            wait_index_38 = (wait_base_38 + dependency_38) * 2
            other_38 = pl.cast(pl.tensor.read(waits, [wait_index_38]), pl.INDEX)
            required_38 = pl.tensor.read(waits, [wait_index_38 + 1])
            target_38 = generation_base + required_38
            flag_38 = other_38 * 64
            pl.system.cacheinvalid(progress, [1], [flag_38])
            initial_38 = pl.tensor.read(progress, [flag_38])
            for (ready_38,) in pl.while_(init_values=(initial_38,)):
                pl.cond(ready_38 < target_38)
                pl.system.cacheinvalid(progress, [1], [flag_38])
                after_38 = pl.yield_(pl.tensor.read(progress, [flag_38]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_38_v4: pl.Scalar[pl.INDEX] = block_38
        stage_38_v5: pl.Scalar[pl.INDEX] = stage_38_v4 * 32
        for (stage_38_v6, (stage_38_v26, stage_38_v27)) in pl.pipeline(stage_38_v5, pl.min(stage_38_v5 + 32, 144), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_38_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_38_v6, 0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_38_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_38_v7)
            stage_38_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_38_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_38_v8, stage_38_v9)
            stage_38_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v10, [1, 8])
            stage_38_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_38_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_38_v12, stage_38_v11)
            stage_38_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_38_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_38_v14, stage_38_v13)
            stage_38_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_38_v13, 0.007874015748031496)
            stage_38_v17 = pl.tile.store(stage_38_v16, [0, stage_38_v6], stage_38_v26)
            stage_38_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v15, [8, 1])
            stage_38_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_38_v6, 0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_38_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_38_v19, stage_38_v18)
            stage_38_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_38_v20, target_type=pl.INT32, mode='rint')
            stage_38_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_38_v21, target_type=pl.FP16, mode='round')
            stage_38_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_38_v22, target_type=pl.INT8, mode='trunc')
            stage_38_v24 = pl.tile.store(stage_38_v23, [stage_38_v6, 0], stage_38_v27)
            (stage_38_v28, stage_38_v29) = pl.yield_(stage_38_v17, stage_38_v24)
        if 144 <= stage_38_v5 + 32:
            for (stage_38_v30, (stage_38_v37,)) in pl.range(144, 256, 8, init_values=(stage_38_v29,)):
                stage_38_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_38_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_38_v31, target_type=pl.INT8, mode='trunc')
                stage_38_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_38_v30, 8)
                stage_38_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_38_v33, 1024])] = pl.tile.set_validshape(stage_38_v32, stage_38_v33, 1024)
                stage_38_v35 = pl.tile.store(stage_38_v34, [stage_38_v30, 0], stage_38_v37)
                stage_38_v36 = pl.yield_(stage_38_v35)
            stage_38_v25 = pl.yield_(stage_38_v36)
        else:
            stage_38_v25 = pl.yield_(stage_38_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_38 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_38)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_39 = 32832 + core_id * 6
    mapped_block_39 = pl.cast(pl.tensor.read(records, [record_39]), pl.INDEX)
    block_39 = pl.max(mapped_block_39, 0)
    if mapped_block_39 >= 0:
        subslot_39 = pl.cast(pl.tensor.read(records, [record_39 + 1]), pl.INDEX)
        done_39 = pl.tensor.read(records, [record_39 + 2])
        wait_base_39 = pl.cast(pl.tensor.read(records, [record_39 + 3]), pl.INDEX)
        wait_count_39 = pl.cast(pl.tensor.read(records, [record_39 + 4]), pl.INDEX)
        publish_39 = pl.tensor.read(records, [record_39 + 5])
        for dependency_39 in pl.range(wait_count_39):
            wait_index_39 = (wait_base_39 + dependency_39) * 2
            other_39 = pl.cast(pl.tensor.read(waits, [wait_index_39]), pl.INDEX)
            required_39 = pl.tensor.read(waits, [wait_index_39 + 1])
            target_39 = generation_base + required_39
            flag_39 = other_39 * 64
            pl.system.cacheinvalid(progress, [1], [flag_39])
            initial_39 = pl.tensor.read(progress, [flag_39])
            for (ready_39,) in pl.while_(init_values=(initial_39,)):
                pl.cond(ready_39 < target_39)
                pl.system.cacheinvalid(progress, [1], [flag_39])
                after_39 = pl.yield_(pl.tensor.read(progress, [flag_39]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_39_v4: pl.Scalar[pl.INDEX] = block_39
        stage_39_v5: pl.Scalar[pl.INDEX] = stage_39_v4 * 32
        for (stage_39_v6, (stage_39_v26, stage_39_v27)) in pl.pipeline(stage_39_v5, pl.min(stage_39_v5 + 32, 144), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_39_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_39_v6, 3072], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_39_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_39_v7)
            stage_39_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_39_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_39_v8, stage_39_v9)
            stage_39_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_39_v10, [1, 8])
            stage_39_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_39_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_39_v12, stage_39_v11)
            stage_39_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_39_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_39_v14, stage_39_v13)
            stage_39_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_39_v13, 0.007874015748031496)
            stage_39_v17 = pl.tile.store(stage_39_v16, [3, stage_39_v6], stage_39_v26)
            stage_39_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_39_v15, [8, 1])
            stage_39_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_39_v6, 3072], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_39_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_39_v19, stage_39_v18)
            stage_39_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_39_v20, target_type=pl.INT32, mode='rint')
            stage_39_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_39_v21, target_type=pl.FP16, mode='round')
            stage_39_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_39_v22, target_type=pl.INT8, mode='trunc')
            stage_39_v24 = pl.tile.store(stage_39_v23, [stage_39_v6, 3072], stage_39_v27)
            (stage_39_v28, stage_39_v29) = pl.yield_(stage_39_v17, stage_39_v24)
        if 144 <= stage_39_v5 + 32:
            for (stage_39_v30, (stage_39_v37,)) in pl.range(144, 256, 8, init_values=(stage_39_v29,)):
                stage_39_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_39_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_39_v31, target_type=pl.INT8, mode='trunc')
                stage_39_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_39_v30, 8)
                stage_39_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_39_v33, 1024])] = pl.tile.set_validshape(stage_39_v32, stage_39_v33, 1024)
                stage_39_v35 = pl.tile.store(stage_39_v34, [stage_39_v30, 3072], stage_39_v37)
                stage_39_v36 = pl.yield_(stage_39_v35)
            stage_39_v25 = pl.yield_(stage_39_v36)
        else:
            stage_39_v25 = pl.yield_(stage_39_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_39 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_39)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_40 = 33264 + core_id * 6
    mapped_block_40 = pl.cast(pl.tensor.read(records, [record_40]), pl.INDEX)
    block_40 = pl.max(mapped_block_40, 0)
    if mapped_block_40 >= 0:
        subslot_40 = pl.cast(pl.tensor.read(records, [record_40 + 1]), pl.INDEX)
        done_40 = pl.tensor.read(records, [record_40 + 2])
        wait_base_40 = pl.cast(pl.tensor.read(records, [record_40 + 3]), pl.INDEX)
        wait_count_40 = pl.cast(pl.tensor.read(records, [record_40 + 4]), pl.INDEX)
        publish_40 = pl.tensor.read(records, [record_40 + 5])
        for dependency_40 in pl.range(wait_count_40):
            wait_index_40 = (wait_base_40 + dependency_40) * 2
            other_40 = pl.cast(pl.tensor.read(waits, [wait_index_40]), pl.INDEX)
            required_40 = pl.tensor.read(waits, [wait_index_40 + 1])
            target_40 = generation_base + required_40
            flag_40 = other_40 * 64
            pl.system.cacheinvalid(progress, [1], [flag_40])
            initial_40 = pl.tensor.read(progress, [flag_40])
            for (ready_40,) in pl.while_(init_values=(initial_40,)):
                pl.cond(ready_40 < target_40)
                pl.system.cacheinvalid(progress, [1], [flag_40])
                after_40 = pl.yield_(pl.tensor.read(progress, [flag_40]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_40_v4: pl.Scalar[pl.INDEX] = block_40
        stage_40_v5: pl.Scalar[pl.INDEX] = stage_40_v4 * 32
        for (stage_40_v6, (stage_40_v26, stage_40_v27)) in pl.pipeline(stage_40_v5, pl.min(stage_40_v5 + 32, 144), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_40_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_40_v6, 1024], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_40_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_40_v7)
            stage_40_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_40_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_40_v8, stage_40_v9)
            stage_40_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_40_v10, [1, 8])
            stage_40_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_40_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_40_v12, stage_40_v11)
            stage_40_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_40_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_40_v14, stage_40_v13)
            stage_40_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_40_v13, 0.007874015748031496)
            stage_40_v17 = pl.tile.store(stage_40_v16, [1, stage_40_v6], stage_40_v26)
            stage_40_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_40_v15, [8, 1])
            stage_40_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_40_v6, 1024], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_40_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_40_v19, stage_40_v18)
            stage_40_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_40_v20, target_type=pl.INT32, mode='rint')
            stage_40_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_40_v21, target_type=pl.FP16, mode='round')
            stage_40_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_40_v22, target_type=pl.INT8, mode='trunc')
            stage_40_v24 = pl.tile.store(stage_40_v23, [stage_40_v6, 1024], stage_40_v27)
            (stage_40_v28, stage_40_v29) = pl.yield_(stage_40_v17, stage_40_v24)
        if 144 <= stage_40_v5 + 32:
            for (stage_40_v30, (stage_40_v37,)) in pl.range(144, 256, 8, init_values=(stage_40_v29,)):
                stage_40_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_40_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_40_v31, target_type=pl.INT8, mode='trunc')
                stage_40_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_40_v30, 8)
                stage_40_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_40_v33, 1024])] = pl.tile.set_validshape(stage_40_v32, stage_40_v33, 1024)
                stage_40_v35 = pl.tile.store(stage_40_v34, [stage_40_v30, 1024], stage_40_v37)
                stage_40_v36 = pl.yield_(stage_40_v35)
            stage_40_v25 = pl.yield_(stage_40_v36)
        else:
            stage_40_v25 = pl.yield_(stage_40_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_40 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_40)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_41 = 33696 + core_id * 6
    mapped_block_41 = pl.cast(pl.tensor.read(records, [record_41]), pl.INDEX)
    block_41 = pl.max(mapped_block_41, 0)
    if mapped_block_41 >= 0:
        subslot_41 = pl.cast(pl.tensor.read(records, [record_41 + 1]), pl.INDEX)
        done_41 = pl.tensor.read(records, [record_41 + 2])
        wait_base_41 = pl.cast(pl.tensor.read(records, [record_41 + 3]), pl.INDEX)
        wait_count_41 = pl.cast(pl.tensor.read(records, [record_41 + 4]), pl.INDEX)
        publish_41 = pl.tensor.read(records, [record_41 + 5])
        for dependency_41 in pl.range(wait_count_41):
            wait_index_41 = (wait_base_41 + dependency_41) * 2
            other_41 = pl.cast(pl.tensor.read(waits, [wait_index_41]), pl.INDEX)
            required_41 = pl.tensor.read(waits, [wait_index_41 + 1])
            target_41 = generation_base + required_41
            flag_41 = other_41 * 64
            pl.system.cacheinvalid(progress, [1], [flag_41])
            initial_41 = pl.tensor.read(progress, [flag_41])
            for (ready_41,) in pl.while_(init_values=(initial_41,)):
                pl.cond(ready_41 < target_41)
                pl.system.cacheinvalid(progress, [1], [flag_41])
                after_41 = pl.yield_(pl.tensor.read(progress, [flag_41]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_41_v8: pl.Scalar[pl.INDEX] = block_41
        stage_41_v9: pl.Scalar[pl.INDEX] = stage_41_v8
        stage_41_v10: pl.Scalar[pl.INDEX] = stage_41_v8 - stage_41_v9
        stage_41_v11: pl.Scalar[pl.INDEX] = stage_41_v10 * 4096
        stage_41_v12: pl.Scalar[pl.INDEX] = stage_41_v9 * 4
        stage_41_v13: pl.Tile[[4096], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wo_b_scale, [stage_41_v11], [4096], [4096], target_memory=pl.Mem.Vec)
        stage_41_v14: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_41_v13, [1, 4096])
        for (stage_41_v15, (stage_41_v99,)) in pl.range(stage_41_v12, pl.min(stage_41_v12 + 4, 144), init_values=(view_129,)):
            stage_41_v16: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 4096], dtype=pl.FP32, value=0.0)
            for (stage_41_v100, (stage_41_v108,)) in pl.pipeline(8, stage=2, init_values=(stage_41_v16,)):
                stage_41_v101: pl.Scalar[pl.INDEX] = stage_41_v100 * 4096 + stage_41_v11
                stage_41_v102: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_66, [stage_41_v15, stage_41_v101], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                stage_41_v103: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_65, [stage_41_v100, stage_41_v15])
                stage_41_v104: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_41_v102, target_type=pl.FP32, mode='none')
                stage_41_v105: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_41_v104, stage_41_v103)
                stage_41_v106: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v108, stage_41_v105)
                stage_41_v107: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_41_v106)
            stage_41_v17: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_41_v107, stage_41_v14)
            stage_41_v18: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.create([1, 4096], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            stage_41_v19: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_41_v17, target_type=pl.BF16, mode='rint')
            stage_41_v20: pl.Scalar[pl.INDEX] = pl.min(144 - stage_41_v15, 1)
            stage_41_v21: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_41_v19, target_type=pl.FP32, mode='round')
            stage_41_v22: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.load(view_3, [stage_41_v15, stage_41_v11], [1, 4096], [stage_41_v20, pl.min(pl.max(16384 - stage_41_v11, 0), 4096)], target_memory=pl.Mem.Vec)
            stage_41_v23: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.cast(stage_41_v22, target_type=pl.FP32, mode='round')
            stage_41_v24: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.load(view_3, [stage_41_v15, stage_41_v11 + 4096], [1, 4096], [stage_41_v20, pl.min(pl.max(12288 - stage_41_v11, 0), 4096)], target_memory=pl.Mem.Vec)
            stage_41_v25: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.cast(stage_41_v24, target_type=pl.FP32, mode='round')
            stage_41_v26: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.load(view_3, [stage_41_v15, stage_41_v11 + 8192], [1, 4096], [stage_41_v20, pl.min(pl.max(8192 - stage_41_v11, 0), 4096)], target_memory=pl.Mem.Vec)
            stage_41_v27: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.cast(stage_41_v26, target_type=pl.FP32, mode='round')
            stage_41_v28: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.load(view_3, [stage_41_v15, stage_41_v11 + 12288], [1, 4096], [stage_41_v20, pl.min(pl.max(4096 - stage_41_v11, 0), 4096)], target_memory=pl.Mem.Vec)
            stage_41_v29: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.cast(stage_41_v28, target_type=pl.FP32, mode='round')
            stage_41_v30: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_0, [stage_41_v15, 0])
            stage_41_v31: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_41_v21, stage_41_v30)
            stage_41_v32: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 0])
            stage_41_v33: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v23, stage_41_v32)
            stage_41_v34: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v31, stage_41_v33)
            stage_41_v35: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 4])
            stage_41_v36: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v25, stage_41_v35)
            stage_41_v37: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v34, stage_41_v36)
            stage_41_v38: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 8])
            stage_41_v39: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v27, stage_41_v38)
            stage_41_v40: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v37, stage_41_v39)
            stage_41_v41: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 12])
            stage_41_v42: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v29, stage_41_v41)
            stage_41_v43: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v40, stage_41_v42)
            stage_41_v44: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_41_v43, target_type=pl.BF16, mode='rint')
            stage_41_v45: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, 4096])] = pl.tile.set_validshape(stage_41_v44, stage_41_v20, 4096)
            stage_41_v46 = pl.tile.store(stage_41_v45, [stage_41_v15, stage_41_v11], stage_41_v99)
            stage_41_v47: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_0, [stage_41_v15, 1])
            stage_41_v48: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_41_v21, stage_41_v47)
            stage_41_v49: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 1])
            stage_41_v50: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v23, stage_41_v49)
            stage_41_v51: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v48, stage_41_v50)
            stage_41_v52: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 5])
            stage_41_v53: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v25, stage_41_v52)
            stage_41_v54: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v51, stage_41_v53)
            stage_41_v55: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 9])
            stage_41_v56: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v27, stage_41_v55)
            stage_41_v57: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v54, stage_41_v56)
            stage_41_v58: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 13])
            stage_41_v59: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v29, stage_41_v58)
            stage_41_v60: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v57, stage_41_v59)
            stage_41_v61: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_41_v60, target_type=pl.BF16, mode='rint')
            stage_41_v62: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, 4096])] = pl.tile.set_validshape(stage_41_v61, stage_41_v20, 4096)
            stage_41_v63 = pl.tile.store(stage_41_v62, [stage_41_v15, stage_41_v11 + 4096], stage_41_v46)
            stage_41_v64: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_0, [stage_41_v15, 2])
            stage_41_v65: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_41_v21, stage_41_v64)
            stage_41_v66: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 2])
            stage_41_v67: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v23, stage_41_v66)
            stage_41_v68: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v65, stage_41_v67)
            stage_41_v69: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 6])
            stage_41_v70: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v25, stage_41_v69)
            stage_41_v71: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v68, stage_41_v70)
            stage_41_v72: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 10])
            stage_41_v73: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v27, stage_41_v72)
            stage_41_v74: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v71, stage_41_v73)
            stage_41_v75: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 14])
            stage_41_v76: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v29, stage_41_v75)
            stage_41_v77: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v74, stage_41_v76)
            stage_41_v78: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_41_v77, target_type=pl.BF16, mode='rint')
            stage_41_v79: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, 4096])] = pl.tile.set_validshape(stage_41_v78, stage_41_v20, 4096)
            stage_41_v80 = pl.tile.store(stage_41_v79, [stage_41_v15, stage_41_v11 + 8192], stage_41_v63)
            stage_41_v81: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_0, [stage_41_v15, 3])
            stage_41_v82: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_41_v21, stage_41_v81)
            stage_41_v83: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 3])
            stage_41_v84: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v23, stage_41_v83)
            stage_41_v85: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v82, stage_41_v84)
            stage_41_v86: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 7])
            stage_41_v87: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v25, stage_41_v86)
            stage_41_v88: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v85, stage_41_v87)
            stage_41_v89: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 11])
            stage_41_v90: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v27, stage_41_v89)
            stage_41_v91: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v88, stage_41_v90)
            stage_41_v92: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_41_v15, 15])
            stage_41_v93: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_41_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_41_v29, stage_41_v92)
            stage_41_v94: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_41_v91, stage_41_v93)
            stage_41_v95: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_41_v94, target_type=pl.BF16, mode='rint')
            stage_41_v96: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v20, 4096])] = pl.tile.set_validshape(stage_41_v95, stage_41_v20, 4096)
            stage_41_v97 = pl.tile.store(stage_41_v96, [stage_41_v15, stage_41_v11 + 12288], stage_41_v80)
            stage_41_v98 = pl.yield_(stage_41_v97)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_41 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_41)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    return (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_129, progress)

@pl.jit
def entry(temporary_4: pl.InOut[pl.Tensor[[144, 1], pl.FP32]], temporary_5: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]], view_2: pl.InOut[pl.Tensor[[144, 16384], pl.FP32]], view_3: pl.Tensor[[144, 16384], pl.BF16], temporary_7: pl.InOut[pl.Tensor[[576, 32], pl.FP32]], arg_hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], temporary_8: pl.InOut[pl.Tensor[[144, 32], pl.FP32]], arg_hc_attn_scale: pl.Tensor[[3], pl.FP32], temporary_0: pl.InOut[pl.Tensor[[144, 4], pl.FP32]], temporary_9: pl.InOut[pl.Tensor[[8, 8], pl.FP32]], temporary_6: pl.InOut[pl.Tensor[[144, 8], pl.FP32]], arg_hc_attn_base: pl.Tensor[[24], pl.FP32], view_12: pl.Tensor[[1, 24], pl.FP32], temporary_1: pl.InOut[pl.Tensor[[144, 16], pl.FP32]], temporary_10: pl.InOut[pl.Tensor[[8, 32], pl.FP32]], temporary_11: pl.InOut[pl.Tensor[[144, 4096], pl.BF16]], temporary_2: pl.InOut[pl.Tensor[[144, 4096], pl.BF16]], temporary_12: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]], arg_attn_norm_w: pl.Tensor[[4096], pl.BF16], arg_cmp_seq_lens: pl.Tensor[[24], pl.INT32], arg_cmp_query_start_loc: pl.Tensor[[25], pl.INT32], temporary_14: pl.InOut[pl.Tensor[[24], pl.INT32]], arg_kv_seq_lens: pl.Tensor[[24], pl.INT32], arg_idx_query_start_loc: pl.Tensor[[25], pl.INT32], temporary_15: pl.InOut[pl.Tensor[[24], pl.INT32]], arg_freqs_sin: pl.Tensor[[144, 64], pl.FP32], temporary_13: pl.InOut[pl.Tensor[[144, 64], pl.FP32]], temporary_20: pl.InOut[pl.Tensor[[144, 64], pl.FP32]], temporary_21: pl.InOut[pl.Tensor[[144, 64], pl.INT32]], temporary_24: pl.InOut[pl.Tensor[[144, 1024], pl.FP32]], arg_wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ], view_31: pl.Tensor[[1, 1024], pl.BF16], temporary_23: pl.InOut[pl.Tensor[[512, 1], pl.FP32]], temporary_19: pl.InOut[pl.Tensor[[144, 1], pl.FP32]], temporary_22: pl.InOut[pl.Tensor[[512, 1024], pl.INT8]], temporary_18: pl.InOut[pl.Tensor[[144, 1024], pl.INT8]], temporary_25: pl.InOut[pl.Tensor[[256, 32768], pl.INT32]], arg_wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ], arg_freqs_cos: pl.Tensor[[144, 64], pl.FP32], arg_wq_b_scale: pl.Tensor[[32768], pl.FP32], view_40: pl.InOut[pl.Tensor[[144, 32768], pl.BF16]], temporary_26: pl.InOut[pl.Tensor[[144, 512], pl.FP32]], arg_wkv: pl.Tensor[[4096, 512], pl.BF16], temporary_17: pl.InOut[pl.Tensor[[144, 512], pl.BF16]], arg_gamma_ckv: pl.Tensor[[512], pl.BF16], view_45: pl.InOut[pl.Tensor[[6944, 512], pl.BF16]], arg_ori_slot_mapping: pl.Tensor[[144, 2], pl.INT32], temporary_29: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], temporary_30: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], arg_cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], arg_cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], temporary_28: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], arg_position_ids: pl.Tensor[[144], pl.INT64], arg_cmp_ape: pl.Tensor[[4, 1024], pl.FP32], arg_state_block_table: pl.Tensor[[24, 4100], pl.INT32], arg_compress_state: pl.InOut[pl.Tensor[[217, 8192], pl.FP32]], arg_state_slot_mapping: pl.Tensor[[144, 2], pl.INT32], arg_cmp_freqs_cos: pl.Tensor[[60, 64], pl.FP32], arg_cmp_freqs_sin: pl.Tensor[[60, 64], pl.FP32], temporary_31: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], view_60: pl.Tensor[[1, 512], pl.BF16], view_61: pl.InOut[pl.Tensor[[50720, 512], pl.BF16]], temporary_27: pl.InOut[pl.Tensor[[144, 512], pl.FP32]], arg_cmp_slot_mapping: pl.Tensor[[60, 2], pl.INT32], temporary_34: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], temporary_35: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], arg_inner_wkv: pl.Tensor[[256, 4096], pl.BF16], arg_inner_wgate: pl.Tensor[[256, 4096], pl.BF16], temporary_36: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_inner_ape: pl.Tensor[[4, 256], pl.FP32], arg_inner_state_block_table: pl.Tensor[[24, 4100], pl.INT32], arg_inner_compress_state: pl.InOut[pl.Tensor[[217, 1040], pl.FP32]], arg_inner_state_slot_mapping: pl.Tensor[[144, 2], pl.INT32], temporary_33: pl.InOut[pl.Tensor[[384, 128], pl.BF16]], arg_inner_freqs_cos: pl.Tensor[[60, 64], pl.FP32], arg_inner_freqs_sin: pl.Tensor[[60, 64], pl.FP32], view_76: pl.Tensor[[1, 128], pl.BF16], temporary_38: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_hadamard_idx: pl.Tensor[[128, 128], pl.BF16], temporary_37: pl.InOut[pl.Tensor[[384, 1], pl.FP32]], arg_idx_native_kv_cache: pl.InOut[pl.Tensor[[1585, 4160], pl.INT8]], temporary_32: pl.InOut[pl.Tensor[[144, 128], pl.FP32]], arg_idx_slot_mapping: pl.Tensor[[60, 2], pl.INT32], temporary_42: pl.InOut[pl.Tensor[[384, 8192], pl.INT32]], arg_idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], view_85: pl.InOut[pl.Tensor[[384, 8192], pl.BF16]], arg_idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], temporary_43: pl.InOut[pl.Tensor[[9216, 128], pl.FP32]], temporary_41: pl.InOut[pl.Tensor[[24576, 128], pl.BF16]], temporary_39: pl.InOut[pl.Tensor[[24576, 128], pl.INT8]], temporary_40: pl.InOut[pl.Tensor[[24576, 1], pl.FP32]], temporary_45: pl.InOut[pl.Tensor[[1536, 64], pl.FP32]], arg_weights_proj: pl.Tensor[[4096, 64], pl.BF16], temporary_44: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], view_94: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_48: pl.InOut[pl.Tensor[[1024, 384], pl.FP16]], temporary_50: pl.InOut[pl.Tensor[[256], pl.INT64]], arg_idx_block_table: pl.Tensor[[24, 66], pl.INT32], view_98: pl.InOut[pl.Tensor[[51512, 128], pl.INT8]], view_99: pl.InOut[pl.Tensor[[51512, 128], pl.INT8]], temporary_49: pl.InOut[pl.Tensor[[288, 768], pl.FP32]], temporary_47: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_46: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32]], arg_idx_topk_scores: pl.InOut[pl.Tensor[[144, 512], pl.FP32]], arg_idx_topk: pl.InOut[pl.Tensor[[144, 512], pl.INT32]], temporary_53: pl.InOut[pl.Tensor[[144, 512], pl.INT32]], temporary_52: pl.InOut[pl.Tensor[[144, 640], pl.FP32]], view_107: pl.Tensor[[144, 1], pl.INT64], temporary_54: pl.InOut[pl.Tensor[[144, 16], pl.INT32]], arg_ori_block_table: pl.Tensor[[24, 258], pl.INT32], temporary_55: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_62: pl.InOut[pl.Tensor[[256], pl.INT64]], view_112: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_56: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_57: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], temporary_58: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], temporary_59: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], view_117: pl.Tensor[[64, 1], pl.FP32], arg_cmp_block_table: pl.Tensor[[24, 66], pl.INT32], temporary_60: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_61: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], arg_wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ], temporary_63: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_65: pl.InOut[pl.Tensor[[8, 384], pl.FP32]], temporary_64: pl.InOut[pl.Tensor[[384, 8192], pl.INT8]], arg_wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ], temporary_66: pl.InOut[pl.Tensor[[384, 32768], pl.INT32]], arg_wo_b_scale: pl.Tensor[[4096], pl.FP32], view_129: pl.InOut[pl.Tensor[[144, 16384], pl.BF16]], progress: pl.InOut[pl.Tensor[[4608], pl.INT64]], records: pl.Tensor[[34128], pl.INT64], waits: pl.Tensor[[38978], pl.INT64]) -> tuple[pl.Tensor[[144, 1], pl.FP32], pl.Tensor[[8, 16384], pl.FP32], pl.Tensor[[144, 16384], pl.FP32], pl.Tensor[[576, 32], pl.FP32], pl.Tensor[[144, 32], pl.FP32], pl.Tensor[[144, 4], pl.FP32], pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[144, 8], pl.FP32], pl.Tensor[[144, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32], pl.Tensor[[144, 4096], pl.BF16], pl.Tensor[[144, 4096], pl.BF16], pl.Tensor[[8, 4096], pl.BF16], pl.Tensor[[24], pl.INT32], pl.Tensor[[24], pl.INT32], pl.Tensor[[144, 64], pl.FP32], pl.Tensor[[144, 64], pl.FP32], pl.Tensor[[144, 64], pl.INT32], pl.Tensor[[144, 1024], pl.FP32], pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[144, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[144, 1024], pl.INT8], pl.Tensor[[256, 32768], pl.INT32], pl.Tensor[[144, 32768], pl.BF16], pl.Tensor[[144, 512], pl.FP32], pl.Tensor[[144, 512], pl.BF16], pl.Tensor[[6944, 512], pl.BF16], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[217, 8192], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[50720, 512], pl.BF16], pl.Tensor[[144, 512], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[217, 1040], pl.FP32], pl.Tensor[[384, 128], pl.BF16], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[1585, 4160], pl.INT8], pl.Tensor[[144, 128], pl.FP32], pl.Tensor[[384, 8192], pl.INT32], pl.Tensor[[384, 8192], pl.BF16], pl.Tensor[[9216, 128], pl.FP32], pl.Tensor[[24576, 128], pl.BF16], pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Tensor[[1536, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[1024, 384], pl.FP16], pl.Tensor[[256], pl.INT64], pl.Tensor[[51512, 128], pl.INT8], pl.Tensor[[51512, 128], pl.INT8], pl.Tensor[[288, 768], pl.FP32], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Tensor[[144, 512], pl.FP32], pl.Tensor[[144, 512], pl.INT32], pl.Tensor[[144, 512], pl.INT32], pl.Tensor[[144, 640], pl.FP32], pl.Tensor[[144, 16], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[256], pl.INT64], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8], pl.Tensor[[384, 32768], pl.INT32], pl.Tensor[[144, 16384], pl.BF16], pl.Tensor[[4608], pl.INT64]]:
    pl.func_attr({'mx_tensor_views_blocked': True})
    with pl.spmd(24, sync_start=True, name_hint='csa_static_full_card'):
        with pl.cluster(name_hint='csa_static_group'):
            (aic_return_temporary_4, aic_return_temporary_5, aic_return_view_2, aic_return_temporary_7, aic_return_temporary_8, aic_return_temporary_0, aic_return_temporary_9, aic_return_temporary_6, aic_return_temporary_1, aic_return_temporary_10, aic_return_temporary_11, aic_return_temporary_2, aic_return_temporary_12, aic_return_temporary_14, aic_return_temporary_15, aic_return_temporary_13, aic_return_temporary_20, aic_return_temporary_21, aic_return_temporary_24, aic_return_temporary_23, aic_return_temporary_19, aic_return_temporary_22, aic_return_temporary_18, aic_return_temporary_25, aic_return_view_40, aic_return_temporary_26, aic_return_temporary_17, aic_return_view_45, aic_return_temporary_29, aic_return_temporary_30, aic_return_temporary_28, aic_return_arg_compress_state, aic_return_temporary_31, aic_return_view_61, aic_return_temporary_27, aic_return_temporary_34, aic_return_temporary_35, aic_return_temporary_36, aic_return_arg_inner_compress_state, aic_return_temporary_33, aic_return_temporary_38, aic_return_temporary_37, aic_return_arg_idx_native_kv_cache, aic_return_temporary_32, aic_return_temporary_42, aic_return_view_85, aic_return_temporary_43, aic_return_temporary_41, aic_return_temporary_39, aic_return_temporary_40, aic_return_temporary_45, aic_return_temporary_44, aic_return_view_94, aic_return_temporary_48, aic_return_temporary_50, aic_return_view_98, aic_return_view_99, aic_return_temporary_49, aic_return_temporary_47, aic_return_temporary_46, aic_return_arg_idx_topk_scores, aic_return_arg_idx_topk, aic_return_temporary_53, aic_return_temporary_52, aic_return_temporary_54, aic_return_temporary_55, aic_return_temporary_62, aic_return_view_112, aic_return_temporary_56, aic_return_temporary_57, aic_return_temporary_58, aic_return_temporary_59, aic_return_temporary_60, aic_return_temporary_61, aic_return_temporary_51, aic_return_temporary_63, aic_return_temporary_65, aic_return_temporary_64, aic_return_temporary_66, aic_return_view_129, aic_return_progress) = csa_static_aic(temporary_4, temporary_5, view_2, view_3, temporary_7, arg_hc_attn_fn, temporary_8, arg_hc_attn_scale, temporary_0, temporary_9, temporary_6, arg_hc_attn_base, view_12, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, arg_attn_norm_w, arg_cmp_seq_lens, arg_cmp_query_start_loc, temporary_14, arg_kv_seq_lens, arg_idx_query_start_loc, temporary_15, arg_freqs_sin, temporary_13, temporary_20, temporary_21, temporary_24, arg_wq_a, view_31, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, arg_wq_b, arg_freqs_cos, arg_wq_b_scale, view_40, temporary_26, arg_wkv, temporary_17, arg_gamma_ckv, view_45, arg_ori_slot_mapping, temporary_29, temporary_30, arg_cmp_wkv, arg_cmp_wgate, temporary_28, arg_position_ids, arg_cmp_ape, arg_state_block_table, arg_compress_state, arg_state_slot_mapping, arg_cmp_freqs_cos, arg_cmp_freqs_sin, temporary_31, view_60, view_61, temporary_27, arg_cmp_slot_mapping, temporary_34, temporary_35, arg_inner_wkv, arg_inner_wgate, temporary_36, arg_inner_ape, arg_inner_state_block_table, arg_inner_compress_state, arg_inner_state_slot_mapping, temporary_33, arg_inner_freqs_cos, arg_inner_freqs_sin, view_76, temporary_38, arg_hadamard_idx, temporary_37, arg_idx_native_kv_cache, temporary_32, arg_idx_slot_mapping, temporary_42, arg_idx_wq_b, view_85, arg_idx_wq_b_scale, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, arg_weights_proj, temporary_44, view_94, temporary_48, temporary_50, arg_idx_block_table, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, view_107, temporary_54, arg_ori_block_table, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, view_117, arg_cmp_block_table, temporary_60, temporary_61, temporary_51, arg_wo_a, temporary_63, temporary_65, temporary_64, arg_wo_b, temporary_66, arg_wo_b_scale, view_129, progress, records, waits)
            (aiv_return_temporary_4, aiv_return_temporary_5, aiv_return_view_2, aiv_return_temporary_7, aiv_return_temporary_8, aiv_return_temporary_0, aiv_return_temporary_9, aiv_return_temporary_6, aiv_return_temporary_1, aiv_return_temporary_10, aiv_return_temporary_11, aiv_return_temporary_2, aiv_return_temporary_12, aiv_return_temporary_14, aiv_return_temporary_15, aiv_return_temporary_13, aiv_return_temporary_20, aiv_return_temporary_21, aiv_return_temporary_24, aiv_return_temporary_23, aiv_return_temporary_19, aiv_return_temporary_22, aiv_return_temporary_18, aiv_return_temporary_25, aiv_return_view_40, aiv_return_temporary_26, aiv_return_temporary_17, aiv_return_view_45, aiv_return_temporary_29, aiv_return_temporary_30, aiv_return_temporary_28, aiv_return_arg_compress_state, aiv_return_temporary_31, aiv_return_view_61, aiv_return_temporary_27, aiv_return_temporary_34, aiv_return_temporary_35, aiv_return_temporary_36, aiv_return_arg_inner_compress_state, aiv_return_temporary_33, aiv_return_temporary_38, aiv_return_temporary_37, aiv_return_arg_idx_native_kv_cache, aiv_return_temporary_32, aiv_return_temporary_42, aiv_return_view_85, aiv_return_temporary_43, aiv_return_temporary_41, aiv_return_temporary_39, aiv_return_temporary_40, aiv_return_temporary_45, aiv_return_temporary_44, aiv_return_view_94, aiv_return_temporary_48, aiv_return_temporary_50, aiv_return_view_98, aiv_return_view_99, aiv_return_temporary_49, aiv_return_temporary_47, aiv_return_temporary_46, aiv_return_arg_idx_topk_scores, aiv_return_arg_idx_topk, aiv_return_temporary_53, aiv_return_temporary_52, aiv_return_temporary_54, aiv_return_temporary_55, aiv_return_temporary_62, aiv_return_view_112, aiv_return_temporary_56, aiv_return_temporary_57, aiv_return_temporary_58, aiv_return_temporary_59, aiv_return_temporary_60, aiv_return_temporary_61, aiv_return_temporary_51, aiv_return_temporary_63, aiv_return_temporary_65, aiv_return_temporary_64, aiv_return_temporary_66, aiv_return_view_129, aiv_return_progress) = csa_static_aiv(temporary_4, temporary_5, view_2, view_3, temporary_7, arg_hc_attn_fn, temporary_8, arg_hc_attn_scale, temporary_0, temporary_9, temporary_6, arg_hc_attn_base, view_12, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, arg_attn_norm_w, arg_cmp_seq_lens, arg_cmp_query_start_loc, temporary_14, arg_kv_seq_lens, arg_idx_query_start_loc, temporary_15, arg_freqs_sin, temporary_13, temporary_20, temporary_21, temporary_24, arg_wq_a, view_31, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, arg_wq_b, arg_freqs_cos, arg_wq_b_scale, view_40, temporary_26, arg_wkv, temporary_17, arg_gamma_ckv, view_45, arg_ori_slot_mapping, temporary_29, temporary_30, arg_cmp_wkv, arg_cmp_wgate, temporary_28, arg_position_ids, arg_cmp_ape, arg_state_block_table, arg_compress_state, arg_state_slot_mapping, arg_cmp_freqs_cos, arg_cmp_freqs_sin, temporary_31, view_60, view_61, temporary_27, arg_cmp_slot_mapping, temporary_34, temporary_35, arg_inner_wkv, arg_inner_wgate, temporary_36, arg_inner_ape, arg_inner_state_block_table, arg_inner_compress_state, arg_inner_state_slot_mapping, temporary_33, arg_inner_freqs_cos, arg_inner_freqs_sin, view_76, temporary_38, arg_hadamard_idx, temporary_37, arg_idx_native_kv_cache, temporary_32, arg_idx_slot_mapping, temporary_42, arg_idx_wq_b, view_85, arg_idx_wq_b_scale, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, arg_weights_proj, temporary_44, view_94, temporary_48, temporary_50, arg_idx_block_table, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, view_107, temporary_54, arg_ori_block_table, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, view_117, arg_cmp_block_table, temporary_60, temporary_61, temporary_51, arg_wo_a, temporary_63, temporary_65, temporary_64, arg_wo_b, temporary_66, arg_wo_b_scale, view_129, progress, records, waits)
    return (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_129, progress)
