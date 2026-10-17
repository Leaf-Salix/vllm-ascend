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
def csa_static_aic(temporary_4: pl.InOut[pl.Tensor[[96, 1], pl.FP32]], temporary_5: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]], view_2: pl.InOut[pl.Tensor[[96, 16384], pl.FP32]], view_3: pl.Tensor[[96, 16384], pl.BF16], temporary_7: pl.InOut[pl.Tensor[[384, 32], pl.FP32]], arg_hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], temporary_8: pl.InOut[pl.Tensor[[96, 32], pl.FP32]], arg_hc_attn_scale: pl.Tensor[[3], pl.FP32], temporary_0: pl.InOut[pl.Tensor[[96, 4], pl.FP32]], temporary_9: pl.InOut[pl.Tensor[[8, 8], pl.FP32]], temporary_6: pl.InOut[pl.Tensor[[96, 8], pl.FP32]], arg_hc_attn_base: pl.Tensor[[24], pl.FP32], view_12: pl.Tensor[[1, 24], pl.FP32], temporary_1: pl.InOut[pl.Tensor[[96, 16], pl.FP32]], temporary_10: pl.InOut[pl.Tensor[[8, 32], pl.FP32]], temporary_11: pl.InOut[pl.Tensor[[96, 4096], pl.BF16]], temporary_2: pl.InOut[pl.Tensor[[96, 4096], pl.BF16]], temporary_12: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]], arg_attn_norm_w: pl.Tensor[[4096], pl.BF16], arg_cmp_seq_lens: pl.Tensor[[16], pl.INT32], arg_cmp_query_start_loc: pl.Tensor[[17], pl.INT32], temporary_14: pl.InOut[pl.Tensor[[16], pl.INT32]], arg_kv_seq_lens: pl.Tensor[[16], pl.INT32], arg_idx_query_start_loc: pl.Tensor[[17], pl.INT32], temporary_15: pl.InOut[pl.Tensor[[16], pl.INT32]], arg_freqs_sin: pl.Tensor[[96, 64], pl.FP32], temporary_13: pl.InOut[pl.Tensor[[96, 64], pl.FP32]], temporary_20: pl.InOut[pl.Tensor[[96, 64], pl.FP32]], temporary_21: pl.InOut[pl.Tensor[[96, 64], pl.INT32]], temporary_24: pl.InOut[pl.Tensor[[96, 1024], pl.FP32]], arg_wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ], view_31: pl.Tensor[[1, 1024], pl.BF16], temporary_23: pl.InOut[pl.Tensor[[512, 1], pl.FP32]], temporary_19: pl.InOut[pl.Tensor[[96, 1], pl.FP32]], temporary_22: pl.InOut[pl.Tensor[[512, 1024], pl.INT8]], temporary_18: pl.InOut[pl.Tensor[[96, 1024], pl.INT8]], temporary_25: pl.InOut[pl.Tensor[[128, 32768], pl.INT32]], arg_wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ], arg_freqs_cos: pl.Tensor[[96, 64], pl.FP32], arg_wq_b_scale: pl.Tensor[[32768], pl.FP32], view_40: pl.InOut[pl.Tensor[[96, 32768], pl.BF16]], temporary_26: pl.InOut[pl.Tensor[[96, 512], pl.FP32]], arg_wkv: pl.Tensor[[4096, 512], pl.BF16], temporary_17: pl.InOut[pl.Tensor[[96, 512], pl.BF16]], arg_gamma_ckv: pl.Tensor[[512], pl.BF16], view_45: pl.InOut[pl.Tensor[[4640, 512], pl.BF16]], arg_ori_slot_mapping: pl.Tensor[[96, 2], pl.INT32], temporary_29: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], temporary_30: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], arg_cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], arg_cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], temporary_28: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], arg_position_ids: pl.Tensor[[96], pl.INT64], arg_cmp_ape: pl.Tensor[[4, 1024], pl.FP32], arg_state_block_table: pl.Tensor[[16, 65540], pl.INT32], arg_compress_state: pl.InOut[pl.Tensor[[145, 8192], pl.FP32]], arg_state_slot_mapping: pl.Tensor[[96, 2], pl.INT32], arg_cmp_freqs_cos: pl.Tensor[[40, 64], pl.FP32], arg_cmp_freqs_sin: pl.Tensor[[40, 64], pl.FP32], temporary_31: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], view_60: pl.Tensor[[1, 512], pl.BF16], view_61: pl.InOut[pl.Tensor[[525344, 512], pl.BF16]], temporary_27: pl.InOut[pl.Tensor[[96, 512], pl.FP32]], arg_cmp_slot_mapping: pl.Tensor[[40, 2], pl.INT32], temporary_34: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], temporary_35: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], arg_inner_wkv: pl.Tensor[[256, 4096], pl.BF16], arg_inner_wgate: pl.Tensor[[256, 4096], pl.BF16], temporary_36: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_inner_ape: pl.Tensor[[4, 256], pl.FP32], arg_inner_state_block_table: pl.Tensor[[16, 65540], pl.INT32], arg_inner_compress_state: pl.InOut[pl.Tensor[[145, 1040], pl.FP32]], arg_inner_state_slot_mapping: pl.Tensor[[96, 2], pl.INT32], temporary_33: pl.InOut[pl.Tensor[[384, 128], pl.BF16]], arg_inner_freqs_cos: pl.Tensor[[40, 64], pl.FP32], arg_inner_freqs_sin: pl.Tensor[[40, 64], pl.FP32], view_76: pl.Tensor[[1, 128], pl.BF16], temporary_38: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_hadamard_idx: pl.Tensor[[128, 128], pl.BF16], temporary_37: pl.InOut[pl.Tensor[[384, 1], pl.FP32]], arg_idx_native_kv_cache: pl.InOut[pl.Tensor[[16417, 4160], pl.INT8]], temporary_32: pl.InOut[pl.Tensor[[96, 128], pl.FP32]], arg_idx_slot_mapping: pl.Tensor[[40, 2], pl.INT32], temporary_42: pl.InOut[pl.Tensor[[384, 8192], pl.INT32]], arg_idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], view_85: pl.InOut[pl.Tensor[[384, 8192], pl.BF16]], arg_idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], temporary_43: pl.InOut[pl.Tensor[[6144, 128], pl.FP32]], temporary_41: pl.InOut[pl.Tensor[[24576, 128], pl.BF16]], temporary_39: pl.InOut[pl.Tensor[[24576, 128], pl.INT8]], temporary_40: pl.InOut[pl.Tensor[[24576, 1], pl.FP32]], temporary_45: pl.InOut[pl.Tensor[[1536, 64], pl.FP32]], arg_weights_proj: pl.Tensor[[4096, 64], pl.BF16], temporary_44: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], view_94: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_48: pl.InOut[pl.Tensor[[1024, 384], pl.FP16]], temporary_50: pl.InOut[pl.Tensor[[256], pl.INT64]], arg_idx_block_table: pl.Tensor[[16, 1026], pl.INT32], view_98: pl.InOut[pl.Tensor[[533552, 128], pl.INT8]], view_99: pl.InOut[pl.Tensor[[533552, 128], pl.INT8]], temporary_49: pl.InOut[pl.Tensor[[288, 1024], pl.FP32]], temporary_46: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32]], arg_idx_topk_scores: pl.InOut[pl.Tensor[[96, 512], pl.FP32]], arg_idx_topk: pl.InOut[pl.Tensor[[96, 512], pl.INT32]], temporary_53: pl.InOut[pl.Tensor[[96, 512], pl.INT32]], temporary_52: pl.InOut[pl.Tensor[[96, 640], pl.FP32]], view_106: pl.Tensor[[96, 1], pl.INT64], temporary_54: pl.InOut[pl.Tensor[[96, 16], pl.INT32]], arg_ori_block_table: pl.Tensor[[16, 4098], pl.INT32], temporary_55: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_62: pl.InOut[pl.Tensor[[256], pl.INT64]], view_111: pl.InOut[pl.Tensor[[6144, 512], pl.BF16]], temporary_56: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_57: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], temporary_58: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], temporary_59: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], view_116: pl.Tensor[[64, 1], pl.FP32], arg_cmp_block_table: pl.Tensor[[16, 1026], pl.INT32], temporary_60: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_61: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], arg_wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ], temporary_63: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_65: pl.InOut[pl.Tensor[[8, 384], pl.FP32]], temporary_64: pl.InOut[pl.Tensor[[384, 8192], pl.INT8]], arg_wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ], temporary_66: pl.InOut[pl.Tensor[[384, 32768], pl.INT32]], arg_wo_b_scale: pl.Tensor[[4096], pl.FP32], view_128: pl.InOut[pl.Tensor[[96, 16384], pl.BF16]], progress: pl.InOut[pl.Tensor[[4608], pl.INT64]], records: pl.Tensor[[36288], pl.INT64], waits: pl.Tensor[[33112], pl.INT64]) -> tuple[pl.Tensor[[96, 1], pl.FP32], pl.Tensor[[8, 16384], pl.FP32], pl.Tensor[[96, 16384], pl.FP32], pl.Tensor[[384, 32], pl.FP32], pl.Tensor[[96, 32], pl.FP32], pl.Tensor[[96, 4], pl.FP32], pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[96, 8], pl.FP32], pl.Tensor[[96, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32], pl.Tensor[[96, 4096], pl.BF16], pl.Tensor[[96, 4096], pl.BF16], pl.Tensor[[8, 4096], pl.BF16], pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.INT32], pl.Tensor[[96, 64], pl.FP32], pl.Tensor[[96, 64], pl.FP32], pl.Tensor[[96, 64], pl.INT32], pl.Tensor[[96, 1024], pl.FP32], pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[96, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[96, 1024], pl.INT8], pl.Tensor[[128, 32768], pl.INT32], pl.Tensor[[96, 32768], pl.BF16], pl.Tensor[[96, 512], pl.FP32], pl.Tensor[[96, 512], pl.BF16], pl.Tensor[[4640, 512], pl.BF16], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[145, 8192], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[525344, 512], pl.BF16], pl.Tensor[[96, 512], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[145, 1040], pl.FP32], pl.Tensor[[384, 128], pl.BF16], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[16417, 4160], pl.INT8], pl.Tensor[[96, 128], pl.FP32], pl.Tensor[[384, 8192], pl.INT32], pl.Tensor[[384, 8192], pl.BF16], pl.Tensor[[6144, 128], pl.FP32], pl.Tensor[[24576, 128], pl.BF16], pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Tensor[[1536, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[1024, 384], pl.FP16], pl.Tensor[[256], pl.INT64], pl.Tensor[[533552, 128], pl.INT8], pl.Tensor[[533552, 128], pl.INT8], pl.Tensor[[288, 1024], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Tensor[[96, 512], pl.FP32], pl.Tensor[[96, 512], pl.INT32], pl.Tensor[[96, 512], pl.INT32], pl.Tensor[[96, 640], pl.FP32], pl.Tensor[[96, 16], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[256], pl.INT64], pl.Tensor[[6144, 512], pl.BF16], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8], pl.Tensor[[384, 32768], pl.INT32], pl.Tensor[[96, 16384], pl.BF16], pl.Tensor[[4608], pl.INT64]]:
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
        for (stage_0_v5, (stage_0_v14,)) in pl.range(stage_0_v4, 24, 24, init_values=(temporary_7,)):
            stage_0_v6: pl.Scalar[pl.INDEX] = stage_0_v5 // 4 * 16
            stage_0_v7: pl.Scalar[pl.INDEX] = stage_0_v5 % 4
            stage_0_v8: pl.Scalar[pl.INDEX] = stage_0_v7 * 4096
            stage_0_v9: pl.Scalar[pl.INDEX] = pl.min(96 - stage_0_v6, 16)
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
            stage_0_v11: pl.Scalar[pl.INDEX] = stage_0_v7 * 96 + stage_0_v6
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
        for (stage_1_v5, (stage_1_v14,)) in pl.range(stage_1_v4, 24, 24, init_values=(temporary_7,)):
            stage_1_v6: pl.Scalar[pl.INDEX] = stage_1_v5 // 4 * 16
            stage_1_v7: pl.Scalar[pl.INDEX] = stage_1_v5 % 4
            stage_1_v8: pl.Scalar[pl.INDEX] = stage_1_v7 * 4096
            stage_1_v9: pl.Scalar[pl.INDEX] = pl.min(96 - stage_1_v6, 16)
            stage_1_v10: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_1_v15, (stage_1_v24,)) in pl.pipeline(16, stage=2, init_values=(stage_1_v10,)):
                stage_1_v16: pl.Scalar[pl.INDEX] = stage_1_v8 + stage_1_v15 * 256
                stage_1_v17: pl.Tile[[16, 256], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[stage_1_v9, 256])] = pl.tile.load(view_2, [stage_1_v6, stage_1_v16], [16, 256], [stage_1_v9, 256], target_memory=pl.Mem.Mat)
                stage_1_v18: pl.Tile[[32, 256], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[24, 256])] = pl.tile.load(arg_hc_attn_fn, [0, stage_1_v16], [32, 256], [24, 256], target_memory=pl.Mem.Mat)
                stage_1_v19: pl.Tile[[256, 32], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[256, 24], blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_1_v18)
                stage_1_v20: pl.Tile[[16, 256], pl.FP32, pl.Mem.Left, pl.TileView(valid_shape=[stage_1_v9, 256])] = pl.tile.move(stage_1_v17, target_memory=pl.Mem.Left)
                stage_1_v21: pl.Tile[[256, 32], pl.FP32, pl.Mem.Right, pl.TileView(valid_shape=[256, 24])] = pl.tile.move(stage_1_v19, target_memory=pl.Mem.Right)
                stage_1_v22: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_1_v24, stage_1_v20, stage_1_v21, stage_1_v15 == 0)
                stage_1_v23: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_1_v22)
            stage_1_v11: pl.Scalar[pl.INDEX] = stage_1_v7 * 96 + stage_1_v6
            stage_1_v12 = pl.tile.store(stage_1_v23, [stage_1_v11, 0], stage_1_v14)
            stage_1_v13 = pl.yield_(stage_1_v12)
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
        stage_2_v5: pl.Scalar[pl.INDEX] = 0
        stage_2_v6: pl.Scalar[pl.INDEX] = stage_2_v4 % 8 * 128
        stage_2_v7: pl.Scalar[pl.INDEX] = stage_2_v4 // 8
        for (stage_2_v8, (stage_2_v17,)) in pl.range(stage_2_v7 * 32, 96, 3 * 32, init_values=(temporary_24,)):
            stage_2_v9: pl.Scalar[pl.INDEX] = 0 + stage_2_v8
            stage_2_v10 = pl.tensor.create([32, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for (stage_2_v18, (stage_2_v33,)) in pl.pipeline(16, stage=2, init_values=(stage_2_v10,)):
                stage_2_v19: pl.Scalar[pl.INDEX] = stage_2_v5 + stage_2_v18 * 256
                stage_2_v20 = pl.tensor.slice(temporary_2, [32, 256], [stage_2_v9, stage_2_v19])
                stage_2_v21 = pl.tensor.slice(arg_wq_a, [128, 256], [stage_2_v6, stage_2_v19])
                stage_2_v22 = pl.tensor.matmul_acc(stage_2_v33, stage_2_v20, stage_2_v21, init_cond=stage_2_v18 == 0, a_trans=False, b_trans=True)
                stage_2_v23 = pl.yield_(stage_2_v22)
            stage_2_v11 = pl.tensor.assemble(stage_2_v17, stage_2_v23, [stage_2_v8, stage_2_v6])
            stage_2_v12 = pl.yield_(stage_2_v11)
        for (stage_2_v13, (stage_2_v24,)) in pl.range(96 + stage_2_v7 * 16, 96, 3 * 16, init_values=(stage_2_v12,)):
            stage_2_v14 = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for (stage_2_v25, (stage_2_v34,)) in pl.pipeline(16, stage=2, init_values=(stage_2_v14,)):
                stage_2_v26: pl.Scalar[pl.INDEX] = stage_2_v5 + stage_2_v25 * 256
                stage_2_v27: pl.Scalar[pl.INDEX] = pl.min(96 - stage_2_v13, 16)
                stage_2_v28: pl.Scalar[pl.INDEX] = 0 + stage_2_v13
                stage_2_v29 = pl.tensor.slice(temporary_2, [16, 256], [stage_2_v28, stage_2_v26], [stage_2_v27, 256])
                stage_2_v30 = pl.tensor.slice(arg_wq_a, [128, 256], [stage_2_v6, stage_2_v26])
                stage_2_v31 = pl.tensor.matmul_acc(stage_2_v34, stage_2_v29, stage_2_v30, init_cond=stage_2_v25 == 0, a_trans=False, b_trans=True)
                stage_2_v32 = pl.yield_(stage_2_v31)
            stage_2_v15 = pl.tensor.assemble(stage_2_v24, stage_2_v32, [stage_2_v13, stage_2_v6])
            stage_2_v16 = pl.yield_(stage_2_v15)
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
        stage_3_v4: pl.Scalar[pl.INDEX] = block_3
        for (stage_3_v5, (stage_3_v14,)) in pl.range(stage_3_v4, 6 * 4, 8, init_values=(temporary_45,)):
            stage_3_v6: pl.Scalar[pl.INDEX] = stage_3_v5 // 4
            stage_3_v7: pl.Scalar[pl.INDEX] = stage_3_v5 - stage_3_v6 * 4
            stage_3_v8: pl.Scalar[pl.INDEX] = stage_3_v6 * 16
            stage_3_v9: pl.Scalar[pl.INDEX] = pl.min(96 - stage_3_v8, 16)
            stage_3_v10: pl.Scalar[pl.INDEX] = stage_3_v7 * 1024
            stage_3_v11: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_3_v15, (stage_3_v20,)) in pl.range(2, init_values=(stage_3_v11,)):
                stage_3_v16: pl.Scalar[pl.INDEX] = stage_3_v10 + stage_3_v15 * 512
                stage_3_v17: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_3_v9, 512])] = pl.tile.load(temporary_2, [stage_3_v8, stage_3_v16], [16, 512], [stage_3_v9, 512], target_memory=pl.Mem.Mat)
                stage_3_v18: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_weights_proj, [stage_3_v16, 0], [512, 64], [512, 64], target_memory=pl.Mem.Mat)
                for (stage_3_v21, (stage_3_v26,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_3_v20,)):
                    stage_3_v22: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_3_v9, pl.const(16, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_3_v17, 0, stage_3_v21, [16, 256], target_memory=pl.Mem.Left)
                    stage_3_v23: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_3_v18, stage_3_v21, 0, [256, 64], target_memory=pl.Mem.Right)
                    stage_3_v24: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_3_v26, stage_3_v22, stage_3_v23, stage_3_v15 == 0 and stage_3_v21 == 0)
                    stage_3_v25: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_3_v24)
                stage_3_v19: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_3_v25)
            stage_3_v12 = pl.tile.store(stage_3_v19, [stage_3_v7 * 384 + stage_3_v8, 0], stage_3_v14)
            stage_3_v13 = pl.yield_(stage_3_v12)
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
        for (stage_4_v7, (stage_4_v14, stage_4_v15)) in pl.range(stage_4_v6, 96 // 2, 24, init_values=(temporary_34, temporary_35)):
            stage_4_v8: pl.Scalar[pl.INDEX] = stage_4_v7 // 8 * 16
            stage_4_v9: pl.Scalar[pl.INDEX] = stage_4_v7 % 8 * 32
            stage_4_v10: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            stage_4_v11: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_4_v16, (stage_4_v31, stage_4_v32)) in pl.pipeline(8, stage=2, init_values=(stage_4_v10, stage_4_v11)):
                stage_4_v17: pl.Scalar[pl.INDEX] = stage_4_v16 * 512
                stage_4_v18: pl.Scalar[pl.INDEX] = pl.min(96 - stage_4_v8, 16)
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
        for (stage_5_v7, (stage_5_v14, stage_5_v15)) in pl.range(stage_5_v6, 96 // 2, 24, init_values=(temporary_34, temporary_35)):
            stage_5_v8: pl.Scalar[pl.INDEX] = stage_5_v7 // 8 * 16
            stage_5_v9: pl.Scalar[pl.INDEX] = stage_5_v7 % 8 * 32
            stage_5_v10: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            stage_5_v11: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_5_v16, (stage_5_v31, stage_5_v32)) in pl.pipeline(8, stage=2, init_values=(stage_5_v10, stage_5_v11)):
                stage_5_v17: pl.Scalar[pl.INDEX] = stage_5_v16 * 512
                stage_5_v18: pl.Scalar[pl.INDEX] = pl.min(96 - stage_5_v8, 16)
                stage_5_v19: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_5_v18, 512])] = pl.tile.load(temporary_2, [stage_5_v8, stage_5_v17], [16, 512], [stage_5_v18, 512], target_memory=pl.Mem.Mat)
                stage_5_v20: pl.Tile[[32, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_inner_wkv, [stage_5_v9, stage_5_v17], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
                stage_5_v21: pl.Tile[[32, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_inner_wgate, [stage_5_v9, stage_5_v17], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
                stage_5_v22: pl.Tile[[512, 32], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_5_v20)
                stage_5_v23: pl.Tile[[16, 512], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[stage_5_v18, 512])] = pl.tile.move(stage_5_v19, target_memory=pl.Mem.Left)
                stage_5_v24: pl.Tile[[512, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_5_v22, target_memory=pl.Mem.Right)
                stage_5_v25: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_5_v31, stage_5_v23, stage_5_v24, stage_5_v16 == 0)
                stage_5_v26: pl.Tile[[512, 32], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_5_v21)
                stage_5_v27: pl.Tile[[512, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_5_v26, target_memory=pl.Mem.Right)
                stage_5_v28: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_5_v32, stage_5_v23, stage_5_v27, stage_5_v16 == 0)
                (stage_5_v33, stage_5_v34) = pl.yield_(stage_5_v25, stage_5_v28)
            stage_5_v12 = pl.tile.store(stage_5_v33, [stage_5_v8, stage_5_v9], stage_5_v14)
            stage_5_v13 = pl.tile.store(stage_5_v34, [stage_5_v8, stage_5_v9], stage_5_v15)
            (stage_5_v29, stage_5_v30) = pl.yield_(stage_5_v12, stage_5_v13)
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
        stage_6_v4: pl.Scalar[pl.INDEX] = block_6
        stage_6_v5: pl.Scalar[pl.INDEX] = stage_6_v4 // 3 * 128
        stage_6_v6: pl.Scalar[pl.INDEX] = 0
        stage_6_v7: pl.Scalar[pl.INDEX] = stage_6_v4 % 3
        for (stage_6_v8, (stage_6_v17,)) in pl.range(stage_6_v7 * 32, 96, 3 * 32, init_values=(temporary_26,)):
            stage_6_v9: pl.Scalar[pl.INDEX] = 0 + stage_6_v8
            stage_6_v10: pl.Tile[[32, 128], pl.FP32, pl.Mem.Acc] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_6_v18, (stage_6_v31,)) in pl.pipeline(8, stage=2, init_values=(stage_6_v10,)):
                stage_6_v19: pl.Scalar[pl.INDEX] = stage_6_v6 + stage_6_v18 * 512
                stage_6_v20: pl.Tile[[32, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(temporary_2, [stage_6_v9, stage_6_v19], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
                stage_6_v21: pl.Tile[[512, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_wkv, [stage_6_v19, stage_6_v5], [512, 128], [512, 128], target_memory=pl.Mem.Mat, cache=1)
                for (stage_6_v32, (stage_6_v43,)) in pl.pipeline(0, 512, 128, stage=2, init_values=(stage_6_v31,)):
                    stage_6_v33: pl.Tile[[32, 128], pl.BF16, pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(stage_6_v20, 0, stage_6_v32, [32, 128], target_memory=pl.Mem.Left)
                    stage_6_v34: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_6_v21, stage_6_v32, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_6_v35: pl.Tile[[32, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_6_v43, stage_6_v33, stage_6_v34, stage_6_v18 == 0 and stage_6_v32 == 0)
                    stage_6_v36: pl.Tile[[32, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_6_v35)
                stage_6_v22: pl.Tile[[32, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_6_v36)
            stage_6_v11 = pl.tile.store(stage_6_v22, [stage_6_v8, stage_6_v5], stage_6_v17)
            stage_6_v12 = pl.yield_(stage_6_v11)
        for (stage_6_v13, (stage_6_v23,)) in pl.range(96 + stage_6_v7 * 16, 96, 3 * 16, init_values=(stage_6_v12,)):
            stage_6_v14: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_6_v24, (stage_6_v37,)) in pl.pipeline(8, stage=2, init_values=(stage_6_v14,)):
                stage_6_v25: pl.Scalar[pl.INDEX] = stage_6_v6 + stage_6_v24 * 512
                stage_6_v26: pl.Scalar[pl.INDEX] = pl.min(96 - stage_6_v13, 16)
                stage_6_v27: pl.Scalar[pl.INDEX] = 0 + stage_6_v13
                stage_6_v28: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_6_v26, 512])] = pl.tile.load(temporary_2, [stage_6_v27, stage_6_v25], [16, 512], [stage_6_v26, 512], target_memory=pl.Mem.Mat)
                stage_6_v29: pl.Tile[[512, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_wkv, [stage_6_v25, stage_6_v5], [512, 128], [512, 128], target_memory=pl.Mem.Mat, cache=1)
                for (stage_6_v38, (stage_6_v44,)) in pl.pipeline(0, 512, 128, stage=2, init_values=(stage_6_v37,)):
                    stage_6_v39: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_6_v26, pl.const(16, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_6_v28, 0, stage_6_v38, [16, 128], target_memory=pl.Mem.Left)
                    stage_6_v40: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_6_v29, stage_6_v38, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_6_v41: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_6_v44, stage_6_v39, stage_6_v40, stage_6_v24 == 0 and stage_6_v38 == 0)
                    stage_6_v42: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_6_v41)
                stage_6_v30: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_6_v42)
            stage_6_v15 = pl.tile.store(stage_6_v30, [stage_6_v13, stage_6_v5], stage_6_v23)
            stage_6_v16 = pl.yield_(stage_6_v15)
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
        for (stage_7_v5, (stage_7_v9,)) in pl.range(stage_7_v4, 32, 24, init_values=(temporary_42,)):
            stage_7_v6: pl.Scalar[pl.INDEX] = stage_7_v5 * 256
            stage_7_v7: pl.Tile[[1024, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_idx_wq_b, [0, stage_7_v6], [1024, 256], [1024, 256], target_memory=pl.Mem.Mat)
            for (stage_7_v10, (stage_7_v18,)) in pl.range(2, init_values=(stage_7_v9,)):
                stage_7_v11: pl.Scalar[pl.INDEX] = stage_7_v10 * 64
                stage_7_v12: pl.Scalar[pl.INDEX] = pl.min(96 - stage_7_v11, 64)
                stage_7_v13: pl.Tile[[64, 1024], pl.INT8, pl.Mem.Mat, pl.TileView(valid_shape=[stage_7_v12, 1024])] = pl.tile.load(temporary_18, [stage_7_v11, 0], [64, 1024], [stage_7_v12, 1024], target_memory=pl.Mem.Mat)
                stage_7_v14: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc, compact=True)
                stage_7_v15: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_7_v12, 256], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_7_v14, stage_7_v12, 256)
                for (stage_7_v19, (stage_7_v24,)) in pl.pipeline(0, 1024, 128, stage=2, init_values=(stage_7_v15,)):
                    stage_7_v20: pl.Tile[[64, 128], pl.INT8, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_7_v12, pl.const(64, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_7_v13, 0, stage_7_v19, [64, 128], target_memory=pl.Mem.Left)
                    stage_7_v21: pl.Tile[[128, 256], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_7_v7, stage_7_v19, 0, [128, 256], target_memory=pl.Mem.Right)
                    stage_7_v22: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_7_v12, 256], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_7_v24, stage_7_v20, stage_7_v21, stage_7_v19 == 0)
                    stage_7_v23: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_7_v12, 256], compact=pl.CompactMode.normal)] = pl.yield_(stage_7_v22)
                stage_7_v16 = pl.tile.store(stage_7_v23, [stage_7_v11, stage_7_v6], stage_7_v18)
                stage_7_v17 = pl.yield_(stage_7_v16)
            stage_7_v8 = pl.yield_(stage_7_v17)
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
        stage_8_v6: pl.Scalar[pl.INDEX] = block_8
        for (stage_8_v7, (stage_8_v19, stage_8_v20)) in pl.range(stage_8_v6, 128 // 4, 24, init_values=(temporary_29, temporary_30)):
            stage_8_v8: pl.Scalar[pl.INDEX] = stage_8_v7 // 16 * 64
            stage_8_v9: pl.Scalar[pl.INDEX] = stage_8_v7 % 16 * 64
            stage_8_v10: pl.Scalar[pl.INDEX] = pl.min(96 - stage_8_v8, 64)
            stage_8_v11: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_8_v12: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_8_v13: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            stage_8_v14: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_8_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_8_v13, stage_8_v10, 64)
            stage_8_v15: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            stage_8_v16: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_8_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_8_v15, stage_8_v10, 64)
            for (stage_8_v21, (stage_8_v28, stage_8_v29)) in pl.pipeline(8, stage=2, init_values=(stage_8_v14, stage_8_v16)):
                stage_8_v22: pl.Scalar[pl.INDEX] = stage_8_v21 * 512
                stage_8_v23: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_8_v10, 512])] = pl.tile.load(temporary_2, [stage_8_v8, stage_8_v22], [64, 512], [stage_8_v10, 512], target_memory=pl.Mem.Mat)
                stage_8_v24: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_cmp_wkv, [stage_8_v9, stage_8_v22], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
                stage_8_v25: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_cmp_wgate, [stage_8_v9, stage_8_v22], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
                if stage_8_v21 == 0:
                    stage_8_v30: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_8_v24)
                    stage_8_v31: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    stage_8_v32: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_8_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_8_v31, stage_8_v10, 64)
                    for (stage_8_v40, (stage_8_v62,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_8_v32,)):
                        stage_8_v41: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_8_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_8_v23, 0, stage_8_v40, [64, 256], target_memory=pl.Mem.Left)
                        stage_8_v42: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_8_v30, stage_8_v40, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_8_v43: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_8_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_8_v62, stage_8_v41, stage_8_v42, stage_8_v40 == 0)
                        stage_8_v44: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_8_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_8_v43)
                    stage_8_v33: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_8_v25)
                    stage_8_v34: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    stage_8_v35: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_8_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_8_v34, stage_8_v10, 64)
                    for (stage_8_v45, (stage_8_v63,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_8_v35,)):
                        stage_8_v46: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_8_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_8_v23, 0, stage_8_v45, [64, 256], target_memory=pl.Mem.Left)
                        stage_8_v47: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_8_v33, stage_8_v45, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_8_v48: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_8_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_8_v63, stage_8_v46, stage_8_v47, stage_8_v45 == 0)
                        stage_8_v49: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_8_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_8_v48)
                    (stage_8_v50, stage_8_v51) = pl.yield_(stage_8_v44, stage_8_v49)
                else:
                    stage_8_v36: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_8_v24)
                    for (stage_8_v52, (stage_8_v64,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_8_v28,)):
                        stage_8_v53: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_8_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_8_v23, 0, stage_8_v52, [64, 256], target_memory=pl.Mem.Left)
                        stage_8_v54: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_8_v36, stage_8_v52, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_8_v55: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_8_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_8_v64, stage_8_v53, stage_8_v54)
                        stage_8_v56: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_8_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_8_v55)
                    stage_8_v37: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_8_v25)
                    for (stage_8_v57, (stage_8_v65,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_8_v29,)):
                        stage_8_v58: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_8_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_8_v23, 0, stage_8_v57, [64, 256], target_memory=pl.Mem.Left)
                        stage_8_v59: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_8_v37, stage_8_v57, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_8_v60: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_8_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_8_v65, stage_8_v58, stage_8_v59)
                        stage_8_v61: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_8_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_8_v60)
                    (stage_8_v50, stage_8_v51) = pl.yield_(stage_8_v56, stage_8_v61)
                (stage_8_v38, stage_8_v39) = pl.yield_(stage_8_v50, stage_8_v51)
            stage_8_v17 = pl.tile.store(stage_8_v38, [stage_8_v8, stage_8_v9], stage_8_v19)
            stage_8_v18 = pl.tile.store(stage_8_v39, [stage_8_v8, stage_8_v9], stage_8_v20)
            (stage_8_v26, stage_8_v27) = pl.yield_(stage_8_v17, stage_8_v18)
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
        for (stage_9_v5, (stage_9_v9,)) in pl.range((19 - stage_9_v4 + 128) // 20, init_values=(temporary_25,)):
            stage_9_v6: pl.Scalar[pl.INDEX] = stage_9_v4 + stage_9_v5 * 20
            stage_9_v7: pl.Scalar[pl.INDEX] = stage_9_v6 * 256
            for (stage_9_v10, (stage_9_v17,)) in pl.range(0, 128, 128, init_values=(stage_9_v9,)):
                stage_9_v11: pl.Scalar[pl.INDEX] = pl.min(96 - stage_9_v10, 128)
                stage_9_v12 = pl.tensor.slice(temporary_22, [128, 256], [stage_9_v10, 0], [stage_9_v11, 256])
                stage_9_v13 = pl.tensor.slice(arg_wq_b, [256, 256], [0, stage_9_v7])
                stage_9_v14 = pl.tensor.matmul(stage_9_v12, stage_9_v13, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                for (stage_9_v18, (stage_9_v23,)) in pl.pipeline(256, 1024, 256, stage=2, init_values=(stage_9_v14,)):
                    stage_9_v19 = pl.tensor.slice(temporary_22, [128, 256], [stage_9_v10, stage_9_v18], [stage_9_v11, 256])
                    stage_9_v20 = pl.tensor.slice(arg_wq_b, [256, 256], [stage_9_v18, stage_9_v7])
                    stage_9_v21 = pl.tensor.matmul_acc(stage_9_v23, stage_9_v19, stage_9_v20, a_trans=False, b_trans=False)
                    stage_9_v22 = pl.yield_(stage_9_v21)
                stage_9_v15 = pl.tensor.assemble(stage_9_v17, stage_9_v22, [stage_9_v10, stage_9_v7])
                stage_9_v16 = pl.yield_(stage_9_v15)
            stage_9_v8 = pl.yield_(stage_9_v16)
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
        stage_10_v6: pl.Scalar[pl.INDEX] = block_10
        for (stage_10_v7, (stage_10_v19, stage_10_v20)) in pl.range(stage_10_v6, 128 // 4, 24, init_values=(temporary_29, temporary_30)):
            stage_10_v8: pl.Scalar[pl.INDEX] = stage_10_v7 // 16 * 64
            stage_10_v9: pl.Scalar[pl.INDEX] = stage_10_v7 % 16 * 64
            stage_10_v10: pl.Scalar[pl.INDEX] = pl.min(96 - stage_10_v8, 64)
            stage_10_v11: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_10_v12: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_10_v13: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            stage_10_v14: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_10_v13, stage_10_v10, 64)
            stage_10_v15: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            stage_10_v16: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_10_v15, stage_10_v10, 64)
            for (stage_10_v21, (stage_10_v28, stage_10_v29)) in pl.pipeline(8, stage=2, init_values=(stage_10_v14, stage_10_v16)):
                stage_10_v22: pl.Scalar[pl.INDEX] = stage_10_v21 * 512
                stage_10_v23: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_10_v10, 512])] = pl.tile.load(temporary_2, [stage_10_v8, stage_10_v22], [64, 512], [stage_10_v10, 512], target_memory=pl.Mem.Mat)
                stage_10_v24: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_cmp_wkv, [stage_10_v9, stage_10_v22], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
                stage_10_v25: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_cmp_wgate, [stage_10_v9, stage_10_v22], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
                if stage_10_v21 == 0:
                    stage_10_v30: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_10_v24)
                    stage_10_v31: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    stage_10_v32: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_10_v31, stage_10_v10, 64)
                    for (stage_10_v40, (stage_10_v62,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_10_v32,)):
                        stage_10_v41: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_10_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_10_v23, 0, stage_10_v40, [64, 256], target_memory=pl.Mem.Left)
                        stage_10_v42: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_10_v30, stage_10_v40, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_10_v43: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_10_v62, stage_10_v41, stage_10_v42, stage_10_v40 == 0)
                        stage_10_v44: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_10_v43)
                    stage_10_v33: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_10_v25)
                    stage_10_v34: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    stage_10_v35: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_10_v34, stage_10_v10, 64)
                    for (stage_10_v45, (stage_10_v63,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_10_v35,)):
                        stage_10_v46: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_10_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_10_v23, 0, stage_10_v45, [64, 256], target_memory=pl.Mem.Left)
                        stage_10_v47: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_10_v33, stage_10_v45, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_10_v48: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_10_v63, stage_10_v46, stage_10_v47, stage_10_v45 == 0)
                        stage_10_v49: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_10_v48)
                    (stage_10_v50, stage_10_v51) = pl.yield_(stage_10_v44, stage_10_v49)
                else:
                    stage_10_v36: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_10_v24)
                    for (stage_10_v52, (stage_10_v64,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_10_v28,)):
                        stage_10_v53: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_10_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_10_v23, 0, stage_10_v52, [64, 256], target_memory=pl.Mem.Left)
                        stage_10_v54: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_10_v36, stage_10_v52, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_10_v55: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_10_v64, stage_10_v53, stage_10_v54)
                        stage_10_v56: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_10_v55)
                    stage_10_v37: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_10_v25)
                    for (stage_10_v57, (stage_10_v65,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_10_v29,)):
                        stage_10_v58: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_10_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_10_v23, 0, stage_10_v57, [64, 256], target_memory=pl.Mem.Left)
                        stage_10_v59: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_10_v37, stage_10_v57, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_10_v60: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_10_v65, stage_10_v58, stage_10_v59)
                        stage_10_v61: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_10_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_10_v60)
                    (stage_10_v50, stage_10_v51) = pl.yield_(stage_10_v56, stage_10_v61)
                (stage_10_v38, stage_10_v39) = pl.yield_(stage_10_v50, stage_10_v51)
            stage_10_v17 = pl.tile.store(stage_10_v38, [stage_10_v8, stage_10_v9], stage_10_v19)
            stage_10_v18 = pl.tile.store(stage_10_v39, [stage_10_v8, stage_10_v9], stage_10_v20)
            (stage_10_v26, stage_10_v27) = pl.yield_(stage_10_v17, stage_10_v18)
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
            for (stage_11_v10, (stage_11_v18,)) in pl.range(2, init_values=(stage_11_v9,)):
                stage_11_v11: pl.Scalar[pl.INDEX] = stage_11_v10 * 64
                stage_11_v12: pl.Scalar[pl.INDEX] = pl.min(96 - stage_11_v11, 64)
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
        stage_12_v4: pl.Scalar[pl.INDEX] = block_12
        for (stage_12_v5, (stage_12_v9,)) in pl.range((19 - stage_12_v4 + 128) // 20, init_values=(temporary_25,)):
            stage_12_v6: pl.Scalar[pl.INDEX] = stage_12_v4 + stage_12_v5 * 20
            stage_12_v7: pl.Scalar[pl.INDEX] = stage_12_v6 * 256
            for (stage_12_v10, (stage_12_v17,)) in pl.range(0, 128, 128, init_values=(stage_12_v9,)):
                stage_12_v11: pl.Scalar[pl.INDEX] = pl.min(96 - stage_12_v10, 128)
                stage_12_v12 = pl.tensor.slice(temporary_22, [128, 256], [stage_12_v10, 0], [stage_12_v11, 256])
                stage_12_v13 = pl.tensor.slice(arg_wq_b, [256, 256], [0, stage_12_v7])
                stage_12_v14 = pl.tensor.matmul(stage_12_v12, stage_12_v13, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                for (stage_12_v18, (stage_12_v23,)) in pl.pipeline(256, 1024, 256, stage=2, init_values=(stage_12_v14,)):
                    stage_12_v19 = pl.tensor.slice(temporary_22, [128, 256], [stage_12_v10, stage_12_v18], [stage_12_v11, 256])
                    stage_12_v20 = pl.tensor.slice(arg_wq_b, [256, 256], [stage_12_v18, stage_12_v7])
                    stage_12_v21 = pl.tensor.matmul_acc(stage_12_v23, stage_12_v19, stage_12_v20, a_trans=False, b_trans=False)
                    stage_12_v22 = pl.yield_(stage_12_v21)
                stage_12_v15 = pl.tensor.assemble(stage_12_v17, stage_12_v22, [stage_12_v10, stage_12_v7])
                stage_12_v16 = pl.yield_(stage_12_v15)
            stage_12_v8 = pl.yield_(stage_12_v16)
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
        for (stage_13_v5, (stage_13_v9,)) in pl.range((19 - stage_13_v4 + 128) // 20, init_values=(temporary_25,)):
            stage_13_v6: pl.Scalar[pl.INDEX] = stage_13_v4 + stage_13_v5 * 20
            stage_13_v7: pl.Scalar[pl.INDEX] = stage_13_v6 * 256
            for (stage_13_v10, (stage_13_v17,)) in pl.range(0, 128, 128, init_values=(stage_13_v9,)):
                stage_13_v11: pl.Scalar[pl.INDEX] = pl.min(96 - stage_13_v10, 128)
                stage_13_v12 = pl.tensor.slice(temporary_22, [128, 256], [stage_13_v10, 0], [stage_13_v11, 256])
                stage_13_v13 = pl.tensor.slice(arg_wq_b, [256, 256], [0, stage_13_v7])
                stage_13_v14 = pl.tensor.matmul(stage_13_v12, stage_13_v13, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                for (stage_13_v18, (stage_13_v23,)) in pl.pipeline(256, 1024, 256, stage=2, init_values=(stage_13_v14,)):
                    stage_13_v19 = pl.tensor.slice(temporary_22, [128, 256], [stage_13_v10, stage_13_v18], [stage_13_v11, 256])
                    stage_13_v20 = pl.tensor.slice(arg_wq_b, [256, 256], [stage_13_v18, stage_13_v7])
                    stage_13_v21 = pl.tensor.matmul_acc(stage_13_v23, stage_13_v19, stage_13_v20, a_trans=False, b_trans=False)
                    stage_13_v22 = pl.yield_(stage_13_v21)
                stage_13_v15 = pl.tensor.assemble(stage_13_v17, stage_13_v22, [stage_13_v10, stage_13_v7])
                stage_13_v16 = pl.yield_(stage_13_v15)
            stage_13_v8 = pl.yield_(stage_13_v16)
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
        for (stage_14_v5, (stage_14_v9,)) in pl.range(stage_14_v4, 32, 24, init_values=(temporary_42,)):
            stage_14_v6: pl.Scalar[pl.INDEX] = stage_14_v5 * 256
            stage_14_v7: pl.Tile[[1024, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_idx_wq_b, [0, stage_14_v6], [1024, 256], [1024, 256], target_memory=pl.Mem.Mat)
            for (stage_14_v10, (stage_14_v18,)) in pl.range(2, init_values=(stage_14_v9,)):
                stage_14_v11: pl.Scalar[pl.INDEX] = stage_14_v10 * 64
                stage_14_v12: pl.Scalar[pl.INDEX] = pl.min(96 - stage_14_v11, 64)
                stage_14_v13: pl.Tile[[64, 1024], pl.INT8, pl.Mem.Mat, pl.TileView(valid_shape=[stage_14_v12, 1024])] = pl.tile.load(temporary_18, [stage_14_v11, 0], [64, 1024], [stage_14_v12, 1024], target_memory=pl.Mem.Mat)
                stage_14_v14: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc, compact=True)
                stage_14_v15: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_14_v12, 256], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_14_v14, stage_14_v12, 256)
                for (stage_14_v19, (stage_14_v24,)) in pl.pipeline(0, 1024, 128, stage=2, init_values=(stage_14_v15,)):
                    stage_14_v20: pl.Tile[[64, 128], pl.INT8, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_14_v12, pl.const(64, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_14_v13, 0, stage_14_v19, [64, 128], target_memory=pl.Mem.Left)
                    stage_14_v21: pl.Tile[[128, 256], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_14_v7, stage_14_v19, 0, [128, 256], target_memory=pl.Mem.Right)
                    stage_14_v22: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_14_v12, 256], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_14_v24, stage_14_v20, stage_14_v21, stage_14_v19 == 0)
                    stage_14_v23: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_14_v12, 256], compact=pl.CompactMode.normal)] = pl.yield_(stage_14_v22)
                stage_14_v16 = pl.tile.store(stage_14_v23, [stage_14_v11, stage_14_v6], stage_14_v18)
                stage_14_v17 = pl.yield_(stage_14_v16)
            stage_14_v8 = pl.yield_(stage_14_v17)
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
        stage_15_v4: pl.Scalar[pl.INDEX] = block_15
        stage_15_v5: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_hadamard_idx, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat)
        for (stage_15_v6, (stage_15_v14,)) in pl.range(stage_15_v4, 6144 // 64, 24, init_values=(temporary_43,)):
            stage_15_v7: pl.Scalar[pl.INDEX] = stage_15_v6 * 64
            stage_15_v8: pl.Tile[[64, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(temporary_41, [stage_15_v7, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat)
            stage_15_v9: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(stage_15_v8, target_memory=pl.Mem.Left)
            stage_15_v10: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_15_v5, target_memory=pl.Mem.Right)
            stage_15_v11: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_15_v9, stage_15_v10)
            stage_15_v12 = pl.tile.store(stage_15_v11, [stage_15_v7, 0], stage_15_v14)
            stage_15_v13 = pl.yield_(stage_15_v12)
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
        stage_16_v4: pl.Scalar[pl.INDEX] = block_16
        stage_16_v5: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_hadamard_idx, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat)
        for (stage_16_v6, (stage_16_v14,)) in pl.range(stage_16_v4, 6144 // 64, 24, init_values=(temporary_43,)):
            stage_16_v7: pl.Scalar[pl.INDEX] = stage_16_v6 * 64
            stage_16_v8: pl.Tile[[64, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(temporary_41, [stage_16_v7, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat)
            stage_16_v9: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(stage_16_v8, target_memory=pl.Mem.Left)
            stage_16_v10: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_16_v5, target_memory=pl.Mem.Right)
            stage_16_v11: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_16_v9, stage_16_v10)
            stage_16_v12 = pl.tile.store(stage_16_v11, [stage_16_v7, 0], stage_16_v14)
            stage_16_v13 = pl.yield_(stage_16_v12)
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
        for (stage_17_v3, (stage_17_v6,)) in pl.range(0, 128, 64, init_values=(temporary_38,)):
            stage_17_v4: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_hadamard_idx, [0, stage_17_v3], [128, 64], [128, 64], target_memory=pl.Mem.Mat)
            for (stage_17_v7, (stage_17_v16,)) in pl.range(2, init_values=(stage_17_v6,)):
                stage_17_v8: pl.Scalar[pl.INDEX] = stage_17_v7 * 16
                stage_17_v9: pl.Scalar[pl.INDEX] = pl.min(32 - stage_17_v8, 16)
                stage_17_v10: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_17_v9, 128])] = pl.tile.load(temporary_33, [stage_17_v8, 0], [16, 128], [stage_17_v9, 128], target_memory=pl.Mem.Mat)
                stage_17_v11: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[stage_17_v9, 128])] = pl.tile.move(stage_17_v10, target_memory=pl.Mem.Left)
                stage_17_v12: pl.Tile[[128, 64], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_17_v4, target_memory=pl.Mem.Right)
                stage_17_v13: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_17_v9, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul(stage_17_v11, stage_17_v12)
                stage_17_v14 = pl.tile.store(stage_17_v13, [stage_17_v8, stage_17_v3], stage_17_v16)
                stage_17_v15 = pl.yield_(stage_17_v14)
            stage_17_v5 = pl.yield_(stage_17_v15)
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
        stage_18_v12: pl.Scalar[pl.INDEX] = block_18
        stage_18_v13: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        stage_18_v14: pl.Scalar[pl.INDEX] = 0
        for (stage_18_v25, (stage_18_v39,)) in pl.range(stage_18_v13 // 6, init_values=(stage_18_v14,)):
            stage_18_v26: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_18_v25])
            stage_18_v27: pl.Scalar[pl.INDEX] = pl.max(stage_18_v39, pl.cast(stage_18_v26, pl.INDEX) // 4)
            stage_18_v28: pl.Scalar[pl.INDEX] = pl.yield_(stage_18_v27)
        stage_18_v15: pl.Scalar[pl.INDEX] = pl.min(stage_18_v28, 262144)
        stage_18_v16: pl.Scalar[pl.INDEX] = (stage_18_v15 + 1023) // 1024
        stage_18_v17: pl.Scalar[pl.INDEX] = pl.max((stage_18_v15 + 8191) // 8192, 1)
        stage_18_v18: pl.Scalar[pl.INDEX] = 8
        stage_18_v19: pl.Scalar[pl.INDEX] = 0
        stage_18_v20: pl.Scalar[pl.INDEX] = 1
        if stage_18_v13 // 6 * 3 == 48:
            stage_18_v29: pl.Scalar[pl.INDEX] = 3
            stage_18_v30: pl.Scalar[pl.INDEX] = pl.yield_(stage_18_v29)
        else:
            if stage_18_v13 // 6 * 3 == 24:
                stage_18_v40: pl.Scalar[pl.INDEX] = 3
                stage_18_v41: pl.Scalar[pl.INDEX] = pl.yield_(stage_18_v40)
            else:
                if stage_18_v13 // 6 * 6 == 24:
                    stage_18_v62: pl.Scalar[pl.INDEX] = 6
                    stage_18_v63: pl.Scalar[pl.INDEX] = pl.yield_(stage_18_v62)
                else:
                    stage_18_v63: pl.Scalar[pl.INDEX] = pl.yield_(stage_18_v20)
                stage_18_v41: pl.Scalar[pl.INDEX] = pl.yield_(stage_18_v63)
            stage_18_v30: pl.Scalar[pl.INDEX] = pl.yield_(stage_18_v41)
        if 1 < stage_18_v30:
            stage_18_v31: pl.Scalar[pl.INDEX] = (stage_18_v17 + stage_18_v30 - 1) // stage_18_v30 * stage_18_v30
            if stage_18_v31 <= 32:
                stage_18_v42: pl.Scalar[pl.INDEX] = stage_18_v31
                stage_18_v43: pl.Scalar[pl.INDEX] = pl.max(stage_18_v16 // stage_18_v31, 1)
                stage_18_v44: pl.Scalar[pl.INDEX] = stage_18_v16 % stage_18_v31
                (stage_18_v64, stage_18_v65, stage_18_v66) = pl.yield_(stage_18_v44, stage_18_v42, stage_18_v43)
            else:
                (stage_18_v64, stage_18_v65, stage_18_v66) = pl.yield_(stage_18_v19, stage_18_v17, stage_18_v18)
            (stage_18_v45, stage_18_v46, stage_18_v47) = pl.yield_(stage_18_v64, stage_18_v65, stage_18_v66)
        else:
            (stage_18_v45, stage_18_v46, stage_18_v47) = pl.yield_(stage_18_v19, stage_18_v17, stage_18_v18)
        stage_18_v21: pl.Scalar[pl.INDEX] = stage_18_v46
        stage_18_v22: pl.Scalar[pl.INDEX] = stage_18_v47
        stage_18_v23: pl.Scalar[pl.INDEX] = stage_18_v45
        pl.system.set_ffts(temporary_50)
        for stage_18_v24 in pl.range(stage_18_v12, stage_18_v13 // 6 * stage_18_v21, 24):
            if stage_18_v13 < 144:
                stage_18_v48: pl.Scalar[pl.INDEX] = stage_18_v24 % (stage_18_v13 // 6) * 6
                stage_18_v49: pl.Scalar[pl.INDEX] = stage_18_v24 // (stage_18_v13 // 6)
                (stage_18_v67, stage_18_v68) = pl.yield_(stage_18_v49, stage_18_v48)
            else:
                stage_18_v50: pl.Scalar[pl.INDEX] = stage_18_v24 // stage_18_v21 * 6
                stage_18_v51: pl.Scalar[pl.INDEX] = stage_18_v24 % stage_18_v21
                (stage_18_v67, stage_18_v68) = pl.yield_(stage_18_v51, stage_18_v50)
            stage_18_v32: pl.Scalar[pl.INDEX] = stage_18_v68 // 6
            stage_18_v33: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_18_v68 + 5])
            stage_18_v34: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_18_v32])
            stage_18_v35: pl.Scalar[pl.INDEX] = pl.cast(stage_18_v34, pl.INDEX) // 4
            stage_18_v36: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_18_v35, (stage_18_v33 + 1) // 4), 262144), 0)
            stage_18_v37: pl.Scalar[pl.INDEX] = (stage_18_v67 * stage_18_v22 + pl.min(stage_18_v67, stage_18_v23)) * 1024
            stage_18_v38: pl.Scalar[pl.INDEX] = (stage_18_v22 + pl.cast(stage_18_v67 < stage_18_v23, pl.INDEX)) * 1024
            if stage_18_v37 < stage_18_v36:
                stage_18_v52: pl.Scalar[pl.INDEX] = pl.min(stage_18_v38, stage_18_v36 - stage_18_v37)
                stage_18_v53: pl.Scalar[pl.INDEX] = (stage_18_v52 + 1023) // 1024
                stage_18_v54: pl.Scalar[pl.INDEX] = pl.min(stage_18_v53 * 512, 4096)
                stage_18_v55: pl.Scalar[pl.INDEX] = (stage_18_v54 + 511) // 512
                stage_18_v56: pl.Tile[[384, 128], pl.INT8, pl.Mem.Mat] = pl.tile.load(temporary_39, [stage_18_v68 * 64, 0], [384, 128], [384, 128], target_memory=pl.Mem.Mat)
                stage_18_v57: pl.Tile[[16, 384], pl.FP16, pl.Mem.Mat] = pl.tile.load(temporary_48, [stage_18_v68 // 6 * 16, 0], [16, 384], [16, 384], target_memory=pl.Mem.Mat)
                stage_18_v58: pl.Tile[[384, 128], pl.INT8, pl.Mem.Left] = pl.tile.move(stage_18_v56, target_memory=pl.Mem.Left)
                stage_18_v59: pl.Tile[[16, 384], pl.FP16, pl.Mem.Left] = pl.tile.move(stage_18_v57, target_memory=pl.Mem.Left)
                for stage_18_v60 in pl.range(stage_18_v55):
                    if 2 <= stage_18_v60:
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    stage_18_v69: pl.Scalar[pl.INDEX] = stage_18_v60 * 1024
                    stage_18_v70: pl.Scalar[pl.INDEX] = stage_18_v12 * 12 + stage_18_v60 % 2 * 6
                    stage_18_v71: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_18_v72: pl.Scalar[pl.INDEX] = 0
                    stage_18_v73: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v72
                    stage_18_v74: pl.Scalar[pl.INDEX] = pl.min(stage_18_v73 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v75: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v74])
                    stage_18_v76: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v75, pl.INDEX), 0)
                    stage_18_v77: pl.Scalar[pl.INDEX] = stage_18_v76 * 4160
                    if stage_18_v77 % 128 == 0:
                        stage_18_v469: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v71, view_98, [0, 0], [stage_18_v77 // 128, 0], [32, 128], transpose=False)
                        stage_18_v470: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v469)
                    else:
                        stage_18_v471: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v71, view_99, [0, 0], [stage_18_v77 // 128, 0], [32, 128], transpose=False)
                        stage_18_v470: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v471)
                    stage_18_v78: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v72 + 32
                    stage_18_v79: pl.Scalar[pl.INDEX] = pl.min(stage_18_v78 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v80: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v79])
                    stage_18_v81: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v80, pl.INDEX), 0)
                    stage_18_v82: pl.Scalar[pl.INDEX] = stage_18_v81 * 4160
                    if stage_18_v82 % 128 == 0:
                        stage_18_v472: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v470, view_98, [32, 0], [stage_18_v82 // 128, 0], [32, 128], transpose=False)
                        stage_18_v473: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v472)
                    else:
                        stage_18_v474: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v470, view_99, [32, 0], [stage_18_v82 // 128, 0], [32, 128], transpose=False)
                        stage_18_v473: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v474)
                    stage_18_v83: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v473)
                    stage_18_v84: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v83, 0, 0, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v85: pl.Scalar[pl.INDEX] = 64
                    stage_18_v86: pl.Scalar[pl.INDEX] = 64
                    stage_18_v87: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v85
                    stage_18_v88: pl.Scalar[pl.INDEX] = pl.min(stage_18_v87 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v89: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v88])
                    stage_18_v90: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v89, pl.INDEX), 0)
                    stage_18_v91: pl.Scalar[pl.INDEX] = stage_18_v90 * 4160
                    if stage_18_v91 % 128 == 0:
                        stage_18_v475: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v473, view_98, [stage_18_v86, 0], [stage_18_v91 // 128, 0], [32, 128], transpose=False)
                        stage_18_v476: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v475)
                    else:
                        stage_18_v477: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v473, view_99, [stage_18_v86, 0], [stage_18_v91 // 128, 0], [32, 128], transpose=False)
                        stage_18_v476: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v477)
                    stage_18_v92: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v85 + 32
                    stage_18_v93: pl.Scalar[pl.INDEX] = pl.min(stage_18_v92 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v94: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v93])
                    stage_18_v95: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v94, pl.INDEX), 0)
                    stage_18_v96: pl.Scalar[pl.INDEX] = stage_18_v95 * 4160
                    if stage_18_v96 % 128 == 0:
                        stage_18_v478: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v476, view_98, [stage_18_v86 + 32, 0], [stage_18_v96 // 128, 0], [32, 128], transpose=False)
                        stage_18_v479: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v478)
                    else:
                        stage_18_v480: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v476, view_99, [stage_18_v86 + 32, 0], [stage_18_v96 // 128, 0], [32, 128], transpose=False)
                        stage_18_v479: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v480)
                    stage_18_v97: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v479)
                    stage_18_v98: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v97, 0, stage_18_v86, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v99: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v84
                    stage_18_v100: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v101: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v99)
                    stage_18_v102: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v100, stage_18_v101, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v103: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v102
                    stage_18_v104: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v98
                    stage_18_v105: pl.Scalar[pl.INDEX] = 64
                    stage_18_v106: pl.Scalar[pl.INDEX] = 128
                    stage_18_v107: pl.Scalar[pl.INDEX] = 0
                    stage_18_v108: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v106
                    stage_18_v109: pl.Scalar[pl.INDEX] = pl.min(stage_18_v108 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v110: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v109])
                    stage_18_v111: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v110, pl.INDEX), 0)
                    stage_18_v112: pl.Scalar[pl.INDEX] = stage_18_v111 * 4160
                    if stage_18_v112 % 128 == 0:
                        stage_18_v481: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v479, view_98, [stage_18_v107, 0], [stage_18_v112 // 128, 0], [32, 128], transpose=False)
                        stage_18_v482: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v481)
                    else:
                        stage_18_v483: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v479, view_99, [stage_18_v107, 0], [stage_18_v112 // 128, 0], [32, 128], transpose=False)
                        stage_18_v482: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v483)
                    stage_18_v113: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v106 + 32
                    stage_18_v114: pl.Scalar[pl.INDEX] = pl.min(stage_18_v113 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v115: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v114])
                    stage_18_v116: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v115, pl.INDEX), 0)
                    stage_18_v117: pl.Scalar[pl.INDEX] = stage_18_v116 * 4160
                    if stage_18_v117 % 128 == 0:
                        stage_18_v484: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v482, view_98, [stage_18_v107 + 32, 0], [stage_18_v117 // 128, 0], [32, 128], transpose=False)
                        stage_18_v485: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v484)
                    else:
                        stage_18_v486: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v482, view_99, [stage_18_v107 + 32, 0], [stage_18_v117 // 128, 0], [32, 128], transpose=False)
                        stage_18_v485: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v486)
                    stage_18_v118: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v485)
                    stage_18_v119: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v118, 0, stage_18_v107, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v120: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v104
                    stage_18_v121: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v122: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v103, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v123: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v120)
                    stage_18_v124: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v122)
                    stage_18_v125: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v121, stage_18_v123, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v126: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v124, 6, 64)
                    stage_18_v127 = pl.tile.store(stage_18_v126, [stage_18_v70, stage_18_v105 - 64], temporary_49)
                    stage_18_v128: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v125
                    stage_18_v129: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v119
                    stage_18_v130: pl.Scalar[pl.INDEX] = 128
                    stage_18_v131: pl.Scalar[pl.INDEX] = 192
                    stage_18_v132: pl.Scalar[pl.INDEX] = 64
                    stage_18_v133: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v131
                    stage_18_v134: pl.Scalar[pl.INDEX] = pl.min(stage_18_v133 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v135: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v134])
                    stage_18_v136: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v135, pl.INDEX), 0)
                    stage_18_v137: pl.Scalar[pl.INDEX] = stage_18_v136 * 4160
                    if stage_18_v137 % 128 == 0:
                        stage_18_v487: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v485, view_98, [stage_18_v132, 0], [stage_18_v137 // 128, 0], [32, 128], transpose=False)
                        stage_18_v488: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v487)
                    else:
                        stage_18_v489: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v485, view_99, [stage_18_v132, 0], [stage_18_v137 // 128, 0], [32, 128], transpose=False)
                        stage_18_v488: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v489)
                    stage_18_v138: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v131 + 32
                    stage_18_v139: pl.Scalar[pl.INDEX] = pl.min(stage_18_v138 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v140: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v139])
                    stage_18_v141: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v140, pl.INDEX), 0)
                    stage_18_v142: pl.Scalar[pl.INDEX] = stage_18_v141 * 4160
                    if stage_18_v142 % 128 == 0:
                        stage_18_v490: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v488, view_98, [stage_18_v132 + 32, 0], [stage_18_v142 // 128, 0], [32, 128], transpose=False)
                        stage_18_v491: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v490)
                    else:
                        stage_18_v492: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v488, view_99, [stage_18_v132 + 32, 0], [stage_18_v142 // 128, 0], [32, 128], transpose=False)
                        stage_18_v491: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v492)
                    stage_18_v143: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v491)
                    stage_18_v144: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v143, 0, stage_18_v132, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v145: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v129
                    stage_18_v146: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v147: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v128, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v148: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v145)
                    stage_18_v149: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v147)
                    stage_18_v150: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v146, stage_18_v148, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v151: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v149, 6, 64)
                    stage_18_v152 = pl.tile.store(stage_18_v151, [stage_18_v70, stage_18_v130 - 64], temporary_49)
                    stage_18_v153: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v150
                    stage_18_v154: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v144
                    stage_18_v155: pl.Scalar[pl.INDEX] = 192
                    stage_18_v156: pl.Scalar[pl.INDEX] = 256
                    stage_18_v157: pl.Scalar[pl.INDEX] = 0
                    stage_18_v158: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v156
                    stage_18_v159: pl.Scalar[pl.INDEX] = pl.min(stage_18_v158 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v160: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v159])
                    stage_18_v161: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v160, pl.INDEX), 0)
                    stage_18_v162: pl.Scalar[pl.INDEX] = stage_18_v161 * 4160
                    if stage_18_v162 % 128 == 0:
                        stage_18_v493: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v491, view_98, [stage_18_v157, 0], [stage_18_v162 // 128, 0], [32, 128], transpose=False)
                        stage_18_v494: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v493)
                    else:
                        stage_18_v495: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v491, view_99, [stage_18_v157, 0], [stage_18_v162 // 128, 0], [32, 128], transpose=False)
                        stage_18_v494: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v495)
                    stage_18_v163: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v156 + 32
                    stage_18_v164: pl.Scalar[pl.INDEX] = pl.min(stage_18_v163 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v165: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v164])
                    stage_18_v166: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v165, pl.INDEX), 0)
                    stage_18_v167: pl.Scalar[pl.INDEX] = stage_18_v166 * 4160
                    if stage_18_v167 % 128 == 0:
                        stage_18_v496: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v494, view_98, [stage_18_v157 + 32, 0], [stage_18_v167 // 128, 0], [32, 128], transpose=False)
                        stage_18_v497: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v496)
                    else:
                        stage_18_v498: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v494, view_99, [stage_18_v157 + 32, 0], [stage_18_v167 // 128, 0], [32, 128], transpose=False)
                        stage_18_v497: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v498)
                    stage_18_v168: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v497)
                    stage_18_v169: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v168, 0, stage_18_v157, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v170: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v154
                    stage_18_v171: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v172: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v153, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v173: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v170)
                    stage_18_v174: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v172)
                    stage_18_v175: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v171, stage_18_v173, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v176: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v174, 6, 64)
                    stage_18_v177 = pl.tile.store(stage_18_v176, [stage_18_v70, stage_18_v155 - 64], temporary_49)
                    stage_18_v178: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v175
                    stage_18_v179: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v169
                    stage_18_v180: pl.Scalar[pl.INDEX] = 256
                    stage_18_v181: pl.Scalar[pl.INDEX] = 320
                    stage_18_v182: pl.Scalar[pl.INDEX] = 64
                    stage_18_v183: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v181
                    stage_18_v184: pl.Scalar[pl.INDEX] = pl.min(stage_18_v183 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v185: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v184])
                    stage_18_v186: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v185, pl.INDEX), 0)
                    stage_18_v187: pl.Scalar[pl.INDEX] = stage_18_v186 * 4160
                    if stage_18_v187 % 128 == 0:
                        stage_18_v499: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v497, view_98, [stage_18_v182, 0], [stage_18_v187 // 128, 0], [32, 128], transpose=False)
                        stage_18_v500: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v499)
                    else:
                        stage_18_v501: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v497, view_99, [stage_18_v182, 0], [stage_18_v187 // 128, 0], [32, 128], transpose=False)
                        stage_18_v500: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v501)
                    stage_18_v188: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v181 + 32
                    stage_18_v189: pl.Scalar[pl.INDEX] = pl.min(stage_18_v188 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v190: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v189])
                    stage_18_v191: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v190, pl.INDEX), 0)
                    stage_18_v192: pl.Scalar[pl.INDEX] = stage_18_v191 * 4160
                    if stage_18_v192 % 128 == 0:
                        stage_18_v502: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v500, view_98, [stage_18_v182 + 32, 0], [stage_18_v192 // 128, 0], [32, 128], transpose=False)
                        stage_18_v503: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v502)
                    else:
                        stage_18_v504: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v500, view_99, [stage_18_v182 + 32, 0], [stage_18_v192 // 128, 0], [32, 128], transpose=False)
                        stage_18_v503: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v504)
                    stage_18_v193: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v503)
                    stage_18_v194: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v193, 0, stage_18_v182, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v195: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v179
                    stage_18_v196: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v197: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v178, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v198: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v195)
                    stage_18_v199: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v197)
                    stage_18_v200: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v196, stage_18_v198, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v201: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v199, 6, 64)
                    stage_18_v202 = pl.tile.store(stage_18_v201, [stage_18_v70, stage_18_v180 - 64], temporary_49)
                    stage_18_v203: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v200
                    stage_18_v204: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v194
                    stage_18_v205: pl.Scalar[pl.INDEX] = 320
                    stage_18_v206: pl.Scalar[pl.INDEX] = 384
                    stage_18_v207: pl.Scalar[pl.INDEX] = 0
                    stage_18_v208: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v206
                    stage_18_v209: pl.Scalar[pl.INDEX] = pl.min(stage_18_v208 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v210: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v209])
                    stage_18_v211: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v210, pl.INDEX), 0)
                    stage_18_v212: pl.Scalar[pl.INDEX] = stage_18_v211 * 4160
                    if stage_18_v212 % 128 == 0:
                        stage_18_v505: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v503, view_98, [stage_18_v207, 0], [stage_18_v212 // 128, 0], [32, 128], transpose=False)
                        stage_18_v506: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v505)
                    else:
                        stage_18_v507: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v503, view_99, [stage_18_v207, 0], [stage_18_v212 // 128, 0], [32, 128], transpose=False)
                        stage_18_v506: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v507)
                    stage_18_v213: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v206 + 32
                    stage_18_v214: pl.Scalar[pl.INDEX] = pl.min(stage_18_v213 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v215: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v214])
                    stage_18_v216: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v215, pl.INDEX), 0)
                    stage_18_v217: pl.Scalar[pl.INDEX] = stage_18_v216 * 4160
                    if stage_18_v217 % 128 == 0:
                        stage_18_v508: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v506, view_98, [stage_18_v207 + 32, 0], [stage_18_v217 // 128, 0], [32, 128], transpose=False)
                        stage_18_v509: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v508)
                    else:
                        stage_18_v510: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v506, view_99, [stage_18_v207 + 32, 0], [stage_18_v217 // 128, 0], [32, 128], transpose=False)
                        stage_18_v509: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v510)
                    stage_18_v218: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v509)
                    stage_18_v219: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v218, 0, stage_18_v207, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v220: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v204
                    stage_18_v221: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v222: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v203, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v223: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v220)
                    stage_18_v224: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v222)
                    stage_18_v225: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v221, stage_18_v223, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v226: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v224, 6, 64)
                    stage_18_v227 = pl.tile.store(stage_18_v226, [stage_18_v70, stage_18_v205 - 64], temporary_49)
                    stage_18_v228: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v225
                    stage_18_v229: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v219
                    stage_18_v230: pl.Scalar[pl.INDEX] = 384
                    stage_18_v231: pl.Scalar[pl.INDEX] = 448
                    stage_18_v232: pl.Scalar[pl.INDEX] = 64
                    stage_18_v233: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v231
                    stage_18_v234: pl.Scalar[pl.INDEX] = pl.min(stage_18_v233 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v235: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v234])
                    stage_18_v236: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v235, pl.INDEX), 0)
                    stage_18_v237: pl.Scalar[pl.INDEX] = stage_18_v236 * 4160
                    if stage_18_v237 % 128 == 0:
                        stage_18_v511: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v509, view_98, [stage_18_v232, 0], [stage_18_v237 // 128, 0], [32, 128], transpose=False)
                        stage_18_v512: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v511)
                    else:
                        stage_18_v513: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v509, view_99, [stage_18_v232, 0], [stage_18_v237 // 128, 0], [32, 128], transpose=False)
                        stage_18_v512: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v513)
                    stage_18_v238: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v231 + 32
                    stage_18_v239: pl.Scalar[pl.INDEX] = pl.min(stage_18_v238 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v240: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v239])
                    stage_18_v241: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v240, pl.INDEX), 0)
                    stage_18_v242: pl.Scalar[pl.INDEX] = stage_18_v241 * 4160
                    if stage_18_v242 % 128 == 0:
                        stage_18_v514: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v512, view_98, [stage_18_v232 + 32, 0], [stage_18_v242 // 128, 0], [32, 128], transpose=False)
                        stage_18_v515: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v514)
                    else:
                        stage_18_v516: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v512, view_99, [stage_18_v232 + 32, 0], [stage_18_v242 // 128, 0], [32, 128], transpose=False)
                        stage_18_v515: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v516)
                    stage_18_v243: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v515)
                    stage_18_v244: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v243, 0, stage_18_v232, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v245: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v229
                    stage_18_v246: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v247: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v228, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v248: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v245)
                    stage_18_v249: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v247)
                    stage_18_v250: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v246, stage_18_v248, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v251: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v249, 6, 64)
                    stage_18_v252 = pl.tile.store(stage_18_v251, [stage_18_v70, stage_18_v230 - 64], temporary_49)
                    stage_18_v253: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v250
                    stage_18_v254: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v244
                    stage_18_v255: pl.Scalar[pl.INDEX] = 448
                    stage_18_v256: pl.Scalar[pl.INDEX] = 512
                    stage_18_v257: pl.Scalar[pl.INDEX] = 0
                    stage_18_v258: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v256
                    stage_18_v259: pl.Scalar[pl.INDEX] = pl.min(stage_18_v258 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v260: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v259])
                    stage_18_v261: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v260, pl.INDEX), 0)
                    stage_18_v262: pl.Scalar[pl.INDEX] = stage_18_v261 * 4160
                    if stage_18_v262 % 128 == 0:
                        stage_18_v517: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v515, view_98, [stage_18_v257, 0], [stage_18_v262 // 128, 0], [32, 128], transpose=False)
                        stage_18_v518: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v517)
                    else:
                        stage_18_v519: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v515, view_99, [stage_18_v257, 0], [stage_18_v262 // 128, 0], [32, 128], transpose=False)
                        stage_18_v518: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v519)
                    stage_18_v263: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v256 + 32
                    stage_18_v264: pl.Scalar[pl.INDEX] = pl.min(stage_18_v263 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v265: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v264])
                    stage_18_v266: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v265, pl.INDEX), 0)
                    stage_18_v267: pl.Scalar[pl.INDEX] = stage_18_v266 * 4160
                    if stage_18_v267 % 128 == 0:
                        stage_18_v520: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v518, view_98, [stage_18_v257 + 32, 0], [stage_18_v267 // 128, 0], [32, 128], transpose=False)
                        stage_18_v521: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v520)
                    else:
                        stage_18_v522: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v518, view_99, [stage_18_v257 + 32, 0], [stage_18_v267 // 128, 0], [32, 128], transpose=False)
                        stage_18_v521: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v522)
                    stage_18_v268: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v521)
                    stage_18_v269: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v268, 0, stage_18_v257, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v270: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v254
                    stage_18_v271: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v272: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v253, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v273: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v270)
                    stage_18_v274: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v272)
                    stage_18_v275: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v271, stage_18_v273, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v276: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v274, 6, 64)
                    stage_18_v277 = pl.tile.store(stage_18_v276, [stage_18_v70, stage_18_v255 - 64], temporary_49)
                    stage_18_v278: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v275
                    stage_18_v279: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v269
                    stage_18_v280: pl.Scalar[pl.INDEX] = 512
                    stage_18_v281: pl.Scalar[pl.INDEX] = 576
                    stage_18_v282: pl.Scalar[pl.INDEX] = 64
                    stage_18_v283: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v281
                    stage_18_v284: pl.Scalar[pl.INDEX] = pl.min(stage_18_v283 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v285: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v284])
                    stage_18_v286: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v285, pl.INDEX), 0)
                    stage_18_v287: pl.Scalar[pl.INDEX] = stage_18_v286 * 4160
                    if stage_18_v287 % 128 == 0:
                        stage_18_v523: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v521, view_98, [stage_18_v282, 0], [stage_18_v287 // 128, 0], [32, 128], transpose=False)
                        stage_18_v524: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v523)
                    else:
                        stage_18_v525: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v521, view_99, [stage_18_v282, 0], [stage_18_v287 // 128, 0], [32, 128], transpose=False)
                        stage_18_v524: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v525)
                    stage_18_v288: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v281 + 32
                    stage_18_v289: pl.Scalar[pl.INDEX] = pl.min(stage_18_v288 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v290: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v289])
                    stage_18_v291: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v290, pl.INDEX), 0)
                    stage_18_v292: pl.Scalar[pl.INDEX] = stage_18_v291 * 4160
                    if stage_18_v292 % 128 == 0:
                        stage_18_v526: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v524, view_98, [stage_18_v282 + 32, 0], [stage_18_v292 // 128, 0], [32, 128], transpose=False)
                        stage_18_v527: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v526)
                    else:
                        stage_18_v528: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v524, view_99, [stage_18_v282 + 32, 0], [stage_18_v292 // 128, 0], [32, 128], transpose=False)
                        stage_18_v527: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v528)
                    stage_18_v293: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v527)
                    stage_18_v294: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v293, 0, stage_18_v282, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v295: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v279
                    stage_18_v296: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v297: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v278, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v298: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v295)
                    stage_18_v299: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v297)
                    stage_18_v300: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v296, stage_18_v298, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v301: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v299, 6, 64)
                    stage_18_v302 = pl.tile.store(stage_18_v301, [stage_18_v70, stage_18_v280 - 64], temporary_49)
                    stage_18_v303: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v300
                    stage_18_v304: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v294
                    stage_18_v305: pl.Scalar[pl.INDEX] = 576
                    stage_18_v306: pl.Scalar[pl.INDEX] = 640
                    stage_18_v307: pl.Scalar[pl.INDEX] = 0
                    stage_18_v308: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v306
                    stage_18_v309: pl.Scalar[pl.INDEX] = pl.min(stage_18_v308 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v310: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v309])
                    stage_18_v311: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v310, pl.INDEX), 0)
                    stage_18_v312: pl.Scalar[pl.INDEX] = stage_18_v311 * 4160
                    if stage_18_v312 % 128 == 0:
                        stage_18_v529: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v527, view_98, [stage_18_v307, 0], [stage_18_v312 // 128, 0], [32, 128], transpose=False)
                        stage_18_v530: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v529)
                    else:
                        stage_18_v531: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v527, view_99, [stage_18_v307, 0], [stage_18_v312 // 128, 0], [32, 128], transpose=False)
                        stage_18_v530: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v531)
                    stage_18_v313: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v306 + 32
                    stage_18_v314: pl.Scalar[pl.INDEX] = pl.min(stage_18_v313 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v315: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v314])
                    stage_18_v316: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v315, pl.INDEX), 0)
                    stage_18_v317: pl.Scalar[pl.INDEX] = stage_18_v316 * 4160
                    if stage_18_v317 % 128 == 0:
                        stage_18_v532: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v530, view_98, [stage_18_v307 + 32, 0], [stage_18_v317 // 128, 0], [32, 128], transpose=False)
                        stage_18_v533: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v532)
                    else:
                        stage_18_v534: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v530, view_99, [stage_18_v307 + 32, 0], [stage_18_v317 // 128, 0], [32, 128], transpose=False)
                        stage_18_v533: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v534)
                    stage_18_v318: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v533)
                    stage_18_v319: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v318, 0, stage_18_v307, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v320: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v304
                    stage_18_v321: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v322: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v303, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v323: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v320)
                    stage_18_v324: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v322)
                    stage_18_v325: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v321, stage_18_v323, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v326: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v324, 6, 64)
                    stage_18_v327 = pl.tile.store(stage_18_v326, [stage_18_v70, stage_18_v305 - 64], temporary_49)
                    stage_18_v328: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v325
                    stage_18_v329: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v319
                    stage_18_v330: pl.Scalar[pl.INDEX] = 640
                    stage_18_v331: pl.Scalar[pl.INDEX] = 704
                    stage_18_v332: pl.Scalar[pl.INDEX] = 64
                    stage_18_v333: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v331
                    stage_18_v334: pl.Scalar[pl.INDEX] = pl.min(stage_18_v333 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v335: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v334])
                    stage_18_v336: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v335, pl.INDEX), 0)
                    stage_18_v337: pl.Scalar[pl.INDEX] = stage_18_v336 * 4160
                    if stage_18_v337 % 128 == 0:
                        stage_18_v535: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v533, view_98, [stage_18_v332, 0], [stage_18_v337 // 128, 0], [32, 128], transpose=False)
                        stage_18_v536: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v535)
                    else:
                        stage_18_v537: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v533, view_99, [stage_18_v332, 0], [stage_18_v337 // 128, 0], [32, 128], transpose=False)
                        stage_18_v536: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v537)
                    stage_18_v338: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v331 + 32
                    stage_18_v339: pl.Scalar[pl.INDEX] = pl.min(stage_18_v338 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v340: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v339])
                    stage_18_v341: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v340, pl.INDEX), 0)
                    stage_18_v342: pl.Scalar[pl.INDEX] = stage_18_v341 * 4160
                    if stage_18_v342 % 128 == 0:
                        stage_18_v538: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v536, view_98, [stage_18_v332 + 32, 0], [stage_18_v342 // 128, 0], [32, 128], transpose=False)
                        stage_18_v539: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v538)
                    else:
                        stage_18_v540: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v536, view_99, [stage_18_v332 + 32, 0], [stage_18_v342 // 128, 0], [32, 128], transpose=False)
                        stage_18_v539: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v540)
                    stage_18_v343: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v539)
                    stage_18_v344: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v343, 0, stage_18_v332, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v345: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v329
                    stage_18_v346: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v347: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v328, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v348: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v345)
                    stage_18_v349: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v347)
                    stage_18_v350: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v346, stage_18_v348, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v351: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v349, 6, 64)
                    stage_18_v352 = pl.tile.store(stage_18_v351, [stage_18_v70, stage_18_v330 - 64], temporary_49)
                    stage_18_v353: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v350
                    stage_18_v354: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v344
                    stage_18_v355: pl.Scalar[pl.INDEX] = 704
                    stage_18_v356: pl.Scalar[pl.INDEX] = 768
                    stage_18_v357: pl.Scalar[pl.INDEX] = 0
                    stage_18_v358: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v356
                    stage_18_v359: pl.Scalar[pl.INDEX] = pl.min(stage_18_v358 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v360: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v359])
                    stage_18_v361: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v360, pl.INDEX), 0)
                    stage_18_v362: pl.Scalar[pl.INDEX] = stage_18_v361 * 4160
                    if stage_18_v362 % 128 == 0:
                        stage_18_v541: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v539, view_98, [stage_18_v357, 0], [stage_18_v362 // 128, 0], [32, 128], transpose=False)
                        stage_18_v542: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v541)
                    else:
                        stage_18_v543: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v539, view_99, [stage_18_v357, 0], [stage_18_v362 // 128, 0], [32, 128], transpose=False)
                        stage_18_v542: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v543)
                    stage_18_v363: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v356 + 32
                    stage_18_v364: pl.Scalar[pl.INDEX] = pl.min(stage_18_v363 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v365: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v364])
                    stage_18_v366: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v365, pl.INDEX), 0)
                    stage_18_v367: pl.Scalar[pl.INDEX] = stage_18_v366 * 4160
                    if stage_18_v367 % 128 == 0:
                        stage_18_v544: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v542, view_98, [stage_18_v357 + 32, 0], [stage_18_v367 // 128, 0], [32, 128], transpose=False)
                        stage_18_v545: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v544)
                    else:
                        stage_18_v546: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v542, view_99, [stage_18_v357 + 32, 0], [stage_18_v367 // 128, 0], [32, 128], transpose=False)
                        stage_18_v545: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v546)
                    stage_18_v368: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v545)
                    stage_18_v369: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v368, 0, stage_18_v357, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v370: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v354
                    stage_18_v371: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v372: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v353, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v373: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v370)
                    stage_18_v374: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v372)
                    stage_18_v375: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v371, stage_18_v373, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v376: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v374, 6, 64)
                    stage_18_v377 = pl.tile.store(stage_18_v376, [stage_18_v70, stage_18_v355 - 64], temporary_49)
                    stage_18_v378: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v375
                    stage_18_v379: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v369
                    stage_18_v380: pl.Scalar[pl.INDEX] = 768
                    stage_18_v381: pl.Scalar[pl.INDEX] = 832
                    stage_18_v382: pl.Scalar[pl.INDEX] = 64
                    stage_18_v383: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v381
                    stage_18_v384: pl.Scalar[pl.INDEX] = pl.min(stage_18_v383 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v385: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v384])
                    stage_18_v386: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v385, pl.INDEX), 0)
                    stage_18_v387: pl.Scalar[pl.INDEX] = stage_18_v386 * 4160
                    if stage_18_v387 % 128 == 0:
                        stage_18_v547: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v545, view_98, [stage_18_v382, 0], [stage_18_v387 // 128, 0], [32, 128], transpose=False)
                        stage_18_v548: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v547)
                    else:
                        stage_18_v549: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v545, view_99, [stage_18_v382, 0], [stage_18_v387 // 128, 0], [32, 128], transpose=False)
                        stage_18_v548: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v549)
                    stage_18_v388: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v381 + 32
                    stage_18_v389: pl.Scalar[pl.INDEX] = pl.min(stage_18_v388 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v390: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v389])
                    stage_18_v391: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v390, pl.INDEX), 0)
                    stage_18_v392: pl.Scalar[pl.INDEX] = stage_18_v391 * 4160
                    if stage_18_v392 % 128 == 0:
                        stage_18_v550: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v548, view_98, [stage_18_v382 + 32, 0], [stage_18_v392 // 128, 0], [32, 128], transpose=False)
                        stage_18_v551: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v550)
                    else:
                        stage_18_v552: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v548, view_99, [stage_18_v382 + 32, 0], [stage_18_v392 // 128, 0], [32, 128], transpose=False)
                        stage_18_v551: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v552)
                    stage_18_v393: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v551)
                    stage_18_v394: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v393, 0, stage_18_v382, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v395: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v379
                    stage_18_v396: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v397: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v378, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v398: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v395)
                    stage_18_v399: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v397)
                    stage_18_v400: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v396, stage_18_v398, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v401: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v399, 6, 64)
                    stage_18_v402 = pl.tile.store(stage_18_v401, [stage_18_v70, stage_18_v380 - 64], temporary_49)
                    stage_18_v403: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v400
                    stage_18_v404: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v394
                    stage_18_v405: pl.Scalar[pl.INDEX] = 832
                    stage_18_v406: pl.Scalar[pl.INDEX] = 896
                    stage_18_v407: pl.Scalar[pl.INDEX] = 0
                    stage_18_v408: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v406
                    stage_18_v409: pl.Scalar[pl.INDEX] = pl.min(stage_18_v408 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v410: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v409])
                    stage_18_v411: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v410, pl.INDEX), 0)
                    stage_18_v412: pl.Scalar[pl.INDEX] = stage_18_v411 * 4160
                    if stage_18_v412 % 128 == 0:
                        stage_18_v553: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v551, view_98, [stage_18_v407, 0], [stage_18_v412 // 128, 0], [32, 128], transpose=False)
                        stage_18_v554: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v553)
                    else:
                        stage_18_v555: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v551, view_99, [stage_18_v407, 0], [stage_18_v412 // 128, 0], [32, 128], transpose=False)
                        stage_18_v554: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v555)
                    stage_18_v413: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v406 + 32
                    stage_18_v414: pl.Scalar[pl.INDEX] = pl.min(stage_18_v413 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v415: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v414])
                    stage_18_v416: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v415, pl.INDEX), 0)
                    stage_18_v417: pl.Scalar[pl.INDEX] = stage_18_v416 * 4160
                    if stage_18_v417 % 128 == 0:
                        stage_18_v556: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v554, view_98, [stage_18_v407 + 32, 0], [stage_18_v417 // 128, 0], [32, 128], transpose=False)
                        stage_18_v557: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v556)
                    else:
                        stage_18_v558: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v554, view_99, [stage_18_v407 + 32, 0], [stage_18_v417 // 128, 0], [32, 128], transpose=False)
                        stage_18_v557: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v558)
                    stage_18_v418: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v557)
                    stage_18_v419: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v418, 0, stage_18_v407, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v420: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v404
                    stage_18_v421: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v422: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v403, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v423: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v420)
                    stage_18_v424: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v422)
                    stage_18_v425: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v421, stage_18_v423, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v426: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v424, 6, 64)
                    stage_18_v427 = pl.tile.store(stage_18_v426, [stage_18_v70, stage_18_v405 - 64], temporary_49)
                    stage_18_v428: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v425
                    stage_18_v429: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v419
                    stage_18_v430: pl.Scalar[pl.INDEX] = 896
                    stage_18_v431: pl.Scalar[pl.INDEX] = 960
                    stage_18_v432: pl.Scalar[pl.INDEX] = 64
                    stage_18_v433: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v431
                    stage_18_v434: pl.Scalar[pl.INDEX] = pl.min(stage_18_v433 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v435: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v434])
                    stage_18_v436: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v435, pl.INDEX), 0)
                    stage_18_v437: pl.Scalar[pl.INDEX] = stage_18_v436 * 4160
                    if stage_18_v437 % 128 == 0:
                        stage_18_v559: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v557, view_98, [stage_18_v432, 0], [stage_18_v437 // 128, 0], [32, 128], transpose=False)
                        stage_18_v560: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v559)
                    else:
                        stage_18_v561: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v557, view_99, [stage_18_v432, 0], [stage_18_v437 // 128, 0], [32, 128], transpose=False)
                        stage_18_v560: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v561)
                    stage_18_v438: pl.Scalar[pl.INDEX] = stage_18_v37 + stage_18_v69 + stage_18_v431 + 32
                    stage_18_v439: pl.Scalar[pl.INDEX] = pl.min(stage_18_v438 // 32, pl.max((stage_18_v35 - 1) // 32, 0))
                    stage_18_v440: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_18_v32, stage_18_v439])
                    stage_18_v441: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_18_v440, pl.INDEX), 0)
                    stage_18_v442: pl.Scalar[pl.INDEX] = stage_18_v441 * 4160
                    if stage_18_v442 % 128 == 0:
                        stage_18_v562: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v560, view_98, [stage_18_v432 + 32, 0], [stage_18_v442 // 128, 0], [32, 128], transpose=False)
                        stage_18_v563: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v562)
                    else:
                        stage_18_v564: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_18_v560, view_99, [stage_18_v432 + 32, 0], [stage_18_v442 // 128, 0], [32, 128], transpose=False)
                        stage_18_v563: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_18_v564)
                    stage_18_v443: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_18_v563)
                    stage_18_v444: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_18_v443, 0, stage_18_v432, [128, 64], target_memory=pl.Mem.Right)
                    stage_18_v445: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v429
                    stage_18_v446: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v447: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v428, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v448: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v445)
                    stage_18_v449: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v447)
                    stage_18_v450: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v446, stage_18_v448, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v451: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v449, 6, 64)
                    stage_18_v452 = pl.tile.store(stage_18_v451, [stage_18_v70, stage_18_v430 - 64], temporary_49)
                    stage_18_v453: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v450
                    stage_18_v454: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v444
                    stage_18_v455: pl.Scalar[pl.INDEX] = 960
                    stage_18_v456: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_18_v454
                    stage_18_v457: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_18_v458: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v453, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v459: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v58, stage_18_v456)
                    stage_18_v460: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v458)
                    stage_18_v461: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_18_v457, stage_18_v459, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_18_v462: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v460, 6, 64)
                    stage_18_v463 = pl.tile.store(stage_18_v462, [stage_18_v70, stage_18_v455 - 64], temporary_49)
                    stage_18_v464: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_18_v461
                    stage_18_v465: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_18_v464, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    stage_18_v466: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_18_v59, stage_18_v465)
                    stage_18_v467: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_18_v466, 6, 64)
                    stage_18_v468 = pl.tile.store(stage_18_v467, [stage_18_v70, 960], temporary_49)
                    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                for stage_18_v61 in pl.range(pl.min(stage_18_v55, 2)):
                    pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
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
        stage_19_v21: pl.Scalar[pl.INDEX] = block_19
        pl.system.set_ffts(temporary_62)
        stage_19_v22: pl.Scalar[pl.INDEX] = (96 - stage_19_v21 + 23) // 24
        stage_19_v23: pl.Scalar[pl.INDEX] = stage_19_v22 * 5
        stage_19_v24: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.create([64, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat)
        stage_19_v25: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.tile.create([384, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat)
        for (stage_19_v26, (stage_19_v27, stage_19_v28)) in pl.range(stage_19_v23 + 2, init_values=(stage_19_v25, stage_19_v24)):
            if stage_19_v26 < stage_19_v23:
                stage_19_v29: pl.Scalar[pl.INDEX] = stage_19_v26 % 5
                stage_19_v30: pl.Scalar[pl.INDEX] = stage_19_v21 + stage_19_v26 // 5 * 24
                if stage_19_v29 == 0:
                    stage_19_v38: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.gather_row(stage_19_v28, view_111, [0, 0], [stage_19_v30 * 64, 0], [64, 512], transpose=False)
                    stage_19_v39: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.yield_(stage_19_v38)
                else:
                    stage_19_v39: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.yield_(stage_19_v28)
                stage_19_v31: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_19_v30, stage_19_v29])
                if 0 < pl.cast(stage_19_v31, pl.INDEX):
                    stage_19_v40: pl.Scalar[pl.INDEX] = stage_19_v21 * 3 + stage_19_v26 % 3
                    stage_19_v41: pl.Scalar[pl.INDEX] = stage_19_v40 * 128
                    stage_19_v42: pl.Scalar[pl.INDEX] = stage_19_v40 * 64
                    pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    stage_19_v43: pl.Scalar[pl.INDEX] = stage_19_v26 % 3 * 128
                    stage_19_v44: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.tile.gather_row(stage_19_v27, temporary_56, [stage_19_v43, 0], [stage_19_v41, 0], [128, 512], transpose=False)
                    stage_19_v45: pl.Tile[[512, 384], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_19_v44)
                    stage_19_v46: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                    for (stage_19_v71, (stage_19_v76,)) in pl.pipeline(0, 512, 128, stage=2, init_values=(stage_19_v46,)):
                        stage_19_v72: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(stage_19_v39, 0, stage_19_v71, [64, 128], target_memory=pl.Mem.Left)
                        stage_19_v73: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_19_v45, stage_19_v71, stage_19_v43, [128, 128], target_memory=pl.Mem.Right)
                        stage_19_v74: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_19_v76, stage_19_v72, stage_19_v73, stage_19_v71 == 0)
                        stage_19_v75: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_19_v74)
                    stage_19_v47 = pl.tile.store(stage_19_v75, [stage_19_v42, 0], temporary_57)
                    pl.system.sync_set(1, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                    stage_19_v48: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.yield_(stage_19_v44)
                else:
                    stage_19_v48: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.yield_(stage_19_v27)
                (stage_19_v49, stage_19_v50) = pl.yield_(stage_19_v48, stage_19_v39)
            else:
                (stage_19_v49, stage_19_v50) = pl.yield_(stage_19_v27, stage_19_v28)
            if 2 <= stage_19_v26:
                stage_19_v32: pl.Scalar[pl.INDEX] = stage_19_v26 - 2
                stage_19_v33: pl.Scalar[pl.INDEX] = stage_19_v32 % 5
                stage_19_v34: pl.Scalar[pl.INDEX] = stage_19_v21 + stage_19_v32 // 5 * 24
                stage_19_v35: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_19_v34, stage_19_v33])
                if 0 < pl.cast(stage_19_v35, pl.INDEX):
                    stage_19_v51: pl.Scalar[pl.INDEX] = stage_19_v21 * 3 + stage_19_v32 % 3
                    stage_19_v52: pl.Scalar[pl.INDEX] = stage_19_v51 * 64
                    pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    stage_19_v53: pl.Tile[[64, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(temporary_58, [stage_19_v52, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat)
                    stage_19_v54: pl.Scalar[pl.INDEX] = stage_19_v32 % 3 * 128
                    stage_19_v55: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(stage_19_v53, target_memory=pl.Mem.Left)
                    stage_19_v56: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_19_v49, stage_19_v54, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_19_v57: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_19_v55, stage_19_v56)
                    stage_19_v58: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_19_v49, stage_19_v54, 128, [128, 128], target_memory=pl.Mem.Right)
                    stage_19_v59: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_19_v55, stage_19_v58)
                    stage_19_v60 = pl.tile.store(stage_19_v57, [stage_19_v52, 0], temporary_59)
                    stage_19_v61: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = stage_19_v59
                    stage_19_v62: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_19_v49, stage_19_v54, 256, [128, 128], target_memory=pl.Mem.Right)
                    stage_19_v63: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_19_v55, stage_19_v62)
                    stage_19_v64 = pl.tile.store(stage_19_v61, [stage_19_v52, 128], temporary_59)
                    stage_19_v65: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = stage_19_v63
                    stage_19_v66: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_19_v49, stage_19_v54, 384, [128, 128], target_memory=pl.Mem.Right)
                    stage_19_v67: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_19_v55, stage_19_v66)
                    stage_19_v68 = pl.tile.store(stage_19_v65, [stage_19_v52, 256], temporary_59)
                    stage_19_v69: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = stage_19_v67
                    stage_19_v70 = pl.tile.store(stage_19_v69, [stage_19_v52, 384], temporary_59)
                    pl.system.sync_set(3, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            (stage_19_v36, stage_19_v37) = pl.yield_(stage_19_v49, stage_19_v50)
        pl.system.set_ffts(temporary_62)
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
        stage_20_v5: pl.Scalar[pl.INDEX] = stage_20_v4 // 8
        stage_20_v6: pl.Scalar[pl.INDEX] = stage_20_v4 % 8
        stage_20_v7: pl.Scalar[pl.INDEX] = stage_20_v5 * 128
        stage_20_v8: pl.Scalar[pl.INDEX] = pl.min(96 - stage_20_v7, 128)
        stage_20_v9: pl.Scalar[pl.INDEX] = 2688 + stage_20_v7
        stage_20_v10: pl.Scalar[pl.INDEX] = stage_20_v6 * 128
        stage_20_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_20_v9, 0], [stage_20_v8, 256])
        stage_20_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [7, 0, stage_20_v10])
        stage_20_v13 = pl.tensor.matmul(stage_20_v11, stage_20_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_20_v15, (stage_20_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_20_v13,)):
            stage_20_v16: pl.Scalar[pl.INDEX] = stage_20_v15 * 256
            stage_20_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_20_v9, stage_20_v16], [stage_20_v8, 256])
            stage_20_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [7, stage_20_v16, stage_20_v10])
            stage_20_v19 = pl.tensor.matmul_acc(stage_20_v21, stage_20_v17, stage_20_v18, a_trans=False, b_trans=False)
            stage_20_v20 = pl.yield_(stage_20_v19)
        stage_20_v14 = pl.tensor.assemble(temporary_63, stage_20_v20, [stage_20_v7, 7168 + stage_20_v10])
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
        stage_21_v5: pl.Scalar[pl.INDEX] = stage_21_v4 // 8
        stage_21_v6: pl.Scalar[pl.INDEX] = stage_21_v4 % 8
        stage_21_v7: pl.Scalar[pl.INDEX] = stage_21_v5 * 128
        stage_21_v8: pl.Scalar[pl.INDEX] = pl.min(96 - stage_21_v7, 128)
        stage_21_v9: pl.Scalar[pl.INDEX] = 2304 + stage_21_v7
        stage_21_v10: pl.Scalar[pl.INDEX] = stage_21_v6 * 128
        stage_21_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_21_v9, 0], [stage_21_v8, 256])
        stage_21_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [6, 0, stage_21_v10])
        stage_21_v13 = pl.tensor.matmul(stage_21_v11, stage_21_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_21_v15, (stage_21_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_21_v13,)):
            stage_21_v16: pl.Scalar[pl.INDEX] = stage_21_v15 * 256
            stage_21_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_21_v9, stage_21_v16], [stage_21_v8, 256])
            stage_21_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [6, stage_21_v16, stage_21_v10])
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
        stage_22_v5: pl.Scalar[pl.INDEX] = stage_22_v4 // 8
        stage_22_v6: pl.Scalar[pl.INDEX] = stage_22_v4 % 8
        stage_22_v7: pl.Scalar[pl.INDEX] = stage_22_v5 * 128
        stage_22_v8: pl.Scalar[pl.INDEX] = pl.min(96 - stage_22_v7, 128)
        stage_22_v9: pl.Scalar[pl.INDEX] = 1920 + stage_22_v7
        stage_22_v10: pl.Scalar[pl.INDEX] = stage_22_v6 * 128
        stage_22_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_22_v9, 0], [stage_22_v8, 256])
        stage_22_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [5, 0, stage_22_v10])
        stage_22_v13 = pl.tensor.matmul(stage_22_v11, stage_22_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_22_v15, (stage_22_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_22_v13,)):
            stage_22_v16: pl.Scalar[pl.INDEX] = stage_22_v15 * 256
            stage_22_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_22_v9, stage_22_v16], [stage_22_v8, 256])
            stage_22_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [5, stage_22_v16, stage_22_v10])
            stage_22_v19 = pl.tensor.matmul_acc(stage_22_v21, stage_22_v17, stage_22_v18, a_trans=False, b_trans=False)
            stage_22_v20 = pl.yield_(stage_22_v19)
        stage_22_v14 = pl.tensor.assemble(temporary_63, stage_22_v20, [stage_22_v7, 5120 + stage_22_v10])
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
        stage_23_v5: pl.Scalar[pl.INDEX] = stage_23_v4 // 8
        stage_23_v6: pl.Scalar[pl.INDEX] = stage_23_v4 % 8
        stage_23_v7: pl.Scalar[pl.INDEX] = stage_23_v5 * 128
        stage_23_v8: pl.Scalar[pl.INDEX] = pl.min(96 - stage_23_v7, 128)
        stage_23_v9: pl.Scalar[pl.INDEX] = 2688 + stage_23_v7
        stage_23_v10: pl.Scalar[pl.INDEX] = stage_23_v6 * 128
        stage_23_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_23_v9, 0], [stage_23_v8, 256])
        stage_23_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [7, 0, stage_23_v10])
        stage_23_v13 = pl.tensor.matmul(stage_23_v11, stage_23_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_23_v15, (stage_23_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_23_v13,)):
            stage_23_v16: pl.Scalar[pl.INDEX] = stage_23_v15 * 256
            stage_23_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_23_v9, stage_23_v16], [stage_23_v8, 256])
            stage_23_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [7, stage_23_v16, stage_23_v10])
            stage_23_v19 = pl.tensor.matmul_acc(stage_23_v21, stage_23_v17, stage_23_v18, a_trans=False, b_trans=False)
            stage_23_v20 = pl.yield_(stage_23_v19)
        stage_23_v14 = pl.tensor.assemble(temporary_63, stage_23_v20, [stage_23_v7, 7168 + stage_23_v10])
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
        stage_24_v5: pl.Scalar[pl.INDEX] = stage_24_v4 // 8
        stage_24_v6: pl.Scalar[pl.INDEX] = stage_24_v4 % 8
        stage_24_v7: pl.Scalar[pl.INDEX] = stage_24_v5 * 128
        stage_24_v8: pl.Scalar[pl.INDEX] = pl.min(96 - stage_24_v7, 128)
        stage_24_v9: pl.Scalar[pl.INDEX] = 1536 + stage_24_v7
        stage_24_v10: pl.Scalar[pl.INDEX] = stage_24_v6 * 128
        stage_24_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_24_v9, 0], [stage_24_v8, 256])
        stage_24_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [4, 0, stage_24_v10])
        stage_24_v13 = pl.tensor.matmul(stage_24_v11, stage_24_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_24_v15, (stage_24_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_24_v13,)):
            stage_24_v16: pl.Scalar[pl.INDEX] = stage_24_v15 * 256
            stage_24_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_24_v9, stage_24_v16], [stage_24_v8, 256])
            stage_24_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [4, stage_24_v16, stage_24_v10])
            stage_24_v19 = pl.tensor.matmul_acc(stage_24_v21, stage_24_v17, stage_24_v18, a_trans=False, b_trans=False)
            stage_24_v20 = pl.yield_(stage_24_v19)
        stage_24_v14 = pl.tensor.assemble(temporary_63, stage_24_v20, [stage_24_v7, 4096 + stage_24_v10])
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
        stage_25_v5: pl.Scalar[pl.INDEX] = stage_25_v4 // 8
        stage_25_v6: pl.Scalar[pl.INDEX] = stage_25_v4 % 8
        stage_25_v7: pl.Scalar[pl.INDEX] = stage_25_v5 * 128
        stage_25_v8: pl.Scalar[pl.INDEX] = pl.min(96 - stage_25_v7, 128)
        stage_25_v9: pl.Scalar[pl.INDEX] = 1152 + stage_25_v7
        stage_25_v10: pl.Scalar[pl.INDEX] = stage_25_v6 * 128
        stage_25_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_25_v9, 0], [stage_25_v8, 256])
        stage_25_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [3, 0, stage_25_v10])
        stage_25_v13 = pl.tensor.matmul(stage_25_v11, stage_25_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_25_v15, (stage_25_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_25_v13,)):
            stage_25_v16: pl.Scalar[pl.INDEX] = stage_25_v15 * 256
            stage_25_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_25_v9, stage_25_v16], [stage_25_v8, 256])
            stage_25_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [3, stage_25_v16, stage_25_v10])
            stage_25_v19 = pl.tensor.matmul_acc(stage_25_v21, stage_25_v17, stage_25_v18, a_trans=False, b_trans=False)
            stage_25_v20 = pl.yield_(stage_25_v19)
        stage_25_v14 = pl.tensor.assemble(temporary_63, stage_25_v20, [stage_25_v7, 3072 + stage_25_v10])
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
        stage_26_v5: pl.Scalar[pl.INDEX] = stage_26_v4 // 8
        stage_26_v6: pl.Scalar[pl.INDEX] = stage_26_v4 % 8
        stage_26_v7: pl.Scalar[pl.INDEX] = stage_26_v5 * 128
        stage_26_v8: pl.Scalar[pl.INDEX] = pl.min(96 - stage_26_v7, 128)
        stage_26_v9: pl.Scalar[pl.INDEX] = 0 + stage_26_v7
        stage_26_v10: pl.Scalar[pl.INDEX] = stage_26_v6 * 128
        stage_26_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_26_v9, 0], [stage_26_v8, 256])
        stage_26_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [0, 0, stage_26_v10])
        stage_26_v13 = pl.tensor.matmul(stage_26_v11, stage_26_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_26_v15, (stage_26_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_26_v13,)):
            stage_26_v16: pl.Scalar[pl.INDEX] = stage_26_v15 * 256
            stage_26_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_26_v9, stage_26_v16], [stage_26_v8, 256])
            stage_26_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [0, stage_26_v16, stage_26_v10])
            stage_26_v19 = pl.tensor.matmul_acc(stage_26_v21, stage_26_v17, stage_26_v18, a_trans=False, b_trans=False)
            stage_26_v20 = pl.yield_(stage_26_v19)
        stage_26_v14 = pl.tensor.assemble(temporary_63, stage_26_v20, [stage_26_v7, 0 + stage_26_v10])
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
        stage_27_v5: pl.Scalar[pl.INDEX] = stage_27_v4 // 8
        stage_27_v6: pl.Scalar[pl.INDEX] = stage_27_v4 % 8
        stage_27_v7: pl.Scalar[pl.INDEX] = stage_27_v5 * 128
        stage_27_v8: pl.Scalar[pl.INDEX] = pl.min(96 - stage_27_v7, 128)
        stage_27_v9: pl.Scalar[pl.INDEX] = 768 + stage_27_v7
        stage_27_v10: pl.Scalar[pl.INDEX] = stage_27_v6 * 128
        stage_27_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_27_v9, 0], [stage_27_v8, 256])
        stage_27_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [2, 0, stage_27_v10])
        stage_27_v13 = pl.tensor.matmul(stage_27_v11, stage_27_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_27_v15, (stage_27_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_27_v13,)):
            stage_27_v16: pl.Scalar[pl.INDEX] = stage_27_v15 * 256
            stage_27_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_27_v9, stage_27_v16], [stage_27_v8, 256])
            stage_27_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [2, stage_27_v16, stage_27_v10])
            stage_27_v19 = pl.tensor.matmul_acc(stage_27_v21, stage_27_v17, stage_27_v18, a_trans=False, b_trans=False)
            stage_27_v20 = pl.yield_(stage_27_v19)
        stage_27_v14 = pl.tensor.assemble(temporary_63, stage_27_v20, [stage_27_v7, 2048 + stage_27_v10])
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
        stage_28_v5: pl.Scalar[pl.INDEX] = stage_28_v4 // 8
        stage_28_v6: pl.Scalar[pl.INDEX] = stage_28_v4 % 8
        stage_28_v7: pl.Scalar[pl.INDEX] = stage_28_v5 * 128
        stage_28_v8: pl.Scalar[pl.INDEX] = pl.min(96 - stage_28_v7, 128)
        stage_28_v9: pl.Scalar[pl.INDEX] = 1152 + stage_28_v7
        stage_28_v10: pl.Scalar[pl.INDEX] = stage_28_v6 * 128
        stage_28_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_28_v9, 0], [stage_28_v8, 256])
        stage_28_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [3, 0, stage_28_v10])
        stage_28_v13 = pl.tensor.matmul(stage_28_v11, stage_28_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_28_v15, (stage_28_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_28_v13,)):
            stage_28_v16: pl.Scalar[pl.INDEX] = stage_28_v15 * 256
            stage_28_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_28_v9, stage_28_v16], [stage_28_v8, 256])
            stage_28_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [3, stage_28_v16, stage_28_v10])
            stage_28_v19 = pl.tensor.matmul_acc(stage_28_v21, stage_28_v17, stage_28_v18, a_trans=False, b_trans=False)
            stage_28_v20 = pl.yield_(stage_28_v19)
        stage_28_v14 = pl.tensor.assemble(temporary_63, stage_28_v20, [stage_28_v7, 3072 + stage_28_v10])
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
        stage_29_v5: pl.Scalar[pl.INDEX] = stage_29_v4 // 8
        stage_29_v6: pl.Scalar[pl.INDEX] = stage_29_v4 % 8
        stage_29_v7: pl.Scalar[pl.INDEX] = stage_29_v5 * 128
        stage_29_v8: pl.Scalar[pl.INDEX] = pl.min(96 - stage_29_v7, 128)
        stage_29_v9: pl.Scalar[pl.INDEX] = 384 + stage_29_v7
        stage_29_v10: pl.Scalar[pl.INDEX] = stage_29_v6 * 128
        stage_29_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_29_v9, 0], [stage_29_v8, 256])
        stage_29_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [1, 0, stage_29_v10])
        stage_29_v13 = pl.tensor.matmul(stage_29_v11, stage_29_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_29_v15, (stage_29_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_29_v13,)):
            stage_29_v16: pl.Scalar[pl.INDEX] = stage_29_v15 * 256
            stage_29_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_29_v9, stage_29_v16], [stage_29_v8, 256])
            stage_29_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [1, stage_29_v16, stage_29_v10])
            stage_29_v19 = pl.tensor.matmul_acc(stage_29_v21, stage_29_v17, stage_29_v18, a_trans=False, b_trans=False)
            stage_29_v20 = pl.yield_(stage_29_v19)
        stage_29_v14 = pl.tensor.assemble(temporary_63, stage_29_v20, [stage_29_v7, 1024 + stage_29_v10])
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
        for stage_30_v6 in pl.range(1):
            stage_30_v7: pl.Scalar[pl.INDEX] = stage_30_v6 * 96
            stage_30_v8: pl.Tile[[96, 1024], pl.INT8, pl.Mem.Mat] = pl.tile.load(temporary_64, [stage_30_v7, 7168], [96, 1024], [96, 1024], target_memory=pl.Mem.Mat)
            for stage_30_v9 in pl.range(2):
                stage_30_v10: pl.Scalar[pl.INDEX] = stage_30_v5 + stage_30_v9 * 256
                stage_30_v11: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.create([96, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
                for (stage_30_v12, (stage_30_v19,)) in pl.pipeline(4, stage=2, init_values=(stage_30_v11,)):
                    stage_30_v13: pl.Scalar[pl.INDEX] = stage_30_v12 * 256
                    stage_30_v14: pl.Scalar[pl.INDEX] = pl.max(7, 0) * 1024 + stage_30_v13
                    stage_30_v15: pl.Tile[[256, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_wo_b, [stage_30_v14, stage_30_v10], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
                    stage_30_v16: pl.Tile[[96, 256], pl.INT8, pl.Mem.Mat] = pl.tile.slice(stage_30_v8, [96, 256], [0, stage_30_v13])
                    stage_30_v17: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_30_v19, stage_30_v16, stage_30_v15, stage_30_v12 == 0)
                    stage_30_v18: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.yield_(stage_30_v17)
                pl.tile.store(stage_30_v18, [stage_30_v7, 7 * 4096 + stage_30_v10], temporary_66)
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
        for stage_31_v6 in pl.range(1):
            stage_31_v7: pl.Scalar[pl.INDEX] = stage_31_v6 * 96
            stage_31_v8: pl.Tile[[96, 1024], pl.INT8, pl.Mem.Mat] = pl.tile.load(temporary_64, [stage_31_v7, 4096], [96, 1024], [96, 1024], target_memory=pl.Mem.Mat)
            for stage_31_v9 in pl.range(2):
                stage_31_v10: pl.Scalar[pl.INDEX] = stage_31_v5 + stage_31_v9 * 256
                stage_31_v11: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.create([96, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
                for (stage_31_v12, (stage_31_v19,)) in pl.pipeline(4, stage=2, init_values=(stage_31_v11,)):
                    stage_31_v13: pl.Scalar[pl.INDEX] = stage_31_v12 * 256
                    stage_31_v14: pl.Scalar[pl.INDEX] = pl.max(4, 0) * 1024 + stage_31_v13
                    stage_31_v15: pl.Tile[[256, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_wo_b, [stage_31_v14, stage_31_v10], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
                    stage_31_v16: pl.Tile[[96, 256], pl.INT8, pl.Mem.Mat] = pl.tile.slice(stage_31_v8, [96, 256], [0, stage_31_v13])
                    stage_31_v17: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_31_v19, stage_31_v16, stage_31_v15, stage_31_v12 == 0)
                    stage_31_v18: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.yield_(stage_31_v17)
                pl.tile.store(stage_31_v18, [stage_31_v7, 4 * 4096 + stage_31_v10], temporary_66)
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
        for stage_32_v6 in pl.range(1):
            stage_32_v7: pl.Scalar[pl.INDEX] = stage_32_v6 * 96
            stage_32_v8: pl.Tile[[96, 1024], pl.INT8, pl.Mem.Mat] = pl.tile.load(temporary_64, [stage_32_v7, 5120], [96, 1024], [96, 1024], target_memory=pl.Mem.Mat)
            for stage_32_v9 in pl.range(2):
                stage_32_v10: pl.Scalar[pl.INDEX] = stage_32_v5 + stage_32_v9 * 256
                stage_32_v11: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.create([96, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
                for (stage_32_v12, (stage_32_v19,)) in pl.pipeline(4, stage=2, init_values=(stage_32_v11,)):
                    stage_32_v13: pl.Scalar[pl.INDEX] = stage_32_v12 * 256
                    stage_32_v14: pl.Scalar[pl.INDEX] = pl.max(5, 0) * 1024 + stage_32_v13
                    stage_32_v15: pl.Tile[[256, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_wo_b, [stage_32_v14, stage_32_v10], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
                    stage_32_v16: pl.Tile[[96, 256], pl.INT8, pl.Mem.Mat] = pl.tile.slice(stage_32_v8, [96, 256], [0, stage_32_v13])
                    stage_32_v17: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_32_v19, stage_32_v16, stage_32_v15, stage_32_v12 == 0)
                    stage_32_v18: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.yield_(stage_32_v17)
                pl.tile.store(stage_32_v18, [stage_32_v7, 5 * 4096 + stage_32_v10], temporary_66)
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
        for stage_33_v6 in pl.range(1):
            stage_33_v7: pl.Scalar[pl.INDEX] = stage_33_v6 * 96
            stage_33_v8: pl.Tile[[96, 1024], pl.INT8, pl.Mem.Mat] = pl.tile.load(temporary_64, [stage_33_v7, 0], [96, 1024], [96, 1024], target_memory=pl.Mem.Mat)
            for stage_33_v9 in pl.range(2):
                stage_33_v10: pl.Scalar[pl.INDEX] = stage_33_v5 + stage_33_v9 * 256
                stage_33_v11: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.create([96, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
                for (stage_33_v12, (stage_33_v19,)) in pl.pipeline(4, stage=2, init_values=(stage_33_v11,)):
                    stage_33_v13: pl.Scalar[pl.INDEX] = stage_33_v12 * 256
                    stage_33_v14: pl.Scalar[pl.INDEX] = pl.max(0, 0) * 1024 + stage_33_v13
                    stage_33_v15: pl.Tile[[256, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_wo_b, [stage_33_v14, stage_33_v10], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
                    stage_33_v16: pl.Tile[[96, 256], pl.INT8, pl.Mem.Mat] = pl.tile.slice(stage_33_v8, [96, 256], [0, stage_33_v13])
                    stage_33_v17: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_33_v19, stage_33_v16, stage_33_v15, stage_33_v12 == 0)
                    stage_33_v18: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.yield_(stage_33_v17)
                pl.tile.store(stage_33_v18, [stage_33_v7, 0 * 4096 + stage_33_v10], temporary_66)
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
        for stage_34_v6 in pl.range(1):
            stage_34_v7: pl.Scalar[pl.INDEX] = stage_34_v6 * 96
            stage_34_v8: pl.Tile[[96, 1024], pl.INT8, pl.Mem.Mat] = pl.tile.load(temporary_64, [stage_34_v7, 2048], [96, 1024], [96, 1024], target_memory=pl.Mem.Mat)
            for stage_34_v9 in pl.range(2):
                stage_34_v10: pl.Scalar[pl.INDEX] = stage_34_v5 + stage_34_v9 * 256
                stage_34_v11: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.create([96, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
                for (stage_34_v12, (stage_34_v19,)) in pl.pipeline(4, stage=2, init_values=(stage_34_v11,)):
                    stage_34_v13: pl.Scalar[pl.INDEX] = stage_34_v12 * 256
                    stage_34_v14: pl.Scalar[pl.INDEX] = pl.max(2, 0) * 1024 + stage_34_v13
                    stage_34_v15: pl.Tile[[256, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_wo_b, [stage_34_v14, stage_34_v10], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
                    stage_34_v16: pl.Tile[[96, 256], pl.INT8, pl.Mem.Mat] = pl.tile.slice(stage_34_v8, [96, 256], [0, stage_34_v13])
                    stage_34_v17: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_34_v19, stage_34_v16, stage_34_v15, stage_34_v12 == 0)
                    stage_34_v18: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.yield_(stage_34_v17)
                pl.tile.store(stage_34_v18, [stage_34_v7, 2 * 4096 + stage_34_v10], temporary_66)
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
        stage_35_v5: pl.Scalar[pl.INDEX] = stage_35_v4 // 8
        stage_35_v6: pl.Scalar[pl.INDEX] = stage_35_v4 % 8
        stage_35_v7: pl.Scalar[pl.INDEX] = stage_35_v5 * 128
        stage_35_v8: pl.Scalar[pl.INDEX] = pl.min(96 - stage_35_v7, 128)
        stage_35_v9: pl.Scalar[pl.INDEX] = 384 + stage_35_v7
        stage_35_v10: pl.Scalar[pl.INDEX] = stage_35_v6 * 128
        stage_35_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_35_v9, 0], [stage_35_v8, 256])
        stage_35_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [1, 0, stage_35_v10])
        stage_35_v13 = pl.tensor.matmul(stage_35_v11, stage_35_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_35_v15, (stage_35_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_35_v13,)):
            stage_35_v16: pl.Scalar[pl.INDEX] = stage_35_v15 * 256
            stage_35_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_35_v9, stage_35_v16], [stage_35_v8, 256])
            stage_35_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 128], [1, stage_35_v16, stage_35_v10])
            stage_35_v19 = pl.tensor.matmul_acc(stage_35_v21, stage_35_v17, stage_35_v18, a_trans=False, b_trans=False)
            stage_35_v20 = pl.yield_(stage_35_v19)
        stage_35_v14 = pl.tensor.assemble(temporary_63, stage_35_v20, [stage_35_v7, 1024 + stage_35_v10])
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
        for stage_36_v6 in pl.range(1):
            stage_36_v7: pl.Scalar[pl.INDEX] = stage_36_v6 * 96
            stage_36_v8: pl.Tile[[96, 1024], pl.INT8, pl.Mem.Mat] = pl.tile.load(temporary_64, [stage_36_v7, 6144], [96, 1024], [96, 1024], target_memory=pl.Mem.Mat)
            for stage_36_v9 in pl.range(2):
                stage_36_v10: pl.Scalar[pl.INDEX] = stage_36_v5 + stage_36_v9 * 256
                stage_36_v11: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.create([96, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
                for (stage_36_v12, (stage_36_v19,)) in pl.pipeline(4, stage=2, init_values=(stage_36_v11,)):
                    stage_36_v13: pl.Scalar[pl.INDEX] = stage_36_v12 * 256
                    stage_36_v14: pl.Scalar[pl.INDEX] = pl.max(6, 0) * 1024 + stage_36_v13
                    stage_36_v15: pl.Tile[[256, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_wo_b, [stage_36_v14, stage_36_v10], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
                    stage_36_v16: pl.Tile[[96, 256], pl.INT8, pl.Mem.Mat] = pl.tile.slice(stage_36_v8, [96, 256], [0, stage_36_v13])
                    stage_36_v17: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_36_v19, stage_36_v16, stage_36_v15, stage_36_v12 == 0)
                    stage_36_v18: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.yield_(stage_36_v17)
                pl.tile.store(stage_36_v18, [stage_36_v7, 6 * 4096 + stage_36_v10], temporary_66)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_36 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_36)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_37 = 15984 + core_id * 6
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
        stage_37_v5: pl.Scalar[pl.INDEX] = stage_37_v4 * 512
        for stage_37_v6 in pl.range(1):
            stage_37_v7: pl.Scalar[pl.INDEX] = stage_37_v6 * 96
            stage_37_v8: pl.Tile[[96, 1024], pl.INT8, pl.Mem.Mat] = pl.tile.load(temporary_64, [stage_37_v7, 3072], [96, 1024], [96, 1024], target_memory=pl.Mem.Mat)
            for stage_37_v9 in pl.range(2):
                stage_37_v10: pl.Scalar[pl.INDEX] = stage_37_v5 + stage_37_v9 * 256
                stage_37_v11: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.create([96, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
                for (stage_37_v12, (stage_37_v19,)) in pl.pipeline(4, stage=2, init_values=(stage_37_v11,)):
                    stage_37_v13: pl.Scalar[pl.INDEX] = stage_37_v12 * 256
                    stage_37_v14: pl.Scalar[pl.INDEX] = pl.max(3, 0) * 1024 + stage_37_v13
                    stage_37_v15: pl.Tile[[256, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_wo_b, [stage_37_v14, stage_37_v10], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
                    stage_37_v16: pl.Tile[[96, 256], pl.INT8, pl.Mem.Mat] = pl.tile.slice(stage_37_v8, [96, 256], [0, stage_37_v13])
                    stage_37_v17: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_37_v19, stage_37_v16, stage_37_v15, stage_37_v12 == 0)
                    stage_37_v18: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.yield_(stage_37_v17)
                pl.tile.store(stage_37_v18, [stage_37_v7, 3 * 4096 + stage_37_v10], temporary_66)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_37 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_37)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_38 = 16416 + core_id * 6
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
        stage_38_v5: pl.Scalar[pl.INDEX] = stage_38_v4 * 512
        for stage_38_v6 in pl.range(1):
            stage_38_v7: pl.Scalar[pl.INDEX] = stage_38_v6 * 96
            stage_38_v8: pl.Tile[[96, 1024], pl.INT8, pl.Mem.Mat] = pl.tile.load(temporary_64, [stage_38_v7, 1024], [96, 1024], [96, 1024], target_memory=pl.Mem.Mat)
            for stage_38_v9 in pl.range(2):
                stage_38_v10: pl.Scalar[pl.INDEX] = stage_38_v5 + stage_38_v9 * 256
                stage_38_v11: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.create([96, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
                for (stage_38_v12, (stage_38_v19,)) in pl.pipeline(4, stage=2, init_values=(stage_38_v11,)):
                    stage_38_v13: pl.Scalar[pl.INDEX] = stage_38_v12 * 256
                    stage_38_v14: pl.Scalar[pl.INDEX] = pl.max(1, 0) * 1024 + stage_38_v13
                    stage_38_v15: pl.Tile[[256, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_wo_b, [stage_38_v14, stage_38_v10], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
                    stage_38_v16: pl.Tile[[96, 256], pl.INT8, pl.Mem.Mat] = pl.tile.slice(stage_38_v8, [96, 256], [0, stage_38_v13])
                    stage_38_v17: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_38_v19, stage_38_v16, stage_38_v15, stage_38_v12 == 0)
                    stage_38_v18: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.yield_(stage_38_v17)
                pl.tile.store(stage_38_v18, [stage_38_v7, 1 * 4096 + stage_38_v10], temporary_66)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_38 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_38)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_39 = 16848 + core_id * 6
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
        stage_39_v5: pl.Scalar[pl.INDEX] = stage_39_v4 * 512
        for stage_39_v6 in pl.range(1):
            stage_39_v7: pl.Scalar[pl.INDEX] = stage_39_v6 * 96
            stage_39_v8: pl.Tile[[96, 1024], pl.INT8, pl.Mem.Mat] = pl.tile.load(temporary_64, [stage_39_v7, 0], [96, 1024], [96, 1024], target_memory=pl.Mem.Mat)
            for stage_39_v9 in pl.range(2):
                stage_39_v10: pl.Scalar[pl.INDEX] = stage_39_v5 + stage_39_v9 * 256
                stage_39_v11: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.create([96, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
                for (stage_39_v12, (stage_39_v19,)) in pl.pipeline(4, stage=2, init_values=(stage_39_v11,)):
                    stage_39_v13: pl.Scalar[pl.INDEX] = stage_39_v12 * 256
                    stage_39_v14: pl.Scalar[pl.INDEX] = pl.max(0, 0) * 1024 + stage_39_v13
                    stage_39_v15: pl.Tile[[256, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_wo_b, [stage_39_v14, stage_39_v10], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
                    stage_39_v16: pl.Tile[[96, 256], pl.INT8, pl.Mem.Mat] = pl.tile.slice(stage_39_v8, [96, 256], [0, stage_39_v13])
                    stage_39_v17: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_39_v19, stage_39_v16, stage_39_v15, stage_39_v12 == 0)
                    stage_39_v18: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.yield_(stage_39_v17)
                pl.tile.store(stage_39_v18, [stage_39_v7, 0 * 4096 + stage_39_v10], temporary_66)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_39 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_39)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    return (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_111, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_128, progress)

@pl.jit.incore
def csa_static_aiv(temporary_4: pl.InOut[pl.Tensor[[96, 1], pl.FP32]], temporary_5: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]], view_2: pl.InOut[pl.Tensor[[96, 16384], pl.FP32]], view_3: pl.Tensor[[96, 16384], pl.BF16], temporary_7: pl.InOut[pl.Tensor[[384, 32], pl.FP32]], arg_hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], temporary_8: pl.InOut[pl.Tensor[[96, 32], pl.FP32]], arg_hc_attn_scale: pl.Tensor[[3], pl.FP32], temporary_0: pl.InOut[pl.Tensor[[96, 4], pl.FP32]], temporary_9: pl.InOut[pl.Tensor[[8, 8], pl.FP32]], temporary_6: pl.InOut[pl.Tensor[[96, 8], pl.FP32]], arg_hc_attn_base: pl.Tensor[[24], pl.FP32], view_12: pl.Tensor[[1, 24], pl.FP32], temporary_1: pl.InOut[pl.Tensor[[96, 16], pl.FP32]], temporary_10: pl.InOut[pl.Tensor[[8, 32], pl.FP32]], temporary_11: pl.InOut[pl.Tensor[[96, 4096], pl.BF16]], temporary_2: pl.InOut[pl.Tensor[[96, 4096], pl.BF16]], temporary_12: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]], arg_attn_norm_w: pl.Tensor[[4096], pl.BF16], arg_cmp_seq_lens: pl.Tensor[[16], pl.INT32], arg_cmp_query_start_loc: pl.Tensor[[17], pl.INT32], temporary_14: pl.InOut[pl.Tensor[[16], pl.INT32]], arg_kv_seq_lens: pl.Tensor[[16], pl.INT32], arg_idx_query_start_loc: pl.Tensor[[17], pl.INT32], temporary_15: pl.InOut[pl.Tensor[[16], pl.INT32]], arg_freqs_sin: pl.Tensor[[96, 64], pl.FP32], temporary_13: pl.InOut[pl.Tensor[[96, 64], pl.FP32]], temporary_20: pl.InOut[pl.Tensor[[96, 64], pl.FP32]], temporary_21: pl.InOut[pl.Tensor[[96, 64], pl.INT32]], temporary_24: pl.InOut[pl.Tensor[[96, 1024], pl.FP32]], arg_wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ], view_31: pl.Tensor[[1, 1024], pl.BF16], temporary_23: pl.InOut[pl.Tensor[[512, 1], pl.FP32]], temporary_19: pl.InOut[pl.Tensor[[96, 1], pl.FP32]], temporary_22: pl.InOut[pl.Tensor[[512, 1024], pl.INT8]], temporary_18: pl.InOut[pl.Tensor[[96, 1024], pl.INT8]], temporary_25: pl.InOut[pl.Tensor[[128, 32768], pl.INT32]], arg_wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ], arg_freqs_cos: pl.Tensor[[96, 64], pl.FP32], arg_wq_b_scale: pl.Tensor[[32768], pl.FP32], view_40: pl.InOut[pl.Tensor[[96, 32768], pl.BF16]], temporary_26: pl.InOut[pl.Tensor[[96, 512], pl.FP32]], arg_wkv: pl.Tensor[[4096, 512], pl.BF16], temporary_17: pl.InOut[pl.Tensor[[96, 512], pl.BF16]], arg_gamma_ckv: pl.Tensor[[512], pl.BF16], view_45: pl.InOut[pl.Tensor[[4640, 512], pl.BF16]], arg_ori_slot_mapping: pl.Tensor[[96, 2], pl.INT32], temporary_29: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], temporary_30: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], arg_cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], arg_cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], temporary_28: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], arg_position_ids: pl.Tensor[[96], pl.INT64], arg_cmp_ape: pl.Tensor[[4, 1024], pl.FP32], arg_state_block_table: pl.Tensor[[16, 65540], pl.INT32], arg_compress_state: pl.InOut[pl.Tensor[[145, 8192], pl.FP32]], arg_state_slot_mapping: pl.Tensor[[96, 2], pl.INT32], arg_cmp_freqs_cos: pl.Tensor[[40, 64], pl.FP32], arg_cmp_freqs_sin: pl.Tensor[[40, 64], pl.FP32], temporary_31: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], view_60: pl.Tensor[[1, 512], pl.BF16], view_61: pl.InOut[pl.Tensor[[525344, 512], pl.BF16]], temporary_27: pl.InOut[pl.Tensor[[96, 512], pl.FP32]], arg_cmp_slot_mapping: pl.Tensor[[40, 2], pl.INT32], temporary_34: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], temporary_35: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], arg_inner_wkv: pl.Tensor[[256, 4096], pl.BF16], arg_inner_wgate: pl.Tensor[[256, 4096], pl.BF16], temporary_36: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_inner_ape: pl.Tensor[[4, 256], pl.FP32], arg_inner_state_block_table: pl.Tensor[[16, 65540], pl.INT32], arg_inner_compress_state: pl.InOut[pl.Tensor[[145, 1040], pl.FP32]], arg_inner_state_slot_mapping: pl.Tensor[[96, 2], pl.INT32], temporary_33: pl.InOut[pl.Tensor[[384, 128], pl.BF16]], arg_inner_freqs_cos: pl.Tensor[[40, 64], pl.FP32], arg_inner_freqs_sin: pl.Tensor[[40, 64], pl.FP32], view_76: pl.Tensor[[1, 128], pl.BF16], temporary_38: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_hadamard_idx: pl.Tensor[[128, 128], pl.BF16], temporary_37: pl.InOut[pl.Tensor[[384, 1], pl.FP32]], arg_idx_native_kv_cache: pl.InOut[pl.Tensor[[16417, 4160], pl.INT8]], temporary_32: pl.InOut[pl.Tensor[[96, 128], pl.FP32]], arg_idx_slot_mapping: pl.Tensor[[40, 2], pl.INT32], temporary_42: pl.InOut[pl.Tensor[[384, 8192], pl.INT32]], arg_idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], view_85: pl.InOut[pl.Tensor[[384, 8192], pl.BF16]], arg_idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], temporary_43: pl.InOut[pl.Tensor[[6144, 128], pl.FP32]], temporary_41: pl.InOut[pl.Tensor[[24576, 128], pl.BF16]], temporary_39: pl.InOut[pl.Tensor[[24576, 128], pl.INT8]], temporary_40: pl.InOut[pl.Tensor[[24576, 1], pl.FP32]], temporary_45: pl.InOut[pl.Tensor[[1536, 64], pl.FP32]], arg_weights_proj: pl.Tensor[[4096, 64], pl.BF16], temporary_44: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], view_94: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_48: pl.InOut[pl.Tensor[[1024, 384], pl.FP16]], temporary_50: pl.InOut[pl.Tensor[[256], pl.INT64]], arg_idx_block_table: pl.Tensor[[16, 1026], pl.INT32], view_98: pl.InOut[pl.Tensor[[533552, 128], pl.INT8]], view_99: pl.InOut[pl.Tensor[[533552, 128], pl.INT8]], temporary_49: pl.InOut[pl.Tensor[[288, 1024], pl.FP32]], temporary_46: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32]], arg_idx_topk_scores: pl.InOut[pl.Tensor[[96, 512], pl.FP32]], arg_idx_topk: pl.InOut[pl.Tensor[[96, 512], pl.INT32]], temporary_53: pl.InOut[pl.Tensor[[96, 512], pl.INT32]], temporary_52: pl.InOut[pl.Tensor[[96, 640], pl.FP32]], view_106: pl.Tensor[[96, 1], pl.INT64], temporary_54: pl.InOut[pl.Tensor[[96, 16], pl.INT32]], arg_ori_block_table: pl.Tensor[[16, 4098], pl.INT32], temporary_55: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_62: pl.InOut[pl.Tensor[[256], pl.INT64]], view_111: pl.InOut[pl.Tensor[[6144, 512], pl.BF16]], temporary_56: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_57: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], temporary_58: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], temporary_59: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], view_116: pl.Tensor[[64, 1], pl.FP32], arg_cmp_block_table: pl.Tensor[[16, 1026], pl.INT32], temporary_60: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_61: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], arg_wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ], temporary_63: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_65: pl.InOut[pl.Tensor[[8, 384], pl.FP32]], temporary_64: pl.InOut[pl.Tensor[[384, 8192], pl.INT8]], arg_wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ], temporary_66: pl.InOut[pl.Tensor[[384, 32768], pl.INT32]], arg_wo_b_scale: pl.Tensor[[4096], pl.FP32], view_128: pl.InOut[pl.Tensor[[96, 16384], pl.BF16]], progress: pl.InOut[pl.Tensor[[4608], pl.INT64]], records: pl.Tensor[[36288], pl.INT64], waits: pl.Tensor[[33112], pl.INT64]) -> tuple[pl.Tensor[[96, 1], pl.FP32], pl.Tensor[[8, 16384], pl.FP32], pl.Tensor[[96, 16384], pl.FP32], pl.Tensor[[384, 32], pl.FP32], pl.Tensor[[96, 32], pl.FP32], pl.Tensor[[96, 4], pl.FP32], pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[96, 8], pl.FP32], pl.Tensor[[96, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32], pl.Tensor[[96, 4096], pl.BF16], pl.Tensor[[96, 4096], pl.BF16], pl.Tensor[[8, 4096], pl.BF16], pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.INT32], pl.Tensor[[96, 64], pl.FP32], pl.Tensor[[96, 64], pl.FP32], pl.Tensor[[96, 64], pl.INT32], pl.Tensor[[96, 1024], pl.FP32], pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[96, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[96, 1024], pl.INT8], pl.Tensor[[128, 32768], pl.INT32], pl.Tensor[[96, 32768], pl.BF16], pl.Tensor[[96, 512], pl.FP32], pl.Tensor[[96, 512], pl.BF16], pl.Tensor[[4640, 512], pl.BF16], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[145, 8192], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[525344, 512], pl.BF16], pl.Tensor[[96, 512], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[145, 1040], pl.FP32], pl.Tensor[[384, 128], pl.BF16], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[16417, 4160], pl.INT8], pl.Tensor[[96, 128], pl.FP32], pl.Tensor[[384, 8192], pl.INT32], pl.Tensor[[384, 8192], pl.BF16], pl.Tensor[[6144, 128], pl.FP32], pl.Tensor[[24576, 128], pl.BF16], pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Tensor[[1536, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[1024, 384], pl.FP16], pl.Tensor[[256], pl.INT64], pl.Tensor[[533552, 128], pl.INT8], pl.Tensor[[533552, 128], pl.INT8], pl.Tensor[[288, 1024], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Tensor[[96, 512], pl.FP32], pl.Tensor[[96, 512], pl.INT32], pl.Tensor[[96, 512], pl.INT32], pl.Tensor[[96, 640], pl.FP32], pl.Tensor[[96, 16], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[256], pl.INT64], pl.Tensor[[6144, 512], pl.BF16], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8], pl.Tensor[[384, 32768], pl.INT32], pl.Tensor[[96, 16384], pl.BF16], pl.Tensor[[4608], pl.INT64]]:
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
    record_0 = 17280 + core_id * 6
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
        stage_0_v6: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        stage_0_v7: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_cmp_seq_lens, 0)
        for (stage_0_v10, (stage_0_v24,)) in pl.range(stage_0_v7, init_values=(stage_0_v6,)):
            stage_0_v11: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_query_start_loc, [stage_0_v10])
            stage_0_v12: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_query_start_loc, [stage_0_v10 + 1])
            stage_0_v13: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_seq_lens, [stage_0_v10])
            stage_0_v14: pl.Scalar[pl.INT32] = stage_0_v13 + stage_0_v11 - stage_0_v12
            pl.tensor.write(temporary_14, [stage_0_v10], pl.cast(stage_0_v24 - pl.cast(stage_0_v14, pl.INDEX) // 4 - 1, pl.INT32))
            stage_0_v15: pl.Scalar[pl.INDEX] = stage_0_v24 + pl.cast(stage_0_v13, pl.INDEX) // 4 - pl.cast(stage_0_v14, pl.INDEX) // 4
            stage_0_v16: pl.Scalar[pl.INDEX] = pl.yield_(stage_0_v15)
        stage_0_v8: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        stage_0_v9: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_kv_seq_lens, 0)
        for (stage_0_v17, (stage_0_v25,)) in pl.range(stage_0_v9, init_values=(stage_0_v8,)):
            stage_0_v18: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_query_start_loc, [stage_0_v17])
            stage_0_v19: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_query_start_loc, [stage_0_v17 + 1])
            stage_0_v20: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_0_v17])
            stage_0_v21: pl.Scalar[pl.INT32] = stage_0_v20 + stage_0_v18 - stage_0_v19
            pl.tensor.write(temporary_15, [stage_0_v17], pl.cast(stage_0_v25 - pl.cast(stage_0_v21, pl.INDEX) // 4 - 1, pl.INT32))
            stage_0_v22: pl.Scalar[pl.INDEX] = stage_0_v25 + pl.cast(stage_0_v20, pl.INDEX) // 4 - pl.cast(stage_0_v21, pl.INDEX) // 4
            stage_0_v23: pl.Scalar[pl.INDEX] = pl.yield_(stage_0_v22)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_0 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_0)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_1 = 17712 + core_id * 6
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
        stage_1_v5: pl.Scalar[pl.INDEX] = block_1
        for (stage_1_v6, (stage_1_v16, stage_1_v17, stage_1_v18)) in pl.range(stage_1_v5, 12, pl.min(12, 48), init_values=(temporary_4, temporary_5, view_2)):
            stage_1_v7: pl.Scalar[pl.INDEX] = stage_1_v6 * 8
            stage_1_v8: pl.Scalar[pl.INDEX] = pl.min(96 - stage_1_v7, 8)
            stage_1_v9: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
            for (stage_1_v19, (stage_1_v26, stage_1_v27, stage_1_v28)) in pl.pipeline(32, stage=4, init_values=(stage_1_v9, stage_1_v17, stage_1_v18)):
                stage_1_v20: pl.Scalar[pl.INDEX] = stage_1_v19 * 512
                stage_1_v21: pl.Tile[[8, 512], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_1_v8, 512])] = pl.tile.load(view_3, [stage_1_v7, stage_1_v20], [8, 512], [stage_1_v8, 512], target_memory=pl.Mem.Vec)
                stage_1_v22: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_1_v8, 512])] = pl.tile.cast(stage_1_v21, target_type=pl.FP32, mode='round')
                if stage_1_v8 == 8:
                    stage_1_v29 = pl.tile.store(stage_1_v22, [stage_1_v7, stage_1_v20], stage_1_v28)
                    stage_1_v30: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_1_v8, 512])] = pl.tile.mul(stage_1_v22, stage_1_v22)
                    stage_1_v31: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_1_v32: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_1_v8, 1])] = pl.tile.row_sum(stage_1_v30, stage_1_v31)
                    stage_1_v33: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_1_v8])] = pl.tile.reshape(stage_1_v32, [1, 8])
                    stage_1_v34: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_1_v26, stage_1_v33)
                    (stage_1_v47, stage_1_v48, stage_1_v49) = pl.yield_(stage_1_v34, stage_1_v27, stage_1_v29)
                else:
                    stage_1_v35: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_1_v8, 512])] = pl.tile.set_validshape(stage_1_v22, stage_1_v8, 512)
                    stage_1_v36: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_1_v35, pad_value=pl.PadValue.zero)
                    stage_1_v37: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.mul(stage_1_v36, stage_1_v36)
                    stage_1_v38: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_1_v39: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_1_v37, stage_1_v38)
                    stage_1_v40: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_1_v39, [1, 8])
                    stage_1_v41: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_1_v26, stage_1_v40)
                    stage_1_v42 = pl.tile.store(stage_1_v22, [0, stage_1_v20], stage_1_v27)
                    stage_1_v43: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_1_v8, 512])] = pl.tile.load(stage_1_v42, [0, stage_1_v20], [8, 512], [stage_1_v8, 512], target_memory=pl.Mem.Vec)
                    pl.tile.store(stage_1_v43, [stage_1_v7, stage_1_v20], stage_1_v28)
                    (stage_1_v47, stage_1_v48, stage_1_v49) = pl.yield_(stage_1_v41, stage_1_v42, stage_1_v28)
                (stage_1_v44, stage_1_v45, stage_1_v46) = pl.yield_(stage_1_v47, stage_1_v48, stage_1_v49)
            stage_1_v10: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_1_v44, 6.103515625e-05)
            stage_1_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_1_v10, 1e-06)
            stage_1_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_1_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(stage_1_v11, stage_1_v12)
            stage_1_v14: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_1_v13, [8, 1])
            stage_1_v15 = pl.tile.store(stage_1_v14, [stage_1_v7, 0], stage_1_v16)
            (stage_1_v23, stage_1_v24, stage_1_v25) = pl.yield_(stage_1_v15, stage_1_v45, stage_1_v46)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_1 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_1)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_2 = 18144 + core_id * 6
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
        for (stage_2_v20, (stage_2_v23, stage_2_v24)) in pl.range(stage_2_v4, 12, pl.min(12, 48), init_values=(temporary_20, temporary_21)):
            stage_2_v21: pl.Scalar[pl.INDEX] = stage_2_v20 * 8
            stage_2_v22: pl.Scalar[pl.INDEX] = pl.min(96 - stage_2_v21, 8)
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
    record_3 = 18576 + core_id * 6
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
        for stage_3_v4 in pl.range(stage_3_v3, 16, pl.min(16, 16)):
            stage_3_v5: pl.Scalar[pl.INDEX] = stage_3_v4 * 6
            stage_3_v6: pl.Scalar[pl.INDEX] = pl.min(96 - stage_3_v5, 6)
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
    record_4 = 19008 + core_id * 6
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
        for stage_4_v4 in pl.range(stage_4_v3, 24, pl.min(24, 16)):
            stage_4_v5: pl.Scalar[pl.INDEX] = stage_4_v4 * 4
            stage_4_v6: pl.Scalar[pl.INDEX] = pl.min(96 - stage_4_v5, 4)
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
    record_5 = 19440 + core_id * 6
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
            stage_5_v8: pl.Scalar[pl.INDEX] = stage_5_v7 * 96 + stage_5_v4
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
    record_6 = 19872 + core_id * 6
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
        stage_6_v8: pl.Scalar[pl.INDEX] = block_6
        stage_6_v9: pl.Scalar[pl.FP32] = pl.tensor.read(arg_hc_attn_scale, [0])
        stage_6_v10: pl.Scalar[pl.FP32] = pl.tensor.read(arg_hc_attn_scale, [1])
        for (stage_6_v11, (stage_6_v40, stage_6_v41, stage_6_v42)) in pl.range(stage_6_v8, 12, pl.min(12, 16), init_values=(temporary_0, temporary_9, temporary_6)):
            stage_6_v12: pl.Scalar[pl.INDEX] = stage_6_v11 * 8
            stage_6_v13: pl.Scalar[pl.INDEX] = pl.min(96 - stage_6_v12, 8)
            stage_6_v14: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_4, [stage_6_v12, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
            stage_6_v15: pl.Tile[[8], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_hc_attn_base, [0], [8], [8], target_memory=pl.Mem.Vec)
            stage_6_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_6_v15, [1, 8])
            stage_6_v17: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_8, [stage_6_v12, 0], [8, 8], [8, 8], target_memory=pl.Mem.Vec)
            stage_6_v18: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_6_v17, stage_6_v14)
            stage_6_v19: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_6_v18, stage_6_v9)
            stage_6_v20: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand(stage_6_v19, stage_6_v16)
            stage_6_v21: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_6_v19, stage_6_v20)
            stage_6_v22: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_6_v21)
            stage_6_v23: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_6_v22)
            stage_6_v24: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_6_v23, 1.0)
            stage_6_v25: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_6_v24)
            stage_6_v26: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_6_v25, 1e-06)
            stage_6_v27 = pl.tile.store(stage_6_v26, [stage_6_v12, 0], stage_6_v42)
            stage_6_v28: pl.Tile[[8], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_hc_attn_base, [4], [8], [8], target_memory=pl.Mem.Vec)
            stage_6_v29: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_6_v28, [1, 8])
            stage_6_v30: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_8, [stage_6_v12, 4], [8, 8], [8, 8], target_memory=pl.Mem.Vec)
            stage_6_v31: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_6_v30, stage_6_v14)
            stage_6_v32: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_6_v31, stage_6_v10)
            stage_6_v33: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand(stage_6_v32, stage_6_v29)
            stage_6_v34: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_6_v32, stage_6_v33)
            stage_6_v35: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_6_v34)
            stage_6_v36: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_6_v35)
            stage_6_v37: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_6_v36, 1.0)
            stage_6_v38: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_6_v37)
            stage_6_v39: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_6_v38, 2.0)
            if stage_6_v13 == 8:
                stage_6_v43: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4])] = pl.tile.slice(stage_6_v39, [8, 8], [0, 0], [8, 4])
                stage_6_v44 = pl.tile.store(stage_6_v43, [stage_6_v12, 0], stage_6_v40)
                (stage_6_v50, stage_6_v51) = pl.yield_(stage_6_v44, stage_6_v41)
            else:
                stage_6_v45 = pl.tile.store(stage_6_v39, [0, 0], stage_6_v41)
                stage_6_v46: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v13, 4])] = pl.tile.load(stage_6_v45, [0, 0], [8, 8], [stage_6_v13, 4], target_memory=pl.Mem.Vec)
                pl.tile.store(stage_6_v46, [stage_6_v12, 0], stage_6_v40)
                (stage_6_v50, stage_6_v51) = pl.yield_(stage_6_v40, stage_6_v45)
            (stage_6_v47, stage_6_v48, stage_6_v49) = pl.yield_(stage_6_v50, stage_6_v51, stage_6_v27)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_6 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_6)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_7 = 20304 + core_id * 6
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
        stage_7_v7: pl.Scalar[pl.INDEX] = block_7
        stage_7_v8: pl.Scalar[pl.FP32] = pl.tensor.read(arg_hc_attn_scale, [2])
        stage_7_v9: pl.Scalar[pl.INDEX] = stage_7_v7 * 8
        stage_7_v10: pl.Scalar[pl.INDEX] = pl.min(96 - stage_7_v9, 8)
        stage_7_v11: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 1])] = pl.tile.load(temporary_4, [stage_7_v9, 0], [8, 1], [stage_7_v10, 1], target_memory=pl.Mem.Vec)
        stage_7_v12: pl.Scalar[pl.INDEX] = 8
        stage_7_v13: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.load(temporary_8, [stage_7_v9, stage_7_v12], [8, 8], [stage_7_v10, 4], target_memory=pl.Mem.Vec)
        stage_7_v14: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.load(temporary_8, [stage_7_v9, stage_7_v12 + 4], [8, 8], [stage_7_v10, 4], target_memory=pl.Mem.Vec)
        stage_7_v15: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.load(temporary_8, [stage_7_v9, stage_7_v12 + 8], [8, 8], [stage_7_v10, 4], target_memory=pl.Mem.Vec)
        stage_7_v16: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.load(temporary_8, [stage_7_v9, stage_7_v12 + 12], [8, 8], [stage_7_v10, 4], target_memory=pl.Mem.Vec)
        stage_7_v17: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(view_12, [0, stage_7_v12], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        stage_7_v18: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(view_12, [0, stage_7_v12 + 4], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        stage_7_v19: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(view_12, [0, stage_7_v12 + 8], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        stage_7_v20: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(view_12, [0, stage_7_v12 + 12], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        stage_7_v21: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.row_expand_mul(stage_7_v13, stage_7_v11)
        stage_7_v22: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.muls(stage_7_v21, stage_7_v8)
        stage_7_v23: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.col_expand(stage_7_v13, stage_7_v17)
        stage_7_v24: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.add(stage_7_v22, stage_7_v23)
        stage_7_v25: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.row_expand_mul(stage_7_v14, stage_7_v11)
        stage_7_v26: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.muls(stage_7_v25, stage_7_v8)
        stage_7_v27: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.col_expand(stage_7_v14, stage_7_v18)
        stage_7_v28: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.add(stage_7_v26, stage_7_v27)
        stage_7_v29: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.row_expand_mul(stage_7_v15, stage_7_v11)
        stage_7_v30: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.muls(stage_7_v29, stage_7_v8)
        stage_7_v31: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.col_expand(stage_7_v15, stage_7_v19)
        stage_7_v32: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.add(stage_7_v30, stage_7_v31)
        stage_7_v33: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.row_expand_mul(stage_7_v16, stage_7_v11)
        stage_7_v34: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.muls(stage_7_v33, stage_7_v8)
        stage_7_v35: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.col_expand(stage_7_v16, stage_7_v20)
        stage_7_v36: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.add(stage_7_v34, stage_7_v35)
        stage_7_v37: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_7_v24, pad_value=pl.PadValue.min)
        stage_7_v38: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_7_v28, pad_value=pl.PadValue.min)
        stage_7_v39: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_7_v32, pad_value=pl.PadValue.min)
        stage_7_v40: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_7_v36, pad_value=pl.PadValue.min)
        stage_7_v41: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_7_v42: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_7_v43: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_7_v37, stage_7_v41)
        stage_7_v44: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_7_v38, stage_7_v41)
        stage_7_v45: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_7_v39, stage_7_v41)
        stage_7_v46: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_7_v40, stage_7_v41)
        stage_7_v47: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(stage_7_v37, stage_7_v43)
        stage_7_v48: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(stage_7_v47)
        stage_7_v49: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(stage_7_v38, stage_7_v44)
        stage_7_v50: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(stage_7_v49)
        stage_7_v51: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(stage_7_v39, stage_7_v45)
        stage_7_v52: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(stage_7_v51)
        stage_7_v53: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(stage_7_v40, stage_7_v46)
        stage_7_v54: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(stage_7_v53)
        stage_7_v55: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_7_v48, stage_7_v42)
        stage_7_v56: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_7_v50, stage_7_v42)
        stage_7_v57: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_7_v52, stage_7_v42)
        stage_7_v58: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_7_v54, stage_7_v42)
        stage_7_v59: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(stage_7_v48, stage_7_v55)
        stage_7_v60: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(stage_7_v59, 1e-06)
        stage_7_v61: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(stage_7_v50, stage_7_v56)
        stage_7_v62: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(stage_7_v61, 1e-06)
        stage_7_v63: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(stage_7_v52, stage_7_v57)
        stage_7_v64: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(stage_7_v63, 1e-06)
        stage_7_v65: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(stage_7_v54, stage_7_v58)
        stage_7_v66: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(stage_7_v65, 1e-06)
        stage_7_v67: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_7_v60, 8, 4)
        stage_7_v68: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_7_v62, 8, 4)
        stage_7_v69: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_7_v64, 8, 4)
        stage_7_v70: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_7_v66, 8, 4)
        stage_7_v71: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_7_v67, pad_value=pl.PadValue.zero)
        stage_7_v72: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_7_v68, pad_value=pl.PadValue.zero)
        stage_7_v73: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_7_v69, pad_value=pl.PadValue.zero)
        stage_7_v74: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_7_v70, pad_value=pl.PadValue.zero)
        stage_7_v75: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_7_v76: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_7_v71, stage_7_v72)
        stage_7_v77: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_7_v73, stage_7_v74)
        stage_7_v78: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_7_v76, stage_7_v77)
        stage_7_v79: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(stage_7_v78, 1e-06)
        stage_7_v80: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_7_v71, stage_7_v79)
        stage_7_v81: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_7_v72, stage_7_v79)
        stage_7_v82: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_7_v73, stage_7_v79)
        stage_7_v83: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_7_v74, stage_7_v79)
        for (stage_7_v84, (stage_7_v133, stage_7_v134, stage_7_v135, stage_7_v136)) in pl.pipeline(19, stage=2, init_values=(stage_7_v80, stage_7_v81, stage_7_v82, stage_7_v83)):
            stage_7_v85: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_7_v133, stage_7_v75)
            stage_7_v86: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_7_v85, [1, 8])
            stage_7_v87: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_7_v86, 1e-06)
            stage_7_v88: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_7_v87, [8, 1])
            stage_7_v89: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_7_v134, stage_7_v75)
            stage_7_v90: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_7_v89, [1, 8])
            stage_7_v91: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_7_v90, 1e-06)
            stage_7_v92: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_7_v91, [8, 1])
            stage_7_v93: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_7_v135, stage_7_v75)
            stage_7_v94: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_7_v93, [1, 8])
            stage_7_v95: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_7_v94, 1e-06)
            stage_7_v96: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_7_v95, [8, 1])
            stage_7_v97: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_7_v136, stage_7_v75)
            stage_7_v98: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_7_v97, [1, 8])
            stage_7_v99: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_7_v98, 1e-06)
            stage_7_v100: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_7_v99, [8, 1])
            stage_7_v101: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(stage_7_v133, stage_7_v88)
            stage_7_v102: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(stage_7_v134, stage_7_v92)
            stage_7_v103: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(stage_7_v135, stage_7_v96)
            stage_7_v104: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(stage_7_v136, stage_7_v100)
            stage_7_v105: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_7_v101, stage_7_v102)
            stage_7_v106: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_7_v103, stage_7_v104)
            stage_7_v107: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_7_v105, stage_7_v106)
            stage_7_v108: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(stage_7_v107, 1e-06)
            stage_7_v109: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_7_v101, stage_7_v108)
            stage_7_v110: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_7_v102, stage_7_v108)
            stage_7_v111: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_7_v103, stage_7_v108)
            stage_7_v112: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_7_v104, stage_7_v108)
            (stage_7_v137, stage_7_v138, stage_7_v139, stage_7_v140) = pl.yield_(stage_7_v109, stage_7_v110, stage_7_v111, stage_7_v112)
        if stage_7_v10 == 8:
            stage_7_v113: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(stage_7_v137, 8, 4)
            stage_7_v114: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(stage_7_v138, 8, 4)
            stage_7_v115: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(stage_7_v139, 8, 4)
            stage_7_v116: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(stage_7_v140, 8, 4)
            stage_7_v117 = pl.tile.store(stage_7_v113, [stage_7_v9, 0], temporary_1)
            stage_7_v118 = pl.tile.store(stage_7_v114, [stage_7_v9, 4], temporary_1)
            stage_7_v119 = pl.tile.store(stage_7_v115, [stage_7_v9, 8], temporary_1)
            stage_7_v120 = pl.tile.store(stage_7_v116, [stage_7_v9, 12], temporary_1)
        else:
            stage_7_v121 = pl.tile.store(stage_7_v137, [0, 0], temporary_10)
            stage_7_v122 = pl.tile.store(stage_7_v138, [0, 8], temporary_10)
            stage_7_v123 = pl.tile.store(stage_7_v139, [0, 16], temporary_10)
            stage_7_v124 = pl.tile.store(stage_7_v140, [0, 24], temporary_10)
            stage_7_v125: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.load(temporary_10, [0, 0], [8, 8], [stage_7_v10, 4], target_memory=pl.Mem.Vec)
            stage_7_v126: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.load(temporary_10, [0, 8], [8, 8], [stage_7_v10, 4], target_memory=pl.Mem.Vec)
            stage_7_v127: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.load(temporary_10, [0, 16], [8, 8], [stage_7_v10, 4], target_memory=pl.Mem.Vec)
            stage_7_v128: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_7_v10, 4])] = pl.tile.load(temporary_10, [0, 24], [8, 8], [stage_7_v10, 4], target_memory=pl.Mem.Vec)
            stage_7_v129 = pl.tile.store(stage_7_v125, [stage_7_v9, 0], temporary_1)
            stage_7_v130 = pl.tile.store(stage_7_v126, [stage_7_v9, 4], temporary_1)
            stage_7_v131 = pl.tile.store(stage_7_v127, [stage_7_v9, 8], temporary_1)
            stage_7_v132 = pl.tile.store(stage_7_v128, [stage_7_v9, 12], temporary_1)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_7 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_7)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_8 = 20736 + core_id * 6
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
        stage_8_v9: pl.Scalar[pl.INDEX] = pl.min(96 - stage_8_v8, 8)
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
    record_9 = 21168 + core_id * 6
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
        stage_9_v7: pl.Scalar[pl.INDEX] = block_9
        stage_9_v8: pl.Tile[[1, 1024], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_31, [0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
        stage_9_v9: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_9_v8, target_type=pl.FP32, mode='round')
        for stage_9_v10 in pl.range(stage_9_v7, 12, 12):
            stage_9_v11: pl.Scalar[pl.INDEX] = stage_9_v10 * 8
            stage_9_v12: pl.Scalar[pl.INDEX] = pl.min(96 - stage_9_v11, 8)
            stage_9_v13: pl.Scalar[pl.INDEX] = 0 + stage_9_v11
            stage_9_v14: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_24, [stage_9_v11, 0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_9_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
            stage_9_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
            for (stage_9_v32, (stage_9_v58, stage_9_v59)) in pl.pipeline(0, 1024, 256, stage=2, init_values=(stage_9_v16, stage_9_v15)):
                stage_9_v33: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_9_v14, 0, stage_9_v32, [8, 256], target_memory=pl.Mem.Vec)
                stage_9_v34: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_9_v33, stage_9_v33)
                stage_9_v35: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_9_v36: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_9_v34, stage_9_v35)
                stage_9_v37: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_9_v36, [1, 8])
                stage_9_v38: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_9_v59, stage_9_v37)
                stage_9_v39: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_9_v9, 0, stage_9_v32, [1, 256], target_memory=pl.Mem.Vec)
                stage_9_v40: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_9_v33, stage_9_v39)
                stage_9_v41: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_9_v40)
                stage_9_v42: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_9_v43: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_9_v41, stage_9_v42)
                stage_9_v44: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_9_v43, [1, 8])
                stage_9_v45: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_9_v58, stage_9_v44)
                (stage_9_v60, stage_9_v61) = pl.yield_(stage_9_v45, stage_9_v38)
            stage_9_v17: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_9_v61, 0.0009765625)
            stage_9_v18: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_9_v17, 1e-06)
            stage_9_v19: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_9_v20: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(stage_9_v18, stage_9_v19)
            stage_9_v21: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_9_v20, [8, 1])
            stage_9_v22: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_9_v23: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_9_v20, stage_9_v60)
            stage_9_v24: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_9_v22, stage_9_v23)
            stage_9_v25: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_9_v26: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_9_v25, stage_9_v24)
            stage_9_v27: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_9_v26, [8, 1])
            stage_9_v28: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_9_v26)
            stage_9_v29: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_9_v28, [8, 1])
            stage_9_v30 = pl.tile.store(stage_9_v29, [stage_9_v11, 0], temporary_23)
            if stage_9_v12 == 8:
                stage_9_v46 = pl.tile.store(stage_9_v29, [stage_9_v13, 0], temporary_19)
            else:
                stage_9_v47: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_9_v12, 1])] = pl.tile.slice(stage_9_v29, [8, 1], [0, 0], valid_shape=[stage_9_v12, 1])
                stage_9_v48 = pl.tile.store(stage_9_v47, [stage_9_v13, 0], temporary_19)
            for stage_9_v31 in pl.pipeline(0, 1024, 256, stage=2):
                stage_9_v49: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_9_v14, 0, stage_9_v31, [8, 256], target_memory=pl.Mem.Vec)
                stage_9_v50: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_9_v9, 0, stage_9_v31, [1, 256], target_memory=pl.Mem.Vec)
                stage_9_v51: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_9_v49, stage_9_v21)
                stage_9_v52: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_9_v51, stage_9_v50)
                stage_9_v53: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_9_v52, stage_9_v27)
                stage_9_v54: pl.Tile[[8, 256], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_9_v53, target_type=pl.INT32, mode='rint')
                stage_9_v55: pl.Tile[[8, 256], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_9_v54, target_type=pl.FP16, mode='round')
                stage_9_v56: pl.Tile[[8, 256], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_9_v55, target_type=pl.INT8, mode='trunc')
                stage_9_v57 = pl.tile.store(stage_9_v56, [stage_9_v11, stage_9_v31], temporary_22)
                if stage_9_v12 == 8:
                    stage_9_v62 = pl.tile.store(stage_9_v56, [stage_9_v13, stage_9_v31], temporary_18)
                else:
                    stage_9_v63: pl.Tile[[8, 256], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_9_v12, 256])] = pl.tile.set_validshape(stage_9_v56, stage_9_v12, 256)
                    stage_9_v64 = pl.tile.store(stage_9_v63, [stage_9_v13, stage_9_v31], temporary_18)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_9 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_9)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_10 = 21600 + core_id * 6
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
        stage_10_v3: pl.Scalar[pl.INDEX] = block_10
        stage_10_v4: pl.Scalar[pl.INDEX] = stage_10_v3 * 16
        stage_10_v5: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_45, [stage_10_v4, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_10_v6: pl.Scalar[pl.INDEX] = stage_10_v4 + 384
        stage_10_v7: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_45, [stage_10_v6, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_10_v8: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_10_v5, stage_10_v7)
        stage_10_v9: pl.Scalar[pl.INDEX] = stage_10_v4 + 768
        stage_10_v10: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_45, [stage_10_v9, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_10_v11: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_10_v8, stage_10_v10)
        stage_10_v12: pl.Scalar[pl.INDEX] = stage_10_v4 + 1152
        stage_10_v13: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_45, [stage_10_v12, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_10_v14: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_10_v11, stage_10_v13)
        stage_10_v15: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_10_v14, 0.011048543456039806)
        stage_10_v16 = pl.tile.store(stage_10_v15, [stage_10_v4, 0], temporary_44)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_10 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_10)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_11 = 22032 + core_id * 6
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
        stage_11_v7: pl.Scalar[pl.INDEX] = block_11
        stage_11_v8: pl.Scalar[pl.INDEX] = stage_11_v7 * 32
        stage_11_v9: pl.Scalar[pl.INDEX] = pl.min(96 - stage_11_v8, 32)
        stage_11_v10: pl.Scalar[pl.INDEX] = 0 + stage_11_v8
        if stage_11_v9 == 32:
            stage_11_v11: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=0.0)
            for (stage_11_v85, (stage_11_v121,)) in pl.pipeline(0, 512, 64, stage=2, init_values=(stage_11_v11,)):
                stage_11_v86: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_26, [stage_11_v8, stage_11_v85], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
                stage_11_v87: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_11_v86, stage_11_v86)
                stage_11_v88: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_11_v121, stage_11_v87)
                stage_11_v89: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_11_v88)
            stage_11_v12: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_11_v13: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_11_v89, stage_11_v12)
            stage_11_v14: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_11_v13, [1, 32])
            stage_11_v15: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_11_v14, 0.001953125)
            stage_11_v16: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_11_v15, 1e-06)
            stage_11_v17: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_11_v16)
            stage_11_v18: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=1.0)
            stage_11_v19: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_11_v18, stage_11_v17)
            stage_11_v20: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_11_v19, [32, 1])
            for (stage_11_v90, (stage_11_v122,)) in pl.pipeline(0, 448, 64, stage=2, init_values=(temporary_17,)):
                stage_11_v91: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_26, [stage_11_v8, stage_11_v90], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
                stage_11_v92: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_gamma_ckv, [stage_11_v90], [64], [64], target_memory=pl.Mem.Vec)
                stage_11_v93: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_11_v92, target_type=pl.FP32, mode='round')
                stage_11_v94: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_11_v93, [1, 64])
                stage_11_v95: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_11_v91, stage_11_v20)
                stage_11_v96: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_11_v95, stage_11_v94)
                stage_11_v97: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_11_v96, target_type=pl.BF16, mode='rint')
                stage_11_v98 = pl.tile.store(stage_11_v97, [stage_11_v10, stage_11_v90], stage_11_v122)
                stage_11_v99 = pl.yield_(stage_11_v98)
            stage_11_v21: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_gamma_ckv, [448], [64], [64], target_memory=pl.Mem.Vec)
            stage_11_v22: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_11_v21, target_type=pl.FP32, mode='round')
            stage_11_v23: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_11_v22, [1, 64])
            stage_11_v24: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_26, [stage_11_v8, 448], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
            stage_11_v25: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_11_v24, stage_11_v20)
            stage_11_v26: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_11_v25, stage_11_v23)
            stage_11_v27: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_11_v10, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
            stage_11_v28: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_20, [stage_11_v10, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
            stage_11_v29: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_21, [stage_11_v10, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
            stage_11_v30: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for (stage_11_v100, (stage_11_v123,)) in pl.range(32, init_values=(stage_11_v30,)):
                stage_11_v101: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_11_v26, [1, 64], [stage_11_v100, 0], [1, 64])
                stage_11_v102: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_11_v29, [1, 64], [stage_11_v100, 0], [1, 64])
                stage_11_v103: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                stage_11_v104: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_11_v101, stage_11_v102, stage_11_v103)
                stage_11_v105: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_11_v123, stage_11_v104, [stage_11_v100, 0])
                stage_11_v106: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_11_v105)
            stage_11_v31: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = stage_11_v106
            stage_11_v32: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_11_v26, stage_11_v27)
            stage_11_v33: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_11_v31, stage_11_v28)
            stage_11_v34: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_11_v32, stage_11_v33)
            stage_11_v35: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_11_v34, target_type=pl.BF16, mode='rint')
            stage_11_v36 = pl.tile.store(stage_11_v35, [stage_11_v10, 448], stage_11_v99)
        else:
            stage_11_v37: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_11_v38: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=0.0)
            for (stage_11_v107, (stage_11_v124,)) in pl.pipeline(0, 512, 64, stage=2, init_values=(stage_11_v38,)):
                stage_11_v108: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.load(temporary_26, [stage_11_v8, stage_11_v107], [32, 64], [stage_11_v9, 64], target_memory=pl.Mem.Vec)
                stage_11_v109: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.mul(stage_11_v108, stage_11_v108)
                stage_11_v110: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_11_v124, stage_11_v109)
                stage_11_v111: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_11_v110)
            stage_11_v39: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_11_v111, stage_11_v37)
            stage_11_v40: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_11_v39, [1, 32])
            stage_11_v41: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_11_v40, 0.001953125)
            stage_11_v42: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_11_v41, 1e-06)
            stage_11_v43: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_11_v42)
            stage_11_v44: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=1.0)
            stage_11_v45: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_11_v44, stage_11_v43)
            stage_11_v46: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_11_v45, [32, 1])
            for stage_11_v47 in pl.pipeline(0, 448, 64, stage=2):
                stage_11_v112: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.load(temporary_26, [stage_11_v8, stage_11_v47], [32, 64], [stage_11_v9, 64], target_memory=pl.Mem.Vec)
                stage_11_v113: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_gamma_ckv, [stage_11_v47], [64], [64], target_memory=pl.Mem.Vec)
                stage_11_v114: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_11_v113, target_type=pl.FP32, mode='round')
                stage_11_v115: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_11_v114, [1, 64])
                stage_11_v116: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.row_expand_mul(stage_11_v112, stage_11_v46)
                stage_11_v117: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.col_expand_mul(stage_11_v116, stage_11_v115)
                stage_11_v118: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.cast(stage_11_v117, target_type=pl.BF16, mode='rint')
                stage_11_v119: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.set_validshape(stage_11_v118, stage_11_v9, 64)
                stage_11_v120 = pl.tile.store(stage_11_v119, [stage_11_v10, stage_11_v47], temporary_17)
            stage_11_v48: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_gamma_ckv, [448], [64], [64], target_memory=pl.Mem.Vec)
            stage_11_v49: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_11_v48, target_type=pl.FP32, mode='round')
            stage_11_v50: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_11_v49, [1, 64])
            stage_11_v51: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.load(temporary_26, [stage_11_v8, 448], [32, 64], [stage_11_v9, 64], target_memory=pl.Mem.Vec)
            stage_11_v52: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.row_expand_mul(stage_11_v51, stage_11_v46)
            stage_11_v53: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.col_expand_mul(stage_11_v52, stage_11_v50)
            stage_11_v54: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.load(arg_freqs_cos, [stage_11_v10, 0], [32, 64], [stage_11_v9, 64], target_memory=pl.Mem.Vec)
            stage_11_v55: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.load(temporary_20, [stage_11_v10, 0], [32, 64], [stage_11_v9, 64], target_memory=pl.Mem.Vec)
            stage_11_v56: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=1.0)
            stage_11_v57: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            stage_11_v58: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_11_v57, target_type=pl.FP32, mode='round')
            stage_11_v59: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_11_v56, stage_11_v58)
            stage_11_v60: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_11_v59, 0.5)
            stage_11_v61: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_11_v60, target_type=pl.INT32, mode='trunc')
            stage_11_v62: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_11_v61, target_type=pl.FP32, mode='round')
            stage_11_v63: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_11_v62, 2.0)
            stage_11_v64: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_11_v59, stage_11_v63)
            stage_11_v65: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_11_v59, 1.0)
            stage_11_v66: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_11_v64, 2.0)
            stage_11_v67: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_11_v65, stage_11_v66)
            stage_11_v68: pl.Tile[[1, 32], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 32], dtype=pl.INT32, descending=False)
            stage_11_v69: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_11_v68, target_type=pl.FP32, mode='round')
            stage_11_v70: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_11_v69, 64.0)
            stage_11_v71: pl.Tile[[64, 32], pl.FP32, pl.Mem.Vec] = pl.tile.full([64, 32], dtype=pl.FP32, value=1.0)
            stage_11_v72: pl.Tile[[64, 32], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_11_v71, stage_11_v70)
            stage_11_v73: pl.Tile[[64, 32], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_11_v74: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(stage_11_v72, 0, 1, stage_11_v73)
            stage_11_v75: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_11_v67, stage_11_v74)
            stage_11_v76: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_11_v75, target_type=pl.INT32, mode='round')
            stage_11_v77: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            stage_11_v78: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_11_v53, stage_11_v76, stage_11_v77)
            stage_11_v79: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.mul(stage_11_v53, stage_11_v54)
            stage_11_v80: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_11_v78, stage_11_v55)
            stage_11_v81: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.add(stage_11_v79, stage_11_v80)
            stage_11_v82: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.cast(stage_11_v81, target_type=pl.BF16, mode='rint')
            stage_11_v83: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v9, 64])] = pl.tile.set_validshape(stage_11_v82, stage_11_v9, 64)
            stage_11_v84 = pl.tile.store(stage_11_v83, [stage_11_v10, 448], temporary_17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_11 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_11)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_12 = 22464 + core_id * 6
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
        stage_12_v4: pl.Scalar[pl.INDEX] = block_12
        for (stage_12_v5, (stage_12_v8,)) in pl.range(stage_12_v4, 12, 8, init_values=(view_45,)):
            stage_12_v6: pl.Scalar[pl.INDEX] = stage_12_v5 * 8
            for (stage_12_v9, (stage_12_v14,)) in pl.range(pl.min(96 - stage_12_v6, 8), init_values=(stage_12_v8,)):
                stage_12_v10: pl.Scalar[pl.INDEX] = stage_12_v6 + stage_12_v9
                stage_12_v11: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_slot_mapping, [stage_12_v10, 0])
                stage_12_v12: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_slot_mapping, [stage_12_v10, 1])
                if 0 <= pl.cast(stage_12_v11, pl.INDEX) and 0 <= pl.cast(stage_12_v12, pl.INDEX):
                    stage_12_v15: pl.Scalar[pl.INDEX] = pl.cast(stage_12_v11, pl.INDEX) * 32 + pl.cast(stage_12_v12, pl.INDEX)
                    stage_12_v16: pl.Tile[[1, 512], pl.BF16, pl.Mem.Vec] = pl.tile.load(temporary_17, [stage_12_v10, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    stage_12_v17 = pl.tile.store(stage_12_v16, [stage_12_v15, 0], stage_12_v14)
                    stage_12_v18 = pl.yield_(stage_12_v17)
                else:
                    stage_12_v18 = pl.yield_(stage_12_v14)
                stage_12_v13 = pl.yield_(stage_12_v18)
            stage_12_v7 = pl.yield_(stage_12_v13)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_12 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_12)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_13 = 22896 + core_id * 6
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
        stage_13_v8: pl.Scalar[pl.INDEX] = block_13
        for (stage_13_v9, (stage_13_v12,)) in pl.range(stage_13_v8, 16, 16, init_values=(temporary_36,)):
            stage_13_v10: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_13_v9 * 6])
            for (stage_13_v13, (stage_13_v19,)) in pl.range(6, init_values=(stage_13_v12,)):
                stage_13_v14: pl.Scalar[pl.INDEX] = stage_13_v9 * 6 + stage_13_v13
                stage_13_v15: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_13_v14])
                stage_13_v16: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
                stage_13_v17 = pl.tile.store(stage_13_v16, [stage_13_v14, 0], stage_13_v19)
                if (stage_13_v15 + 1) % 4 == 0:
                    stage_13_v20: pl.Scalar[pl.INT64] = stage_13_v15 - 8 + 1
                    for (stage_13_v22, (stage_13_v33,)) in pl.range(0, 128, 64, init_values=(stage_13_v17,)):
                        stage_13_v23: pl.Scalar[pl.INDEX] = pl.cast(stage_13_v15 % 4, pl.INDEX)
                        stage_13_v24: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_35, [stage_13_v14, stage_13_v22 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                        stage_13_v25: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_ape, [stage_13_v23, stage_13_v22 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                        stage_13_v26: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_13_v24, stage_13_v25)
                        stage_13_v27: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_13_v26, stage_13_v26)
                        stage_13_v28: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_13_v27)
                        stage_13_v29: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_34, [stage_13_v14, stage_13_v22 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                        for (stage_13_v34, (stage_13_v50, stage_13_v51, stage_13_v52)) in pl.range(7, init_values=(stage_13_v28, stage_13_v26, stage_13_v29)):
                            stage_13_v35: pl.Scalar[pl.INT64] = stage_13_v20 + stage_13_v34
                            stage_13_v36: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
                            stage_13_v37: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=-3.4028234663852886e+38)
                            stage_13_v38: pl.Scalar[pl.INDEX] = 0
                            if 4 <= stage_13_v34:
                                stage_13_v53: pl.Scalar[pl.INDEX] = 128
                                stage_13_v54: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v53)
                            else:
                                stage_13_v54: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v38)
                            if 0 <= stage_13_v35 and stage_13_v35 < stage_13_v10:
                                stage_13_v55: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
                                stage_13_v56: pl.Scalar[pl.INT32] = pl.tensor.read(arg_inner_state_block_table, [stage_13_v9, stage_13_v35 // 2])
                                stage_13_v57: pl.Scalar[pl.INDEX] = pl.cast(stage_13_v56, pl.INDEX)
                                if 0 <= stage_13_v57:
                                    stage_13_v61: pl.Scalar[pl.INDEX] = pl.cast(stage_13_v35 % 2, pl.INDEX) * 512 + stage_13_v54 + stage_13_v22
                                    stage_13_v62: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_compress_state, [stage_13_v57, stage_13_v61], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    stage_13_v63: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_compress_state, [stage_13_v57, stage_13_v61 + 256], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    (stage_13_v74, stage_13_v75) = pl.yield_(stage_13_v63, stage_13_v62)
                                else:
                                    (stage_13_v74, stage_13_v75) = pl.yield_(stage_13_v55, stage_13_v36)
                                (stage_13_v64, stage_13_v65) = pl.yield_(stage_13_v74, stage_13_v75)
                            else:
                                (stage_13_v64, stage_13_v65) = pl.yield_(stage_13_v37, stage_13_v36)
                            if stage_13_v10 <= stage_13_v35:
                                if stage_13_v35 <= stage_13_v15:
                                    stage_13_v66: pl.Scalar[pl.INDEX] = stage_13_v9 * 6 + stage_13_v35 - stage_13_v10
                                    stage_13_v67: pl.Scalar[pl.INDEX] = pl.cast(stage_13_v35 % 4, pl.INDEX)
                                    stage_13_v68: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_34, [stage_13_v66, stage_13_v54 + stage_13_v22], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    stage_13_v69: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_35, [stage_13_v66, stage_13_v54 + stage_13_v22], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    stage_13_v70: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_ape, [stage_13_v67, stage_13_v54 + stage_13_v22], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    stage_13_v71: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_13_v69, stage_13_v70)
                                    (stage_13_v76, stage_13_v77) = pl.yield_(stage_13_v71, stage_13_v68)
                                else:
                                    (stage_13_v76, stage_13_v77) = pl.yield_(stage_13_v64, stage_13_v65)
                                (stage_13_v72, stage_13_v73) = pl.yield_(stage_13_v76, stage_13_v77)
                            else:
                                (stage_13_v72, stage_13_v73) = pl.yield_(stage_13_v64, stage_13_v65)
                            stage_13_v39: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_13_v51, stage_13_v72)
                            stage_13_v40: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_13_v51, stage_13_v39)
                            stage_13_v41: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_13_v40)
                            stage_13_v42: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_13_v72, stage_13_v39)
                            stage_13_v43: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_13_v42)
                            stage_13_v44: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_13_v41, stage_13_v50)
                            stage_13_v45: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_13_v44, stage_13_v43)
                            stage_13_v46: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_13_v52, stage_13_v41)
                            stage_13_v47: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_13_v73, stage_13_v43)
                            stage_13_v48: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_13_v46, stage_13_v47)
                            stage_13_v49: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = stage_13_v39
                            (stage_13_v58, stage_13_v59, stage_13_v60) = pl.yield_(stage_13_v45, stage_13_v49, stage_13_v48)
                        stage_13_v30: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_13_v60, stage_13_v58)
                        stage_13_v31 = pl.tile.store(stage_13_v30, [stage_13_v14, stage_13_v22], stage_13_v33)
                        stage_13_v32 = pl.yield_(stage_13_v31)
                    stage_13_v21 = pl.yield_(stage_13_v32)
                else:
                    stage_13_v21 = pl.yield_(stage_13_v17)
                stage_13_v18 = pl.yield_(stage_13_v21)
            stage_13_v11 = pl.yield_(stage_13_v18)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_13 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_13)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_14 = 23328 + core_id * 6
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
        stage_14_v9: pl.Scalar[pl.INDEX] = block_14
        for (stage_14_v10, (stage_14_v14,)) in pl.range(stage_14_v9, 16, 16, init_values=(temporary_28,)):
            stage_14_v11: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_14_v10 * 6])
            stage_14_v12: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_seq_lens, [stage_14_v10])
            for (stage_14_v15, (stage_14_v21,)) in pl.range(6, init_values=(stage_14_v14,)):
                stage_14_v16: pl.Scalar[pl.INDEX] = stage_14_v10 * 6 + stage_14_v15
                stage_14_v17: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_14_v16])
                stage_14_v18: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                stage_14_v19 = pl.tile.store(stage_14_v18, [stage_14_v16, 0], stage_14_v21)
                if 0 < pl.cast(stage_14_v12, pl.INDEX) and (stage_14_v17 + 1) % 4 == 0:
                    stage_14_v22: pl.Scalar[pl.INT64] = stage_14_v17 - 8 + 1
                    stage_14_v23: pl.Scalar[pl.INDEX] = pl.cast(stage_14_v17 % 4, pl.INDEX)
                    stage_14_v24: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_30, [stage_14_v16, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    stage_14_v25: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_cmp_ape, [stage_14_v23, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    stage_14_v26: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_14_v24, stage_14_v25)
                    stage_14_v27: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_14_v26, stage_14_v26)
                    stage_14_v28: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_14_v27)
                    stage_14_v29: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_29, [stage_14_v16, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    for (stage_14_v31, (stage_14_v47, stage_14_v48, stage_14_v49)) in pl.range(7, init_values=(stage_14_v28, stage_14_v26, stage_14_v29)):
                        stage_14_v32: pl.Scalar[pl.INT64] = stage_14_v22 + stage_14_v31
                        stage_14_v33: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                        stage_14_v34: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e+38)
                        stage_14_v35: pl.Scalar[pl.INDEX] = 0
                        if 4 <= stage_14_v31:
                            stage_14_v50: pl.Scalar[pl.INDEX] = 512
                            stage_14_v51: pl.Scalar[pl.INDEX] = pl.yield_(stage_14_v50)
                        else:
                            stage_14_v51: pl.Scalar[pl.INDEX] = pl.yield_(stage_14_v35)
                        if 0 <= stage_14_v32 and stage_14_v32 < stage_14_v11:
                            stage_14_v52: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                            stage_14_v53: pl.Scalar[pl.INT32] = pl.tensor.read(arg_state_block_table, [stage_14_v10, stage_14_v32 // 2])
                            stage_14_v54: pl.Scalar[pl.INDEX] = pl.cast(stage_14_v53, pl.INDEX)
                            if 0 <= stage_14_v54:
                                stage_14_v58: pl.Scalar[pl.INDEX] = pl.cast(stage_14_v32 % 2, pl.INDEX) * 2048 + stage_14_v51
                                stage_14_v59: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_compress_state, [stage_14_v54, stage_14_v58], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                stage_14_v60: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_compress_state, [stage_14_v54, stage_14_v58 + 1024], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                (stage_14_v71, stage_14_v72) = pl.yield_(stage_14_v60, stage_14_v59)
                            else:
                                (stage_14_v71, stage_14_v72) = pl.yield_(stage_14_v52, stage_14_v33)
                            (stage_14_v61, stage_14_v62) = pl.yield_(stage_14_v71, stage_14_v72)
                        else:
                            (stage_14_v61, stage_14_v62) = pl.yield_(stage_14_v34, stage_14_v33)
                        if stage_14_v11 <= stage_14_v32:
                            if stage_14_v32 <= stage_14_v17:
                                stage_14_v63: pl.Scalar[pl.INDEX] = stage_14_v10 * 6 + stage_14_v32 - stage_14_v11
                                stage_14_v64: pl.Scalar[pl.INDEX] = pl.cast(stage_14_v32 % 4, pl.INDEX)
                                stage_14_v65: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_29, [stage_14_v63, stage_14_v51], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                stage_14_v66: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_30, [stage_14_v63, stage_14_v51], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                stage_14_v67: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_cmp_ape, [stage_14_v64, stage_14_v51], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                stage_14_v68: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_14_v66, stage_14_v67)
                                (stage_14_v73, stage_14_v74) = pl.yield_(stage_14_v68, stage_14_v65)
                            else:
                                (stage_14_v73, stage_14_v74) = pl.yield_(stage_14_v61, stage_14_v62)
                            (stage_14_v69, stage_14_v70) = pl.yield_(stage_14_v73, stage_14_v74)
                        else:
                            (stage_14_v69, stage_14_v70) = pl.yield_(stage_14_v61, stage_14_v62)
                        stage_14_v36: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_14_v48, stage_14_v69)
                        stage_14_v37: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_14_v48, stage_14_v36)
                        stage_14_v38: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_14_v37)
                        stage_14_v39: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_14_v69, stage_14_v36)
                        stage_14_v40: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_14_v39)
                        stage_14_v41: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_14_v38, stage_14_v47)
                        stage_14_v42: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_14_v41, stage_14_v40)
                        stage_14_v43: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_14_v49, stage_14_v38)
                        stage_14_v44: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_14_v70, stage_14_v40)
                        stage_14_v45: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_14_v43, stage_14_v44)
                        stage_14_v46: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = stage_14_v36
                        (stage_14_v55, stage_14_v56, stage_14_v57) = pl.yield_(stage_14_v42, stage_14_v46, stage_14_v45)
                    stage_14_v30: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_14_v57, stage_14_v55)
                    pl.tile.store(stage_14_v30, [stage_14_v16, 0], stage_14_v19)
                stage_14_v20 = pl.yield_(stage_14_v19)
            stage_14_v13 = pl.yield_(stage_14_v20)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_14 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_14)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_15 = 23760 + core_id * 6
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
        stage_15_v8: pl.Scalar[pl.INDEX] = block_15
        stage_15_v9: pl.Scalar[pl.INDEX] = 4
        if 384 <= 96 and 96 % 384 == 0:
            stage_15_v11: pl.Scalar[pl.INDEX] = 64
            stage_15_v12: pl.Scalar[pl.INDEX] = pl.yield_(stage_15_v11)
        else:
            if 96 <= 96 and 96 % 96 == 0:
                stage_15_v16: pl.Scalar[pl.INDEX] = 16
                stage_15_v17: pl.Scalar[pl.INDEX] = pl.yield_(stage_15_v16)
            else:
                stage_15_v17: pl.Scalar[pl.INDEX] = pl.yield_(stage_15_v9)
            stage_15_v12: pl.Scalar[pl.INDEX] = pl.yield_(stage_15_v17)
        for stage_15_v10 in pl.range(stage_15_v8, (96 + 7) // 8 * (64 // stage_15_v12), 48):
            stage_15_v13: pl.Scalar[pl.INDEX] = stage_15_v10 % (64 // stage_15_v12) * stage_15_v12
            stage_15_v14: pl.Scalar[pl.INDEX] = stage_15_v10 // (64 // stage_15_v12) * 8
            stage_15_v15: pl.Scalar[pl.INDEX] = 0 + stage_15_v14
            if stage_15_v14 + 8 <= 96:
                stage_15_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_23, [stage_15_v14, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_15_v19: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_15_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_15_v20: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_20, [stage_15_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_15_v21: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_21, [stage_15_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_15_v22: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                stage_15_v23: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v22, target_type=pl.FP32, mode='round')
                stage_15_v24: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v23, 0.015625)
                stage_15_v25: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_15_v24, target_type=pl.INT32, mode='trunc')
                stage_15_v26: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v25, target_type=pl.FP32, mode='round')
                stage_15_v27: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v26, 64.0)
                stage_15_v28: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v21, [1, 512])
                stage_15_v29: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v28, target_type=pl.FP32, mode='round')
                stage_15_v30: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_15_v29, stage_15_v27)
                stage_15_v31: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_15_v30, target_type=pl.INT32, mode='round')
                stage_15_v32: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 512], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_15_v33 in pl.pipeline(stage_15_v12, stage=2):
                    stage_15_v62: pl.Scalar[pl.INDEX] = stage_15_v13 + stage_15_v33
                    stage_15_v63: pl.Scalar[pl.INDEX] = stage_15_v62 * 512
                    stage_15_v64: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_25, [stage_15_v14, stage_15_v63], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    stage_15_v65: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wq_b_scale, [stage_15_v63], [512], [512], target_memory=pl.Mem.Vec)
                    stage_15_v66: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v65, [1, 512])
                    stage_15_v67: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v64, target_type=pl.FP32, mode='none')
                    stage_15_v68: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_15_v67, stage_15_v18)
                    stage_15_v69: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_15_v68, stage_15_v66)
                    stage_15_v70: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_15_v69, stage_15_v69)
                    stage_15_v71: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_15_v72: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_15_v70, stage_15_v71)
                    stage_15_v73: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v72, [1, 8])
                    stage_15_v74: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v73, 0.001953125)
                    stage_15_v75: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_15_v74, 1e-06)
                    stage_15_v76: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_15_v77: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(stage_15_v75, stage_15_v76)
                    stage_15_v78: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v77, [8, 1])
                    stage_15_v79: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_15_v69, [8, 448], [0, 0])
                    stage_15_v80: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_15_v79, stage_15_v78)
                    stage_15_v81: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_15_v80, target_type=pl.BF16, mode='rint')
                    stage_15_v82 = pl.tile.store(stage_15_v81, [stage_15_v15, stage_15_v63], view_40)
                    stage_15_v83: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_15_v69, [8, 64], [0, 448])
                    stage_15_v84: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_15_v83, stage_15_v78)
                    stage_15_v85: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v84, [1, 512])
                    stage_15_v86: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_15_v85, stage_15_v31, stage_15_v32)
                    stage_15_v87: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v86, [8, 64])
                    stage_15_v88: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_15_v84, stage_15_v19)
                    stage_15_v89: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_15_v87, stage_15_v20)
                    stage_15_v90: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_15_v88, stage_15_v89)
                    stage_15_v91: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_15_v90, target_type=pl.BF16, mode='rint')
                    stage_15_v92 = pl.tile.store(stage_15_v91, [stage_15_v15, stage_15_v63 + 448], view_40)
            else:
                stage_15_v34: pl.Scalar[pl.INDEX] = 96 - stage_15_v14
                stage_15_v35: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_23, [stage_15_v14, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_15_v36: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v34, 64])] = pl.tile.load(arg_freqs_cos, [stage_15_v15, 0], [8, 64], [stage_15_v34, 64], target_memory=pl.Mem.Vec)
                stage_15_v37: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v34, 64])] = pl.tile.load(temporary_20, [stage_15_v15, 0], [8, 64], [stage_15_v34, 64], target_memory=pl.Mem.Vec)
                stage_15_v38: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                stage_15_v39: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                stage_15_v40: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v39, target_type=pl.FP32, mode='round')
                stage_15_v41: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_15_v38, stage_15_v40)
                stage_15_v42: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v41, 0.5)
                stage_15_v43: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_15_v42, target_type=pl.INT32, mode='trunc')
                stage_15_v44: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v43, target_type=pl.FP32, mode='round')
                stage_15_v45: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v44, 2.0)
                stage_15_v46: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_15_v41, stage_15_v45)
                stage_15_v47: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_15_v41, 1.0)
                stage_15_v48: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v46, 2.0)
                stage_15_v49: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_15_v47, stage_15_v48)
                stage_15_v50: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False)
                stage_15_v51: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v50, target_type=pl.FP32, mode='round')
                stage_15_v52: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v51, 64.0)
                stage_15_v53: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                stage_15_v54: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_15_v53, stage_15_v52)
                stage_15_v55: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_15_v56: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(stage_15_v54, 0, 1, stage_15_v55)
                stage_15_v57: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_15_v49, stage_15_v56)
                stage_15_v58: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_15_v57, target_type=pl.INT32, mode='round')
                stage_15_v59: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_15_v60: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_15_v61 in pl.range(4):
                    stage_15_v93: pl.Scalar[pl.INDEX] = stage_15_v13 + stage_15_v61
                    stage_15_v94: pl.Scalar[pl.INDEX] = stage_15_v93 * 512
                    stage_15_v95: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_25, [stage_15_v14, stage_15_v94], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    stage_15_v96: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wq_b_scale, [stage_15_v94], [512], [512], target_memory=pl.Mem.Vec)
                    stage_15_v97: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v96, [1, 512])
                    stage_15_v98: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v95, target_type=pl.FP32, mode='none')
                    stage_15_v99: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_15_v98, stage_15_v35)
                    stage_15_v100: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_15_v99, stage_15_v97)
                    stage_15_v101: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_15_v100, stage_15_v100)
                    stage_15_v102: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_15_v101, stage_15_v59)
                    stage_15_v103: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v102, [1, 8])
                    stage_15_v104: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v103, 0.001953125)
                    stage_15_v105: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v104, [8, 1])
                    stage_15_v106: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v105, [1, 8])
                    stage_15_v107: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_15_v106, 1e-06)
                    stage_15_v108: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v107, [8, 1])
                    stage_15_v109: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v108, [1, 8])
                    stage_15_v110: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_15_v109)
                    stage_15_v111: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v110, [8, 1])
                    stage_15_v112: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v111, [1, 8])
                    stage_15_v113: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_15_v112)
                    stage_15_v114: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v113, [8, 1])
                    stage_15_v115: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_15_v100, [8, 448], [0, 0])
                    stage_15_v116: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_15_v115, stage_15_v114)
                    stage_15_v117: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_15_v116, target_type=pl.BF16, mode='rint')
                    stage_15_v118: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v34, 448])] = pl.tile.set_validshape(stage_15_v117, stage_15_v34, 448)
                    stage_15_v119 = pl.tile.store(stage_15_v118, [stage_15_v15, stage_15_v94], view_40)
                    stage_15_v120: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_15_v100, [8, 64], [0, 448])
                    stage_15_v121: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_15_v120, stage_15_v114)
                    stage_15_v122: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_15_v121, stage_15_v58, stage_15_v60)
                    stage_15_v123: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v34, 64])] = pl.tile.set_validshape(stage_15_v121, stage_15_v34, 64)
                    stage_15_v124: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v34, 64])] = pl.tile.set_validshape(stage_15_v122, stage_15_v34, 64)
                    stage_15_v125: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v34, 64])] = pl.tile.mul(stage_15_v123, stage_15_v36)
                    stage_15_v126: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v34, 64])] = pl.tile.mul(stage_15_v124, stage_15_v37)
                    stage_15_v127: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v34, 64])] = pl.tile.add(stage_15_v125, stage_15_v126)
                    stage_15_v128: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v34, 64])] = pl.tile.cast(stage_15_v127, target_type=pl.BF16, mode='rint')
                    stage_15_v129: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_15_v34, 64])] = pl.tile.set_validshape(stage_15_v128, stage_15_v34, 64)
                    stage_15_v130 = pl.tile.store(stage_15_v129, [stage_15_v15, stage_15_v94 + 448], view_40)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_15 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_15)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_16 = 24192 + core_id * 6
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
        stage_16_v1: pl.Tile[[1, 512], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_45, [0, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
        stage_16_v2 = pl.tile.store(stage_16_v1, [0, 0], view_45)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_16 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_16)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_17 = 24624 + core_id * 6
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
        stage_17_v9: pl.Scalar[pl.INDEX] = 4
        if 384 <= 96 and 96 % 384 == 0:
            stage_17_v11: pl.Scalar[pl.INDEX] = 64
            stage_17_v12: pl.Scalar[pl.INDEX] = pl.yield_(stage_17_v11)
        else:
            if 96 <= 96 and 96 % 96 == 0:
                stage_17_v16: pl.Scalar[pl.INDEX] = 16
                stage_17_v17: pl.Scalar[pl.INDEX] = pl.yield_(stage_17_v16)
            else:
                stage_17_v17: pl.Scalar[pl.INDEX] = pl.yield_(stage_17_v9)
            stage_17_v12: pl.Scalar[pl.INDEX] = pl.yield_(stage_17_v17)
        for stage_17_v10 in pl.range(stage_17_v8, (96 + 7) // 8 * (64 // stage_17_v12), 48):
            stage_17_v13: pl.Scalar[pl.INDEX] = stage_17_v10 % (64 // stage_17_v12) * stage_17_v12
            stage_17_v14: pl.Scalar[pl.INDEX] = stage_17_v10 // (64 // stage_17_v12) * 8
            stage_17_v15: pl.Scalar[pl.INDEX] = 0 + stage_17_v14
            if stage_17_v14 + 8 <= 96:
                stage_17_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_23, [stage_17_v14, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_17_v19: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_17_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_17_v20: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_20, [stage_17_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_17_v21: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_21, [stage_17_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_17_v22: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                stage_17_v23: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_17_v22, target_type=pl.FP32, mode='round')
                stage_17_v24: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_17_v23, 0.015625)
                stage_17_v25: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_17_v24, target_type=pl.INT32, mode='trunc')
                stage_17_v26: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_17_v25, target_type=pl.FP32, mode='round')
                stage_17_v27: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_17_v26, 64.0)
                stage_17_v28: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v21, [1, 512])
                stage_17_v29: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_17_v28, target_type=pl.FP32, mode='round')
                stage_17_v30: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_17_v29, stage_17_v27)
                stage_17_v31: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_17_v30, target_type=pl.INT32, mode='round')
                stage_17_v32: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 512], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_17_v33 in pl.pipeline(stage_17_v12, stage=2):
                    stage_17_v62: pl.Scalar[pl.INDEX] = stage_17_v13 + stage_17_v33
                    stage_17_v63: pl.Scalar[pl.INDEX] = stage_17_v62 * 512
                    stage_17_v64: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_25, [stage_17_v14, stage_17_v63], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    stage_17_v65: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wq_b_scale, [stage_17_v63], [512], [512], target_memory=pl.Mem.Vec)
                    stage_17_v66: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v65, [1, 512])
                    stage_17_v67: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_17_v64, target_type=pl.FP32, mode='none')
                    stage_17_v68: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_17_v67, stage_17_v18)
                    stage_17_v69: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_17_v68, stage_17_v66)
                    stage_17_v70: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_17_v69, stage_17_v69)
                    stage_17_v71: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_17_v72: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_17_v70, stage_17_v71)
                    stage_17_v73: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v72, [1, 8])
                    stage_17_v74: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_17_v73, 0.001953125)
                    stage_17_v75: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_17_v74, 1e-06)
                    stage_17_v76: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_17_v77: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(stage_17_v75, stage_17_v76)
                    stage_17_v78: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v77, [8, 1])
                    stage_17_v79: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_17_v69, [8, 448], [0, 0])
                    stage_17_v80: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_17_v79, stage_17_v78)
                    stage_17_v81: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_17_v80, target_type=pl.BF16, mode='rint')
                    stage_17_v82 = pl.tile.store(stage_17_v81, [stage_17_v15, stage_17_v63], view_40)
                    stage_17_v83: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_17_v69, [8, 64], [0, 448])
                    stage_17_v84: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_17_v83, stage_17_v78)
                    stage_17_v85: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v84, [1, 512])
                    stage_17_v86: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_17_v85, stage_17_v31, stage_17_v32)
                    stage_17_v87: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v86, [8, 64])
                    stage_17_v88: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_17_v84, stage_17_v19)
                    stage_17_v89: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_17_v87, stage_17_v20)
                    stage_17_v90: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_17_v88, stage_17_v89)
                    stage_17_v91: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_17_v90, target_type=pl.BF16, mode='rint')
                    stage_17_v92 = pl.tile.store(stage_17_v91, [stage_17_v15, stage_17_v63 + 448], view_40)
            else:
                stage_17_v34: pl.Scalar[pl.INDEX] = 96 - stage_17_v14
                stage_17_v35: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_23, [stage_17_v14, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_17_v36: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_17_v34, 64])] = pl.tile.load(arg_freqs_cos, [stage_17_v15, 0], [8, 64], [stage_17_v34, 64], target_memory=pl.Mem.Vec)
                stage_17_v37: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_17_v34, 64])] = pl.tile.load(temporary_20, [stage_17_v15, 0], [8, 64], [stage_17_v34, 64], target_memory=pl.Mem.Vec)
                stage_17_v38: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                stage_17_v39: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                stage_17_v40: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_17_v39, target_type=pl.FP32, mode='round')
                stage_17_v41: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_17_v38, stage_17_v40)
                stage_17_v42: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_17_v41, 0.5)
                stage_17_v43: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_17_v42, target_type=pl.INT32, mode='trunc')
                stage_17_v44: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_17_v43, target_type=pl.FP32, mode='round')
                stage_17_v45: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_17_v44, 2.0)
                stage_17_v46: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_17_v41, stage_17_v45)
                stage_17_v47: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_17_v41, 1.0)
                stage_17_v48: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_17_v46, 2.0)
                stage_17_v49: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_17_v47, stage_17_v48)
                stage_17_v50: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False)
                stage_17_v51: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_17_v50, target_type=pl.FP32, mode='round')
                stage_17_v52: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_17_v51, 64.0)
                stage_17_v53: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                stage_17_v54: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_17_v53, stage_17_v52)
                stage_17_v55: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_17_v56: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(stage_17_v54, 0, 1, stage_17_v55)
                stage_17_v57: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_17_v49, stage_17_v56)
                stage_17_v58: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_17_v57, target_type=pl.INT32, mode='round')
                stage_17_v59: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_17_v60: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_17_v61 in pl.range(4):
                    stage_17_v93: pl.Scalar[pl.INDEX] = stage_17_v13 + stage_17_v61
                    stage_17_v94: pl.Scalar[pl.INDEX] = stage_17_v93 * 512
                    stage_17_v95: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_25, [stage_17_v14, stage_17_v94], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    stage_17_v96: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wq_b_scale, [stage_17_v94], [512], [512], target_memory=pl.Mem.Vec)
                    stage_17_v97: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v96, [1, 512])
                    stage_17_v98: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_17_v95, target_type=pl.FP32, mode='none')
                    stage_17_v99: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_17_v98, stage_17_v35)
                    stage_17_v100: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_17_v99, stage_17_v97)
                    stage_17_v101: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_17_v100, stage_17_v100)
                    stage_17_v102: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_17_v101, stage_17_v59)
                    stage_17_v103: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v102, [1, 8])
                    stage_17_v104: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_17_v103, 0.001953125)
                    stage_17_v105: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v104, [8, 1])
                    stage_17_v106: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v105, [1, 8])
                    stage_17_v107: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_17_v106, 1e-06)
                    stage_17_v108: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v107, [8, 1])
                    stage_17_v109: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v108, [1, 8])
                    stage_17_v110: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_17_v109)
                    stage_17_v111: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v110, [8, 1])
                    stage_17_v112: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v111, [1, 8])
                    stage_17_v113: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_17_v112)
                    stage_17_v114: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_17_v113, [8, 1])
                    stage_17_v115: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_17_v100, [8, 448], [0, 0])
                    stage_17_v116: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_17_v115, stage_17_v114)
                    stage_17_v117: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_17_v116, target_type=pl.BF16, mode='rint')
                    stage_17_v118: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_17_v34, 448])] = pl.tile.set_validshape(stage_17_v117, stage_17_v34, 448)
                    stage_17_v119 = pl.tile.store(stage_17_v118, [stage_17_v15, stage_17_v94], view_40)
                    stage_17_v120: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_17_v100, [8, 64], [0, 448])
                    stage_17_v121: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_17_v120, stage_17_v114)
                    stage_17_v122: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_17_v121, stage_17_v58, stage_17_v60)
                    stage_17_v123: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_17_v34, 64])] = pl.tile.set_validshape(stage_17_v121, stage_17_v34, 64)
                    stage_17_v124: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_17_v34, 64])] = pl.tile.set_validshape(stage_17_v122, stage_17_v34, 64)
                    stage_17_v125: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_17_v34, 64])] = pl.tile.mul(stage_17_v123, stage_17_v36)
                    stage_17_v126: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_17_v34, 64])] = pl.tile.mul(stage_17_v124, stage_17_v37)
                    stage_17_v127: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_17_v34, 64])] = pl.tile.add(stage_17_v125, stage_17_v126)
                    stage_17_v128: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_17_v34, 64])] = pl.tile.cast(stage_17_v127, target_type=pl.BF16, mode='rint')
                    stage_17_v129: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_17_v34, 64])] = pl.tile.set_validshape(stage_17_v128, stage_17_v34, 64)
                    stage_17_v130 = pl.tile.store(stage_17_v129, [stage_17_v15, stage_17_v94 + 448], view_40)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_17 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_17)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_18 = 25056 + core_id * 6
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
        stage_18_v7: pl.Scalar[pl.INDEX] = block_18
        stage_18_v8: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        stage_18_v9: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_18_v10: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_18_v9, target_type=pl.FP32, mode='round')
        stage_18_v11: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_18_v8, stage_18_v10)
        stage_18_v12: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_18_v11, 0.5)
        stage_18_v13: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_18_v12, target_type=pl.INT32, mode='trunc')
        stage_18_v14: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_18_v13, target_type=pl.FP32, mode='round')
        stage_18_v15: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_18_v14, 2.0)
        stage_18_v16: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_18_v11, stage_18_v15)
        stage_18_v17: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_18_v11, 1.0)
        stage_18_v18: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_18_v16, 2.0)
        stage_18_v19: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_18_v17, stage_18_v18)
        stage_18_v20: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_18_v19, target_type=pl.INT32, mode='round')
        for (stage_18_v21, (stage_18_v25,)) in pl.range(stage_18_v7, 192, 48, init_values=(view_85,)):
            stage_18_v22: pl.Scalar[pl.INDEX] = stage_18_v21 % 16 * 4
            stage_18_v23: pl.Scalar[pl.INDEX] = stage_18_v21 // 16 * 8
            if stage_18_v23 + 8 <= 96:
                stage_18_v26: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_19, [stage_18_v23, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_18_v27: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_18_v23, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_18_v28: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_13, [stage_18_v23, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_18_v29: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                stage_18_v30: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                stage_18_v31: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.rems(stage_18_v29, pl.const(2, pl.INT32), stage_18_v30)
                stage_18_v32: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_18_v31, pl.const(2, pl.INT32))
                stage_18_v33: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.sub(stage_18_v29, stage_18_v32)
                stage_18_v34: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_18_v33, pl.const(65, pl.INT32))
                stage_18_v35: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                stage_18_v36: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.col_expand(stage_18_v35, stage_18_v34)
                stage_18_v37: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False)
                stage_18_v38: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_18_v37, pl.const(128, pl.INT32))
                stage_18_v39: pl.Tile[[8, 1], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v38, [8, 1])
                stage_18_v40: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.row_expand_add(stage_18_v36, stage_18_v39)
                stage_18_v41: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v40, [1, 512])
                stage_18_v42: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 512], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_18_v43 in pl.pipeline(4, stage=2):
                    stage_18_v46: pl.Scalar[pl.INDEX] = (stage_18_v22 + stage_18_v43) * 128
                    stage_18_v47: pl.Tile[[128], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_idx_wq_b_scale, [stage_18_v46], [128], [128], target_memory=pl.Mem.Vec)
                    stage_18_v48: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v47, [1, 128])
                    stage_18_v49: pl.Tile[[8, 128], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_42, [stage_18_v23, stage_18_v46], [8, 128], [8, 128], target_memory=pl.Mem.Vec)
                    stage_18_v50: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_18_v49, target_type=pl.FP32, mode='none')
                    stage_18_v51: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_18_v50, stage_18_v26)
                    stage_18_v52: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_18_v51, stage_18_v48)
                    stage_18_v53: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_18_v52, [8, 64], [0, 0])
                    stage_18_v54: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_18_v53, target_type=pl.BF16, mode='rint')
                    stage_18_v55: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_18_v52, [8, 64], [0, 64])
                    stage_18_v56: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v52, [1, 1024])
                    stage_18_v57: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_18_v56, stage_18_v41, stage_18_v42)
                    stage_18_v58: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v57, [8, 64])
                    stage_18_v59: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_18_v55, stage_18_v27)
                    stage_18_v60: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_18_v58, stage_18_v28)
                    stage_18_v61: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_18_v59, stage_18_v60)
                    stage_18_v62: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_18_v61, target_type=pl.BF16, mode='rint')
                    pl.tile.store(stage_18_v54, [stage_18_v23, stage_18_v46], stage_18_v25)
                    pl.tile.store(stage_18_v62, [stage_18_v23, stage_18_v46 + 64], stage_18_v25)
                stage_18_v44 = pl.yield_(stage_18_v25)
            else:
                stage_18_v45: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_18_v20, [1, 64], [0, 0])
                for (stage_18_v63, (stage_18_v68,)) in pl.range(stage_18_v23, 96, init_values=(stage_18_v25,)):
                    stage_18_v64: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_19, [stage_18_v63, 0])
                    stage_18_v65: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_18_v63, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_18_v66: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_13, [stage_18_v63, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    for (stage_18_v69, (stage_18_v89,)) in pl.pipeline(4, stage=2, init_values=(stage_18_v68,)):
                        stage_18_v70: pl.Scalar[pl.INDEX] = (stage_18_v22 + stage_18_v69) * 128
                        stage_18_v71: pl.Tile[[128], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_idx_wq_b_scale, [stage_18_v70], [128], [128], target_memory=pl.Mem.Vec)
                        stage_18_v72: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v71, [1, 128])
                        stage_18_v73: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_42, [stage_18_v63, stage_18_v70], [1, 128], [1, 128], target_memory=pl.Mem.Vec)
                        stage_18_v74: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_18_v73, target_type=pl.FP32, mode='none')
                        stage_18_v75: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_18_v74, stage_18_v64)
                        stage_18_v76: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_18_v75, stage_18_v72)
                        stage_18_v77: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_18_v76, [1, 64], [0, 0])
                        stage_18_v78: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_18_v77, target_type=pl.BF16, mode='rint')
                        stage_18_v79: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_18_v76, [1, 64], [0, 64])
                        stage_18_v80: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        for (stage_18_v90, (stage_18_v97,)) in pl.range(1, init_values=(stage_18_v80,)):
                            stage_18_v91: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_18_v79, [1, 64], [stage_18_v90, 0], [1, 64])
                            stage_18_v92: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_18_v45, [1, 64], [stage_18_v90, 0], [1, 64])
                            stage_18_v93: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                            stage_18_v94: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_18_v91, stage_18_v92, stage_18_v93)
                            stage_18_v95: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_18_v97, stage_18_v94, [stage_18_v90, 0])
                            stage_18_v96: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_18_v95)
                        stage_18_v81: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = stage_18_v96
                        stage_18_v82: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_18_v79, stage_18_v65)
                        stage_18_v83: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_18_v81, stage_18_v66)
                        stage_18_v84: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_18_v82, stage_18_v83)
                        stage_18_v85: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_18_v84, target_type=pl.BF16, mode='rint')
                        stage_18_v86 = pl.tile.store(stage_18_v78, [stage_18_v63, stage_18_v70], stage_18_v89)
                        stage_18_v87 = pl.tile.store(stage_18_v85, [stage_18_v63, stage_18_v70 + 64], stage_18_v86)
                        stage_18_v88 = pl.yield_(stage_18_v87)
                    stage_18_v67 = pl.yield_(stage_18_v88)
                stage_18_v44 = pl.yield_(stage_18_v67)
            stage_18_v24 = pl.yield_(stage_18_v44)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_18 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_18)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_19 = 25488 + core_id * 6
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
        stage_19_v2: pl.Scalar[pl.INDEX] = block_19
        stage_19_v3: pl.Scalar[pl.INDEX] = stage_19_v2 * 2
        stage_19_v4: pl.Tile[[2, 128], pl.BF16, pl.Mem.Vec] = pl.tile.full([2, 128], dtype=pl.BF16, value=0.0)
        stage_19_v5 = pl.tile.store(stage_19_v4, [stage_19_v3, 0], temporary_33)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_19 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_19)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_20 = 25920 + core_id * 6
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
        for (stage_20_v8, (stage_20_v10,)) in pl.range(stage_20_v7, 16, 16, init_values=(arg_compress_state,)):
            for (stage_20_v11, (stage_20_v16,)) in pl.range(6, init_values=(stage_20_v10,)):
                stage_20_v12: pl.Scalar[pl.INDEX] = stage_20_v8 * 6 + stage_20_v11
                stage_20_v13: pl.Scalar[pl.INT32] = pl.tensor.read(arg_state_slot_mapping, [stage_20_v12, 0])
                stage_20_v14: pl.Scalar[pl.INT32] = pl.tensor.read(arg_state_slot_mapping, [stage_20_v12, 1])
                if 0 <= pl.cast(stage_20_v13, pl.INDEX) and 0 <= pl.cast(stage_20_v14, pl.INDEX):
                    stage_20_v17: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_20_v12])
                    stage_20_v18: pl.Scalar[pl.INDEX] = pl.cast(stage_20_v13, pl.INDEX)
                    stage_20_v19: pl.Scalar[pl.INDEX] = pl.cast(stage_20_v14, pl.INDEX) * 2048
                    stage_20_v20: pl.Scalar[pl.INDEX] = pl.cast(stage_20_v17 % 4, pl.INDEX)
                    stage_20_v21: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_29, [stage_20_v12, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_20_v22 = pl.tile.store(stage_20_v21, [stage_20_v18, stage_20_v19], stage_20_v16)
                    stage_20_v23: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_30, [stage_20_v12, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_20_v24: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_cmp_ape, [stage_20_v20, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_20_v25: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_20_v23, stage_20_v24)
                    stage_20_v26 = pl.tile.store(stage_20_v25, [stage_20_v18, stage_20_v19 + 1024], stage_20_v22)
                    stage_20_v27 = pl.yield_(stage_20_v26)
                else:
                    stage_20_v27 = pl.yield_(stage_20_v16)
                stage_20_v15 = pl.yield_(stage_20_v27)
            stage_20_v9 = pl.yield_(stage_20_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_20 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_20)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_21 = 26352 + core_id * 6
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
        stage_21_v12: pl.Scalar[pl.INDEX] = block_21
        stage_21_v13: pl.Scalar[pl.INDEX] = stage_21_v12 * 16
        stage_21_v14: pl.Scalar[pl.INDEX] = pl.min(96 - stage_21_v13, 16)
        stage_21_v15: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        stage_21_v16: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
        for (stage_21_v71, (stage_21_v93, stage_21_v94)) in pl.range(stage_21_v14, init_values=(stage_21_v15, stage_21_v16)):
            stage_21_v72: pl.Scalar[pl.INDEX] = stage_21_v13 + stage_21_v71
            stage_21_v73: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_21_v72])
            if (stage_21_v73 + 1) % 4 == 0:
                stage_21_v95: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_14, [stage_21_v72 // 6])
                stage_21_v96: pl.Scalar[pl.INDEX] = pl.cast(stage_21_v95, pl.INDEX) + pl.cast((stage_21_v73 + 1) // 4, pl.INDEX)
                stage_21_v97: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_cmp_freqs_cos, 0)
                if stage_21_v96 < stage_21_v97:
                    stage_21_v112: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather_row(stage_21_v93, arg_cmp_freqs_cos, [stage_21_v71, 0], [stage_21_v96, 0], [1, 64], transpose=False)
                    stage_21_v113: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather_row(stage_21_v94, arg_cmp_freqs_sin, [stage_21_v71, 0], [stage_21_v96, 0], [1, 64], transpose=False)
                    (stage_21_v123, stage_21_v124) = pl.yield_(stage_21_v112, stage_21_v113)
                else:
                    (stage_21_v123, stage_21_v124) = pl.yield_(stage_21_v93, stage_21_v94)
                (stage_21_v114, stage_21_v115) = pl.yield_(stage_21_v123, stage_21_v124)
            else:
                (stage_21_v114, stage_21_v115) = pl.yield_(stage_21_v93, stage_21_v94)
            (stage_21_v98, stage_21_v99) = pl.yield_(stage_21_v114, stage_21_v115)
        stage_21_v17: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_21_v98
        stage_21_v18: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_21_v99
        stage_21_v19: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_28, [stage_21_v13, 0], [16, 256], [16, 256], target_memory=pl.Mem.Vec)
        stage_21_v20: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_28, [stage_21_v13, 256], [16, 256], [16, 256], target_memory=pl.Mem.Vec)
        stage_21_v21: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_21_v19, stage_21_v19)
        stage_21_v22: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_21_v20, stage_21_v20)
        stage_21_v23: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_21_v21, stage_21_v22)
        stage_21_v24: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_21_v23, [16, 128], [0, 0])
        stage_21_v25: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_21_v23, [16, 128], [0, 128])
        stage_21_v26: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_21_v24, stage_21_v25)
        stage_21_v27: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_21_v26, [16, 64], [0, 0])
        stage_21_v28: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_21_v26, [16, 64], [0, 64])
        stage_21_v29: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_21_v27, stage_21_v28)
        stage_21_v30: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_21_v31: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_21_v29, stage_21_v30)
        stage_21_v32: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_21_v31, [1, 16])
        stage_21_v33: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_21_v32, 0.001953125)
        stage_21_v34: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_21_v33, [16, 1])
        stage_21_v35: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_21_v34, [1, 16])
        stage_21_v36: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_21_v35, 1e-06)
        stage_21_v37: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_21_v36, [16, 1])
        stage_21_v38: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_21_v37, [1, 16])
        stage_21_v39: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_21_v38)
        stage_21_v40: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_21_v39, [16, 1])
        stage_21_v41: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_21_v40, [1, 16])
        stage_21_v42: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_21_v41)
        stage_21_v43: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_21_v42, [16, 1])
        for (stage_21_v74, (stage_21_v100,)) in pl.range(0, 448, 64, init_values=(temporary_31,)):
            stage_21_v75: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_28, [stage_21_v13, stage_21_v74], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            stage_21_v76: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_60, [0, stage_21_v74], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
            stage_21_v77: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_21_v76, target_type=pl.FP32, mode='round')
            stage_21_v78: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_21_v75, stage_21_v43)
            stage_21_v79: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_21_v78, stage_21_v77)
            stage_21_v80 = pl.tile.store(stage_21_v79, [stage_21_v13, stage_21_v74], stage_21_v100)
            stage_21_v81 = pl.yield_(stage_21_v80)
        stage_21_v44: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_28, [stage_21_v13, 448], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_21_v45: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_60, [0, 448], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
        stage_21_v46: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_21_v45, target_type=pl.FP32, mode='round')
        stage_21_v47: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_21_v44, stage_21_v43)
        stage_21_v48: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_21_v47, stage_21_v46)
        stage_21_v49: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        stage_21_v50: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_21_v51: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_21_v50, target_type=pl.FP32, mode='round')
        stage_21_v52: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_21_v49, stage_21_v51)
        stage_21_v53: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_21_v52, 0.5)
        stage_21_v54: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_21_v53, target_type=pl.INT32, mode='trunc')
        stage_21_v55: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_21_v54, target_type=pl.FP32, mode='round')
        stage_21_v56: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_21_v55, 2.0)
        stage_21_v57: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_21_v52, stage_21_v56)
        stage_21_v58: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_21_v52, 1.0)
        stage_21_v59: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_21_v57, 2.0)
        stage_21_v60: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_21_v58, stage_21_v59)
        stage_21_v61: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_21_v60, target_type=pl.INT32, mode='round')
        stage_21_v62: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        for (stage_21_v82, (stage_21_v101,)) in pl.range(16, init_values=(stage_21_v62,)):
            stage_21_v83: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_21_v48, [1, 64], [stage_21_v82, 0], [1, 64])
            stage_21_v84: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_21_v61, [1, 64], [stage_21_v82, 0], [1, 64])
            stage_21_v85: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            stage_21_v86: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_21_v83, stage_21_v84, stage_21_v85)
            stage_21_v87: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_21_v101, stage_21_v86, [stage_21_v82, 0])
            stage_21_v88: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_21_v87)
        stage_21_v63: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_21_v88
        stage_21_v64: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_21_v57, 2.0)
        stage_21_v65: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_21_v64, 1.0)
        stage_21_v66: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_21_v18, stage_21_v65)
        stage_21_v67: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_21_v48, stage_21_v17)
        stage_21_v68: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_21_v63, stage_21_v66)
        stage_21_v69: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_21_v67, stage_21_v68)
        stage_21_v70 = pl.tile.store(stage_21_v69, [stage_21_v13, 448], stage_21_v81)
        for (stage_21_v89, (stage_21_v102, stage_21_v103)) in pl.range(stage_21_v14, init_values=(view_61, temporary_27)):
            stage_21_v90: pl.Scalar[pl.INDEX] = stage_21_v13 + stage_21_v89
            stage_21_v91: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_21_v90])
            stage_21_v92: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_seq_lens, [stage_21_v90 // 6])
            if 0 < pl.cast(stage_21_v92, pl.INDEX) and (stage_21_v91 + 1) % 4 == 0:
                stage_21_v104: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_14, [stage_21_v90 // 6])
                stage_21_v105: pl.Scalar[pl.INDEX] = pl.cast(stage_21_v104, pl.INDEX) + pl.cast((stage_21_v91 + 1) // 4, pl.INDEX)
                stage_21_v106: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_cmp_slot_mapping, 0)
                stage_21_v107: pl.Scalar[pl.INDEX] = pl.min(stage_21_v105, stage_21_v106 - 1)
                stage_21_v108: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_slot_mapping, [stage_21_v107, 0])
                stage_21_v109: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_slot_mapping, [stage_21_v107, 1])
                if stage_21_v105 < stage_21_v106 and 0 <= pl.cast(stage_21_v108, pl.INDEX) and (0 <= pl.cast(stage_21_v109, pl.INDEX)):
                    stage_21_v116: pl.Scalar[pl.INDEX] = pl.cast(stage_21_v108, pl.INDEX) * 32 + pl.cast(stage_21_v109, pl.INDEX)
                    stage_21_v117: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(stage_21_v70, [stage_21_v90, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    stage_21_v118 = pl.tile.store(stage_21_v117, [stage_21_v90, 0], stage_21_v103)
                    stage_21_v119: pl.Tile[[1, 512], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_21_v117, target_type=pl.BF16, mode='rint')
                    stage_21_v120 = pl.tile.store(stage_21_v119, [stage_21_v116, 0], stage_21_v102)
                    (stage_21_v125, stage_21_v126) = pl.yield_(stage_21_v120, stage_21_v118)
                else:
                    (stage_21_v125, stage_21_v126) = pl.yield_(stage_21_v102, stage_21_v103)
                (stage_21_v121, stage_21_v122) = pl.yield_(stage_21_v125, stage_21_v126)
            else:
                (stage_21_v121, stage_21_v122) = pl.yield_(stage_21_v102, stage_21_v103)
            (stage_21_v110, stage_21_v111) = pl.yield_(stage_21_v121, stage_21_v122)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_21 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_21)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_22 = 26784 + core_id * 6
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
        stage_22_v7: pl.Scalar[pl.INDEX] = block_22
        stage_22_v8: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        stage_22_v9: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_22_v10: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_22_v9, target_type=pl.FP32, mode='round')
        stage_22_v11: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_22_v8, stage_22_v10)
        stage_22_v12: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_22_v11, 0.5)
        stage_22_v13: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_22_v12, target_type=pl.INT32, mode='trunc')
        stage_22_v14: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_22_v13, target_type=pl.FP32, mode='round')
        stage_22_v15: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_22_v14, 2.0)
        stage_22_v16: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_22_v11, stage_22_v15)
        stage_22_v17: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_22_v11, 1.0)
        stage_22_v18: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_22_v16, 2.0)
        stage_22_v19: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_22_v17, stage_22_v18)
        stage_22_v20: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_22_v19, target_type=pl.INT32, mode='round')
        for (stage_22_v21, (stage_22_v25,)) in pl.range(stage_22_v7, 192, 48, init_values=(view_85,)):
            stage_22_v22: pl.Scalar[pl.INDEX] = stage_22_v21 % 16 * 4
            stage_22_v23: pl.Scalar[pl.INDEX] = stage_22_v21 // 16 * 8
            if stage_22_v23 + 8 <= 96:
                stage_22_v26: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_19, [stage_22_v23, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_22_v27: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_22_v23, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_22_v28: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_13, [stage_22_v23, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_22_v29: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                stage_22_v30: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                stage_22_v31: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.rems(stage_22_v29, pl.const(2, pl.INT32), stage_22_v30)
                stage_22_v32: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_22_v31, pl.const(2, pl.INT32))
                stage_22_v33: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.sub(stage_22_v29, stage_22_v32)
                stage_22_v34: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_22_v33, pl.const(65, pl.INT32))
                stage_22_v35: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                stage_22_v36: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.col_expand(stage_22_v35, stage_22_v34)
                stage_22_v37: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False)
                stage_22_v38: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_22_v37, pl.const(128, pl.INT32))
                stage_22_v39: pl.Tile[[8, 1], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_22_v38, [8, 1])
                stage_22_v40: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.row_expand_add(stage_22_v36, stage_22_v39)
                stage_22_v41: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_22_v40, [1, 512])
                stage_22_v42: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 512], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_22_v43 in pl.pipeline(4, stage=2):
                    stage_22_v46: pl.Scalar[pl.INDEX] = (stage_22_v22 + stage_22_v43) * 128
                    stage_22_v47: pl.Tile[[128], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_idx_wq_b_scale, [stage_22_v46], [128], [128], target_memory=pl.Mem.Vec)
                    stage_22_v48: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_22_v47, [1, 128])
                    stage_22_v49: pl.Tile[[8, 128], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_42, [stage_22_v23, stage_22_v46], [8, 128], [8, 128], target_memory=pl.Mem.Vec)
                    stage_22_v50: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_22_v49, target_type=pl.FP32, mode='none')
                    stage_22_v51: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_22_v50, stage_22_v26)
                    stage_22_v52: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_22_v51, stage_22_v48)
                    stage_22_v53: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_22_v52, [8, 64], [0, 0])
                    stage_22_v54: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_22_v53, target_type=pl.BF16, mode='rint')
                    stage_22_v55: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_22_v52, [8, 64], [0, 64])
                    stage_22_v56: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_22_v52, [1, 1024])
                    stage_22_v57: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_22_v56, stage_22_v41, stage_22_v42)
                    stage_22_v58: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_22_v57, [8, 64])
                    stage_22_v59: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_22_v55, stage_22_v27)
                    stage_22_v60: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_22_v58, stage_22_v28)
                    stage_22_v61: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_22_v59, stage_22_v60)
                    stage_22_v62: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_22_v61, target_type=pl.BF16, mode='rint')
                    pl.tile.store(stage_22_v54, [stage_22_v23, stage_22_v46], stage_22_v25)
                    pl.tile.store(stage_22_v62, [stage_22_v23, stage_22_v46 + 64], stage_22_v25)
                stage_22_v44 = pl.yield_(stage_22_v25)
            else:
                stage_22_v45: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_22_v20, [1, 64], [0, 0])
                for (stage_22_v63, (stage_22_v68,)) in pl.range(stage_22_v23, 96, init_values=(stage_22_v25,)):
                    stage_22_v64: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_19, [stage_22_v63, 0])
                    stage_22_v65: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_22_v63, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_22_v66: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_13, [stage_22_v63, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    for (stage_22_v69, (stage_22_v89,)) in pl.pipeline(4, stage=2, init_values=(stage_22_v68,)):
                        stage_22_v70: pl.Scalar[pl.INDEX] = (stage_22_v22 + stage_22_v69) * 128
                        stage_22_v71: pl.Tile[[128], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_idx_wq_b_scale, [stage_22_v70], [128], [128], target_memory=pl.Mem.Vec)
                        stage_22_v72: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_22_v71, [1, 128])
                        stage_22_v73: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_42, [stage_22_v63, stage_22_v70], [1, 128], [1, 128], target_memory=pl.Mem.Vec)
                        stage_22_v74: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_22_v73, target_type=pl.FP32, mode='none')
                        stage_22_v75: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_22_v74, stage_22_v64)
                        stage_22_v76: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_22_v75, stage_22_v72)
                        stage_22_v77: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_22_v76, [1, 64], [0, 0])
                        stage_22_v78: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_22_v77, target_type=pl.BF16, mode='rint')
                        stage_22_v79: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_22_v76, [1, 64], [0, 64])
                        stage_22_v80: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        for (stage_22_v90, (stage_22_v97,)) in pl.range(1, init_values=(stage_22_v80,)):
                            stage_22_v91: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_22_v79, [1, 64], [stage_22_v90, 0], [1, 64])
                            stage_22_v92: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_22_v45, [1, 64], [stage_22_v90, 0], [1, 64])
                            stage_22_v93: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                            stage_22_v94: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_22_v91, stage_22_v92, stage_22_v93)
                            stage_22_v95: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_22_v97, stage_22_v94, [stage_22_v90, 0])
                            stage_22_v96: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_22_v95)
                        stage_22_v81: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = stage_22_v96
                        stage_22_v82: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_22_v79, stage_22_v65)
                        stage_22_v83: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_22_v81, stage_22_v66)
                        stage_22_v84: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_22_v82, stage_22_v83)
                        stage_22_v85: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_22_v84, target_type=pl.BF16, mode='rint')
                        stage_22_v86 = pl.tile.store(stage_22_v78, [stage_22_v63, stage_22_v70], stage_22_v89)
                        stage_22_v87 = pl.tile.store(stage_22_v85, [stage_22_v63, stage_22_v70 + 64], stage_22_v86)
                        stage_22_v88 = pl.yield_(stage_22_v87)
                    stage_22_v67 = pl.yield_(stage_22_v88)
                stage_22_v44 = pl.yield_(stage_22_v67)
            stage_22_v24 = pl.yield_(stage_22_v44)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_22 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_22)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_23 = 27216 + core_id * 6
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
        stage_23_v2: pl.Scalar[pl.INDEX] = block_23
        stage_23_v3: pl.Scalar[pl.INDEX] = stage_23_v2 * 2
        stage_23_v4: pl.Tile[[2, 128], pl.BF16, pl.Mem.Vec] = pl.tile.full([2, 128], dtype=pl.BF16, value=0.0)
        stage_23_v5 = pl.tile.store(stage_23_v4, [stage_23_v3, 0], temporary_33)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_23 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_23)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_24 = 27648 + core_id * 6
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
        stage_24_v7: pl.Scalar[pl.INDEX] = block_24
        for (stage_24_v8, (stage_24_v10,)) in pl.range(stage_24_v7, 16, 16, init_values=(arg_inner_compress_state,)):
            for (stage_24_v11, (stage_24_v16,)) in pl.range(6, init_values=(stage_24_v10,)):
                stage_24_v12: pl.Scalar[pl.INDEX] = stage_24_v8 * 6 + stage_24_v11
                stage_24_v13: pl.Scalar[pl.INT32] = pl.tensor.read(arg_inner_state_slot_mapping, [stage_24_v12, 0])
                stage_24_v14: pl.Scalar[pl.INT32] = pl.tensor.read(arg_inner_state_slot_mapping, [stage_24_v12, 1])
                if 0 <= pl.cast(stage_24_v13, pl.INDEX) and 0 <= pl.cast(stage_24_v14, pl.INDEX):
                    stage_24_v17: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_24_v12])
                    stage_24_v18: pl.Scalar[pl.INDEX] = pl.cast(stage_24_v13, pl.INDEX)
                    stage_24_v19: pl.Scalar[pl.INDEX] = pl.cast(stage_24_v14, pl.INDEX) * 512
                    stage_24_v20: pl.Scalar[pl.INDEX] = pl.cast(stage_24_v17 % 4, pl.INDEX)
                    stage_24_v21: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_34, [stage_24_v12, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                    stage_24_v22 = pl.tile.store(stage_24_v21, [stage_24_v18, stage_24_v19], stage_24_v16)
                    stage_24_v23: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_35, [stage_24_v12, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                    stage_24_v24: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_ape, [stage_24_v20, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                    stage_24_v25: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_24_v23, stage_24_v24)
                    stage_24_v26 = pl.tile.store(stage_24_v25, [stage_24_v18, stage_24_v19 + 256], stage_24_v22)
                    stage_24_v27 = pl.yield_(stage_24_v26)
                else:
                    stage_24_v27 = pl.yield_(stage_24_v16)
                stage_24_v15 = pl.yield_(stage_24_v27)
            stage_24_v9 = pl.yield_(stage_24_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_24 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_24)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_25 = 28080 + core_id * 6
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
        for (stage_25_v5, (stage_25_v21, stage_25_v22)) in pl.range(stage_25_v4, 6144 // 64, 48, init_values=(temporary_39, temporary_40)):
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
    record_26 = 28512 + core_id * 6
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
        stage_26_v8: pl.Scalar[pl.INDEX] = block_26
        stage_26_v9: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        stage_26_v10: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_26_v11: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_26_v10, target_type=pl.FP32, mode='round')
        stage_26_v12: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_26_v9, stage_26_v11)
        stage_26_v13: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_26_v12, 0.5)
        stage_26_v14: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_26_v13, target_type=pl.INT32, mode='trunc')
        stage_26_v15: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_26_v14, target_type=pl.FP32, mode='round')
        stage_26_v16: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_26_v15, 2.0)
        stage_26_v17: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_26_v12, stage_26_v16)
        stage_26_v18: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_26_v12, 1.0)
        stage_26_v19: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_26_v17, 2.0)
        stage_26_v20: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_26_v18, stage_26_v19)
        stage_26_v21: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_26_v20, target_type=pl.INT32, mode='round')
        for (stage_26_v22, (stage_26_v69,)) in pl.range(stage_26_v8, 6, 2, init_values=(temporary_33,)):
            stage_26_v23: pl.Scalar[pl.INDEX] = stage_26_v22 * 16
            stage_26_v24: pl.Scalar[pl.INDEX] = pl.min(96 - stage_26_v23, 16)
            stage_26_v25: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
            stage_26_v26: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
            for (stage_26_v70, (stage_26_v84, stage_26_v85)) in pl.range(stage_26_v24, init_values=(stage_26_v25, stage_26_v26)):
                stage_26_v71: pl.Scalar[pl.INDEX] = stage_26_v23 + stage_26_v70
                stage_26_v72: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_26_v71])
                if (stage_26_v72 + 1) % 4 == 0:
                    stage_26_v86: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_15, [stage_26_v71 // 6])
                    stage_26_v87: pl.Scalar[pl.INDEX] = pl.cast(stage_26_v86, pl.INDEX) + pl.cast((stage_26_v72 + 1) // 4, pl.INDEX)
                    stage_26_v88: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_inner_freqs_cos, 0)
                    if stage_26_v87 < stage_26_v88:
                        stage_26_v102: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather_row(stage_26_v84, arg_inner_freqs_cos, [stage_26_v70, 0], [stage_26_v87, 0], [1, 64], transpose=False)
                        stage_26_v103: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather_row(stage_26_v85, arg_inner_freqs_sin, [stage_26_v70, 0], [stage_26_v87, 0], [1, 64], transpose=False)
                        (stage_26_v106, stage_26_v107) = pl.yield_(stage_26_v102, stage_26_v103)
                    else:
                        (stage_26_v106, stage_26_v107) = pl.yield_(stage_26_v84, stage_26_v85)
                    (stage_26_v104, stage_26_v105) = pl.yield_(stage_26_v106, stage_26_v107)
                else:
                    (stage_26_v104, stage_26_v105) = pl.yield_(stage_26_v84, stage_26_v85)
                (stage_26_v89, stage_26_v90) = pl.yield_(stage_26_v104, stage_26_v105)
            stage_26_v27: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_26_v89
            stage_26_v28: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_26_v90
            stage_26_v29: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_36, [stage_26_v23, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            stage_26_v30: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_36, [stage_26_v23, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            stage_26_v31: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_26_v29, stage_26_v29)
            stage_26_v32: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_26_v30, stage_26_v30)
            stage_26_v33: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_26_v31, stage_26_v32)
            stage_26_v34: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_26_v35: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_26_v33, stage_26_v34)
            stage_26_v36: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_26_v35, [1, 16])
            stage_26_v37: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_26_v36, 0.0078125)
            stage_26_v38: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_26_v37, [16, 1])
            stage_26_v39: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_26_v38, [1, 16])
            stage_26_v40: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_26_v39, 1e-06)
            stage_26_v41: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_26_v40, [16, 1])
            stage_26_v42: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_26_v41, [1, 16])
            stage_26_v43: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_26_v42)
            stage_26_v44: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_26_v43, [16, 1])
            stage_26_v45: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_26_v44, [1, 16])
            stage_26_v46: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_26_v45)
            stage_26_v47: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_26_v46, [16, 1])
            stage_26_v48: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_36, [stage_26_v23, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            stage_26_v49: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_76, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
            stage_26_v50: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_26_v49, target_type=pl.FP32, mode='round')
            stage_26_v51: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_26_v48, stage_26_v47)
            stage_26_v52: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_26_v51, stage_26_v50)
            stage_26_v53: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_26_v52, target_type=pl.BF16, mode='rint')
            stage_26_v54: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_36, [stage_26_v23, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            stage_26_v55: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_76, [0, 64], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
            stage_26_v56: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_26_v55, target_type=pl.FP32, mode='round')
            stage_26_v57: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_26_v54, stage_26_v47)
            stage_26_v58: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_26_v57, stage_26_v56)
            stage_26_v59: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for (stage_26_v73, (stage_26_v91,)) in pl.range(16, init_values=(stage_26_v59,)):
                stage_26_v74: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_26_v58, [1, 64], [stage_26_v73, 0], [1, 64])
                stage_26_v75: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_26_v21, [1, 64], [stage_26_v73, 0], [1, 64])
                stage_26_v76: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                stage_26_v77: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_26_v74, stage_26_v75, stage_26_v76)
                stage_26_v78: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_26_v91, stage_26_v77, [stage_26_v73, 0])
                stage_26_v79: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_26_v78)
            stage_26_v60: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_26_v79
            stage_26_v61: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_26_v17, 2.0)
            stage_26_v62: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_26_v61, 1.0)
            stage_26_v63: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_26_v28, stage_26_v62)
            stage_26_v64: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_26_v58, stage_26_v27)
            stage_26_v65: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_26_v60, stage_26_v63)
            stage_26_v66: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_26_v64, stage_26_v65)
            stage_26_v67: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_26_v66, target_type=pl.BF16, mode='rint')
            for (stage_26_v80, (stage_26_v92,)) in pl.range(stage_26_v24, init_values=(stage_26_v69,)):
                stage_26_v81: pl.Scalar[pl.INDEX] = stage_26_v23 + stage_26_v80
                stage_26_v82: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_26_v81])
                if (stage_26_v82 + 1) % 4 == 0:
                    stage_26_v93: pl.Scalar[pl.INDEX] = stage_26_v81 // 6
                    stage_26_v94: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_26_v93 * 6])
                    stage_26_v95: pl.Scalar[pl.INDEX] = 3 - stage_26_v94 % 4
                    stage_26_v96: pl.Scalar[pl.INDEX] = stage_26_v93 * 2 + (stage_26_v81 % 6 - stage_26_v95) // 4
                    stage_26_v97: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_26_v53, [1, 64], [stage_26_v80, 0])
                    stage_26_v98 = pl.tile.store(stage_26_v97, [stage_26_v96, 0], stage_26_v92)
                    stage_26_v99: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_26_v67, [1, 64], [stage_26_v80, 0])
                    stage_26_v100 = pl.tile.store(stage_26_v99, [stage_26_v96, 64], stage_26_v98)
                    stage_26_v101 = pl.yield_(stage_26_v100)
                else:
                    stage_26_v101 = pl.yield_(stage_26_v92)
                stage_26_v83 = pl.yield_(stage_26_v101)
            stage_26_v68 = pl.yield_(stage_26_v83)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_26 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_26)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_27 = 28944 + core_id * 6
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
        for (stage_27_v5, (stage_27_v21, stage_27_v22)) in pl.range(stage_27_v4, 6144 // 64, 48, init_values=(temporary_39, temporary_40)):
            stage_27_v6: pl.Scalar[pl.INDEX] = stage_27_v5 * 64
            stage_27_v7: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_43, [stage_27_v6, 0], [64, 128], [64, 128], target_memory=pl.Mem.Vec)
            stage_27_v8: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_27_v7, 0.08838834764831845)
            stage_27_v9: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0001)
            for (stage_27_v23, (stage_27_v35,)) in pl.range(0, 128, 64, init_values=(stage_27_v9,)):
                stage_27_v24: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_27_v8, 0, stage_27_v23, [64, 64], target_memory=pl.Mem.Vec)
                stage_27_v25: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_27_v24)
                stage_27_v26: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_27_v8, 0, stage_27_v23, [64, 64], target_memory=pl.Mem.Vec)
                stage_27_v27: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_27_v26, stage_27_v25)
                stage_27_v28: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_27_v29: pl.Tile[[64, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_27_v27, stage_27_v28)
                stage_27_v30: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_27_v29, [1, 64])
                stage_27_v31: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_27_v35, stage_27_v30)
                stage_27_v32: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_27_v31)
            stage_27_v10: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=127.0)
            stage_27_v11: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_27_v10, stage_27_v32)
            stage_27_v12: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_27_v11)
            stage_27_v13: pl.Tile[[64, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_27_v12, [64, 1])
            stage_27_v14 = pl.tile.store(stage_27_v13, [stage_27_v6, 0], stage_27_v22)
            stage_27_v15: pl.Tile[[64, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_27_v11, [64, 1])
            stage_27_v16: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_27_v8, stage_27_v15)
            stage_27_v17: pl.Tile[[64, 128], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_27_v16, target_type=pl.INT32, mode='rint')
            stage_27_v18: pl.Tile[[64, 128], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_27_v17, target_type=pl.FP16, mode='round')
            stage_27_v19: pl.Tile[[64, 128], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_27_v18, target_type=pl.INT8, mode='trunc')
            stage_27_v20 = pl.tile.store(stage_27_v19, [stage_27_v6, 0], stage_27_v21)
            (stage_27_v33, stage_27_v34) = pl.yield_(stage_27_v20, stage_27_v14)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_27 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_27)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_28 = 29376 + core_id * 6
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
        stage_28_v9: pl.Scalar[pl.INDEX] = block_28
        stage_28_v10: pl.Scalar[pl.INDEX] = stage_28_v9 * 16
        stage_28_v11: pl.Scalar[pl.INDEX] = pl.min(32 - stage_28_v10, 16)
        stage_28_v12: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_38, [stage_28_v10, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec)
        stage_28_v13: pl.Tile[[16, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_28_v12, target_type=pl.BF16, mode='rint')
        stage_28_v14: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_28_v13, target_type=pl.FP32, mode='round')
        stage_28_v15: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_28_v14, 0.08838834764831845)
        stage_28_v16: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_28_v15)
        stage_28_v17: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_28_v18: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_28_v16, stage_28_v17)
        stage_28_v19: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_28_v18, [1, 16])
        stage_28_v20: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0001)
        stage_28_v21: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_28_v19, stage_28_v20)
        stage_28_v22: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=127.0)
        stage_28_v23: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_28_v22, stage_28_v21)
        stage_28_v24: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_28_v23)
        stage_28_v25: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_28_v24, [16, 1])
        stage_28_v26: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_28_v23, [16, 1])
        stage_28_v27 = pl.tile.store(stage_28_v25, [stage_28_v10, 0], temporary_37)
        stage_28_v28: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_28_v15, stage_28_v26)
        stage_28_v29: pl.Tile[[16, 128], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_28_v28, target_type=pl.INT32, mode='rint')
        stage_28_v30: pl.Tile[[16, 128], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_28_v29, target_type=pl.FP16, mode='round')
        stage_28_v31: pl.Tile[[16, 128], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_28_v30, target_type=pl.INT8, mode='trunc')
        for (stage_28_v32, (stage_28_v38, stage_28_v39)) in pl.range(stage_28_v11, init_values=(arg_idx_native_kv_cache, temporary_32)):
            stage_28_v33: pl.Scalar[pl.INDEX] = stage_28_v10 + stage_28_v32
            stage_28_v34: pl.Scalar[pl.INDEX] = stage_28_v33 // 2
            stage_28_v35: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_28_v34 * 6])
            stage_28_v36: pl.Scalar[pl.INDEX] = stage_28_v33 % 2 * 4 - stage_28_v35 % 4 + 3
            stage_28_v37: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_28_v34])
            if 0 < pl.cast(stage_28_v37, pl.INDEX) and stage_28_v36 < 6:
                stage_28_v40: pl.Scalar[pl.INDEX] = stage_28_v34 * 6 + stage_28_v36
                stage_28_v41: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_28_v40])
                stage_28_v42: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_15, [stage_28_v34])
                stage_28_v43: pl.Scalar[pl.INDEX] = pl.cast(stage_28_v42, pl.INDEX) + pl.cast((stage_28_v41 + 1) // 4, pl.INDEX)
                stage_28_v44: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_idx_slot_mapping, 0)
                stage_28_v45: pl.Scalar[pl.INDEX] = pl.min(stage_28_v43, stage_28_v44 - 1)
                stage_28_v46: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_slot_mapping, [stage_28_v45, 0])
                stage_28_v47: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_slot_mapping, [stage_28_v45, 1])
                if stage_28_v43 < stage_28_v44 and 0 <= pl.cast(stage_28_v46, pl.INDEX) and (0 <= pl.cast(stage_28_v47, pl.INDEX)):
                    stage_28_v50: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_28_v15, [1, 128], [stage_28_v32, 0])
                    stage_28_v51 = pl.tile.store(stage_28_v50, [stage_28_v40, 0], stage_28_v39)
                    stage_28_v52: pl.Scalar[pl.INDEX] = pl.cast(stage_28_v46, pl.INDEX)
                    stage_28_v53: pl.Scalar[pl.INDEX] = pl.cast(stage_28_v47, pl.INDEX) * 128
                    stage_28_v54: pl.Tile[[1, 128], pl.INT8, pl.Mem.Vec] = pl.tile.slice(stage_28_v31, [1, 128], [stage_28_v32, 0])
                    stage_28_v55 = pl.tile.store(stage_28_v54, [stage_28_v52, stage_28_v53], stage_28_v38)
                    (stage_28_v58, stage_28_v59) = pl.yield_(stage_28_v55, stage_28_v51)
                else:
                    (stage_28_v58, stage_28_v59) = pl.yield_(stage_28_v38, stage_28_v39)
                (stage_28_v56, stage_28_v57) = pl.yield_(stage_28_v58, stage_28_v59)
            else:
                (stage_28_v56, stage_28_v57) = pl.yield_(stage_28_v38, stage_28_v39)
            (stage_28_v48, stage_28_v49) = pl.yield_(stage_28_v56, stage_28_v57)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_28 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_28)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_29 = 29808 + core_id * 6
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
        for stage_29_v6 in pl.range(32):
            stage_29_v7: pl.Scalar[pl.INDEX] = stage_29_v6 // 2
            stage_29_v8: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v7 * 6])
            stage_29_v9: pl.Scalar[pl.INDEX] = stage_29_v6 % 2 * 4 - stage_29_v8 % 4 + 3
            stage_29_v10: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_29_v7])
            if 0 < pl.cast(stage_29_v10, pl.INDEX) and stage_29_v9 < 6:
                stage_29_v11: pl.Scalar[pl.INDEX] = stage_29_v7 * 6 + stage_29_v9
                stage_29_v12: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v11])
                stage_29_v13: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_15, [stage_29_v7])
                stage_29_v14: pl.Scalar[pl.INDEX] = pl.cast(stage_29_v13, pl.INDEX) + pl.cast((stage_29_v12 + 1) // 4, pl.INDEX)
                stage_29_v15: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_idx_slot_mapping, 0)
                stage_29_v16: pl.Scalar[pl.INDEX] = pl.min(stage_29_v14, stage_29_v15 - 1)
                stage_29_v17: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_slot_mapping, [stage_29_v16, 0])
                stage_29_v18: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_slot_mapping, [stage_29_v16, 1])
                if stage_29_v14 < stage_29_v15 and 0 <= pl.cast(stage_29_v17, pl.INDEX) and (0 <= pl.cast(stage_29_v18, pl.INDEX)):
                    stage_29_v19: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_37, [stage_29_v6, 0])
                    stage_29_v20: pl.Scalar[pl.FP16] = pl.cast(stage_29_v19, pl.FP16)
                    stage_29_v21: pl.Scalar[pl.INDEX] = pl.cast(stage_29_v17, pl.INDEX)
                    stage_29_v22: pl.Tile[[1, 64], pl.INT8, pl.Mem.Vec] = pl.tile.load(arg_idx_native_kv_cache, [stage_29_v21, 4096], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_29_v23: pl.Tile[[1, 32], pl.FP16, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v22, dtype=pl.FP16)
                    pl.tile.write(stage_29_v23, [0, stage_29_v18], stage_29_v20)
                    stage_29_v24: pl.Tile[[1, 64], pl.INT8, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v23, dtype=pl.INT8)
                    stage_29_v25 = pl.tile.store(stage_29_v24, [stage_29_v21, 4096], arg_idx_native_kv_cache)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_29 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_29)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_30 = 30240 + core_id * 6
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
        stage_30_v5: pl.Scalar[pl.INDEX] = block_30
        stage_30_v6: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        for stage_30_v7 in pl.range(stage_30_v5, stage_30_v6 // 6, 48):
            stage_30_v8: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.full([96, 64], dtype=pl.FP16, value=0.0)
            stage_30_v9: pl.Scalar[pl.INDEX] = stage_30_v7 * 6
            stage_30_v10: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_94, [stage_30_v9, 0], [6, 64], [6, 64], target_memory=pl.Mem.Vec)
            stage_30_v11: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_44, [stage_30_v9, 0], [6, 64], [6, 64], target_memory=pl.Mem.Vec)
            stage_30_v12: pl.Tile[[6, 64], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_30_v10, target_type=pl.FP16, mode='rint')
            stage_30_v13: pl.Tile[[6, 64], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_30_v11, target_type=pl.FP16, mode='rint')
            stage_30_v14: pl.Tile[[6, 64], pl.FP16, pl.Mem.Vec] = pl.tile.mul(stage_30_v12, stage_30_v13)
            stage_30_v15: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_30_v14, 0, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_30_v16: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_30_v8, stage_30_v15, [0, 0])
            stage_30_v17: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_30_v14, 1, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_30_v18: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_30_v16, stage_30_v17, [7, 0])
            stage_30_v19: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_30_v14, 2, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_30_v20: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_30_v18, stage_30_v19, [14, 0])
            stage_30_v21: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_30_v14, 3, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_30_v22: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_30_v20, stage_30_v21, [21, 0])
            stage_30_v23: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_30_v14, 4, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_30_v24: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_30_v22, stage_30_v23, [28, 0])
            stage_30_v25: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_30_v14, 5, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_30_v26: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_30_v24, stage_30_v25, [35, 0])
            stage_30_v27: pl.Tile[[16, 384], pl.FP16, pl.Mem.Vec] = pl.tile.reshape(stage_30_v26, [16, 384])
            stage_30_v28 = pl.tile.store(stage_30_v27, [stage_30_v7 * 16, 0], temporary_48)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_30 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_30)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_31 = 30672 + core_id * 6
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
        stage_31_v13: pl.Scalar[pl.INDEX] = block_31
        stage_31_v14: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        stage_31_v15: pl.Scalar[pl.INDEX] = 0
        for (stage_31_v27, (stage_31_v41,)) in pl.range(stage_31_v14 // 6, init_values=(stage_31_v15,)):
            stage_31_v28: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_31_v27])
            stage_31_v29: pl.Scalar[pl.INDEX] = pl.max(stage_31_v41, pl.cast(stage_31_v28, pl.INDEX) // 4)
            stage_31_v30: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v29)
        stage_31_v16: pl.Scalar[pl.INDEX] = pl.min(stage_31_v30, 262144)
        stage_31_v17: pl.Scalar[pl.INDEX] = (stage_31_v16 + 1023) // 1024
        stage_31_v18: pl.Scalar[pl.INDEX] = pl.max((stage_31_v16 + 8191) // 8192, 1)
        stage_31_v19: pl.Scalar[pl.INDEX] = 8
        stage_31_v20: pl.Scalar[pl.INDEX] = 0
        stage_31_v21: pl.Scalar[pl.INDEX] = 1
        if stage_31_v14 // 6 * 3 == 48:
            stage_31_v31: pl.Scalar[pl.INDEX] = 3
            stage_31_v32: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v31)
        else:
            if stage_31_v14 // 6 * 3 == 24:
                stage_31_v42: pl.Scalar[pl.INDEX] = 3
                stage_31_v43: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v42)
            else:
                if stage_31_v14 // 6 * 6 == 24:
                    stage_31_v67: pl.Scalar[pl.INDEX] = 6
                    stage_31_v68: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v67)
                else:
                    stage_31_v68: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v21)
                stage_31_v43: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v68)
            stage_31_v32: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v43)
        if 1 < stage_31_v32:
            stage_31_v33: pl.Scalar[pl.INDEX] = (stage_31_v18 + stage_31_v32 - 1) // stage_31_v32 * stage_31_v32
            if stage_31_v33 <= 32:
                stage_31_v44: pl.Scalar[pl.INDEX] = stage_31_v33
                stage_31_v45: pl.Scalar[pl.INDEX] = pl.max(stage_31_v17 // stage_31_v33, 1)
                stage_31_v46: pl.Scalar[pl.INDEX] = stage_31_v17 % stage_31_v33
                (stage_31_v69, stage_31_v70, stage_31_v71) = pl.yield_(stage_31_v46, stage_31_v44, stage_31_v45)
            else:
                (stage_31_v69, stage_31_v70, stage_31_v71) = pl.yield_(stage_31_v20, stage_31_v18, stage_31_v19)
            (stage_31_v47, stage_31_v48, stage_31_v49) = pl.yield_(stage_31_v69, stage_31_v70, stage_31_v71)
        else:
            (stage_31_v47, stage_31_v48, stage_31_v49) = pl.yield_(stage_31_v20, stage_31_v18, stage_31_v19)
        stage_31_v22: pl.Scalar[pl.INDEX] = stage_31_v48
        stage_31_v23: pl.Scalar[pl.INDEX] = stage_31_v49
        stage_31_v24: pl.Scalar[pl.INDEX] = stage_31_v47
        pl.system.set_ffts(temporary_50)
        stage_31_v25: pl.Scalar[pl.INDEX] = subslot_31
        for stage_31_v26 in pl.range(stage_31_v13, stage_31_v14 // 6 * stage_31_v22, 24):
            if stage_31_v14 < 144:
                stage_31_v50: pl.Scalar[pl.INDEX] = stage_31_v26 % (stage_31_v14 // 6) * 6
                stage_31_v51: pl.Scalar[pl.INDEX] = stage_31_v26 // (stage_31_v14 // 6)
                (stage_31_v72, stage_31_v73) = pl.yield_(stage_31_v51, stage_31_v50)
            else:
                stage_31_v52: pl.Scalar[pl.INDEX] = stage_31_v26 // stage_31_v22 * 6
                stage_31_v53: pl.Scalar[pl.INDEX] = stage_31_v26 % stage_31_v22
                (stage_31_v72, stage_31_v73) = pl.yield_(stage_31_v53, stage_31_v52)
            stage_31_v34: pl.Scalar[pl.INDEX] = stage_31_v73 // 6
            stage_31_v35: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_31_v34])
            stage_31_v36: pl.Scalar[pl.INDEX] = pl.cast(stage_31_v35, pl.INDEX) // 4
            stage_31_v37: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_31_v73 + 5])
            stage_31_v38: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_31_v36, (stage_31_v37 + 1) // 4), 262144), 0)
            stage_31_v39: pl.Scalar[pl.INDEX] = (stage_31_v72 * stage_31_v23 + pl.min(stage_31_v72, stage_31_v24)) * 1024
            stage_31_v40: pl.Scalar[pl.INDEX] = (stage_31_v23 + pl.cast(stage_31_v72 < stage_31_v24, pl.INDEX)) * 1024
            if stage_31_v39 < stage_31_v38:
                stage_31_v54: pl.Scalar[pl.INDEX] = pl.min(stage_31_v40, stage_31_v38 - stage_31_v39)
                stage_31_v55: pl.Scalar[pl.INDEX] = (stage_31_v54 + 1023) // 1024
                stage_31_v56: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([3, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
                stage_31_v57: pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([6, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
                for (stage_31_v74, (stage_31_v90, stage_31_v91)) in pl.range(stage_31_v55, init_values=(stage_31_v56, stage_31_v57)):
                    stage_31_v75: pl.Scalar[pl.INDEX] = stage_31_v39 + stage_31_v74 * 1024
                    stage_31_v76: pl.Scalar[pl.INDEX] = stage_31_v13 * 12 + stage_31_v74 % 2 * 6 + stage_31_v25 * 3
                    stage_31_v77: pl.Tile[[1, 2048], pl.INT8, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.INT8, target_memory=pl.Mem.Vec)
                    for (stage_31_v92, (stage_31_v116,)) in pl.range(32, init_values=(stage_31_v77,)):
                        stage_31_v93: pl.Scalar[pl.INDEX] = pl.min(stage_31_v75 // 32 + stage_31_v92, pl.max((stage_31_v36 - 1) // 32, 0))
                        stage_31_v94: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_31_v34, stage_31_v93])
                        stage_31_v95: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_31_v94, pl.INDEX), 0)
                        stage_31_v96: pl.Tile[[1, 2048], pl.INT8, pl.Mem.Vec] = pl.tile.gather_row(stage_31_v116, arg_idx_native_kv_cache, [0, stage_31_v92 * 64], [stage_31_v95, 4096], [1, 64], transpose=False)
                        stage_31_v97: pl.Tile[[1, 2048], pl.INT8, pl.Mem.Vec] = pl.yield_(stage_31_v96)
                    stage_31_v78: pl.Tile[[1, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_31_v97, dtype=pl.FP16)
                    stage_31_v79: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_31_v78, target_type=pl.FP32, mode='round')
                    pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    stage_31_v80: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_49, [stage_31_v76, 0], [3, 1024], [3, 1024], target_memory=pl.Mem.Vec)
                    pl.system.sync_set(1, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV)
                    stage_31_v81: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v80, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                    stage_31_v82: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_31_v81, stage_31_v79)
                    stage_31_v83: pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_31_v91, stage_31_v82, [stage_31_v74 % 2, 0])
                    stage_31_v84: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v80, 1, 0, [1, 1024], target_memory=pl.Mem.Vec)
                    stage_31_v85: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_31_v84, stage_31_v79)
                    stage_31_v86: pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_31_v83, stage_31_v85, [stage_31_v74 % 2 + 2, 0])
                    stage_31_v87: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v80, 2, 0, [1, 1024], target_memory=pl.Mem.Vec)
                    stage_31_v88: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_31_v87, stage_31_v79)
                    stage_31_v89: pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_31_v86, stage_31_v88, [stage_31_v74 % 2 + 4, 0])
                    if stage_31_v74 % 2 == 1 or stage_31_v74 + 1 == stage_31_v55:
                        stage_31_v98: pl.Scalar[pl.INDEX] = stage_31_v39 + stage_31_v74 // 2 * 2048
                        stage_31_v99: pl.Scalar[pl.INDEX] = stage_31_v74 % 2 * 1024 + 1024
                        stage_31_v100: pl.Tile[[3, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v89, [3, 2048])
                        stage_31_v101: pl.Scalar[pl.INDEX] = stage_31_v25 * 3
                        stage_31_v102: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_31_v73 + stage_31_v101])
                        stage_31_v103: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_31_v36, (stage_31_v102 + 1) // 4), 262144), 0)
                        stage_31_v104: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_31_v103 - stage_31_v98, stage_31_v99), 0)
                        if 0 < stage_31_v104:
                            stage_31_v117: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v100, 0, 0, [1, 2048], target_memory=pl.Mem.Vec)
                            stage_31_v118: pl.Scalar[pl.INT32] = pl.cast(stage_31_v98, pl.INT32)
                            if stage_31_v104 <= 512:
                                stage_31_v126: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                                stage_31_v127: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_31_v126, stage_31_v118)
                                stage_31_v128: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v117, 0, 0, [1, 512], target_memory=pl.Mem.Vec)
                                stage_31_v129: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v104])] = pl.tile.slice(stage_31_v128, [1, 512], [0, 0], valid_shape=[1, stage_31_v104])
                                stage_31_v130: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_31_v129, pad_value=pl.PadValue.min)
                                stage_31_v131: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_31_v130, -3.4028234663852886e+38)
                                stage_31_v132: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_31_v127, dtype=pl.UINT32)
                                stage_31_v133: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_31_v131, stage_31_v132)
                                stage_31_v134: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v133, pl.const(64, pl.INT32))
                                stage_31_v135: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v134, pl.const(256, pl.INT32))
                                stage_31_v136: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v135, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                stage_31_v137: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v136)
                            else:
                                if stage_31_v104 <= 1024:
                                    stage_31_v183: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                                    stage_31_v184: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_31_v183, stage_31_v118)
                                    stage_31_v185: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v117, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                    stage_31_v186: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v104])] = pl.tile.slice(stage_31_v185, [1, 1024], [0, 0], valid_shape=[1, stage_31_v104])
                                    stage_31_v187: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_31_v186, pad_value=pl.PadValue.min)
                                    stage_31_v188: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_31_v187, -3.4028234663852886e+38)
                                    stage_31_v189: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_31_v184, dtype=pl.UINT32)
                                    stage_31_v190: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_31_v188, stage_31_v189)
                                    stage_31_v191: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v190, pl.const(64, pl.INT32))
                                    stage_31_v192: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v191, pl.const(256, pl.INT32))
                                    stage_31_v193: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_31_v192, [1, 1024], [0, 0])
                                    stage_31_v194: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_31_v192, [1, 1024], [0, 1024])
                                    stage_31_v195: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                    stage_31_v196: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_31_v193, stage_31_v194, stage_31_v195, exhausted=False)
                                    stage_31_v197: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v196, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                    stage_31_v198: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v197)
                                else:
                                    stage_31_v199: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                                    stage_31_v200: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_31_v199, stage_31_v118)
                                    stage_31_v201: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v117, 0, 0, [1, 2048], target_memory=pl.Mem.Vec)
                                    stage_31_v202: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v104])] = pl.tile.slice(stage_31_v201, [1, 2048], [0, 0], valid_shape=[1, stage_31_v104])
                                    stage_31_v203: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_31_v202, pad_value=pl.PadValue.min)
                                    stage_31_v204: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_31_v203, -3.4028234663852886e+38)
                                    stage_31_v205: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_31_v200, dtype=pl.UINT32)
                                    stage_31_v206: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_31_v204, stage_31_v205)
                                    stage_31_v207: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v206, pl.const(64, pl.INT32))
                                    stage_31_v208: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v207, pl.const(256, pl.INT32))
                                    stage_31_v209: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v208, pl.const(1024, pl.INT32))
                                    stage_31_v210: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v209, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                    stage_31_v198: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v210)
                                stage_31_v137: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v198)
                            if stage_31_v74 < 2:
                                stage_31_v138: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_31_v90, stage_31_v137, [0, 0])
                                stage_31_v139: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v138)
                            else:
                                stage_31_v140: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v90, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                stage_31_v141: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                stage_31_v142: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(stage_31_v137, stage_31_v140, stage_31_v141, exhausted=False)
                                stage_31_v143: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v142, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                stage_31_v144: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_31_v90, stage_31_v143, [0, 0])
                                stage_31_v139: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v144)
                            stage_31_v119: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v139)
                        else:
                            stage_31_v119: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v90)
                        stage_31_v105: pl.Scalar[pl.INDEX] = stage_31_v25 * 3 + 1
                        stage_31_v106: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_31_v73 + stage_31_v105])
                        stage_31_v107: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_31_v36, (stage_31_v106 + 1) // 4), 262144), 0)
                        stage_31_v108: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_31_v107 - stage_31_v98, stage_31_v99), 0)
                        if 0 < stage_31_v108:
                            stage_31_v120: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v100, 1, 0, [1, 2048], target_memory=pl.Mem.Vec)
                            stage_31_v121: pl.Scalar[pl.INT32] = pl.cast(stage_31_v98, pl.INT32)
                            if stage_31_v108 <= 512:
                                stage_31_v145: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                                stage_31_v146: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_31_v145, stage_31_v121)
                                stage_31_v147: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v120, 0, 0, [1, 512], target_memory=pl.Mem.Vec)
                                stage_31_v148: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v108])] = pl.tile.slice(stage_31_v147, [1, 512], [0, 0], valid_shape=[1, stage_31_v108])
                                stage_31_v149: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_31_v148, pad_value=pl.PadValue.min)
                                stage_31_v150: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_31_v149, -3.4028234663852886e+38)
                                stage_31_v151: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_31_v146, dtype=pl.UINT32)
                                stage_31_v152: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_31_v150, stage_31_v151)
                                stage_31_v153: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v152, pl.const(64, pl.INT32))
                                stage_31_v154: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v153, pl.const(256, pl.INT32))
                                stage_31_v155: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v154, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                stage_31_v156: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v155)
                            else:
                                if stage_31_v108 <= 1024:
                                    stage_31_v211: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                                    stage_31_v212: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_31_v211, stage_31_v121)
                                    stage_31_v213: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v120, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                    stage_31_v214: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v108])] = pl.tile.slice(stage_31_v213, [1, 1024], [0, 0], valid_shape=[1, stage_31_v108])
                                    stage_31_v215: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_31_v214, pad_value=pl.PadValue.min)
                                    stage_31_v216: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_31_v215, -3.4028234663852886e+38)
                                    stage_31_v217: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_31_v212, dtype=pl.UINT32)
                                    stage_31_v218: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_31_v216, stage_31_v217)
                                    stage_31_v219: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v218, pl.const(64, pl.INT32))
                                    stage_31_v220: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v219, pl.const(256, pl.INT32))
                                    stage_31_v221: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_31_v220, [1, 1024], [0, 0])
                                    stage_31_v222: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_31_v220, [1, 1024], [0, 1024])
                                    stage_31_v223: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                    stage_31_v224: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_31_v221, stage_31_v222, stage_31_v223, exhausted=False)
                                    stage_31_v225: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v224, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                    stage_31_v226: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v225)
                                else:
                                    stage_31_v227: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                                    stage_31_v228: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_31_v227, stage_31_v121)
                                    stage_31_v229: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v120, 0, 0, [1, 2048], target_memory=pl.Mem.Vec)
                                    stage_31_v230: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v108])] = pl.tile.slice(stage_31_v229, [1, 2048], [0, 0], valid_shape=[1, stage_31_v108])
                                    stage_31_v231: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_31_v230, pad_value=pl.PadValue.min)
                                    stage_31_v232: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_31_v231, -3.4028234663852886e+38)
                                    stage_31_v233: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_31_v228, dtype=pl.UINT32)
                                    stage_31_v234: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_31_v232, stage_31_v233)
                                    stage_31_v235: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v234, pl.const(64, pl.INT32))
                                    stage_31_v236: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v235, pl.const(256, pl.INT32))
                                    stage_31_v237: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v236, pl.const(1024, pl.INT32))
                                    stage_31_v238: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v237, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                    stage_31_v226: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v238)
                                stage_31_v156: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v226)
                            if stage_31_v74 < 2:
                                stage_31_v157: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_31_v119, stage_31_v156, [1, 0])
                                stage_31_v158: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v157)
                            else:
                                stage_31_v159: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v119, 1, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                stage_31_v160: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                stage_31_v161: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(stage_31_v156, stage_31_v159, stage_31_v160, exhausted=False)
                                stage_31_v162: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v161, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                stage_31_v163: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_31_v119, stage_31_v162, [1, 0])
                                stage_31_v158: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v163)
                            stage_31_v122: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v158)
                        else:
                            stage_31_v122: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v119)
                        stage_31_v109: pl.Scalar[pl.INDEX] = stage_31_v25 * 3 + 2
                        stage_31_v110: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_31_v73 + stage_31_v109])
                        stage_31_v111: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_31_v36, (stage_31_v110 + 1) // 4), 262144), 0)
                        stage_31_v112: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_31_v111 - stage_31_v98, stage_31_v99), 0)
                        if 0 < stage_31_v112:
                            stage_31_v123: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v100, 2, 0, [1, 2048], target_memory=pl.Mem.Vec)
                            stage_31_v124: pl.Scalar[pl.INT32] = pl.cast(stage_31_v98, pl.INT32)
                            if stage_31_v112 <= 512:
                                stage_31_v164: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                                stage_31_v165: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_31_v164, stage_31_v124)
                                stage_31_v166: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v123, 0, 0, [1, 512], target_memory=pl.Mem.Vec)
                                stage_31_v167: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v112])] = pl.tile.slice(stage_31_v166, [1, 512], [0, 0], valid_shape=[1, stage_31_v112])
                                stage_31_v168: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_31_v167, pad_value=pl.PadValue.min)
                                stage_31_v169: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_31_v168, -3.4028234663852886e+38)
                                stage_31_v170: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_31_v165, dtype=pl.UINT32)
                                stage_31_v171: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_31_v169, stage_31_v170)
                                stage_31_v172: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v171, pl.const(64, pl.INT32))
                                stage_31_v173: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v172, pl.const(256, pl.INT32))
                                stage_31_v174: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v173, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                stage_31_v175: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v174)
                            else:
                                if stage_31_v112 <= 1024:
                                    stage_31_v239: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                                    stage_31_v240: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_31_v239, stage_31_v124)
                                    stage_31_v241: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v123, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                    stage_31_v242: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v112])] = pl.tile.slice(stage_31_v241, [1, 1024], [0, 0], valid_shape=[1, stage_31_v112])
                                    stage_31_v243: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_31_v242, pad_value=pl.PadValue.min)
                                    stage_31_v244: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_31_v243, -3.4028234663852886e+38)
                                    stage_31_v245: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_31_v240, dtype=pl.UINT32)
                                    stage_31_v246: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_31_v244, stage_31_v245)
                                    stage_31_v247: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v246, pl.const(64, pl.INT32))
                                    stage_31_v248: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v247, pl.const(256, pl.INT32))
                                    stage_31_v249: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_31_v248, [1, 1024], [0, 0])
                                    stage_31_v250: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_31_v248, [1, 1024], [0, 1024])
                                    stage_31_v251: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                    stage_31_v252: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_31_v249, stage_31_v250, stage_31_v251, exhausted=False)
                                    stage_31_v253: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v252, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                    stage_31_v254: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v253)
                                else:
                                    stage_31_v255: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                                    stage_31_v256: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_31_v255, stage_31_v124)
                                    stage_31_v257: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v123, 0, 0, [1, 2048], target_memory=pl.Mem.Vec)
                                    stage_31_v258: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_31_v112])] = pl.tile.slice(stage_31_v257, [1, 2048], [0, 0], valid_shape=[1, stage_31_v112])
                                    stage_31_v259: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_31_v258, pad_value=pl.PadValue.min)
                                    stage_31_v260: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_31_v259, -3.4028234663852886e+38)
                                    stage_31_v261: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_31_v256, dtype=pl.UINT32)
                                    stage_31_v262: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_31_v260, stage_31_v261)
                                    stage_31_v263: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v262, pl.const(64, pl.INT32))
                                    stage_31_v264: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v263, pl.const(256, pl.INT32))
                                    stage_31_v265: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_31_v264, pl.const(1024, pl.INT32))
                                    stage_31_v266: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v265, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                    stage_31_v254: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v266)
                                stage_31_v175: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v254)
                            if stage_31_v74 < 2:
                                stage_31_v176: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_31_v122, stage_31_v175, [2, 0])
                                stage_31_v177: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v176)
                            else:
                                stage_31_v178: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v122, 2, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                stage_31_v179: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                stage_31_v180: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(stage_31_v175, stage_31_v178, stage_31_v179, exhausted=False)
                                stage_31_v181: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v180, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                stage_31_v182: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_31_v122, stage_31_v181, [2, 0])
                                stage_31_v177: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v182)
                            stage_31_v125: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v177)
                        else:
                            stage_31_v125: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v122)
                        stage_31_v113: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v125)
                    else:
                        stage_31_v113: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_31_v90)
                    (stage_31_v114, stage_31_v115) = pl.yield_(stage_31_v113, stage_31_v89)
                stage_31_v58: pl.Scalar[pl.INDEX] = stage_31_v25 * 3
                stage_31_v59: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v114, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                stage_31_v60 = pl.tile.store(stage_31_v59, [(stage_31_v73 + stage_31_v58) * 64 + stage_31_v72, 0], temporary_46)
                stage_31_v61: pl.Scalar[pl.INDEX] = stage_31_v25 * 3 + 1
                stage_31_v62: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v114, 1, 0, [1, 1024], target_memory=pl.Mem.Vec)
                stage_31_v63 = pl.tile.store(stage_31_v62, [(stage_31_v73 + stage_31_v61) * 64 + stage_31_v72, 0], temporary_46)
                stage_31_v64: pl.Scalar[pl.INDEX] = stage_31_v25 * 3 + 2
                stage_31_v65: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_31_v114, 2, 0, [1, 1024], target_memory=pl.Mem.Vec)
                stage_31_v66 = pl.tile.store(stage_31_v65, [(stage_31_v73 + stage_31_v64) * 64 + stage_31_v72, 0], temporary_46)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_31 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_31)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_32 = 31104 + core_id * 6
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
        stage_32_v6: pl.Scalar[pl.INDEX] = block_32
        stage_32_v7: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        stage_32_v8: pl.Scalar[pl.INDEX] = 0
        for (stage_32_v18, (stage_32_v33,)) in pl.range(stage_32_v7 // 6, init_values=(stage_32_v8,)):
            stage_32_v19: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_32_v18])
            stage_32_v20: pl.Scalar[pl.INDEX] = pl.max(stage_32_v33, pl.cast(stage_32_v19, pl.INDEX) // 4)
            stage_32_v21: pl.Scalar[pl.INDEX] = pl.yield_(stage_32_v20)
        stage_32_v9: pl.Scalar[pl.INDEX] = stage_32_v7 // 2
        if 24 <= stage_32_v7:
            stage_32_v22: pl.Scalar[pl.INDEX] = stage_32_v7 // 6
            stage_32_v23: pl.Scalar[pl.INDEX] = pl.yield_(stage_32_v22)
        else:
            stage_32_v23: pl.Scalar[pl.INDEX] = pl.yield_(stage_32_v9)
        stage_32_v10: pl.Scalar[pl.INDEX] = (pl.min(stage_32_v21, 262144) + 1023) // 1024
        stage_32_v11: pl.Scalar[pl.INDEX] = pl.max((pl.min(stage_32_v21, 262144) + 8191) // 8192, 1)
        stage_32_v12: pl.Scalar[pl.INDEX] = 8
        stage_32_v13: pl.Scalar[pl.INDEX] = 0
        stage_32_v14: pl.Scalar[pl.INDEX] = 1
        if stage_32_v23 * 3 == 48:
            stage_32_v24: pl.Scalar[pl.INDEX] = 3
            stage_32_v25: pl.Scalar[pl.INDEX] = pl.yield_(stage_32_v24)
        else:
            if stage_32_v23 * 3 == 24:
                stage_32_v34: pl.Scalar[pl.INDEX] = 3
                stage_32_v35: pl.Scalar[pl.INDEX] = pl.yield_(stage_32_v34)
            else:
                if stage_32_v23 * 6 == 24:
                    stage_32_v54: pl.Scalar[pl.INDEX] = 6
                    stage_32_v55: pl.Scalar[pl.INDEX] = pl.yield_(stage_32_v54)
                else:
                    stage_32_v55: pl.Scalar[pl.INDEX] = pl.yield_(stage_32_v14)
                stage_32_v35: pl.Scalar[pl.INDEX] = pl.yield_(stage_32_v55)
            stage_32_v25: pl.Scalar[pl.INDEX] = pl.yield_(stage_32_v35)
        if 1 < stage_32_v25:
            stage_32_v26: pl.Scalar[pl.INDEX] = (stage_32_v11 + stage_32_v25 - 1) // stage_32_v25 * stage_32_v25
            if stage_32_v26 <= 32:
                stage_32_v36: pl.Scalar[pl.INDEX] = pl.max(stage_32_v10 // stage_32_v26, 1)
                stage_32_v37: pl.Scalar[pl.INDEX] = stage_32_v10 % stage_32_v26
                (stage_32_v56, stage_32_v57) = pl.yield_(stage_32_v37, stage_32_v36)
            else:
                (stage_32_v56, stage_32_v57) = pl.yield_(stage_32_v13, stage_32_v12)
            (stage_32_v38, stage_32_v39) = pl.yield_(stage_32_v56, stage_32_v57)
        else:
            (stage_32_v38, stage_32_v39) = pl.yield_(stage_32_v13, stage_32_v12)
        stage_32_v15: pl.Scalar[pl.INDEX] = stage_32_v39
        stage_32_v16: pl.Scalar[pl.INDEX] = stage_32_v38
        for stage_32_v17 in pl.range(stage_32_v6, stage_32_v7, 48):
            stage_32_v27: pl.Scalar[pl.INDEX] = stage_32_v17 // 6
            stage_32_v28: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_32_v17])
            stage_32_v29: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_32_v27])
            stage_32_v30: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v29, pl.INDEX) // 4
            stage_32_v31: pl.Scalar[pl.INDEX] = pl.min(stage_32_v30, (stage_32_v28 + 1) // 4)
            stage_32_v32: pl.Scalar[pl.INDEX] = pl.min(stage_32_v31, 262144)
            if 0 < stage_32_v32:
                stage_32_v40: pl.Scalar[pl.INDEX] = (stage_32_v32 + 1023) // 1024
                stage_32_v41: pl.Scalar[pl.INDEX] = stage_32_v16 * (stage_32_v15 + 1)
                stage_32_v42: pl.Scalar[pl.INDEX] = pl.min(stage_32_v16, (stage_32_v40 + stage_32_v15) // (stage_32_v15 + 1))
                stage_32_v43: pl.Scalar[pl.INDEX] = pl.max(stage_32_v40 - stage_32_v41, 0)
                stage_32_v44: pl.Scalar[pl.INDEX] = stage_32_v42 + (stage_32_v43 + stage_32_v15 - 1) // stage_32_v15
                stage_32_v45: pl.Scalar[pl.INDEX] = stage_32_v44 * 2
                stage_32_v46: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
                if 24 <= stage_32_v46:
                    stage_32_v58: pl.Scalar[pl.INDEX] = stage_32_v44
                    stage_32_v59: pl.Scalar[pl.INDEX] = pl.yield_(stage_32_v58)
                else:
                    stage_32_v59: pl.Scalar[pl.INDEX] = pl.yield_(stage_32_v45)
                stage_32_v47: pl.Scalar[pl.INDEX] = stage_32_v17 * 64
                stage_32_v48: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_32_v47, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                for (stage_32_v60, (stage_32_v65,)) in pl.range(1, stage_32_v59, 3, init_values=(stage_32_v48,)):
                    stage_32_v61: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_32_v47 + stage_32_v60, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    if stage_32_v60 + 2 < stage_32_v59:
                        stage_32_v66: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_32_v47 + stage_32_v60 + 1, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                        stage_32_v67: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_32_v47 + stage_32_v60 + 2, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                        stage_32_v68: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 4096], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_32_v69: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(stage_32_v67, stage_32_v66, stage_32_v61, stage_32_v65, stage_32_v68, exhausted=True)
                        stage_32_v70: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_32_v69, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                        stage_32_v71: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_32_v70)
                    else:
                        if stage_32_v60 + 1 < stage_32_v59:
                            stage_32_v73: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_32_v47 + stage_32_v60 + 1, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                            stage_32_v74: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 3072], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                            stage_32_v75: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(stage_32_v73, stage_32_v61, stage_32_v65, stage_32_v74, exhausted=True)
                            stage_32_v76: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_32_v75, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                            stage_32_v77: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_32_v76)
                        else:
                            stage_32_v78: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                            stage_32_v79: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(stage_32_v61, stage_32_v65, stage_32_v78, exhausted=False)
                            stage_32_v80: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_32_v79, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                            stage_32_v77: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_32_v80)
                        stage_32_v71: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_32_v77)
                    stage_32_v62: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_32_v71)
                stage_32_v49: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = stage_32_v62
                stage_32_v50: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather_mask(stage_32_v49, mask_pattern=1, output_dtype=pl.FP32)
                pl.tile.store(stage_32_v50, [stage_32_v17, 0], arg_idx_topk_scores)
                stage_32_v51: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.gather_mask(stage_32_v49, mask_pattern=2, output_dtype=pl.INT32)
                if 512 <= stage_32_v32:
                    pl.tile.store(stage_32_v51, [stage_32_v17, 0], arg_idx_topk)
                else:
                    stage_32_v63: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                    for stage_32_v64 in pl.range(stage_32_v32):
                        stage_32_v72: pl.Scalar[pl.INT32] = pl.tile.read(stage_32_v51, [0, stage_32_v64])
                        pl.tile.write(stage_32_v63, [0, stage_32_v64], stage_32_v72)
                    pl.tile.store(stage_32_v63, [stage_32_v17, 0], arg_idx_topk)
            else:
                stage_32_v52: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e+38)
                pl.tile.store(stage_32_v52, [stage_32_v17, 0], arg_idx_topk_scores)
                stage_32_v53: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                pl.tile.store(stage_32_v53, [stage_32_v17, 0], arg_idx_topk)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_32 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_32)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_33 = 31536 + core_id * 6
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
        stage_33_v8: pl.Scalar[pl.INDEX] = block_33
        for (stage_33_v9, (stage_33_v63, stage_33_v64)) in pl.range(stage_33_v8 * 8, 96, 128, init_values=(temporary_53, temporary_52)):
            stage_33_v10: pl.Scalar[pl.INDEX] = pl.min(96 - stage_33_v9, 8)
            stage_33_v11: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.load(arg_idx_topk, [stage_33_v9, 0], [8, 512], [stage_33_v10, 512], target_memory=pl.Mem.Vec)
            stage_33_v12: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.cast(stage_33_v11, target_type=pl.FP32, mode='round')
            stage_33_v13: pl.Tile[[8, 1], pl.INT64, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 1])] = pl.tile.load(view_106, [stage_33_v9, 0], [8, 1], [stage_33_v10, 1], target_memory=pl.Mem.Vec)
            stage_33_v14: pl.Tile[[1, 8], pl.INT64, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_33_v10])] = pl.tile.reshape(stage_33_v13, [1, 8])
            stage_33_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_33_v10])] = pl.tile.cast(stage_33_v14, target_type=pl.FP32, mode='round')
            stage_33_v16: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 1])] = pl.tile.reshape(stage_33_v15, [8, 1])
            stage_33_v17: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_33_v10])] = pl.tile.reshape(stage_33_v16, [1, 8])
            stage_33_v18: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_33_v10])] = pl.tile.adds(stage_33_v17, 1.0)
            stage_33_v19: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 1])] = pl.tile.reshape(stage_33_v18, [8, 1])
            stage_33_v20: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_33_v10])] = pl.tile.reshape(stage_33_v19, [1, 8])
            stage_33_v21: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_33_v10])] = pl.tile.muls(stage_33_v20, 0.25)
            stage_33_v22: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 1])] = pl.tile.reshape(stage_33_v21, [8, 1])
            stage_33_v23: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_33_v10])] = pl.tile.reshape(stage_33_v22, [1, 8])
            stage_33_v24: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_33_v10])] = pl.tile.cast(stage_33_v23, target_type=pl.INT32, mode='trunc')
            stage_33_v25: pl.Tile[[8, 1], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 1])] = pl.tile.reshape(stage_33_v24, [8, 1])
            stage_33_v26: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_33_v10])] = pl.tile.reshape(stage_33_v25, [1, 8])
            stage_33_v27: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_33_v10])] = pl.tile.cast(stage_33_v26, target_type=pl.FP32, mode='round')
            stage_33_v28: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 1])] = pl.tile.reshape(stage_33_v27, [8, 1])
            stage_33_v29: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=1.0)
            stage_33_v30: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_33_v29, stage_33_v28)
            stage_33_v31: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.adds(stage_33_v12, 1.0)
            stage_33_v32: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.maximums(stage_33_v31, 0.0)
            stage_33_v33: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.minimums(stage_33_v32, 1.0)
            stage_33_v34: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_33_v30, stage_33_v12)
            stage_33_v35: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.maximums(stage_33_v34, 0.0)
            stage_33_v36: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(stage_33_v35, 1.0)
            stage_33_v37: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.mul(stage_33_v33, stage_33_v36)
            stage_33_v38: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.adds(stage_33_v12, 1.0)
            stage_33_v39: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.mul(stage_33_v37, stage_33_v38)
            stage_33_v40: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.subs(stage_33_v39, 1.0)
            stage_33_v41: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.cast(stage_33_v40, target_type=pl.INT32, mode='round')
            stage_33_v42: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.set_validshape(stage_33_v41, stage_33_v10, 512)
            stage_33_v43 = pl.tile.store(stage_33_v42, [stage_33_v9, 0], stage_33_v63)
            stage_33_v44: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.minimums(stage_33_v40, 0.0)
            stage_33_v45: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.muls(stage_33_v44, 1e+20)
            stage_33_v46: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 512])] = pl.tile.set_validshape(stage_33_v45, stage_33_v10, 512)
            stage_33_v47 = pl.tile.store(stage_33_v46, [stage_33_v9, 128], stage_33_v64)
            for stage_33_v48 in pl.range(1, 5):
                stage_33_v65: pl.Scalar[pl.INDEX] = (stage_33_v48 - 1) * 128
                stage_33_v66: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(stage_33_v10, pl.const(8, pl.INDEX)), 128])] = pl.tile.slice(stage_33_v37, [8, 128], [0, stage_33_v65], [stage_33_v10, 128])
                stage_33_v67: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_33_v68: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(stage_33_v10, pl.const(8, pl.INDEX)), 1])] = pl.tile.row_max(stage_33_v66, stage_33_v67)
                for stage_33_v69 in pl.range(stage_33_v10):
                    stage_33_v90: pl.Scalar[pl.FP32] = pl.tile.read(stage_33_v68, [stage_33_v69, 0])
                    stage_33_v91: pl.Scalar[pl.INT32] = pl.cast(stage_33_v90, pl.INT32)
                    pl.tensor.write(temporary_54, [stage_33_v9 + stage_33_v69, stage_33_v48], stage_33_v91)
            stage_33_v49: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 128], dtype=pl.INT32, descending=False)
            stage_33_v50: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_33_v49, target_type=pl.FP32, mode='round')
            stage_33_v51: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for stage_33_v52 in pl.range(8):
                stage_33_v70: pl.Scalar[pl.INDEX] = stage_33_v9 + pl.min(stage_33_v52, stage_33_v10 - 1)
                stage_33_v71: pl.Scalar[pl.INDEX] = stage_33_v70 // 6
                stage_33_v72: pl.Scalar[pl.INT64] = pl.tensor.read(view_106, [stage_33_v70, 0])
                stage_33_v73: pl.Scalar[pl.INDEX] = pl.cast(stage_33_v72, pl.INDEX)
                stage_33_v74: pl.Scalar[pl.INDEX] = pl.min(stage_33_v73 + 1, 128)
                stage_33_v75: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_33_v71])
                if pl.cast(stage_33_v75, pl.INDEX) <= 0:
                    stage_33_v92: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
                    stage_33_v93: pl.Scalar[pl.INDEX] = pl.yield_(stage_33_v92)
                else:
                    stage_33_v93: pl.Scalar[pl.INDEX] = pl.yield_(stage_33_v74)
                pl.tile.write(stage_33_v51, [stage_33_v52, 0], pl.cast(pl.cast(stage_33_v93, pl.INT32), pl.FP32))
            stage_33_v53: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 128], dtype=pl.FP32, value=1.0)
            stage_33_v54: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_33_v53, stage_33_v50)
            stage_33_v55: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_sub(stage_33_v54, stage_33_v51)
            stage_33_v56: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_33_v55)
            stage_33_v57: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.maximums(stage_33_v56, 0.0)
            stage_33_v58: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(stage_33_v57, 1.0)
            stage_33_v59: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_33_v58, 1.0)
            stage_33_v60: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_33_v59, 1e+20)
            stage_33_v61: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v10, 128])] = pl.tile.set_validshape(stage_33_v60, stage_33_v10, 128)
            stage_33_v62 = pl.tile.store(stage_33_v61, [stage_33_v9, 0], stage_33_v47)
            for (stage_33_v76, (stage_33_v94,)) in pl.range(stage_33_v10, init_values=(stage_33_v62,)):
                stage_33_v77: pl.Scalar[pl.INDEX] = stage_33_v9 + stage_33_v76
                stage_33_v78: pl.Scalar[pl.INDEX] = stage_33_v77 // 6
                stage_33_v79: pl.Scalar[pl.INT64] = pl.tensor.read(view_106, [stage_33_v77, 0])
                stage_33_v80: pl.Scalar[pl.INDEX] = pl.cast(stage_33_v79, pl.INDEX)
                stage_33_v81: pl.Scalar[pl.INDEX] = pl.min(stage_33_v80 + 1, 128)
                stage_33_v82: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_33_v78])
                if pl.cast(stage_33_v82, pl.INDEX) <= 0:
                    stage_33_v95: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
                    stage_33_v96: pl.Scalar[pl.INDEX] = pl.yield_(stage_33_v95)
                else:
                    stage_33_v96: pl.Scalar[pl.INDEX] = pl.yield_(stage_33_v81)
                stage_33_v83: pl.Scalar[pl.INDEX] = stage_33_v80 - stage_33_v96 + 1
                stage_33_v84: pl.Scalar[pl.INDEX] = stage_33_v83 % 32
                stage_33_v85: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
                stage_33_v86: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
                for (stage_33_v97, (stage_33_v107, stage_33_v108)) in pl.range(5, init_values=(stage_33_v85, stage_33_v86)):
                    stage_33_v98: pl.Scalar[pl.INDEX] = pl.max(stage_33_v97 * 32 - stage_33_v84, 0)
                    stage_33_v99: pl.Scalar[pl.INDEX] = pl.min(stage_33_v97 * 32 - stage_33_v84 + 32, stage_33_v96)
                    if stage_33_v98 < stage_33_v99:
                        stage_33_v109: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_33_v78, (stage_33_v83 + stage_33_v98) // 32])
                        if 0 <= pl.cast(stage_33_v109, pl.INDEX):
                            stage_33_v116: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
                            (stage_33_v123, stage_33_v124) = pl.yield_(stage_33_v116, stage_33_v108)
                        else:
                            stage_33_v117: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
                            (stage_33_v123, stage_33_v124) = pl.yield_(stage_33_v107, stage_33_v117)
                        (stage_33_v118, stage_33_v119) = pl.yield_(stage_33_v123, stage_33_v124)
                    else:
                        (stage_33_v118, stage_33_v119) = pl.yield_(stage_33_v107, stage_33_v108)
                    (stage_33_v110, stage_33_v111) = pl.yield_(stage_33_v118, stage_33_v119)
                pl.tensor.write(temporary_54, [stage_33_v77, 0], stage_33_v110)
                if pl.cast(stage_33_v111, pl.INDEX) == 1:
                    stage_33_v100: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 128], dtype=pl.INT32, descending=False)
                    stage_33_v101: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_33_v100, target_type=pl.FP32, mode='round')
                    stage_33_v102: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
                    for (stage_33_v112, (stage_33_v120,)) in pl.range(5, init_values=(stage_33_v102,)):
                        stage_33_v113: pl.Scalar[pl.INDEX] = pl.max(stage_33_v112 * 32 - stage_33_v84, 0)
                        stage_33_v114: pl.Scalar[pl.INDEX] = pl.min(stage_33_v112 * 32 - stage_33_v84 + 32, stage_33_v96)
                        if stage_33_v113 < stage_33_v114:
                            stage_33_v121: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_33_v78, (stage_33_v83 + stage_33_v113) // 32])
                            if 0 <= pl.cast(stage_33_v121, pl.INDEX):
                                stage_33_v125: pl.Scalar[pl.FP32] = pl.cast(pl.cast(stage_33_v113, pl.INT32), pl.FP32)
                                stage_33_v126: pl.Scalar[pl.FP32] = pl.cast(pl.cast(stage_33_v114, pl.INT32), pl.FP32)
                                stage_33_v127: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_33_v101, stage_33_v125)
                                stage_33_v128: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_33_v127, 1.0)
                                stage_33_v129: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.maximums(stage_33_v128, 0.0)
                                stage_33_v130: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(stage_33_v129, 1.0)
                                stage_33_v131: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_33_v101)
                                stage_33_v132: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_33_v131, stage_33_v126)
                                stage_33_v133: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.maximums(stage_33_v132, 0.0)
                                stage_33_v134: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(stage_33_v133, 1.0)
                                stage_33_v135: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_33_v130, stage_33_v134)
                                stage_33_v136: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_33_v120, stage_33_v135)
                                stage_33_v137: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_33_v136)
                            else:
                                stage_33_v137: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_33_v120)
                            stage_33_v122: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_33_v137)
                        else:
                            stage_33_v122: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_33_v120)
                        stage_33_v115: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_33_v122)
                    stage_33_v103: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_33_v115, 1.0)
                    stage_33_v104: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_33_v103, 1e+20)
                    stage_33_v105 = pl.tile.store(stage_33_v104, [stage_33_v77, 0], stage_33_v94)
                    stage_33_v106 = pl.yield_(stage_33_v105)
                else:
                    stage_33_v106 = pl.yield_(stage_33_v94)
                stage_33_v87 = pl.yield_(stage_33_v106)
            (stage_33_v88, stage_33_v89) = pl.yield_(stage_33_v43, stage_33_v87)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_33 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_33)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_34 = 31968 + core_id * 6
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
        stage_34_v22: pl.Scalar[pl.INDEX] = block_34
        pl.system.set_ffts(temporary_62)
        stage_34_v23: pl.Scalar[pl.INDEX] = (96 - stage_34_v22 + 23) // 24
        stage_34_v24: pl.Scalar[pl.INDEX] = stage_34_v23 * 5
        stage_34_v25: pl.Scalar[pl.INDEX] = subslot_34
        pl.system.set_ffts(temporary_62)
        stage_34_v26: pl.Scalar[pl.INDEX] = stage_34_v25 * 32
        stage_34_v27: pl.Scalar[pl.INDEX] = stage_34_v25 * 64
        stage_34_v28: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_34_v29: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        stage_34_v30: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.rems(stage_34_v28, pl.const(2, pl.INT32), stage_34_v29)
        stage_34_v31: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_34_v30, pl.const(2, pl.INT32))
        stage_34_v32: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.sub(stage_34_v28, stage_34_v31)
        stage_34_v33: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_34_v32, pl.const(449, pl.INT32))
        stage_34_v34: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        stage_34_v35: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.col_expand(stage_34_v34, stage_34_v33)
        stage_34_v36: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 16], dtype=pl.INT32, descending=False)
        stage_34_v37: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_34_v36, pl.const(512, pl.INT32))
        stage_34_v38: pl.Tile[[16, 1], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v37, [16, 1])
        stage_34_v39: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.row_expand_add(stage_34_v35, stage_34_v38)
        stage_34_v40: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v39, [1, 1024])
        stage_34_v41: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_116, [stage_34_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
        stage_34_v42: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v41, [1, 32])
        stage_34_v43: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_34_v42, 0.0)
        stage_34_v44: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v43, [32, 1])
        stage_34_v45: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        stage_34_v46: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        for (stage_34_v47, (stage_34_v51, stage_34_v52, stage_34_v53, stage_34_v54)) in pl.range(stage_34_v24 + 3, init_values=(stage_34_v41, stage_34_v44, stage_34_v45, stage_34_v46)):
            stage_34_v48: pl.Scalar[pl.INDEX] = stage_34_v22 + stage_34_v47 // 5 * 24
            stage_34_v49: pl.Scalar[pl.INDEX] = stage_34_v48 // 6
            stage_34_v50: pl.Scalar[pl.INDEX] = stage_34_v47 % 5
            if stage_34_v47 < stage_34_v24:
                stage_34_v55: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_34_v48, stage_34_v50])
                if 0 < pl.cast(stage_34_v55, pl.INDEX):
                    stage_34_v72: pl.Scalar[pl.INDEX] = stage_34_v22 * 3 + stage_34_v47 % 3
                    stage_34_v73: pl.Scalar[pl.INDEX] = stage_34_v72 * 128
                    stage_34_v74: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
                    if stage_34_v50 == 0:
                        stage_34_v256: pl.Scalar[pl.INT64] = pl.tensor.read(view_106, [stage_34_v48, 0])
                        stage_34_v257: pl.Scalar[pl.INDEX] = pl.cast(stage_34_v256, pl.INDEX)
                        stage_34_v258: pl.Scalar[pl.INDEX] = pl.min(stage_34_v257 + 1, 128)
                        stage_34_v259: pl.Scalar[pl.INDEX] = stage_34_v257 - stage_34_v258 + 1
                        stage_34_v260: pl.Scalar[pl.INDEX] = stage_34_v259 % 32
                        stage_34_v261: pl.Scalar[pl.INDEX] = pl.min(pl.max(stage_34_v258 - stage_34_v27, 0), 64)
                        stage_34_v262: pl.Scalar[pl.INDEX] = 0
                        stage_34_v263: pl.Scalar[pl.INDEX] = pl.min(32 - stage_34_v260 - stage_34_v27, stage_34_v261)
                        if stage_34_v262 < stage_34_v263:
                            stage_34_v273: pl.Scalar[pl.INDEX] = stage_34_v259 + stage_34_v27 + stage_34_v262
                            stage_34_v274: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_34_v49, stage_34_v273 // 32])
                            if 0 <= pl.cast(stage_34_v274, pl.INDEX):
                                stage_34_v291: pl.Scalar[pl.INDEX] = pl.cast(stage_34_v274, pl.INDEX) * 32 + stage_34_v273 % 32
                                stage_34_v292: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_34_v74, view_45, [stage_34_v262, 0], [stage_34_v291, 0], [64, 512], valid_shape=[stage_34_v263 - stage_34_v262, 512], transpose=False)
                                stage_34_v293: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v292
                            else:
                                stage_34_v293: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v74
                            stage_34_v275: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v293
                        else:
                            stage_34_v275: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v74
                        stage_34_v264: pl.Scalar[pl.INDEX] = pl.max(32 - stage_34_v260 - stage_34_v27, 0)
                        stage_34_v265: pl.Scalar[pl.INDEX] = pl.min(64 - stage_34_v260 - stage_34_v27, stage_34_v261)
                        if stage_34_v264 < stage_34_v265:
                            stage_34_v276: pl.Scalar[pl.INDEX] = stage_34_v259 + stage_34_v27 + stage_34_v264
                            stage_34_v277: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_34_v49, stage_34_v276 // 32])
                            if 0 <= pl.cast(stage_34_v277, pl.INDEX):
                                stage_34_v294: pl.Scalar[pl.INDEX] = pl.cast(stage_34_v277, pl.INDEX) * 32 + stage_34_v276 % 32
                                stage_34_v295: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_34_v275, view_45, [stage_34_v264, 0], [stage_34_v294, 0], [64, 512], valid_shape=[stage_34_v265 - stage_34_v264, 512], transpose=False)
                                stage_34_v296: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v295
                            else:
                                stage_34_v296: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v275
                            stage_34_v278: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v296
                        else:
                            stage_34_v278: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v275
                        stage_34_v266: pl.Scalar[pl.INDEX] = pl.max(64 - stage_34_v260 - stage_34_v27, 0)
                        stage_34_v267: pl.Scalar[pl.INDEX] = pl.min(96 - stage_34_v260 - stage_34_v27, stage_34_v261)
                        if stage_34_v266 < stage_34_v267:
                            stage_34_v279: pl.Scalar[pl.INDEX] = stage_34_v259 + stage_34_v27 + stage_34_v266
                            stage_34_v280: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_34_v49, stage_34_v279 // 32])
                            if 0 <= pl.cast(stage_34_v280, pl.INDEX):
                                stage_34_v297: pl.Scalar[pl.INDEX] = pl.cast(stage_34_v280, pl.INDEX) * 32 + stage_34_v279 % 32
                                stage_34_v298: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_34_v278, view_45, [stage_34_v266, 0], [stage_34_v297, 0], [64, 512], valid_shape=[stage_34_v267 - stage_34_v266, 512], transpose=False)
                                stage_34_v299: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v298
                            else:
                                stage_34_v299: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v278
                            stage_34_v281: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v299
                        else:
                            stage_34_v281: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v278
                        stage_34_v268: pl.Scalar[pl.INDEX] = pl.max(96 - stage_34_v260 - stage_34_v27, 0)
                        stage_34_v269: pl.Scalar[pl.INDEX] = pl.min(128 - stage_34_v260 - stage_34_v27, stage_34_v261)
                        if stage_34_v268 < stage_34_v269:
                            stage_34_v282: pl.Scalar[pl.INDEX] = stage_34_v259 + stage_34_v27 + stage_34_v268
                            stage_34_v283: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_34_v49, stage_34_v282 // 32])
                            if 0 <= pl.cast(stage_34_v283, pl.INDEX):
                                stage_34_v300: pl.Scalar[pl.INDEX] = pl.cast(stage_34_v283, pl.INDEX) * 32 + stage_34_v282 % 32
                                stage_34_v301: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_34_v281, view_45, [stage_34_v268, 0], [stage_34_v300, 0], [64, 512], valid_shape=[stage_34_v269 - stage_34_v268, 512], transpose=False)
                                stage_34_v302: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v301
                            else:
                                stage_34_v302: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v281
                            stage_34_v284: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v302
                        else:
                            stage_34_v284: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v281
                        stage_34_v270: pl.Scalar[pl.INDEX] = pl.max(128 - stage_34_v260 - stage_34_v27, 0)
                        stage_34_v271: pl.Scalar[pl.INDEX] = pl.min(160 - stage_34_v260 - stage_34_v27, stage_34_v261)
                        if stage_34_v270 < stage_34_v271:
                            stage_34_v285: pl.Scalar[pl.INDEX] = stage_34_v259 + stage_34_v27 + stage_34_v270
                            stage_34_v286: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_34_v49, stage_34_v285 // 32])
                            if 0 <= pl.cast(stage_34_v286, pl.INDEX):
                                stage_34_v303: pl.Scalar[pl.INDEX] = pl.cast(stage_34_v286, pl.INDEX) * 32 + stage_34_v285 % 32
                                stage_34_v304: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_34_v284, view_45, [stage_34_v270, 0], [stage_34_v303, 0], [64, 512], valid_shape=[stage_34_v271 - stage_34_v270, 512], transpose=False)
                                stage_34_v305: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v304
                            else:
                                stage_34_v305: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v284
                            stage_34_v287: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v305
                        else:
                            stage_34_v287: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v284
                        stage_34_v272: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v287
                    else:
                        stage_34_v74: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
                        for (stage_34_v288, (stage_34_v306,)) in pl.range(64, init_values=(stage_34_v74,)):
                            stage_34_v289: pl.Scalar[pl.INDEX] = (stage_34_v50 - 1) * 128 + stage_34_v27 + stage_34_v288
                            if stage_34_v289 < 512:
                                stage_34_v307: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_53, [stage_34_v48, stage_34_v289])
                                if 0 <= pl.cast(stage_34_v307, pl.INDEX):
                                    stage_34_v309: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_block_table, [stage_34_v49, pl.cast(stage_34_v307, pl.INDEX) // 32])
                                    stage_34_v310: pl.Scalar[pl.INDEX] = pl.cast(stage_34_v309, pl.INDEX)
                                    stage_34_v311: pl.Scalar[pl.INDEX] = stage_34_v310 * 32 + pl.cast(stage_34_v307, pl.INDEX) % 32
                                    stage_34_v312: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_34_v306, view_61, [stage_34_v288, 0], [stage_34_v311, 0], [1, 512], transpose=False)
                                    stage_34_v313: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v312
                                else:
                                    stage_34_v313: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v306
                                stage_34_v308: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v313
                            else:
                                stage_34_v308: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v306
                            stage_34_v290: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.yield_(stage_34_v308)
                        stage_34_v272: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_34_v290
                    stage_34_v75 = pl.tile.store(stage_34_v272, [stage_34_v73 + stage_34_v27, 0], temporary_56)
            if 144 <= 96:
                if 0 < stage_34_v47 and stage_34_v47 <= stage_34_v24:
                    stage_34_v76: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_34_v22 + (stage_34_v47 - 1) // 5 * 24, (stage_34_v47 - 1) % 5])
                    if 0 < pl.cast(stage_34_v76, pl.INDEX):
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                if stage_34_v47 < stage_34_v24:
                    stage_34_v77: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_34_v48, stage_34_v50])
                    if 0 < pl.cast(stage_34_v77, pl.INDEX):
                        pl.system.sync_set(0, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if 0 < stage_34_v47 and stage_34_v47 <= stage_34_v24:
                stage_34_v56: pl.Scalar[pl.INDEX] = stage_34_v47 - 1
                stage_34_v57: pl.Scalar[pl.INDEX] = stage_34_v56 % 5
                stage_34_v58: pl.Scalar[pl.INDEX] = stage_34_v22 + stage_34_v56 // 5 * 24
                stage_34_v59: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_34_v58, stage_34_v57])
                if 0 < pl.cast(stage_34_v59, pl.INDEX):
                    stage_34_v78: pl.Scalar[pl.INDEX] = stage_34_v22 * 3 + stage_34_v56 % 3
                    stage_34_v79: pl.Scalar[pl.INDEX] = stage_34_v78 * 64
                    stage_34_v80: pl.Scalar[pl.INDEX] = stage_34_v57 * 128
                    if 96 < 144:
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    stage_34_v81: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_34_v82: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_57, [stage_34_v79 + stage_34_v26, 0], [32, 128], [32, 128], target_memory=pl.Mem.Vec)
                    stage_34_v83: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_52, [stage_34_v58, stage_34_v80], [1, 128], [1, 128], target_memory=pl.Mem.Vec)
                    stage_34_v84: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_34_v82, 0.04419417382415922)
                    stage_34_v85: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_add(stage_34_v84, stage_34_v83)
                    stage_34_v86: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_34_v85, stage_34_v81)
                    stage_34_v87: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_sub(stage_34_v85, stage_34_v86)
                    stage_34_v88: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_34_v87)
                    stage_34_v89: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_34_v88, stage_34_v81)
                    stage_34_v90: pl.Tile[[32, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_34_v88, target_type=pl.BF16, mode='rint')
                    stage_34_v91 = pl.tile.store(stage_34_v90, [stage_34_v79 + stage_34_v26, 0], temporary_58)
                    stage_34_v92 = pl.tile.store(stage_34_v86, [stage_34_v79 + stage_34_v26, 0], temporary_60)
                    stage_34_v93 = pl.tile.store(stage_34_v89, [stage_34_v79 + stage_34_v26, 0], temporary_61)
                    pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if 96 < 144:
                if stage_34_v47 < stage_34_v24:
                    stage_34_v94: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_34_v48, stage_34_v50])
                    if 0 < pl.cast(stage_34_v94, pl.INDEX):
                        pl.system.sync_set(0, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if 3 <= stage_34_v47:
                stage_34_v60: pl.Scalar[pl.INDEX] = stage_34_v47 - 2 - 1
                stage_34_v61: pl.Scalar[pl.INDEX] = stage_34_v60 % 5
                stage_34_v62: pl.Scalar[pl.INDEX] = stage_34_v22 + stage_34_v60 // 5 * 24
                stage_34_v63: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_34_v62, stage_34_v61])
                if 0 < pl.cast(stage_34_v63, pl.INDEX):
                    stage_34_v95: pl.Scalar[pl.INDEX] = stage_34_v22 * 3 + stage_34_v60 % 3
                    stage_34_v96: pl.Scalar[pl.INDEX] = stage_34_v95 * 64
                    pl.system.sync_wait(3, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    stage_34_v97: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_60, [stage_34_v96 + stage_34_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
                    stage_34_v98: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_61, [stage_34_v96 + stage_34_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
                    stage_34_v99: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v51, [1, 32])
                    stage_34_v100: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v97, [1, 32])
                    stage_34_v101: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_34_v99, stage_34_v100)
                    stage_34_v102: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v101, [32, 1])
                    stage_34_v103: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v51, [1, 32])
                    stage_34_v104: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v102, [1, 32])
                    stage_34_v105: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_34_v103, stage_34_v104)
                    stage_34_v106: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v105, [32, 1])
                    stage_34_v107: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v106, [1, 32])
                    stage_34_v108: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_34_v107)
                    stage_34_v109: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v108, [32, 1])
                    stage_34_v110: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v97, [1, 32])
                    stage_34_v111: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v102, [1, 32])
                    stage_34_v112: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_34_v110, stage_34_v111)
                    stage_34_v113: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v112, [32, 1])
                    stage_34_v114: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v113, [1, 32])
                    stage_34_v115: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_34_v114)
                    stage_34_v116: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v115, [32, 1])
                    stage_34_v117: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v109, [1, 32])
                    stage_34_v118: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v52, [1, 32])
                    stage_34_v119: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_34_v117, stage_34_v118)
                    stage_34_v120: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v119, [32, 1])
                    stage_34_v121: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v116, [1, 32])
                    stage_34_v122: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v98, [1, 32])
                    stage_34_v123: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_34_v121, stage_34_v122)
                    stage_34_v124: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v123, [32, 1])
                    stage_34_v125: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v120, [1, 32])
                    stage_34_v126: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v124, [1, 32])
                    stage_34_v127: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_34_v125, stage_34_v126)
                    stage_34_v128: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v127, [32, 1])
                    stage_34_v129: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_59, [stage_34_v96 + stage_34_v26, 0], [32, 256], [32, 256], target_memory=pl.Mem.Vec)
                    stage_34_v130: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_34_v53, stage_34_v109)
                    stage_34_v131: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_34_v129, stage_34_v116)
                    stage_34_v132: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_34_v130, stage_34_v131)
                    stage_34_v133: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_59, [stage_34_v96 + stage_34_v26, 256], [32, 256], [32, 256], target_memory=pl.Mem.Vec)
                    stage_34_v134: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_34_v54, stage_34_v109)
                    stage_34_v135: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_34_v133, stage_34_v116)
                    stage_34_v136: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_34_v134, stage_34_v135)
                    stage_34_v137 = stage_34_v102
                    stage_34_v138 = stage_34_v128
                    stage_34_v139 = stage_34_v132
                    stage_34_v140 = stage_34_v136
                else:
                    stage_34_v137 = stage_34_v51
                    stage_34_v138 = stage_34_v52
                    stage_34_v139 = stage_34_v53
                    stage_34_v140 = stage_34_v54
                if stage_34_v61 == 4:
                    stage_34_v141: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 1024], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    stage_34_v142: pl.Scalar[pl.INDEX] = 0
                    stage_34_v143: pl.Scalar[pl.INDEX] = stage_34_v26 + stage_34_v142
                    stage_34_v144: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v137, [1, 32])
                    stage_34_v145: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_34_v144, 0, stage_34_v142, [1, 16], target_memory=pl.Mem.Vec)
                    stage_34_v146: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v145, [16, 1])
                    stage_34_v147: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v138, [1, 32])
                    stage_34_v148: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_34_v147, 0, stage_34_v142, [1, 16], target_memory=pl.Mem.Vec)
                    stage_34_v149: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v148, [16, 1])
                    stage_34_v150: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_34_v139, [16, 256], [stage_34_v142, 0])
                    stage_34_v151: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_34_v140, [16, 256], [stage_34_v142, 0])
                    stage_34_v152: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.concat(stage_34_v150, stage_34_v151)
                    stage_34_v153: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_116, [stage_34_v143, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    stage_34_v154: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v146, [1, 16])
                    stage_34_v155: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v146, [1, 16])
                    stage_34_v156: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_34_v154, stage_34_v155)
                    stage_34_v157: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v156, [16, 1])
                    stage_34_v158: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v157, [1, 16])
                    stage_34_v159: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v153, [1, 16])
                    stage_34_v160: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_34_v158, stage_34_v159)
                    stage_34_v161: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v160, [16, 1])
                    stage_34_v162: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v161, [1, 16])
                    stage_34_v163: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v146, [1, 16])
                    stage_34_v164: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_34_v162, stage_34_v163)
                    stage_34_v165: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v164, [16, 1])
                    stage_34_v166: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v165, [1, 16])
                    stage_34_v167: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_34_v166)
                    stage_34_v168: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v167, [16, 1])
                    stage_34_v169: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v149, [1, 16])
                    stage_34_v170: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v168, [1, 16])
                    stage_34_v171: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_34_v169, stage_34_v170)
                    stage_34_v172: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v171, [16, 1])
                    stage_34_v173: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_div(stage_34_v152, stage_34_v172)
                    stage_34_v174: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_34_v173, target_type=pl.BF16, mode='rint')
                    stage_34_v175: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_34_v62, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_34_v176: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_55, [stage_34_v62, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_34_v177: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v173, [1, 8192])
                    stage_34_v178: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_34_v177, stage_34_v40, stage_34_v141)
                    stage_34_v179: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v178, [16, 64])
                    stage_34_v180: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_34_v173, 0, 448, [16, 64], target_memory=pl.Mem.Vec)
                    stage_34_v181: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_34_v180, stage_34_v175)
                    stage_34_v182: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_34_v179, stage_34_v176)
                    stage_34_v183: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_34_v181, stage_34_v182)
                    stage_34_v184: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_34_v183, target_type=pl.BF16, mode='rint')
                    stage_34_v185: pl.Tile[[16, 448], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_34_v174, [16, 448], [0, 0])
                    stage_34_v186: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.concat(stage_34_v185, stage_34_v184)
                    stage_34_v187: pl.Tile[[2, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.reshape(stage_34_v186, [2, 4096])
                    stage_34_v188: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_34_v187, [1, 4096], [0, 0])
                    stage_34_v189: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_34_v187, [1, 4096], [1, 0])
                    stage_34_v190: pl.Scalar[pl.INDEX] = stage_34_v143 // 8 * 384 + stage_34_v62
                    stage_34_v191: pl.Scalar[pl.INDEX] = stage_34_v190 + 384
                    stage_34_v192 = pl.tile.store(stage_34_v188, [stage_34_v190, 0], temporary_51)
                    stage_34_v193 = pl.tile.store(stage_34_v189, [stage_34_v191, 0], temporary_51)
                    stage_34_v194: pl.Scalar[pl.INDEX] = 16
                    stage_34_v195: pl.Scalar[pl.INDEX] = stage_34_v26 + stage_34_v194
                    stage_34_v196: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v137, [1, 32])
                    stage_34_v197: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_34_v196, 0, stage_34_v194, [1, 16], target_memory=pl.Mem.Vec)
                    stage_34_v198: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v197, [16, 1])
                    stage_34_v199: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v138, [1, 32])
                    stage_34_v200: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_34_v199, 0, stage_34_v194, [1, 16], target_memory=pl.Mem.Vec)
                    stage_34_v201: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v200, [16, 1])
                    stage_34_v202: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_34_v139, [16, 256], [stage_34_v194, 0])
                    stage_34_v203: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_34_v140, [16, 256], [stage_34_v194, 0])
                    stage_34_v204: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.concat(stage_34_v202, stage_34_v203)
                    stage_34_v205: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_116, [stage_34_v195, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    stage_34_v206: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v198, [1, 16])
                    stage_34_v207: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v198, [1, 16])
                    stage_34_v208: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_34_v206, stage_34_v207)
                    stage_34_v209: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v208, [16, 1])
                    stage_34_v210: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v209, [1, 16])
                    stage_34_v211: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v205, [1, 16])
                    stage_34_v212: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_34_v210, stage_34_v211)
                    stage_34_v213: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v212, [16, 1])
                    stage_34_v214: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v213, [1, 16])
                    stage_34_v215: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v198, [1, 16])
                    stage_34_v216: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_34_v214, stage_34_v215)
                    stage_34_v217: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v216, [16, 1])
                    stage_34_v218: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v217, [1, 16])
                    stage_34_v219: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_34_v218)
                    stage_34_v220: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v219, [16, 1])
                    stage_34_v221: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v201, [1, 16])
                    stage_34_v222: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v220, [1, 16])
                    stage_34_v223: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_34_v221, stage_34_v222)
                    stage_34_v224: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v223, [16, 1])
                    stage_34_v225: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_div(stage_34_v204, stage_34_v224)
                    stage_34_v226: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_34_v225, target_type=pl.BF16, mode='rint')
                    stage_34_v227: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_34_v62, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_34_v228: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_55, [stage_34_v62, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_34_v229: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v225, [1, 8192])
                    stage_34_v230: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_34_v229, stage_34_v40, stage_34_v141)
                    stage_34_v231: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v230, [16, 64])
                    stage_34_v232: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_34_v225, 0, 448, [16, 64], target_memory=pl.Mem.Vec)
                    stage_34_v233: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_34_v232, stage_34_v227)
                    stage_34_v234: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_34_v231, stage_34_v228)
                    stage_34_v235: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_34_v233, stage_34_v234)
                    stage_34_v236: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_34_v235, target_type=pl.BF16, mode='rint')
                    stage_34_v237: pl.Tile[[16, 448], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_34_v226, [16, 448], [0, 0])
                    stage_34_v238: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.concat(stage_34_v237, stage_34_v236)
                    stage_34_v239: pl.Tile[[2, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.reshape(stage_34_v238, [2, 4096])
                    stage_34_v240: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_34_v239, [1, 4096], [0, 0])
                    stage_34_v241: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_34_v239, [1, 4096], [1, 0])
                    stage_34_v242: pl.Scalar[pl.INDEX] = stage_34_v195 // 8 * 384 + stage_34_v62
                    stage_34_v243: pl.Scalar[pl.INDEX] = stage_34_v242 + 384
                    stage_34_v244 = pl.tile.store(stage_34_v240, [stage_34_v242, 0], temporary_51)
                    stage_34_v245 = pl.tile.store(stage_34_v241, [stage_34_v243, 0], temporary_51)
                    stage_34_v246: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_116, [stage_34_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
                    stage_34_v247: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v246, [1, 32])
                    stage_34_v248: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_34_v247, 0.0)
                    stage_34_v249: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_34_v248, [32, 1])
                    stage_34_v250: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
                    stage_34_v251: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
                    stage_34_v252 = stage_34_v246
                    stage_34_v253 = stage_34_v249
                    stage_34_v254 = stage_34_v250
                    stage_34_v255 = stage_34_v251
                else:
                    stage_34_v252 = stage_34_v137
                    stage_34_v253 = stage_34_v138
                    stage_34_v254 = stage_34_v139
                    stage_34_v255 = stage_34_v140
                stage_34_v64 = stage_34_v252
                stage_34_v65 = stage_34_v253
                stage_34_v66 = stage_34_v254
                stage_34_v67 = stage_34_v255
            else:
                stage_34_v64 = stage_34_v51
                stage_34_v65 = stage_34_v52
                stage_34_v66 = stage_34_v53
                stage_34_v67 = stage_34_v54
            (stage_34_v68, stage_34_v69, stage_34_v70, stage_34_v71) = pl.yield_(stage_34_v64, stage_34_v65, stage_34_v66, stage_34_v67)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_34 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_34)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_35 = 32400 + core_id * 6
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
        for (stage_35_v6, (stage_35_v26, stage_35_v27)) in pl.pipeline(stage_35_v5, pl.min(stage_35_v5 + 32, 96), 8, stage=2, init_values=(temporary_65, temporary_64)):
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
        if 96 <= stage_35_v5 + 32:
            for (stage_35_v30, (stage_35_v37,)) in pl.range(96, 96, 8, init_values=(stage_35_v29,)):
                stage_35_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_35_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_35_v31, target_type=pl.INT8, mode='trunc')
                stage_35_v33: pl.Scalar[pl.INDEX] = pl.min(96 - stage_35_v30, 8)
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
    record_36 = 32832 + core_id * 6
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
        for (stage_36_v6, (stage_36_v26, stage_36_v27)) in pl.pipeline(stage_36_v5, pl.min(stage_36_v5 + 32, 96), 8, stage=2, init_values=(temporary_65, temporary_64)):
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
        if 96 <= stage_36_v5 + 32:
            for (stage_36_v30, (stage_36_v37,)) in pl.range(96, 96, 8, init_values=(stage_36_v29,)):
                stage_36_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_36_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_36_v31, target_type=pl.INT8, mode='trunc')
                stage_36_v33: pl.Scalar[pl.INDEX] = pl.min(96 - stage_36_v30, 8)
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
    record_37 = 33264 + core_id * 6
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
        for (stage_37_v6, (stage_37_v26, stage_37_v27)) in pl.pipeline(stage_37_v5, pl.min(stage_37_v5 + 32, 96), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_37_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_37_v6, 7168], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_37_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_37_v7)
            stage_37_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_37_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_37_v8, stage_37_v9)
            stage_37_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_37_v10, [1, 8])
            stage_37_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_37_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_37_v12, stage_37_v11)
            stage_37_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_37_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_37_v14, stage_37_v13)
            stage_37_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_37_v13, 0.007874015748031496)
            stage_37_v17 = pl.tile.store(stage_37_v16, [7, stage_37_v6], stage_37_v26)
            stage_37_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_37_v15, [8, 1])
            stage_37_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_37_v6, 7168], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_37_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_37_v19, stage_37_v18)
            stage_37_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_37_v20, target_type=pl.INT32, mode='rint')
            stage_37_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_37_v21, target_type=pl.FP16, mode='round')
            stage_37_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_37_v22, target_type=pl.INT8, mode='trunc')
            stage_37_v24 = pl.tile.store(stage_37_v23, [stage_37_v6, 7168], stage_37_v27)
            (stage_37_v28, stage_37_v29) = pl.yield_(stage_37_v17, stage_37_v24)
        if 96 <= stage_37_v5 + 32:
            for (stage_37_v30, (stage_37_v37,)) in pl.range(96, 96, 8, init_values=(stage_37_v29,)):
                stage_37_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_37_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_37_v31, target_type=pl.INT8, mode='trunc')
                stage_37_v33: pl.Scalar[pl.INDEX] = pl.min(96 - stage_37_v30, 8)
                stage_37_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_37_v33, 1024])] = pl.tile.set_validshape(stage_37_v32, stage_37_v33, 1024)
                stage_37_v35 = pl.tile.store(stage_37_v34, [stage_37_v30, 7168], stage_37_v37)
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
    record_38 = 33696 + core_id * 6
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
        for (stage_38_v6, (stage_38_v26, stage_38_v27)) in pl.pipeline(stage_38_v5, pl.min(stage_38_v5 + 32, 96), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_38_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_38_v6, 4096], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_38_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_38_v7)
            stage_38_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_38_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_38_v8, stage_38_v9)
            stage_38_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v10, [1, 8])
            stage_38_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_38_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_38_v12, stage_38_v11)
            stage_38_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_38_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_38_v14, stage_38_v13)
            stage_38_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_38_v13, 0.007874015748031496)
            stage_38_v17 = pl.tile.store(stage_38_v16, [4, stage_38_v6], stage_38_v26)
            stage_38_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v15, [8, 1])
            stage_38_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_38_v6, 4096], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_38_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_38_v19, stage_38_v18)
            stage_38_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_38_v20, target_type=pl.INT32, mode='rint')
            stage_38_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_38_v21, target_type=pl.FP16, mode='round')
            stage_38_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_38_v22, target_type=pl.INT8, mode='trunc')
            stage_38_v24 = pl.tile.store(stage_38_v23, [stage_38_v6, 4096], stage_38_v27)
            (stage_38_v28, stage_38_v29) = pl.yield_(stage_38_v17, stage_38_v24)
        if 96 <= stage_38_v5 + 32:
            for (stage_38_v30, (stage_38_v37,)) in pl.range(96, 96, 8, init_values=(stage_38_v29,)):
                stage_38_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_38_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_38_v31, target_type=pl.INT8, mode='trunc')
                stage_38_v33: pl.Scalar[pl.INDEX] = pl.min(96 - stage_38_v30, 8)
                stage_38_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_38_v33, 1024])] = pl.tile.set_validshape(stage_38_v32, stage_38_v33, 1024)
                stage_38_v35 = pl.tile.store(stage_38_v34, [stage_38_v30, 4096], stage_38_v37)
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
    record_39 = 34128 + core_id * 6
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
        for (stage_39_v6, (stage_39_v26, stage_39_v27)) in pl.pipeline(stage_39_v5, pl.min(stage_39_v5 + 32, 96), 8, stage=2, init_values=(temporary_65, temporary_64)):
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
        if 96 <= stage_39_v5 + 32:
            for (stage_39_v30, (stage_39_v37,)) in pl.range(96, 96, 8, init_values=(stage_39_v29,)):
                stage_39_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_39_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_39_v31, target_type=pl.INT8, mode='trunc')
                stage_39_v33: pl.Scalar[pl.INDEX] = pl.min(96 - stage_39_v30, 8)
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
    record_40 = 34560 + core_id * 6
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
        for (stage_40_v6, (stage_40_v26, stage_40_v27)) in pl.pipeline(stage_40_v5, pl.min(stage_40_v5 + 32, 96), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_40_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_40_v6, 0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_40_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_40_v7)
            stage_40_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_40_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_40_v8, stage_40_v9)
            stage_40_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_40_v10, [1, 8])
            stage_40_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_40_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_40_v12, stage_40_v11)
            stage_40_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_40_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_40_v14, stage_40_v13)
            stage_40_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_40_v13, 0.007874015748031496)
            stage_40_v17 = pl.tile.store(stage_40_v16, [0, stage_40_v6], stage_40_v26)
            stage_40_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_40_v15, [8, 1])
            stage_40_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_40_v6, 0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_40_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_40_v19, stage_40_v18)
            stage_40_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_40_v20, target_type=pl.INT32, mode='rint')
            stage_40_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_40_v21, target_type=pl.FP16, mode='round')
            stage_40_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_40_v22, target_type=pl.INT8, mode='trunc')
            stage_40_v24 = pl.tile.store(stage_40_v23, [stage_40_v6, 0], stage_40_v27)
            (stage_40_v28, stage_40_v29) = pl.yield_(stage_40_v17, stage_40_v24)
        if 96 <= stage_40_v5 + 32:
            for (stage_40_v30, (stage_40_v37,)) in pl.range(96, 96, 8, init_values=(stage_40_v29,)):
                stage_40_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_40_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_40_v31, target_type=pl.INT8, mode='trunc')
                stage_40_v33: pl.Scalar[pl.INDEX] = pl.min(96 - stage_40_v30, 8)
                stage_40_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_40_v33, 1024])] = pl.tile.set_validshape(stage_40_v32, stage_40_v33, 1024)
                stage_40_v35 = pl.tile.store(stage_40_v34, [stage_40_v30, 0], stage_40_v37)
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
    record_41 = 34992 + core_id * 6
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
        stage_41_v4: pl.Scalar[pl.INDEX] = block_41
        stage_41_v5: pl.Scalar[pl.INDEX] = stage_41_v4 * 32
        for (stage_41_v6, (stage_41_v26, stage_41_v27)) in pl.pipeline(stage_41_v5, pl.min(stage_41_v5 + 32, 96), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_41_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_41_v6, 2048], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_41_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_41_v7)
            stage_41_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_41_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_41_v8, stage_41_v9)
            stage_41_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_41_v10, [1, 8])
            stage_41_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_41_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_41_v12, stage_41_v11)
            stage_41_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_41_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_41_v14, stage_41_v13)
            stage_41_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_41_v13, 0.007874015748031496)
            stage_41_v17 = pl.tile.store(stage_41_v16, [2, stage_41_v6], stage_41_v26)
            stage_41_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_41_v15, [8, 1])
            stage_41_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_41_v6, 2048], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_41_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_41_v19, stage_41_v18)
            stage_41_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_41_v20, target_type=pl.INT32, mode='rint')
            stage_41_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_41_v21, target_type=pl.FP16, mode='round')
            stage_41_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_41_v22, target_type=pl.INT8, mode='trunc')
            stage_41_v24 = pl.tile.store(stage_41_v23, [stage_41_v6, 2048], stage_41_v27)
            (stage_41_v28, stage_41_v29) = pl.yield_(stage_41_v17, stage_41_v24)
        if 96 <= stage_41_v5 + 32:
            for (stage_41_v30, (stage_41_v37,)) in pl.range(96, 96, 8, init_values=(stage_41_v29,)):
                stage_41_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_41_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_41_v31, target_type=pl.INT8, mode='trunc')
                stage_41_v33: pl.Scalar[pl.INDEX] = pl.min(96 - stage_41_v30, 8)
                stage_41_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v33, 1024])] = pl.tile.set_validshape(stage_41_v32, stage_41_v33, 1024)
                stage_41_v35 = pl.tile.store(stage_41_v34, [stage_41_v30, 2048], stage_41_v37)
                stage_41_v36 = pl.yield_(stage_41_v35)
            stage_41_v25 = pl.yield_(stage_41_v36)
        else:
            stage_41_v25 = pl.yield_(stage_41_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_41 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_41)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_42 = 35424 + core_id * 6
    mapped_block_42 = pl.cast(pl.tensor.read(records, [record_42]), pl.INDEX)
    block_42 = pl.max(mapped_block_42, 0)
    if mapped_block_42 >= 0:
        subslot_42 = pl.cast(pl.tensor.read(records, [record_42 + 1]), pl.INDEX)
        done_42 = pl.tensor.read(records, [record_42 + 2])
        wait_base_42 = pl.cast(pl.tensor.read(records, [record_42 + 3]), pl.INDEX)
        wait_count_42 = pl.cast(pl.tensor.read(records, [record_42 + 4]), pl.INDEX)
        publish_42 = pl.tensor.read(records, [record_42 + 5])
        for dependency_42 in pl.range(wait_count_42):
            wait_index_42 = (wait_base_42 + dependency_42) * 2
            other_42 = pl.cast(pl.tensor.read(waits, [wait_index_42]), pl.INDEX)
            required_42 = pl.tensor.read(waits, [wait_index_42 + 1])
            target_42 = generation_base + required_42
            flag_42 = other_42 * 64
            pl.system.cacheinvalid(progress, [1], [flag_42])
            initial_42 = pl.tensor.read(progress, [flag_42])
            for (ready_42,) in pl.while_(init_values=(initial_42,)):
                pl.cond(ready_42 < target_42)
                pl.system.cacheinvalid(progress, [1], [flag_42])
                after_42 = pl.yield_(pl.tensor.read(progress, [flag_42]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_42_v4: pl.Scalar[pl.INDEX] = block_42
        stage_42_v5: pl.Scalar[pl.INDEX] = stage_42_v4 * 32
        for (stage_42_v6, (stage_42_v26, stage_42_v27)) in pl.pipeline(stage_42_v5, pl.min(stage_42_v5 + 32, 96), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_42_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_42_v6, 1024], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_42_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_42_v7)
            stage_42_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_42_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_42_v8, stage_42_v9)
            stage_42_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_42_v10, [1, 8])
            stage_42_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_42_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_42_v12, stage_42_v11)
            stage_42_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_42_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_42_v14, stage_42_v13)
            stage_42_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_42_v13, 0.007874015748031496)
            stage_42_v17 = pl.tile.store(stage_42_v16, [1, stage_42_v6], stage_42_v26)
            stage_42_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_42_v15, [8, 1])
            stage_42_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_42_v6, 1024], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_42_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_42_v19, stage_42_v18)
            stage_42_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_42_v20, target_type=pl.INT32, mode='rint')
            stage_42_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_42_v21, target_type=pl.FP16, mode='round')
            stage_42_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_42_v22, target_type=pl.INT8, mode='trunc')
            stage_42_v24 = pl.tile.store(stage_42_v23, [stage_42_v6, 1024], stage_42_v27)
            (stage_42_v28, stage_42_v29) = pl.yield_(stage_42_v17, stage_42_v24)
        if 96 <= stage_42_v5 + 32:
            for (stage_42_v30, (stage_42_v37,)) in pl.range(96, 96, 8, init_values=(stage_42_v29,)):
                stage_42_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_42_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_42_v31, target_type=pl.INT8, mode='trunc')
                stage_42_v33: pl.Scalar[pl.INDEX] = pl.min(96 - stage_42_v30, 8)
                stage_42_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_42_v33, 1024])] = pl.tile.set_validshape(stage_42_v32, stage_42_v33, 1024)
                stage_42_v35 = pl.tile.store(stage_42_v34, [stage_42_v30, 1024], stage_42_v37)
                stage_42_v36 = pl.yield_(stage_42_v35)
            stage_42_v25 = pl.yield_(stage_42_v36)
        else:
            stage_42_v25 = pl.yield_(stage_42_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_42 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_42)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_43 = 35856 + core_id * 6
    mapped_block_43 = pl.cast(pl.tensor.read(records, [record_43]), pl.INDEX)
    block_43 = pl.max(mapped_block_43, 0)
    if mapped_block_43 >= 0:
        subslot_43 = pl.cast(pl.tensor.read(records, [record_43 + 1]), pl.INDEX)
        done_43 = pl.tensor.read(records, [record_43 + 2])
        wait_base_43 = pl.cast(pl.tensor.read(records, [record_43 + 3]), pl.INDEX)
        wait_count_43 = pl.cast(pl.tensor.read(records, [record_43 + 4]), pl.INDEX)
        publish_43 = pl.tensor.read(records, [record_43 + 5])
        for dependency_43 in pl.range(wait_count_43):
            wait_index_43 = (wait_base_43 + dependency_43) * 2
            other_43 = pl.cast(pl.tensor.read(waits, [wait_index_43]), pl.INDEX)
            required_43 = pl.tensor.read(waits, [wait_index_43 + 1])
            target_43 = generation_base + required_43
            flag_43 = other_43 * 64
            pl.system.cacheinvalid(progress, [1], [flag_43])
            initial_43 = pl.tensor.read(progress, [flag_43])
            for (ready_43,) in pl.while_(init_values=(initial_43,)):
                pl.cond(ready_43 < target_43)
                pl.system.cacheinvalid(progress, [1], [flag_43])
                after_43 = pl.yield_(pl.tensor.read(progress, [flag_43]))
        pl.system.fence()
        pl.system.cacheinvalid()
        stage_43_v8: pl.Scalar[pl.INDEX] = block_43
        stage_43_v9: pl.Scalar[pl.INDEX] = stage_43_v8
        stage_43_v10: pl.Scalar[pl.INDEX] = stage_43_v8 - stage_43_v9
        stage_43_v11: pl.Scalar[pl.INDEX] = stage_43_v10 * 4096
        stage_43_v12: pl.Scalar[pl.INDEX] = stage_43_v9 * 2
        stage_43_v13: pl.Tile[[4096], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wo_b_scale, [stage_43_v11], [4096], [4096], target_memory=pl.Mem.Vec)
        stage_43_v14: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_43_v13, [1, 4096])
        for (stage_43_v15, (stage_43_v99,)) in pl.range(stage_43_v12, pl.min(stage_43_v12 + 2, 96), init_values=(view_128,)):
            stage_43_v16: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 4096], dtype=pl.FP32, value=0.0)
            for (stage_43_v100, (stage_43_v108,)) in pl.pipeline(8, stage=2, init_values=(stage_43_v16,)):
                stage_43_v101: pl.Scalar[pl.INDEX] = stage_43_v100 * 4096 + stage_43_v11
                stage_43_v102: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_66, [stage_43_v15, stage_43_v101], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                stage_43_v103: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_65, [stage_43_v100, stage_43_v15])
                stage_43_v104: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_43_v102, target_type=pl.FP32, mode='none')
                stage_43_v105: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_43_v104, stage_43_v103)
                stage_43_v106: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v108, stage_43_v105)
                stage_43_v107: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_43_v106)
            stage_43_v17: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_43_v107, stage_43_v14)
            stage_43_v18: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.create([1, 4096], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            stage_43_v19: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_43_v17, target_type=pl.BF16, mode='rint')
            stage_43_v20: pl.Scalar[pl.INDEX] = pl.min(96 - stage_43_v15, 1)
            stage_43_v21: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_43_v19, target_type=pl.FP32, mode='round')
            stage_43_v22: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.load(view_3, [stage_43_v15, stage_43_v11], [1, 4096], [stage_43_v20, pl.min(pl.max(16384 - stage_43_v11, 0), 4096)], target_memory=pl.Mem.Vec)
            stage_43_v23: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.cast(stage_43_v22, target_type=pl.FP32, mode='round')
            stage_43_v24: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.load(view_3, [stage_43_v15, stage_43_v11 + 4096], [1, 4096], [stage_43_v20, pl.min(pl.max(12288 - stage_43_v11, 0), 4096)], target_memory=pl.Mem.Vec)
            stage_43_v25: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.cast(stage_43_v24, target_type=pl.FP32, mode='round')
            stage_43_v26: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.load(view_3, [stage_43_v15, stage_43_v11 + 8192], [1, 4096], [stage_43_v20, pl.min(pl.max(8192 - stage_43_v11, 0), 4096)], target_memory=pl.Mem.Vec)
            stage_43_v27: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.cast(stage_43_v26, target_type=pl.FP32, mode='round')
            stage_43_v28: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.load(view_3, [stage_43_v15, stage_43_v11 + 12288], [1, 4096], [stage_43_v20, pl.min(pl.max(4096 - stage_43_v11, 0), 4096)], target_memory=pl.Mem.Vec)
            stage_43_v29: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.cast(stage_43_v28, target_type=pl.FP32, mode='round')
            stage_43_v30: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_0, [stage_43_v15, 0])
            stage_43_v31: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_43_v21, stage_43_v30)
            stage_43_v32: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 0])
            stage_43_v33: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v23, stage_43_v32)
            stage_43_v34: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v31, stage_43_v33)
            stage_43_v35: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 4])
            stage_43_v36: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v25, stage_43_v35)
            stage_43_v37: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v34, stage_43_v36)
            stage_43_v38: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 8])
            stage_43_v39: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v27, stage_43_v38)
            stage_43_v40: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v37, stage_43_v39)
            stage_43_v41: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 12])
            stage_43_v42: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v29, stage_43_v41)
            stage_43_v43: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v40, stage_43_v42)
            stage_43_v44: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_43_v43, target_type=pl.BF16, mode='rint')
            stage_43_v45: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, 4096])] = pl.tile.set_validshape(stage_43_v44, stage_43_v20, 4096)
            stage_43_v46 = pl.tile.store(stage_43_v45, [stage_43_v15, stage_43_v11], stage_43_v99)
            stage_43_v47: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_0, [stage_43_v15, 1])
            stage_43_v48: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_43_v21, stage_43_v47)
            stage_43_v49: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 1])
            stage_43_v50: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v23, stage_43_v49)
            stage_43_v51: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v48, stage_43_v50)
            stage_43_v52: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 5])
            stage_43_v53: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v25, stage_43_v52)
            stage_43_v54: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v51, stage_43_v53)
            stage_43_v55: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 9])
            stage_43_v56: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v27, stage_43_v55)
            stage_43_v57: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v54, stage_43_v56)
            stage_43_v58: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 13])
            stage_43_v59: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v29, stage_43_v58)
            stage_43_v60: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v57, stage_43_v59)
            stage_43_v61: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_43_v60, target_type=pl.BF16, mode='rint')
            stage_43_v62: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, 4096])] = pl.tile.set_validshape(stage_43_v61, stage_43_v20, 4096)
            stage_43_v63 = pl.tile.store(stage_43_v62, [stage_43_v15, stage_43_v11 + 4096], stage_43_v46)
            stage_43_v64: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_0, [stage_43_v15, 2])
            stage_43_v65: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_43_v21, stage_43_v64)
            stage_43_v66: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 2])
            stage_43_v67: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v23, stage_43_v66)
            stage_43_v68: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v65, stage_43_v67)
            stage_43_v69: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 6])
            stage_43_v70: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v25, stage_43_v69)
            stage_43_v71: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v68, stage_43_v70)
            stage_43_v72: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 10])
            stage_43_v73: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v27, stage_43_v72)
            stage_43_v74: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v71, stage_43_v73)
            stage_43_v75: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 14])
            stage_43_v76: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v29, stage_43_v75)
            stage_43_v77: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v74, stage_43_v76)
            stage_43_v78: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_43_v77, target_type=pl.BF16, mode='rint')
            stage_43_v79: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, 4096])] = pl.tile.set_validshape(stage_43_v78, stage_43_v20, 4096)
            stage_43_v80 = pl.tile.store(stage_43_v79, [stage_43_v15, stage_43_v11 + 8192], stage_43_v63)
            stage_43_v81: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_0, [stage_43_v15, 3])
            stage_43_v82: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_43_v21, stage_43_v81)
            stage_43_v83: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 3])
            stage_43_v84: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v23, stage_43_v83)
            stage_43_v85: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v82, stage_43_v84)
            stage_43_v86: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 7])
            stage_43_v87: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v25, stage_43_v86)
            stage_43_v88: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v85, stage_43_v87)
            stage_43_v89: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 11])
            stage_43_v90: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v27, stage_43_v89)
            stage_43_v91: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v88, stage_43_v90)
            stage_43_v92: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_43_v15, 15])
            stage_43_v93: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_43_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_43_v29, stage_43_v92)
            stage_43_v94: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_43_v91, stage_43_v93)
            stage_43_v95: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_43_v94, target_type=pl.BF16, mode='rint')
            stage_43_v96: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v20, 4096])] = pl.tile.set_validshape(stage_43_v95, stage_43_v20, 4096)
            stage_43_v97 = pl.tile.store(stage_43_v96, [stage_43_v15, stage_43_v11 + 12288], stage_43_v80)
            stage_43_v98 = pl.yield_(stage_43_v97)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_43 != 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_43)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    return (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_111, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_128, progress)

@pl.jit.inline
def csa_static_group(temporary_4: pl.InOut[pl.Tensor[[96, 1], pl.FP32]], temporary_5: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]], view_2: pl.InOut[pl.Tensor[[96, 16384], pl.FP32]], view_3: pl.Tensor[[96, 16384], pl.BF16], temporary_7: pl.InOut[pl.Tensor[[384, 32], pl.FP32]], arg_hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], temporary_8: pl.InOut[pl.Tensor[[96, 32], pl.FP32]], arg_hc_attn_scale: pl.Tensor[[3], pl.FP32], temporary_0: pl.InOut[pl.Tensor[[96, 4], pl.FP32]], temporary_9: pl.InOut[pl.Tensor[[8, 8], pl.FP32]], temporary_6: pl.InOut[pl.Tensor[[96, 8], pl.FP32]], arg_hc_attn_base: pl.Tensor[[24], pl.FP32], view_12: pl.Tensor[[1, 24], pl.FP32], temporary_1: pl.InOut[pl.Tensor[[96, 16], pl.FP32]], temporary_10: pl.InOut[pl.Tensor[[8, 32], pl.FP32]], temporary_11: pl.InOut[pl.Tensor[[96, 4096], pl.BF16]], temporary_2: pl.InOut[pl.Tensor[[96, 4096], pl.BF16]], temporary_12: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]], arg_attn_norm_w: pl.Tensor[[4096], pl.BF16], arg_cmp_seq_lens: pl.Tensor[[16], pl.INT32], arg_cmp_query_start_loc: pl.Tensor[[17], pl.INT32], temporary_14: pl.InOut[pl.Tensor[[16], pl.INT32]], arg_kv_seq_lens: pl.Tensor[[16], pl.INT32], arg_idx_query_start_loc: pl.Tensor[[17], pl.INT32], temporary_15: pl.InOut[pl.Tensor[[16], pl.INT32]], arg_freqs_sin: pl.Tensor[[96, 64], pl.FP32], temporary_13: pl.InOut[pl.Tensor[[96, 64], pl.FP32]], temporary_20: pl.InOut[pl.Tensor[[96, 64], pl.FP32]], temporary_21: pl.InOut[pl.Tensor[[96, 64], pl.INT32]], temporary_24: pl.InOut[pl.Tensor[[96, 1024], pl.FP32]], arg_wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ], view_31: pl.Tensor[[1, 1024], pl.BF16], temporary_23: pl.InOut[pl.Tensor[[512, 1], pl.FP32]], temporary_19: pl.InOut[pl.Tensor[[96, 1], pl.FP32]], temporary_22: pl.InOut[pl.Tensor[[512, 1024], pl.INT8]], temporary_18: pl.InOut[pl.Tensor[[96, 1024], pl.INT8]], temporary_25: pl.InOut[pl.Tensor[[128, 32768], pl.INT32]], arg_wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ], arg_freqs_cos: pl.Tensor[[96, 64], pl.FP32], arg_wq_b_scale: pl.Tensor[[32768], pl.FP32], view_40: pl.InOut[pl.Tensor[[96, 32768], pl.BF16]], temporary_26: pl.InOut[pl.Tensor[[96, 512], pl.FP32]], arg_wkv: pl.Tensor[[4096, 512], pl.BF16], temporary_17: pl.InOut[pl.Tensor[[96, 512], pl.BF16]], arg_gamma_ckv: pl.Tensor[[512], pl.BF16], view_45: pl.InOut[pl.Tensor[[4640, 512], pl.BF16]], arg_ori_slot_mapping: pl.Tensor[[96, 2], pl.INT32], temporary_29: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], temporary_30: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], arg_cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], arg_cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], temporary_28: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], arg_position_ids: pl.Tensor[[96], pl.INT64], arg_cmp_ape: pl.Tensor[[4, 1024], pl.FP32], arg_state_block_table: pl.Tensor[[16, 65540], pl.INT32], arg_compress_state: pl.InOut[pl.Tensor[[145, 8192], pl.FP32]], arg_state_slot_mapping: pl.Tensor[[96, 2], pl.INT32], arg_cmp_freqs_cos: pl.Tensor[[40, 64], pl.FP32], arg_cmp_freqs_sin: pl.Tensor[[40, 64], pl.FP32], temporary_31: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], view_60: pl.Tensor[[1, 512], pl.BF16], view_61: pl.InOut[pl.Tensor[[525344, 512], pl.BF16]], temporary_27: pl.InOut[pl.Tensor[[96, 512], pl.FP32]], arg_cmp_slot_mapping: pl.Tensor[[40, 2], pl.INT32], temporary_34: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], temporary_35: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], arg_inner_wkv: pl.Tensor[[256, 4096], pl.BF16], arg_inner_wgate: pl.Tensor[[256, 4096], pl.BF16], temporary_36: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_inner_ape: pl.Tensor[[4, 256], pl.FP32], arg_inner_state_block_table: pl.Tensor[[16, 65540], pl.INT32], arg_inner_compress_state: pl.InOut[pl.Tensor[[145, 1040], pl.FP32]], arg_inner_state_slot_mapping: pl.Tensor[[96, 2], pl.INT32], temporary_33: pl.InOut[pl.Tensor[[384, 128], pl.BF16]], arg_inner_freqs_cos: pl.Tensor[[40, 64], pl.FP32], arg_inner_freqs_sin: pl.Tensor[[40, 64], pl.FP32], view_76: pl.Tensor[[1, 128], pl.BF16], temporary_38: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_hadamard_idx: pl.Tensor[[128, 128], pl.BF16], temporary_37: pl.InOut[pl.Tensor[[384, 1], pl.FP32]], arg_idx_native_kv_cache: pl.InOut[pl.Tensor[[16417, 4160], pl.INT8]], temporary_32: pl.InOut[pl.Tensor[[96, 128], pl.FP32]], arg_idx_slot_mapping: pl.Tensor[[40, 2], pl.INT32], temporary_42: pl.InOut[pl.Tensor[[384, 8192], pl.INT32]], arg_idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], view_85: pl.InOut[pl.Tensor[[384, 8192], pl.BF16]], arg_idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], temporary_43: pl.InOut[pl.Tensor[[6144, 128], pl.FP32]], temporary_41: pl.InOut[pl.Tensor[[24576, 128], pl.BF16]], temporary_39: pl.InOut[pl.Tensor[[24576, 128], pl.INT8]], temporary_40: pl.InOut[pl.Tensor[[24576, 1], pl.FP32]], temporary_45: pl.InOut[pl.Tensor[[1536, 64], pl.FP32]], arg_weights_proj: pl.Tensor[[4096, 64], pl.BF16], temporary_44: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], view_94: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_48: pl.InOut[pl.Tensor[[1024, 384], pl.FP16]], temporary_50: pl.InOut[pl.Tensor[[256], pl.INT64]], arg_idx_block_table: pl.Tensor[[16, 1026], pl.INT32], view_98: pl.InOut[pl.Tensor[[533552, 128], pl.INT8]], view_99: pl.InOut[pl.Tensor[[533552, 128], pl.INT8]], temporary_49: pl.InOut[pl.Tensor[[288, 1024], pl.FP32]], temporary_46: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32]], arg_idx_topk_scores: pl.InOut[pl.Tensor[[96, 512], pl.FP32]], arg_idx_topk: pl.InOut[pl.Tensor[[96, 512], pl.INT32]], temporary_53: pl.InOut[pl.Tensor[[96, 512], pl.INT32]], temporary_52: pl.InOut[pl.Tensor[[96, 640], pl.FP32]], view_106: pl.Tensor[[96, 1], pl.INT64], temporary_54: pl.InOut[pl.Tensor[[96, 16], pl.INT32]], arg_ori_block_table: pl.Tensor[[16, 4098], pl.INT32], temporary_55: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_62: pl.InOut[pl.Tensor[[256], pl.INT64]], view_111: pl.InOut[pl.Tensor[[6144, 512], pl.BF16]], temporary_56: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_57: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], temporary_58: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], temporary_59: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], view_116: pl.Tensor[[64, 1], pl.FP32], arg_cmp_block_table: pl.Tensor[[16, 1026], pl.INT32], temporary_60: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_61: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], arg_wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ], temporary_63: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_65: pl.InOut[pl.Tensor[[8, 384], pl.FP32]], temporary_64: pl.InOut[pl.Tensor[[384, 8192], pl.INT8]], arg_wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ], temporary_66: pl.InOut[pl.Tensor[[384, 32768], pl.INT32]], arg_wo_b_scale: pl.Tensor[[4096], pl.FP32], view_128: pl.InOut[pl.Tensor[[96, 16384], pl.BF16]], progress: pl.InOut[pl.Tensor[[4608], pl.INT64]], records: pl.Tensor[[36288], pl.INT64], waits: pl.Tensor[[33112], pl.INT64]) -> tuple[pl.Tensor[[96, 1], pl.FP32], pl.Tensor[[8, 16384], pl.FP32], pl.Tensor[[96, 16384], pl.FP32], pl.Tensor[[384, 32], pl.FP32], pl.Tensor[[96, 32], pl.FP32], pl.Tensor[[96, 4], pl.FP32], pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[96, 8], pl.FP32], pl.Tensor[[96, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32], pl.Tensor[[96, 4096], pl.BF16], pl.Tensor[[96, 4096], pl.BF16], pl.Tensor[[8, 4096], pl.BF16], pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.INT32], pl.Tensor[[96, 64], pl.FP32], pl.Tensor[[96, 64], pl.FP32], pl.Tensor[[96, 64], pl.INT32], pl.Tensor[[96, 1024], pl.FP32], pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[96, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[96, 1024], pl.INT8], pl.Tensor[[128, 32768], pl.INT32], pl.Tensor[[96, 32768], pl.BF16], pl.Tensor[[96, 512], pl.FP32], pl.Tensor[[96, 512], pl.BF16], pl.Tensor[[4640, 512], pl.BF16], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[145, 8192], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[525344, 512], pl.BF16], pl.Tensor[[96, 512], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[145, 1040], pl.FP32], pl.Tensor[[384, 128], pl.BF16], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[16417, 4160], pl.INT8], pl.Tensor[[96, 128], pl.FP32], pl.Tensor[[384, 8192], pl.INT32], pl.Tensor[[384, 8192], pl.BF16], pl.Tensor[[6144, 128], pl.FP32], pl.Tensor[[24576, 128], pl.BF16], pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Tensor[[1536, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[1024, 384], pl.FP16], pl.Tensor[[256], pl.INT64], pl.Tensor[[533552, 128], pl.INT8], pl.Tensor[[533552, 128], pl.INT8], pl.Tensor[[288, 1024], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Tensor[[96, 512], pl.FP32], pl.Tensor[[96, 512], pl.INT32], pl.Tensor[[96, 512], pl.INT32], pl.Tensor[[96, 640], pl.FP32], pl.Tensor[[96, 16], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[256], pl.INT64], pl.Tensor[[6144, 512], pl.BF16], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8], pl.Tensor[[384, 32768], pl.INT32], pl.Tensor[[96, 16384], pl.BF16], pl.Tensor[[4608], pl.INT64]]:
    pl.func_attr({'mx_tensor_views_blocked': True})
    with pl.cluster(name_hint='csa_static_group'):
        (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_111, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_128, progress) = csa_static_aic(temporary_4, temporary_5, view_2, view_3, temporary_7, arg_hc_attn_fn, temporary_8, arg_hc_attn_scale, temporary_0, temporary_9, temporary_6, arg_hc_attn_base, view_12, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, arg_attn_norm_w, arg_cmp_seq_lens, arg_cmp_query_start_loc, temporary_14, arg_kv_seq_lens, arg_idx_query_start_loc, temporary_15, arg_freqs_sin, temporary_13, temporary_20, temporary_21, temporary_24, arg_wq_a, view_31, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, arg_wq_b, arg_freqs_cos, arg_wq_b_scale, view_40, temporary_26, arg_wkv, temporary_17, arg_gamma_ckv, view_45, arg_ori_slot_mapping, temporary_29, temporary_30, arg_cmp_wkv, arg_cmp_wgate, temporary_28, arg_position_ids, arg_cmp_ape, arg_state_block_table, arg_compress_state, arg_state_slot_mapping, arg_cmp_freqs_cos, arg_cmp_freqs_sin, temporary_31, view_60, view_61, temporary_27, arg_cmp_slot_mapping, temporary_34, temporary_35, arg_inner_wkv, arg_inner_wgate, temporary_36, arg_inner_ape, arg_inner_state_block_table, arg_inner_compress_state, arg_inner_state_slot_mapping, temporary_33, arg_inner_freqs_cos, arg_inner_freqs_sin, view_76, temporary_38, arg_hadamard_idx, temporary_37, arg_idx_native_kv_cache, temporary_32, arg_idx_slot_mapping, temporary_42, arg_idx_wq_b, view_85, arg_idx_wq_b_scale, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, arg_weights_proj, temporary_44, view_94, temporary_48, temporary_50, arg_idx_block_table, view_98, view_99, temporary_49, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, view_106, temporary_54, arg_ori_block_table, temporary_55, temporary_62, view_111, temporary_56, temporary_57, temporary_58, temporary_59, view_116, arg_cmp_block_table, temporary_60, temporary_61, temporary_51, arg_wo_a, temporary_63, temporary_65, temporary_64, arg_wo_b, temporary_66, arg_wo_b_scale, view_128, progress, records, waits)
        (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_111, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_128, progress) = csa_static_aiv(temporary_4, temporary_5, view_2, view_3, temporary_7, arg_hc_attn_fn, temporary_8, arg_hc_attn_scale, temporary_0, temporary_9, temporary_6, arg_hc_attn_base, view_12, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, arg_attn_norm_w, arg_cmp_seq_lens, arg_cmp_query_start_loc, temporary_14, arg_kv_seq_lens, arg_idx_query_start_loc, temporary_15, arg_freqs_sin, temporary_13, temporary_20, temporary_21, temporary_24, arg_wq_a, view_31, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, arg_wq_b, arg_freqs_cos, arg_wq_b_scale, view_40, temporary_26, arg_wkv, temporary_17, arg_gamma_ckv, view_45, arg_ori_slot_mapping, temporary_29, temporary_30, arg_cmp_wkv, arg_cmp_wgate, temporary_28, arg_position_ids, arg_cmp_ape, arg_state_block_table, arg_compress_state, arg_state_slot_mapping, arg_cmp_freqs_cos, arg_cmp_freqs_sin, temporary_31, view_60, view_61, temporary_27, arg_cmp_slot_mapping, temporary_34, temporary_35, arg_inner_wkv, arg_inner_wgate, temporary_36, arg_inner_ape, arg_inner_state_block_table, arg_inner_compress_state, arg_inner_state_slot_mapping, temporary_33, arg_inner_freqs_cos, arg_inner_freqs_sin, view_76, temporary_38, arg_hadamard_idx, temporary_37, arg_idx_native_kv_cache, temporary_32, arg_idx_slot_mapping, temporary_42, arg_idx_wq_b, view_85, arg_idx_wq_b_scale, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, arg_weights_proj, temporary_44, view_94, temporary_48, temporary_50, arg_idx_block_table, view_98, view_99, temporary_49, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, view_106, temporary_54, arg_ori_block_table, temporary_55, temporary_62, view_111, temporary_56, temporary_57, temporary_58, temporary_59, view_116, arg_cmp_block_table, temporary_60, temporary_61, temporary_51, arg_wo_a, temporary_63, temporary_65, temporary_64, arg_wo_b, temporary_66, arg_wo_b_scale, view_128, progress, records, waits)
    return (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_111, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_128, progress)

@pl.jit
def entry(temporary_4: pl.InOut[pl.Tensor[[96, 1], pl.FP32]], temporary_5: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]], view_2: pl.InOut[pl.Tensor[[96, 16384], pl.FP32]], view_3: pl.Tensor[[96, 16384], pl.BF16], temporary_7: pl.InOut[pl.Tensor[[384, 32], pl.FP32]], arg_hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], temporary_8: pl.InOut[pl.Tensor[[96, 32], pl.FP32]], arg_hc_attn_scale: pl.Tensor[[3], pl.FP32], temporary_0: pl.InOut[pl.Tensor[[96, 4], pl.FP32]], temporary_9: pl.InOut[pl.Tensor[[8, 8], pl.FP32]], temporary_6: pl.InOut[pl.Tensor[[96, 8], pl.FP32]], arg_hc_attn_base: pl.Tensor[[24], pl.FP32], view_12: pl.Tensor[[1, 24], pl.FP32], temporary_1: pl.InOut[pl.Tensor[[96, 16], pl.FP32]], temporary_10: pl.InOut[pl.Tensor[[8, 32], pl.FP32]], temporary_11: pl.InOut[pl.Tensor[[96, 4096], pl.BF16]], temporary_2: pl.InOut[pl.Tensor[[96, 4096], pl.BF16]], temporary_12: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]], arg_attn_norm_w: pl.Tensor[[4096], pl.BF16], arg_cmp_seq_lens: pl.Tensor[[16], pl.INT32], arg_cmp_query_start_loc: pl.Tensor[[17], pl.INT32], temporary_14: pl.InOut[pl.Tensor[[16], pl.INT32]], arg_kv_seq_lens: pl.Tensor[[16], pl.INT32], arg_idx_query_start_loc: pl.Tensor[[17], pl.INT32], temporary_15: pl.InOut[pl.Tensor[[16], pl.INT32]], arg_freqs_sin: pl.Tensor[[96, 64], pl.FP32], temporary_13: pl.InOut[pl.Tensor[[96, 64], pl.FP32]], temporary_20: pl.InOut[pl.Tensor[[96, 64], pl.FP32]], temporary_21: pl.InOut[pl.Tensor[[96, 64], pl.INT32]], temporary_24: pl.InOut[pl.Tensor[[96, 1024], pl.FP32]], arg_wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ], view_31: pl.Tensor[[1, 1024], pl.BF16], temporary_23: pl.InOut[pl.Tensor[[512, 1], pl.FP32]], temporary_19: pl.InOut[pl.Tensor[[96, 1], pl.FP32]], temporary_22: pl.InOut[pl.Tensor[[512, 1024], pl.INT8]], temporary_18: pl.InOut[pl.Tensor[[96, 1024], pl.INT8]], temporary_25: pl.InOut[pl.Tensor[[128, 32768], pl.INT32]], arg_wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ], arg_freqs_cos: pl.Tensor[[96, 64], pl.FP32], arg_wq_b_scale: pl.Tensor[[32768], pl.FP32], view_40: pl.InOut[pl.Tensor[[96, 32768], pl.BF16]], temporary_26: pl.InOut[pl.Tensor[[96, 512], pl.FP32]], arg_wkv: pl.Tensor[[4096, 512], pl.BF16], temporary_17: pl.InOut[pl.Tensor[[96, 512], pl.BF16]], arg_gamma_ckv: pl.Tensor[[512], pl.BF16], view_45: pl.InOut[pl.Tensor[[4640, 512], pl.BF16]], arg_ori_slot_mapping: pl.Tensor[[96, 2], pl.INT32], temporary_29: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], temporary_30: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], arg_cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], arg_cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], temporary_28: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], arg_position_ids: pl.Tensor[[96], pl.INT64], arg_cmp_ape: pl.Tensor[[4, 1024], pl.FP32], arg_state_block_table: pl.Tensor[[16, 65540], pl.INT32], arg_compress_state: pl.InOut[pl.Tensor[[145, 8192], pl.FP32]], arg_state_slot_mapping: pl.Tensor[[96, 2], pl.INT32], arg_cmp_freqs_cos: pl.Tensor[[40, 64], pl.FP32], arg_cmp_freqs_sin: pl.Tensor[[40, 64], pl.FP32], temporary_31: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], view_60: pl.Tensor[[1, 512], pl.BF16], view_61: pl.InOut[pl.Tensor[[525344, 512], pl.BF16]], temporary_27: pl.InOut[pl.Tensor[[96, 512], pl.FP32]], arg_cmp_slot_mapping: pl.Tensor[[40, 2], pl.INT32], temporary_34: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], temporary_35: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], arg_inner_wkv: pl.Tensor[[256, 4096], pl.BF16], arg_inner_wgate: pl.Tensor[[256, 4096], pl.BF16], temporary_36: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_inner_ape: pl.Tensor[[4, 256], pl.FP32], arg_inner_state_block_table: pl.Tensor[[16, 65540], pl.INT32], arg_inner_compress_state: pl.InOut[pl.Tensor[[145, 1040], pl.FP32]], arg_inner_state_slot_mapping: pl.Tensor[[96, 2], pl.INT32], temporary_33: pl.InOut[pl.Tensor[[384, 128], pl.BF16]], arg_inner_freqs_cos: pl.Tensor[[40, 64], pl.FP32], arg_inner_freqs_sin: pl.Tensor[[40, 64], pl.FP32], view_76: pl.Tensor[[1, 128], pl.BF16], temporary_38: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_hadamard_idx: pl.Tensor[[128, 128], pl.BF16], temporary_37: pl.InOut[pl.Tensor[[384, 1], pl.FP32]], arg_idx_native_kv_cache: pl.InOut[pl.Tensor[[16417, 4160], pl.INT8]], temporary_32: pl.InOut[pl.Tensor[[96, 128], pl.FP32]], arg_idx_slot_mapping: pl.Tensor[[40, 2], pl.INT32], temporary_42: pl.InOut[pl.Tensor[[384, 8192], pl.INT32]], arg_idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], view_85: pl.InOut[pl.Tensor[[384, 8192], pl.BF16]], arg_idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], temporary_43: pl.InOut[pl.Tensor[[6144, 128], pl.FP32]], temporary_41: pl.InOut[pl.Tensor[[24576, 128], pl.BF16]], temporary_39: pl.InOut[pl.Tensor[[24576, 128], pl.INT8]], temporary_40: pl.InOut[pl.Tensor[[24576, 1], pl.FP32]], temporary_45: pl.InOut[pl.Tensor[[1536, 64], pl.FP32]], arg_weights_proj: pl.Tensor[[4096, 64], pl.BF16], temporary_44: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], view_94: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_48: pl.InOut[pl.Tensor[[1024, 384], pl.FP16]], temporary_50: pl.InOut[pl.Tensor[[256], pl.INT64]], arg_idx_block_table: pl.Tensor[[16, 1026], pl.INT32], temporary_49: pl.InOut[pl.Tensor[[288, 1024], pl.FP32]], temporary_46: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32]], arg_idx_topk_scores: pl.InOut[pl.Tensor[[96, 512], pl.FP32]], arg_idx_topk: pl.InOut[pl.Tensor[[96, 512], pl.INT32]], temporary_53: pl.InOut[pl.Tensor[[96, 512], pl.INT32]], temporary_52: pl.InOut[pl.Tensor[[96, 640], pl.FP32]], view_106: pl.Tensor[[96, 1], pl.INT64], temporary_54: pl.InOut[pl.Tensor[[96, 16], pl.INT32]], arg_ori_block_table: pl.Tensor[[16, 4098], pl.INT32], temporary_55: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_62: pl.InOut[pl.Tensor[[256], pl.INT64]], view_111: pl.InOut[pl.Tensor[[6144, 512], pl.BF16]], temporary_56: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_57: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], temporary_58: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], temporary_59: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], view_116: pl.Tensor[[64, 1], pl.FP32], arg_cmp_block_table: pl.Tensor[[16, 1026], pl.INT32], temporary_60: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_61: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], arg_wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ], temporary_63: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_65: pl.InOut[pl.Tensor[[8, 384], pl.FP32]], temporary_64: pl.InOut[pl.Tensor[[384, 8192], pl.INT8]], arg_wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ], temporary_66: pl.InOut[pl.Tensor[[384, 32768], pl.INT32]], arg_wo_b_scale: pl.Tensor[[4096], pl.FP32], view_128: pl.InOut[pl.Tensor[[96, 16384], pl.BF16]], progress: pl.InOut[pl.Tensor[[4608], pl.INT64]], records: pl.Tensor[[36288], pl.INT64], waits: pl.Tensor[[33112], pl.INT64]) -> tuple[pl.Tensor[[96, 1], pl.FP32], pl.Tensor[[8, 16384], pl.FP32], pl.Tensor[[96, 16384], pl.FP32], pl.Tensor[[384, 32], pl.FP32], pl.Tensor[[96, 32], pl.FP32], pl.Tensor[[96, 4], pl.FP32], pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[96, 8], pl.FP32], pl.Tensor[[96, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32], pl.Tensor[[96, 4096], pl.BF16], pl.Tensor[[96, 4096], pl.BF16], pl.Tensor[[8, 4096], pl.BF16], pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.INT32], pl.Tensor[[96, 64], pl.FP32], pl.Tensor[[96, 64], pl.FP32], pl.Tensor[[96, 64], pl.INT32], pl.Tensor[[96, 1024], pl.FP32], pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[96, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[96, 1024], pl.INT8], pl.Tensor[[128, 32768], pl.INT32], pl.Tensor[[96, 32768], pl.BF16], pl.Tensor[[96, 512], pl.FP32], pl.Tensor[[96, 512], pl.BF16], pl.Tensor[[4640, 512], pl.BF16], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[145, 8192], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[525344, 512], pl.BF16], pl.Tensor[[96, 512], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[145, 1040], pl.FP32], pl.Tensor[[384, 128], pl.BF16], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[16417, 4160], pl.INT8], pl.Tensor[[96, 128], pl.FP32], pl.Tensor[[384, 8192], pl.INT32], pl.Tensor[[384, 8192], pl.BF16], pl.Tensor[[6144, 128], pl.FP32], pl.Tensor[[24576, 128], pl.BF16], pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Tensor[[1536, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[1024, 384], pl.FP16], pl.Tensor[[256], pl.INT64], pl.Tensor[[288, 1024], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Tensor[[96, 512], pl.FP32], pl.Tensor[[96, 512], pl.INT32], pl.Tensor[[96, 512], pl.INT32], pl.Tensor[[96, 640], pl.FP32], pl.Tensor[[96, 16], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[256], pl.INT64], pl.Tensor[[6144, 512], pl.BF16], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8], pl.Tensor[[384, 32768], pl.INT32], pl.Tensor[[96, 16384], pl.BF16], pl.Tensor[[4608], pl.INT64]]:
    pl.func_attr({'mx_tensor_views_blocked': True})
    view_98_flat = pl.reshape(arg_idx_native_kv_cache, [68294720])
    view_98_span = pl.slice(view_98_flat, [68294656], [0])
    view_98 = pl.reshape(view_98_span, [533552, 128])
    view_99_flat = pl.reshape(arg_idx_native_kv_cache, [68294720])
    view_99_span = pl.slice(view_99_flat, [68294656], [64])
    view_99 = pl.reshape(view_99_span, [533552, 128])
    with pl.spmd(24, sync_start=True, name_hint='csa_static_full_card'):
        (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_111, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_128, progress) = csa_static_group(temporary_4, temporary_5, view_2, view_3, temporary_7, arg_hc_attn_fn, temporary_8, arg_hc_attn_scale, temporary_0, temporary_9, temporary_6, arg_hc_attn_base, view_12, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, arg_attn_norm_w, arg_cmp_seq_lens, arg_cmp_query_start_loc, temporary_14, arg_kv_seq_lens, arg_idx_query_start_loc, temporary_15, arg_freqs_sin, temporary_13, temporary_20, temporary_21, temporary_24, arg_wq_a, view_31, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, arg_wq_b, arg_freqs_cos, arg_wq_b_scale, view_40, temporary_26, arg_wkv, temporary_17, arg_gamma_ckv, view_45, arg_ori_slot_mapping, temporary_29, temporary_30, arg_cmp_wkv, arg_cmp_wgate, temporary_28, arg_position_ids, arg_cmp_ape, arg_state_block_table, arg_compress_state, arg_state_slot_mapping, arg_cmp_freqs_cos, arg_cmp_freqs_sin, temporary_31, view_60, view_61, temporary_27, arg_cmp_slot_mapping, temporary_34, temporary_35, arg_inner_wkv, arg_inner_wgate, temporary_36, arg_inner_ape, arg_inner_state_block_table, arg_inner_compress_state, arg_inner_state_slot_mapping, temporary_33, arg_inner_freqs_cos, arg_inner_freqs_sin, view_76, temporary_38, arg_hadamard_idx, temporary_37, arg_idx_native_kv_cache, temporary_32, arg_idx_slot_mapping, temporary_42, arg_idx_wq_b, view_85, arg_idx_wq_b_scale, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, arg_weights_proj, temporary_44, view_94, temporary_48, temporary_50, arg_idx_block_table, view_98, view_99, temporary_49, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, view_106, temporary_54, arg_ori_block_table, temporary_55, temporary_62, view_111, temporary_56, temporary_57, temporary_58, temporary_59, view_116, arg_cmp_block_table, temporary_60, temporary_61, temporary_51, arg_wo_a, temporary_63, temporary_65, temporary_64, arg_wo_b, temporary_66, arg_wo_b_scale, view_128, progress, records, waits)
    return (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, temporary_49, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_111, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_128, progress)
