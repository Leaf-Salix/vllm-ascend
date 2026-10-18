# SPDX-License-Identifier: Apache-2.0
"""历史静态CSA参考：H=8192，B=32，S=6。

整体思路：把原多task的数学体与逐核依赖排成一个24 AIC+48 AIV的MIX任务。
依次包含HC_pre、Q/K/V及压缩投影、Indexer/Top-K、稀疏Attention、输出投影、HC_post；
成员可以按原依赖重叠，源文件的排列顺序不是所有核之间的阶段屏障。
输入输出、Tile与编排形状为该历史规格冻结；位置、长度、页表和cache/state仍是
传入的device Tensor。GM progress每核独占128个INT32，1024为代际步长；
records/waits保存本规格逐核调度，工作区需由调用者独占并在首次使用前初始化。
阅读顺序：先看本目录layout.json中的协议、entry.bindings.json的入参合同，
再看csa_static_aic/csa_static_aiv的逐核成员与末尾entry。同步由共享signals.py提供，
cached100等待的生成源码处理保存在wait_codegen.py，不能仅凭导入认为它已启用。
此文件仅作长期参考；正式动态入口另在deepseek_v4_flash_csa_spmd目录。
"""
# 按形状特化的 CSA 单 SPMD；新增分片在核内循环，复用核体。
import pypto.language as pl
from ...signals import csa_signal_notify, csa_signal_wait
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
def csa_static_aic(temporary_4: pl.InOut[pl.Tensor[[192, 1], pl.FP32]], temporary_5: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]], view_2: pl.InOut[pl.Tensor[[192, 16384], pl.FP32]], view_3: pl.Tensor[[192, 16384], pl.BF16], temporary_7: pl.InOut[pl.Tensor[[768, 32], pl.FP32]], arg_hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], temporary_8: pl.InOut[pl.Tensor[[192, 32], pl.FP32]], arg_hc_attn_scale: pl.Tensor[[3], pl.FP32], temporary_0: pl.InOut[pl.Tensor[[192, 4], pl.FP32]], temporary_9: pl.InOut[pl.Tensor[[8, 8], pl.FP32]], temporary_6: pl.InOut[pl.Tensor[[192, 8], pl.FP32]], arg_hc_attn_base: pl.Tensor[[24], pl.FP32], view_12: pl.Tensor[[1, 24], pl.FP32], temporary_1: pl.InOut[pl.Tensor[[192, 16], pl.FP32]], temporary_10: pl.InOut[pl.Tensor[[8, 32], pl.FP32]], temporary_11: pl.InOut[pl.Tensor[[192, 4096], pl.BF16]], temporary_2: pl.InOut[pl.Tensor[[192, 4096], pl.BF16]], temporary_12: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]], arg_attn_norm_w: pl.Tensor[[4096], pl.BF16], arg_cmp_seq_lens: pl.Tensor[[32], pl.INT32], arg_cmp_query_start_loc: pl.Tensor[[33], pl.INT32], temporary_14: pl.InOut[pl.Tensor[[32], pl.INT32]], arg_kv_seq_lens: pl.Tensor[[32], pl.INT32], arg_idx_query_start_loc: pl.Tensor[[33], pl.INT32], temporary_15: pl.InOut[pl.Tensor[[32], pl.INT32]], arg_freqs_sin: pl.Tensor[[192, 64], pl.FP32], temporary_13: pl.InOut[pl.Tensor[[192, 64], pl.FP32]], temporary_20: pl.InOut[pl.Tensor[[192, 64], pl.FP32]], temporary_21: pl.InOut[pl.Tensor[[192, 64], pl.INT32]], temporary_24: pl.InOut[pl.Tensor[[192, 1024], pl.FP32]], arg_wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ], view_31: pl.Tensor[[1, 1024], pl.BF16], temporary_23: pl.InOut[pl.Tensor[[512, 1], pl.FP32]], temporary_19: pl.InOut[pl.Tensor[[192, 1], pl.FP32]], temporary_22: pl.InOut[pl.Tensor[[512, 1024], pl.INT8]], temporary_18: pl.InOut[pl.Tensor[[192, 1024], pl.INT8]], temporary_25: pl.InOut[pl.Tensor[[256, 32768], pl.INT32]], arg_wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ], arg_freqs_cos: pl.Tensor[[192, 64], pl.FP32], arg_wq_b_scale: pl.Tensor[[32768], pl.FP32], view_40: pl.InOut[pl.Tensor[[192, 32768], pl.BF16]], temporary_26: pl.InOut[pl.Tensor[[192, 512], pl.FP32]], arg_wkv: pl.Tensor[[4096, 512], pl.BF16], temporary_17: pl.InOut[pl.Tensor[[192, 512], pl.BF16]], arg_gamma_ckv: pl.Tensor[[512], pl.BF16], view_45: pl.InOut[pl.Tensor[[9248, 512], pl.BF16]], arg_ori_slot_mapping: pl.Tensor[[192, 2], pl.INT32], temporary_29: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], temporary_30: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], arg_cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], arg_cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], temporary_28: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], arg_position_ids: pl.Tensor[[192], pl.INT64], arg_cmp_ape: pl.Tensor[[4, 1024], pl.FP32], arg_state_block_table: pl.Tensor[[32, 4100], pl.INT32], arg_compress_state: pl.InOut[pl.Tensor[[289, 8192], pl.FP32]], arg_state_slot_mapping: pl.Tensor[[192, 2], pl.INT32], arg_cmp_freqs_cos: pl.Tensor[[80, 64], pl.FP32], arg_cmp_freqs_sin: pl.Tensor[[80, 64], pl.FP32], temporary_31: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], view_60: pl.Tensor[[1, 512], pl.BF16], view_61: pl.InOut[pl.Tensor[[67616, 512], pl.BF16]], temporary_27: pl.InOut[pl.Tensor[[192, 512], pl.FP32]], arg_cmp_slot_mapping: pl.Tensor[[80, 2], pl.INT32], temporary_34: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], temporary_35: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], arg_inner_wkv: pl.Tensor[[256, 4096], pl.BF16], arg_inner_wgate: pl.Tensor[[256, 4096], pl.BF16], temporary_36: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_inner_ape: pl.Tensor[[4, 256], pl.FP32], arg_inner_state_block_table: pl.Tensor[[32, 4100], pl.INT32], arg_inner_compress_state: pl.InOut[pl.Tensor[[289, 1040], pl.FP32]], arg_inner_state_slot_mapping: pl.Tensor[[192, 2], pl.INT32], temporary_33: pl.InOut[pl.Tensor[[384, 128], pl.BF16]], arg_inner_freqs_cos: pl.Tensor[[80, 64], pl.FP32], arg_inner_freqs_sin: pl.Tensor[[80, 64], pl.FP32], view_76: pl.Tensor[[1, 128], pl.BF16], temporary_38: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_hadamard_idx: pl.Tensor[[128, 128], pl.BF16], temporary_37: pl.InOut[pl.Tensor[[384, 1], pl.FP32]], arg_idx_native_kv_cache: pl.InOut[pl.Tensor[[2113, 4160], pl.INT8]], temporary_32: pl.InOut[pl.Tensor[[192, 128], pl.FP32]], arg_idx_slot_mapping: pl.Tensor[[80, 2], pl.INT32], temporary_42: pl.InOut[pl.Tensor[[384, 8192], pl.INT32]], arg_idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], view_85: pl.InOut[pl.Tensor[[384, 8192], pl.BF16]], arg_idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], temporary_43: pl.InOut[pl.Tensor[[12288, 128], pl.FP32]], temporary_41: pl.InOut[pl.Tensor[[24576, 128], pl.BF16]], temporary_39: pl.InOut[pl.Tensor[[24576, 128], pl.INT8]], temporary_40: pl.InOut[pl.Tensor[[24576, 1], pl.FP32]], temporary_45: pl.InOut[pl.Tensor[[1536, 64], pl.FP32]], arg_weights_proj: pl.Tensor[[4096, 64], pl.BF16], temporary_44: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], view_94: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_48: pl.InOut[pl.Tensor[[3072, 128], pl.FP16]], temporary_50: pl.InOut[pl.Tensor[[256], pl.INT64]], arg_idx_block_table: pl.Tensor[[32, 66], pl.INT32], view_98: pl.InOut[pl.Tensor[[68672, 128], pl.INT8]], view_99: pl.InOut[pl.Tensor[[68672, 128], pl.INT8]], temporary_49: pl.InOut[pl.Tensor[[96, 768], pl.FP32]], temporary_47: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_46: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32]], arg_idx_topk_scores: pl.InOut[pl.Tensor[[192, 512], pl.FP32]], arg_idx_topk: pl.InOut[pl.Tensor[[192, 512], pl.INT32]], temporary_53: pl.InOut[pl.Tensor[[192, 512], pl.INT32]], temporary_52: pl.InOut[pl.Tensor[[192, 640], pl.FP32]], view_107: pl.Tensor[[192, 1], pl.INT64], temporary_54: pl.InOut[pl.Tensor[[192, 16], pl.INT32]], arg_ori_block_table: pl.Tensor[[32, 258], pl.INT32], temporary_55: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_62: pl.InOut[pl.Tensor[[256], pl.INT64]], view_112: pl.InOut[pl.Tensor[[12288, 512], pl.BF16]], temporary_56: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_57: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], temporary_58: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], temporary_59: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], view_117: pl.Tensor[[64, 1], pl.FP32], arg_cmp_block_table: pl.Tensor[[32, 66], pl.INT32], temporary_60: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_61: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], arg_wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ], temporary_63: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_65: pl.InOut[pl.Tensor[[8, 384], pl.FP32]], temporary_64: pl.InOut[pl.Tensor[[384, 8192], pl.INT8]], arg_wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ], temporary_66: pl.InOut[pl.Tensor[[384, 32768], pl.INT32]], arg_wo_b_scale: pl.Tensor[[4096], pl.FP32], view_129: pl.InOut[pl.Tensor[[192, 16384], pl.BF16]], progress: pl.InOut[pl.Tensor[[9216], pl.INT32]], records: pl.Tensor[[33696], pl.INT64], waits: pl.Tensor[[49300], pl.INT64]) -> tuple[pl.Tensor[[192, 1], pl.FP32], pl.Tensor[[8, 16384], pl.FP32], pl.Tensor[[192, 16384], pl.FP32], pl.Tensor[[768, 32], pl.FP32], pl.Tensor[[192, 32], pl.FP32], pl.Tensor[[192, 4], pl.FP32], pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[192, 8], pl.FP32], pl.Tensor[[192, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32], pl.Tensor[[192, 4096], pl.BF16], pl.Tensor[[192, 4096], pl.BF16], pl.Tensor[[8, 4096], pl.BF16], pl.Tensor[[32], pl.INT32], pl.Tensor[[32], pl.INT32], pl.Tensor[[192, 64], pl.FP32], pl.Tensor[[192, 64], pl.FP32], pl.Tensor[[192, 64], pl.INT32], pl.Tensor[[192, 1024], pl.FP32], pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[192, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[192, 1024], pl.INT8], pl.Tensor[[256, 32768], pl.INT32], pl.Tensor[[192, 32768], pl.BF16], pl.Tensor[[192, 512], pl.FP32], pl.Tensor[[192, 512], pl.BF16], pl.Tensor[[9248, 512], pl.BF16], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[289, 8192], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[67616, 512], pl.BF16], pl.Tensor[[192, 512], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[289, 1040], pl.FP32], pl.Tensor[[384, 128], pl.BF16], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[2113, 4160], pl.INT8], pl.Tensor[[192, 128], pl.FP32], pl.Tensor[[384, 8192], pl.INT32], pl.Tensor[[384, 8192], pl.BF16], pl.Tensor[[12288, 128], pl.FP32], pl.Tensor[[24576, 128], pl.BF16], pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Tensor[[1536, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[3072, 128], pl.FP16], pl.Tensor[[256], pl.INT64], pl.Tensor[[68672, 128], pl.INT8], pl.Tensor[[68672, 128], pl.INT8], pl.Tensor[[96, 768], pl.FP32], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Tensor[[192, 512], pl.FP32], pl.Tensor[[192, 512], pl.INT32], pl.Tensor[[192, 512], pl.INT32], pl.Tensor[[192, 640], pl.FP32], pl.Tensor[[192, 16], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[256], pl.INT64], pl.Tensor[[12288, 512], pl.BF16], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8], pl.Tensor[[384, 32768], pl.INT32], pl.Tensor[[192, 16384], pl.BF16], pl.Tensor[[9216], pl.INT32]]:
    pl.func_attr({'mx_tensor_views_blocked': True})
    group_id = pl.tile.get_block_idx()
    core_id = group_id
    own_flag = core_id * 128
    pl.system.cacheinvalid(progress, [1], [own_flag])
    pl.system.fence()
    prior_progress = pl.tensor.read(progress, [own_flag])
    generation_base = (prior_progress // 1024 + 1) * 1024
    progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base, pl.INT32))
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
            flag_0 = other_0 * 128
            progress = csa_signal_wait(progress, flag_0, pl.cast(target_0, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_0_v4: pl.Scalar[pl.INDEX] = block_0
        for (stage_0_v5, (stage_0_v14,)) in pl.range(stage_0_v4, 48, 24, init_values=(temporary_7,)):
            stage_0_v6: pl.Scalar[pl.INDEX] = stage_0_v5 // 4 * 16
            stage_0_v7: pl.Scalar[pl.INDEX] = stage_0_v5 % 4
            stage_0_v8: pl.Scalar[pl.INDEX] = stage_0_v7 * 4096
            stage_0_v9: pl.Scalar[pl.INDEX] = pl.min(192 - stage_0_v6, 16)
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
            stage_0_v11: pl.Scalar[pl.INDEX] = stage_0_v7 * 192 + stage_0_v6
            stage_0_v12 = pl.tile.store(stage_0_v23, [stage_0_v11, 0], stage_0_v14)
            stage_0_v13 = pl.yield_(stage_0_v12)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_0 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_0, pl.INT32))
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
            flag_1 = other_1 * 128
            progress = csa_signal_wait(progress, flag_1, pl.cast(target_1, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_1_v4: pl.Scalar[pl.INDEX] = block_1
        stage_1_v5: pl.Scalar[pl.INDEX] = 0
        stage_1_v6: pl.Scalar[pl.INDEX] = stage_1_v4 % 8 * 128
        stage_1_v7: pl.Scalar[pl.INDEX] = stage_1_v4 // 8
        for (stage_1_v8, (stage_1_v17,)) in pl.range(stage_1_v7 * 64, 192, 3 * 64, init_values=(temporary_24,)):
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
        for (stage_1_v13, (stage_1_v24,)) in pl.range(192 + stage_1_v7 * 16, 192, 3 * 16, init_values=(stage_1_v12,)):
            stage_1_v14 = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for (stage_1_v25, (stage_1_v34,)) in pl.pipeline(16, stage=2, init_values=(stage_1_v14,)):
                stage_1_v26: pl.Scalar[pl.INDEX] = stage_1_v5 + stage_1_v25 * 256
                stage_1_v27: pl.Scalar[pl.INDEX] = pl.min(192 - stage_1_v13, 16)
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
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_1, pl.INT32))
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
            flag_2 = other_2 * 128
            progress = csa_signal_wait(progress, flag_2, pl.cast(target_2, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_2_v6: pl.Scalar[pl.INDEX] = block_2
        for (stage_2_v7, (stage_2_v14, stage_2_v15)) in pl.range(stage_2_v6, 192 // 2, 24, init_values=(temporary_34, temporary_35)):
            stage_2_v8: pl.Scalar[pl.INDEX] = stage_2_v7 // 8 * 16
            stage_2_v9: pl.Scalar[pl.INDEX] = stage_2_v7 % 8 * 32
            stage_2_v10: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            stage_2_v11: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_2_v16, (stage_2_v31, stage_2_v32)) in pl.pipeline(8, stage=2, init_values=(stage_2_v10, stage_2_v11)):
                stage_2_v17: pl.Scalar[pl.INDEX] = stage_2_v16 * 512
                stage_2_v18: pl.Scalar[pl.INDEX] = pl.min(192 - stage_2_v8, 16)
                stage_2_v19: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_2_v18, 512])] = pl.tile.load(temporary_2, [stage_2_v8, stage_2_v17], [16, 512], [stage_2_v18, 512], target_memory=pl.Mem.Mat)
                stage_2_v20: pl.Tile[[32, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_inner_wkv, [stage_2_v9, stage_2_v17], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
                stage_2_v21: pl.Tile[[32, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_inner_wgate, [stage_2_v9, stage_2_v17], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
                stage_2_v22: pl.Tile[[512, 32], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_2_v20)
                stage_2_v23: pl.Tile[[16, 512], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[stage_2_v18, 512])] = pl.tile.move(stage_2_v19, target_memory=pl.Mem.Left)
                stage_2_v24: pl.Tile[[512, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_2_v22, target_memory=pl.Mem.Right)
                stage_2_v25: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_2_v31, stage_2_v23, stage_2_v24, stage_2_v16 == 0)
                stage_2_v26: pl.Tile[[512, 32], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_2_v21)
                stage_2_v27: pl.Tile[[512, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_2_v26, target_memory=pl.Mem.Right)
                stage_2_v28: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_2_v32, stage_2_v23, stage_2_v27, stage_2_v16 == 0)
                (stage_2_v33, stage_2_v34) = pl.yield_(stage_2_v25, stage_2_v28)
            stage_2_v12 = pl.tile.store(stage_2_v33, [stage_2_v8, stage_2_v9], stage_2_v14)
            stage_2_v13 = pl.tile.store(stage_2_v34, [stage_2_v8, stage_2_v9], stage_2_v15)
            (stage_2_v29, stage_2_v30) = pl.yield_(stage_2_v12, stage_2_v13)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_2 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_2, pl.INT32))
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
            flag_3 = other_3 * 128
            progress = csa_signal_wait(progress, flag_3, pl.cast(target_3, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_3_v4: pl.Scalar[pl.INDEX] = block_3
        for (stage_3_v5, (stage_3_v9,)) in pl.range(stage_3_v4, 32, 24, init_values=(temporary_42,)):
            stage_3_v6: pl.Scalar[pl.INDEX] = stage_3_v5 * 256
            stage_3_v7: pl.Tile[[1024, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(arg_idx_wq_b, [0, stage_3_v6], [1024, 256], [1024, 256], target_memory=pl.Mem.Mat)
            for (stage_3_v10, (stage_3_v18,)) in pl.range(3, init_values=(stage_3_v9,)):
                stage_3_v11: pl.Scalar[pl.INDEX] = stage_3_v10 * 64
                stage_3_v12: pl.Scalar[pl.INDEX] = pl.min(192 - stage_3_v11, 64)
                stage_3_v13: pl.Tile[[64, 1024], pl.INT8, pl.Mem.Mat, pl.TileView(valid_shape=[stage_3_v12, 1024])] = pl.tile.load(temporary_18, [stage_3_v11, 0], [64, 1024], [stage_3_v12, 1024], target_memory=pl.Mem.Mat)
                stage_3_v14: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc, compact=True)
                stage_3_v15: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_3_v12, 256], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_3_v14, stage_3_v12, 256)
                for (stage_3_v19, (stage_3_v24,)) in pl.pipeline(0, 1024, 128, stage=2, init_values=(stage_3_v15,)):
                    stage_3_v20: pl.Tile[[64, 128], pl.INT8, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_3_v12, pl.const(64, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_3_v13, 0, stage_3_v19, [64, 128], target_memory=pl.Mem.Left)
                    stage_3_v21: pl.Tile[[128, 256], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_3_v7, stage_3_v19, 0, [128, 256], target_memory=pl.Mem.Right)
                    stage_3_v22: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_3_v12, 256], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_3_v24, stage_3_v20, stage_3_v21, stage_3_v19 == 0)
                    stage_3_v23: pl.Tile[[64, 256], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_3_v12, 256], compact=pl.CompactMode.normal)] = pl.yield_(stage_3_v22)
                stage_3_v16 = pl.tile.store(stage_3_v23, [stage_3_v11, stage_3_v6], stage_3_v18)
                stage_3_v17 = pl.yield_(stage_3_v16)
            stage_3_v8 = pl.yield_(stage_3_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_3 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_3, pl.INT32))
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
            flag_4 = other_4 * 128
            progress = csa_signal_wait(progress, flag_4, pl.cast(target_4, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_4_v6: pl.Scalar[pl.INDEX] = block_4
        for (stage_4_v7, (stage_4_v19, stage_4_v20)) in pl.range(stage_4_v6, 192 // 4, 24, init_values=(temporary_29, temporary_30)):
            stage_4_v8: pl.Scalar[pl.INDEX] = stage_4_v7 // 16 * 64
            stage_4_v9: pl.Scalar[pl.INDEX] = stage_4_v7 % 16 * 64
            stage_4_v10: pl.Scalar[pl.INDEX] = pl.min(192 - stage_4_v8, 64)
            stage_4_v11: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_4_v12: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_4_v13: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            stage_4_v14: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_4_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_4_v13, stage_4_v10, 64)
            stage_4_v15: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            stage_4_v16: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_4_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_4_v15, stage_4_v10, 64)
            for (stage_4_v21, (stage_4_v28, stage_4_v29)) in pl.pipeline(8, stage=2, init_values=(stage_4_v14, stage_4_v16)):
                stage_4_v22: pl.Scalar[pl.INDEX] = stage_4_v21 * 512
                stage_4_v23: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_4_v10, 512])] = pl.tile.load(temporary_2, [stage_4_v8, stage_4_v22], [64, 512], [stage_4_v10, 512], target_memory=pl.Mem.Mat)
                stage_4_v24: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_cmp_wkv, [stage_4_v9, stage_4_v22], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
                stage_4_v25: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_cmp_wgate, [stage_4_v9, stage_4_v22], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
                if stage_4_v21 == 0:
                    stage_4_v30: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_4_v24)
                    stage_4_v31: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    stage_4_v32: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_4_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_4_v31, stage_4_v10, 64)
                    for (stage_4_v40, (stage_4_v62,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_4_v32,)):
                        stage_4_v41: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_4_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_4_v23, 0, stage_4_v40, [64, 256], target_memory=pl.Mem.Left)
                        stage_4_v42: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_4_v30, stage_4_v40, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_4_v43: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_4_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_4_v62, stage_4_v41, stage_4_v42, stage_4_v40 == 0)
                        stage_4_v44: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_4_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_4_v43)
                    stage_4_v33: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_4_v25)
                    stage_4_v34: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    stage_4_v35: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_4_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(stage_4_v34, stage_4_v10, 64)
                    for (stage_4_v45, (stage_4_v63,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_4_v35,)):
                        stage_4_v46: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_4_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_4_v23, 0, stage_4_v45, [64, 256], target_memory=pl.Mem.Left)
                        stage_4_v47: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_4_v33, stage_4_v45, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_4_v48: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_4_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_4_v63, stage_4_v46, stage_4_v47, stage_4_v45 == 0)
                        stage_4_v49: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_4_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_4_v48)
                    (stage_4_v50, stage_4_v51) = pl.yield_(stage_4_v44, stage_4_v49)
                else:
                    stage_4_v36: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_4_v24)
                    for (stage_4_v52, (stage_4_v64,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_4_v28,)):
                        stage_4_v53: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_4_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_4_v23, 0, stage_4_v52, [64, 256], target_memory=pl.Mem.Left)
                        stage_4_v54: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_4_v36, stage_4_v52, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_4_v55: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_4_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_4_v64, stage_4_v53, stage_4_v54)
                        stage_4_v56: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_4_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_4_v55)
                    stage_4_v37: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_4_v25)
                    for (stage_4_v57, (stage_4_v65,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_4_v29,)):
                        stage_4_v58: pl.Tile[[64, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_4_v10, pl.const(64, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_4_v23, 0, stage_4_v57, [64, 256], target_memory=pl.Mem.Left)
                        stage_4_v59: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_4_v37, stage_4_v57, 0, [256, 64], target_memory=pl.Mem.Right)
                        stage_4_v60: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_4_v10, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(stage_4_v65, stage_4_v58, stage_4_v59)
                        stage_4_v61: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_4_v10, 64], compact=pl.CompactMode.normal)] = pl.yield_(stage_4_v60)
                    (stage_4_v50, stage_4_v51) = pl.yield_(stage_4_v56, stage_4_v61)
                (stage_4_v38, stage_4_v39) = pl.yield_(stage_4_v50, stage_4_v51)
            stage_4_v17 = pl.tile.store(stage_4_v38, [stage_4_v8, stage_4_v9], stage_4_v19)
            stage_4_v18 = pl.tile.store(stage_4_v39, [stage_4_v8, stage_4_v9], stage_4_v20)
            (stage_4_v26, stage_4_v27) = pl.yield_(stage_4_v17, stage_4_v18)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_4 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_4, pl.INT32))
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
            flag_5 = other_5 * 128
            progress = csa_signal_wait(progress, flag_5, pl.cast(target_5, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_5_v4: pl.Scalar[pl.INDEX] = block_5
        stage_5_v5: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_hadamard_idx, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat)
        for (stage_5_v6, (stage_5_v14,)) in pl.range(stage_5_v4, 12288 // 64, 24, init_values=(temporary_43,)):
            stage_5_v7: pl.Scalar[pl.INDEX] = stage_5_v6 * 64
            stage_5_v8: pl.Tile[[64, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(temporary_41, [stage_5_v7, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat)
            stage_5_v9: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(stage_5_v8, target_memory=pl.Mem.Left)
            stage_5_v10: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_5_v5, target_memory=pl.Mem.Right)
            stage_5_v11: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_5_v9, stage_5_v10)
            stage_5_v12 = pl.tile.store(stage_5_v11, [stage_5_v7, 0], stage_5_v14)
            stage_5_v13 = pl.yield_(stage_5_v12)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_5 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_5, pl.INT32))
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
            flag_6 = other_6 * 128
            progress = csa_signal_wait(progress, flag_6, pl.cast(target_6, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_6_v4: pl.Scalar[pl.INDEX] = block_6
        for (stage_6_v5, (stage_6_v9,)) in pl.range((19 - stage_6_v4 + 128) // 20, init_values=(temporary_25,)):
            stage_6_v6: pl.Scalar[pl.INDEX] = stage_6_v4 + stage_6_v5 * 20
            stage_6_v7: pl.Scalar[pl.INDEX] = stage_6_v6 * 256
            for (stage_6_v10, (stage_6_v17,)) in pl.range(0, 256, 128, init_values=(stage_6_v9,)):
                stage_6_v11: pl.Scalar[pl.INDEX] = pl.min(192 - stage_6_v10, 128)
                stage_6_v12 = pl.tensor.slice(temporary_22, [128, 256], [stage_6_v10, 0], [stage_6_v11, 256])
                stage_6_v13 = pl.tensor.slice(arg_wq_b, [256, 256], [0, stage_6_v7])
                stage_6_v14 = pl.tensor.matmul(stage_6_v12, stage_6_v13, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                for (stage_6_v18, (stage_6_v23,)) in pl.pipeline(256, 1024, 256, stage=2, init_values=(stage_6_v14,)):
                    stage_6_v19 = pl.tensor.slice(temporary_22, [128, 256], [stage_6_v10, stage_6_v18], [stage_6_v11, 256])
                    stage_6_v20 = pl.tensor.slice(arg_wq_b, [256, 256], [stage_6_v18, stage_6_v7])
                    stage_6_v21 = pl.tensor.matmul_acc(stage_6_v23, stage_6_v19, stage_6_v20, a_trans=False, b_trans=False)
                    stage_6_v22 = pl.yield_(stage_6_v21)
                stage_6_v15 = pl.tensor.assemble(stage_6_v17, stage_6_v22, [stage_6_v10, stage_6_v7])
                stage_6_v16 = pl.yield_(stage_6_v15)
            stage_6_v8 = pl.yield_(stage_6_v16)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_6 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_6, pl.INT32))
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
            flag_7 = other_7 * 128
            progress = csa_signal_wait(progress, flag_7, pl.cast(target_7, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        for (stage_7_v3, (stage_7_v6,)) in pl.range(0, 128, 64, init_values=(temporary_38,)):
            stage_7_v4: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_hadamard_idx, [0, stage_7_v3], [128, 64], [128, 64], target_memory=pl.Mem.Mat)
            for (stage_7_v7, (stage_7_v16,)) in pl.range(4, init_values=(stage_7_v6,)):
                stage_7_v8: pl.Scalar[pl.INDEX] = stage_7_v7 * 16
                stage_7_v9: pl.Scalar[pl.INDEX] = pl.min(64 - stage_7_v8, 16)
                stage_7_v10: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_7_v9, 128])] = pl.tile.load(temporary_33, [stage_7_v8, 0], [16, 128], [stage_7_v9, 128], target_memory=pl.Mem.Mat)
                stage_7_v11: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[stage_7_v9, 128])] = pl.tile.move(stage_7_v10, target_memory=pl.Mem.Left)
                stage_7_v12: pl.Tile[[128, 64], pl.BF16, pl.Mem.Right] = pl.tile.move(stage_7_v4, target_memory=pl.Mem.Right)
                stage_7_v13: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[stage_7_v9, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul(stage_7_v11, stage_7_v12)
                stage_7_v14 = pl.tile.store(stage_7_v13, [stage_7_v8, stage_7_v3], stage_7_v16)
                stage_7_v15 = pl.yield_(stage_7_v14)
            stage_7_v5 = pl.yield_(stage_7_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_7 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_7, pl.INT32))
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
            flag_8 = other_8 * 128
            progress = csa_signal_wait(progress, flag_8, pl.cast(target_8, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_8_v4: pl.Scalar[pl.INDEX] = block_8
        for (stage_8_v5, (stage_8_v14,)) in pl.range(stage_8_v4, 12 * 4, 8, init_values=(temporary_45,)):
            stage_8_v6: pl.Scalar[pl.INDEX] = stage_8_v5 // 4
            stage_8_v7: pl.Scalar[pl.INDEX] = stage_8_v5 - stage_8_v6 * 4
            stage_8_v8: pl.Scalar[pl.INDEX] = stage_8_v6 * 16
            stage_8_v9: pl.Scalar[pl.INDEX] = pl.min(192 - stage_8_v8, 16)
            stage_8_v10: pl.Scalar[pl.INDEX] = stage_8_v7 * 1024
            stage_8_v11: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_8_v15, (stage_8_v20,)) in pl.range(2, init_values=(stage_8_v11,)):
                stage_8_v16: pl.Scalar[pl.INDEX] = stage_8_v10 + stage_8_v15 * 512
                stage_8_v17: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_8_v9, 512])] = pl.tile.load(temporary_2, [stage_8_v8, stage_8_v16], [16, 512], [stage_8_v9, 512], target_memory=pl.Mem.Mat)
                stage_8_v18: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_weights_proj, [stage_8_v16, 0], [512, 64], [512, 64], target_memory=pl.Mem.Mat)
                for (stage_8_v21, (stage_8_v26,)) in pl.pipeline(0, 512, 256, stage=2, init_values=(stage_8_v20,)):
                    stage_8_v22: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_8_v9, pl.const(16, pl.INDEX)), 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_8_v17, 0, stage_8_v21, [16, 256], target_memory=pl.Mem.Left)
                    stage_8_v23: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_8_v18, stage_8_v21, 0, [256, 64], target_memory=pl.Mem.Right)
                    stage_8_v24: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_8_v26, stage_8_v22, stage_8_v23, stage_8_v15 == 0 and stage_8_v21 == 0)
                    stage_8_v25: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_8_v24)
                stage_8_v19: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_8_v25)
            stage_8_v12 = pl.tile.store(stage_8_v19, [stage_8_v7 * 384 + stage_8_v8, 0], stage_8_v14)
            stage_8_v13 = pl.yield_(stage_8_v12)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_8 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_8, pl.INT32))
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
            flag_9 = other_9 * 128
            progress = csa_signal_wait(progress, flag_9, pl.cast(target_9, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_9_v4: pl.Scalar[pl.INDEX] = block_9
        for (stage_9_v5, (stage_9_v9,)) in pl.range((19 - stage_9_v4 + 128) // 20, init_values=(temporary_25,)):
            stage_9_v6: pl.Scalar[pl.INDEX] = stage_9_v4 + stage_9_v5 * 20
            stage_9_v7: pl.Scalar[pl.INDEX] = stage_9_v6 * 256
            for (stage_9_v10, (stage_9_v17,)) in pl.range(0, 256, 128, init_values=(stage_9_v9,)):
                stage_9_v11: pl.Scalar[pl.INDEX] = pl.min(192 - stage_9_v10, 128)
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
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_9, pl.INT32))
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
            flag_10 = other_10 * 128
            progress = csa_signal_wait(progress, flag_10, pl.cast(target_10, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_10_v13: pl.Scalar[pl.INDEX] = block_10
        stage_10_v14: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        stage_10_v15: pl.Scalar[pl.INDEX] = 0
        for (stage_10_v19, (stage_10_v30,)) in pl.range(stage_10_v14 // 6, init_values=(stage_10_v15,)):
            stage_10_v20: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_10_v19])
            stage_10_v21: pl.Scalar[pl.INDEX] = pl.max(stage_10_v30, pl.cast(stage_10_v20, pl.INDEX) // 4)
            stage_10_v22: pl.Scalar[pl.INDEX] = pl.yield_(stage_10_v21)
        stage_10_v16: pl.Scalar[pl.INDEX] = pl.min(stage_10_v22, 262144)
        stage_10_v17: pl.Scalar[pl.INDEX] = pl.max((stage_10_v16 + 8191) // 8192, 1)
        pl.system.set_ffts(temporary_50)
        for stage_10_v18 in pl.range(stage_10_v13, stage_10_v14 // 2 * stage_10_v17, 24):
            if stage_10_v14 < 48:
                stage_10_v31: pl.Scalar[pl.INDEX] = stage_10_v18 % (stage_10_v14 // 2) * 2
                stage_10_v32: pl.Scalar[pl.INDEX] = stage_10_v18 // (stage_10_v14 // 2)
                (stage_10_v45, stage_10_v46) = pl.yield_(stage_10_v32, stage_10_v31)
            else:
                stage_10_v33: pl.Scalar[pl.INDEX] = stage_10_v18 // stage_10_v17 * 2
                stage_10_v34: pl.Scalar[pl.INDEX] = stage_10_v18 % stage_10_v17
                (stage_10_v45, stage_10_v46) = pl.yield_(stage_10_v34, stage_10_v33)
            stage_10_v23: pl.Scalar[pl.INDEX] = stage_10_v46 // 6
            stage_10_v24: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_10_v46 + 1])
            stage_10_v25: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_10_v23])
            stage_10_v26: pl.Scalar[pl.INDEX] = pl.cast(stage_10_v25, pl.INDEX) // 4
            stage_10_v27: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_10_v26, (stage_10_v24 + 1) // 4), 262144), 0)
            stage_10_v28: pl.Scalar[pl.INDEX] = stage_10_v45 * 8192
            stage_10_v29: pl.Scalar[pl.INDEX] = 8192
            if stage_10_v28 < stage_10_v27:
                stage_10_v35: pl.Scalar[pl.INDEX] = pl.min(stage_10_v29, stage_10_v27 - stage_10_v28)
                stage_10_v36: pl.Scalar[pl.INDEX] = (stage_10_v35 + 767) // 768
                stage_10_v37: pl.Scalar[pl.INDEX] = pl.min(stage_10_v36 * 384, 4096)
                stage_10_v38: pl.Scalar[pl.INDEX] = (stage_10_v37 + 383) // 384
                stage_10_v39: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.load(temporary_39, [stage_10_v46 * 64, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat)
                stage_10_v40: pl.Tile[[16, 128], pl.FP16, pl.Mem.Mat] = pl.tile.load(temporary_48, [stage_10_v46 // 2 * 16, 0], [16, 128], [16, 128], target_memory=pl.Mem.Mat)
                stage_10_v41: pl.Tile[[128, 128], pl.INT8, pl.Mem.Left] = pl.tile.move(stage_10_v39, target_memory=pl.Mem.Left)
                stage_10_v42: pl.Tile[[16, 128], pl.FP16, pl.Mem.Left] = pl.tile.move(stage_10_v40, target_memory=pl.Mem.Left)
                for stage_10_v43 in pl.range(stage_10_v38):
                    if 2 <= stage_10_v43:
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    stage_10_v47: pl.Scalar[pl.INDEX] = stage_10_v43 * 384
                    stage_10_v48: pl.Scalar[pl.INDEX] = stage_10_v13 * 4 + stage_10_v43 % 2 * 2
                    stage_10_v49: pl.Scalar[pl.INDEX] = 0
                    stage_10_v50: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_10_v51: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v49
                    stage_10_v52: pl.Scalar[pl.INDEX] = pl.min(stage_10_v51 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v53: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v52])
                    stage_10_v54: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v53, pl.INDEX), 0)
                    stage_10_v55: pl.Scalar[pl.INDEX] = stage_10_v54 * 4160
                    if stage_10_v55 % 128 == 0:
                        stage_10_v252: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v50, view_98, [0, 0], [stage_10_v55 // 128, 0], [32, 128], transpose=False)
                        stage_10_v253: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v252)
                    else:
                        stage_10_v254: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v50, view_99, [0, 0], [stage_10_v55 // 128, 0], [32, 128], transpose=False)
                        stage_10_v253: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v254)
                    stage_10_v56: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v49 + 32
                    stage_10_v57: pl.Scalar[pl.INDEX] = pl.min(stage_10_v56 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v58: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v57])
                    stage_10_v59: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v58, pl.INDEX), 0)
                    stage_10_v60: pl.Scalar[pl.INDEX] = stage_10_v59 * 4160
                    if stage_10_v60 % 128 == 0:
                        stage_10_v255: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v253, view_98, [32, 0], [stage_10_v60 // 128, 0], [32, 128], transpose=False)
                        stage_10_v256: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v255)
                    else:
                        stage_10_v257: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v253, view_99, [32, 0], [stage_10_v60 // 128, 0], [32, 128], transpose=False)
                        stage_10_v256: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v257)
                    stage_10_v61: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v49 + 64
                    stage_10_v62: pl.Scalar[pl.INDEX] = pl.min(stage_10_v61 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v63: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v62])
                    stage_10_v64: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v63, pl.INDEX), 0)
                    stage_10_v65: pl.Scalar[pl.INDEX] = stage_10_v64 * 4160
                    if stage_10_v65 % 128 == 0:
                        stage_10_v258: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v256, view_98, [64, 0], [stage_10_v65 // 128, 0], [32, 128], transpose=False)
                        stage_10_v259: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v258)
                    else:
                        stage_10_v260: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v256, view_99, [64, 0], [stage_10_v65 // 128, 0], [32, 128], transpose=False)
                        stage_10_v259: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v260)
                    stage_10_v66: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v49 + 96
                    stage_10_v67: pl.Scalar[pl.INDEX] = pl.min(stage_10_v66 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v68: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v67])
                    stage_10_v69: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v68, pl.INDEX), 0)
                    stage_10_v70: pl.Scalar[pl.INDEX] = stage_10_v69 * 4160
                    if stage_10_v70 % 128 == 0:
                        stage_10_v261: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v259, view_98, [96, 0], [stage_10_v70 // 128, 0], [32, 128], transpose=False)
                        stage_10_v262: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v261)
                    else:
                        stage_10_v263: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v259, view_99, [96, 0], [stage_10_v70 // 128, 0], [32, 128], transpose=False)
                        stage_10_v262: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v263)
                    stage_10_v71: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_10_v262)
                    stage_10_v72: pl.Tile[[128, 128], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_10_v71, target_memory=pl.Mem.Right)
                    stage_10_v73: pl.Tile[[128, 128], pl.INT8, pl.Mem.Right] = stage_10_v72
                    stage_10_v74: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_10_v75: pl.Tile[[128, 128], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_10_v41, stage_10_v73)
                    stage_10_v76: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_10_v74, stage_10_v75, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_10_v77: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = stage_10_v76
                    stage_10_v78: pl.Scalar[pl.INDEX] = 128
                    stage_10_v79: pl.Scalar[pl.INDEX] = 128
                    stage_10_v80: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_10_v81: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v79
                    stage_10_v82: pl.Scalar[pl.INDEX] = pl.min(stage_10_v81 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v83: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v82])
                    stage_10_v84: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v83, pl.INDEX), 0)
                    stage_10_v85: pl.Scalar[pl.INDEX] = stage_10_v84 * 4160
                    if stage_10_v85 % 128 == 0:
                        stage_10_v264: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v80, view_98, [0, 0], [stage_10_v85 // 128, 0], [32, 128], transpose=False)
                        stage_10_v265: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v264)
                    else:
                        stage_10_v266: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v80, view_99, [0, 0], [stage_10_v85 // 128, 0], [32, 128], transpose=False)
                        stage_10_v265: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v266)
                    stage_10_v86: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v79 + 32
                    stage_10_v87: pl.Scalar[pl.INDEX] = pl.min(stage_10_v86 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v88: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v87])
                    stage_10_v89: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v88, pl.INDEX), 0)
                    stage_10_v90: pl.Scalar[pl.INDEX] = stage_10_v89 * 4160
                    if stage_10_v90 % 128 == 0:
                        stage_10_v267: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v265, view_98, [32, 0], [stage_10_v90 // 128, 0], [32, 128], transpose=False)
                        stage_10_v268: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v267)
                    else:
                        stage_10_v269: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v265, view_99, [32, 0], [stage_10_v90 // 128, 0], [32, 128], transpose=False)
                        stage_10_v268: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v269)
                    stage_10_v91: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v79 + 64
                    stage_10_v92: pl.Scalar[pl.INDEX] = pl.min(stage_10_v91 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v93: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v92])
                    stage_10_v94: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v93, pl.INDEX), 0)
                    stage_10_v95: pl.Scalar[pl.INDEX] = stage_10_v94 * 4160
                    if stage_10_v95 % 128 == 0:
                        stage_10_v270: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v268, view_98, [64, 0], [stage_10_v95 // 128, 0], [32, 128], transpose=False)
                        stage_10_v271: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v270)
                    else:
                        stage_10_v272: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v268, view_99, [64, 0], [stage_10_v95 // 128, 0], [32, 128], transpose=False)
                        stage_10_v271: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v272)
                    stage_10_v96: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v79 + 96
                    stage_10_v97: pl.Scalar[pl.INDEX] = pl.min(stage_10_v96 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v98: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v97])
                    stage_10_v99: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v98, pl.INDEX), 0)
                    stage_10_v100: pl.Scalar[pl.INDEX] = stage_10_v99 * 4160
                    if stage_10_v100 % 128 == 0:
                        stage_10_v273: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v271, view_98, [96, 0], [stage_10_v100 // 128, 0], [32, 128], transpose=False)
                        stage_10_v274: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v273)
                    else:
                        stage_10_v275: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v271, view_99, [96, 0], [stage_10_v100 // 128, 0], [32, 128], transpose=False)
                        stage_10_v274: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v275)
                    stage_10_v101: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_10_v274)
                    stage_10_v102: pl.Tile[[128, 128], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_10_v101, target_memory=pl.Mem.Right)
                    stage_10_v103: pl.Tile[[128, 128], pl.INT8, pl.Mem.Right] = stage_10_v102
                    stage_10_v104: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_10_v105: pl.Tile[[128, 128], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_10_v77, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_10_v106: pl.Tile[[128, 128], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_10_v41, stage_10_v103)
                    stage_10_v107: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_10_v42, stage_10_v105)
                    stage_10_v108: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_10_v104, stage_10_v106, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_10_v109: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[2, 128])] = pl.tile.set_validshape(stage_10_v107, 2, 128)
                    stage_10_v110 = pl.tile.store(stage_10_v109, [stage_10_v48, stage_10_v78 - 128], temporary_49)
                    stage_10_v111: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = stage_10_v108
                    stage_10_v112: pl.Scalar[pl.INDEX] = 256
                    stage_10_v113: pl.Scalar[pl.INDEX] = 256
                    stage_10_v114: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_10_v115: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v113
                    stage_10_v116: pl.Scalar[pl.INDEX] = pl.min(stage_10_v115 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v117: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v116])
                    stage_10_v118: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v117, pl.INDEX), 0)
                    stage_10_v119: pl.Scalar[pl.INDEX] = stage_10_v118 * 4160
                    if stage_10_v119 % 128 == 0:
                        stage_10_v276: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v114, view_98, [0, 0], [stage_10_v119 // 128, 0], [32, 128], transpose=False)
                        stage_10_v277: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v276)
                    else:
                        stage_10_v278: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v114, view_99, [0, 0], [stage_10_v119 // 128, 0], [32, 128], transpose=False)
                        stage_10_v277: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v278)
                    stage_10_v120: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v113 + 32
                    stage_10_v121: pl.Scalar[pl.INDEX] = pl.min(stage_10_v120 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v122: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v121])
                    stage_10_v123: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v122, pl.INDEX), 0)
                    stage_10_v124: pl.Scalar[pl.INDEX] = stage_10_v123 * 4160
                    if stage_10_v124 % 128 == 0:
                        stage_10_v279: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v277, view_98, [32, 0], [stage_10_v124 // 128, 0], [32, 128], transpose=False)
                        stage_10_v280: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v279)
                    else:
                        stage_10_v281: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v277, view_99, [32, 0], [stage_10_v124 // 128, 0], [32, 128], transpose=False)
                        stage_10_v280: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v281)
                    stage_10_v125: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v113 + 64
                    stage_10_v126: pl.Scalar[pl.INDEX] = pl.min(stage_10_v125 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v127: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v126])
                    stage_10_v128: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v127, pl.INDEX), 0)
                    stage_10_v129: pl.Scalar[pl.INDEX] = stage_10_v128 * 4160
                    if stage_10_v129 % 128 == 0:
                        stage_10_v282: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v280, view_98, [64, 0], [stage_10_v129 // 128, 0], [32, 128], transpose=False)
                        stage_10_v283: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v282)
                    else:
                        stage_10_v284: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v280, view_99, [64, 0], [stage_10_v129 // 128, 0], [32, 128], transpose=False)
                        stage_10_v283: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v284)
                    stage_10_v130: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v113 + 96
                    stage_10_v131: pl.Scalar[pl.INDEX] = pl.min(stage_10_v130 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v132: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v131])
                    stage_10_v133: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v132, pl.INDEX), 0)
                    stage_10_v134: pl.Scalar[pl.INDEX] = stage_10_v133 * 4160
                    if stage_10_v134 % 128 == 0:
                        stage_10_v285: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v283, view_98, [96, 0], [stage_10_v134 // 128, 0], [32, 128], transpose=False)
                        stage_10_v286: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v285)
                    else:
                        stage_10_v287: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v283, view_99, [96, 0], [stage_10_v134 // 128, 0], [32, 128], transpose=False)
                        stage_10_v286: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v287)
                    stage_10_v135: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_10_v286)
                    stage_10_v136: pl.Tile[[128, 128], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_10_v135, target_memory=pl.Mem.Right)
                    stage_10_v137: pl.Tile[[128, 128], pl.INT8, pl.Mem.Right] = stage_10_v136
                    stage_10_v138: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_10_v139: pl.Tile[[128, 128], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_10_v111, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_10_v140: pl.Tile[[128, 128], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_10_v41, stage_10_v137)
                    stage_10_v141: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_10_v42, stage_10_v139)
                    stage_10_v142: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_10_v138, stage_10_v140, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_10_v143: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[2, 128])] = pl.tile.set_validshape(stage_10_v141, 2, 128)
                    stage_10_v144 = pl.tile.store(stage_10_v143, [stage_10_v48, stage_10_v112 - 128], temporary_49)
                    stage_10_v145: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = stage_10_v142
                    stage_10_v146: pl.Scalar[pl.INDEX] = 384
                    stage_10_v147: pl.Scalar[pl.INDEX] = 384
                    stage_10_v148: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_10_v149: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v147 // 384 * stage_10_v37 + stage_10_v147 % 384
                    stage_10_v150: pl.Scalar[pl.INDEX] = pl.min(stage_10_v149 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v151: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v150])
                    stage_10_v152: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v151, pl.INDEX), 0)
                    stage_10_v153: pl.Scalar[pl.INDEX] = stage_10_v152 * 4160
                    if stage_10_v153 % 128 == 0:
                        stage_10_v288: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v148, view_98, [0, 0], [stage_10_v153 // 128, 0], [32, 128], transpose=False)
                        stage_10_v289: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v288)
                    else:
                        stage_10_v290: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v148, view_99, [0, 0], [stage_10_v153 // 128, 0], [32, 128], transpose=False)
                        stage_10_v289: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v290)
                    stage_10_v154: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v147 // 384 * stage_10_v37 + stage_10_v147 % 384 + 32
                    stage_10_v155: pl.Scalar[pl.INDEX] = pl.min(stage_10_v154 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v156: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v155])
                    stage_10_v157: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v156, pl.INDEX), 0)
                    stage_10_v158: pl.Scalar[pl.INDEX] = stage_10_v157 * 4160
                    if stage_10_v158 % 128 == 0:
                        stage_10_v291: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v289, view_98, [32, 0], [stage_10_v158 // 128, 0], [32, 128], transpose=False)
                        stage_10_v292: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v291)
                    else:
                        stage_10_v293: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v289, view_99, [32, 0], [stage_10_v158 // 128, 0], [32, 128], transpose=False)
                        stage_10_v292: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v293)
                    stage_10_v159: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v147 // 384 * stage_10_v37 + stage_10_v147 % 384 + 64
                    stage_10_v160: pl.Scalar[pl.INDEX] = pl.min(stage_10_v159 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v161: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v160])
                    stage_10_v162: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v161, pl.INDEX), 0)
                    stage_10_v163: pl.Scalar[pl.INDEX] = stage_10_v162 * 4160
                    if stage_10_v163 % 128 == 0:
                        stage_10_v294: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v292, view_98, [64, 0], [stage_10_v163 // 128, 0], [32, 128], transpose=False)
                        stage_10_v295: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v294)
                    else:
                        stage_10_v296: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v292, view_99, [64, 0], [stage_10_v163 // 128, 0], [32, 128], transpose=False)
                        stage_10_v295: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v296)
                    stage_10_v164: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v147 // 384 * stage_10_v37 + stage_10_v147 % 384 + 96
                    stage_10_v165: pl.Scalar[pl.INDEX] = pl.min(stage_10_v164 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v166: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v165])
                    stage_10_v167: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v166, pl.INDEX), 0)
                    stage_10_v168: pl.Scalar[pl.INDEX] = stage_10_v167 * 4160
                    if stage_10_v168 % 128 == 0:
                        stage_10_v297: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v295, view_98, [96, 0], [stage_10_v168 // 128, 0], [32, 128], transpose=False)
                        stage_10_v298: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v297)
                    else:
                        stage_10_v299: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v295, view_99, [96, 0], [stage_10_v168 // 128, 0], [32, 128], transpose=False)
                        stage_10_v298: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v299)
                    stage_10_v169: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_10_v298)
                    stage_10_v170: pl.Tile[[128, 128], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_10_v169, target_memory=pl.Mem.Right)
                    stage_10_v171: pl.Tile[[128, 128], pl.INT8, pl.Mem.Right] = stage_10_v170
                    stage_10_v172: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_10_v173: pl.Tile[[128, 128], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_10_v145, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_10_v174: pl.Tile[[128, 128], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_10_v41, stage_10_v171)
                    stage_10_v175: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_10_v42, stage_10_v173)
                    stage_10_v176: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_10_v172, stage_10_v174, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_10_v177: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[2, 128])] = pl.tile.set_validshape(stage_10_v175, 2, 128)
                    stage_10_v178 = pl.tile.store(stage_10_v177, [stage_10_v48, stage_10_v146 - 128], temporary_49)
                    stage_10_v179: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = stage_10_v176
                    stage_10_v180: pl.Scalar[pl.INDEX] = 512
                    stage_10_v181: pl.Scalar[pl.INDEX] = 512
                    stage_10_v182: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_10_v183: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v181 // 384 * stage_10_v37 + stage_10_v181 % 384
                    stage_10_v184: pl.Scalar[pl.INDEX] = pl.min(stage_10_v183 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v185: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v184])
                    stage_10_v186: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v185, pl.INDEX), 0)
                    stage_10_v187: pl.Scalar[pl.INDEX] = stage_10_v186 * 4160
                    if stage_10_v187 % 128 == 0:
                        stage_10_v300: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v182, view_98, [0, 0], [stage_10_v187 // 128, 0], [32, 128], transpose=False)
                        stage_10_v301: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v300)
                    else:
                        stage_10_v302: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v182, view_99, [0, 0], [stage_10_v187 // 128, 0], [32, 128], transpose=False)
                        stage_10_v301: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v302)
                    stage_10_v188: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v181 // 384 * stage_10_v37 + stage_10_v181 % 384 + 32
                    stage_10_v189: pl.Scalar[pl.INDEX] = pl.min(stage_10_v188 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v190: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v189])
                    stage_10_v191: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v190, pl.INDEX), 0)
                    stage_10_v192: pl.Scalar[pl.INDEX] = stage_10_v191 * 4160
                    if stage_10_v192 % 128 == 0:
                        stage_10_v303: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v301, view_98, [32, 0], [stage_10_v192 // 128, 0], [32, 128], transpose=False)
                        stage_10_v304: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v303)
                    else:
                        stage_10_v305: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v301, view_99, [32, 0], [stage_10_v192 // 128, 0], [32, 128], transpose=False)
                        stage_10_v304: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v305)
                    stage_10_v193: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v181 // 384 * stage_10_v37 + stage_10_v181 % 384 + 64
                    stage_10_v194: pl.Scalar[pl.INDEX] = pl.min(stage_10_v193 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v195: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v194])
                    stage_10_v196: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v195, pl.INDEX), 0)
                    stage_10_v197: pl.Scalar[pl.INDEX] = stage_10_v196 * 4160
                    if stage_10_v197 % 128 == 0:
                        stage_10_v306: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v304, view_98, [64, 0], [stage_10_v197 // 128, 0], [32, 128], transpose=False)
                        stage_10_v307: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v306)
                    else:
                        stage_10_v308: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v304, view_99, [64, 0], [stage_10_v197 // 128, 0], [32, 128], transpose=False)
                        stage_10_v307: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v308)
                    stage_10_v198: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v181 // 384 * stage_10_v37 + stage_10_v181 % 384 + 96
                    stage_10_v199: pl.Scalar[pl.INDEX] = pl.min(stage_10_v198 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v200: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v199])
                    stage_10_v201: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v200, pl.INDEX), 0)
                    stage_10_v202: pl.Scalar[pl.INDEX] = stage_10_v201 * 4160
                    if stage_10_v202 % 128 == 0:
                        stage_10_v309: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v307, view_98, [96, 0], [stage_10_v202 // 128, 0], [32, 128], transpose=False)
                        stage_10_v310: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v309)
                    else:
                        stage_10_v311: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v307, view_99, [96, 0], [stage_10_v202 // 128, 0], [32, 128], transpose=False)
                        stage_10_v310: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v311)
                    stage_10_v203: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_10_v310)
                    stage_10_v204: pl.Tile[[128, 128], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_10_v203, target_memory=pl.Mem.Right)
                    stage_10_v205: pl.Tile[[128, 128], pl.INT8, pl.Mem.Right] = stage_10_v204
                    stage_10_v206: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_10_v207: pl.Tile[[128, 128], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_10_v179, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_10_v208: pl.Tile[[128, 128], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_10_v41, stage_10_v205)
                    stage_10_v209: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_10_v42, stage_10_v207)
                    stage_10_v210: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_10_v206, stage_10_v208, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_10_v211: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[2, 128])] = pl.tile.set_validshape(stage_10_v209, 2, 128)
                    stage_10_v212 = pl.tile.store(stage_10_v211, [stage_10_v48, stage_10_v180 - 128], temporary_49)
                    stage_10_v213: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = stage_10_v210
                    stage_10_v214: pl.Scalar[pl.INDEX] = 640
                    stage_10_v215: pl.Scalar[pl.INDEX] = 640
                    stage_10_v216: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    stage_10_v217: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v215 // 384 * stage_10_v37 + stage_10_v215 % 384
                    stage_10_v218: pl.Scalar[pl.INDEX] = pl.min(stage_10_v217 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v219: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v218])
                    stage_10_v220: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v219, pl.INDEX), 0)
                    stage_10_v221: pl.Scalar[pl.INDEX] = stage_10_v220 * 4160
                    if stage_10_v221 % 128 == 0:
                        stage_10_v312: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v216, view_98, [0, 0], [stage_10_v221 // 128, 0], [32, 128], transpose=False)
                        stage_10_v313: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v312)
                    else:
                        stage_10_v314: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v216, view_99, [0, 0], [stage_10_v221 // 128, 0], [32, 128], transpose=False)
                        stage_10_v313: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v314)
                    stage_10_v222: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v215 // 384 * stage_10_v37 + stage_10_v215 % 384 + 32
                    stage_10_v223: pl.Scalar[pl.INDEX] = pl.min(stage_10_v222 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v224: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v223])
                    stage_10_v225: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v224, pl.INDEX), 0)
                    stage_10_v226: pl.Scalar[pl.INDEX] = stage_10_v225 * 4160
                    if stage_10_v226 % 128 == 0:
                        stage_10_v315: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v313, view_98, [32, 0], [stage_10_v226 // 128, 0], [32, 128], transpose=False)
                        stage_10_v316: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v315)
                    else:
                        stage_10_v317: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v313, view_99, [32, 0], [stage_10_v226 // 128, 0], [32, 128], transpose=False)
                        stage_10_v316: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v317)
                    stage_10_v227: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v215 // 384 * stage_10_v37 + stage_10_v215 % 384 + 64
                    stage_10_v228: pl.Scalar[pl.INDEX] = pl.min(stage_10_v227 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v229: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v228])
                    stage_10_v230: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v229, pl.INDEX), 0)
                    stage_10_v231: pl.Scalar[pl.INDEX] = stage_10_v230 * 4160
                    if stage_10_v231 % 128 == 0:
                        stage_10_v318: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v316, view_98, [64, 0], [stage_10_v231 // 128, 0], [32, 128], transpose=False)
                        stage_10_v319: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v318)
                    else:
                        stage_10_v320: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v316, view_99, [64, 0], [stage_10_v231 // 128, 0], [32, 128], transpose=False)
                        stage_10_v319: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v320)
                    stage_10_v232: pl.Scalar[pl.INDEX] = stage_10_v28 + stage_10_v47 + stage_10_v215 // 384 * stage_10_v37 + stage_10_v215 % 384 + 96
                    stage_10_v233: pl.Scalar[pl.INDEX] = pl.min(stage_10_v232 // 32, pl.max((stage_10_v26 - 1) // 32, 0))
                    stage_10_v234: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_10_v23, stage_10_v233])
                    stage_10_v235: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_10_v234, pl.INDEX), 0)
                    stage_10_v236: pl.Scalar[pl.INDEX] = stage_10_v235 * 4160
                    if stage_10_v236 % 128 == 0:
                        stage_10_v321: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v319, view_98, [96, 0], [stage_10_v236 // 128, 0], [32, 128], transpose=False)
                        stage_10_v322: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v321)
                    else:
                        stage_10_v323: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_10_v319, view_99, [96, 0], [stage_10_v236 // 128, 0], [32, 128], transpose=False)
                        stage_10_v322: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_10_v323)
                    stage_10_v237: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_10_v322)
                    stage_10_v238: pl.Tile[[128, 128], pl.INT8, pl.Mem.Right] = pl.tile.move(stage_10_v237, target_memory=pl.Mem.Right)
                    stage_10_v239: pl.Tile[[128, 128], pl.INT8, pl.Mem.Right] = stage_10_v238
                    stage_10_v240: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    stage_10_v241: pl.Tile[[128, 128], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_10_v213, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_10_v242: pl.Tile[[128, 128], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_10_v41, stage_10_v239)
                    stage_10_v243: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_10_v42, stage_10_v241)
                    stage_10_v244: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_10_v240, stage_10_v242, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                    stage_10_v245: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[2, 128])] = pl.tile.set_validshape(stage_10_v243, 2, 128)
                    stage_10_v246 = pl.tile.store(stage_10_v245, [stage_10_v48, stage_10_v214 - 128], temporary_49)
                    stage_10_v247: pl.Tile[[128, 128], pl.FP16, pl.Mem.Mat] = stage_10_v244
                    stage_10_v248: pl.Tile[[128, 128], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_10_v247, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_10_v249: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_10_v42, stage_10_v248)
                    stage_10_v250: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[2, 128])] = pl.tile.set_validshape(stage_10_v249, 2, 128)
                    stage_10_v251 = pl.tile.store(stage_10_v250, [stage_10_v48, 640], temporary_49)
                    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                for stage_10_v44 in pl.range(pl.min(stage_10_v38, 2)):
                    pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_10 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_10, pl.INT32))
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
            flag_11 = other_11 * 128
            progress = csa_signal_wait(progress, flag_11, pl.cast(target_11, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_11_v4: pl.Scalar[pl.INDEX] = block_11
        stage_11_v5: pl.Scalar[pl.INDEX] = stage_11_v4 // 3 * 128
        stage_11_v6: pl.Scalar[pl.INDEX] = 0
        stage_11_v7: pl.Scalar[pl.INDEX] = stage_11_v4 % 3
        for (stage_11_v8, (stage_11_v17,)) in pl.range(stage_11_v7 * 64, 192, 3 * 64, init_values=(temporary_26,)):
            stage_11_v9: pl.Scalar[pl.INDEX] = 0 + stage_11_v8
            stage_11_v10: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_11_v18, (stage_11_v31,)) in pl.pipeline(8, stage=2, init_values=(stage_11_v10,)):
                stage_11_v19: pl.Scalar[pl.INDEX] = stage_11_v6 + stage_11_v18 * 512
                stage_11_v20: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(temporary_2, [stage_11_v9, stage_11_v19], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
                stage_11_v21: pl.Tile[[512, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_wkv, [stage_11_v19, stage_11_v5], [512, 128], [512, 128], target_memory=pl.Mem.Mat, cache=1)
                for (stage_11_v32, (stage_11_v43,)) in pl.pipeline(0, 512, 128, stage=2, init_values=(stage_11_v31,)):
                    stage_11_v33: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(stage_11_v20, 0, stage_11_v32, [64, 128], target_memory=pl.Mem.Left)
                    stage_11_v34: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_11_v21, stage_11_v32, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_11_v35: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_11_v43, stage_11_v33, stage_11_v34, stage_11_v18 == 0 and stage_11_v32 == 0)
                    stage_11_v36: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_11_v35)
                stage_11_v22: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_11_v36)
            stage_11_v11 = pl.tile.store(stage_11_v22, [stage_11_v8, stage_11_v5], stage_11_v17)
            stage_11_v12 = pl.yield_(stage_11_v11)
        for (stage_11_v13, (stage_11_v23,)) in pl.range(192 + stage_11_v7 * 16, 192, 3 * 16, init_values=(stage_11_v12,)):
            stage_11_v14: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_11_v24, (stage_11_v37,)) in pl.pipeline(8, stage=2, init_values=(stage_11_v14,)):
                stage_11_v25: pl.Scalar[pl.INDEX] = stage_11_v6 + stage_11_v24 * 512
                stage_11_v26: pl.Scalar[pl.INDEX] = pl.min(192 - stage_11_v13, 16)
                stage_11_v27: pl.Scalar[pl.INDEX] = 0 + stage_11_v13
                stage_11_v28: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_11_v26, 512])] = pl.tile.load(temporary_2, [stage_11_v27, stage_11_v25], [16, 512], [stage_11_v26, 512], target_memory=pl.Mem.Mat)
                stage_11_v29: pl.Tile[[512, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_wkv, [stage_11_v25, stage_11_v5], [512, 128], [512, 128], target_memory=pl.Mem.Mat, cache=1)
                for (stage_11_v38, (stage_11_v44,)) in pl.pipeline(0, 512, 128, stage=2, init_values=(stage_11_v37,)):
                    stage_11_v39: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_11_v26, pl.const(16, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_11_v28, 0, stage_11_v38, [16, 128], target_memory=pl.Mem.Left)
                    stage_11_v40: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_11_v29, stage_11_v38, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_11_v41: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_11_v44, stage_11_v39, stage_11_v40, stage_11_v24 == 0 and stage_11_v38 == 0)
                    stage_11_v42: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_11_v41)
                stage_11_v30: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_11_v42)
            stage_11_v15 = pl.tile.store(stage_11_v30, [stage_11_v13, stage_11_v5], stage_11_v23)
            stage_11_v16 = pl.yield_(stage_11_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_11 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_11, pl.INT32))
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
            flag_12 = other_12 * 128
            progress = csa_signal_wait(progress, flag_12, pl.cast(target_12, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_12_v4: pl.Scalar[pl.INDEX] = block_12
        stage_12_v5: pl.Scalar[pl.INDEX] = stage_12_v4 // 3 * 128
        stage_12_v6: pl.Scalar[pl.INDEX] = 0
        stage_12_v7: pl.Scalar[pl.INDEX] = stage_12_v4 % 3
        for (stage_12_v8, (stage_12_v17,)) in pl.range(stage_12_v7 * 64, 192, 3 * 64, init_values=(temporary_26,)):
            stage_12_v9: pl.Scalar[pl.INDEX] = 0 + stage_12_v8
            stage_12_v10: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_12_v18, (stage_12_v31,)) in pl.pipeline(8, stage=2, init_values=(stage_12_v10,)):
                stage_12_v19: pl.Scalar[pl.INDEX] = stage_12_v6 + stage_12_v18 * 512
                stage_12_v20: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(temporary_2, [stage_12_v9, stage_12_v19], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
                stage_12_v21: pl.Tile[[512, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_wkv, [stage_12_v19, stage_12_v5], [512, 128], [512, 128], target_memory=pl.Mem.Mat, cache=1)
                for (stage_12_v32, (stage_12_v43,)) in pl.pipeline(0, 512, 128, stage=2, init_values=(stage_12_v31,)):
                    stage_12_v33: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(stage_12_v20, 0, stage_12_v32, [64, 128], target_memory=pl.Mem.Left)
                    stage_12_v34: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_12_v21, stage_12_v32, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_12_v35: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_12_v43, stage_12_v33, stage_12_v34, stage_12_v18 == 0 and stage_12_v32 == 0)
                    stage_12_v36: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_12_v35)
                stage_12_v22: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_12_v36)
            stage_12_v11 = pl.tile.store(stage_12_v22, [stage_12_v8, stage_12_v5], stage_12_v17)
            stage_12_v12 = pl.yield_(stage_12_v11)
        for (stage_12_v13, (stage_12_v23,)) in pl.range(192 + stage_12_v7 * 16, 192, 3 * 16, init_values=(stage_12_v12,)):
            stage_12_v14: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for (stage_12_v24, (stage_12_v37,)) in pl.pipeline(8, stage=2, init_values=(stage_12_v14,)):
                stage_12_v25: pl.Scalar[pl.INDEX] = stage_12_v6 + stage_12_v24 * 512
                stage_12_v26: pl.Scalar[pl.INDEX] = pl.min(192 - stage_12_v13, 16)
                stage_12_v27: pl.Scalar[pl.INDEX] = 0 + stage_12_v13
                stage_12_v28: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[stage_12_v26, 512])] = pl.tile.load(temporary_2, [stage_12_v27, stage_12_v25], [16, 512], [stage_12_v26, 512], target_memory=pl.Mem.Mat)
                stage_12_v29: pl.Tile[[512, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(arg_wkv, [stage_12_v25, stage_12_v5], [512, 128], [512, 128], target_memory=pl.Mem.Mat, cache=1)
                for (stage_12_v38, (stage_12_v44,)) in pl.pipeline(0, 512, 128, stage=2, init_values=(stage_12_v37,)):
                    stage_12_v39: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(stage_12_v26, pl.const(16, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(stage_12_v28, 0, stage_12_v38, [16, 128], target_memory=pl.Mem.Left)
                    stage_12_v40: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_12_v29, stage_12_v38, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_12_v41: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_12_v44, stage_12_v39, stage_12_v40, stage_12_v24 == 0 and stage_12_v38 == 0)
                    stage_12_v42: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_12_v41)
                stage_12_v30: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_12_v42)
            stage_12_v15 = pl.tile.store(stage_12_v30, [stage_12_v13, stage_12_v5], stage_12_v23)
            stage_12_v16 = pl.yield_(stage_12_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_12 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_12, pl.INT32))
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
            flag_13 = other_13 * 128
            progress = csa_signal_wait(progress, flag_13, pl.cast(target_13, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_13_v21: pl.Scalar[pl.INDEX] = block_13
        pl.system.set_ffts(temporary_62)
        stage_13_v22: pl.Scalar[pl.INDEX] = (192 - stage_13_v21 + 23) // 24
        stage_13_v23: pl.Scalar[pl.INDEX] = stage_13_v22 * 5
        stage_13_v24: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.create([64, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat)
        stage_13_v25: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.tile.create([384, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat)
        for (stage_13_v26, (stage_13_v27, stage_13_v28)) in pl.range(stage_13_v23 + 2, init_values=(stage_13_v25, stage_13_v24)):
            if stage_13_v26 < stage_13_v23:
                stage_13_v29: pl.Scalar[pl.INDEX] = stage_13_v26 % 5
                stage_13_v30: pl.Scalar[pl.INDEX] = stage_13_v21 + stage_13_v26 // 5 * 24
                if stage_13_v29 == 0:
                    stage_13_v38: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v28, view_112, [0, 0], [stage_13_v30 * 64, 0], [64, 512], transpose=False)
                    stage_13_v39: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.yield_(stage_13_v38)
                else:
                    stage_13_v39: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.yield_(stage_13_v28)
                stage_13_v31: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_13_v30, stage_13_v29])
                if 0 < pl.cast(stage_13_v31, pl.INDEX):
                    stage_13_v40: pl.Scalar[pl.INDEX] = stage_13_v21 * 3 + stage_13_v26 % 3
                    stage_13_v41: pl.Scalar[pl.INDEX] = stage_13_v40 * 128
                    stage_13_v42: pl.Scalar[pl.INDEX] = stage_13_v40 * 64
                    pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    stage_13_v43: pl.Scalar[pl.INDEX] = stage_13_v26 % 3 * 128
                    stage_13_v44: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v27, temporary_56, [stage_13_v43, 0], [stage_13_v41, 0], [128, 512], transpose=False)
                    stage_13_v45: pl.Tile[[512, 384], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v44)
                    stage_13_v46: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                    for (stage_13_v71, (stage_13_v76,)) in pl.pipeline(0, 512, 128, stage=2, init_values=(stage_13_v46,)):
                        stage_13_v72: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(stage_13_v39, 0, stage_13_v71, [64, 128], target_memory=pl.Mem.Left)
                        stage_13_v73: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_13_v45, stage_13_v71, stage_13_v43, [128, 128], target_memory=pl.Mem.Right)
                        stage_13_v74: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(stage_13_v76, stage_13_v72, stage_13_v73, stage_13_v71 == 0)
                        stage_13_v75: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(stage_13_v74)
                    stage_13_v47 = pl.tile.store(stage_13_v75, [stage_13_v42, 0], temporary_57)
                    pl.system.sync_set(1, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                    stage_13_v48: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.yield_(stage_13_v44)
                else:
                    stage_13_v48: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.yield_(stage_13_v27)
                (stage_13_v49, stage_13_v50) = pl.yield_(stage_13_v48, stage_13_v39)
            else:
                (stage_13_v49, stage_13_v50) = pl.yield_(stage_13_v27, stage_13_v28)
            if 2 <= stage_13_v26:
                stage_13_v32: pl.Scalar[pl.INDEX] = stage_13_v26 - 2
                stage_13_v33: pl.Scalar[pl.INDEX] = stage_13_v32 % 5
                stage_13_v34: pl.Scalar[pl.INDEX] = stage_13_v21 + stage_13_v32 // 5 * 24
                stage_13_v35: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_13_v34, stage_13_v33])
                if 0 < pl.cast(stage_13_v35, pl.INDEX):
                    stage_13_v51: pl.Scalar[pl.INDEX] = stage_13_v21 * 3 + stage_13_v32 % 3
                    stage_13_v52: pl.Scalar[pl.INDEX] = stage_13_v51 * 64
                    pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    stage_13_v53: pl.Tile[[64, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(temporary_58, [stage_13_v52, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat)
                    stage_13_v54: pl.Scalar[pl.INDEX] = stage_13_v32 % 3 * 128
                    stage_13_v55: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(stage_13_v53, target_memory=pl.Mem.Left)
                    stage_13_v56: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_13_v49, stage_13_v54, 0, [128, 128], target_memory=pl.Mem.Right)
                    stage_13_v57: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v55, stage_13_v56)
                    stage_13_v58: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_13_v49, stage_13_v54, 128, [128, 128], target_memory=pl.Mem.Right)
                    stage_13_v59: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v55, stage_13_v58)
                    stage_13_v60 = pl.tile.store(stage_13_v57, [stage_13_v52, 0], temporary_59)
                    stage_13_v61: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = stage_13_v59
                    stage_13_v62: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_13_v49, stage_13_v54, 256, [128, 128], target_memory=pl.Mem.Right)
                    stage_13_v63: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v55, stage_13_v62)
                    stage_13_v64 = pl.tile.store(stage_13_v61, [stage_13_v52, 128], temporary_59)
                    stage_13_v65: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = stage_13_v63
                    stage_13_v66: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(stage_13_v49, stage_13_v54, 384, [128, 128], target_memory=pl.Mem.Right)
                    stage_13_v67: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v55, stage_13_v66)
                    stage_13_v68 = pl.tile.store(stage_13_v65, [stage_13_v52, 256], temporary_59)
                    stage_13_v69: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = stage_13_v67
                    stage_13_v70 = pl.tile.store(stage_13_v69, [stage_13_v52, 384], temporary_59)
                    pl.system.sync_set(3, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            (stage_13_v36, stage_13_v37) = pl.yield_(stage_13_v49, stage_13_v50)
        pl.system.set_ffts(temporary_62)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_13 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_13, pl.INT32))
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
            flag_14 = other_14 * 128
            progress = csa_signal_wait(progress, flag_14, pl.cast(target_14, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_14_v4: pl.Scalar[pl.INDEX] = block_14
        stage_14_v5: pl.Scalar[pl.INDEX] = stage_14_v4 // 4
        stage_14_v6: pl.Scalar[pl.INDEX] = stage_14_v4 % 4
        stage_14_v7: pl.Scalar[pl.INDEX] = stage_14_v5 * 128
        stage_14_v8: pl.Scalar[pl.INDEX] = pl.min(192 - stage_14_v7, 128)
        stage_14_v9: pl.Scalar[pl.INDEX] = 1920 + stage_14_v7
        stage_14_v10: pl.Scalar[pl.INDEX] = stage_14_v6 * 256
        stage_14_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_14_v9, 0], [stage_14_v8, 256])
        stage_14_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [5, 0, stage_14_v10])
        stage_14_v13 = pl.tensor.matmul(stage_14_v11, stage_14_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_14_v15, (stage_14_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_14_v13,)):
            stage_14_v16: pl.Scalar[pl.INDEX] = stage_14_v15 * 256
            stage_14_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_14_v9, stage_14_v16], [stage_14_v8, 256])
            stage_14_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [5, stage_14_v16, stage_14_v10])
            stage_14_v19 = pl.tensor.matmul_acc(stage_14_v21, stage_14_v17, stage_14_v18, a_trans=False, b_trans=False)
            stage_14_v20 = pl.yield_(stage_14_v19)
        stage_14_v14 = pl.tensor.assemble(temporary_63, stage_14_v20, [stage_14_v7, 5120 + stage_14_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_14 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_14, pl.INT32))
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
            flag_15 = other_15 * 128
            progress = csa_signal_wait(progress, flag_15, pl.cast(target_15, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_15_v4: pl.Scalar[pl.INDEX] = block_15
        stage_15_v5: pl.Scalar[pl.INDEX] = stage_15_v4 // 4
        stage_15_v6: pl.Scalar[pl.INDEX] = stage_15_v4 % 4
        stage_15_v7: pl.Scalar[pl.INDEX] = stage_15_v5 * 128
        stage_15_v8: pl.Scalar[pl.INDEX] = pl.min(192 - stage_15_v7, 128)
        stage_15_v9: pl.Scalar[pl.INDEX] = 384 + stage_15_v7
        stage_15_v10: pl.Scalar[pl.INDEX] = stage_15_v6 * 256
        stage_15_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_15_v9, 0], [stage_15_v8, 256])
        stage_15_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [1, 0, stage_15_v10])
        stage_15_v13 = pl.tensor.matmul(stage_15_v11, stage_15_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_15_v15, (stage_15_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_15_v13,)):
            stage_15_v16: pl.Scalar[pl.INDEX] = stage_15_v15 * 256
            stage_15_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_15_v9, stage_15_v16], [stage_15_v8, 256])
            stage_15_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [1, stage_15_v16, stage_15_v10])
            stage_15_v19 = pl.tensor.matmul_acc(stage_15_v21, stage_15_v17, stage_15_v18, a_trans=False, b_trans=False)
            stage_15_v20 = pl.yield_(stage_15_v19)
        stage_15_v14 = pl.tensor.assemble(temporary_63, stage_15_v20, [stage_15_v7, 1024 + stage_15_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_15 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_15, pl.INT32))
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
            flag_16 = other_16 * 128
            progress = csa_signal_wait(progress, flag_16, pl.cast(target_16, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_16_v4: pl.Scalar[pl.INDEX] = block_16
        stage_16_v5: pl.Scalar[pl.INDEX] = stage_16_v4 // 4
        stage_16_v6: pl.Scalar[pl.INDEX] = stage_16_v4 % 4
        stage_16_v7: pl.Scalar[pl.INDEX] = stage_16_v5 * 128
        stage_16_v8: pl.Scalar[pl.INDEX] = pl.min(192 - stage_16_v7, 128)
        stage_16_v9: pl.Scalar[pl.INDEX] = 0 + stage_16_v7
        stage_16_v10: pl.Scalar[pl.INDEX] = stage_16_v6 * 256
        stage_16_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_16_v9, 0], [stage_16_v8, 256])
        stage_16_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [0, 0, stage_16_v10])
        stage_16_v13 = pl.tensor.matmul(stage_16_v11, stage_16_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_16_v15, (stage_16_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_16_v13,)):
            stage_16_v16: pl.Scalar[pl.INDEX] = stage_16_v15 * 256
            stage_16_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_16_v9, stage_16_v16], [stage_16_v8, 256])
            stage_16_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [0, stage_16_v16, stage_16_v10])
            stage_16_v19 = pl.tensor.matmul_acc(stage_16_v21, stage_16_v17, stage_16_v18, a_trans=False, b_trans=False)
            stage_16_v20 = pl.yield_(stage_16_v19)
        stage_16_v14 = pl.tensor.assemble(temporary_63, stage_16_v20, [stage_16_v7, 0 + stage_16_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_16 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_16, pl.INT32))
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
            flag_17 = other_17 * 128
            progress = csa_signal_wait(progress, flag_17, pl.cast(target_17, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_17_v4: pl.Scalar[pl.INDEX] = block_17
        stage_17_v5: pl.Scalar[pl.INDEX] = stage_17_v4 * 512
        for (stage_17_v6, (stage_17_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_17_v7: pl.Scalar[pl.INDEX] = stage_17_v5 + stage_17_v6 * 256
            stage_17_v8: pl.Scalar[pl.INDEX] = pl.max(4, 0) * 1024
            stage_17_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_17_v8, stage_17_v7])
            for (stage_17_v12, (stage_17_v18,)) in pl.range(2, init_values=(stage_17_v11,)):
                stage_17_v13: pl.Scalar[pl.INDEX] = stage_17_v12 * 128
                stage_17_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_17_v13, 4096])
                stage_17_v15 = pl.tensor.matmul(stage_17_v14, stage_17_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_17_v16 = pl.tensor.assemble(stage_17_v18, stage_17_v15, [stage_17_v13, 4 * 4096 + stage_17_v7])
                stage_17_v17 = pl.yield_(stage_17_v16)
            stage_17_v10 = pl.yield_(stage_17_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_17 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_17, pl.INT32))
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
            flag_18 = other_18 * 128
            progress = csa_signal_wait(progress, flag_18, pl.cast(target_18, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_18_v4: pl.Scalar[pl.INDEX] = block_18
        stage_18_v5: pl.Scalar[pl.INDEX] = stage_18_v4 * 512
        for (stage_18_v6, (stage_18_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_18_v7: pl.Scalar[pl.INDEX] = stage_18_v5 + stage_18_v6 * 256
            stage_18_v8: pl.Scalar[pl.INDEX] = pl.max(0, 0) * 1024
            stage_18_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_18_v8, stage_18_v7])
            for (stage_18_v12, (stage_18_v18,)) in pl.range(2, init_values=(stage_18_v11,)):
                stage_18_v13: pl.Scalar[pl.INDEX] = stage_18_v12 * 128
                stage_18_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_18_v13, 0])
                stage_18_v15 = pl.tensor.matmul(stage_18_v14, stage_18_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_18_v16 = pl.tensor.assemble(stage_18_v18, stage_18_v15, [stage_18_v13, 0 * 4096 + stage_18_v7])
                stage_18_v17 = pl.yield_(stage_18_v16)
            stage_18_v10 = pl.yield_(stage_18_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_18 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_18, pl.INT32))
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
            flag_19 = other_19 * 128
            progress = csa_signal_wait(progress, flag_19, pl.cast(target_19, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_19_v4: pl.Scalar[pl.INDEX] = block_19
        stage_19_v5: pl.Scalar[pl.INDEX] = stage_19_v4 // 4
        stage_19_v6: pl.Scalar[pl.INDEX] = stage_19_v4 % 4
        stage_19_v7: pl.Scalar[pl.INDEX] = stage_19_v5 * 128
        stage_19_v8: pl.Scalar[pl.INDEX] = pl.min(192 - stage_19_v7, 128)
        stage_19_v9: pl.Scalar[pl.INDEX] = 2304 + stage_19_v7
        stage_19_v10: pl.Scalar[pl.INDEX] = stage_19_v6 * 256
        stage_19_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_19_v9, 0], [stage_19_v8, 256])
        stage_19_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [6, 0, stage_19_v10])
        stage_19_v13 = pl.tensor.matmul(stage_19_v11, stage_19_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_19_v15, (stage_19_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_19_v13,)):
            stage_19_v16: pl.Scalar[pl.INDEX] = stage_19_v15 * 256
            stage_19_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_19_v9, stage_19_v16], [stage_19_v8, 256])
            stage_19_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [6, stage_19_v16, stage_19_v10])
            stage_19_v19 = pl.tensor.matmul_acc(stage_19_v21, stage_19_v17, stage_19_v18, a_trans=False, b_trans=False)
            stage_19_v20 = pl.yield_(stage_19_v19)
        stage_19_v14 = pl.tensor.assemble(temporary_63, stage_19_v20, [stage_19_v7, 6144 + stage_19_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_19 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_19, pl.INT32))
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
            flag_20 = other_20 * 128
            progress = csa_signal_wait(progress, flag_20, pl.cast(target_20, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_20_v4: pl.Scalar[pl.INDEX] = block_20
        stage_20_v5: pl.Scalar[pl.INDEX] = stage_20_v4 // 4
        stage_20_v6: pl.Scalar[pl.INDEX] = stage_20_v4 % 4
        stage_20_v7: pl.Scalar[pl.INDEX] = stage_20_v5 * 128
        stage_20_v8: pl.Scalar[pl.INDEX] = pl.min(192 - stage_20_v7, 128)
        stage_20_v9: pl.Scalar[pl.INDEX] = 768 + stage_20_v7
        stage_20_v10: pl.Scalar[pl.INDEX] = stage_20_v6 * 256
        stage_20_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_20_v9, 0], [stage_20_v8, 256])
        stage_20_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [2, 0, stage_20_v10])
        stage_20_v13 = pl.tensor.matmul(stage_20_v11, stage_20_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_20_v15, (stage_20_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_20_v13,)):
            stage_20_v16: pl.Scalar[pl.INDEX] = stage_20_v15 * 256
            stage_20_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_20_v9, stage_20_v16], [stage_20_v8, 256])
            stage_20_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [2, stage_20_v16, stage_20_v10])
            stage_20_v19 = pl.tensor.matmul_acc(stage_20_v21, stage_20_v17, stage_20_v18, a_trans=False, b_trans=False)
            stage_20_v20 = pl.yield_(stage_20_v19)
        stage_20_v14 = pl.tensor.assemble(temporary_63, stage_20_v20, [stage_20_v7, 2048 + stage_20_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_20 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_20, pl.INT32))
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
            flag_21 = other_21 * 128
            progress = csa_signal_wait(progress, flag_21, pl.cast(target_21, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_21_v4: pl.Scalar[pl.INDEX] = block_21
        stage_21_v5: pl.Scalar[pl.INDEX] = stage_21_v4 * 512
        for (stage_21_v6, (stage_21_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_21_v7: pl.Scalar[pl.INDEX] = stage_21_v5 + stage_21_v6 * 256
            stage_21_v8: pl.Scalar[pl.INDEX] = pl.max(6, 0) * 1024
            stage_21_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_21_v8, stage_21_v7])
            for (stage_21_v12, (stage_21_v18,)) in pl.range(2, init_values=(stage_21_v11,)):
                stage_21_v13: pl.Scalar[pl.INDEX] = stage_21_v12 * 128
                stage_21_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_21_v13, 6144])
                stage_21_v15 = pl.tensor.matmul(stage_21_v14, stage_21_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_21_v16 = pl.tensor.assemble(stage_21_v18, stage_21_v15, [stage_21_v13, 6 * 4096 + stage_21_v7])
                stage_21_v17 = pl.yield_(stage_21_v16)
            stage_21_v10 = pl.yield_(stage_21_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_21 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_21, pl.INT32))
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
            flag_22 = other_22 * 128
            progress = csa_signal_wait(progress, flag_22, pl.cast(target_22, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_22_v4: pl.Scalar[pl.INDEX] = block_22
        stage_22_v5: pl.Scalar[pl.INDEX] = stage_22_v4 * 512
        for (stage_22_v6, (stage_22_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_22_v7: pl.Scalar[pl.INDEX] = stage_22_v5 + stage_22_v6 * 256
            stage_22_v8: pl.Scalar[pl.INDEX] = pl.max(5, 0) * 1024
            stage_22_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_22_v8, stage_22_v7])
            for (stage_22_v12, (stage_22_v18,)) in pl.range(2, init_values=(stage_22_v11,)):
                stage_22_v13: pl.Scalar[pl.INDEX] = stage_22_v12 * 128
                stage_22_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_22_v13, 5120])
                stage_22_v15 = pl.tensor.matmul(stage_22_v14, stage_22_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_22_v16 = pl.tensor.assemble(stage_22_v18, stage_22_v15, [stage_22_v13, 5 * 4096 + stage_22_v7])
                stage_22_v17 = pl.yield_(stage_22_v16)
            stage_22_v10 = pl.yield_(stage_22_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_22 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_22, pl.INT32))
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
            flag_23 = other_23 * 128
            progress = csa_signal_wait(progress, flag_23, pl.cast(target_23, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_23_v4: pl.Scalar[pl.INDEX] = block_23
        stage_23_v5: pl.Scalar[pl.INDEX] = stage_23_v4 * 512
        for (stage_23_v6, (stage_23_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_23_v7: pl.Scalar[pl.INDEX] = stage_23_v5 + stage_23_v6 * 256
            stage_23_v8: pl.Scalar[pl.INDEX] = pl.max(2, 0) * 1024
            stage_23_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_23_v8, stage_23_v7])
            for (stage_23_v12, (stage_23_v18,)) in pl.range(2, init_values=(stage_23_v11,)):
                stage_23_v13: pl.Scalar[pl.INDEX] = stage_23_v12 * 128
                stage_23_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_23_v13, 2048])
                stage_23_v15 = pl.tensor.matmul(stage_23_v14, stage_23_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_23_v16 = pl.tensor.assemble(stage_23_v18, stage_23_v15, [stage_23_v13, 2 * 4096 + stage_23_v7])
                stage_23_v17 = pl.yield_(stage_23_v16)
            stage_23_v10 = pl.yield_(stage_23_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_23 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_23, pl.INT32))
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
            flag_24 = other_24 * 128
            progress = csa_signal_wait(progress, flag_24, pl.cast(target_24, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_24_v4: pl.Scalar[pl.INDEX] = block_24
        stage_24_v5: pl.Scalar[pl.INDEX] = stage_24_v4 * 512
        for (stage_24_v6, (stage_24_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_24_v7: pl.Scalar[pl.INDEX] = stage_24_v5 + stage_24_v6 * 256
            stage_24_v8: pl.Scalar[pl.INDEX] = pl.max(1, 0) * 1024
            stage_24_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_24_v8, stage_24_v7])
            for (stage_24_v12, (stage_24_v18,)) in pl.range(2, init_values=(stage_24_v11,)):
                stage_24_v13: pl.Scalar[pl.INDEX] = stage_24_v12 * 128
                stage_24_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_24_v13, 1024])
                stage_24_v15 = pl.tensor.matmul(stage_24_v14, stage_24_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_24_v16 = pl.tensor.assemble(stage_24_v18, stage_24_v15, [stage_24_v13, 1 * 4096 + stage_24_v7])
                stage_24_v17 = pl.yield_(stage_24_v16)
            stage_24_v10 = pl.yield_(stage_24_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_24 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_24, pl.INT32))
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
            flag_25 = other_25 * 128
            progress = csa_signal_wait(progress, flag_25, pl.cast(target_25, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_25_v4: pl.Scalar[pl.INDEX] = block_25
        stage_25_v5: pl.Scalar[pl.INDEX] = stage_25_v4 // 4
        stage_25_v6: pl.Scalar[pl.INDEX] = stage_25_v4 % 4
        stage_25_v7: pl.Scalar[pl.INDEX] = stage_25_v5 * 128
        stage_25_v8: pl.Scalar[pl.INDEX] = pl.min(192 - stage_25_v7, 128)
        stage_25_v9: pl.Scalar[pl.INDEX] = 2688 + stage_25_v7
        stage_25_v10: pl.Scalar[pl.INDEX] = stage_25_v6 * 256
        stage_25_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_25_v9, 0], [stage_25_v8, 256])
        stage_25_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [7, 0, stage_25_v10])
        stage_25_v13 = pl.tensor.matmul(stage_25_v11, stage_25_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_25_v15, (stage_25_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_25_v13,)):
            stage_25_v16: pl.Scalar[pl.INDEX] = stage_25_v15 * 256
            stage_25_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_25_v9, stage_25_v16], [stage_25_v8, 256])
            stage_25_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [7, stage_25_v16, stage_25_v10])
            stage_25_v19 = pl.tensor.matmul_acc(stage_25_v21, stage_25_v17, stage_25_v18, a_trans=False, b_trans=False)
            stage_25_v20 = pl.yield_(stage_25_v19)
        stage_25_v14 = pl.tensor.assemble(temporary_63, stage_25_v20, [stage_25_v7, 7168 + stage_25_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_25 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_25, pl.INT32))
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
            flag_26 = other_26 * 128
            progress = csa_signal_wait(progress, flag_26, pl.cast(target_26, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_26_v4: pl.Scalar[pl.INDEX] = block_26
        stage_26_v5: pl.Scalar[pl.INDEX] = stage_26_v4 // 4
        stage_26_v6: pl.Scalar[pl.INDEX] = stage_26_v4 % 4
        stage_26_v7: pl.Scalar[pl.INDEX] = stage_26_v5 * 128
        stage_26_v8: pl.Scalar[pl.INDEX] = pl.min(192 - stage_26_v7, 128)
        stage_26_v9: pl.Scalar[pl.INDEX] = 1536 + stage_26_v7
        stage_26_v10: pl.Scalar[pl.INDEX] = stage_26_v6 * 256
        stage_26_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_26_v9, 0], [stage_26_v8, 256])
        stage_26_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [4, 0, stage_26_v10])
        stage_26_v13 = pl.tensor.matmul(stage_26_v11, stage_26_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_26_v15, (stage_26_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_26_v13,)):
            stage_26_v16: pl.Scalar[pl.INDEX] = stage_26_v15 * 256
            stage_26_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_26_v9, stage_26_v16], [stage_26_v8, 256])
            stage_26_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [4, stage_26_v16, stage_26_v10])
            stage_26_v19 = pl.tensor.matmul_acc(stage_26_v21, stage_26_v17, stage_26_v18, a_trans=False, b_trans=False)
            stage_26_v20 = pl.yield_(stage_26_v19)
        stage_26_v14 = pl.tensor.assemble(temporary_63, stage_26_v20, [stage_26_v7, 4096 + stage_26_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_26 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_26, pl.INT32))
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
            flag_27 = other_27 * 128
            progress = csa_signal_wait(progress, flag_27, pl.cast(target_27, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_27_v4: pl.Scalar[pl.INDEX] = block_27
        stage_27_v5: pl.Scalar[pl.INDEX] = stage_27_v4 // 4
        stage_27_v6: pl.Scalar[pl.INDEX] = stage_27_v4 % 4
        stage_27_v7: pl.Scalar[pl.INDEX] = stage_27_v5 * 128
        stage_27_v8: pl.Scalar[pl.INDEX] = pl.min(192 - stage_27_v7, 128)
        stage_27_v9: pl.Scalar[pl.INDEX] = 1152 + stage_27_v7
        stage_27_v10: pl.Scalar[pl.INDEX] = stage_27_v6 * 256
        stage_27_v11 = pl.tensor.slice(temporary_51, [128, 256], [stage_27_v9, 0], [stage_27_v8, 256])
        stage_27_v12 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [3, 0, stage_27_v10])
        stage_27_v13 = pl.tensor.matmul(stage_27_v11, stage_27_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (stage_27_v15, (stage_27_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_27_v13,)):
            stage_27_v16: pl.Scalar[pl.INDEX] = stage_27_v15 * 256
            stage_27_v17 = pl.tensor.slice(temporary_51, [128, 256], [stage_27_v9, stage_27_v16], [stage_27_v8, 256])
            stage_27_v18 = pl.tensor.slice(arg_wo_a, [1, 256, 256], [3, stage_27_v16, stage_27_v10])
            stage_27_v19 = pl.tensor.matmul_acc(stage_27_v21, stage_27_v17, stage_27_v18, a_trans=False, b_trans=False)
            stage_27_v20 = pl.yield_(stage_27_v19)
        stage_27_v14 = pl.tensor.assemble(temporary_63, stage_27_v20, [stage_27_v7, 3072 + stage_27_v10])
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_27 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_27, pl.INT32))
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
            flag_28 = other_28 * 128
            progress = csa_signal_wait(progress, flag_28, pl.cast(target_28, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
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
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_28, pl.INT32))
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
            flag_29 = other_29 * 128
            progress = csa_signal_wait(progress, flag_29, pl.cast(target_29, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_29_v4: pl.Scalar[pl.INDEX] = block_29
        stage_29_v5: pl.Scalar[pl.INDEX] = stage_29_v4 * 512
        for (stage_29_v6, (stage_29_v11,)) in pl.range(2, init_values=(temporary_66,)):
            stage_29_v7: pl.Scalar[pl.INDEX] = stage_29_v5 + stage_29_v6 * 256
            stage_29_v8: pl.Scalar[pl.INDEX] = pl.max(3, 0) * 1024
            stage_29_v9 = pl.tensor.slice(arg_wo_b, [1024, 256], [stage_29_v8, stage_29_v7])
            for (stage_29_v12, (stage_29_v18,)) in pl.range(2, init_values=(stage_29_v11,)):
                stage_29_v13: pl.Scalar[pl.INDEX] = stage_29_v12 * 128
                stage_29_v14 = pl.tensor.slice(temporary_64, [128, 1024], [stage_29_v13, 3072])
                stage_29_v15 = pl.tensor.matmul(stage_29_v14, stage_29_v9, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                stage_29_v16 = pl.tensor.assemble(stage_29_v18, stage_29_v15, [stage_29_v13, 3 * 4096 + stage_29_v7])
                stage_29_v17 = pl.yield_(stage_29_v16)
            stage_29_v10 = pl.yield_(stage_29_v17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_29 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_29, pl.INT32))
    return (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_129, progress)

@pl.jit.incore
def csa_static_aiv(temporary_4: pl.InOut[pl.Tensor[[192, 1], pl.FP32]], temporary_5: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]], view_2: pl.InOut[pl.Tensor[[192, 16384], pl.FP32]], view_3: pl.Tensor[[192, 16384], pl.BF16], temporary_7: pl.InOut[pl.Tensor[[768, 32], pl.FP32]], arg_hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], temporary_8: pl.InOut[pl.Tensor[[192, 32], pl.FP32]], arg_hc_attn_scale: pl.Tensor[[3], pl.FP32], temporary_0: pl.InOut[pl.Tensor[[192, 4], pl.FP32]], temporary_9: pl.InOut[pl.Tensor[[8, 8], pl.FP32]], temporary_6: pl.InOut[pl.Tensor[[192, 8], pl.FP32]], arg_hc_attn_base: pl.Tensor[[24], pl.FP32], view_12: pl.Tensor[[1, 24], pl.FP32], temporary_1: pl.InOut[pl.Tensor[[192, 16], pl.FP32]], temporary_10: pl.InOut[pl.Tensor[[8, 32], pl.FP32]], temporary_11: pl.InOut[pl.Tensor[[192, 4096], pl.BF16]], temporary_2: pl.InOut[pl.Tensor[[192, 4096], pl.BF16]], temporary_12: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]], arg_attn_norm_w: pl.Tensor[[4096], pl.BF16], arg_cmp_seq_lens: pl.Tensor[[32], pl.INT32], arg_cmp_query_start_loc: pl.Tensor[[33], pl.INT32], temporary_14: pl.InOut[pl.Tensor[[32], pl.INT32]], arg_kv_seq_lens: pl.Tensor[[32], pl.INT32], arg_idx_query_start_loc: pl.Tensor[[33], pl.INT32], temporary_15: pl.InOut[pl.Tensor[[32], pl.INT32]], arg_freqs_sin: pl.Tensor[[192, 64], pl.FP32], temporary_13: pl.InOut[pl.Tensor[[192, 64], pl.FP32]], temporary_20: pl.InOut[pl.Tensor[[192, 64], pl.FP32]], temporary_21: pl.InOut[pl.Tensor[[192, 64], pl.INT32]], temporary_24: pl.InOut[pl.Tensor[[192, 1024], pl.FP32]], arg_wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ], view_31: pl.Tensor[[1, 1024], pl.BF16], temporary_23: pl.InOut[pl.Tensor[[512, 1], pl.FP32]], temporary_19: pl.InOut[pl.Tensor[[192, 1], pl.FP32]], temporary_22: pl.InOut[pl.Tensor[[512, 1024], pl.INT8]], temporary_18: pl.InOut[pl.Tensor[[192, 1024], pl.INT8]], temporary_25: pl.InOut[pl.Tensor[[256, 32768], pl.INT32]], arg_wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ], arg_freqs_cos: pl.Tensor[[192, 64], pl.FP32], arg_wq_b_scale: pl.Tensor[[32768], pl.FP32], view_40: pl.InOut[pl.Tensor[[192, 32768], pl.BF16]], temporary_26: pl.InOut[pl.Tensor[[192, 512], pl.FP32]], arg_wkv: pl.Tensor[[4096, 512], pl.BF16], temporary_17: pl.InOut[pl.Tensor[[192, 512], pl.BF16]], arg_gamma_ckv: pl.Tensor[[512], pl.BF16], view_45: pl.InOut[pl.Tensor[[9248, 512], pl.BF16]], arg_ori_slot_mapping: pl.Tensor[[192, 2], pl.INT32], temporary_29: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], temporary_30: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], arg_cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], arg_cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], temporary_28: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], arg_position_ids: pl.Tensor[[192], pl.INT64], arg_cmp_ape: pl.Tensor[[4, 1024], pl.FP32], arg_state_block_table: pl.Tensor[[32, 4100], pl.INT32], arg_compress_state: pl.InOut[pl.Tensor[[289, 8192], pl.FP32]], arg_state_slot_mapping: pl.Tensor[[192, 2], pl.INT32], arg_cmp_freqs_cos: pl.Tensor[[80, 64], pl.FP32], arg_cmp_freqs_sin: pl.Tensor[[80, 64], pl.FP32], temporary_31: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], view_60: pl.Tensor[[1, 512], pl.BF16], view_61: pl.InOut[pl.Tensor[[67616, 512], pl.BF16]], temporary_27: pl.InOut[pl.Tensor[[192, 512], pl.FP32]], arg_cmp_slot_mapping: pl.Tensor[[80, 2], pl.INT32], temporary_34: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], temporary_35: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], arg_inner_wkv: pl.Tensor[[256, 4096], pl.BF16], arg_inner_wgate: pl.Tensor[[256, 4096], pl.BF16], temporary_36: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_inner_ape: pl.Tensor[[4, 256], pl.FP32], arg_inner_state_block_table: pl.Tensor[[32, 4100], pl.INT32], arg_inner_compress_state: pl.InOut[pl.Tensor[[289, 1040], pl.FP32]], arg_inner_state_slot_mapping: pl.Tensor[[192, 2], pl.INT32], temporary_33: pl.InOut[pl.Tensor[[384, 128], pl.BF16]], arg_inner_freqs_cos: pl.Tensor[[80, 64], pl.FP32], arg_inner_freqs_sin: pl.Tensor[[80, 64], pl.FP32], view_76: pl.Tensor[[1, 128], pl.BF16], temporary_38: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_hadamard_idx: pl.Tensor[[128, 128], pl.BF16], temporary_37: pl.InOut[pl.Tensor[[384, 1], pl.FP32]], arg_idx_native_kv_cache: pl.InOut[pl.Tensor[[2113, 4160], pl.INT8]], temporary_32: pl.InOut[pl.Tensor[[192, 128], pl.FP32]], arg_idx_slot_mapping: pl.Tensor[[80, 2], pl.INT32], temporary_42: pl.InOut[pl.Tensor[[384, 8192], pl.INT32]], arg_idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], view_85: pl.InOut[pl.Tensor[[384, 8192], pl.BF16]], arg_idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], temporary_43: pl.InOut[pl.Tensor[[12288, 128], pl.FP32]], temporary_41: pl.InOut[pl.Tensor[[24576, 128], pl.BF16]], temporary_39: pl.InOut[pl.Tensor[[24576, 128], pl.INT8]], temporary_40: pl.InOut[pl.Tensor[[24576, 1], pl.FP32]], temporary_45: pl.InOut[pl.Tensor[[1536, 64], pl.FP32]], arg_weights_proj: pl.Tensor[[4096, 64], pl.BF16], temporary_44: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], view_94: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_48: pl.InOut[pl.Tensor[[3072, 128], pl.FP16]], temporary_50: pl.InOut[pl.Tensor[[256], pl.INT64]], arg_idx_block_table: pl.Tensor[[32, 66], pl.INT32], view_98: pl.InOut[pl.Tensor[[68672, 128], pl.INT8]], view_99: pl.InOut[pl.Tensor[[68672, 128], pl.INT8]], temporary_49: pl.InOut[pl.Tensor[[96, 768], pl.FP32]], temporary_47: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_46: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32]], arg_idx_topk_scores: pl.InOut[pl.Tensor[[192, 512], pl.FP32]], arg_idx_topk: pl.InOut[pl.Tensor[[192, 512], pl.INT32]], temporary_53: pl.InOut[pl.Tensor[[192, 512], pl.INT32]], temporary_52: pl.InOut[pl.Tensor[[192, 640], pl.FP32]], view_107: pl.Tensor[[192, 1], pl.INT64], temporary_54: pl.InOut[pl.Tensor[[192, 16], pl.INT32]], arg_ori_block_table: pl.Tensor[[32, 258], pl.INT32], temporary_55: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_62: pl.InOut[pl.Tensor[[256], pl.INT64]], view_112: pl.InOut[pl.Tensor[[12288, 512], pl.BF16]], temporary_56: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_57: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], temporary_58: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], temporary_59: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], view_117: pl.Tensor[[64, 1], pl.FP32], arg_cmp_block_table: pl.Tensor[[32, 66], pl.INT32], temporary_60: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_61: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], arg_wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ], temporary_63: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_65: pl.InOut[pl.Tensor[[8, 384], pl.FP32]], temporary_64: pl.InOut[pl.Tensor[[384, 8192], pl.INT8]], arg_wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ], temporary_66: pl.InOut[pl.Tensor[[384, 32768], pl.INT32]], arg_wo_b_scale: pl.Tensor[[4096], pl.FP32], view_129: pl.InOut[pl.Tensor[[192, 16384], pl.BF16]], progress: pl.InOut[pl.Tensor[[9216], pl.INT32]], records: pl.Tensor[[33696], pl.INT64], waits: pl.Tensor[[49300], pl.INT64]) -> tuple[pl.Tensor[[192, 1], pl.FP32], pl.Tensor[[8, 16384], pl.FP32], pl.Tensor[[192, 16384], pl.FP32], pl.Tensor[[768, 32], pl.FP32], pl.Tensor[[192, 32], pl.FP32], pl.Tensor[[192, 4], pl.FP32], pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[192, 8], pl.FP32], pl.Tensor[[192, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32], pl.Tensor[[192, 4096], pl.BF16], pl.Tensor[[192, 4096], pl.BF16], pl.Tensor[[8, 4096], pl.BF16], pl.Tensor[[32], pl.INT32], pl.Tensor[[32], pl.INT32], pl.Tensor[[192, 64], pl.FP32], pl.Tensor[[192, 64], pl.FP32], pl.Tensor[[192, 64], pl.INT32], pl.Tensor[[192, 1024], pl.FP32], pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[192, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[192, 1024], pl.INT8], pl.Tensor[[256, 32768], pl.INT32], pl.Tensor[[192, 32768], pl.BF16], pl.Tensor[[192, 512], pl.FP32], pl.Tensor[[192, 512], pl.BF16], pl.Tensor[[9248, 512], pl.BF16], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[289, 8192], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[67616, 512], pl.BF16], pl.Tensor[[192, 512], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[289, 1040], pl.FP32], pl.Tensor[[384, 128], pl.BF16], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[2113, 4160], pl.INT8], pl.Tensor[[192, 128], pl.FP32], pl.Tensor[[384, 8192], pl.INT32], pl.Tensor[[384, 8192], pl.BF16], pl.Tensor[[12288, 128], pl.FP32], pl.Tensor[[24576, 128], pl.BF16], pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Tensor[[1536, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[3072, 128], pl.FP16], pl.Tensor[[256], pl.INT64], pl.Tensor[[68672, 128], pl.INT8], pl.Tensor[[68672, 128], pl.INT8], pl.Tensor[[96, 768], pl.FP32], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Tensor[[192, 512], pl.FP32], pl.Tensor[[192, 512], pl.INT32], pl.Tensor[[192, 512], pl.INT32], pl.Tensor[[192, 640], pl.FP32], pl.Tensor[[192, 16], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[256], pl.INT64], pl.Tensor[[12288, 512], pl.BF16], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8], pl.Tensor[[384, 32768], pl.INT32], pl.Tensor[[192, 16384], pl.BF16], pl.Tensor[[9216], pl.INT32]]:
    pl.func_attr({'mx_tensor_views_blocked': True, 'split_aiv': True})
    group_id = pl.tile.get_block_idx()
    lane_id = pl.tile.get_subblock_idx()
    core_id = 24 + group_id * 2 + lane_id
    own_flag = core_id * 128
    pl.system.cacheinvalid(progress, [1], [own_flag])
    pl.system.fence()
    prior_progress = pl.tensor.read(progress, [own_flag])
    generation_base = (prior_progress // 1024 + 1) * 1024
    progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base, pl.INT32))
    record_0 = 12960 + core_id * 6
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
            flag_0 = other_0 * 128
            progress = csa_signal_wait(progress, flag_0, pl.cast(target_0, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_0_v5: pl.Scalar[pl.INDEX] = block_0
        for (stage_0_v6, (stage_0_v16, stage_0_v17, stage_0_v18)) in pl.range(stage_0_v5, 24, pl.min(24, 48), init_values=(temporary_4, temporary_5, view_2)):
            stage_0_v7: pl.Scalar[pl.INDEX] = stage_0_v6 * 8
            stage_0_v8: pl.Scalar[pl.INDEX] = pl.min(192 - stage_0_v7, 8)
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
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_0, pl.INT32))
    record_1 = 13392 + core_id * 6
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
            flag_1 = other_1 * 128
            progress = csa_signal_wait(progress, flag_1, pl.cast(target_1, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_1_v4: pl.Scalar[pl.INDEX] = block_1
        stage_1_v5: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        stage_1_v6: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_1_v7: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_1_v6, target_type=pl.FP32, mode='round')
        stage_1_v8: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_1_v5, stage_1_v7)
        stage_1_v9: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_1_v8, 0.5)
        stage_1_v10: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_1_v9, target_type=pl.INT32, mode='trunc')
        stage_1_v11: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_1_v10, target_type=pl.FP32, mode='round')
        stage_1_v12: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_1_v11, 2.0)
        stage_1_v13: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_1_v8, stage_1_v12)
        stage_1_v14: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_1_v8, 1.0)
        stage_1_v15: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_1_v13, 2.0)
        stage_1_v16: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_1_v14, stage_1_v15)
        stage_1_v17: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_1_v16, target_type=pl.INT32, mode='round')
        stage_1_v18: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_1_v13, 2.0)
        stage_1_v19: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_1_v18, 1.0)
        for (stage_1_v20, (stage_1_v23, stage_1_v24)) in pl.range(stage_1_v4, 24, pl.min(24, 48), init_values=(temporary_20, temporary_21)):
            stage_1_v21: pl.Scalar[pl.INDEX] = stage_1_v20 * 8
            stage_1_v22: pl.Scalar[pl.INDEX] = pl.min(192 - stage_1_v21, 8)
            if stage_1_v22 == 8:
                stage_1_v25: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_sin, [stage_1_v21, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_1_v26: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_1_v25, stage_1_v19)
                stage_1_v27 = pl.tile.store(stage_1_v26, [stage_1_v21, 0], stage_1_v23)
                stage_1_v28 = pl.tile.store(stage_1_v17, [stage_1_v21, 0], stage_1_v24)
                (stage_1_v51, stage_1_v52) = pl.yield_(stage_1_v27, stage_1_v28)
            else:
                stage_1_v29: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_1_v22, 64])] = pl.tile.load(arg_freqs_sin, [stage_1_v21, 0], [8, 64], [stage_1_v22, 64], target_memory=pl.Mem.Vec)
                stage_1_v30: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                stage_1_v31: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                stage_1_v32: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_1_v31, target_type=pl.FP32, mode='round')
                stage_1_v33: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_1_v30, stage_1_v32)
                stage_1_v34: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_1_v33, 0.5)
                stage_1_v35: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_1_v34, target_type=pl.INT32, mode='trunc')
                stage_1_v36: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_1_v35, target_type=pl.FP32, mode='round')
                stage_1_v37: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_1_v36, 2.0)
                stage_1_v38: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_1_v33, stage_1_v37)
                stage_1_v39: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_1_v33, 1.0)
                stage_1_v40: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_1_v38, 2.0)
                stage_1_v41: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_1_v39, stage_1_v40)
                stage_1_v42: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_1_v38, 2.0)
                stage_1_v43: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_1_v42, 1.0)
                stage_1_v44: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_1_v22, 64])] = pl.tile.set_validshape(stage_1_v43, stage_1_v22, 64)
                stage_1_v45: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_1_v22, 64])] = pl.tile.mul(stage_1_v29, stage_1_v44)
                stage_1_v46: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_1_v22, 64])] = pl.tile.set_validshape(stage_1_v45, stage_1_v22, 64)
                pl.tile.store(stage_1_v46, [stage_1_v21, 0], stage_1_v23)
                stage_1_v47: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_1_v41, target_type=pl.INT32, mode='round')
                stage_1_v48: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_1_v22, 64])] = pl.tile.set_validshape(stage_1_v47, stage_1_v22, 64)
                pl.tile.store(stage_1_v48, [stage_1_v21, 0], stage_1_v24)
                (stage_1_v51, stage_1_v52) = pl.yield_(stage_1_v23, stage_1_v24)
            (stage_1_v49, stage_1_v50) = pl.yield_(stage_1_v51, stage_1_v52)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_1 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_1, pl.INT32))
    record_2 = 13824 + core_id * 6
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
            flag_2 = other_2 * 128
            progress = csa_signal_wait(progress, flag_2, pl.cast(target_2, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_2_v6: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        stage_2_v7: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_cmp_seq_lens, 0)
        for (stage_2_v10, (stage_2_v24,)) in pl.range(stage_2_v7, init_values=(stage_2_v6,)):
            stage_2_v11: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_query_start_loc, [stage_2_v10])
            stage_2_v12: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_query_start_loc, [stage_2_v10 + 1])
            stage_2_v13: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_seq_lens, [stage_2_v10])
            stage_2_v14: pl.Scalar[pl.INT32] = stage_2_v13 + stage_2_v11 - stage_2_v12
            pl.tensor.write(temporary_14, [stage_2_v10], pl.cast(stage_2_v24 - pl.cast(stage_2_v14, pl.INDEX) // 4 - 1, pl.INT32))
            stage_2_v15: pl.Scalar[pl.INDEX] = stage_2_v24 + pl.cast(stage_2_v13, pl.INDEX) // 4 - pl.cast(stage_2_v14, pl.INDEX) // 4
            stage_2_v16: pl.Scalar[pl.INDEX] = pl.yield_(stage_2_v15)
        stage_2_v8: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        stage_2_v9: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_kv_seq_lens, 0)
        for (stage_2_v17, (stage_2_v25,)) in pl.range(stage_2_v9, init_values=(stage_2_v8,)):
            stage_2_v18: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_query_start_loc, [stage_2_v17])
            stage_2_v19: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_query_start_loc, [stage_2_v17 + 1])
            stage_2_v20: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_2_v17])
            stage_2_v21: pl.Scalar[pl.INT32] = stage_2_v20 + stage_2_v18 - stage_2_v19
            pl.tensor.write(temporary_15, [stage_2_v17], pl.cast(stage_2_v25 - pl.cast(stage_2_v21, pl.INDEX) // 4 - 1, pl.INT32))
            stage_2_v22: pl.Scalar[pl.INDEX] = stage_2_v25 + pl.cast(stage_2_v20, pl.INDEX) // 4 - pl.cast(stage_2_v21, pl.INDEX) // 4
            stage_2_v23: pl.Scalar[pl.INDEX] = pl.yield_(stage_2_v22)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_2 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_2, pl.INT32))
    record_3 = 14256 + core_id * 6
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
            flag_3 = other_3 * 128
            progress = csa_signal_wait(progress, flag_3, pl.cast(target_3, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_3_v3: pl.Scalar[pl.INDEX] = block_3
        for stage_3_v4 in pl.range(stage_3_v3, 32, pl.min(32, 16)):
            stage_3_v5: pl.Scalar[pl.INDEX] = stage_3_v4 * 6
            stage_3_v6: pl.Scalar[pl.INDEX] = pl.min(192 - stage_3_v5, 6)
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
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_3, pl.INT32))
    record_4 = 14688 + core_id * 6
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
            flag_4 = other_4 * 128
            progress = csa_signal_wait(progress, flag_4, pl.cast(target_4, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_4_v4: pl.Scalar[pl.INDEX] = block_4
        stage_4_v5: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        stage_4_v6: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_4_v7: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_4_v6, target_type=pl.FP32, mode='round')
        stage_4_v8: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_4_v5, stage_4_v7)
        stage_4_v9: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_4_v8, 0.5)
        stage_4_v10: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_4_v9, target_type=pl.INT32, mode='trunc')
        stage_4_v11: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_4_v10, target_type=pl.FP32, mode='round')
        stage_4_v12: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_4_v11, 2.0)
        stage_4_v13: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_4_v8, stage_4_v12)
        stage_4_v14: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_4_v8, 1.0)
        stage_4_v15: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_4_v13, 2.0)
        stage_4_v16: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_4_v14, stage_4_v15)
        stage_4_v17: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_4_v16, target_type=pl.INT32, mode='round')
        stage_4_v18: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_4_v13, 2.0)
        stage_4_v19: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_4_v18, 1.0)
        for (stage_4_v20, (stage_4_v23, stage_4_v24)) in pl.range(stage_4_v4, 24, pl.min(24, 48), init_values=(temporary_20, temporary_21)):
            stage_4_v21: pl.Scalar[pl.INDEX] = stage_4_v20 * 8
            stage_4_v22: pl.Scalar[pl.INDEX] = pl.min(192 - stage_4_v21, 8)
            if stage_4_v22 == 8:
                stage_4_v25: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_sin, [stage_4_v21, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_4_v26: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_4_v25, stage_4_v19)
                stage_4_v27 = pl.tile.store(stage_4_v26, [stage_4_v21, 0], stage_4_v23)
                stage_4_v28 = pl.tile.store(stage_4_v17, [stage_4_v21, 0], stage_4_v24)
                (stage_4_v51, stage_4_v52) = pl.yield_(stage_4_v27, stage_4_v28)
            else:
                stage_4_v29: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_4_v22, 64])] = pl.tile.load(arg_freqs_sin, [stage_4_v21, 0], [8, 64], [stage_4_v22, 64], target_memory=pl.Mem.Vec)
                stage_4_v30: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                stage_4_v31: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                stage_4_v32: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_4_v31, target_type=pl.FP32, mode='round')
                stage_4_v33: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_4_v30, stage_4_v32)
                stage_4_v34: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_4_v33, 0.5)
                stage_4_v35: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_4_v34, target_type=pl.INT32, mode='trunc')
                stage_4_v36: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_4_v35, target_type=pl.FP32, mode='round')
                stage_4_v37: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_4_v36, 2.0)
                stage_4_v38: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_4_v33, stage_4_v37)
                stage_4_v39: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_4_v33, 1.0)
                stage_4_v40: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_4_v38, 2.0)
                stage_4_v41: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_4_v39, stage_4_v40)
                stage_4_v42: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_4_v38, 2.0)
                stage_4_v43: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_4_v42, 1.0)
                stage_4_v44: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_4_v22, 64])] = pl.tile.set_validshape(stage_4_v43, stage_4_v22, 64)
                stage_4_v45: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_4_v22, 64])] = pl.tile.mul(stage_4_v29, stage_4_v44)
                stage_4_v46: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_4_v22, 64])] = pl.tile.set_validshape(stage_4_v45, stage_4_v22, 64)
                pl.tile.store(stage_4_v46, [stage_4_v21, 0], stage_4_v23)
                stage_4_v47: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_4_v41, target_type=pl.INT32, mode='round')
                stage_4_v48: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_4_v22, 64])] = pl.tile.set_validshape(stage_4_v47, stage_4_v22, 64)
                pl.tile.store(stage_4_v48, [stage_4_v21, 0], stage_4_v24)
                (stage_4_v51, stage_4_v52) = pl.yield_(stage_4_v23, stage_4_v24)
            (stage_4_v49, stage_4_v50) = pl.yield_(stage_4_v51, stage_4_v52)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_4 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_4, pl.INT32))
    record_5 = 15120 + core_id * 6
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
            flag_5 = other_5 * 128
            progress = csa_signal_wait(progress, flag_5, pl.cast(target_5, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_5_v3: pl.Scalar[pl.INDEX] = block_5
        for stage_5_v4 in pl.range(stage_5_v3, 48, pl.min(48, 16)):
            stage_5_v5: pl.Scalar[pl.INDEX] = stage_5_v4 * 4
            stage_5_v6: pl.Scalar[pl.INDEX] = pl.min(192 - stage_5_v5, 4)
            stage_5_v7: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([4, 64], dtype=pl.FP32, value=1.0)
            stage_5_v8: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            stage_5_v9: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_5_v8, target_type=pl.FP32, mode='round')
            stage_5_v10: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_5_v7, stage_5_v9)
            stage_5_v11: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_5_v10, 0.5)
            stage_5_v12: pl.Tile[[4, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_5_v11, target_type=pl.INT32, mode='trunc')
            stage_5_v13: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_5_v12, target_type=pl.FP32, mode='round')
            stage_5_v14: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_5_v13, 2.0)
            stage_5_v15: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_5_v10, stage_5_v14)
            stage_5_v16: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_5_v15, 2.0)
            stage_5_v17: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_5_v16, 1.0)
            stage_5_v18: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_5_v6, 64])] = pl.tile.load(arg_freqs_sin, [stage_5_v5, 0], [4, 64], [stage_5_v6, 64], target_memory=pl.Mem.Vec)
            stage_5_v19: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_5_v6, 64])] = pl.tile.set_validshape(stage_5_v17, stage_5_v6, 64)
            stage_5_v20: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_5_v6, 64])] = pl.tile.mul(stage_5_v18, stage_5_v19)
            stage_5_v21 = pl.tile.store(stage_5_v20, [stage_5_v5, 0], temporary_13)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_5 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_5, pl.INT32))
    record_6 = 15552 + core_id * 6
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
            flag_6 = other_6 * 128
            progress = csa_signal_wait(progress, flag_6, pl.cast(target_6, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_6_v3: pl.Scalar[pl.INDEX] = block_6
        for stage_6_v4 in pl.range(stage_6_v3, 32, pl.min(32, 16)):
            stage_6_v5: pl.Scalar[pl.INDEX] = stage_6_v4 * 6
            stage_6_v6: pl.Scalar[pl.INDEX] = pl.min(192 - stage_6_v5, 6)
            stage_6_v7: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([6, 64], dtype=pl.FP32, value=1.0)
            stage_6_v8: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            stage_6_v9: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_6_v8, target_type=pl.FP32, mode='round')
            stage_6_v10: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_6_v7, stage_6_v9)
            stage_6_v11: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_6_v10, 0.5)
            stage_6_v12: pl.Tile[[6, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_6_v11, target_type=pl.INT32, mode='trunc')
            stage_6_v13: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_6_v12, target_type=pl.FP32, mode='round')
            stage_6_v14: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_6_v13, 2.0)
            stage_6_v15: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_6_v10, stage_6_v14)
            stage_6_v16: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_6_v15, 2.0)
            stage_6_v17: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_6_v16, 1.0)
            stage_6_v18: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_6_v17)
            stage_6_v19: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v6, 64])] = pl.tile.load(arg_freqs_sin, [stage_6_v5, 0], [6, 64], [stage_6_v6, 64], target_memory=pl.Mem.Vec)
            stage_6_v20: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v6, 64])] = pl.tile.set_validshape(stage_6_v18, stage_6_v6, 64)
            stage_6_v21: pl.Tile[[6, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_6_v6, 64])] = pl.tile.mul(stage_6_v19, stage_6_v20)
            stage_6_v22 = pl.tile.store(stage_6_v21, [stage_6_v5, 0], temporary_55)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_6 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_6, pl.INT32))
    record_7 = 15984 + core_id * 6
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
            flag_7 = other_7 * 128
            progress = csa_signal_wait(progress, flag_7, pl.cast(target_7, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_7_v3: pl.Scalar[pl.INDEX] = block_7
        stage_7_v4: pl.Scalar[pl.INDEX] = stage_7_v3 * 16
        stage_7_v5: pl.Tile[[16, 32], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_7, [stage_7_v4, 0], [16, 32], [16, 32], target_memory=pl.Mem.Vec)
        for (stage_7_v7, (stage_7_v12,)) in pl.range(1, 4, init_values=(stage_7_v5,)):
            stage_7_v8: pl.Scalar[pl.INDEX] = stage_7_v7 * 192 + stage_7_v4
            stage_7_v9: pl.Tile[[16, 32], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_7, [stage_7_v8, 0], [16, 32], [16, 32], target_memory=pl.Mem.Vec)
            stage_7_v10: pl.Tile[[16, 32], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_7_v12, stage_7_v9)
            stage_7_v11: pl.Tile[[16, 32], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_7_v10)
        stage_7_v6 = pl.tile.store(stage_7_v11, [stage_7_v4, 0], temporary_8)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_7 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_7, pl.INT32))
    record_8 = 16416 + core_id * 6
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
            flag_8 = other_8 * 128
            progress = csa_signal_wait(progress, flag_8, pl.cast(target_8, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_8_v7: pl.Scalar[pl.INDEX] = block_8
        stage_8_v8: pl.Scalar[pl.FP32] = pl.tensor.read(arg_hc_attn_scale, [2])
        stage_8_v9: pl.Scalar[pl.INDEX] = stage_8_v7 * 8
        stage_8_v10: pl.Scalar[pl.INDEX] = pl.min(192 - stage_8_v9, 8)
        stage_8_v11: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 1])] = pl.tile.load(temporary_4, [stage_8_v9, 0], [8, 1], [stage_8_v10, 1], target_memory=pl.Mem.Vec)
        stage_8_v12: pl.Scalar[pl.INDEX] = 8
        stage_8_v13: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.load(temporary_8, [stage_8_v9, stage_8_v12], [8, 8], [stage_8_v10, 4], target_memory=pl.Mem.Vec)
        stage_8_v14: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.load(temporary_8, [stage_8_v9, stage_8_v12 + 4], [8, 8], [stage_8_v10, 4], target_memory=pl.Mem.Vec)
        stage_8_v15: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.load(temporary_8, [stage_8_v9, stage_8_v12 + 8], [8, 8], [stage_8_v10, 4], target_memory=pl.Mem.Vec)
        stage_8_v16: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.load(temporary_8, [stage_8_v9, stage_8_v12 + 12], [8, 8], [stage_8_v10, 4], target_memory=pl.Mem.Vec)
        stage_8_v17: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(view_12, [0, stage_8_v12], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        stage_8_v18: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(view_12, [0, stage_8_v12 + 4], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        stage_8_v19: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(view_12, [0, stage_8_v12 + 8], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        stage_8_v20: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(view_12, [0, stage_8_v12 + 12], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        stage_8_v21: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.row_expand_mul(stage_8_v13, stage_8_v11)
        stage_8_v22: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.muls(stage_8_v21, stage_8_v8)
        stage_8_v23: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.col_expand(stage_8_v13, stage_8_v17)
        stage_8_v24: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.add(stage_8_v22, stage_8_v23)
        stage_8_v25: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.row_expand_mul(stage_8_v14, stage_8_v11)
        stage_8_v26: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.muls(stage_8_v25, stage_8_v8)
        stage_8_v27: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.col_expand(stage_8_v14, stage_8_v18)
        stage_8_v28: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.add(stage_8_v26, stage_8_v27)
        stage_8_v29: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.row_expand_mul(stage_8_v15, stage_8_v11)
        stage_8_v30: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.muls(stage_8_v29, stage_8_v8)
        stage_8_v31: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.col_expand(stage_8_v15, stage_8_v19)
        stage_8_v32: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.add(stage_8_v30, stage_8_v31)
        stage_8_v33: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.row_expand_mul(stage_8_v16, stage_8_v11)
        stage_8_v34: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.muls(stage_8_v33, stage_8_v8)
        stage_8_v35: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.col_expand(stage_8_v16, stage_8_v20)
        stage_8_v36: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.add(stage_8_v34, stage_8_v35)
        stage_8_v37: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_8_v24, pad_value=pl.PadValue.min)
        stage_8_v38: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_8_v28, pad_value=pl.PadValue.min)
        stage_8_v39: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_8_v32, pad_value=pl.PadValue.min)
        stage_8_v40: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_8_v36, pad_value=pl.PadValue.min)
        stage_8_v41: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_8_v42: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_8_v43: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_8_v37, stage_8_v41)
        stage_8_v44: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_8_v38, stage_8_v41)
        stage_8_v45: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_8_v39, stage_8_v41)
        stage_8_v46: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_8_v40, stage_8_v41)
        stage_8_v47: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(stage_8_v37, stage_8_v43)
        stage_8_v48: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(stage_8_v47)
        stage_8_v49: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(stage_8_v38, stage_8_v44)
        stage_8_v50: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(stage_8_v49)
        stage_8_v51: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(stage_8_v39, stage_8_v45)
        stage_8_v52: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(stage_8_v51)
        stage_8_v53: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(stage_8_v40, stage_8_v46)
        stage_8_v54: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(stage_8_v53)
        stage_8_v55: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_8_v48, stage_8_v42)
        stage_8_v56: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_8_v50, stage_8_v42)
        stage_8_v57: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_8_v52, stage_8_v42)
        stage_8_v58: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_8_v54, stage_8_v42)
        stage_8_v59: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(stage_8_v48, stage_8_v55)
        stage_8_v60: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(stage_8_v59, 1e-06)
        stage_8_v61: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(stage_8_v50, stage_8_v56)
        stage_8_v62: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(stage_8_v61, 1e-06)
        stage_8_v63: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(stage_8_v52, stage_8_v57)
        stage_8_v64: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(stage_8_v63, 1e-06)
        stage_8_v65: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(stage_8_v54, stage_8_v58)
        stage_8_v66: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(stage_8_v65, 1e-06)
        stage_8_v67: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_8_v60, 8, 4)
        stage_8_v68: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_8_v62, 8, 4)
        stage_8_v69: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_8_v64, 8, 4)
        stage_8_v70: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_8_v66, 8, 4)
        stage_8_v71: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_8_v67, pad_value=pl.PadValue.zero)
        stage_8_v72: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_8_v68, pad_value=pl.PadValue.zero)
        stage_8_v73: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_8_v69, pad_value=pl.PadValue.zero)
        stage_8_v74: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(stage_8_v70, pad_value=pl.PadValue.zero)
        stage_8_v75: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_8_v76: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_8_v71, stage_8_v72)
        stage_8_v77: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_8_v73, stage_8_v74)
        stage_8_v78: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_8_v76, stage_8_v77)
        stage_8_v79: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(stage_8_v78, 1e-06)
        stage_8_v80: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_8_v71, stage_8_v79)
        stage_8_v81: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_8_v72, stage_8_v79)
        stage_8_v82: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_8_v73, stage_8_v79)
        stage_8_v83: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_8_v74, stage_8_v79)
        for (stage_8_v84, (stage_8_v133, stage_8_v134, stage_8_v135, stage_8_v136)) in pl.pipeline(19, stage=2, init_values=(stage_8_v80, stage_8_v81, stage_8_v82, stage_8_v83)):
            stage_8_v85: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_8_v133, stage_8_v75)
            stage_8_v86: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_8_v85, [1, 8])
            stage_8_v87: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_8_v86, 1e-06)
            stage_8_v88: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_8_v87, [8, 1])
            stage_8_v89: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_8_v134, stage_8_v75)
            stage_8_v90: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_8_v89, [1, 8])
            stage_8_v91: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_8_v90, 1e-06)
            stage_8_v92: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_8_v91, [8, 1])
            stage_8_v93: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_8_v135, stage_8_v75)
            stage_8_v94: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_8_v93, [1, 8])
            stage_8_v95: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_8_v94, 1e-06)
            stage_8_v96: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_8_v95, [8, 1])
            stage_8_v97: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_8_v136, stage_8_v75)
            stage_8_v98: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_8_v97, [1, 8])
            stage_8_v99: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_8_v98, 1e-06)
            stage_8_v100: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_8_v99, [8, 1])
            stage_8_v101: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(stage_8_v133, stage_8_v88)
            stage_8_v102: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(stage_8_v134, stage_8_v92)
            stage_8_v103: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(stage_8_v135, stage_8_v96)
            stage_8_v104: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(stage_8_v136, stage_8_v100)
            stage_8_v105: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_8_v101, stage_8_v102)
            stage_8_v106: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_8_v103, stage_8_v104)
            stage_8_v107: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(stage_8_v105, stage_8_v106)
            stage_8_v108: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(stage_8_v107, 1e-06)
            stage_8_v109: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_8_v101, stage_8_v108)
            stage_8_v110: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_8_v102, stage_8_v108)
            stage_8_v111: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_8_v103, stage_8_v108)
            stage_8_v112: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(stage_8_v104, stage_8_v108)
            (stage_8_v137, stage_8_v138, stage_8_v139, stage_8_v140) = pl.yield_(stage_8_v109, stage_8_v110, stage_8_v111, stage_8_v112)
        if stage_8_v10 == 8:
            stage_8_v113: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(stage_8_v137, 8, 4)
            stage_8_v114: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(stage_8_v138, 8, 4)
            stage_8_v115: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(stage_8_v139, 8, 4)
            stage_8_v116: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(stage_8_v140, 8, 4)
            stage_8_v117 = pl.tile.store(stage_8_v113, [stage_8_v9, 0], temporary_1)
            stage_8_v118 = pl.tile.store(stage_8_v114, [stage_8_v9, 4], temporary_1)
            stage_8_v119 = pl.tile.store(stage_8_v115, [stage_8_v9, 8], temporary_1)
            stage_8_v120 = pl.tile.store(stage_8_v116, [stage_8_v9, 12], temporary_1)
        else:
            stage_8_v121 = pl.tile.store(stage_8_v137, [0, 0], temporary_10)
            stage_8_v122 = pl.tile.store(stage_8_v138, [0, 8], temporary_10)
            stage_8_v123 = pl.tile.store(stage_8_v139, [0, 16], temporary_10)
            stage_8_v124 = pl.tile.store(stage_8_v140, [0, 24], temporary_10)
            stage_8_v125: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.load(temporary_10, [0, 0], [8, 8], [stage_8_v10, 4], target_memory=pl.Mem.Vec)
            stage_8_v126: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.load(temporary_10, [0, 8], [8, 8], [stage_8_v10, 4], target_memory=pl.Mem.Vec)
            stage_8_v127: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.load(temporary_10, [0, 16], [8, 8], [stage_8_v10, 4], target_memory=pl.Mem.Vec)
            stage_8_v128: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_8_v10, 4])] = pl.tile.load(temporary_10, [0, 24], [8, 8], [stage_8_v10, 4], target_memory=pl.Mem.Vec)
            stage_8_v129 = pl.tile.store(stage_8_v125, [stage_8_v9, 0], temporary_1)
            stage_8_v130 = pl.tile.store(stage_8_v126, [stage_8_v9, 4], temporary_1)
            stage_8_v131 = pl.tile.store(stage_8_v127, [stage_8_v9, 8], temporary_1)
            stage_8_v132 = pl.tile.store(stage_8_v128, [stage_8_v9, 12], temporary_1)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_8 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_8, pl.INT32))
    record_9 = 16848 + core_id * 6
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
            flag_9 = other_9 * 128
            progress = csa_signal_wait(progress, flag_9, pl.cast(target_9, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_9_v8: pl.Scalar[pl.INDEX] = block_9
        stage_9_v9: pl.Scalar[pl.FP32] = pl.tensor.read(arg_hc_attn_scale, [0])
        stage_9_v10: pl.Scalar[pl.FP32] = pl.tensor.read(arg_hc_attn_scale, [1])
        for (stage_9_v11, (stage_9_v40, stage_9_v41, stage_9_v42)) in pl.range(stage_9_v8, 24, pl.min(24, 16), init_values=(temporary_0, temporary_9, temporary_6)):
            stage_9_v12: pl.Scalar[pl.INDEX] = stage_9_v11 * 8
            stage_9_v13: pl.Scalar[pl.INDEX] = pl.min(192 - stage_9_v12, 8)
            stage_9_v14: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_4, [stage_9_v12, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
            stage_9_v15: pl.Tile[[8], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_hc_attn_base, [0], [8], [8], target_memory=pl.Mem.Vec)
            stage_9_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_9_v15, [1, 8])
            stage_9_v17: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_8, [stage_9_v12, 0], [8, 8], [8, 8], target_memory=pl.Mem.Vec)
            stage_9_v18: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_9_v17, stage_9_v14)
            stage_9_v19: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_9_v18, stage_9_v9)
            stage_9_v20: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand(stage_9_v19, stage_9_v16)
            stage_9_v21: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_9_v19, stage_9_v20)
            stage_9_v22: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_9_v21)
            stage_9_v23: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_9_v22)
            stage_9_v24: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_9_v23, 1.0)
            stage_9_v25: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_9_v24)
            stage_9_v26: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_9_v25, 1e-06)
            stage_9_v27 = pl.tile.store(stage_9_v26, [stage_9_v12, 0], stage_9_v42)
            stage_9_v28: pl.Tile[[8], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_hc_attn_base, [4], [8], [8], target_memory=pl.Mem.Vec)
            stage_9_v29: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_9_v28, [1, 8])
            stage_9_v30: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_8, [stage_9_v12, 4], [8, 8], [8, 8], target_memory=pl.Mem.Vec)
            stage_9_v31: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_9_v30, stage_9_v14)
            stage_9_v32: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_9_v31, stage_9_v10)
            stage_9_v33: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand(stage_9_v32, stage_9_v29)
            stage_9_v34: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_9_v32, stage_9_v33)
            stage_9_v35: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_9_v34)
            stage_9_v36: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_9_v35)
            stage_9_v37: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_9_v36, 1.0)
            stage_9_v38: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_9_v37)
            stage_9_v39: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_9_v38, 2.0)
            if stage_9_v13 == 8:
                stage_9_v43: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4])] = pl.tile.slice(stage_9_v39, [8, 8], [0, 0], [8, 4])
                stage_9_v44 = pl.tile.store(stage_9_v43, [stage_9_v12, 0], stage_9_v40)
                (stage_9_v50, stage_9_v51) = pl.yield_(stage_9_v44, stage_9_v41)
            else:
                stage_9_v45 = pl.tile.store(stage_9_v39, [0, 0], stage_9_v41)
                stage_9_v46: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_9_v13, 4])] = pl.tile.load(stage_9_v45, [0, 0], [8, 8], [stage_9_v13, 4], target_memory=pl.Mem.Vec)
                pl.tile.store(stage_9_v46, [stage_9_v12, 0], stage_9_v40)
                (stage_9_v50, stage_9_v51) = pl.yield_(stage_9_v40, stage_9_v45)
            (stage_9_v47, stage_9_v48, stage_9_v49) = pl.yield_(stage_9_v50, stage_9_v51, stage_9_v27)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_9 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_9, pl.INT32))
    record_10 = 17280 + core_id * 6
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
            flag_10 = other_10 * 128
            progress = csa_signal_wait(progress, flag_10, pl.cast(target_10, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_10_v7: pl.Scalar[pl.INDEX] = block_10
        stage_10_v8: pl.Scalar[pl.INDEX] = stage_10_v7 * 8
        stage_10_v9: pl.Scalar[pl.INDEX] = pl.min(192 - stage_10_v8, 8)
        stage_10_v10: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_6, [stage_10_v8, 0], [8, 8], [8, 8], target_memory=pl.Mem.Vec)
        stage_10_v11: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_10_v12: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(stage_10_v10, 0, 1, stage_10_v11)
        stage_10_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_10_v12, [1, 8], [0, 0])
        stage_10_v14: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_10_v13, [8, 1])
        stage_10_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_10_v12, [1, 8], [1, 0])
        stage_10_v16: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_10_v15, [8, 1])
        stage_10_v17: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_10_v12, [1, 8], [2, 0])
        stage_10_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_10_v17, [8, 1])
        stage_10_v19: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_10_v12, [1, 8], [3, 0])
        stage_10_v20: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_10_v19, [8, 1])
        stage_10_v21: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for (stage_10_v31, (stage_10_v64, stage_10_v65)) in pl.pipeline(16, stage=2, init_values=(stage_10_v21, temporary_11)):
            stage_10_v32: pl.Scalar[pl.INDEX] = stage_10_v31 * 256
            stage_10_v33: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.load(view_2, [stage_10_v8, stage_10_v32], [8, 256], [stage_10_v9, 256], target_memory=pl.Mem.Vec)
            stage_10_v34: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.load(view_2, [stage_10_v8, stage_10_v32 + 4096], [8, 256], [stage_10_v9, 256], target_memory=pl.Mem.Vec)
            stage_10_v35: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.load(view_2, [stage_10_v8, stage_10_v32 + 8192], [8, 256], [stage_10_v9, 256], target_memory=pl.Mem.Vec)
            stage_10_v36: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.load(view_2, [stage_10_v8, stage_10_v32 + 12288], [8, 256], [stage_10_v9, 256], target_memory=pl.Mem.Vec)
            stage_10_v37: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.row_expand_mul(stage_10_v33, stage_10_v14)
            stage_10_v38: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.row_expand_mul(stage_10_v34, stage_10_v16)
            stage_10_v39: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.row_expand_mul(stage_10_v35, stage_10_v18)
            stage_10_v40: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.row_expand_mul(stage_10_v36, stage_10_v20)
            stage_10_v41: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.add(stage_10_v37, stage_10_v38)
            stage_10_v42: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.add(stage_10_v39, stage_10_v40)
            stage_10_v43: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.add(stage_10_v41, stage_10_v42)
            stage_10_v44: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.cast(stage_10_v43, target_type=pl.BF16, mode='rint')
            stage_10_v45 = pl.tile.store(stage_10_v44, [stage_10_v8, stage_10_v32], stage_10_v65)
            stage_10_v46: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.cast(stage_10_v44, target_type=pl.FP32, mode='round')
            stage_10_v47: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.mul(stage_10_v46, stage_10_v46)
            stage_10_v48: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_10_v49: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 1])] = pl.tile.row_sum(stage_10_v47, stage_10_v48)
            stage_10_v50: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_10_v9])] = pl.tile.reshape(stage_10_v49, [1, 8])
            stage_10_v51: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_10_v64, stage_10_v50)
            (stage_10_v66, stage_10_v67) = pl.yield_(stage_10_v51, stage_10_v45)
        stage_10_v22: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_10_v23: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_10_v22, stage_10_v66, [0, 0])
        stage_10_v24: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_10_v25: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_10_v23, 0.000244140625)
        stage_10_v26: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_10_v25, 1e-06)
        stage_10_v27: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_10_v28: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(stage_10_v26, stage_10_v27)
        stage_10_v29: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = stage_10_v28
        stage_10_v30: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_10_v24, stage_10_v29, [0, 0])
        for (stage_10_v52, (stage_10_v68, stage_10_v69)) in pl.pipeline(16, stage=2, init_values=(temporary_2, temporary_12)):
            stage_10_v53: pl.Scalar[pl.INDEX] = stage_10_v52 * 256
            stage_10_v54: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec] = pl.tile.load(stage_10_v67, [stage_10_v8, stage_10_v53], [8, 256], [8, 256], target_memory=pl.Mem.Vec)
            stage_10_v55: pl.Tile[[256], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_attn_norm_w, [stage_10_v53], [256], [256], target_memory=pl.Mem.Vec)
            stage_10_v56: pl.Tile[[1, 256], pl.BF16, pl.Mem.Vec] = pl.tile.reshape(stage_10_v55, [1, 256])
            stage_10_v57: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_10_v54, target_type=pl.FP32, mode='round')
            stage_10_v58: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_10_v30, [8, 1])
            stage_10_v59: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_10_v56, target_type=pl.FP32, mode='round')
            stage_10_v60: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_10_v57, stage_10_v58)
            stage_10_v61: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_10_v60, stage_10_v59)
            stage_10_v62: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_10_v61, target_type=pl.BF16, mode='rint')
            stage_10_v63: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec] = stage_10_v62
            if stage_10_v9 == 8:
                stage_10_v70 = pl.tile.store(stage_10_v63, [stage_10_v8, stage_10_v53], stage_10_v68)
                (stage_10_v75, stage_10_v76) = pl.yield_(stage_10_v70, stage_10_v69)
            else:
                stage_10_v71 = pl.tile.store(stage_10_v63, [0, stage_10_v53], stage_10_v69)
                stage_10_v72: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_10_v9, 256])] = pl.tile.load(stage_10_v71, [0, stage_10_v53], [8, 256], [stage_10_v9, 256], target_memory=pl.Mem.Vec)
                pl.tile.store(stage_10_v72, [stage_10_v8, stage_10_v53], stage_10_v68)
                (stage_10_v75, stage_10_v76) = pl.yield_(stage_10_v68, stage_10_v71)
            (stage_10_v73, stage_10_v74) = pl.yield_(stage_10_v75, stage_10_v76)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_10 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_10, pl.INT32))
    record_11 = 17712 + core_id * 6
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
            flag_11 = other_11 * 128
            progress = csa_signal_wait(progress, flag_11, pl.cast(target_11, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_11_v7: pl.Scalar[pl.INDEX] = block_11
        stage_11_v8: pl.Tile[[1, 1024], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_31, [0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
        stage_11_v9: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_11_v8, target_type=pl.FP32, mode='round')
        for stage_11_v10 in pl.range(stage_11_v7, 24, 16):
            stage_11_v11: pl.Scalar[pl.INDEX] = stage_11_v10 * 8
            stage_11_v12: pl.Scalar[pl.INDEX] = pl.min(192 - stage_11_v11, 8)
            stage_11_v13: pl.Scalar[pl.INDEX] = 0 + stage_11_v11
            stage_11_v14: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_24, [stage_11_v11, 0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_11_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
            stage_11_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
            for (stage_11_v32, (stage_11_v58, stage_11_v59)) in pl.pipeline(0, 1024, 256, stage=2, init_values=(stage_11_v16, stage_11_v15)):
                stage_11_v33: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_11_v14, 0, stage_11_v32, [8, 256], target_memory=pl.Mem.Vec)
                stage_11_v34: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_11_v33, stage_11_v33)
                stage_11_v35: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_11_v36: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_11_v34, stage_11_v35)
                stage_11_v37: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_11_v36, [1, 8])
                stage_11_v38: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_11_v59, stage_11_v37)
                stage_11_v39: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_11_v9, 0, stage_11_v32, [1, 256], target_memory=pl.Mem.Vec)
                stage_11_v40: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_11_v33, stage_11_v39)
                stage_11_v41: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_11_v40)
                stage_11_v42: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_11_v43: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_11_v41, stage_11_v42)
                stage_11_v44: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_11_v43, [1, 8])
                stage_11_v45: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_11_v58, stage_11_v44)
                (stage_11_v60, stage_11_v61) = pl.yield_(stage_11_v45, stage_11_v38)
            stage_11_v17: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_11_v61, 0.0009765625)
            stage_11_v18: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_11_v17, 1e-06)
            stage_11_v19: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_11_v20: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(stage_11_v18, stage_11_v19)
            stage_11_v21: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_11_v20, [8, 1])
            stage_11_v22: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_11_v23: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_11_v20, stage_11_v60)
            stage_11_v24: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_11_v22, stage_11_v23)
            stage_11_v25: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_11_v26: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_11_v25, stage_11_v24)
            stage_11_v27: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_11_v26, [8, 1])
            stage_11_v28: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_11_v26)
            stage_11_v29: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_11_v28, [8, 1])
            stage_11_v30 = pl.tile.store(stage_11_v29, [stage_11_v11, 0], temporary_23)
            if stage_11_v12 == 8:
                stage_11_v46 = pl.tile.store(stage_11_v29, [stage_11_v13, 0], temporary_19)
            else:
                stage_11_v47: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v12, 1])] = pl.tile.slice(stage_11_v29, [8, 1], [0, 0], valid_shape=[stage_11_v12, 1])
                stage_11_v48 = pl.tile.store(stage_11_v47, [stage_11_v13, 0], temporary_19)
            for stage_11_v31 in pl.pipeline(0, 1024, 256, stage=2):
                stage_11_v49: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_11_v14, 0, stage_11_v31, [8, 256], target_memory=pl.Mem.Vec)
                stage_11_v50: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_11_v9, 0, stage_11_v31, [1, 256], target_memory=pl.Mem.Vec)
                stage_11_v51: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_11_v49, stage_11_v21)
                stage_11_v52: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_11_v51, stage_11_v50)
                stage_11_v53: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_11_v52, stage_11_v27)
                stage_11_v54: pl.Tile[[8, 256], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_11_v53, target_type=pl.INT32, mode='rint')
                stage_11_v55: pl.Tile[[8, 256], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_11_v54, target_type=pl.FP16, mode='round')
                stage_11_v56: pl.Tile[[8, 256], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_11_v55, target_type=pl.INT8, mode='trunc')
                stage_11_v57 = pl.tile.store(stage_11_v56, [stage_11_v11, stage_11_v31], temporary_22)
                if stage_11_v12 == 8:
                    stage_11_v62 = pl.tile.store(stage_11_v56, [stage_11_v13, stage_11_v31], temporary_18)
                else:
                    stage_11_v63: pl.Tile[[8, 256], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_11_v12, 256])] = pl.tile.set_validshape(stage_11_v56, stage_11_v12, 256)
                    stage_11_v64 = pl.tile.store(stage_11_v63, [stage_11_v13, stage_11_v31], temporary_18)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_11 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_11, pl.INT32))
    record_12 = 18144 + core_id * 6
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
            flag_12 = other_12 * 128
            progress = csa_signal_wait(progress, flag_12, pl.cast(target_12, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_12_v8: pl.Scalar[pl.INDEX] = block_12
        for (stage_12_v9, (stage_12_v12,)) in pl.range(stage_12_v8, 32, 32, init_values=(temporary_36,)):
            stage_12_v10: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_12_v9 * 6])
            for (stage_12_v13, (stage_12_v19,)) in pl.range(6, init_values=(stage_12_v12,)):
                stage_12_v14: pl.Scalar[pl.INDEX] = stage_12_v9 * 6 + stage_12_v13
                stage_12_v15: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_12_v14])
                stage_12_v16: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
                stage_12_v17 = pl.tile.store(stage_12_v16, [stage_12_v14, 0], stage_12_v19)
                if (stage_12_v15 + 1) % 4 == 0:
                    stage_12_v20: pl.Scalar[pl.INT64] = stage_12_v15 - 8 + 1
                    for (stage_12_v22, (stage_12_v33,)) in pl.range(0, 128, 64, init_values=(stage_12_v17,)):
                        stage_12_v23: pl.Scalar[pl.INDEX] = pl.cast(stage_12_v15 % 4, pl.INDEX)
                        stage_12_v24: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_35, [stage_12_v14, stage_12_v22 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                        stage_12_v25: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_ape, [stage_12_v23, stage_12_v22 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                        stage_12_v26: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_12_v24, stage_12_v25)
                        stage_12_v27: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_12_v26, stage_12_v26)
                        stage_12_v28: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_12_v27)
                        stage_12_v29: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_34, [stage_12_v14, stage_12_v22 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                        for (stage_12_v34, (stage_12_v50, stage_12_v51, stage_12_v52)) in pl.range(7, init_values=(stage_12_v28, stage_12_v26, stage_12_v29)):
                            stage_12_v35: pl.Scalar[pl.INT64] = stage_12_v20 + stage_12_v34
                            stage_12_v36: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
                            stage_12_v37: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=-3.4028234663852886e+38)
                            stage_12_v38: pl.Scalar[pl.INDEX] = 0
                            if 4 <= stage_12_v34:
                                stage_12_v53: pl.Scalar[pl.INDEX] = 128
                                stage_12_v54: pl.Scalar[pl.INDEX] = pl.yield_(stage_12_v53)
                            else:
                                stage_12_v54: pl.Scalar[pl.INDEX] = pl.yield_(stage_12_v38)
                            if 0 <= stage_12_v35 and stage_12_v35 < stage_12_v10:
                                stage_12_v55: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
                                stage_12_v56: pl.Scalar[pl.INT32] = pl.tensor.read(arg_inner_state_block_table, [stage_12_v9, stage_12_v35 // 2])
                                stage_12_v57: pl.Scalar[pl.INDEX] = pl.cast(stage_12_v56, pl.INDEX)
                                if 0 <= stage_12_v57:
                                    stage_12_v61: pl.Scalar[pl.INDEX] = pl.cast(stage_12_v35 % 2, pl.INDEX) * 512 + stage_12_v54 + stage_12_v22
                                    stage_12_v62: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_compress_state, [stage_12_v57, stage_12_v61], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    stage_12_v63: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_compress_state, [stage_12_v57, stage_12_v61 + 256], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    (stage_12_v74, stage_12_v75) = pl.yield_(stage_12_v63, stage_12_v62)
                                else:
                                    (stage_12_v74, stage_12_v75) = pl.yield_(stage_12_v55, stage_12_v36)
                                (stage_12_v64, stage_12_v65) = pl.yield_(stage_12_v74, stage_12_v75)
                            else:
                                (stage_12_v64, stage_12_v65) = pl.yield_(stage_12_v37, stage_12_v36)
                            if stage_12_v10 <= stage_12_v35:
                                if stage_12_v35 <= stage_12_v15:
                                    stage_12_v66: pl.Scalar[pl.INDEX] = stage_12_v9 * 6 + stage_12_v35 - stage_12_v10
                                    stage_12_v67: pl.Scalar[pl.INDEX] = pl.cast(stage_12_v35 % 4, pl.INDEX)
                                    stage_12_v68: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_34, [stage_12_v66, stage_12_v54 + stage_12_v22], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    stage_12_v69: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_35, [stage_12_v66, stage_12_v54 + stage_12_v22], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    stage_12_v70: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_ape, [stage_12_v67, stage_12_v54 + stage_12_v22], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                    stage_12_v71: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_12_v69, stage_12_v70)
                                    (stage_12_v76, stage_12_v77) = pl.yield_(stage_12_v71, stage_12_v68)
                                else:
                                    (stage_12_v76, stage_12_v77) = pl.yield_(stage_12_v64, stage_12_v65)
                                (stage_12_v72, stage_12_v73) = pl.yield_(stage_12_v76, stage_12_v77)
                            else:
                                (stage_12_v72, stage_12_v73) = pl.yield_(stage_12_v64, stage_12_v65)
                            stage_12_v39: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_12_v51, stage_12_v72)
                            stage_12_v40: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_12_v51, stage_12_v39)
                            stage_12_v41: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_12_v40)
                            stage_12_v42: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_12_v72, stage_12_v39)
                            stage_12_v43: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_12_v42)
                            stage_12_v44: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_12_v41, stage_12_v50)
                            stage_12_v45: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_12_v44, stage_12_v43)
                            stage_12_v46: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_12_v52, stage_12_v41)
                            stage_12_v47: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_12_v73, stage_12_v43)
                            stage_12_v48: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_12_v46, stage_12_v47)
                            stage_12_v49: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = stage_12_v39
                            (stage_12_v58, stage_12_v59, stage_12_v60) = pl.yield_(stage_12_v45, stage_12_v49, stage_12_v48)
                        stage_12_v30: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_12_v60, stage_12_v58)
                        stage_12_v31 = pl.tile.store(stage_12_v30, [stage_12_v14, stage_12_v22], stage_12_v33)
                        stage_12_v32 = pl.yield_(stage_12_v31)
                    stage_12_v21 = pl.yield_(stage_12_v32)
                else:
                    stage_12_v21 = pl.yield_(stage_12_v17)
                stage_12_v18 = pl.yield_(stage_12_v21)
            stage_12_v11 = pl.yield_(stage_12_v18)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_12 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_12, pl.INT32))
    record_13 = 18576 + core_id * 6
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
            flag_13 = other_13 * 128
            progress = csa_signal_wait(progress, flag_13, pl.cast(target_13, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_13_v2: pl.Scalar[pl.INDEX] = block_13
        stage_13_v3: pl.Scalar[pl.INDEX] = stage_13_v2 * 2
        stage_13_v4: pl.Tile[[2, 128], pl.BF16, pl.Mem.Vec] = pl.tile.full([2, 128], dtype=pl.BF16, value=0.0)
        stage_13_v5 = pl.tile.store(stage_13_v4, [stage_13_v3, 0], temporary_33)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_13 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_13, pl.INT32))
    record_14 = 19008 + core_id * 6
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
            flag_14 = other_14 * 128
            progress = csa_signal_wait(progress, flag_14, pl.cast(target_14, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_14_v7: pl.Scalar[pl.INDEX] = block_14
        for (stage_14_v8, (stage_14_v10,)) in pl.range(stage_14_v7, 32, 32, init_values=(arg_inner_compress_state,)):
            for (stage_14_v11, (stage_14_v16,)) in pl.range(6, init_values=(stage_14_v10,)):
                stage_14_v12: pl.Scalar[pl.INDEX] = stage_14_v8 * 6 + stage_14_v11
                stage_14_v13: pl.Scalar[pl.INT32] = pl.tensor.read(arg_inner_state_slot_mapping, [stage_14_v12, 0])
                stage_14_v14: pl.Scalar[pl.INT32] = pl.tensor.read(arg_inner_state_slot_mapping, [stage_14_v12, 1])
                if 0 <= pl.cast(stage_14_v13, pl.INDEX) and 0 <= pl.cast(stage_14_v14, pl.INDEX):
                    stage_14_v17: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_14_v12])
                    stage_14_v18: pl.Scalar[pl.INDEX] = pl.cast(stage_14_v13, pl.INDEX)
                    stage_14_v19: pl.Scalar[pl.INDEX] = pl.cast(stage_14_v14, pl.INDEX) * 512
                    stage_14_v20: pl.Scalar[pl.INDEX] = pl.cast(stage_14_v17 % 4, pl.INDEX)
                    stage_14_v21: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_34, [stage_14_v12, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                    stage_14_v22 = pl.tile.store(stage_14_v21, [stage_14_v18, stage_14_v19], stage_14_v16)
                    stage_14_v23: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_35, [stage_14_v12, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                    stage_14_v24: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_ape, [stage_14_v20, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                    stage_14_v25: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_14_v23, stage_14_v24)
                    stage_14_v26 = pl.tile.store(stage_14_v25, [stage_14_v18, stage_14_v19 + 256], stage_14_v22)
                    stage_14_v27 = pl.yield_(stage_14_v26)
                else:
                    stage_14_v27 = pl.yield_(stage_14_v16)
                stage_14_v15 = pl.yield_(stage_14_v27)
            stage_14_v9 = pl.yield_(stage_14_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_14 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_14, pl.INT32))
    record_15 = 19440 + core_id * 6
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
            flag_15 = other_15 * 128
            progress = csa_signal_wait(progress, flag_15, pl.cast(target_15, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_15_v7: pl.Scalar[pl.INDEX] = block_15
        stage_15_v8: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        stage_15_v9: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_15_v10: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v9, target_type=pl.FP32, mode='round')
        stage_15_v11: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_15_v8, stage_15_v10)
        stage_15_v12: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v11, 0.5)
        stage_15_v13: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_15_v12, target_type=pl.INT32, mode='trunc')
        stage_15_v14: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v13, target_type=pl.FP32, mode='round')
        stage_15_v15: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v14, 2.0)
        stage_15_v16: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_15_v11, stage_15_v15)
        stage_15_v17: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_15_v11, 1.0)
        stage_15_v18: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v16, 2.0)
        stage_15_v19: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_15_v17, stage_15_v18)
        stage_15_v20: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_15_v19, target_type=pl.INT32, mode='round')
        for (stage_15_v21, (stage_15_v25,)) in pl.range(stage_15_v7, 384, 48, init_values=(view_85,)):
            stage_15_v22: pl.Scalar[pl.INDEX] = stage_15_v21 % 16 * 4
            stage_15_v23: pl.Scalar[pl.INDEX] = stage_15_v21 // 16 * 8
            if stage_15_v23 + 8 <= 192:
                stage_15_v26: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_19, [stage_15_v23, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_15_v27: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_15_v23, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_15_v28: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_13, [stage_15_v23, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_15_v29: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                stage_15_v30: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                stage_15_v31: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.rems(stage_15_v29, pl.const(2, pl.INT32), stage_15_v30)
                stage_15_v32: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_15_v31, pl.const(2, pl.INT32))
                stage_15_v33: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.sub(stage_15_v29, stage_15_v32)
                stage_15_v34: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_15_v33, pl.const(65, pl.INT32))
                stage_15_v35: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                stage_15_v36: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.col_expand(stage_15_v35, stage_15_v34)
                stage_15_v37: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False)
                stage_15_v38: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_15_v37, pl.const(128, pl.INT32))
                stage_15_v39: pl.Tile[[8, 1], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v38, [8, 1])
                stage_15_v40: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.row_expand_add(stage_15_v36, stage_15_v39)
                stage_15_v41: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v40, [1, 512])
                stage_15_v42: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 512], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_15_v43 in pl.pipeline(4, stage=2):
                    stage_15_v46: pl.Scalar[pl.INDEX] = (stage_15_v22 + stage_15_v43) * 128
                    stage_15_v47: pl.Tile[[128], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_idx_wq_b_scale, [stage_15_v46], [128], [128], target_memory=pl.Mem.Vec)
                    stage_15_v48: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v47, [1, 128])
                    stage_15_v49: pl.Tile[[8, 128], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_42, [stage_15_v23, stage_15_v46], [8, 128], [8, 128], target_memory=pl.Mem.Vec)
                    stage_15_v50: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v49, target_type=pl.FP32, mode='none')
                    stage_15_v51: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_15_v50, stage_15_v26)
                    stage_15_v52: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_15_v51, stage_15_v48)
                    stage_15_v53: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_15_v52, [8, 64], [0, 0])
                    stage_15_v54: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_15_v53, target_type=pl.BF16, mode='rint')
                    stage_15_v55: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_15_v52, [8, 64], [0, 64])
                    stage_15_v56: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v52, [1, 1024])
                    stage_15_v57: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_15_v56, stage_15_v41, stage_15_v42)
                    stage_15_v58: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v57, [8, 64])
                    stage_15_v59: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_15_v55, stage_15_v27)
                    stage_15_v60: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_15_v58, stage_15_v28)
                    stage_15_v61: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_15_v59, stage_15_v60)
                    stage_15_v62: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_15_v61, target_type=pl.BF16, mode='rint')
                    pl.tile.store(stage_15_v54, [stage_15_v23, stage_15_v46], stage_15_v25)
                    pl.tile.store(stage_15_v62, [stage_15_v23, stage_15_v46 + 64], stage_15_v25)
                stage_15_v44 = pl.yield_(stage_15_v25)
            else:
                stage_15_v45: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_15_v20, [1, 64], [0, 0])
                for (stage_15_v63, (stage_15_v68,)) in pl.range(stage_15_v23, 192, init_values=(stage_15_v25,)):
                    stage_15_v64: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_19, [stage_15_v63, 0])
                    stage_15_v65: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_15_v63, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_15_v66: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_13, [stage_15_v63, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    for (stage_15_v69, (stage_15_v89,)) in pl.pipeline(4, stage=2, init_values=(stage_15_v68,)):
                        stage_15_v70: pl.Scalar[pl.INDEX] = (stage_15_v22 + stage_15_v69) * 128
                        stage_15_v71: pl.Tile[[128], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_idx_wq_b_scale, [stage_15_v70], [128], [128], target_memory=pl.Mem.Vec)
                        stage_15_v72: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_15_v71, [1, 128])
                        stage_15_v73: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_42, [stage_15_v63, stage_15_v70], [1, 128], [1, 128], target_memory=pl.Mem.Vec)
                        stage_15_v74: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_15_v73, target_type=pl.FP32, mode='none')
                        stage_15_v75: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_15_v74, stage_15_v64)
                        stage_15_v76: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_15_v75, stage_15_v72)
                        stage_15_v77: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_15_v76, [1, 64], [0, 0])
                        stage_15_v78: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_15_v77, target_type=pl.BF16, mode='rint')
                        stage_15_v79: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_15_v76, [1, 64], [0, 64])
                        stage_15_v80: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        for (stage_15_v90, (stage_15_v97,)) in pl.range(1, init_values=(stage_15_v80,)):
                            stage_15_v91: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_15_v79, [1, 64], [stage_15_v90, 0], [1, 64])
                            stage_15_v92: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_15_v45, [1, 64], [stage_15_v90, 0], [1, 64])
                            stage_15_v93: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                            stage_15_v94: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_15_v91, stage_15_v92, stage_15_v93)
                            stage_15_v95: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_15_v97, stage_15_v94, [stage_15_v90, 0])
                            stage_15_v96: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_15_v95)
                        stage_15_v81: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = stage_15_v96
                        stage_15_v82: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_15_v79, stage_15_v65)
                        stage_15_v83: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_15_v81, stage_15_v66)
                        stage_15_v84: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_15_v82, stage_15_v83)
                        stage_15_v85: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_15_v84, target_type=pl.BF16, mode='rint')
                        stage_15_v86 = pl.tile.store(stage_15_v78, [stage_15_v63, stage_15_v70], stage_15_v89)
                        stage_15_v87 = pl.tile.store(stage_15_v85, [stage_15_v63, stage_15_v70 + 64], stage_15_v86)
                        stage_15_v88 = pl.yield_(stage_15_v87)
                    stage_15_v67 = pl.yield_(stage_15_v88)
                stage_15_v44 = pl.yield_(stage_15_v67)
            stage_15_v24 = pl.yield_(stage_15_v44)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_15 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_15, pl.INT32))
    record_16 = 19872 + core_id * 6
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
            flag_16 = other_16 * 128
            progress = csa_signal_wait(progress, flag_16, pl.cast(target_16, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_16_v7: pl.Scalar[pl.INDEX] = block_16
        for (stage_16_v8, (stage_16_v10,)) in pl.range(stage_16_v7, 32, 32, init_values=(arg_inner_compress_state,)):
            for (stage_16_v11, (stage_16_v16,)) in pl.range(6, init_values=(stage_16_v10,)):
                stage_16_v12: pl.Scalar[pl.INDEX] = stage_16_v8 * 6 + stage_16_v11
                stage_16_v13: pl.Scalar[pl.INT32] = pl.tensor.read(arg_inner_state_slot_mapping, [stage_16_v12, 0])
                stage_16_v14: pl.Scalar[pl.INT32] = pl.tensor.read(arg_inner_state_slot_mapping, [stage_16_v12, 1])
                if 0 <= pl.cast(stage_16_v13, pl.INDEX) and 0 <= pl.cast(stage_16_v14, pl.INDEX):
                    stage_16_v17: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_16_v12])
                    stage_16_v18: pl.Scalar[pl.INDEX] = pl.cast(stage_16_v13, pl.INDEX)
                    stage_16_v19: pl.Scalar[pl.INDEX] = pl.cast(stage_16_v14, pl.INDEX) * 512
                    stage_16_v20: pl.Scalar[pl.INDEX] = pl.cast(stage_16_v17 % 4, pl.INDEX)
                    stage_16_v21: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_34, [stage_16_v12, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                    stage_16_v22 = pl.tile.store(stage_16_v21, [stage_16_v18, stage_16_v19], stage_16_v16)
                    stage_16_v23: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_35, [stage_16_v12, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                    stage_16_v24: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_inner_ape, [stage_16_v20, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                    stage_16_v25: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_16_v23, stage_16_v24)
                    stage_16_v26 = pl.tile.store(stage_16_v25, [stage_16_v18, stage_16_v19 + 256], stage_16_v22)
                    stage_16_v27 = pl.yield_(stage_16_v26)
                else:
                    stage_16_v27 = pl.yield_(stage_16_v16)
                stage_16_v15 = pl.yield_(stage_16_v27)
            stage_16_v9 = pl.yield_(stage_16_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_16 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_16, pl.INT32))
    record_17 = 20304 + core_id * 6
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
            flag_17 = other_17 * 128
            progress = csa_signal_wait(progress, flag_17, pl.cast(target_17, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
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
        for (stage_17_v22, (stage_17_v69,)) in pl.range(stage_17_v8, 12, 2, init_values=(temporary_33,)):
            stage_17_v23: pl.Scalar[pl.INDEX] = stage_17_v22 * 16
            stage_17_v24: pl.Scalar[pl.INDEX] = pl.min(192 - stage_17_v23, 16)
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
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_17, pl.INT32))
    record_18 = 20736 + core_id * 6
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
            flag_18 = other_18 * 128
            progress = csa_signal_wait(progress, flag_18, pl.cast(target_18, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_18_v4: pl.Scalar[pl.INDEX] = block_18
        for (stage_18_v5, (stage_18_v21, stage_18_v22)) in pl.range(stage_18_v4, 12288 // 64, 48, init_values=(temporary_39, temporary_40)):
            stage_18_v6: pl.Scalar[pl.INDEX] = stage_18_v5 * 64
            stage_18_v7: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_43, [stage_18_v6, 0], [64, 128], [64, 128], target_memory=pl.Mem.Vec)
            stage_18_v8: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_18_v7, 0.08838834764831845)
            stage_18_v9: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0001)
            for (stage_18_v23, (stage_18_v35,)) in pl.range(0, 128, 64, init_values=(stage_18_v9,)):
                stage_18_v24: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_18_v8, 0, stage_18_v23, [64, 64], target_memory=pl.Mem.Vec)
                stage_18_v25: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_18_v24)
                stage_18_v26: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_18_v8, 0, stage_18_v23, [64, 64], target_memory=pl.Mem.Vec)
                stage_18_v27: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_18_v26, stage_18_v25)
                stage_18_v28: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_18_v29: pl.Tile[[64, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_18_v27, stage_18_v28)
                stage_18_v30: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v29, [1, 64])
                stage_18_v31: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_18_v35, stage_18_v30)
                stage_18_v32: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_18_v31)
            stage_18_v10: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=127.0)
            stage_18_v11: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_18_v10, stage_18_v32)
            stage_18_v12: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_18_v11)
            stage_18_v13: pl.Tile[[64, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v12, [64, 1])
            stage_18_v14 = pl.tile.store(stage_18_v13, [stage_18_v6, 0], stage_18_v22)
            stage_18_v15: pl.Tile[[64, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_18_v11, [64, 1])
            stage_18_v16: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_18_v8, stage_18_v15)
            stage_18_v17: pl.Tile[[64, 128], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_18_v16, target_type=pl.INT32, mode='rint')
            stage_18_v18: pl.Tile[[64, 128], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_18_v17, target_type=pl.FP16, mode='round')
            stage_18_v19: pl.Tile[[64, 128], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_18_v18, target_type=pl.INT8, mode='trunc')
            stage_18_v20 = pl.tile.store(stage_18_v19, [stage_18_v6, 0], stage_18_v21)
            (stage_18_v33, stage_18_v34) = pl.yield_(stage_18_v20, stage_18_v14)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_18 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_18, pl.INT32))
    record_19 = 21168 + core_id * 6
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
            flag_19 = other_19 * 128
            progress = csa_signal_wait(progress, flag_19, pl.cast(target_19, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_19_v3: pl.Scalar[pl.INDEX] = block_19
        stage_19_v4: pl.Scalar[pl.INDEX] = stage_19_v3 * 16
        stage_19_v5: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_45, [stage_19_v4, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_19_v6: pl.Scalar[pl.INDEX] = stage_19_v4 + 384
        stage_19_v7: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_45, [stage_19_v6, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_19_v8: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_19_v5, stage_19_v7)
        stage_19_v9: pl.Scalar[pl.INDEX] = stage_19_v4 + 768
        stage_19_v10: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_45, [stage_19_v9, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_19_v11: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_19_v8, stage_19_v10)
        stage_19_v12: pl.Scalar[pl.INDEX] = stage_19_v4 + 1152
        stage_19_v13: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_45, [stage_19_v12, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_19_v14: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_19_v11, stage_19_v13)
        stage_19_v15: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_19_v14, 0.011048543456039806)
        stage_19_v16 = pl.tile.store(stage_19_v15, [stage_19_v4, 0], temporary_44)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_19 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_19, pl.INT32))
    record_20 = 21600 + core_id * 6
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
            flag_20 = other_20 * 128
            progress = csa_signal_wait(progress, flag_20, pl.cast(target_20, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_20_v9: pl.Scalar[pl.INDEX] = block_20
        stage_20_v10: pl.Scalar[pl.INDEX] = stage_20_v9 * 16
        stage_20_v11: pl.Scalar[pl.INDEX] = pl.min(64 - stage_20_v10, 16)
        stage_20_v12: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_38, [stage_20_v10, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec)
        stage_20_v13: pl.Tile[[16, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_20_v12, target_type=pl.BF16, mode='rint')
        stage_20_v14: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_20_v13, target_type=pl.FP32, mode='round')
        stage_20_v15: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_20_v14, 0.08838834764831845)
        stage_20_v16: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_20_v15)
        stage_20_v17: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_20_v18: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_20_v16, stage_20_v17)
        stage_20_v19: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_20_v18, [1, 16])
        stage_20_v20: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0001)
        stage_20_v21: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_20_v19, stage_20_v20)
        stage_20_v22: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=127.0)
        stage_20_v23: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_20_v22, stage_20_v21)
        stage_20_v24: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_20_v23)
        stage_20_v25: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_20_v24, [16, 1])
        stage_20_v26: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_20_v23, [16, 1])
        stage_20_v27 = pl.tile.store(stage_20_v25, [stage_20_v10, 0], temporary_37)
        stage_20_v28: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_20_v15, stage_20_v26)
        stage_20_v29: pl.Tile[[16, 128], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_20_v28, target_type=pl.INT32, mode='rint')
        stage_20_v30: pl.Tile[[16, 128], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_20_v29, target_type=pl.FP16, mode='round')
        stage_20_v31: pl.Tile[[16, 128], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_20_v30, target_type=pl.INT8, mode='trunc')
        for (stage_20_v32, (stage_20_v38, stage_20_v39)) in pl.range(stage_20_v11, init_values=(arg_idx_native_kv_cache, temporary_32)):
            stage_20_v33: pl.Scalar[pl.INDEX] = stage_20_v10 + stage_20_v32
            stage_20_v34: pl.Scalar[pl.INDEX] = stage_20_v33 // 2
            stage_20_v35: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_20_v34 * 6])
            stage_20_v36: pl.Scalar[pl.INDEX] = stage_20_v33 % 2 * 4 - stage_20_v35 % 4 + 3
            stage_20_v37: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_20_v34])
            if 0 < pl.cast(stage_20_v37, pl.INDEX) and stage_20_v36 < 6:
                stage_20_v40: pl.Scalar[pl.INDEX] = stage_20_v34 * 6 + stage_20_v36
                stage_20_v41: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_20_v40])
                stage_20_v42: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_15, [stage_20_v34])
                stage_20_v43: pl.Scalar[pl.INDEX] = pl.cast(stage_20_v42, pl.INDEX) + pl.cast((stage_20_v41 + 1) // 4, pl.INDEX)
                stage_20_v44: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_idx_slot_mapping, 0)
                stage_20_v45: pl.Scalar[pl.INDEX] = pl.min(stage_20_v43, stage_20_v44 - 1)
                stage_20_v46: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_slot_mapping, [stage_20_v45, 0])
                stage_20_v47: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_slot_mapping, [stage_20_v45, 1])
                if stage_20_v43 < stage_20_v44 and 0 <= pl.cast(stage_20_v46, pl.INDEX) and (0 <= pl.cast(stage_20_v47, pl.INDEX)):
                    stage_20_v50: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_20_v15, [1, 128], [stage_20_v32, 0])
                    stage_20_v51 = pl.tile.store(stage_20_v50, [stage_20_v40, 0], stage_20_v39)
                    stage_20_v52: pl.Scalar[pl.INDEX] = pl.cast(stage_20_v46, pl.INDEX)
                    stage_20_v53: pl.Scalar[pl.INDEX] = pl.cast(stage_20_v47, pl.INDEX) * 128
                    stage_20_v54: pl.Tile[[1, 128], pl.INT8, pl.Mem.Vec] = pl.tile.slice(stage_20_v31, [1, 128], [stage_20_v32, 0])
                    stage_20_v55 = pl.tile.store(stage_20_v54, [stage_20_v52, stage_20_v53], stage_20_v38)
                    (stage_20_v58, stage_20_v59) = pl.yield_(stage_20_v55, stage_20_v51)
                else:
                    (stage_20_v58, stage_20_v59) = pl.yield_(stage_20_v38, stage_20_v39)
                (stage_20_v56, stage_20_v57) = pl.yield_(stage_20_v58, stage_20_v59)
            else:
                (stage_20_v56, stage_20_v57) = pl.yield_(stage_20_v38, stage_20_v39)
            (stage_20_v48, stage_20_v49) = pl.yield_(stage_20_v56, stage_20_v57)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_20 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_20, pl.INT32))
    record_21 = 22032 + core_id * 6
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
            flag_21 = other_21 * 128
            progress = csa_signal_wait(progress, flag_21, pl.cast(target_21, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        for stage_21_v6 in pl.range(64):
            stage_21_v7: pl.Scalar[pl.INDEX] = stage_21_v6 // 2
            stage_21_v8: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_21_v7 * 6])
            stage_21_v9: pl.Scalar[pl.INDEX] = stage_21_v6 % 2 * 4 - stage_21_v8 % 4 + 3
            stage_21_v10: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_21_v7])
            if 0 < pl.cast(stage_21_v10, pl.INDEX) and stage_21_v9 < 6:
                stage_21_v11: pl.Scalar[pl.INDEX] = stage_21_v7 * 6 + stage_21_v9
                stage_21_v12: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_21_v11])
                stage_21_v13: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_15, [stage_21_v7])
                stage_21_v14: pl.Scalar[pl.INDEX] = pl.cast(stage_21_v13, pl.INDEX) + pl.cast((stage_21_v12 + 1) // 4, pl.INDEX)
                stage_21_v15: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_idx_slot_mapping, 0)
                stage_21_v16: pl.Scalar[pl.INDEX] = pl.min(stage_21_v14, stage_21_v15 - 1)
                stage_21_v17: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_slot_mapping, [stage_21_v16, 0])
                stage_21_v18: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_slot_mapping, [stage_21_v16, 1])
                if stage_21_v14 < stage_21_v15 and 0 <= pl.cast(stage_21_v17, pl.INDEX) and (0 <= pl.cast(stage_21_v18, pl.INDEX)):
                    stage_21_v19: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_37, [stage_21_v6, 0])
                    stage_21_v20: pl.Scalar[pl.FP16] = pl.cast(stage_21_v19, pl.FP16)
                    stage_21_v21: pl.Scalar[pl.INDEX] = pl.cast(stage_21_v17, pl.INDEX)
                    stage_21_v22: pl.Tile[[1, 64], pl.INT8, pl.Mem.Vec] = pl.tile.load(arg_idx_native_kv_cache, [stage_21_v21, 4096], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_21_v23: pl.Tile[[1, 32], pl.FP16, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_21_v22, dtype=pl.FP16)
                    pl.tile.write(stage_21_v23, [0, stage_21_v18], stage_21_v20)
                    stage_21_v24: pl.Tile[[1, 64], pl.INT8, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_21_v23, dtype=pl.INT8)
                    stage_21_v25 = pl.tile.store(stage_21_v24, [stage_21_v21, 4096], arg_idx_native_kv_cache)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_21 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_21, pl.INT32))
    record_22 = 22464 + core_id * 6
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
            flag_22 = other_22 * 128
            progress = csa_signal_wait(progress, flag_22, pl.cast(target_22, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_22_v5: pl.Scalar[pl.INDEX] = block_22
        stage_22_v6: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        for stage_22_v7 in pl.range(stage_22_v5, stage_22_v6 // 2, 48):
            stage_22_v8: pl.Tile[[32, 64], pl.FP16, pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP16, value=0.0)
            stage_22_v9: pl.Scalar[pl.INDEX] = stage_22_v7 * 2
            stage_22_v10: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_94, [stage_22_v9, 0], [2, 64], [2, 64], target_memory=pl.Mem.Vec)
            stage_22_v11: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_44, [stage_22_v9, 0], [2, 64], [2, 64], target_memory=pl.Mem.Vec)
            stage_22_v12: pl.Tile[[2, 64], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_22_v10, target_type=pl.FP16, mode='rint')
            stage_22_v13: pl.Tile[[2, 64], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_22_v11, target_type=pl.FP16, mode='rint')
            stage_22_v14: pl.Tile[[2, 64], pl.FP16, pl.Mem.Vec] = pl.tile.mul(stage_22_v12, stage_22_v13)
            stage_22_v15: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_22_v14, 0, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_22_v16: pl.Tile[[32, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_22_v8, stage_22_v15, [0, 0])
            stage_22_v17: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_22_v14, 1, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_22_v18: pl.Tile[[32, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_22_v16, stage_22_v17, [3, 0])
            stage_22_v19: pl.Tile[[16, 128], pl.FP16, pl.Mem.Vec] = pl.tile.reshape(stage_22_v18, [16, 128])
            stage_22_v20 = pl.tile.store(stage_22_v19, [stage_22_v7 * 16, 0], temporary_48)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_22 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_22, pl.INT32))
    record_23 = 22896 + core_id * 6
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
            flag_23 = other_23 * 128
            progress = csa_signal_wait(progress, flag_23, pl.cast(target_23, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_23_v9: pl.Scalar[pl.INDEX] = block_23
        for (stage_23_v10, (stage_23_v14,)) in pl.range(stage_23_v9, 32, 32, init_values=(temporary_28,)):
            stage_23_v11: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_23_v10 * 6])
            stage_23_v12: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_seq_lens, [stage_23_v10])
            for (stage_23_v15, (stage_23_v21,)) in pl.range(6, init_values=(stage_23_v14,)):
                stage_23_v16: pl.Scalar[pl.INDEX] = stage_23_v10 * 6 + stage_23_v15
                stage_23_v17: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_23_v16])
                stage_23_v18: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                stage_23_v19 = pl.tile.store(stage_23_v18, [stage_23_v16, 0], stage_23_v21)
                if 0 < pl.cast(stage_23_v12, pl.INDEX) and (stage_23_v17 + 1) % 4 == 0:
                    stage_23_v22: pl.Scalar[pl.INT64] = stage_23_v17 - 8 + 1
                    stage_23_v23: pl.Scalar[pl.INDEX] = pl.cast(stage_23_v17 % 4, pl.INDEX)
                    stage_23_v24: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_30, [stage_23_v16, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    stage_23_v25: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_cmp_ape, [stage_23_v23, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    stage_23_v26: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_23_v24, stage_23_v25)
                    stage_23_v27: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_23_v26, stage_23_v26)
                    stage_23_v28: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_23_v27)
                    stage_23_v29: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_29, [stage_23_v16, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    for (stage_23_v31, (stage_23_v47, stage_23_v48, stage_23_v49)) in pl.range(7, init_values=(stage_23_v28, stage_23_v26, stage_23_v29)):
                        stage_23_v32: pl.Scalar[pl.INT64] = stage_23_v22 + stage_23_v31
                        stage_23_v33: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                        stage_23_v34: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e+38)
                        stage_23_v35: pl.Scalar[pl.INDEX] = 0
                        if 4 <= stage_23_v31:
                            stage_23_v50: pl.Scalar[pl.INDEX] = 512
                            stage_23_v51: pl.Scalar[pl.INDEX] = pl.yield_(stage_23_v50)
                        else:
                            stage_23_v51: pl.Scalar[pl.INDEX] = pl.yield_(stage_23_v35)
                        if 0 <= stage_23_v32 and stage_23_v32 < stage_23_v11:
                            stage_23_v52: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                            stage_23_v53: pl.Scalar[pl.INT32] = pl.tensor.read(arg_state_block_table, [stage_23_v10, stage_23_v32 // 2])
                            stage_23_v54: pl.Scalar[pl.INDEX] = pl.cast(stage_23_v53, pl.INDEX)
                            if 0 <= stage_23_v54:
                                stage_23_v58: pl.Scalar[pl.INDEX] = pl.cast(stage_23_v32 % 2, pl.INDEX) * 2048 + stage_23_v51
                                stage_23_v59: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_compress_state, [stage_23_v54, stage_23_v58], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                stage_23_v60: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_compress_state, [stage_23_v54, stage_23_v58 + 1024], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                (stage_23_v71, stage_23_v72) = pl.yield_(stage_23_v60, stage_23_v59)
                            else:
                                (stage_23_v71, stage_23_v72) = pl.yield_(stage_23_v52, stage_23_v33)
                            (stage_23_v61, stage_23_v62) = pl.yield_(stage_23_v71, stage_23_v72)
                        else:
                            (stage_23_v61, stage_23_v62) = pl.yield_(stage_23_v34, stage_23_v33)
                        if stage_23_v11 <= stage_23_v32:
                            if stage_23_v32 <= stage_23_v17:
                                stage_23_v63: pl.Scalar[pl.INDEX] = stage_23_v10 * 6 + stage_23_v32 - stage_23_v11
                                stage_23_v64: pl.Scalar[pl.INDEX] = pl.cast(stage_23_v32 % 4, pl.INDEX)
                                stage_23_v65: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_29, [stage_23_v63, stage_23_v51], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                stage_23_v66: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_30, [stage_23_v63, stage_23_v51], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                stage_23_v67: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_cmp_ape, [stage_23_v64, stage_23_v51], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                                stage_23_v68: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_23_v66, stage_23_v67)
                                (stage_23_v73, stage_23_v74) = pl.yield_(stage_23_v68, stage_23_v65)
                            else:
                                (stage_23_v73, stage_23_v74) = pl.yield_(stage_23_v61, stage_23_v62)
                            (stage_23_v69, stage_23_v70) = pl.yield_(stage_23_v73, stage_23_v74)
                        else:
                            (stage_23_v69, stage_23_v70) = pl.yield_(stage_23_v61, stage_23_v62)
                        stage_23_v36: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_23_v48, stage_23_v69)
                        stage_23_v37: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_23_v48, stage_23_v36)
                        stage_23_v38: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_23_v37)
                        stage_23_v39: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_23_v69, stage_23_v36)
                        stage_23_v40: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_23_v39)
                        stage_23_v41: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_23_v38, stage_23_v47)
                        stage_23_v42: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_23_v41, stage_23_v40)
                        stage_23_v43: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_23_v49, stage_23_v38)
                        stage_23_v44: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_23_v70, stage_23_v40)
                        stage_23_v45: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_23_v43, stage_23_v44)
                        stage_23_v46: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = stage_23_v36
                        (stage_23_v55, stage_23_v56, stage_23_v57) = pl.yield_(stage_23_v42, stage_23_v46, stage_23_v45)
                    stage_23_v30: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_23_v57, stage_23_v55)
                    pl.tile.store(stage_23_v30, [stage_23_v16, 0], stage_23_v19)
                stage_23_v20 = pl.yield_(stage_23_v19)
            stage_23_v13 = pl.yield_(stage_23_v20)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_23 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_23, pl.INT32))
    record_24 = 23328 + core_id * 6
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
            flag_24 = other_24 * 128
            progress = csa_signal_wait(progress, flag_24, pl.cast(target_24, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_24_v5: pl.Scalar[pl.INDEX] = block_24
        stage_24_v6: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        for stage_24_v7 in pl.range(stage_24_v5, stage_24_v6 // 2, 48):
            stage_24_v8: pl.Tile[[32, 64], pl.FP16, pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP16, value=0.0)
            stage_24_v9: pl.Scalar[pl.INDEX] = stage_24_v7 * 2
            stage_24_v10: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_94, [stage_24_v9, 0], [2, 64], [2, 64], target_memory=pl.Mem.Vec)
            stage_24_v11: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_44, [stage_24_v9, 0], [2, 64], [2, 64], target_memory=pl.Mem.Vec)
            stage_24_v12: pl.Tile[[2, 64], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_24_v10, target_type=pl.FP16, mode='rint')
            stage_24_v13: pl.Tile[[2, 64], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_24_v11, target_type=pl.FP16, mode='rint')
            stage_24_v14: pl.Tile[[2, 64], pl.FP16, pl.Mem.Vec] = pl.tile.mul(stage_24_v12, stage_24_v13)
            stage_24_v15: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_24_v14, 0, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_24_v16: pl.Tile[[32, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_24_v8, stage_24_v15, [0, 0])
            stage_24_v17: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.extract(stage_24_v14, 1, 0, [1, 64], target_memory=pl.Mem.Vec)
            stage_24_v18: pl.Tile[[32, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(stage_24_v16, stage_24_v17, [3, 0])
            stage_24_v19: pl.Tile[[16, 128], pl.FP16, pl.Mem.Vec] = pl.tile.reshape(stage_24_v18, [16, 128])
            stage_24_v20 = pl.tile.store(stage_24_v19, [stage_24_v7 * 16, 0], temporary_48)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_24 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_24, pl.INT32))
    record_25 = 23760 + core_id * 6
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
            flag_25 = other_25 * 128
            progress = csa_signal_wait(progress, flag_25, pl.cast(target_25, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_25_v14: pl.Scalar[pl.INDEX] = block_25
        stage_25_v15: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        stage_25_v16: pl.Scalar[pl.INDEX] = 0
        for (stage_25_v21, (stage_25_v32,)) in pl.range(stage_25_v15 // 6, init_values=(stage_25_v16,)):
            stage_25_v22: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_25_v21])
            stage_25_v23: pl.Scalar[pl.INDEX] = pl.max(stage_25_v32, pl.cast(stage_25_v22, pl.INDEX) // 4)
            stage_25_v24: pl.Scalar[pl.INDEX] = pl.yield_(stage_25_v23)
        stage_25_v17: pl.Scalar[pl.INDEX] = pl.min(stage_25_v24, 262144)
        stage_25_v18: pl.Scalar[pl.INDEX] = pl.max((stage_25_v17 + 8191) // 8192, 1)
        pl.system.set_ffts(temporary_50)
        stage_25_v19: pl.Scalar[pl.INDEX] = subslot_25
        for stage_25_v20 in pl.range(stage_25_v14, stage_25_v15 // 2 * stage_25_v18, 24):
            if stage_25_v15 < 48:
                stage_25_v33: pl.Scalar[pl.INDEX] = stage_25_v20 % (stage_25_v15 // 2) * 2
                stage_25_v34: pl.Scalar[pl.INDEX] = stage_25_v20 // (stage_25_v15 // 2)
                (stage_25_v53, stage_25_v54) = pl.yield_(stage_25_v34, stage_25_v33)
            else:
                stage_25_v35: pl.Scalar[pl.INDEX] = stage_25_v20 // stage_25_v18 * 2
                stage_25_v36: pl.Scalar[pl.INDEX] = stage_25_v20 % stage_25_v18
                (stage_25_v53, stage_25_v54) = pl.yield_(stage_25_v36, stage_25_v35)
            stage_25_v25: pl.Scalar[pl.INDEX] = stage_25_v54 // 6
            stage_25_v26: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_25_v54 + 1])
            stage_25_v27: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_25_v25])
            stage_25_v28: pl.Scalar[pl.INDEX] = pl.cast(stage_25_v27, pl.INDEX) // 4
            stage_25_v29: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_25_v28, (stage_25_v26 + 1) // 4), 262144), 0)
            stage_25_v30: pl.Scalar[pl.INDEX] = stage_25_v53 * 8192
            stage_25_v31: pl.Scalar[pl.INDEX] = 8192
            if stage_25_v30 < stage_25_v29:
                stage_25_v37: pl.Scalar[pl.INDEX] = pl.min(stage_25_v31, stage_25_v29 - stage_25_v30)
                stage_25_v38: pl.Scalar[pl.INDEX] = (stage_25_v37 + 767) // 768
                stage_25_v39: pl.Scalar[pl.INDEX] = pl.min(stage_25_v38 * 384, 4096)
                stage_25_v40: pl.Scalar[pl.INDEX] = (stage_25_v39 + 383) // 384
                stage_25_v41: pl.Scalar[pl.INDEX] = stage_25_v19 * stage_25_v39
                for stage_25_v42 in pl.range(stage_25_v40):
                    stage_25_v55: pl.Scalar[pl.INDEX] = stage_25_v42 * 384
                    stage_25_v56: pl.Scalar[pl.INDEX] = stage_25_v14 * 4 + stage_25_v42 % 2 * 2
                    stage_25_v57: pl.Tile[[1, 768], pl.INT8, pl.Mem.Vec] = pl.tile.create([1, 768], dtype=pl.INT8, target_memory=pl.Mem.Vec)
                    for (stage_25_v77, (stage_25_v108,)) in pl.range(12, init_values=(stage_25_v57,)):
                        stage_25_v78: pl.Scalar[pl.INDEX] = stage_25_v30 + stage_25_v55 + stage_25_v41 + stage_25_v77 * 32
                        stage_25_v79: pl.Scalar[pl.INDEX] = pl.min(stage_25_v78 // 32, pl.max((stage_25_v28 - 1) // 32, 0))
                        stage_25_v80: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_25_v25, stage_25_v79])
                        stage_25_v81: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_25_v80, pl.INDEX), 0)
                        stage_25_v82: pl.Tile[[1, 768], pl.INT8, pl.Mem.Vec] = pl.tile.gather_row(stage_25_v108, arg_idx_native_kv_cache, [0, stage_25_v77 * 64], [stage_25_v81, 4096], [1, 64], transpose=False)
                        stage_25_v83: pl.Tile[[1, 768], pl.INT8, pl.Mem.Vec] = pl.yield_(stage_25_v82)
                    stage_25_v58: pl.Tile[[1, 384], pl.FP16, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_25_v83, dtype=pl.FP16)
                    stage_25_v59: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_25_v58, target_type=pl.FP32, mode='round')
                    pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    stage_25_v60: pl.Tile[[2, 384], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_49, [stage_25_v56, stage_25_v19 * 384], [2, 384], [2, 384], target_memory=pl.Mem.Vec)
                    pl.system.sync_set(1, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV)
                    stage_25_v61: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_25_v54])
                    stage_25_v62: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_25_v28, (stage_25_v61 + 1) // 4), 262144), 0)
                    stage_25_v63: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_25_v62 - stage_25_v30 - stage_25_v55 - stage_25_v41, 384), 0)
                    stage_25_v64: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_25_v60, [1, 384], [0, 0])
                    stage_25_v65: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_25_v64, stage_25_v59)
                    if 0 < stage_25_v63:
                        stage_25_v84: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v63])] = pl.tile.set_validshape(stage_25_v65, 1, stage_25_v63)
                        stage_25_v85 = pl.tile.store(stage_25_v84, [stage_25_v14 * 4 + stage_25_v19, stage_25_v55], temporary_47)
                    stage_25_v66: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_25_v54 + 1])
                    stage_25_v67: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_25_v28, (stage_25_v66 + 1) // 4), 262144), 0)
                    stage_25_v68: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_25_v67 - stage_25_v30 - stage_25_v55 - stage_25_v41, 384), 0)
                    stage_25_v69: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_25_v60, [1, 384], [1, 0])
                    stage_25_v70: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_25_v69, stage_25_v59)
                    if 0 < stage_25_v68:
                        stage_25_v86: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v68])] = pl.tile.set_validshape(stage_25_v70, 1, stage_25_v68)
                        stage_25_v87 = pl.tile.store(stage_25_v86, [stage_25_v14 * 4 + stage_25_v19 + 2, stage_25_v55], temporary_47)
                stage_25_v43: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_25_v54])
                stage_25_v44: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_25_v28, (stage_25_v43 + 1) // 4), 262144), 0)
                stage_25_v45: pl.Scalar[pl.INDEX] = stage_25_v30 + stage_25_v41
                stage_25_v46: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_25_v44 - stage_25_v45, stage_25_v39), 0)
                stage_25_v47: pl.Scalar[pl.INDEX] = stage_25_v54 * 64 + stage_25_v53 * 2 + stage_25_v19
                if 0 < stage_25_v46:
                    stage_25_v71: pl.Scalar[pl.INT32] = pl.cast(stage_25_v45, pl.INT32)
                    if stage_25_v46 <= 512:
                        stage_25_v88: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                        stage_25_v89: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_25_v88, stage_25_v71)
                        stage_25_v90: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v46])] = pl.tile.load(temporary_47, [stage_25_v14 * 4 + stage_25_v19, 0], [1, 512], [1, stage_25_v46], target_memory=pl.Mem.Vec)
                        stage_25_v91: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_25_v90, pad_value=pl.PadValue.min)
                        stage_25_v92: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_25_v91, -3.4028234663852886e+38)
                        stage_25_v93: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_25_v89, dtype=pl.UINT32)
                        stage_25_v94: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_25_v92, stage_25_v93)
                        stage_25_v95: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v94, pl.const(64, pl.INT32))
                        stage_25_v96: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v95, pl.const(256, pl.INT32))
                        stage_25_v97 = pl.tile.store(stage_25_v96, [stage_25_v47, 0], temporary_46)
                    elif stage_25_v46 <= 1024:
                        stage_25_v109: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                        stage_25_v110: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_25_v109, stage_25_v71)
                        stage_25_v111: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v46])] = pl.tile.load(temporary_47, [stage_25_v14 * 4 + stage_25_v19, 0], [1, 1024], [1, stage_25_v46], target_memory=pl.Mem.Vec)
                        stage_25_v112: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_25_v111, pad_value=pl.PadValue.min)
                        stage_25_v113: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_25_v112, -3.4028234663852886e+38)
                        stage_25_v114: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_25_v110, dtype=pl.UINT32)
                        stage_25_v115: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_25_v113, stage_25_v114)
                        stage_25_v116: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v115, pl.const(64, pl.INT32))
                        stage_25_v117: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v116, pl.const(256, pl.INT32))
                        stage_25_v118: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v117, [1, 1024], [0, 0])
                        stage_25_v119: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v117, [1, 1024], [0, 1024])
                        stage_25_v120: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_25_v121: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_25_v118, stage_25_v119, stage_25_v120, exhausted=False)
                        stage_25_v122: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v121, [1, 1024], [0, 0])
                        stage_25_v123 = pl.tile.store(stage_25_v122, [stage_25_v47, 0], temporary_46)
                    elif stage_25_v46 <= 2048:
                        stage_25_v139: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                        stage_25_v140: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_25_v139, stage_25_v71)
                        stage_25_v141: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v46])] = pl.tile.load(temporary_47, [stage_25_v14 * 4 + stage_25_v19, 0], [1, 2048], [1, stage_25_v46], target_memory=pl.Mem.Vec)
                        stage_25_v142: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_25_v141, pad_value=pl.PadValue.min)
                        stage_25_v143: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_25_v142, -3.4028234663852886e+38)
                        stage_25_v144: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_25_v140, dtype=pl.UINT32)
                        stage_25_v145: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_25_v143, stage_25_v144)
                        stage_25_v146: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v145, pl.const(64, pl.INT32))
                        stage_25_v147: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v146, pl.const(256, pl.INT32))
                        stage_25_v148: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v147, pl.const(1024, pl.INT32))
                        stage_25_v149: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v148, [1, 1024], [0, 0])
                        stage_25_v150 = pl.tile.store(stage_25_v149, [stage_25_v47, 0], temporary_46)
                    elif stage_25_v46 <= 2560:
                        stage_25_v163: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2560], dtype=pl.INT32, descending=False)
                        stage_25_v164: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_25_v163, stage_25_v71)
                        stage_25_v165: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v46])] = pl.tile.load(temporary_47, [stage_25_v14 * 4 + stage_25_v19, 0], [1, 2560], [1, stage_25_v46], target_memory=pl.Mem.Vec)
                        stage_25_v166: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_25_v165, pad_value=pl.PadValue.min)
                        stage_25_v167: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_25_v166, -3.4028234663852886e+38)
                        stage_25_v168: pl.Tile[[1, 2560], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_25_v164, dtype=pl.UINT32)
                        stage_25_v169: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_25_v167, stage_25_v168)
                        stage_25_v170: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v169, pl.const(64, pl.INT32))
                        stage_25_v171: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v170, pl.const(256, pl.INT32))
                        stage_25_v172: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_25_v171, 1, 4096)
                        stage_25_v173: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v172, pl.const(1024, pl.INT32))
                        stage_25_v174: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v173, [1, 1024], [0, 0])
                        stage_25_v175: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v171, [1, 1024], [0, 4096])
                        stage_25_v176: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_25_v177: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_25_v175, stage_25_v174, stage_25_v176, exhausted=False)
                        stage_25_v178: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v177, [1, 1024], [0, 0])
                        stage_25_v179 = pl.tile.store(stage_25_v178, [stage_25_v47, 0], temporary_46)
                    elif stage_25_v46 <= 3072:
                        stage_25_v197: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 3072], dtype=pl.INT32, descending=False)
                        stage_25_v198: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_25_v197, stage_25_v71)
                        stage_25_v199: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v46])] = pl.tile.load(temporary_47, [stage_25_v14 * 4 + stage_25_v19, 0], [1, 3072], [1, stage_25_v46], target_memory=pl.Mem.Vec)
                        stage_25_v200: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_25_v199, pad_value=pl.PadValue.min)
                        stage_25_v201: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_25_v200, -3.4028234663852886e+38)
                        stage_25_v202: pl.Tile[[1, 3072], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_25_v198, dtype=pl.UINT32)
                        stage_25_v203: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_25_v201, stage_25_v202)
                        stage_25_v204: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v203, pl.const(64, pl.INT32))
                        stage_25_v205: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v204, pl.const(256, pl.INT32))
                        stage_25_v206: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_25_v205, 1, 4096)
                        stage_25_v207: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v206, pl.const(1024, pl.INT32))
                        stage_25_v208: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v207, [1, 1024], [0, 0])
                        stage_25_v209: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v205, [1, 1024], [0, 4096])
                        stage_25_v210: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v205, [1, 1024], [0, 5120])
                        stage_25_v211: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 3072], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_25_v212: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_25_v209, stage_25_v210, stage_25_v208, stage_25_v211, exhausted=False)
                        stage_25_v213: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v212, [1, 1024], [0, 0])
                        stage_25_v214 = pl.tile.store(stage_25_v213, [stage_25_v47, 0], temporary_46)
                    else:
                        stage_25_v215: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 4096], dtype=pl.INT32, descending=False)
                        stage_25_v216: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_25_v215, stage_25_v71)
                        stage_25_v217: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v46])] = pl.tile.load(temporary_47, [stage_25_v14 * 4 + stage_25_v19, 0], [1, 4096], [1, stage_25_v46], target_memory=pl.Mem.Vec)
                        stage_25_v218: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_25_v217, pad_value=pl.PadValue.min)
                        stage_25_v219: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_25_v218, -3.4028234663852886e+38)
                        stage_25_v220: pl.Tile[[1, 4096], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_25_v216, dtype=pl.UINT32)
                        stage_25_v221: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_25_v219, stage_25_v220)
                        stage_25_v222: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v221, pl.const(64, pl.INT32))
                        stage_25_v223: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v222, pl.const(256, pl.INT32))
                        stage_25_v224: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v223, pl.const(1024, pl.INT32))
                        stage_25_v225: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v224, [1, 1024], [0, 0])
                        stage_25_v226: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v224, [1, 1024], [0, 4096])
                        stage_25_v227: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_25_v228: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_25_v226, stage_25_v225, stage_25_v227, exhausted=False)
                        stage_25_v229: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v228, [1, 1024], [0, 0])
                        stage_25_v230 = pl.tile.store(stage_25_v229, [stage_25_v47, 0], temporary_46)
                else:
                    stage_25_v72: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
                    stage_25_v73 = pl.tile.store(stage_25_v72, [stage_25_v47, 0], temporary_46)
                stage_25_v48: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_25_v54 + 1])
                stage_25_v49: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_25_v28, (stage_25_v48 + 1) // 4), 262144), 0)
                stage_25_v50: pl.Scalar[pl.INDEX] = stage_25_v30 + stage_25_v41
                stage_25_v51: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_25_v49 - stage_25_v50, stage_25_v39), 0)
                stage_25_v52: pl.Scalar[pl.INDEX] = stage_25_v54 * 64 + stage_25_v53 * 2 + stage_25_v19 + 64
                if 0 < stage_25_v51:
                    stage_25_v74: pl.Scalar[pl.INT32] = pl.cast(stage_25_v50, pl.INT32)
                    if stage_25_v51 <= 512:
                        stage_25_v98: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                        stage_25_v99: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_25_v98, stage_25_v74)
                        stage_25_v100: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v51])] = pl.tile.load(temporary_47, [stage_25_v14 * 4 + stage_25_v19 + 2, 0], [1, 512], [1, stage_25_v51], target_memory=pl.Mem.Vec)
                        stage_25_v101: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_25_v100, pad_value=pl.PadValue.min)
                        stage_25_v102: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_25_v101, -3.4028234663852886e+38)
                        stage_25_v103: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_25_v99, dtype=pl.UINT32)
                        stage_25_v104: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_25_v102, stage_25_v103)
                        stage_25_v105: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v104, pl.const(64, pl.INT32))
                        stage_25_v106: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v105, pl.const(256, pl.INT32))
                        stage_25_v107 = pl.tile.store(stage_25_v106, [stage_25_v52, 0], temporary_46)
                    elif stage_25_v51 <= 1024:
                        stage_25_v124: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                        stage_25_v125: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_25_v124, stage_25_v74)
                        stage_25_v126: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v51])] = pl.tile.load(temporary_47, [stage_25_v14 * 4 + stage_25_v19 + 2, 0], [1, 1024], [1, stage_25_v51], target_memory=pl.Mem.Vec)
                        stage_25_v127: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_25_v126, pad_value=pl.PadValue.min)
                        stage_25_v128: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_25_v127, -3.4028234663852886e+38)
                        stage_25_v129: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_25_v125, dtype=pl.UINT32)
                        stage_25_v130: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_25_v128, stage_25_v129)
                        stage_25_v131: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v130, pl.const(64, pl.INT32))
                        stage_25_v132: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v131, pl.const(256, pl.INT32))
                        stage_25_v133: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v132, [1, 1024], [0, 0])
                        stage_25_v134: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v132, [1, 1024], [0, 1024])
                        stage_25_v135: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_25_v136: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_25_v133, stage_25_v134, stage_25_v135, exhausted=False)
                        stage_25_v137: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v136, [1, 1024], [0, 0])
                        stage_25_v138 = pl.tile.store(stage_25_v137, [stage_25_v52, 0], temporary_46)
                    elif stage_25_v51 <= 2048:
                        stage_25_v151: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                        stage_25_v152: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_25_v151, stage_25_v74)
                        stage_25_v153: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v51])] = pl.tile.load(temporary_47, [stage_25_v14 * 4 + stage_25_v19 + 2, 0], [1, 2048], [1, stage_25_v51], target_memory=pl.Mem.Vec)
                        stage_25_v154: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_25_v153, pad_value=pl.PadValue.min)
                        stage_25_v155: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_25_v154, -3.4028234663852886e+38)
                        stage_25_v156: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_25_v152, dtype=pl.UINT32)
                        stage_25_v157: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_25_v155, stage_25_v156)
                        stage_25_v158: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v157, pl.const(64, pl.INT32))
                        stage_25_v159: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v158, pl.const(256, pl.INT32))
                        stage_25_v160: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v159, pl.const(1024, pl.INT32))
                        stage_25_v161: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v160, [1, 1024], [0, 0])
                        stage_25_v162 = pl.tile.store(stage_25_v161, [stage_25_v52, 0], temporary_46)
                    elif stage_25_v51 <= 2560:
                        stage_25_v180: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2560], dtype=pl.INT32, descending=False)
                        stage_25_v181: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_25_v180, stage_25_v74)
                        stage_25_v182: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v51])] = pl.tile.load(temporary_47, [stage_25_v14 * 4 + stage_25_v19 + 2, 0], [1, 2560], [1, stage_25_v51], target_memory=pl.Mem.Vec)
                        stage_25_v183: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_25_v182, pad_value=pl.PadValue.min)
                        stage_25_v184: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_25_v183, -3.4028234663852886e+38)
                        stage_25_v185: pl.Tile[[1, 2560], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_25_v181, dtype=pl.UINT32)
                        stage_25_v186: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_25_v184, stage_25_v185)
                        stage_25_v187: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v186, pl.const(64, pl.INT32))
                        stage_25_v188: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v187, pl.const(256, pl.INT32))
                        stage_25_v189: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_25_v188, 1, 4096)
                        stage_25_v190: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v189, pl.const(1024, pl.INT32))
                        stage_25_v191: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v190, [1, 1024], [0, 0])
                        stage_25_v192: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v188, [1, 1024], [0, 4096])
                        stage_25_v193: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_25_v194: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_25_v192, stage_25_v191, stage_25_v193, exhausted=False)
                        stage_25_v195: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v194, [1, 1024], [0, 0])
                        stage_25_v196 = pl.tile.store(stage_25_v195, [stage_25_v52, 0], temporary_46)
                    elif stage_25_v51 <= 3072:
                        stage_25_v231: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 3072], dtype=pl.INT32, descending=False)
                        stage_25_v232: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_25_v231, stage_25_v74)
                        stage_25_v233: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v51])] = pl.tile.load(temporary_47, [stage_25_v14 * 4 + stage_25_v19 + 2, 0], [1, 3072], [1, stage_25_v51], target_memory=pl.Mem.Vec)
                        stage_25_v234: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_25_v233, pad_value=pl.PadValue.min)
                        stage_25_v235: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_25_v234, -3.4028234663852886e+38)
                        stage_25_v236: pl.Tile[[1, 3072], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_25_v232, dtype=pl.UINT32)
                        stage_25_v237: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_25_v235, stage_25_v236)
                        stage_25_v238: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v237, pl.const(64, pl.INT32))
                        stage_25_v239: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v238, pl.const(256, pl.INT32))
                        stage_25_v240: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_25_v239, 1, 4096)
                        stage_25_v241: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v240, pl.const(1024, pl.INT32))
                        stage_25_v242: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v241, [1, 1024], [0, 0])
                        stage_25_v243: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v239, [1, 1024], [0, 4096])
                        stage_25_v244: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v239, [1, 1024], [0, 5120])
                        stage_25_v245: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 3072], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_25_v246: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_25_v243, stage_25_v244, stage_25_v242, stage_25_v245, exhausted=False)
                        stage_25_v247: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v246, [1, 1024], [0, 0])
                        stage_25_v248 = pl.tile.store(stage_25_v247, [stage_25_v52, 0], temporary_46)
                    else:
                        stage_25_v249: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 4096], dtype=pl.INT32, descending=False)
                        stage_25_v250: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_25_v249, stage_25_v74)
                        stage_25_v251: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_25_v51])] = pl.tile.load(temporary_47, [stage_25_v14 * 4 + stage_25_v19 + 2, 0], [1, 4096], [1, stage_25_v51], target_memory=pl.Mem.Vec)
                        stage_25_v252: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_25_v251, pad_value=pl.PadValue.min)
                        stage_25_v253: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_25_v252, -3.4028234663852886e+38)
                        stage_25_v254: pl.Tile[[1, 4096], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_25_v250, dtype=pl.UINT32)
                        stage_25_v255: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_25_v253, stage_25_v254)
                        stage_25_v256: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v255, pl.const(64, pl.INT32))
                        stage_25_v257: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v256, pl.const(256, pl.INT32))
                        stage_25_v258: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_25_v257, pl.const(1024, pl.INT32))
                        stage_25_v259: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v258, [1, 1024], [0, 0])
                        stage_25_v260: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v258, [1, 1024], [0, 4096])
                        stage_25_v261: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_25_v262: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_25_v260, stage_25_v259, stage_25_v261, exhausted=False)
                        stage_25_v263: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_25_v262, [1, 1024], [0, 0])
                        stage_25_v264 = pl.tile.store(stage_25_v263, [stage_25_v52, 0], temporary_46)
                else:
                    stage_25_v75: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
                    stage_25_v76 = pl.tile.store(stage_25_v75, [stage_25_v52, 0], temporary_46)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_25 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_25, pl.INT32))
    record_26 = 24192 + core_id * 6
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
            flag_26 = other_26 * 128
            progress = csa_signal_wait(progress, flag_26, pl.cast(target_26, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_26_v7: pl.Scalar[pl.INDEX] = block_26
        for (stage_26_v8, (stage_26_v10,)) in pl.range(stage_26_v7, 32, 32, init_values=(arg_compress_state,)):
            for (stage_26_v11, (stage_26_v16,)) in pl.range(6, init_values=(stage_26_v10,)):
                stage_26_v12: pl.Scalar[pl.INDEX] = stage_26_v8 * 6 + stage_26_v11
                stage_26_v13: pl.Scalar[pl.INT32] = pl.tensor.read(arg_state_slot_mapping, [stage_26_v12, 0])
                stage_26_v14: pl.Scalar[pl.INT32] = pl.tensor.read(arg_state_slot_mapping, [stage_26_v12, 1])
                if 0 <= pl.cast(stage_26_v13, pl.INDEX) and 0 <= pl.cast(stage_26_v14, pl.INDEX):
                    stage_26_v17: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_26_v12])
                    stage_26_v18: pl.Scalar[pl.INDEX] = pl.cast(stage_26_v13, pl.INDEX)
                    stage_26_v19: pl.Scalar[pl.INDEX] = pl.cast(stage_26_v14, pl.INDEX) * 2048
                    stage_26_v20: pl.Scalar[pl.INDEX] = pl.cast(stage_26_v17 % 4, pl.INDEX)
                    stage_26_v21: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_29, [stage_26_v12, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_26_v22 = pl.tile.store(stage_26_v21, [stage_26_v18, stage_26_v19], stage_26_v16)
                    stage_26_v23: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_30, [stage_26_v12, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_26_v24: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_cmp_ape, [stage_26_v20, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_26_v25: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_26_v23, stage_26_v24)
                    stage_26_v26 = pl.tile.store(stage_26_v25, [stage_26_v18, stage_26_v19 + 1024], stage_26_v22)
                    stage_26_v27 = pl.yield_(stage_26_v26)
                else:
                    stage_26_v27 = pl.yield_(stage_26_v16)
                stage_26_v15 = pl.yield_(stage_26_v27)
            stage_26_v9 = pl.yield_(stage_26_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_26 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_26, pl.INT32))
    record_27 = 24624 + core_id * 6
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
            flag_27 = other_27 * 128
            progress = csa_signal_wait(progress, flag_27, pl.cast(target_27, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_27_v12: pl.Scalar[pl.INDEX] = block_27
        stage_27_v13: pl.Scalar[pl.INDEX] = stage_27_v12 * 16
        stage_27_v14: pl.Scalar[pl.INDEX] = pl.min(192 - stage_27_v13, 16)
        stage_27_v15: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        stage_27_v16: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
        for (stage_27_v71, (stage_27_v93, stage_27_v94)) in pl.range(stage_27_v14, init_values=(stage_27_v15, stage_27_v16)):
            stage_27_v72: pl.Scalar[pl.INDEX] = stage_27_v13 + stage_27_v71
            stage_27_v73: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_27_v72])
            if (stage_27_v73 + 1) % 4 == 0:
                stage_27_v95: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_14, [stage_27_v72 // 6])
                stage_27_v96: pl.Scalar[pl.INDEX] = pl.cast(stage_27_v95, pl.INDEX) + pl.cast((stage_27_v73 + 1) // 4, pl.INDEX)
                stage_27_v97: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_cmp_freqs_cos, 0)
                if stage_27_v96 < stage_27_v97:
                    stage_27_v112: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather_row(stage_27_v93, arg_cmp_freqs_cos, [stage_27_v71, 0], [stage_27_v96, 0], [1, 64], transpose=False)
                    stage_27_v113: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather_row(stage_27_v94, arg_cmp_freqs_sin, [stage_27_v71, 0], [stage_27_v96, 0], [1, 64], transpose=False)
                    (stage_27_v123, stage_27_v124) = pl.yield_(stage_27_v112, stage_27_v113)
                else:
                    (stage_27_v123, stage_27_v124) = pl.yield_(stage_27_v93, stage_27_v94)
                (stage_27_v114, stage_27_v115) = pl.yield_(stage_27_v123, stage_27_v124)
            else:
                (stage_27_v114, stage_27_v115) = pl.yield_(stage_27_v93, stage_27_v94)
            (stage_27_v98, stage_27_v99) = pl.yield_(stage_27_v114, stage_27_v115)
        stage_27_v17: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_27_v98
        stage_27_v18: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_27_v99
        stage_27_v19: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_28, [stage_27_v13, 0], [16, 256], [16, 256], target_memory=pl.Mem.Vec)
        stage_27_v20: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_28, [stage_27_v13, 256], [16, 256], [16, 256], target_memory=pl.Mem.Vec)
        stage_27_v21: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_27_v19, stage_27_v19)
        stage_27_v22: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_27_v20, stage_27_v20)
        stage_27_v23: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_27_v21, stage_27_v22)
        stage_27_v24: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_27_v23, [16, 128], [0, 0])
        stage_27_v25: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_27_v23, [16, 128], [0, 128])
        stage_27_v26: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_27_v24, stage_27_v25)
        stage_27_v27: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_27_v26, [16, 64], [0, 0])
        stage_27_v28: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_27_v26, [16, 64], [0, 64])
        stage_27_v29: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_27_v27, stage_27_v28)
        stage_27_v30: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        stage_27_v31: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_27_v29, stage_27_v30)
        stage_27_v32: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_27_v31, [1, 16])
        stage_27_v33: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_27_v32, 0.001953125)
        stage_27_v34: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_27_v33, [16, 1])
        stage_27_v35: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_27_v34, [1, 16])
        stage_27_v36: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_27_v35, 1e-06)
        stage_27_v37: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_27_v36, [16, 1])
        stage_27_v38: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_27_v37, [1, 16])
        stage_27_v39: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_27_v38)
        stage_27_v40: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_27_v39, [16, 1])
        stage_27_v41: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_27_v40, [1, 16])
        stage_27_v42: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_27_v41)
        stage_27_v43: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_27_v42, [16, 1])
        for (stage_27_v74, (stage_27_v100,)) in pl.range(0, 448, 64, init_values=(temporary_31,)):
            stage_27_v75: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_28, [stage_27_v13, stage_27_v74], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            stage_27_v76: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_60, [0, stage_27_v74], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
            stage_27_v77: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_27_v76, target_type=pl.FP32, mode='round')
            stage_27_v78: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_27_v75, stage_27_v43)
            stage_27_v79: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_27_v78, stage_27_v77)
            stage_27_v80 = pl.tile.store(stage_27_v79, [stage_27_v13, stage_27_v74], stage_27_v100)
            stage_27_v81 = pl.yield_(stage_27_v80)
        stage_27_v44: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_28, [stage_27_v13, 448], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        stage_27_v45: pl.Tile[[1, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_60, [0, 448], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
        stage_27_v46: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_27_v45, target_type=pl.FP32, mode='round')
        stage_27_v47: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_27_v44, stage_27_v43)
        stage_27_v48: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_27_v47, stage_27_v46)
        stage_27_v49: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        stage_27_v50: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_27_v51: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_27_v50, target_type=pl.FP32, mode='round')
        stage_27_v52: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_27_v49, stage_27_v51)
        stage_27_v53: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_27_v52, 0.5)
        stage_27_v54: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_27_v53, target_type=pl.INT32, mode='trunc')
        stage_27_v55: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_27_v54, target_type=pl.FP32, mode='round')
        stage_27_v56: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_27_v55, 2.0)
        stage_27_v57: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_27_v52, stage_27_v56)
        stage_27_v58: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_27_v52, 1.0)
        stage_27_v59: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_27_v57, 2.0)
        stage_27_v60: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_27_v58, stage_27_v59)
        stage_27_v61: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_27_v60, target_type=pl.INT32, mode='round')
        stage_27_v62: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        for (stage_27_v82, (stage_27_v101,)) in pl.range(16, init_values=(stage_27_v62,)):
            stage_27_v83: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_27_v48, [1, 64], [stage_27_v82, 0], [1, 64])
            stage_27_v84: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_27_v61, [1, 64], [stage_27_v82, 0], [1, 64])
            stage_27_v85: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            stage_27_v86: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_27_v83, stage_27_v84, stage_27_v85)
            stage_27_v87: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_27_v101, stage_27_v86, [stage_27_v82, 0])
            stage_27_v88: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_27_v87)
        stage_27_v63: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = stage_27_v88
        stage_27_v64: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_27_v57, 2.0)
        stage_27_v65: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_27_v64, 1.0)
        stage_27_v66: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_27_v18, stage_27_v65)
        stage_27_v67: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_27_v48, stage_27_v17)
        stage_27_v68: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_27_v63, stage_27_v66)
        stage_27_v69: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_27_v67, stage_27_v68)
        stage_27_v70 = pl.tile.store(stage_27_v69, [stage_27_v13, 448], stage_27_v81)
        for (stage_27_v89, (stage_27_v102, stage_27_v103)) in pl.range(stage_27_v14, init_values=(view_61, temporary_27)):
            stage_27_v90: pl.Scalar[pl.INDEX] = stage_27_v13 + stage_27_v89
            stage_27_v91: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_27_v90])
            stage_27_v92: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_seq_lens, [stage_27_v90 // 6])
            if 0 < pl.cast(stage_27_v92, pl.INDEX) and (stage_27_v91 + 1) % 4 == 0:
                stage_27_v104: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_14, [stage_27_v90 // 6])
                stage_27_v105: pl.Scalar[pl.INDEX] = pl.cast(stage_27_v104, pl.INDEX) + pl.cast((stage_27_v91 + 1) // 4, pl.INDEX)
                stage_27_v106: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_cmp_slot_mapping, 0)
                stage_27_v107: pl.Scalar[pl.INDEX] = pl.min(stage_27_v105, stage_27_v106 - 1)
                stage_27_v108: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_slot_mapping, [stage_27_v107, 0])
                stage_27_v109: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_slot_mapping, [stage_27_v107, 1])
                if stage_27_v105 < stage_27_v106 and 0 <= pl.cast(stage_27_v108, pl.INDEX) and (0 <= pl.cast(stage_27_v109, pl.INDEX)):
                    stage_27_v116: pl.Scalar[pl.INDEX] = pl.cast(stage_27_v108, pl.INDEX) * 32 + pl.cast(stage_27_v109, pl.INDEX)
                    stage_27_v117: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(stage_27_v70, [stage_27_v90, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    stage_27_v118 = pl.tile.store(stage_27_v117, [stage_27_v90, 0], stage_27_v103)
                    stage_27_v119: pl.Tile[[1, 512], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_27_v117, target_type=pl.BF16, mode='rint')
                    stage_27_v120 = pl.tile.store(stage_27_v119, [stage_27_v116, 0], stage_27_v102)
                    (stage_27_v125, stage_27_v126) = pl.yield_(stage_27_v120, stage_27_v118)
                else:
                    (stage_27_v125, stage_27_v126) = pl.yield_(stage_27_v102, stage_27_v103)
                (stage_27_v121, stage_27_v122) = pl.yield_(stage_27_v125, stage_27_v126)
            else:
                (stage_27_v121, stage_27_v122) = pl.yield_(stage_27_v102, stage_27_v103)
            (stage_27_v110, stage_27_v111) = pl.yield_(stage_27_v121, stage_27_v122)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_27 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_27, pl.INT32))
    record_28 = 25056 + core_id * 6
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
            flag_28 = other_28 * 128
            progress = csa_signal_wait(progress, flag_28, pl.cast(target_28, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_28_v7: pl.Scalar[pl.INDEX] = block_28
        for (stage_28_v8, (stage_28_v10,)) in pl.range(stage_28_v7, 32, 32, init_values=(arg_compress_state,)):
            for (stage_28_v11, (stage_28_v16,)) in pl.range(6, init_values=(stage_28_v10,)):
                stage_28_v12: pl.Scalar[pl.INDEX] = stage_28_v8 * 6 + stage_28_v11
                stage_28_v13: pl.Scalar[pl.INT32] = pl.tensor.read(arg_state_slot_mapping, [stage_28_v12, 0])
                stage_28_v14: pl.Scalar[pl.INT32] = pl.tensor.read(arg_state_slot_mapping, [stage_28_v12, 1])
                if 0 <= pl.cast(stage_28_v13, pl.INDEX) and 0 <= pl.cast(stage_28_v14, pl.INDEX):
                    stage_28_v17: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_28_v12])
                    stage_28_v18: pl.Scalar[pl.INDEX] = pl.cast(stage_28_v13, pl.INDEX)
                    stage_28_v19: pl.Scalar[pl.INDEX] = pl.cast(stage_28_v14, pl.INDEX) * 2048
                    stage_28_v20: pl.Scalar[pl.INDEX] = pl.cast(stage_28_v17 % 4, pl.INDEX)
                    stage_28_v21: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_29, [stage_28_v12, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_28_v22 = pl.tile.store(stage_28_v21, [stage_28_v18, stage_28_v19], stage_28_v16)
                    stage_28_v23: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_30, [stage_28_v12, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_28_v24: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_cmp_ape, [stage_28_v20, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_28_v25: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_28_v23, stage_28_v24)
                    stage_28_v26 = pl.tile.store(stage_28_v25, [stage_28_v18, stage_28_v19 + 1024], stage_28_v22)
                    stage_28_v27 = pl.yield_(stage_28_v26)
                else:
                    stage_28_v27 = pl.yield_(stage_28_v16)
                stage_28_v15 = pl.yield_(stage_28_v27)
            stage_28_v9 = pl.yield_(stage_28_v15)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_28 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_28, pl.INT32))
    record_29 = 25488 + core_id * 6
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
            flag_29 = other_29 * 128
            progress = csa_signal_wait(progress, flag_29, pl.cast(target_29, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
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
        for stage_29_v20 in pl.range(stage_29_v14, stage_29_v15 // 2 * stage_29_v18, 24):
            if stage_29_v15 < 48:
                stage_29_v33: pl.Scalar[pl.INDEX] = stage_29_v20 % (stage_29_v15 // 2) * 2
                stage_29_v34: pl.Scalar[pl.INDEX] = stage_29_v20 // (stage_29_v15 // 2)
                (stage_29_v53, stage_29_v54) = pl.yield_(stage_29_v34, stage_29_v33)
            else:
                stage_29_v35: pl.Scalar[pl.INDEX] = stage_29_v20 // stage_29_v18 * 2
                stage_29_v36: pl.Scalar[pl.INDEX] = stage_29_v20 % stage_29_v18
                (stage_29_v53, stage_29_v54) = pl.yield_(stage_29_v36, stage_29_v35)
            stage_29_v25: pl.Scalar[pl.INDEX] = stage_29_v54 // 6
            stage_29_v26: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v54 + 1])
            stage_29_v27: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_29_v25])
            stage_29_v28: pl.Scalar[pl.INDEX] = pl.cast(stage_29_v27, pl.INDEX) // 4
            stage_29_v29: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v26 + 1) // 4), 262144), 0)
            stage_29_v30: pl.Scalar[pl.INDEX] = stage_29_v53 * 8192
            stage_29_v31: pl.Scalar[pl.INDEX] = 8192
            if stage_29_v30 < stage_29_v29:
                stage_29_v37: pl.Scalar[pl.INDEX] = pl.min(stage_29_v31, stage_29_v29 - stage_29_v30)
                stage_29_v38: pl.Scalar[pl.INDEX] = (stage_29_v37 + 767) // 768
                stage_29_v39: pl.Scalar[pl.INDEX] = pl.min(stage_29_v38 * 384, 4096)
                stage_29_v40: pl.Scalar[pl.INDEX] = (stage_29_v39 + 383) // 384
                stage_29_v41: pl.Scalar[pl.INDEX] = stage_29_v19 * stage_29_v39
                for stage_29_v42 in pl.range(stage_29_v40):
                    stage_29_v55: pl.Scalar[pl.INDEX] = stage_29_v42 * 384
                    stage_29_v56: pl.Scalar[pl.INDEX] = stage_29_v14 * 4 + stage_29_v42 % 2 * 2
                    stage_29_v57: pl.Tile[[1, 768], pl.INT8, pl.Mem.Vec] = pl.tile.create([1, 768], dtype=pl.INT8, target_memory=pl.Mem.Vec)
                    for (stage_29_v77, (stage_29_v108,)) in pl.range(12, init_values=(stage_29_v57,)):
                        stage_29_v78: pl.Scalar[pl.INDEX] = stage_29_v30 + stage_29_v55 + stage_29_v41 + stage_29_v77 * 32
                        stage_29_v79: pl.Scalar[pl.INDEX] = pl.min(stage_29_v78 // 32, pl.max((stage_29_v28 - 1) // 32, 0))
                        stage_29_v80: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_29_v25, stage_29_v79])
                        stage_29_v81: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_29_v80, pl.INDEX), 0)
                        stage_29_v82: pl.Tile[[1, 768], pl.INT8, pl.Mem.Vec] = pl.tile.gather_row(stage_29_v108, arg_idx_native_kv_cache, [0, stage_29_v77 * 64], [stage_29_v81, 4096], [1, 64], transpose=False)
                        stage_29_v83: pl.Tile[[1, 768], pl.INT8, pl.Mem.Vec] = pl.yield_(stage_29_v82)
                    stage_29_v58: pl.Tile[[1, 384], pl.FP16, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v83, dtype=pl.FP16)
                    stage_29_v59: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_29_v58, target_type=pl.FP32, mode='round')
                    pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    stage_29_v60: pl.Tile[[2, 384], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_49, [stage_29_v56, stage_29_v19 * 384], [2, 384], [2, 384], target_memory=pl.Mem.Vec)
                    pl.system.sync_set(1, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV)
                    stage_29_v61: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v54])
                    stage_29_v62: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v61 + 1) // 4), 262144), 0)
                    stage_29_v63: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v62 - stage_29_v30 - stage_29_v55 - stage_29_v41, 384), 0)
                    stage_29_v64: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_29_v60, [1, 384], [0, 0])
                    stage_29_v65: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_29_v64, stage_29_v59)
                    if 0 < stage_29_v63:
                        stage_29_v84: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v63])] = pl.tile.set_validshape(stage_29_v65, 1, stage_29_v63)
                        stage_29_v85 = pl.tile.store(stage_29_v84, [stage_29_v14 * 4 + stage_29_v19, stage_29_v55], temporary_47)
                    stage_29_v66: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v54 + 1])
                    stage_29_v67: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v66 + 1) // 4), 262144), 0)
                    stage_29_v68: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v67 - stage_29_v30 - stage_29_v55 - stage_29_v41, 384), 0)
                    stage_29_v69: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_29_v60, [1, 384], [1, 0])
                    stage_29_v70: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_29_v69, stage_29_v59)
                    if 0 < stage_29_v68:
                        stage_29_v86: pl.Tile[[1, 384], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v68])] = pl.tile.set_validshape(stage_29_v70, 1, stage_29_v68)
                        stage_29_v87 = pl.tile.store(stage_29_v86, [stage_29_v14 * 4 + stage_29_v19 + 2, stage_29_v55], temporary_47)
                stage_29_v43: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v54])
                stage_29_v44: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v43 + 1) // 4), 262144), 0)
                stage_29_v45: pl.Scalar[pl.INDEX] = stage_29_v30 + stage_29_v41
                stage_29_v46: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v44 - stage_29_v45, stage_29_v39), 0)
                stage_29_v47: pl.Scalar[pl.INDEX] = stage_29_v54 * 64 + stage_29_v53 * 2 + stage_29_v19
                if 0 < stage_29_v46:
                    stage_29_v71: pl.Scalar[pl.INT32] = pl.cast(stage_29_v45, pl.INT32)
                    if stage_29_v46 <= 512:
                        stage_29_v88: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                        stage_29_v89: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v88, stage_29_v71)
                        stage_29_v90: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v46])] = pl.tile.load(temporary_47, [stage_29_v14 * 4 + stage_29_v19, 0], [1, 512], [1, stage_29_v46], target_memory=pl.Mem.Vec)
                        stage_29_v91: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v90, pad_value=pl.PadValue.min)
                        stage_29_v92: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v91, -3.4028234663852886e+38)
                        stage_29_v93: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v89, dtype=pl.UINT32)
                        stage_29_v94: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v92, stage_29_v93)
                        stage_29_v95: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v94, pl.const(64, pl.INT32))
                        stage_29_v96: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v95, pl.const(256, pl.INT32))
                        stage_29_v97 = pl.tile.store(stage_29_v96, [stage_29_v47, 0], temporary_46)
                    elif stage_29_v46 <= 1024:
                        stage_29_v109: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                        stage_29_v110: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v109, stage_29_v71)
                        stage_29_v111: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v46])] = pl.tile.load(temporary_47, [stage_29_v14 * 4 + stage_29_v19, 0], [1, 1024], [1, stage_29_v46], target_memory=pl.Mem.Vec)
                        stage_29_v112: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v111, pad_value=pl.PadValue.min)
                        stage_29_v113: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v112, -3.4028234663852886e+38)
                        stage_29_v114: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v110, dtype=pl.UINT32)
                        stage_29_v115: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v113, stage_29_v114)
                        stage_29_v116: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v115, pl.const(64, pl.INT32))
                        stage_29_v117: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v116, pl.const(256, pl.INT32))
                        stage_29_v118: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v117, [1, 1024], [0, 0])
                        stage_29_v119: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v117, [1, 1024], [0, 1024])
                        stage_29_v120: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v121: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v118, stage_29_v119, stage_29_v120, exhausted=False)
                        stage_29_v122: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v121, [1, 1024], [0, 0])
                        stage_29_v123 = pl.tile.store(stage_29_v122, [stage_29_v47, 0], temporary_46)
                    elif stage_29_v46 <= 2048:
                        stage_29_v139: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                        stage_29_v140: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v139, stage_29_v71)
                        stage_29_v141: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v46])] = pl.tile.load(temporary_47, [stage_29_v14 * 4 + stage_29_v19, 0], [1, 2048], [1, stage_29_v46], target_memory=pl.Mem.Vec)
                        stage_29_v142: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v141, pad_value=pl.PadValue.min)
                        stage_29_v143: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v142, -3.4028234663852886e+38)
                        stage_29_v144: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v140, dtype=pl.UINT32)
                        stage_29_v145: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v143, stage_29_v144)
                        stage_29_v146: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v145, pl.const(64, pl.INT32))
                        stage_29_v147: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v146, pl.const(256, pl.INT32))
                        stage_29_v148: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v147, pl.const(1024, pl.INT32))
                        stage_29_v149: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v148, [1, 1024], [0, 0])
                        stage_29_v150 = pl.tile.store(stage_29_v149, [stage_29_v47, 0], temporary_46)
                    elif stage_29_v46 <= 2560:
                        stage_29_v163: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2560], dtype=pl.INT32, descending=False)
                        stage_29_v164: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v163, stage_29_v71)
                        stage_29_v165: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v46])] = pl.tile.load(temporary_47, [stage_29_v14 * 4 + stage_29_v19, 0], [1, 2560], [1, stage_29_v46], target_memory=pl.Mem.Vec)
                        stage_29_v166: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v165, pad_value=pl.PadValue.min)
                        stage_29_v167: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v166, -3.4028234663852886e+38)
                        stage_29_v168: pl.Tile[[1, 2560], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v164, dtype=pl.UINT32)
                        stage_29_v169: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v167, stage_29_v168)
                        stage_29_v170: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v169, pl.const(64, pl.INT32))
                        stage_29_v171: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v170, pl.const(256, pl.INT32))
                        stage_29_v172: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v171, 1, 4096)
                        stage_29_v173: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v172, pl.const(1024, pl.INT32))
                        stage_29_v174: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v173, [1, 1024], [0, 0])
                        stage_29_v175: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v171, [1, 1024], [0, 4096])
                        stage_29_v176: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v177: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v175, stage_29_v174, stage_29_v176, exhausted=False)
                        stage_29_v178: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v177, [1, 1024], [0, 0])
                        stage_29_v179 = pl.tile.store(stage_29_v178, [stage_29_v47, 0], temporary_46)
                    elif stage_29_v46 <= 3072:
                        stage_29_v197: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 3072], dtype=pl.INT32, descending=False)
                        stage_29_v198: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v197, stage_29_v71)
                        stage_29_v199: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v46])] = pl.tile.load(temporary_47, [stage_29_v14 * 4 + stage_29_v19, 0], [1, 3072], [1, stage_29_v46], target_memory=pl.Mem.Vec)
                        stage_29_v200: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v199, pad_value=pl.PadValue.min)
                        stage_29_v201: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v200, -3.4028234663852886e+38)
                        stage_29_v202: pl.Tile[[1, 3072], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v198, dtype=pl.UINT32)
                        stage_29_v203: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v201, stage_29_v202)
                        stage_29_v204: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v203, pl.const(64, pl.INT32))
                        stage_29_v205: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v204, pl.const(256, pl.INT32))
                        stage_29_v206: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v205, 1, 4096)
                        stage_29_v207: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v206, pl.const(1024, pl.INT32))
                        stage_29_v208: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v207, [1, 1024], [0, 0])
                        stage_29_v209: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v205, [1, 1024], [0, 4096])
                        stage_29_v210: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v205, [1, 1024], [0, 5120])
                        stage_29_v211: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 3072], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v212: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v209, stage_29_v210, stage_29_v208, stage_29_v211, exhausted=False)
                        stage_29_v213: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v212, [1, 1024], [0, 0])
                        stage_29_v214 = pl.tile.store(stage_29_v213, [stage_29_v47, 0], temporary_46)
                    else:
                        stage_29_v215: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 4096], dtype=pl.INT32, descending=False)
                        stage_29_v216: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v215, stage_29_v71)
                        stage_29_v217: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v46])] = pl.tile.load(temporary_47, [stage_29_v14 * 4 + stage_29_v19, 0], [1, 4096], [1, stage_29_v46], target_memory=pl.Mem.Vec)
                        stage_29_v218: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v217, pad_value=pl.PadValue.min)
                        stage_29_v219: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v218, -3.4028234663852886e+38)
                        stage_29_v220: pl.Tile[[1, 4096], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v216, dtype=pl.UINT32)
                        stage_29_v221: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v219, stage_29_v220)
                        stage_29_v222: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v221, pl.const(64, pl.INT32))
                        stage_29_v223: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v222, pl.const(256, pl.INT32))
                        stage_29_v224: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v223, pl.const(1024, pl.INT32))
                        stage_29_v225: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v224, [1, 1024], [0, 0])
                        stage_29_v226: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v224, [1, 1024], [0, 4096])
                        stage_29_v227: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v228: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v226, stage_29_v225, stage_29_v227, exhausted=False)
                        stage_29_v229: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v228, [1, 1024], [0, 0])
                        stage_29_v230 = pl.tile.store(stage_29_v229, [stage_29_v47, 0], temporary_46)
                else:
                    stage_29_v72: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
                    stage_29_v73 = pl.tile.store(stage_29_v72, [stage_29_v47, 0], temporary_46)
                stage_29_v48: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_29_v54 + 1])
                stage_29_v49: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_29_v28, (stage_29_v48 + 1) // 4), 262144), 0)
                stage_29_v50: pl.Scalar[pl.INDEX] = stage_29_v30 + stage_29_v41
                stage_29_v51: pl.Scalar[pl.INDEX] = pl.max(pl.min(stage_29_v49 - stage_29_v50, stage_29_v39), 0)
                stage_29_v52: pl.Scalar[pl.INDEX] = stage_29_v54 * 64 + stage_29_v53 * 2 + stage_29_v19 + 64
                if 0 < stage_29_v51:
                    stage_29_v74: pl.Scalar[pl.INT32] = pl.cast(stage_29_v50, pl.INT32)
                    if stage_29_v51 <= 512:
                        stage_29_v98: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                        stage_29_v99: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v98, stage_29_v74)
                        stage_29_v100: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v51])] = pl.tile.load(temporary_47, [stage_29_v14 * 4 + stage_29_v19 + 2, 0], [1, 512], [1, stage_29_v51], target_memory=pl.Mem.Vec)
                        stage_29_v101: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v100, pad_value=pl.PadValue.min)
                        stage_29_v102: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v101, -3.4028234663852886e+38)
                        stage_29_v103: pl.Tile[[1, 512], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v99, dtype=pl.UINT32)
                        stage_29_v104: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v102, stage_29_v103)
                        stage_29_v105: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v104, pl.const(64, pl.INT32))
                        stage_29_v106: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v105, pl.const(256, pl.INT32))
                        stage_29_v107 = pl.tile.store(stage_29_v106, [stage_29_v52, 0], temporary_46)
                    elif stage_29_v51 <= 1024:
                        stage_29_v124: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False)
                        stage_29_v125: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v124, stage_29_v74)
                        stage_29_v126: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v51])] = pl.tile.load(temporary_47, [stage_29_v14 * 4 + stage_29_v19 + 2, 0], [1, 1024], [1, stage_29_v51], target_memory=pl.Mem.Vec)
                        stage_29_v127: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v126, pad_value=pl.PadValue.min)
                        stage_29_v128: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v127, -3.4028234663852886e+38)
                        stage_29_v129: pl.Tile[[1, 1024], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v125, dtype=pl.UINT32)
                        stage_29_v130: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v128, stage_29_v129)
                        stage_29_v131: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v130, pl.const(64, pl.INT32))
                        stage_29_v132: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v131, pl.const(256, pl.INT32))
                        stage_29_v133: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v132, [1, 1024], [0, 0])
                        stage_29_v134: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v132, [1, 1024], [0, 1024])
                        stage_29_v135: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v136: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v133, stage_29_v134, stage_29_v135, exhausted=False)
                        stage_29_v137: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v136, [1, 1024], [0, 0])
                        stage_29_v138 = pl.tile.store(stage_29_v137, [stage_29_v52, 0], temporary_46)
                    elif stage_29_v51 <= 2048:
                        stage_29_v151: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False)
                        stage_29_v152: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v151, stage_29_v74)
                        stage_29_v153: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v51])] = pl.tile.load(temporary_47, [stage_29_v14 * 4 + stage_29_v19 + 2, 0], [1, 2048], [1, stage_29_v51], target_memory=pl.Mem.Vec)
                        stage_29_v154: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v153, pad_value=pl.PadValue.min)
                        stage_29_v155: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v154, -3.4028234663852886e+38)
                        stage_29_v156: pl.Tile[[1, 2048], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v152, dtype=pl.UINT32)
                        stage_29_v157: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v155, stage_29_v156)
                        stage_29_v158: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v157, pl.const(64, pl.INT32))
                        stage_29_v159: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v158, pl.const(256, pl.INT32))
                        stage_29_v160: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v159, pl.const(1024, pl.INT32))
                        stage_29_v161: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v160, [1, 1024], [0, 0])
                        stage_29_v162 = pl.tile.store(stage_29_v161, [stage_29_v52, 0], temporary_46)
                    elif stage_29_v51 <= 2560:
                        stage_29_v180: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 2560], dtype=pl.INT32, descending=False)
                        stage_29_v181: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v180, stage_29_v74)
                        stage_29_v182: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v51])] = pl.tile.load(temporary_47, [stage_29_v14 * 4 + stage_29_v19 + 2, 0], [1, 2560], [1, stage_29_v51], target_memory=pl.Mem.Vec)
                        stage_29_v183: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v182, pad_value=pl.PadValue.min)
                        stage_29_v184: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v183, -3.4028234663852886e+38)
                        stage_29_v185: pl.Tile[[1, 2560], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v181, dtype=pl.UINT32)
                        stage_29_v186: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v184, stage_29_v185)
                        stage_29_v187: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v186, pl.const(64, pl.INT32))
                        stage_29_v188: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v187, pl.const(256, pl.INT32))
                        stage_29_v189: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v188, 1, 4096)
                        stage_29_v190: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v189, pl.const(1024, pl.INT32))
                        stage_29_v191: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v190, [1, 1024], [0, 0])
                        stage_29_v192: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v188, [1, 1024], [0, 4096])
                        stage_29_v193: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v194: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v192, stage_29_v191, stage_29_v193, exhausted=False)
                        stage_29_v195: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v194, [1, 1024], [0, 0])
                        stage_29_v196 = pl.tile.store(stage_29_v195, [stage_29_v52, 0], temporary_46)
                    elif stage_29_v51 <= 3072:
                        stage_29_v231: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 3072], dtype=pl.INT32, descending=False)
                        stage_29_v232: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v231, stage_29_v74)
                        stage_29_v233: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v51])] = pl.tile.load(temporary_47, [stage_29_v14 * 4 + stage_29_v19 + 2, 0], [1, 3072], [1, stage_29_v51], target_memory=pl.Mem.Vec)
                        stage_29_v234: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v233, pad_value=pl.PadValue.min)
                        stage_29_v235: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v234, -3.4028234663852886e+38)
                        stage_29_v236: pl.Tile[[1, 3072], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v232, dtype=pl.UINT32)
                        stage_29_v237: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v235, stage_29_v236)
                        stage_29_v238: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v237, pl.const(64, pl.INT32))
                        stage_29_v239: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v238, pl.const(256, pl.INT32))
                        stage_29_v240: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(stage_29_v239, 1, 4096)
                        stage_29_v241: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v240, pl.const(1024, pl.INT32))
                        stage_29_v242: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v241, [1, 1024], [0, 0])
                        stage_29_v243: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v239, [1, 1024], [0, 4096])
                        stage_29_v244: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v239, [1, 1024], [0, 5120])
                        stage_29_v245: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 3072], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v246: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v243, stage_29_v244, stage_29_v242, stage_29_v245, exhausted=False)
                        stage_29_v247: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v246, [1, 1024], [0, 0])
                        stage_29_v248 = pl.tile.store(stage_29_v247, [stage_29_v52, 0], temporary_46)
                    else:
                        stage_29_v249: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 4096], dtype=pl.INT32, descending=False)
                        stage_29_v250: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_29_v249, stage_29_v74)
                        stage_29_v251: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_29_v51])] = pl.tile.load(temporary_47, [stage_29_v14 * 4 + stage_29_v19 + 2, 0], [1, 4096], [1, stage_29_v51], target_memory=pl.Mem.Vec)
                        stage_29_v252: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(stage_29_v251, pad_value=pl.PadValue.min)
                        stage_29_v253: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(stage_29_v252, -3.4028234663852886e+38)
                        stage_29_v254: pl.Tile[[1, 4096], pl.UINT32, pl.Mem.Vec] = pl.tile.reinterpret_view(stage_29_v250, dtype=pl.UINT32)
                        stage_29_v255: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(stage_29_v253, stage_29_v254)
                        stage_29_v256: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v255, pl.const(64, pl.INT32))
                        stage_29_v257: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v256, pl.const(256, pl.INT32))
                        stage_29_v258: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(stage_29_v257, pl.const(1024, pl.INT32))
                        stage_29_v259: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v258, [1, 1024], [0, 0])
                        stage_29_v260: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v258, [1, 1024], [0, 4096])
                        stage_29_v261: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        stage_29_v262: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(stage_29_v260, stage_29_v259, stage_29_v261, exhausted=False)
                        stage_29_v263: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(stage_29_v262, [1, 1024], [0, 0])
                        stage_29_v264 = pl.tile.store(stage_29_v263, [stage_29_v52, 0], temporary_46)
                else:
                    stage_29_v75: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 1024], dtype=pl.FP32, value=-3.4028234663852886e+38)
                    stage_29_v76 = pl.tile.store(stage_29_v75, [stage_29_v52, 0], temporary_46)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_29 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_29, pl.INT32))
    record_30 = 25920 + core_id * 6
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
            flag_30 = other_30 * 128
            progress = csa_signal_wait(progress, flag_30, pl.cast(target_30, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_30_v8: pl.Scalar[pl.INDEX] = block_30
        stage_30_v9: pl.Scalar[pl.INDEX] = 4
        if 384 <= 192 and 192 % 384 == 0:
            stage_30_v11: pl.Scalar[pl.INDEX] = 64
            stage_30_v12: pl.Scalar[pl.INDEX] = pl.yield_(stage_30_v11)
        else:
            if 96 <= 192 and 192 % 96 == 0:
                stage_30_v16: pl.Scalar[pl.INDEX] = 16
                stage_30_v17: pl.Scalar[pl.INDEX] = pl.yield_(stage_30_v16)
            else:
                stage_30_v17: pl.Scalar[pl.INDEX] = pl.yield_(stage_30_v9)
            stage_30_v12: pl.Scalar[pl.INDEX] = pl.yield_(stage_30_v17)
        for stage_30_v10 in pl.range(stage_30_v8, (192 + 7) // 8 * (64 // stage_30_v12), 48):
            stage_30_v13: pl.Scalar[pl.INDEX] = stage_30_v10 % (64 // stage_30_v12) * stage_30_v12
            stage_30_v14: pl.Scalar[pl.INDEX] = stage_30_v10 // (64 // stage_30_v12) * 8
            stage_30_v15: pl.Scalar[pl.INDEX] = 0 + stage_30_v14
            if stage_30_v14 + 8 <= 192:
                stage_30_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_23, [stage_30_v14, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_30_v19: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_30_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_30_v20: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_20, [stage_30_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_30_v21: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_21, [stage_30_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_30_v22: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                stage_30_v23: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_30_v22, target_type=pl.FP32, mode='round')
                stage_30_v24: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_30_v23, 0.015625)
                stage_30_v25: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_30_v24, target_type=pl.INT32, mode='trunc')
                stage_30_v26: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_30_v25, target_type=pl.FP32, mode='round')
                stage_30_v27: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_30_v26, 64.0)
                stage_30_v28: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v21, [1, 512])
                stage_30_v29: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_30_v28, target_type=pl.FP32, mode='round')
                stage_30_v30: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_30_v29, stage_30_v27)
                stage_30_v31: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_30_v30, target_type=pl.INT32, mode='round')
                stage_30_v32: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 512], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_30_v33 in pl.pipeline(stage_30_v12, stage=2):
                    stage_30_v62: pl.Scalar[pl.INDEX] = stage_30_v13 + stage_30_v33
                    stage_30_v63: pl.Scalar[pl.INDEX] = stage_30_v62 * 512
                    stage_30_v64: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_25, [stage_30_v14, stage_30_v63], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    stage_30_v65: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wq_b_scale, [stage_30_v63], [512], [512], target_memory=pl.Mem.Vec)
                    stage_30_v66: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v65, [1, 512])
                    stage_30_v67: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_30_v64, target_type=pl.FP32, mode='none')
                    stage_30_v68: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_30_v67, stage_30_v18)
                    stage_30_v69: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_30_v68, stage_30_v66)
                    stage_30_v70: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_30_v69, stage_30_v69)
                    stage_30_v71: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_30_v72: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_30_v70, stage_30_v71)
                    stage_30_v73: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v72, [1, 8])
                    stage_30_v74: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_30_v73, 0.001953125)
                    stage_30_v75: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_30_v74, 1e-06)
                    stage_30_v76: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_30_v77: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(stage_30_v75, stage_30_v76)
                    stage_30_v78: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v77, [8, 1])
                    stage_30_v79: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_30_v69, [8, 448], [0, 0])
                    stage_30_v80: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_30_v79, stage_30_v78)
                    stage_30_v81: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_30_v80, target_type=pl.BF16, mode='rint')
                    stage_30_v82 = pl.tile.store(stage_30_v81, [stage_30_v15, stage_30_v63], view_40)
                    stage_30_v83: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_30_v69, [8, 64], [0, 448])
                    stage_30_v84: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_30_v83, stage_30_v78)
                    stage_30_v85: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v84, [1, 512])
                    stage_30_v86: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_30_v85, stage_30_v31, stage_30_v32)
                    stage_30_v87: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v86, [8, 64])
                    stage_30_v88: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_30_v84, stage_30_v19)
                    stage_30_v89: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_30_v87, stage_30_v20)
                    stage_30_v90: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_30_v88, stage_30_v89)
                    stage_30_v91: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_30_v90, target_type=pl.BF16, mode='rint')
                    stage_30_v92 = pl.tile.store(stage_30_v91, [stage_30_v15, stage_30_v63 + 448], view_40)
            else:
                stage_30_v34: pl.Scalar[pl.INDEX] = 192 - stage_30_v14
                stage_30_v35: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_23, [stage_30_v14, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_30_v36: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_30_v34, 64])] = pl.tile.load(arg_freqs_cos, [stage_30_v15, 0], [8, 64], [stage_30_v34, 64], target_memory=pl.Mem.Vec)
                stage_30_v37: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_30_v34, 64])] = pl.tile.load(temporary_20, [stage_30_v15, 0], [8, 64], [stage_30_v34, 64], target_memory=pl.Mem.Vec)
                stage_30_v38: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                stage_30_v39: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                stage_30_v40: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_30_v39, target_type=pl.FP32, mode='round')
                stage_30_v41: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_30_v38, stage_30_v40)
                stage_30_v42: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_30_v41, 0.5)
                stage_30_v43: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_30_v42, target_type=pl.INT32, mode='trunc')
                stage_30_v44: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_30_v43, target_type=pl.FP32, mode='round')
                stage_30_v45: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_30_v44, 2.0)
                stage_30_v46: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_30_v41, stage_30_v45)
                stage_30_v47: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_30_v41, 1.0)
                stage_30_v48: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_30_v46, 2.0)
                stage_30_v49: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_30_v47, stage_30_v48)
                stage_30_v50: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False)
                stage_30_v51: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_30_v50, target_type=pl.FP32, mode='round')
                stage_30_v52: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_30_v51, 64.0)
                stage_30_v53: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                stage_30_v54: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_30_v53, stage_30_v52)
                stage_30_v55: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_30_v56: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(stage_30_v54, 0, 1, stage_30_v55)
                stage_30_v57: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_30_v49, stage_30_v56)
                stage_30_v58: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_30_v57, target_type=pl.INT32, mode='round')
                stage_30_v59: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_30_v60: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_30_v61 in pl.range(4):
                    stage_30_v93: pl.Scalar[pl.INDEX] = stage_30_v13 + stage_30_v61
                    stage_30_v94: pl.Scalar[pl.INDEX] = stage_30_v93 * 512
                    stage_30_v95: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_25, [stage_30_v14, stage_30_v94], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    stage_30_v96: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wq_b_scale, [stage_30_v94], [512], [512], target_memory=pl.Mem.Vec)
                    stage_30_v97: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v96, [1, 512])
                    stage_30_v98: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_30_v95, target_type=pl.FP32, mode='none')
                    stage_30_v99: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_30_v98, stage_30_v35)
                    stage_30_v100: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_30_v99, stage_30_v97)
                    stage_30_v101: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_30_v100, stage_30_v100)
                    stage_30_v102: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_30_v101, stage_30_v59)
                    stage_30_v103: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v102, [1, 8])
                    stage_30_v104: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_30_v103, 0.001953125)
                    stage_30_v105: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v104, [8, 1])
                    stage_30_v106: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v105, [1, 8])
                    stage_30_v107: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_30_v106, 1e-06)
                    stage_30_v108: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v107, [8, 1])
                    stage_30_v109: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v108, [1, 8])
                    stage_30_v110: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_30_v109)
                    stage_30_v111: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v110, [8, 1])
                    stage_30_v112: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v111, [1, 8])
                    stage_30_v113: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_30_v112)
                    stage_30_v114: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_30_v113, [8, 1])
                    stage_30_v115: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_30_v100, [8, 448], [0, 0])
                    stage_30_v116: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_30_v115, stage_30_v114)
                    stage_30_v117: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_30_v116, target_type=pl.BF16, mode='rint')
                    stage_30_v118: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_30_v34, 448])] = pl.tile.set_validshape(stage_30_v117, stage_30_v34, 448)
                    stage_30_v119 = pl.tile.store(stage_30_v118, [stage_30_v15, stage_30_v94], view_40)
                    stage_30_v120: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_30_v100, [8, 64], [0, 448])
                    stage_30_v121: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_30_v120, stage_30_v114)
                    stage_30_v122: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_30_v121, stage_30_v58, stage_30_v60)
                    stage_30_v123: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_30_v34, 64])] = pl.tile.set_validshape(stage_30_v121, stage_30_v34, 64)
                    stage_30_v124: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_30_v34, 64])] = pl.tile.set_validshape(stage_30_v122, stage_30_v34, 64)
                    stage_30_v125: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_30_v34, 64])] = pl.tile.mul(stage_30_v123, stage_30_v36)
                    stage_30_v126: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_30_v34, 64])] = pl.tile.mul(stage_30_v124, stage_30_v37)
                    stage_30_v127: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_30_v34, 64])] = pl.tile.add(stage_30_v125, stage_30_v126)
                    stage_30_v128: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_30_v34, 64])] = pl.tile.cast(stage_30_v127, target_type=pl.BF16, mode='rint')
                    stage_30_v129: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_30_v34, 64])] = pl.tile.set_validshape(stage_30_v128, stage_30_v34, 64)
                    stage_30_v130 = pl.tile.store(stage_30_v129, [stage_30_v15, stage_30_v94 + 448], view_40)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_30 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_30, pl.INT32))
    record_31 = 26352 + core_id * 6
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
            flag_31 = other_31 * 128
            progress = csa_signal_wait(progress, flag_31, pl.cast(target_31, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_31_v8: pl.Scalar[pl.INDEX] = block_31
        stage_31_v9: pl.Scalar[pl.INDEX] = 4
        if 384 <= 192 and 192 % 384 == 0:
            stage_31_v11: pl.Scalar[pl.INDEX] = 64
            stage_31_v12: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v11)
        else:
            if 96 <= 192 and 192 % 96 == 0:
                stage_31_v16: pl.Scalar[pl.INDEX] = 16
                stage_31_v17: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v16)
            else:
                stage_31_v17: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v9)
            stage_31_v12: pl.Scalar[pl.INDEX] = pl.yield_(stage_31_v17)
        for stage_31_v10 in pl.range(stage_31_v8, (192 + 7) // 8 * (64 // stage_31_v12), 48):
            stage_31_v13: pl.Scalar[pl.INDEX] = stage_31_v10 % (64 // stage_31_v12) * stage_31_v12
            stage_31_v14: pl.Scalar[pl.INDEX] = stage_31_v10 // (64 // stage_31_v12) * 8
            stage_31_v15: pl.Scalar[pl.INDEX] = 0 + stage_31_v14
            if stage_31_v14 + 8 <= 192:
                stage_31_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_23, [stage_31_v14, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_31_v19: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_31_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_31_v20: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_20, [stage_31_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_31_v21: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_21, [stage_31_v15, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                stage_31_v22: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False)
                stage_31_v23: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_31_v22, target_type=pl.FP32, mode='round')
                stage_31_v24: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_31_v23, 0.015625)
                stage_31_v25: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_31_v24, target_type=pl.INT32, mode='trunc')
                stage_31_v26: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_31_v25, target_type=pl.FP32, mode='round')
                stage_31_v27: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_31_v26, 64.0)
                stage_31_v28: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v21, [1, 512])
                stage_31_v29: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_31_v28, target_type=pl.FP32, mode='round')
                stage_31_v30: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_31_v29, stage_31_v27)
                stage_31_v31: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_31_v30, target_type=pl.INT32, mode='round')
                stage_31_v32: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 512], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_31_v33 in pl.pipeline(stage_31_v12, stage=2):
                    stage_31_v62: pl.Scalar[pl.INDEX] = stage_31_v13 + stage_31_v33
                    stage_31_v63: pl.Scalar[pl.INDEX] = stage_31_v62 * 512
                    stage_31_v64: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_25, [stage_31_v14, stage_31_v63], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    stage_31_v65: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wq_b_scale, [stage_31_v63], [512], [512], target_memory=pl.Mem.Vec)
                    stage_31_v66: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v65, [1, 512])
                    stage_31_v67: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_31_v64, target_type=pl.FP32, mode='none')
                    stage_31_v68: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_31_v67, stage_31_v18)
                    stage_31_v69: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_31_v68, stage_31_v66)
                    stage_31_v70: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_31_v69, stage_31_v69)
                    stage_31_v71: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_31_v72: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_31_v70, stage_31_v71)
                    stage_31_v73: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v72, [1, 8])
                    stage_31_v74: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_31_v73, 0.001953125)
                    stage_31_v75: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_31_v74, 1e-06)
                    stage_31_v76: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_31_v77: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(stage_31_v75, stage_31_v76)
                    stage_31_v78: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v77, [8, 1])
                    stage_31_v79: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_31_v69, [8, 448], [0, 0])
                    stage_31_v80: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_31_v79, stage_31_v78)
                    stage_31_v81: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_31_v80, target_type=pl.BF16, mode='rint')
                    stage_31_v82 = pl.tile.store(stage_31_v81, [stage_31_v15, stage_31_v63], view_40)
                    stage_31_v83: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_31_v69, [8, 64], [0, 448])
                    stage_31_v84: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_31_v83, stage_31_v78)
                    stage_31_v85: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v84, [1, 512])
                    stage_31_v86: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_31_v85, stage_31_v31, stage_31_v32)
                    stage_31_v87: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v86, [8, 64])
                    stage_31_v88: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_31_v84, stage_31_v19)
                    stage_31_v89: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_31_v87, stage_31_v20)
                    stage_31_v90: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_31_v88, stage_31_v89)
                    stage_31_v91: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_31_v90, target_type=pl.BF16, mode='rint')
                    stage_31_v92 = pl.tile.store(stage_31_v91, [stage_31_v15, stage_31_v63 + 448], view_40)
            else:
                stage_31_v34: pl.Scalar[pl.INDEX] = 192 - stage_31_v14
                stage_31_v35: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_23, [stage_31_v14, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                stage_31_v36: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v34, 64])] = pl.tile.load(arg_freqs_cos, [stage_31_v15, 0], [8, 64], [stage_31_v34, 64], target_memory=pl.Mem.Vec)
                stage_31_v37: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v34, 64])] = pl.tile.load(temporary_20, [stage_31_v15, 0], [8, 64], [stage_31_v34, 64], target_memory=pl.Mem.Vec)
                stage_31_v38: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                stage_31_v39: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                stage_31_v40: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_31_v39, target_type=pl.FP32, mode='round')
                stage_31_v41: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_31_v38, stage_31_v40)
                stage_31_v42: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_31_v41, 0.5)
                stage_31_v43: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_31_v42, target_type=pl.INT32, mode='trunc')
                stage_31_v44: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_31_v43, target_type=pl.FP32, mode='round')
                stage_31_v45: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_31_v44, 2.0)
                stage_31_v46: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_31_v41, stage_31_v45)
                stage_31_v47: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_31_v41, 1.0)
                stage_31_v48: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_31_v46, 2.0)
                stage_31_v49: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_31_v47, stage_31_v48)
                stage_31_v50: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False)
                stage_31_v51: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_31_v50, target_type=pl.FP32, mode='round')
                stage_31_v52: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_31_v51, 64.0)
                stage_31_v53: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                stage_31_v54: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_31_v53, stage_31_v52)
                stage_31_v55: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_31_v56: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(stage_31_v54, 0, 1, stage_31_v55)
                stage_31_v57: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_31_v49, stage_31_v56)
                stage_31_v58: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_31_v57, target_type=pl.INT32, mode='round')
                stage_31_v59: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_31_v60: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for stage_31_v61 in pl.range(4):
                    stage_31_v93: pl.Scalar[pl.INDEX] = stage_31_v13 + stage_31_v61
                    stage_31_v94: pl.Scalar[pl.INDEX] = stage_31_v93 * 512
                    stage_31_v95: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_25, [stage_31_v14, stage_31_v94], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    stage_31_v96: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wq_b_scale, [stage_31_v94], [512], [512], target_memory=pl.Mem.Vec)
                    stage_31_v97: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v96, [1, 512])
                    stage_31_v98: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_31_v95, target_type=pl.FP32, mode='none')
                    stage_31_v99: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_31_v98, stage_31_v35)
                    stage_31_v100: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_31_v99, stage_31_v97)
                    stage_31_v101: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_31_v100, stage_31_v100)
                    stage_31_v102: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_31_v101, stage_31_v59)
                    stage_31_v103: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v102, [1, 8])
                    stage_31_v104: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_31_v103, 0.001953125)
                    stage_31_v105: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v104, [8, 1])
                    stage_31_v106: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v105, [1, 8])
                    stage_31_v107: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_31_v106, 1e-06)
                    stage_31_v108: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v107, [8, 1])
                    stage_31_v109: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v108, [1, 8])
                    stage_31_v110: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_31_v109)
                    stage_31_v111: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v110, [8, 1])
                    stage_31_v112: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v111, [1, 8])
                    stage_31_v113: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(stage_31_v112)
                    stage_31_v114: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_31_v113, [8, 1])
                    stage_31_v115: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_31_v100, [8, 448], [0, 0])
                    stage_31_v116: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_31_v115, stage_31_v114)
                    stage_31_v117: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_31_v116, target_type=pl.BF16, mode='rint')
                    stage_31_v118: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v34, 448])] = pl.tile.set_validshape(stage_31_v117, stage_31_v34, 448)
                    stage_31_v119 = pl.tile.store(stage_31_v118, [stage_31_v15, stage_31_v94], view_40)
                    stage_31_v120: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_31_v100, [8, 64], [0, 448])
                    stage_31_v121: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_31_v120, stage_31_v114)
                    stage_31_v122: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_31_v121, stage_31_v58, stage_31_v60)
                    stage_31_v123: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v34, 64])] = pl.tile.set_validshape(stage_31_v121, stage_31_v34, 64)
                    stage_31_v124: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v34, 64])] = pl.tile.set_validshape(stage_31_v122, stage_31_v34, 64)
                    stage_31_v125: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v34, 64])] = pl.tile.mul(stage_31_v123, stage_31_v36)
                    stage_31_v126: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v34, 64])] = pl.tile.mul(stage_31_v124, stage_31_v37)
                    stage_31_v127: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v34, 64])] = pl.tile.add(stage_31_v125, stage_31_v126)
                    stage_31_v128: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v34, 64])] = pl.tile.cast(stage_31_v127, target_type=pl.BF16, mode='rint')
                    stage_31_v129: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_31_v34, 64])] = pl.tile.set_validshape(stage_31_v128, stage_31_v34, 64)
                    stage_31_v130 = pl.tile.store(stage_31_v129, [stage_31_v15, stage_31_v94 + 448], view_40)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_31 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_31, pl.INT32))
    record_32 = 26784 + core_id * 6
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
            flag_32 = other_32 * 128
            progress = csa_signal_wait(progress, flag_32, pl.cast(target_32, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_32_v6: pl.Scalar[pl.INDEX] = block_32
        stage_32_v7: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        for stage_32_v8 in pl.range(stage_32_v6, stage_32_v7, 48):
            stage_32_v9: pl.Scalar[pl.INDEX] = stage_32_v8 // 6
            stage_32_v10: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_32_v8])
            stage_32_v11: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_32_v9])
            stage_32_v12: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v11, pl.INDEX) // 4
            stage_32_v13: pl.Scalar[pl.INDEX] = pl.min(stage_32_v12, (stage_32_v10 + 1) // 4)
            stage_32_v14: pl.Scalar[pl.INDEX] = pl.min(stage_32_v13, 262144)
            if 0 < stage_32_v14:
                stage_32_v15: pl.Scalar[pl.INDEX] = (stage_32_v14 + 8191) // 8192
                stage_32_v16: pl.Scalar[pl.INDEX] = stage_32_v15 * 2
                stage_32_v17: pl.Scalar[pl.INDEX] = stage_32_v8 * 64
                for stage_32_v18 in pl.range(1, stage_32_v16):
                    stage_32_v25: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_32_v17, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_32_v26: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_32_v17 + stage_32_v18, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_32_v27: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_32_v28: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(stage_32_v26, stage_32_v25, stage_32_v27, exhausted=False)
                    stage_32_v29: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_32_v28, [1, 1024], [0, 0])
                    pl.tile.store(stage_32_v29, [stage_32_v17, 0], temporary_46)
                stage_32_v19: pl.Scalar[pl.INDEX] = stage_32_v17
                stage_32_v20: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_32_v19, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                stage_32_v21: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather_mask(stage_32_v20, mask_pattern=1, output_dtype=pl.FP32)
                pl.tile.store(stage_32_v21, [stage_32_v8, 0], arg_idx_topk_scores)
                stage_32_v22: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.gather_mask(stage_32_v20, mask_pattern=2, output_dtype=pl.INT32)
                if 512 <= stage_32_v14:
                    pl.tile.store(stage_32_v22, [stage_32_v8, 0], arg_idx_topk)
                else:
                    stage_32_v30: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                    for stage_32_v31 in pl.range(stage_32_v14):
                        stage_32_v32: pl.Scalar[pl.INT32] = pl.tile.read(stage_32_v22, [0, stage_32_v31])
                        pl.tile.write(stage_32_v30, [0, stage_32_v31], stage_32_v32)
                    pl.tile.store(stage_32_v30, [stage_32_v8, 0], arg_idx_topk)
            else:
                stage_32_v23: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e+38)
                pl.tile.store(stage_32_v23, [stage_32_v8, 0], arg_idx_topk_scores)
                stage_32_v24: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                pl.tile.store(stage_32_v24, [stage_32_v8, 0], arg_idx_topk)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_32 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_32, pl.INT32))
    record_33 = 27216 + core_id * 6
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
            flag_33 = other_33 * 128
            progress = csa_signal_wait(progress, flag_33, pl.cast(target_33, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_33_v7: pl.Scalar[pl.INDEX] = block_33
        stage_33_v8: pl.Scalar[pl.INDEX] = stage_33_v7 * 32
        stage_33_v9: pl.Scalar[pl.INDEX] = pl.min(192 - stage_33_v8, 32)
        stage_33_v10: pl.Scalar[pl.INDEX] = 0 + stage_33_v8
        if stage_33_v9 == 32:
            stage_33_v11: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=0.0)
            for (stage_33_v85, (stage_33_v121,)) in pl.pipeline(0, 512, 64, stage=2, init_values=(stage_33_v11,)):
                stage_33_v86: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_26, [stage_33_v8, stage_33_v85], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
                stage_33_v87: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_33_v86, stage_33_v86)
                stage_33_v88: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_33_v121, stage_33_v87)
                stage_33_v89: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_33_v88)
            stage_33_v12: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_33_v13: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_33_v89, stage_33_v12)
            stage_33_v14: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_33_v13, [1, 32])
            stage_33_v15: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_33_v14, 0.001953125)
            stage_33_v16: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_33_v15, 1e-06)
            stage_33_v17: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_33_v16)
            stage_33_v18: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=1.0)
            stage_33_v19: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_33_v18, stage_33_v17)
            stage_33_v20: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_33_v19, [32, 1])
            for (stage_33_v90, (stage_33_v122,)) in pl.pipeline(0, 448, 64, stage=2, init_values=(temporary_17,)):
                stage_33_v91: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_26, [stage_33_v8, stage_33_v90], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
                stage_33_v92: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_gamma_ckv, [stage_33_v90], [64], [64], target_memory=pl.Mem.Vec)
                stage_33_v93: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_33_v92, target_type=pl.FP32, mode='round')
                stage_33_v94: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_33_v93, [1, 64])
                stage_33_v95: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_33_v91, stage_33_v20)
                stage_33_v96: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_33_v95, stage_33_v94)
                stage_33_v97: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_33_v96, target_type=pl.BF16, mode='rint')
                stage_33_v98 = pl.tile.store(stage_33_v97, [stage_33_v10, stage_33_v90], stage_33_v122)
                stage_33_v99 = pl.yield_(stage_33_v98)
            stage_33_v21: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_gamma_ckv, [448], [64], [64], target_memory=pl.Mem.Vec)
            stage_33_v22: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_33_v21, target_type=pl.FP32, mode='round')
            stage_33_v23: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_33_v22, [1, 64])
            stage_33_v24: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_26, [stage_33_v8, 448], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
            stage_33_v25: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_33_v24, stage_33_v20)
            stage_33_v26: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_33_v25, stage_33_v23)
            stage_33_v27: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_33_v10, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
            stage_33_v28: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_20, [stage_33_v10, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
            stage_33_v29: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_21, [stage_33_v10, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
            stage_33_v30: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for (stage_33_v100, (stage_33_v123,)) in pl.range(32, init_values=(stage_33_v30,)):
                stage_33_v101: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_33_v26, [1, 64], [stage_33_v100, 0], [1, 64])
                stage_33_v102: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.slice(stage_33_v29, [1, 64], [stage_33_v100, 0], [1, 64])
                stage_33_v103: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                stage_33_v104: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_33_v101, stage_33_v102, stage_33_v103)
                stage_33_v105: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(stage_33_v123, stage_33_v104, [stage_33_v100, 0])
                stage_33_v106: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_33_v105)
            stage_33_v31: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = stage_33_v106
            stage_33_v32: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_33_v26, stage_33_v27)
            stage_33_v33: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_33_v31, stage_33_v28)
            stage_33_v34: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_33_v32, stage_33_v33)
            stage_33_v35: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_33_v34, target_type=pl.BF16, mode='rint')
            stage_33_v36 = pl.tile.store(stage_33_v35, [stage_33_v10, 448], stage_33_v99)
        else:
            stage_33_v37: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_33_v38: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=0.0)
            for (stage_33_v107, (stage_33_v124,)) in pl.pipeline(0, 512, 64, stage=2, init_values=(stage_33_v38,)):
                stage_33_v108: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.load(temporary_26, [stage_33_v8, stage_33_v107], [32, 64], [stage_33_v9, 64], target_memory=pl.Mem.Vec)
                stage_33_v109: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.mul(stage_33_v108, stage_33_v108)
                stage_33_v110: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_33_v124, stage_33_v109)
                stage_33_v111: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_33_v110)
            stage_33_v39: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_33_v111, stage_33_v37)
            stage_33_v40: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_33_v39, [1, 32])
            stage_33_v41: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_33_v40, 0.001953125)
            stage_33_v42: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_33_v41, 1e-06)
            stage_33_v43: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(stage_33_v42)
            stage_33_v44: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=1.0)
            stage_33_v45: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_33_v44, stage_33_v43)
            stage_33_v46: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_33_v45, [32, 1])
            for stage_33_v47 in pl.pipeline(0, 448, 64, stage=2):
                stage_33_v112: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.load(temporary_26, [stage_33_v8, stage_33_v47], [32, 64], [stage_33_v9, 64], target_memory=pl.Mem.Vec)
                stage_33_v113: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_gamma_ckv, [stage_33_v47], [64], [64], target_memory=pl.Mem.Vec)
                stage_33_v114: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_33_v113, target_type=pl.FP32, mode='round')
                stage_33_v115: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_33_v114, [1, 64])
                stage_33_v116: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.row_expand_mul(stage_33_v112, stage_33_v46)
                stage_33_v117: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.col_expand_mul(stage_33_v116, stage_33_v115)
                stage_33_v118: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.cast(stage_33_v117, target_type=pl.BF16, mode='rint')
                stage_33_v119: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.set_validshape(stage_33_v118, stage_33_v9, 64)
                stage_33_v120 = pl.tile.store(stage_33_v119, [stage_33_v10, stage_33_v47], temporary_17)
            stage_33_v48: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(arg_gamma_ckv, [448], [64], [64], target_memory=pl.Mem.Vec)
            stage_33_v49: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_33_v48, target_type=pl.FP32, mode='round')
            stage_33_v50: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_33_v49, [1, 64])
            stage_33_v51: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.load(temporary_26, [stage_33_v8, 448], [32, 64], [stage_33_v9, 64], target_memory=pl.Mem.Vec)
            stage_33_v52: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.row_expand_mul(stage_33_v51, stage_33_v46)
            stage_33_v53: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.col_expand_mul(stage_33_v52, stage_33_v50)
            stage_33_v54: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.load(arg_freqs_cos, [stage_33_v10, 0], [32, 64], [stage_33_v9, 64], target_memory=pl.Mem.Vec)
            stage_33_v55: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.load(temporary_20, [stage_33_v10, 0], [32, 64], [stage_33_v9, 64], target_memory=pl.Mem.Vec)
            stage_33_v56: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=1.0)
            stage_33_v57: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            stage_33_v58: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_33_v57, target_type=pl.FP32, mode='round')
            stage_33_v59: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_33_v56, stage_33_v58)
            stage_33_v60: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_33_v59, 0.5)
            stage_33_v61: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_33_v60, target_type=pl.INT32, mode='trunc')
            stage_33_v62: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_33_v61, target_type=pl.FP32, mode='round')
            stage_33_v63: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_33_v62, 2.0)
            stage_33_v64: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_33_v59, stage_33_v63)
            stage_33_v65: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_33_v59, 1.0)
            stage_33_v66: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_33_v64, 2.0)
            stage_33_v67: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_33_v65, stage_33_v66)
            stage_33_v68: pl.Tile[[1, 32], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 32], dtype=pl.INT32, descending=False)
            stage_33_v69: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_33_v68, target_type=pl.FP32, mode='round')
            stage_33_v70: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_33_v69, 64.0)
            stage_33_v71: pl.Tile[[64, 32], pl.FP32, pl.Mem.Vec] = pl.tile.full([64, 32], dtype=pl.FP32, value=1.0)
            stage_33_v72: pl.Tile[[64, 32], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_33_v71, stage_33_v70)
            stage_33_v73: pl.Tile[[64, 32], pl.FP32, pl.Mem.Vec] = pl.tile.create([64, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_33_v74: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(stage_33_v72, 0, 1, stage_33_v73)
            stage_33_v75: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_33_v67, stage_33_v74)
            stage_33_v76: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_33_v75, target_type=pl.INT32, mode='round')
            stage_33_v77: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            stage_33_v78: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_33_v53, stage_33_v76, stage_33_v77)
            stage_33_v79: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.mul(stage_33_v53, stage_33_v54)
            stage_33_v80: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_33_v78, stage_33_v55)
            stage_33_v81: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.add(stage_33_v79, stage_33_v80)
            stage_33_v82: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.cast(stage_33_v81, target_type=pl.BF16, mode='rint')
            stage_33_v83: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_33_v9, 64])] = pl.tile.set_validshape(stage_33_v82, stage_33_v9, 64)
            stage_33_v84 = pl.tile.store(stage_33_v83, [stage_33_v10, 448], temporary_17)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_33 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_33, pl.INT32))
    record_34 = 27648 + core_id * 6
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
            flag_34 = other_34 * 128
            progress = csa_signal_wait(progress, flag_34, pl.cast(target_34, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_34_v6: pl.Scalar[pl.INDEX] = block_34
        stage_34_v7: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
        for stage_34_v8 in pl.range(stage_34_v6, stage_34_v7, 48):
            stage_34_v9: pl.Scalar[pl.INDEX] = stage_34_v8 // 6
            stage_34_v10: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_34_v8])
            stage_34_v11: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_34_v9])
            stage_34_v12: pl.Scalar[pl.INDEX] = pl.cast(stage_34_v11, pl.INDEX) // 4
            stage_34_v13: pl.Scalar[pl.INDEX] = pl.min(stage_34_v12, (stage_34_v10 + 1) // 4)
            stage_34_v14: pl.Scalar[pl.INDEX] = pl.min(stage_34_v13, 262144)
            if 0 < stage_34_v14:
                stage_34_v15: pl.Scalar[pl.INDEX] = (stage_34_v14 + 8191) // 8192
                stage_34_v16: pl.Scalar[pl.INDEX] = stage_34_v15 * 2
                stage_34_v17: pl.Scalar[pl.INDEX] = stage_34_v8 * 64
                for stage_34_v18 in pl.range(1, stage_34_v16):
                    stage_34_v25: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_34_v17, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_34_v26: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_34_v17 + stage_34_v18, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    stage_34_v27: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_34_v28: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(stage_34_v26, stage_34_v25, stage_34_v27, exhausted=False)
                    stage_34_v29: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_34_v28, [1, 1024], [0, 0])
                    pl.tile.store(stage_34_v29, [stage_34_v17, 0], temporary_46)
                stage_34_v19: pl.Scalar[pl.INDEX] = stage_34_v17
                stage_34_v20: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_46, [stage_34_v19, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                stage_34_v21: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather_mask(stage_34_v20, mask_pattern=1, output_dtype=pl.FP32)
                pl.tile.store(stage_34_v21, [stage_34_v8, 0], arg_idx_topk_scores)
                stage_34_v22: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.gather_mask(stage_34_v20, mask_pattern=2, output_dtype=pl.INT32)
                if 512 <= stage_34_v14:
                    pl.tile.store(stage_34_v22, [stage_34_v8, 0], arg_idx_topk)
                else:
                    stage_34_v30: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                    for stage_34_v31 in pl.range(stage_34_v14):
                        stage_34_v32: pl.Scalar[pl.INT32] = pl.tile.read(stage_34_v22, [0, stage_34_v31])
                        pl.tile.write(stage_34_v30, [0, stage_34_v31], stage_34_v32)
                    pl.tile.store(stage_34_v30, [stage_34_v8, 0], arg_idx_topk)
            else:
                stage_34_v23: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e+38)
                pl.tile.store(stage_34_v23, [stage_34_v8, 0], arg_idx_topk_scores)
                stage_34_v24: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                pl.tile.store(stage_34_v24, [stage_34_v8, 0], arg_idx_topk)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_34 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_34, pl.INT32))
    record_35 = 28080 + core_id * 6
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
            flag_35 = other_35 * 128
            progress = csa_signal_wait(progress, flag_35, pl.cast(target_35, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_35_v4: pl.Scalar[pl.INDEX] = block_35
        for (stage_35_v5, (stage_35_v8,)) in pl.range(stage_35_v4, 24, 8, init_values=(view_45,)):
            stage_35_v6: pl.Scalar[pl.INDEX] = stage_35_v5 * 8
            for (stage_35_v9, (stage_35_v14,)) in pl.range(pl.min(192 - stage_35_v6, 8), init_values=(stage_35_v8,)):
                stage_35_v10: pl.Scalar[pl.INDEX] = stage_35_v6 + stage_35_v9
                stage_35_v11: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_slot_mapping, [stage_35_v10, 0])
                stage_35_v12: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_slot_mapping, [stage_35_v10, 1])
                if 0 <= pl.cast(stage_35_v11, pl.INDEX) and 0 <= pl.cast(stage_35_v12, pl.INDEX):
                    stage_35_v15: pl.Scalar[pl.INDEX] = pl.cast(stage_35_v11, pl.INDEX) * 32 + pl.cast(stage_35_v12, pl.INDEX)
                    stage_35_v16: pl.Tile[[1, 512], pl.BF16, pl.Mem.Vec] = pl.tile.load(temporary_17, [stage_35_v10, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                    stage_35_v17 = pl.tile.store(stage_35_v16, [stage_35_v15, 0], stage_35_v14)
                    stage_35_v18 = pl.yield_(stage_35_v17)
                else:
                    stage_35_v18 = pl.yield_(stage_35_v14)
                stage_35_v13 = pl.yield_(stage_35_v18)
            stage_35_v7 = pl.yield_(stage_35_v13)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_35 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_35, pl.INT32))
    record_36 = 28512 + core_id * 6
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
            flag_36 = other_36 * 128
            progress = csa_signal_wait(progress, flag_36, pl.cast(target_36, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_36_v8: pl.Scalar[pl.INDEX] = block_36
        for (stage_36_v9, (stage_36_v63, stage_36_v64)) in pl.range(stage_36_v8 * 8, 192, 128, init_values=(temporary_53, temporary_52)):
            stage_36_v10: pl.Scalar[pl.INDEX] = pl.min(192 - stage_36_v9, 8)
            stage_36_v11: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.load(arg_idx_topk, [stage_36_v9, 0], [8, 512], [stage_36_v10, 512], target_memory=pl.Mem.Vec)
            stage_36_v12: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.cast(stage_36_v11, target_type=pl.FP32, mode='round')
            stage_36_v13: pl.Tile[[8, 1], pl.INT64, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 1])] = pl.tile.load(view_107, [stage_36_v9, 0], [8, 1], [stage_36_v10, 1], target_memory=pl.Mem.Vec)
            stage_36_v14: pl.Tile[[1, 8], pl.INT64, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_36_v10])] = pl.tile.reshape(stage_36_v13, [1, 8])
            stage_36_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_36_v10])] = pl.tile.cast(stage_36_v14, target_type=pl.FP32, mode='round')
            stage_36_v16: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 1])] = pl.tile.reshape(stage_36_v15, [8, 1])
            stage_36_v17: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_36_v10])] = pl.tile.reshape(stage_36_v16, [1, 8])
            stage_36_v18: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_36_v10])] = pl.tile.adds(stage_36_v17, 1.0)
            stage_36_v19: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 1])] = pl.tile.reshape(stage_36_v18, [8, 1])
            stage_36_v20: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_36_v10])] = pl.tile.reshape(stage_36_v19, [1, 8])
            stage_36_v21: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_36_v10])] = pl.tile.muls(stage_36_v20, 0.25)
            stage_36_v22: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 1])] = pl.tile.reshape(stage_36_v21, [8, 1])
            stage_36_v23: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_36_v10])] = pl.tile.reshape(stage_36_v22, [1, 8])
            stage_36_v24: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_36_v10])] = pl.tile.cast(stage_36_v23, target_type=pl.INT32, mode='trunc')
            stage_36_v25: pl.Tile[[8, 1], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 1])] = pl.tile.reshape(stage_36_v24, [8, 1])
            stage_36_v26: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_36_v10])] = pl.tile.reshape(stage_36_v25, [1, 8])
            stage_36_v27: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, stage_36_v10])] = pl.tile.cast(stage_36_v26, target_type=pl.FP32, mode='round')
            stage_36_v28: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 1])] = pl.tile.reshape(stage_36_v27, [8, 1])
            stage_36_v29: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=1.0)
            stage_36_v30: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_36_v29, stage_36_v28)
            stage_36_v31: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.adds(stage_36_v12, 1.0)
            stage_36_v32: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.maximums(stage_36_v31, 0.0)
            stage_36_v33: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.minimums(stage_36_v32, 1.0)
            stage_36_v34: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_36_v30, stage_36_v12)
            stage_36_v35: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.maximums(stage_36_v34, 0.0)
            stage_36_v36: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(stage_36_v35, 1.0)
            stage_36_v37: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.mul(stage_36_v33, stage_36_v36)
            stage_36_v38: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.adds(stage_36_v12, 1.0)
            stage_36_v39: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.mul(stage_36_v37, stage_36_v38)
            stage_36_v40: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.subs(stage_36_v39, 1.0)
            stage_36_v41: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.cast(stage_36_v40, target_type=pl.INT32, mode='round')
            stage_36_v42: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.set_validshape(stage_36_v41, stage_36_v10, 512)
            stage_36_v43 = pl.tile.store(stage_36_v42, [stage_36_v9, 0], stage_36_v63)
            stage_36_v44: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.minimums(stage_36_v40, 0.0)
            stage_36_v45: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.muls(stage_36_v44, 1e+20)
            stage_36_v46: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 512])] = pl.tile.set_validshape(stage_36_v45, stage_36_v10, 512)
            stage_36_v47 = pl.tile.store(stage_36_v46, [stage_36_v9, 128], stage_36_v64)
            for stage_36_v48 in pl.range(1, 5):
                stage_36_v65: pl.Scalar[pl.INDEX] = (stage_36_v48 - 1) * 128
                stage_36_v66: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(stage_36_v10, pl.const(8, pl.INDEX)), 128])] = pl.tile.slice(stage_36_v37, [8, 128], [0, stage_36_v65], [stage_36_v10, 128])
                stage_36_v67: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                stage_36_v68: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(stage_36_v10, pl.const(8, pl.INDEX)), 1])] = pl.tile.row_max(stage_36_v66, stage_36_v67)
                for stage_36_v69 in pl.range(stage_36_v10):
                    stage_36_v90: pl.Scalar[pl.FP32] = pl.tile.read(stage_36_v68, [stage_36_v69, 0])
                    stage_36_v91: pl.Scalar[pl.INT32] = pl.cast(stage_36_v90, pl.INT32)
                    pl.tensor.write(temporary_54, [stage_36_v9 + stage_36_v69, stage_36_v48], stage_36_v91)
            stage_36_v49: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 128], dtype=pl.INT32, descending=False)
            stage_36_v50: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_36_v49, target_type=pl.FP32, mode='round')
            stage_36_v51: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for stage_36_v52 in pl.range(8):
                stage_36_v70: pl.Scalar[pl.INDEX] = stage_36_v9 + pl.min(stage_36_v52, stage_36_v10 - 1)
                stage_36_v71: pl.Scalar[pl.INDEX] = stage_36_v70 // 6
                stage_36_v72: pl.Scalar[pl.INT64] = pl.tensor.read(view_107, [stage_36_v70, 0])
                stage_36_v73: pl.Scalar[pl.INDEX] = pl.cast(stage_36_v72, pl.INDEX)
                stage_36_v74: pl.Scalar[pl.INDEX] = pl.min(stage_36_v73 + 1, 128)
                stage_36_v75: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_36_v71])
                if pl.cast(stage_36_v75, pl.INDEX) <= 0:
                    stage_36_v92: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
                    stage_36_v93: pl.Scalar[pl.INDEX] = pl.yield_(stage_36_v92)
                else:
                    stage_36_v93: pl.Scalar[pl.INDEX] = pl.yield_(stage_36_v74)
                pl.tile.write(stage_36_v51, [stage_36_v52, 0], pl.cast(pl.cast(stage_36_v93, pl.INT32), pl.FP32))
            stage_36_v53: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 128], dtype=pl.FP32, value=1.0)
            stage_36_v54: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_36_v53, stage_36_v50)
            stage_36_v55: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_sub(stage_36_v54, stage_36_v51)
            stage_36_v56: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_36_v55)
            stage_36_v57: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.maximums(stage_36_v56, 0.0)
            stage_36_v58: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(stage_36_v57, 1.0)
            stage_36_v59: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_36_v58, 1.0)
            stage_36_v60: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_36_v59, 1e+20)
            stage_36_v61: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_36_v10, 128])] = pl.tile.set_validshape(stage_36_v60, stage_36_v10, 128)
            stage_36_v62 = pl.tile.store(stage_36_v61, [stage_36_v9, 0], stage_36_v47)
            for (stage_36_v76, (stage_36_v94,)) in pl.range(stage_36_v10, init_values=(stage_36_v62,)):
                stage_36_v77: pl.Scalar[pl.INDEX] = stage_36_v9 + stage_36_v76
                stage_36_v78: pl.Scalar[pl.INDEX] = stage_36_v77 // 6
                stage_36_v79: pl.Scalar[pl.INT64] = pl.tensor.read(view_107, [stage_36_v77, 0])
                stage_36_v80: pl.Scalar[pl.INDEX] = pl.cast(stage_36_v79, pl.INDEX)
                stage_36_v81: pl.Scalar[pl.INDEX] = pl.min(stage_36_v80 + 1, 128)
                stage_36_v82: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_36_v78])
                if pl.cast(stage_36_v82, pl.INDEX) <= 0:
                    stage_36_v95: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
                    stage_36_v96: pl.Scalar[pl.INDEX] = pl.yield_(stage_36_v95)
                else:
                    stage_36_v96: pl.Scalar[pl.INDEX] = pl.yield_(stage_36_v81)
                stage_36_v83: pl.Scalar[pl.INDEX] = stage_36_v80 - stage_36_v96 + 1
                stage_36_v84: pl.Scalar[pl.INDEX] = stage_36_v83 % 32
                stage_36_v85: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
                stage_36_v86: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
                for (stage_36_v97, (stage_36_v107, stage_36_v108)) in pl.range(5, init_values=(stage_36_v85, stage_36_v86)):
                    stage_36_v98: pl.Scalar[pl.INDEX] = pl.max(stage_36_v97 * 32 - stage_36_v84, 0)
                    stage_36_v99: pl.Scalar[pl.INDEX] = pl.min(stage_36_v97 * 32 - stage_36_v84 + 32, stage_36_v96)
                    if stage_36_v98 < stage_36_v99:
                        stage_36_v109: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_36_v78, (stage_36_v83 + stage_36_v98) // 32])
                        if 0 <= pl.cast(stage_36_v109, pl.INDEX):
                            stage_36_v116: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
                            (stage_36_v123, stage_36_v124) = pl.yield_(stage_36_v116, stage_36_v108)
                        else:
                            stage_36_v117: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
                            (stage_36_v123, stage_36_v124) = pl.yield_(stage_36_v107, stage_36_v117)
                        (stage_36_v118, stage_36_v119) = pl.yield_(stage_36_v123, stage_36_v124)
                    else:
                        (stage_36_v118, stage_36_v119) = pl.yield_(stage_36_v107, stage_36_v108)
                    (stage_36_v110, stage_36_v111) = pl.yield_(stage_36_v118, stage_36_v119)
                pl.tensor.write(temporary_54, [stage_36_v77, 0], stage_36_v110)
                if pl.cast(stage_36_v111, pl.INDEX) == 1:
                    stage_36_v100: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 128], dtype=pl.INT32, descending=False)
                    stage_36_v101: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_36_v100, target_type=pl.FP32, mode='round')
                    stage_36_v102: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
                    for (stage_36_v112, (stage_36_v120,)) in pl.range(5, init_values=(stage_36_v102,)):
                        stage_36_v113: pl.Scalar[pl.INDEX] = pl.max(stage_36_v112 * 32 - stage_36_v84, 0)
                        stage_36_v114: pl.Scalar[pl.INDEX] = pl.min(stage_36_v112 * 32 - stage_36_v84 + 32, stage_36_v96)
                        if stage_36_v113 < stage_36_v114:
                            stage_36_v121: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_36_v78, (stage_36_v83 + stage_36_v113) // 32])
                            if 0 <= pl.cast(stage_36_v121, pl.INDEX):
                                stage_36_v125: pl.Scalar[pl.FP32] = pl.cast(pl.cast(stage_36_v113, pl.INT32), pl.FP32)
                                stage_36_v126: pl.Scalar[pl.FP32] = pl.cast(pl.cast(stage_36_v114, pl.INT32), pl.FP32)
                                stage_36_v127: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_36_v101, stage_36_v125)
                                stage_36_v128: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_36_v127, 1.0)
                                stage_36_v129: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.maximums(stage_36_v128, 0.0)
                                stage_36_v130: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(stage_36_v129, 1.0)
                                stage_36_v131: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.neg(stage_36_v101)
                                stage_36_v132: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.adds(stage_36_v131, stage_36_v126)
                                stage_36_v133: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.maximums(stage_36_v132, 0.0)
                                stage_36_v134: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(stage_36_v133, 1.0)
                                stage_36_v135: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_36_v130, stage_36_v134)
                                stage_36_v136: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_36_v120, stage_36_v135)
                                stage_36_v137: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_36_v136)
                            else:
                                stage_36_v137: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_36_v120)
                            stage_36_v122: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_36_v137)
                        else:
                            stage_36_v122: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_36_v120)
                        stage_36_v115: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_36_v122)
                    stage_36_v103: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.subs(stage_36_v115, 1.0)
                    stage_36_v104: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_36_v103, 1e+20)
                    stage_36_v105 = pl.tile.store(stage_36_v104, [stage_36_v77, 0], stage_36_v94)
                    stage_36_v106 = pl.yield_(stage_36_v105)
                else:
                    stage_36_v106 = pl.yield_(stage_36_v94)
                stage_36_v87 = pl.yield_(stage_36_v106)
            (stage_36_v88, stage_36_v89) = pl.yield_(stage_36_v43, stage_36_v87)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_36 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_36, pl.INT32))
    record_37 = 28944 + core_id * 6
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
            flag_37 = other_37 * 128
            progress = csa_signal_wait(progress, flag_37, pl.cast(target_37, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_37_v1: pl.Tile[[1, 512], pl.BF16, pl.Mem.Vec] = pl.tile.load(view_45, [0, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
        stage_37_v2 = pl.tile.store(stage_37_v1, [0, 0], view_45)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_37 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_37, pl.INT32))
    record_38 = 29376 + core_id * 6
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
            flag_38 = other_38 * 128
            progress = csa_signal_wait(progress, flag_38, pl.cast(target_38, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_38_v22: pl.Scalar[pl.INDEX] = block_38
        pl.system.set_ffts(temporary_62)
        stage_38_v23: pl.Scalar[pl.INDEX] = (192 - stage_38_v22 + 23) // 24
        stage_38_v24: pl.Scalar[pl.INDEX] = stage_38_v23 * 5
        stage_38_v25: pl.Scalar[pl.INDEX] = subslot_38
        pl.system.set_ffts(temporary_62)
        stage_38_v26: pl.Scalar[pl.INDEX] = stage_38_v25 * 32
        stage_38_v27: pl.Scalar[pl.INDEX] = stage_38_v25 * 64
        stage_38_v28: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        stage_38_v29: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        stage_38_v30: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.rems(stage_38_v28, pl.const(2, pl.INT32), stage_38_v29)
        stage_38_v31: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_38_v30, pl.const(2, pl.INT32))
        stage_38_v32: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.sub(stage_38_v28, stage_38_v31)
        stage_38_v33: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.adds(stage_38_v32, pl.const(449, pl.INT32))
        stage_38_v34: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        stage_38_v35: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.col_expand(stage_38_v34, stage_38_v33)
        stage_38_v36: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 16], dtype=pl.INT32, descending=False)
        stage_38_v37: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.muls(stage_38_v36, pl.const(512, pl.INT32))
        stage_38_v38: pl.Tile[[16, 1], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v37, [16, 1])
        stage_38_v39: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.row_expand_add(stage_38_v35, stage_38_v38)
        stage_38_v40: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v39, [1, 1024])
        stage_38_v41: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_117, [stage_38_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
        stage_38_v42: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v41, [1, 32])
        stage_38_v43: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_38_v42, 0.0)
        stage_38_v44: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v43, [32, 1])
        stage_38_v45: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        stage_38_v46: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        for (stage_38_v47, (stage_38_v51, stage_38_v52, stage_38_v53, stage_38_v54)) in pl.range(stage_38_v24 + 3, init_values=(stage_38_v41, stage_38_v44, stage_38_v45, stage_38_v46)):
            stage_38_v48: pl.Scalar[pl.INDEX] = stage_38_v22 + stage_38_v47 // 5 * 24
            stage_38_v49: pl.Scalar[pl.INDEX] = stage_38_v48 // 6
            stage_38_v50: pl.Scalar[pl.INDEX] = stage_38_v47 % 5
            if stage_38_v47 < stage_38_v24:
                stage_38_v55: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_38_v48, stage_38_v50])
                if 0 < pl.cast(stage_38_v55, pl.INDEX):
                    stage_38_v72: pl.Scalar[pl.INDEX] = stage_38_v22 * 3 + stage_38_v47 % 3
                    stage_38_v73: pl.Scalar[pl.INDEX] = stage_38_v72 * 128
                    stage_38_v74: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
                    if stage_38_v50 == 0:
                        stage_38_v256: pl.Scalar[pl.INT64] = pl.tensor.read(view_107, [stage_38_v48, 0])
                        stage_38_v257: pl.Scalar[pl.INDEX] = pl.cast(stage_38_v256, pl.INDEX)
                        stage_38_v258: pl.Scalar[pl.INDEX] = pl.min(stage_38_v257 + 1, 128)
                        stage_38_v259: pl.Scalar[pl.INDEX] = stage_38_v257 - stage_38_v258 + 1
                        stage_38_v260: pl.Scalar[pl.INDEX] = stage_38_v259 % 32
                        stage_38_v261: pl.Scalar[pl.INDEX] = pl.min(pl.max(stage_38_v258 - stage_38_v27, 0), 64)
                        stage_38_v262: pl.Scalar[pl.INDEX] = 0
                        stage_38_v263: pl.Scalar[pl.INDEX] = pl.min(32 - stage_38_v260 - stage_38_v27, stage_38_v261)
                        if stage_38_v262 < stage_38_v263:
                            stage_38_v273: pl.Scalar[pl.INDEX] = stage_38_v259 + stage_38_v27 + stage_38_v262
                            stage_38_v274: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_38_v49, stage_38_v273 // 32])
                            if 0 <= pl.cast(stage_38_v274, pl.INDEX):
                                stage_38_v291: pl.Scalar[pl.INDEX] = pl.cast(stage_38_v274, pl.INDEX) * 32 + stage_38_v273 % 32
                                stage_38_v292: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_38_v74, view_45, [stage_38_v262, 0], [stage_38_v291, 0], [64, 512], valid_shape=[stage_38_v263 - stage_38_v262, 512], transpose=False)
                                stage_38_v293: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v292
                            else:
                                stage_38_v293: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v74
                            stage_38_v275: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v293
                        else:
                            stage_38_v275: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v74
                        stage_38_v264: pl.Scalar[pl.INDEX] = pl.max(32 - stage_38_v260 - stage_38_v27, 0)
                        stage_38_v265: pl.Scalar[pl.INDEX] = pl.min(64 - stage_38_v260 - stage_38_v27, stage_38_v261)
                        if stage_38_v264 < stage_38_v265:
                            stage_38_v276: pl.Scalar[pl.INDEX] = stage_38_v259 + stage_38_v27 + stage_38_v264
                            stage_38_v277: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_38_v49, stage_38_v276 // 32])
                            if 0 <= pl.cast(stage_38_v277, pl.INDEX):
                                stage_38_v294: pl.Scalar[pl.INDEX] = pl.cast(stage_38_v277, pl.INDEX) * 32 + stage_38_v276 % 32
                                stage_38_v295: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_38_v275, view_45, [stage_38_v264, 0], [stage_38_v294, 0], [64, 512], valid_shape=[stage_38_v265 - stage_38_v264, 512], transpose=False)
                                stage_38_v296: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v295
                            else:
                                stage_38_v296: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v275
                            stage_38_v278: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v296
                        else:
                            stage_38_v278: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v275
                        stage_38_v266: pl.Scalar[pl.INDEX] = pl.max(64 - stage_38_v260 - stage_38_v27, 0)
                        stage_38_v267: pl.Scalar[pl.INDEX] = pl.min(96 - stage_38_v260 - stage_38_v27, stage_38_v261)
                        if stage_38_v266 < stage_38_v267:
                            stage_38_v279: pl.Scalar[pl.INDEX] = stage_38_v259 + stage_38_v27 + stage_38_v266
                            stage_38_v280: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_38_v49, stage_38_v279 // 32])
                            if 0 <= pl.cast(stage_38_v280, pl.INDEX):
                                stage_38_v297: pl.Scalar[pl.INDEX] = pl.cast(stage_38_v280, pl.INDEX) * 32 + stage_38_v279 % 32
                                stage_38_v298: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_38_v278, view_45, [stage_38_v266, 0], [stage_38_v297, 0], [64, 512], valid_shape=[stage_38_v267 - stage_38_v266, 512], transpose=False)
                                stage_38_v299: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v298
                            else:
                                stage_38_v299: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v278
                            stage_38_v281: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v299
                        else:
                            stage_38_v281: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v278
                        stage_38_v268: pl.Scalar[pl.INDEX] = pl.max(96 - stage_38_v260 - stage_38_v27, 0)
                        stage_38_v269: pl.Scalar[pl.INDEX] = pl.min(128 - stage_38_v260 - stage_38_v27, stage_38_v261)
                        if stage_38_v268 < stage_38_v269:
                            stage_38_v282: pl.Scalar[pl.INDEX] = stage_38_v259 + stage_38_v27 + stage_38_v268
                            stage_38_v283: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_38_v49, stage_38_v282 // 32])
                            if 0 <= pl.cast(stage_38_v283, pl.INDEX):
                                stage_38_v300: pl.Scalar[pl.INDEX] = pl.cast(stage_38_v283, pl.INDEX) * 32 + stage_38_v282 % 32
                                stage_38_v301: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_38_v281, view_45, [stage_38_v268, 0], [stage_38_v300, 0], [64, 512], valid_shape=[stage_38_v269 - stage_38_v268, 512], transpose=False)
                                stage_38_v302: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v301
                            else:
                                stage_38_v302: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v281
                            stage_38_v284: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v302
                        else:
                            stage_38_v284: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v281
                        stage_38_v270: pl.Scalar[pl.INDEX] = pl.max(128 - stage_38_v260 - stage_38_v27, 0)
                        stage_38_v271: pl.Scalar[pl.INDEX] = pl.min(160 - stage_38_v260 - stage_38_v27, stage_38_v261)
                        if stage_38_v270 < stage_38_v271:
                            stage_38_v285: pl.Scalar[pl.INDEX] = stage_38_v259 + stage_38_v27 + stage_38_v270
                            stage_38_v286: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table, [stage_38_v49, stage_38_v285 // 32])
                            if 0 <= pl.cast(stage_38_v286, pl.INDEX):
                                stage_38_v303: pl.Scalar[pl.INDEX] = pl.cast(stage_38_v286, pl.INDEX) * 32 + stage_38_v285 % 32
                                stage_38_v304: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_38_v284, view_45, [stage_38_v270, 0], [stage_38_v303, 0], [64, 512], valid_shape=[stage_38_v271 - stage_38_v270, 512], transpose=False)
                                stage_38_v305: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v304
                            else:
                                stage_38_v305: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v284
                            stage_38_v287: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v305
                        else:
                            stage_38_v287: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v284
                        stage_38_v272: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v287
                    else:
                        stage_38_v74: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
                        for (stage_38_v288, (stage_38_v306,)) in pl.range(64, init_values=(stage_38_v74,)):
                            stage_38_v289: pl.Scalar[pl.INDEX] = (stage_38_v50 - 1) * 128 + stage_38_v27 + stage_38_v288
                            if stage_38_v289 < 512:
                                stage_38_v307: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_53, [stage_38_v48, stage_38_v289])
                                if 0 <= pl.cast(stage_38_v307, pl.INDEX):
                                    stage_38_v309: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_block_table, [stage_38_v49, pl.cast(stage_38_v307, pl.INDEX) // 32])
                                    stage_38_v310: pl.Scalar[pl.INDEX] = pl.cast(stage_38_v309, pl.INDEX)
                                    stage_38_v311: pl.Scalar[pl.INDEX] = stage_38_v310 * 32 + pl.cast(stage_38_v307, pl.INDEX) % 32
                                    stage_38_v312: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(stage_38_v306, view_61, [stage_38_v288, 0], [stage_38_v311, 0], [1, 512], transpose=False)
                                    stage_38_v313: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v312
                                else:
                                    stage_38_v313: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v306
                                stage_38_v308: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v313
                            else:
                                stage_38_v308: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v306
                            stage_38_v290: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.yield_(stage_38_v308)
                        stage_38_v272: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = stage_38_v290
                    stage_38_v75 = pl.tile.store(stage_38_v272, [stage_38_v73 + stage_38_v27, 0], temporary_56)
            if 144 <= 192:
                if 0 < stage_38_v47 and stage_38_v47 <= stage_38_v24:
                    stage_38_v76: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_38_v22 + (stage_38_v47 - 1) // 5 * 24, (stage_38_v47 - 1) % 5])
                    if 0 < pl.cast(stage_38_v76, pl.INDEX):
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                if stage_38_v47 < stage_38_v24:
                    stage_38_v77: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_38_v48, stage_38_v50])
                    if 0 < pl.cast(stage_38_v77, pl.INDEX):
                        pl.system.sync_set(0, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if 0 < stage_38_v47 and stage_38_v47 <= stage_38_v24:
                stage_38_v56: pl.Scalar[pl.INDEX] = stage_38_v47 - 1
                stage_38_v57: pl.Scalar[pl.INDEX] = stage_38_v56 % 5
                stage_38_v58: pl.Scalar[pl.INDEX] = stage_38_v22 + stage_38_v56 // 5 * 24
                stage_38_v59: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_38_v58, stage_38_v57])
                if 0 < pl.cast(stage_38_v59, pl.INDEX):
                    stage_38_v78: pl.Scalar[pl.INDEX] = stage_38_v22 * 3 + stage_38_v56 % 3
                    stage_38_v79: pl.Scalar[pl.INDEX] = stage_38_v78 * 64
                    stage_38_v80: pl.Scalar[pl.INDEX] = stage_38_v57 * 128
                    if 192 < 144:
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    stage_38_v81: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    stage_38_v82: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_57, [stage_38_v79 + stage_38_v26, 0], [32, 128], [32, 128], target_memory=pl.Mem.Vec)
                    stage_38_v83: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_52, [stage_38_v58, stage_38_v80], [1, 128], [1, 128], target_memory=pl.Mem.Vec)
                    stage_38_v84: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_38_v82, 0.04419417382415922)
                    stage_38_v85: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_add(stage_38_v84, stage_38_v83)
                    stage_38_v86: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_38_v85, stage_38_v81)
                    stage_38_v87: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_sub(stage_38_v85, stage_38_v86)
                    stage_38_v88: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_38_v87)
                    stage_38_v89: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(stage_38_v88, stage_38_v81)
                    stage_38_v90: pl.Tile[[32, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_38_v88, target_type=pl.BF16, mode='rint')
                    stage_38_v91 = pl.tile.store(stage_38_v90, [stage_38_v79 + stage_38_v26, 0], temporary_58)
                    stage_38_v92 = pl.tile.store(stage_38_v86, [stage_38_v79 + stage_38_v26, 0], temporary_60)
                    stage_38_v93 = pl.tile.store(stage_38_v89, [stage_38_v79 + stage_38_v26, 0], temporary_61)
                    pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if 192 < 144:
                if stage_38_v47 < stage_38_v24:
                    stage_38_v94: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_38_v48, stage_38_v50])
                    if 0 < pl.cast(stage_38_v94, pl.INDEX):
                        pl.system.sync_set(0, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if 3 <= stage_38_v47:
                stage_38_v60: pl.Scalar[pl.INDEX] = stage_38_v47 - 2 - 1
                stage_38_v61: pl.Scalar[pl.INDEX] = stage_38_v60 % 5
                stage_38_v62: pl.Scalar[pl.INDEX] = stage_38_v22 + stage_38_v60 // 5 * 24
                stage_38_v63: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54, [stage_38_v62, stage_38_v61])
                if 0 < pl.cast(stage_38_v63, pl.INDEX):
                    stage_38_v95: pl.Scalar[pl.INDEX] = stage_38_v22 * 3 + stage_38_v60 % 3
                    stage_38_v96: pl.Scalar[pl.INDEX] = stage_38_v95 * 64
                    pl.system.sync_wait(3, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    stage_38_v97: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_60, [stage_38_v96 + stage_38_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
                    stage_38_v98: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_61, [stage_38_v96 + stage_38_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
                    stage_38_v99: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v51, [1, 32])
                    stage_38_v100: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v97, [1, 32])
                    stage_38_v101: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_38_v99, stage_38_v100)
                    stage_38_v102: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v101, [32, 1])
                    stage_38_v103: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v51, [1, 32])
                    stage_38_v104: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v102, [1, 32])
                    stage_38_v105: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_38_v103, stage_38_v104)
                    stage_38_v106: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v105, [32, 1])
                    stage_38_v107: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v106, [1, 32])
                    stage_38_v108: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_38_v107)
                    stage_38_v109: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v108, [32, 1])
                    stage_38_v110: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v97, [1, 32])
                    stage_38_v111: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v102, [1, 32])
                    stage_38_v112: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_38_v110, stage_38_v111)
                    stage_38_v113: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v112, [32, 1])
                    stage_38_v114: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v113, [1, 32])
                    stage_38_v115: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_38_v114)
                    stage_38_v116: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v115, [32, 1])
                    stage_38_v117: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v109, [1, 32])
                    stage_38_v118: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v52, [1, 32])
                    stage_38_v119: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_38_v117, stage_38_v118)
                    stage_38_v120: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v119, [32, 1])
                    stage_38_v121: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v116, [1, 32])
                    stage_38_v122: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v98, [1, 32])
                    stage_38_v123: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.mul(stage_38_v121, stage_38_v122)
                    stage_38_v124: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v123, [32, 1])
                    stage_38_v125: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v120, [1, 32])
                    stage_38_v126: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v124, [1, 32])
                    stage_38_v127: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_38_v125, stage_38_v126)
                    stage_38_v128: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v127, [32, 1])
                    stage_38_v129: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_59, [stage_38_v96 + stage_38_v26, 0], [32, 256], [32, 256], target_memory=pl.Mem.Vec)
                    stage_38_v130: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_38_v53, stage_38_v109)
                    stage_38_v131: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_38_v129, stage_38_v116)
                    stage_38_v132: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_38_v130, stage_38_v131)
                    stage_38_v133: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_59, [stage_38_v96 + stage_38_v26, 256], [32, 256], [32, 256], target_memory=pl.Mem.Vec)
                    stage_38_v134: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_38_v54, stage_38_v109)
                    stage_38_v135: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_38_v133, stage_38_v116)
                    stage_38_v136: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_38_v134, stage_38_v135)
                    stage_38_v137 = stage_38_v102
                    stage_38_v138 = stage_38_v128
                    stage_38_v139 = stage_38_v132
                    stage_38_v140 = stage_38_v136
                else:
                    stage_38_v137 = stage_38_v51
                    stage_38_v138 = stage_38_v52
                    stage_38_v139 = stage_38_v53
                    stage_38_v140 = stage_38_v54
                if stage_38_v61 == 4:
                    stage_38_v141: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.create([1, 1024], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    stage_38_v142: pl.Scalar[pl.INDEX] = 0
                    stage_38_v143: pl.Scalar[pl.INDEX] = stage_38_v26 + stage_38_v142
                    stage_38_v144: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v137, [1, 32])
                    stage_38_v145: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_38_v144, 0, stage_38_v142, [1, 16], target_memory=pl.Mem.Vec)
                    stage_38_v146: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v145, [16, 1])
                    stage_38_v147: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v138, [1, 32])
                    stage_38_v148: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_38_v147, 0, stage_38_v142, [1, 16], target_memory=pl.Mem.Vec)
                    stage_38_v149: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v148, [16, 1])
                    stage_38_v150: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_38_v139, [16, 256], [stage_38_v142, 0])
                    stage_38_v151: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_38_v140, [16, 256], [stage_38_v142, 0])
                    stage_38_v152: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.concat(stage_38_v150, stage_38_v151)
                    stage_38_v153: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_117, [stage_38_v143, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    stage_38_v154: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v146, [1, 16])
                    stage_38_v155: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v146, [1, 16])
                    stage_38_v156: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_38_v154, stage_38_v155)
                    stage_38_v157: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v156, [16, 1])
                    stage_38_v158: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v157, [1, 16])
                    stage_38_v159: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v153, [1, 16])
                    stage_38_v160: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_38_v158, stage_38_v159)
                    stage_38_v161: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v160, [16, 1])
                    stage_38_v162: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v161, [1, 16])
                    stage_38_v163: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v146, [1, 16])
                    stage_38_v164: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_38_v162, stage_38_v163)
                    stage_38_v165: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v164, [16, 1])
                    stage_38_v166: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v165, [1, 16])
                    stage_38_v167: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_38_v166)
                    stage_38_v168: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v167, [16, 1])
                    stage_38_v169: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v149, [1, 16])
                    stage_38_v170: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v168, [1, 16])
                    stage_38_v171: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_38_v169, stage_38_v170)
                    stage_38_v172: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v171, [16, 1])
                    stage_38_v173: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_div(stage_38_v152, stage_38_v172)
                    stage_38_v174: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_38_v173, target_type=pl.BF16, mode='rint')
                    stage_38_v175: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_38_v62, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_38_v176: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_55, [stage_38_v62, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_38_v177: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v173, [1, 8192])
                    stage_38_v178: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_38_v177, stage_38_v40, stage_38_v141)
                    stage_38_v179: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v178, [16, 64])
                    stage_38_v180: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_38_v173, 0, 448, [16, 64], target_memory=pl.Mem.Vec)
                    stage_38_v181: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_38_v180, stage_38_v175)
                    stage_38_v182: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_38_v179, stage_38_v176)
                    stage_38_v183: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_38_v181, stage_38_v182)
                    stage_38_v184: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_38_v183, target_type=pl.BF16, mode='rint')
                    stage_38_v185: pl.Tile[[16, 448], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_38_v174, [16, 448], [0, 0])
                    stage_38_v186: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.concat(stage_38_v185, stage_38_v184)
                    stage_38_v187: pl.Tile[[2, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.reshape(stage_38_v186, [2, 4096])
                    stage_38_v188: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_38_v187, [1, 4096], [0, 0])
                    stage_38_v189: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_38_v187, [1, 4096], [1, 0])
                    stage_38_v190: pl.Scalar[pl.INDEX] = stage_38_v143 // 8 * 384 + stage_38_v62
                    stage_38_v191: pl.Scalar[pl.INDEX] = stage_38_v190 + 384
                    stage_38_v192 = pl.tile.store(stage_38_v188, [stage_38_v190, 0], temporary_51)
                    stage_38_v193 = pl.tile.store(stage_38_v189, [stage_38_v191, 0], temporary_51)
                    stage_38_v194: pl.Scalar[pl.INDEX] = 16
                    stage_38_v195: pl.Scalar[pl.INDEX] = stage_38_v26 + stage_38_v194
                    stage_38_v196: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v137, [1, 32])
                    stage_38_v197: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_38_v196, 0, stage_38_v194, [1, 16], target_memory=pl.Mem.Vec)
                    stage_38_v198: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v197, [16, 1])
                    stage_38_v199: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v138, [1, 32])
                    stage_38_v200: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_38_v199, 0, stage_38_v194, [1, 16], target_memory=pl.Mem.Vec)
                    stage_38_v201: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v200, [16, 1])
                    stage_38_v202: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_38_v139, [16, 256], [stage_38_v194, 0])
                    stage_38_v203: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(stage_38_v140, [16, 256], [stage_38_v194, 0])
                    stage_38_v204: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.concat(stage_38_v202, stage_38_v203)
                    stage_38_v205: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_117, [stage_38_v195, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    stage_38_v206: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v198, [1, 16])
                    stage_38_v207: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v198, [1, 16])
                    stage_38_v208: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_38_v206, stage_38_v207)
                    stage_38_v209: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v208, [16, 1])
                    stage_38_v210: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v209, [1, 16])
                    stage_38_v211: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v205, [1, 16])
                    stage_38_v212: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_38_v210, stage_38_v211)
                    stage_38_v213: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v212, [16, 1])
                    stage_38_v214: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v213, [1, 16])
                    stage_38_v215: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v198, [1, 16])
                    stage_38_v216: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sub(stage_38_v214, stage_38_v215)
                    stage_38_v217: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v216, [16, 1])
                    stage_38_v218: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v217, [1, 16])
                    stage_38_v219: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.exp(stage_38_v218)
                    stage_38_v220: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v219, [16, 1])
                    stage_38_v221: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v201, [1, 16])
                    stage_38_v222: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v220, [1, 16])
                    stage_38_v223: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_38_v221, stage_38_v222)
                    stage_38_v224: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v223, [16, 1])
                    stage_38_v225: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_div(stage_38_v204, stage_38_v224)
                    stage_38_v226: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_38_v225, target_type=pl.BF16, mode='rint')
                    stage_38_v227: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_freqs_cos, [stage_38_v62, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_38_v228: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_55, [stage_38_v62, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    stage_38_v229: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v225, [1, 8192])
                    stage_38_v230: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.gather(stage_38_v229, stage_38_v40, stage_38_v141)
                    stage_38_v231: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v230, [16, 64])
                    stage_38_v232: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(stage_38_v225, 0, 448, [16, 64], target_memory=pl.Mem.Vec)
                    stage_38_v233: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_38_v232, stage_38_v227)
                    stage_38_v234: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_38_v231, stage_38_v228)
                    stage_38_v235: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_38_v233, stage_38_v234)
                    stage_38_v236: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_38_v235, target_type=pl.BF16, mode='rint')
                    stage_38_v237: pl.Tile[[16, 448], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_38_v226, [16, 448], [0, 0])
                    stage_38_v238: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.concat(stage_38_v237, stage_38_v236)
                    stage_38_v239: pl.Tile[[2, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.reshape(stage_38_v238, [2, 4096])
                    stage_38_v240: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_38_v239, [1, 4096], [0, 0])
                    stage_38_v241: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(stage_38_v239, [1, 4096], [1, 0])
                    stage_38_v242: pl.Scalar[pl.INDEX] = stage_38_v195 // 8 * 384 + stage_38_v62
                    stage_38_v243: pl.Scalar[pl.INDEX] = stage_38_v242 + 384
                    stage_38_v244 = pl.tile.store(stage_38_v240, [stage_38_v242, 0], temporary_51)
                    stage_38_v245 = pl.tile.store(stage_38_v241, [stage_38_v243, 0], temporary_51)
                    stage_38_v246: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_117, [stage_38_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
                    stage_38_v247: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v246, [1, 32])
                    stage_38_v248: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_38_v247, 0.0)
                    stage_38_v249: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_38_v248, [32, 1])
                    stage_38_v250: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
                    stage_38_v251: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
                    stage_38_v252 = stage_38_v246
                    stage_38_v253 = stage_38_v249
                    stage_38_v254 = stage_38_v250
                    stage_38_v255 = stage_38_v251
                else:
                    stage_38_v252 = stage_38_v137
                    stage_38_v253 = stage_38_v138
                    stage_38_v254 = stage_38_v139
                    stage_38_v255 = stage_38_v140
                stage_38_v64 = stage_38_v252
                stage_38_v65 = stage_38_v253
                stage_38_v66 = stage_38_v254
                stage_38_v67 = stage_38_v255
            else:
                stage_38_v64 = stage_38_v51
                stage_38_v65 = stage_38_v52
                stage_38_v66 = stage_38_v53
                stage_38_v67 = stage_38_v54
            (stage_38_v68, stage_38_v69, stage_38_v70, stage_38_v71) = pl.yield_(stage_38_v64, stage_38_v65, stage_38_v66, stage_38_v67)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_38 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_38, pl.INT32))
    record_39 = 29808 + core_id * 6
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
            flag_39 = other_39 * 128
            progress = csa_signal_wait(progress, flag_39, pl.cast(target_39, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_39_v4: pl.Scalar[pl.INDEX] = block_39
        stage_39_v5: pl.Scalar[pl.INDEX] = stage_39_v4 * 32
        for (stage_39_v6, (stage_39_v26, stage_39_v27)) in pl.pipeline(stage_39_v5, pl.min(stage_39_v5 + 32, 192), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_39_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_39_v6, 6144], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_39_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_39_v7)
            stage_39_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_39_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_39_v8, stage_39_v9)
            stage_39_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_39_v10, [1, 8])
            stage_39_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_39_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_39_v12, stage_39_v11)
            stage_39_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_39_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_39_v14, stage_39_v13)
            stage_39_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_39_v13, 0.007874015748031496)
            stage_39_v17 = pl.tile.store(stage_39_v16, [6, stage_39_v6], stage_39_v26)
            stage_39_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_39_v15, [8, 1])
            stage_39_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_39_v6, 6144], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_39_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_39_v19, stage_39_v18)
            stage_39_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_39_v20, target_type=pl.INT32, mode='rint')
            stage_39_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_39_v21, target_type=pl.FP16, mode='round')
            stage_39_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_39_v22, target_type=pl.INT8, mode='trunc')
            stage_39_v24 = pl.tile.store(stage_39_v23, [stage_39_v6, 6144], stage_39_v27)
            (stage_39_v28, stage_39_v29) = pl.yield_(stage_39_v17, stage_39_v24)
        if 192 <= stage_39_v5 + 32:
            for (stage_39_v30, (stage_39_v37,)) in pl.range(192, 256, 8, init_values=(stage_39_v29,)):
                stage_39_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_39_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_39_v31, target_type=pl.INT8, mode='trunc')
                stage_39_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_39_v30, 8)
                stage_39_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_39_v33, 1024])] = pl.tile.set_validshape(stage_39_v32, stage_39_v33, 1024)
                stage_39_v35 = pl.tile.store(stage_39_v34, [stage_39_v30, 6144], stage_39_v37)
                stage_39_v36 = pl.yield_(stage_39_v35)
            stage_39_v25 = pl.yield_(stage_39_v36)
        else:
            stage_39_v25 = pl.yield_(stage_39_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_39 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_39, pl.INT32))
    record_40 = 30240 + core_id * 6
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
            flag_40 = other_40 * 128
            progress = csa_signal_wait(progress, flag_40, pl.cast(target_40, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_40_v4: pl.Scalar[pl.INDEX] = block_40
        stage_40_v5: pl.Scalar[pl.INDEX] = stage_40_v4 * 32
        for (stage_40_v6, (stage_40_v26, stage_40_v27)) in pl.pipeline(stage_40_v5, pl.min(stage_40_v5 + 32, 192), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_40_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_40_v6, 7168], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_40_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_40_v7)
            stage_40_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_40_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_40_v8, stage_40_v9)
            stage_40_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_40_v10, [1, 8])
            stage_40_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_40_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_40_v12, stage_40_v11)
            stage_40_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_40_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_40_v14, stage_40_v13)
            stage_40_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_40_v13, 0.007874015748031496)
            stage_40_v17 = pl.tile.store(stage_40_v16, [7, stage_40_v6], stage_40_v26)
            stage_40_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_40_v15, [8, 1])
            stage_40_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_40_v6, 7168], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_40_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_40_v19, stage_40_v18)
            stage_40_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_40_v20, target_type=pl.INT32, mode='rint')
            stage_40_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_40_v21, target_type=pl.FP16, mode='round')
            stage_40_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_40_v22, target_type=pl.INT8, mode='trunc')
            stage_40_v24 = pl.tile.store(stage_40_v23, [stage_40_v6, 7168], stage_40_v27)
            (stage_40_v28, stage_40_v29) = pl.yield_(stage_40_v17, stage_40_v24)
        if 192 <= stage_40_v5 + 32:
            for (stage_40_v30, (stage_40_v37,)) in pl.range(192, 256, 8, init_values=(stage_40_v29,)):
                stage_40_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_40_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_40_v31, target_type=pl.INT8, mode='trunc')
                stage_40_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_40_v30, 8)
                stage_40_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_40_v33, 1024])] = pl.tile.set_validshape(stage_40_v32, stage_40_v33, 1024)
                stage_40_v35 = pl.tile.store(stage_40_v34, [stage_40_v30, 7168], stage_40_v37)
                stage_40_v36 = pl.yield_(stage_40_v35)
            stage_40_v25 = pl.yield_(stage_40_v36)
        else:
            stage_40_v25 = pl.yield_(stage_40_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_40 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_40, pl.INT32))
    record_41 = 30672 + core_id * 6
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
            flag_41 = other_41 * 128
            progress = csa_signal_wait(progress, flag_41, pl.cast(target_41, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_41_v4: pl.Scalar[pl.INDEX] = block_41
        stage_41_v5: pl.Scalar[pl.INDEX] = stage_41_v4 * 32
        for (stage_41_v6, (stage_41_v26, stage_41_v27)) in pl.pipeline(stage_41_v5, pl.min(stage_41_v5 + 32, 192), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_41_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_41_v6, 5120], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_41_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_41_v7)
            stage_41_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_41_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_41_v8, stage_41_v9)
            stage_41_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_41_v10, [1, 8])
            stage_41_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_41_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_41_v12, stage_41_v11)
            stage_41_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_41_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_41_v14, stage_41_v13)
            stage_41_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_41_v13, 0.007874015748031496)
            stage_41_v17 = pl.tile.store(stage_41_v16, [5, stage_41_v6], stage_41_v26)
            stage_41_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_41_v15, [8, 1])
            stage_41_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_41_v6, 5120], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_41_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_41_v19, stage_41_v18)
            stage_41_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_41_v20, target_type=pl.INT32, mode='rint')
            stage_41_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_41_v21, target_type=pl.FP16, mode='round')
            stage_41_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_41_v22, target_type=pl.INT8, mode='trunc')
            stage_41_v24 = pl.tile.store(stage_41_v23, [stage_41_v6, 5120], stage_41_v27)
            (stage_41_v28, stage_41_v29) = pl.yield_(stage_41_v17, stage_41_v24)
        if 192 <= stage_41_v5 + 32:
            for (stage_41_v30, (stage_41_v37,)) in pl.range(192, 256, 8, init_values=(stage_41_v29,)):
                stage_41_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_41_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_41_v31, target_type=pl.INT8, mode='trunc')
                stage_41_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_41_v30, 8)
                stage_41_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_41_v33, 1024])] = pl.tile.set_validshape(stage_41_v32, stage_41_v33, 1024)
                stage_41_v35 = pl.tile.store(stage_41_v34, [stage_41_v30, 5120], stage_41_v37)
                stage_41_v36 = pl.yield_(stage_41_v35)
            stage_41_v25 = pl.yield_(stage_41_v36)
        else:
            stage_41_v25 = pl.yield_(stage_41_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_41 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_41, pl.INT32))
    record_42 = 31104 + core_id * 6
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
            flag_42 = other_42 * 128
            progress = csa_signal_wait(progress, flag_42, pl.cast(target_42, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_42_v4: pl.Scalar[pl.INDEX] = block_42
        stage_42_v5: pl.Scalar[pl.INDEX] = stage_42_v4 * 32
        for (stage_42_v6, (stage_42_v26, stage_42_v27)) in pl.pipeline(stage_42_v5, pl.min(stage_42_v5 + 32, 192), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_42_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_42_v6, 4096], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_42_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_42_v7)
            stage_42_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_42_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_42_v8, stage_42_v9)
            stage_42_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_42_v10, [1, 8])
            stage_42_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_42_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_42_v12, stage_42_v11)
            stage_42_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_42_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_42_v14, stage_42_v13)
            stage_42_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_42_v13, 0.007874015748031496)
            stage_42_v17 = pl.tile.store(stage_42_v16, [4, stage_42_v6], stage_42_v26)
            stage_42_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_42_v15, [8, 1])
            stage_42_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_42_v6, 4096], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_42_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_42_v19, stage_42_v18)
            stage_42_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_42_v20, target_type=pl.INT32, mode='rint')
            stage_42_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_42_v21, target_type=pl.FP16, mode='round')
            stage_42_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_42_v22, target_type=pl.INT8, mode='trunc')
            stage_42_v24 = pl.tile.store(stage_42_v23, [stage_42_v6, 4096], stage_42_v27)
            (stage_42_v28, stage_42_v29) = pl.yield_(stage_42_v17, stage_42_v24)
        if 192 <= stage_42_v5 + 32:
            for (stage_42_v30, (stage_42_v37,)) in pl.range(192, 256, 8, init_values=(stage_42_v29,)):
                stage_42_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_42_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_42_v31, target_type=pl.INT8, mode='trunc')
                stage_42_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_42_v30, 8)
                stage_42_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_42_v33, 1024])] = pl.tile.set_validshape(stage_42_v32, stage_42_v33, 1024)
                stage_42_v35 = pl.tile.store(stage_42_v34, [stage_42_v30, 4096], stage_42_v37)
                stage_42_v36 = pl.yield_(stage_42_v35)
            stage_42_v25 = pl.yield_(stage_42_v36)
        else:
            stage_42_v25 = pl.yield_(stage_42_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_42 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_42, pl.INT32))
    record_43 = 31536 + core_id * 6
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
            flag_43 = other_43 * 128
            progress = csa_signal_wait(progress, flag_43, pl.cast(target_43, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_43_v4: pl.Scalar[pl.INDEX] = block_43
        stage_43_v5: pl.Scalar[pl.INDEX] = stage_43_v4 * 32
        for (stage_43_v6, (stage_43_v26, stage_43_v27)) in pl.pipeline(stage_43_v5, pl.min(stage_43_v5 + 32, 192), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_43_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_43_v6, 1024], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_43_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_43_v7)
            stage_43_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_43_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_43_v8, stage_43_v9)
            stage_43_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_43_v10, [1, 8])
            stage_43_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_43_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_43_v12, stage_43_v11)
            stage_43_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_43_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_43_v14, stage_43_v13)
            stage_43_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_43_v13, 0.007874015748031496)
            stage_43_v17 = pl.tile.store(stage_43_v16, [1, stage_43_v6], stage_43_v26)
            stage_43_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_43_v15, [8, 1])
            stage_43_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_43_v6, 1024], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_43_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_43_v19, stage_43_v18)
            stage_43_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_43_v20, target_type=pl.INT32, mode='rint')
            stage_43_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_43_v21, target_type=pl.FP16, mode='round')
            stage_43_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_43_v22, target_type=pl.INT8, mode='trunc')
            stage_43_v24 = pl.tile.store(stage_43_v23, [stage_43_v6, 1024], stage_43_v27)
            (stage_43_v28, stage_43_v29) = pl.yield_(stage_43_v17, stage_43_v24)
        if 192 <= stage_43_v5 + 32:
            for (stage_43_v30, (stage_43_v37,)) in pl.range(192, 256, 8, init_values=(stage_43_v29,)):
                stage_43_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_43_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_43_v31, target_type=pl.INT8, mode='trunc')
                stage_43_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_43_v30, 8)
                stage_43_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_43_v33, 1024])] = pl.tile.set_validshape(stage_43_v32, stage_43_v33, 1024)
                stage_43_v35 = pl.tile.store(stage_43_v34, [stage_43_v30, 1024], stage_43_v37)
                stage_43_v36 = pl.yield_(stage_43_v35)
            stage_43_v25 = pl.yield_(stage_43_v36)
        else:
            stage_43_v25 = pl.yield_(stage_43_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_43 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_43, pl.INT32))
    record_44 = 31968 + core_id * 6
    mapped_block_44 = pl.cast(pl.tensor.read(records, [record_44]), pl.INDEX)
    block_44 = pl.max(mapped_block_44, 0)
    if mapped_block_44 >= 0:
        subslot_44 = pl.cast(pl.tensor.read(records, [record_44 + 1]), pl.INDEX)
        done_44 = pl.tensor.read(records, [record_44 + 2])
        wait_base_44 = pl.cast(pl.tensor.read(records, [record_44 + 3]), pl.INDEX)
        wait_count_44 = pl.cast(pl.tensor.read(records, [record_44 + 4]), pl.INDEX)
        publish_44 = pl.tensor.read(records, [record_44 + 5])
        for dependency_44 in pl.range(wait_count_44):
            wait_index_44 = (wait_base_44 + dependency_44) * 2
            other_44 = pl.cast(pl.tensor.read(waits, [wait_index_44]), pl.INDEX)
            required_44 = pl.tensor.read(waits, [wait_index_44 + 1])
            target_44 = generation_base + required_44
            flag_44 = other_44 * 128
            progress = csa_signal_wait(progress, flag_44, pl.cast(target_44, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_44_v4: pl.Scalar[pl.INDEX] = block_44
        stage_44_v5: pl.Scalar[pl.INDEX] = stage_44_v4 * 32
        for (stage_44_v6, (stage_44_v26, stage_44_v27)) in pl.pipeline(stage_44_v5, pl.min(stage_44_v5 + 32, 192), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_44_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_44_v6, 2048], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_44_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_44_v7)
            stage_44_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_44_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_44_v8, stage_44_v9)
            stage_44_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_44_v10, [1, 8])
            stage_44_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_44_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_44_v12, stage_44_v11)
            stage_44_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_44_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_44_v14, stage_44_v13)
            stage_44_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_44_v13, 0.007874015748031496)
            stage_44_v17 = pl.tile.store(stage_44_v16, [2, stage_44_v6], stage_44_v26)
            stage_44_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_44_v15, [8, 1])
            stage_44_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_44_v6, 2048], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_44_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_44_v19, stage_44_v18)
            stage_44_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_44_v20, target_type=pl.INT32, mode='rint')
            stage_44_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_44_v21, target_type=pl.FP16, mode='round')
            stage_44_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_44_v22, target_type=pl.INT8, mode='trunc')
            stage_44_v24 = pl.tile.store(stage_44_v23, [stage_44_v6, 2048], stage_44_v27)
            (stage_44_v28, stage_44_v29) = pl.yield_(stage_44_v17, stage_44_v24)
        if 192 <= stage_44_v5 + 32:
            for (stage_44_v30, (stage_44_v37,)) in pl.range(192, 256, 8, init_values=(stage_44_v29,)):
                stage_44_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_44_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_44_v31, target_type=pl.INT8, mode='trunc')
                stage_44_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_44_v30, 8)
                stage_44_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_44_v33, 1024])] = pl.tile.set_validshape(stage_44_v32, stage_44_v33, 1024)
                stage_44_v35 = pl.tile.store(stage_44_v34, [stage_44_v30, 2048], stage_44_v37)
                stage_44_v36 = pl.yield_(stage_44_v35)
            stage_44_v25 = pl.yield_(stage_44_v36)
        else:
            stage_44_v25 = pl.yield_(stage_44_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_44 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_44, pl.INT32))
    record_45 = 32400 + core_id * 6
    mapped_block_45 = pl.cast(pl.tensor.read(records, [record_45]), pl.INDEX)
    block_45 = pl.max(mapped_block_45, 0)
    if mapped_block_45 >= 0:
        subslot_45 = pl.cast(pl.tensor.read(records, [record_45 + 1]), pl.INDEX)
        done_45 = pl.tensor.read(records, [record_45 + 2])
        wait_base_45 = pl.cast(pl.tensor.read(records, [record_45 + 3]), pl.INDEX)
        wait_count_45 = pl.cast(pl.tensor.read(records, [record_45 + 4]), pl.INDEX)
        publish_45 = pl.tensor.read(records, [record_45 + 5])
        for dependency_45 in pl.range(wait_count_45):
            wait_index_45 = (wait_base_45 + dependency_45) * 2
            other_45 = pl.cast(pl.tensor.read(waits, [wait_index_45]), pl.INDEX)
            required_45 = pl.tensor.read(waits, [wait_index_45 + 1])
            target_45 = generation_base + required_45
            flag_45 = other_45 * 128
            progress = csa_signal_wait(progress, flag_45, pl.cast(target_45, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_45_v4: pl.Scalar[pl.INDEX] = block_45
        stage_45_v5: pl.Scalar[pl.INDEX] = stage_45_v4 * 32
        for (stage_45_v6, (stage_45_v26, stage_45_v27)) in pl.pipeline(stage_45_v5, pl.min(stage_45_v5 + 32, 192), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_45_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_45_v6, 3072], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_45_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_45_v7)
            stage_45_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_45_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_45_v8, stage_45_v9)
            stage_45_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_45_v10, [1, 8])
            stage_45_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_45_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_45_v12, stage_45_v11)
            stage_45_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_45_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_45_v14, stage_45_v13)
            stage_45_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_45_v13, 0.007874015748031496)
            stage_45_v17 = pl.tile.store(stage_45_v16, [3, stage_45_v6], stage_45_v26)
            stage_45_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_45_v15, [8, 1])
            stage_45_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_45_v6, 3072], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_45_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_45_v19, stage_45_v18)
            stage_45_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_45_v20, target_type=pl.INT32, mode='rint')
            stage_45_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_45_v21, target_type=pl.FP16, mode='round')
            stage_45_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_45_v22, target_type=pl.INT8, mode='trunc')
            stage_45_v24 = pl.tile.store(stage_45_v23, [stage_45_v6, 3072], stage_45_v27)
            (stage_45_v28, stage_45_v29) = pl.yield_(stage_45_v17, stage_45_v24)
        if 192 <= stage_45_v5 + 32:
            for (stage_45_v30, (stage_45_v37,)) in pl.range(192, 256, 8, init_values=(stage_45_v29,)):
                stage_45_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_45_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_45_v31, target_type=pl.INT8, mode='trunc')
                stage_45_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_45_v30, 8)
                stage_45_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_45_v33, 1024])] = pl.tile.set_validshape(stage_45_v32, stage_45_v33, 1024)
                stage_45_v35 = pl.tile.store(stage_45_v34, [stage_45_v30, 3072], stage_45_v37)
                stage_45_v36 = pl.yield_(stage_45_v35)
            stage_45_v25 = pl.yield_(stage_45_v36)
        else:
            stage_45_v25 = pl.yield_(stage_45_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_45 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_45, pl.INT32))
    record_46 = 32832 + core_id * 6
    mapped_block_46 = pl.cast(pl.tensor.read(records, [record_46]), pl.INDEX)
    block_46 = pl.max(mapped_block_46, 0)
    if mapped_block_46 >= 0:
        subslot_46 = pl.cast(pl.tensor.read(records, [record_46 + 1]), pl.INDEX)
        done_46 = pl.tensor.read(records, [record_46 + 2])
        wait_base_46 = pl.cast(pl.tensor.read(records, [record_46 + 3]), pl.INDEX)
        wait_count_46 = pl.cast(pl.tensor.read(records, [record_46 + 4]), pl.INDEX)
        publish_46 = pl.tensor.read(records, [record_46 + 5])
        for dependency_46 in pl.range(wait_count_46):
            wait_index_46 = (wait_base_46 + dependency_46) * 2
            other_46 = pl.cast(pl.tensor.read(waits, [wait_index_46]), pl.INDEX)
            required_46 = pl.tensor.read(waits, [wait_index_46 + 1])
            target_46 = generation_base + required_46
            flag_46 = other_46 * 128
            progress = csa_signal_wait(progress, flag_46, pl.cast(target_46, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_46_v4: pl.Scalar[pl.INDEX] = block_46
        stage_46_v5: pl.Scalar[pl.INDEX] = stage_46_v4 * 32
        for (stage_46_v6, (stage_46_v26, stage_46_v27)) in pl.pipeline(stage_46_v5, pl.min(stage_46_v5 + 32, 192), 8, stage=2, init_values=(temporary_65, temporary_64)):
            stage_46_v7: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_46_v6, 0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_46_v8: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.abs(stage_46_v7)
            stage_46_v9: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_46_v10: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(stage_46_v8, stage_46_v9)
            stage_46_v11: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_46_v10, [1, 8])
            stage_46_v12: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            stage_46_v13: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(stage_46_v12, stage_46_v11)
            stage_46_v14: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            stage_46_v15: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(stage_46_v14, stage_46_v13)
            stage_46_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_46_v13, 0.007874015748031496)
            stage_46_v17 = pl.tile.store(stage_46_v16, [0, stage_46_v6], stage_46_v26)
            stage_46_v18: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_46_v15, [8, 1])
            stage_46_v19: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(temporary_63, [stage_46_v6, 0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            stage_46_v20: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(stage_46_v19, stage_46_v18)
            stage_46_v21: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.cast(stage_46_v20, target_type=pl.INT32, mode='rint')
            stage_46_v22: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.cast(stage_46_v21, target_type=pl.FP16, mode='round')
            stage_46_v23: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_46_v22, target_type=pl.INT8, mode='trunc')
            stage_46_v24 = pl.tile.store(stage_46_v23, [stage_46_v6, 0], stage_46_v27)
            (stage_46_v28, stage_46_v29) = pl.yield_(stage_46_v17, stage_46_v24)
        if 192 <= stage_46_v5 + 32:
            for (stage_46_v30, (stage_46_v37,)) in pl.range(192, 256, 8, init_values=(stage_46_v29,)):
                stage_46_v31: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                stage_46_v32: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(stage_46_v31, target_type=pl.INT8, mode='trunc')
                stage_46_v33: pl.Scalar[pl.INDEX] = pl.min(256 - stage_46_v30, 8)
                stage_46_v34: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[stage_46_v33, 1024])] = pl.tile.set_validshape(stage_46_v32, stage_46_v33, 1024)
                stage_46_v35 = pl.tile.store(stage_46_v34, [stage_46_v30, 0], stage_46_v37)
                stage_46_v36 = pl.yield_(stage_46_v35)
            stage_46_v25 = pl.yield_(stage_46_v36)
        else:
            stage_46_v25 = pl.yield_(stage_46_v29)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_46 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_46, pl.INT32))
    record_47 = 33264 + core_id * 6
    mapped_block_47 = pl.cast(pl.tensor.read(records, [record_47]), pl.INDEX)
    block_47 = pl.max(mapped_block_47, 0)
    if mapped_block_47 >= 0:
        subslot_47 = pl.cast(pl.tensor.read(records, [record_47 + 1]), pl.INDEX)
        done_47 = pl.tensor.read(records, [record_47 + 2])
        wait_base_47 = pl.cast(pl.tensor.read(records, [record_47 + 3]), pl.INDEX)
        wait_count_47 = pl.cast(pl.tensor.read(records, [record_47 + 4]), pl.INDEX)
        publish_47 = pl.tensor.read(records, [record_47 + 5])
        for dependency_47 in pl.range(wait_count_47):
            wait_index_47 = (wait_base_47 + dependency_47) * 2
            other_47 = pl.cast(pl.tensor.read(waits, [wait_index_47]), pl.INDEX)
            required_47 = pl.tensor.read(waits, [wait_index_47 + 1])
            target_47 = generation_base + required_47
            flag_47 = other_47 * 128
            progress = csa_signal_wait(progress, flag_47, pl.cast(target_47, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        stage_47_v8: pl.Scalar[pl.INDEX] = block_47
        stage_47_v9: pl.Scalar[pl.INDEX] = stage_47_v8
        stage_47_v10: pl.Scalar[pl.INDEX] = stage_47_v8 - stage_47_v9
        stage_47_v11: pl.Scalar[pl.INDEX] = stage_47_v10 * 4096
        stage_47_v12: pl.Scalar[pl.INDEX] = stage_47_v9 * 4
        stage_47_v13: pl.Tile[[4096], pl.FP32, pl.Mem.Vec] = pl.tile.load(arg_wo_b_scale, [stage_47_v11], [4096], [4096], target_memory=pl.Mem.Vec)
        stage_47_v14: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(stage_47_v13, [1, 4096])
        for (stage_47_v15, (stage_47_v99,)) in pl.range(stage_47_v12, pl.min(stage_47_v12 + 4, 192), init_values=(view_129,)):
            stage_47_v16: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 4096], dtype=pl.FP32, value=0.0)
            for (stage_47_v100, (stage_47_v108,)) in pl.pipeline(8, stage=2, init_values=(stage_47_v16,)):
                stage_47_v101: pl.Scalar[pl.INDEX] = stage_47_v100 * 4096 + stage_47_v11
                stage_47_v102: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.load(temporary_66, [stage_47_v15, stage_47_v101], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                stage_47_v103: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_65, [stage_47_v100, stage_47_v15])
                stage_47_v104: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_47_v102, target_type=pl.FP32, mode='none')
                stage_47_v105: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_47_v104, stage_47_v103)
                stage_47_v106: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v108, stage_47_v105)
                stage_47_v107: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.yield_(stage_47_v106)
            stage_47_v17: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(stage_47_v107, stage_47_v14)
            stage_47_v18: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.create([1, 4096], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            stage_47_v19: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_47_v17, target_type=pl.BF16, mode='rint')
            stage_47_v20: pl.Scalar[pl.INDEX] = pl.min(192 - stage_47_v15, 1)
            stage_47_v21: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.cast(stage_47_v19, target_type=pl.FP32, mode='round')
            stage_47_v22: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.load(view_3, [stage_47_v15, stage_47_v11], [1, 4096], [stage_47_v20, pl.min(pl.max(16384 - stage_47_v11, 0), 4096)], target_memory=pl.Mem.Vec)
            stage_47_v23: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.cast(stage_47_v22, target_type=pl.FP32, mode='round')
            stage_47_v24: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.load(view_3, [stage_47_v15, stage_47_v11 + 4096], [1, 4096], [stage_47_v20, pl.min(pl.max(12288 - stage_47_v11, 0), 4096)], target_memory=pl.Mem.Vec)
            stage_47_v25: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.cast(stage_47_v24, target_type=pl.FP32, mode='round')
            stage_47_v26: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.load(view_3, [stage_47_v15, stage_47_v11 + 8192], [1, 4096], [stage_47_v20, pl.min(pl.max(8192 - stage_47_v11, 0), 4096)], target_memory=pl.Mem.Vec)
            stage_47_v27: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.cast(stage_47_v26, target_type=pl.FP32, mode='round')
            stage_47_v28: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.load(view_3, [stage_47_v15, stage_47_v11 + 12288], [1, 4096], [stage_47_v20, pl.min(pl.max(4096 - stage_47_v11, 0), 4096)], target_memory=pl.Mem.Vec)
            stage_47_v29: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.cast(stage_47_v28, target_type=pl.FP32, mode='round')
            stage_47_v30: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_0, [stage_47_v15, 0])
            stage_47_v31: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_47_v21, stage_47_v30)
            stage_47_v32: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 0])
            stage_47_v33: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v23, stage_47_v32)
            stage_47_v34: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v31, stage_47_v33)
            stage_47_v35: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 4])
            stage_47_v36: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v25, stage_47_v35)
            stage_47_v37: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v34, stage_47_v36)
            stage_47_v38: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 8])
            stage_47_v39: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v27, stage_47_v38)
            stage_47_v40: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v37, stage_47_v39)
            stage_47_v41: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 12])
            stage_47_v42: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v29, stage_47_v41)
            stage_47_v43: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v40, stage_47_v42)
            stage_47_v44: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_47_v43, target_type=pl.BF16, mode='rint')
            stage_47_v45: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, 4096])] = pl.tile.set_validshape(stage_47_v44, stage_47_v20, 4096)
            stage_47_v46 = pl.tile.store(stage_47_v45, [stage_47_v15, stage_47_v11], stage_47_v99)
            stage_47_v47: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_0, [stage_47_v15, 1])
            stage_47_v48: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_47_v21, stage_47_v47)
            stage_47_v49: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 1])
            stage_47_v50: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v23, stage_47_v49)
            stage_47_v51: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v48, stage_47_v50)
            stage_47_v52: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 5])
            stage_47_v53: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v25, stage_47_v52)
            stage_47_v54: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v51, stage_47_v53)
            stage_47_v55: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 9])
            stage_47_v56: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v27, stage_47_v55)
            stage_47_v57: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v54, stage_47_v56)
            stage_47_v58: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 13])
            stage_47_v59: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v29, stage_47_v58)
            stage_47_v60: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v57, stage_47_v59)
            stage_47_v61: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_47_v60, target_type=pl.BF16, mode='rint')
            stage_47_v62: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, 4096])] = pl.tile.set_validshape(stage_47_v61, stage_47_v20, 4096)
            stage_47_v63 = pl.tile.store(stage_47_v62, [stage_47_v15, stage_47_v11 + 4096], stage_47_v46)
            stage_47_v64: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_0, [stage_47_v15, 2])
            stage_47_v65: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_47_v21, stage_47_v64)
            stage_47_v66: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 2])
            stage_47_v67: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v23, stage_47_v66)
            stage_47_v68: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v65, stage_47_v67)
            stage_47_v69: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 6])
            stage_47_v70: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v25, stage_47_v69)
            stage_47_v71: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v68, stage_47_v70)
            stage_47_v72: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 10])
            stage_47_v73: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v27, stage_47_v72)
            stage_47_v74: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v71, stage_47_v73)
            stage_47_v75: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 14])
            stage_47_v76: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v29, stage_47_v75)
            stage_47_v77: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v74, stage_47_v76)
            stage_47_v78: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_47_v77, target_type=pl.BF16, mode='rint')
            stage_47_v79: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, 4096])] = pl.tile.set_validshape(stage_47_v78, stage_47_v20, 4096)
            stage_47_v80 = pl.tile.store(stage_47_v79, [stage_47_v15, stage_47_v11 + 8192], stage_47_v63)
            stage_47_v81: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_0, [stage_47_v15, 3])
            stage_47_v82: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.muls(stage_47_v21, stage_47_v81)
            stage_47_v83: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 3])
            stage_47_v84: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(16384, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v23, stage_47_v83)
            stage_47_v85: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v82, stage_47_v84)
            stage_47_v86: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 7])
            stage_47_v87: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(12288, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v25, stage_47_v86)
            stage_47_v88: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v85, stage_47_v87)
            stage_47_v89: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 11])
            stage_47_v90: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(8192, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v27, stage_47_v89)
            stage_47_v91: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v88, stage_47_v90)
            stage_47_v92: pl.Scalar[pl.FP32] = pl.tensor.read(temporary_1, [stage_47_v15, 15])
            stage_47_v93: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, pl.min(pl.max(pl.const(4096, pl.INDEX) - stage_47_v11, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX))])] = pl.tile.muls(stage_47_v29, stage_47_v92)
            stage_47_v94: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(stage_47_v91, stage_47_v93)
            stage_47_v95: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(stage_47_v94, target_type=pl.BF16, mode='rint')
            stage_47_v96: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[stage_47_v20, 4096])] = pl.tile.set_validshape(stage_47_v95, stage_47_v20, 4096)
            stage_47_v97 = pl.tile.store(stage_47_v96, [stage_47_v15, stage_47_v11 + 12288], stage_47_v80)
            stage_47_v98 = pl.yield_(stage_47_v97)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_47 != 0:
            progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_47, pl.INT32))
    return (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_129, progress)

@pl.jit.inline
def csa_static_group(temporary_4: pl.InOut[pl.Tensor[[192, 1], pl.FP32]], temporary_5: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]], view_2: pl.InOut[pl.Tensor[[192, 16384], pl.FP32]], view_3: pl.Tensor[[192, 16384], pl.BF16], temporary_7: pl.InOut[pl.Tensor[[768, 32], pl.FP32]], arg_hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], temporary_8: pl.InOut[pl.Tensor[[192, 32], pl.FP32]], arg_hc_attn_scale: pl.Tensor[[3], pl.FP32], temporary_0: pl.InOut[pl.Tensor[[192, 4], pl.FP32]], temporary_9: pl.InOut[pl.Tensor[[8, 8], pl.FP32]], temporary_6: pl.InOut[pl.Tensor[[192, 8], pl.FP32]], arg_hc_attn_base: pl.Tensor[[24], pl.FP32], view_12: pl.Tensor[[1, 24], pl.FP32], temporary_1: pl.InOut[pl.Tensor[[192, 16], pl.FP32]], temporary_10: pl.InOut[pl.Tensor[[8, 32], pl.FP32]], temporary_11: pl.InOut[pl.Tensor[[192, 4096], pl.BF16]], temporary_2: pl.InOut[pl.Tensor[[192, 4096], pl.BF16]], temporary_12: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]], arg_attn_norm_w: pl.Tensor[[4096], pl.BF16], arg_cmp_seq_lens: pl.Tensor[[32], pl.INT32], arg_cmp_query_start_loc: pl.Tensor[[33], pl.INT32], temporary_14: pl.InOut[pl.Tensor[[32], pl.INT32]], arg_kv_seq_lens: pl.Tensor[[32], pl.INT32], arg_idx_query_start_loc: pl.Tensor[[33], pl.INT32], temporary_15: pl.InOut[pl.Tensor[[32], pl.INT32]], arg_freqs_sin: pl.Tensor[[192, 64], pl.FP32], temporary_13: pl.InOut[pl.Tensor[[192, 64], pl.FP32]], temporary_20: pl.InOut[pl.Tensor[[192, 64], pl.FP32]], temporary_21: pl.InOut[pl.Tensor[[192, 64], pl.INT32]], temporary_24: pl.InOut[pl.Tensor[[192, 1024], pl.FP32]], arg_wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ], view_31: pl.Tensor[[1, 1024], pl.BF16], temporary_23: pl.InOut[pl.Tensor[[512, 1], pl.FP32]], temporary_19: pl.InOut[pl.Tensor[[192, 1], pl.FP32]], temporary_22: pl.InOut[pl.Tensor[[512, 1024], pl.INT8]], temporary_18: pl.InOut[pl.Tensor[[192, 1024], pl.INT8]], temporary_25: pl.InOut[pl.Tensor[[256, 32768], pl.INT32]], arg_wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ], arg_freqs_cos: pl.Tensor[[192, 64], pl.FP32], arg_wq_b_scale: pl.Tensor[[32768], pl.FP32], view_40: pl.InOut[pl.Tensor[[192, 32768], pl.BF16]], temporary_26: pl.InOut[pl.Tensor[[192, 512], pl.FP32]], arg_wkv: pl.Tensor[[4096, 512], pl.BF16], temporary_17: pl.InOut[pl.Tensor[[192, 512], pl.BF16]], arg_gamma_ckv: pl.Tensor[[512], pl.BF16], view_45: pl.InOut[pl.Tensor[[9248, 512], pl.BF16]], arg_ori_slot_mapping: pl.Tensor[[192, 2], pl.INT32], temporary_29: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], temporary_30: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], arg_cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], arg_cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], temporary_28: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], arg_position_ids: pl.Tensor[[192], pl.INT64], arg_cmp_ape: pl.Tensor[[4, 1024], pl.FP32], arg_state_block_table: pl.Tensor[[32, 4100], pl.INT32], arg_compress_state: pl.InOut[pl.Tensor[[289, 8192], pl.FP32]], arg_state_slot_mapping: pl.Tensor[[192, 2], pl.INT32], arg_cmp_freqs_cos: pl.Tensor[[80, 64], pl.FP32], arg_cmp_freqs_sin: pl.Tensor[[80, 64], pl.FP32], temporary_31: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], view_60: pl.Tensor[[1, 512], pl.BF16], view_61: pl.InOut[pl.Tensor[[67616, 512], pl.BF16]], temporary_27: pl.InOut[pl.Tensor[[192, 512], pl.FP32]], arg_cmp_slot_mapping: pl.Tensor[[80, 2], pl.INT32], temporary_34: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], temporary_35: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], arg_inner_wkv: pl.Tensor[[256, 4096], pl.BF16], arg_inner_wgate: pl.Tensor[[256, 4096], pl.BF16], temporary_36: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_inner_ape: pl.Tensor[[4, 256], pl.FP32], arg_inner_state_block_table: pl.Tensor[[32, 4100], pl.INT32], arg_inner_compress_state: pl.InOut[pl.Tensor[[289, 1040], pl.FP32]], arg_inner_state_slot_mapping: pl.Tensor[[192, 2], pl.INT32], temporary_33: pl.InOut[pl.Tensor[[384, 128], pl.BF16]], arg_inner_freqs_cos: pl.Tensor[[80, 64], pl.FP32], arg_inner_freqs_sin: pl.Tensor[[80, 64], pl.FP32], view_76: pl.Tensor[[1, 128], pl.BF16], temporary_38: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_hadamard_idx: pl.Tensor[[128, 128], pl.BF16], temporary_37: pl.InOut[pl.Tensor[[384, 1], pl.FP32]], arg_idx_native_kv_cache: pl.InOut[pl.Tensor[[2113, 4160], pl.INT8]], temporary_32: pl.InOut[pl.Tensor[[192, 128], pl.FP32]], arg_idx_slot_mapping: pl.Tensor[[80, 2], pl.INT32], temporary_42: pl.InOut[pl.Tensor[[384, 8192], pl.INT32]], arg_idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], view_85: pl.InOut[pl.Tensor[[384, 8192], pl.BF16]], arg_idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], temporary_43: pl.InOut[pl.Tensor[[12288, 128], pl.FP32]], temporary_41: pl.InOut[pl.Tensor[[24576, 128], pl.BF16]], temporary_39: pl.InOut[pl.Tensor[[24576, 128], pl.INT8]], temporary_40: pl.InOut[pl.Tensor[[24576, 1], pl.FP32]], temporary_45: pl.InOut[pl.Tensor[[1536, 64], pl.FP32]], arg_weights_proj: pl.Tensor[[4096, 64], pl.BF16], temporary_44: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], view_94: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_48: pl.InOut[pl.Tensor[[3072, 128], pl.FP16]], temporary_50: pl.InOut[pl.Tensor[[256], pl.INT64]], arg_idx_block_table: pl.Tensor[[32, 66], pl.INT32], view_98: pl.InOut[pl.Tensor[[68672, 128], pl.INT8]], view_99: pl.InOut[pl.Tensor[[68672, 128], pl.INT8]], temporary_49: pl.InOut[pl.Tensor[[96, 768], pl.FP32]], temporary_47: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_46: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32]], arg_idx_topk_scores: pl.InOut[pl.Tensor[[192, 512], pl.FP32]], arg_idx_topk: pl.InOut[pl.Tensor[[192, 512], pl.INT32]], temporary_53: pl.InOut[pl.Tensor[[192, 512], pl.INT32]], temporary_52: pl.InOut[pl.Tensor[[192, 640], pl.FP32]], view_107: pl.Tensor[[192, 1], pl.INT64], temporary_54: pl.InOut[pl.Tensor[[192, 16], pl.INT32]], arg_ori_block_table: pl.Tensor[[32, 258], pl.INT32], temporary_55: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_62: pl.InOut[pl.Tensor[[256], pl.INT64]], view_112: pl.InOut[pl.Tensor[[12288, 512], pl.BF16]], temporary_56: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_57: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], temporary_58: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], temporary_59: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], view_117: pl.Tensor[[64, 1], pl.FP32], arg_cmp_block_table: pl.Tensor[[32, 66], pl.INT32], temporary_60: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_61: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], arg_wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ], temporary_63: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_65: pl.InOut[pl.Tensor[[8, 384], pl.FP32]], temporary_64: pl.InOut[pl.Tensor[[384, 8192], pl.INT8]], arg_wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ], temporary_66: pl.InOut[pl.Tensor[[384, 32768], pl.INT32]], arg_wo_b_scale: pl.Tensor[[4096], pl.FP32], view_129: pl.InOut[pl.Tensor[[192, 16384], pl.BF16]], progress: pl.InOut[pl.Tensor[[9216], pl.INT32]], records: pl.Tensor[[33696], pl.INT64], waits: pl.Tensor[[49300], pl.INT64]) -> tuple[pl.Tensor[[192, 1], pl.FP32], pl.Tensor[[8, 16384], pl.FP32], pl.Tensor[[192, 16384], pl.FP32], pl.Tensor[[768, 32], pl.FP32], pl.Tensor[[192, 32], pl.FP32], pl.Tensor[[192, 4], pl.FP32], pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[192, 8], pl.FP32], pl.Tensor[[192, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32], pl.Tensor[[192, 4096], pl.BF16], pl.Tensor[[192, 4096], pl.BF16], pl.Tensor[[8, 4096], pl.BF16], pl.Tensor[[32], pl.INT32], pl.Tensor[[32], pl.INT32], pl.Tensor[[192, 64], pl.FP32], pl.Tensor[[192, 64], pl.FP32], pl.Tensor[[192, 64], pl.INT32], pl.Tensor[[192, 1024], pl.FP32], pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[192, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[192, 1024], pl.INT8], pl.Tensor[[256, 32768], pl.INT32], pl.Tensor[[192, 32768], pl.BF16], pl.Tensor[[192, 512], pl.FP32], pl.Tensor[[192, 512], pl.BF16], pl.Tensor[[9248, 512], pl.BF16], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[289, 8192], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[67616, 512], pl.BF16], pl.Tensor[[192, 512], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[289, 1040], pl.FP32], pl.Tensor[[384, 128], pl.BF16], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[2113, 4160], pl.INT8], pl.Tensor[[192, 128], pl.FP32], pl.Tensor[[384, 8192], pl.INT32], pl.Tensor[[384, 8192], pl.BF16], pl.Tensor[[12288, 128], pl.FP32], pl.Tensor[[24576, 128], pl.BF16], pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Tensor[[1536, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[3072, 128], pl.FP16], pl.Tensor[[256], pl.INT64], pl.Tensor[[68672, 128], pl.INT8], pl.Tensor[[68672, 128], pl.INT8], pl.Tensor[[96, 768], pl.FP32], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Tensor[[192, 512], pl.FP32], pl.Tensor[[192, 512], pl.INT32], pl.Tensor[[192, 512], pl.INT32], pl.Tensor[[192, 640], pl.FP32], pl.Tensor[[192, 16], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[256], pl.INT64], pl.Tensor[[12288, 512], pl.BF16], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8], pl.Tensor[[384, 32768], pl.INT32], pl.Tensor[[192, 16384], pl.BF16], pl.Tensor[[9216], pl.INT32]]:
    pl.func_attr({'mx_tensor_views_blocked': True})
    with pl.cluster(name_hint='csa_static_group'):
        (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_129, progress) = csa_static_aic(temporary_4, temporary_5, view_2, view_3, temporary_7, arg_hc_attn_fn, temporary_8, arg_hc_attn_scale, temporary_0, temporary_9, temporary_6, arg_hc_attn_base, view_12, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, arg_attn_norm_w, arg_cmp_seq_lens, arg_cmp_query_start_loc, temporary_14, arg_kv_seq_lens, arg_idx_query_start_loc, temporary_15, arg_freqs_sin, temporary_13, temporary_20, temporary_21, temporary_24, arg_wq_a, view_31, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, arg_wq_b, arg_freqs_cos, arg_wq_b_scale, view_40, temporary_26, arg_wkv, temporary_17, arg_gamma_ckv, view_45, arg_ori_slot_mapping, temporary_29, temporary_30, arg_cmp_wkv, arg_cmp_wgate, temporary_28, arg_position_ids, arg_cmp_ape, arg_state_block_table, arg_compress_state, arg_state_slot_mapping, arg_cmp_freqs_cos, arg_cmp_freqs_sin, temporary_31, view_60, view_61, temporary_27, arg_cmp_slot_mapping, temporary_34, temporary_35, arg_inner_wkv, arg_inner_wgate, temporary_36, arg_inner_ape, arg_inner_state_block_table, arg_inner_compress_state, arg_inner_state_slot_mapping, temporary_33, arg_inner_freqs_cos, arg_inner_freqs_sin, view_76, temporary_38, arg_hadamard_idx, temporary_37, arg_idx_native_kv_cache, temporary_32, arg_idx_slot_mapping, temporary_42, arg_idx_wq_b, view_85, arg_idx_wq_b_scale, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, arg_weights_proj, temporary_44, view_94, temporary_48, temporary_50, arg_idx_block_table, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, view_107, temporary_54, arg_ori_block_table, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, view_117, arg_cmp_block_table, temporary_60, temporary_61, temporary_51, arg_wo_a, temporary_63, temporary_65, temporary_64, arg_wo_b, temporary_66, arg_wo_b_scale, view_129, progress, records, waits)
        (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_129, progress) = csa_static_aiv(temporary_4, temporary_5, view_2, view_3, temporary_7, arg_hc_attn_fn, temporary_8, arg_hc_attn_scale, temporary_0, temporary_9, temporary_6, arg_hc_attn_base, view_12, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, arg_attn_norm_w, arg_cmp_seq_lens, arg_cmp_query_start_loc, temporary_14, arg_kv_seq_lens, arg_idx_query_start_loc, temporary_15, arg_freqs_sin, temporary_13, temporary_20, temporary_21, temporary_24, arg_wq_a, view_31, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, arg_wq_b, arg_freqs_cos, arg_wq_b_scale, view_40, temporary_26, arg_wkv, temporary_17, arg_gamma_ckv, view_45, arg_ori_slot_mapping, temporary_29, temporary_30, arg_cmp_wkv, arg_cmp_wgate, temporary_28, arg_position_ids, arg_cmp_ape, arg_state_block_table, arg_compress_state, arg_state_slot_mapping, arg_cmp_freqs_cos, arg_cmp_freqs_sin, temporary_31, view_60, view_61, temporary_27, arg_cmp_slot_mapping, temporary_34, temporary_35, arg_inner_wkv, arg_inner_wgate, temporary_36, arg_inner_ape, arg_inner_state_block_table, arg_inner_compress_state, arg_inner_state_slot_mapping, temporary_33, arg_inner_freqs_cos, arg_inner_freqs_sin, view_76, temporary_38, arg_hadamard_idx, temporary_37, arg_idx_native_kv_cache, temporary_32, arg_idx_slot_mapping, temporary_42, arg_idx_wq_b, view_85, arg_idx_wq_b_scale, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, arg_weights_proj, temporary_44, view_94, temporary_48, temporary_50, arg_idx_block_table, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, view_107, temporary_54, arg_ori_block_table, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, view_117, arg_cmp_block_table, temporary_60, temporary_61, temporary_51, arg_wo_a, temporary_63, temporary_65, temporary_64, arg_wo_b, temporary_66, arg_wo_b_scale, view_129, progress, records, waits)
    return (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_129, progress)

@pl.jit
def entry(temporary_4: pl.InOut[pl.Tensor[[192, 1], pl.FP32]], temporary_5: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]], view_2: pl.InOut[pl.Tensor[[192, 16384], pl.FP32]], view_3: pl.Tensor[[192, 16384], pl.BF16], temporary_7: pl.InOut[pl.Tensor[[768, 32], pl.FP32]], arg_hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], temporary_8: pl.InOut[pl.Tensor[[192, 32], pl.FP32]], arg_hc_attn_scale: pl.Tensor[[3], pl.FP32], temporary_0: pl.InOut[pl.Tensor[[192, 4], pl.FP32]], temporary_9: pl.InOut[pl.Tensor[[8, 8], pl.FP32]], temporary_6: pl.InOut[pl.Tensor[[192, 8], pl.FP32]], arg_hc_attn_base: pl.Tensor[[24], pl.FP32], view_12: pl.Tensor[[1, 24], pl.FP32], temporary_1: pl.InOut[pl.Tensor[[192, 16], pl.FP32]], temporary_10: pl.InOut[pl.Tensor[[8, 32], pl.FP32]], temporary_11: pl.InOut[pl.Tensor[[192, 4096], pl.BF16]], temporary_2: pl.InOut[pl.Tensor[[192, 4096], pl.BF16]], temporary_12: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]], arg_attn_norm_w: pl.Tensor[[4096], pl.BF16], arg_cmp_seq_lens: pl.Tensor[[32], pl.INT32], arg_cmp_query_start_loc: pl.Tensor[[33], pl.INT32], temporary_14: pl.InOut[pl.Tensor[[32], pl.INT32]], arg_kv_seq_lens: pl.Tensor[[32], pl.INT32], arg_idx_query_start_loc: pl.Tensor[[33], pl.INT32], temporary_15: pl.InOut[pl.Tensor[[32], pl.INT32]], arg_freqs_sin: pl.Tensor[[192, 64], pl.FP32], temporary_13: pl.InOut[pl.Tensor[[192, 64], pl.FP32]], temporary_20: pl.InOut[pl.Tensor[[192, 64], pl.FP32]], temporary_21: pl.InOut[pl.Tensor[[192, 64], pl.INT32]], temporary_24: pl.InOut[pl.Tensor[[192, 1024], pl.FP32]], arg_wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ], view_31: pl.Tensor[[1, 1024], pl.BF16], temporary_23: pl.InOut[pl.Tensor[[512, 1], pl.FP32]], temporary_19: pl.InOut[pl.Tensor[[192, 1], pl.FP32]], temporary_22: pl.InOut[pl.Tensor[[512, 1024], pl.INT8]], temporary_18: pl.InOut[pl.Tensor[[192, 1024], pl.INT8]], temporary_25: pl.InOut[pl.Tensor[[256, 32768], pl.INT32]], arg_wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ], arg_freqs_cos: pl.Tensor[[192, 64], pl.FP32], arg_wq_b_scale: pl.Tensor[[32768], pl.FP32], view_40: pl.InOut[pl.Tensor[[192, 32768], pl.BF16]], temporary_26: pl.InOut[pl.Tensor[[192, 512], pl.FP32]], arg_wkv: pl.Tensor[[4096, 512], pl.BF16], temporary_17: pl.InOut[pl.Tensor[[192, 512], pl.BF16]], arg_gamma_ckv: pl.Tensor[[512], pl.BF16], view_45: pl.InOut[pl.Tensor[[9248, 512], pl.BF16]], arg_ori_slot_mapping: pl.Tensor[[192, 2], pl.INT32], temporary_29: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], temporary_30: pl.InOut[pl.Tensor[[384, 1024], pl.FP32]], arg_cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], arg_cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], temporary_28: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], arg_position_ids: pl.Tensor[[192], pl.INT64], arg_cmp_ape: pl.Tensor[[4, 1024], pl.FP32], arg_state_block_table: pl.Tensor[[32, 4100], pl.INT32], arg_compress_state: pl.InOut[pl.Tensor[[289, 8192], pl.FP32]], arg_state_slot_mapping: pl.Tensor[[192, 2], pl.INT32], arg_cmp_freqs_cos: pl.Tensor[[80, 64], pl.FP32], arg_cmp_freqs_sin: pl.Tensor[[80, 64], pl.FP32], temporary_31: pl.InOut[pl.Tensor[[384, 512], pl.FP32]], view_60: pl.Tensor[[1, 512], pl.BF16], view_61: pl.InOut[pl.Tensor[[67616, 512], pl.BF16]], temporary_27: pl.InOut[pl.Tensor[[192, 512], pl.FP32]], arg_cmp_slot_mapping: pl.Tensor[[80, 2], pl.INT32], temporary_34: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], temporary_35: pl.InOut[pl.Tensor[[384, 256], pl.FP32]], arg_inner_wkv: pl.Tensor[[256, 4096], pl.BF16], arg_inner_wgate: pl.Tensor[[256, 4096], pl.BF16], temporary_36: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_inner_ape: pl.Tensor[[4, 256], pl.FP32], arg_inner_state_block_table: pl.Tensor[[32, 4100], pl.INT32], arg_inner_compress_state: pl.InOut[pl.Tensor[[289, 1040], pl.FP32]], arg_inner_state_slot_mapping: pl.Tensor[[192, 2], pl.INT32], temporary_33: pl.InOut[pl.Tensor[[384, 128], pl.BF16]], arg_inner_freqs_cos: pl.Tensor[[80, 64], pl.FP32], arg_inner_freqs_sin: pl.Tensor[[80, 64], pl.FP32], view_76: pl.Tensor[[1, 128], pl.BF16], temporary_38: pl.InOut[pl.Tensor[[384, 128], pl.FP32]], arg_hadamard_idx: pl.Tensor[[128, 128], pl.BF16], temporary_37: pl.InOut[pl.Tensor[[384, 1], pl.FP32]], arg_idx_native_kv_cache: pl.InOut[pl.Tensor[[2113, 4160], pl.INT8]], temporary_32: pl.InOut[pl.Tensor[[192, 128], pl.FP32]], arg_idx_slot_mapping: pl.Tensor[[80, 2], pl.INT32], temporary_42: pl.InOut[pl.Tensor[[384, 8192], pl.INT32]], arg_idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], view_85: pl.InOut[pl.Tensor[[384, 8192], pl.BF16]], arg_idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], temporary_43: pl.InOut[pl.Tensor[[12288, 128], pl.FP32]], temporary_41: pl.InOut[pl.Tensor[[24576, 128], pl.BF16]], temporary_39: pl.InOut[pl.Tensor[[24576, 128], pl.INT8]], temporary_40: pl.InOut[pl.Tensor[[24576, 1], pl.FP32]], temporary_45: pl.InOut[pl.Tensor[[1536, 64], pl.FP32]], arg_weights_proj: pl.Tensor[[4096, 64], pl.BF16], temporary_44: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], view_94: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_48: pl.InOut[pl.Tensor[[3072, 128], pl.FP16]], temporary_50: pl.InOut[pl.Tensor[[256], pl.INT64]], arg_idx_block_table: pl.Tensor[[32, 66], pl.INT32], temporary_49: pl.InOut[pl.Tensor[[96, 768], pl.FP32]], temporary_47: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_46: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32]], arg_idx_topk_scores: pl.InOut[pl.Tensor[[192, 512], pl.FP32]], arg_idx_topk: pl.InOut[pl.Tensor[[192, 512], pl.INT32]], temporary_53: pl.InOut[pl.Tensor[[192, 512], pl.INT32]], temporary_52: pl.InOut[pl.Tensor[[192, 640], pl.FP32]], view_107: pl.Tensor[[192, 1], pl.INT64], temporary_54: pl.InOut[pl.Tensor[[192, 16], pl.INT32]], arg_ori_block_table: pl.Tensor[[32, 258], pl.INT32], temporary_55: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_62: pl.InOut[pl.Tensor[[256], pl.INT64]], view_112: pl.InOut[pl.Tensor[[12288, 512], pl.BF16]], temporary_56: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_57: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], temporary_58: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], temporary_59: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], view_117: pl.Tensor[[64, 1], pl.FP32], arg_cmp_block_table: pl.Tensor[[32, 66], pl.INT32], temporary_60: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_61: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], arg_wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ], temporary_63: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], temporary_65: pl.InOut[pl.Tensor[[8, 384], pl.FP32]], temporary_64: pl.InOut[pl.Tensor[[384, 8192], pl.INT8]], arg_wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ], temporary_66: pl.InOut[pl.Tensor[[384, 32768], pl.INT32]], arg_wo_b_scale: pl.Tensor[[4096], pl.FP32], view_129: pl.InOut[pl.Tensor[[192, 16384], pl.BF16]], progress: pl.InOut[pl.Tensor[[9216], pl.INT32]], records: pl.Tensor[[33696], pl.INT64], waits: pl.Tensor[[49300], pl.INT64]) -> tuple[pl.Tensor[[192, 1], pl.FP32], pl.Tensor[[8, 16384], pl.FP32], pl.Tensor[[192, 16384], pl.FP32], pl.Tensor[[768, 32], pl.FP32], pl.Tensor[[192, 32], pl.FP32], pl.Tensor[[192, 4], pl.FP32], pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[192, 8], pl.FP32], pl.Tensor[[192, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32], pl.Tensor[[192, 4096], pl.BF16], pl.Tensor[[192, 4096], pl.BF16], pl.Tensor[[8, 4096], pl.BF16], pl.Tensor[[32], pl.INT32], pl.Tensor[[32], pl.INT32], pl.Tensor[[192, 64], pl.FP32], pl.Tensor[[192, 64], pl.FP32], pl.Tensor[[192, 64], pl.INT32], pl.Tensor[[192, 1024], pl.FP32], pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[192, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[192, 1024], pl.INT8], pl.Tensor[[256, 32768], pl.INT32], pl.Tensor[[192, 32768], pl.BF16], pl.Tensor[[192, 512], pl.FP32], pl.Tensor[[192, 512], pl.BF16], pl.Tensor[[9248, 512], pl.BF16], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 1024], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[289, 8192], pl.FP32], pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[67616, 512], pl.BF16], pl.Tensor[[192, 512], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[289, 1040], pl.FP32], pl.Tensor[[384, 128], pl.BF16], pl.Tensor[[384, 128], pl.FP32], pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[2113, 4160], pl.INT8], pl.Tensor[[192, 128], pl.FP32], pl.Tensor[[384, 8192], pl.INT32], pl.Tensor[[384, 8192], pl.BF16], pl.Tensor[[12288, 128], pl.FP32], pl.Tensor[[24576, 128], pl.BF16], pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Tensor[[1536, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[3072, 128], pl.FP16], pl.Tensor[[256], pl.INT64], pl.Tensor[[96, 768], pl.FP32], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Tensor[[192, 512], pl.FP32], pl.Tensor[[192, 512], pl.INT32], pl.Tensor[[192, 512], pl.INT32], pl.Tensor[[192, 640], pl.FP32], pl.Tensor[[192, 16], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[256], pl.INT64], pl.Tensor[[12288, 512], pl.BF16], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8], pl.Tensor[[384, 32768], pl.INT32], pl.Tensor[[192, 16384], pl.BF16], pl.Tensor[[9216], pl.INT32]]:
    pl.func_attr({'mx_tensor_views_blocked': True})
    view_98_flat = pl.reshape(arg_idx_native_kv_cache, [8790080])
    view_98_span = pl.slice(view_98_flat, [8790016], [0])
    view_98 = pl.reshape(view_98_span, [68672, 128])
    view_99_flat = pl.reshape(arg_idx_native_kv_cache, [8790080])
    view_99_span = pl.slice(view_99_flat, [8790016], [64])
    view_99 = pl.reshape(view_99_span, [68672, 128])
    with pl.spmd(24, sync_start=True, name_hint='csa_static_full_card'):
        (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_129, progress) = csa_static_group(temporary_4, temporary_5, view_2, view_3, temporary_7, arg_hc_attn_fn, temporary_8, arg_hc_attn_scale, temporary_0, temporary_9, temporary_6, arg_hc_attn_base, view_12, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, arg_attn_norm_w, arg_cmp_seq_lens, arg_cmp_query_start_loc, temporary_14, arg_kv_seq_lens, arg_idx_query_start_loc, temporary_15, arg_freqs_sin, temporary_13, temporary_20, temporary_21, temporary_24, arg_wq_a, view_31, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, arg_wq_b, arg_freqs_cos, arg_wq_b_scale, view_40, temporary_26, arg_wkv, temporary_17, arg_gamma_ckv, view_45, arg_ori_slot_mapping, temporary_29, temporary_30, arg_cmp_wkv, arg_cmp_wgate, temporary_28, arg_position_ids, arg_cmp_ape, arg_state_block_table, arg_compress_state, arg_state_slot_mapping, arg_cmp_freqs_cos, arg_cmp_freqs_sin, temporary_31, view_60, view_61, temporary_27, arg_cmp_slot_mapping, temporary_34, temporary_35, arg_inner_wkv, arg_inner_wgate, temporary_36, arg_inner_ape, arg_inner_state_block_table, arg_inner_compress_state, arg_inner_state_slot_mapping, temporary_33, arg_inner_freqs_cos, arg_inner_freqs_sin, view_76, temporary_38, arg_hadamard_idx, temporary_37, arg_idx_native_kv_cache, temporary_32, arg_idx_slot_mapping, temporary_42, arg_idx_wq_b, view_85, arg_idx_wq_b_scale, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, arg_weights_proj, temporary_44, view_94, temporary_48, temporary_50, arg_idx_block_table, view_98, view_99, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, view_107, temporary_54, arg_ori_block_table, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, view_117, arg_cmp_block_table, temporary_60, temporary_61, temporary_51, arg_wo_a, temporary_63, temporary_65, temporary_64, arg_wo_b, temporary_66, arg_wo_b_scale, view_129, progress, records, waits)
    return (temporary_4, temporary_5, view_2, temporary_7, temporary_8, temporary_0, temporary_9, temporary_6, temporary_1, temporary_10, temporary_11, temporary_2, temporary_12, temporary_14, temporary_15, temporary_13, temporary_20, temporary_21, temporary_24, temporary_23, temporary_19, temporary_22, temporary_18, temporary_25, view_40, temporary_26, temporary_17, view_45, temporary_29, temporary_30, temporary_28, arg_compress_state, temporary_31, view_61, temporary_27, temporary_34, temporary_35, temporary_36, arg_inner_compress_state, temporary_33, temporary_38, temporary_37, arg_idx_native_kv_cache, temporary_32, temporary_42, view_85, temporary_43, temporary_41, temporary_39, temporary_40, temporary_45, temporary_44, view_94, temporary_48, temporary_50, temporary_49, temporary_47, temporary_46, arg_idx_topk_scores, arg_idx_topk, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, view_112, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51, temporary_63, temporary_65, temporary_64, temporary_66, view_129, progress)
