# Copyright (c) PyPTO Contributors.
# CANN Open Software License Agreement Version 2.0; see repository LICENSE.
# CSA 编排与逻辑调用合同。只由 AST 绑定器读取，不直接导入执行。
# root 保留完整形状/视图/依赖编排；十个 NZ 函数保留重新 lowering 所需数学体。
# 其余函数只有原签名和返回别名；真实物理数学体统一在 kernels.py。
# 参数顺序保持原逻辑调用顺序，不等同于 kernels.py 的显式 worker 参数顺序。
import pypto.language as pl
B_DYN = pl.dynamic('B_DYN')
CMP_BLOCK_NUM_DYN = pl.dynamic('CMP_BLOCK_NUM_DYN')
COMPRESSED_ROWS_DYN = pl.dynamic('COMPRESSED_ROWS_DYN')
COMPRESSED_TABLE_COLUMNS_DYN = pl.dynamic('COMPRESSED_TABLE_COLUMNS_DYN')
HC_PAD_ROWS_DYN = pl.dynamic('HC_PAD_ROWS_DYN')
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
QKV_KV_MM_T_DYN = pl.dynamic('QKV_KV_MM_T_DYN')
QKV_QPROJ_MM_T_DYN = pl.dynamic('QKV_QPROJ_MM_T_DYN')
QUERY_BOUNDS_DYN = pl.dynamic('QUERY_BOUNDS_DYN')
RMS_NORM_CHUNK_D_DYN = pl.dynamic('RMS_NORM_CHUNK_D_DYN')
STATE_PAGE_ELEMENTS_DYN = pl.dynamic('STATE_PAGE_ELEMENTS_DYN')
STATE_TABLE_COLUMNS_DYN = pl.dynamic('STATE_TABLE_COLUMNS_DYN')
T_DYN = pl.dynamic('T_DYN')
_jitdyn_hc_mix_norm_x_mixed_d0 = pl.dynamic('_jitdyn_hc_mix_norm_x_mixed_d0')

@pl.program
class CSAOrchestration:

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def _proj_b_mm_nz_kernel(o_r_i8_pad__ssa_v0: pl.Tensor[[384, 8192], pl.INT8], wo_b__ssa_v0: pl.Tensor[[8192, 4096], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], partials__ssa_v0: pl.Tensor[[384, 32768], pl.INT32], g__ssa_v0: pl.Scalar[pl.INDEX], col_g__ssa_v0: pl.Scalar[pl.INDEX], proj_b_t_rows__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[384, 32768], pl.INT32]:
        t__tmp_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        d0__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v0 * 512
        for tb__idx_v0 in pl.range(proj_b_t_rows__ssa_v0):
            t0__ssa_v0: pl.Scalar[pl.INDEX] = tb__idx_v0 * 32
            activation_l1__ssa_v0: pl.Tile[[32, 1024], pl.INT8, pl.Mem.Mat] = pl.tile.load(o_r_i8_pad__ssa_v0, [t0__ssa_v0, col_g__ssa_v0], [32, 1024], [32, 1024], target_memory=pl.Mem.Mat)
            for nf__idx_v0 in pl.range(2):
                n0__ssa_v0: pl.Scalar[pl.INDEX] = d0__ssa_v0 + nf__idx_v0 * 256
                acc_b__ssa_v0: pl.Tile[[32, 256], pl.INT32, pl.Mem.Acc] = pl.tile.create([32, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
                for (kb__idx_v0, (acc_b__iter_v1,)) in pl.pipeline(4, stage=2, init_values=(acc_b__ssa_v0,)):
                    k0__ssa_v0: pl.Scalar[pl.INDEX] = kb__idx_v0 * 256
                    wk0__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(g__ssa_v0, 0) * 1024 + k0__ssa_v0
                    b_weight_l1__ssa_v0: pl.Tile[[256, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(wo_b__ssa_v0, [wk0__ssa_v0, n0__ssa_v0], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
                    b_act_l1__ssa_v0: pl.Tile[[32, 256], pl.INT8, pl.Mem.Mat] = pl.tile.slice(activation_l1__ssa_v0, [32, 256], [0, k0__ssa_v0])
                    acc_b__ssa_v3: pl.Tile[[32, 256], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(acc_b__iter_v1, b_act_l1__ssa_v0, b_weight_l1__ssa_v0, kb__idx_v0 == 0)
                    acc_b__rv_v2: pl.Tile[[32, 256], pl.INT32, pl.Mem.Acc] = pl.yield_(acc_b__ssa_v3)
                pl.tile.store(acc_b__rv_v2, [t0__ssa_v0, g__ssa_v0 * 4096 + n0__ssa_v0], partials__ssa_v0)
        return partials__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def _proj_b_mm_nz_kernel__2(o_r_i8_pad__ssa_v0: pl.Tensor[[384, 8192], pl.INT8], wo_b__ssa_v0: pl.Tensor[[8192, 4096], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], partials__ssa_v0: pl.Tensor[[384, 32768], pl.INT32], g__ssa_v0: pl.Scalar[pl.INDEX], col_g__ssa_v0: pl.Scalar[pl.INDEX], proj_b_t_rows__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[384, 32768], pl.INT32]:
        t__tmp_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        d0__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v0 * 512
        for tb__idx_v0 in pl.range(proj_b_t_rows__ssa_v0):
            t0__ssa_v0: pl.Scalar[pl.INDEX] = tb__idx_v0 * 96
            activation_l1__ssa_v0: pl.Tile[[96, 1024], pl.INT8, pl.Mem.Mat] = pl.tile.load(o_r_i8_pad__ssa_v0, [t0__ssa_v0, col_g__ssa_v0], [96, 1024], [96, 1024], target_memory=pl.Mem.Mat)
            for nf__idx_v0 in pl.range(2):
                n0__ssa_v0: pl.Scalar[pl.INDEX] = d0__ssa_v0 + nf__idx_v0 * 256
                acc_b__ssa_v0: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.create([96, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
                for (kb__idx_v0, (acc_b__iter_v1,)) in pl.pipeline(4, stage=2, init_values=(acc_b__ssa_v0,)):
                    k0__ssa_v0: pl.Scalar[pl.INDEX] = kb__idx_v0 * 256
                    wk0__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(g__ssa_v0, 0) * 1024 + k0__ssa_v0
                    b_weight_l1__ssa_v0: pl.Tile[[256, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(wo_b__ssa_v0, [wk0__ssa_v0, n0__ssa_v0], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
                    b_act_l1__ssa_v0: pl.Tile[[96, 256], pl.INT8, pl.Mem.Mat] = pl.tile.slice(activation_l1__ssa_v0, [96, 256], [0, k0__ssa_v0])
                    acc_b__ssa_v3: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(acc_b__iter_v1, b_act_l1__ssa_v0, b_weight_l1__ssa_v0, kb__idx_v0 == 0)
                    acc_b__rv_v2: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.yield_(acc_b__ssa_v3)
                pl.tile.store(acc_b__rv_v2, [t0__ssa_v0, g__ssa_v0 * 4096 + n0__ssa_v0], partials__ssa_v0)
        return partials__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def _proj_b_mm_nz_kernel__3(o_r_i8_pad__ssa_v0: pl.Tensor[[384, 8192], pl.INT8], wo_b__ssa_v0: pl.Tensor[[8192, 4096], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], partials__ssa_v0: pl.Tensor[[384, 32768], pl.INT32], g__ssa_v0: pl.Scalar[pl.INDEX], col_g__ssa_v0: pl.Scalar[pl.INDEX], proj_b_t_rows__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[384, 32768], pl.INT32]:
        t__tmp_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        d0__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v0 * 512
        for (nf__idx_v0, (partials__iter_v1,)) in pl.range(2, init_values=(partials__ssa_v0,)):
            n0__ssa_v1: pl.Scalar[pl.INDEX] = d0__ssa_v0 + nf__idx_v0 * 256
            weight_k0__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(g__ssa_v0, 0) * 1024
            weight_resident__ssa_v0: pl.Tensor[[1024, 256], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wo_b__ssa_v0, [1024, 256], [weight_k0__ssa_v0, n0__ssa_v1])
            for (tb__idx_v0, (partials__iter_v3,)) in pl.range(proj_b_t_rows__ssa_v0, init_values=(partials__iter_v1,)):
                t0__ssa_v1: pl.Scalar[pl.INDEX] = tb__idx_v0 * 128
                b_act_full__ssa_v0: pl.Tensor[[128, 1024], pl.INT8] = pl.tensor.slice(o_r_i8_pad__ssa_v0, [128, 1024], [t0__ssa_v1, col_g__ssa_v0])
                full_acc__ssa_v0: pl.Tensor[[128, 256], pl.INT32] = pl.tensor.matmul(b_act_full__ssa_v0, weight_resident__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                partials__ssa_v5: pl.Tensor[[384, 32768], pl.INT32] = pl.tensor.assemble(partials__iter_v3, full_acc__ssa_v0, [t0__ssa_v1, g__ssa_v0 * 4096 + n0__ssa_v1])
                partials__rv_v4: pl.Tensor[[384, 32768], pl.INT32] = pl.yield_(partials__ssa_v5)
            partials__rv_v2: pl.Tensor[[384, 32768], pl.INT32] = pl.yield_(partials__rv_v4)
        return partials__rv_v2

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def comb_sinkhorn(hc_attn_scale__ssa_v0: pl.Tensor[[3], pl.FP32], t_dim_inline440__ssa_v0: pl.Scalar[pl.INDEX], inv_rms__rv_v2: pl.Tensor[[hc_padded_rows__ssa_v0, 1], pl.FP32], mixes_raw_inline448__ssa_v1: pl.Tensor[[t_linear_inline461__ssa_v0, 32], pl.FP32], hc_base_2d_inline462__ssa_v0: pl.Tensor[[1, 24], pl.FP32], comb_t__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 16], pl.FP32]], comb_tail_store_inline449__ssa_v0: pl.InOut[pl.Tensor[[8, 32], pl.FP32]]) -> tuple[pl.Tensor[[T_DYN, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32]]:
        return (comb_t__ssa_v0, comb_tail_store_inline449__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def compress_state_commit(compress_state__ssa_v0: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32], b_dim_inline3818__ssa_v0: pl.Scalar[pl.INDEX], commit_workers_inline3840__ssa_v0: pl.Scalar[pl.INDEX], s_dim_inline3837__ssa_v0: pl.Scalar[pl.INDEX], state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32], position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], cmp4_kv_proj_pad_inline3836__ssa_v0: pl.Tensor[[384, 1024], pl.FP32], cmp4_score_proj_pad_inline3838__ssa_v0: pl.Tensor[[384, 1024], pl.FP32], cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def compress_state_commit_0(inner_compress_state__ssa_v0: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32], b_dim_inline1638_inline3904__ssa_v0: pl.Scalar[pl.INDEX], commit_workers_inline1662_inline3965__ssa_v0: pl.Scalar[pl.INDEX], s_dim_inline1634_inline3899__ssa_v0: pl.Scalar[pl.INDEX], inner_state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32], position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], kv_proj_pad_inline3911__rv_v2: pl.Tensor[[384, 256], pl.FP32], score_proj_pad_inline3938__rv_v2: pl.Tensor[[384, 256], pl.FP32], inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def csa_cache_writeback(kv_cache_flat__ssa_v0: pl.Tensor[[ORI_BLOCK_NUM_DYN * pl.const(32, pl.INDEX), 512], pl.BF16], wb_blocks__ssa_v0: pl.Scalar[pl.INDEX], T_DYN: pl.Scalar[pl.INDEX], ori_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32], kv__ssa_v0: pl.Tensor[[T_DYN, 512], pl.BF16]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def csa_rope_sign(rope_sign_blocks__ssa_v0: pl.Scalar[pl.INDEX], T_DYN: pl.Scalar[pl.INDEX], freqs_sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32], idx_sin_signed__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 64], pl.FP32]]) -> pl.Tensor[[T_DYN, 64], pl.FP32]:
        return idx_sin_signed__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def csa_row_offsets(cmp_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], cmp_query_start_loc__ssa_v0: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32], cmp_row_offsets__ssa_v0: pl.Out[pl.Tensor[[B_DYN], pl.INT32]], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], idx_query_start_loc__ssa_v0: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32], idx_row_offsets__ssa_v0: pl.Out[pl.Tensor[[B_DYN], pl.INT32]]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def csa_slots_build_valid_qk_plan(cmp_sparse_indices_inline4329__ssa_v0: pl.Tensor[[t_dim_inline4335__ssa_v0, 512], pl.INT32], sparse_bias_inline4301__ssa_v0: pl.Tensor[[t_dim_inline4335__ssa_v0, 640], pl.FP32], t_dim_inline4335__ssa_v0: pl.Scalar[pl.INDEX], index_indices__ssa_v0: pl.Tensor[[T_DYN, 512], pl.INT32], position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT64], valid_block_mask_inline4413__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline4335__ssa_v0, 16], pl.INT32]], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], ori_block_table__ssa_v0: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32]) -> tuple[pl.Tensor[[t_dim_inline4335__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline4335__ssa_v0, 640], pl.FP32]]:
        return (cmp_sparse_indices_inline4329__ssa_v0, sparse_bias_inline4301__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def hc_pre_linear(mixes_partials_inline487__ssa_v0: pl.Tensor[[t_linear_inline461__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32], linear_units_inline444__ssa_v0: pl.Scalar[pl.INDEX], linear_workers_inline432__ssa_v0: pl.Scalar[pl.INDEX], t_dim_inline440__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline486__ssa_v0: pl.Tensor[[t_dim_inline440__ssa_v0, 16384], pl.FP32], hc_attn_fn__ssa_v0: pl.Tensor[[24, 16384], pl.FP32], t_linear_inline461__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[t_linear_inline461__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32]:
        return mixes_partials_inline487__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def hc_pre_linear_reduce(mixes_partials_inline487__rv_v2: pl.Tensor[[t_linear_inline461__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32], t_linear_inline461__ssa_v0: pl.Scalar[pl.INDEX], mixes_raw_inline448__ssa_v0: pl.Out[pl.Tensor[[t_linear_inline461__ssa_v0, 32], pl.FP32]]) -> pl.Tensor[[t_linear_inline461__ssa_v0, 32], pl.FP32]:
        return mixes_raw_inline448__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def hc_widen_rms(inv_rms__ssa_v0: pl.Tensor[[hc_padded_rows__ssa_v0, 1], pl.FP32], widen_tail__ssa_v0: pl.Tensor[[8, 16384], pl.FP32], x_hc32_flat__ssa_v0: pl.Tensor[[T_DYN, 16384], pl.FP32], widen_rows__ssa_v0: pl.Scalar[pl.INDEX], T_DYN: pl.Scalar[pl.INDEX], x_hc_flat__ssa_v0: pl.Tensor[[T_DYN, 16384], pl.BF16]) -> pl.Tensor[[hc_padded_rows__ssa_v0, 1], pl.FP32]:
        return inv_rms__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def idx_kv_scale_commit(compact_rows_inline4000__ssa_v0: pl.Scalar[pl.INDEX], position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], idx_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], idx_slot_mapping__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 2], pl.INT32], idx_kv_scale_values_inline3994__ssa_v1: pl.Tensor[[384, 1], pl.FP32], idx_native_kv_cache__rv_v2: pl.InOut[pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]]) -> pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]:
        return idx_native_kv_cache__rv_v2

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def idx_qr_dequant_rope(qr_bf16_2d_inline1995_inline4100__ssa_v0: pl.Tensor[[384, 8192], pl.BF16], dq_rope_units_inline2000_inline4054__ssa_v0: pl.Scalar[pl.INDEX], dq_rope_workers_inline2016_inline4062__ssa_v0: pl.Scalar[pl.INDEX], bs_inline1997_inline4109__ssa_v0: pl.Scalar[pl.INDEX], qr_scale__ssa_v0: pl.Tensor[[T_DYN, 1], pl.FP32], freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32], idx_sin_signed__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32], idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32], qr_acc_pad_inline2007_inline4067__rv_v2: pl.Tensor[[384, 8192], pl.INT32]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def idx_qr_proj_matmul(qr_acc_pad_inline2007_inline4067__ssa_v0: pl.Tensor[[384, 8192], pl.INT32], idx_wq_b__ssa_v0: pl.Tensor[[1024, 8192], pl.INT8], row_blocks_inline2004_inline4047__ssa_v0: pl.Scalar[pl.INDEX], bs_inline1997_inline4109__ssa_v0: pl.Scalar[pl.INDEX], qr__ssa_v0: pl.Tensor[[T_DYN, 1024], pl.INT8]) -> pl.Tensor[[384, 8192], pl.INT32]:
        return qr_acc_pad_inline2007_inline4067__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_boundary_init(normed_kv_inline574__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.BF16]]) -> pl.Tensor[[384, 128], pl.BF16]:
        return normed_kv_inline574__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_head_coefficients(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], coefficient_scales_inline1285_inline8894__ssa_v0: pl.Tensor[[384, 64], pl.FP32], weights_inline4197__ssa_v0: pl.Tensor[[384, 64], pl.FP32], coefficients_inline1281_inline8756__ssa_v0: pl.Out[pl.Tensor[[1024, 384], pl.FP16]]) -> pl.Tensor[[1024, 384], pl.FP16]:
        return coefficients_inline1281_inline8756__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_head_coefficients_0(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], coefficient_scales_inline2881_inline9074__ssa_v0: pl.Tensor[[384, 64], pl.FP32], weights_inline4197__ssa_v0: pl.Tensor[[384, 64], pl.FP32], coefficients_inline2879_inline9040__ssa_v0: pl.Out[pl.Tensor[[3072, 128], pl.FP16]]) -> pl.Tensor[[3072, 128], pl.FP16]:
        return coefficients_inline2879_inline9040__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_head_coefficients_1(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], coefficient_scales_inline1739_inline9508__ssa_v0: pl.Tensor[[384, 64], pl.FP32], weights_inline4197__ssa_v0: pl.Tensor[[384, 64], pl.FP32], coefficients_inline1735_inline9271__ssa_v0: pl.Out[pl.Tensor[[1024, 384], pl.FP16]]) -> pl.Tensor[[1024, 384], pl.FP16]:
        return coefficients_inline1735_inline9271__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_head_coefficients_2(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], coefficient_scales_inline3037_inline9683__ssa_v0: pl.Tensor[[384, 64], pl.FP32], weights_inline4197__ssa_v0: pl.Tensor[[384, 64], pl.FP32], coefficients_inline3035_inline9660__ssa_v0: pl.Out[pl.Tensor[[3072, 128], pl.FP16]]) -> pl.Tensor[[3072, 128], pl.FP16]:
        return coefficients_inline3035_inline9660__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_score_topk_leaf(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], score_arena_inline3255_inline4168__ssa_v0: pl.Tensor[[384, 8192], pl.FP32], qr_hadamard_i8_inline583__rv_v2: pl.Tensor[[24576, 128], pl.INT8], qr_hadamard_scale_dq_inline582__rv_v2: pl.Tensor[[24576, 1], pl.FP32], weights_inline4197__ssa_v0: pl.Tensor[[384, 64], pl.FP32], idx_block_table__ssa_v0: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32], native_page_bytes_inline3220_inline4177__ssa_v0: pl.Scalar[pl.INDEX], idx_kv_cache_inline3234_inline4199__ssa_v0: pl.Tensor[[key_rows_inline3222_inline4169__ssa_v0, 128], pl.INT8], idx_kv_cache_shift64_inline3230_inline4201__ssa_v0: pl.Tensor[[shifted_rows_inline3223_inline4178__ssa_v0, 128], pl.INT8], idx_native_kv_cache__rv_v2: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], pair_arena_inline3226_inline4202__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32]]) -> tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        return (score_arena_inline3255_inline4168__ssa_v0, pair_arena_inline3226_inline4202__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_score_topk_native_pair(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], buf_score_ffts_inline8757__ssa_v0: pl.Tensor[[256], pl.INT64], qr_hadamard_i8_inline583__rv_v2: pl.Tensor[[24576, 128], pl.INT8], coefficients_inline8733__ssa_v0: pl.Tensor[[1024, 384], pl.FP16], idx_block_table__ssa_v0: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32], native_page_bytes_inline8726__ssa_v0: pl.Scalar[pl.INDEX], idx_kv_cache_inline8854__ssa_v0: pl.Tensor[[key_rows_inline8656__ssa_v0, 128], pl.INT8], idx_kv_cache_shift64_inline8790__ssa_v0: pl.Tensor[[shifted_rows_inline8748__ssa_v0, 128], pl.INT8], buf_score_transfer_inline8845__ssa_v0: pl.InOut[pl.Tensor[[288, 1024], pl.FP32]], idx_native_kv_cache__rv_v2: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], pair_arena_inline3226_inline4202__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32]]) -> tuple[pl.Tensor[[288, 1024], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        return (buf_score_transfer_inline8845__ssa_v0, pair_arena_inline3226_inline4202__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_score_topk_native_pair_0(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], buf_score_ffts_inline8997__ssa_v0: pl.Tensor[[256], pl.INT64], qr_hadamard_i8_inline583__rv_v2: pl.Tensor[[24576, 128], pl.INT8], coefficients_inline9000__ssa_v0: pl.Tensor[[3072, 128], pl.FP16], idx_block_table__ssa_v0: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32], native_page_bytes_inline9037__ssa_v0: pl.Scalar[pl.INDEX], idx_kv_cache_inline9127__ssa_v0: pl.Tensor[[key_rows_inline9041__ssa_v0, 128], pl.INT8], idx_kv_cache_shift64_inline9079__ssa_v0: pl.Tensor[[shifted_rows_inline9108__ssa_v0, 128], pl.INT8], buf_score_transfer_inline9167__ssa_v0: pl.InOut[pl.Tensor[[96, 1024], pl.FP32]], idx_native_kv_cache__rv_v2: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], score_arena_inline3255_inline4168__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], pair_arena_inline3226_inline4202__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32]]) -> tuple[pl.Tensor[[96, 1024], pl.FP32], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        return (buf_score_transfer_inline9167__ssa_v0, score_arena_inline3255_inline4168__ssa_v0, pair_arena_inline3226_inline4202__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_score_topk_native_pair_1(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], buf_score_ffts_inline9502__ssa_v0: pl.Tensor[[256], pl.INT64], qr_hadamard_i8_inline583__rv_v2: pl.Tensor[[24576, 128], pl.INT8], coefficients_inline9363__ssa_v0: pl.Tensor[[1024, 384], pl.FP16], idx_block_table__ssa_v0: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32], native_page_bytes_inline9325__ssa_v0: pl.Scalar[pl.INDEX], idx_kv_cache_inline9317__ssa_v0: pl.Tensor[[key_rows_inline9391__ssa_v0, 128], pl.INT8], idx_kv_cache_shift64_inline9293__ssa_v0: pl.Tensor[[shifted_rows_inline9361__ssa_v0, 128], pl.INT8], buf_score_transfer_inline9382__ssa_v0: pl.InOut[pl.Tensor[[288, 768], pl.FP32]], idx_native_kv_cache__rv_v2: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], score_arena_inline3255_inline4168__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], pair_arena_inline3226_inline4202__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32]]) -> tuple[pl.Tensor[[288, 768], pl.FP32], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        return (buf_score_transfer_inline9382__ssa_v0, score_arena_inline3255_inline4168__ssa_v0, pair_arena_inline3226_inline4202__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_score_topk_native_pair_2(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], buf_score_ffts_inline9751__ssa_v0: pl.Tensor[[256], pl.INT64], qr_hadamard_i8_inline583__rv_v2: pl.Tensor[[24576, 128], pl.INT8], coefficients_inline9652__ssa_v0: pl.Tensor[[3072, 128], pl.FP16], idx_block_table__ssa_v0: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32], native_page_bytes_inline9662__ssa_v0: pl.Scalar[pl.INDEX], idx_kv_cache_inline9701__ssa_v0: pl.Tensor[[key_rows_inline9688__ssa_v0, 128], pl.INT8], idx_kv_cache_shift64_inline9817__ssa_v0: pl.Tensor[[shifted_rows_inline9696__ssa_v0, 128], pl.INT8], buf_score_transfer_inline9767__ssa_v0: pl.InOut[pl.Tensor[[96, 768], pl.FP32]], idx_native_kv_cache__rv_v2: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], score_arena_inline3255_inline4168__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], pair_arena_inline3226_inline4202__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32]]) -> tuple[pl.Tensor[[96, 768], pl.FP32], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        return (buf_score_transfer_inline9767__ssa_v0, score_arena_inline3255_inline4168__ssa_v0, pair_arena_inline3226_inline4202__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_topk_query_merge(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], pair_arena__ssa_v0: pl.Tensor[[24576, 1024], pl.FP32], topk_scores__ssa_v0: pl.Tensor[[T_DYN, 512], pl.FP32], topk_indices__ssa_v0: pl.Tensor[[T_DYN, 512], pl.INT32]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_topk_query_merge__2(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], pair_arena__ssa_v0: pl.Tensor[[24576, 1024], pl.FP32], topk_scores__ssa_v0: pl.Tensor[[T_DYN, 512], pl.FP32], topk_indices__ssa_v0: pl.Tensor[[T_DYN, 512], pl.INT32]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_topk_single_leaf_publish(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], score_arena__ssa_v0: pl.Tensor[[384, 8192], pl.FP32], topk_scores__ssa_v0: pl.Tensor[[T_DYN, 512], pl.FP32], topk_indices__ssa_v0: pl.Tensor[[T_DYN, 512], pl.INT32]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def kv_and_cache_write(compact_rows_inline4000__ssa_v0: pl.Scalar[pl.INDEX], kv_final_inline3998__rv_v2: pl.Tensor[[384, 128], pl.FP32], idx_kv_scale_values_inline3994__ssa_v0: pl.Out[pl.Tensor[[384, 1], pl.FP32]], idx_native_kv_cache__ssa_v0: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], kv_flat_inline4003__ssa_v0: pl.Tensor[[T_DYN, 128], pl.FP32], position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], idx_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], idx_slot_mapping__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 2], pl.INT32]) -> tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]]:
        return (idx_kv_scale_values_inline3994__ssa_v0, idx_native_kv_cache__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def kv_hadamard(kv_final_inline3998__ssa_v0: pl.Tensor[[384, 128], pl.FP32], hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16], rms_blocks_inline4008__ssa_v0: pl.Scalar[pl.INDEX], compact_rows_inline4000__ssa_v0: pl.Scalar[pl.INDEX], normed_kv_inline574__rv_v3: pl.Tensor[[384, 128], pl.BF16]) -> pl.Tensor[[384, 128], pl.FP32]:
        return kv_final_inline3998__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def kv_proj_matmul(kv_m_groups_inline1466_inline3638__ssa_v0: pl.Scalar[pl.INDEX], kv_fp32_inline3665__ssa_v0: pl.Tensor[[t_matmul_inline3663__ssa_v0, 512], pl.FP32], kv_full_rows_inline1452_inline3639__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline3655__idx_v0: pl.Scalar[pl.INDEX], x_view_inline1451_inline3640__ssa_v0: pl.Tensor[[t_dim_inline1450_inline3644__ssa_v0, 4096], pl.BF16], wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16], t_matmul_inline3663__ssa_v0: pl.Scalar[pl.INDEX], tile_rows_inline3661__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[t_matmul_inline3663__ssa_v0, 512], pl.FP32]:
        return kv_fp32_inline3665__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def kv_proj_matmul_0(kv_m_groups_inline1490_inline3660__ssa_v0: pl.Scalar[pl.INDEX], kv_fp32_inline3665__ssa_v0: pl.Tensor[[t_matmul_inline3663__ssa_v0, 512], pl.FP32], kv_full_rows_inline1475_inline3651__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline3655__idx_v0: pl.Scalar[pl.INDEX], x_view_inline1476_inline3614__ssa_v0: pl.Tensor[[t_dim_inline1474_inline3632__ssa_v0, 4096], pl.BF16], wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16], t_matmul_inline3663__ssa_v0: pl.Scalar[pl.INDEX], tile_rows_inline3661__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[t_matmul_inline3663__ssa_v0, 512], pl.FP32]:
        return kv_fp32_inline3665__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def kv_rms_norm_rope(tile_rows_inline3661__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline3655__idx_v0: pl.Scalar[pl.INDEX], kv_fp32_inline3665__phi_v23: pl.Tensor[[t_matmul_inline3663__ssa_v0, 512], pl.FP32], kv_view_inline3736__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline3668__ssa_v0, 512], pl.BF16]], gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16], freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32], q_rope_sin_signed_inline558__ssa_v0: pl.Tensor[[t_dim_inline560__ssa_v0, 64], pl.FP32], q_rope_swap_idx_inline557__ssa_v0: pl.Tensor[[t_dim_inline560__ssa_v0, 64], pl.INT32]) -> pl.Tensor[[t_dim_inline3668__ssa_v0, 512], pl.BF16]:
        return kv_view_inline3736__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def kv_score_proj(cmp4_kv_proj_pad_inline281_inline3787__ssa_v0: pl.Tensor[[384, 1024], pl.FP32], cmp4_score_proj_pad_inline284_inline3769__ssa_v0: pl.Tensor[[384, 1024], pl.FP32], t_matmul_inline274_inline3773__ssa_v0: pl.Scalar[pl.INDEX], bs_inline273_inline3759__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline278_inline3772__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16], cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16], cmp_wgate__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def kv_score_proj_0(kv_proj_pad_inline3911__ssa_v0: pl.Tensor[[384, 256], pl.FP32], score_proj_pad_inline3938__ssa_v0: pl.Tensor[[384, 256], pl.FP32], t_matmul_inline1594_inline3937__ssa_v0: pl.Scalar[pl.INDEX], bs_inline1593_inline3928__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline1595_inline3942__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16], inner_wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16], inner_wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16]) -> tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32]]:
        return (kv_proj_pad_inline3911__ssa_v0, score_proj_pad_inline3938__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def kv_touch(ori_kv_flat_inline4326__ssa_v0: pl.InOut[pl.Tensor[[ori_block_num_inline4369__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16]]) -> pl.Tensor[[ori_block_num_inline4369__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16]:
        return ori_kv_flat_inline4326__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def mix_x_rms_norm(t_dim_inline529__ssa_v0: pl.Scalar[pl.INDEX], pre_val_store__rv_v2: pl.Tensor[[hc_padded_rows__ssa_v0, 8], pl.FP32], x_mixed_inline534__ssa_v0: pl.Tensor[[t_pad_inline523__ssa_v0, 4096], pl.BF16], x_flat_inline525__ssa_v0: pl.Tensor[[t_dim_inline529__ssa_v0, 16384], pl.FP32], x_normed_t__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16], x_normed_tail_store_inline516__ssa_v0: pl.Tensor[[8, 4096], pl.BF16], attn_norm_w__ssa_v0: pl.Tensor[[4096], pl.BF16]) -> pl.Tensor[[T_DYN, 4096], pl.BF16]:
        return x_normed_t__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def proj_a_mm(t_dim_inline4518__ssa_v0: pl.Scalar[pl.INDEX], row_base_o_inline4519__ssa_v0: pl.Scalar[pl.INDEX], o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16], wo_a__ssa_v0: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], g_inline4502__idx_v0: pl.Scalar[pl.INDEX], o_r_pad_inline4514__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.FP32]], out_col_g_inline4498__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({'cache_policy': [(3, 1)]})
        pa_unit_inline351_inline4491__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pa_rb_inline338_inline4487__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline351_inline4491__ssa_v0 // 8
        nf_inline337_inline4486__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline351_inline4491__ssa_v0 % 8
        pa_r0_inline344_inline4524__ssa_v0: pl.Scalar[pl.INDEX] = pa_rb_inline338_inline4487__ssa_v0 * 128
        pa_rows_inline333_inline4485__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline4518__ssa_v0 - pa_r0_inline344_inline4524__ssa_v0, 128)
        pa_src0_inline330_inline4483__ssa_v0: pl.Scalar[pl.INDEX] = row_base_o_inline4519__ssa_v0 + pa_r0_inline344_inline4524__ssa_v0
        n0_inline346_inline4500__ssa_v0: pl.Scalar[pl.INDEX] = nf_inline337_inline4486__ssa_v0 * 128
        xa_first_inline327_inline4484__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows_inline333_inline4485__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(o_packed_heads__ssa_v1, [128, 256], [pa_src0_inline330_inline4483__ssa_v0, 0], [pa_rows_inline333_inline4485__ssa_v0, 256])
        wa_first_inline348_inline4494__ssa_v0: pl.Tensor[[1, 256, 128], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wo_a__ssa_v0, [1, 256, 128], [g_inline4502__idx_v0, 0, n0_inline346_inline4500__ssa_v0])
        acc_a_inline350_inline4506__ssa_v0: pl.Tensor[[1, 128, 128], pl.FP32] = pl.tensor.matmul(xa_first_inline327_inline4484__ssa_v0, wa_first_inline348_inline4494__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (kb_inline353_inline4541__idx_v0, (acc_a_inline350_inline4506__iter_v1,)) in pl.pipeline(1, 16, stage=2, init_values=(acc_a_inline350_inline4506__ssa_v0,)):
            k0_inline354_inline4495__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline353_inline4541__idx_v0 * 256
            xa_k_chunk_inline331_inline4479__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows_inline333_inline4485__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(o_packed_heads__ssa_v1, [128, 256], [pa_src0_inline330_inline4483__ssa_v0, k0_inline354_inline4495__ssa_v0], [pa_rows_inline333_inline4485__ssa_v0, 256])
            wa_k_chunk_inline324_inline4527__ssa_v0: pl.Tensor[[1, 256, 128], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wo_a__ssa_v0, [1, 256, 128], [g_inline4502__idx_v0, k0_inline354_inline4495__ssa_v0, n0_inline346_inline4500__ssa_v0])
            acc_a_inline350_inline4506__ssa_v3: pl.Tensor[[1, 128, 128], pl.FP32] = pl.tensor.matmul_acc(acc_a_inline350_inline4506__iter_v1, xa_k_chunk_inline331_inline4479__ssa_v0, wa_k_chunk_inline324_inline4527__ssa_v0, a_trans=False, b_trans=False)
            acc_a_inline350_inline4506__rv_v2: pl.Tensor[[1, 128, 128], pl.FP32] = pl.yield_(acc_a_inline350_inline4506__ssa_v3)
        stored_inline323_inline4542__ssa_v0: pl.Tensor[[384, 8192], pl.FP32] = pl.tensor.assemble(o_r_pad_inline4514__iter_v1, acc_a_inline350_inline4506__rv_v2, [pa_r0_inline344_inline4524__ssa_v0, out_col_g_inline4498__ssa_v0 + n0_inline346_inline4500__ssa_v0])
        return o_r_pad_inline4514__iter_v1

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def proj_a_mm_0(t_dim_inline4627__ssa_v0: pl.Scalar[pl.INDEX], row_base_o_inline4605__ssa_v0: pl.Scalar[pl.INDEX], o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16], wo_a__ssa_v0: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], g_inline4648__idx_v0: pl.Scalar[pl.INDEX], o_r_pad_inline4642__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.FP32]], out_col_g_inline4626__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({'cache_policy': [(3, 1)]})
        pa_unit_inline616_inline4620__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pa_rb_inline603_inline4636__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline616_inline4620__ssa_v0 // 8
        nf_inline602_inline4615__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline616_inline4620__ssa_v0 % 8
        pa_r0_inline609_inline4640__ssa_v0: pl.Scalar[pl.INDEX] = pa_rb_inline603_inline4636__ssa_v0 * 128
        pa_rows_inline598_inline4613__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline4627__ssa_v0 - pa_r0_inline609_inline4640__ssa_v0, 128)
        pa_src0_inline595_inline4609__ssa_v0: pl.Scalar[pl.INDEX] = row_base_o_inline4605__ssa_v0 + pa_r0_inline609_inline4640__ssa_v0
        n0_inline611_inline4643__ssa_v0: pl.Scalar[pl.INDEX] = nf_inline602_inline4615__ssa_v0 * 128
        xa_first_inline592_inline4659__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows_inline598_inline4613__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(o_packed_heads__ssa_v1, [128, 256], [pa_src0_inline595_inline4609__ssa_v0, 0], [pa_rows_inline598_inline4613__ssa_v0, 256])
        wa_first_inline613_inline4622__ssa_v0: pl.Tensor[[1, 256, 128], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wo_a__ssa_v0, [1, 256, 128], [g_inline4648__idx_v0, 0, n0_inline611_inline4643__ssa_v0])
        acc_a_inline615_inline4670__ssa_v0: pl.Tensor[[1, 128, 128], pl.FP32] = pl.tensor.matmul(xa_first_inline592_inline4659__ssa_v0, wa_first_inline613_inline4622__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (kb_inline618_inline4672__idx_v0, (acc_a_inline615_inline4670__iter_v1,)) in pl.pipeline(1, 16, stage=2, init_values=(acc_a_inline615_inline4670__ssa_v0,)):
            k0_inline619_inline4623__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline618_inline4672__idx_v0 * 256
            xa_k_chunk_inline596_inline4602__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows_inline598_inline4613__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(o_packed_heads__ssa_v1, [128, 256], [pa_src0_inline595_inline4609__ssa_v0, k0_inline619_inline4623__ssa_v0], [pa_rows_inline598_inline4613__ssa_v0, 256])
            wa_k_chunk_inline589_inline4614__ssa_v0: pl.Tensor[[1, 256, 128], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wo_a__ssa_v0, [1, 256, 128], [g_inline4648__idx_v0, k0_inline619_inline4623__ssa_v0, n0_inline611_inline4643__ssa_v0])
            acc_a_inline615_inline4670__ssa_v3: pl.Tensor[[1, 128, 128], pl.FP32] = pl.tensor.matmul_acc(acc_a_inline615_inline4670__iter_v1, xa_k_chunk_inline596_inline4602__ssa_v0, wa_k_chunk_inline589_inline4614__ssa_v0, a_trans=False, b_trans=False)
            acc_a_inline615_inline4670__rv_v2: pl.Tensor[[1, 128, 128], pl.FP32] = pl.yield_(acc_a_inline615_inline4670__ssa_v3)
        stored_inline588_inline4607__ssa_v0: pl.Tensor[[384, 8192], pl.FP32] = pl.tensor.assemble(o_r_pad_inline4642__iter_v1, acc_a_inline615_inline4670__rv_v2, [pa_r0_inline609_inline4640__ssa_v0, out_col_g_inline4626__ssa_v0 + n0_inline611_inline4643__ssa_v0])
        return o_r_pad_inline4642__iter_v1

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def proj_a_mm_1(t_dim_inline4745__ssa_v0: pl.Scalar[pl.INDEX], row_base_o_inline4772__ssa_v0: pl.Scalar[pl.INDEX], o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16], wo_a__ssa_v0: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], g_inline4747__idx_v0: pl.Scalar[pl.INDEX], o_r_pad_inline4770__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.FP32]], out_col_g_inline4735__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({'cache_policy': [(3, 1)]})
        pa_unit_inline1254_inline4751__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pa_rb_inline1241_inline4721__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline1254_inline4751__ssa_v0 // 8
        nf_inline1240_inline4737__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline1254_inline4751__ssa_v0 % 8
        pa_r0_inline1247_inline4724__ssa_v0: pl.Scalar[pl.INDEX] = pa_rb_inline1241_inline4721__ssa_v0 * 128
        pa_rows_inline1236_inline4766__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline4745__ssa_v0 - pa_r0_inline1247_inline4724__ssa_v0, 128)
        pa_src0_inline1233_inline4716__ssa_v0: pl.Scalar[pl.INDEX] = row_base_o_inline4772__ssa_v0 + pa_r0_inline1247_inline4724__ssa_v0
        n0_inline1249_inline4719__ssa_v0: pl.Scalar[pl.INDEX] = nf_inline1240_inline4737__ssa_v0 * 128
        xa_first_inline1230_inline4781__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows_inline1236_inline4766__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(o_packed_heads__ssa_v1, [128, 256], [pa_src0_inline1233_inline4716__ssa_v0, 0], [pa_rows_inline1236_inline4766__ssa_v0, 256])
        wa_first_inline1251_inline4755__ssa_v0: pl.Tensor[[1, 256, 128], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wo_a__ssa_v0, [1, 256, 128], [g_inline4747__idx_v0, 0, n0_inline1249_inline4719__ssa_v0])
        acc_a_inline1253_inline4743__ssa_v0: pl.Tensor[[1, 128, 128], pl.FP32] = pl.tensor.matmul(xa_first_inline1230_inline4781__ssa_v0, wa_first_inline1251_inline4755__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (kb_inline1256_inline4783__idx_v0, (acc_a_inline1253_inline4743__iter_v1,)) in pl.pipeline(1, 16, stage=2, init_values=(acc_a_inline1253_inline4743__ssa_v0,)):
            k0_inline1257_inline4757__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline1256_inline4783__idx_v0 * 256
            xa_k_chunk_inline1234_inline4713__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows_inline1236_inline4766__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(o_packed_heads__ssa_v1, [128, 256], [pa_src0_inline1233_inline4716__ssa_v0, k0_inline1257_inline4757__ssa_v0], [pa_rows_inline1236_inline4766__ssa_v0, 256])
            wa_k_chunk_inline1227_inline4778__ssa_v0: pl.Tensor[[1, 256, 128], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wo_a__ssa_v0, [1, 256, 128], [g_inline4747__idx_v0, k0_inline1257_inline4757__ssa_v0, n0_inline1249_inline4719__ssa_v0])
            acc_a_inline1253_inline4743__ssa_v3: pl.Tensor[[1, 128, 128], pl.FP32] = pl.tensor.matmul_acc(acc_a_inline1253_inline4743__iter_v1, xa_k_chunk_inline1234_inline4713__ssa_v0, wa_k_chunk_inline1227_inline4778__ssa_v0, a_trans=False, b_trans=False)
            acc_a_inline1253_inline4743__rv_v2: pl.Tensor[[1, 128, 128], pl.FP32] = pl.yield_(acc_a_inline1253_inline4743__ssa_v3)
        stored_inline1226_inline4785__ssa_v0: pl.Tensor[[384, 8192], pl.FP32] = pl.tensor.assemble(o_r_pad_inline4770__iter_v1, acc_a_inline1253_inline4743__rv_v2, [pa_r0_inline1247_inline4724__ssa_v0, out_col_g_inline4735__ssa_v0 + n0_inline1249_inline4719__ssa_v0])
        return o_r_pad_inline4770__iter_v1

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def proj_a_mm_2(t_dim_inline4891__ssa_v0: pl.Scalar[pl.INDEX], row_base_o_inline4886__ssa_v0: pl.Scalar[pl.INDEX], o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16], wo_a__ssa_v0: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], g_inline4868__idx_v0: pl.Scalar[pl.INDEX], o_r_pad_inline4871__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.FP32]], out_col_g_inline4870__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({'cache_policy': [(3, 1)]})
        pa_unit_inline656_inline4867__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pa_rb_inline653_inline4866__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline656_inline4867__ssa_v0 // 4
        nf_inline650_inline4863__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline656_inline4867__ssa_v0 % 4
        pa_r0_inline652_inline4902__ssa_v0: pl.Scalar[pl.INDEX] = pa_rb_inline653_inline4866__ssa_v0 * 128
        pa_rows_inline655_inline4853__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline4891__ssa_v0 - pa_r0_inline652_inline4902__ssa_v0, 128)
        pa_src0_inline648_inline4858__ssa_v0: pl.Scalar[pl.INDEX] = row_base_o_inline4886__ssa_v0 + pa_r0_inline652_inline4902__ssa_v0
        n0_inline645_inline4885__ssa_v0: pl.Scalar[pl.INDEX] = nf_inline650_inline4863__ssa_v0 * 256
        xa_first_inline647_inline4864__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows_inline655_inline4853__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(o_packed_heads__ssa_v1, [128, 256], [pa_src0_inline648_inline4858__ssa_v0, 0], [pa_rows_inline655_inline4853__ssa_v0, 256])
        wa_first_inline667_inline4900__ssa_v0: pl.Tensor[[1, 256, 256], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wo_a__ssa_v0, [1, 256, 256], [g_inline4868__idx_v0, 0, n0_inline645_inline4885__ssa_v0])
        acc_a_inline673_inline4888__ssa_v0: pl.Tensor[[1, 128, 256], pl.FP32] = pl.tensor.matmul(xa_first_inline647_inline4864__ssa_v0, wa_first_inline667_inline4900__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
        for (kb_inline664_inline4909__idx_v0, (acc_a_inline673_inline4888__iter_v1,)) in pl.pipeline(1, 16, stage=2, init_values=(acc_a_inline673_inline4888__ssa_v0,)):
            k0_inline659_inline4905__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline664_inline4909__idx_v0 * 256
            xa_k_chunk_inline643_inline4903__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows_inline655_inline4853__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(o_packed_heads__ssa_v1, [128, 256], [pa_src0_inline648_inline4858__ssa_v0, k0_inline659_inline4905__ssa_v0], [pa_rows_inline655_inline4853__ssa_v0, 256])
            wa_k_chunk_inline642_inline4875__ssa_v0: pl.Tensor[[1, 256, 256], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wo_a__ssa_v0, [1, 256, 256], [g_inline4868__idx_v0, k0_inline659_inline4905__ssa_v0, n0_inline645_inline4885__ssa_v0])
            acc_a_inline673_inline4888__ssa_v3: pl.Tensor[[1, 128, 256], pl.FP32] = pl.tensor.matmul_acc(acc_a_inline673_inline4888__iter_v1, xa_k_chunk_inline643_inline4903__ssa_v0, wa_k_chunk_inline642_inline4875__ssa_v0, a_trans=False, b_trans=False)
            acc_a_inline673_inline4888__rv_v2: pl.Tensor[[1, 128, 256], pl.FP32] = pl.yield_(acc_a_inline673_inline4888__ssa_v3)
        stored_inline668_inline4904__ssa_v0: pl.Tensor[[384, 8192], pl.FP32] = pl.tensor.assemble(o_r_pad_inline4871__iter_v1, acc_a_inline673_inline4888__rv_v2, [pa_r0_inline652_inline4902__ssa_v0, out_col_g_inline4870__ssa_v0 + n0_inline645_inline4885__ssa_v0])
        return o_r_pad_inline4871__iter_v1

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def proj_b_act_hc_post(act_row_step_inline4515__ssa_v1: pl.Scalar[pl.INDEX], wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32], y_flat_inline4468__ssa_v0: pl.Tensor[[t_dim_inline4518__ssa_v0, 16384], pl.BF16], t_dim_inline4518__ssa_v0: pl.Scalar[pl.INDEX], partials_inline4512__rv_v2: pl.Tensor[[384, 32768], pl.INT32], act_scale_dq_inline4531__rv_v2: pl.Tensor[[8, 384], pl.FP32], residual_flat_inline4469__ssa_v0: pl.Tensor[[t_dim_inline4518__ssa_v0, 16384], pl.BF16], post_t__rv_v2: pl.Tensor[[T_DYN, 4], pl.FP32], comb_t__ssa_v0: pl.Tensor[[T_DYN, 16], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def proj_b_act_hc_post_0(act_row_step_inline4633__ssa_v0: pl.Scalar[pl.INDEX], wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32], y_flat_inline4588__ssa_v0: pl.Tensor[[t_dim_inline4627__ssa_v0, 16384], pl.BF16], t_dim_inline4627__ssa_v0: pl.Scalar[pl.INDEX], partials_inline4641__rv_v2: pl.Tensor[[384, 32768], pl.INT32], act_scale_dq_inline4655__rv_v2: pl.Tensor[[8, 384], pl.FP32], residual_flat_inline4663__ssa_v0: pl.Tensor[[t_dim_inline4627__ssa_v0, 16384], pl.BF16], post_t__rv_v2: pl.Tensor[[T_DYN, 4], pl.FP32], comb_t__ssa_v0: pl.Tensor[[T_DYN, 16], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def proj_b_act_hc_post_1(act_row_step_inline4746__ssa_v1: pl.Scalar[pl.INDEX], wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32], y_flat_inline4761__ssa_v0: pl.Tensor[[t_dim_inline4745__ssa_v0, 16384], pl.BF16], t_dim_inline4745__ssa_v0: pl.Scalar[pl.INDEX], partials_inline4759__rv_v2: pl.Tensor[[384, 32768], pl.INT32], act_scale_dq_inline4765__rv_v2: pl.Tensor[[8, 384], pl.FP32], residual_flat_inline4703__ssa_v0: pl.Tensor[[t_dim_inline4745__ssa_v0, 16384], pl.BF16], post_t__rv_v2: pl.Tensor[[T_DYN, 4], pl.FP32], comb_t__ssa_v0: pl.Tensor[[T_DYN, 16], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def proj_b_act_hc_post_2(act_row_step_inline4887__ssa_v1: pl.Scalar[pl.INDEX], wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32], y_flat_inline4862__ssa_v0: pl.Tensor[[t_dim_inline4891__ssa_v0, 16384], pl.BF16], t_dim_inline4891__ssa_v0: pl.Scalar[pl.INDEX], partials_inline4880__rv_v2: pl.Tensor[[384, 32768], pl.INT32], act_scale_dq_inline4892__rv_v2: pl.Tensor[[8, 384], pl.FP32], residual_flat_inline4834__ssa_v0: pl.Tensor[[t_dim_inline4891__ssa_v0, 16384], pl.BF16], post_t__rv_v2: pl.Tensor[[T_DYN, 4], pl.FP32], comb_t__ssa_v0: pl.Tensor[[T_DYN, 16], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def q_rope_prepare(rope_sin_signed_view_inline3575__ssa_v0: pl.Tensor[[t_dim_inline3567__ssa_v0, 64], pl.FP32], rope_swap_idx_view_inline3564__ssa_v0: pl.Tensor[[t_dim_inline3567__ssa_v0, 64], pl.INT32], token_tiles_inline3560__ssa_v0: pl.Scalar[pl.INDEX], t_dim_inline3567__ssa_v0: pl.Scalar[pl.INDEX], rope_sin_view_inline3559__ssa_v0: pl.Tensor[[t_dim_inline3567__ssa_v0, 64], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def qk_pv(ffts_workspace_inline4361__ssa_v0: pl.Tensor[[256], pl.INT64], t_dim_inline4335__ssa_v0: pl.Scalar[pl.INDEX], q_flat_inline4389__ssa_v0: pl.Tensor[[t_heads_inline4306__ssa_v0, 512], pl.BF16], valid_block_mask_inline4413__ssa_v0: pl.Tensor[[t_dim_inline4335__ssa_v0, 16], pl.INT32], kv_transfer_inline4342__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], score_transfer_inline4398__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], probability_transfer_inline4399__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], pv_transfer_inline4320__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], attn_sink_col_inline4382__ssa_v0: pl.Tensor[[64, 1], pl.FP32], position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT64], ori_block_table__ssa_v0: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32], ori_kv_flat_inline4326__ssa_v1: pl.Tensor[[ori_block_num_inline4369__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16], cmp_sparse_indices_inline4329__rv_v2: pl.Tensor[[t_dim_inline4335__ssa_v0, 512], pl.INT32], cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32], cmp_kv_flat_inline4388__ssa_v0: pl.Tensor[[cmp_block_num_inline4371__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16], sparse_bias_inline4301__rv_v2: pl.Tensor[[t_dim_inline4335__ssa_v0, 640], pl.FP32], mi_transfer_inline4318__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], li_transfer_inline4403__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32], rope_sin_signed_inline4321__ssa_v0: pl.Tensor[[384, 64], pl.FP32], o_packed_heads__ssa_v0: pl.Out[pl.Tensor[[3072, 4096], pl.BF16]]) -> tuple[pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16]]:
        return (kv_transfer_inline4342__ssa_v0, score_transfer_inline4398__ssa_v0, probability_transfer_inline4399__ssa_v0, pv_transfer_inline4320__ssa_v0, mi_transfer_inline4318__ssa_v0, li_transfer_inline4403__ssa_v0, o_packed_heads__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def qproj_dequant_rms_nope_rope(tile_rows_inline11_inline3587__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline12_inline3596__idx_v0: pl.Scalar[pl.INDEX], qr_scale_pad_store_inline3589__ssa_v0: pl.Tensor[[512, 1], pl.FP32], freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32], q_rope_sin_signed_inline558__ssa_v0: pl.Tensor[[t_dim_inline560__ssa_v0, 64], pl.FP32], q_rope_swap_idx_inline557__ssa_v0: pl.Tensor[[t_dim_inline560__ssa_v0, 64], pl.INT32], q_proj_i32_inline7_inline3585__ssa_v6: pl.Tensor[[qproj_t_matmul_inline10_inline3586__ssa_v0, 32768], pl.INT32], wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32], q_flat_inline8548__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline8556__ssa_v0, 32768], pl.BF16]]) -> pl.Tensor[[t_dim_inline8556__ssa_v0, 32768], pl.BF16]:
        return q_flat_inline8548__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def qproj_matmul(q_proj_i32_inline7_inline3585__ssa_v0: pl.Tensor[[qproj_t_matmul_inline10_inline3586__ssa_v0, 32768], pl.INT32], qproj_t_matmul_inline8513__ssa_v0: pl.Scalar[pl.INDEX], tile_rows_inline11_inline3587__ssa_v0: pl.Scalar[pl.INDEX], qr_i8_matmul_inline3588__ssa_v0: pl.Tensor[[512, 1024], pl.INT8], wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)]) -> pl.Tensor[[qproj_t_matmul_inline10_inline3586__ssa_v0, 32768], pl.INT32]:
        pl.func_attr({'cache_policy': [(4, 1)]})
        qproj_worker_inline8514__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for (qproj_round_inline8515__idx_v0, (q_proj_i32_inline7_inline3585__iter_v1,)) in pl.range((19 - qproj_worker_inline8514__ssa_v0 + 128) // 20, init_values=(q_proj_i32_inline7_inline3585__ssa_v0,)):
            qproj_n_idx_inline8522__ssa_v0: pl.Scalar[pl.INDEX] = qproj_worker_inline8514__ssa_v0 + qproj_round_inline8515__idx_v0 * 20
            w_col0_inline8519__ssa_v0: pl.Scalar[pl.INDEX] = qproj_n_idx_inline8522__ssa_v0 * 256
            for (t0_inline8524__idx_v0, (q_proj_i32_inline7_inline3585__iter_v3,)) in pl.range(0, qproj_t_matmul_inline8513__ssa_v0, 128, init_values=(q_proj_i32_inline7_inline3585__iter_v1,)):
                m_rows_inline8520__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline11_inline3587__ssa_v0 - t0_inline8524__idx_v0, 128)
                qr_first_inline8521__ssa_v0: pl.Tensor[[128, 256], pl.INT8, pl.TensorView(valid_shape=[m_rows_inline8520__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(qr_i8_matmul_inline3588__ssa_v0, [128, 256], [t0_inline8524__idx_v0, 0], [m_rows_inline8520__ssa_v0, 256])
                wq_first_inline8518__ssa_v0: pl.Tensor[[256, 256], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wq_b__ssa_v0, [256, 256], [0, w_col0_inline8519__ssa_v0])
                col_acc_inline8523__ssa_v0: pl.Tensor[[128, 256], pl.INT32] = pl.tensor.matmul(qr_first_inline8521__ssa_v0, wq_first_inline8518__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                for (k0_inline8525__idx_v0, (col_acc_inline8523__iter_v1,)) in pl.pipeline(256, 1024, 256, stage=2, init_values=(col_acc_inline8523__ssa_v0,)):
                    qr_chunk_inline8516__ssa_v0: pl.Tensor[[128, 256], pl.INT8, pl.TensorView(valid_shape=[m_rows_inline8520__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(qr_i8_matmul_inline3588__ssa_v0, [128, 256], [t0_inline8524__idx_v0, k0_inline8525__idx_v0], [m_rows_inline8520__ssa_v0, 256])
                    wq_chunk_inline8512__ssa_v0: pl.Tensor[[256, 256], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wq_b__ssa_v0, [256, 256], [k0_inline8525__idx_v0, w_col0_inline8519__ssa_v0])
                    col_acc_inline8523__ssa_v3: pl.Tensor[[128, 256], pl.INT32] = pl.tensor.matmul_acc(col_acc_inline8523__iter_v1, qr_chunk_inline8516__ssa_v0, wq_chunk_inline8512__ssa_v0, a_trans=False, b_trans=False)
                    col_acc_inline8523__rv_v2: pl.Tensor[[128, 256], pl.INT32] = pl.yield_(col_acc_inline8523__ssa_v3)
                q_proj_i32_inline7_inline3585__ssa_v5: pl.Tensor[[QKV_QPROJ_MM_T_DYN, 32768], pl.INT32] = pl.tensor.assemble(q_proj_i32_inline7_inline3585__iter_v3, col_acc_inline8523__rv_v2, [t0_inline8524__idx_v0, w_col0_inline8519__ssa_v0])
                q_proj_i32_inline7_inline3585__rv_v4: pl.Tensor[[qproj_t_matmul_inline10_inline3586__ssa_v0, 32768], pl.INT32] = pl.yield_(q_proj_i32_inline7_inline3585__ssa_v5)
            q_proj_i32_inline7_inline3585__rv_v2: pl.Tensor[[qproj_t_matmul_inline10_inline3586__ssa_v0, 32768], pl.INT32] = pl.yield_(q_proj_i32_inline7_inline3585__rv_v4)
        return q_proj_i32_inline7_inline3585__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def qr_hadamard_matmul(hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16], qh_acc_gm_inline2065_inline4066__ssa_v0: pl.Tensor[[bs_heads_inline2066_inline4048__ssa_v0, 128], pl.FP32], bs_heads_inline2066_inline4048__ssa_v0: pl.Scalar[pl.INDEX], qr_bf16_inline4070__ssa_v0: pl.Tensor[[24576, 128], pl.BF16]) -> pl.Tensor[[bs_heads_inline2066_inline4048__ssa_v0, 128], pl.FP32]:
        return qh_acc_gm_inline2065_inline4066__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def qr_hadamard_quant(qr_hadamard_i8_inline583__ssa_v0: pl.Tensor[[24576, 128], pl.INT8], qr_hadamard_scale_dq_inline582__ssa_v0: pl.Tensor[[24576, 1], pl.FP32], bs_heads_inline2066_inline4048__ssa_v0: pl.Scalar[pl.INDEX], qh_acc_gm_inline2065_inline4066__rv_v2: pl.Tensor[[bs_heads_inline2066_inline4048__ssa_v0, 128], pl.FP32]) -> tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32]]:
        return (qr_hadamard_i8_inline583__ssa_v0, qr_hadamard_scale_dq_inline582__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def qr_proj_matmul(qr_fp32_inline1_inline3598__ssa_v0: pl.Tensor[[qr_t_matmul_inline2_inline3597__ssa_v0, 1024], pl.FP32], qr_full_rows_inline53_inline8466__ssa_v0: pl.Scalar[pl.INDEX], qr_m_groups_inline58_inline8419__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline3_inline3594__idx_v0: pl.Scalar[pl.INDEX], x_view_inline50_inline8453__ssa_v0: pl.Tensor[[qa_tokens_inline48_inline8448__ssa_v0, 4096], pl.BF16], wq_a__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], qr_t_matmul_inline54_inline8450__ssa_v0: pl.Scalar[pl.INDEX], tile_rows_inline4_inline3595__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[qr_t_matmul_inline2_inline3597__ssa_v0, 1024], pl.FP32]:
        pl.func_attr({'cache_policy': [(5, 1)]})
        qbg_idx_inline63_inline8457__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qr_k_base_inline65_inline8426__ssa_v0: pl.Scalar[pl.INDEX] = 0
        q_a_col0_inline66_inline8410__ssa_v0: pl.Scalar[pl.INDEX] = qbg_idx_inline63_inline8457__ssa_v0 % 8 * 128
        qr_m_group_inline67_inline8440__ssa_v0: pl.Scalar[pl.INDEX] = qbg_idx_inline63_inline8457__ssa_v0 // 8
        for (dense_t0_inline47_inline8449__idx_v0, (qr_fp32_inline1_inline3598__iter_v15,)) in pl.range(qr_m_group_inline67_inline8440__ssa_v0 * 32, qr_full_rows_inline53_inline8466__ssa_v0, qr_m_groups_inline58_inline8419__ssa_v0 * 32, init_values=(qr_fp32_inline1_inline3598__ssa_v0,)):
            dense_x0_inline45_inline8464__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline3_inline3594__idx_v0 + dense_t0_inline47_inline8449__idx_v0
            dense_acc_inline44_inline8432__ssa_v0: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.create([32, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for (dense_k_inline57_inline8458__idx_v0, (dense_acc_inline44_inline8432__iter_v1,)) in pl.pipeline(16, stage=2, init_values=(dense_acc_inline44_inline8432__ssa_v0,)):
                dense_d0_inline43_inline8465__ssa_v0: pl.Scalar[pl.INDEX] = qr_k_base_inline65_inline8426__ssa_v0 + dense_k_inline57_inline8458__idx_v0 * 256
                dense_x_inline46_inline8467__ssa_v0: pl.Tensor[[32, 256], pl.BF16] = pl.tensor.slice(x_view_inline50_inline8453__ssa_v0, [32, 256], [dense_x0_inline45_inline8464__ssa_v0, dense_d0_inline43_inline8465__ssa_v0])
                dense_w_inline64_inline8462__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wq_a__ssa_v0, [128, 256], [q_a_col0_inline66_inline8410__ssa_v0, dense_d0_inline43_inline8465__ssa_v0])
                dense_acc_inline44_inline8432__ssa_v3: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.matmul_acc(dense_acc_inline44_inline8432__iter_v1, dense_x_inline46_inline8467__ssa_v0, dense_w_inline64_inline8462__ssa_v0, init_cond=dense_k_inline57_inline8458__idx_v0 == 0, a_trans=False, b_trans=True)
                dense_acc_inline44_inline8432__rv_v2: pl.Tensor[[32, 128], pl.FP32] = pl.yield_(dense_acc_inline44_inline8432__ssa_v3)
            qr_fp32_inline1_inline3598__ssa_v17: pl.Tensor[[QKV_QPROJ_MM_T_DYN, 1024], pl.FP32] = pl.tensor.assemble(qr_fp32_inline1_inline3598__iter_v15, dense_acc_inline44_inline8432__rv_v2, [dense_t0_inline47_inline8449__idx_v0, q_a_col0_inline66_inline8410__ssa_v0])
            qr_fp32_inline1_inline3598__rv_v16: pl.Tensor[[qr_t_matmul_inline2_inline3597__ssa_v0, 1024], pl.FP32] = pl.yield_(qr_fp32_inline1_inline3598__ssa_v17)
        for (t0_inline49_inline8420__idx_v0, (qr_fp32_inline1_inline3598__iter_v18,)) in pl.range(qr_full_rows_inline53_inline8466__ssa_v0 + qr_m_group_inline67_inline8440__ssa_v0 * 16, qr_t_matmul_inline54_inline8450__ssa_v0, qr_m_groups_inline58_inline8419__ssa_v0 * 16, init_values=(qr_fp32_inline1_inline3598__rv_v16,)):
            q_acc_inline42_inline8413__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for (db_inline52_inline8421__idx_v0, (q_acc_inline42_inline8413__iter_v1,)) in pl.pipeline(16, stage=2, init_values=(q_acc_inline42_inline8413__ssa_v0,)):
                qr_d0_inline56_inline8468__ssa_v0: pl.Scalar[pl.INDEX] = qr_k_base_inline65_inline8426__ssa_v0 + db_inline52_inline8421__idx_v0 * 256
                qr_rows_inline51_inline8452__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline4_inline3595__ssa_v0 - t0_inline49_inline8420__idx_v0, 16)
                x_t0_inline41_inline8429__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline3_inline3594__idx_v0 + t0_inline49_inline8420__idx_v0
                q_x_chunk_bf16_inline59_inline8470__ssa_v0: pl.Tensor[[16, 256], pl.BF16, pl.TensorView(valid_shape=[qr_rows_inline51_inline8452__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_view_inline50_inline8453__ssa_v0, [16, 256], [x_t0_inline41_inline8429__ssa_v0, qr_d0_inline56_inline8468__ssa_v0], [qr_rows_inline51_inline8452__ssa_v0, 256])
                w_chunk_inline55_inline8422__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wq_a__ssa_v0, [128, 256], [q_a_col0_inline66_inline8410__ssa_v0, qr_d0_inline56_inline8468__ssa_v0])
                q_acc_inline42_inline8413__ssa_v3: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul_acc(q_acc_inline42_inline8413__iter_v1, q_x_chunk_bf16_inline59_inline8470__ssa_v0, w_chunk_inline55_inline8422__ssa_v0, init_cond=db_inline52_inline8421__idx_v0 == 0, a_trans=False, b_trans=True)
                q_acc_inline42_inline8413__rv_v2: pl.Tensor[[16, 128], pl.FP32] = pl.yield_(q_acc_inline42_inline8413__ssa_v3)
            qr_fp32_inline1_inline3598__ssa_v20: pl.Tensor[[QKV_QPROJ_MM_T_DYN, 1024], pl.FP32] = pl.tensor.assemble(qr_fp32_inline1_inline3598__iter_v18, q_acc_inline42_inline8413__rv_v2, [t0_inline49_inline8420__idx_v0, q_a_col0_inline66_inline8410__ssa_v0])
            qr_fp32_inline1_inline3598__rv_v19: pl.Tensor[[qr_t_matmul_inline2_inline3597__ssa_v0, 1024], pl.FP32] = pl.yield_(qr_fp32_inline1_inline3598__ssa_v20)
        return qr_fp32_inline1_inline3598__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def qr_proj_matmul_0(qr_fp32_inline1_inline3598__ssa_v0: pl.Tensor[[qr_t_matmul_inline2_inline3597__ssa_v0, 1024], pl.FP32], qr_full_rows_inline82_inline8408__ssa_v0: pl.Scalar[pl.INDEX], qr_m_groups_inline80_inline8407__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline3_inline3594__idx_v0: pl.Scalar[pl.INDEX], x_view_inline78_inline8472__ssa_v0: pl.Tensor[[qa_tokens_inline79_inline8471__ssa_v0, 4096], pl.BF16], wq_a__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], qr_t_matmul_inline84_inline8427__ssa_v0: pl.Scalar[pl.INDEX], tile_rows_inline4_inline3595__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[qr_t_matmul_inline2_inline3597__ssa_v0, 1024], pl.FP32]:
        pl.func_attr({'cache_policy': [(5, 1)]})
        qbg_idx_inline91_inline8404__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qr_k_base_inline87_inline8403__ssa_v0: pl.Scalar[pl.INDEX] = 0
        q_a_col0_inline94_inline8456__ssa_v0: pl.Scalar[pl.INDEX] = qbg_idx_inline91_inline8404__ssa_v0 % 8 * 128
        qr_m_group_inline92_inline8402__ssa_v0: pl.Scalar[pl.INDEX] = qbg_idx_inline91_inline8404__ssa_v0 // 8
        for (dense_t0_inline77_inline8401__idx_v0, (qr_fp32_inline1_inline3598__iter_v25,)) in pl.range(qr_m_group_inline92_inline8402__ssa_v0 * 64, qr_full_rows_inline82_inline8408__ssa_v0, qr_m_groups_inline80_inline8407__ssa_v0 * 64, init_values=(qr_fp32_inline1_inline3598__ssa_v0,)):
            dense_x0_inline76_inline8400__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline3_inline3594__idx_v0 + dense_t0_inline77_inline8401__idx_v0
            dense_acc_inline88_inline8438__ssa_v0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.create([64, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for (dense_k_inline75_inline8399__idx_v0, (dense_acc_inline88_inline8438__iter_v1,)) in pl.pipeline(16, stage=2, init_values=(dense_acc_inline88_inline8438__ssa_v0,)):
                dense_d0_inline74_inline8398__ssa_v0: pl.Scalar[pl.INDEX] = qr_k_base_inline87_inline8403__ssa_v0 + dense_k_inline75_inline8399__idx_v0 * 256
                dense_x_inline93_inline8397__ssa_v0: pl.Tensor[[64, 256], pl.BF16] = pl.tensor.slice(x_view_inline78_inline8472__ssa_v0, [64, 256], [dense_x0_inline76_inline8400__ssa_v0, dense_d0_inline74_inline8398__ssa_v0])
                dense_w_inline73_inline8396__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wq_a__ssa_v0, [128, 256], [q_a_col0_inline94_inline8456__ssa_v0, dense_d0_inline74_inline8398__ssa_v0])
                dense_acc_inline88_inline8438__ssa_v3: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul_acc(dense_acc_inline88_inline8438__iter_v1, dense_x_inline93_inline8397__ssa_v0, dense_w_inline73_inline8396__ssa_v0, init_cond=dense_k_inline75_inline8399__idx_v0 == 0, a_trans=False, b_trans=True)
                dense_acc_inline88_inline8438__rv_v2: pl.Tensor[[64, 128], pl.FP32] = pl.yield_(dense_acc_inline88_inline8438__ssa_v3)
            qr_fp32_inline1_inline3598__ssa_v27: pl.Tensor[[QKV_QPROJ_MM_T_DYN, 1024], pl.FP32] = pl.tensor.assemble(qr_fp32_inline1_inline3598__iter_v25, dense_acc_inline88_inline8438__rv_v2, [dense_t0_inline77_inline8401__idx_v0, q_a_col0_inline94_inline8456__ssa_v0])
            qr_fp32_inline1_inline3598__rv_v26: pl.Tensor[[qr_t_matmul_inline2_inline3597__ssa_v0, 1024], pl.FP32] = pl.yield_(qr_fp32_inline1_inline3598__ssa_v27)
        for (t0_inline72_inline8395__idx_v0, (qr_fp32_inline1_inline3598__iter_v28,)) in pl.range(qr_full_rows_inline82_inline8408__ssa_v0 + qr_m_group_inline92_inline8402__ssa_v0 * 16, qr_t_matmul_inline84_inline8427__ssa_v0, qr_m_groups_inline80_inline8407__ssa_v0 * 16, init_values=(qr_fp32_inline1_inline3598__rv_v26,)):
            q_acc_inline71_inline8394__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for (db_inline69_inline8393__idx_v0, (q_acc_inline71_inline8394__iter_v1,)) in pl.pipeline(16, stage=2, init_values=(q_acc_inline71_inline8394__ssa_v0,)):
                qr_d0_inline68_inline8392__ssa_v0: pl.Scalar[pl.INDEX] = qr_k_base_inline87_inline8403__ssa_v0 + db_inline69_inline8393__idx_v0 * 256
                qr_rows_inline70_inline8391__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline4_inline3595__ssa_v0 - t0_inline72_inline8395__idx_v0, 16)
                x_t0_inline83_inline8436__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline3_inline3594__idx_v0 + t0_inline72_inline8395__idx_v0
                q_x_chunk_bf16_inline81_inline8428__ssa_v0: pl.Tensor[[16, 256], pl.BF16, pl.TensorView(valid_shape=[qr_rows_inline70_inline8391__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_view_inline78_inline8472__ssa_v0, [16, 256], [x_t0_inline83_inline8436__ssa_v0, qr_d0_inline68_inline8392__ssa_v0], [qr_rows_inline70_inline8391__ssa_v0, 256])
                w_chunk_inline90_inline8434__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wq_a__ssa_v0, [128, 256], [q_a_col0_inline94_inline8456__ssa_v0, qr_d0_inline68_inline8392__ssa_v0])
                q_acc_inline71_inline8394__ssa_v3: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul_acc(q_acc_inline71_inline8394__iter_v1, q_x_chunk_bf16_inline81_inline8428__ssa_v0, w_chunk_inline90_inline8434__ssa_v0, init_cond=db_inline69_inline8393__idx_v0 == 0, a_trans=False, b_trans=True)
                q_acc_inline71_inline8394__rv_v2: pl.Tensor[[16, 128], pl.FP32] = pl.yield_(q_acc_inline71_inline8394__ssa_v3)
            qr_fp32_inline1_inline3598__ssa_v30: pl.Tensor[[QKV_QPROJ_MM_T_DYN, 1024], pl.FP32] = pl.tensor.assemble(qr_fp32_inline1_inline3598__iter_v28, q_acc_inline71_inline8394__rv_v2, [t0_inline72_inline8395__idx_v0, q_a_col0_inline94_inline8456__ssa_v0])
            qr_fp32_inline1_inline3598__rv_v29: pl.Tensor[[qr_t_matmul_inline2_inline3597__ssa_v0, 1024], pl.FP32] = pl.yield_(qr_fp32_inline1_inline3598__ssa_v30)
        return qr_fp32_inline1_inline3598__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def qr_rms_norm_quant(gamma_cq_view_inline8485__ssa_v0: pl.Tensor[[1, 1024], pl.BF16], qr_token_tiles_inline8500__ssa_v0: pl.Scalar[pl.INDEX], qr_norm_workers_inline8483__ssa_v0: pl.Scalar[pl.INDEX], tile_rows_inline4_inline3595__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline3_inline3594__idx_v0: pl.Scalar[pl.INDEX], qr_fp32_inline1_inline3598__phi_v31: pl.Tensor[[qr_t_matmul_inline2_inline3597__ssa_v0, 1024], pl.FP32], qr_scale_pad_store_inline3589__ssa_v0: pl.Out[pl.Tensor[[512, 1], pl.FP32]], qr_scale_view_inline8490__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline8493__ssa_v0, 1], pl.FP32]], qr_i8_matmul_inline3588__ssa_v0: pl.Out[pl.Tensor[[512, 1024], pl.INT8]], qr_view_inline8492__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline8493__ssa_v0, 1024], pl.INT8]]) -> tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[t_dim_inline8493__ssa_v0, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[t_dim_inline8493__ssa_v0, 1024], pl.INT8]]:
        return (qr_scale_pad_store_inline3589__ssa_v0, qr_scale_view_inline8490__ssa_v0, qr_i8_matmul_inline3588__ssa_v0, qr_view_inline8492__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def quant(act_scale_dq_inline4531__iter_v1: pl.Tensor[[8, 384], pl.FP32], o_r_i8_pad_inline4534__iter_v1: pl.Tensor[[384, 8192], pl.INT8], t_dim_inline4518__ssa_v0: pl.Scalar[pl.INDEX], o_r_pad_inline4514__ssa_v3: pl.Tensor[[384, 8192], pl.FP32], col_g_inline4488__ssa_v0: pl.Scalar[pl.INDEX], g_inline4502__idx_v0: pl.Scalar[pl.INDEX], proj_b_padded_rows_inline4526__ssa_v0: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8]]:
        return (act_scale_dq_inline4531__iter_v1, o_r_i8_pad_inline4534__iter_v1)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def quant_0(act_scale_dq_inline4655__iter_v1: pl.Tensor[[8, 384], pl.FP32], o_r_i8_pad_inline4645__iter_v1: pl.Tensor[[384, 8192], pl.INT8], t_dim_inline4627__ssa_v0: pl.Scalar[pl.INDEX], o_r_pad_inline4642__ssa_v3: pl.Tensor[[384, 8192], pl.FP32], col_g_inline4637__ssa_v0: pl.Scalar[pl.INDEX], g_inline4648__idx_v0: pl.Scalar[pl.INDEX], proj_b_padded_rows_inline4630__ssa_v0: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8]]:
        return (act_scale_dq_inline4655__iter_v1, o_r_i8_pad_inline4645__iter_v1)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def quant_1(act_scale_dq_inline4765__iter_v1: pl.Tensor[[8, 384], pl.FP32], o_r_i8_pad_inline4738__iter_v1: pl.Tensor[[384, 8192], pl.INT8], t_dim_inline4745__ssa_v0: pl.Scalar[pl.INDEX], o_r_pad_inline4770__ssa_v3: pl.Tensor[[384, 8192], pl.FP32], col_g_inline4791__ssa_v0: pl.Scalar[pl.INDEX], g_inline4747__idx_v0: pl.Scalar[pl.INDEX], proj_b_padded_rows_inline4780__ssa_v0: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8]]:
        return (act_scale_dq_inline4765__iter_v1, o_r_i8_pad_inline4738__iter_v1)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def quant_2(act_scale_dq_inline4892__iter_v1: pl.Tensor[[8, 384], pl.FP32], o_r_i8_pad_inline4874__iter_v1: pl.Tensor[[384, 8192], pl.INT8], t_dim_inline4891__ssa_v0: pl.Scalar[pl.INDEX], o_r_pad_inline4871__ssa_v3: pl.Tensor[[384, 8192], pl.FP32], col_g_inline4865__ssa_v0: pl.Scalar[pl.INDEX], g_inline4868__idx_v0: pl.Scalar[pl.INDEX], proj_b_padded_rows_inline4872__ssa_v0: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8]]:
        return (act_scale_dq_inline4892__iter_v1, o_r_i8_pad_inline4874__iter_v1)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def rmsnorm_rope(normed_kv_inline574__ssa_v1: pl.Tensor[[384, 128], pl.BF16], rms_blocks_inline1646_inline3901__ssa_v0: pl.Scalar[pl.INDEX], rms_workers_inline1639_inline3967__ssa_v0: pl.Scalar[pl.INDEX], bs_inline1644_inline3932__ssa_v0: pl.Scalar[pl.INDEX], position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], idx_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], inner_freqs_cos__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 64], pl.FP32], inner_freqs_sin__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 64], pl.FP32], pooled_kv_inline3908__rv_v2: pl.Tensor[[384, 128], pl.FP32], norm_w_2d_inline1659_inline3897__ssa_v0: pl.Tensor[[1, 128], pl.BF16]) -> pl.Tensor[[384, 128], pl.BF16]:
        return normed_kv_inline574__ssa_v1

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def rmsnorm_rope_cache_write(bs_inline3815__ssa_v0: pl.Scalar[pl.INDEX], position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], cmp_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], cmp_freqs_cos__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 64], pl.FP32], cmp_freqs_sin__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 64], pl.FP32], pooled_kv_inline567__rv_v2: pl.Tensor[[384, 512], pl.FP32], normed_kv_inline3852__ssa_v0: pl.Tensor[[384, 512], pl.FP32], norm_w_2d_inline3844__ssa_v0: pl.Tensor[[1, 512], pl.BF16], cmp_kv_cache_flat_inline3834__ssa_v0: pl.Tensor[[cmp_block_num_inline3819__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16], kv_flat_inline3830__ssa_v0: pl.Tensor[[T_DYN, 512], pl.FP32], cmp_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], cmp_slot_mapping__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 2], pl.INT32]):
        return

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def rope_cs(rope_cs_blocks_inline4315__ssa_v0: pl.Scalar[pl.INDEX], t_dim_inline4335__ssa_v0: pl.Scalar[pl.INDEX], freqs_sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32], rope_sin_signed_inline4321__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32]]) -> pl.Tensor[[384, 64], pl.FP32]:
        return rope_sin_signed_inline4321__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def scatter_softmax_pool(pooled_kv_inline567__ssa_v0: pl.Tensor[[384, 512], pl.FP32], b_dim_inline308_inline3788__ssa_v0: pl.Scalar[pl.INDEX], pool_workers_inline297_inline3797__ssa_v0: pl.Scalar[pl.INDEX], position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], s_dim_inline307_inline3792__ssa_v0: pl.Scalar[pl.INDEX], cmp_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], cmp4_score_proj_pad_inline299_inline3775__ssa_v0: pl.Tensor[[384, 1024], pl.FP32], cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32], cmp4_kv_proj_pad_inline300_inline3767__ssa_v0: pl.Tensor[[384, 1024], pl.FP32], state_block_table__ssa_v0: pl.Tensor[[B_DYN, STATE_TABLE_COLUMNS_DYN], pl.INT32], compress_state__ssa_v0: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32]) -> tuple[pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[384, 512], pl.FP32]]:
        return (pooled_kv_inline567__ssa_v0, pooled_kv_inline567__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def scatter_softmax_pool_0(pooled_kv_inline3908__ssa_v0: pl.Tensor[[384, 128], pl.FP32], b_dim_inline1638_inline3904__ssa_v0: pl.Scalar[pl.INDEX], pool_workers_inline1636_inline3917__ssa_v0: pl.Scalar[pl.INDEX], position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], s_dim_inline1634_inline3899__ssa_v0: pl.Scalar[pl.INDEX], score_proj_pad_inline3938__rv_v2: pl.Tensor[[384, 256], pl.FP32], inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32], kv_proj_pad_inline3911__rv_v2: pl.Tensor[[384, 256], pl.FP32], inner_state_block_table__ssa_v0: pl.Tensor[[B_DYN, INNER_STATE_TABLE_COLUMNS_DYN], pl.INT32], inner_compress_state__ssa_v0: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32]) -> pl.Tensor[[384, 128], pl.FP32]:
        return pooled_kv_inline3908__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def split_pre_post(hc_attn_scale__ssa_v0: pl.Tensor[[3], pl.FP32], post_t__ssa_v0: pl.Tensor[[T_DYN, 4], pl.FP32], post_tail_store_inline491__ssa_v0: pl.Tensor[[8, 8], pl.FP32], pre_val_store__ssa_v0: pl.Tensor[[hc_padded_rows__ssa_v0, 8], pl.FP32], token_tiles_inline443__ssa_v0: pl.Scalar[pl.INDEX], t_dim_inline440__ssa_v0: pl.Scalar[pl.INDEX], inv_rms__rv_v2: pl.Tensor[[hc_padded_rows__ssa_v0, 1], pl.FP32], hc_attn_base__ssa_v0: pl.Tensor[[24], pl.FP32], mixes_raw_inline448__ssa_v1: pl.Tensor[[t_linear_inline461__ssa_v0, 32], pl.FP32]) -> tuple[pl.Tensor[[T_DYN, 4], pl.FP32], pl.Tensor[[hc_padded_rows__ssa_v0, 8], pl.FP32]]:
        return (post_t__ssa_v0, pre_val_store__ssa_v0)

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def weights_proj(weights_partial_inline3185_inline4185__ssa_v0: pl.Tensor[[1536, 64], pl.FP32], row_blocks_inline3179_inline4165__ssa_v0: pl.Scalar[pl.INDEX], bs_inline3187_inline4159__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline3180_inline4167__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16], weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16]) -> pl.Tensor[[1536, 64], pl.FP32]:
        return weights_partial_inline3185_inline4185__ssa_v0

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def weights_proj_reduce(weights_partial_inline3185_inline4185__rv_v2: pl.Tensor[[1536, 64], pl.FP32], weights_inline3183_inline4184__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32]]) -> pl.Tensor[[384, 64], pl.FP32]:
        return weights_inline3183_inline4184__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def _decode_csa_tp1_layer(self, x_hc__ssa_v0: pl.Tensor[[T_DYN, 4, 4096], pl.BF16], hc_attn_fn__ssa_v0: pl.Tensor[[24, 16384], pl.FP32], hc_attn_scale__ssa_v0: pl.Tensor[[3], pl.FP32], hc_attn_base__ssa_v0: pl.Tensor[[24], pl.FP32], attn_norm_w__ssa_v0: pl.Tensor[[4096], pl.BF16], wq_a__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32], wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16], gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16], gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16], freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32], freqs_sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32], cmp_freqs_cos__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 64], pl.FP32], cmp_freqs_sin__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 64], pl.FP32], inner_freqs_cos__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 64], pl.FP32], inner_freqs_sin__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 64], pl.FP32], cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16], cmp_wgate__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16], cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32], cmp_norm_w__ssa_v0: pl.Tensor[[512], pl.BF16], compress_state__ssa_v0: pl.InOut[pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32]], state_block_table__ssa_v0: pl.Tensor[[B_DYN, STATE_TABLE_COLUMNS_DYN], pl.INT32], idx_wq_b__ssa_v0: pl.Tensor[[1024, 8192], pl.INT8], idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32], weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16], hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16], inner_wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16], inner_wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16], inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32], inner_norm_w__ssa_v0: pl.Tensor[[128], pl.BF16], inner_compress_state__ssa_v0: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32]], inner_state_block_table__ssa_v0: pl.Tensor[[B_DYN, INNER_STATE_TABLE_COLUMNS_DYN], pl.INT32], kv_cache__ssa_v0: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16]], cmp_kv__ssa_v0: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16]], cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32], idx_native_kv_cache__ssa_v0: pl.InOut[pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]], idx_block_table__ssa_v0: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32], ori_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32], ori_block_table__ssa_v0: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32], cmp_slot_mapping__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 2], pl.INT32], idx_slot_mapping__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 2], pl.INT32], state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32], inner_state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32], position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], cmp_query_start_loc__ssa_v0: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32], cmp_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], idx_query_start_loc__ssa_v0: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32], attn_sink__ssa_v0: pl.Tensor[[64], pl.FP32], wo_a__ssa_v0: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], wo_b__ssa_v0: pl.Tensor[[8192, 4096], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32], idx_topk_scores__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32]], idx_topk__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.INT32]], x_out__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 4, 4096], pl.BF16]]) -> pl.Tensor[[T_DYN, 512], pl.FP32]:
        t__tmp_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv_seq_lens__ssa_v0, 0)
        for (batch_inline377__idx_v0, (maximum_inline378__iter_v1,)) in pl.range(t__tmp_v0, init_values=(0,)):
            t__tmp_v1: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_inline377__idx_v0])
            maximum_inline378__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(maximum_inline378__iter_v1, pl.cast(t__tmp_v1, pl.INDEX))
            maximum_inline378__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(maximum_inline378__ssa_v3)
        max_seq_len__ssa_v0: pl.Scalar[pl.INDEX] = maximum_inline378__rv_v2
        post_t__ssa_v0: pl.Tensor[[T_DYN, 4], pl.FP32] = pl.tensor.create([T_DYN, 4], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        comb_t__ssa_v0: pl.Tensor[[T_DYN, 16], pl.FP32] = pl.tensor.create([T_DYN, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        x_normed_t__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16] = pl.tensor.create([T_DYN, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        x_hc32__ssa_v0: pl.Tensor[[T_DYN, 4, 4096], pl.FP32] = pl.tensor.create([T_DYN, 4, 4096], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        x_hc_flat__ssa_v0: pl.Tensor[[T_DYN, 16384], pl.BF16] = pl.tensor.reshape(x_hc__ssa_v0, [T_DYN, 16384])
        x_hc32_flat__ssa_v0: pl.Tensor[[T_DYN, 16384], pl.FP32] = pl.tensor.reshape(x_hc32__ssa_v0, [T_DYN, 16384])
        widen_rows__ssa_v0: pl.Scalar[pl.INDEX] = (T_DYN + 7) // 8
        hc_padded_rows__ssa_v0: pl.Scalar[pl.INDEX] = (T_DYN + 15) // 16 * 16
        inv_rms__ssa_v0: pl.Tensor[[hc_padded_rows__ssa_v0, 1], pl.FP32] = pl.tensor.create([hc_padded_rows__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        widen_tail__ssa_v0: pl.Tensor[[8, 16384], pl.FP32] = pl.tensor.create([8, 16384], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.spmd(pl.min(widen_rows__ssa_v0, 48), name_hint='hc_widen_rms_spmd', allow_early_resolve=True) as _widen_tid__ssa_v0:
            inv_rms__rv_v2: pl.Tensor[[hc_padded_rows__ssa_v0, 1], pl.FP32] = self.hc_widen_rms(inv_rms__ssa_v0, widen_tail__ssa_v0, x_hc32_flat__ssa_v0, widen_rows__ssa_v0, T_DYN, x_hc_flat__ssa_v0)
        with pl.scope():
            pre_val_store__ssa_v0: pl.Tensor[[hc_padded_rows__ssa_v0, 8], pl.FP32] = pl.tensor.create([hc_padded_rows__ssa_v0, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            t_dim_inline440__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_hc32__ssa_v0, 0)
            token_tiles_inline443__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline440__ssa_v0 + 7) // 8
            t_linear_inline461__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline440__ssa_v0 + 15) // 16 * 16
            x_flat_inline486__ssa_v0: pl.Tensor[[t_dim_inline440__ssa_v0, 16384], pl.FP32] = pl.tensor.reshape(x_hc32__ssa_v0, [t_dim_inline440__ssa_v0, 16384])
            hc_base_2d_inline462__ssa_v0: pl.Tensor[[1, 24], pl.FP32] = pl.tensor.reshape(hc_attn_base__ssa_v0, [1, 24])
            mixes_partials_inline487__ssa_v0: pl.Tensor[[t_linear_inline461__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32] = pl.tensor.create([t_linear_inline461__ssa_v0 * 4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            linear_units_inline444__ssa_v0: pl.Scalar[pl.INDEX] = t_linear_inline461__ssa_v0 // 16 * 4
            linear_workers_inline432__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(linear_units_inline444__ssa_v0, 24)
            with pl.spmd(linear_workers_inline432__ssa_v0, name_hint='hc_pre_linear_spmd', allow_early_resolve=True):
                mixes_partials_inline487__rv_v2: pl.Tensor[[t_linear_inline461__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32] = self.hc_pre_linear(mixes_partials_inline487__ssa_v0, linear_units_inline444__ssa_v0, linear_workers_inline432__ssa_v0, t_dim_inline440__ssa_v0, x_flat_inline486__ssa_v0, hc_attn_fn__ssa_v0, t_linear_inline461__ssa_v0)
            mixes_raw_inline448__ssa_v0: pl.Tensor[[t_linear_inline461__ssa_v0, 32], pl.FP32] = pl.tensor.create([t_linear_inline461__ssa_v0, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(t_linear_inline461__ssa_v0 // 16, name_hint='hc_pre_linear_reduce_spmd', allow_early_resolve=True):
                mixes_raw_inline448__ssa_v1: pl.Tensor[[t_linear_inline461__ssa_v0, 32], pl.FP32] = self.hc_pre_linear_reduce(mixes_partials_inline487__rv_v2, t_linear_inline461__ssa_v0, mixes_raw_inline448__ssa_v0)
            post_tail_store_inline491__ssa_v0: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(pl.min(token_tiles_inline443__ssa_v0, 16), name_hint='split_pre_post_spmd', allow_early_resolve=True):
                ret__tmp_v0: pl.Tuple[pl.Tensor[[T_DYN, 4], pl.FP32], pl.Tensor[[hc_padded_rows__ssa_v0, 8], pl.FP32]] = self.split_pre_post(hc_attn_scale__ssa_v0, post_t__ssa_v0, post_tail_store_inline491__ssa_v0, pre_val_store__ssa_v0, token_tiles_inline443__ssa_v0, t_dim_inline440__ssa_v0, inv_rms__rv_v2, hc_attn_base__ssa_v0, mixes_raw_inline448__ssa_v1)
                post_t__rv_v2: pl.Tensor[[T_DYN, 4], pl.FP32] = ret__tmp_v0[0]
                pre_val_store__rv_v2: pl.Tensor[[hc_padded_rows__ssa_v0, 8], pl.FP32] = ret__tmp_v0[1]
            comb_tail_store_inline449__ssa_v0: pl.Tensor[[8, 32], pl.FP32] = pl.tensor.create([8, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(token_tiles_inline443__ssa_v0, name_hint='comb_sinkhorn_spmd', allow_early_resolve=True):
                ret__tmp_v0_1: pl.Tuple[pl.Tensor[[T_DYN, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32]] = self.comb_sinkhorn(hc_attn_scale__ssa_v0, t_dim_inline440__ssa_v0, inv_rms__rv_v2, mixes_raw_inline448__ssa_v1, hc_base_2d_inline462__ssa_v0, comb_t__ssa_v0, comb_tail_store_inline449__ssa_v0)
                comb_t__ssa_v1: pl.Tensor[[T_DYN, 16], pl.FP32] = ret__tmp_v0_1[0]
                comb_tail_store_inline449__ssa_v1: pl.Tensor[[8, 32], pl.FP32] = ret__tmp_v0_1[1]
            t_dim_inline529__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_hc32__ssa_v0, 0)
            x_flat_inline525__ssa_v0: pl.Tensor[[t_dim_inline529__ssa_v0, 16384], pl.FP32] = pl.tensor.reshape(x_hc32__ssa_v0, [t_dim_inline529__ssa_v0, 16384])
            t_pad_inline523__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline529__ssa_v0 + 7) // 8 * 8
            x_mixed_inline534__ssa_v0: pl.Tensor[[t_pad_inline523__ssa_v0, 4096], pl.BF16] = pl.tensor.create([t_pad_inline523__ssa_v0, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            x_normed_tail_store_inline516__ssa_v0: pl.Tensor[[8, 4096], pl.BF16] = pl.tensor.create([8, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            with pl.spmd(t_pad_inline523__ssa_v0 // 8, name_hint='mix_x_rms_norm_spmd', allow_early_resolve=True) as mixed_tid_inline520__ssa_v0:
                x_normed_t__rv_v2: pl.Tensor[[T_DYN, 4096], pl.BF16] = self.mix_x_rms_norm(t_dim_inline529__ssa_v0, pre_val_store__rv_v2, x_mixed_inline534__ssa_v0, x_flat_inline525__ssa_v0, x_normed_t__ssa_v0, x_normed_tail_store_inline516__ssa_v0, attn_norm_w__ssa_v0)
        wb_blocks__ssa_v0: pl.Scalar[pl.INDEX] = (T_DYN + 7) // 8
        idx_sin_signed__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32] = pl.tensor.create([T_DYN, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        cmp_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32] = pl.tensor.create([B_DYN], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        idx_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32] = pl.tensor.create([B_DYN], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        ret__tmp_v0_2: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.submit(self.csa_row_offsets, cmp_seq_lens__ssa_v0, cmp_query_start_loc__ssa_v0, cmp_row_offsets__ssa_v0, kv_seq_lens__ssa_v0, idx_query_start_loc__ssa_v0, idx_row_offsets__ssa_v0, allow_early_resolve=True)
        offsets_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_2[0]
        rope_sign_blocks__ssa_v0: pl.Scalar[pl.INDEX] = (T_DYN + 3) // 4
        with pl.spmd(pl.min(rope_sign_blocks__ssa_v0, 16), name_hint='csa_rope_sign_spmd', deps=[offsets_tid__ssa_v0], allow_early_resolve=True) as rope_tid__ssa_v0:
            idx_sin_signed__ssa_v1: pl.Tensor[[T_DYN, 64], pl.FP32] = self.csa_rope_sign(rope_sign_blocks__ssa_v0, T_DYN, freqs_sin__ssa_v0, idx_sin_signed__ssa_v0)
        q__ssa_v0: pl.Tensor[[T_DYN, 64, 512], pl.BF16] = pl.tensor.create([T_DYN, 64, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        kv__ssa_v0: pl.Tensor[[T_DYN, 512], pl.BF16] = pl.tensor.create([T_DYN, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        qr__ssa_v0: pl.Tensor[[T_DYN, 1024], pl.INT8] = pl.tensor.create([T_DYN, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND)
        qr_scale__ssa_v0: pl.Tensor[[T_DYN, 1], pl.FP32] = pl.tensor.create([T_DYN, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT64] = pl.tensor.reshape(position_ids__ssa_v0, [T_DYN, 1])
        with pl.scope():
            late_dep__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[rope_tid__ssa_v0])
            t_dim_inline560__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__rv_v2, 0)
            q_rope_sin_signed_inline558__ssa_v0: pl.Tensor[[t_dim_inline560__ssa_v0, 64], pl.FP32] = pl.tensor.create([t_dim_inline560__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            q_rope_swap_idx_inline557__ssa_v0: pl.Tensor[[t_dim_inline560__ssa_v0, 64], pl.INT32] = pl.tensor.create([t_dim_inline560__ssa_v0, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            t_dim_inline3567__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(freqs_sin__ssa_v0, 0)
            rope_sin_view_inline3559__ssa_v0: pl.Tensor[[t_dim_inline3567__ssa_v0, 64], pl.FP32] = pl.tensor.reshape(freqs_sin__ssa_v0, [t_dim_inline3567__ssa_v0, 64])
            rope_sin_signed_view_inline3575__ssa_v0: pl.Tensor[[t_dim_inline3567__ssa_v0, 64], pl.FP32] = pl.tensor.reshape(q_rope_sin_signed_inline558__ssa_v0, [t_dim_inline3567__ssa_v0, 64])
            rope_swap_idx_view_inline3564__ssa_v0: pl.Tensor[[t_dim_inline3567__ssa_v0, 64], pl.INT32] = pl.tensor.reshape(q_rope_swap_idx_inline557__ssa_v0, [t_dim_inline3567__ssa_v0, 64])
            token_tiles_inline3560__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline3567__ssa_v0 + 7) // 8
            with pl.spmd(pl.min(token_tiles_inline3560__ssa_v0, 48), name_hint='q_rope_prepare_spmd', allow_early_resolve=True):
                self.q_rope_prepare(rope_sin_signed_view_inline3575__ssa_v0, rope_swap_idx_view_inline3564__ssa_v0, token_tiles_inline3560__ssa_v0, t_dim_inline3567__ssa_v0, rope_sin_view_inline3559__ssa_v0)
            qr_i8_matmul_inline3588__ssa_v0: pl.Tensor[[512, 1024], pl.INT8] = pl.tensor.create([512, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND)
            qr_scale_pad_store_inline3589__ssa_v0: pl.Tensor[[512, 1], pl.FP32] = pl.tensor.create([512, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
            qa_tids_inline6_inline3592__ssa_v0: pl.Array[1, pl.TASK_ID] = pl.array.create(1, dtype=pl.TASK_ID)
            t__tmp_v78: pl.Scalar[pl.TASK_ID] = pl.system.task_invalid()
            qa_tids_inline6_inline3592__ssa_v1: pl.Array[1, pl.TASK_ID] = pl.array.update_element(qa_tids_inline6_inline3592__ssa_v0, 0, t__tmp_v78)
            t_dim_inline5_inline3593__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__rv_v2, 0)
            for (tile_base_inline3_inline3594__idx_v0, (qa_tids_inline6_inline3592__iter_v2, qr_scale_pad_store_inline3589__iter_v1, qr_i8_matmul_inline3588__iter_v1)) in pl.range(0, t_dim_inline5_inline3593__ssa_v0, 512, init_values=(qa_tids_inline6_inline3592__ssa_v1, qr_scale_pad_store_inline3589__ssa_v0, qr_i8_matmul_inline3588__ssa_v0)):
                tile_rows_inline4_inline3595__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline5_inline3593__ssa_v0 - tile_base_inline3_inline3594__idx_v0, 512)
                with pl.scope():
                    qr_t_matmul_inline2_inline3597__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline4_inline3595__ssa_v0 + 15) // 16 * 16
                    qr_fp32_inline1_inline3598__ssa_v0: pl.Tensor[[qr_t_matmul_inline2_inline3597__ssa_v0, 1024], pl.FP32] = pl.tensor.create([qr_t_matmul_inline2_inline3597__ssa_v0, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    if tile_rows_inline4_inline3595__ssa_v0 < 128:
                        qa_tokens_inline48_inline8448__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__rv_v2, 0)
                        x_view_inline50_inline8453__ssa_v0: pl.Tensor[[qa_tokens_inline48_inline8448__ssa_v0, 4096], pl.BF16] = pl.tensor.reshape(x_normed_t__rv_v2, [qa_tokens_inline48_inline8448__ssa_v0, 4096])
                        qr_t_matmul_inline54_inline8450__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline4_inline3595__ssa_v0 + 15) // 16 * 16
                        qr_full_rows_inline53_inline8466__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline4_inline3595__ssa_v0 // 32 * 32
                        qr_m_groups_inline58_inline8419__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(tile_rows_inline4_inline3595__ssa_v0 // 32, 1)
                        with pl.spmd(qr_m_groups_inline58_inline8419__ssa_v0 * 8, name_hint='qr_proj_matmul_spmd', allow_early_resolve=True) as qa_tid_inline61_inline8460__ssa_v0:
                            qr_fp32_inline1_inline3598__rv_v19: pl.Tensor[[qr_t_matmul_inline2_inline3597__ssa_v0, 1024], pl.FP32] = self.qr_proj_matmul(qr_fp32_inline1_inline3598__ssa_v0, qr_full_rows_inline53_inline8466__ssa_v0, qr_m_groups_inline58_inline8419__ssa_v0, tile_base_inline3_inline3594__idx_v0, x_view_inline50_inline8453__ssa_v0, wq_a__ssa_v0, qr_t_matmul_inline54_inline8450__ssa_v0, tile_rows_inline4_inline3595__ssa_v0)
                        qa_tid_inline8447__ssa_v1: pl.Scalar[pl.TASK_ID] = qa_tid_inline61_inline8460__ssa_v0
                        (qa_tid_inline8447__phi_v3, qr_fp32_inline1_inline3598__phi_v31) = pl.yield_(qa_tid_inline8447__ssa_v1, qr_fp32_inline1_inline3598__rv_v19)
                    else:
                        qa_tokens_inline79_inline8471__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__rv_v2, 0)
                        x_view_inline78_inline8472__ssa_v0: pl.Tensor[[qa_tokens_inline79_inline8471__ssa_v0, 4096], pl.BF16] = pl.tensor.reshape(x_normed_t__rv_v2, [qa_tokens_inline79_inline8471__ssa_v0, 4096])
                        qr_t_matmul_inline84_inline8427__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline4_inline3595__ssa_v0 + 15) // 16 * 16
                        qr_full_rows_inline82_inline8408__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline4_inline3595__ssa_v0 // 64 * 64
                        qr_m_groups_inline80_inline8407__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline4_inline3595__ssa_v0 // 64, 3)
                        with pl.spmd(qr_m_groups_inline80_inline8407__ssa_v0 * 8, name_hint='qr_proj_matmul_spmd', allow_early_resolve=True) as qa_tid_inline89_inline8411__ssa_v0:
                            qr_fp32_inline1_inline3598__rv_v29: pl.Tensor[[qr_t_matmul_inline2_inline3597__ssa_v0, 1024], pl.FP32] = self.qr_proj_matmul_0(qr_fp32_inline1_inline3598__ssa_v0, qr_full_rows_inline82_inline8408__ssa_v0, qr_m_groups_inline80_inline8407__ssa_v0, tile_base_inline3_inline3594__idx_v0, x_view_inline78_inline8472__ssa_v0, wq_a__ssa_v0, qr_t_matmul_inline84_inline8427__ssa_v0, tile_rows_inline4_inline3595__ssa_v0)
                        qa_tid_inline8447__ssa_v2: pl.Scalar[pl.TASK_ID] = qa_tid_inline89_inline8411__ssa_v0
                        (qa_tid_inline8447__phi_v3, qr_fp32_inline1_inline3598__phi_v31) = pl.yield_(qa_tid_inline8447__ssa_v2, qr_fp32_inline1_inline3598__rv_v29)
                    qa_task_inline0_inline3599__ssa_v0: pl.Scalar[pl.TASK_ID] = qa_tid_inline8447__phi_v3
                    qa_tids_inline6_inline3592__ssa_v4: pl.Array[1, pl.TASK_ID] = pl.array.update_element(qa_tids_inline6_inline3592__iter_v2, 0, qa_task_inline0_inline3599__ssa_v0)
                    t_dim_inline8493__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(qr__ssa_v0, 0)
                    qr_view_inline8492__ssa_v0: pl.Tensor[[t_dim_inline8493__ssa_v0, 1024], pl.INT8] = pl.tensor.reshape(qr__ssa_v0, [t_dim_inline8493__ssa_v0, 1024])
                    qr_scale_view_inline8490__ssa_v0: pl.Tensor[[t_dim_inline8493__ssa_v0, 1], pl.FP32] = pl.tensor.reshape(qr_scale__ssa_v0, [t_dim_inline8493__ssa_v0, 1])
                    gamma_cq_view_inline8485__ssa_v0: pl.Tensor[[1, 1024], pl.BF16] = pl.tensor.reshape(gamma_cq__ssa_v0, [1, 1024])
                    qr_token_tiles_inline8500__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline4_inline3595__ssa_v0 + 7) // 8
                    qr_norm_workers_inline8483__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(qr_token_tiles_inline8500__ssa_v0, 16)
                    with pl.spmd(qr_norm_workers_inline8483__ssa_v0, name_hint='qr_rms_norm_quant_spmd', allow_early_resolve=True):
                        ret__tmp_v0_3: pl.Tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[t_dim_inline8493__ssa_v0, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[t_dim_inline8493__ssa_v0, 1024], pl.INT8]] = self.qr_rms_norm_quant(gamma_cq_view_inline8485__ssa_v0, qr_token_tiles_inline8500__ssa_v0, qr_norm_workers_inline8483__ssa_v0, tile_rows_inline4_inline3595__ssa_v0, tile_base_inline3_inline3594__idx_v0, qr_fp32_inline1_inline3598__phi_v31, qr_scale_pad_store_inline3589__iter_v1, qr_scale_view_inline8490__ssa_v0, qr_i8_matmul_inline3588__iter_v1, qr_view_inline8492__ssa_v0)
                        qr_scale_pad_store_inline3589__ssa_v1: pl.Tensor[[512, 1], pl.FP32] = ret__tmp_v0_3[0]
                        qr_scale_view_inline8490__ssa_v1: pl.Tensor[[t_dim_inline8493__ssa_v0, 1], pl.FP32] = ret__tmp_v0_3[1]
                        qr_i8_matmul_inline3588__ssa_v1: pl.Tensor[[512, 1024], pl.INT8] = ret__tmp_v0_3[2]
                        qr_view_inline8492__ssa_v1: pl.Tensor[[t_dim_inline8493__ssa_v0, 1024], pl.INT8] = ret__tmp_v0_3[3]
                (qa_tids_inline6_inline3592__rv_v3, qr_scale_pad_store_inline3589__rv_v1, qr_i8_matmul_inline3588__rv_v1) = pl.yield_(qa_tids_inline6_inline3592__ssa_v4, qr_scale_pad_store_inline3589__ssa_v1, qr_i8_matmul_inline3588__ssa_v1)
            qa_tid_inline3590__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.array.get_element(qa_tids_inline6_inline3592__rv_v3, 0)
            q_seq_dep_inline3601__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
            t_dim_inline13_inline3591__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__rv_v2, 0)
            for tile_base_inline12_inline3596__idx_v0 in pl.range(0, t_dim_inline13_inline3591__ssa_v0, 512):
                tile_rows_inline11_inline3587__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline13_inline3591__ssa_v0 - tile_base_inline12_inline3596__idx_v0, 512)
                with pl.scope():
                    qproj_t_matmul_inline10_inline3586__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline11_inline3587__ssa_v0 + 127) // 128 * 128
                    q_proj_i32_inline7_inline3585__ssa_v0: pl.Tensor[[qproj_t_matmul_inline10_inline3586__ssa_v0, 32768], pl.INT32] = pl.tensor.create([qproj_t_matmul_inline10_inline3586__ssa_v0, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                    qproj_t_matmul_inline8513__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q_proj_i32_inline7_inline3585__ssa_v0, 0)
                    with pl.spmd(20, name_hint='qproj_matmul_spmd', deps=[q_seq_dep_inline3601__ssa_v0]) as qproj_tid_inline8517__ssa_v0:
                        q_proj_i32_inline7_inline3585__rv_v2: pl.Tensor[[qproj_t_matmul_inline10_inline3586__ssa_v0, 32768], pl.INT32] = self.qproj_matmul(q_proj_i32_inline7_inline3585__ssa_v0, qproj_t_matmul_inline8513__ssa_v0, tile_rows_inline11_inline3587__ssa_v0, qr_i8_matmul_inline3588__rv_v1, wq_b__ssa_v0)
                    q_proj_i32_inline7_inline3585__ssa_v6: pl.Tensor[[qproj_t_matmul_inline10_inline3586__ssa_v0, 32768], pl.INT32] = q_proj_i32_inline7_inline3585__rv_v2
                    t_dim_inline8556__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q__ssa_v0, 0)
                    q_flat_inline8548__ssa_v0: pl.Tensor[[t_dim_inline8556__ssa_v0, 32768], pl.BF16] = pl.tensor.reshape(q__ssa_v0, [t_dim_inline8556__ssa_v0, 32768])
                    with pl.spmd(48, name_hint='qproj_dequant_rms_nope_rope_spmd', allow_early_resolve=True):
                        q_flat_inline8548__ssa_v1: pl.Tensor[[t_dim_inline8556__ssa_v0, 32768], pl.BF16] = self.qproj_dequant_rms_nope_rope(tile_rows_inline11_inline3587__ssa_v0, tile_base_inline12_inline3596__idx_v0, qr_scale_pad_store_inline3589__rv_v1, freqs_cos__ssa_v0, q_rope_sin_signed_inline558__ssa_v0, q_rope_swap_idx_inline557__ssa_v0, q_proj_i32_inline7_inline3585__ssa_v6, wq_b_scale__ssa_v0, q_flat_inline8548__ssa_v0)
            qa_tid_inline559__ssa_v0: pl.Scalar[pl.TASK_ID] = qa_tid_inline3590__ssa_v0
            t_dim_inline3668__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__rv_v2, 0)
            for tile_base_inline3655__idx_v0 in pl.range(0, t_dim_inline3668__ssa_v0, 512):
                tile_rows_inline3661__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline3668__ssa_v0 - tile_base_inline3655__idx_v0, 512)
                with pl.scope():
                    t_matmul_inline3663__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline3661__ssa_v0 + 15) // 16 * 16
                    kv_fp32_inline3665__ssa_v0: pl.Tensor[[t_matmul_inline3663__ssa_v0, 512], pl.FP32] = pl.tensor.create([t_matmul_inline3663__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    if tile_rows_inline3661__ssa_v0 < 128:
                        t_dim_inline1450_inline3644__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__rv_v2, 0)
                        x_view_inline1451_inline3640__ssa_v0: pl.Tensor[[t_dim_inline1450_inline3644__ssa_v0, 4096], pl.BF16] = pl.tensor.reshape(x_normed_t__rv_v2, [t_dim_inline1450_inline3644__ssa_v0, 4096])
                        kv_full_rows_inline1452_inline3639__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline3661__ssa_v0 // 32 * 32
                        kv_m_groups_inline1466_inline3638__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(tile_rows_inline3661__ssa_v0 // 32, 1)
                        with pl.spmd(kv_m_groups_inline1466_inline3638__ssa_v0 * 4, name_hint='kv_proj_matmul_spmd', deps=[late_dep__ssa_v0], allow_early_resolve=True) as _kv_tid_inline1455_inline3694__ssa_v0:
                            kv_fp32_inline3665__rv_v15: pl.Tensor[[t_matmul_inline3663__ssa_v0, 512], pl.FP32] = self.kv_proj_matmul(kv_m_groups_inline1466_inline3638__ssa_v0, kv_fp32_inline3665__ssa_v0, kv_full_rows_inline1452_inline3639__ssa_v0, tile_base_inline3655__idx_v0, x_view_inline1451_inline3640__ssa_v0, wkv__ssa_v0, t_matmul_inline3663__ssa_v0, tile_rows_inline3661__ssa_v0)
                        kv_fp32_inline3665__phi_v23: pl.Tensor[[t_matmul_inline3663__ssa_v0, 512], pl.FP32] = pl.yield_(kv_fp32_inline3665__rv_v15)
                    else:
                        t_dim_inline1474_inline3632__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__rv_v2, 0)
                        x_view_inline1476_inline3614__ssa_v0: pl.Tensor[[t_dim_inline1474_inline3632__ssa_v0, 4096], pl.BF16] = pl.tensor.reshape(x_normed_t__rv_v2, [t_dim_inline1474_inline3632__ssa_v0, 4096])
                        kv_full_rows_inline1475_inline3651__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline3661__ssa_v0 // 64 * 64
                        kv_m_groups_inline1490_inline3660__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline3661__ssa_v0 // 64, 3)
                        with pl.spmd(kv_m_groups_inline1490_inline3660__ssa_v0 * 4, name_hint='kv_proj_matmul_spmd', deps=[late_dep__ssa_v0], allow_early_resolve=True) as _kv_tid_inline1477_inline3703__ssa_v0:
                            kv_fp32_inline3665__rv_v21: pl.Tensor[[t_matmul_inline3663__ssa_v0, 512], pl.FP32] = self.kv_proj_matmul_0(kv_m_groups_inline1490_inline3660__ssa_v0, kv_fp32_inline3665__ssa_v0, kv_full_rows_inline1475_inline3651__ssa_v0, tile_base_inline3655__idx_v0, x_view_inline1476_inline3614__ssa_v0, wkv__ssa_v0, t_matmul_inline3663__ssa_v0, tile_rows_inline3661__ssa_v0)
                        kv_fp32_inline3665__phi_v23: pl.Tensor[[t_matmul_inline3663__ssa_v0, 512], pl.FP32] = pl.yield_(kv_fp32_inline3665__rv_v21)
                    kv_view_inline3736__ssa_v0: pl.Tensor[[t_dim_inline3668__ssa_v0, 512], pl.BF16] = pl.tensor.reshape(kv__ssa_v0, [t_dim_inline3668__ssa_v0, 512])
                    kv_token_tiles_inline3726__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline3661__ssa_v0 + 31) // 32
                    with pl.spmd(kv_token_tiles_inline3726__ssa_v0, sync_start=True, name_hint='kv_rms_norm_rope_spmd', allow_early_resolve=True):
                        kv_view_inline3736__ssa_v1: pl.Tensor[[t_dim_inline3668__ssa_v0, 512], pl.BF16] = self.kv_rms_norm_rope(tile_rows_inline3661__ssa_v0, tile_base_inline3655__idx_v0, kv_fp32_inline3665__phi_v23, kv_view_inline3736__ssa_v0, gamma_ckv__ssa_v0, freqs_cos__ssa_v0, q_rope_sin_signed_inline558__ssa_v0, q_rope_swap_idx_inline557__ssa_v0)
            q__ssa_v1: pl.Tensor[[T_DYN, 64, 512], pl.BF16] = q__ssa_v0
            qa_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = qa_tid_inline559__ssa_v0
            kv_cache_flat__ssa_v0: pl.Tensor[[ORI_BLOCK_NUM_DYN * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(kv_cache__ssa_v0, [ORI_BLOCK_NUM_DYN * 32, 512])
            with pl.spmd(8, name_hint='csa_cache_writeback_spmd', allow_early_resolve=True):
                self.csa_cache_writeback(kv_cache_flat__ssa_v0, wb_blocks__ssa_v0, T_DYN, ori_slot_mapping__ssa_v0, kv__ssa_v0)
            compressor_dep__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[late_dep__ssa_v0, qa_tid__ssa_v0])
            cmp_out__ssa_v0: pl.Tensor[[T_DYN, 512], pl.FP32] = pl.tensor.create([T_DYN, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            pooled_kv_inline567__ssa_v0: pl.Tensor[[384, 512], pl.FP32] = pl.tensor.create([384, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            kv_proj_pad_inline565__ssa_v0: pl.Tensor[[384, 1024], pl.FP32] = pl.tensor.create([384, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            score_proj_pad_inline563__ssa_v0: pl.Tensor[[384, 1024], pl.FP32] = pl.tensor.create([384, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            bs_inline273_inline3759__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__rv_v2, 0)
            t_matmul_inline274_inline3773__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline273_inline3759__ssa_v0 + 63) // 64 * 64
            x_flat_inline278_inline3772__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16] = x_normed_t__rv_v2
            cmp4_kv_proj_pad_inline281_inline3787__ssa_v0: pl.Tensor[[384, 1024], pl.FP32] = kv_proj_pad_inline565__ssa_v0
            cmp4_score_proj_pad_inline284_inline3769__ssa_v0: pl.Tensor[[384, 1024], pl.FP32] = score_proj_pad_inline563__ssa_v0
            with pl.spmd(24, name_hint='kv_score_proj_spmd', deps=[compressor_dep__ssa_v0], allow_early_resolve=True) as _kv_score_tid_inline275_inline3765__ssa_v0:
                self.kv_score_proj(cmp4_kv_proj_pad_inline281_inline3787__ssa_v0, cmp4_score_proj_pad_inline284_inline3769__ssa_v0, t_matmul_inline274_inline3773__ssa_v0, bs_inline273_inline3759__ssa_v0, x_flat_inline278_inline3772__ssa_v0, cmp_wkv__ssa_v0, cmp_wgate__ssa_v0)
            projection_tid_inline3785__ssa_v0: pl.Scalar[pl.TASK_ID] = _kv_score_tid_inline275_inline3765__ssa_v0
            b_dim_inline308_inline3788__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(state_block_table__ssa_v0, 0)
            bs_inline312_inline3790__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            s_dim_inline307_inline3792__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline312_inline3790__ssa_v0 // b_dim_inline308_inline3788__ssa_v0
            cmp4_kv_proj_pad_inline300_inline3767__ssa_v0: pl.Tensor[[384, 1024], pl.FP32] = kv_proj_pad_inline565__ssa_v0
            cmp4_score_proj_pad_inline299_inline3775__ssa_v0: pl.Tensor[[384, 1024], pl.FP32] = score_proj_pad_inline563__ssa_v0
            _kv_score_tid_inline298_inline3795__ssa_v0: pl.Scalar[pl.TASK_ID] = projection_tid_inline3785__ssa_v0
            pool_workers_inline297_inline3797__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline308_inline3788__ssa_v0, 48)
            with pl.spmd(pool_workers_inline297_inline3797__ssa_v0, name_hint='scatter_softmax_pool_spmd', deps=[_kv_score_tid_inline298_inline3795__ssa_v0], allow_early_resolve=True) as pool_tid_inline295_inline3779__ssa_v0:
                ret__tmp_v0_4: pl.Tuple[pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[384, 512], pl.FP32]] = self.scatter_softmax_pool(pooled_kv_inline567__ssa_v0, b_dim_inline308_inline3788__ssa_v0, pool_workers_inline297_inline3797__ssa_v0, position_ids__ssa_v0, s_dim_inline307_inline3792__ssa_v0, cmp_seq_lens__ssa_v0, cmp4_score_proj_pad_inline299_inline3775__ssa_v0, cmp_ape__ssa_v0, cmp4_kv_proj_pad_inline300_inline3767__ssa_v0, state_block_table__ssa_v0, compress_state__ssa_v0)
                pooled_kv_inline567__rv_v2: pl.Tensor[[384, 512], pl.FP32] = ret__tmp_v0_4[0]
                pooled_kv_inline567__ssa_v5: pl.Tensor[[384, 512], pl.FP32] = ret__tmp_v0_4[1]
            pool_tid_inline3764__ssa_v0: pl.Scalar[pl.TASK_ID] = pool_tid_inline295_inline3779__ssa_v0
            projection_ready_tid_inline3746__ssa_v0: pl.Scalar[pl.TASK_ID] = _kv_score_tid_inline298_inline3795__ssa_v0
            pool_tid_inline566__ssa_v0: pl.Scalar[pl.TASK_ID] = pool_tid_inline3764__ssa_v0
            kv_score_tid_inline564__ssa_v0: pl.Scalar[pl.TASK_ID] = projection_ready_tid_inline3746__ssa_v0
            b_dim_inline3818__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(state_block_table__ssa_v0, 0)
            bs_inline3815__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            s_dim_inline3837__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline3815__ssa_v0 // b_dim_inline3818__ssa_v0
            rms_blocks_inline3824__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline3815__ssa_v0 + 15) // 16
            cmp_block_num_inline3819__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
            kv_flat_inline3830__ssa_v0: pl.Tensor[[T_DYN, 512], pl.FP32] = cmp_out__ssa_v0
            cmp_kv_cache_flat_inline3834__ssa_v0: pl.Tensor[[cmp_block_num_inline3819__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(cmp_kv__ssa_v0, [cmp_block_num_inline3819__ssa_v0 * 32, 512])
            cmp4_kv_proj_pad_inline3836__ssa_v0: pl.Tensor[[384, 1024], pl.FP32] = kv_proj_pad_inline565__ssa_v0
            cmp4_score_proj_pad_inline3838__ssa_v0: pl.Tensor[[384, 1024], pl.FP32] = score_proj_pad_inline563__ssa_v0
            commit_workers_inline3840__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline3818__ssa_v0, 48)
            with pl.spmd(commit_workers_inline3840__ssa_v0, name_hint='compress_state_commit_spmd', deps=[pool_tid_inline566__ssa_v0, pool_tid_inline566__ssa_v0]):
                self.compress_state_commit(compress_state__ssa_v0, b_dim_inline3818__ssa_v0, commit_workers_inline3840__ssa_v0, s_dim_inline3837__ssa_v0, state_slot_mapping__ssa_v0, position_ids__ssa_v0, cmp4_kv_proj_pad_inline3836__ssa_v0, cmp4_score_proj_pad_inline3838__ssa_v0, cmp_ape__ssa_v0)
            normed_kv_inline3852__ssa_v0: pl.Tensor[[384, 512], pl.FP32] = pl.tensor.create([384, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            norm_w_2d_inline3844__ssa_v0: pl.Tensor[[1, 512], pl.BF16] = pl.tensor.reshape(cmp_norm_w__ssa_v0, [1, 512])
            with pl.spmd(rms_blocks_inline3824__ssa_v0, name_hint='rmsnorm_rope_cache_write_spmd', deps=[pool_tid_inline566__ssa_v0, pool_tid_inline566__ssa_v0], allow_early_resolve=True) as cache_write_tid_inline3807__ssa_v0:
                self.rmsnorm_rope_cache_write(bs_inline3815__ssa_v0, position_ids__ssa_v0, cmp_row_offsets__ssa_v0, cmp_freqs_cos__ssa_v0, cmp_freqs_sin__ssa_v0, pooled_kv_inline567__rv_v2, normed_kv_inline3852__ssa_v0, norm_w_2d_inline3844__ssa_v0, cmp_kv_cache_flat_inline3834__ssa_v0, kv_flat_inline3830__ssa_v0, cmp_seq_lens__ssa_v0, cmp_slot_mapping__ssa_v0)
            cmp_out__ssa_v1: pl.Tensor[[T_DYN, 512], pl.FP32] = cmp_out__ssa_v0
            cmp_kv_score_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = kv_score_tid_inline564__ssa_v0
            idx_kv_unused__ssa_v0: pl.Tensor[[T_DYN, 128], pl.FP32] = pl.tensor.create([T_DYN, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            normed_kv_inline574__ssa_v0: pl.Tensor[[384, 128], pl.BF16] = pl.tensor.create([384, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            kv_proj_pad_inline3911__ssa_v0: pl.Tensor[[384, 256], pl.FP32] = pl.tensor.create([384, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            score_proj_pad_inline3938__ssa_v0: pl.Tensor[[384, 256], pl.FP32] = pl.tensor.create([384, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            pooled_kv_inline3908__ssa_v0: pl.Tensor[[384, 128], pl.FP32] = pl.tensor.create([384, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            bs_inline1593_inline3928__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__rv_v2, 0)
            t_matmul_inline1594_inline3937__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline1593_inline3928__ssa_v0 + 15) // 16 * 16
            x_flat_inline1595_inline3942__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16] = x_normed_t__rv_v2
            with pl.spmd(24, name_hint='kv_score_proj_spmd', deps=[late_dep__ssa_v0, late_dep__ssa_v0], allow_early_resolve=True) as _kv_score_tid_inline1600_inline3947__ssa_v0:
                ret__tmp_v0_5: pl.Tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32]] = self.kv_score_proj_0(kv_proj_pad_inline3911__ssa_v0, score_proj_pad_inline3938__ssa_v0, t_matmul_inline1594_inline3937__ssa_v0, bs_inline1593_inline3928__ssa_v0, x_flat_inline1595_inline3942__ssa_v0, inner_wkv__ssa_v0, inner_wgate__ssa_v0)
                kv_proj_pad_inline3911__rv_v2: pl.Tensor[[384, 256], pl.FP32] = ret__tmp_v0_5[0]
                score_proj_pad_inline3938__rv_v2: pl.Tensor[[384, 256], pl.FP32] = ret__tmp_v0_5[1]
            projection_tid_inline3905__ssa_v0: pl.Scalar[pl.TASK_ID] = _kv_score_tid_inline1600_inline3947__ssa_v0
            b_dim_inline1638_inline3904__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(inner_state_block_table__ssa_v0, 0)
            bs_inline1644_inline3932__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            s_dim_inline1634_inline3899__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline1644_inline3932__ssa_v0 // b_dim_inline1638_inline3904__ssa_v0
            rms_blocks_inline1646_inline3901__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline1644_inline3932__ssa_v0 + 15) // 16
            _kv_score_tid_inline1668_inline3918__ssa_v0: pl.Scalar[pl.TASK_ID] = projection_tid_inline3905__ssa_v0
            pool_workers_inline1636_inline3917__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline1638_inline3904__ssa_v0, 48)
            with pl.spmd(pool_workers_inline1636_inline3917__ssa_v0, name_hint='scatter_softmax_pool_spmd', deps=[_kv_score_tid_inline1668_inline3918__ssa_v0], allow_early_resolve=True) as pool_tid_inline1650_inline3906__ssa_v0:
                pooled_kv_inline3908__rv_v2: pl.Tensor[[384, 128], pl.FP32] = self.scatter_softmax_pool_0(pooled_kv_inline3908__ssa_v0, b_dim_inline1638_inline3904__ssa_v0, pool_workers_inline1636_inline3917__ssa_v0, position_ids__ssa_v0, s_dim_inline1634_inline3899__ssa_v0, score_proj_pad_inline3938__rv_v2, inner_ape__ssa_v0, kv_proj_pad_inline3911__rv_v2, inner_state_block_table__ssa_v0, inner_compress_state__ssa_v0)
            commit_workers_inline1662_inline3965__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline1638_inline3904__ssa_v0, 48)
            with pl.spmd(commit_workers_inline1662_inline3965__ssa_v0, name_hint='compress_state_commit_spmd', deps=[pool_tid_inline1650_inline3906__ssa_v0]):
                self.compress_state_commit_0(inner_compress_state__ssa_v0, b_dim_inline1638_inline3904__ssa_v0, commit_workers_inline1662_inline3965__ssa_v0, s_dim_inline1634_inline3899__ssa_v0, inner_state_slot_mapping__ssa_v0, position_ids__ssa_v0, kv_proj_pad_inline3911__rv_v2, score_proj_pad_inline3938__rv_v2, inner_ape__ssa_v0)
            norm_w_2d_inline1659_inline3897__ssa_v0: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.reshape(inner_norm_w__ssa_v0, [1, 128])
            with pl.spmd(b_dim_inline1638_inline3904__ssa_v0, name_hint='indexer_boundary_init_spmd', deps=[pool_tid_inline1650_inline3906__ssa_v0], allow_early_resolve=True) as boundary_init_tid_inline1671_inline3896__ssa_v0:
                normed_kv_inline574__ssa_v1: pl.Tensor[[384, 128], pl.BF16] = self.indexer_boundary_init(normed_kv_inline574__ssa_v0)
            rms_workers_inline1639_inline3967__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(rms_blocks_inline1646_inline3901__ssa_v0, 2)
            with pl.spmd(rms_workers_inline1639_inline3967__ssa_v0, name_hint='rmsnorm_rope_spmd', deps=[pool_tid_inline1650_inline3906__ssa_v0, boundary_init_tid_inline1671_inline3896__ssa_v0], allow_early_resolve=True) as rms_tid_inline1674_inline3903__ssa_v0:
                normed_kv_inline574__rv_v3: pl.Tensor[[384, 128], pl.BF16] = self.rmsnorm_rope(normed_kv_inline574__ssa_v1, rms_blocks_inline1646_inline3901__ssa_v0, rms_workers_inline1639_inline3967__ssa_v0, bs_inline1644_inline3932__ssa_v0, position_ids__ssa_v0, idx_row_offsets__ssa_v0, inner_freqs_cos__ssa_v0, inner_freqs_sin__ssa_v0, pooled_kv_inline3908__rv_v2, norm_w_2d_inline1659_inline3897__ssa_v0)
            rms_tid_inline3867__ssa_v0: pl.Scalar[pl.TASK_ID] = rms_tid_inline1674_inline3903__ssa_v0
            rms_tid_inline571__ssa_v0: pl.Scalar[pl.TASK_ID] = rms_tid_inline3867__ssa_v0
            bs_inline4012__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            compact_rows_inline4000__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline4012__ssa_v0 // 6 * 2
            rms_blocks_inline4008__ssa_v0: pl.Scalar[pl.INDEX] = (compact_rows_inline4000__ssa_v0 + 15) // 16
            kv_flat_inline4003__ssa_v0: pl.Tensor[[T_DYN, 128], pl.FP32] = idx_kv_unused__ssa_v0
            idx_kv_scale_values_inline3994__ssa_v0: pl.Tensor[[384, 1], pl.FP32] = pl.tensor.create([384, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            kv_final_inline3998__ssa_v0: pl.Tensor[[384, 128], pl.FP32] = pl.tensor.create([384, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            ret__tmp_v0_6: pl.Tuple[pl.Tensor[[384, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(self.kv_hadamard, kv_final_inline3998__ssa_v0, hadamard_idx__ssa_v0, rms_blocks_inline4008__ssa_v0, compact_rows_inline4000__ssa_v0, normed_kv_inline574__rv_v3, deps=[rms_tid_inline571__ssa_v0, cmp_kv_score_tid__ssa_v0], allow_early_resolve=True)
            kv_final_inline3998__rv_v2: pl.Tensor[[384, 128], pl.FP32] = ret__tmp_v0_6[0]
            hadamard_tid_inline3992__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_6[1]
            with pl.spmd(rms_blocks_inline4008__ssa_v0, name_hint='kv_and_cache_write_spmd', deps=[hadamard_tid_inline3992__ssa_v0], allow_early_resolve=True) as _write_tid_inline4011__ssa_v0:
                ret__tmp_v0_7: pl.Tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]] = self.kv_and_cache_write(compact_rows_inline4000__ssa_v0, kv_final_inline3998__rv_v2, idx_kv_scale_values_inline3994__ssa_v0, idx_native_kv_cache__ssa_v0, kv_flat_inline4003__ssa_v0, position_ids__ssa_v0, kv_seq_lens__ssa_v0, idx_row_offsets__ssa_v0, idx_slot_mapping__ssa_v0)
                idx_kv_scale_values_inline3994__ssa_v1: pl.Tensor[[384, 1], pl.FP32] = ret__tmp_v0_7[0]
                idx_native_kv_cache__rv_v2: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8] = ret__tmp_v0_7[1]
            ret__tmp_v0_8: pl.Tuple[pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.submit(self.idx_kv_scale_commit, compact_rows_inline4000__ssa_v0, position_ids__ssa_v0, kv_seq_lens__ssa_v0, idx_row_offsets__ssa_v0, idx_slot_mapping__ssa_v0, idx_kv_scale_values_inline3994__ssa_v1, idx_native_kv_cache__rv_v2, deps=[_write_tid_inline4011__ssa_v0], allow_early_resolve=True)
            idx_native_kv_cache__rv_v3: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8] = ret__tmp_v0_8[0]
            scale_commit_tid_inline3977__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_8[1]
            write_tid_inline568__ssa_v0: pl.Scalar[pl.TASK_ID] = scale_commit_tid_inline3977__ssa_v0
            idx_cache_write_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = write_tid_inline568__ssa_v0
            with pl.scope():
                qr_hadamard_i8_inline583__ssa_v0: pl.Tensor[[24576, 128], pl.INT8] = pl.tensor.create([24576, 128], dtype=pl.INT8, layout=pl.TensorLayout.ND)
                qr_hadamard_scale_dq_inline582__ssa_v0: pl.Tensor[[24576, 1], pl.FP32] = pl.tensor.create([24576, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                qr_bf16_inline4070__ssa_v0: pl.Tensor[[24576, 128], pl.BF16] = pl.tensor.create([24576, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                bs_inline1997_inline4109__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__rv_v2, 0)
                row_blocks_inline2004_inline4047__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline1997_inline4109__ssa_v0 + 63) // 64
                qr_acc_pad_inline2007_inline4067__ssa_v0: pl.Tensor[[384, 8192], pl.INT32] = pl.tensor.create([384, 8192], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                with pl.spmd(24, name_hint='idx_qr_proj_matmul_spmd', allow_early_resolve=True) as idx_qr_mm_tid_inline2011_inline4057__ssa_v0:
                    qr_acc_pad_inline2007_inline4067__rv_v2: pl.Tensor[[384, 8192], pl.INT32] = self.idx_qr_proj_matmul(qr_acc_pad_inline2007_inline4067__ssa_v0, idx_wq_b__ssa_v0, row_blocks_inline2004_inline4047__ssa_v0, bs_inline1997_inline4109__ssa_v0, qr__ssa_v0)
                qr_bf16_2d_inline1995_inline4100__ssa_v0: pl.Tensor[[384, 8192], pl.BF16] = pl.tensor.reshape(qr_bf16_inline4070__ssa_v0, [384, 8192])
                dq_rope_units_inline2000_inline4054__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline1997_inline4109__ssa_v0 + 7) // 8 * 16
                dq_rope_workers_inline2016_inline4062__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(dq_rope_units_inline2000_inline4054__ssa_v0, 48)
                with pl.spmd(dq_rope_workers_inline2016_inline4062__ssa_v0, name_hint='idx_qr_dequant_rope_spmd', allow_early_resolve=True):
                    self.idx_qr_dequant_rope(qr_bf16_2d_inline1995_inline4100__ssa_v0, dq_rope_units_inline2000_inline4054__ssa_v0, dq_rope_workers_inline2016_inline4062__ssa_v0, bs_inline1997_inline4109__ssa_v0, qr_scale__ssa_v0, freqs_cos__ssa_v0, idx_sin_signed__ssa_v1, idx_wq_b_scale__ssa_v0, qr_acc_pad_inline2007_inline4067__rv_v2)
                idx_qr_mm_tid_inline4040__ssa_v0: pl.Scalar[pl.TASK_ID] = idx_qr_mm_tid_inline2011_inline4057__ssa_v0
                bs_inline2053_inline4099__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__rv_v2, 0)
                bs_heads_inline2066_inline4048__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline2053_inline4099__ssa_v0 * 64
                qh_acc_gm_inline2065_inline4066__ssa_v0: pl.Tensor[[bs_heads_inline2066_inline4048__ssa_v0, 128], pl.FP32] = pl.tensor.create([bs_heads_inline2066_inline4048__ssa_v0, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                with pl.spmd(24, name_hint='qr_hadamard_matmul_spmd', deps=[idx_qr_mm_tid_inline4040__ssa_v0], allow_early_resolve=True) as qh_mm_tid_inline2063_inline4055__ssa_v0:
                    qh_acc_gm_inline2065_inline4066__rv_v2: pl.Tensor[[bs_heads_inline2066_inline4048__ssa_v0, 128], pl.FP32] = self.qr_hadamard_matmul(hadamard_idx__ssa_v0, qh_acc_gm_inline2065_inline4066__ssa_v0, bs_heads_inline2066_inline4048__ssa_v0, qr_bf16_inline4070__ssa_v0)
                with pl.spmd(48, name_hint='qr_hadamard_quant_spmd', allow_early_resolve=True) as qh_quant_tid_inline2049_inline4078__ssa_v0:
                    ret__tmp_v0_9: pl.Tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32]] = self.qr_hadamard_quant(qr_hadamard_i8_inline583__ssa_v0, qr_hadamard_scale_dq_inline582__ssa_v0, bs_heads_inline2066_inline4048__ssa_v0, qh_acc_gm_inline2065_inline4066__rv_v2)
                    qr_hadamard_i8_inline583__rv_v2: pl.Tensor[[24576, 128], pl.INT8] = ret__tmp_v0_9[0]
                    qr_hadamard_scale_dq_inline582__rv_v2: pl.Tensor[[24576, 1], pl.FP32] = ret__tmp_v0_9[1]
                qh_quant_tid_inline4098__ssa_v0: pl.Scalar[pl.TASK_ID] = qh_quant_tid_inline2049_inline4078__ssa_v0
                qh_quant_tid_inline579__ssa_v0: pl.Scalar[pl.TASK_ID] = qh_quant_tid_inline4098__ssa_v0
                weights_gate_dep_inline576__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
                bs_inline3187_inline4159__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__rv_v2, 0)
                row_blocks_inline3179_inline4165__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline3187_inline4159__ssa_v0 + 15) // 16
                x_flat_inline3180_inline4167__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16] = x_normed_t__rv_v2
                weights_inline3183_inline4184__ssa_v0: pl.Tensor[[384, 64], pl.FP32] = pl.tensor.create([384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                weights_partial_inline3185_inline4185__ssa_v0: pl.Tensor[[1536, 64], pl.FP32] = pl.tensor.create([1536, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                with pl.spmd(8, name_hint='weights_proj_spmd', deps=[weights_gate_dep_inline576__ssa_v0], allow_early_resolve=True) as _weights_tid_inline3189_inline4190__ssa_v0:
                    weights_partial_inline3185_inline4185__rv_v2: pl.Tensor[[1536, 64], pl.FP32] = self.weights_proj(weights_partial_inline3185_inline4185__ssa_v0, row_blocks_inline3179_inline4165__ssa_v0, bs_inline3187_inline4159__ssa_v0, x_flat_inline3180_inline4167__ssa_v0, weights_proj__ssa_v0)
                with pl.spmd(row_blocks_inline3179_inline4165__ssa_v0, name_hint='weights_proj_reduce_spmd', allow_early_resolve=True) as weights_tid_inline3175_inline4162__ssa_v0:
                    weights_inline3183_inline4184__ssa_v1: pl.Tensor[[384, 64], pl.FP32] = self.weights_proj_reduce(weights_partial_inline3185_inline4185__rv_v2, weights_inline3183_inline4184__ssa_v0)
                weights_inline4197__ssa_v0: pl.Tensor[[384, 64], pl.FP32] = weights_inline3183_inline4184__ssa_v1
                weights_tid_inline4157__ssa_v0: pl.Scalar[pl.TASK_ID] = weights_tid_inline3175_inline4162__ssa_v0
                native_page_bytes_inline3220_inline4177__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_native_kv_cache__rv_v3, 1)
                t__tmp_v277: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_native_kv_cache__rv_v3, 0)
                cache_bytes_inline3221_inline4156__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v277 * native_page_bytes_inline3220_inline4177__ssa_v0
                cache_flat_inline3231_inline4172__ssa_v0: pl.Tensor[[cache_bytes_inline3221_inline4156__ssa_v0], pl.INT8] = pl.tensor.reshape(idx_native_kv_cache__rv_v3, [cache_bytes_inline3221_inline4156__ssa_v0])
                key_rows_inline3222_inline4169__ssa_v0: pl.Scalar[pl.INDEX] = cache_bytes_inline3221_inline4156__ssa_v0 // 128
                shifted_rows_inline3223_inline4178__ssa_v0: pl.Scalar[pl.INDEX] = (cache_bytes_inline3221_inline4156__ssa_v0 - 64) // 128
                t__tmp_v278: pl.Tensor[[key_rows_inline3222_inline4169__ssa_v0 * pl.const(128, pl.INDEX)], pl.INT8] = pl.tensor.slice(cache_flat_inline3231_inline4172__ssa_v0, [key_rows_inline3222_inline4169__ssa_v0 * 128], [0])
                idx_kv_cache_inline3234_inline4199__ssa_v0: pl.Tensor[[key_rows_inline3222_inline4169__ssa_v0, 128], pl.INT8] = pl.tensor.reshape(t__tmp_v278, [key_rows_inline3222_inline4169__ssa_v0, 128])
                t__tmp_v279: pl.Tensor[[shifted_rows_inline3223_inline4178__ssa_v0 * pl.const(128, pl.INDEX)], pl.INT8] = pl.tensor.slice(cache_flat_inline3231_inline4172__ssa_v0, [shifted_rows_inline3223_inline4178__ssa_v0 * 128], [64])
                idx_kv_cache_shift64_inline3230_inline4201__ssa_v0: pl.Tensor[[shifted_rows_inline3223_inline4178__ssa_v0, 128], pl.INT8] = pl.tensor.reshape(t__tmp_v279, [shifted_rows_inline3223_inline4178__ssa_v0, 128])
                pair_arena_inline3226_inline4202__ssa_v0: pl.Tensor[[24576, 1024], pl.FP32] = pl.tensor.create([24576, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                score_arena_inline3255_inline4168__ssa_v0: pl.Tensor[[384, 8192], pl.FP32] = pl.tensor.create([384, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                max_topk_cache_len_inline3209_inline4203__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(max_seq_len__ssa_v0, pl.INDEX) // 4
                if 2048 <= max_topk_cache_len_inline3209_inline4203__ssa_v0:
                    if 8192 < max_topk_cache_len_inline3209_inline4203__ssa_v0:
                        t__tmp_v280: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
                        if 24 <= t__tmp_v280:
                            native_page_bytes_inline8726__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_native_kv_cache__rv_v3, 1)
                            t__tmp_v281: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_native_kv_cache__rv_v3, 0)
                            cache_bytes_inline8786__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v281 * native_page_bytes_inline8726__ssa_v0
                            cache_flat_inline8754__ssa_v0: pl.Tensor[[cache_bytes_inline8786__ssa_v0], pl.INT8] = pl.tensor.reshape(idx_native_kv_cache__rv_v3, [cache_bytes_inline8786__ssa_v0])
                            key_rows_inline8656__ssa_v0: pl.Scalar[pl.INDEX] = cache_bytes_inline8786__ssa_v0 // 128
                            shifted_rows_inline8748__ssa_v0: pl.Scalar[pl.INDEX] = (cache_bytes_inline8786__ssa_v0 - 64) // 128
                            t__tmp_v282: pl.Tensor[[key_rows_inline8656__ssa_v0 * pl.const(128, pl.INDEX)], pl.INT8] = pl.tensor.slice(cache_flat_inline8754__ssa_v0, [key_rows_inline8656__ssa_v0 * 128], [0])
                            idx_kv_cache_inline8854__ssa_v0: pl.Tensor[[key_rows_inline8656__ssa_v0, 128], pl.INT8] = pl.tensor.reshape(t__tmp_v282, [key_rows_inline8656__ssa_v0, 128])
                            t__tmp_v283: pl.Tensor[[shifted_rows_inline8748__ssa_v0 * pl.const(128, pl.INDEX)], pl.INT8] = pl.tensor.slice(cache_flat_inline8754__ssa_v0, [shifted_rows_inline8748__ssa_v0 * 128], [64])
                            idx_kv_cache_shift64_inline8790__ssa_v0: pl.Tensor[[shifted_rows_inline8748__ssa_v0, 128], pl.INT8] = pl.tensor.reshape(t__tmp_v283, [shifted_rows_inline8748__ssa_v0, 128])
                            coefficients_inline1281_inline8756__ssa_v0: pl.Tensor[[1024, 384], pl.FP16] = pl.tensor.create([1024, 384], dtype=pl.FP16, layout=pl.TensorLayout.ND)
                            coefficient_scales_inline1285_inline8894__ssa_v0: pl.Tensor[[384, 64], pl.FP32] = pl.tensor.reshape(qr_hadamard_scale_dq_inline582__rv_v2, [384, 64])
                            t__tmp_v284: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
                            coefficient_workers_inline1282_inline8781__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t__tmp_v284 // 6, 48)
                            with pl.spmd(coefficient_workers_inline1282_inline8781__ssa_v0, name_hint='indexer_head_coefficients_spmd', deps=[qh_quant_tid_inline579__ssa_v0, weights_tid_inline4157__ssa_v0], allow_early_resolve=True) as coefficients_tid_inline1287_inline8922__ssa_v0:
                                coefficients_inline1281_inline8756__ssa_v1: pl.Tensor[[1024, 384], pl.FP16] = self.indexer_head_coefficients(position_ids__ssa_v0, coefficient_scales_inline1285_inline8894__ssa_v0, weights_inline4197__ssa_v0, coefficients_inline1281_inline8756__ssa_v0)
                            coefficients_inline8733__ssa_v0: pl.Tensor[[1024, 384], pl.FP16] = coefficients_inline1281_inline8756__ssa_v1
                            coefficients_tid_inline8758__ssa_v0: pl.Scalar[pl.TASK_ID] = coefficients_tid_inline1287_inline8922__ssa_v0
                            buf_score_transfer_inline8845__ssa_v0: pl.Tensor[[288, 1024], pl.FP32] = pl.tensor.create([288, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            buf_score_ffts_inline8757__ssa_v0: pl.Tensor[[256], pl.INT64] = pl.tensor.create([256], dtype=pl.INT64, layout=pl.TensorLayout.ND)
                            with pl.spmd(24, sync_start=True, name_hint='indexer_score_topk_native_pair_spmd', deps=[coefficients_tid_inline8758__ssa_v0, idx_cache_write_tid__ssa_v0]) as buffered_leaf_tid_inline8730__ssa_v0:
                                ret__tmp_v0_10: pl.Tuple[pl.Tensor[[288, 1024], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]] = self.indexer_score_topk_native_pair(position_ids__ssa_v0, kv_seq_lens__ssa_v0, buf_score_ffts_inline8757__ssa_v0, qr_hadamard_i8_inline583__rv_v2, coefficients_inline8733__ssa_v0, idx_block_table__ssa_v0, native_page_bytes_inline8726__ssa_v0, idx_kv_cache_inline8854__ssa_v0, idx_kv_cache_shift64_inline8790__ssa_v0, buf_score_transfer_inline8845__ssa_v0, idx_native_kv_cache__rv_v3, pair_arena_inline3226_inline4202__ssa_v0)
                                buf_score_transfer_inline8845__ssa_v1: pl.Tensor[[288, 1024], pl.FP32] = ret__tmp_v0_10[0]
                                pair_arena_inline3226_inline4202__ssa_v1: pl.Tensor[[24576, 1024], pl.FP32] = ret__tmp_v0_10[1]
                            score_tid_inline3240_inline4204__ssa_v0: pl.Scalar[pl.TASK_ID] = buffered_leaf_tid_inline8730__ssa_v0
                            (score_tid_inline3240_inline4204__phi_v2, pair_arena_inline3226_inline4202__rv_v1, score_arena_inline3255_inline4168__rv_v1) = pl.yield_(score_tid_inline3240_inline4204__ssa_v0, pair_arena_inline3226_inline4202__ssa_v1, score_arena_inline3255_inline4168__ssa_v0)
                        else:
                            native_page_bytes_inline9037__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_native_kv_cache__rv_v3, 1)
                            t__tmp_v380: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_native_kv_cache__rv_v3, 0)
                            cache_bytes_inline9065__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v380 * native_page_bytes_inline9037__ssa_v0
                            cache_flat_inline9053__ssa_v0: pl.Tensor[[cache_bytes_inline9065__ssa_v0], pl.INT8] = pl.tensor.reshape(idx_native_kv_cache__rv_v3, [cache_bytes_inline9065__ssa_v0])
                            key_rows_inline9041__ssa_v0: pl.Scalar[pl.INDEX] = cache_bytes_inline9065__ssa_v0 // 128
                            shifted_rows_inline9108__ssa_v0: pl.Scalar[pl.INDEX] = (cache_bytes_inline9065__ssa_v0 - 64) // 128
                            t__tmp_v381: pl.Tensor[[key_rows_inline9041__ssa_v0 * pl.const(128, pl.INDEX)], pl.INT8] = pl.tensor.slice(cache_flat_inline9053__ssa_v0, [key_rows_inline9041__ssa_v0 * 128], [0])
                            idx_kv_cache_inline9127__ssa_v0: pl.Tensor[[key_rows_inline9041__ssa_v0, 128], pl.INT8] = pl.tensor.reshape(t__tmp_v381, [key_rows_inline9041__ssa_v0, 128])
                            t__tmp_v382: pl.Tensor[[shifted_rows_inline9108__ssa_v0 * pl.const(128, pl.INDEX)], pl.INT8] = pl.tensor.slice(cache_flat_inline9053__ssa_v0, [shifted_rows_inline9108__ssa_v0 * 128], [64])
                            idx_kv_cache_shift64_inline9079__ssa_v0: pl.Tensor[[shifted_rows_inline9108__ssa_v0, 128], pl.INT8] = pl.tensor.reshape(t__tmp_v382, [shifted_rows_inline9108__ssa_v0, 128])
                            coefficients_inline2879_inline9040__ssa_v0: pl.Tensor[[3072, 128], pl.FP16] = pl.tensor.create([3072, 128], dtype=pl.FP16, layout=pl.TensorLayout.ND)
                            coefficient_scales_inline2881_inline9074__ssa_v0: pl.Tensor[[384, 64], pl.FP32] = pl.tensor.reshape(qr_hadamard_scale_dq_inline582__rv_v2, [384, 64])
                            t__tmp_v383: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
                            coefficient_workers_inline2884_inline9021__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t__tmp_v383 // 2, 48)
                            with pl.spmd(coefficient_workers_inline2884_inline9021__ssa_v0, name_hint='indexer_head_coefficients_spmd', deps=[qh_quant_tid_inline579__ssa_v0, weights_tid_inline4157__ssa_v0], allow_early_resolve=True) as coefficients_tid_inline2883_inline9066__ssa_v0:
                                coefficients_inline2879_inline9040__ssa_v1: pl.Tensor[[3072, 128], pl.FP16] = self.indexer_head_coefficients_0(position_ids__ssa_v0, coefficient_scales_inline2881_inline9074__ssa_v0, weights_inline4197__ssa_v0, coefficients_inline2879_inline9040__ssa_v0)
                            coefficients_inline9000__ssa_v0: pl.Tensor[[3072, 128], pl.FP16] = coefficients_inline2879_inline9040__ssa_v1
                            coefficients_tid_inline9052__ssa_v0: pl.Scalar[pl.TASK_ID] = coefficients_tid_inline2883_inline9066__ssa_v0
                            buf_score_transfer_inline9167__ssa_v0: pl.Tensor[[96, 1024], pl.FP32] = pl.tensor.create([96, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            buf_score_ffts_inline8997__ssa_v0: pl.Tensor[[256], pl.INT64] = pl.tensor.create([256], dtype=pl.INT64, layout=pl.TensorLayout.ND)
                            with pl.spmd(24, sync_start=True, name_hint='indexer_score_topk_native_pair_spmd', deps=[coefficients_tid_inline9052__ssa_v0, idx_cache_write_tid__ssa_v0]) as buffered_leaf_tid_inline9043__ssa_v0:
                                ret__tmp_v0_11: pl.Tuple[pl.Tensor[[96, 1024], pl.FP32], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]] = self.indexer_score_topk_native_pair_0(position_ids__ssa_v0, kv_seq_lens__ssa_v0, buf_score_ffts_inline8997__ssa_v0, qr_hadamard_i8_inline583__rv_v2, coefficients_inline9000__ssa_v0, idx_block_table__ssa_v0, native_page_bytes_inline9037__ssa_v0, idx_kv_cache_inline9127__ssa_v0, idx_kv_cache_shift64_inline9079__ssa_v0, buf_score_transfer_inline9167__ssa_v0, idx_native_kv_cache__rv_v3, score_arena_inline3255_inline4168__ssa_v0, pair_arena_inline3226_inline4202__ssa_v0)
                                buf_score_transfer_inline9167__ssa_v1: pl.Tensor[[96, 1024], pl.FP32] = ret__tmp_v0_11[0]
                                score_arena_inline3255_inline4168__ssa_v1: pl.Tensor[[384, 8192], pl.FP32] = ret__tmp_v0_11[1]
                                pair_arena_inline3226_inline4202__ssa_v2: pl.Tensor[[24576, 1024], pl.FP32] = ret__tmp_v0_11[2]
                            score_tid_inline3240_inline4204__ssa_v1: pl.Scalar[pl.TASK_ID] = buffered_leaf_tid_inline9043__ssa_v0
                            (score_tid_inline3240_inline4204__phi_v2, pair_arena_inline3226_inline4202__rv_v1, score_arena_inline3255_inline4168__rv_v1) = pl.yield_(score_tid_inline3240_inline4204__ssa_v1, pair_arena_inline3226_inline4202__ssa_v2, score_arena_inline3255_inline4168__ssa_v1)
                        (score_tid_inline3240_inline4204__phi_v6, pair_arena_inline3226_inline4202__rv_v3, score_arena_inline3255_inline4168__rv_v5) = pl.yield_(score_tid_inline3240_inline4204__phi_v2, pair_arena_inline3226_inline4202__rv_v1, score_arena_inline3255_inline4168__rv_v1)
                    else:
                        short_queries_inline3239_inline4183__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
                        short_pair_waves_inline3232_inline4205__ssa_v0: pl.Scalar[pl.INDEX] = (short_queries_inline3239_inline4183__ssa_v0 + 46) // 48
                        short_request_waves_inline3218_inline4206__ssa_v0: pl.Scalar[pl.INDEX] = (short_queries_inline3239_inline4183__ssa_v0 + 138) // 144
                        short_use_request_inline3219_inline4208__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
                        if 144 <= short_queries_inline3239_inline4183__ssa_v0:
                            if short_request_waves_inline3218_inline4206__ssa_v0 * 24 <= short_pair_waves_inline3232_inline4205__ssa_v0 * 10:
                                short_use_request_inline3219_inline4208__ssa_v1: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
                                short_use_request_inline3219_inline4208__phi_v2: pl.Scalar[pl.INT32] = pl.yield_(short_use_request_inline3219_inline4208__ssa_v1)
                            else:
                                short_use_request_inline3219_inline4208__phi_v2: pl.Scalar[pl.INT32] = pl.yield_(short_use_request_inline3219_inline4208__ssa_v0)
                            short_use_request_inline3219_inline4208__phi_v3: pl.Scalar[pl.INT32] = pl.yield_(short_use_request_inline3219_inline4208__phi_v2)
                        else:
                            short_use_request_inline3219_inline4208__phi_v3: pl.Scalar[pl.INT32] = pl.yield_(short_use_request_inline3219_inline4208__ssa_v0)
                        if pl.cast(short_use_request_inline3219_inline4208__phi_v3, pl.INDEX) == 1:
                            native_page_bytes_inline9325__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_native_kv_cache__rv_v3, 1)
                            t__tmp_v472: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_native_kv_cache__rv_v3, 0)
                            cache_bytes_inline9379__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v472 * native_page_bytes_inline9325__ssa_v0
                            cache_flat_inline9405__ssa_v0: pl.Tensor[[cache_bytes_inline9379__ssa_v0], pl.INT8] = pl.tensor.reshape(idx_native_kv_cache__rv_v3, [cache_bytes_inline9379__ssa_v0])
                            key_rows_inline9391__ssa_v0: pl.Scalar[pl.INDEX] = cache_bytes_inline9379__ssa_v0 // 128
                            shifted_rows_inline9361__ssa_v0: pl.Scalar[pl.INDEX] = (cache_bytes_inline9379__ssa_v0 - 64) // 128
                            t__tmp_v473: pl.Tensor[[key_rows_inline9391__ssa_v0 * pl.const(128, pl.INDEX)], pl.INT8] = pl.tensor.slice(cache_flat_inline9405__ssa_v0, [key_rows_inline9391__ssa_v0 * 128], [0])
                            idx_kv_cache_inline9317__ssa_v0: pl.Tensor[[key_rows_inline9391__ssa_v0, 128], pl.INT8] = pl.tensor.reshape(t__tmp_v473, [key_rows_inline9391__ssa_v0, 128])
                            t__tmp_v474: pl.Tensor[[shifted_rows_inline9361__ssa_v0 * pl.const(128, pl.INDEX)], pl.INT8] = pl.tensor.slice(cache_flat_inline9405__ssa_v0, [shifted_rows_inline9361__ssa_v0 * 128], [64])
                            idx_kv_cache_shift64_inline9293__ssa_v0: pl.Tensor[[shifted_rows_inline9361__ssa_v0, 128], pl.INT8] = pl.tensor.reshape(t__tmp_v474, [shifted_rows_inline9361__ssa_v0, 128])
                            coefficients_inline1735_inline9271__ssa_v0: pl.Tensor[[1024, 384], pl.FP16] = pl.tensor.create([1024, 384], dtype=pl.FP16, layout=pl.TensorLayout.ND)
                            coefficient_scales_inline1739_inline9508__ssa_v0: pl.Tensor[[384, 64], pl.FP32] = pl.tensor.reshape(qr_hadamard_scale_dq_inline582__rv_v2, [384, 64])
                            t__tmp_v475: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
                            coefficient_workers_inline1736_inline9396__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t__tmp_v475 // 6, 48)
                            with pl.spmd(coefficient_workers_inline1736_inline9396__ssa_v0, name_hint='indexer_head_coefficients_spmd', deps=[qh_quant_tid_inline579__ssa_v0, weights_tid_inline4157__ssa_v0], allow_early_resolve=True) as coefficients_tid_inline1741_inline9350__ssa_v0:
                                coefficients_inline1735_inline9271__ssa_v1: pl.Tensor[[1024, 384], pl.FP16] = self.indexer_head_coefficients_1(position_ids__ssa_v0, coefficient_scales_inline1739_inline9508__ssa_v0, weights_inline4197__ssa_v0, coefficients_inline1735_inline9271__ssa_v0)
                            coefficients_inline9363__ssa_v0: pl.Tensor[[1024, 384], pl.FP16] = coefficients_inline1735_inline9271__ssa_v1
                            coefficients_tid_inline9362__ssa_v0: pl.Scalar[pl.TASK_ID] = coefficients_tid_inline1741_inline9350__ssa_v0
                            buf_score_transfer_inline9382__ssa_v0: pl.Tensor[[288, 768], pl.FP32] = pl.tensor.create([288, 768], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            buf_score_ffts_inline9502__ssa_v0: pl.Tensor[[256], pl.INT64] = pl.tensor.create([256], dtype=pl.INT64, layout=pl.TensorLayout.ND)
                            with pl.spmd(24, name_hint='indexer_score_topk_native_pair_spmd', deps=[coefficients_tid_inline9362__ssa_v0, idx_cache_write_tid__ssa_v0], allow_early_resolve=True) as buffered_leaf_tid_inline9323__ssa_v0:
                                ret__tmp_v0_12: pl.Tuple[pl.Tensor[[288, 768], pl.FP32], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]] = self.indexer_score_topk_native_pair_1(position_ids__ssa_v0, kv_seq_lens__ssa_v0, buf_score_ffts_inline9502__ssa_v0, qr_hadamard_i8_inline583__rv_v2, coefficients_inline9363__ssa_v0, idx_block_table__ssa_v0, native_page_bytes_inline9325__ssa_v0, idx_kv_cache_inline9317__ssa_v0, idx_kv_cache_shift64_inline9293__ssa_v0, buf_score_transfer_inline9382__ssa_v0, idx_native_kv_cache__rv_v3, score_arena_inline3255_inline4168__ssa_v0, pair_arena_inline3226_inline4202__ssa_v0)
                                buf_score_transfer_inline9382__ssa_v1: pl.Tensor[[288, 768], pl.FP32] = ret__tmp_v0_12[0]
                                score_arena_inline3255_inline4168__ssa_v2: pl.Tensor[[384, 8192], pl.FP32] = ret__tmp_v0_12[1]
                                pair_arena_inline3226_inline4202__ssa_v3: pl.Tensor[[24576, 1024], pl.FP32] = ret__tmp_v0_12[2]
                            score_tid_inline3240_inline4204__ssa_v3: pl.Scalar[pl.TASK_ID] = buffered_leaf_tid_inline9323__ssa_v0
                            (score_tid_inline3240_inline4204__phi_v5, score_arena_inline3255_inline4168__rv_v3, pair_arena_inline3226_inline4202__rv_v2) = pl.yield_(score_tid_inline3240_inline4204__ssa_v3, score_arena_inline3255_inline4168__ssa_v2, pair_arena_inline3226_inline4202__ssa_v3)
                        else:
                            native_page_bytes_inline9662__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_native_kv_cache__rv_v3, 1)
                            t__tmp_v636: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_native_kv_cache__rv_v3, 0)
                            cache_bytes_inline9651__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v636 * native_page_bytes_inline9662__ssa_v0
                            cache_flat_inline9822__ssa_v0: pl.Tensor[[cache_bytes_inline9651__ssa_v0], pl.INT8] = pl.tensor.reshape(idx_native_kv_cache__rv_v3, [cache_bytes_inline9651__ssa_v0])
                            key_rows_inline9688__ssa_v0: pl.Scalar[pl.INDEX] = cache_bytes_inline9651__ssa_v0 // 128
                            shifted_rows_inline9696__ssa_v0: pl.Scalar[pl.INDEX] = (cache_bytes_inline9651__ssa_v0 - 64) // 128
                            t__tmp_v637: pl.Tensor[[key_rows_inline9688__ssa_v0 * pl.const(128, pl.INDEX)], pl.INT8] = pl.tensor.slice(cache_flat_inline9822__ssa_v0, [key_rows_inline9688__ssa_v0 * 128], [0])
                            idx_kv_cache_inline9701__ssa_v0: pl.Tensor[[key_rows_inline9688__ssa_v0, 128], pl.INT8] = pl.tensor.reshape(t__tmp_v637, [key_rows_inline9688__ssa_v0, 128])
                            t__tmp_v638: pl.Tensor[[shifted_rows_inline9696__ssa_v0 * pl.const(128, pl.INDEX)], pl.INT8] = pl.tensor.slice(cache_flat_inline9822__ssa_v0, [shifted_rows_inline9696__ssa_v0 * 128], [64])
                            idx_kv_cache_shift64_inline9817__ssa_v0: pl.Tensor[[shifted_rows_inline9696__ssa_v0, 128], pl.INT8] = pl.tensor.reshape(t__tmp_v638, [shifted_rows_inline9696__ssa_v0, 128])
                            coefficients_inline3035_inline9660__ssa_v0: pl.Tensor[[3072, 128], pl.FP16] = pl.tensor.create([3072, 128], dtype=pl.FP16, layout=pl.TensorLayout.ND)
                            coefficient_scales_inline3037_inline9683__ssa_v0: pl.Tensor[[384, 64], pl.FP32] = pl.tensor.reshape(qr_hadamard_scale_dq_inline582__rv_v2, [384, 64])
                            t__tmp_v639: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
                            coefficient_workers_inline3040_inline9790__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t__tmp_v639 // 2, 48)
                            with pl.spmd(coefficient_workers_inline3040_inline9790__ssa_v0, name_hint='indexer_head_coefficients_spmd', deps=[qh_quant_tid_inline579__ssa_v0, weights_tid_inline4157__ssa_v0], allow_early_resolve=True) as coefficients_tid_inline3039_inline9697__ssa_v0:
                                coefficients_inline3035_inline9660__ssa_v1: pl.Tensor[[3072, 128], pl.FP16] = self.indexer_head_coefficients_2(position_ids__ssa_v0, coefficient_scales_inline3037_inline9683__ssa_v0, weights_inline4197__ssa_v0, coefficients_inline3035_inline9660__ssa_v0)
                            coefficients_inline9652__ssa_v0: pl.Tensor[[3072, 128], pl.FP16] = coefficients_inline3035_inline9660__ssa_v1
                            coefficients_tid_inline9736__ssa_v0: pl.Scalar[pl.TASK_ID] = coefficients_tid_inline3039_inline9697__ssa_v0
                            buf_score_transfer_inline9767__ssa_v0: pl.Tensor[[96, 768], pl.FP32] = pl.tensor.create([96, 768], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            buf_score_ffts_inline9751__ssa_v0: pl.Tensor[[256], pl.INT64] = pl.tensor.create([256], dtype=pl.INT64, layout=pl.TensorLayout.ND)
                            with pl.spmd(24, name_hint='indexer_score_topk_native_pair_spmd', deps=[coefficients_tid_inline9736__ssa_v0, idx_cache_write_tid__ssa_v0], allow_early_resolve=True) as buffered_leaf_tid_inline9699__ssa_v0:
                                ret__tmp_v0_13: pl.Tuple[pl.Tensor[[96, 768], pl.FP32], pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]] = self.indexer_score_topk_native_pair_2(position_ids__ssa_v0, kv_seq_lens__ssa_v0, buf_score_ffts_inline9751__ssa_v0, qr_hadamard_i8_inline583__rv_v2, coefficients_inline9652__ssa_v0, idx_block_table__ssa_v0, native_page_bytes_inline9662__ssa_v0, idx_kv_cache_inline9701__ssa_v0, idx_kv_cache_shift64_inline9817__ssa_v0, buf_score_transfer_inline9767__ssa_v0, idx_native_kv_cache__rv_v3, score_arena_inline3255_inline4168__ssa_v0, pair_arena_inline3226_inline4202__ssa_v0)
                                buf_score_transfer_inline9767__ssa_v1: pl.Tensor[[96, 768], pl.FP32] = ret__tmp_v0_13[0]
                                score_arena_inline3255_inline4168__ssa_v3: pl.Tensor[[384, 8192], pl.FP32] = ret__tmp_v0_13[1]
                                pair_arena_inline3226_inline4202__ssa_v4: pl.Tensor[[24576, 1024], pl.FP32] = ret__tmp_v0_13[2]
                            score_tid_inline3240_inline4204__ssa_v4: pl.Scalar[pl.TASK_ID] = buffered_leaf_tid_inline9699__ssa_v0
                            (score_tid_inline3240_inline4204__phi_v5, score_arena_inline3255_inline4168__rv_v3, pair_arena_inline3226_inline4202__rv_v2) = pl.yield_(score_tid_inline3240_inline4204__ssa_v4, score_arena_inline3255_inline4168__ssa_v3, pair_arena_inline3226_inline4202__ssa_v4)
                        (score_tid_inline3240_inline4204__phi_v6, pair_arena_inline3226_inline4202__rv_v3, score_arena_inline3255_inline4168__rv_v5) = pl.yield_(score_tid_inline3240_inline4204__phi_v5, pair_arena_inline3226_inline4202__rv_v2, score_arena_inline3255_inline4168__rv_v3)
                    (score_arena_inline3255_inline4168__phi_v8, score_tid_inline3240_inline4204__phi_v8, pair_arena_inline3226_inline4202__rv_v4, score_arena_inline3255_inline4168__rv_v6) = pl.yield_(score_arena_inline3255_inline4168__rv_v5, score_tid_inline3240_inline4204__phi_v6, pair_arena_inline3226_inline4202__rv_v3, score_arena_inline3255_inline4168__rv_v5)
                else:
                    with pl.spmd(24, name_hint='indexer_score_topk_leaf_spmd', deps=[qh_quant_tid_inline579__ssa_v0, weights_tid_inline4157__ssa_v0, idx_cache_write_tid__ssa_v0], allow_early_resolve=True) as direct_leaf_tid_inline3236_inline4215__ssa_v0:
                        ret__tmp_v0_14: pl.Tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]] = self.indexer_score_topk_leaf(position_ids__ssa_v0, kv_seq_lens__ssa_v0, score_arena_inline3255_inline4168__ssa_v0, qr_hadamard_i8_inline583__rv_v2, qr_hadamard_scale_dq_inline582__rv_v2, weights_inline4197__ssa_v0, idx_block_table__ssa_v0, native_page_bytes_inline3220_inline4177__ssa_v0, idx_kv_cache_inline3234_inline4199__ssa_v0, idx_kv_cache_shift64_inline3230_inline4201__ssa_v0, idx_native_kv_cache__rv_v3, pair_arena_inline3226_inline4202__ssa_v0)
                        score_arena_inline3255_inline4168__rv_v2: pl.Tensor[[384, 8192], pl.FP32] = ret__tmp_v0_14[0]
                        pair_arena_inline3226_inline4202__ssa_v5: pl.Tensor[[24576, 1024], pl.FP32] = ret__tmp_v0_14[1]
                    score_tid_inline3240_inline4204__ssa_v7: pl.Scalar[pl.TASK_ID] = direct_leaf_tid_inline3236_inline4215__ssa_v0
                    (score_arena_inline3255_inline4168__phi_v8, score_tid_inline3240_inline4204__phi_v8, pair_arena_inline3226_inline4202__rv_v4, score_arena_inline3255_inline4168__rv_v6) = pl.yield_(score_arena_inline3255_inline4168__rv_v2, score_tid_inline3240_inline4204__ssa_v7, pair_arena_inline3226_inline4202__ssa_v5, score_arena_inline3255_inline4168__ssa_v0)
                with pl.scope():
                    if max_topk_cache_len_inline3209_inline4203__ssa_v0 < 2048:
                        with pl.spmd(48, name_hint='indexer_topk_single_leaf_publish', deps=[score_tid_inline3240_inline4204__phi_v8], allow_early_resolve=True):
                            self.indexer_topk_single_leaf_publish(position_ids__ssa_v0, kv_seq_lens__ssa_v0, score_arena_inline3255_inline4168__phi_v8, idx_topk_scores__ssa_v0, idx_topk__ssa_v0)
                    elif max_topk_cache_len_inline3209_inline4203__ssa_v0 <= 8192:
                        with pl.spmd(48, name_hint='indexer_topk_query_merge', deps=[score_tid_inline3240_inline4204__phi_v8], allow_early_resolve=True):
                            self.indexer_topk_query_merge(position_ids__ssa_v0, kv_seq_lens__ssa_v0, pair_arena_inline3226_inline4202__rv_v4, idx_topk_scores__ssa_v0, idx_topk__ssa_v0)
                    else:
                        with pl.spmd(48, name_hint='indexer_topk_query_merge', deps=[score_tid_inline3240_inline4204__phi_v8], allow_early_resolve=True):
                            self.indexer_topk_query_merge__2(position_ids__ssa_v0, kv_seq_lens__ssa_v0, pair_arena_inline3226_inline4202__rv_v4, idx_topk_scores__ssa_v0, idx_topk__ssa_v0)
                idx_topk_scores__ssa_v1: pl.Tensor[[T_DYN, 512], pl.FP32] = idx_topk_scores__ssa_v0
                idx_topk__ssa_v1: pl.Tensor[[T_DYN, 512], pl.INT32] = idx_topk__ssa_v0
                idx_topk_scores__ssa_v2: pl.Tensor[[T_DYN, 512], pl.FP32] = idx_topk_scores__ssa_v1
                idx_topk__ssa_v2: pl.Tensor[[T_DYN, 512], pl.INT32] = idx_topk__ssa_v1
                _index_scores__ssa_v0: pl.Tensor[[T_DYN, 512], pl.FP32] = idx_topk_scores__ssa_v2
                index_indices__ssa_v0: pl.Tensor[[T_DYN, 512], pl.INT32] = idx_topk__ssa_v2
            o_packed_heads__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16] = pl.tensor.create([3072, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            ori_block_num_inline4369__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv_cache__ssa_v0, 0)
            t_dim_inline4335__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q__ssa_v1, 0)
            t_heads_inline4306__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline4335__ssa_v0 * 64
            rope_cs_blocks_inline4315__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4335__ssa_v0 + 5) // 6
            ori_kv_flat_inline4326__ssa_v0: pl.Tensor[[ori_block_num_inline4369__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(kv_cache__ssa_v0, [ori_block_num_inline4369__ssa_v0 * 32, 512])
            ret__tmp_v0_15: pl.Tuple[pl.Tensor[[ori_block_num_inline4369__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.submit(self.kv_touch, ori_kv_flat_inline4326__ssa_v0, allow_early_resolve=True)
            ori_kv_flat_inline4326__ssa_v1: pl.Tensor[[ori_block_num_inline4369__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = ret__tmp_v0_15[0]
            tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_15[1]
            sparse_bias_inline4301__ssa_v0: pl.Tensor[[t_dim_inline4335__ssa_v0, 640], pl.FP32] = pl.tensor.create([t_dim_inline4335__ssa_v0, 640], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            cmp_sparse_indices_inline4329__ssa_v0: pl.Tensor[[t_dim_inline4335__ssa_v0, 512], pl.INT32] = pl.tensor.create([t_dim_inline4335__ssa_v0, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            valid_block_mask_inline4413__ssa_v0: pl.Tensor[[t_dim_inline4335__ssa_v0, 16], pl.INT32] = pl.tensor.create([t_dim_inline4335__ssa_v0, 16], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            with pl.spmd(16, name_hint='csa_slots_build_valid_qk_plan_spmd', allow_early_resolve=True) as qk_plan_tid_inline4317__ssa_v0:
                ret__tmp_v0_16: pl.Tuple[pl.Tensor[[t_dim_inline4335__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline4335__ssa_v0, 640], pl.FP32]] = self.csa_slots_build_valid_qk_plan(cmp_sparse_indices_inline4329__ssa_v0, sparse_bias_inline4301__ssa_v0, t_dim_inline4335__ssa_v0, index_indices__ssa_v0, position_ids_t1__ssa_v0, valid_block_mask_inline4413__ssa_v0, kv_seq_lens__ssa_v0, ori_block_table__ssa_v0)
                cmp_sparse_indices_inline4329__rv_v2: pl.Tensor[[t_dim_inline4335__ssa_v0, 512], pl.INT32] = ret__tmp_v0_16[0]
                sparse_bias_inline4301__rv_v2: pl.Tensor[[t_dim_inline4335__ssa_v0, 640], pl.FP32] = ret__tmp_v0_16[1]
            rope_sin_signed_inline4321__ssa_v0: pl.Tensor[[384, 64], pl.FP32] = pl.tensor.create([384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(pl.min(rope_cs_blocks_inline4315__ssa_v0, 16), name_hint='rope_cs_spmd', allow_early_resolve=True) as rope_tid_inline4397__ssa_v0:
                rope_sin_signed_inline4321__ssa_v1: pl.Tensor[[384, 64], pl.FP32] = self.rope_cs(rope_cs_blocks_inline4315__ssa_v0, t_dim_inline4335__ssa_v0, freqs_sin__ssa_v0, rope_sin_signed_inline4321__ssa_v0)
            cmp_block_num_inline4371__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
            cmp_kv_flat_inline4388__ssa_v0: pl.Tensor[[cmp_block_num_inline4371__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(cmp_kv__ssa_v0, [cmp_block_num_inline4371__ssa_v0 * 32, 512])
            q_flat_inline4389__ssa_v0: pl.Tensor[[t_heads_inline4306__ssa_v0, 512], pl.BF16] = pl.tensor.reshape(q__ssa_v1, [t_heads_inline4306__ssa_v0, 512])
            attn_sink_col_inline4382__ssa_v0: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.reshape(attn_sink__ssa_v0, [64, 1])
            kv_transfer_inline4342__ssa_v0: pl.Tensor[[9216, 512], pl.BF16] = pl.tensor.create([9216, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            score_transfer_inline4398__ssa_v0: pl.Tensor[[4608, 128], pl.FP32] = pl.tensor.create([4608, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            probability_transfer_inline4399__ssa_v0: pl.Tensor[[4608, 128], pl.BF16] = pl.tensor.create([4608, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            pv_transfer_inline4320__ssa_v0: pl.Tensor[[4608, 512], pl.FP32] = pl.tensor.create([4608, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            mi_transfer_inline4318__ssa_v0: pl.Tensor[[4608, 1], pl.FP32] = pl.tensor.create([4608, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            li_transfer_inline4403__ssa_v0: pl.Tensor[[4608, 1], pl.FP32] = pl.tensor.create([4608, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            ffts_workspace_inline4361__ssa_v0: pl.Tensor[[256], pl.INT64] = pl.tensor.create([256], dtype=pl.INT64, layout=pl.TensorLayout.ND)
            with pl.spmd(24, name_hint='qk_pv_spmd', deps=[qk_plan_tid_inline4317__ssa_v0, rope_tid_inline4397__ssa_v0], allow_early_resolve=True) as qk_tid_inline4404__ssa_v0:
                ret__tmp_v0_17: pl.Tuple[pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16]] = self.qk_pv(ffts_workspace_inline4361__ssa_v0, t_dim_inline4335__ssa_v0, q_flat_inline4389__ssa_v0, valid_block_mask_inline4413__ssa_v0, kv_transfer_inline4342__ssa_v0, score_transfer_inline4398__ssa_v0, probability_transfer_inline4399__ssa_v0, pv_transfer_inline4320__ssa_v0, attn_sink_col_inline4382__ssa_v0, position_ids_t1__ssa_v0, ori_block_table__ssa_v0, ori_kv_flat_inline4326__ssa_v1, cmp_sparse_indices_inline4329__rv_v2, cmp_block_table__ssa_v0, cmp_kv_flat_inline4388__ssa_v0, sparse_bias_inline4301__rv_v2, mi_transfer_inline4318__ssa_v0, li_transfer_inline4403__ssa_v0, freqs_cos__ssa_v0, rope_sin_signed_inline4321__ssa_v1, o_packed_heads__ssa_v0)
                kv_transfer_inline4342__ssa_v1: pl.Tensor[[9216, 512], pl.BF16] = ret__tmp_v0_17[0]
                score_transfer_inline4398__ssa_v1: pl.Tensor[[4608, 128], pl.FP32] = ret__tmp_v0_17[1]
                probability_transfer_inline4399__ssa_v1: pl.Tensor[[4608, 128], pl.BF16] = ret__tmp_v0_17[2]
                pv_transfer_inline4320__ssa_v1: pl.Tensor[[4608, 512], pl.FP32] = ret__tmp_v0_17[3]
                mi_transfer_inline4318__ssa_v1: pl.Tensor[[4608, 1], pl.FP32] = ret__tmp_v0_17[4]
                li_transfer_inline4403__ssa_v1: pl.Tensor[[4608, 1], pl.FP32] = ret__tmp_v0_17[5]
                o_packed_heads__ssa_v2: pl.Tensor[[3072, 4096], pl.BF16] = ret__tmp_v0_17[6]
            packed_inline585__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16] = o_packed_heads__ssa_v2
            ready_inline584__ssa_v0: pl.Scalar[pl.TASK_ID] = qk_tid_inline4404__ssa_v0
            o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16] = packed_inline585__ssa_v0
            heads_dep__ssa_v0: pl.Scalar[pl.TASK_ID] = ready_inline584__ssa_v0
            with pl.scope():
                t_dim_inline587__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_out__ssa_v0, 0)
                if t_dim_inline587__ssa_v0 <= 32:
                    t_dim_inline4518__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_out__ssa_v0, 0)
                    act_row_step_inline4515__ssa_v1: pl.Scalar[pl.INDEX] = 4
                    act_t_blks_inline4525__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4518__ssa_v0 + act_row_step_inline4515__ssa_v1 - 1) // act_row_step_inline4515__ssa_v1
                    proj_a_rows_inline4559__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4518__ssa_v0 + 127) // 128
                    proj_b_t_rows_inline4504__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4518__ssa_v0 + 31) // 32
                    proj_b_padded_rows_inline4526__ssa_v0: pl.Scalar[pl.INDEX] = proj_b_t_rows_inline4504__ssa_v0 * 32
                    o_r_pad_inline4514__ssa_v0: pl.Tensor[[384, 8192], pl.FP32] = pl.tensor.create([384, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    o_r_i8_pad_inline4534__ssa_v0: pl.Tensor[[384, 8192], pl.INT8] = pl.tensor.create([384, 8192], dtype=pl.INT8, layout=pl.TensorLayout.ND)
                    act_scale_dq_inline4531__ssa_v0: pl.Tensor[[8, 384], pl.FP32] = pl.tensor.create([8, 384], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    partials_inline4512__ssa_v0: pl.Tensor[[384, 32768], pl.INT32] = pl.tensor.create([384, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                    proj_b_tids_inline4535__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
                    with pl.scope(mode=pl.ScopeMode.MANUAL):
                        for (g_inline4502__idx_v0, (act_scale_dq_inline4531__iter_v1, o_r_i8_pad_inline4534__iter_v1, o_r_pad_inline4514__iter_v1, partials_inline4512__iter_v1, proj_b_tids_inline4535__iter_v1)) in pl.parallel(8, init_values=(act_scale_dq_inline4531__ssa_v0, o_r_i8_pad_inline4534__ssa_v0, o_r_pad_inline4514__ssa_v0, partials_inline4512__ssa_v0, proj_b_tids_inline4535__ssa_v0)):
                            row_base_o_inline4519__ssa_v0: pl.Scalar[pl.INDEX] = g_inline4502__idx_v0 * 384
                            out_col_g_inline4498__ssa_v0: pl.Scalar[pl.INDEX] = g_inline4502__idx_v0 * 1024
                            with pl.spmd(proj_a_rows_inline4559__ssa_v0 * 8, name_hint='proj_a_mm_spmd', deps=[heads_dep__ssa_v0], allow_early_resolve=True) as pa_tid_inline341_inline4530__ssa_v0:
                                stored_inline323_inline4542__ssa_v0: pl.Tensor[[384, 8192], pl.FP32] = self.proj_a_mm(t_dim_inline4518__ssa_v0, row_base_o_inline4519__ssa_v0, o_packed_heads__ssa_v1, wo_a__ssa_v0, g_inline4502__idx_v0, o_r_pad_inline4514__iter_v1, out_col_g_inline4498__ssa_v0)
                            o_r_pad_inline4514__ssa_v3: pl.Tensor[[384, 8192], pl.FP32] = stored_inline323_inline4542__ssa_v0
                            pa_tid_inline4543__ssa_v0: pl.Scalar[pl.TASK_ID] = pa_tid_inline341_inline4530__ssa_v0
                            col_g_inline4488__ssa_v0: pl.Scalar[pl.INDEX] = g_inline4502__idx_v0 * 1024
                            with pl.spmd((t_dim_inline4518__ssa_v0 + 31) // 32, name_hint='quant_spmd', deps=[pa_tid_inline4543__ssa_v0], allow_early_resolve=True) as q_tid_inline4547__ssa_v0:
                                ret__tmp_v0_18: pl.Tuple[pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8]] = self.quant(act_scale_dq_inline4531__iter_v1, o_r_i8_pad_inline4534__iter_v1, t_dim_inline4518__ssa_v0, o_r_pad_inline4514__ssa_v3, col_g_inline4488__ssa_v0, g_inline4502__idx_v0, proj_b_padded_rows_inline4526__ssa_v0)
                                act_scale_dq_inline4531__rv_v4: pl.Tensor[[8, 384], pl.FP32] = ret__tmp_v0_18[0]
                                o_r_i8_pad_inline4534__phi_v9: pl.Tensor[[384, 8192], pl.INT8] = ret__tmp_v0_18[1]
                            with pl.spmd(8, name_hint='proj_b_mm', deps=[q_tid_inline4547__ssa_v0], allow_early_resolve=True) as pb_tid_inline355_inline4470__ssa_v0:
                                partials_inline4512__ssa_v3: pl.Tensor[[384, 32768], pl.INT32] = self._proj_b_mm_nz_kernel(o_r_i8_pad_inline4534__phi_v9, wo_b__ssa_v0, partials_inline4512__iter_v1, g_inline4502__idx_v0, col_g_inline4488__ssa_v0, proj_b_t_rows_inline4504__ssa_v0)
                            partials_inline4512__ssa_v4: pl.Tensor[[384, 32768], pl.INT32] = partials_inline4512__ssa_v3
                            pb_tid_inline4523__ssa_v0: pl.Scalar[pl.TASK_ID] = pb_tid_inline355_inline4470__ssa_v0
                            proj_b_tids_inline4535__ssa_v3: pl.Array[8, pl.TASK_ID] = pl.array.update_element(proj_b_tids_inline4535__iter_v1, g_inline4502__idx_v0, pb_tid_inline4523__ssa_v0)
                            (act_scale_dq_inline4531__rv_v2, o_r_i8_pad_inline4534__rv_v2, o_r_pad_inline4514__rv_v2, partials_inline4512__rv_v2, proj_b_tids_inline4535__rv_v2) = pl.yield_(act_scale_dq_inline4531__rv_v4, o_r_i8_pad_inline4534__phi_v9, o_r_pad_inline4514__ssa_v3, partials_inline4512__ssa_v4, proj_b_tids_inline4535__ssa_v3)
                    residual_flat_inline4469__ssa_v0: pl.Tensor[[t_dim_inline4518__ssa_v0, 16384], pl.BF16] = pl.tensor.reshape(x_hc__ssa_v0, [t_dim_inline4518__ssa_v0, 16384])
                    y_flat_inline4468__ssa_v0: pl.Tensor[[t_dim_inline4518__ssa_v0, 16384], pl.BF16] = pl.tensor.reshape(x_out__ssa_v0, [t_dim_inline4518__ssa_v0, 16384])
                    _submit_deps_buf_inline4537__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
                    t__tmp_v842: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4535__rv_v2, 0)
                    _submit_deps_buf_inline4466__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4537__ssa_v0, 0, t__tmp_v842)
                    t__tmp_v843: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4535__rv_v2, 1)
                    _submit_deps_buf_inline4465__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4466__ssa_v0, 1, t__tmp_v843)
                    t__tmp_v844: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4535__rv_v2, 2)
                    _submit_deps_buf_inline4464__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4465__ssa_v0, 2, t__tmp_v844)
                    t__tmp_v845: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4535__rv_v2, 3)
                    _submit_deps_buf_inline4489__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4464__ssa_v0, 3, t__tmp_v845)
                    t__tmp_v846: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4535__rv_v2, 4)
                    _submit_deps_buf_inline4463__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4489__ssa_v0, 4, t__tmp_v846)
                    t__tmp_v847: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4535__rv_v2, 5)
                    _submit_deps_buf_inline4480__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4463__ssa_v0, 5, t__tmp_v847)
                    t__tmp_v848: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4535__rv_v2, 6)
                    _submit_deps_buf_inline4529__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4480__ssa_v0, 6, t__tmp_v848)
                    t__tmp_v849: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4535__rv_v2, 7)
                    _submit_deps_buf_inline4510__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4529__ssa_v0, 7, t__tmp_v849)
                    with pl.spmd(act_t_blks_inline4525__ssa_v0, name_hint='proj_b_act_hc_post_spmd', deps=[_submit_deps_buf_inline4510__ssa_v0], allow_early_resolve=True) as _act_tid_inline4493__ssa_v0:
                        self.proj_b_act_hc_post(act_row_step_inline4515__ssa_v1, wo_b_scale__ssa_v0, y_flat_inline4468__ssa_v0, t_dim_inline4518__ssa_v0, partials_inline4512__rv_v2, act_scale_dq_inline4531__rv_v2, residual_flat_inline4469__ssa_v0, post_t__rv_v2, comb_t__ssa_v1)
                elif t_dim_inline587__ssa_v0 == 96:
                    t_dim_inline4627__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_out__ssa_v0, 0)
                    act_row_step_inline4633__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4627__ssa_v0 + 47) // 48
                    act_t_blks_inline4652__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4627__ssa_v0 + act_row_step_inline4633__ssa_v0 - 1) // act_row_step_inline4633__ssa_v0
                    proj_a_rows_inline4621__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4627__ssa_v0 + 127) // 128
                    proj_b_t_rows_inline4612__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4627__ssa_v0 + 95) // 96
                    proj_b_padded_rows_inline4630__ssa_v0: pl.Scalar[pl.INDEX] = proj_b_t_rows_inline4612__ssa_v0 * 96
                    o_r_pad_inline4642__ssa_v0: pl.Tensor[[384, 8192], pl.FP32] = pl.tensor.create([384, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    o_r_i8_pad_inline4645__ssa_v0: pl.Tensor[[384, 8192], pl.INT8] = pl.tensor.create([384, 8192], dtype=pl.INT8, layout=pl.TensorLayout.ND)
                    act_scale_dq_inline4655__ssa_v0: pl.Tensor[[8, 384], pl.FP32] = pl.tensor.create([8, 384], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    partials_inline4641__ssa_v0: pl.Tensor[[384, 32768], pl.INT32] = pl.tensor.create([384, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                    proj_b_tids_inline4599__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
                    with pl.scope(mode=pl.ScopeMode.MANUAL):
                        for (g_inline4648__idx_v0, (act_scale_dq_inline4655__iter_v1, o_r_i8_pad_inline4645__iter_v1, o_r_pad_inline4642__iter_v1, partials_inline4641__iter_v1, proj_b_tids_inline4599__iter_v1)) in pl.parallel(8, init_values=(act_scale_dq_inline4655__ssa_v0, o_r_i8_pad_inline4645__ssa_v0, o_r_pad_inline4642__ssa_v0, partials_inline4641__ssa_v0, proj_b_tids_inline4599__ssa_v0)):
                            row_base_o_inline4605__ssa_v0: pl.Scalar[pl.INDEX] = g_inline4648__idx_v0 * 384
                            out_col_g_inline4626__ssa_v0: pl.Scalar[pl.INDEX] = g_inline4648__idx_v0 * 1024
                            with pl.spmd(proj_a_rows_inline4621__ssa_v0 * 8, name_hint='proj_a_mm_spmd', deps=[heads_dep__ssa_v0], allow_early_resolve=True) as pa_tid_inline606_inline4635__ssa_v0:
                                stored_inline588_inline4607__ssa_v0: pl.Tensor[[384, 8192], pl.FP32] = self.proj_a_mm_0(t_dim_inline4627__ssa_v0, row_base_o_inline4605__ssa_v0, o_packed_heads__ssa_v1, wo_a__ssa_v0, g_inline4648__idx_v0, o_r_pad_inline4642__iter_v1, out_col_g_inline4626__ssa_v0)
                            o_r_pad_inline4642__ssa_v3: pl.Tensor[[384, 8192], pl.FP32] = stored_inline588_inline4607__ssa_v0
                            pa_tid_inline4647__ssa_v0: pl.Scalar[pl.TASK_ID] = pa_tid_inline606_inline4635__ssa_v0
                            col_g_inline4637__ssa_v0: pl.Scalar[pl.INDEX] = g_inline4648__idx_v0 * 1024
                            with pl.spmd((t_dim_inline4627__ssa_v0 + 31) // 32, name_hint='quant_spmd', deps=[pa_tid_inline4647__ssa_v0], allow_early_resolve=True) as q_tid_inline4674__ssa_v0:
                                ret__tmp_v0_19: pl.Tuple[pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8]] = self.quant_0(act_scale_dq_inline4655__iter_v1, o_r_i8_pad_inline4645__iter_v1, t_dim_inline4627__ssa_v0, o_r_pad_inline4642__ssa_v3, col_g_inline4637__ssa_v0, g_inline4648__idx_v0, proj_b_padded_rows_inline4630__ssa_v0)
                                act_scale_dq_inline4655__rv_v4: pl.Tensor[[8, 384], pl.FP32] = ret__tmp_v0_19[0]
                                o_r_i8_pad_inline4645__phi_v9: pl.Tensor[[384, 8192], pl.INT8] = ret__tmp_v0_19[1]
                            with pl.spmd(8, name_hint='proj_b_mm', deps=[q_tid_inline4674__ssa_v0], allow_early_resolve=True) as pb_tid_inline620_inline4589__ssa_v0:
                                partials_inline4641__ssa_v3: pl.Tensor[[384, 32768], pl.INT32] = self._proj_b_mm_nz_kernel__2(o_r_i8_pad_inline4645__phi_v9, wo_b__ssa_v0, partials_inline4641__iter_v1, g_inline4648__idx_v0, col_g_inline4637__ssa_v0, proj_b_t_rows_inline4612__ssa_v0)
                            partials_inline4641__ssa_v4: pl.Tensor[[384, 32768], pl.INT32] = partials_inline4641__ssa_v3
                            pb_tid_inline4666__ssa_v0: pl.Scalar[pl.TASK_ID] = pb_tid_inline620_inline4589__ssa_v0
                            proj_b_tids_inline4599__ssa_v3: pl.Array[8, pl.TASK_ID] = pl.array.update_element(proj_b_tids_inline4599__iter_v1, g_inline4648__idx_v0, pb_tid_inline4666__ssa_v0)
                            (act_scale_dq_inline4655__rv_v2, o_r_i8_pad_inline4645__rv_v2, o_r_pad_inline4642__rv_v2, partials_inline4641__rv_v2, proj_b_tids_inline4599__rv_v2) = pl.yield_(act_scale_dq_inline4655__rv_v4, o_r_i8_pad_inline4645__phi_v9, o_r_pad_inline4642__ssa_v3, partials_inline4641__ssa_v4, proj_b_tids_inline4599__ssa_v3)
                    residual_flat_inline4663__ssa_v0: pl.Tensor[[t_dim_inline4627__ssa_v0, 16384], pl.BF16] = pl.tensor.reshape(x_hc__ssa_v0, [t_dim_inline4627__ssa_v0, 16384])
                    y_flat_inline4588__ssa_v0: pl.Tensor[[t_dim_inline4627__ssa_v0, 16384], pl.BF16] = pl.tensor.reshape(x_out__ssa_v0, [t_dim_inline4627__ssa_v0, 16384])
                    _submit_deps_buf_inline4654__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
                    t__tmp_v864: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4599__rv_v2, 0)
                    _submit_deps_buf_inline4587__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4654__ssa_v0, 0, t__tmp_v864)
                    t__tmp_v865: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4599__rv_v2, 1)
                    _submit_deps_buf_inline4675__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4587__ssa_v0, 1, t__tmp_v865)
                    t__tmp_v866: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4599__rv_v2, 2)
                    _submit_deps_buf_inline4585__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4675__ssa_v0, 2, t__tmp_v866)
                    t__tmp_v867: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4599__rv_v2, 3)
                    _submit_deps_buf_inline4629__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4585__ssa_v0, 3, t__tmp_v867)
                    t__tmp_v868: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4599__rv_v2, 4)
                    _submit_deps_buf_inline4662__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4629__ssa_v0, 4, t__tmp_v868)
                    t__tmp_v869: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4599__rv_v2, 5)
                    _submit_deps_buf_inline4624__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4662__ssa_v0, 5, t__tmp_v869)
                    t__tmp_v870: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4599__rv_v2, 6)
                    _submit_deps_buf_inline4584__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4624__ssa_v0, 6, t__tmp_v870)
                    t__tmp_v871: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4599__rv_v2, 7)
                    _submit_deps_buf_inline4582__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4584__ssa_v0, 7, t__tmp_v871)
                    with pl.spmd(act_t_blks_inline4652__ssa_v0, sync_start=True, name_hint='proj_b_act_hc_post_spmd', deps=[_submit_deps_buf_inline4582__ssa_v0], allow_early_resolve=True) as _act_tid_inline4617__ssa_v0:
                        self.proj_b_act_hc_post_0(act_row_step_inline4633__ssa_v0, wo_b_scale__ssa_v0, y_flat_inline4588__ssa_v0, t_dim_inline4627__ssa_v0, partials_inline4641__rv_v2, act_scale_dq_inline4655__rv_v2, residual_flat_inline4663__ssa_v0, post_t__rv_v2, comb_t__ssa_v1)
                elif t_dim_inline587__ssa_v0 <= 96:
                    t_dim_inline4745__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_out__ssa_v0, 0)
                    act_row_step_inline4746__ssa_v1: pl.Scalar[pl.INDEX] = 4
                    act_t_blks_inline4762__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4745__ssa_v0 + act_row_step_inline4746__ssa_v1 - 1) // act_row_step_inline4746__ssa_v1
                    proj_a_rows_inline4756__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4745__ssa_v0 + 127) // 128
                    proj_b_t_rows_inline4750__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4745__ssa_v0 + 95) // 96
                    proj_b_padded_rows_inline4780__ssa_v0: pl.Scalar[pl.INDEX] = proj_b_t_rows_inline4750__ssa_v0 * 96
                    o_r_pad_inline4770__ssa_v0: pl.Tensor[[384, 8192], pl.FP32] = pl.tensor.create([384, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    o_r_i8_pad_inline4738__ssa_v0: pl.Tensor[[384, 8192], pl.INT8] = pl.tensor.create([384, 8192], dtype=pl.INT8, layout=pl.TensorLayout.ND)
                    act_scale_dq_inline4765__ssa_v0: pl.Tensor[[8, 384], pl.FP32] = pl.tensor.create([8, 384], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    partials_inline4759__ssa_v0: pl.Tensor[[384, 32768], pl.INT32] = pl.tensor.create([384, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                    proj_b_tids_inline4768__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
                    with pl.scope(mode=pl.ScopeMode.MANUAL):
                        for (g_inline4747__idx_v0, (act_scale_dq_inline4765__iter_v1, o_r_i8_pad_inline4738__iter_v1, o_r_pad_inline4770__iter_v1, partials_inline4759__iter_v1, proj_b_tids_inline4768__iter_v1)) in pl.parallel(8, init_values=(act_scale_dq_inline4765__ssa_v0, o_r_i8_pad_inline4738__ssa_v0, o_r_pad_inline4770__ssa_v0, partials_inline4759__ssa_v0, proj_b_tids_inline4768__ssa_v0)):
                            row_base_o_inline4772__ssa_v0: pl.Scalar[pl.INDEX] = g_inline4747__idx_v0 * 384
                            out_col_g_inline4735__ssa_v0: pl.Scalar[pl.INDEX] = g_inline4747__idx_v0 * 1024
                            with pl.spmd(proj_a_rows_inline4756__ssa_v0 * 8, name_hint='proj_a_mm_spmd', deps=[heads_dep__ssa_v0], allow_early_resolve=True) as pa_tid_inline1244_inline4720__ssa_v0:
                                stored_inline1226_inline4785__ssa_v0: pl.Tensor[[384, 8192], pl.FP32] = self.proj_a_mm_1(t_dim_inline4745__ssa_v0, row_base_o_inline4772__ssa_v0, o_packed_heads__ssa_v1, wo_a__ssa_v0, g_inline4747__idx_v0, o_r_pad_inline4770__iter_v1, out_col_g_inline4735__ssa_v0)
                            o_r_pad_inline4770__ssa_v3: pl.Tensor[[384, 8192], pl.FP32] = stored_inline1226_inline4785__ssa_v0
                            pa_tid_inline4789__ssa_v0: pl.Scalar[pl.TASK_ID] = pa_tid_inline1244_inline4720__ssa_v0
                            col_g_inline4791__ssa_v0: pl.Scalar[pl.INDEX] = g_inline4747__idx_v0 * 1024
                            with pl.spmd((t_dim_inline4745__ssa_v0 + 31) // 32, name_hint='quant_spmd', deps=[pa_tid_inline4789__ssa_v0], allow_early_resolve=True) as q_tid_inline4793__ssa_v0:
                                ret__tmp_v0_20: pl.Tuple[pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8]] = self.quant_1(act_scale_dq_inline4765__iter_v1, o_r_i8_pad_inline4738__iter_v1, t_dim_inline4745__ssa_v0, o_r_pad_inline4770__ssa_v3, col_g_inline4791__ssa_v0, g_inline4747__idx_v0, proj_b_padded_rows_inline4780__ssa_v0)
                                act_scale_dq_inline4765__rv_v4: pl.Tensor[[8, 384], pl.FP32] = ret__tmp_v0_20[0]
                                o_r_i8_pad_inline4738__phi_v9: pl.Tensor[[384, 8192], pl.INT8] = ret__tmp_v0_20[1]
                            with pl.spmd(8, name_hint='proj_b_mm', deps=[q_tid_inline4793__ssa_v0], allow_early_resolve=True) as pb_tid_inline1258_inline4704__ssa_v0:
                                partials_inline4759__ssa_v3: pl.Tensor[[384, 32768], pl.INT32] = self._proj_b_mm_nz_kernel__2(o_r_i8_pad_inline4738__phi_v9, wo_b__ssa_v0, partials_inline4759__iter_v1, g_inline4747__idx_v0, col_g_inline4791__ssa_v0, proj_b_t_rows_inline4750__ssa_v0)
                            partials_inline4759__ssa_v4: pl.Tensor[[384, 32768], pl.INT32] = partials_inline4759__ssa_v3
                            pb_tid_inline4715__ssa_v0: pl.Scalar[pl.TASK_ID] = pb_tid_inline1258_inline4704__ssa_v0
                            proj_b_tids_inline4768__ssa_v3: pl.Array[8, pl.TASK_ID] = pl.array.update_element(proj_b_tids_inline4768__iter_v1, g_inline4747__idx_v0, pb_tid_inline4715__ssa_v0)
                            (act_scale_dq_inline4765__rv_v2, o_r_i8_pad_inline4738__rv_v2, o_r_pad_inline4770__rv_v2, partials_inline4759__rv_v2, proj_b_tids_inline4768__rv_v2) = pl.yield_(act_scale_dq_inline4765__rv_v4, o_r_i8_pad_inline4738__phi_v9, o_r_pad_inline4770__ssa_v3, partials_inline4759__ssa_v4, proj_b_tids_inline4768__ssa_v3)
                    residual_flat_inline4703__ssa_v0: pl.Tensor[[t_dim_inline4745__ssa_v0, 16384], pl.BF16] = pl.tensor.reshape(x_hc__ssa_v0, [t_dim_inline4745__ssa_v0, 16384])
                    y_flat_inline4761__ssa_v0: pl.Tensor[[t_dim_inline4745__ssa_v0, 16384], pl.BF16] = pl.tensor.reshape(x_out__ssa_v0, [t_dim_inline4745__ssa_v0, 16384])
                    _submit_deps_buf_inline4702__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
                    t__tmp_v886: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4768__rv_v2, 0)
                    _submit_deps_buf_inline4701__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4702__ssa_v0, 0, t__tmp_v886)
                    t__tmp_v887: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4768__rv_v2, 1)
                    _submit_deps_buf_inline4700__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4701__ssa_v0, 1, t__tmp_v887)
                    t__tmp_v888: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4768__rv_v2, 2)
                    _submit_deps_buf_inline4779__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4700__ssa_v0, 2, t__tmp_v888)
                    t__tmp_v889: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4768__rv_v2, 3)
                    _submit_deps_buf_inline4697__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4779__ssa_v0, 3, t__tmp_v889)
                    t__tmp_v890: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4768__rv_v2, 4)
                    _submit_deps_buf_inline4771__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4697__ssa_v0, 4, t__tmp_v890)
                    t__tmp_v891: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4768__rv_v2, 5)
                    _submit_deps_buf_inline4695__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4771__ssa_v0, 5, t__tmp_v891)
                    t__tmp_v892: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4768__rv_v2, 6)
                    _submit_deps_buf_inline4799__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4695__ssa_v0, 6, t__tmp_v892)
                    t__tmp_v893: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4768__rv_v2, 7)
                    _submit_deps_buf_inline4694__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4799__ssa_v0, 7, t__tmp_v893)
                    with pl.spmd(act_t_blks_inline4762__ssa_v0, name_hint='proj_b_act_hc_post_spmd', deps=[_submit_deps_buf_inline4694__ssa_v0], allow_early_resolve=True) as _act_tid_inline4693__ssa_v0:
                        self.proj_b_act_hc_post_1(act_row_step_inline4746__ssa_v1, wo_b_scale__ssa_v0, y_flat_inline4761__ssa_v0, t_dim_inline4745__ssa_v0, partials_inline4759__rv_v2, act_scale_dq_inline4765__rv_v2, residual_flat_inline4703__ssa_v0, post_t__rv_v2, comb_t__ssa_v1)
                else:
                    t_dim_inline4891__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_out__ssa_v0, 0)
                    act_row_step_inline4887__ssa_v1: pl.Scalar[pl.INDEX] = 4
                    act_t_blks_inline4889__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4891__ssa_v0 + act_row_step_inline4887__ssa_v1 - 1) // act_row_step_inline4887__ssa_v1
                    proj_a_rows_inline4848__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4891__ssa_v0 + 127) // 128
                    proj_b_t_rows_inline4877__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline4891__ssa_v0 + 127) // 128
                    proj_b_padded_rows_inline4872__ssa_v0: pl.Scalar[pl.INDEX] = proj_b_t_rows_inline4877__ssa_v0 * 128
                    o_r_pad_inline4871__ssa_v0: pl.Tensor[[384, 8192], pl.FP32] = pl.tensor.create([384, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    o_r_i8_pad_inline4874__ssa_v0: pl.Tensor[[384, 8192], pl.INT8] = pl.tensor.create([384, 8192], dtype=pl.INT8, layout=pl.TensorLayout.ND)
                    act_scale_dq_inline4892__ssa_v0: pl.Tensor[[8, 384], pl.FP32] = pl.tensor.create([8, 384], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    partials_inline4880__ssa_v0: pl.Tensor[[384, 32768], pl.INT32] = pl.tensor.create([384, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                    proj_b_tids_inline4893__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
                    with pl.scope(mode=pl.ScopeMode.MANUAL):
                        for (g_inline4868__idx_v0, (act_scale_dq_inline4892__iter_v1, o_r_i8_pad_inline4874__iter_v1, o_r_pad_inline4871__iter_v1, partials_inline4880__iter_v1, proj_b_tids_inline4893__iter_v1)) in pl.parallel(8, init_values=(act_scale_dq_inline4892__ssa_v0, o_r_i8_pad_inline4874__ssa_v0, o_r_pad_inline4871__ssa_v0, partials_inline4880__ssa_v0, proj_b_tids_inline4893__ssa_v0)):
                            row_base_o_inline4886__ssa_v0: pl.Scalar[pl.INDEX] = g_inline4868__idx_v0 * 384
                            out_col_g_inline4870__ssa_v0: pl.Scalar[pl.INDEX] = g_inline4868__idx_v0 * 1024
                            with pl.spmd(proj_a_rows_inline4848__ssa_v0 * 4, name_hint='proj_a_mm_spmd', deps=[heads_dep__ssa_v0], allow_early_resolve=True) as pa_tid_inline660_inline4873__ssa_v0:
                                stored_inline668_inline4904__ssa_v0: pl.Tensor[[384, 8192], pl.FP32] = self.proj_a_mm_2(t_dim_inline4891__ssa_v0, row_base_o_inline4886__ssa_v0, o_packed_heads__ssa_v1, wo_a__ssa_v0, g_inline4868__idx_v0, o_r_pad_inline4871__iter_v1, out_col_g_inline4870__ssa_v0)
                            o_r_pad_inline4871__ssa_v3: pl.Tensor[[384, 8192], pl.FP32] = stored_inline668_inline4904__ssa_v0
                            pa_tid_inline4906__ssa_v0: pl.Scalar[pl.TASK_ID] = pa_tid_inline660_inline4873__ssa_v0
                            col_g_inline4865__ssa_v0: pl.Scalar[pl.INDEX] = g_inline4868__idx_v0 * 1024
                            with pl.spmd((t_dim_inline4891__ssa_v0 + 31) // 32, name_hint='quant_spmd', deps=[pa_tid_inline4906__ssa_v0], allow_early_resolve=True) as q_tid_inline4910__ssa_v0:
                                ret__tmp_v0_21: pl.Tuple[pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8]] = self.quant_2(act_scale_dq_inline4892__iter_v1, o_r_i8_pad_inline4874__iter_v1, t_dim_inline4891__ssa_v0, o_r_pad_inline4871__ssa_v3, col_g_inline4865__ssa_v0, g_inline4868__idx_v0, proj_b_padded_rows_inline4872__ssa_v0)
                                act_scale_dq_inline4892__rv_v4: pl.Tensor[[8, 384], pl.FP32] = ret__tmp_v0_21[0]
                                o_r_i8_pad_inline4874__phi_v9: pl.Tensor[[384, 8192], pl.INT8] = ret__tmp_v0_21[1]
                            with pl.spmd(8, name_hint='proj_b_mm', deps=[q_tid_inline4910__ssa_v0], allow_early_resolve=True) as pb_tid_inline674_inline4837__ssa_v0:
                                partials_inline4880__ssa_v3: pl.Tensor[[384, 32768], pl.INT32] = self._proj_b_mm_nz_kernel__3(o_r_i8_pad_inline4874__phi_v9, wo_b__ssa_v0, partials_inline4880__iter_v1, g_inline4868__idx_v0, col_g_inline4865__ssa_v0, proj_b_t_rows_inline4877__ssa_v0)
                            partials_inline4880__ssa_v4: pl.Tensor[[384, 32768], pl.INT32] = partials_inline4880__ssa_v3
                            pb_tid_inline4835__ssa_v0: pl.Scalar[pl.TASK_ID] = pb_tid_inline674_inline4837__ssa_v0
                            proj_b_tids_inline4893__ssa_v3: pl.Array[8, pl.TASK_ID] = pl.array.update_element(proj_b_tids_inline4893__iter_v1, g_inline4868__idx_v0, pb_tid_inline4835__ssa_v0)
                            (act_scale_dq_inline4892__rv_v2, o_r_i8_pad_inline4874__rv_v2, o_r_pad_inline4871__rv_v2, partials_inline4880__rv_v2, proj_b_tids_inline4893__rv_v2) = pl.yield_(act_scale_dq_inline4892__rv_v4, o_r_i8_pad_inline4874__phi_v9, o_r_pad_inline4871__ssa_v3, partials_inline4880__ssa_v4, proj_b_tids_inline4893__ssa_v3)
                    residual_flat_inline4834__ssa_v0: pl.Tensor[[t_dim_inline4891__ssa_v0, 16384], pl.BF16] = pl.tensor.reshape(x_hc__ssa_v0, [t_dim_inline4891__ssa_v0, 16384])
                    y_flat_inline4862__ssa_v0: pl.Tensor[[t_dim_inline4891__ssa_v0, 16384], pl.BF16] = pl.tensor.reshape(x_out__ssa_v0, [t_dim_inline4891__ssa_v0, 16384])
                    _submit_deps_buf_inline4861__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
                    t__tmp_v908: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4893__rv_v2, 0)
                    _submit_deps_buf_inline4833__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4861__ssa_v0, 0, t__tmp_v908)
                    t__tmp_v909: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4893__rv_v2, 1)
                    _submit_deps_buf_inline4852__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4833__ssa_v0, 1, t__tmp_v909)
                    t__tmp_v910: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4893__rv_v2, 2)
                    _submit_deps_buf_inline4831__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4852__ssa_v0, 2, t__tmp_v910)
                    t__tmp_v911: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4893__rv_v2, 3)
                    _submit_deps_buf_inline4829__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4831__ssa_v0, 3, t__tmp_v911)
                    t__tmp_v912: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4893__rv_v2, 4)
                    _submit_deps_buf_inline4827__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4829__ssa_v0, 4, t__tmp_v912)
                    t__tmp_v913: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4893__rv_v2, 5)
                    _submit_deps_buf_inline4825__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4827__ssa_v0, 5, t__tmp_v913)
                    t__tmp_v914: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4893__rv_v2, 6)
                    _submit_deps_buf_inline4824__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4825__ssa_v0, 6, t__tmp_v914)
                    t__tmp_v915: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline4893__rv_v2, 7)
                    _submit_deps_buf_inline4823__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4824__ssa_v0, 7, t__tmp_v915)
                    with pl.spmd(act_t_blks_inline4889__ssa_v0, name_hint='proj_b_act_hc_post_spmd', deps=[_submit_deps_buf_inline4823__ssa_v0], allow_early_resolve=True) as _act_tid_inline4821__ssa_v0:
                        self.proj_b_act_hc_post_2(act_row_step_inline4887__ssa_v1, wo_b_scale__ssa_v0, y_flat_inline4862__ssa_v0, t_dim_inline4891__ssa_v0, partials_inline4880__rv_v2, act_scale_dq_inline4892__rv_v2, residual_flat_inline4834__ssa_v0, post_t__rv_v2, comb_t__ssa_v1)
                layer_output__ssa_v0: pl.Tensor[[T_DYN, 4, 4096], pl.BF16] = x_out__ssa_v0
        return layer_output__ssa_v0
