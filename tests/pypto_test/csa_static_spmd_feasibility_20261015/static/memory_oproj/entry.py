# 只供 CPU 编译的两个 AIC 阶段分配诊断；不允许单独上卡。
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

@pl.program
class CSAAllocatorFocusedProbe:

    @pl.function(type=pl.FunctionType.InCore)
    def csa_static_aic(self, temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], arg_wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], temporary_63: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], records: pl.Tensor[[31968], pl.INT64]) -> tuple[pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32]]:
        pl.func_attr({'mx_tensor_views_blocked': True})
        group_id = pl.tile.get_block_idx()
        core_id = group_id
        record_15 = 6480 + core_id * 6
        mapped_block_15 = pl.cast(pl.tensor.read(records, [record_15]), pl.INDEX)
        block_15 = pl.max(mapped_block_15, 0)
        if mapped_block_15 >= 0:
            subslot_15 = pl.cast(pl.tensor.read(records, [record_15 + 1]), pl.INDEX)
            stage_15_v4: pl.Scalar[pl.INDEX] = block_15
            stage_15_v5: pl.Scalar[pl.INDEX] = stage_15_v4 // 8
            stage_15_v6: pl.Scalar[pl.INDEX] = stage_15_v4 % 8
            stage_15_v7: pl.Scalar[pl.INDEX] = stage_15_v5 * 128
            stage_15_v8: pl.Scalar[pl.INDEX] = pl.min(24 - stage_15_v7, 128)
            stage_15_v9: pl.Scalar[pl.INDEX] = 2688 + stage_15_v7
            stage_15_v10: pl.Scalar[pl.INDEX] = stage_15_v6 * 128
            stage_15_v11: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[stage_15_v8, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(temporary_51, [128, 256], [stage_15_v9, 0], [stage_15_v8, 256])
            stage_15_v12: pl.Tensor[[1, 256, 128], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(arg_wo_a, [1, 256, 128], [7, 0, stage_15_v10])
            stage_15_v13: pl.Tensor[[1, 128, 128], pl.FP32] = pl.tensor.matmul(stage_15_v11, stage_15_v12, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
            for (stage_15_v15, (stage_15_v21,)) in pl.pipeline(1, 16, stage=2, init_values=(stage_15_v13,)):
                stage_15_v16: pl.Scalar[pl.INDEX] = stage_15_v15 * 256
                stage_15_v17: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[stage_15_v8, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(temporary_51, [128, 256], [stage_15_v9, stage_15_v16], [stage_15_v8, 256])
                stage_15_v18: pl.Tensor[[1, 256, 128], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(arg_wo_a, [1, 256, 128], [7, stage_15_v16, stage_15_v10])
                stage_15_v19: pl.Tensor[[1, 128, 128], pl.FP32] = pl.tensor.matmul_acc(stage_15_v21, stage_15_v17, stage_15_v18, a_trans=False, b_trans=False)
                stage_15_v20: pl.Tensor[[1, 128, 128], pl.FP32] = pl.yield_(stage_15_v19)
            stage_15_v14: pl.Tensor[[384, 8192], pl.FP32] = pl.tensor.assemble(temporary_63, stage_15_v20, [stage_15_v7, 7168 + stage_15_v10])
            pl.system.fence()
            pl.system.cacheinvalid()
            pl.system.fence()
        return (temporary_51, temporary_63)

    @pl.function(type=pl.FunctionType.Orchestration)
    def entry(self, temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], arg_wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)], temporary_63: pl.InOut[pl.Tensor[[384, 8192], pl.FP32]], records: pl.Tensor[[31968], pl.INT64]) -> tuple[pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32]]:
        pl.func_attr({'mx_tensor_views_blocked': True})
        with pl.spmd(24, sync_start=True, name_hint='csa_static_full_card'):
            (temporary_51, temporary_63) = self.csa_static_aic(temporary_51, arg_wo_a, temporary_63, records)
        return (temporary_51, temporary_63)
