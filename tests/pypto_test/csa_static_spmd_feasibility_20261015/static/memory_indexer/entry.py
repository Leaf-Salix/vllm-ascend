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
    def csa_static_aic(self, arg_kv_seq_lens: pl.Tensor[[4], pl.INT32], arg_position_ids: pl.Tensor[[24], pl.INT64], temporary_39: pl.InOut[pl.Tensor[[24576, 128], pl.INT8]], temporary_48: pl.InOut[pl.Tensor[[1024, 384], pl.FP16]], temporary_50: pl.InOut[pl.Tensor[[256], pl.INT64]], arg_idx_block_table: pl.Tensor[[4, 1026], pl.INT32], view_98: pl.InOut[pl.Tensor[[133412, 128], pl.INT8]], view_99: pl.InOut[pl.Tensor[[133412, 128], pl.INT8]], temporary_49: pl.InOut[pl.Tensor[[288, 1024], pl.FP32]], records: pl.Tensor[[31968], pl.INT64]) -> tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[1024, 384], pl.FP16], pl.Tensor[[256], pl.INT64], pl.Tensor[[133412, 128], pl.INT8], pl.Tensor[[133412, 128], pl.INT8], pl.Tensor[[288, 1024], pl.FP32]]:
        pl.func_attr({'mx_tensor_views_blocked': True})
        group_id = pl.tile.get_block_idx()
        core_id = group_id
        record_13 = 5616 + core_id * 6
        mapped_block_13 = pl.cast(pl.tensor.read(records, [record_13]), pl.INDEX)
        block_13 = pl.max(mapped_block_13, 0)
        if mapped_block_13 >= 0:
            subslot_13 = pl.cast(pl.tensor.read(records, [record_13 + 1]), pl.INDEX)
            stage_13_v12: pl.Scalar[pl.INDEX] = block_13
            stage_13_v13: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids, 0)
            stage_13_v14: pl.Scalar[pl.INDEX] = 0
            for (stage_13_v25, (stage_13_v39,)) in pl.range(stage_13_v13 // 6, init_values=(stage_13_v14,)):
                stage_13_v26: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_13_v25])
                stage_13_v27: pl.Scalar[pl.INDEX] = pl.max(stage_13_v39, pl.cast(stage_13_v26, pl.INDEX) // 4)
                stage_13_v28: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v27)
            stage_13_v15: pl.Scalar[pl.INDEX] = pl.min(stage_13_v28, 262144)
            stage_13_v16: pl.Scalar[pl.INDEX] = (stage_13_v15 + 1023) // 1024
            stage_13_v17: pl.Scalar[pl.INDEX] = pl.max((stage_13_v15 + 8191) // 8192, 1)
            stage_13_v18: pl.Scalar[pl.INDEX] = 8
            stage_13_v19: pl.Scalar[pl.INDEX] = 0
            stage_13_v20: pl.Scalar[pl.INDEX] = 1
            if stage_13_v13 // 6 * 3 == 48:
                stage_13_v29: pl.Scalar[pl.INDEX] = 3
                stage_13_v30: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v29)
            else:
                if stage_13_v13 // 6 * 3 == 24:
                    stage_13_v40: pl.Scalar[pl.INDEX] = 3
                    stage_13_v41: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v40)
                else:
                    if stage_13_v13 // 6 * 6 == 24:
                        stage_13_v62: pl.Scalar[pl.INDEX] = 6
                        stage_13_v63: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v62)
                    else:
                        stage_13_v63: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v20)
                    stage_13_v41: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v63)
                stage_13_v30: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v41)
            if 1 < stage_13_v30:
                stage_13_v31: pl.Scalar[pl.INDEX] = (stage_13_v17 + stage_13_v30 - 1) // stage_13_v30 * stage_13_v30
                if stage_13_v31 <= 32:
                    stage_13_v42: pl.Scalar[pl.INDEX] = stage_13_v31
                    stage_13_v43: pl.Scalar[pl.INDEX] = pl.max(stage_13_v16 // stage_13_v31, 1)
                    stage_13_v44: pl.Scalar[pl.INDEX] = stage_13_v16 % stage_13_v31
                    (stage_13_v64, stage_13_v65, stage_13_v66) = pl.yield_(stage_13_v44, stage_13_v42, stage_13_v43)
                else:
                    (stage_13_v64, stage_13_v65, stage_13_v66) = pl.yield_(stage_13_v19, stage_13_v17, stage_13_v18)
                (stage_13_v45, stage_13_v46, stage_13_v47) = pl.yield_(stage_13_v64, stage_13_v65, stage_13_v66)
            else:
                (stage_13_v45, stage_13_v46, stage_13_v47) = pl.yield_(stage_13_v19, stage_13_v17, stage_13_v18)
            stage_13_v21: pl.Scalar[pl.INDEX] = stage_13_v46
            stage_13_v22: pl.Scalar[pl.INDEX] = stage_13_v47
            stage_13_v23: pl.Scalar[pl.INDEX] = stage_13_v45
            pl.system.set_ffts(temporary_50)
            for stage_13_v24 in pl.range(stage_13_v12, stage_13_v13 // 6 * stage_13_v21, 24):
                if stage_13_v13 < 144:
                    stage_13_v48: pl.Scalar[pl.INDEX] = stage_13_v24 % (stage_13_v13 // 6) * 6
                    stage_13_v49: pl.Scalar[pl.INDEX] = stage_13_v24 // (stage_13_v13 // 6)
                    (stage_13_v67, stage_13_v68) = pl.yield_(stage_13_v49, stage_13_v48)
                else:
                    stage_13_v50: pl.Scalar[pl.INDEX] = stage_13_v24 // stage_13_v21 * 6
                    stage_13_v51: pl.Scalar[pl.INDEX] = stage_13_v24 % stage_13_v21
                    (stage_13_v67, stage_13_v68) = pl.yield_(stage_13_v51, stage_13_v50)
                stage_13_v32: pl.Scalar[pl.INDEX] = stage_13_v68 // 6
                stage_13_v33: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids, [stage_13_v68 + 5])
                stage_13_v34: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens, [stage_13_v32])
                stage_13_v35: pl.Scalar[pl.INDEX] = pl.cast(stage_13_v34, pl.INDEX) // 4
                stage_13_v36: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_13_v35, (stage_13_v33 + 1) // 4), 262144), 0)
                stage_13_v37: pl.Scalar[pl.INDEX] = (stage_13_v67 * stage_13_v22 + pl.min(stage_13_v67, stage_13_v23)) * 1024
                stage_13_v38: pl.Scalar[pl.INDEX] = (stage_13_v22 + pl.cast(stage_13_v67 < stage_13_v23, pl.INDEX)) * 1024
                if stage_13_v37 < stage_13_v36:
                    stage_13_v52: pl.Scalar[pl.INDEX] = pl.min(stage_13_v38, stage_13_v36 - stage_13_v37)
                    stage_13_v53: pl.Scalar[pl.INDEX] = (stage_13_v52 + 1023) // 1024
                    stage_13_v54: pl.Scalar[pl.INDEX] = pl.min(stage_13_v53 * 512, 4096)
                    stage_13_v55: pl.Scalar[pl.INDEX] = (stage_13_v54 + 511) // 512
                    stage_13_v56: pl.Tile[[384, 128], pl.INT8, pl.Mem.Mat] = pl.tile.load(temporary_39, [stage_13_v68 * 64, 0], [384, 128], [384, 128], target_memory=pl.Mem.Mat)
                    stage_13_v57: pl.Tile[[16, 384], pl.FP16, pl.Mem.Mat] = pl.tile.load(temporary_48, [stage_13_v68 // 6 * 16, 0], [16, 384], [16, 384], target_memory=pl.Mem.Mat)
                    stage_13_v58: pl.Tile[[384, 128], pl.INT8, pl.Mem.Left] = pl.tile.move(stage_13_v56, target_memory=pl.Mem.Left)
                    stage_13_v59: pl.Tile[[16, 384], pl.FP16, pl.Mem.Left] = pl.tile.move(stage_13_v57, target_memory=pl.Mem.Left)
                    for stage_13_v60 in pl.range(stage_13_v55):
                        if 2 <= stage_13_v60:
                            pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        stage_13_v69: pl.Scalar[pl.INDEX] = stage_13_v60 * 1024
                        stage_13_v70: pl.Scalar[pl.INDEX] = stage_13_v12 * 12 + stage_13_v60 % 2 * 6
                        stage_13_v71: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                        stage_13_v72: pl.Scalar[pl.INDEX] = 0
                        stage_13_v73: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v72
                        stage_13_v74: pl.Scalar[pl.INDEX] = pl.min(stage_13_v73 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v75: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v74])
                        stage_13_v76: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v75, pl.INDEX), 0)
                        stage_13_v77: pl.Scalar[pl.INDEX] = stage_13_v76 * 4160
                        if stage_13_v77 % 128 == 0:
                            stage_13_v469: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v71, view_98, [0, 0], [stage_13_v77 // 128, 0], [32, 128], transpose=False)
                            stage_13_v470: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v469)
                        else:
                            stage_13_v471: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v71, view_99, [0, 0], [stage_13_v77 // 128, 0], [32, 128], transpose=False)
                            stage_13_v470: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v471)
                        stage_13_v78: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v72 + 32
                        stage_13_v79: pl.Scalar[pl.INDEX] = pl.min(stage_13_v78 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v80: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v79])
                        stage_13_v81: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v80, pl.INDEX), 0)
                        stage_13_v82: pl.Scalar[pl.INDEX] = stage_13_v81 * 4160
                        if stage_13_v82 % 128 == 0:
                            stage_13_v472: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v470, view_98, [32, 0], [stage_13_v82 // 128, 0], [32, 128], transpose=False)
                            stage_13_v473: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v472)
                        else:
                            stage_13_v474: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v470, view_99, [32, 0], [stage_13_v82 // 128, 0], [32, 128], transpose=False)
                            stage_13_v473: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v474)
                        stage_13_v83: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v473)
                        stage_13_v84: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v83, 0, 0, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v85: pl.Scalar[pl.INDEX] = 64
                        stage_13_v86: pl.Scalar[pl.INDEX] = 64
                        stage_13_v87: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v85
                        stage_13_v88: pl.Scalar[pl.INDEX] = pl.min(stage_13_v87 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v89: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v88])
                        stage_13_v90: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v89, pl.INDEX), 0)
                        stage_13_v91: pl.Scalar[pl.INDEX] = stage_13_v90 * 4160
                        if stage_13_v91 % 128 == 0:
                            stage_13_v475: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v473, view_98, [stage_13_v86, 0], [stage_13_v91 // 128, 0], [32, 128], transpose=False)
                            stage_13_v476: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v475)
                        else:
                            stage_13_v477: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v473, view_99, [stage_13_v86, 0], [stage_13_v91 // 128, 0], [32, 128], transpose=False)
                            stage_13_v476: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v477)
                        stage_13_v92: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v85 + 32
                        stage_13_v93: pl.Scalar[pl.INDEX] = pl.min(stage_13_v92 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v94: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v93])
                        stage_13_v95: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v94, pl.INDEX), 0)
                        stage_13_v96: pl.Scalar[pl.INDEX] = stage_13_v95 * 4160
                        if stage_13_v96 % 128 == 0:
                            stage_13_v478: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v476, view_98, [stage_13_v86 + 32, 0], [stage_13_v96 // 128, 0], [32, 128], transpose=False)
                            stage_13_v479: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v478)
                        else:
                            stage_13_v480: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v476, view_99, [stage_13_v86 + 32, 0], [stage_13_v96 // 128, 0], [32, 128], transpose=False)
                            stage_13_v479: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v480)
                        stage_13_v97: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v479)
                        stage_13_v98: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v97, 0, stage_13_v86, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v99: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v84
                        stage_13_v100: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v101: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v99)
                        stage_13_v102: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v100, stage_13_v101, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v103: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v102
                        stage_13_v104: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v98
                        stage_13_v105: pl.Scalar[pl.INDEX] = 64
                        stage_13_v106: pl.Scalar[pl.INDEX] = 128
                        stage_13_v107: pl.Scalar[pl.INDEX] = 0
                        stage_13_v108: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v106
                        stage_13_v109: pl.Scalar[pl.INDEX] = pl.min(stage_13_v108 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v110: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v109])
                        stage_13_v111: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v110, pl.INDEX), 0)
                        stage_13_v112: pl.Scalar[pl.INDEX] = stage_13_v111 * 4160
                        if stage_13_v112 % 128 == 0:
                            stage_13_v481: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v479, view_98, [stage_13_v107, 0], [stage_13_v112 // 128, 0], [32, 128], transpose=False)
                            stage_13_v482: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v481)
                        else:
                            stage_13_v483: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v479, view_99, [stage_13_v107, 0], [stage_13_v112 // 128, 0], [32, 128], transpose=False)
                            stage_13_v482: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v483)
                        stage_13_v113: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v106 + 32
                        stage_13_v114: pl.Scalar[pl.INDEX] = pl.min(stage_13_v113 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v115: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v114])
                        stage_13_v116: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v115, pl.INDEX), 0)
                        stage_13_v117: pl.Scalar[pl.INDEX] = stage_13_v116 * 4160
                        if stage_13_v117 % 128 == 0:
                            stage_13_v484: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v482, view_98, [stage_13_v107 + 32, 0], [stage_13_v117 // 128, 0], [32, 128], transpose=False)
                            stage_13_v485: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v484)
                        else:
                            stage_13_v486: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v482, view_99, [stage_13_v107 + 32, 0], [stage_13_v117 // 128, 0], [32, 128], transpose=False)
                            stage_13_v485: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v486)
                        stage_13_v118: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v485)
                        stage_13_v119: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v118, 0, stage_13_v107, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v120: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v104
                        stage_13_v121: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v122: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v103, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v123: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v120)
                        stage_13_v124: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v122)
                        stage_13_v125: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v121, stage_13_v123, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v126: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v124, 6, 64)
                        stage_13_v127: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v126, [stage_13_v70, stage_13_v105 - 64], temporary_49)
                        stage_13_v128: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v125
                        stage_13_v129: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v119
                        stage_13_v130: pl.Scalar[pl.INDEX] = 128
                        stage_13_v131: pl.Scalar[pl.INDEX] = 192
                        stage_13_v132: pl.Scalar[pl.INDEX] = 64
                        stage_13_v133: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v131
                        stage_13_v134: pl.Scalar[pl.INDEX] = pl.min(stage_13_v133 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v135: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v134])
                        stage_13_v136: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v135, pl.INDEX), 0)
                        stage_13_v137: pl.Scalar[pl.INDEX] = stage_13_v136 * 4160
                        if stage_13_v137 % 128 == 0:
                            stage_13_v487: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v485, view_98, [stage_13_v132, 0], [stage_13_v137 // 128, 0], [32, 128], transpose=False)
                            stage_13_v488: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v487)
                        else:
                            stage_13_v489: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v485, view_99, [stage_13_v132, 0], [stage_13_v137 // 128, 0], [32, 128], transpose=False)
                            stage_13_v488: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v489)
                        stage_13_v138: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v131 + 32
                        stage_13_v139: pl.Scalar[pl.INDEX] = pl.min(stage_13_v138 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v140: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v139])
                        stage_13_v141: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v140, pl.INDEX), 0)
                        stage_13_v142: pl.Scalar[pl.INDEX] = stage_13_v141 * 4160
                        if stage_13_v142 % 128 == 0:
                            stage_13_v490: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v488, view_98, [stage_13_v132 + 32, 0], [stage_13_v142 // 128, 0], [32, 128], transpose=False)
                            stage_13_v491: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v490)
                        else:
                            stage_13_v492: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v488, view_99, [stage_13_v132 + 32, 0], [stage_13_v142 // 128, 0], [32, 128], transpose=False)
                            stage_13_v491: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v492)
                        stage_13_v143: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v491)
                        stage_13_v144: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v143, 0, stage_13_v132, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v145: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v129
                        stage_13_v146: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v147: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v128, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v148: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v145)
                        stage_13_v149: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v147)
                        stage_13_v150: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v146, stage_13_v148, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v151: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v149, 6, 64)
                        stage_13_v152: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v151, [stage_13_v70, stage_13_v130 - 64], temporary_49)
                        stage_13_v153: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v150
                        stage_13_v154: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v144
                        stage_13_v155: pl.Scalar[pl.INDEX] = 192
                        stage_13_v156: pl.Scalar[pl.INDEX] = 256
                        stage_13_v157: pl.Scalar[pl.INDEX] = 0
                        stage_13_v158: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v156
                        stage_13_v159: pl.Scalar[pl.INDEX] = pl.min(stage_13_v158 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v160: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v159])
                        stage_13_v161: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v160, pl.INDEX), 0)
                        stage_13_v162: pl.Scalar[pl.INDEX] = stage_13_v161 * 4160
                        if stage_13_v162 % 128 == 0:
                            stage_13_v493: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v491, view_98, [stage_13_v157, 0], [stage_13_v162 // 128, 0], [32, 128], transpose=False)
                            stage_13_v494: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v493)
                        else:
                            stage_13_v495: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v491, view_99, [stage_13_v157, 0], [stage_13_v162 // 128, 0], [32, 128], transpose=False)
                            stage_13_v494: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v495)
                        stage_13_v163: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v156 + 32
                        stage_13_v164: pl.Scalar[pl.INDEX] = pl.min(stage_13_v163 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v165: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v164])
                        stage_13_v166: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v165, pl.INDEX), 0)
                        stage_13_v167: pl.Scalar[pl.INDEX] = stage_13_v166 * 4160
                        if stage_13_v167 % 128 == 0:
                            stage_13_v496: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v494, view_98, [stage_13_v157 + 32, 0], [stage_13_v167 // 128, 0], [32, 128], transpose=False)
                            stage_13_v497: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v496)
                        else:
                            stage_13_v498: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v494, view_99, [stage_13_v157 + 32, 0], [stage_13_v167 // 128, 0], [32, 128], transpose=False)
                            stage_13_v497: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v498)
                        stage_13_v168: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v497)
                        stage_13_v169: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v168, 0, stage_13_v157, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v170: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v154
                        stage_13_v171: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v172: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v153, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v173: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v170)
                        stage_13_v174: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v172)
                        stage_13_v175: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v171, stage_13_v173, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v176: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v174, 6, 64)
                        stage_13_v177: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v176, [stage_13_v70, stage_13_v155 - 64], temporary_49)
                        stage_13_v178: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v175
                        stage_13_v179: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v169
                        stage_13_v180: pl.Scalar[pl.INDEX] = 256
                        stage_13_v181: pl.Scalar[pl.INDEX] = 320
                        stage_13_v182: pl.Scalar[pl.INDEX] = 64
                        stage_13_v183: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v181
                        stage_13_v184: pl.Scalar[pl.INDEX] = pl.min(stage_13_v183 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v185: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v184])
                        stage_13_v186: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v185, pl.INDEX), 0)
                        stage_13_v187: pl.Scalar[pl.INDEX] = stage_13_v186 * 4160
                        if stage_13_v187 % 128 == 0:
                            stage_13_v499: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v497, view_98, [stage_13_v182, 0], [stage_13_v187 // 128, 0], [32, 128], transpose=False)
                            stage_13_v500: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v499)
                        else:
                            stage_13_v501: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v497, view_99, [stage_13_v182, 0], [stage_13_v187 // 128, 0], [32, 128], transpose=False)
                            stage_13_v500: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v501)
                        stage_13_v188: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v181 + 32
                        stage_13_v189: pl.Scalar[pl.INDEX] = pl.min(stage_13_v188 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v190: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v189])
                        stage_13_v191: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v190, pl.INDEX), 0)
                        stage_13_v192: pl.Scalar[pl.INDEX] = stage_13_v191 * 4160
                        if stage_13_v192 % 128 == 0:
                            stage_13_v502: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v500, view_98, [stage_13_v182 + 32, 0], [stage_13_v192 // 128, 0], [32, 128], transpose=False)
                            stage_13_v503: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v502)
                        else:
                            stage_13_v504: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v500, view_99, [stage_13_v182 + 32, 0], [stage_13_v192 // 128, 0], [32, 128], transpose=False)
                            stage_13_v503: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v504)
                        stage_13_v193: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v503)
                        stage_13_v194: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v193, 0, stage_13_v182, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v195: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v179
                        stage_13_v196: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v197: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v178, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v198: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v195)
                        stage_13_v199: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v197)
                        stage_13_v200: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v196, stage_13_v198, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v201: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v199, 6, 64)
                        stage_13_v202: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v201, [stage_13_v70, stage_13_v180 - 64], temporary_49)
                        stage_13_v203: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v200
                        stage_13_v204: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v194
                        stage_13_v205: pl.Scalar[pl.INDEX] = 320
                        stage_13_v206: pl.Scalar[pl.INDEX] = 384
                        stage_13_v207: pl.Scalar[pl.INDEX] = 0
                        stage_13_v208: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v206
                        stage_13_v209: pl.Scalar[pl.INDEX] = pl.min(stage_13_v208 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v210: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v209])
                        stage_13_v211: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v210, pl.INDEX), 0)
                        stage_13_v212: pl.Scalar[pl.INDEX] = stage_13_v211 * 4160
                        if stage_13_v212 % 128 == 0:
                            stage_13_v505: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v503, view_98, [stage_13_v207, 0], [stage_13_v212 // 128, 0], [32, 128], transpose=False)
                            stage_13_v506: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v505)
                        else:
                            stage_13_v507: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v503, view_99, [stage_13_v207, 0], [stage_13_v212 // 128, 0], [32, 128], transpose=False)
                            stage_13_v506: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v507)
                        stage_13_v213: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v206 + 32
                        stage_13_v214: pl.Scalar[pl.INDEX] = pl.min(stage_13_v213 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v215: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v214])
                        stage_13_v216: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v215, pl.INDEX), 0)
                        stage_13_v217: pl.Scalar[pl.INDEX] = stage_13_v216 * 4160
                        if stage_13_v217 % 128 == 0:
                            stage_13_v508: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v506, view_98, [stage_13_v207 + 32, 0], [stage_13_v217 // 128, 0], [32, 128], transpose=False)
                            stage_13_v509: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v508)
                        else:
                            stage_13_v510: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v506, view_99, [stage_13_v207 + 32, 0], [stage_13_v217 // 128, 0], [32, 128], transpose=False)
                            stage_13_v509: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v510)
                        stage_13_v218: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v509)
                        stage_13_v219: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v218, 0, stage_13_v207, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v220: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v204
                        stage_13_v221: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v222: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v203, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v223: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v220)
                        stage_13_v224: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v222)
                        stage_13_v225: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v221, stage_13_v223, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v226: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v224, 6, 64)
                        stage_13_v227: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v226, [stage_13_v70, stage_13_v205 - 64], temporary_49)
                        stage_13_v228: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v225
                        stage_13_v229: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v219
                        stage_13_v230: pl.Scalar[pl.INDEX] = 384
                        stage_13_v231: pl.Scalar[pl.INDEX] = 448
                        stage_13_v232: pl.Scalar[pl.INDEX] = 64
                        stage_13_v233: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v231
                        stage_13_v234: pl.Scalar[pl.INDEX] = pl.min(stage_13_v233 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v235: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v234])
                        stage_13_v236: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v235, pl.INDEX), 0)
                        stage_13_v237: pl.Scalar[pl.INDEX] = stage_13_v236 * 4160
                        if stage_13_v237 % 128 == 0:
                            stage_13_v511: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v509, view_98, [stage_13_v232, 0], [stage_13_v237 // 128, 0], [32, 128], transpose=False)
                            stage_13_v512: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v511)
                        else:
                            stage_13_v513: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v509, view_99, [stage_13_v232, 0], [stage_13_v237 // 128, 0], [32, 128], transpose=False)
                            stage_13_v512: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v513)
                        stage_13_v238: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v231 + 32
                        stage_13_v239: pl.Scalar[pl.INDEX] = pl.min(stage_13_v238 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v240: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v239])
                        stage_13_v241: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v240, pl.INDEX), 0)
                        stage_13_v242: pl.Scalar[pl.INDEX] = stage_13_v241 * 4160
                        if stage_13_v242 % 128 == 0:
                            stage_13_v514: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v512, view_98, [stage_13_v232 + 32, 0], [stage_13_v242 // 128, 0], [32, 128], transpose=False)
                            stage_13_v515: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v514)
                        else:
                            stage_13_v516: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v512, view_99, [stage_13_v232 + 32, 0], [stage_13_v242 // 128, 0], [32, 128], transpose=False)
                            stage_13_v515: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v516)
                        stage_13_v243: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v515)
                        stage_13_v244: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v243, 0, stage_13_v232, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v245: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v229
                        stage_13_v246: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v247: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v228, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v248: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v245)
                        stage_13_v249: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v247)
                        stage_13_v250: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v246, stage_13_v248, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v251: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v249, 6, 64)
                        stage_13_v252: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v251, [stage_13_v70, stage_13_v230 - 64], temporary_49)
                        stage_13_v253: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v250
                        stage_13_v254: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v244
                        stage_13_v255: pl.Scalar[pl.INDEX] = 448
                        stage_13_v256: pl.Scalar[pl.INDEX] = 512
                        stage_13_v257: pl.Scalar[pl.INDEX] = 0
                        stage_13_v258: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v256
                        stage_13_v259: pl.Scalar[pl.INDEX] = pl.min(stage_13_v258 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v260: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v259])
                        stage_13_v261: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v260, pl.INDEX), 0)
                        stage_13_v262: pl.Scalar[pl.INDEX] = stage_13_v261 * 4160
                        if stage_13_v262 % 128 == 0:
                            stage_13_v517: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v515, view_98, [stage_13_v257, 0], [stage_13_v262 // 128, 0], [32, 128], transpose=False)
                            stage_13_v518: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v517)
                        else:
                            stage_13_v519: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v515, view_99, [stage_13_v257, 0], [stage_13_v262 // 128, 0], [32, 128], transpose=False)
                            stage_13_v518: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v519)
                        stage_13_v263: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v256 + 32
                        stage_13_v264: pl.Scalar[pl.INDEX] = pl.min(stage_13_v263 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v265: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v264])
                        stage_13_v266: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v265, pl.INDEX), 0)
                        stage_13_v267: pl.Scalar[pl.INDEX] = stage_13_v266 * 4160
                        if stage_13_v267 % 128 == 0:
                            stage_13_v520: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v518, view_98, [stage_13_v257 + 32, 0], [stage_13_v267 // 128, 0], [32, 128], transpose=False)
                            stage_13_v521: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v520)
                        else:
                            stage_13_v522: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v518, view_99, [stage_13_v257 + 32, 0], [stage_13_v267 // 128, 0], [32, 128], transpose=False)
                            stage_13_v521: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v522)
                        stage_13_v268: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v521)
                        stage_13_v269: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v268, 0, stage_13_v257, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v270: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v254
                        stage_13_v271: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v272: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v253, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v273: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v270)
                        stage_13_v274: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v272)
                        stage_13_v275: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v271, stage_13_v273, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v276: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v274, 6, 64)
                        stage_13_v277: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v276, [stage_13_v70, stage_13_v255 - 64], temporary_49)
                        stage_13_v278: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v275
                        stage_13_v279: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v269
                        stage_13_v280: pl.Scalar[pl.INDEX] = 512
                        stage_13_v281: pl.Scalar[pl.INDEX] = 576
                        stage_13_v282: pl.Scalar[pl.INDEX] = 64
                        stage_13_v283: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v281
                        stage_13_v284: pl.Scalar[pl.INDEX] = pl.min(stage_13_v283 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v285: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v284])
                        stage_13_v286: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v285, pl.INDEX), 0)
                        stage_13_v287: pl.Scalar[pl.INDEX] = stage_13_v286 * 4160
                        if stage_13_v287 % 128 == 0:
                            stage_13_v523: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v521, view_98, [stage_13_v282, 0], [stage_13_v287 // 128, 0], [32, 128], transpose=False)
                            stage_13_v524: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v523)
                        else:
                            stage_13_v525: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v521, view_99, [stage_13_v282, 0], [stage_13_v287 // 128, 0], [32, 128], transpose=False)
                            stage_13_v524: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v525)
                        stage_13_v288: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v281 + 32
                        stage_13_v289: pl.Scalar[pl.INDEX] = pl.min(stage_13_v288 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v290: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v289])
                        stage_13_v291: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v290, pl.INDEX), 0)
                        stage_13_v292: pl.Scalar[pl.INDEX] = stage_13_v291 * 4160
                        if stage_13_v292 % 128 == 0:
                            stage_13_v526: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v524, view_98, [stage_13_v282 + 32, 0], [stage_13_v292 // 128, 0], [32, 128], transpose=False)
                            stage_13_v527: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v526)
                        else:
                            stage_13_v528: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v524, view_99, [stage_13_v282 + 32, 0], [stage_13_v292 // 128, 0], [32, 128], transpose=False)
                            stage_13_v527: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v528)
                        stage_13_v293: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v527)
                        stage_13_v294: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v293, 0, stage_13_v282, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v295: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v279
                        stage_13_v296: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v297: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v278, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v298: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v295)
                        stage_13_v299: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v297)
                        stage_13_v300: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v296, stage_13_v298, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v301: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v299, 6, 64)
                        stage_13_v302: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v301, [stage_13_v70, stage_13_v280 - 64], temporary_49)
                        stage_13_v303: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v300
                        stage_13_v304: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v294
                        stage_13_v305: pl.Scalar[pl.INDEX] = 576
                        stage_13_v306: pl.Scalar[pl.INDEX] = 640
                        stage_13_v307: pl.Scalar[pl.INDEX] = 0
                        stage_13_v308: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v306
                        stage_13_v309: pl.Scalar[pl.INDEX] = pl.min(stage_13_v308 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v310: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v309])
                        stage_13_v311: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v310, pl.INDEX), 0)
                        stage_13_v312: pl.Scalar[pl.INDEX] = stage_13_v311 * 4160
                        if stage_13_v312 % 128 == 0:
                            stage_13_v529: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v527, view_98, [stage_13_v307, 0], [stage_13_v312 // 128, 0], [32, 128], transpose=False)
                            stage_13_v530: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v529)
                        else:
                            stage_13_v531: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v527, view_99, [stage_13_v307, 0], [stage_13_v312 // 128, 0], [32, 128], transpose=False)
                            stage_13_v530: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v531)
                        stage_13_v313: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v306 + 32
                        stage_13_v314: pl.Scalar[pl.INDEX] = pl.min(stage_13_v313 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v315: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v314])
                        stage_13_v316: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v315, pl.INDEX), 0)
                        stage_13_v317: pl.Scalar[pl.INDEX] = stage_13_v316 * 4160
                        if stage_13_v317 % 128 == 0:
                            stage_13_v532: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v530, view_98, [stage_13_v307 + 32, 0], [stage_13_v317 // 128, 0], [32, 128], transpose=False)
                            stage_13_v533: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v532)
                        else:
                            stage_13_v534: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v530, view_99, [stage_13_v307 + 32, 0], [stage_13_v317 // 128, 0], [32, 128], transpose=False)
                            stage_13_v533: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v534)
                        stage_13_v318: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v533)
                        stage_13_v319: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v318, 0, stage_13_v307, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v320: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v304
                        stage_13_v321: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v322: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v303, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v323: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v320)
                        stage_13_v324: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v322)
                        stage_13_v325: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v321, stage_13_v323, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v326: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v324, 6, 64)
                        stage_13_v327: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v326, [stage_13_v70, stage_13_v305 - 64], temporary_49)
                        stage_13_v328: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v325
                        stage_13_v329: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v319
                        stage_13_v330: pl.Scalar[pl.INDEX] = 640
                        stage_13_v331: pl.Scalar[pl.INDEX] = 704
                        stage_13_v332: pl.Scalar[pl.INDEX] = 64
                        stage_13_v333: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v331
                        stage_13_v334: pl.Scalar[pl.INDEX] = pl.min(stage_13_v333 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v335: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v334])
                        stage_13_v336: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v335, pl.INDEX), 0)
                        stage_13_v337: pl.Scalar[pl.INDEX] = stage_13_v336 * 4160
                        if stage_13_v337 % 128 == 0:
                            stage_13_v535: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v533, view_98, [stage_13_v332, 0], [stage_13_v337 // 128, 0], [32, 128], transpose=False)
                            stage_13_v536: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v535)
                        else:
                            stage_13_v537: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v533, view_99, [stage_13_v332, 0], [stage_13_v337 // 128, 0], [32, 128], transpose=False)
                            stage_13_v536: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v537)
                        stage_13_v338: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v331 + 32
                        stage_13_v339: pl.Scalar[pl.INDEX] = pl.min(stage_13_v338 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v340: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v339])
                        stage_13_v341: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v340, pl.INDEX), 0)
                        stage_13_v342: pl.Scalar[pl.INDEX] = stage_13_v341 * 4160
                        if stage_13_v342 % 128 == 0:
                            stage_13_v538: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v536, view_98, [stage_13_v332 + 32, 0], [stage_13_v342 // 128, 0], [32, 128], transpose=False)
                            stage_13_v539: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v538)
                        else:
                            stage_13_v540: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v536, view_99, [stage_13_v332 + 32, 0], [stage_13_v342 // 128, 0], [32, 128], transpose=False)
                            stage_13_v539: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v540)
                        stage_13_v343: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v539)
                        stage_13_v344: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v343, 0, stage_13_v332, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v345: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v329
                        stage_13_v346: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v347: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v328, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v348: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v345)
                        stage_13_v349: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v347)
                        stage_13_v350: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v346, stage_13_v348, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v351: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v349, 6, 64)
                        stage_13_v352: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v351, [stage_13_v70, stage_13_v330 - 64], temporary_49)
                        stage_13_v353: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v350
                        stage_13_v354: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v344
                        stage_13_v355: pl.Scalar[pl.INDEX] = 704
                        stage_13_v356: pl.Scalar[pl.INDEX] = 768
                        stage_13_v357: pl.Scalar[pl.INDEX] = 0
                        stage_13_v358: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v356
                        stage_13_v359: pl.Scalar[pl.INDEX] = pl.min(stage_13_v358 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v360: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v359])
                        stage_13_v361: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v360, pl.INDEX), 0)
                        stage_13_v362: pl.Scalar[pl.INDEX] = stage_13_v361 * 4160
                        if stage_13_v362 % 128 == 0:
                            stage_13_v541: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v539, view_98, [stage_13_v357, 0], [stage_13_v362 // 128, 0], [32, 128], transpose=False)
                            stage_13_v542: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v541)
                        else:
                            stage_13_v543: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v539, view_99, [stage_13_v357, 0], [stage_13_v362 // 128, 0], [32, 128], transpose=False)
                            stage_13_v542: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v543)
                        stage_13_v363: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v356 + 32
                        stage_13_v364: pl.Scalar[pl.INDEX] = pl.min(stage_13_v363 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v365: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v364])
                        stage_13_v366: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v365, pl.INDEX), 0)
                        stage_13_v367: pl.Scalar[pl.INDEX] = stage_13_v366 * 4160
                        if stage_13_v367 % 128 == 0:
                            stage_13_v544: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v542, view_98, [stage_13_v357 + 32, 0], [stage_13_v367 // 128, 0], [32, 128], transpose=False)
                            stage_13_v545: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v544)
                        else:
                            stage_13_v546: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v542, view_99, [stage_13_v357 + 32, 0], [stage_13_v367 // 128, 0], [32, 128], transpose=False)
                            stage_13_v545: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v546)
                        stage_13_v368: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v545)
                        stage_13_v369: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v368, 0, stage_13_v357, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v370: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v354
                        stage_13_v371: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v372: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v353, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v373: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v370)
                        stage_13_v374: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v372)
                        stage_13_v375: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v371, stage_13_v373, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v376: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v374, 6, 64)
                        stage_13_v377: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v376, [stage_13_v70, stage_13_v355 - 64], temporary_49)
                        stage_13_v378: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v375
                        stage_13_v379: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v369
                        stage_13_v380: pl.Scalar[pl.INDEX] = 768
                        stage_13_v381: pl.Scalar[pl.INDEX] = 832
                        stage_13_v382: pl.Scalar[pl.INDEX] = 64
                        stage_13_v383: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v381
                        stage_13_v384: pl.Scalar[pl.INDEX] = pl.min(stage_13_v383 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v385: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v384])
                        stage_13_v386: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v385, pl.INDEX), 0)
                        stage_13_v387: pl.Scalar[pl.INDEX] = stage_13_v386 * 4160
                        if stage_13_v387 % 128 == 0:
                            stage_13_v547: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v545, view_98, [stage_13_v382, 0], [stage_13_v387 // 128, 0], [32, 128], transpose=False)
                            stage_13_v548: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v547)
                        else:
                            stage_13_v549: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v545, view_99, [stage_13_v382, 0], [stage_13_v387 // 128, 0], [32, 128], transpose=False)
                            stage_13_v548: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v549)
                        stage_13_v388: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v381 + 32
                        stage_13_v389: pl.Scalar[pl.INDEX] = pl.min(stage_13_v388 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v390: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v389])
                        stage_13_v391: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v390, pl.INDEX), 0)
                        stage_13_v392: pl.Scalar[pl.INDEX] = stage_13_v391 * 4160
                        if stage_13_v392 % 128 == 0:
                            stage_13_v550: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v548, view_98, [stage_13_v382 + 32, 0], [stage_13_v392 // 128, 0], [32, 128], transpose=False)
                            stage_13_v551: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v550)
                        else:
                            stage_13_v552: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v548, view_99, [stage_13_v382 + 32, 0], [stage_13_v392 // 128, 0], [32, 128], transpose=False)
                            stage_13_v551: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v552)
                        stage_13_v393: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v551)
                        stage_13_v394: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v393, 0, stage_13_v382, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v395: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v379
                        stage_13_v396: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v397: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v378, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v398: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v395)
                        stage_13_v399: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v397)
                        stage_13_v400: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v396, stage_13_v398, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v401: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v399, 6, 64)
                        stage_13_v402: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v401, [stage_13_v70, stage_13_v380 - 64], temporary_49)
                        stage_13_v403: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v400
                        stage_13_v404: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v394
                        stage_13_v405: pl.Scalar[pl.INDEX] = 832
                        stage_13_v406: pl.Scalar[pl.INDEX] = 896
                        stage_13_v407: pl.Scalar[pl.INDEX] = 0
                        stage_13_v408: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v406
                        stage_13_v409: pl.Scalar[pl.INDEX] = pl.min(stage_13_v408 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v410: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v409])
                        stage_13_v411: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v410, pl.INDEX), 0)
                        stage_13_v412: pl.Scalar[pl.INDEX] = stage_13_v411 * 4160
                        if stage_13_v412 % 128 == 0:
                            stage_13_v553: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v551, view_98, [stage_13_v407, 0], [stage_13_v412 // 128, 0], [32, 128], transpose=False)
                            stage_13_v554: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v553)
                        else:
                            stage_13_v555: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v551, view_99, [stage_13_v407, 0], [stage_13_v412 // 128, 0], [32, 128], transpose=False)
                            stage_13_v554: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v555)
                        stage_13_v413: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v406 + 32
                        stage_13_v414: pl.Scalar[pl.INDEX] = pl.min(stage_13_v413 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v415: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v414])
                        stage_13_v416: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v415, pl.INDEX), 0)
                        stage_13_v417: pl.Scalar[pl.INDEX] = stage_13_v416 * 4160
                        if stage_13_v417 % 128 == 0:
                            stage_13_v556: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v554, view_98, [stage_13_v407 + 32, 0], [stage_13_v417 // 128, 0], [32, 128], transpose=False)
                            stage_13_v557: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v556)
                        else:
                            stage_13_v558: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v554, view_99, [stage_13_v407 + 32, 0], [stage_13_v417 // 128, 0], [32, 128], transpose=False)
                            stage_13_v557: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v558)
                        stage_13_v418: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v557)
                        stage_13_v419: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v418, 0, stage_13_v407, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v420: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v404
                        stage_13_v421: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v422: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v403, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v423: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v420)
                        stage_13_v424: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v422)
                        stage_13_v425: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v421, stage_13_v423, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v426: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v424, 6, 64)
                        stage_13_v427: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v426, [stage_13_v70, stage_13_v405 - 64], temporary_49)
                        stage_13_v428: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v425
                        stage_13_v429: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v419
                        stage_13_v430: pl.Scalar[pl.INDEX] = 896
                        stage_13_v431: pl.Scalar[pl.INDEX] = 960
                        stage_13_v432: pl.Scalar[pl.INDEX] = 64
                        stage_13_v433: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v431
                        stage_13_v434: pl.Scalar[pl.INDEX] = pl.min(stage_13_v433 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v435: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v434])
                        stage_13_v436: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v435, pl.INDEX), 0)
                        stage_13_v437: pl.Scalar[pl.INDEX] = stage_13_v436 * 4160
                        if stage_13_v437 % 128 == 0:
                            stage_13_v559: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v557, view_98, [stage_13_v432, 0], [stage_13_v437 // 128, 0], [32, 128], transpose=False)
                            stage_13_v560: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v559)
                        else:
                            stage_13_v561: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v557, view_99, [stage_13_v432, 0], [stage_13_v437 // 128, 0], [32, 128], transpose=False)
                            stage_13_v560: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v561)
                        stage_13_v438: pl.Scalar[pl.INDEX] = stage_13_v37 + stage_13_v69 + stage_13_v431 + 32
                        stage_13_v439: pl.Scalar[pl.INDEX] = pl.min(stage_13_v438 // 32, pl.max((stage_13_v35 - 1) // 32, 0))
                        stage_13_v440: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table, [stage_13_v32, stage_13_v439])
                        stage_13_v441: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v440, pl.INDEX), 0)
                        stage_13_v442: pl.Scalar[pl.INDEX] = stage_13_v441 * 4160
                        if stage_13_v442 % 128 == 0:
                            stage_13_v562: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v560, view_98, [stage_13_v432 + 32, 0], [stage_13_v442 // 128, 0], [32, 128], transpose=False)
                            stage_13_v563: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v562)
                        else:
                            stage_13_v564: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(stage_13_v560, view_99, [stage_13_v432 + 32, 0], [stage_13_v442 // 128, 0], [32, 128], transpose=False)
                            stage_13_v563: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.yield_(stage_13_v564)
                        stage_13_v443: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(stage_13_v563)
                        stage_13_v444: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(stage_13_v443, 0, stage_13_v432, [128, 64], target_memory=pl.Mem.Right)
                        stage_13_v445: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v429
                        stage_13_v446: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v447: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v428, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v448: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v445)
                        stage_13_v449: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v447)
                        stage_13_v450: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v446, stage_13_v448, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v451: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v449, 6, 64)
                        stage_13_v452: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v451, [stage_13_v70, stage_13_v430 - 64], temporary_49)
                        stage_13_v453: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v450
                        stage_13_v454: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v444
                        stage_13_v455: pl.Scalar[pl.INDEX] = 960
                        stage_13_v456: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = stage_13_v454
                        stage_13_v457: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        stage_13_v458: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v453, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v459: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v58, stage_13_v456)
                        stage_13_v460: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v458)
                        stage_13_v461: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(stage_13_v457, stage_13_v459, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        stage_13_v462: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v460, 6, 64)
                        stage_13_v463: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v462, [stage_13_v70, stage_13_v455 - 64], temporary_49)
                        stage_13_v464: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = stage_13_v461
                        stage_13_v465: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(stage_13_v464, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                        stage_13_v466: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(stage_13_v59, stage_13_v465)
                        stage_13_v467: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(stage_13_v466, 6, 64)
                        stage_13_v468: pl.Tensor[[288, 1024], pl.FP32] = pl.tile.store(stage_13_v467, [stage_13_v70, 960], temporary_49)
                        pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                    for stage_13_v61 in pl.range(pl.min(stage_13_v55, 2)):
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
            pl.system.fence()
            pl.system.cacheinvalid()
            pl.system.fence()
        return (temporary_39, temporary_48, temporary_50, view_98, view_99, temporary_49)

    @pl.function(type=pl.FunctionType.Orchestration)
    def entry(self, arg_kv_seq_lens: pl.Tensor[[4], pl.INT32], arg_position_ids: pl.Tensor[[24], pl.INT64], temporary_39: pl.InOut[pl.Tensor[[24576, 128], pl.INT8]], temporary_48: pl.InOut[pl.Tensor[[1024, 384], pl.FP16]], temporary_50: pl.InOut[pl.Tensor[[256], pl.INT64]], arg_idx_block_table: pl.Tensor[[4, 1026], pl.INT32], view_98: pl.InOut[pl.Tensor[[133412, 128], pl.INT8]], view_99: pl.InOut[pl.Tensor[[133412, 128], pl.INT8]], temporary_49: pl.InOut[pl.Tensor[[288, 1024], pl.FP32]], records: pl.Tensor[[31968], pl.INT64]) -> tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[1024, 384], pl.FP16], pl.Tensor[[256], pl.INT64], pl.Tensor[[133412, 128], pl.INT8], pl.Tensor[[133412, 128], pl.INT8], pl.Tensor[[288, 1024], pl.FP32]]:
        pl.func_attr({'mx_tensor_views_blocked': True})
        with pl.spmd(24, sync_start=True, name_hint='csa_static_full_card'):
            (temporary_39, temporary_48, temporary_50, view_98, view_99, temporary_49) = self.csa_static_aic(arg_kv_seq_lens, arg_position_ids, temporary_39, temporary_48, temporary_50, arg_idx_block_table, view_98, view_99, temporary_49, records)
        return (temporary_39, temporary_48, temporary_50, view_98, view_99, temporary_49)
