# 只供 CPU 编译的阶段分配诊断；不允许单独上卡。
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
    def csa_static_aiv(self, arg_freqs_cos: pl.Tensor[[24, 64], pl.FP32], view_45: pl.InOut[pl.Tensor[[1184, 512], pl.BF16]], view_61: pl.InOut[pl.Tensor[[131360, 512], pl.BF16]], temporary_53: pl.InOut[pl.Tensor[[24, 512], pl.INT32]], temporary_52: pl.InOut[pl.Tensor[[24, 640], pl.FP32]], view_106: pl.Tensor[[24, 1], pl.INT64], temporary_54: pl.InOut[pl.Tensor[[24, 16], pl.INT32]], arg_ori_block_table: pl.Tensor[[4, 4098], pl.INT32], temporary_55: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_62: pl.InOut[pl.Tensor[[256], pl.INT64]], temporary_56: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_57: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], temporary_58: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], temporary_59: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], view_116: pl.Tensor[[64, 1], pl.FP32], arg_cmp_block_table: pl.Tensor[[4, 1026], pl.INT32], temporary_60: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_61: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], records: pl.Tensor[[31968], pl.INT64]) -> tuple[pl.Tensor[[1184, 512], pl.BF16], pl.Tensor[[131360, 512], pl.BF16], pl.Tensor[[24, 512], pl.INT32], pl.Tensor[[24, 640], pl.FP32], pl.Tensor[[24, 16], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[256], pl.INT64], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16]]:
        pl.func_attr({'mx_tensor_views_blocked': True, 'split_aiv': True})
        group_id = pl.tile.get_block_idx()
        lane_id = pl.tile.get_subblock_idx()
        core_id = 24 + group_id * 2 + lane_id
        record_32 = 27648 + core_id * 6
        mapped_block_32 = pl.cast(pl.tensor.read(records, [record_32]), pl.INDEX)
        block_32 = pl.max(mapped_block_32, 0)
        if mapped_block_32 >= 0:
            subslot_32 = pl.cast(pl.tensor.read(records, [record_32 + 1]), pl.INDEX)
            stage_32_v22: pl.Scalar[pl.INDEX] = block_32
            pl.system.set_ffts(temporary_62)
            stage_32_v23: pl.Scalar[pl.INDEX] = (24 - stage_32_v22 + 23) // 24
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
            stage_32_v41: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_116, [stage_32_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
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
                            stage_32_v256: pl.Scalar[pl.INT64] = pl.tensor.read(view_106, [stage_32_v48, 0])
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
                        stage_32_v75: pl.Tensor[[9216, 512], pl.BF16] = pl.tile.store(stage_32_v272, [stage_32_v73 + stage_32_v27, 0], temporary_56)
                if 144 <= 24:
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
                        if 24 < 144:
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
                        stage_32_v91: pl.Tensor[[4608, 128], pl.BF16] = pl.tile.store(stage_32_v90, [stage_32_v79 + stage_32_v26, 0], temporary_58)
                        stage_32_v92: pl.Tensor[[4608, 1], pl.FP32] = pl.tile.store(stage_32_v86, [stage_32_v79 + stage_32_v26, 0], temporary_60)
                        stage_32_v93: pl.Tensor[[4608, 1], pl.FP32] = pl.tile.store(stage_32_v89, [stage_32_v79 + stage_32_v26, 0], temporary_61)
                        pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                if 24 < 144:
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
                        stage_32_v153: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_116, [stage_32_v143, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
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
                        stage_32_v192: pl.Tensor[[3072, 4096], pl.BF16] = pl.tile.store(stage_32_v188, [stage_32_v190, 0], temporary_51)
                        stage_32_v193: pl.Tensor[[3072, 4096], pl.BF16] = pl.tile.store(stage_32_v189, [stage_32_v191, 0], temporary_51)
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
                        stage_32_v205: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_116, [stage_32_v195, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
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
                        stage_32_v244: pl.Tensor[[3072, 4096], pl.BF16] = pl.tile.store(stage_32_v240, [stage_32_v242, 0], temporary_51)
                        stage_32_v245: pl.Tensor[[3072, 4096], pl.BF16] = pl.tile.store(stage_32_v241, [stage_32_v243, 0], temporary_51)
                        stage_32_v246: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(view_116, [stage_32_v26, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
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
        return (view_45, view_61, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51)

    @pl.function(type=pl.FunctionType.Orchestration)
    def entry(self, arg_freqs_cos: pl.Tensor[[24, 64], pl.FP32], view_45: pl.InOut[pl.Tensor[[1184, 512], pl.BF16]], view_61: pl.InOut[pl.Tensor[[131360, 512], pl.BF16]], temporary_53: pl.InOut[pl.Tensor[[24, 512], pl.INT32]], temporary_52: pl.InOut[pl.Tensor[[24, 640], pl.FP32]], view_106: pl.Tensor[[24, 1], pl.INT64], temporary_54: pl.InOut[pl.Tensor[[24, 16], pl.INT32]], arg_ori_block_table: pl.Tensor[[4, 4098], pl.INT32], temporary_55: pl.InOut[pl.Tensor[[384, 64], pl.FP32]], temporary_62: pl.InOut[pl.Tensor[[256], pl.INT64]], temporary_56: pl.InOut[pl.Tensor[[9216, 512], pl.BF16]], temporary_57: pl.InOut[pl.Tensor[[4608, 128], pl.FP32]], temporary_58: pl.InOut[pl.Tensor[[4608, 128], pl.BF16]], temporary_59: pl.InOut[pl.Tensor[[4608, 512], pl.FP32]], view_116: pl.Tensor[[64, 1], pl.FP32], arg_cmp_block_table: pl.Tensor[[4, 1026], pl.INT32], temporary_60: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_61: pl.InOut[pl.Tensor[[4608, 1], pl.FP32]], temporary_51: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16]], records: pl.Tensor[[31968], pl.INT64]) -> tuple[pl.Tensor[[1184, 512], pl.BF16], pl.Tensor[[131360, 512], pl.BF16], pl.Tensor[[24, 512], pl.INT32], pl.Tensor[[24, 640], pl.FP32], pl.Tensor[[24, 16], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[256], pl.INT64], pl.Tensor[[9216, 512], pl.BF16], pl.Tensor[[4608, 128], pl.FP32], pl.Tensor[[4608, 128], pl.BF16], pl.Tensor[[4608, 512], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[4608, 1], pl.FP32], pl.Tensor[[3072, 4096], pl.BF16]]:
        pl.func_attr({'mx_tensor_views_blocked': True})
        with pl.spmd(24, sync_start=True, name_hint='csa_static_full_card'):
            (view_45, view_61, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51) = self.csa_static_aiv(arg_freqs_cos, view_45, view_61, temporary_53, temporary_52, view_106, temporary_54, arg_ori_block_table, temporary_55, temporary_62, temporary_56, temporary_57, temporary_58, temporary_59, view_116, arg_cmp_block_table, temporary_60, temporary_61, temporary_51, records)
        return (view_45, view_61, temporary_53, temporary_52, temporary_54, temporary_55, temporary_62, temporary_56, temporary_57, temporary_58, temporary_59, temporary_60, temporary_61, temporary_51)
