# pypto.program: CSAAllocatorFocusedProbe
import pypto.language as pl


@pl.program
class CSAAllocatorFocusedProbe:
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def csa_static_aic(
        arg_kv_seq_lens__ssa_v0: pl.Tensor[[4], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16)],
        arg_position_ids__ssa_v0: pl.Tensor[[24], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 192)],
        temporary_39__ssa_v0: pl.InOut[pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 3145728)]],
        temporary_48__ssa_v0: pl.InOut[pl.Tensor[[1024, 384], pl.FP16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 786432)]],
        temporary_50__ssa_v0: pl.InOut[pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2048)]],
        arg_idx_block_table__ssa_v0: pl.Tensor[[4, 1026], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16416)],
        view_98__ssa_v0: pl.InOut[pl.Tensor[[133412, 128], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 17076736)]],
        view_99__ssa_v0: pl.InOut[pl.Tensor[[133412, 128], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 17076736)]],
        temporary_49__ssa_v0: pl.InOut[pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)]],
        temporary_51__ssa_v0: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 25165824)]],
        arg_wo_a__ssa_v0: pl.Tensor[
            [8, 64, 256, 16, 16], pl.BF16, pl.TensorView(stride=[4194304, 65536, 256, 16, 1], layout=pl.TensorLayout.NZ), pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 67108864)
        ],
        temporary_63__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 12582912)]],
        records__ssa_v0: pl.Tensor[[31968], pl.INT64, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 255744)],
    ) -> tuple[
        pl.Tensor[[24576, 128], pl.INT8],
        pl.Tensor[[1024, 384], pl.FP16],
        pl.Tensor[[256], pl.INT64],
        pl.Tensor[[133412, 128], pl.INT8],
        pl.Tensor[[133412, 128], pl.INT8],
        pl.Tensor[[288, 1024], pl.FP32],
        pl.Tensor[[3072, 4096], pl.BF16],
        pl.Tensor[[384, 8192], pl.FP32],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True, "nz_tensor_views_blocked": True})
        mem_mat_13: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_mat_14: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 12288)
        mem_left_15: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 49152)
        mem_left_16: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 12288)
        mem_mat_17: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_right_20: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_right_23: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_24: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_acc_25: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_right_28: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_29: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_30: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_31: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_32: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_35: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_36: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_37: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_38: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_39: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_42: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_43: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_44: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_45: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_46: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_49: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_50: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_51: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_52: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_53: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_56: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_57: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_58: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_59: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_60: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_63: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_64: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_65: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_66: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_67: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_70: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_71: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_72: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_73: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_74: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_77: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_78: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_79: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_80: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_81: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_84: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_85: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_86: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_87: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_88: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_91: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_92: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_93: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_94: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_95: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_98: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_99: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_100: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_101: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_102: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_105: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_106: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_107: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_108: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_109: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_112: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_113: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_114: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_115: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_116: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_119: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_mat_120: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_121: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_122: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_123: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_mat_124: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_right_125: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_126: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_acc_127: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_right_128: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 49152)
        mem_acc_129: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_mat_130: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_131: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_acc_132: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 65536)
        mem_left_133: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_134: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_135: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_136: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_mat_138: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_139: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_140: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_141: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_142: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_143: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_144: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_145: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_147: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_148: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_149: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_150: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_mat_153: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_154: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_155: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_156: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_157: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_158: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        group_id__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        core_id__ssa_v0: pl.Scalar[pl.INDEX] = group_id__ssa_v0
        record_13__ssa_v0: pl.Scalar[pl.INDEX] = core_id__ssa_v0 * 6 + 5616
        t__tile: pl.Scalar[pl.INT64] = pl.tensor.read(records__ssa_v0, [record_13__ssa_v0])
        mapped_block_13__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX)
        block_13__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(mapped_block_13__ssa_v0, 0)
        if 0 <= mapped_block_13__ssa_v0:
            t__tile_1: pl.Scalar[pl.INT64] = pl.tensor.read(records__ssa_v0, [record_13__ssa_v0 + 1])
            subslot_13__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
            stage_13_v12__ssa_v0: pl.Scalar[pl.INDEX] = block_13__ssa_v0
            stage_13_v13__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(arg_position_ids__ssa_v0, 0)
            stage_13_v14__ssa_v0: pl.Scalar[pl.INDEX] = 0
            for stage_13_v25__idx_v0, (stage_13_v39,) in pl.range(stage_13_v13__ssa_v0 // 6, init_values=(stage_13_v14__ssa_v0,)):
                stage_13_v26__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens__ssa_v0, [stage_13_v25__idx_v0])
                stage_13_v27__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(stage_13_v39, pl.cast(stage_13_v26__tile, pl.INDEX) // 4)
                stage_13_v28: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v27__ssa_v0)
            stage_13_v15__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v28, 262144)
            stage_13_v16__ssa_v0: pl.Scalar[pl.INDEX] = (stage_13_v15__ssa_v0 + 1023) // 1024
            stage_13_v17__ssa_v0: pl.Scalar[pl.INDEX] = pl.max((stage_13_v15__ssa_v0 + 8191) // 8192, 1)
            stage_13_v18__ssa_v0: pl.Scalar[pl.INDEX] = 8
            stage_13_v19__ssa_v0: pl.Scalar[pl.INDEX] = 0
            stage_13_v20__ssa_v0: pl.Scalar[pl.INDEX] = 1
            if stage_13_v13__ssa_v0 // 6 * 3 == 48:
                stage_13_v29__ssa_v0: pl.Scalar[pl.INDEX] = 3
                stage_13_v30__rv_v0: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v29__ssa_v0)
            else:
                if stage_13_v13__ssa_v0 // 6 * 3 == 24:
                    stage_13_v40__ssa_v0: pl.Scalar[pl.INDEX] = 3
                    stage_13_v41__rv_v0: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v40__ssa_v0)
                else:
                    if stage_13_v13__ssa_v0 // 6 * 6 == 24:
                        stage_13_v62__ssa_v0: pl.Scalar[pl.INDEX] = 6
                        stage_13_v63__rv_v0: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v62__ssa_v0)
                    else:
                        stage_13_v63__rv_v0: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v20__ssa_v0)
                    stage_13_v41__rv_v0: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v63__rv_v0)
                stage_13_v30__rv_v0: pl.Scalar[pl.INDEX] = pl.yield_(stage_13_v41__rv_v0)
            if 1 < stage_13_v30__rv_v0:
                stage_13_v31__ssa_v0: pl.Scalar[pl.INDEX] = (stage_13_v17__ssa_v0 + stage_13_v30__rv_v0 - 1) // stage_13_v30__rv_v0 * stage_13_v30__rv_v0
                if stage_13_v31__ssa_v0 <= 32:
                    stage_13_v42__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v31__ssa_v0
                    stage_13_v43__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(stage_13_v16__ssa_v0 // stage_13_v31__ssa_v0, 1)
                    stage_13_v44__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v16__ssa_v0 % stage_13_v31__ssa_v0
                    stage_13_v64__rv_v0, stage_13_v65__rv_v0, stage_13_v66__rv_v0 = pl.yield_(stage_13_v44__ssa_v0, stage_13_v42__ssa_v0, stage_13_v43__ssa_v0)
                else:
                    stage_13_v64__rv_v0, stage_13_v65__rv_v0, stage_13_v66__rv_v0 = pl.yield_(stage_13_v19__ssa_v0, stage_13_v17__ssa_v0, stage_13_v18__ssa_v0)
                stage_13_v45__rv_v0, stage_13_v46__rv_v0, stage_13_v47__rv_v0 = pl.yield_(stage_13_v64__rv_v0, stage_13_v65__rv_v0, stage_13_v66__rv_v0)
            else:
                stage_13_v45__rv_v0, stage_13_v46__rv_v0, stage_13_v47__rv_v0 = pl.yield_(stage_13_v19__ssa_v0, stage_13_v17__ssa_v0, stage_13_v18__ssa_v0)
            stage_13_v21__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v46__rv_v0
            stage_13_v22__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v47__rv_v0
            stage_13_v23__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v45__rv_v0
            pl.system.set_ffts(temporary_50__ssa_v0)
            for stage_13_v24__idx_v0 in pl.range(stage_13_v12__ssa_v0, stage_13_v13__ssa_v0 // 6 * stage_13_v21__ssa_v0, 24):
                if stage_13_v13__ssa_v0 < 144:
                    stage_13_v48__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v24__idx_v0 % (stage_13_v13__ssa_v0 // 6) * 6
                    stage_13_v49__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v24__idx_v0 // (stage_13_v13__ssa_v0 // 6)
                    stage_13_v67__rv_v0, stage_13_v68__rv_v0 = pl.yield_(stage_13_v49__ssa_v0, stage_13_v48__ssa_v0)
                else:
                    stage_13_v50__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v24__idx_v0 // stage_13_v21__ssa_v0 * 6
                    stage_13_v51__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v24__idx_v0 % stage_13_v21__ssa_v0
                    stage_13_v67__rv_v0, stage_13_v68__rv_v0 = pl.yield_(stage_13_v51__ssa_v0, stage_13_v50__ssa_v0)
                stage_13_v32__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v68__rv_v0 // 6
                stage_13_v33__tile: pl.Scalar[pl.INT64] = pl.tensor.read(arg_position_ids__ssa_v0, [stage_13_v68__rv_v0 + 5])
                stage_13_v34__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_kv_seq_lens__ssa_v0, [stage_13_v32__ssa_v0])
                stage_13_v35__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(stage_13_v34__tile, pl.INDEX) // 4
                stage_13_v36__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(stage_13_v35__ssa_v0, (stage_13_v33__tile + 1) // 4), 262144), 0)
                stage_13_v37__ssa_v0: pl.Scalar[pl.INDEX] = (stage_13_v67__rv_v0 * stage_13_v22__ssa_v0 + pl.min(stage_13_v67__rv_v0, stage_13_v23__ssa_v0)) * 1024
                stage_13_v38__ssa_v0: pl.Scalar[pl.INDEX] = (stage_13_v22__ssa_v0 + pl.cast(stage_13_v67__rv_v0 < stage_13_v23__ssa_v0, pl.INDEX)) * 1024
                if stage_13_v37__ssa_v0 < stage_13_v36__ssa_v0:
                    stage_13_v52__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v38__ssa_v0, stage_13_v36__ssa_v0 - stage_13_v37__ssa_v0)
                    stage_13_v53__ssa_v0: pl.Scalar[pl.INDEX] = (stage_13_v52__ssa_v0 + 1023) // 1024
                    stage_13_v54__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v53__ssa_v0 * 512, 4096)
                    stage_13_v55__ssa_v0: pl.Scalar[pl.INDEX] = (stage_13_v54__ssa_v0 + 511) // 512
                    stage_13_v56__ssa_v0: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_13, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.load(
                        temporary_39__ssa_v0, [stage_13_v68__rv_v0 * 64, 0], [384, 128], [384, 128], target_memory=pl.Mem.Mat
                    )
                    stage_13_v57__ssa_v0: pl.Tile[[16, 384], pl.FP16, pl.MemRef(mem_mat_14, pl.const(0, pl.INT64), 12288), pl.Mem.Mat] = pl.tile.load(
                        temporary_48__ssa_v0, [stage_13_v68__rv_v0 // 6 * 16, 0], [16, 384], [16, 384], target_memory=pl.Mem.Mat
                    )
                    stage_13_v58__ssa_v0: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_left_15, pl.const(0, pl.INT64), 49152), pl.Mem.Left] = pl.tile.move(
                        stage_13_v56__ssa_v0, target_memory=pl.Mem.Left
                    )
                    stage_13_v59__ssa_v0: pl.Tile[[16, 384], pl.FP16, pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 12288), pl.Mem.Left] = pl.tile.move(stage_13_v57__ssa_v0, target_memory=pl.Mem.Left)
                    for stage_13_v60__idx_v0 in pl.range(stage_13_v55__ssa_v0):
                        if 2 <= stage_13_v60__idx_v0:
                            pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        stage_13_v69__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v60__idx_v0 * 1024
                        stage_13_v70__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v12__ssa_v0 * 12 + stage_13_v60__idx_v0 % 2 * 6
                        stage_13_v71__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.create(
                            [128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat
                        )
                        stage_13_v72__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        stage_13_v73__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v72__ssa_v0
                        stage_13_v74__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v73__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v75__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v74__ssa_v0])
                        stage_13_v76__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v75__tile, pl.INDEX), 0)
                        stage_13_v77__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v76__ssa_v0 * 4160
                        if stage_13_v77__ssa_v0 % 128 == 0:
                            stage_13_v469__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v71__ssa_v0, view_98__ssa_v0, [0, 0], [stage_13_v77__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v470__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v469__ssa_v0)
                        else:
                            stage_13_v471__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v71__ssa_v0, view_99__ssa_v0, [0, 0], [stage_13_v77__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v470__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v471__ssa_v0)
                        stage_13_v78__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v72__ssa_v0 + 32
                        stage_13_v79__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v78__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v80__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v79__ssa_v0])
                        stage_13_v81__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v80__tile, pl.INDEX), 0)
                        stage_13_v82__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v81__ssa_v0 * 4160
                        if stage_13_v82__ssa_v0 % 128 == 0:
                            stage_13_v472__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v470__rv_v0, view_98__ssa_v0, [32, 0], [stage_13_v82__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v473__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v472__ssa_v0)
                        else:
                            stage_13_v474__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v470__rv_v0, view_99__ssa_v0, [32, 0], [stage_13_v82__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v473__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v474__ssa_v0)
                        stage_13_v83__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v473__rv_v0)
                        stage_13_v84__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_20, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v83__ssa_v0, 0, 0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v85__ssa_v0: pl.Scalar[pl.INDEX] = 64
                        stage_13_v86__ssa_v0: pl.Scalar[pl.INDEX] = 64
                        stage_13_v87__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v85__ssa_v0
                        stage_13_v88__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v87__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v89__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v88__ssa_v0])
                        stage_13_v90__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v89__tile, pl.INDEX), 0)
                        stage_13_v91__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v90__ssa_v0 * 4160
                        if stage_13_v91__ssa_v0 % 128 == 0:
                            stage_13_v475__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v473__rv_v0, view_98__ssa_v0, [stage_13_v86__ssa_v0, 0], [stage_13_v91__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v476__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v475__ssa_v0)
                        else:
                            stage_13_v477__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v473__rv_v0, view_99__ssa_v0, [stage_13_v86__ssa_v0, 0], [stage_13_v91__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v476__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v477__ssa_v0)
                        stage_13_v92__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v85__ssa_v0 + 32
                        stage_13_v93__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v92__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v94__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v93__ssa_v0])
                        stage_13_v95__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v94__tile, pl.INDEX), 0)
                        stage_13_v96__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v95__ssa_v0 * 4160
                        if stage_13_v96__ssa_v0 % 128 == 0:
                            stage_13_v478__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v476__rv_v0, view_98__ssa_v0, [stage_13_v86__ssa_v0 + 32, 0], [stage_13_v96__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v479__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v478__ssa_v0)
                        else:
                            stage_13_v480__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v476__rv_v0, view_99__ssa_v0, [stage_13_v86__ssa_v0 + 32, 0], [stage_13_v96__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v479__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v480__ssa_v0)
                        stage_13_v97__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v479__rv_v0)
                        stage_13_v98__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v97__ssa_v0, 0, stage_13_v86__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v99__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_20, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v84__ssa_v0
                        stage_13_v100__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_24, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v101__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_25, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v99__ssa_v0
                        )
                        stage_13_v102__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_24, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v100__ssa_v0, stage_13_v101__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v103__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_24, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v102__ssa_v0
                        stage_13_v104__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v98__ssa_v0
                        stage_13_v105__ssa_v0: pl.Scalar[pl.INDEX] = 64
                        stage_13_v106__ssa_v0: pl.Scalar[pl.INDEX] = 128
                        stage_13_v107__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        stage_13_v108__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v106__ssa_v0
                        stage_13_v109__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v108__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v110__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v109__ssa_v0])
                        stage_13_v111__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v110__tile, pl.INDEX), 0)
                        stage_13_v112__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v111__ssa_v0 * 4160
                        if stage_13_v112__ssa_v0 % 128 == 0:
                            stage_13_v481__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v479__rv_v0, view_98__ssa_v0, [stage_13_v107__ssa_v0, 0], [stage_13_v112__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v482__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v481__ssa_v0)
                        else:
                            stage_13_v483__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v479__rv_v0, view_99__ssa_v0, [stage_13_v107__ssa_v0, 0], [stage_13_v112__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v482__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v483__ssa_v0)
                        stage_13_v113__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v106__ssa_v0 + 32
                        stage_13_v114__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v113__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v115__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v114__ssa_v0])
                        stage_13_v116__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v115__tile, pl.INDEX), 0)
                        stage_13_v117__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v116__ssa_v0 * 4160
                        if stage_13_v117__ssa_v0 % 128 == 0:
                            stage_13_v484__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v482__rv_v0, view_98__ssa_v0, [stage_13_v107__ssa_v0 + 32, 0], [stage_13_v117__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v485__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v484__ssa_v0)
                        else:
                            stage_13_v486__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v482__rv_v0, view_99__ssa_v0, [stage_13_v107__ssa_v0 + 32, 0], [stage_13_v117__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v485__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v486__ssa_v0)
                        stage_13_v118__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v485__rv_v0)
                        stage_13_v119__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_28, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v118__ssa_v0, 0, stage_13_v107__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v120__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v104__ssa_v0
                        stage_13_v121__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_29, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v122__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_30, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v103__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v123__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v120__ssa_v0
                        )
                        stage_13_v124__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_32, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v122__ssa_v0)
                        stage_13_v125__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_29, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v121__ssa_v0, stage_13_v123__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v126__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_32, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v124__ssa_v0, 6, 64
                        )
                        stage_13_v127__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v126__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v105__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v128__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_29, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v125__ssa_v0
                        stage_13_v129__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_28, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v119__ssa_v0
                        stage_13_v130__ssa_v0: pl.Scalar[pl.INDEX] = 128
                        stage_13_v131__ssa_v0: pl.Scalar[pl.INDEX] = 192
                        stage_13_v132__ssa_v0: pl.Scalar[pl.INDEX] = 64
                        stage_13_v133__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v131__ssa_v0
                        stage_13_v134__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v133__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v135__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v134__ssa_v0])
                        stage_13_v136__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v135__tile, pl.INDEX), 0)
                        stage_13_v137__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v136__ssa_v0 * 4160
                        if stage_13_v137__ssa_v0 % 128 == 0:
                            stage_13_v487__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v485__rv_v0, view_98__ssa_v0, [stage_13_v132__ssa_v0, 0], [stage_13_v137__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v488__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v487__ssa_v0)
                        else:
                            stage_13_v489__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v485__rv_v0, view_99__ssa_v0, [stage_13_v132__ssa_v0, 0], [stage_13_v137__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v488__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v489__ssa_v0)
                        stage_13_v138__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v131__ssa_v0 + 32
                        stage_13_v139__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v138__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v140__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v139__ssa_v0])
                        stage_13_v141__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v140__tile, pl.INDEX), 0)
                        stage_13_v142__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v141__ssa_v0 * 4160
                        if stage_13_v142__ssa_v0 % 128 == 0:
                            stage_13_v490__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v488__rv_v0, view_98__ssa_v0, [stage_13_v132__ssa_v0 + 32, 0], [stage_13_v142__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v491__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v490__ssa_v0)
                        else:
                            stage_13_v492__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v488__rv_v0, view_99__ssa_v0, [stage_13_v132__ssa_v0 + 32, 0], [stage_13_v142__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v491__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v492__ssa_v0)
                        stage_13_v143__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v491__rv_v0)
                        stage_13_v144__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_35, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v143__ssa_v0, 0, stage_13_v132__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v145__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_28, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v129__ssa_v0
                        stage_13_v146__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_36, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v147__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_37, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v128__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v148__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_38, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v145__ssa_v0
                        )
                        stage_13_v149__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v147__ssa_v0)
                        stage_13_v150__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_36, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v146__ssa_v0, stage_13_v148__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v151__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v149__ssa_v0, 6, 64
                        )
                        stage_13_v152__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v151__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v130__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v153__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_36, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v150__ssa_v0
                        stage_13_v154__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_35, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v144__ssa_v0
                        stage_13_v155__ssa_v0: pl.Scalar[pl.INDEX] = 192
                        stage_13_v156__ssa_v0: pl.Scalar[pl.INDEX] = 256
                        stage_13_v157__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        stage_13_v158__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v156__ssa_v0
                        stage_13_v159__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v158__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v160__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v159__ssa_v0])
                        stage_13_v161__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v160__tile, pl.INDEX), 0)
                        stage_13_v162__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v161__ssa_v0 * 4160
                        if stage_13_v162__ssa_v0 % 128 == 0:
                            stage_13_v493__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v491__rv_v0, view_98__ssa_v0, [stage_13_v157__ssa_v0, 0], [stage_13_v162__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v494__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v493__ssa_v0)
                        else:
                            stage_13_v495__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v491__rv_v0, view_99__ssa_v0, [stage_13_v157__ssa_v0, 0], [stage_13_v162__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v494__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v495__ssa_v0)
                        stage_13_v163__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v156__ssa_v0 + 32
                        stage_13_v164__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v163__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v165__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v164__ssa_v0])
                        stage_13_v166__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v165__tile, pl.INDEX), 0)
                        stage_13_v167__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v166__ssa_v0 * 4160
                        if stage_13_v167__ssa_v0 % 128 == 0:
                            stage_13_v496__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v494__rv_v0, view_98__ssa_v0, [stage_13_v157__ssa_v0 + 32, 0], [stage_13_v167__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v497__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v496__ssa_v0)
                        else:
                            stage_13_v498__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v494__rv_v0, view_99__ssa_v0, [stage_13_v157__ssa_v0 + 32, 0], [stage_13_v167__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v497__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v498__ssa_v0)
                        stage_13_v168__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v497__rv_v0)
                        stage_13_v169__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_42, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v168__ssa_v0, 0, stage_13_v157__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v170__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_35, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v154__ssa_v0
                        stage_13_v171__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_43, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v172__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_44, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v153__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v173__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_45, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v170__ssa_v0
                        )
                        stage_13_v174__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_46, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v172__ssa_v0)
                        stage_13_v175__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_43, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v171__ssa_v0, stage_13_v173__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v176__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_46, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v174__ssa_v0, 6, 64
                        )
                        stage_13_v177__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v176__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v155__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v178__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_43, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v175__ssa_v0
                        stage_13_v179__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_42, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v169__ssa_v0
                        stage_13_v180__ssa_v0: pl.Scalar[pl.INDEX] = 256
                        stage_13_v181__ssa_v0: pl.Scalar[pl.INDEX] = 320
                        stage_13_v182__ssa_v0: pl.Scalar[pl.INDEX] = 64
                        stage_13_v183__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v181__ssa_v0
                        stage_13_v184__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v183__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v185__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v184__ssa_v0])
                        stage_13_v186__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v185__tile, pl.INDEX), 0)
                        stage_13_v187__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v186__ssa_v0 * 4160
                        if stage_13_v187__ssa_v0 % 128 == 0:
                            stage_13_v499__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v497__rv_v0, view_98__ssa_v0, [stage_13_v182__ssa_v0, 0], [stage_13_v187__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v500__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v499__ssa_v0)
                        else:
                            stage_13_v501__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v497__rv_v0, view_99__ssa_v0, [stage_13_v182__ssa_v0, 0], [stage_13_v187__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v500__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v501__ssa_v0)
                        stage_13_v188__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v181__ssa_v0 + 32
                        stage_13_v189__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v188__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v190__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v189__ssa_v0])
                        stage_13_v191__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v190__tile, pl.INDEX), 0)
                        stage_13_v192__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v191__ssa_v0 * 4160
                        if stage_13_v192__ssa_v0 % 128 == 0:
                            stage_13_v502__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v500__rv_v0, view_98__ssa_v0, [stage_13_v182__ssa_v0 + 32, 0], [stage_13_v192__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v503__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v502__ssa_v0)
                        else:
                            stage_13_v504__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v500__rv_v0, view_99__ssa_v0, [stage_13_v182__ssa_v0 + 32, 0], [stage_13_v192__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v503__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v504__ssa_v0)
                        stage_13_v193__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v503__rv_v0)
                        stage_13_v194__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_49, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v193__ssa_v0, 0, stage_13_v182__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v195__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_42, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v179__ssa_v0
                        stage_13_v196__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_50, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v197__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_51, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v178__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v198__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_52, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v195__ssa_v0
                        )
                        stage_13_v199__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_53, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v197__ssa_v0)
                        stage_13_v200__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_50, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v196__ssa_v0, stage_13_v198__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v201__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_53, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v199__ssa_v0, 6, 64
                        )
                        stage_13_v202__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v201__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v180__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v203__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_50, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v200__ssa_v0
                        stage_13_v204__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_49, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v194__ssa_v0
                        stage_13_v205__ssa_v0: pl.Scalar[pl.INDEX] = 320
                        stage_13_v206__ssa_v0: pl.Scalar[pl.INDEX] = 384
                        stage_13_v207__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        stage_13_v208__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v206__ssa_v0
                        stage_13_v209__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v208__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v210__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v209__ssa_v0])
                        stage_13_v211__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v210__tile, pl.INDEX), 0)
                        stage_13_v212__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v211__ssa_v0 * 4160
                        if stage_13_v212__ssa_v0 % 128 == 0:
                            stage_13_v505__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v503__rv_v0, view_98__ssa_v0, [stage_13_v207__ssa_v0, 0], [stage_13_v212__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v506__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v505__ssa_v0)
                        else:
                            stage_13_v507__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v503__rv_v0, view_99__ssa_v0, [stage_13_v207__ssa_v0, 0], [stage_13_v212__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v506__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v507__ssa_v0)
                        stage_13_v213__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v206__ssa_v0 + 32
                        stage_13_v214__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v213__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v215__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v214__ssa_v0])
                        stage_13_v216__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v215__tile, pl.INDEX), 0)
                        stage_13_v217__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v216__ssa_v0 * 4160
                        if stage_13_v217__ssa_v0 % 128 == 0:
                            stage_13_v508__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v506__rv_v0, view_98__ssa_v0, [stage_13_v207__ssa_v0 + 32, 0], [stage_13_v217__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v509__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v508__ssa_v0)
                        else:
                            stage_13_v510__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v506__rv_v0, view_99__ssa_v0, [stage_13_v207__ssa_v0 + 32, 0], [stage_13_v217__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v509__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v510__ssa_v0)
                        stage_13_v218__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v509__rv_v0)
                        stage_13_v219__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_56, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v218__ssa_v0, 0, stage_13_v207__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v220__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_49, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v204__ssa_v0
                        stage_13_v221__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_57, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v222__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_58, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v203__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v223__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_59, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v220__ssa_v0
                        )
                        stage_13_v224__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_60, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v222__ssa_v0)
                        stage_13_v225__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_57, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v221__ssa_v0, stage_13_v223__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v226__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_60, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v224__ssa_v0, 6, 64
                        )
                        stage_13_v227__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v226__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v205__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v228__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_57, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v225__ssa_v0
                        stage_13_v229__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_56, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v219__ssa_v0
                        stage_13_v230__ssa_v0: pl.Scalar[pl.INDEX] = 384
                        stage_13_v231__ssa_v0: pl.Scalar[pl.INDEX] = 448
                        stage_13_v232__ssa_v0: pl.Scalar[pl.INDEX] = 64
                        stage_13_v233__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v231__ssa_v0
                        stage_13_v234__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v233__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v235__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v234__ssa_v0])
                        stage_13_v236__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v235__tile, pl.INDEX), 0)
                        stage_13_v237__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v236__ssa_v0 * 4160
                        if stage_13_v237__ssa_v0 % 128 == 0:
                            stage_13_v511__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v509__rv_v0, view_98__ssa_v0, [stage_13_v232__ssa_v0, 0], [stage_13_v237__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v512__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v511__ssa_v0)
                        else:
                            stage_13_v513__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v509__rv_v0, view_99__ssa_v0, [stage_13_v232__ssa_v0, 0], [stage_13_v237__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v512__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v513__ssa_v0)
                        stage_13_v238__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v231__ssa_v0 + 32
                        stage_13_v239__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v238__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v240__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v239__ssa_v0])
                        stage_13_v241__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v240__tile, pl.INDEX), 0)
                        stage_13_v242__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v241__ssa_v0 * 4160
                        if stage_13_v242__ssa_v0 % 128 == 0:
                            stage_13_v514__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v512__rv_v0, view_98__ssa_v0, [stage_13_v232__ssa_v0 + 32, 0], [stage_13_v242__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v515__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v514__ssa_v0)
                        else:
                            stage_13_v516__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v512__rv_v0, view_99__ssa_v0, [stage_13_v232__ssa_v0 + 32, 0], [stage_13_v242__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v515__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v516__ssa_v0)
                        stage_13_v243__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v515__rv_v0)
                        stage_13_v244__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_63, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v243__ssa_v0, 0, stage_13_v232__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v245__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_56, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v229__ssa_v0
                        stage_13_v246__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_64, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v247__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_65, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v228__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v248__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_66, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v245__ssa_v0
                        )
                        stage_13_v249__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_67, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v247__ssa_v0)
                        stage_13_v250__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_64, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v246__ssa_v0, stage_13_v248__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v251__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_67, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v249__ssa_v0, 6, 64
                        )
                        stage_13_v252__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v251__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v230__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v253__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_64, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v250__ssa_v0
                        stage_13_v254__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_63, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v244__ssa_v0
                        stage_13_v255__ssa_v0: pl.Scalar[pl.INDEX] = 448
                        stage_13_v256__ssa_v0: pl.Scalar[pl.INDEX] = 512
                        stage_13_v257__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        stage_13_v258__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v256__ssa_v0
                        stage_13_v259__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v258__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v260__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v259__ssa_v0])
                        stage_13_v261__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v260__tile, pl.INDEX), 0)
                        stage_13_v262__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v261__ssa_v0 * 4160
                        if stage_13_v262__ssa_v0 % 128 == 0:
                            stage_13_v517__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v515__rv_v0, view_98__ssa_v0, [stage_13_v257__ssa_v0, 0], [stage_13_v262__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v518__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v517__ssa_v0)
                        else:
                            stage_13_v519__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v515__rv_v0, view_99__ssa_v0, [stage_13_v257__ssa_v0, 0], [stage_13_v262__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v518__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v519__ssa_v0)
                        stage_13_v263__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v256__ssa_v0 + 32
                        stage_13_v264__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v263__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v265__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v264__ssa_v0])
                        stage_13_v266__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v265__tile, pl.INDEX), 0)
                        stage_13_v267__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v266__ssa_v0 * 4160
                        if stage_13_v267__ssa_v0 % 128 == 0:
                            stage_13_v520__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v518__rv_v0, view_98__ssa_v0, [stage_13_v257__ssa_v0 + 32, 0], [stage_13_v267__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v521__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v520__ssa_v0)
                        else:
                            stage_13_v522__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v518__rv_v0, view_99__ssa_v0, [stage_13_v257__ssa_v0 + 32, 0], [stage_13_v267__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v521__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v522__ssa_v0)
                        stage_13_v268__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v521__rv_v0)
                        stage_13_v269__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_70, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v268__ssa_v0, 0, stage_13_v257__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v270__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_63, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v254__ssa_v0
                        stage_13_v271__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_71, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v272__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_72, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v253__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v273__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_73, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v270__ssa_v0
                        )
                        stage_13_v274__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_74, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v272__ssa_v0)
                        stage_13_v275__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_71, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v271__ssa_v0, stage_13_v273__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v276__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_74, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v274__ssa_v0, 6, 64
                        )
                        stage_13_v277__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v276__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v255__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v278__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_71, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v275__ssa_v0
                        stage_13_v279__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_70, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v269__ssa_v0
                        stage_13_v280__ssa_v0: pl.Scalar[pl.INDEX] = 512
                        stage_13_v281__ssa_v0: pl.Scalar[pl.INDEX] = 576
                        stage_13_v282__ssa_v0: pl.Scalar[pl.INDEX] = 64
                        stage_13_v283__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v281__ssa_v0
                        stage_13_v284__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v283__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v285__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v284__ssa_v0])
                        stage_13_v286__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v285__tile, pl.INDEX), 0)
                        stage_13_v287__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v286__ssa_v0 * 4160
                        if stage_13_v287__ssa_v0 % 128 == 0:
                            stage_13_v523__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v521__rv_v0, view_98__ssa_v0, [stage_13_v282__ssa_v0, 0], [stage_13_v287__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v524__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v523__ssa_v0)
                        else:
                            stage_13_v525__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v521__rv_v0, view_99__ssa_v0, [stage_13_v282__ssa_v0, 0], [stage_13_v287__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v524__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v525__ssa_v0)
                        stage_13_v288__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v281__ssa_v0 + 32
                        stage_13_v289__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v288__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v290__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v289__ssa_v0])
                        stage_13_v291__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v290__tile, pl.INDEX), 0)
                        stage_13_v292__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v291__ssa_v0 * 4160
                        if stage_13_v292__ssa_v0 % 128 == 0:
                            stage_13_v526__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v524__rv_v0, view_98__ssa_v0, [stage_13_v282__ssa_v0 + 32, 0], [stage_13_v292__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v527__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v526__ssa_v0)
                        else:
                            stage_13_v528__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v524__rv_v0, view_99__ssa_v0, [stage_13_v282__ssa_v0 + 32, 0], [stage_13_v292__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v527__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v528__ssa_v0)
                        stage_13_v293__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v527__rv_v0)
                        stage_13_v294__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_77, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v293__ssa_v0, 0, stage_13_v282__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v295__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_70, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v279__ssa_v0
                        stage_13_v296__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_78, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v297__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_79, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v278__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v298__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_80, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v295__ssa_v0
                        )
                        stage_13_v299__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_81, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v297__ssa_v0)
                        stage_13_v300__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_78, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v296__ssa_v0, stage_13_v298__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v301__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_81, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v299__ssa_v0, 6, 64
                        )
                        stage_13_v302__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v301__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v280__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v303__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_78, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v300__ssa_v0
                        stage_13_v304__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_77, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v294__ssa_v0
                        stage_13_v305__ssa_v0: pl.Scalar[pl.INDEX] = 576
                        stage_13_v306__ssa_v0: pl.Scalar[pl.INDEX] = 640
                        stage_13_v307__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        stage_13_v308__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v306__ssa_v0
                        stage_13_v309__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v308__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v310__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v309__ssa_v0])
                        stage_13_v311__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v310__tile, pl.INDEX), 0)
                        stage_13_v312__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v311__ssa_v0 * 4160
                        if stage_13_v312__ssa_v0 % 128 == 0:
                            stage_13_v529__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v527__rv_v0, view_98__ssa_v0, [stage_13_v307__ssa_v0, 0], [stage_13_v312__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v530__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v529__ssa_v0)
                        else:
                            stage_13_v531__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v527__rv_v0, view_99__ssa_v0, [stage_13_v307__ssa_v0, 0], [stage_13_v312__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v530__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v531__ssa_v0)
                        stage_13_v313__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v306__ssa_v0 + 32
                        stage_13_v314__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v313__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v315__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v314__ssa_v0])
                        stage_13_v316__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v315__tile, pl.INDEX), 0)
                        stage_13_v317__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v316__ssa_v0 * 4160
                        if stage_13_v317__ssa_v0 % 128 == 0:
                            stage_13_v532__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v530__rv_v0, view_98__ssa_v0, [stage_13_v307__ssa_v0 + 32, 0], [stage_13_v317__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v533__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v532__ssa_v0)
                        else:
                            stage_13_v534__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v530__rv_v0, view_99__ssa_v0, [stage_13_v307__ssa_v0 + 32, 0], [stage_13_v317__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v533__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v534__ssa_v0)
                        stage_13_v318__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v533__rv_v0)
                        stage_13_v319__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_84, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v318__ssa_v0, 0, stage_13_v307__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v320__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_77, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v304__ssa_v0
                        stage_13_v321__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_85, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v322__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_86, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v303__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v323__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_87, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v320__ssa_v0
                        )
                        stage_13_v324__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_88, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v322__ssa_v0)
                        stage_13_v325__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_85, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v321__ssa_v0, stage_13_v323__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v326__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_88, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v324__ssa_v0, 6, 64
                        )
                        stage_13_v327__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v326__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v305__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v328__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_85, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v325__ssa_v0
                        stage_13_v329__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_84, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v319__ssa_v0
                        stage_13_v330__ssa_v0: pl.Scalar[pl.INDEX] = 640
                        stage_13_v331__ssa_v0: pl.Scalar[pl.INDEX] = 704
                        stage_13_v332__ssa_v0: pl.Scalar[pl.INDEX] = 64
                        stage_13_v333__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v331__ssa_v0
                        stage_13_v334__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v333__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v335__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v334__ssa_v0])
                        stage_13_v336__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v335__tile, pl.INDEX), 0)
                        stage_13_v337__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v336__ssa_v0 * 4160
                        if stage_13_v337__ssa_v0 % 128 == 0:
                            stage_13_v535__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v533__rv_v0, view_98__ssa_v0, [stage_13_v332__ssa_v0, 0], [stage_13_v337__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v536__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v535__ssa_v0)
                        else:
                            stage_13_v537__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v533__rv_v0, view_99__ssa_v0, [stage_13_v332__ssa_v0, 0], [stage_13_v337__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v536__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v537__ssa_v0)
                        stage_13_v338__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v331__ssa_v0 + 32
                        stage_13_v339__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v338__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v340__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v339__ssa_v0])
                        stage_13_v341__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v340__tile, pl.INDEX), 0)
                        stage_13_v342__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v341__ssa_v0 * 4160
                        if stage_13_v342__ssa_v0 % 128 == 0:
                            stage_13_v538__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v536__rv_v0, view_98__ssa_v0, [stage_13_v332__ssa_v0 + 32, 0], [stage_13_v342__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v539__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v538__ssa_v0)
                        else:
                            stage_13_v540__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v536__rv_v0, view_99__ssa_v0, [stage_13_v332__ssa_v0 + 32, 0], [stage_13_v342__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v539__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v540__ssa_v0)
                        stage_13_v343__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v539__rv_v0)
                        stage_13_v344__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_91, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v343__ssa_v0, 0, stage_13_v332__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v345__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_84, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v329__ssa_v0
                        stage_13_v346__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_92, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v347__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_93, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v328__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v348__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_94, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v345__ssa_v0
                        )
                        stage_13_v349__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_95, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v347__ssa_v0)
                        stage_13_v350__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_92, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v346__ssa_v0, stage_13_v348__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v351__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_95, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v349__ssa_v0, 6, 64
                        )
                        stage_13_v352__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v351__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v330__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v353__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_92, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v350__ssa_v0
                        stage_13_v354__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_91, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v344__ssa_v0
                        stage_13_v355__ssa_v0: pl.Scalar[pl.INDEX] = 704
                        stage_13_v356__ssa_v0: pl.Scalar[pl.INDEX] = 768
                        stage_13_v357__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        stage_13_v358__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v356__ssa_v0
                        stage_13_v359__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v358__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v360__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v359__ssa_v0])
                        stage_13_v361__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v360__tile, pl.INDEX), 0)
                        stage_13_v362__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v361__ssa_v0 * 4160
                        if stage_13_v362__ssa_v0 % 128 == 0:
                            stage_13_v541__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v539__rv_v0, view_98__ssa_v0, [stage_13_v357__ssa_v0, 0], [stage_13_v362__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v542__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v541__ssa_v0)
                        else:
                            stage_13_v543__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v539__rv_v0, view_99__ssa_v0, [stage_13_v357__ssa_v0, 0], [stage_13_v362__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v542__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v543__ssa_v0)
                        stage_13_v363__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v356__ssa_v0 + 32
                        stage_13_v364__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v363__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v365__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v364__ssa_v0])
                        stage_13_v366__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v365__tile, pl.INDEX), 0)
                        stage_13_v367__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v366__ssa_v0 * 4160
                        if stage_13_v367__ssa_v0 % 128 == 0:
                            stage_13_v544__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v542__rv_v0, view_98__ssa_v0, [stage_13_v357__ssa_v0 + 32, 0], [stage_13_v367__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v545__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v544__ssa_v0)
                        else:
                            stage_13_v546__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v542__rv_v0, view_99__ssa_v0, [stage_13_v357__ssa_v0 + 32, 0], [stage_13_v367__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v545__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v546__ssa_v0)
                        stage_13_v368__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v545__rv_v0)
                        stage_13_v369__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_98, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v368__ssa_v0, 0, stage_13_v357__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v370__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_91, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v354__ssa_v0
                        stage_13_v371__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_99, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v372__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_100, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v353__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v373__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_101, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v370__ssa_v0
                        )
                        stage_13_v374__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_102, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v372__ssa_v0)
                        stage_13_v375__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_99, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v371__ssa_v0, stage_13_v373__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v376__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_102, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v374__ssa_v0, 6, 64
                        )
                        stage_13_v377__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v376__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v355__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v378__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_99, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v375__ssa_v0
                        stage_13_v379__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_98, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v369__ssa_v0
                        stage_13_v380__ssa_v0: pl.Scalar[pl.INDEX] = 768
                        stage_13_v381__ssa_v0: pl.Scalar[pl.INDEX] = 832
                        stage_13_v382__ssa_v0: pl.Scalar[pl.INDEX] = 64
                        stage_13_v383__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v381__ssa_v0
                        stage_13_v384__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v383__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v385__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v384__ssa_v0])
                        stage_13_v386__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v385__tile, pl.INDEX), 0)
                        stage_13_v387__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v386__ssa_v0 * 4160
                        if stage_13_v387__ssa_v0 % 128 == 0:
                            stage_13_v547__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v545__rv_v0, view_98__ssa_v0, [stage_13_v382__ssa_v0, 0], [stage_13_v387__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v548__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v547__ssa_v0)
                        else:
                            stage_13_v549__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v545__rv_v0, view_99__ssa_v0, [stage_13_v382__ssa_v0, 0], [stage_13_v387__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v548__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v549__ssa_v0)
                        stage_13_v388__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v381__ssa_v0 + 32
                        stage_13_v389__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v388__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v390__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v389__ssa_v0])
                        stage_13_v391__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v390__tile, pl.INDEX), 0)
                        stage_13_v392__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v391__ssa_v0 * 4160
                        if stage_13_v392__ssa_v0 % 128 == 0:
                            stage_13_v550__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v548__rv_v0, view_98__ssa_v0, [stage_13_v382__ssa_v0 + 32, 0], [stage_13_v392__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v551__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v550__ssa_v0)
                        else:
                            stage_13_v552__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v548__rv_v0, view_99__ssa_v0, [stage_13_v382__ssa_v0 + 32, 0], [stage_13_v392__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v551__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v552__ssa_v0)
                        stage_13_v393__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v551__rv_v0)
                        stage_13_v394__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_105, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v393__ssa_v0, 0, stage_13_v382__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v395__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_98, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v379__ssa_v0
                        stage_13_v396__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_106, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v397__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_107, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v378__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v398__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_108, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v395__ssa_v0
                        )
                        stage_13_v399__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_109, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v397__ssa_v0)
                        stage_13_v400__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_106, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v396__ssa_v0, stage_13_v398__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v401__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_109, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v399__ssa_v0, 6, 64
                        )
                        stage_13_v402__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v401__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v380__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v403__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_106, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v400__ssa_v0
                        stage_13_v404__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_105, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v394__ssa_v0
                        stage_13_v405__ssa_v0: pl.Scalar[pl.INDEX] = 832
                        stage_13_v406__ssa_v0: pl.Scalar[pl.INDEX] = 896
                        stage_13_v407__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        stage_13_v408__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v406__ssa_v0
                        stage_13_v409__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v408__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v410__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v409__ssa_v0])
                        stage_13_v411__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v410__tile, pl.INDEX), 0)
                        stage_13_v412__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v411__ssa_v0 * 4160
                        if stage_13_v412__ssa_v0 % 128 == 0:
                            stage_13_v553__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v551__rv_v0, view_98__ssa_v0, [stage_13_v407__ssa_v0, 0], [stage_13_v412__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v554__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v553__ssa_v0)
                        else:
                            stage_13_v555__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v551__rv_v0, view_99__ssa_v0, [stage_13_v407__ssa_v0, 0], [stage_13_v412__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v554__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v555__ssa_v0)
                        stage_13_v413__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v406__ssa_v0 + 32
                        stage_13_v414__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v413__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v415__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v414__ssa_v0])
                        stage_13_v416__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v415__tile, pl.INDEX), 0)
                        stage_13_v417__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v416__ssa_v0 * 4160
                        if stage_13_v417__ssa_v0 % 128 == 0:
                            stage_13_v556__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v554__rv_v0, view_98__ssa_v0, [stage_13_v407__ssa_v0 + 32, 0], [stage_13_v417__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v557__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v556__ssa_v0)
                        else:
                            stage_13_v558__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v554__rv_v0, view_99__ssa_v0, [stage_13_v407__ssa_v0 + 32, 0], [stage_13_v417__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v557__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v558__ssa_v0)
                        stage_13_v418__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v557__rv_v0)
                        stage_13_v419__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_112, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v418__ssa_v0, 0, stage_13_v407__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v420__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_105, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v404__ssa_v0
                        stage_13_v421__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_113, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v422__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_114, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v403__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v423__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_115, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v420__ssa_v0
                        )
                        stage_13_v424__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_116, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v422__ssa_v0)
                        stage_13_v425__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_113, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v421__ssa_v0, stage_13_v423__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v426__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_116, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v424__ssa_v0, 6, 64
                        )
                        stage_13_v427__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v426__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v405__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v428__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_113, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v425__ssa_v0
                        stage_13_v429__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_112, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v419__ssa_v0
                        stage_13_v430__ssa_v0: pl.Scalar[pl.INDEX] = 896
                        stage_13_v431__ssa_v0: pl.Scalar[pl.INDEX] = 960
                        stage_13_v432__ssa_v0: pl.Scalar[pl.INDEX] = 64
                        stage_13_v433__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v431__ssa_v0
                        stage_13_v434__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v433__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v435__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v434__ssa_v0])
                        stage_13_v436__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v435__tile, pl.INDEX), 0)
                        stage_13_v437__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v436__ssa_v0 * 4160
                        if stage_13_v437__ssa_v0 % 128 == 0:
                            stage_13_v559__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v557__rv_v0, view_98__ssa_v0, [stage_13_v432__ssa_v0, 0], [stage_13_v437__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v560__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v559__ssa_v0)
                        else:
                            stage_13_v561__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v557__rv_v0, view_99__ssa_v0, [stage_13_v432__ssa_v0, 0], [stage_13_v437__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v560__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v561__ssa_v0)
                        stage_13_v438__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v37__ssa_v0 + stage_13_v69__ssa_v0 + stage_13_v431__ssa_v0 + 32
                        stage_13_v439__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_13_v438__ssa_v0 // 32, pl.max((stage_13_v35__ssa_v0 - 1) // 32, 0))
                        stage_13_v440__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_idx_block_table__ssa_v0, [stage_13_v32__ssa_v0, stage_13_v439__ssa_v0])
                        stage_13_v441__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(stage_13_v440__tile, pl.INDEX), 0)
                        stage_13_v442__ssa_v0: pl.Scalar[pl.INDEX] = stage_13_v441__ssa_v0 * 4160
                        if stage_13_v442__ssa_v0 % 128 == 0:
                            stage_13_v562__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v560__rv_v0, view_98__ssa_v0, [stage_13_v432__ssa_v0 + 32, 0], [stage_13_v442__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v563__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v562__ssa_v0)
                        else:
                            stage_13_v564__ssa_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.gather_row(
                                stage_13_v560__rv_v0, view_99__ssa_v0, [stage_13_v432__ssa_v0 + 32, 0], [stage_13_v442__ssa_v0 // 128, 0], [32, 128], transpose=False
                            )
                            stage_13_v563__rv_v0: pl.Tile[[128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.yield_(stage_13_v564__ssa_v0)
                        stage_13_v443__ssa_v0: pl.Tile[
                            [128, 128], pl.INT8, pl.MemRef(mem_mat_17, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(stage_13_v563__rv_v0)
                        stage_13_v444__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_119, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v443__ssa_v0, 0, stage_13_v432__ssa_v0, [128, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v445__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_112, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v429__ssa_v0
                        stage_13_v446__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_120, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v447__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_121, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v428__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v448__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_122, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v445__ssa_v0
                        )
                        stage_13_v449__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_123, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v447__ssa_v0)
                        stage_13_v450__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_120, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v446__ssa_v0, stage_13_v448__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v451__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_123, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v449__ssa_v0, 6, 64
                        )
                        stage_13_v452__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v451__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v430__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v453__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_120, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v450__ssa_v0
                        stage_13_v454__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_119, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v444__ssa_v0
                        stage_13_v455__ssa_v0: pl.Scalar[pl.INDEX] = 960
                        stage_13_v456__ssa_v0: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_119, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = stage_13_v454__ssa_v0
                        stage_13_v457__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_124, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                            [384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                        )
                        stage_13_v458__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_125, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v453__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v459__ssa_v0: pl.Tile[[384, 64], pl.INT32, pl.MemRef(mem_acc_126, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul(
                            stage_13_v58__ssa_v0, stage_13_v456__ssa_v0
                        )
                        stage_13_v460__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_127, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v458__ssa_v0)
                        stage_13_v461__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_124, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.assemble(
                            stage_13_v457__ssa_v0, stage_13_v459__ssa_v0, [0, 0], pre_quant=0.0009765625, pre_relu=True
                        )
                        stage_13_v462__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_127, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v460__ssa_v0, 6, 64
                        )
                        stage_13_v463__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v462__ssa_v0, [stage_13_v70__ssa_v0, stage_13_v455__ssa_v0 - 64], temporary_49__ssa_v0
                        )
                        stage_13_v464__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_mat_124, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = stage_13_v461__ssa_v0
                        stage_13_v465__ssa_v0: pl.Tile[[384, 64], pl.FP16, pl.MemRef(mem_right_128, pl.const(0, pl.INT64), 49152), pl.Mem.Right] = pl.tile.extract(
                            stage_13_v464__ssa_v0, 0, 0, [384, 64], target_memory=pl.Mem.Right
                        )
                        stage_13_v466__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_129, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(stage_13_v59__ssa_v0, stage_13_v465__ssa_v0)
                        stage_13_v467__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_129, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[6, 64])] = pl.tile.set_validshape(
                            stage_13_v466__ssa_v0, 6, 64
                        )
                        stage_13_v468__ssa_v0: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_13_v467__ssa_v0, [stage_13_v70__ssa_v0, 960], temporary_49__ssa_v0
                        )
                        pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                    for stage_13_v61__idx_v0 in pl.range(pl.min(stage_13_v55__ssa_v0, 2)):
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
            pl.system.fence()
            pl.system.cacheinvalid()
            pl.system.fence()
        record_15__ssa_v0: pl.Scalar[pl.INDEX] = core_id__ssa_v0 * 6 + 6480
        t__tile_2: pl.Scalar[pl.INT64] = pl.tensor.read(records__ssa_v0, [record_15__ssa_v0])
        mapped_block_15__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
        block_15__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(mapped_block_15__ssa_v0, 0)
        if 0 <= mapped_block_15__ssa_v0:
            t__tile_3: pl.Scalar[pl.INT64] = pl.tensor.read(records__ssa_v0, [record_15__ssa_v0 + 1])
            subslot_15__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_3, pl.INDEX)
            stage_15_v4__ssa_v0: pl.Scalar[pl.INDEX] = block_15__ssa_v0
            stage_15_v5__ssa_v0: pl.Scalar[pl.INDEX] = stage_15_v4__ssa_v0 // 8
            stage_15_v6__ssa_v0: pl.Scalar[pl.INDEX] = stage_15_v4__ssa_v0 % 8
            stage_15_v7__ssa_v0: pl.Scalar[pl.INDEX] = stage_15_v5__ssa_v0 * 128
            stage_15_v8__ssa_v0: pl.Scalar[pl.INDEX] = 24 - stage_15_v7__ssa_v0
            stage_15_v9__ssa_v0: pl.Scalar[pl.INDEX] = stage_15_v7__ssa_v0 + 2688
            stage_15_v10__ssa_v0: pl.Scalar[pl.INDEX] = stage_15_v6__ssa_v0 * 128
            stage_15_v11__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_130, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[stage_15_v8__ssa_v0, 256])] = pl.tile.load(
                temporary_51__ssa_v0, [stage_15_v9__ssa_v0, 0], [128, 256], [stage_15_v8__ssa_v0, 256], target_memory=pl.Mem.Mat
            )
            stage_15_v12__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_131, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                arg_wo_a__ssa_v0, [7, stage_15_v10__ssa_v0 // 16, 0, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat
            )
            stage_15_v13__tile_l0_init_storage: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = (
                pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            )
            stage_15_v13__tile_l0_init: pl.Tile[
                [128, 128],
                pl.FP32,
                pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                pl.Mem.Acc,
                pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
            ] = pl.tile.set_validshape(stage_15_v13__tile_l0_init_storage, pl.min(stage_15_v8__ssa_v0, 128), 128)
            for stage_15_v13__tile_l0_ko, (stage_15_v13__tile_l0_c,) in pl.range(0, 256, 256, init_values=(stage_15_v13__tile_l0_init,)):
                stage_15_v13__tile_l0_a: pl.Tile[
                    [128, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_133, pl.const(0, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(stage_15_v11__tile, 0, stage_15_v13__tile_l0_ko, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "0:0"})
                stage_15_v13__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_134, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    stage_15_v12__tile, stage_15_v13__tile_l0_ko, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:0"}
                )
                stage_15_v13__tile_l0_a_1: pl.Tile[
                    [128, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_135, pl.const(0, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(stage_15_v11__tile, 0, stage_15_v13__tile_l0_ko + 128, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "0:1"})
                stage_15_v13__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_136, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    stage_15_v12__tile, stage_15_v13__tile_l0_ko + 128, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:1"}
                )
                stage_15_v13__tile_l0_c_acc: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.tile.matmul_acc(stage_15_v13__tile_l0_c, stage_15_v13__tile_l0_a, stage_15_v13__tile_l0_b, stage_15_v13__tile_l0_ko == 0)
                stage_15_v13__tile_l0_c_acc_1: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.tile.matmul_acc(stage_15_v13__tile_l0_c_acc, stage_15_v13__tile_l0_a_1, stage_15_v13__tile_l0_b_1, stage_15_v13__tile_l0_ko + 128 == 0)
                stage_15_v13__tile: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.yield_(stage_15_v13__tile_l0_c_acc_1)
            for stage_15_v15__idx_v0, (stage_15_v21,) in pl.range(1, 15, 2, init_values=(stage_15_v13__tile,)):
                stage_15_v16__ssa_v0: pl.Scalar[pl.INDEX] = stage_15_v15__idx_v0 * 256
                stage_15_v16__ssa_v0_1: pl.Scalar[pl.INDEX] = (stage_15_v15__idx_v0 + 1) * 256
                stage_15_v17__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_138, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[stage_15_v8__ssa_v0, 256])] = pl.tile.load(
                    temporary_51__ssa_v0, [stage_15_v9__ssa_v0, stage_15_v16__ssa_v0], [128, 256], [stage_15_v8__ssa_v0, 256], target_memory=pl.Mem.Mat, attrs={"pipeline_membership": "2:0"}
                )
                stage_15_v18__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_139, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    arg_wo_a__ssa_v0,
                    [7, stage_15_v10__ssa_v0 // 16, stage_15_v16__ssa_v0 // 16, 0, 0],
                    [1, 8, 16, 16, 16],
                    [1, 8, 16, 16, 16],
                    target_memory=pl.Mem.Mat,
                    attrs={"pipeline_membership": "2:0"},
                )
                stage_15_v17__tile_1: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_140, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[stage_15_v8__ssa_v0, 256])] = (
                    pl.tile.load(
                        temporary_51__ssa_v0, [stage_15_v9__ssa_v0, stage_15_v16__ssa_v0_1], [128, 256], [stage_15_v8__ssa_v0, 256], target_memory=pl.Mem.Mat, attrs={"pipeline_membership": "2:1"}
                    )
                )
                stage_15_v18__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_141, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    arg_wo_a__ssa_v0,
                    [7, stage_15_v10__ssa_v0 // 16, stage_15_v16__ssa_v0_1 // 16, 0, 0],
                    [1, 8, 16, 16, 16],
                    [1, 8, 16, 16, 16],
                    target_memory=pl.Mem.Mat,
                    attrs={"pipeline_membership": "2:1"},
                )
                for stage_15_v21_l0_ko, (stage_15_v21_l0_c,) in pl.range(0, 256, 256, init_values=(stage_15_v21,)):
                    stage_15_v21_l0_a: pl.Tile[
                        [128, 128],
                        pl.BF16,
                        pl.MemRef(mem_left_142, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(stage_15_v17__tile, 0, stage_15_v21_l0_ko, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "1:0;2:0"})
                    stage_15_v21_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_143, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        stage_15_v18__tile, stage_15_v21_l0_ko, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:0;2:0"}
                    )
                    stage_15_v21_l0_a_1: pl.Tile[
                        [128, 128],
                        pl.BF16,
                        pl.MemRef(mem_left_144, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(stage_15_v17__tile, 0, stage_15_v21_l0_ko + 128, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "1:1;2:0"})
                    stage_15_v21_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_145, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        stage_15_v18__tile, stage_15_v21_l0_ko + 128, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:1;2:0"}
                    )
                    stage_15_v21_l0_c_acc: pl.Tile[
                        [128, 128],
                        pl.FP32,
                        pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(stage_15_v21_l0_c, stage_15_v21_l0_a, stage_15_v21_l0_b)
                    stage_15_v21_l0_c_acc_1: pl.Tile[
                        [128, 128],
                        pl.FP32,
                        pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(stage_15_v21_l0_c_acc, stage_15_v21_l0_a_1, stage_15_v21_l0_b_1)
                    stage_15_v21_1: pl.Tile[
                        [128, 128],
                        pl.FP32,
                        pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                    ] = pl.yield_(stage_15_v21_l0_c_acc_1)
                for stage_15_v21_l0_ko_1, (stage_15_v21_l0_c_1,) in pl.range(0, 256, 256, init_values=(stage_15_v21_1,)):
                    stage_15_v21_l0_a_2: pl.Tile[
                        [128, 128],
                        pl.BF16,
                        pl.MemRef(mem_left_147, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(stage_15_v17__tile_1, 0, stage_15_v21_l0_ko_1, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "1:0;2:1"})
                    stage_15_v21_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_148, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        stage_15_v18__tile_1, stage_15_v21_l0_ko_1, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:0;2:1"}
                    )
                    stage_15_v21_l0_a_3: pl.Tile[
                        [128, 128],
                        pl.BF16,
                        pl.MemRef(mem_left_149, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(stage_15_v17__tile_1, 0, stage_15_v21_l0_ko_1 + 128, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "1:1;2:1"})
                    stage_15_v21_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_150, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        stage_15_v18__tile_1, stage_15_v21_l0_ko_1 + 128, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:1;2:1"}
                    )
                    stage_15_v21_l0_c_acc_2: pl.Tile[
                        [128, 128],
                        pl.FP32,
                        pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(stage_15_v21_l0_c_1, stage_15_v21_l0_a_2, stage_15_v21_l0_b_2)
                    stage_15_v21_l0_c_acc_3: pl.Tile[
                        [128, 128],
                        pl.FP32,
                        pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(stage_15_v21_l0_c_acc_2, stage_15_v21_l0_a_3, stage_15_v21_l0_b_3)
                    stage_15_v21_2: pl.Tile[
                        [128, 128],
                        pl.FP32,
                        pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                    ] = pl.yield_(stage_15_v21_l0_c_acc_3)
                stage_15_v20_main: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.yield_(stage_15_v21_2)
            stage_15_v16__ssa_v0_2: pl.Scalar[pl.INDEX] = pl.const(15, pl.INDEX) * pl.const(256, pl.INDEX)
            stage_15_v17__tile_2: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_153, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[stage_15_v8__ssa_v0, 256])] = pl.tile.load(
                temporary_51__ssa_v0, [stage_15_v9__ssa_v0, stage_15_v16__ssa_v0_2], [128, 256], [stage_15_v8__ssa_v0, 256], target_memory=pl.Mem.Mat, attrs={"pipeline_membership": "3:0"}
            )
            stage_15_v18__tile_2: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_154, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                arg_wo_a__ssa_v0,
                [7, stage_15_v10__ssa_v0 // 16, stage_15_v16__ssa_v0_2 // 16, 0, 0],
                [1, 8, 16, 16, 16],
                [1, 8, 16, 16, 16],
                target_memory=pl.Mem.Mat,
                attrs={"pipeline_membership": "3:0"},
            )
            for stage_15_v21_l0_ko_2, (stage_15_v21_l0_c_2,) in pl.range(0, 256, 256, init_values=(stage_15_v20_main,)):
                stage_15_v21_l0_a_4: pl.Tile[
                    [128, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_155, pl.const(0, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(stage_15_v17__tile_2, 0, stage_15_v21_l0_ko_2, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "1:0;3:0"})
                stage_15_v21_l0_b_4: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_156, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    stage_15_v18__tile_2, stage_15_v21_l0_ko_2, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:0;3:0"}
                )
                stage_15_v21_l0_a_5: pl.Tile[
                    [128, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_157, pl.const(0, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(stage_15_v17__tile_2, 0, stage_15_v21_l0_ko_2 + 128, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "1:1;3:0"})
                stage_15_v21_l0_b_5: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_158, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    stage_15_v18__tile_2, stage_15_v21_l0_ko_2 + 128, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:1;3:0"}
                )
                stage_15_v21_l0_c_acc_4: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.tile.matmul_acc(stage_15_v21_l0_c_2, stage_15_v21_l0_a_4, stage_15_v21_l0_b_4)
                stage_15_v21_l0_c_acc_5: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.tile.matmul_acc(stage_15_v21_l0_c_acc_4, stage_15_v21_l0_a_5, stage_15_v21_l0_b_5)
                stage_15_v21_3: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.yield_(stage_15_v21_l0_c_acc_5)
            stage_15_v20: pl.Tile[
                [128, 128],
                pl.FP32,
                pl.MemRef(mem_acc_132, pl.const(0, pl.INT64), 65536),
                pl.Mem.Acc,
                pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
            ] = stage_15_v21_3
            stage_15_v14__tile: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
                stage_15_v20, [stage_15_v7__ssa_v0, stage_15_v10__ssa_v0 + 7168], temporary_63__ssa_v0
            )
            pl.system.fence()
            pl.system.cacheinvalid()
            pl.system.fence()
        return temporary_39__ssa_v0, temporary_48__ssa_v0, temporary_50__ssa_v0, view_98__ssa_v0, view_99__ssa_v0, temporary_49__ssa_v0, temporary_51__ssa_v0, temporary_63__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_static_full_card(
        self,
        arg_kv_seq_lens__ssa_v0: pl.Tensor[[4], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16)],
        arg_position_ids__ssa_v0: pl.Tensor[[24], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 192)],
        temporary_39__ssa_v0: pl.InOut[pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 3145728)]],
        temporary_48__ssa_v0: pl.InOut[pl.Tensor[[1024, 384], pl.FP16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 786432)]],
        temporary_50__ssa_v0: pl.InOut[pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2048)]],
        arg_idx_block_table__ssa_v0: pl.Tensor[[4, 1026], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16416)],
        view_98__ssa_v0: pl.InOut[pl.Tensor[[133412, 128], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 17076736)]],
        view_99__ssa_v0: pl.InOut[pl.Tensor[[133412, 128], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 17076736)]],
        temporary_49__ssa_v0: pl.InOut[pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)]],
        temporary_51__ssa_v0: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 25165824)]],
        arg_wo_a__ssa_v0: pl.Tensor[
            [8, 64, 256, 16, 16], pl.BF16, pl.TensorView(stride=[4194304, 65536, 256, 16, 1], layout=pl.TensorLayout.NZ), pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 67108864)
        ],
        temporary_63__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 12582912)]],
        records__ssa_v0: pl.Tensor[[31968], pl.INT64, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 255744)],
    ) -> tuple[
        pl.Tensor[[24576, 128], pl.INT8],
        pl.Tensor[[1024, 384], pl.FP16],
        pl.Tensor[[256], pl.INT64],
        pl.Tensor[[133412, 128], pl.INT8],
        pl.Tensor[[133412, 128], pl.INT8],
        pl.Tensor[[288, 1024], pl.FP32],
        pl.Tensor[[3072, 4096], pl.BF16],
        pl.Tensor[[384, 8192], pl.FP32],
    ]:
        pl.func_attr({"nz_tensor_views_blocked": True, "mx_tensor_views_blocked": True})
        _tuple_tmp__ssa_v0: pl.Tuple[
            pl.Tensor[[24576, 128], pl.INT8],
            pl.Tensor[[1024, 384], pl.FP16],
            pl.Tensor[[256], pl.INT64],
            pl.Tensor[[133412, 128], pl.INT8],
            pl.Tensor[[133412, 128], pl.INT8],
            pl.Tensor[[288, 1024], pl.FP32],
            pl.Tensor[[3072, 4096], pl.BF16],
            pl.Tensor[[384, 8192], pl.FP32],
        ] = self.csa_static_aic(
            arg_kv_seq_lens__ssa_v0,
            arg_position_ids__ssa_v0,
            temporary_39__ssa_v0,
            temporary_48__ssa_v0,
            temporary_50__ssa_v0,
            arg_idx_block_table__ssa_v0,
            view_98__ssa_v0,
            view_99__ssa_v0,
            temporary_49__ssa_v0,
            temporary_51__ssa_v0,
            arg_wo_a__ssa_v0,
            temporary_63__ssa_v0,
            records__ssa_v0,
        )
        temporary_39__ssa_v1: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 3145728)] = _tuple_tmp__ssa_v0[0]
        temporary_48__ssa_v1: pl.Tensor[[1024, 384], pl.FP16, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 786432)] = _tuple_tmp__ssa_v0[1]
        temporary_50__ssa_v1: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 2048)] = _tuple_tmp__ssa_v0[2]
        view_98__ssa_v1: pl.Tensor[[133412, 128], pl.INT8, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 17076736)] = _tuple_tmp__ssa_v0[3]
        view_99__ssa_v1: pl.Tensor[[133412, 128], pl.INT8, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 17076736)] = _tuple_tmp__ssa_v0[4]
        temporary_49__ssa_v1: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 1179648)] = _tuple_tmp__ssa_v0[5]
        temporary_51__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 25165824)] = _tuple_tmp__ssa_v0[6]
        temporary_63__ssa_v1: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 12582912)] = _tuple_tmp__ssa_v0[7]
        return temporary_39__ssa_v0, temporary_48__ssa_v0, temporary_50__ssa_v0, view_98__ssa_v0, view_99__ssa_v0, temporary_49__ssa_v0, temporary_51__ssa_v0, temporary_63__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(
        self,
        arg_kv_seq_lens__ssa_v0: pl.Tensor[[4], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16)],
        arg_position_ids__ssa_v0: pl.Tensor[[24], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 192)],
        temporary_39__ssa_v0: pl.InOut[pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 3145728)]],
        temporary_48__ssa_v0: pl.InOut[pl.Tensor[[1024, 384], pl.FP16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 786432)]],
        temporary_50__ssa_v0: pl.InOut[pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2048)]],
        arg_idx_block_table__ssa_v0: pl.Tensor[[4, 1026], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16416)],
        view_98__ssa_v0: pl.InOut[pl.Tensor[[133412, 128], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 17076736)]],
        view_99__ssa_v0: pl.InOut[pl.Tensor[[133412, 128], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 17076736)]],
        temporary_49__ssa_v0: pl.InOut[pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1179648)]],
        temporary_51__ssa_v0: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 25165824)]],
        arg_wo_a__ssa_v0: pl.Tensor[
            [8, 64, 256, 16, 16], pl.BF16, pl.TensorView(stride=[4194304, 65536, 256, 16, 1], layout=pl.TensorLayout.NZ), pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 67108864)
        ],
        temporary_63__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 12582912)]],
        records__ssa_v0: pl.Tensor[[31968], pl.INT64, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 255744)],
    ) -> tuple[
        pl.Tensor[[24576, 128], pl.INT8],
        pl.Tensor[[1024, 384], pl.FP16],
        pl.Tensor[[256], pl.INT64],
        pl.Tensor[[133412, 128], pl.INT8],
        pl.Tensor[[133412, 128], pl.INT8],
        pl.Tensor[[288, 1024], pl.FP32],
        pl.Tensor[[3072, 4096], pl.BF16],
        pl.Tensor[[384, 8192], pl.FP32],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True, "nz_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[
            pl.Tensor[[24576, 128], pl.INT8],
            pl.Tensor[[1024, 384], pl.FP16],
            pl.Tensor[[256], pl.INT64],
            pl.Tensor[[133412, 128], pl.INT8],
            pl.Tensor[[133412, 128], pl.INT8],
            pl.Tensor[[288, 1024], pl.FP32],
            pl.Tensor[[3072, 4096], pl.BF16],
            pl.Tensor[[384, 8192], pl.FP32],
        ] = self.csa_static_full_card(
            arg_kv_seq_lens__ssa_v0,
            arg_position_ids__ssa_v0,
            temporary_39__ssa_v0,
            temporary_48__ssa_v0,
            temporary_50__ssa_v0,
            arg_idx_block_table__ssa_v0,
            view_98__ssa_v0,
            view_99__ssa_v0,
            temporary_49__ssa_v0,
            temporary_51__ssa_v0,
            arg_wo_a__ssa_v0,
            temporary_63__ssa_v0,
            records__ssa_v0,
            attrs={"core_num": 24, "sync_start": True},
        )
        temporary_39__ssa_v1: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0[0]
        temporary_48__ssa_v1: pl.Tensor[[1024, 384], pl.FP16, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 786432)] = ret__tmp_v0[1]
        temporary_50__ssa_v1: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 2048)] = ret__tmp_v0[2]
        view_98__ssa_v1: pl.Tensor[[133412, 128], pl.INT8, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 17076736)] = ret__tmp_v0[3]
        view_99__ssa_v1: pl.Tensor[[133412, 128], pl.INT8, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 17076736)] = ret__tmp_v0[4]
        temporary_49__ssa_v1: pl.Tensor[[288, 1024], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 1179648)] = ret__tmp_v0[5]
        temporary_51__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 25165824)] = ret__tmp_v0[6]
        temporary_63__ssa_v1: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0[7]
        return temporary_39__ssa_v1, temporary_48__ssa_v1, temporary_50__ssa_v1, view_98__ssa_v1, view_99__ssa_v1, temporary_49__ssa_v1, temporary_51__ssa_v1, temporary_63__ssa_v1
