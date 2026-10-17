# pypto.program: CSAAllocatorFocusedProbe
import pypto.language as pl


@pl.program
class CSAAllocatorFocusedProbe:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_static_aiv(
        arg_freqs_cos__ssa_v0: pl.Tensor[[24, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6144)],
        view_45__ssa_v0: pl.InOut[pl.Tensor[[1184, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1212416)]],
        view_61__ssa_v0: pl.InOut[pl.Tensor[[131360, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 134512640)]],
        temporary_53__ssa_v0: pl.InOut[pl.Tensor[[24, 512], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 49152)]],
        temporary_52__ssa_v0: pl.InOut[pl.Tensor[[24, 640], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 61440)]],
        view_106__ssa_v0: pl.Tensor[[24, 1], pl.INT64, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 192)],
        temporary_54__ssa_v0: pl.InOut[pl.Tensor[[24, 16], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 1536)]],
        arg_ori_block_table__ssa_v0: pl.Tensor[[4, 4098], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 65568)],
        temporary_55__ssa_v0: pl.InOut[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 98304)]],
        temporary_62__ssa_v0: pl.InOut[pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 2048)]],
        temporary_56__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 9437184)]],
        temporary_57__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 2359296)]],
        temporary_58__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 1179648)]],
        temporary_59__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 9437184)]],
        view_116__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 256)],
        arg_cmp_block_table__ssa_v0: pl.Tensor[[4, 1026], pl.INT32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 16416)],
        temporary_60__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        temporary_61__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 18432)]],
        temporary_51__ssa_v0: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 25165824)]],
        records__ssa_v0: pl.Tensor[[31968], pl.INT64, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 255744)],
    ) -> tuple[
        pl.Tensor[[1184, 512], pl.BF16],
        pl.Tensor[[131360, 512], pl.BF16],
        pl.Tensor[[24, 512], pl.INT32],
        pl.Tensor[[24, 640], pl.FP32],
        pl.Tensor[[24, 16], pl.INT32],
        pl.Tensor[[384, 64], pl.FP32],
        pl.Tensor[[256], pl.INT64],
        pl.Tensor[[9216, 512], pl.BF16],
        pl.Tensor[[4608, 128], pl.FP32],
        pl.Tensor[[4608, 128], pl.BF16],
        pl.Tensor[[4608, 512], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[3072, 4096], pl.BF16],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True, "dual_aiv_dispatch": True})
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 768)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_25: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_26: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_28: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_29: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 768)
        mem_vec_30: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_31: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_32: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_33: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_34: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_35: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_36: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_37: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        mem_vec_48: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        mem_vec_49: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        mem_vec_51: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        mem_vec_55: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_56: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_57: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_58: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_59: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_60: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_61: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_62: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_63: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_64: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_65: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_66: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_67: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_68: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_69: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_70: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_71: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_72: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_73: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_74: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_75: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_76: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_77: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_78: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_79: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_80: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_81: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_82: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_83: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_84: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_85: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_86: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_91: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_92: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_93: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_94: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_95: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_96: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_97: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_98: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_99: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_100: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_101: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_102: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_103: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_104: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_105: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_106: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_107: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_108: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_109: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_110: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_111: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_112: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_113: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_114: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_115: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_116: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_117: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_118: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_119: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_120: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_121: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_122: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_123: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_124: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_125: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_126: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_127: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_128: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_129: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_130: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_131: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_132: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_133: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_134: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_135: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_140: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_141: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_142: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_143: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        group_id__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        lane_id__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
        core_id__ssa_v0: pl.Scalar[pl.INDEX] = group_id__ssa_v0 * 2 + lane_id__ssa_v0 + 24
        record_32__ssa_v0: pl.Scalar[pl.INDEX] = core_id__ssa_v0 * 6 + 27648
        t__tile: pl.Scalar[pl.INT64] = pl.tensor.read(records__ssa_v0, [record_32__ssa_v0])
        mapped_block_32__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX)
        block_32__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(mapped_block_32__ssa_v0, 0)
        if 0 <= mapped_block_32__ssa_v0:
            t__tile_1: pl.Scalar[pl.INT64] = pl.tensor.read(records__ssa_v0, [record_32__ssa_v0 + 1])
            subslot_32__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
            stage_32_v22__ssa_v0: pl.Scalar[pl.INDEX] = block_32__ssa_v0
            pl.system.set_ffts(temporary_62__ssa_v0)
            stage_32_v23__ssa_v0: pl.Scalar[pl.INDEX] = (24 - stage_32_v22__ssa_v0 + 23) // 24
            stage_32_v24__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v23__ssa_v0 * 5
            stage_32_v25__ssa_v0: pl.Scalar[pl.INDEX] = subslot_32__ssa_v0
            pl.system.set_ffts(temporary_62__ssa_v0)
            stage_32_v26__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v25__ssa_v0 * 32
            stage_32_v27__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v25__ssa_v0 * 64
            stage_32_v28__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_32_v28__ssa_v0: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_21, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
                pl.const(0, pl.INT32), [1, 64], tmp=stage_32_v28__ci_tmp_v0, dtype=pl.INT32, descending=False
            )
            stage_32_v29__ssa_v0: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_22, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            stage_32_v30__ssa_v0: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_23, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.rems(
                stage_32_v28__ssa_v0, pl.const(2, pl.INT32), stage_32_v29__ssa_v0
            )
            stage_32_v31__ssa_v0: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_24, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.muls(stage_32_v30__ssa_v0, pl.const(2, pl.INT32))
            stage_32_v32__ssa_v0: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_25, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(stage_32_v28__ssa_v0, stage_32_v31__ssa_v0)
            stage_32_v33__ssa_v0: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_26, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.adds(stage_32_v32__ssa_v0, pl.const(449, pl.INT32))
            stage_32_v34__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            stage_32_v35__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_28, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand(stage_32_v34__ssa_v0, stage_32_v33__ssa_v0)
            stage_32_v36__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_29, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            stage_32_v36__ssa_v0: pl.Tile[[1, 16], pl.INT32, pl.MemRef(mem_vec_30, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.ci(
                pl.const(0, pl.INT32), [1, 16], tmp=stage_32_v36__ci_tmp_v1, dtype=pl.INT32, descending=False
            )
            stage_32_v37__ssa_v0: pl.Tile[[1, 16], pl.INT32, pl.MemRef(mem_vec_31, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(stage_32_v36__ssa_v0, pl.const(512, pl.INT32))
            stage_32_v38__ssa_v0: pl.Tile[[16, 1], pl.INT32, pl.MemRef(mem_vec_31, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v37__ssa_v0, [16, 1])
            stage_32_v39__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_32, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_add(stage_32_v35__ssa_v0, stage_32_v38__ssa_v0)
            stage_32_v40__ssa_v0: pl.Tile[[1, 1024], pl.INT32, pl.MemRef(mem_vec_32, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.reshape(stage_32_v39__ssa_v0, [1, 1024])
            stage_32_v41__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                view_116__ssa_v0, [stage_32_v26__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
            )
            stage_32_v42__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_33, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v41__ssa_v0, [1, 32])
            stage_32_v43__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(stage_32_v42__ssa_v0, 0.0)
            stage_32_v44__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v43__ssa_v0, [32, 1])
            stage_32_v45__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_35, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
            stage_32_v46__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_36, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
            for stage_32_v47__idx_v0, (stage_32_v51, stage_32_v52, stage_32_v53, stage_32_v54) in pl.range(
                stage_32_v24__ssa_v0 + 3, init_values=(stage_32_v41__ssa_v0, stage_32_v44__ssa_v0, stage_32_v45__ssa_v0, stage_32_v46__ssa_v0)
            ):
                stage_32_v48__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v22__ssa_v0 + stage_32_v47__idx_v0 // 5 * 24
                stage_32_v49__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v48__ssa_v0 // 6
                stage_32_v50__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v47__idx_v0 % 5
                if stage_32_v47__idx_v0 < stage_32_v24__ssa_v0:
                    stage_32_v55__tile: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54__ssa_v0, [stage_32_v48__ssa_v0, stage_32_v50__ssa_v0])
                    if 0 < pl.cast(stage_32_v55__tile, pl.INDEX):
                        stage_32_v72__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v22__ssa_v0 * 3 + stage_32_v47__idx_v0 % 3
                        stage_32_v73__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v72__ssa_v0 * 128
                        stage_32_v74__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
                        if stage_32_v50__ssa_v0 == 0:
                            stage_32_v256__tile: pl.Scalar[pl.INT64] = pl.tensor.read(view_106__ssa_v0, [stage_32_v48__ssa_v0, 0])
                            stage_32_v257__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v256__tile, pl.INDEX)
                            stage_32_v258__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(stage_32_v257__ssa_v0 + 1, 128)
                            stage_32_v259__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v257__ssa_v0 - stage_32_v258__ssa_v0 + 1
                            stage_32_v260__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v259__ssa_v0 % 32
                            stage_32_v261__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(stage_32_v258__ssa_v0 - stage_32_v27__ssa_v0, 0), 64)
                            stage_32_v262__ssa_v0: pl.Scalar[pl.INDEX] = 0
                            stage_32_v263__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32 - stage_32_v260__ssa_v0 - stage_32_v27__ssa_v0, stage_32_v261__ssa_v0)
                            if stage_32_v262__ssa_v0 < stage_32_v263__ssa_v0:
                                stage_32_v273__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v259__ssa_v0 + stage_32_v27__ssa_v0 + stage_32_v262__ssa_v0
                                stage_32_v274__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table__ssa_v0, [stage_32_v49__ssa_v0, stage_32_v273__ssa_v0 // 32])
                                if 0 <= pl.cast(stage_32_v274__tile, pl.INDEX):
                                    stage_32_v291__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v274__tile, pl.INDEX) * 32 + stage_32_v273__ssa_v0 % 32
                                    stage_32_v292__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        stage_32_v74__ssa_v0,
                                        view_45__ssa_v0,
                                        [stage_32_v262__ssa_v0, 0],
                                        [stage_32_v291__ssa_v0, 0],
                                        [64, 512],
                                        valid_shape=[stage_32_v263__ssa_v0 - stage_32_v262__ssa_v0, 512],
                                        transpose=False,
                                    )
                                    stage_32_v293__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v292__ssa_v0
                                    stage_32_v293__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v293__ssa_v0)
                                else:
                                    stage_32_v293__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v74__ssa_v0
                                    stage_32_v293__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v293__ssa_v1)
                                stage_32_v275__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v293__phi_v2
                                stage_32_v275__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v275__ssa_v0)
                            else:
                                stage_32_v275__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v74__ssa_v0
                                stage_32_v275__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v275__ssa_v1)
                            stage_32_v264__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(32 - stage_32_v260__ssa_v0 - stage_32_v27__ssa_v0, 0)
                            stage_32_v265__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(64 - stage_32_v260__ssa_v0 - stage_32_v27__ssa_v0, stage_32_v261__ssa_v0)
                            if stage_32_v264__ssa_v0 < stage_32_v265__ssa_v0:
                                stage_32_v276__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v259__ssa_v0 + stage_32_v27__ssa_v0 + stage_32_v264__ssa_v0
                                stage_32_v277__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table__ssa_v0, [stage_32_v49__ssa_v0, stage_32_v276__ssa_v0 // 32])
                                if 0 <= pl.cast(stage_32_v277__tile, pl.INDEX):
                                    stage_32_v294__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v277__tile, pl.INDEX) * 32 + stage_32_v276__ssa_v0 % 32
                                    stage_32_v295__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        stage_32_v275__phi_v2,
                                        view_45__ssa_v0,
                                        [stage_32_v264__ssa_v0, 0],
                                        [stage_32_v294__ssa_v0, 0],
                                        [64, 512],
                                        valid_shape=[stage_32_v265__ssa_v0 - stage_32_v264__ssa_v0, 512],
                                        transpose=False,
                                    )
                                    stage_32_v296__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v295__ssa_v0
                                    stage_32_v296__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v296__ssa_v0)
                                else:
                                    stage_32_v296__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v275__phi_v2
                                    stage_32_v296__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v296__ssa_v1)
                                stage_32_v278__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v296__phi_v2
                                stage_32_v278__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v278__ssa_v0)
                            else:
                                stage_32_v278__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v275__phi_v2
                                stage_32_v278__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v278__ssa_v1)
                            stage_32_v266__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(64 - stage_32_v260__ssa_v0 - stage_32_v27__ssa_v0, 0)
                            stage_32_v267__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(96 - stage_32_v260__ssa_v0 - stage_32_v27__ssa_v0, stage_32_v261__ssa_v0)
                            if stage_32_v266__ssa_v0 < stage_32_v267__ssa_v0:
                                stage_32_v279__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v259__ssa_v0 + stage_32_v27__ssa_v0 + stage_32_v266__ssa_v0
                                stage_32_v280__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table__ssa_v0, [stage_32_v49__ssa_v0, stage_32_v279__ssa_v0 // 32])
                                if 0 <= pl.cast(stage_32_v280__tile, pl.INDEX):
                                    stage_32_v297__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v280__tile, pl.INDEX) * 32 + stage_32_v279__ssa_v0 % 32
                                    stage_32_v298__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        stage_32_v278__phi_v2,
                                        view_45__ssa_v0,
                                        [stage_32_v266__ssa_v0, 0],
                                        [stage_32_v297__ssa_v0, 0],
                                        [64, 512],
                                        valid_shape=[stage_32_v267__ssa_v0 - stage_32_v266__ssa_v0, 512],
                                        transpose=False,
                                    )
                                    stage_32_v299__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v298__ssa_v0
                                    stage_32_v299__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v299__ssa_v0)
                                else:
                                    stage_32_v299__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v278__phi_v2
                                    stage_32_v299__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v299__ssa_v1)
                                stage_32_v281__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v299__phi_v2
                                stage_32_v281__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v281__ssa_v0)
                            else:
                                stage_32_v281__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v278__phi_v2
                                stage_32_v281__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v281__ssa_v1)
                            stage_32_v268__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(96 - stage_32_v260__ssa_v0 - stage_32_v27__ssa_v0, 0)
                            stage_32_v269__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(128 - stage_32_v260__ssa_v0 - stage_32_v27__ssa_v0, stage_32_v261__ssa_v0)
                            if stage_32_v268__ssa_v0 < stage_32_v269__ssa_v0:
                                stage_32_v282__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v259__ssa_v0 + stage_32_v27__ssa_v0 + stage_32_v268__ssa_v0
                                stage_32_v283__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table__ssa_v0, [stage_32_v49__ssa_v0, stage_32_v282__ssa_v0 // 32])
                                if 0 <= pl.cast(stage_32_v283__tile, pl.INDEX):
                                    stage_32_v300__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v283__tile, pl.INDEX) * 32 + stage_32_v282__ssa_v0 % 32
                                    stage_32_v301__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        stage_32_v281__phi_v2,
                                        view_45__ssa_v0,
                                        [stage_32_v268__ssa_v0, 0],
                                        [stage_32_v300__ssa_v0, 0],
                                        [64, 512],
                                        valid_shape=[stage_32_v269__ssa_v0 - stage_32_v268__ssa_v0, 512],
                                        transpose=False,
                                    )
                                    stage_32_v302__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v301__ssa_v0
                                    stage_32_v302__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v302__ssa_v0)
                                else:
                                    stage_32_v302__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v281__phi_v2
                                    stage_32_v302__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v302__ssa_v1)
                                stage_32_v284__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v302__phi_v2
                                stage_32_v284__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v284__ssa_v0)
                            else:
                                stage_32_v284__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v281__phi_v2
                                stage_32_v284__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v284__ssa_v1)
                            stage_32_v270__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(128 - stage_32_v260__ssa_v0 - stage_32_v27__ssa_v0, 0)
                            stage_32_v271__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(160 - stage_32_v260__ssa_v0 - stage_32_v27__ssa_v0, stage_32_v261__ssa_v0)
                            if stage_32_v270__ssa_v0 < stage_32_v271__ssa_v0:
                                stage_32_v285__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v259__ssa_v0 + stage_32_v27__ssa_v0 + stage_32_v270__ssa_v0
                                stage_32_v286__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_ori_block_table__ssa_v0, [stage_32_v49__ssa_v0, stage_32_v285__ssa_v0 // 32])
                                if 0 <= pl.cast(stage_32_v286__tile, pl.INDEX):
                                    stage_32_v303__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v286__tile, pl.INDEX) * 32 + stage_32_v285__ssa_v0 % 32
                                    stage_32_v304__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        stage_32_v284__phi_v2,
                                        view_45__ssa_v0,
                                        [stage_32_v270__ssa_v0, 0],
                                        [stage_32_v303__ssa_v0, 0],
                                        [64, 512],
                                        valid_shape=[stage_32_v271__ssa_v0 - stage_32_v270__ssa_v0, 512],
                                        transpose=False,
                                    )
                                    stage_32_v305__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v304__ssa_v0
                                    stage_32_v305__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v305__ssa_v0)
                                else:
                                    stage_32_v305__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v284__phi_v2
                                    stage_32_v305__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v305__ssa_v1)
                                stage_32_v287__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v305__phi_v2
                                stage_32_v287__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v287__ssa_v0)
                            else:
                                stage_32_v287__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v284__phi_v2
                                stage_32_v287__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v287__ssa_v1)
                            stage_32_v272__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v287__phi_v2
                            stage_32_v272__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v272__ssa_v0)
                        else:
                            stage_32_v74__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_48, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
                            for stage_32_v288__idx_v0, (stage_32_v306,) in pl.range(64, init_values=(stage_32_v74__ssa_v1,)):
                                stage_32_v289__ssa_v0: pl.Scalar[pl.INDEX] = (stage_32_v50__ssa_v0 - 1) * 128 + stage_32_v27__ssa_v0 + stage_32_v288__idx_v0
                                if stage_32_v289__ssa_v0 < 512:
                                    stage_32_v307__tile: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_53__ssa_v0, [stage_32_v48__ssa_v0, stage_32_v289__ssa_v0])
                                    if 0 <= pl.cast(stage_32_v307__tile, pl.INDEX):
                                        stage_32_v309__tile: pl.Scalar[pl.INT32] = pl.tensor.read(arg_cmp_block_table__ssa_v0, [stage_32_v49__ssa_v0, pl.cast(stage_32_v307__tile, pl.INDEX) // 32])
                                        stage_32_v310__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(stage_32_v309__tile, pl.INDEX)
                                        stage_32_v311__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v310__ssa_v0 * 32 + pl.cast(stage_32_v307__tile, pl.INDEX) % 32
                                        stage_32_v312__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_48, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                            stage_32_v306, view_61__ssa_v0, [stage_32_v288__idx_v0, 0], [stage_32_v311__ssa_v0, 0], [1, 512], transpose=False
                                        )
                                        stage_32_v313__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_48, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v312__ssa_v0
                                        stage_32_v313__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_48, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v313__ssa_v0)
                                    else:
                                        stage_32_v313__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_48, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v306
                                        stage_32_v313__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_48, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v313__ssa_v1)
                                    stage_32_v308__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_48, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v313__phi_v2
                                    stage_32_v308__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_48, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v308__ssa_v0)
                                else:
                                    stage_32_v308__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_48, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v306
                                    stage_32_v308__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_48, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v308__ssa_v1)
                                stage_32_v290: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_48, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v308__phi_v2)
                            stage_32_v272__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_48, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = stage_32_v290
                            stage_32_v272__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_37, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(stage_32_v272__ssa_v1)
                        stage_32_v75__ssa_v0: pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 9437184)] = pl.tile.store(
                            stage_32_v272__phi_v2, [stage_32_v73__ssa_v0 + stage_32_v27__ssa_v0, 0], temporary_56__ssa_v0
                        )
                if 0 < stage_32_v47__idx_v0 and stage_32_v47__idx_v0 <= stage_32_v24__ssa_v0:
                    stage_32_v56__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v47__idx_v0 - 1
                    stage_32_v57__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v56__ssa_v0 % 5
                    stage_32_v58__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v22__ssa_v0 + stage_32_v56__ssa_v0 // 5 * 24
                    stage_32_v59__tile: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54__ssa_v0, [stage_32_v58__ssa_v0, stage_32_v57__ssa_v0])
                    if 0 < pl.cast(stage_32_v59__tile, pl.INDEX):
                        stage_32_v78__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v22__ssa_v0 * 3 + stage_32_v56__ssa_v0 % 3
                        stage_32_v79__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v78__ssa_v0 * 64
                        stage_32_v80__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v57__ssa_v0 * 128
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        stage_32_v81__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_55, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                            [32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec
                        )
                        stage_32_v82__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_56, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                            temporary_57__ssa_v0, [stage_32_v79__ssa_v0 + stage_32_v26__ssa_v0, 0], [32, 128], [32, 128], target_memory=pl.Mem.Vec
                        )
                        stage_32_v83__ssa_v0: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_57, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                            temporary_52__ssa_v0, [stage_32_v58__ssa_v0, stage_32_v80__ssa_v0], [1, 128], [1, 128], target_memory=pl.Mem.Vec
                        )
                        stage_32_v84__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_58, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(stage_32_v82__ssa_v0, 0.044194173824159223)
                        stage_32_v85__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_59, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_add(
                            stage_32_v84__ssa_v0, stage_32_v83__ssa_v0
                        )
                        stage_32_v86__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_max(stage_32_v85__ssa_v0, stage_32_v81__ssa_v0)
                        stage_32_v87__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_61, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_sub(
                            stage_32_v85__ssa_v0, stage_32_v86__ssa_v0
                        )
                        stage_32_v88__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_62, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.exp(stage_32_v87__ssa_v0)
                        stage_32_v89__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_63, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(stage_32_v88__ssa_v0, stage_32_v81__ssa_v0)
                        stage_32_v90__ssa_v0: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_vec_64, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                            stage_32_v88__ssa_v0, target_type=pl.BF16, mode="rint"
                        )
                        stage_32_v91__ssa_v0: pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            stage_32_v90__ssa_v0, [stage_32_v79__ssa_v0 + stage_32_v26__ssa_v0, 0], temporary_58__ssa_v0
                        )
                        stage_32_v92__ssa_v0: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)] = pl.tile.store(
                            stage_32_v86__ssa_v0, [stage_32_v79__ssa_v0 + stage_32_v26__ssa_v0, 0], temporary_60__ssa_v0
                        )
                        stage_32_v93__ssa_v0: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 18432)] = pl.tile.store(
                            stage_32_v89__ssa_v0, [stage_32_v79__ssa_v0 + stage_32_v26__ssa_v0, 0], temporary_61__ssa_v0
                        )
                        pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                if stage_32_v47__idx_v0 < stage_32_v24__ssa_v0:
                    stage_32_v94__tile: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54__ssa_v0, [stage_32_v48__ssa_v0, stage_32_v50__ssa_v0])
                    if 0 < pl.cast(stage_32_v94__tile, pl.INDEX):
                        pl.system.sync_set(0, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                if 3 <= stage_32_v47__idx_v0:
                    stage_32_v60__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v47__idx_v0 - 2 - 1
                    stage_32_v61__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v60__ssa_v0 % 5
                    stage_32_v62__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v22__ssa_v0 + stage_32_v60__ssa_v0 // 5 * 24
                    stage_32_v63__tile: pl.Scalar[pl.INT32] = pl.tensor.read(temporary_54__ssa_v0, [stage_32_v62__ssa_v0, stage_32_v61__ssa_v0])
                    if 0 < pl.cast(stage_32_v63__tile, pl.INDEX):
                        stage_32_v95__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v22__ssa_v0 * 3 + stage_32_v60__ssa_v0 % 3
                        stage_32_v96__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v95__ssa_v0 * 64
                        pl.system.sync_wait(3, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        stage_32_v97__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_65, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                            temporary_60__ssa_v0, [stage_32_v96__ssa_v0 + stage_32_v26__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                        )
                        stage_32_v98__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_66, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                            temporary_61__ssa_v0, [stage_32_v96__ssa_v0 + stage_32_v26__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                        )
                        stage_32_v99__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_33, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v51, [1, 32])
                        stage_32_v100__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_65, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v97__ssa_v0, [1, 32])
                        stage_32_v101__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_67, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.maximum(stage_32_v99__ssa_v0, stage_32_v100__ssa_v0)
                        stage_32_v102__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_67, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v101__ssa_v0, [32, 1])
                        stage_32_v103__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_33, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v51, [1, 32])
                        stage_32_v104__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_67, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v102__ssa_v0, [1, 32])
                        stage_32_v105__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_68, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.sub(stage_32_v103__ssa_v0, stage_32_v104__ssa_v0)
                        stage_32_v106__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_68, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v105__ssa_v0, [32, 1])
                        stage_32_v107__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_68, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v106__ssa_v0, [1, 32])
                        stage_32_v108__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_69, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.exp(stage_32_v107__ssa_v0)
                        stage_32_v109__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_69, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v108__ssa_v0, [32, 1])
                        stage_32_v110__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_65, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v97__ssa_v0, [1, 32])
                        stage_32_v111__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_67, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v102__ssa_v0, [1, 32])
                        stage_32_v112__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_70, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.sub(stage_32_v110__ssa_v0, stage_32_v111__ssa_v0)
                        stage_32_v113__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_70, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v112__ssa_v0, [32, 1])
                        stage_32_v114__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_70, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v113__ssa_v0, [1, 32])
                        stage_32_v115__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_71, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.exp(stage_32_v114__ssa_v0)
                        stage_32_v116__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_71, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v115__ssa_v0, [32, 1])
                        stage_32_v117__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_69, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v109__ssa_v0, [1, 32])
                        stage_32_v118__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v52, [1, 32])
                        stage_32_v119__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_72, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.mul(stage_32_v117__ssa_v0, stage_32_v118__ssa_v0)
                        stage_32_v120__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_72, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v119__ssa_v0, [32, 1])
                        stage_32_v121__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_71, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v116__ssa_v0, [1, 32])
                        stage_32_v122__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_66, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v98__ssa_v0, [1, 32])
                        stage_32_v123__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_73, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.mul(stage_32_v121__ssa_v0, stage_32_v122__ssa_v0)
                        stage_32_v124__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_73, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v123__ssa_v0, [32, 1])
                        stage_32_v125__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_72, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v120__ssa_v0, [1, 32])
                        stage_32_v126__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_73, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v124__ssa_v0, [1, 32])
                        stage_32_v127__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_74, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(stage_32_v125__ssa_v0, stage_32_v126__ssa_v0)
                        stage_32_v128__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_74, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v127__ssa_v0, [32, 1])
                        stage_32_v129__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_75, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                            temporary_59__ssa_v0, [stage_32_v96__ssa_v0 + stage_32_v26__ssa_v0, 0], [32, 256], [32, 256], target_memory=pl.Mem.Vec
                        )
                        stage_32_v130__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_76, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                            stage_32_v53, stage_32_v109__ssa_v0
                        )
                        stage_32_v131__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_77, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                            stage_32_v129__ssa_v0, stage_32_v116__ssa_v0
                        )
                        stage_32_v132__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_78, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(stage_32_v130__ssa_v0, stage_32_v131__ssa_v0)
                        stage_32_v133__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_79, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                            temporary_59__ssa_v0, [stage_32_v96__ssa_v0 + stage_32_v26__ssa_v0, 256], [32, 256], [32, 256], target_memory=pl.Mem.Vec
                        )
                        stage_32_v134__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_80, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                            stage_32_v54, stage_32_v109__ssa_v0
                        )
                        stage_32_v135__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_81, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                            stage_32_v133__ssa_v0, stage_32_v116__ssa_v0
                        )
                        stage_32_v136__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_82, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(stage_32_v134__ssa_v0, stage_32_v135__ssa_v0)
                        stage_32_v137__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_67, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = stage_32_v102__ssa_v0
                        stage_32_v138__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_74, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = stage_32_v128__ssa_v0
                        stage_32_v139__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_78, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = stage_32_v132__ssa_v0
                        stage_32_v140__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_82, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = stage_32_v136__ssa_v0
                        stage_32_v137__phi_v2, stage_32_v138__phi_v2, stage_32_v139__phi_v2, stage_32_v140__phi_v2 = pl.yield_(
                            stage_32_v137__ssa_v0, stage_32_v138__ssa_v0, stage_32_v139__ssa_v0, stage_32_v140__ssa_v0
                        )
                    else:
                        stage_32_v137__ssa_v1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = stage_32_v51
                        stage_32_v138__ssa_v1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = stage_32_v52
                        stage_32_v139__ssa_v1: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_35, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = stage_32_v53
                        stage_32_v140__ssa_v1: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_36, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = stage_32_v54
                        stage_32_v137__phi_v2, stage_32_v138__phi_v2, stage_32_v139__phi_v2, stage_32_v140__phi_v2 = pl.yield_(
                            stage_32_v137__ssa_v1, stage_32_v138__ssa_v1, stage_32_v139__ssa_v1, stage_32_v140__ssa_v1
                        )
                    if stage_32_v61__ssa_v0 == 4:
                        stage_32_v141__ssa_v0: pl.Tile[[1, 1024], pl.INT32, pl.MemRef(mem_vec_91, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create(
                            [1, 1024], dtype=pl.INT32, target_memory=pl.Mem.Vec
                        )
                        stage_32_v142__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        stage_32_v143__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v26__ssa_v0 + stage_32_v142__ssa_v0
                        stage_32_v144__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_67, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v137__phi_v2, [1, 32])
                        stage_32_v145__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_92, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.extract(
                            stage_32_v144__ssa_v0, 0, stage_32_v142__ssa_v0, [1, 16], target_memory=pl.Mem.Vec
                        )
                        stage_32_v146__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_92, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v145__ssa_v0, [16, 1])
                        stage_32_v147__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_74, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v138__phi_v2, [1, 32])
                        stage_32_v148__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_93, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.extract(
                            stage_32_v147__ssa_v0, 0, stage_32_v142__ssa_v0, [1, 16], target_memory=pl.Mem.Vec
                        )
                        stage_32_v149__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_93, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v148__ssa_v0, [16, 1])
                        stage_32_v150__ssa_v0: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_vec_78, stage_32_v142__ssa_v0 * 256 * 4, 16384), pl.Mem.Vec] = pl.tile.slice(
                            stage_32_v139__phi_v2, [16, 256], [stage_32_v142__ssa_v0, 0]
                        )
                        stage_32_v151__ssa_v0: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_vec_82, stage_32_v142__ssa_v0 * 256 * 4, 16384), pl.Mem.Vec] = pl.tile.slice(
                            stage_32_v140__phi_v2, [16, 256], [stage_32_v142__ssa_v0, 0]
                        )
                        stage_32_v152__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_94, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.concat(
                            stage_32_v150__ssa_v0, stage_32_v151__ssa_v0
                        )
                        stage_32_v153__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_95, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                            view_116__ssa_v0, [stage_32_v143__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec
                        )
                        stage_32_v154__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_92, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v146__ssa_v0, [1, 16])
                        stage_32_v155__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_92, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v146__ssa_v0, [1, 16])
                        stage_32_v156__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_96, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(stage_32_v154__ssa_v0, stage_32_v155__ssa_v0)
                        stage_32_v157__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_96, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v156__ssa_v0, [16, 1])
                        stage_32_v158__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_96, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v157__ssa_v0, [1, 16])
                        stage_32_v159__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_95, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v153__ssa_v0, [1, 16])
                        stage_32_v160__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_97, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(stage_32_v158__ssa_v0, stage_32_v159__ssa_v0)
                        stage_32_v161__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_97, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v160__ssa_v0, [16, 1])
                        stage_32_v162__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_97, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v161__ssa_v0, [1, 16])
                        stage_32_v163__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_92, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v146__ssa_v0, [1, 16])
                        stage_32_v164__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_98, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(stage_32_v162__ssa_v0, stage_32_v163__ssa_v0)
                        stage_32_v165__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_98, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v164__ssa_v0, [16, 1])
                        stage_32_v166__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_98, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v165__ssa_v0, [1, 16])
                        stage_32_v167__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_99, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(stage_32_v166__ssa_v0)
                        stage_32_v168__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_99, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v167__ssa_v0, [16, 1])
                        stage_32_v169__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_93, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v149__ssa_v0, [1, 16])
                        stage_32_v170__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_99, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v168__ssa_v0, [1, 16])
                        stage_32_v171__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_100, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(stage_32_v169__ssa_v0, stage_32_v170__ssa_v0)
                        stage_32_v172__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_100, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v171__ssa_v0, [16, 1])
                        stage_32_v173__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_101, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_div(
                            stage_32_v152__ssa_v0, stage_32_v172__ssa_v0
                        )
                        stage_32_v174__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_102, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                            stage_32_v173__ssa_v0, target_type=pl.BF16, mode="rint"
                        )
                        stage_32_v175__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_103, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                            arg_freqs_cos__ssa_v0, [stage_32_v62__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                        )
                        stage_32_v176__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_104, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                            temporary_55__ssa_v0, [stage_32_v62__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                        )
                        stage_32_v177__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_101, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.reshape(stage_32_v173__ssa_v0, [1, 8192])
                        stage_32_v178__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.gather(
                            stage_32_v177__ssa_v0, stage_32_v40__ssa_v0, stage_32_v141__ssa_v0
                        )
                        stage_32_v179__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.reshape(stage_32_v178__ssa_v0, [16, 64])
                        stage_32_v180__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_106, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.extract(
                            stage_32_v173__ssa_v0, 0, 448, [16, 64], target_memory=pl.Mem.Vec
                        )
                        stage_32_v181__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                            stage_32_v180__ssa_v0, stage_32_v175__ssa_v0
                        )
                        stage_32_v182__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                            stage_32_v179__ssa_v0, stage_32_v176__ssa_v0
                        )
                        stage_32_v183__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_109, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(stage_32_v181__ssa_v0, stage_32_v182__ssa_v0)
                        stage_32_v184__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_110, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                            stage_32_v183__ssa_v0, target_type=pl.BF16, mode="rint"
                        )
                        stage_32_v185__ssa_v0: pl.Tile[[16, 448], pl.BF16, pl.MemRef(mem_vec_102, 0, 16256), pl.Mem.Vec] = pl.tile.slice(stage_32_v174__ssa_v0, [16, 448], [0, 0])
                        stage_32_v186__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_111, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.concat(
                            stage_32_v185__ssa_v0, stage_32_v184__ssa_v0
                        )
                        stage_32_v187__ssa_v0: pl.Tile[[2, 4096], pl.BF16, pl.MemRef(mem_vec_111, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.reshape(stage_32_v186__ssa_v0, [2, 4096])
                        stage_32_v188__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_111, 0, 8192), pl.Mem.Vec] = pl.tile.slice(stage_32_v187__ssa_v0, [1, 4096], [0, 0])
                        stage_32_v189__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_111, 8192, 8192), pl.Mem.Vec] = pl.tile.slice(stage_32_v187__ssa_v0, [1, 4096], [1, 0])
                        stage_32_v190__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v143__ssa_v0 // 8 * 384 + stage_32_v62__ssa_v0
                        stage_32_v191__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v190__ssa_v0 + 384
                        stage_32_v192__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                            stage_32_v188__ssa_v0, [stage_32_v190__ssa_v0, 0], temporary_51__ssa_v0
                        )
                        stage_32_v193__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                            stage_32_v189__ssa_v0, [stage_32_v191__ssa_v0, 0], temporary_51__ssa_v0
                        )
                        stage_32_v194__ssa_v0: pl.Scalar[pl.INDEX] = 16
                        stage_32_v195__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v26__ssa_v0 + stage_32_v194__ssa_v0
                        stage_32_v196__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_67, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v137__phi_v2, [1, 32])
                        stage_32_v197__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_112, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.extract(
                            stage_32_v196__ssa_v0, 0, stage_32_v194__ssa_v0, [1, 16], target_memory=pl.Mem.Vec
                        )
                        stage_32_v198__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_112, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v197__ssa_v0, [16, 1])
                        stage_32_v199__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_74, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v138__phi_v2, [1, 32])
                        stage_32_v200__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_113, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.extract(
                            stage_32_v199__ssa_v0, 0, stage_32_v194__ssa_v0, [1, 16], target_memory=pl.Mem.Vec
                        )
                        stage_32_v201__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_113, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v200__ssa_v0, [16, 1])
                        stage_32_v202__ssa_v0: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_vec_78, stage_32_v194__ssa_v0 * 256 * 4, 16384), pl.Mem.Vec] = pl.tile.slice(
                            stage_32_v139__phi_v2, [16, 256], [stage_32_v194__ssa_v0, 0]
                        )
                        stage_32_v203__ssa_v0: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_vec_82, stage_32_v194__ssa_v0 * 256 * 4, 16384), pl.Mem.Vec] = pl.tile.slice(
                            stage_32_v140__phi_v2, [16, 256], [stage_32_v194__ssa_v0, 0]
                        )
                        stage_32_v204__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_114, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.concat(
                            stage_32_v202__ssa_v0, stage_32_v203__ssa_v0
                        )
                        stage_32_v205__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_115, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                            view_116__ssa_v0, [stage_32_v195__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec
                        )
                        stage_32_v206__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_112, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v198__ssa_v0, [1, 16])
                        stage_32_v207__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_112, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v198__ssa_v0, [1, 16])
                        stage_32_v208__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_116, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(stage_32_v206__ssa_v0, stage_32_v207__ssa_v0)
                        stage_32_v209__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_116, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v208__ssa_v0, [16, 1])
                        stage_32_v210__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_116, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v209__ssa_v0, [1, 16])
                        stage_32_v211__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_115, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v205__ssa_v0, [1, 16])
                        stage_32_v212__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_117, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(stage_32_v210__ssa_v0, stage_32_v211__ssa_v0)
                        stage_32_v213__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_117, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v212__ssa_v0, [16, 1])
                        stage_32_v214__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_117, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v213__ssa_v0, [1, 16])
                        stage_32_v215__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_112, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v198__ssa_v0, [1, 16])
                        stage_32_v216__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_118, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(stage_32_v214__ssa_v0, stage_32_v215__ssa_v0)
                        stage_32_v217__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_118, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v216__ssa_v0, [16, 1])
                        stage_32_v218__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_118, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v217__ssa_v0, [1, 16])
                        stage_32_v219__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(stage_32_v218__ssa_v0)
                        stage_32_v220__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v219__ssa_v0, [16, 1])
                        stage_32_v221__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_113, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v201__ssa_v0, [1, 16])
                        stage_32_v222__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v220__ssa_v0, [1, 16])
                        stage_32_v223__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_120, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(stage_32_v221__ssa_v0, stage_32_v222__ssa_v0)
                        stage_32_v224__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_120, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(stage_32_v223__ssa_v0, [16, 1])
                        stage_32_v225__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_121, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_div(
                            stage_32_v204__ssa_v0, stage_32_v224__ssa_v0
                        )
                        stage_32_v226__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_122, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                            stage_32_v225__ssa_v0, target_type=pl.BF16, mode="rint"
                        )
                        stage_32_v227__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_123, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                            arg_freqs_cos__ssa_v0, [stage_32_v62__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                        )
                        stage_32_v228__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_124, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                            temporary_55__ssa_v0, [stage_32_v62__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                        )
                        stage_32_v229__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_121, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.reshape(stage_32_v225__ssa_v0, [1, 8192])
                        stage_32_v230__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_125, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.gather(
                            stage_32_v229__ssa_v0, stage_32_v40__ssa_v0, stage_32_v141__ssa_v0
                        )
                        stage_32_v231__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_125, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.reshape(stage_32_v230__ssa_v0, [16, 64])
                        stage_32_v232__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_126, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.extract(
                            stage_32_v225__ssa_v0, 0, 448, [16, 64], target_memory=pl.Mem.Vec
                        )
                        stage_32_v233__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_127, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                            stage_32_v232__ssa_v0, stage_32_v227__ssa_v0
                        )
                        stage_32_v234__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_128, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                            stage_32_v231__ssa_v0, stage_32_v228__ssa_v0
                        )
                        stage_32_v235__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_129, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(stage_32_v233__ssa_v0, stage_32_v234__ssa_v0)
                        stage_32_v236__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_130, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                            stage_32_v235__ssa_v0, target_type=pl.BF16, mode="rint"
                        )
                        stage_32_v237__ssa_v0: pl.Tile[[16, 448], pl.BF16, pl.MemRef(mem_vec_122, 0, 16256), pl.Mem.Vec] = pl.tile.slice(stage_32_v226__ssa_v0, [16, 448], [0, 0])
                        stage_32_v238__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_131, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.concat(
                            stage_32_v237__ssa_v0, stage_32_v236__ssa_v0
                        )
                        stage_32_v239__ssa_v0: pl.Tile[[2, 4096], pl.BF16, pl.MemRef(mem_vec_131, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.reshape(stage_32_v238__ssa_v0, [2, 4096])
                        stage_32_v240__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_131, 0, 8192), pl.Mem.Vec] = pl.tile.slice(stage_32_v239__ssa_v0, [1, 4096], [0, 0])
                        stage_32_v241__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_131, 8192, 8192), pl.Mem.Vec] = pl.tile.slice(stage_32_v239__ssa_v0, [1, 4096], [1, 0])
                        stage_32_v242__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v195__ssa_v0 // 8 * 384 + stage_32_v62__ssa_v0
                        stage_32_v243__ssa_v0: pl.Scalar[pl.INDEX] = stage_32_v242__ssa_v0 + 384
                        stage_32_v244__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                            stage_32_v240__ssa_v0, [stage_32_v242__ssa_v0, 0], temporary_51__ssa_v0
                        )
                        stage_32_v245__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                            stage_32_v241__ssa_v0, [stage_32_v243__ssa_v0, 0], temporary_51__ssa_v0
                        )
                        stage_32_v246__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                            view_116__ssa_v0, [stage_32_v26__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                        )
                        stage_32_v247__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v246__ssa_v0, [1, 32])
                        stage_32_v248__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(stage_32_v247__ssa_v0, 0.0)
                        stage_32_v249__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(stage_32_v248__ssa_v0, [32, 1])
                        stage_32_v250__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_134, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
                        stage_32_v251__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_135, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
                        stage_32_v252__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = stage_32_v246__ssa_v0
                        stage_32_v253__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = stage_32_v249__ssa_v0
                        stage_32_v254__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_134, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = stage_32_v250__ssa_v0
                        stage_32_v255__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_135, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = stage_32_v251__ssa_v0
                        stage_32_v252__phi_v2, stage_32_v253__phi_v2, stage_32_v254__phi_v2, stage_32_v255__phi_v2 = pl.yield_(
                            stage_32_v252__ssa_v0, stage_32_v253__ssa_v0, stage_32_v254__ssa_v0, stage_32_v255__ssa_v0
                        )
                    else:
                        stage_32_v252__ssa_v1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_67, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = stage_32_v137__phi_v2
                        stage_32_v253__ssa_v1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_74, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = stage_32_v138__phi_v2
                        stage_32_v254__ssa_v1: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_78, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = stage_32_v139__phi_v2
                        stage_32_v255__ssa_v1: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_82, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = stage_32_v140__phi_v2
                        stage_32_v252__phi_v2, stage_32_v253__phi_v2, stage_32_v254__phi_v2, stage_32_v255__phi_v2 = pl.yield_(
                            stage_32_v252__ssa_v1, stage_32_v253__ssa_v1, stage_32_v254__ssa_v1, stage_32_v255__ssa_v1
                        )
                    stage_32_v64__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = stage_32_v252__phi_v2
                    stage_32_v65__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = stage_32_v253__phi_v2
                    stage_32_v66__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_134, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = stage_32_v254__phi_v2
                    stage_32_v67__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_135, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = stage_32_v255__phi_v2
                    stage_32_v64__phi_v2, stage_32_v65__phi_v2, stage_32_v66__phi_v2, stage_32_v67__phi_v2 = pl.yield_(
                        stage_32_v64__ssa_v0, stage_32_v65__ssa_v0, stage_32_v66__ssa_v0, stage_32_v67__ssa_v0
                    )
                else:
                    stage_32_v64__ssa_v1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = stage_32_v51
                    stage_32_v65__ssa_v1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = stage_32_v52
                    stage_32_v66__ssa_v1: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_35, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = stage_32_v53
                    stage_32_v67__ssa_v1: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_36, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = stage_32_v54
                    stage_32_v64__phi_v2, stage_32_v65__phi_v2, stage_32_v66__phi_v2, stage_32_v67__phi_v2 = pl.yield_(
                        stage_32_v64__ssa_v1, stage_32_v65__ssa_v1, stage_32_v66__ssa_v1, stage_32_v67__ssa_v1
                    )
                stage_32_v68, stage_32_v69, stage_32_v70, stage_32_v71 = pl.yield_(stage_32_v64__phi_v2, stage_32_v65__phi_v2, stage_32_v66__phi_v2, stage_32_v67__phi_v2)
            pl.system.fence()
            pl.system.cacheinvalid()
            pl.system.fence()
        return (
            view_45__ssa_v0,
            view_61__ssa_v0,
            temporary_53__ssa_v0,
            temporary_52__ssa_v0,
            temporary_54__ssa_v0,
            temporary_55__ssa_v0,
            temporary_62__ssa_v0,
            temporary_56__ssa_v0,
            temporary_57__ssa_v0,
            temporary_58__ssa_v0,
            temporary_59__ssa_v0,
            temporary_60__ssa_v0,
            temporary_61__ssa_v0,
            temporary_51__ssa_v0,
        )

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_static_full_card(
        self,
        arg_freqs_cos__ssa_v0: pl.Tensor[[24, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6144)],
        view_45__ssa_v0: pl.InOut[pl.Tensor[[1184, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1212416)]],
        view_61__ssa_v0: pl.InOut[pl.Tensor[[131360, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 134512640)]],
        temporary_53__ssa_v0: pl.InOut[pl.Tensor[[24, 512], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 49152)]],
        temporary_52__ssa_v0: pl.InOut[pl.Tensor[[24, 640], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 61440)]],
        view_106__ssa_v0: pl.Tensor[[24, 1], pl.INT64, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 192)],
        temporary_54__ssa_v0: pl.InOut[pl.Tensor[[24, 16], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 1536)]],
        arg_ori_block_table__ssa_v0: pl.Tensor[[4, 4098], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 65568)],
        temporary_55__ssa_v0: pl.InOut[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 98304)]],
        temporary_62__ssa_v0: pl.InOut[pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 2048)]],
        temporary_56__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 9437184)]],
        temporary_57__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 2359296)]],
        temporary_58__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 1179648)]],
        temporary_59__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 9437184)]],
        view_116__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 256)],
        arg_cmp_block_table__ssa_v0: pl.Tensor[[4, 1026], pl.INT32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 16416)],
        temporary_60__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        temporary_61__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 18432)]],
        temporary_51__ssa_v0: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 25165824)]],
        records__ssa_v0: pl.Tensor[[31968], pl.INT64, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 255744)],
    ) -> tuple[
        pl.Tensor[[1184, 512], pl.BF16],
        pl.Tensor[[131360, 512], pl.BF16],
        pl.Tensor[[24, 512], pl.INT32],
        pl.Tensor[[24, 640], pl.FP32],
        pl.Tensor[[24, 16], pl.INT32],
        pl.Tensor[[384, 64], pl.FP32],
        pl.Tensor[[256], pl.INT64],
        pl.Tensor[[9216, 512], pl.BF16],
        pl.Tensor[[4608, 128], pl.FP32],
        pl.Tensor[[4608, 128], pl.BF16],
        pl.Tensor[[4608, 512], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[3072, 4096], pl.BF16],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        _tuple_tmp__ssa_v0: pl.Tuple[
            pl.Tensor[[1184, 512], pl.BF16],
            pl.Tensor[[131360, 512], pl.BF16],
            pl.Tensor[[24, 512], pl.INT32],
            pl.Tensor[[24, 640], pl.FP32],
            pl.Tensor[[24, 16], pl.INT32],
            pl.Tensor[[384, 64], pl.FP32],
            pl.Tensor[[256], pl.INT64],
            pl.Tensor[[9216, 512], pl.BF16],
            pl.Tensor[[4608, 128], pl.FP32],
            pl.Tensor[[4608, 128], pl.BF16],
            pl.Tensor[[4608, 512], pl.FP32],
            pl.Tensor[[4608, 1], pl.FP32],
            pl.Tensor[[4608, 1], pl.FP32],
            pl.Tensor[[3072, 4096], pl.BF16],
        ] = self.csa_static_aiv(
            arg_freqs_cos__ssa_v0,
            view_45__ssa_v0,
            view_61__ssa_v0,
            temporary_53__ssa_v0,
            temporary_52__ssa_v0,
            view_106__ssa_v0,
            temporary_54__ssa_v0,
            arg_ori_block_table__ssa_v0,
            temporary_55__ssa_v0,
            temporary_62__ssa_v0,
            temporary_56__ssa_v0,
            temporary_57__ssa_v0,
            temporary_58__ssa_v0,
            temporary_59__ssa_v0,
            view_116__ssa_v0,
            arg_cmp_block_table__ssa_v0,
            temporary_60__ssa_v0,
            temporary_61__ssa_v0,
            temporary_51__ssa_v0,
            records__ssa_v0,
        )
        view_45__ssa_v1: pl.Tensor[[1184, 512], pl.BF16, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 1212416)] = _tuple_tmp__ssa_v0[0]
        view_61__ssa_v1: pl.Tensor[[131360, 512], pl.BF16, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 134512640)] = _tuple_tmp__ssa_v0[1]
        temporary_53__ssa_v1: pl.Tensor[[24, 512], pl.INT32, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 49152)] = _tuple_tmp__ssa_v0[2]
        temporary_52__ssa_v1: pl.Tensor[[24, 640], pl.FP32, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 61440)] = _tuple_tmp__ssa_v0[3]
        temporary_54__ssa_v1: pl.Tensor[[24, 16], pl.INT32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 1536)] = _tuple_tmp__ssa_v0[4]
        temporary_55__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 98304)] = _tuple_tmp__ssa_v0[5]
        temporary_62__ssa_v1: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 2048)] = _tuple_tmp__ssa_v0[6]
        temporary_56__ssa_v1: pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 9437184)] = _tuple_tmp__ssa_v0[7]
        temporary_57__ssa_v1: pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 2359296)] = _tuple_tmp__ssa_v0[8]
        temporary_58__ssa_v1: pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 1179648)] = _tuple_tmp__ssa_v0[9]
        temporary_59__ssa_v1: pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 9437184)] = _tuple_tmp__ssa_v0[10]
        temporary_60__ssa_v1: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 18432)] = _tuple_tmp__ssa_v0[11]
        temporary_61__ssa_v1: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 18432)] = _tuple_tmp__ssa_v0[12]
        temporary_51__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 25165824)] = _tuple_tmp__ssa_v0[13]
        return (
            view_45__ssa_v0,
            view_61__ssa_v0,
            temporary_53__ssa_v0,
            temporary_52__ssa_v0,
            temporary_54__ssa_v0,
            temporary_55__ssa_v0,
            temporary_62__ssa_v0,
            temporary_56__ssa_v0,
            temporary_57__ssa_v0,
            temporary_58__ssa_v0,
            temporary_59__ssa_v0,
            temporary_60__ssa_v0,
            temporary_61__ssa_v0,
            temporary_51__ssa_v0,
        )

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(
        self,
        arg_freqs_cos__ssa_v0: pl.Tensor[[24, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6144)],
        view_45__ssa_v0: pl.InOut[pl.Tensor[[1184, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1212416)]],
        view_61__ssa_v0: pl.InOut[pl.Tensor[[131360, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 134512640)]],
        temporary_53__ssa_v0: pl.InOut[pl.Tensor[[24, 512], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 49152)]],
        temporary_52__ssa_v0: pl.InOut[pl.Tensor[[24, 640], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 61440)]],
        view_106__ssa_v0: pl.Tensor[[24, 1], pl.INT64, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 192)],
        temporary_54__ssa_v0: pl.InOut[pl.Tensor[[24, 16], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 1536)]],
        arg_ori_block_table__ssa_v0: pl.Tensor[[4, 4098], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 65568)],
        temporary_55__ssa_v0: pl.InOut[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 98304)]],
        temporary_62__ssa_v0: pl.InOut[pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 2048)]],
        temporary_56__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 9437184)]],
        temporary_57__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 2359296)]],
        temporary_58__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 1179648)]],
        temporary_59__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 9437184)]],
        view_116__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 256)],
        arg_cmp_block_table__ssa_v0: pl.Tensor[[4, 1026], pl.INT32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 16416)],
        temporary_60__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        temporary_61__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 18432)]],
        temporary_51__ssa_v0: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 25165824)]],
        records__ssa_v0: pl.Tensor[[31968], pl.INT64, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 255744)],
    ) -> tuple[
        pl.Tensor[[1184, 512], pl.BF16],
        pl.Tensor[[131360, 512], pl.BF16],
        pl.Tensor[[24, 512], pl.INT32],
        pl.Tensor[[24, 640], pl.FP32],
        pl.Tensor[[24, 16], pl.INT32],
        pl.Tensor[[384, 64], pl.FP32],
        pl.Tensor[[256], pl.INT64],
        pl.Tensor[[9216, 512], pl.BF16],
        pl.Tensor[[4608, 128], pl.FP32],
        pl.Tensor[[4608, 128], pl.BF16],
        pl.Tensor[[4608, 512], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[3072, 4096], pl.BF16],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[
            pl.Tensor[[1184, 512], pl.BF16],
            pl.Tensor[[131360, 512], pl.BF16],
            pl.Tensor[[24, 512], pl.INT32],
            pl.Tensor[[24, 640], pl.FP32],
            pl.Tensor[[24, 16], pl.INT32],
            pl.Tensor[[384, 64], pl.FP32],
            pl.Tensor[[256], pl.INT64],
            pl.Tensor[[9216, 512], pl.BF16],
            pl.Tensor[[4608, 128], pl.FP32],
            pl.Tensor[[4608, 128], pl.BF16],
            pl.Tensor[[4608, 512], pl.FP32],
            pl.Tensor[[4608, 1], pl.FP32],
            pl.Tensor[[4608, 1], pl.FP32],
            pl.Tensor[[3072, 4096], pl.BF16],
        ] = self.csa_static_full_card(
            arg_freqs_cos__ssa_v0,
            view_45__ssa_v0,
            view_61__ssa_v0,
            temporary_53__ssa_v0,
            temporary_52__ssa_v0,
            view_106__ssa_v0,
            temporary_54__ssa_v0,
            arg_ori_block_table__ssa_v0,
            temporary_55__ssa_v0,
            temporary_62__ssa_v0,
            temporary_56__ssa_v0,
            temporary_57__ssa_v0,
            temporary_58__ssa_v0,
            temporary_59__ssa_v0,
            view_116__ssa_v0,
            arg_cmp_block_table__ssa_v0,
            temporary_60__ssa_v0,
            temporary_61__ssa_v0,
            temporary_51__ssa_v0,
            records__ssa_v0,
            attrs={"core_num": 24, "sync_start": True},
        )
        view_45__ssa_v1: pl.Tensor[[1184, 512], pl.BF16, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 1212416)] = ret__tmp_v0[0]
        view_61__ssa_v1: pl.Tensor[[131360, 512], pl.BF16, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 134512640)] = ret__tmp_v0[1]
        temporary_53__ssa_v1: pl.Tensor[[24, 512], pl.INT32, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 49152)] = ret__tmp_v0[2]
        temporary_52__ssa_v1: pl.Tensor[[24, 640], pl.FP32, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 61440)] = ret__tmp_v0[3]
        temporary_54__ssa_v1: pl.Tensor[[24, 16], pl.INT32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 1536)] = ret__tmp_v0[4]
        temporary_55__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0[5]
        temporary_62__ssa_v1: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 2048)] = ret__tmp_v0[6]
        temporary_56__ssa_v1: pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 9437184)] = ret__tmp_v0[7]
        temporary_57__ssa_v1: pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 2359296)] = ret__tmp_v0[8]
        temporary_58__ssa_v1: pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 1179648)] = ret__tmp_v0[9]
        temporary_59__ssa_v1: pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 9437184)] = ret__tmp_v0[10]
        temporary_60__ssa_v1: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 18432)] = ret__tmp_v0[11]
        temporary_61__ssa_v1: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 18432)] = ret__tmp_v0[12]
        temporary_51__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 25165824)] = ret__tmp_v0[13]
        return (
            view_45__ssa_v1,
            view_61__ssa_v1,
            temporary_53__ssa_v1,
            temporary_52__ssa_v1,
            temporary_54__ssa_v1,
            temporary_55__ssa_v1,
            temporary_62__ssa_v1,
            temporary_56__ssa_v1,
            temporary_57__ssa_v1,
            temporary_58__ssa_v1,
            temporary_59__ssa_v1,
            temporary_60__ssa_v1,
            temporary_61__ssa_v1,
            temporary_51__ssa_v1,
        )
