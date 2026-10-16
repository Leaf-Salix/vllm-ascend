# pypto.program: CSAAllocatorFocusedProbe
import pypto.language as pl


@pl.program
class CSAAllocatorFocusedProbe:
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def csa_static_aic(
        temporary_51__ssa_v0: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)]],
        arg_wo_a__ssa_v0: pl.Tensor[
            [8, 64, 256, 16, 16], pl.BF16, pl.TensorView(stride=[4194304, 65536, 256, 16, 1], layout=pl.TensorLayout.NZ), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)
        ],
        temporary_63__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        records__ssa_v0: pl.Tensor[[31968], pl.INT64, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 255744)],
    ) -> tuple[pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True, "nz_tensor_views_blocked": True})
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_acc_6: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 65536)
        mem_left_7: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_8: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_9: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_10: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_mat_12: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_13: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_14: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_15: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_16: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_17: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_18: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_19: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_21: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_22: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_23: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_24: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_mat_27: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_28: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_29: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_30: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_31: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_32: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        group_id__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        core_id__ssa_v0: pl.Scalar[pl.INDEX] = group_id__ssa_v0
        record_15__ssa_v0: pl.Scalar[pl.INDEX] = core_id__ssa_v0 * 6 + 6480
        t__tile: pl.Scalar[pl.INT64] = pl.tensor.read(records__ssa_v0, [record_15__ssa_v0])
        mapped_block_15__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX)
        block_15__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(mapped_block_15__ssa_v0, 0)
        if 0 <= mapped_block_15__ssa_v0:
            t__tile_1: pl.Scalar[pl.INT64] = pl.tensor.read(records__ssa_v0, [record_15__ssa_v0 + 1])
            subslot_15__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
            stage_15_v4__ssa_v0: pl.Scalar[pl.INDEX] = block_15__ssa_v0
            stage_15_v5__ssa_v0: pl.Scalar[pl.INDEX] = stage_15_v4__ssa_v0 // 8
            stage_15_v6__ssa_v0: pl.Scalar[pl.INDEX] = stage_15_v4__ssa_v0 % 8
            stage_15_v7__ssa_v0: pl.Scalar[pl.INDEX] = stage_15_v5__ssa_v0 * 128
            stage_15_v8__ssa_v0: pl.Scalar[pl.INDEX] = 24 - stage_15_v7__ssa_v0
            stage_15_v9__ssa_v0: pl.Scalar[pl.INDEX] = stage_15_v7__ssa_v0 + 2688
            stage_15_v10__ssa_v0: pl.Scalar[pl.INDEX] = stage_15_v6__ssa_v0 * 128
            stage_15_v11__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[stage_15_v8__ssa_v0, 256])] = pl.tile.load(
                temporary_51__ssa_v0, [stage_15_v9__ssa_v0, 0], [128, 256], [stage_15_v8__ssa_v0, 256], target_memory=pl.Mem.Mat
            )
            stage_15_v12__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_5, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                arg_wo_a__ssa_v0, [7, stage_15_v10__ssa_v0 // 16, 0, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat
            )
            stage_15_v13__tile_l0_init_storage: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = (
                pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            )
            stage_15_v13__tile_l0_init: pl.Tile[
                [128, 128],
                pl.FP32,
                pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                pl.Mem.Acc,
                pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
            ] = pl.tile.set_validshape(stage_15_v13__tile_l0_init_storage, pl.min(stage_15_v8__ssa_v0, 128), 128)
            for stage_15_v13__tile_l0_ko, (stage_15_v13__tile_l0_c,) in pl.range(0, 256, 256, init_values=(stage_15_v13__tile_l0_init,)):
                stage_15_v13__tile_l0_a: pl.Tile[
                    [128, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_7, pl.const(0, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(stage_15_v11__tile, 0, stage_15_v13__tile_l0_ko, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "0:0"})
                stage_15_v13__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_8, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    stage_15_v12__tile, stage_15_v13__tile_l0_ko, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:0"}
                )
                stage_15_v13__tile_l0_a_1: pl.Tile[
                    [128, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_9, pl.const(0, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(stage_15_v11__tile, 0, stage_15_v13__tile_l0_ko + 128, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "0:1"})
                stage_15_v13__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    stage_15_v12__tile, stage_15_v13__tile_l0_ko + 128, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:1"}
                )
                stage_15_v13__tile_l0_c_acc: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.tile.matmul_acc(stage_15_v13__tile_l0_c, stage_15_v13__tile_l0_a, stage_15_v13__tile_l0_b, stage_15_v13__tile_l0_ko == 0)
                stage_15_v13__tile_l0_c_acc_1: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.tile.matmul_acc(stage_15_v13__tile_l0_c_acc, stage_15_v13__tile_l0_a_1, stage_15_v13__tile_l0_b_1, stage_15_v13__tile_l0_ko + 128 == 0)
                stage_15_v13__tile: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.yield_(stage_15_v13__tile_l0_c_acc_1)
            for stage_15_v15__idx_v0, (stage_15_v21,) in pl.range(1, 15, 2, init_values=(stage_15_v13__tile,)):
                stage_15_v16__ssa_v0: pl.Scalar[pl.INDEX] = stage_15_v15__idx_v0 * 256
                stage_15_v16__ssa_v0_1: pl.Scalar[pl.INDEX] = (stage_15_v15__idx_v0 + 1) * 256
                stage_15_v17__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_12, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[stage_15_v8__ssa_v0, 256])] = pl.tile.load(
                    temporary_51__ssa_v0, [stage_15_v9__ssa_v0, stage_15_v16__ssa_v0], [128, 256], [stage_15_v8__ssa_v0, 256], target_memory=pl.Mem.Mat, attrs={"pipeline_membership": "2:0"}
                )
                stage_15_v18__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_13, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    arg_wo_a__ssa_v0,
                    [7, stage_15_v10__ssa_v0 // 16, stage_15_v16__ssa_v0 // 16, 0, 0],
                    [1, 8, 16, 16, 16],
                    [1, 8, 16, 16, 16],
                    target_memory=pl.Mem.Mat,
                    attrs={"pipeline_membership": "2:0"},
                )
                stage_15_v17__tile_1: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_14, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[stage_15_v8__ssa_v0, 256])] = pl.tile.load(
                    temporary_51__ssa_v0, [stage_15_v9__ssa_v0, stage_15_v16__ssa_v0_1], [128, 256], [stage_15_v8__ssa_v0, 256], target_memory=pl.Mem.Mat, attrs={"pipeline_membership": "2:1"}
                )
                stage_15_v18__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_15, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
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
                        pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(stage_15_v17__tile, 0, stage_15_v21_l0_ko, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "1:0;2:0"})
                    stage_15_v21_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_17, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        stage_15_v18__tile, stage_15_v21_l0_ko, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:0;2:0"}
                    )
                    stage_15_v21_l0_a_1: pl.Tile[
                        [128, 128],
                        pl.BF16,
                        pl.MemRef(mem_left_18, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(stage_15_v17__tile, 0, stage_15_v21_l0_ko + 128, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "1:1;2:0"})
                    stage_15_v21_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_19, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        stage_15_v18__tile, stage_15_v21_l0_ko + 128, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:1;2:0"}
                    )
                    stage_15_v21_l0_c_acc: pl.Tile[
                        [128, 128],
                        pl.FP32,
                        pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(stage_15_v21_l0_c, stage_15_v21_l0_a, stage_15_v21_l0_b)
                    stage_15_v21_l0_c_acc_1: pl.Tile[
                        [128, 128],
                        pl.FP32,
                        pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(stage_15_v21_l0_c_acc, stage_15_v21_l0_a_1, stage_15_v21_l0_b_1)
                    stage_15_v21_1: pl.Tile[
                        [128, 128],
                        pl.FP32,
                        pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                    ] = pl.yield_(stage_15_v21_l0_c_acc_1)
                for stage_15_v21_l0_ko_1, (stage_15_v21_l0_c_1,) in pl.range(0, 256, 256, init_values=(stage_15_v21_1,)):
                    stage_15_v21_l0_a_2: pl.Tile[
                        [128, 128],
                        pl.BF16,
                        pl.MemRef(mem_left_21, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(stage_15_v17__tile_1, 0, stage_15_v21_l0_ko_1, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "1:0;2:1"})
                    stage_15_v21_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_22, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        stage_15_v18__tile_1, stage_15_v21_l0_ko_1, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:0;2:1"}
                    )
                    stage_15_v21_l0_a_3: pl.Tile[
                        [128, 128],
                        pl.BF16,
                        pl.MemRef(mem_left_23, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(stage_15_v17__tile_1, 0, stage_15_v21_l0_ko_1 + 128, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "1:1;2:1"})
                    stage_15_v21_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_24, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        stage_15_v18__tile_1, stage_15_v21_l0_ko_1 + 128, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:1;2:1"}
                    )
                    stage_15_v21_l0_c_acc_2: pl.Tile[
                        [128, 128],
                        pl.FP32,
                        pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(stage_15_v21_l0_c_1, stage_15_v21_l0_a_2, stage_15_v21_l0_b_2)
                    stage_15_v21_l0_c_acc_3: pl.Tile[
                        [128, 128],
                        pl.FP32,
                        pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(stage_15_v21_l0_c_acc_2, stage_15_v21_l0_a_3, stage_15_v21_l0_b_3)
                    stage_15_v21_2: pl.Tile[
                        [128, 128],
                        pl.FP32,
                        pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                    ] = pl.yield_(stage_15_v21_l0_c_acc_3)
                stage_15_v20_main: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.yield_(stage_15_v21_2)
            stage_15_v16__ssa_v0_2: pl.Scalar[pl.INDEX] = pl.const(15, pl.INDEX) * pl.const(256, pl.INDEX)
            stage_15_v17__tile_2: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_27, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[stage_15_v8__ssa_v0, 256])] = pl.tile.load(
                temporary_51__ssa_v0, [stage_15_v9__ssa_v0, stage_15_v16__ssa_v0_2], [128, 256], [stage_15_v8__ssa_v0, 256], target_memory=pl.Mem.Mat, attrs={"pipeline_membership": "3:0"}
            )
            stage_15_v18__tile_2: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_28, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
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
                    pl.MemRef(mem_left_29, pl.const(0, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(stage_15_v17__tile_2, 0, stage_15_v21_l0_ko_2, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "1:0;3:0"})
                stage_15_v21_l0_b_4: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_30, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    stage_15_v18__tile_2, stage_15_v21_l0_ko_2, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:0;3:0"}
                )
                stage_15_v21_l0_a_5: pl.Tile[
                    [128, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_31, pl.const(0, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(stage_15_v17__tile_2, 0, stage_15_v21_l0_ko_2 + 128, [128, 128], target_memory=pl.Mem.Left, attrs={"pipeline_membership": "1:1;3:0"})
                stage_15_v21_l0_b_5: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_32, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    stage_15_v18__tile_2, stage_15_v21_l0_ko_2 + 128, 0, [128, 128], target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:1;3:0"}
                )
                stage_15_v21_l0_c_acc_4: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.tile.matmul_acc(stage_15_v21_l0_c_2, stage_15_v21_l0_a_4, stage_15_v21_l0_b_4)
                stage_15_v21_l0_c_acc_5: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.tile.matmul_acc(stage_15_v21_l0_c_acc_4, stage_15_v21_l0_a_5, stage_15_v21_l0_b_5)
                stage_15_v21_3: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
                ] = pl.yield_(stage_15_v21_l0_c_acc_5)
            stage_15_v20: pl.Tile[
                [128, 128],
                pl.FP32,
                pl.MemRef(mem_acc_6, pl.const(0, pl.INT64), 65536),
                pl.Mem.Acc,
                pl.TileView(valid_shape=[pl.min(stage_15_v8__ssa_v0, pl.const(128, pl.INDEX)), 128], compact=pl.CompactMode.normal),
            ] = stage_15_v21_3
            stage_15_v14__tile: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
                stage_15_v20, [stage_15_v7__ssa_v0, stage_15_v10__ssa_v0 + 7168], temporary_63__ssa_v0
            )
            pl.system.fence()
            pl.system.cacheinvalid()
            pl.system.fence()
        return temporary_51__ssa_v0, temporary_63__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_static_full_card(
        self,
        temporary_51__ssa_v0: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)]],
        arg_wo_a__ssa_v0: pl.Tensor[
            [8, 64, 256, 16, 16], pl.BF16, pl.TensorView(stride=[4194304, 65536, 256, 16, 1], layout=pl.TensorLayout.NZ), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)
        ],
        temporary_63__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        records__ssa_v0: pl.Tensor[[31968], pl.INT64, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 255744)],
    ) -> tuple[pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32]]:
        pl.func_attr({"nz_tensor_views_blocked": True, "mx_tensor_views_blocked": True})
        _tuple_tmp__ssa_v0: pl.Tuple[pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32]] = self.csa_static_aic(
            temporary_51__ssa_v0, arg_wo_a__ssa_v0, temporary_63__ssa_v0, records__ssa_v0
        )
        temporary_51__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 25165824)] = _tuple_tmp__ssa_v0[0]
        temporary_63__ssa_v1: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 12582912)] = _tuple_tmp__ssa_v0[1]
        return temporary_51__ssa_v0, temporary_63__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(
        self,
        temporary_51__ssa_v0: pl.InOut[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)]],
        arg_wo_a__ssa_v0: pl.Tensor[
            [8, 64, 256, 16, 16], pl.BF16, pl.TensorView(stride=[4194304, 65536, 256, 16, 1], layout=pl.TensorLayout.NZ), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)
        ],
        temporary_63__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        records__ssa_v0: pl.Tensor[[31968], pl.INT64, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 255744)],
    ) -> tuple[pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True, "nz_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[3072, 4096], pl.BF16], pl.Tensor[[384, 8192], pl.FP32]] = self.csa_static_full_card(
            temporary_51__ssa_v0, arg_wo_a__ssa_v0, temporary_63__ssa_v0, records__ssa_v0, attrs={"core_num": 24, "sync_start": True}
        )
        temporary_51__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 25165824)] = ret__tmp_v0[0]
        temporary_63__ssa_v1: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0[1]
        return temporary_51__ssa_v1, temporary_63__ssa_v1
