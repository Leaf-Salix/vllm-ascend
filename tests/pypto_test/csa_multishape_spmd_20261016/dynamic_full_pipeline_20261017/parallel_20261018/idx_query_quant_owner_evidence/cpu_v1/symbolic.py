# pypto.program: _jit_decode_csa_spmd
import pypto.language as pl

CSA_FULL_ATTN_PLAN_ROWS = pl.dynamic("CSA_FULL_ATTN_PLAN_ROWS")
CSA_FULL_B = pl.dynamic("CSA_FULL_B")
CSA_FULL_CACHE_ROWS = pl.dynamic("CSA_FULL_CACHE_ROWS")
CSA_FULL_COEFF_COLS = pl.dynamic("CSA_FULL_COEFF_COLS")
CSA_FULL_COEFF_ROWS = pl.dynamic("CSA_FULL_COEFF_ROWS")
CSA_FULL_COMPRESSED_COLS = pl.dynamic("CSA_FULL_COMPRESSED_COLS")
CSA_FULL_GROUP_OFFSET_ROWS = pl.dynamic("CSA_FULL_GROUP_OFFSET_ROWS")
CSA_FULL_HEAD_RING_ROWS = pl.dynamic("CSA_FULL_HEAD_RING_ROWS")
CSA_FULL_IC = pl.dynamic("CSA_FULL_IC")
CSA_FULL_ICACHE_BYTES = pl.dynamic("CSA_FULL_ICACHE_BYTES")
CSA_FULL_ICACHE_PAGES = pl.dynamic("CSA_FULL_ICACHE_PAGES")
CSA_FULL_ICPAD = pl.dynamic("CSA_FULL_ICPAD")
CSA_FULL_IDX_HEAD_ROWS = pl.dynamic("CSA_FULL_IDX_HEAD_ROWS")
CSA_FULL_IDX_PROJ_ROWS = pl.dynamic("CSA_FULL_IDX_PROJ_ROWS")
CSA_FULL_IDX_ROPE_HEAD_ROWS = pl.dynamic("CSA_FULL_IDX_ROPE_HEAD_ROWS")
CSA_FULL_IDX_ROPE_ROWS = pl.dynamic("CSA_FULL_IDX_ROPE_ROWS")
CSA_FULL_IDX_SCALE_T_ROWS = pl.dynamic("CSA_FULL_IDX_SCALE_T_ROWS")
CSA_FULL_INDEXER_COLS = pl.dynamic("CSA_FULL_INDEXER_COLS")
CSA_FULL_INDEX_TRANSFER_COLS = pl.dynamic("CSA_FULL_INDEX_TRANSFER_COLS")
CSA_FULL_INDEX_TRANSFER_ROWS = pl.dynamic("CSA_FULL_INDEX_TRANSFER_ROWS")
CSA_FULL_IPOOL = pl.dynamic("CSA_FULL_IPOOL")
CSA_FULL_IPROJ = pl.dynamic("CSA_FULL_IPROJ")
CSA_FULL_IQ = pl.dynamic("CSA_FULL_IQ")
CSA_FULL_ISCALES = pl.dynamic("CSA_FULL_ISCALES")
CSA_FULL_ISLOTS = pl.dynamic("CSA_FULL_ISLOTS")
CSA_FULL_ISTATE_ELEMS = pl.dynamic("CSA_FULL_ISTATE_ELEMS")
CSA_FULL_ISTATE_PAGES = pl.dynamic("CSA_FULL_ISTATE_PAGES")
CSA_FULL_ITABLE_COLS = pl.dynamic("CSA_FULL_ITABLE_COLS")
CSA_FULL_KEY0_ROWS = pl.dynamic("CSA_FULL_KEY0_ROWS")
CSA_FULL_KEY64_ROWS = pl.dynamic("CSA_FULL_KEY64_ROWS")
CSA_FULL_KV_RING_ROWS = pl.dynamic("CSA_FULL_KV_RING_ROWS")
CSA_FULL_KV_ROWS = pl.dynamic("CSA_FULL_KV_ROWS")
CSA_FULL_MC = pl.dynamic("CSA_FULL_MC")
CSA_FULL_MCACHE_ROWS = pl.dynamic("CSA_FULL_MCACHE_ROWS")
CSA_FULL_MNORM = pl.dynamic("CSA_FULL_MNORM")
CSA_FULL_MPOOL = pl.dynamic("CSA_FULL_MPOOL")
CSA_FULL_MPROJ = pl.dynamic("CSA_FULL_MPROJ")
CSA_FULL_MQ = pl.dynamic("CSA_FULL_MQ")
CSA_FULL_MSLOTS = pl.dynamic("CSA_FULL_MSLOTS")
CSA_FULL_MSTATE_ELEMS = pl.dynamic("CSA_FULL_MSTATE_ELEMS")
CSA_FULL_MSTATE_PAGES = pl.dynamic("CSA_FULL_MSTATE_PAGES")
CSA_FULL_MTABLE_COLS = pl.dynamic("CSA_FULL_MTABLE_COLS")
CSA_FULL_ORIGINAL_COLS = pl.dynamic("CSA_FULL_ORIGINAL_COLS")
CSA_FULL_O_MM_ROWS = pl.dynamic("CSA_FULL_O_MM_ROWS")
CSA_FULL_O_SCALE_ROWS = pl.dynamic("CSA_FULL_O_SCALE_ROWS")
CSA_FULL_P16 = pl.dynamic("CSA_FULL_P16")
CSA_FULL_P8 = pl.dynamic("CSA_FULL_P8")
CSA_FULL_PACK_ROWS = pl.dynamic("CSA_FULL_PACK_ROWS")
CSA_FULL_PAIR_ROWS = pl.dynamic("CSA_FULL_PAIR_ROWS")
CSA_FULL_PARTIAL = pl.dynamic("CSA_FULL_PARTIAL")
CSA_FULL_QPROJ_ROWS = pl.dynamic("CSA_FULL_QPROJ_ROWS")
CSA_FULL_QR_ROWS = pl.dynamic("CSA_FULL_QR_ROWS")
CSA_FULL_QUERY_HEAD_ROWS = pl.dynamic("CSA_FULL_QUERY_HEAD_ROWS")
CSA_FULL_SCORE_ROWS = pl.dynamic("CSA_FULL_SCORE_ROWS")
CSA_FULL_T = pl.dynamic("CSA_FULL_T")
CSA_FULL_WEIGHTS_PARTIAL_ROWS = pl.dynamic("CSA_FULL_WEIGHTS_PARTIAL_ROWS")
CSA_FULL_WEIGHTS_ROWS = pl.dynamic("CSA_FULL_WEIGHTS_ROWS")
CSA_FULL_WORDS = pl.dynamic("CSA_FULL_WORDS")
RMS_NORM_CHUNK_D_DYN = pl.dynamic("RMS_NORM_CHUNK_D_DYN")


@pl.program
class _jit_decode_csa_spmd:
    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def _o_b_nz_groups_stage(
        self,
        o_r_i8_pad: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8],
        wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        partials: pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32],
        tokens: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
        group_begin: pl.Scalar[pl.INDEX],
        group_end: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32]:
        # NZ：owner=(g*8+dc)%WORKERS；每任务保持全部实际行块的原驻留次序。
        proj_b_t_rows: pl.Scalar[pl.INDEX] = (tokens + 96 - 1) // 96
        worker: pl.Scalar[pl.INDEX] = pl.max(worker_id, 0)
        group_first: pl.Scalar[pl.INDEX] = pl.max(group_begin, 0) * 8
        group_limit: pl.Scalar[pl.INDEX] = group_end * 8
        for round_id in pl.range(group_first // 24, (group_limit - worker + 24 - 1) // 24):
            work: pl.Scalar[pl.INDEX] = worker + round_id * 24
            if work >= group_first:
                g: pl.Scalar[pl.INDEX] = work // 8
                col_g: pl.Scalar[pl.INDEX] = g * 1024
                d0: pl.Scalar[pl.INDEX] = work % 8 * 512
                if pl.const(96, pl.INDEX) <= pl.const(96, pl.INDEX):
                    for tb in pl.range(proj_b_t_rows):
                        t0: pl.Scalar[pl.INDEX] = tb * 96
                        activation_l1: pl.Tile[[96, 1024], pl.INT8, pl.Mem.Mat] = pl.tile.load(o_r_i8_pad, [t0, col_g], [96, 1024], [96, 1024], target_memory=pl.Mem.Mat)
                        for nf in pl.range(2):
                            n0: pl.Scalar[pl.INDEX] = d0 + nf * 256
                            acc_b: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.create([96, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
                            for kb in pl.pipeline(4, stage=2):
                                k0: pl.Scalar[pl.INDEX] = kb * 256
                                wk0: pl.Scalar[pl.INDEX] = pl.max(g, 0) * 1024 + k0
                                b_weight_l1: pl.Tile[[256, 256], pl.INT8, pl.Mem.Mat] = pl.tile.load(wo_b, [wk0, n0], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
                                b_act_l1: pl.Tile[[96, 256], pl.INT8, pl.Mem.Mat] = pl.tile.slice(activation_l1, [96, 256], [0, k0])
                                acc_b: pl.Tile[[96, 256], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(acc_b, b_act_l1, b_weight_l1, kb == 0)
                            pl.tile.store(acc_b, [t0, g * 4096 + n0], partials)
                else:
                    for nf_1 in pl.range(2):
                        n0: pl.Scalar[pl.INDEX] = d0 + nf_1 * 256
                        weight_k0: pl.Scalar[pl.INDEX] = pl.max(g, 0) * 1024
                        weight_resident: pl.Tensor[[1024, 256], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wo_b, [1024, 256], [weight_k0, n0])
                        for tb_1 in pl.range(proj_b_t_rows):
                            t0: pl.Scalar[pl.INDEX] = tb_1 * 96
                            b_act_full: pl.Tensor[[96, 1024], pl.INT8] = pl.tensor.slice(o_r_i8_pad, [96, 1024], [t0, col_g])
                            full_acc: pl.Tensor[[96, 256], pl.INT32] = pl.tensor.matmul(b_act_full, weight_resident, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                            partials: pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32] = pl.tensor.assemble(partials, full_acc, [t0, g * 4096 + n0])
        return partials

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def _q_b_nz_stage(
        self,
        wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        qr_i8_matmul: pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8],
        q_proj_i32: pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32],
        tokens: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32]:
        # NZ Q_B：显式 AIC 按列块条带领取，原 K 流水不变；调用前必须等待 QR。
        qproj_worker: pl.Scalar[pl.INDEX] = pl.max(worker_id, 0)
        qproj_t_matmul: pl.Scalar[pl.INDEX] = (tokens + 128 - 1) // 128 * 128
        for qproj_round in pl.range((128 - qproj_worker + 24 - 1) // 24):
            qproj_n_idx: pl.Scalar[pl.INDEX] = qproj_worker + qproj_round * 24
            w_col0: pl.Scalar[pl.INDEX] = qproj_n_idx * 256
            for t0 in pl.range(0, qproj_t_matmul, 128):
                m_rows: pl.Scalar[pl.INDEX] = pl.min(128, tokens - t0)
                qr_first: pl.Tensor[[128, 256], pl.INT8, pl.TensorView(valid_shape=[m_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                    qr_i8_matmul, [128, 256], [t0, 0], [m_rows, 256]
                )
                wq_first: pl.Tensor[[256, 256], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wq_b, [256, 256], [0, w_col0])
                col_acc: pl.Tensor[[128, 256], pl.INT32] = pl.tensor.matmul(qr_first, wq_first, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                for k0 in pl.pipeline(256, 1024, 256, stage=2):
                    qr_chunk: pl.Tensor[[128, 256], pl.INT8, pl.TensorView(valid_shape=[m_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                        qr_i8_matmul, [128, 256], [t0, k0], [m_rows, 256]
                    )
                    wq_chunk: pl.Tensor[[256, 256], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wq_b, [256, 256], [k0, w_col0])
                    col_acc: pl.Tensor[[128, 256], pl.INT32] = pl.tensor.matmul_acc(col_acc, qr_chunk, wq_chunk, a_trans=False, b_trans=False)
                q_proj_i32: pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32] = pl.tensor.assemble(q_proj_i32, col_acc, [t0, w_col0])
        return q_proj_i32

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def attention_qk_pv_cube_stage(
        self,
        q_flat: pl.Tensor[[CSA_FULL_QUERY_HEAD_ROWS, 512], pl.BF16],
        valid_block_mask: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32],
        kv_transfer: pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16],
        score_transfer: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32],
        probability_transfer: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16],
        pv_transfer: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32],
        ffts_workspace: pl.Tensor[[256], pl.INT64],
        core_id: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32], pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32]]:
        # AIC侧：QK领先PV两拍；必须与本组两个AIV helper并发。
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(q_flat, 0) // 64
        qk_core: pl.Scalar[pl.INDEX] = core_id
        pl.system.set_ffts(ffts_workspace)
        qk_query_count: pl.Scalar[pl.INDEX] = (t_dim + 24 - 1 - qk_core) // 24
        qk_block_count: pl.Scalar[pl.INDEX] = qk_query_count * 5
        qk_q: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.create([64, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat)
        qk_l1: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.tile.create([384, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat)
        for qk_tick in pl.range(qk_block_count + 2):
            if qk_tick < qk_block_count:
                qk_sb: pl.Scalar[pl.INDEX] = qk_tick % 5
                qk_t: pl.Scalar[pl.INDEX] = qk_core + qk_tick // 5 * 24
                if qk_sb == 0:
                    qk_q: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.gather_row(qk_q, q_flat, [0, 0], [qk_t * 64, 0], [64, 512], transpose=False)
                if pl.cast(pl.tensor.read(valid_block_mask, [qk_t, qk_sb]), pl.INDEX) > 0:
                    qk_slot: pl.Scalar[pl.INDEX] = qk_core * 3 + qk_tick % 3
                    qk_kv_row: pl.Scalar[pl.INDEX] = qk_slot * 128
                    qk_transfer_row: pl.Scalar[pl.INDEX] = qk_slot * 64
                    pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    qk_l1_row: pl.Scalar[pl.INDEX] = qk_tick % 3 * 128
                    qk_l1: pl.Tile[[384, 512], pl.BF16, pl.Mem.Mat] = pl.tile.gather_row(qk_l1, kv_transfer, [qk_l1_row, 0], [qk_kv_row, 0], [128, 512], transpose=False)
                    qk_l1_t: pl.Tile[[512, 384], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(qk_l1)
                    qk_kv_t: pl.Tile[[512, 128], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.slice(
                        qk_l1_t, [512, 128], [0, qk_l1_row]
                    )
                    qk_scores: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(qk_q, qk_kv_t)
                    pl.tile.store(qk_scores, [qk_transfer_row, 0], score_transfer)
                    pl.system.sync_set(3, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            if qk_tick >= 2:
                pv_work: pl.Scalar[pl.INDEX] = qk_tick - 2
                pv_sb: pl.Scalar[pl.INDEX] = pv_work % 5
                pv_t: pl.Scalar[pl.INDEX] = qk_core + pv_work // 5 * 24
                if pl.cast(pl.tensor.read(valid_block_mask, [pv_t, pv_sb]), pl.INDEX) > 0:
                    pv_slot: pl.Scalar[pl.INDEX] = qk_core * 3 + pv_work % 3
                    pv_transfer_row: pl.Scalar[pl.INDEX] = pv_slot * 64
                    pl.system.sync_wait(4, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    pv_probability: pl.Tile[[64, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(probability_transfer, [pv_transfer_row, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat)
                    pv_l1_row: pl.Scalar[pl.INDEX] = pv_work % 3 * 128
                    pv_probability_left: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(pv_probability, target_memory=pl.Mem.Left)
                    pv_first_right: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(qk_l1, pv_l1_row, 0, [128, 128], target_memory=pl.Mem.Right)
                    pv_previous: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(pv_probability_left, pv_first_right)
                    for pv_n in pl.unroll(1, 4):
                        pv_next_right: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(qk_l1, pv_l1_row, pv_n * 128, [128, 128], target_memory=pl.Mem.Right)
                        pv_current: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(pv_probability_left, pv_next_right)
                        pl.tile.store(pv_previous, [pv_transfer_row, (pv_n - 1) * 128], pv_transfer)
                        pv_previous: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pv_current
                    pl.tile.store(pv_previous, [pv_transfer_row, 384], pv_transfer)
                    pl.system.sync_set(5, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
        return score_transfer, pv_transfer

    @pl.function(type=pl.FunctionType.Inline)
    def csa_signal_notify(self, signal: pl.Tensor[[CSA_FULL_WORDS], pl.INT32], offset: pl.Scalar[pl.INDEX], value: pl.Scalar[pl.INT32]) -> pl.Tensor[[CSA_FULL_WORDS], pl.INT32]:
        # 先排空本核流水，再发布指定槽；返回同一 GM Tensor 的引用。
        pl.system.bar_all()
        pl.system.signal_notify(signal, [offset], value)
        return signal

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def begin_invocation(self, signals: pl.Tensor[[CSA_FULL_WORDS], pl.INT32], core_id: pl.Scalar[pl.INDEX]):
        # 每核从自己的阶段 0 槽取下一代；返回 Tensor 引用和 INT32 代际。
        #
        #         这里没有全卡集合。各核从零开始且每调用各推进一次，才保证相同代际。
        #         输入合法性与容量由捕获前的调用层保证，不能在越界后依赖信号自愈。
        #
        own_offset: pl.Scalar[pl.INDEX] = core_id * 128
        pl.system.cacheinvalid(signals, [1], [own_offset])
        pl.system.fence()
        prior: pl.Scalar[pl.INT32] = pl.tensor.read(signals, [own_offset])
        generation: pl.Scalar[pl.INT32] = prior + pl.cast(1, pl.INT32)
        signals_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.csa_signal_notify(signals, own_offset, generation)
        return signals_v1, generation

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def compressor_project_stage(
        self,
        x: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        wkv: pl.Tensor[[1024, 4096], pl.BF16],
        wgate: pl.Tensor[[1024, 4096], pl.BF16],
        kv_proj_pad: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        score_proj_pad: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32], pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32]]:
        # 主压缩投影；一个 AIC worker 独占若干 M64/N64 块及两份完整 K 归约。
        bs: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        t_matmul: pl.Scalar[pl.INDEX] = (bs + 64 - 1) // 64 * 64
        for idx in pl.range(worker_id, t_matmul * 1024 // 4096, 24):
            global_row0: pl.Scalar[pl.INDEX] = idx // 16 * 64
            o0: pl.Scalar[pl.INDEX] = idx % 16 * 64
            x_rows: pl.Scalar[pl.INDEX] = pl.min(64, bs - global_row0)
            kv_acc: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.create([64, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            score_acc: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.create([64, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for kb in pl.pipeline(8, stage=2):
                k0: pl.Scalar[pl.INDEX] = kb * 512
                x_tile: pl.Tensor[[64, 512], pl.BF16, pl.TensorView(valid_shape=[x_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x, [64, 512], [global_row0, k0], [x_rows, 512])
                wkv_tile: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(wkv, [64, 512], [o0, k0])
                wgate_tile: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(wgate, [64, 512], [o0, k0])
                if kb == 0:
                    kv_acc: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.matmul(x_tile, wkv_tile, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                    score_acc: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.matmul(x_tile, wgate_tile, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                else:
                    kv_acc: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.matmul_acc(kv_acc, x_tile, wkv_tile, a_trans=False, b_trans=True)
                    score_acc: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.matmul_acc(score_acc, x_tile, wgate_tile, a_trans=False, b_trans=True)
            kv_proj_pad: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32] = pl.tensor.assemble(kv_proj_pad, kv_acc, [global_row0, o0])
            score_proj_pad: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32] = pl.tensor.assemble(score_proj_pad, score_acc, [global_row0, o0])
        return kv_proj_pad, score_proj_pad

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def hc_project_range(
        self,
        x: pl.Tensor[[CSA_FULL_T, 16384], pl.FP32],
        hc_fn: pl.Tensor[[24, 16384], pl.FP32],
        mixes_partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32],
        group_worker: pl.Scalar[pl.INDEX],
        group_workers: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32]:
        # AIC：M16、四段 K4096，每段仍按原 16 次 K256 顺序累加。
        #
        #         调用前等待 HC_INPUT，确保所有本核输入行的 FP32 写回完成；返回后发布
        #         HC_PARTIAL。逻辑 task=(行块×4+K段)，多波次属于同一组核，不另派任务。
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        t_linear: pl.Scalar[pl.INDEX] = (t_dim + 16 - 1) // 16 * 16
        x_flat: pl.Tensor[[CSA_FULL_T, 16384], pl.FP32] = x
        linear_units: pl.Scalar[pl.INDEX] = t_linear // 16 * 4
        for wave in pl.range((linear_units + group_workers - 1) // group_workers):
            task: pl.Scalar[pl.INDEX] = group_worker + wave * group_workers
            if task < linear_units:
                t0: pl.Scalar[pl.INDEX] = task // 4 * 16
                linear_split: pl.Scalar[pl.INDEX] = task % 4
                k_base: pl.Scalar[pl.INDEX] = linear_split * 4096
                t_rows: pl.Scalar[pl.INDEX] = pl.min(16, t_dim - t0)
                acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.create([16, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for kb in pl.pipeline(16, stage=2):
                    k0: pl.Scalar[pl.INDEX] = k_base + kb * 256
                    x_linear_chunk: pl.Tensor[[16, 256], pl.FP32, pl.TensorView(valid_shape=[t_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                        x_flat, [16, 256], [t0, k0], [t_rows, 256]
                    )
                    w_chunk: pl.Tensor[[32, 256], pl.FP32, pl.TensorView(valid_shape=[24, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(hc_fn, [32, 256], [0, k0], [24, 256])
                    acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul_acc(acc, x_linear_chunk, w_chunk, init_cond=kb == 0, a_trans=False, b_trans=True)
                partial_row0: pl.Scalar[pl.INDEX] = linear_split * t_linear + t0
                mixes_partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32] = pl.tensor.assemble(mixes_partials, acc, [partial_row0, 0])
        return mixes_partials

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_compressor_hadamard_stage(
        self,
        normed: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16],
        hadamard: pl.Tensor[[128, 128], pl.BF16],
        output: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32],
        valid_count: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32]:
        # AIC消费device实际关闭行数；空C不读输入，尾M16仅load valid_shape行。
        rms_blocks: pl.Scalar[pl.INDEX] = (valid_count + 16 - 1) // 16
        padded: pl.Scalar[pl.INDEX] = rms_blocks * 16
        if valid_count > 0 and padded <= pl.tensor.dim(normed, 0) and padded <= pl.tensor.dim(output, 0):
            for logical in pl.range(worker_id, 2 * rms_blocks, 24):
                o0: pl.Scalar[pl.INDEX] = logical // rms_blocks * 64
                had_blk: pl.Scalar[pl.INDEX] = logical % rms_blocks
                hadamard_tile: pl.Tensor[[128, 64], pl.BF16] = pl.tensor.slice(hadamard, [128, 64], [0, o0])
                had_b0: pl.Scalar[pl.INDEX] = had_blk * 16
                had_rows: pl.Scalar[pl.INDEX] = pl.min(16, valid_count - had_b0)
                kv_proj_tile: pl.Tensor[[16, 128], pl.BF16, pl.TensorView(valid_shape=[had_rows, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                    normed, [16, 128], [had_b0, 0], [had_rows, 128]
                )
                kv_hadamard_acc: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul(kv_proj_tile, hadamard_tile, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                output: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32] = pl.tensor.assemble(output, kv_hadamard_acc, [had_b0, o0])
        return output

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_compressor_project_stage(
        self,
        x: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        wkv: pl.Tensor[[256, 4096], pl.BF16],
        wgate: pl.Tensor[[256, 4096], pl.BF16],
        kv_proj_pad: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        score_proj_pad: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32], pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32]]:
        # Indexer压缩投影；M16/N32、BF16转置权重与原init_cond累加顺序不变。
        bs: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        t_matmul: pl.Scalar[pl.INDEX] = (bs + 16 - 1) // 16 * 16
        for idx in pl.range(worker_id, t_matmul * 256 // 512, 24):
            global_row0: pl.Scalar[pl.INDEX] = idx // 8 * 16
            o0: pl.Scalar[pl.INDEX] = idx % 8 * 32
            kv_acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.create([16, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            score_acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.create([16, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for kb in pl.pipeline(8, stage=2):
                k0: pl.Scalar[pl.INDEX] = kb * 512
                x_rows: pl.Scalar[pl.INDEX] = pl.min(16, bs - global_row0)
                x_tile: pl.Tensor[[16, 512], pl.BF16, pl.TensorView(valid_shape=[x_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x, [16, 512], [global_row0, k0], [x_rows, 512])
                wkv_tile: pl.Tensor[[32, 512], pl.BF16] = pl.tensor.slice(wkv, [32, 512], [o0, k0])
                wgate_tile: pl.Tensor[[32, 512], pl.BF16] = pl.tensor.slice(wgate, [32, 512], [o0, k0])
                kv_acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul_acc(kv_acc, x_tile, wkv_tile, init_cond=kb == 0, a_trans=False, b_trans=True)
                score_acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul_acc(score_acc, x_tile, wgate_tile, init_cond=kb == 0, a_trans=False, b_trans=True)
            kv_proj_pad: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32] = pl.tensor.assemble(kv_proj_pad, kv_acc, [global_row0, o0])
            score_proj_pad: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32] = pl.tensor.assemble(score_proj_pad, score_acc, [global_row0, o0])
        return kv_proj_pad, score_proj_pad

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_query_hadamard_stage(
        self,
        qr_bf16: pl.Tensor[[CSA_FULL_IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
        hadamard: pl.Tensor[[128, 128], pl.BF16],
        qh_acc_gm: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32],
        tokens: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32]:
        # AIC：先等BF16 query所有head；64行一块即一个token，保持原完整K128矩阵乘。
        bs_heads: pl.Scalar[pl.INDEX] = tokens * 64
        qh_worker: pl.Scalar[pl.INDEX] = worker_id
        qh_hadamard: pl.Tensor[[128, 128], pl.BF16] = pl.tensor.slice(hadamard, [128, 128], [0, 0])
        for idx in pl.range(qh_worker, bs_heads // 64, 24):
            o0: pl.Scalar[pl.INDEX] = idx * 64
            qh_acc: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul(pl.tensor.slice(qr_bf16, [64, 128], [o0, 0]), qh_hadamard, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
            qh_acc_gm: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32] = pl.tensor.assemble(qh_acc_gm, qh_acc, [o0, 0])
        return qh_acc_gm

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_query_project_stage(
        self, qr: pl.Tensor[[CSA_FULL_T, 1024], pl.INT8], wq_b: pl.Tensor[[1024, 8192], pl.INT8], qr_acc_pad: pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32], worker_id: pl.Scalar[pl.INDEX]
    ) -> pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32]:
        # AIC：按N256条带投影，先等QR；返回后发布query投影完成。
        bs: pl.Scalar[pl.INDEX] = pl.tensor.dim(qr, 0)
        row_blocks: pl.Scalar[pl.INDEX] = (bs + 64 - 1) // 64
        qr_proj_worker: pl.Scalar[pl.INDEX] = worker_id
        for qr_col in pl.range(qr_proj_worker, 32, 24):
            o_base: pl.Scalar[pl.INDEX] = qr_col * 256
            wq_tile: pl.Tensor[[1024, 256], pl.INT8] = pl.tensor.slice(wq_b, [1024, 256], [0, o_base])
            for qr_rb in pl.range(row_blocks):
                qr_r0: pl.Scalar[pl.INDEX] = qr_rb * 64
                qr_rows: pl.Scalar[pl.INDEX] = pl.min(64, bs - qr_r0)
                qr_tile: pl.Tensor[[64, 1024], pl.INT8, pl.TensorView(valid_shape=[qr_rows, 1024], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(qr, [64, 1024], [qr_r0, 0], [qr_rows, 1024])
                qr_acc: pl.Tensor[[64, 256], pl.INT32] = pl.tensor.matmul(qr_tile, wq_tile, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                qr_acc_pad: pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32] = pl.tensor.assemble(qr_acc_pad, qr_acc, [qr_r0, o_base])
        return qr_acc_pad

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_decode_group(
        self,
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        offsets: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32],
        group: pl.Scalar[pl.INDEX],
    ):
        # 从本轮已发布的组描述符返回几何；越界组不读取未初始化的容量尾部。
        owner: pl.Scalar[pl.INDEX] = pl.cast(-1, pl.INDEX)
        first: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        live: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        visible: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        requests: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
        groups: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(offsets, [requests]), pl.INDEX)
        if group >= 0 and group < groups:
            descriptor_base: pl.Scalar[pl.INDEX] = (requests + 1 + 16 - 1) // 16 * 16
            row: pl.Scalar[pl.INDEX] = descriptor_base + group * 4
            owner: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(offsets, [row]), pl.INDEX)
            first: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(offsets, [row + 1]), pl.INDEX)
            live: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(offsets, [row + 2]), pl.INDEX)
            visible: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(offsets, [row + 3]), pl.INDEX)
        return owner, first, live, visible

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_leaf_range(self, leaf: pl.Scalar[pl.INDEX], leaf_tiles: pl.Scalar[pl.INDEX], extra: pl.Scalar[pl.INDEX], visible: pl.Scalar[pl.INDEX]):
        # 先分1024候选的整tile，再按组/单query实际visible截断；返回begin/capacity/valid。
        begin: pl.Scalar[pl.INDEX] = (leaf * leaf_tiles + pl.min(leaf, extra)) * 1024
        capacity: pl.Scalar[pl.INDEX] = (leaf_tiles + pl.cast(leaf < extra, pl.INDEX)) * 1024
        valid: pl.Scalar[pl.INDEX] = pl.max(pl.min(visible - begin, capacity), 0)
        return begin, capacity, valid

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_work_item(self, item: pl.Scalar[pl.INDEX], groups: pl.Scalar[pl.INDEX], leaves: pl.Scalar[pl.INDEX]):
        # 少组时先铺满不同组，多组时同组连续叶；空输入返回-1而不除零。
        group: pl.Scalar[pl.INDEX] = pl.cast(-1, pl.INDEX)
        leaf: pl.Scalar[pl.INDEX] = pl.cast(-1, pl.INDEX)
        if groups > 0 and leaves > 0 and item >= 0 and item < groups * leaves:
            if groups < 24:
                group: pl.Scalar[pl.INDEX] = item % groups
                leaf: pl.Scalar[pl.INDEX] = item // groups
            else:
                group: pl.Scalar[pl.INDEX] = item // leaves
                leaf: pl.Scalar[pl.INDEX] = item % leaves
        return group, leaf

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_score_cube_stage(
        self,
        qr_hadamard_i8: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8],
        coefficients: pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16],
        idx_kv_cache: pl.Tensor[[CSA_FULL_KEY0_ROWS, 128], pl.INT8],
        idx_kv_cache_shift64: pl.Tensor[[CSA_FULL_KEY64_ROWS, 128], pl.INT8],
        idx_native_kv_cache: pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8],
        idx_block_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_INDEXER_COLS], pl.INT32],
        position_ids: pl.Tensor[[CSA_FULL_T], pl.INT64],
        kv_seq_lens: pl.Tensor[[CSA_FULL_B], pl.INT32],
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        offsets: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32],
        plan: pl.Tensor[[8], pl.INT32],
        buf_score_transfer: pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32],
        buf_worker: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32]:
        # AIC生产FP32 score双槽；须与两个vector_lane同MIX并发，外层/运行时已设置同一FFTS域。
        native_page_bytes: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_native_kv_cache, 1)
        groups: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [0]), pl.INDEX)
        buf_max_leaves: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [2]), pl.INDEX)
        buf_leaf_tiles: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [3]), pl.INDEX)
        buf_extra_leaves: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [4]), pl.INDEX)
        for buf_item in pl.range(buf_worker, groups * buf_max_leaves, 24):
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]] = self.indexer_work_item(buf_item, groups, buf_max_leaves)
            buf_group: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            buf_leaf: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            _tuple_tmp_1: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]] = self.indexer_decode_group(
                bounds, kv_seq_lens, position_ids, offsets, buf_group
            )
            buf_batch_idx: pl.Scalar[pl.INDEX] = _tuple_tmp_1[0]
            buf_query: pl.Scalar[pl.INDEX] = _tuple_tmp_1[1]
            buf_live: pl.Scalar[pl.INDEX] = _tuple_tmp_1[2]
            buf_visible_count: pl.Scalar[pl.INDEX] = _tuple_tmp_1[3]
            _tuple_tmp_2: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]] = self.indexer_leaf_range(buf_leaf, buf_leaf_tiles, buf_extra_leaves, buf_visible_count)
            buf_logical_begin: pl.Scalar[pl.INDEX] = _tuple_tmp_2[0]
            buf_leaf_capacity: pl.Scalar[pl.INDEX] = _tuple_tmp_2[1]
            work_valid: pl.Scalar[pl.INDEX] = _tuple_tmp_2[2]
            if work_valid > 0:
                buf_cache_len: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(kv_seq_lens, [buf_batch_idx]), pl.INDEX) // 4
                buf_valid_count: pl.Scalar[pl.INDEX] = pl.min(buf_leaf_capacity, buf_visible_count - buf_logical_begin)
                buf_tile_count: pl.Scalar[pl.INDEX] = (buf_valid_count + 1024 - 1) // 1024
                buf_lane_span: pl.Scalar[pl.INDEX] = pl.min(buf_tile_count * 512, 4096)
                buf_score_iters: pl.Scalar[pl.INDEX] = (buf_lane_span + 512 - 1) // 512
                buf_query_vector: pl.Tile[[384, 128], pl.INT8, pl.Mem.Mat, pl.TileView(valid_shape=[buf_live * pl.const(64, pl.INDEX), 128])] = pl.tile.load(
                    qr_hadamard_i8, [buf_query * 64, 0], [384, 128], [buf_live * 64, 128], target_memory=pl.Mem.Mat
                )
                buf_query_vector_v1: pl.Tile[[384, 128], pl.INT8, pl.Mem.Mat] = pl.tile.set_validshape(buf_query_vector, 384, 128)
                for padding_lane in pl.unroll(6):
                    if padding_lane >= buf_live:
                        buf_query_vector_v1: pl.Tile[[384, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(
                            buf_query_vector_v1, qr_hadamard_i8, [padding_lane * 64, 0], [buf_query * 64, 0], [64, 128], transpose=False
                        )
                buf_coefficients_l1: pl.Tile[[16, 384], pl.FP16, pl.Mem.Mat] = pl.tile.load(coefficients, [buf_group * 16, 0], [16, 384], [16, 384], target_memory=pl.Mem.Mat)
                buf_query_left: pl.Tile[[384, 128], pl.INT8, pl.Mem.Left] = pl.tile.move(buf_query_vector_v1, target_memory=pl.Mem.Left)
                buf_coefficient_pair: pl.Tile[[16, 384], pl.FP16, pl.Mem.Left] = pl.tile.move(buf_coefficients_l1, target_memory=pl.Mem.Left)
                for buf_score_step in pl.range(buf_score_iters):
                    if buf_score_step >= 2:
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    buf_score_begin: pl.Scalar[pl.INDEX] = buf_score_step * 512
                    if True:
                        buf_score_begin: pl.Scalar[pl.INDEX] = buf_score_step * 1024
                    buf_transfer_row: pl.Scalar[pl.INDEX] = buf_worker * 2 * 6 + buf_score_step % 2 * 6
                    buf_previous_l1: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                    buf_current_key: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.create([128, 64], dtype=pl.INT8, target_memory=pl.Mem.Right)
                    buf_prefetched_key: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.create([128, 64], dtype=pl.INT8, target_memory=pl.Mem.Right)
                    buf_key_l1_pool: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                    if pl.const(1, pl.INDEX) > pl.const(0, pl.INDEX):
                        buf_prime_col: pl.Scalar[pl.INDEX] = 0
                        for buf_prime_key_page in pl.unroll(2):
                            buf_prime_key_row: pl.Scalar[pl.INDEX] = buf_logical_begin + buf_score_begin + buf_prime_col // 512 * buf_lane_span + buf_prime_col % 512 + buf_prime_key_page * 32
                            if True:
                                buf_prime_key_row: pl.Scalar[pl.INDEX] = buf_logical_begin + buf_score_begin + buf_prime_col + buf_prime_key_page * 32
                            buf_prime_safe_page: pl.Scalar[pl.INDEX] = pl.min(buf_prime_key_row // 32, pl.max((buf_cache_len - 1) // 32, 0))
                            if buf_prime_safe_page >= 0 and buf_prime_safe_page < pl.tensor.dim(idx_block_table, 1):
                                buf_prime_physical_page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(idx_block_table, [buf_batch_idx, buf_prime_safe_page]), pl.INDEX)
                                if buf_prime_physical_page >= 0 and buf_prime_physical_page < pl.tensor.dim(idx_native_kv_cache, 0):
                                    buf_prime_native_byte: pl.Scalar[pl.INDEX] = buf_prime_physical_page * native_page_bytes
                                    if buf_prime_native_byte % 128 == 0:
                                        buf_key_l1_pool: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(
                                            buf_key_l1_pool, idx_kv_cache, [buf_prime_key_page * 32, 0], [buf_prime_native_byte // 128, 0], [32, 128], transpose=False
                                        )
                                    else:
                                        buf_key_l1_pool: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(
                                            buf_key_l1_pool, idx_kv_cache_shift64, [buf_prime_key_page * 32, 0], [buf_prime_native_byte // 128, 0], [32, 128], transpose=False
                                        )
                        if True:
                            buf_current_key: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(pl.tile.transpose_view(buf_key_l1_pool), 0, 0, [128, 64], target_memory=pl.Mem.Right)
                    for buf_qk_panel in pl.unroll(16):
                        buf_panel_col: pl.Scalar[pl.INDEX] = buf_qk_panel * 64
                        buf_load_col: pl.Scalar[pl.INDEX] = (buf_qk_panel + 1) * 64
                        buf_load_slot: pl.Scalar[pl.INDEX] = (buf_qk_panel + 1) % 2 * 64
                        if pl.const(1, pl.INDEX) > pl.const(0, pl.INDEX):
                            if not True:
                                buf_current_key: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(
                                    pl.tile.transpose_view(buf_key_l1_pool), 0, buf_qk_panel % 2 * 64, [128, 64], target_memory=pl.Mem.Right
                                )
                        if buf_qk_panel + 1 < 16:
                            buf_panel_keys: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
                            for buf_key_page in pl.unroll(2):
                                buf_key_row: pl.Scalar[pl.INDEX] = buf_logical_begin + buf_score_begin + buf_load_col // 512 * buf_lane_span + buf_load_col % 512 + buf_key_page * 32
                                if True:
                                    buf_key_row: pl.Scalar[pl.INDEX] = buf_logical_begin + buf_score_begin + buf_load_col + buf_key_page * 32
                                buf_safe_page: pl.Scalar[pl.INDEX] = pl.min(buf_key_row // 32, pl.max((buf_cache_len - 1) // 32, 0))
                                if buf_safe_page >= 0 and buf_safe_page < pl.tensor.dim(idx_block_table, 1):
                                    buf_physical_page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(idx_block_table, [buf_batch_idx, buf_safe_page]), pl.INDEX)
                                    if buf_physical_page >= 0 and buf_physical_page < pl.tensor.dim(idx_native_kv_cache, 0):
                                        buf_native_byte: pl.Scalar[pl.INDEX] = buf_physical_page * native_page_bytes
                                        if buf_native_byte % 128 == 0:
                                            if pl.const(1, pl.INDEX) > pl.const(0, pl.INDEX):
                                                buf_key_l1_pool: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(
                                                    buf_key_l1_pool, idx_kv_cache, [buf_load_slot + buf_key_page * 32, 0], [buf_native_byte // 128, 0], [32, 128], transpose=False
                                                )
                                            else:
                                                buf_panel_keys: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(
                                                    buf_panel_keys, idx_kv_cache, [buf_key_page * 32, 0], [buf_native_byte // 128, 0], [32, 128], transpose=False
                                                )
                                        else:
                                            if pl.const(1, pl.INDEX) > pl.const(0, pl.INDEX):
                                                buf_key_l1_pool: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(
                                                    buf_key_l1_pool, idx_kv_cache_shift64, [buf_load_slot + buf_key_page * 32, 0], [buf_native_byte // 128, 0], [32, 128], transpose=False
                                                )
                                            else:
                                                buf_panel_keys: pl.Tile[[64, 128], pl.INT8, pl.Mem.Mat] = pl.tile.gather_row(
                                                    buf_panel_keys, idx_kv_cache_shift64, [buf_key_page * 32, 0], [buf_native_byte // 128, 0], [32, 128], transpose=False
                                                )
                            if pl.const(1, pl.INDEX) > pl.const(0, pl.INDEX):
                                if True:
                                    buf_prefetched_key: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.extract(
                                        pl.tile.transpose_view(buf_key_l1_pool), 0, buf_load_slot, [128, 64], target_memory=pl.Mem.Right
                                    )
                            else:
                                buf_prefetched_key: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = pl.tile.move(pl.tile.transpose_view(buf_panel_keys), target_memory=pl.Mem.Right)
                        if pl.const(1, pl.INDEX) > pl.const(0, pl.INDEX):
                            buf_key_panel: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = buf_current_key
                        else:
                            buf_key_panel: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = buf_prefetched_key
                        buf_scores_l1: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
                        if buf_qk_panel > 0:
                            buf_previous_pair: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(buf_previous_l1, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                            buf_scores: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(buf_query_left, buf_key_panel)
                            buf_reduced_pair: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(buf_coefficient_pair, buf_previous_pair)
                            buf_scores_l1: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(buf_scores_l1, buf_scores, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                            pl.tile.store(pl.tile.set_validshape(buf_reduced_pair, 6, 64), [buf_transfer_row, buf_panel_col - 64], buf_score_transfer)
                        else:
                            buf_scores: pl.Tile[[384, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(buf_query_left, buf_key_panel)
                            buf_scores_l1: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = pl.tile.assemble(buf_scores_l1, buf_scores, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                        buf_previous_l1: pl.Tile[[384, 64], pl.FP16, pl.Mem.Mat] = buf_scores_l1
                        if pl.const(1, pl.INDEX) > pl.const(0, pl.INDEX):
                            if True:
                                if buf_qk_panel + 1 < 16:
                                    buf_current_key: pl.Tile[[128, 64], pl.INT8, pl.Mem.Right] = buf_prefetched_key
                    buf_last_pair: pl.Tile[[384, 64], pl.FP16, pl.Mem.Right] = pl.tile.extract(buf_previous_l1, 0, 0, [384, 64], target_memory=pl.Mem.Right)
                    buf_last_scores: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(buf_coefficient_pair, buf_last_pair)
                    pl.tile.store(pl.tile.set_validshape(buf_last_scores, 6, 64), [buf_transfer_row, 960], buf_score_transfer)
                    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                for buf_score_drain in pl.range(pl.min(buf_score_iters, 2)):
                    pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
        return buf_score_transfer

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_weights_project_stage(
        self,
        x: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        weights_proj: pl.Tensor[[4096, 64], pl.BF16],
        weights_partial: pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32]:
        # AIC：等x_normed；M16×四段K1024，每段两个K512原顺序，写动态四份partial。
        bs: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        row_blocks: pl.Scalar[pl.INDEX] = (bs + 16 - 1) // 16
        t_pad: pl.Scalar[pl.INDEX] = row_blocks * 16
        x_flat: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16] = x
        w_worker: pl.Scalar[pl.INDEX] = worker_id
        for w_unit in pl.range(w_worker, 4 * row_blocks, 24):
            w_rb: pl.Scalar[pl.INDEX] = w_unit // 4
            kb: pl.Scalar[pl.INDEX] = w_unit - w_rb * 4
            w_r0: pl.Scalar[pl.INDEX] = w_rb * 16
            w_rows: pl.Scalar[pl.INDEX] = pl.min(16, bs - w_r0)
            k_base: pl.Scalar[pl.INDEX] = kb * 1024
            weights_acc: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.create([16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for db in pl.range(2):
                d0: pl.Scalar[pl.INDEX] = k_base + db * 512
                x_tile: pl.Tensor[[16, 512], pl.BF16, pl.TensorView(valid_shape=[w_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat, [16, 512], [w_r0, d0], [w_rows, 512])
                weights_proj_tile: pl.Tensor[[512, 64], pl.BF16] = pl.tensor.slice(weights_proj, [512, 64], [d0, 0])
                weights_acc: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul_acc(weights_acc, x_tile, weights_proj_tile, init_cond=db == 0, a_trans=False, b_trans=False)
            weights_partial: pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32] = pl.tensor.assemble(weights_partial, weights_acc, [kb * t_pad + w_r0, 0])
        return weights_partial

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def _kv_project_worker(
        self,
        x: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        wkv: pl.Tensor[[4096, 512], pl.BF16],
        kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32],
        tile_base: pl.Scalar[pl.INDEX],
        tile_rows: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32]:
        # AIC 处理当前最多 512 行分段，按列块×行组的逻辑编号分配给物理 worker。
        #
        #         tile_base 是全 T 工作区的全局起点，tile_rows 是本段真实行数；DENSE_M
        #         只控制本段完整行块。每个 worker 对自己领取的所有输出都走完整 K4096，
        #         其余行组由其他核负责，消费者必须等待同一行全部四个 N128 生产者。
        #
        t_matmul: pl.Scalar[pl.INDEX] = (tile_rows + 16 - 1) // 16 * 16
        kv_full_rows: pl.Scalar[pl.INDEX] = tile_rows // 32 * 32
        kv_m_groups: pl.Scalar[pl.INDEX] = pl.min(3, pl.max(1, tile_rows // 32))
        for logical in pl.range(worker_id, 4 * kv_m_groups, 24):
            kv_col0: pl.Scalar[pl.INDEX] = logical // kv_m_groups * 128
            kv_m_group: pl.Scalar[pl.INDEX] = logical % kv_m_groups
            for dense_t0 in pl.range(kv_m_group * 32, kv_full_rows, kv_m_groups * 32):
                dense_x0: pl.Scalar[pl.INDEX] = tile_base + dense_t0
                dense_acc: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.create([32, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for dense_k in pl.pipeline(8, stage=2):
                    dense_d0: pl.Scalar[pl.INDEX] = dense_k * 512
                    dense_x: pl.Tensor[[32, 512], pl.BF16] = pl.tensor.slice(x, [32, 512], [dense_x0, dense_d0])
                    dense_w: pl.Tensor[[512, 128], pl.BF16] = pl.tensor.slice(wkv, [512, 128], [dense_d0, kv_col0])
                    dense_acc: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.matmul_acc(dense_acc, dense_x, dense_w, init_cond=dense_k == 0, a_trans=False, b_trans=False)
                kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32] = pl.tensor.assemble(kv_fp32, dense_acc, [tile_base + dense_t0, kv_col0])
            for t0 in pl.range(kv_full_rows + kv_m_group * 16, t_matmul, kv_m_groups * 16):
                kv_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for db in pl.pipeline(8, stage=2):
                    d0: pl.Scalar[pl.INDEX] = db * 512
                    kv_rows: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows - t0)
                    x_t0: pl.Scalar[pl.INDEX] = tile_base + t0
                    kv_x_chunk_bf16: pl.Tensor[[16, 512], pl.BF16, pl.TensorView(valid_shape=[kv_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                        x, [16, 512], [x_t0, d0], [kv_rows, 512]
                    )
                    wkv_chunk: pl.Tensor[[512, 128], pl.BF16] = pl.tensor.slice(wkv, [512, 128], [d0, kv_col0])
                    kv_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk, init_cond=db == 0, a_trans=False, b_trans=False)
                kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32] = pl.tensor.assemble(kv_fp32, kv_acc, [tile_base + t0, kv_col0])
        return kv_fp32

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def _kv_project_worker__2(
        self,
        x: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        wkv: pl.Tensor[[4096, 512], pl.BF16],
        kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32],
        tile_base: pl.Scalar[pl.INDEX],
        tile_rows: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32]:
        # AIC 处理当前最多 512 行分段，按列块×行组的逻辑编号分配给物理 worker。
        #
        #         tile_base 是全 T 工作区的全局起点，tile_rows 是本段真实行数；DENSE_M
        #         只控制本段完整行块。每个 worker 对自己领取的所有输出都走完整 K4096，
        #         其余行组由其他核负责，消费者必须等待同一行全部四个 N128 生产者。
        #
        t_matmul: pl.Scalar[pl.INDEX] = (tile_rows + 16 - 1) // 16 * 16
        kv_full_rows: pl.Scalar[pl.INDEX] = tile_rows // 64 * 64
        kv_m_groups: pl.Scalar[pl.INDEX] = pl.min(3, pl.max(1, tile_rows // 64))
        for logical in pl.range(worker_id, 4 * kv_m_groups, 24):
            kv_col0: pl.Scalar[pl.INDEX] = logical // kv_m_groups * 128
            kv_m_group: pl.Scalar[pl.INDEX] = logical % kv_m_groups
            for dense_t0 in pl.range(kv_m_group * 64, kv_full_rows, kv_m_groups * 64):
                dense_x0: pl.Scalar[pl.INDEX] = tile_base + dense_t0
                dense_acc: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.create([64, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for dense_k in pl.pipeline(8, stage=2):
                    dense_d0: pl.Scalar[pl.INDEX] = dense_k * 512
                    dense_x: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(x, [64, 512], [dense_x0, dense_d0])
                    dense_w: pl.Tensor[[512, 128], pl.BF16] = pl.tensor.slice(wkv, [512, 128], [dense_d0, kv_col0])
                    dense_acc: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul_acc(dense_acc, dense_x, dense_w, init_cond=dense_k == 0, a_trans=False, b_trans=False)
                kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32] = pl.tensor.assemble(kv_fp32, dense_acc, [tile_base + dense_t0, kv_col0])
            for t0 in pl.range(kv_full_rows + kv_m_group * 16, t_matmul, kv_m_groups * 16):
                kv_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for db in pl.pipeline(8, stage=2):
                    d0: pl.Scalar[pl.INDEX] = db * 512
                    kv_rows: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows - t0)
                    x_t0: pl.Scalar[pl.INDEX] = tile_base + t0
                    kv_x_chunk_bf16: pl.Tensor[[16, 512], pl.BF16, pl.TensorView(valid_shape=[kv_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                        x, [16, 512], [x_t0, d0], [kv_rows, 512]
                    )
                    wkv_chunk: pl.Tensor[[512, 128], pl.BF16] = pl.tensor.slice(wkv, [512, 128], [d0, kv_col0])
                    kv_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk, init_cond=db == 0, a_trans=False, b_trans=False)
                kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32] = pl.tensor.assemble(kv_fp32, kv_acc, [tile_base + t0, kv_col0])
        return kv_fp32

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def kv_project_stage(
        self, x: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16], wkv: pl.Tensor[[4096, 512], pl.BF16], kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32], worker_id: pl.Scalar[pl.INDEX]
    ) -> pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32]:
        # 按实际 T 分段并选择原 M32/M64 策略，所有波次仍在同一 incore 内执行。
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        for tile_base in pl.range(0, tokens, 512):
            tile_rows: pl.Scalar[pl.INDEX] = pl.min(512, tokens - tile_base)
            if tile_rows < 128:
                kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32] = self._kv_project_worker(x, wkv, kv_fp32, tile_base, tile_rows, worker_id)
            else:
                kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32] = self._kv_project_worker__2(x, wkv, kv_fp32, tile_base, tile_rows, worker_id)
        return kv_fp32

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def o_b_stage(
        self,
        o_r_i8_pad: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8],
        wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        partials: pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32],
        tokens: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32]:
        # 全八组兼容入口；与组就绪分支共用数学和动态行循环。
        partials_v1: pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32] = self._o_b_nz_groups_stage(o_r_i8_pad, wo_b, partials, tokens, worker_id, 0, 8)
        return partials_v1

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def o_projection_a_groups_stage_nz(
        self,
        o_packed: pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16],
        wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        o_r_pad: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32],
        tokens: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
        group_begin: pl.Scalar[pl.INDEX],
        group_end: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32]:
        # NZ O_A 的动态组区间；0≤begin≤end≤8，保持全局条带 owner 与原 K 链。
        pack_stride: pl.Scalar[pl.INDEX] = pl.tensor.dim(o_packed, 0) // 8
        row_blocks: pl.Scalar[pl.INDEX] = (tokens + 128 - 1) // 128
        column_blocks: pl.Scalar[pl.INDEX] = 8
        group_first: pl.Scalar[pl.INDEX] = group_begin * row_blocks * column_blocks
        group_limit: pl.Scalar[pl.INDEX] = group_end * row_blocks * column_blocks
        first: pl.Scalar[pl.INDEX] = worker_id + (group_first + 24 - 1 - worker_id) // 24 * 24
        for logical in pl.range(pl.max(0, first), group_limit, 24):
            g: pl.Scalar[pl.INDEX] = pl.max(0, logical // (row_blocks * column_blocks))
            pa_unit: pl.Scalar[pl.INDEX] = logical % (row_blocks * column_blocks)
            pa_rb: pl.Scalar[pl.INDEX] = pa_unit // column_blocks
            nf: pl.Scalar[pl.INDEX] = logical % 8
            pa_r0: pl.Scalar[pl.INDEX] = pa_rb * 128
            pa_rows: pl.Scalar[pl.INDEX] = pl.min(128, tokens - pa_r0)
            pa_src0: pl.Scalar[pl.INDEX] = g * pack_stride + pa_r0
            out_col_g: pl.Scalar[pl.INDEX] = g * 1024
            n0: pl.Scalar[pl.INDEX] = pl.max(0, nf) * 128
            xa_first: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                o_packed, [128, 256], [pa_src0, 0], [pa_rows, 256]
            )
            wa_first: pl.Tensor[[1, 256, 128], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wo_a, [1, 256, 128], [g, 0, n0])
            acc_a: pl.Tensor[[1, 128, 128], pl.FP32] = pl.tensor.matmul(xa_first, wa_first, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
            for kb in pl.pipeline(1, 16, stage=2):
                k0: pl.Scalar[pl.INDEX] = kb * 256
                xa_k_chunk: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                    o_packed, [128, 256], [pa_src0, k0], [pa_rows, 256]
                )
                wa_k_chunk: pl.Tensor[[1, 256, 128], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wo_a, [1, 256, 128], [g, k0, n0])
                acc_a: pl.Tensor[[1, 128, 128], pl.FP32] = pl.tensor.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, a_trans=False, b_trans=False)
            o_r_pad: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32] = pl.tensor.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])
        return o_r_pad

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def o_projection_a_stage_nz(
        self,
        o_packed: pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16],
        wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        o_r_pad: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32],
        tokens: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32]:
        # 全八组兼容入口：与按组通知分支复用同一数学，不另维护计算版本。
        o_r_pad_v1: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32] = self.o_projection_a_groups_stage_nz(o_packed, wo_a, o_r_pad, tokens, worker_id, 0, 8)
        return o_r_pad_v1

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def publish_phase(
        self, signals: pl.Tensor[[CSA_FULL_WORDS], pl.INT32], phase: pl.Scalar[pl.INDEX], core_id: pl.Scalar[pl.INDEX], participant_count: pl.Scalar[pl.INDEX], generation: pl.Scalar[pl.INT32]
    ) -> pl.Tensor[[CSA_FULL_WORDS], pl.INT32]:
        # 完成业务写回后发布本核阶段，不通知或等待无关消费者。
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        offset: pl.Scalar[pl.INDEX] = (phase * participant_count + core_id) * 128
        signals_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.csa_signal_notify(signals, offset, generation)
        return signals_v1

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def _q_a_m16_parallel_worker(
        self,
        x: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        qa_fp32: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32]:
        # 全局M16块按八个N128列任务分核；仅改变owner，不改变原尾块数学。
        #
        #         Host策略只为G6且16<T<32开启，所以正式调用每核至多一块且总权重请求不增。
        #         helper仍按实际T计算循环上界，较大T也可按WORKERS跨波次领取；不得将
        #         T24或两块写死。每块仍K256×16、首段初始化、valid_shape及pad16写回。
        #         新增QA参与核仍承担后继Indexer，不能按核数预记整层性能收益。
        #
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        m16_blocks: pl.Scalar[pl.INDEX] = (tokens + 16 - 1) // 16
        for logical_worker in pl.range(pl.max(worker_id, 0), 8 * m16_blocks, 24):
            column: pl.Scalar[pl.INDEX] = logical_worker % 8 * 128
            row: pl.Scalar[pl.INDEX] = logical_worker // 8 * 16
            m16_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for k_block in pl.pipeline(16, stage=2):
                k0: pl.Scalar[pl.INDEX] = k_block * 256
                valid_rows: pl.Scalar[pl.INDEX] = pl.min(16, tokens - row)
                m16_x: pl.Tensor[[16, 256], pl.BF16, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x, [16, 256], [row, k0], [valid_rows, 256])
                m16_weight: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wq_a, [128, 256], [column, k0])
                m16_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul_acc(m16_acc, m16_x, m16_weight, init_cond=k_block == 0, a_trans=False, b_trans=True)
            qa_fp32: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32] = pl.tensor.assemble(qa_fp32, m16_acc, [row, column])
        return qa_fp32

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def _q_a_worker(
        self,
        x: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        qa_fp32: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32]:
        # 一个物理 AIC 领取原投影的 M/N 逻辑分块，保持完整 K 累加链。
        #
        #         WORKERS 必须为正，worker_id 属于 [0,WORKERS)。同一行需八个列块全部
        #         完成才能归一化；返回 Tensor 只表达数据别名，不表达跨核同步完成。
        #
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        padded_rows: pl.Scalar[pl.INDEX] = (tokens + 16 - 1) // 16 * 16
        full_rows: pl.Scalar[pl.INDEX] = tokens // 32 * 32
        m_groups: pl.Scalar[pl.INDEX] = pl.min(3, pl.max(1, tokens // 32))
        for logical_worker in pl.range(pl.max(worker_id, 0), 8 * m_groups, 24):
            column: pl.Scalar[pl.INDEX] = logical_worker % 8 * 128
            m_group: pl.Scalar[pl.INDEX] = logical_worker // 8
            for row in pl.range(m_group * 32, full_rows, m_groups * 32):
                dense_acc: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.create([32, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for k_block in pl.pipeline(16, stage=2):
                    k0: pl.Scalar[pl.INDEX] = k_block * 256
                    dense_x: pl.Tensor[[32, 256], pl.BF16] = pl.tensor.slice(x, [32, 256], [row, k0])
                    dense_weight: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wq_a, [128, 256], [column, k0])
                    dense_acc: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.matmul_acc(dense_acc, dense_x, dense_weight, init_cond=k_block == 0, a_trans=False, b_trans=True)
                qa_fp32: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32] = pl.tensor.assemble(qa_fp32, dense_acc, [row, column])
            for row_1 in pl.range(full_rows + m_group * 16, padded_rows, m_groups * 16):
                tail_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for k_block_1 in pl.pipeline(16, stage=2):
                    k0: pl.Scalar[pl.INDEX] = k_block_1 * 256
                    valid_rows: pl.Scalar[pl.INDEX] = pl.min(16, tokens - row_1)
                    tail_x: pl.Tensor[[16, 256], pl.BF16, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                        x, [16, 256], [row_1, k0], [valid_rows, 256]
                    )
                    tail_weight: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wq_a, [128, 256], [column, k0])
                    tail_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul_acc(tail_acc, tail_x, tail_weight, init_cond=k_block_1 == 0, a_trans=False, b_trans=True)
                qa_fp32: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32] = pl.tensor.assemble(qa_fp32, tail_acc, [row_1, column])
        return qa_fp32

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def _q_a_worker__2(
        self,
        x: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        qa_fp32: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32]:
        # 一个物理 AIC 领取原投影的 M/N 逻辑分块，保持完整 K 累加链。
        #
        #         WORKERS 必须为正，worker_id 属于 [0,WORKERS)。同一行需八个列块全部
        #         完成才能归一化；返回 Tensor 只表达数据别名，不表达跨核同步完成。
        #
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        padded_rows: pl.Scalar[pl.INDEX] = (tokens + 16 - 1) // 16 * 16
        full_rows: pl.Scalar[pl.INDEX] = tokens // 64 * 64
        m_groups: pl.Scalar[pl.INDEX] = pl.min(3, pl.max(1, tokens // 64))
        for logical_worker in pl.range(pl.max(worker_id, 0), 8 * m_groups, 24):
            column: pl.Scalar[pl.INDEX] = logical_worker % 8 * 128
            m_group: pl.Scalar[pl.INDEX] = logical_worker // 8
            for row in pl.range(m_group * 64, full_rows, m_groups * 64):
                dense_acc: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.create([64, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for k_block in pl.pipeline(16, stage=2):
                    k0: pl.Scalar[pl.INDEX] = k_block * 256
                    dense_x: pl.Tensor[[64, 256], pl.BF16] = pl.tensor.slice(x, [64, 256], [row, k0])
                    dense_weight: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wq_a, [128, 256], [column, k0])
                    dense_acc: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul_acc(dense_acc, dense_x, dense_weight, init_cond=k_block == 0, a_trans=False, b_trans=True)
                qa_fp32: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32] = pl.tensor.assemble(qa_fp32, dense_acc, [row, column])
            for row_1 in pl.range(full_rows + m_group * 16, padded_rows, m_groups * 16):
                tail_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for k_block_1 in pl.pipeline(16, stage=2):
                    k0: pl.Scalar[pl.INDEX] = k_block_1 * 256
                    valid_rows: pl.Scalar[pl.INDEX] = pl.min(16, tokens - row_1)
                    tail_x: pl.Tensor[[16, 256], pl.BF16, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                        x, [16, 256], [row_1, k0], [valid_rows, 256]
                    )
                    tail_weight: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)] = pl.tensor.slice(wq_a, [128, 256], [column, k0])
                    tail_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul_acc(tail_acc, tail_x, tail_weight, init_cond=k_block_1 == 0, a_trans=False, b_trans=True)
                qa_fp32: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32] = pl.tensor.assemble(qa_fp32, tail_acc, [row_1, column])
        return qa_fp32

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def q_a_stage(
        self,
        x: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        qa_fp32: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32]:
        # 由外层 AIC 调用；按实际 T 选 M32/M64，不按固定 batch 生成入口。
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        if False:
            qa_fp32: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32] = self._q_a_m16_parallel_worker(x, wq_a, qa_fp32, worker_id)
        else:
            if tokens < 128:
                qa_fp32: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32] = self._q_a_worker(x, wq_a, qa_fp32, worker_id)
            else:
                qa_fp32: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32] = self._q_a_worker__2(x, wq_a, qa_fp32, worker_id)
        return qa_fp32

    @pl.function(type=pl.FunctionType.Inline)
    def csa_signal_wait(self, signal: pl.Tensor[[CSA_FULL_WORDS], pl.INT32], offset: pl.Scalar[pl.INDEX], expected: pl.Scalar[pl.INT32]) -> pl.Tensor[[CSA_FULL_WORDS], pl.INT32]:
        # 等待该槽达到 expected，不清零；业务数据可见性由外层保证。
        pl.system.signal_wait(signal, [offset], expected)
        return signal

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def wait_producer(
        self, signals: pl.Tensor[[CSA_FULL_WORDS], pl.INT32], phase: pl.Scalar[pl.INDEX], producer_core: pl.Scalar[pl.INDEX], participant_count: pl.Scalar[pl.INDEX], generation: pl.Scalar[pl.INT32]
    ) -> pl.Tensor[[CSA_FULL_WORDS], pl.INT32]:
        # 等待一个实际生产者；不会清零或消费它的信号。
        offset: pl.Scalar[pl.INDEX] = (phase * participant_count + producer_core) * 128
        signals_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.csa_signal_wait(signals, offset, generation)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        return signals_v1

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def wait_row_owners(
        self,
        signals: pl.Tensor[[CSA_FULL_WORDS], pl.INT32],
        phase: pl.Scalar[pl.INDEX],
        participant_count: pl.Scalar[pl.INDEX],
        generation: pl.Scalar[pl.INT32],
        first_row: pl.Scalar[pl.INDEX],
        row_count: pl.Scalar[pl.INDEX],
        rows_per_tile: pl.Scalar[pl.INDEX],
        producer_workers: pl.Scalar[pl.INDEX],
        producer_base: pl.Scalar[pl.INDEX],
        producer_stride: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_WORDS], pl.INT32]:
        # 按行 tile 的轮转所有权，等待覆盖目标行区间的不同生产者。
        #
        #         tile k 归 worker k % producer_workers；其硬件核编号为 base+worker*stride。
        #         连续 tile 的 owner 周期等于 worker 数，因此只需等待 min(tile数,worker数)
        #         个不同 owner。该公式由工作划分决定，与某个历史泳道图的执行次序无关。
        #         所有 divisor 必须为正数；first_row/count 必须是同一个业务 Tensor 的有效范围。
        #
        if row_count > 0:
            first_tile: pl.Scalar[pl.INDEX] = first_row // rows_per_tile
            end_tile: pl.Scalar[pl.INDEX] = (first_row + row_count + rows_per_tile - 1) // rows_per_tile
            owner_count: pl.Scalar[pl.INDEX] = pl.min(end_tile - first_tile, producer_workers)
            for ordinal in pl.range(owner_count):
                owner: pl.Scalar[pl.INDEX] = (first_tile + ordinal) % producer_workers
                core: pl.Scalar[pl.INDEX] = producer_base + owner * producer_stride
                offset: pl.Scalar[pl.INDEX] = (phase * participant_count + core) * 128
                signals: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.csa_signal_wait(signals, offset, generation)
            pl.system.fence()
            pl.system.cacheinvalid()
            pl.system.fence()
        return signals

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def cube_schedule(
        hc_fn: pl.Tensor[[24, 16384], pl.FP32],
        wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        x_fp32: pl.Tensor[[CSA_FULL_T, 16384], pl.FP32],
        partials: pl.InOut[pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32]],
        normed: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        qa: pl.InOut[pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32]],
        qr: pl.Tensor[[CSA_FULL_T, 1024], pl.INT8],
        qr_matmul: pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8],
        wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        q_proj: pl.InOut[pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32]],
        wkv: pl.Tensor[[4096, 512], pl.BF16],
        kv_fp32: pl.InOut[pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32]],
        main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
        main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
        main_value: pl.InOut[pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32]],
        main_score: pl.InOut[pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32]],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        idx_wkv: pl.Tensor[[256, 4096], pl.BF16],
        idx_wgate: pl.Tensor[[256, 4096], pl.BF16],
        idx_value: pl.InOut[pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32]],
        idx_score: pl.InOut[pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32]],
        idx_bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        idx_lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        idx_count: pl.Tensor[[1], pl.INT32],
        idx_normed: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16],
        hadamard: pl.Tensor[[128, 128], pl.BF16],
        idx_had: pl.InOut[pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32]],
        idx_cache: pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8],
        q_heads: pl.Tensor[[CSA_FULL_QUERY_HEAD_ROWS, 512], pl.BF16],
        indexer_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_INDEXER_COLS], pl.INT32],
        idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8],
        idx_query_acc: pl.InOut[pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32]],
        idx_query_rope_heads: pl.Tensor[[CSA_FULL_IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
        idx_query_had: pl.InOut[pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32]],
        idx_query_i8: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8],
        weights_proj: pl.Tensor[[4096, 64], pl.BF16],
        weights_partial: pl.InOut[pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32]],
        group_offsets: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32],
        group_plan: pl.Tensor[[8], pl.INT32],
        coefficients: pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16],
        idx_key0: pl.Tensor[[CSA_FULL_KEY0_ROWS, 128], pl.INT8],
        idx_key64: pl.Tensor[[CSA_FULL_KEY64_ROWS, 128], pl.INT8],
        indexer_transfer: pl.InOut[pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32]],
        attention_mask: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32],
        kv_ring: pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16],
        score_ring: pl.InOut[pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32]],
        probability_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16],
        pv_ring: pl.InOut[pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32]],
        packed: pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16],
        ffts: pl.Tensor[[256], pl.INT64],
        wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        o_r: pl.InOut[pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32]],
        o_quant: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8],
        wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        o_partials: pl.InOut[pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32]],
        signal: pl.InOut[pl.Tensor[[CSA_FULL_WORDS], pl.INT32]],
    ):
        pl.func_attr({"cache_policy": [(1, 1), (8, 1), (10, 1), (50, 1), (53, 1)]})
        partials_gm: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32] = partials
        qa_gm: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32] = qa
        q_proj_gm: pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32] = q_proj
        kv_fp32_gm: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32] = kv_fp32
        main_value_gm: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32] = main_value
        main_score_gm: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32] = main_score
        idx_value_gm: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32] = idx_value
        idx_score_gm: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32] = idx_score
        idx_had_gm: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32] = idx_had
        idx_query_acc_gm: pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32] = idx_query_acc
        idx_query_had_gm: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32] = idx_query_had
        weights_partial_gm: pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32] = weights_partial
        indexer_transfer_gm: pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32] = indexer_transfer
        score_ring_gm: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32] = score_ring
        pv_ring_gm: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32] = pv_ring
        o_r_gm: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32] = o_r
        o_partials_gm: pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32] = o_partials
        signal_gm: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = signal
        worker: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        _tuple_tmp: pl.Tuple[pl.Tensor[[CSA_FULL_WORDS], pl.INT32], pl.Scalar[pl.INT32]] = self.begin_invocation(signal, worker)
        signal: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = _tuple_tmp[0]
        generation: pl.Scalar[pl.INT32] = _tuple_tmp[1]
        pl.system.set_ffts(ffts)
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_fp32, 0)
        if True:
            hc_tasks: pl.Scalar[pl.INDEX] = (tokens + 15) // 16 * 4
            hc_waves: pl.Scalar[pl.INDEX] = (hc_tasks + 24 - 1 - worker) // 24
            for hc_wave in pl.range(pl.min(hc_waves, 4)):
                hc_first: pl.Scalar[pl.INDEX] = (worker + hc_wave * 24) // 4 * 16
                signal: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal, 1, 72, generation, hc_first, pl.min(16, tokens - hc_first), 8, 48, 24, 1)
        else:
            signal: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal, 1, 72, generation, 0, tokens, 8, 48, 24, 1)
        partials_v1: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32] = self.hc_project_range(x_fp32, hc_fn, partials, worker, 24)
        signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal, 2, worker, 72, generation)
        signal_v2: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v1, 6, 72, generation, 0, tokens, 8, 48, 24, 1)
        qa_v1: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32] = self.q_a_stage(normed, wq_a, qa, worker)
        signal_v3: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v2, 7, worker, 72, generation)
        _tuple_tmp_1: pl.Tuple[pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32], pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32]] = self.indexer_compressor_project_stage(
            normed, idx_wkv, idx_wgate, idx_value, idx_score, worker
        )
        idx_value: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32] = _tuple_tmp_1[0]
        idx_score: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32] = _tuple_tmp_1[1]
        signal_v4: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v3, 23, worker, 72, generation)
        signal_v5: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v4, 8, 72, generation, 0, tokens, 8, 48, 24, 1)
        idx_query_acc_v1: pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32] = self.indexer_query_project_stage(qr, idx_wq_b, idx_query_acc, worker)
        signal_v6: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v5, 16, worker, 72, generation)
        weights_partial_v1: pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32] = self.indexer_weights_project_stage(normed, weights_proj, weights_partial, worker)
        signal_v7: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v6, 20, worker, 72, generation)
        kv_fp32_v1: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32] = self.kv_project_stage(normed, wkv, kv_fp32, worker)
        signal_v8: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v7, 12, worker, 72, generation)
        query_rope_tasks: pl.Scalar[pl.INDEX] = (tokens + 7) // 8 * 16
        signal_v9: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v8, 17, 72, generation, 0, query_rope_tasks, 1, 48, 24, 1)
        idx_query_had_v1: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32] = self.indexer_query_hadamard_stage(idx_query_rope_heads, hadamard, idx_query_had, tokens, worker)
        signal_v10: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v9, 18, worker, 72, generation)
        q_proj_v1: pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32] = self._q_b_nz_stage(wq_b, qr_matmul, q_proj, tokens, worker)
        signal_v11: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v10, 10, worker, 72, generation)
        planner: pl.Scalar[pl.INDEX] = pl.min((tokens + 7) // 8, 47)
        signal_v12: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_producer(signal_v11, 15, 24 + planner, 72, generation)
        requests: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_lengths, 0)
        signal_v13: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v12, 29, 72, generation, 0, requests, 1, 48, 24, 1)
        compact_count: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(idx_count, [0]), pl.INDEX)
        idx_had_v1: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32] = self.indexer_compressor_hadamard_stage(idx_normed, hadamard, idx_had, compact_count, worker)
        signal_v14: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v13, 30, worker, 72, generation)
        _tuple_tmp_2: pl.Tuple[pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32], pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32]] = self.compressor_project_stage(
            normed, main_wkv, main_wgate, main_value, main_score, worker
        )
        main_value: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32] = _tuple_tmp_2[0]
        main_score: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32] = _tuple_tmp_2[1]
        signal_v15: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v14, 22, worker, 72, generation)
        signal_v16: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v15, 19, 72, generation, 0, tokens, 1, 48, 24, 1)
        groups: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(group_plan, [0]), pl.INDEX)
        signal_v17: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v16, 33, 72, generation, 0, groups, 1, 48, 24, 1)
        signal_v18: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v17, 32, 72, generation, 0, 48, 1, 48, 24, 1)
        indexer_transfer_v1: pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32] = self.indexer_score_cube_stage(
            idx_query_i8, coefficients, idx_key0, idx_key64, idx_cache, indexer_table, positions, idx_lengths, idx_bounds, group_offsets, group_plan, indexer_transfer, worker
        )
        signal_v19: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v18, 34, worker, 72, generation)
        signal_v20: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v19, 37, 72, generation, 0, tokens, 8, 48, 24, 1)
        signal_v21: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v20, 11, 72, generation, 0, 48, 1, 48, 24, 1)
        _tuple_tmp_3: pl.Tuple[pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32], pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32]] = self.attention_qk_pv_cube_stage(
            q_heads, attention_mask, kv_ring, score_ring, probability_ring, pv_ring, ffts, worker
        )
        score_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32] = _tuple_tmp_3[0]
        pv_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32] = _tuple_tmp_3[1]
        signal_v22: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v21, 38, worker, 72, generation)
        packed_workers: pl.Scalar[pl.INDEX] = pl.min(tokens, 24) * 2
        signal_v23: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v22, 39, 72, generation, 0, packed_workers, 1, 48, 24, 1)
        if True:
            oa_per_group: pl.Scalar[pl.INDEX] = (tokens + 127) // 128 * 8
            for oa_group in pl.range(8):
                oa_first: pl.Scalar[pl.INDEX] = oa_group * oa_per_group
                oa_owned: pl.Scalar[pl.INDEX] = worker + (oa_first + 24 - 1 - worker) // 24 * 24
                if oa_owned < oa_first + oa_per_group:
                    o_r: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32] = self.o_projection_a_groups_stage_nz(packed, wo_a, o_r, tokens, worker, oa_group, oa_group + 1)
                    signal_v23: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v23, 44 + oa_group, worker, 72, generation)
            quant_per_group: pl.Scalar[pl.INDEX] = (tokens + 31) // 32
            for ob_group in pl.range(8):
                ob_first: pl.Scalar[pl.INDEX] = ob_group * 8
                ob_owned: pl.Scalar[pl.INDEX] = worker + (ob_first + 24 - 1 - worker) // 24 * 24
                if ob_owned < ob_first + 8:
                    signal_v23: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v23, 52 + ob_group, 72, generation, ob_group * quant_per_group, quant_per_group, 1, 48, 24, 1)
                    o_partials: pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32] = self._o_b_nz_groups_stage(o_quant, wo_b, o_partials, tokens, worker, ob_group, ob_group + 1)
        else:
            o_r: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32] = self.o_projection_a_stage_nz(packed, wo_a, o_r, tokens, worker)
            signal_v23: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v23, 40, worker, 72, generation)
            quant_tasks: pl.Scalar[pl.INDEX] = 8 * ((tokens + 31) // 32)
            signal_v23: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v23, 41, 72, generation, 0, quant_tasks, 1, 48, 24, 1)
            o_partials: pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32] = self.o_b_stage(o_quant, wo_b, o_partials, tokens, worker)
        signal_v24: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v23, 42, worker, 72, generation)
        return (
            partials_gm,
            qa_gm,
            q_proj_gm,
            kv_fp32_gm,
            main_value_gm,
            main_score_gm,
            idx_value_gm,
            idx_score_gm,
            idx_had_gm,
            idx_query_acc_gm,
            idx_query_had_gm,
            weights_partial_gm,
            indexer_transfer_gm,
            score_ring_gm,
            pv_ring_gm,
            o_r_gm,
            o_partials_gm,
            signal_gm,
        )

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def attention_inverse_rope_sign_stage(
        self, sin: pl.Tensor[[CSA_FULL_T, 64], pl.FP32], signed_sin: pl.Tensor[[CSA_FULL_T, 64], pl.FP32], worker_id: pl.Scalar[pl.INDEX]
    ) -> pl.Tensor[[CSA_FULL_T, 64], pl.FP32]:
        # 逐token逆RoPE预处理；与请求Q分组无关，输出容量尾部保持未写。
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(sin, 0)
        for first in pl.range(worker_id * 8, tokens, 384):
            rows: pl.Scalar[pl.INDEX] = pl.min(8, tokens - first)
            cs_ones: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
            cs_idx_f: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode="round")
            cs_col: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(cs_ones, cs_idx_f)
            cs_dup_i32: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.muls(cs_col, 0.5), target_type=pl.INT32, mode="trunc")
            cs_dup_f: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(cs_dup_i32, target_type=pl.FP32, mode="round")
            cs_lane: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(cs_col, pl.tile.muls(cs_dup_f, 2.0))
            cs_sign: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.neg(pl.tile.subs(pl.tile.muls(cs_lane, 2.0), 1.0))
            cs_sin: pl.Tile[[8, 64], pl.FP32, pl.TileView(valid_shape=[rows, 64])] = pl.tile.load(sin, [first, 0], [8, 64], [rows, 64])
            cs_sign_rows: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.set_validshape(cs_sign, rows, 64)
            pl.tile.store(pl.tile.mul(cs_sin, cs_sign_rows), [first, 0], signed_sin)
        return signed_sin

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def request_bounds(
        self, query_start_loc: pl.Tensor[[CSA_FULL_IQ], pl.INT32], seq_lens: pl.Tensor[[CSA_FULL_B], pl.INT32], request: pl.Scalar[pl.INDEX], token_capacity: pl.Scalar[pl.INDEX]
    ) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]]:
        # 返回 (begin, end, active)；失活时 begin=end=0，禁止据其读取 payload。
        begin: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        end: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        active: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
        if request >= 0 and request < pl.tensor.dim(seq_lens, 0):
            if request + 1 < pl.tensor.dim(query_start_loc, 0):
                length: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(seq_lens, [request]), pl.INDEX)
                if length > 0:
                    raw_begin: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(query_start_loc, [request]), pl.INDEX)
                    raw_end: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(query_start_loc, [request + 1]), pl.INDEX)
                    if raw_begin >= 0 and raw_begin < raw_end and raw_end <= token_capacity:
                        if raw_end - raw_begin <= length:
                            begin: pl.Scalar[pl.INDEX] = raw_begin
                            end: pl.Scalar[pl.INDEX] = raw_end
                            active: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
        return begin, end, active

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def query_request(
        self, query_start_loc: pl.Tensor[[CSA_FULL_IQ], pl.INT32], seq_lens: pl.Tensor[[CSA_FULL_B], pl.INT32], query: pl.Scalar[pl.INDEX], token_capacity: pl.Scalar[pl.INDEX]
    ) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]]:
        # 返回 (request, begin, end, active)；未找到或多重覆盖时 request=-1。
        owner: pl.Scalar[pl.INDEX] = pl.cast(-1, pl.INDEX)
        owner_begin: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        owner_end: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        matches: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        active: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
        if query >= 0 and query < token_capacity:
            for request in pl.range(pl.tensor.dim(seq_lens, 0)):
                _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.request_bounds(query_start_loc, seq_lens, request, token_capacity)
                begin: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
                end: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
                request_active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
                if pl.cast(request_active, pl.INDEX) != 0 and query >= begin and query < end:
                    owner: pl.Scalar[pl.INDEX] = request
                    owner_begin: pl.Scalar[pl.INDEX] = begin
                    owner_end: pl.Scalar[pl.INDEX] = end
                    matches: pl.Scalar[pl.INDEX] = matches + 1
            if matches == 1:
                active: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
            else:
                owner: pl.Scalar[pl.INDEX] = pl.cast(-1, pl.INDEX)
                owner_begin: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
                owner_end: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        return owner, owner_begin, owner_end, active

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def attention_query_geometry(
        self,
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        plan: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32],
        query: pl.Scalar[pl.INDEX],
    ):
        # 读取plan发布的本轮几何，避免两个AIV在五个块中重复O(B)查找。
        #
        #         调用方已等待完整plan，且query在[0,T)中。缓存分支保留owner/位置基本
        #         范围检查；完整Q/L及causal关系由同次plan生产者验证。页表列/物理页保护
        #         仍由实际gather逐地址执行，不因缓存几何跳过。关闭分支保留旧算法供比较。
        #
        request: pl.Scalar[pl.INDEX] = pl.cast(-1, pl.INDEX)
        position: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        active: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
        if False:
            saved_owner: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [query, 5]), pl.INDEX)
            saved_position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [query, 6]), pl.INDEX)
            if saved_owner >= 0 and saved_owner < pl.tensor.dim(lengths, 0) and saved_position >= 0:
                request: pl.Scalar[pl.INDEX] = saved_owner
                position: pl.Scalar[pl.INDEX] = saved_position
                active: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
        else:
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.query_request(bounds, lengths, query, pl.tensor.dim(positions, 0))
            owner: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            begin: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            end: pl.Scalar[pl.INDEX] = _tuple_tmp[2]
            owner_active: pl.Scalar[pl.INT32] = _tuple_tmp[3]
            if pl.cast(owner_active, pl.INDEX) != 0:
                actual_position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [query]), pl.INDEX)
                length: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [owner]), pl.INDEX)
                if actual_position == length - (end - begin) + query - begin:
                    request: pl.Scalar[pl.INDEX] = owner
                    position: pl.Scalar[pl.INDEX] = actual_position
                    active: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
        return request, position, active

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def attention_qk_pv_vector_lane_stage(
        self,
        ori_kv_flat: pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16],
        cmp_kv_flat: pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16],
        ori_block_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_ORIGINAL_COLS], pl.INT32],
        cmp_block_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_COMPRESSED_COLS], pl.INT32],
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        position_ids: pl.Tensor[[CSA_FULL_T], pl.INT64],
        cmp_sparse_indices: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32],
        sparse_bias: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32],
        valid_block_mask: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32],
        attn_sink_col: pl.Tensor[[64, 1], pl.FP32],
        freqs_cos: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_sin_signed: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        kv_transfer: pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16],
        score_transfer: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32],
        probability_transfer: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16],
        pv_transfer: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32],
        mi_transfer: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
        li_transfer: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
        o_packed_heads: pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16],
        ffts_workspace: pl.Tensor[[256], pl.INT64],
        core_id: pl.Scalar[pl.INDEX],
        lane_id: pl.Scalar[pl.INDEX],
    ) -> tuple[
        pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16],
        pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16],
        pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16],
    ]:
        # 双AIV各32头/64候选；同一mask和事件次序保证与AIC配对。
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids, 0)
        qk_core: pl.Scalar[pl.INDEX] = core_id
        qk_aiv: pl.Scalar[pl.INDEX] = lane_id
        qk_query_count: pl.Scalar[pl.INDEX] = (t_dim + 24 - 1 - qk_core) // 24
        qk_block_count: pl.Scalar[pl.INDEX] = qk_query_count * 5
        pl.system.set_ffts(ffts_workspace)
        qk_lane_head: pl.Scalar[pl.INDEX] = qk_aiv * 32
        qk_lane_kv: pl.Scalar[pl.INDEX] = qk_aiv * 64
        m_idx: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        m_rem_tmp: pl.Tile[[1, 64], pl.INT32] = pl.tile.create([1, 64], dtype=pl.INT32)
        m_lane: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.rems(m_idx, pl.const(2, pl.INT32), m_rem_tmp)
        m_swap_row: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.sub(m_idx, pl.tile.muls(m_lane, pl.const(2, pl.INT32))), pl.const(449, pl.INT32))
        m_swap_base: pl.Tile[[16, 64], pl.INT32] = pl.tile.create([16, 64], dtype=pl.INT32)
        m_swap_source: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.col_expand(m_swap_base, m_swap_row)
        m_row_offsets: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.muls(pl.tile.ci(pl.const(0, pl.INT32), [1, 16], dtype=pl.INT32, descending=False), pl.const(512, pl.INT32))
        m_swap_idx: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.row_expand_add(m_swap_source, pl.tile.reshape(m_row_offsets, [16, 1])), [1, 1024])
        running_m: pl.Tile[[32, 1], pl.FP32] = pl.tile.load(attn_sink_col, [qk_lane_head, 0], [32, 1], [32, 1])
        running_l: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(running_m, 0.0)
        running_left: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        running_right: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        for qk_tick, (m_iter, l_iter, left_iter, right_iter) in pl.range(qk_block_count + 2 + 1, init_values=(running_m, running_l, running_left, running_right)):
            qk_t: pl.Scalar[pl.INDEX] = qk_core + qk_tick // 5 * 24
            qk_sb: pl.Scalar[pl.INDEX] = qk_tick % 5
            if qk_tick < qk_block_count:
                if pl.cast(pl.tensor.read(valid_block_mask, [qk_t, qk_sb]), pl.INDEX) > 0:
                    qk_slot: pl.Scalar[pl.INDEX] = qk_core * 3 + qk_tick % 3
                    qk_kv_row: pl.Scalar[pl.INDEX] = qk_slot * 128
                    qk_kv_half: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
                    _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.attention_query_geometry(position_ids, bounds, lengths, valid_block_mask, qk_t)
                    qk_b: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
                    qk_pos: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
                    q_active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
                    if pl.cast(q_active, pl.INDEX) != 0:
                        if qk_sb == 0:
                            qk_win_len: pl.Scalar[pl.INDEX] = pl.min(qk_pos + 1, 128)
                            qk_win_start: pl.Scalar[pl.INDEX] = qk_pos - qk_win_len + 1
                            qk_head: pl.Scalar[pl.INDEX] = qk_win_start % 32
                            qk_rows: pl.Scalar[pl.INDEX] = pl.min(pl.max(qk_win_len - qk_lane_kv, 0), 64)
                            for qk_run in pl.unroll(5):
                                qk_lo: pl.Scalar[pl.INDEX] = pl.max(qk_run * 32 - qk_head - qk_lane_kv, 0)
                                qk_hi: pl.Scalar[pl.INDEX] = pl.min((qk_run + 1) * 32 - qk_head - qk_lane_kv, qk_rows)
                                if qk_hi > qk_lo:
                                    qk_absolute: pl.Scalar[pl.INDEX] = qk_win_start + qk_lane_kv + qk_lo
                                    qk_column: pl.Scalar[pl.INDEX] = qk_absolute // 32
                                    if qk_column >= 0 and qk_column < pl.tensor.dim(ori_block_table, 1):
                                        qk_raw_page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(ori_block_table, [qk_b, qk_column]), pl.INDEX)
                                        if qk_raw_page >= 0 and qk_raw_page < pl.tensor.dim(ori_kv_flat, 0) // 32:
                                            qk_raw_row: pl.Scalar[pl.INDEX] = qk_raw_page * 32 + qk_absolute % 32
                                            qk_kv_half: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(
                                                qk_kv_half, ori_kv_flat, [qk_lo, 0], [qk_raw_row, 0], [64, 512], valid_shape=[qk_hi - qk_lo, 512], transpose=False
                                            )
                        else:
                            for qk_row in pl.range(64):
                                qk_cmp_k: pl.Scalar[pl.INDEX] = (qk_sb - 1) * 128 + qk_lane_kv + qk_row
                                if qk_cmp_k < 512:
                                    qk_ridx: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(cmp_sparse_indices, [qk_t, qk_cmp_k]), pl.INDEX)
                                    if qk_ridx >= 0:
                                        qk_column: pl.Scalar[pl.INDEX] = qk_ridx // 32
                                        if qk_column < pl.tensor.dim(cmp_block_table, 1):
                                            qk_page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(cmp_block_table, [qk_b, qk_column]), pl.INDEX)
                                            if qk_page >= 0 and qk_page < pl.tensor.dim(cmp_kv_flat, 0) // 32:
                                                qk_src: pl.Scalar[pl.INDEX] = qk_page * 32 + qk_ridx % 32
                                                qk_kv_half: pl.Tile[[64, 512], pl.BF16, pl.Mem.Vec] = pl.tile.gather_row(qk_kv_half, cmp_kv_flat, [qk_row, 0], [qk_src, 0], [1, 512], transpose=False)
                    pl.tile.store(qk_kv_half, [qk_kv_row + qk_lane_kv, 0], kv_transfer)
            if t_dim >= 144:
                if qk_tick > 0 and qk_tick <= qk_block_count:
                    if pl.cast(pl.tensor.read(valid_block_mask, [qk_core + (qk_tick - 1) // 5 * 24, (qk_tick - 1) % 5]), pl.INDEX) > 0:
                        pl.system.sync_wait(3, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                if qk_tick < qk_block_count:
                    if pl.cast(pl.tensor.read(valid_block_mask, [qk_t, qk_sb]), pl.INDEX) > 0:
                        pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if qk_tick > 0 and qk_tick <= qk_block_count:
                softmax_work: pl.Scalar[pl.INDEX] = qk_tick - 1
                softmax_sb: pl.Scalar[pl.INDEX] = softmax_work % 5
                softmax_t: pl.Scalar[pl.INDEX] = qk_core + softmax_work // 5 * 24
                if pl.cast(pl.tensor.read(valid_block_mask, [softmax_t, softmax_sb]), pl.INDEX) > 0:
                    qk_slot_v1: pl.Scalar[pl.INDEX] = qk_core * 3 + softmax_work % 3
                    qk_transfer_row: pl.Scalar[pl.INDEX] = qk_slot_v1 * 64
                    qk_s0: pl.Scalar[pl.INDEX] = softmax_sb * 128
                    if t_dim < 144:
                        pl.system.sync_wait(3, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    qk_reduce_tmp: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    qk_scores_half: pl.Tile[[32, 128], pl.FP32] = pl.tile.load(score_transfer, [qk_transfer_row + qk_lane_head, 0], [32, 128], [32, 128])
                    qk_bias: pl.Tile[[1, 128], pl.FP32] = pl.tile.load(sparse_bias, [softmax_t, qk_s0], [1, 128], [1, 128])
                    qk_masked: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_add(pl.tile.muls(qk_scores_half, 0.044194173824159223), qk_bias)
                    qk_mi: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(qk_masked, qk_reduce_tmp)
                    qk_exp: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.exp(pl.tile.row_expand_sub(qk_masked, qk_mi))
                    qk_li: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(qk_exp, qk_reduce_tmp)
                    qk_probability: pl.Tile[[32, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(qk_exp, target_type=pl.BF16, mode="rint")
                    pl.tile.store(qk_probability, [qk_transfer_row + qk_lane_head, 0], probability_transfer)
                    pl.tile.store(qk_mi, [qk_transfer_row + qk_lane_head, 0], mi_transfer)
                    pl.tile.store(qk_li, [qk_transfer_row + qk_lane_head, 0], li_transfer)
                    pl.system.sync_set(4, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if t_dim < 144:
                if qk_tick < qk_block_count:
                    if pl.cast(pl.tensor.read(valid_block_mask, [qk_t, qk_sb]), pl.INDEX) > 0:
                        pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if qk_tick >= 3:
                pv_work: pl.Scalar[pl.INDEX] = qk_tick - 2 - 1
                pv_sb: pl.Scalar[pl.INDEX] = pv_work % 5
                pv_t: pl.Scalar[pl.INDEX] = qk_core + pv_work // 5 * 24
                if pl.cast(pl.tensor.read(valid_block_mask, [pv_t, pv_sb]), pl.INDEX) > 0:
                    pv_slot: pl.Scalar[pl.INDEX] = qk_core * 3 + pv_work % 3
                    pv_transfer_row: pl.Scalar[pl.INDEX] = pv_slot * 64
                    pl.system.sync_wait(5, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    pv_m: pl.Tile[[32, 1], pl.FP32] = pl.tile.load(mi_transfer, [pv_transfer_row + qk_lane_head, 0], [32, 1], [32, 1])
                    pv_l: pl.Tile[[32, 1], pl.FP32] = pl.tile.load(li_transfer, [pv_transfer_row + qk_lane_head, 0], [32, 1], [32, 1])
                    next_m: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(m_iter, pv_m)
                    alpha: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.exp(pl.tile.sub(m_iter, next_m))
                    beta: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.exp(pl.tile.sub(pv_m, next_m))
                    next_l: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.add(pl.tile.mul(alpha, l_iter), pl.tile.mul(beta, pv_l))
                    pv_left: pl.Tile[[32, 256], pl.FP32] = pl.tile.load(pv_transfer, [pv_transfer_row + qk_lane_head, 0], [32, 256], [32, 256])
                    next_left: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(pl.tile.row_expand_mul(left_iter, alpha), pl.tile.row_expand_mul(pv_left, beta))
                    pv_right: pl.Tile[[32, 256], pl.FP32] = pl.tile.load(pv_transfer, [pv_transfer_row + qk_lane_head, 256], [32, 256], [32, 256])
                    next_right: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(pl.tile.row_expand_mul(right_iter, alpha), pl.tile.row_expand_mul(pv_right, beta))
                    m_valid, l_valid, left_valid, right_valid = pl.yield_(next_m, next_l, next_left, next_right)
                else:
                    m_valid, l_valid, left_valid, right_valid = pl.yield_(m_iter, l_iter, left_iter, right_iter)
                if pv_sb == 4:
                    packed_stride: pl.Scalar[pl.INDEX] = pl.tensor.dim(o_packed_heads, 0) // 8
                    m_gather_tmp: pl.Tile[[1, 1024], pl.INT32] = pl.tile.create([1, 1024], dtype=pl.INT32)
                    for pub_h in pl.unroll(2):
                        pub_local_head: pl.Scalar[pl.INDEX] = pub_h * 16
                        m_h0: pl.Scalar[pl.INDEX] = qk_lane_head + pub_local_head
                        m_mi: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.extract(pl.tile.reshape(m_valid, [1, 32]), 0, pub_local_head, [1, 16], target_memory=pl.Mem.Vec), [16, 1])
                        m_li: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.extract(pl.tile.reshape(l_valid, [1, 32]), 0, pub_local_head, [1, 16], target_memory=pl.Mem.Vec), [16, 1])
                        m_left: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(left_valid, [16, 256], [pub_local_head, 0])
                        m_right: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(right_valid, [16, 256], [pub_local_head, 0])
                        m_oi: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.concat(m_left, m_right)
                        n_sink_bias: pl.Tile[[16, 1], pl.FP32] = pl.tile.load(attn_sink_col, [m_h0, 0], [16, 1], [16, 1])
                        n_sink_tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.add(pl.tile.sub(m_mi, m_mi), n_sink_bias)
                        n_denom: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.add(m_li, pl.tile.exp(pl.tile.sub(n_sink_tile, m_mi)))
                        n_full: pl.Tile[[16, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_div(m_oi, n_denom)
                        n_bf16: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.cast(n_full, target_type=pl.BF16, mode="rint")
                        m_rope: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(n_full, [16, 64], [0, 448])
                        m_cos_il: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(freqs_cos, [pv_t, 0], [1, 64], [1, 64])
                        m_sin_signed: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(rope_sin_signed, [pv_t, 0], [1, 64], [1, 64])
                        m_swapped_flat: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.gather(pl.tile.reshape(n_full, [1, 8192]), m_swap_idx, m_gather_tmp)
                        m_swapped: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(m_swapped_flat, [16, 64])
                        m_rot: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(pl.tile.col_expand_mul(m_rope, m_cos_il), pl.tile.col_expand_mul(m_swapped, m_sin_signed))
                        n_rope_bf16: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(m_rot, target_type=pl.BF16, mode="rint")
                        n_full_bf16: pl.Tile[[16, 512], pl.BF16, pl.Mem.Vec] = pl.tile.concat(pl.tile.slice(n_bf16, [16, 448], [0, 0]), n_rope_bf16)
                        n_group_bf16: pl.Tile[[2, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.reshape(n_full_bf16, [2, 4096])
                        n_pack_first: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(n_group_bf16, [1, 4096], [0, 0])
                        n_pack_second: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.slice(n_group_bf16, [1, 4096], [1, 0])
                        n_pack_row: pl.Scalar[pl.INDEX] = m_h0 // 8 * packed_stride + pv_t
                        n_pack_row_second: pl.Scalar[pl.INDEX] = n_pack_row + packed_stride
                        pl.tile.store(n_pack_first, [n_pack_row, 0], o_packed_heads)
                        pl.tile.store(n_pack_second, [n_pack_row_second, 0], o_packed_heads)
                    reset_m: pl.Tile[[32, 1], pl.FP32] = pl.tile.load(attn_sink_col, [qk_lane_head, 0], [32, 1], [32, 1])
                    reset_l: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(reset_m, 0.0)
                    reset_left: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
                    reset_right: pl.Tile[[32, 256], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
                    m_published, l_published, left_published, right_published = pl.yield_(reset_m, reset_l, reset_left, reset_right)
                else:
                    m_published, l_published, left_published, right_published = pl.yield_(m_valid, l_valid, left_valid, right_valid)
                m_after, l_after, left_after, right_after = pl.yield_(m_published, l_published, left_published, right_published)
            else:
                m_after, l_after, left_after, right_after = pl.yield_(m_iter, l_iter, left_iter, right_iter)
            running_m_1, running_l_1, running_left_1, running_right_1 = pl.yield_(m_after, l_after, left_after, right_after)
        return kv_transfer, probability_transfer, mi_transfer, li_transfer, o_packed_heads

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def compressed_pages_all_valid(
        self, table: pl.Tensor[[CSA_FULL_B, CSA_FULL_COMPRESSED_COLS], pl.INT32], request: pl.Scalar[pl.INDEX], length: pl.Scalar[pl.INDEX], physical_pages: pl.Scalar[pl.INDEX]
    ) -> pl.Scalar[pl.INT32]:
        # 保守快速判定：返回1才允许批量通过，返回0仍逐个检查候选，不丢合法页。
        #
        #         一次读取1024个页号并做向量范围检查；最后只读取valid_shape内的列。
        #         当前请求所有query的压缩候选上界不超过length//4，故覆盖该前缀即可。
        #         本函数不接收候选，不改变索引/排序，不在GM保存跨调用可陈旧的判断。
        #
        needed: pl.Scalar[pl.INDEX] = (length // 4 + 32 - 1) // 32
        all_valid: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
        if length <= 16777216 and physical_pages > 0 and physical_pages <= 16777216:
            if needed <= pl.tensor.dim(table, 1):
                all_valid: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
                page_limit: pl.Scalar[pl.FP32] = pl.cast(pl.cast(physical_pages, pl.INT32), pl.FP32)
                reduce_tmp: pl.Tile[[8, 1024], pl.FP32] = pl.tile.create([8, 1024], dtype=pl.FP32)
                for column in pl.range(0, needed, 1024):
                    live: pl.Scalar[pl.INDEX] = pl.min(1024, needed - column)
                    values: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, live])] = pl.tile.cast(
                        pl.tile.load(table, [request, column], [1, 1024], [1, live]), target_type=pl.FP32, mode="round"
                    )
                    lower_ok: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, live])] = pl.tile.minimums(pl.tile.maximums(pl.tile.adds(values, 1.0), 0.0), 1.0)
                    upper_ok: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, live])] = pl.tile.minimums(pl.tile.maximums(pl.tile.adds(pl.tile.neg(values), page_limit), 0.0), 1.0)
                    ones: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, live])] = pl.tile.set_validshape(pl.tile.full([8, 1024], dtype=pl.FP32, value=1.0), 8, live)
                    repeated: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, live])] = pl.tile.set_validshape(
                        pl.tile.col_expand_mul(ones, pl.tile.mul(lower_ok, upper_ok)), 8, live
                    )
                    block_ok: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_min(repeated, reduce_tmp)
                    if pl.tile.read(block_ok, [0, 0]) < 1.0:
                        all_valid: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
        return all_valid

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def attention_slot_plan_stage(
        self,
        topk: pl.Tensor[[CSA_FULL_T, 512], pl.INT32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        original_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_ORIGINAL_COLS], pl.INT32],
        compressed_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_COMPRESSED_COLS], pl.INT32],
        indices: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32],
        bias: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32],
        valid_blocks: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32],
        original_pages: pl.Scalar[pl.INDEX],
        compressed_pages: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32], pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32], pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32]]:
        # 每个八行块独占 indices/bias/mask 行；只访问合法 owner 的 Native 页表。
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(positions, 0)
        for first in pl.range(worker_id * 8, tokens, 384):
            rows: pl.Scalar[pl.INDEX] = pl.min(8, tokens - first)
            c_raw_tile: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 512])] = pl.tile.cast(
                pl.tile.load(topk, [first, 0], [8, 512], [rows, 512]), target_type=pl.FP32, mode="round"
            )
            c_pos: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.full([1, 8], dtype=pl.FP32, value=0.0), [8, 1])
            v_len_col: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.full([1, 8], dtype=pl.FP32, value=0.0), [8, 1])
            owners: pl.Tile[[8, 1], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.full([1, 8], dtype=pl.INT32, value=-1), [8, 1])
            for row in pl.range(rows):
                token: pl.Scalar[pl.INDEX] = first + row
                _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.query_request(bounds, lengths, token, tokens)
                request: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
                begin: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
                end: pl.Scalar[pl.INDEX] = _tuple_tmp[2]
                active: pl.Scalar[pl.INT32] = _tuple_tmp[3]
                if pl.cast(active, pl.INDEX) != 0:
                    position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [token]), pl.INDEX)
                    length: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX)
                    if position == length - (end - begin) + token - begin:
                        pl.tile.write(owners, [row, 0], pl.cast(request, pl.INT32))
                        pl.tile.write(c_pos, [row, 0], pl.cast(pl.cast(position, pl.INT32), pl.FP32))
                        pl.tile.write(v_len_col, [row, 0], pl.cast(pl.cast(pl.min(position + 1, 128), pl.INT32), pl.FP32))
                if False:
                    owner: pl.Scalar[pl.INT32] = pl.tile.read(owners, [row, 0])
                    saved_position: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
                    if pl.cast(owner, pl.INDEX) >= 0:
                        saved_position: pl.Scalar[pl.INT32] = pl.cast(pl.tensor.read(positions, [token]), pl.INT32)
                    pl.tensor.write(valid_blocks, [token, 5], owner)
                    pl.tensor.write(valid_blocks, [token, 6], saved_position)
            c_pos_q: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.cast(
                pl.tile.cast(pl.tile.muls(pl.tile.adds(c_pos, 1.0), 0.25), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32, mode="round"
            )
            c_upper_b: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(pl.tile.full([8, 512], dtype=pl.FP32, value=1.0), c_pos_q)
            c_ge_tile: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 512])] = pl.tile.minimums(pl.tile.maximums(pl.tile.adds(c_raw_tile, 1.0), 0.0), 1.0)
            c_lt_tile: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(pl.tile.maximums(pl.tile.sub(c_upper_b, c_raw_tile), 0.0), 1.0)
            c_mask_tile: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 512])] = pl.tile.mul(c_ge_tile, c_lt_tile)
            c_page_indices: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 512])] = pl.tile.cast(
                pl.tile.subs(pl.tile.mul(c_mask_tile, pl.tile.adds(c_raw_tile, 1.0)), 1.0), target_type=pl.INT32, mode="round"
            )
            c_pages_valid: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=0.0)
            v_columns: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 128], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode="round")
            v_col_tile: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([8, 128], dtype=pl.FP32, value=1.0), v_columns)
            v_valid_tile: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(pl.tile.maximums(pl.tile.neg(pl.tile.row_expand_sub(v_col_tile, v_len_col)), 0.0), 1.0)
            pl.tile.store(pl.tile.set_validshape(pl.tile.muls(pl.tile.subs(v_valid_tile, 1.0), 1e20), rows, 128), [first, 0], bias)
            checked_request: pl.Scalar[pl.INDEX] = pl.cast(-1, pl.INDEX)
            checked_pages_valid: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
            for row_1 in pl.range(rows):
                token: pl.Scalar[pl.INDEX] = first + row_1
                request: pl.Scalar[pl.INDEX] = pl.cast(pl.tile.read(owners, [row_1, 0]), pl.INDEX)
                v_block_valid: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
                if request >= 0:
                    upper: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(pl.tile.read(c_pos_q, [row_1, 0]), pl.INT32), pl.INDEX)
                    if request != checked_request:
                        length: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX)
                        checked_pages_valid: pl.Scalar[pl.INT32] = self.compressed_pages_all_valid(compressed_table, request, length, compressed_pages)
                        checked_request: pl.Scalar[pl.INDEX] = request
                    if pl.cast(checked_pages_valid, pl.INDEX) != 0:
                        final_indices: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(pl.max(rows - row, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)), 512])] = pl.tile.cast(
                            pl.tile.slice(c_page_indices, [1, 512], [row_1, 0], [1, 512]), target_type=pl.FP32, mode="round"
                        )
                        final_ge: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(pl.max(rows - row, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)), 512])] = pl.tile.minimums(
                            pl.tile.maximums(pl.tile.adds(final_indices, 1.0), 0.0), 1.0
                        )
                        upper_float: pl.Scalar[pl.FP32] = pl.cast(pl.cast(upper, pl.INT32), pl.FP32)
                        final_lt: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(pl.max(rows - row, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)), 512])] = pl.tile.minimums(
                            pl.tile.maximums(pl.tile.adds(pl.tile.neg(final_indices), upper_float), 0.0), 1.0
                        )
                        c_pages_valid: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(c_pages_valid, pl.tile.mul(final_ge, final_lt), [row_1, 0])
                    else:
                        for candidate in pl.range(512):
                            raw: pl.Scalar[pl.INDEX] = pl.cast(pl.tile.read(c_page_indices, [row_1, candidate]), pl.INDEX)
                            if raw >= 0 and raw < upper:
                                column: pl.Scalar[pl.INDEX] = raw // 32
                                if column < pl.tensor.dim(compressed_table, 1):
                                    page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(compressed_table, [request, column]), pl.INDEX)
                                    if page >= 0 and page < compressed_pages:
                                        pl.tile.write(c_pages_valid, [row_1, candidate], pl.cast(1.0, pl.FP32))
                    position_v1: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [token]), pl.INDEX)
                    v_length: pl.Scalar[pl.INDEX] = pl.min(position_v1 + 1, 128)
                    v_start: pl.Scalar[pl.INDEX] = position_v1 - v_length + 1
                    v_head: pl.Scalar[pl.INDEX] = v_start % 32
                    v_has_hole: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
                    for v_run in pl.range(5):
                        v_lo: pl.Scalar[pl.INDEX] = pl.max(v_run * 32 - v_head, 0)
                        v_hi: pl.Scalar[pl.INDEX] = pl.min((v_run + 1) * 32 - v_head, v_length)
                        if v_hi > v_lo:
                            column: pl.Scalar[pl.INDEX] = (v_start + v_lo) // 32
                            page_valid: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
                            if column >= 0 and column < pl.tensor.dim(original_table, 1):
                                page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(original_table, [request, column]), pl.INDEX)
                                if page >= 0 and page < original_pages:
                                    page_valid: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
                            if pl.cast(page_valid, pl.INDEX) != 0:
                                v_block_valid: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
                            else:
                                v_has_hole: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
                    if pl.cast(v_has_hole, pl.INDEX) == 1:
                        v_columns_row: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(
                            pl.tile.ci(pl.const(0, pl.INT32), [1, 128], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode="round"
                        )
                        v_valid: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
                        for v_hole_run in pl.range(5):
                            v_hole_lo: pl.Scalar[pl.INDEX] = pl.max(v_hole_run * 32 - v_head, 0)
                            v_hole_hi: pl.Scalar[pl.INDEX] = pl.min((v_hole_run + 1) * 32 - v_head, v_length)
                            if v_hole_hi > v_hole_lo:
                                column_v1: pl.Scalar[pl.INDEX] = (v_start + v_hole_lo) // 32
                                if column_v1 >= 0 and column_v1 < pl.tensor.dim(original_table, 1):
                                    page_v1: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(original_table, [request, column_v1]), pl.INDEX)
                                    if page_v1 >= 0 and page_v1 < original_pages:
                                        v_lo_fp32: pl.Scalar[pl.FP32] = pl.cast(pl.cast(v_hole_lo, pl.INT32), pl.FP32)
                                        v_hi_fp32: pl.Scalar[pl.FP32] = pl.cast(pl.cast(v_hole_hi, pl.INT32), pl.FP32)
                                        v_ge: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(pl.tile.maximums(pl.tile.adds(pl.tile.subs(v_columns_row, v_lo_fp32), 1.0), 0.0), 1.0)
                                        v_lt: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(pl.tile.maximums(pl.tile.adds(pl.tile.neg(v_columns_row), v_hi_fp32), 0.0), 1.0)
                                        v_valid: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(v_valid, pl.tile.mul(v_ge, v_lt))
                        pl.tile.store(pl.tile.muls(pl.tile.subs(v_valid, 1.0), 1e20), [token, 0], bias)
                pl.tensor.write(valid_blocks, [token, 0], v_block_valid)
            c_mask_tile_v1: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 512])] = pl.tile.mul(c_mask_tile, c_pages_valid)
            c_out_tile: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 512])] = pl.tile.subs(pl.tile.mul(c_mask_tile_v1, pl.tile.adds(c_raw_tile, 1.0)), 1.0)
            pl.tile.store(pl.tile.set_validshape(pl.tile.cast(c_out_tile, target_type=pl.INT32, mode="round"), rows, 512), [first, 0], indices)
            pl.tile.store(pl.tile.set_validshape(pl.tile.muls(pl.tile.minimums(c_out_tile, 0.0), 1e20), rows, 512), [first, 128], bias)
            reduce_tmp: pl.Tile[[8, 128], pl.FP32] = pl.tile.create([8, 128], dtype=pl.FP32)
            for block in pl.range(1, 5):
                c_blk_valid: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(rows, pl.const(8, pl.INDEX)), 1])] = pl.tile.row_max(
                    pl.tile.slice(c_mask_tile_v1, [8, 128], [0, (block - 1) * 128], [rows, 128]), reduce_tmp
                )
                for row_2 in pl.range(rows):
                    c_valid: pl.Scalar[pl.INT32] = pl.cast(pl.tile.read(c_blk_valid, [row_2, 0]), pl.INT32)
                    pl.tensor.write(valid_blocks, [first + row_2, block], c_valid)
        return indices, bias, valid_blocks

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def _scan_compact_offsets(
        self, bounds: pl.Tensor[[CSA_FULL_MQ], pl.INT32], lengths: pl.Tensor[[CSA_FULL_B], pl.INT32], offsets: pl.Tensor[[CSA_FULL_B], pl.INT32], token_capacity: pl.Scalar[pl.INDEX]
    ):
        # 只供指定的单写者内联，返回 offsets 别名和核内最终前缀 Scalar。
        prefix: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        for request in pl.range(pl.tensor.dim(lengths, 0)):
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.request_bounds(bounds, lengths, request, token_capacity)
            begin: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            end: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
            value: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
            if pl.cast(active, pl.INDEX) != 0:
                length: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX)
                history_start: pl.Scalar[pl.INDEX] = length - (end - begin)
                value: pl.Scalar[pl.INT32] = pl.cast(prefix - history_start // 4 - 1, pl.INT32)
                prefix: pl.Scalar[pl.INDEX] = prefix + length // 4 - history_start // 4
            pl.tensor.write(offsets, [request], value)
        return offsets, prefix

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def compact_offsets_and_count_stage(
        self,
        bounds: pl.Tensor[[CSA_FULL_MQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        offsets: pl.Tensor[[CSA_FULL_B], pl.INT32],
        valid_count: pl.Tensor[[1], pl.INT32],
        token_capacity: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_B], pl.INT32], pl.Tensor[[1], pl.INT32]]:
        # 同一扫描写 offsets 与实际行数，外层一次发布这两个 GM 输出。
        #
        #         count 来自合法 Q/L 的关闭候选数，不能因 stale position 缩减，否则会
        #         改变后续请求的 Native compact 偏移。norm 先初始化各请求的候选行，再
        #         只覆盖 causal 行；cache 副作用仍逐行核对 position/slot。调用者保证
        #         count 不大于 compact 表容量，不能把容量不足的输入称为有效支持场景。
        #
        if worker_id == 0:
            _tuple_tmp: pl.Tuple[pl.Tensor[[CSA_FULL_B], pl.INT32], pl.Scalar[pl.INDEX]] = self._scan_compact_offsets(bounds, lengths, offsets, token_capacity)
            offsets: pl.Tensor[[CSA_FULL_B], pl.INT32] = _tuple_tmp[0]
            count: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            pl.tensor.write(valid_count, [0], pl.cast(count, pl.INT32))
        return offsets, valid_count

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def load_request_compact_rope_rows(
        self,
        cos: pl.Tensor[[CSA_FULL_IC, 64], pl.FP32],
        sin: pl.Tensor[[CSA_FULL_IC, 64], pl.FP32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        offsets: pl.Tensor[[CSA_FULL_B], pl.INT32],
        request: pl.Scalar[pl.INDEX],
        first: pl.Scalar[pl.INDEX],
        rows: pl.Scalar[pl.INDEX],
    ):
        # 在消费者 AIV 内返回 cos/sin UB tile；first/rows 必须属于同一请求。
        cosine: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        sine: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
        _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
        begin: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
        end: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
        active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
        if pl.cast(active, pl.INDEX) != 0 and rows > 0 and rows <= 16 and first >= begin and first + rows <= end:
            length: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX)
            history_start: pl.Scalar[pl.INDEX] = length - (end - begin)
            compact_offset: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(offsets, [request]), pl.INDEX)
            for row in pl.range(rows):
                token: pl.Scalar[pl.INDEX] = first + row
                position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [token]), pl.INDEX)
                if position == history_start + token - begin and (position + 1) % 4 == 0:
                    compact_row: pl.Scalar[pl.INDEX] = compact_offset + (position + 1) // 4
                    if compact_row >= 0 and compact_row < pl.tensor.dim(cos, 0) and compact_row < pl.tensor.dim(sin, 0):
                        cosine: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather_row(cosine, cos, [row, 0], [compact_row, 0], [1, 64], transpose=False)
                        sine: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather_row(sine, sin, [row, 0], [compact_row, 0], [1, 64], transpose=False)
        return cosine, sine

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def compressor_norm_rope_cache_stage(
        self,
        pooled_kv: pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32],
        norm_w: pl.Tensor[[512], pl.BF16],
        cos: pl.Tensor[[CSA_FULL_MC, 64], pl.FP32],
        sin: pl.Tensor[[CSA_FULL_MC, 64], pl.FP32],
        bounds: pl.Tensor[[CSA_FULL_MQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        compact_offsets: pl.Tensor[[CSA_FULL_B], pl.INT32],
        compact_slots: pl.Tensor[[CSA_FULL_MSLOTS, 2], pl.INT32],
        normed_kv: pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32],
        kv_debug: pl.Tensor[[CSA_FULL_T, 512], pl.FP32],
        cache_flat: pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32], pl.Tensor[[CSA_FULL_T, 512], pl.FP32], pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16]]:
        # AIV请求条带；保留未写区，返回三个原GM别名，由外层发布完成信号。
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(positions, 0)
        for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), 48):
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.request_bounds(bounds, lengths, request, tokens)
            begin: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            end: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
            if pl.cast(active, pl.INDEX) != 0:
                length: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX)
                history_start: pl.Scalar[pl.INDEX] = length - (end - begin)
                compact_offset: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(compact_offsets, [request]), pl.INDEX)
                for first in pl.range(begin, end, 16):
                    rows: pl.Scalar[pl.INDEX] = pl.min(16, end - first)
                    _tuple_tmp_1: pl.Tuple[pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]] = self.load_request_compact_rope_rows(
                        cos, sin, positions, bounds, lengths, compact_offsets, request, first, rows
                    )
                    cos_tile: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = _tuple_tmp_1[0]
                    sin_tile: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = _tuple_tmp_1[1]
                    cos_b: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.set_validshape(cos_tile, rows, 64)
                    sin_b: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.set_validshape(sin_tile, rows, 64)
                    rms_low: pl.Tile[[16, 256], pl.FP32, pl.TileView(valid_shape=[rows, 256])] = pl.tile.load(pooled_kv, [first, 0], [16, 256], [rows, 256])
                    rms_high: pl.Tile[[16, 256], pl.FP32, pl.TileView(valid_shape=[rows, 256])] = pl.tile.load(pooled_kv, [first, 256], [16, 256], [rows, 256])
                    folded4: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 256])] = pl.tile.add(pl.tile.mul(rms_low, rms_low), pl.tile.mul(rms_high, rms_high))
                    folded2: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(rows, pl.const(16, pl.INDEX)), 128])] = pl.tile.add(
                        pl.tile.slice(folded4, [16, 128], [0, 0], [rows, 128]), pl.tile.slice(folded4, [16, 128], [0, 128], [rows, 128])
                    )
                    folded1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(rows, pl.const(16, pl.INDEX)), 64])] = pl.tile.add(
                        pl.tile.slice(folded2, [16, 64], [0, 0], [pl.min(rows, 16), 64]), pl.tile.slice(folded2, [16, 64], [0, 64], [pl.min(rows, 16), 64])
                    )
                    square_sum: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(rows, pl.const(16, pl.INDEX)), 1])] = pl.tile.row_sum(
                        folded1, pl.tile.create([16, 64], dtype=pl.FP32)
                    )
                    variance: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(rows, pl.const(16, pl.INDEX)), 1])] = pl.tile.adds(
                        pl.tile.muls(square_sum, 0.001953125), 9.9999999999999995e-07
                    )
                    inv_rms: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(rows, pl.const(16, pl.INDEX)), 1])] = pl.tile.recip(pl.tile.sqrt(variance))
                    for k0 in pl.range(0, 448, 64):
                        kv_norm_chunk: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[rows, 64])] = pl.tile.load(pooled_kv, [first, k0], [16, 64], [rows, 64])
                        gamma: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.reshape(pl.tile.load(norm_w, [k0], [64], [64]), [1, 64]), target_type=pl.FP32, mode="round")
                        normed_chunk: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.col_expand_mul(pl.tile.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
                        pl.tile.store(pl.tile.set_validshape(normed_chunk, rows, 64), [first, k0], normed_kv)
                    kv_rope_norm: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[rows, 64])] = pl.tile.load(pooled_kv, [first, 448], [16, 64], [rows, 64])
                    gamma_rope: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.reshape(pl.tile.load(norm_w, [448], [64], [64]), [1, 64]), target_type=pl.FP32, mode="round")
                    rope_normed: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.col_expand_mul(pl.tile.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
                    rope_ones: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
                    rope_index: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                    rope_index_f: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(rope_index, target_type=pl.FP32, mode="round")
                    rope_col: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(rope_ones, rope_index_f)
                    rope_dup_f: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32, mode="round")
                    rope_lane: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(rope_col, pl.tile.muls(rope_dup_f, 2.0))
                    rope_swap_idx: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.sub(pl.tile.adds(rope_col, 1.0), pl.tile.muls(rope_lane, 2.0)), target_type=pl.INT32, mode="round")
                    rope_row_offsets: pl.Tile[[16, 1], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(
                        pl.tile.muls(pl.tile.ci(pl.const(0, pl.INT32), [1, 16], dtype=pl.INT32, descending=False), pl.const(64, pl.INT32)), [16, 1]
                    )
                    rope_abs_idx: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.row_expand_add(rope_swap_idx, rope_row_offsets)
                    rope_abs_idx_v1: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.set_validshape(rope_abs_idx, rows, 64)
                    swapped: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(rope_normed, rope_abs_idx_v1, pl.tile.create([16, 64], dtype=pl.INT32))
                    sin_signed: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.mul(sin_b, pl.tile.subs(pl.tile.muls(rope_lane, 2.0), 1.0))
                    rope_rot: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.add(pl.tile.mul(rope_normed, cos_b), pl.tile.mul(swapped, sin_signed))
                    pl.tile.store(pl.tile.set_validshape(rope_rot, rows, 64), [first, 448], normed_kv)
                    for inner in pl.range(rows):
                        token: pl.Scalar[pl.INDEX] = first + inner
                        position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [token]), pl.INDEX)
                        if position == history_start + token - begin and (position + 1) % 4 == 0:
                            compact_row: pl.Scalar[pl.INDEX] = compact_offset + (position + 1) // 4
                            if compact_row >= 0 and compact_row < pl.tensor.dim(compact_slots, 0) and compact_row < pl.tensor.dim(cos, 0) and compact_row < pl.tensor.dim(sin, 0):
                                page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(compact_slots, [compact_row, 0]), pl.INDEX)
                                offset: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(compact_slots, [compact_row, 1]), pl.INDEX)
                                if page >= 0 and page < pl.tensor.dim(cache_flat, 0) // 32 and offset >= 0 and offset < 32:
                                    cache_row: pl.Scalar[pl.INDEX] = page * 32 + offset
                                    if cache_row < pl.tensor.dim(cache_flat, 0):
                                        kv_row_fp32: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.slice(normed_kv, [1, 512], [token, 0])
                                        kv_debug: pl.Tensor[[CSA_FULL_T, 512], pl.FP32] = pl.tensor.assemble(kv_debug, kv_row_fp32, [token, 0])
                                        cache_flat: pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16] = pl.tensor.assemble(
                                            cache_flat, pl.tensor.cast(kv_row_fp32, target_type=pl.BF16, mode="rint"), [cache_row, 0]
                                        )
        return normed_kv, kv_debug, cache_flat

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def compressor_pool_stage(
        self,
        kv_proj: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        score_proj: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        state: pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32],
        state_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_MTABLE_COLS], pl.INT32],
        ape: pl.Tensor[[4, 1024], pl.FP32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        bounds: pl.Tensor[[CSA_FULL_MQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        pooled: pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32]:
        # AIV 主压缩：请求内串行处理关闭窗口，整个 512 列保持原计算粒度。
        for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), 48):
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
            begin: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            end: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
            if pl.cast(active, pl.INDEX) != 0:
                first_position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX) - (end - begin)
                for token in pl.range(begin, end):
                    pl.tile.store(pl.tile.full([1, 512], dtype=pl.FP32, value=0.0), [token, 0], pooled)
                    position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [token]), pl.INDEX)
                    if position == first_position + token - begin and (position + 1) % 4 == 0:
                        window_start: pl.Scalar[pl.INDEX] = position - 8 + 1
                        last_ape: pl.Scalar[pl.INDEX] = position % 4
                        mi: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(pl.tile.load(score_proj, [token, 512], [1, 512], [1, 512]), pl.tile.load(ape, [last_ape, 512], [1, 512], [1, 512]))
                        li: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.exp(pl.tile.sub(mi, mi))
                        oi: pl.Tile[[1, 512], pl.FP32] = pl.tile.load(kv_proj, [token, 512], [1, 512], [1, 512])
                        for state_index in pl.range(7):
                            logical_position: pl.Scalar[pl.INDEX] = window_start + state_index
                            value: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                            score: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e38)
                            half: pl.Scalar[pl.INDEX] = 0
                            if state_index >= 4:
                                half: pl.Scalar[pl.INDEX] = 512
                            if logical_position >= 0 and logical_position < first_position:
                                score: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                                table_column: pl.Scalar[pl.INDEX] = logical_position // 2
                                if table_column < pl.tensor.dim(state_table, 1):
                                    page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(state_table, [request, table_column]), pl.INDEX)
                                    column: pl.Scalar[pl.INDEX] = logical_position % 2 * 2048 + half
                                    if page >= 0 and page < pl.tensor.dim(state, 0):
                                        if column + 1024 + 512 <= pl.tensor.dim(state, 1):
                                            value: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(state, [page, column], [1, 512], [1, 512])
                                            score: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(state, [page, column + 1024], [1, 512], [1, 512])
                            if logical_position >= first_position and logical_position <= position:
                                overlay_token: pl.Scalar[pl.INDEX] = begin + logical_position - first_position
                                ape_row: pl.Scalar[pl.INDEX] = logical_position % 4
                                value: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.load(kv_proj, [overlay_token, half], [1, 512], [1, 512])
                                score: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(
                                    pl.tile.load(score_proj, [overlay_token, half], [1, 512], [1, 512]), pl.tile.load(ape, [ape_row, half], [1, 512], [1, 512])
                                )
                            mi_next: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(mi, score)
                            alpha: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.exp(pl.tile.sub(mi, mi_next))
                            beta: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.exp(pl.tile.sub(score, mi_next))
                            li: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.add(pl.tile.mul(alpha, li), beta)
                            oi: pl.Tile[[1, 512], pl.FP32] = pl.tile.add(pl.tile.mul(oi, alpha), pl.tile.mul(value, beta))
                            mi: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = mi_next
                        pl.tile.store(pl.tile.div(oi, li), [token, 0], pooled)
        return pooled

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def compressor_state_commit_stage(
        self,
        kv_proj: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        score_proj: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        ape: pl.Tensor[[4, 1024], pl.FP32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        bounds: pl.Tensor[[CSA_FULL_MQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        state_slots: pl.Tensor[[CSA_FULL_T, 2], pl.INT32],
        state: pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32]:
        # AIV 主 state 写回；外层先等待所有读取旧 state 的 pool，不只等本核。
        for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), 48):
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
            begin: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            end: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
            if pl.cast(active, pl.INDEX) != 0:
                first_position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX) - (end - begin)
                for token in pl.range(begin, end):
                    position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [token]), pl.INDEX)
                    if position == first_position + token - begin:
                        page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(state_slots, [token, 0]), pl.INDEX)
                        offset: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(state_slots, [token, 1]), pl.INDEX)
                        if page >= 0 and page < pl.tensor.dim(state, 0) and offset >= 0 and offset < 2:
                            column: pl.Scalar[pl.INDEX] = offset * 2048
                            if column + 2048 <= pl.tensor.dim(state, 1):
                                ape_row: pl.Scalar[pl.INDEX] = position % 4
                                value: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(kv_proj, [token, 0], [1, 1024], [1, 1024])
                                score: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.add(
                                    pl.tile.load(score_proj, [token, 0], [1, 1024], [1, 1024]), pl.tile.load(ape, [ape_row, 0], [1, 1024], [1, 1024])
                                )
                                pl.tile.store(value, [page, column], state)
                                pl.tile.store(score, [page, column + 1024], state)
        return state

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def hc_projection_piece__3(self, partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32], raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32], row: pl.Scalar[pl.INDEX], valid_rows: pl.Scalar[pl.INDEX]):
        # 加载消费者需要的M8×N8片段；直接路径保持((p0+p1)+p2)+p3。
        #
        #         VALID_COLUMNS=4用于Sinkhorn，8用于门值，尾行由调用方显式给出。
        #         partials四段各占PAD16行，列仍为32；所有访问只读已由AIC发布的区域。
        #         raw仅在旧路径load；直接路径只取其容量描述符，绝不消费填毒的raw内容。
        #
        if False:
            rows_per_split: pl.Scalar[pl.INDEX] = pl.tensor.dim(raw, 0)
            result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(partials, [row, 8], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
            for split in pl.range(1, 4):
                piece: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(
                    partials, [split * rows_per_split + row, 8], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec
                )
                result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(result, piece)
        else:
            result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(raw, [row, 8], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
        return result

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def hc_projection_piece__4(self, partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32], raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32], row: pl.Scalar[pl.INDEX], valid_rows: pl.Scalar[pl.INDEX]):
        # 加载消费者需要的M8×N8片段；直接路径保持((p0+p1)+p2)+p3。
        #
        #         VALID_COLUMNS=4用于Sinkhorn，8用于门值，尾行由调用方显式给出。
        #         partials四段各占PAD16行，列仍为32；所有访问只读已由AIC发布的区域。
        #         raw仅在旧路径load；直接路径只取其容量描述符，绝不消费填毒的raw内容。
        #
        if False:
            rows_per_split: pl.Scalar[pl.INDEX] = pl.tensor.dim(raw, 0)
            result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(partials, [row, 12], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
            for split in pl.range(1, 4):
                piece: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(
                    partials, [split * rows_per_split + row, 12], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec
                )
                result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(result, piece)
        else:
            result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(raw, [row, 12], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
        return result

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def hc_projection_piece__5(self, partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32], raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32], row: pl.Scalar[pl.INDEX], valid_rows: pl.Scalar[pl.INDEX]):
        # 加载消费者需要的M8×N8片段；直接路径保持((p0+p1)+p2)+p3。
        #
        #         VALID_COLUMNS=4用于Sinkhorn，8用于门值，尾行由调用方显式给出。
        #         partials四段各占PAD16行，列仍为32；所有访问只读已由AIC发布的区域。
        #         raw仅在旧路径load；直接路径只取其容量描述符，绝不消费填毒的raw内容。
        #
        if False:
            rows_per_split: pl.Scalar[pl.INDEX] = pl.tensor.dim(raw, 0)
            result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(partials, [row, 16], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
            for split in pl.range(1, 4):
                piece: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(
                    partials, [split * rows_per_split + row, 16], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec
                )
                result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(result, piece)
        else:
            result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(raw, [row, 16], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
        return result

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def hc_projection_piece__6(self, partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32], raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32], row: pl.Scalar[pl.INDEX], valid_rows: pl.Scalar[pl.INDEX]):
        # 加载消费者需要的M8×N8片段；直接路径保持((p0+p1)+p2)+p3。
        #
        #         VALID_COLUMNS=4用于Sinkhorn，8用于门值，尾行由调用方显式给出。
        #         partials四段各占PAD16行，列仍为32；所有访问只读已由AIC发布的区域。
        #         raw仅在旧路径load；直接路径只取其容量描述符，绝不消费填毒的raw内容。
        #
        if False:
            rows_per_split: pl.Scalar[pl.INDEX] = pl.tensor.dim(raw, 0)
            result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(partials, [row, 20], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
            for split in pl.range(1, 4):
                piece: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(
                    partials, [split * rows_per_split + row, 20], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec
                )
                result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(result, piece)
        else:
            result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(raw, [row, 20], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
        return result

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def hc_comb_range(
        self,
        inv_rms: pl.Tensor[[CSA_FULL_P16, 1], pl.FP32],
        mixes_raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32],
        hc_scale: pl.Tensor[[3], pl.FP32],
        hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
        comb: pl.Tensor[[CSA_FULL_T, 16], pl.FP32],
        comb_tail_store: pl.Tensor[[8, 32], pl.FP32],
        vector_worker: pl.Scalar[pl.INDEX],
        vector_workers: pl.Scalar[pl.INDEX],
        partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32],
    ) -> pl.Tensor[[CSA_FULL_T, 16], pl.FP32]:
        # AIV：CSA 原除法路径的二十轮 Sinkhorn，保持行列归约树。
        #
        #         调用前等待 HC_INPUT 与 HC_RAW（直接归约时改等四个HC_PARTIAL）；与 hc_gates_range/mix 独立，返回后
        #         发布 HC_COMB。不引入 HCA 的 recip 替代路径，不改变迭代次数。
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(comb, 0)
        token_tiles: pl.Scalar[pl.INDEX] = (t_dim + 8 - 1) // 8
        for wave in pl.range((token_tiles + vector_workers - 1) // vector_workers):
            ob: pl.Scalar[pl.INDEX] = vector_worker + wave * vector_workers
            if ob < token_tiles:
                scale2: pl.Scalar[pl.FP32] = pl.tensor.read(hc_scale, [2])
                t0: pl.Scalar[pl.INDEX] = ob * 8
                valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, t_dim - t0)
                inv_col_t: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 1])] = pl.tile.load(inv_rms, [t0, 0], [8, 1], [valid_rows, 1], target_memory=pl.Mem.Vec)
                comb_off: pl.Scalar[pl.INDEX] = 8
                mix_g0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = self.hc_projection_piece__3(partials, mixes_raw, t0, valid_rows)
                mix_g1: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = self.hc_projection_piece__4(partials, mixes_raw, t0, valid_rows)
                mix_g2: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = self.hc_projection_piece__5(partials, mixes_raw, t0, valid_rows)
                mix_g3: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = self.hc_projection_piece__6(partials, mixes_raw, t0, valid_rows)
                cb0: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d, [0, comb_off + 0], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
                cb1: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d, [0, comb_off + 4], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
                cb2: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d, [0, comb_off + 8], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
                cb3: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d, [0, comb_off + 12], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
                row0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(
                    pl.tile.muls(pl.tile.row_expand_mul(mix_g0, inv_col_t), scale2), pl.tile.col_expand(mix_g0, cb0)
                )
                row1: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(
                    pl.tile.muls(pl.tile.row_expand_mul(mix_g1, inv_col_t), scale2), pl.tile.col_expand(mix_g1, cb1)
                )
                row2: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(
                    pl.tile.muls(pl.tile.row_expand_mul(mix_g2, inv_col_t), scale2), pl.tile.col_expand(mix_g2, cb2)
                )
                row3: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(
                    pl.tile.muls(pl.tile.row_expand_mul(mix_g3, inv_col_t), scale2), pl.tile.col_expand(mix_g3, cb3)
                )
                row0_p: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row0, pad_value=pl.PadValue.min)
                row1_p: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row1, pad_value=pl.PadValue.min)
                row2_p: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row2, pad_value=pl.PadValue.min)
                row3_p: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row3, pad_value=pl.PadValue.min)
                row_max_tmp: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                row_sum_tmp: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                row0_max: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row0_p, row_max_tmp)
                row1_max: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row1_p, row_max_tmp)
                row2_max: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row2_p, row_max_tmp)
                row3_max: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row3_p, row_max_tmp)
                row0_exp: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row0_p, row0_max))
                row1_exp: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row1_p, row1_max))
                row2_exp: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row2_p, row2_max))
                row3_exp: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row3_p, row3_max))
                row0_sum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row0_exp, row_sum_tmp)
                row1_sum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row1_exp, row_sum_tmp)
                row2_sum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row2_exp, row_sum_tmp)
                row3_sum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row3_exp, row_sum_tmp)
                row0_soft: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row0_exp, row0_sum), 9.9999999999999995e-07)
                row1_soft: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row1_exp, row1_sum), 9.9999999999999995e-07)
                row2_soft: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row2_exp, row2_sum), 9.9999999999999995e-07)
                row3_soft: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row3_exp, row3_sum), 9.9999999999999995e-07)
                row0_valid: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row0_soft, 8, 4)
                row1_valid: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row1_soft, 8, 4)
                row2_valid: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row2_soft, 8, 4)
                row3_valid: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row3_soft, 8, 4)
                row0_eff: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row0_valid, pad_value=pl.PadValue.zero)
                row1_eff: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row1_valid, pad_value=pl.PadValue.zero)
                row2_eff: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row2_valid, pad_value=pl.PadValue.zero)
                row3_eff: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row3_valid, pad_value=pl.PadValue.zero)
                row_sum_tmp_iter: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                col_sum: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(pl.tile.add(row0_eff, row1_eff), pl.tile.add(row2_eff, row3_eff))
                col_sum_v1: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(col_sum, 9.9999999999999995e-07)
                row0_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row0_eff, col_sum_v1)
                row1_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row1_eff, col_sum_v1)
                row2_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row2_eff, col_sum_v1)
                row3_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row3_eff, col_sum_v1)
                for _sk_it in pl.pipeline(19, stage=2):
                    row0_rowsum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row0_cur, row_sum_tmp_iter), 9.9999999999999995e-07)
                    row1_rowsum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row1_cur, row_sum_tmp_iter), 9.9999999999999995e-07)
                    row2_rowsum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row2_cur, row_sum_tmp_iter), 9.9999999999999995e-07)
                    row3_rowsum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row3_cur, row_sum_tmp_iter), 9.9999999999999995e-07)
                    row0_norm: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row0_cur, row0_rowsum)
                    row1_norm: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row1_cur, row1_rowsum)
                    row2_norm: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row2_cur, row2_rowsum)
                    row3_norm: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row3_cur, row3_rowsum)
                    col_sum_v1: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(pl.tile.add(row0_norm, row1_norm), pl.tile.add(row2_norm, row3_norm))
                    col_sum_v1: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(col_sum_v1, 9.9999999999999995e-07)
                    row0_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row0_norm, col_sum_v1)
                    row1_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row1_norm, col_sum_v1)
                    row2_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row2_norm, col_sum_v1)
                    row3_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row3_norm, col_sum_v1)
                if valid_rows == 8:
                    row0_out: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row0_cur, 8, 4)
                    row1_out: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row1_cur, 8, 4)
                    row2_out: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row2_cur, 8, 4)
                    row3_out: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row3_cur, 8, 4)
                    pl.tile.store(row0_out, [t0, 0], comb)
                    pl.tile.store(row1_out, [t0, 4], comb)
                    pl.tile.store(row2_out, [t0, 8], comb)
                    pl.tile.store(row3_out, [t0, 12], comb)
                else:
                    pl.tile.store(row0_cur, [0, 0], comb_tail_store)
                    pl.tile.store(row1_cur, [0, 8], comb_tail_store)
                    pl.tile.store(row2_cur, [0, 16], comb_tail_store)
                    pl.tile.store(row3_cur, [0, 24], comb_tail_store)
                    row0_tail: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(comb_tail_store, [0, 0], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
                    row1_tail: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(comb_tail_store, [0, 8], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
                    row2_tail: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(
                        comb_tail_store, [0, 16], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec
                    )
                    row3_tail: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(
                        comb_tail_store, [0, 24], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec
                    )
                    pl.tile.store(row0_tail, [t0, 0], comb)
                    pl.tile.store(row1_tail, [t0, 4], comb)
                    pl.tile.store(row2_tail, [t0, 8], comb)
                    pl.tile.store(row3_tail, [t0, 12], comb)
        return comb

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def hc_projection_piece(self, partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32], raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32], row: pl.Scalar[pl.INDEX], valid_rows: pl.Scalar[pl.INDEX]):
        # 加载消费者需要的M8×N8片段；直接路径保持((p0+p1)+p2)+p3。
        #
        #         VALID_COLUMNS=4用于Sinkhorn，8用于门值，尾行由调用方显式给出。
        #         partials四段各占PAD16行，列仍为32；所有访问只读已由AIC发布的区域。
        #         raw仅在旧路径load；直接路径只取其容量描述符，绝不消费填毒的raw内容。
        #
        if False:
            rows_per_split: pl.Scalar[pl.INDEX] = pl.tensor.dim(raw, 0)
            result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.load(partials, [row, 0], [8, 8], [valid_rows, 8], target_memory=pl.Mem.Vec)
            for split in pl.range(1, 4):
                piece: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.load(
                    partials, [split * rows_per_split + row, 0], [8, 8], [valid_rows, 8], target_memory=pl.Mem.Vec
                )
                result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.add(result, piece)
        else:
            result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.load(raw, [row, 0], [8, 8], [valid_rows, 8], target_memory=pl.Mem.Vec)
        return result

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def hc_projection_piece__2(self, partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32], raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32], row: pl.Scalar[pl.INDEX], valid_rows: pl.Scalar[pl.INDEX]):
        # 加载消费者需要的M8×N8片段；直接路径保持((p0+p1)+p2)+p3。
        #
        #         VALID_COLUMNS=4用于Sinkhorn，8用于门值，尾行由调用方显式给出。
        #         partials四段各占PAD16行，列仍为32；所有访问只读已由AIC发布的区域。
        #         raw仅在旧路径load；直接路径只取其容量描述符，绝不消费填毒的raw内容。
        #
        if False:
            rows_per_split: pl.Scalar[pl.INDEX] = pl.tensor.dim(raw, 0)
            result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.load(partials, [row, 4], [8, 8], [valid_rows, 8], target_memory=pl.Mem.Vec)
            for split in pl.range(1, 4):
                piece: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.load(
                    partials, [split * rows_per_split + row, 4], [8, 8], [valid_rows, 8], target_memory=pl.Mem.Vec
                )
                result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.add(result, piece)
        else:
            result: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.load(raw, [row, 4], [8, 8], [valid_rows, 8], target_memory=pl.Mem.Vec)
        return result

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def hc_gates_range(
        self,
        inv_rms: pl.Tensor[[CSA_FULL_P16, 1], pl.FP32],
        mixes_raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32],
        hc_scale: pl.Tensor[[3], pl.FP32],
        hc_base: pl.Tensor[[24], pl.FP32],
        pre_val_store: pl.Tensor[[CSA_FULL_P16, 8], pl.FP32],
        post: pl.Tensor[[CSA_FULL_T, 4], pl.FP32],
        post_tail_store: pl.Tensor[[8, 8], pl.FP32],
        vector_worker: pl.Scalar[pl.INDEX],
        vector_workers: pl.Scalar[pl.INDEX],
        partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32],
    ) -> tuple[pl.Tensor[[CSA_FULL_P16, 8], pl.FP32], pl.Tensor[[CSA_FULL_T, 4], pl.FP32]]:
        # AIV：保留原 pre/post sigmoid、scale 乘法和 eps 顺序。
        #
        #         调用前等待 HC_INPUT 与 HC_RAW（直接归约时改等四个HC_PARTIAL）；返回后发布 HC_GATES，供 mix/norm
        #         和最终 HCpost 消费。pre 按八列物理布局保留，post 只有四列有效。
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(post, 0)
        token_tiles: pl.Scalar[pl.INDEX] = (t_dim + 8 - 1) // 8
        scale0: pl.Scalar[pl.FP32] = pl.tensor.read(hc_scale, [0])
        scale1: pl.Scalar[pl.FP32] = pl.tensor.read(hc_scale, [1])
        for wave in pl.range((token_tiles + vector_workers - 1) // vector_workers):
            ob: pl.Scalar[pl.INDEX] = vector_worker + wave * vector_workers
            if ob < token_tiles:
                t0: pl.Scalar[pl.INDEX] = ob * 8
                valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, t_dim - t0)
                inv_col: pl.Tile[[8, 1], pl.FP32] = pl.tile.load(inv_rms, [t0, 0], [8, 1], [8, 1])
                pre_base: pl.Tile[[1, 8], pl.FP32] = pl.tile.reshape(pl.tile.load(hc_base, [0], [8], [8]), [1, 8])
                pre_raw: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = self.hc_projection_piece(partials, mixes_raw, t0, 8)
                pre_scaled: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.muls(pl.tile.row_expand_mul(pre_raw, inv_col), scale0)
                pre_logits: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.add(pre_scaled, pl.tile.col_expand(pre_scaled, pre_base))
                pre_sig: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.recip(pl.tile.adds(pl.tile.exp(pl.tile.neg(pre_logits)), 1.0))
                pre_val: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.adds(pre_sig, 9.9999999999999995e-07)
                pl.tile.store(pre_val, [t0, 0], pre_val_store)
                post_base: pl.Tile[[1, 8], pl.FP32] = pl.tile.reshape(pl.tile.load(hc_base, [4], [8], [8]), [1, 8])
                post_raw: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = self.hc_projection_piece__2(partials, mixes_raw, t0, 8)
                post_scaled: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.muls(pl.tile.row_expand_mul(post_raw, inv_col), scale1)
                post_logits: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.add(post_scaled, pl.tile.col_expand(post_scaled, post_base))
                post_sig: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.recip(pl.tile.adds(pl.tile.exp(pl.tile.neg(post_logits)), 1.0))
                post_pad: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 8])] = pl.tile.muls(post_sig, 2.0)
                if valid_rows == 8:
                    post_valid: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4])] = pl.tile.set_validshape(post_pad, 8, 4)
                    pl.tile.store(post_valid, [t0, 0], post)
                else:
                    pl.tile.store(post_pad, [0, 0], post_tail_store)
                    post_tile: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(post_tail_store, [0, 0], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
                    pl.tile.store(post_tile, [t0, 0], post)
        return pre_val_store, post

    @pl.function(type=pl.FunctionType.Inline)
    def rms_norm_apply(self, x: pl.Tensor[[CSA_FULL_P8, RMS_NORM_CHUNK_D_DYN], pl.BF16], norm_w: pl.Tensor[[1, RMS_NORM_CHUNK_D_DYN], pl.BF16], inverse: pl.Tensor[[1, 8], pl.FP32]):
        # Apply inverse RMS and norm weights to one BF16 activation chunk.
        x_fp32: pl.Tensor[[CSA_FULL_P8, RMS_NORM_CHUNK_D_DYN], pl.FP32] = pl.tensor.cast(x, target_type=pl.FP32, mode="round")
        inverse_col: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(inverse, [8, 1])
        norm_w_fp32: pl.Tensor[[1, RMS_NORM_CHUNK_D_DYN], pl.FP32] = pl.tensor.cast(norm_w, target_type=pl.FP32, mode="round")
        scaled: pl.Tensor[[CSA_FULL_P8, RMS_NORM_CHUNK_D_DYN], pl.FP32] = pl.tensor.row_expand_mul(x_fp32, inverse_col)
        normed: pl.Tensor[[CSA_FULL_P8, RMS_NORM_CHUNK_D_DYN], pl.FP32] = pl.tensor.col_expand_mul(scaled, norm_w_fp32)
        result: pl.Tensor[[CSA_FULL_P8, RMS_NORM_CHUNK_D_DYN], pl.BF16] = pl.tensor.cast(normed, target_type=pl.BF16, mode="rint")
        return result

    @pl.function(type=pl.FunctionType.Inline)
    def rms_norm_inverse(self, sq_sum: pl.Tensor[[1, 8], pl.FP32]):
        # Compute inverse RMS from the per-token full-width square sums.
        mean: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.muls(sq_sum, 0.000244140625)
        variance: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.adds(mean, 9.9999999999999995e-07)
        inverse: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.rsqrt(variance, high_precision=True)
        return inverse

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def hc_mix_norm_range(
        self,
        x: pl.Tensor[[CSA_FULL_T, 16384], pl.FP32],
        pre_val_store: pl.Tensor[[CSA_FULL_P16, 8], pl.FP32],
        norm_w: pl.Tensor[[4096], pl.BF16],
        x_mixed: pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16],
        x_normed: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        x_normed_tail_store: pl.Tensor[[8, 4096], pl.BF16],
        vector_worker: pl.Scalar[pl.INDEX],
        vector_workers: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16], pl.Tensor[[CSA_FULL_T, 4096], pl.BF16]]:
        # AIV：四路混合、BF16 舍入和完整 D4096 RMSNorm，原序不变。
        #
        #         调用前等待 HC_GATES 以及本核输入行；返回后发布 HC_NORM。每行始终
        #         由一个 worker 完成全部 D，保留16个K256平方和的顺序，不新增跨核RMS。
        #         norm_w 直接接 Native BF16，不在初始化时转换为 FP32。
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        x_flat: pl.Tensor[[CSA_FULL_T, 16384], pl.FP32] = x
        token_tiles: pl.Scalar[pl.INDEX] = (t_dim + 8 - 1) // 8
        for wave in pl.range((token_tiles + vector_workers - 1) // vector_workers):
            token_block: pl.Scalar[pl.INDEX] = vector_worker + wave * vector_workers
            if token_block < token_tiles:
                t0: pl.Scalar[pl.INDEX] = token_block * 8
                valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, t_dim - t0)
                pre_tile: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.slice(pre_val_store, [8, 8], [t0, 0])
                pre_tile_t: pl.Tensor[[8, 8], pl.FP32, pl.TensorView(stride=[1, 8], layout=pl.TensorLayout.DN)] = pl.tensor.transpose(pre_tile, 0, 1)
                pre0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t, [1, 8], [0, 0]), [8, 1])
                pre1: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t, [1, 8], [1, 0]), [8, 1])
                pre2: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t, [1, 8], [2, 0]), [8, 1])
                pre3: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t, [1, 8], [3, 0]), [8, 1])
                sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
                for mix_db in pl.pipeline(16, stage=2):
                    d0: pl.Scalar[pl.INDEX] = mix_db * 256
                    x0: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                        x_flat, [8, 256], [t0, d0], [valid_rows, 256]
                    )
                    x1: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                        x_flat, [8, 256], [t0, 4096 + d0], [valid_rows, 256]
                    )
                    x2: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                        x_flat, [8, 256], [t0, 8192 + d0], [valid_rows, 256]
                    )
                    x3: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                        x_flat, [8, 256], [t0, 12288 + d0], [valid_rows, 256]
                    )
                    y0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x0, pre0)
                    y1: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x1, pre1)
                    y2: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x2, pre2)
                    y3: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x3, pre3)
                    y01: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.add(y0, y1)
                    y23: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.add(y2, y3)
                    y: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.add(y01, y23)
                    y_bf16: pl.Tensor[[8, 256], pl.BF16] = pl.tensor.cast(y, target_type=pl.BF16, mode="rint")
                    x_mixed: pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16] = pl.tensor.assemble(x_mixed, y_bf16, [t0, d0])
                    y_rounded: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.cast(y_bf16, target_type=pl.FP32, mode="round")
                    y_sq: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.mul(y_rounded, y_rounded)
                    y_sq_sum: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.row_sum(y_sq)
                    y_sq_row: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(y_sq_sum, [1, 8])
                    sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(sq_sum, y_sq_row)
                norm_sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.create([1, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                norm_sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.assemble(norm_sq_sum, sq_sum, [0, 0])
                y_inv: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.create([1, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                inverse_result: pl.Tensor[[1, 8], pl.FP32] = self.rms_norm_inverse(norm_sq_sum)
                y_inv: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.assemble(y_inv, inverse_result, [0, 0])
                for norm_db in pl.pipeline(16, stage=2):
                    d0: pl.Scalar[pl.INDEX] = norm_db * 256
                    mixed_input: pl.Tensor[[8, 256], pl.BF16] = pl.tensor.slice(x_mixed, [8, 256], [t0, d0])
                    norm_w_input: pl.Tensor[[256], pl.BF16] = pl.tensor.slice(norm_w, [256], [d0])
                    norm_w_row: pl.Tensor[[1, 256], pl.BF16] = pl.tensor.reshape(norm_w_input, [1, 256])
                    normed_bf16: pl.Tensor[[8, 256], pl.BF16] = self.rms_norm_apply(mixed_input, norm_w_row, y_inv)
                    if valid_rows == 8:
                        x_normed: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16] = pl.tensor.assemble(x_normed, normed_bf16, [t0, d0])
                    else:
                        x_normed_tail_store: pl.Tensor[[8, 4096], pl.BF16] = pl.tensor.assemble(x_normed_tail_store, normed_bf16, [0, d0])
                        normed_valid: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 256])] = pl.tile.load(
                            x_normed_tail_store, [0, d0], [8, 256], [valid_rows, 256], target_memory=pl.Mem.Vec
                        )
                        pl.tile.store(normed_valid, [t0, d0], x_normed)
        return x_mixed, x_normed

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def hc_reduce_range(
        self, mixes_partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32], mixes_raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32], vector_worker: pl.Scalar[pl.INDEX], vector_workers: pl.Scalar[pl.INDEX]
    ) -> pl.Tensor[[CSA_FULL_P16, 32], pl.FP32]:
        # AIV：原四份投影按 ((p0+p1)+p2)+p3 顺序归约。
        #
        #         调用前等待所需四段 HC_PARTIAL；返回后发布 HC_RAW。按完整 M16
        #         归约含 padding 的输出行，不把四份左折叠改成树形 col_sum。
        t_linear: pl.Scalar[pl.INDEX] = pl.tensor.dim(mixes_raw, 0)
        linear_blocks: pl.Scalar[pl.INDEX] = t_linear // 16
        for wave in pl.range((linear_blocks + vector_workers - 1) // vector_workers):
            linear_block: pl.Scalar[pl.INDEX] = vector_worker + wave * vector_workers
            if linear_block < linear_blocks:
                linear_t0: pl.Scalar[pl.INDEX] = linear_block * 16
                mixes_total: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.slice(mixes_partials, [16, 32], [linear_t0, 0])
                for linear_split in pl.range(1, 4):
                    partial_t0: pl.Scalar[pl.INDEX] = linear_split * t_linear + linear_t0
                    partial_tile: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.slice(mixes_partials, [16, 32], [partial_t0, 0])
                    mixes_total: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.add(mixes_total, partial_tile)
                mixes_raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32] = pl.tensor.assemble(mixes_raw, mixes_total, [linear_t0, 0])
        return mixes_raw

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def hc_widen_rms_range(
        self,
        x: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16],
        x_fp32: pl.Tensor[[CSA_FULL_T, 16384], pl.FP32],
        inv_rms: pl.Tensor[[CSA_FULL_P16, 1], pl.FP32],
        widen_tail: pl.Tensor[[8, 16384], pl.FP32],
        vector_worker: pl.Scalar[pl.INDEX],
        vector_workers: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_T, 16384], pl.FP32], pl.Tensor[[CSA_FULL_P16, 1], pl.FP32]]:
        # AIV：BF16 无损扩展并按原 32 个 K512 顺序累加输入平方和。
        #
        #         调用方在本核返回后发布 HC_INPUT；AIC 投影及后续门值消费者等待所需行。
        #         尾行显式恢复 valid_shape 并清零后归约，避免 cast 丢失有效行标记。
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        x_hc_flat: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16] = x
        x_hc32_flat: pl.Tensor[[CSA_FULL_T, 16384], pl.FP32] = x_fp32
        widen_rows: pl.Scalar[pl.INDEX] = (t_dim + 8 - 1) // 8
        for wave in pl.range((widen_rows + vector_workers - 1) // vector_workers):
            widen_blk: pl.Scalar[pl.INDEX] = vector_worker + wave * vector_workers
            if widen_blk < widen_rows:
                w_t0: pl.Scalar[pl.INDEX] = widen_blk * 8
                w_rows: pl.Scalar[pl.INDEX] = pl.min(8, t_dim - w_t0)
                w_sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
                for w_db in pl.pipeline(32, stage=4):
                    w_d0: pl.Scalar[pl.INDEX] = w_db * 512
                    w_src: pl.Tensor[[8, 512], pl.BF16, pl.TensorView(valid_shape=[w_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(
                        x_hc_flat, [8, 512], [w_t0, w_d0], [w_rows, 512]
                    )
                    w_val: pl.Tensor[[8, 512], pl.FP32, pl.TensorView(valid_shape=[w_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.cast(w_src, target_type=pl.FP32, mode="round")
                    if w_rows == 8:
                        x_hc32_flat: pl.Tensor[[CSA_FULL_T, 16384], pl.FP32] = pl.tensor.assemble(x_hc32_flat, w_val, [w_t0, w_d0])
                        w_sq: pl.Tensor[[8, 512], pl.FP32, pl.TensorView(valid_shape=[w_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.mul(w_val, w_val)
                        w_sq_row: pl.Tensor[[1, 8], pl.FP32, pl.TensorView(valid_shape=[1, w_rows], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.reshape(pl.tensor.row_sum(w_sq), [1, 8])
                        w_sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(w_sq_sum, w_sq_row)
                    else:
                        w_valid: pl.Tensor[[8, 512], pl.FP32, pl.TensorView(valid_shape=[w_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.set_validshape(w_val, w_rows, 512)
                        w_clean: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.fillpad(w_valid, pad_value=pl.PadValue.zero)
                        w_sq_tail: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.mul(w_clean, w_clean)
                        w_sq_row_tail: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(w_sq_tail), [1, 8])
                        w_sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(w_sq_sum, w_sq_row_tail)
                        widen_tail: pl.Tensor[[8, 16384], pl.FP32] = pl.tensor.assemble(widen_tail, w_val, [0, w_d0])
                        w_out: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[w_rows, 512])] = pl.tile.load(widen_tail, [0, w_d0], [8, 512], [w_rows, 512], target_memory=pl.Mem.Vec)
                        pl.tile.store(w_out, [w_t0, w_d0], x_hc32_flat)
                w_mean: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.adds(pl.tensor.muls(w_sq_sum, 6.103515625e-05), 9.9999999999999995e-07)
                w_inv: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.rsqrt(w_mean, high_precision=True), [8, 1])
                inv_rms: pl.Tensor[[CSA_FULL_P16, 1], pl.FP32] = pl.tensor.assemble(inv_rms, w_inv, [w_t0, 0])
        return x_fp32, inv_rms

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_compact_quant_cache_stage(
        self,
        values: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        offsets: pl.Tensor[[CSA_FULL_B], pl.INT32],
        slots: pl.Tensor[[CSA_FULL_ISLOTS, 2], pl.INT32],
        scale_values: pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32],
        kv_debug: pl.Tensor[[CSA_FULL_T, 128], pl.FP32],
        cache: pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32], pl.Tensor[[CSA_FULL_T, 128], pl.FP32], pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8]]:
        # AIV按请求量化；只把causal且slot合法的关闭行写到Native key。
        for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), 48):
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
            begin: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            end: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
            if pl.cast(active, pl.INDEX) != 0:
                length: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX)
                history_start: pl.Scalar[pl.INDEX] = length - (end - begin)
                count: pl.Scalar[pl.INDEX] = length // 4 - history_start // 4
                first_token: pl.Scalar[pl.INDEX] = begin + 4 - 1 - history_start % 4
                prefix: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(offsets, [request]), pl.INDEX) + history_start // 4 + 1
                if prefix >= 0 and prefix + count <= pl.tensor.dim(values, 0):
                    if prefix + count <= pl.tensor.dim(scale_values, 0) and prefix + count <= pl.tensor.dim(slots, 0):
                        for local_first in pl.range(0, count, 16):
                            first: pl.Scalar[pl.INDEX] = prefix + local_first
                            rows: pl.Scalar[pl.INDEX] = pl.min(16, count - local_first)
                            raw: pl.Tile[[16, 128], pl.FP32, pl.TileView(valid_shape=[rows, 128])] = pl.tile.load(values, [first, 0], [16, 128], [rows, 128])
                            kv_f32: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 128])] = pl.tile.cast(
                                pl.tile.cast(raw, target_type=pl.BF16, mode="rint"), target_type=pl.FP32, mode="round"
                            )
                            kv_f32_v1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 128])] = pl.tile.muls(kv_f32, 0.088388347648318447)
                            reduce_tmp: pl.Tile[[16, 128], pl.FP32] = pl.tile.create([16, 128], dtype=pl.FP32)
                            amax: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, rows])] = pl.tile.reshape(pl.tile.row_max(pl.tile.abs(kv_f32_v1), reduce_tmp), [1, 16])
                            floor: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, rows])] = pl.tile.set_validshape(pl.tile.full([1, 16], dtype=pl.FP32, value=0.0001), 1, rows)
                            numerator: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, rows])] = pl.tile.set_validshape(pl.tile.full([1, 16], dtype=pl.FP32, value=127.0), 1, rows)
                            amax_v1: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, rows])] = pl.tile.maximum(amax, floor)
                            scale_q_row: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, rows])] = pl.tile.div(numerator, amax_v1)
                            scale_dq_col: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 1])] = pl.tile.reshape(pl.tile.recip(scale_q_row), [16, 1])
                            scale_q_col: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 1])] = pl.tile.reshape(scale_q_row, [16, 1])
                            pl.tile.store(pl.tile.set_validshape(scale_dq_col, rows, 1), [first, 0], scale_values)
                            scaled: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 128])] = pl.tile.row_expand_mul(kv_f32_v1, scale_q_col)
                            quant_i32: pl.Tile[[16, 128], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 128])] = pl.tile.cast(scaled, target_type=pl.INT32, mode="rint")
                            quant_half: pl.Tile[[16, 128], pl.FP16, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 128])] = pl.tile.cast(quant_i32, target_type=pl.FP16, mode="round")
                            quant_i8: pl.Tile[[16, 128], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 128])] = pl.tile.cast(quant_half, target_type=pl.INT8, mode="trunc")
                            for inner in pl.range(rows):
                                token: pl.Scalar[pl.INDEX] = first_token + (local_first + inner) * 4
                                position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [token]), pl.INDEX)
                                if position == history_start + token - begin:
                                    page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(slots, [first + inner, 0]), pl.INDEX)
                                    offset: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(slots, [first + inner, 1]), pl.INDEX)
                                    if page >= 0 and page < pl.tensor.dim(cache, 0) and offset >= 0 and offset < 32:
                                        if pl.tensor.dim(cache, 1) >= 4160:
                                            debug_row: pl.Tile[
                                                [1, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(pl.max(rows - inner, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)), 128])
                                            ] = pl.tile.slice(kv_f32_v1, [1, 128], [inner, 0], [1, 128])
                                            key_row: pl.Tile[
                                                [1, 128], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(pl.max(rows - inner, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)), 128])
                                            ] = pl.tile.slice(quant_i8, [1, 128], [inner, 0], [1, 128])
                                            pl.tile.store(debug_row, [token, 0], kv_debug)
                                            pl.tile.store(key_row, [page, offset * 128], cache)
        return scale_values, kv_debug, cache

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_compressor_norm_rope_stage(
        self,
        pooled: pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32],
        norm_w: pl.Tensor[[128], pl.BF16],
        cos: pl.Tensor[[CSA_FULL_IC, 64], pl.FP32],
        sin: pl.Tensor[[CSA_FULL_IC, 64], pl.FP32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        offsets: pl.Tensor[[CSA_FULL_B], pl.INT32],
        normed: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16]:
        # AIV请求所有者先初始化其真实compact行，再消费本请求已经写好的pooled。
        for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), 48):
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
            begin: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            end: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
            if pl.cast(active, pl.INDEX) != 0:
                length: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX)
                history_start: pl.Scalar[pl.INDEX] = length - (end - begin)
                count: pl.Scalar[pl.INDEX] = length // 4 - history_start // 4
                offset: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(offsets, [request]), pl.INDEX)
                prefix: pl.Scalar[pl.INDEX] = offset + history_start // 4 + 1
                if count > 0 and prefix >= 0 and prefix + count <= pl.tensor.dim(normed, 0):
                    for compact_row in pl.range(prefix, prefix + count):
                        pl.tile.store(pl.tile.full([1, 128], dtype=pl.BF16, value=0.0), [compact_row, 0], normed)
                    if end <= pl.tensor.dim(pooled, 0):
                        rope_ones: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
                        rope_index: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                        rope_index_f: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(rope_index, target_type=pl.FP32, mode="round")
                        rope_col: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(rope_ones, rope_index_f)
                        rope_dup_f: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(
                            pl.tile.cast(pl.tile.muls(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32, mode="round"
                        )
                        rope_lane: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(rope_col, pl.tile.muls(rope_dup_f, 2.0))
                        rope_swap_col: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(rope_col, 1.0), pl.tile.muls(rope_lane, 2.0))
                        row_seed: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(
                            pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 16], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode="round"), 64.0
                        )
                        row_grid: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 16], dtype=pl.FP32, value=1.0), row_seed)
                        row_offset: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(row_grid, 0, 1)
                        rope_swap_idx: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(rope_swap_col, row_offset), target_type=pl.INT32, mode="round")
                        reduce_tmp: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        gather_tmp: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                        gamma_input: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(norm_w, [0], [64], [64], target_memory=pl.Mem.Vec)
                        gamma: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.cast(gamma_input, target_type=pl.FP32, mode="round"), [1, 64])
                        gamma_rope_input: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(norm_w, [64], [64], [64], target_memory=pl.Mem.Vec)
                        gamma_rope: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.cast(gamma_rope_input, target_type=pl.FP32, mode="round"), [1, 64])
                        for b0 in pl.range(begin, end, 16):
                            rows: pl.Scalar[pl.INDEX] = pl.min(16, end - b0)
                            _tuple_tmp_1: pl.Tuple[pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]] = self.load_request_compact_rope_rows(
                                cos, sin, positions, bounds, lengths, offsets, request, b0, rows
                            )
                            cos_b: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = _tuple_tmp_1[0]
                            sin_b: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = _tuple_tmp_1[1]
                            kv_rms_low: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[rows, 64])] = pl.tile.load(pooled, [b0, 0], [16, 64], [rows, 64])
                            kv_rms_high: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[rows, 64])] = pl.tile.load(pooled, [b0, 64], [16, 64], [rows, 64])
                            folded_sq: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.add(
                                pl.tile.mul(kv_rms_low, kv_rms_low), pl.tile.mul(kv_rms_high, kv_rms_high)
                            )
                            square_sum: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 1])] = pl.tile.row_sum(folded_sq, reduce_tmp)
                            variance: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 1])] = pl.tile.adds(pl.tile.muls(square_sum, 0.0078125), 9.9999999999999995e-07)
                            inv_rms: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 1])] = pl.tile.recip(pl.tile.sqrt(variance))
                            normed_chunk: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.col_expand_mul(pl.tile.row_expand_mul(kv_rms_low, inv_rms), gamma)
                            normed_nope: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.cast(normed_chunk, target_type=pl.BF16, mode="rint")
                            rope_normed: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.col_expand_mul(pl.tile.row_expand_mul(kv_rms_high, inv_rms), gamma_rope)
                            swapped: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(rope_normed, rope_swap_idx, gather_tmp)
                            sin_signed: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(sin_b, pl.tile.subs(pl.tile.muls(rope_lane, 2.0), 1.0))
                            rope_rot: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.add(pl.tile.mul(rope_normed, cos_b), pl.tile.mul(swapped, sin_signed))
                            normed_rope: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 64])] = pl.tile.cast(rope_rot, target_type=pl.BF16, mode="rint")
                            for inner in pl.range(rows):
                                token: pl.Scalar[pl.INDEX] = b0 + inner
                                position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [token]), pl.INDEX)
                                expected: pl.Scalar[pl.INDEX] = history_start + token - begin
                                if position == expected and (position + 1) % 4 == 0:
                                    compact_row: pl.Scalar[pl.INDEX] = offset + (position + 1) // 4
                                    if compact_row >= prefix and compact_row < prefix + count:
                                        if compact_row < pl.tensor.dim(cos, 0) and compact_row < pl.tensor.dim(sin, 0):
                                            nope_row: pl.Tile[
                                                [1, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(pl.max(rows - inner, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)), 64])
                                            ] = pl.tile.slice(normed_nope, [1, 64], [inner, 0], [1, 64])
                                            rope_row: pl.Tile[
                                                [1, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(pl.max(rows - inner, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)), 64])
                                            ] = pl.tile.slice(normed_rope, [1, 64], [inner, 0], [1, 64])
                                            pl.tile.store(nope_row, [compact_row, 0], normed)
                                            pl.tile.store(rope_row, [compact_row, 64], normed)
        return normed

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_compressor_pool_stage(
        self,
        kv_proj: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        score_proj: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        state: pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32],
        state_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_ITABLE_COLS], pl.INT32],
        ape: pl.Tensor[[4, 256], pl.FP32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        pooled: pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32]:
        # AIV Indexer 压缩：补齐 Q/L 活动判定，保留两个 64 列窗口的数值顺序。
        for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), 48):
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
            begin: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            end: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
            if pl.cast(active, pl.INDEX) != 0:
                first_position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX) - (end - begin)
                for token in pl.range(begin, end):
                    pl.tile.store(pl.tile.full([1, 128], dtype=pl.FP32, value=0.0), [token, 0], pooled)
                    position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [token]), pl.INDEX)
                    if position == first_position + token - begin and (position + 1) % 4 == 0:
                        window_start: pl.Scalar[pl.INDEX] = position - 8 + 1
                        for h0 in pl.range(0, 128, 64):
                            last_ape: pl.Scalar[pl.INDEX] = position % 4
                            mi: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(
                                pl.tile.load(score_proj, [token, 128 + h0], [1, 64], [1, 64]), pl.tile.load(ape, [last_ape, 128 + h0], [1, 64], [1, 64])
                            )
                            li: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(pl.tile.sub(mi, mi))
                            oi: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(kv_proj, [token, 128 + h0], [1, 64], [1, 64])
                            for state_index in pl.range(7):
                                logical_position: pl.Scalar[pl.INDEX] = window_start + state_index
                                value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
                                score: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=-3.4028234663852886e38)
                                half: pl.Scalar[pl.INDEX] = 0
                                if state_index >= 4:
                                    half: pl.Scalar[pl.INDEX] = 128
                                if logical_position >= 0 and logical_position < first_position:
                                    score: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
                                    table_column: pl.Scalar[pl.INDEX] = logical_position // 2
                                    if table_column < pl.tensor.dim(state_table, 1):
                                        page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(state_table, [request, table_column]), pl.INDEX)
                                        column: pl.Scalar[pl.INDEX] = logical_position % 2 * 512 + half + h0
                                        if page >= 0 and page < pl.tensor.dim(state, 0):
                                            if column + 256 + 64 <= pl.tensor.dim(state, 1):
                                                value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(state, [page, column], [1, 64], [1, 64])
                                                score: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(state, [page, column + 256], [1, 64], [1, 64])
                                if logical_position >= first_position and logical_position <= position:
                                    overlay_token: pl.Scalar[pl.INDEX] = begin + logical_position - first_position
                                    ape_row: pl.Scalar[pl.INDEX] = logical_position % 4
                                    value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(kv_proj, [overlay_token, half + h0], [1, 64], [1, 64])
                                    score: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(
                                        pl.tile.load(score_proj, [overlay_token, half + h0], [1, 64], [1, 64]), pl.tile.load(ape, [ape_row, half + h0], [1, 64], [1, 64])
                                    )
                                mi_next: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(mi, score)
                                alpha: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(pl.tile.sub(mi, mi_next))
                                beta: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(pl.tile.sub(score, mi_next))
                                li: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(pl.tile.mul(alpha, li), beta)
                                oi: pl.Tile[[1, 64], pl.FP32] = pl.tile.add(pl.tile.mul(oi, alpha), pl.tile.mul(value, beta))
                                mi: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = mi_next
                            pl.tile.store(pl.tile.div(oi, li), [token, h0], pooled)
        return pooled

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_compressor_state_commit_stage(
        self,
        kv_proj: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        score_proj: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        ape: pl.Tensor[[4, 256], pl.FP32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        state_slots: pl.Tensor[[CSA_FULL_T, 2], pl.INT32],
        state: pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32]:
        # AIV Indexer state 写回；使用本组 Native slots，不能与主 state 混用。
        for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), 48):
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
            begin: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            end: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
            if pl.cast(active, pl.INDEX) != 0:
                first_position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX) - (end - begin)
                for token in pl.range(begin, end):
                    position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [token]), pl.INDEX)
                    if position == first_position + token - begin:
                        page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(state_slots, [token, 0]), pl.INDEX)
                        offset: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(state_slots, [token, 1]), pl.INDEX)
                        if page >= 0 and page < pl.tensor.dim(state, 0) and offset >= 0 and offset < 2:
                            column: pl.Scalar[pl.INDEX] = offset * 512
                            if column + 512 <= pl.tensor.dim(state, 1):
                                ape_row: pl.Scalar[pl.INDEX] = position % 4
                                value: pl.Tile[[1, 256], pl.FP32] = pl.tile.load(kv_proj, [token, 0], [1, 256], [1, 256])
                                score: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.add(
                                    pl.tile.load(score_proj, [token, 0], [1, 256], [1, 256]), pl.tile.load(ape, [ape_row, 0], [1, 256], [1, 256])
                                )
                                pl.tile.store(value, [page, column], state)
                                pl.tile.store(score, [page, column + 256], state)
        return state

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_long_leaf_plan(self, max_cache_count: pl.Scalar[pl.INDEX], query_groups: pl.Scalar[pl.INDEX]):
        # 沿用原1024粒度平衡公式；组数来自实际请求，不按固定S反算。
        tile_count: pl.Scalar[pl.INDEX] = (max_cache_count + 1024 - 1) // 1024
        leaf_count: pl.Scalar[pl.INDEX] = pl.max((max_cache_count + 8192 - 1) // 8192, 1)
        leaf_tiles: pl.Scalar[pl.INDEX] = 8
        extra_leaves: pl.Scalar[pl.INDEX] = 0
        leaf_multiple: pl.Scalar[pl.INDEX] = 1
        if 3 * query_groups == 48:
            leaf_multiple: pl.Scalar[pl.INDEX] = 3
        else:
            if 3 * query_groups == 24:
                leaf_multiple: pl.Scalar[pl.INDEX] = 3
            else:
                if 6 * query_groups == 24:
                    leaf_multiple: pl.Scalar[pl.INDEX] = 6
        if leaf_multiple > 1:
            balanced_count: pl.Scalar[pl.INDEX] = (leaf_count + leaf_multiple - 1) // leaf_multiple * leaf_multiple
            if balanced_count <= 32:
                leaf_count: pl.Scalar[pl.INDEX] = balanced_count
                leaf_tiles: pl.Scalar[pl.INDEX] = pl.max(tile_count // balanced_count, 1)
                extra_leaves: pl.Scalar[pl.INDEX] = tile_count % balanced_count
        return leaf_count, leaf_tiles, extra_leaves

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_group_plan_stage(
        self,
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        offsets: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32],
        plan: pl.Tensor[[8], pl.INT32],
        tokens: pl.Scalar[pl.INDEX],
        worker: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32], pl.Tensor[[8], pl.INT32]]:
        # worker0写组前缀、描述符和叶计划；容量pad16(B+1)+4*T由入口保证。
        if worker == 0:
            groups: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
            max_history: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
            requests: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
            descriptor_base: pl.Scalar[pl.INDEX] = (requests + 1 + 16 - 1) // 16 * 16
            for request in pl.range(requests):
                pl.tensor.write(offsets, [request], pl.cast(groups, pl.INT32))
                _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.request_bounds(bounds, lengths, request, tokens)
                begin: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
                end: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
                active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
                if pl.cast(active, pl.INDEX) != 0:
                    length: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX)
                    max_history: pl.Scalar[pl.INDEX] = pl.max(max_history, pl.min(length // 4, 262144))
                    group_count: pl.Scalar[pl.INDEX] = (end - begin + 6 - 1) // 6
                    history: pl.Scalar[pl.INDEX] = length - (end - begin)
                    for local_group in pl.range(group_count):
                        first: pl.Scalar[pl.INDEX] = begin + local_group * 6
                        live: pl.Scalar[pl.INDEX] = pl.min(6, end - first)
                        visible: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
                        for lane in pl.range(live):
                            query: pl.Scalar[pl.INDEX] = first + lane
                            position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [query]), pl.INDEX)
                            if position == history + query - begin:
                                lane_visible: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(length // 4, (position + 1) // 4), 262144), 0)
                                visible: pl.Scalar[pl.INDEX] = pl.max(visible, lane_visible)
                        row: pl.Scalar[pl.INDEX] = descriptor_base + (groups + local_group) * 4
                        pl.tensor.write(offsets, [row], pl.cast(request, pl.INT32))
                        pl.tensor.write(offsets, [row + 1], pl.cast(first, pl.INT32))
                        pl.tensor.write(offsets, [row + 2], pl.cast(live, pl.INT32))
                        pl.tensor.write(offsets, [row + 3], pl.cast(visible, pl.INT32))
                    groups: pl.Scalar[pl.INDEX] = groups + group_count
            pl.tensor.write(offsets, [requests], pl.cast(groups, pl.INT32))
            leaves: pl.Scalar[pl.INDEX] = pl.max((max_history + 8192 - 1) // 8192, 1)
            leaf_tiles: pl.Scalar[pl.INDEX] = pl.cast(8, pl.INDEX)
            extra: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
            if True:
                _tuple_tmp_1: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]] = self.indexer_long_leaf_plan(max_history, groups)
                leaves: pl.Scalar[pl.INDEX] = _tuple_tmp_1[0]
                leaf_tiles: pl.Scalar[pl.INDEX] = _tuple_tmp_1[1]
                extra: pl.Scalar[pl.INDEX] = _tuple_tmp_1[2]
            pl.tensor.write(plan, [0], pl.cast(groups, pl.INT32))
            pl.tensor.write(plan, [1], pl.cast(max_history, pl.INT32))
            pl.tensor.write(plan, [2], pl.cast(leaves, pl.INT32))
            pl.tensor.write(plan, [3], pl.cast(leaf_tiles, pl.INT32))
            pl.tensor.write(plan, [4], pl.cast(extra, pl.INT32))
            pl.tensor.write(plan, [5], pl.cast(6, pl.INT32))
            pl.tensor.write(plan, [6], pl.cast(True, pl.INT32))
            pl.tensor.write(plan, [7], pl.cast(1, pl.INT32))
        return offsets, plan

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_query_visibility(
        self,
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        request: pl.Scalar[pl.INDEX],
        query: pl.Scalar[pl.INDEX],
    ):
        # 返回visible/causal；零历史的合法query仍causal=1，不能和dummy混淆。
        visible: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        causal: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
        _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
        begin: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
        end: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
        active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
        if pl.cast(active, pl.INDEX) != 0 and query >= begin and query < end:
            length: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX)
            position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [query]), pl.INDEX)
            if position == length - (end - begin) + query - begin:
                causal: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
                visible: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(length // 4, (position + 1) // 4), 262144), 0)
        return visible, causal

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_head_coefficients_stage(
        self,
        scales: pl.Tensor[[CSA_FULL_IDX_SCALE_T_ROWS, 64], pl.FP32],
        weights: pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        offsets: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32],
        coefficients: pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16],
        worker: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16]:
        # AIV单写者生成每组完整对角系数块；输入scale/weight须已发布且覆盖T。
        requests: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
        groups: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(offsets, [requests]), pl.INDEX)
        for group in pl.range(worker, groups, 48):
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]] = self.indexer_decode_group(bounds, lengths, positions, offsets, group)
            request: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            first: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            live: pl.Scalar[pl.INDEX] = _tuple_tmp[2]
            _visible: pl.Scalar[pl.INDEX] = _tuple_tmp[3]
            coefficient_rows: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.full([96, 64], dtype=pl.FP16, value=0.0)
            for lane in pl.unroll(6):
                if lane < live:
                    _tuple_tmp_1: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.indexer_query_visibility(bounds, lengths, positions, request, first + lane)
                    _visible: pl.Scalar[pl.INDEX] = _tuple_tmp_1[0]
                    causal: pl.Scalar[pl.INT32] = _tuple_tmp_1[1]
                    if pl.cast(causal, pl.INDEX) != 0:
                        query_scales: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(scales, [first + lane, 0], [1, 64], [1, 64])
                        query_weights: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(weights, [first + lane, 0], [1, 64], [1, 64])
                        query_scale_half: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.cast(query_scales, target_type=pl.FP16, mode="rint")
                        query_weight_half: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.cast(query_weights, target_type=pl.FP16, mode="rint")
                        head_coefficients: pl.Tile[[1, 64], pl.FP16, pl.Mem.Vec] = pl.tile.mul(query_scale_half, query_weight_half)
                        coefficient_rows: pl.Tile[[96, 64], pl.FP16, pl.Mem.Vec] = pl.tile.assemble(coefficient_rows, head_coefficients, [lane * 7, 0])
            coefficient_matrix: pl.Tile[[16, 384], pl.FP16, pl.Mem.Vec] = pl.tile.reshape(coefficient_rows, [16, 384])
            coefficients: pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16] = pl.tile.store(coefficient_matrix, [group * 16, 0], coefficients)
        return coefficients

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_visible_root_count(self, visible: pl.Scalar[pl.INDEX], plan: pl.Tensor[[8], pl.INT32]):
        # 返回本query连续有效根数；同源计划不匹配或不可安全读取时返回0。
        roots: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        leaves: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [2]), pl.INDEX)
        leaf_tiles: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [3]), pl.INDEX)
        extra_leaves: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [4]), pl.INDEX)
        max_history: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [1]), pl.INDEX)
        if (
            pl.const(1, pl.INDEX) >= pl.const(1, pl.INDEX)
            and pl.const(1, pl.INDEX) <= pl.const(2, pl.INDEX)
            and pl.cast(pl.tensor.read(plan, [7]), pl.INDEX) == 1
            and pl.tensor.read(plan, [6]) == pl.cast(True, pl.INT32)
        ):
            if visible > 0 and visible <= max_history and max_history <= 262144:
                if leaves > 0 and leaves <= 32 and leaf_tiles > 0:
                    if extra_leaves >= 0 and extra_leaves < leaves:
                        leaf_count: pl.Scalar[pl.INDEX] = (visible + 8192 - 1) // 8192
                        if True:
                            visible_tiles: pl.Scalar[pl.INDEX] = (visible + 1024 - 1) // 1024
                            larger_leaf_tiles: pl.Scalar[pl.INDEX] = extra_leaves * (leaf_tiles + 1)
                            larger_leaf_count: pl.Scalar[pl.INDEX] = pl.min(extra_leaves, (visible_tiles + leaf_tiles) // (leaf_tiles + 1))
                            remaining_tiles: pl.Scalar[pl.INDEX] = pl.max(visible_tiles - larger_leaf_tiles, 0)
                            leaf_count: pl.Scalar[pl.INDEX] = larger_leaf_count + (remaining_tiles + leaf_tiles - 1) // leaf_tiles
                            if leaf_count > 0 and leaf_count <= leaves:
                                roots: pl.Scalar[pl.INDEX] = leaf_count * 1
                        else:
                            if leaf_tiles == 8 and extra_leaves == 0:
                                if leaf_count > 0 and leaf_count <= leaves:
                                    roots: pl.Scalar[pl.INDEX] = leaf_count * 1
        return roots

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def merge2_top512_pairs(self, pair_arena: pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32], left_slot: pl.Scalar[pl.INDEX], right_slot: pl.Scalar[pl.INDEX], output_slot: pl.Scalar[pl.INDEX]):
        # 先读左右512根，再按incoming右根优先合并，写到指定GM行。
        left: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [left_slot, 0], [1, 1024], [1, 1024])
        right: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [right_slot, 0], [1, 1024], [1, 1024])
        merge_tmp: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
        merged_all: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(right, left, merge_tmp, exhausted=False)
        merged: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(merged_all, [1, 1024], [0, 0])
        pl.tile.store(merged, [output_slot, 0], pair_arena)

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def merge_top512_roots(self, pair_arena: pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32], arena_base: pl.Scalar[pl.INDEX], half_count: pl.Scalar[pl.INDEX]):
        # 根0起步，每轮合并至多三个新根，始终只把1024元素root带到下一轮。
        root: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [arena_base, 0], [1, 1024], [1, 1024])
        for child in pl.range(1, half_count, 3):
            incoming1: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [arena_base + child, 0], [1, 1024], [1, 1024])
            if child + 2 < half_count:
                incoming2: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [arena_base + child + 1, 0], [1, 1024], [1, 1024])
                incoming3: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [arena_base + child + 2, 0], [1, 1024], [1, 1024])
                merge_tmp4: pl.Tile[[1, 4096], pl.FP32] = pl.tile.create([1, 4096], dtype=pl.FP32)
                merged4: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(incoming3, incoming2, incoming1, root, merge_tmp4, exhausted=True)
                root: pl.Tile[[1, 1024], pl.FP32] = pl.tile.extract(merged4, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
            else:
                if child + 1 < half_count:
                    incoming2: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [arena_base + child + 1, 0], [1, 1024], [1, 1024])
                    merge_tmp3: pl.Tile[[1, 3072], pl.FP32] = pl.tile.create([1, 3072], dtype=pl.FP32)
                    merged3: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(incoming2, incoming1, root, merge_tmp3, exhausted=True)
                    root: pl.Tile[[1, 1024], pl.FP32] = pl.tile.extract(merged3, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                else:
                    merge_tmp2: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                    merged2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(incoming1, root, merge_tmp2, exhausted=False)
                    root: pl.Tile[[1, 1024], pl.FP32] = pl.tile.extract(merged2, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
        return root

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_pair_publish_stage(
        self,
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        plan: pl.Tensor[[8], pl.INT32],
        pair_arena: pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32],
        topk_scores: pl.Tensor[[CSA_FULL_T, 512], pl.FP32],
        topk_indices: pl.Tensor[[CSA_FULL_T, 512], pl.INT32],
        worker: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_T, 512], pl.FP32], pl.Tensor[[CSA_FULL_T, 512], pl.INT32]]:
        # AIV发布pair根；保持非balanced根0原位归并及原多路平局优先顺序。
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(positions, 0)
        for query in pl.range(worker, tokens, 48):
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.query_request(bounds, lengths, query, tokens)
            request: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            _begin: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            _end: pl.Scalar[pl.INDEX] = _tuple_tmp[2]
            active: pl.Scalar[pl.INT32] = _tuple_tmp[3]
            visible_count: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
            if pl.cast(active, pl.INDEX) != 0:
                _tuple_tmp_1: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.indexer_query_visibility(bounds, lengths, positions, request, query)
                visible_count: pl.Scalar[pl.INDEX] = _tuple_tmp_1[0]
                _causal: pl.Scalar[pl.INT32] = _tuple_tmp_1[1]
            root_count: pl.Scalar[pl.INDEX] = self.indexer_visible_root_count(visible_count, plan)
            arena_base: pl.Scalar[pl.INDEX] = query * 64
            if root_count > 0 and root_count <= 64 and arena_base + root_count <= pl.tensor.dim(pair_arena, 0):
                if True:
                    root_pairs: pl.Tile[[1, 1024], pl.FP32] = self.merge_top512_roots(pair_arena, arena_base, root_count)
                else:
                    for child in pl.range(1, root_count):
                        self.merge2_top512_pairs(pair_arena, arena_base, arena_base + child, arena_base)
                    root_slot: pl.Scalar[pl.INDEX] = arena_base
                    root_pairs: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [root_slot, 0], [1, 1024], [1, 1024])
                root_scores: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather_mask(root_pairs, mask_pattern=1, output_dtype=pl.FP32)
                pl.tile.store(root_scores, [query, 0], topk_scores)
                root_indices: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.gather_mask(root_pairs, mask_pattern=2, output_dtype=pl.INT32)
                if visible_count >= 512:
                    pl.tile.store(root_indices, [query, 0], topk_indices)
                else:
                    output_indices: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                    for lane in pl.range(visible_count):
                        pl.tile.write(output_indices, [0, lane], pl.tile.read(root_indices, [0, lane]))
                    pl.tile.store(output_indices, [query, 0], topk_indices)
            else:
                pl.tile.store(pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e38), [query, 0], topk_scores)
                pl.tile.store(pl.tile.full([1, 512], dtype=pl.INT32, value=-1), [query, 0], topk_indices)
        return topk_scores, topk_indices

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_query_dequant_rope_stage(
        self,
        qr_scale: pl.Tensor[[CSA_FULL_T, 1], pl.FP32],
        wq_b_scale: pl.Tensor[[8192], pl.FP32],
        cos: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        sin: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        qr_acc_pad: pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32],
        qr_bf16_2d: pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16]:
        # AIV：等query INT32投影与QR scale，原反量化/RoPE后写BF16，外层发布供Hadamard消费。
        bs: pl.Scalar[pl.INDEX] = pl.tensor.dim(qr_scale, 0)
        dq_rope_units: pl.Scalar[pl.INDEX] = (bs + 8 - 1) // 8 * 16
        dq_rope_workers: pl.Scalar[pl.INDEX] = pl.min(dq_rope_units, 48)
        if worker_id < dq_rope_workers:
            dq_rope_worker: pl.Scalar[pl.INDEX] = worker_id
            sw_ones: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.full([8, 64], dtype=pl.FP32, value=1.0)
            sw_index: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode="round")
            sw_col: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.col_expand_mul(sw_ones, sw_index)
            sw_dup_f: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.muls(sw_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32, mode="round")
            sw_lane: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(sw_col, pl.tensor.muls(sw_dup_f, 2.0))
            rope_swap_idx: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(pl.tensor.sub(pl.tensor.adds(sw_col, 1.0), pl.tensor.muls(sw_lane, 2.0)), target_type=pl.INT32, mode="round")
            for dq_unit in pl.range(dq_rope_worker, dq_rope_units, dq_rope_workers):
                hg: pl.Scalar[pl.INDEX] = dq_unit % 16 * 4
                dq_t0: pl.Scalar[pl.INDEX] = dq_unit // 16 * 8
                if dq_t0 + 8 <= bs:
                    qr_scale_tile: pl.Tile[[8, 1], pl.FP32] = pl.tile.load(qr_scale, [dq_t0, 0], [8, 1], [8, 1])
                    cos_tile: pl.Tile[[8, 64], pl.FP32] = pl.tile.load(cos, [dq_t0, 0], [8, 64], [8, 64])
                    sin_tile: pl.Tile[[8, 64], pl.FP32] = pl.tile.load(sin, [dq_t0, 0], [8, 64], [8, 64])
                    flat_i: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                    flat_tmp: pl.Tile[[1, 64], pl.INT32] = pl.tile.create([1, 64], dtype=pl.INT32)
                    flat_lane: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.rems(flat_i, pl.const(2, pl.INT32), flat_tmp)
                    swap_row: pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.sub(flat_i, pl.tile.muls(flat_lane, pl.const(2, pl.INT32))), pl.const(65, pl.INT32))
                    swap_base: pl.Tile[[8, 64], pl.INT32] = pl.tile.create([8, 64], dtype=pl.INT32)
                    swap_source: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.col_expand(swap_base, swap_row)
                    row_offsets: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec] = pl.tile.muls(pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False), pl.const(128, pl.INT32))
                    flat_swap: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.row_expand_add(swap_source, pl.tile.reshape(row_offsets, [8, 1])), [1, 512])
                    gather_tmp: pl.Tile[[1, 512], pl.INT32] = pl.tile.create([1, 512], dtype=pl.INT32)
                    for h_inner in pl.pipeline(4, stage=2):
                        h0: pl.Scalar[pl.INDEX] = (hg + h_inner) * 128
                        wq_scale: pl.Tile[[1, 128], pl.FP32] = pl.tile.reshape(pl.tile.load(wq_b_scale, [h0], [128], [128]), [1, 128])
                        acc_fp32: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.load(qr_acc_pad, [dq_t0, h0], [8, 128], [8, 128]), target_type=pl.FP32, mode="none")
                        qr_dequant: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.row_expand_mul(acc_fp32, qr_scale_tile), wq_scale)
                        qr_nope_bf16: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(pl.tile.slice(qr_dequant, [8, 64], [0, 0]), target_type=pl.BF16, mode="rint")
                        qr_rope_slice: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(qr_dequant, [8, 64], [0, 64])
                        qr_swapped_flat: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather(pl.tile.reshape(qr_dequant, [1, 1024]), flat_swap, gather_tmp)
                        qr_swapped: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(qr_swapped_flat, [8, 64])
                        rope_rot: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(pl.tile.mul(qr_rope_slice, cos_tile), pl.tile.mul(qr_swapped, sin_tile))
                        rope_bf16: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(rope_rot, target_type=pl.BF16, mode="rint")
                        pl.tile.store(qr_nope_bf16, [dq_t0, h0], qr_bf16_2d)
                        pl.tile.store(rope_bf16, [dq_t0, h0 + 64], qr_bf16_2d)
                else:
                    tail_swap_idx: pl.Tensor[[1, 64], pl.INT32] = pl.tensor.slice(rope_swap_idx, [1, 64], [0, 0])
                    for tail_t0 in pl.range(dq_t0, bs):
                        tail_qr_scale_value: pl.Scalar[pl.FP32] = pl.tensor.read(qr_scale, [tail_t0, 0])
                        tail_cos_tile: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cos, [1, 64], [tail_t0, 0])
                        tail_sin_tile: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(sin, [1, 64], [tail_t0, 0])
                        for tail_h_inner in pl.pipeline(4, stage=2):
                            tail_h0: pl.Scalar[pl.INDEX] = (hg + tail_h_inner) * 128
                            tail_wq_scale: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(wq_b_scale, [128], [tail_h0]), [1, 128])
                            tail_acc_fp32: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.slice(qr_acc_pad, [1, 128], [tail_t0, tail_h0]), target_type=pl.FP32, mode="none")
                            tail_qr_dequant: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.muls(tail_acc_fp32, tail_qr_scale_value), tail_wq_scale)
                            tail_qr_nope_bf16: pl.Tensor[[1, 64], pl.BF16] = pl.tensor.cast(pl.tensor.slice(tail_qr_dequant, [1, 64], [0, 0]), target_type=pl.BF16, mode="rint")
                            tail_qr_rope_slice: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(tail_qr_dequant, [1, 64], [0, 64])
                            tail_qr_swapped: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.gather(tail_qr_rope_slice, tail_swap_idx, dim=-1)
                            tail_rope_rot: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(tail_qr_rope_slice, tail_cos_tile), pl.tensor.mul(tail_qr_swapped, tail_sin_tile))
                            tail_rope_bf16: pl.Tensor[[1, 64], pl.BF16] = pl.tensor.cast(tail_rope_rot, target_type=pl.BF16, mode="rint")
                            qr_bf16_2d: pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16] = pl.tensor.assemble(qr_bf16_2d, tail_qr_nope_bf16, [tail_t0, tail_h0])
                            qr_bf16_2d: pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16] = pl.tensor.assemble(qr_bf16_2d, tail_rope_bf16, [tail_t0, tail_h0 + 64])
        return qr_bf16_2d

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_query_quant_stage(
        self,
        qh_acc_gm: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32],
        qr_hadamard_i8: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8],
        qr_hadamard_scale_dq: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32],
        tokens: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8], pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32]]:
        # AIV：先等Hadamard FP32，逐64head行归一比例并量化；输出INT8与独立FP32 scale。
        bs_heads: pl.Scalar[pl.INDEX] = tokens * 64
        qh_quant_worker: pl.Scalar[pl.INDEX] = worker_id
        for idx in pl.range(qh_quant_worker, bs_heads // 64, 48):
            o0: pl.Scalar[pl.INDEX] = idx * 64
            qh_full_f32: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.slice(qh_acc_gm, [64, 128], [o0, 0])
            qh_full_f32_v1: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.muls(qh_full_f32, 0.088388347648318447)
            qh_amax: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=0.0001)
            for h0 in pl.range(0, 128, 64):
                qh_a_f32: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.slice(qh_full_f32_v1, [64, 64], [0, h0])
                qh_a_neg: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.neg(qh_a_f32)
                qh_a_abs: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.maximum(qh_a_f32, qh_a_neg)
                qh_a_max_col: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.row_max(qh_a_abs)
                qh_a_max: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(qh_a_max_col, [1, 64])
                qh_amax: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.maximum(qh_amax, qh_a_max)
            qh_scale_numerator: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=127.0)
            qh_scale_quant_row: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.div(qh_scale_numerator, qh_amax)
            qh_scale_recip: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.recip(qh_scale_quant_row)
            qh_scale_dq: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.reshape(qh_scale_recip, [64, 1])
            qr_hadamard_scale_dq: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32] = pl.tensor.assemble(qr_hadamard_scale_dq, qh_scale_dq, [o0, 0])
            qh_scale_quant: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.reshape(qh_scale_quant_row, [64, 1])
            qh_q_scaled: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.row_expand_mul(qh_full_f32_v1, qh_scale_quant)
            qh_q_i32: pl.Tensor[[64, 128], pl.INT32] = pl.tensor.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
            qh_q_half: pl.Tensor[[64, 128], pl.FP16] = pl.tensor.cast(qh_q_i32, target_type=pl.FP16, mode="round")
            qh_i8: pl.Tensor[[64, 128], pl.INT8] = pl.tensor.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
            qr_hadamard_i8: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8] = pl.tensor.assemble(qr_hadamard_i8, qh_i8, [o0, 0])
        return qr_hadamard_i8, qr_hadamard_scale_dq

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_scale_plan_base(self, requests: pl.Scalar[pl.INDEX], tokens: pl.Scalar[pl.INDEX]):
        # 与Host容量公式相同的设备地址表达式；不使用有效group数量缩短固定区。
        group_base: pl.Scalar[pl.INDEX] = (requests + 1 + 16 - 1) // 16 * 16
        group_end: pl.Scalar[pl.INDEX] = group_base + 4 * tokens
        return (group_end + 16 - 1) // 16 * 16

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_scale_page_commit_stage(
        self,
        scale_values: pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32],
        workspace: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32],
        cache: pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8],
        tokens: pl.Scalar[pl.INDEX],
        requests: pl.Scalar[pl.INDEX],
        worker: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8]:
        # 每物理页只有一个writer；逐项保持原Scalar FP16转换及64B Tile读改写。
        base: pl.Scalar[pl.INDEX] = self.indexer_scale_plan_base(requests, tokens)
        initial_node: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(workspace, [base + worker]), pl.INDEX)
        for (node,) in pl.while_(init_values=(initial_node,)):
            pl.cond(node >= 0)
            row: pl.Scalar[pl.INDEX] = base + 48 + node * 4
            page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(workspace, [row]), pl.INDEX)
            offset: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(workspace, [row + 1]), pl.INDEX)
            compact_row: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(workspace, [row + 2]), pl.INDEX)
            scale: pl.Scalar[pl.FP16] = pl.cast(pl.tensor.read(scale_values, [compact_row, 0]), pl.FP16)
            scale_bytes: pl.Tile[[1, 64], pl.INT8] = pl.tile.load(cache, [page, 4096], [1, 64], [1, 64])
            scale_half: pl.Tile[[1, 32], pl.FP16] = pl.tile.reinterpret_view(scale_bytes, dtype=pl.FP16)
            pl.tile.write(scale_half, [0, offset], scale)
            updated: pl.Tile[[1, 64], pl.INT8] = pl.tile.reinterpret_view(scale_half, dtype=pl.INT8)
            pl.tile.store(updated, [page, 4096], cache)
            next_node: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(workspace, [row + 3]), pl.INDEX)
            completed_node: pl.Scalar[pl.INDEX] = pl.yield_(next_node)
        return cache

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_scale_page_plan_stage(
        self,
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        compact_offsets: pl.Tensor[[CSA_FULL_B], pl.INT32],
        slots: pl.Tensor[[CSA_FULL_ISLOTS, 2], pl.INT32],
        cache: pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8],
        workspace: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32],
        scale_capacity: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32]:
        # 仅允许前置唯一planner调用；描述符仅覆盖真实合法的关闭行。
        requests: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(positions, 0)
        base: pl.Scalar[pl.INDEX] = self.indexer_scale_plan_base(requests, tokens)
        for owner in pl.range(48):
            pl.tensor.write(workspace, [base + owner], pl.cast(-1, pl.INT32))
        jobs: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        for request in pl.range(requests):
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.request_bounds(bounds, lengths, request, tokens)
            begin: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            end: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            active: pl.Scalar[pl.INT32] = _tuple_tmp[2]
            if pl.cast(active, pl.INDEX) != 0:
                length: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX)
                history: pl.Scalar[pl.INDEX] = length - (end - begin)
                count: pl.Scalar[pl.INDEX] = length // 4 - history // 4
                first_token: pl.Scalar[pl.INDEX] = begin + 3 - history % 4
                prefix: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(compact_offsets, [request]), pl.INDEX) + history // 4 + 1
                if prefix >= 0 and prefix + count <= scale_capacity and prefix + count <= pl.tensor.dim(slots, 0):
                    for local_row in pl.range(count):
                        token: pl.Scalar[pl.INDEX] = first_token + local_row * 4
                        position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(positions, [token]), pl.INDEX)
                        if position == history + token - begin:
                            page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(slots, [prefix + local_row, 0]), pl.INDEX)
                            offset: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(slots, [prefix + local_row, 1]), pl.INDEX)
                            if page >= 0 and page < pl.tensor.dim(cache, 0) and offset >= 0 and offset < 32:
                                if pl.tensor.dim(cache, 1) >= 4160:
                                    owner: pl.Scalar[pl.INDEX] = page % 48
                                    head: pl.Scalar[pl.INT32] = pl.tensor.read(workspace, [base + owner])
                                    row: pl.Scalar[pl.INDEX] = base + 48 + jobs * 4
                                    pl.tensor.write(workspace, [row], pl.cast(page, pl.INT32))
                                    pl.tensor.write(workspace, [row + 1], pl.cast(offset, pl.INT32))
                                    pl.tensor.write(workspace, [row + 2], pl.cast(prefix + local_row, pl.INT32))
                                    pl.tensor.write(workspace, [row + 3], head)
                                    pl.tensor.write(workspace, [base + owner], pl.cast(jobs, pl.INT32))
                                    jobs: pl.Scalar[pl.INDEX] = jobs + 1
        return workspace

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_topk_half_leaf(
        self,
        score_arena: pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32],
        pair_arena: pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32],
        score_row: pl.Scalar[pl.INDEX],
        logical_begin: pl.Scalar[pl.INDEX],
        valid_count: pl.Scalar[pl.INDEX],
        output_slot: pl.Scalar[pl.INDEX],
    ):
        # 连续半叶1..4096候选转成512项(score,index)根，保持原分支同分顺序。
        logical_begin_i32: pl.Scalar[pl.INT32] = pl.cast(logical_begin, pl.INT32)
        if valid_count <= 512:
            short_indices: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False), logical_begin_i32)
            short_raw: pl.Tile[[1, 512], pl.FP32, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.load(score_arena, [score_row, 0], [1, 512], [1, valid_count])
            short_scores: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(short_raw, pad_value=pl.PadValue.min)
            short_scores_v1: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(short_scores, -3.4028234663852886e38)
            short_pairs: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(short_scores_v1, pl.tile.reinterpret_view(short_indices, dtype=pl.UINT32))
            short_pairs_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(short_pairs, pl.const(64, pl.INT32))
            short_pairs_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(short_pairs_v1, pl.const(256, pl.INT32))
            pl.tile.store(short_pairs_v2, [output_slot, 0], pair_arena)
        else:
            if valid_count <= 1024:
                small_indices: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False), logical_begin_i32)
                small_raw: pl.Tile[[1, 1024], pl.FP32, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.load(score_arena, [score_row, 0], [1, 1024], [1, valid_count])
                small_scores: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(small_raw, pad_value=pl.PadValue.min)
                small_scores_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(small_scores, -3.4028234663852886e38)
                small_pairs: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(small_scores_v1, pl.tile.reinterpret_view(small_indices, dtype=pl.UINT32))
                small_pairs_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(small_pairs, pl.const(64, pl.INT32))
                small_pairs_v2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(small_pairs_v1, pl.const(256, pl.INT32))
                small_left: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(small_pairs_v2, [1, 1024], [0, 0])
                small_right: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(small_pairs_v2, [1, 1024], [0, 1024])
                small_tmp: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                small_merged: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(small_left, small_right, small_tmp, exhausted=False)
                small_top: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(small_merged, [1, 1024], [0, 0])
                pl.tile.store(small_top, [output_slot, 0], pair_arena)
            else:
                if valid_count <= 2048:
                    leaf_indices: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False), logical_begin_i32)
                    leaf_scores_raw: pl.Tile[[1, 2048], pl.FP32, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.load(score_arena, [score_row, 0], [1, 2048], [1, valid_count])
                    leaf_scores: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(leaf_scores_raw, pad_value=pl.PadValue.min)
                    leaf_scores_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(leaf_scores, -3.4028234663852886e38)
                    pairs: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(leaf_scores_v1, pl.tile.reinterpret_view(leaf_indices, dtype=pl.UINT32))
                    pairs_v1: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs, pl.const(64, pl.INT32))
                    pairs_v2: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v1, pl.const(256, pl.INT32))
                    pairs_v3: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v2, pl.const(1024, pl.INT32))
                    pl.tile.store(pl.tile.slice(pairs_v3, [1, 1024], [0, 0]), [output_slot, 0], pair_arena)
                else:
                    if valid_count <= 2560:
                        partial_indices: pl.Tile[[1, 2560], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.ci(pl.const(0, pl.INT32), [1, 2560], dtype=pl.INT32, descending=False), logical_begin_i32)
                        partial_raw: pl.Tile[[1, 2560], pl.FP32, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.load(score_arena, [score_row, 0], [1, 2560], [1, valid_count])
                        partial_scores: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(partial_raw, pad_value=pl.PadValue.min)
                        partial_scores_v1: pl.Tile[[1, 2560], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(partial_scores, -3.4028234663852886e38)
                        partial_pairs: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(
                            partial_scores_v1, pl.tile.reinterpret_view(partial_indices, dtype=pl.UINT32)
                        )
                        partial_pairs_v1: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(partial_pairs, pl.const(64, pl.INT32))
                        partial_pairs_v2: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(partial_pairs_v1, pl.const(256, pl.INT32))
                        partial_prefix: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(partial_pairs_v2, 1, 4096)
                        partial_prefix_v1: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(partial_prefix, pl.const(1024, pl.INT32))
                        partial_left: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(partial_prefix_v1, [1, 1024], [0, 0])
                        partial_right: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(partial_pairs_v2, [1, 1024], [0, 4096])
                        partial_tmp: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                        partial_merged: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(partial_right, partial_left, partial_tmp, exhausted=False)
                        pl.tile.store(pl.tile.slice(partial_merged, [1, 1024], [0, 0]), [output_slot, 0], pair_arena)
                    else:
                        if valid_count <= 3072:
                            triple_indices: pl.Tile[[1, 3072], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.ci(pl.const(0, pl.INT32), [1, 3072], dtype=pl.INT32, descending=False), logical_begin_i32)
                            triple_raw: pl.Tile[[1, 3072], pl.FP32, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.load(score_arena, [score_row, 0], [1, 3072], [1, valid_count])
                            triple_scores: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(triple_raw, pad_value=pl.PadValue.min)
                            triple_scores_v1: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(triple_scores, -3.4028234663852886e38)
                            triple_pairs: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(
                                triple_scores_v1, pl.tile.reinterpret_view(triple_indices, dtype=pl.UINT32)
                            )
                            triple_pairs_v1: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(triple_pairs, pl.const(64, pl.INT32))
                            triple_pairs_v2: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(triple_pairs_v1, pl.const(256, pl.INT32))
                            triple_prefix: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4096], pad=pl.PadValue.min)] = pl.tile.set_validshape(triple_pairs_v2, 1, 4096)
                            triple_prefix_v1: pl.Tile[[1, 6144], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(triple_prefix, pl.const(1024, pl.INT32))
                            triple_left: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(triple_prefix_v1, [1, 1024], [0, 0])
                            triple_right0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(triple_pairs_v2, [1, 1024], [0, 4096])
                            triple_right1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(triple_pairs_v2, [1, 1024], [0, 5120])
                            triple_tmp: pl.Tile[[1, 3072], pl.FP32] = pl.tile.create([1, 3072], dtype=pl.FP32)
                            triple_merged: pl.Tile[[1, 3072], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(
                                triple_right0, triple_right1, triple_left, triple_tmp, exhausted=False
                            )
                            pl.tile.store(pl.tile.slice(triple_merged, [1, 1024], [0, 0]), [output_slot, 0], pair_arena)
                        else:
                            if valid_count <= 4096:
                                medium_indices: pl.Tile[[1, 4096], pl.INT32, pl.Mem.Vec] = pl.tile.adds(
                                    pl.tile.ci(pl.const(0, pl.INT32), [1, 4096], dtype=pl.INT32, descending=False), logical_begin_i32
                                )
                                medium_scores_raw: pl.Tile[[1, 4096], pl.FP32, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.load(score_arena, [score_row, 0], [1, 4096], [1, valid_count])
                                medium_scores: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(medium_scores_raw, pad_value=pl.PadValue.min)
                                medium_scores_v1: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(medium_scores, -3.4028234663852886e38)
                                medium_pairs: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(
                                    medium_scores_v1, pl.tile.reinterpret_view(medium_indices, dtype=pl.UINT32)
                                )
                                medium_pairs_v1: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(medium_pairs, pl.const(64, pl.INT32))
                                medium_pairs_v2: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(medium_pairs_v1, pl.const(256, pl.INT32))
                                medium_pairs_v3: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(medium_pairs_v2, pl.const(1024, pl.INT32))
                                medium_left: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(medium_pairs_v3, [1, 1024], [0, 0])
                                medium_right: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(medium_pairs_v3, [1, 1024], [0, 4096])
                                medium_tmp: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                                medium_merged: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(
                                    medium_right, medium_left, medium_tmp, exhausted=False
                                )
                                pl.tile.store(pl.tile.slice(medium_merged, [1, 1024], [0, 0]), [output_slot, 0], pair_arena)

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_score_vector_lane_stage(
        self,
        idx_native_kv_cache: pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8],
        idx_block_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_INDEXER_COLS], pl.INT32],
        position_ids: pl.Tensor[[CSA_FULL_T], pl.INT64],
        kv_seq_lens: pl.Tensor[[CSA_FULL_B], pl.INT32],
        bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        offsets: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32],
        plan: pl.Tensor[[8], pl.INT32],
        buf_score_transfer: pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32],
        score_arena: pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32],
        pair_arena: pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32],
        buf_score_lane: pl.Scalar[pl.INDEX],
        buf_worker: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32], pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32]]:
        # AIV lane0/1都完整消费同一core事件；只有真实query可写根，空尾lane仍ACK。
        groups: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [0]), pl.INDEX)
        buf_max_leaves: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [2]), pl.INDEX)
        buf_leaf_tiles: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [3]), pl.INDEX)
        buf_extra_leaves: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(plan, [4]), pl.INDEX)
        for buf_item in pl.range(buf_worker, groups * buf_max_leaves, 24):
            _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]] = self.indexer_work_item(buf_item, groups, buf_max_leaves)
            buf_group: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
            buf_leaf: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
            _tuple_tmp_1: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]] = self.indexer_decode_group(
                bounds, kv_seq_lens, position_ids, offsets, buf_group
            )
            buf_batch_idx: pl.Scalar[pl.INDEX] = _tuple_tmp_1[0]
            buf_query: pl.Scalar[pl.INDEX] = _tuple_tmp_1[1]
            buf_live: pl.Scalar[pl.INDEX] = _tuple_tmp_1[2]
            buf_visible_count: pl.Scalar[pl.INDEX] = _tuple_tmp_1[3]
            _tuple_tmp_2: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]] = self.indexer_leaf_range(buf_leaf, buf_leaf_tiles, buf_extra_leaves, buf_visible_count)
            buf_logical_begin: pl.Scalar[pl.INDEX] = _tuple_tmp_2[0]
            buf_leaf_capacity: pl.Scalar[pl.INDEX] = _tuple_tmp_2[1]
            work_valid: pl.Scalar[pl.INDEX] = _tuple_tmp_2[2]
            if work_valid > 0:
                buf_cache_len: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(kv_seq_lens, [buf_batch_idx]), pl.INDEX) // 4
                if True:
                    single_query: pl.Scalar[pl.INDEX] = buf_query
                    single_leaf: pl.Scalar[pl.INDEX] = buf_leaf
                    single_batch: pl.Scalar[pl.INDEX] = buf_batch_idx
                    single_cache_len: pl.Scalar[pl.INDEX] = buf_cache_len
                    single_visible: pl.Scalar[pl.INDEX] = buf_visible_count
                    single_begin: pl.Scalar[pl.INDEX] = buf_logical_begin
                    single_capacity: pl.Scalar[pl.INDEX] = buf_leaf_capacity
                    single_valid: pl.Scalar[pl.INDEX] = pl.min(single_capacity, single_visible - single_begin)
                    single_steps: pl.Scalar[pl.INDEX] = (single_valid + 1024 - 1) // 1024
                    single_roots: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([3, 1024], dtype=pl.FP32, value=-3.4028234663852886e38)
                    single_segments: pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.full([6, 1024], dtype=pl.FP32, value=-3.4028234663852886e38)
                    for single_step in pl.range(single_steps):
                        single_score_begin: pl.Scalar[pl.INDEX] = single_begin + single_step * 1024
                        single_transfer_row: pl.Scalar[pl.INDEX] = buf_worker * 2 * 6 + single_step % 2 * 6 + buf_score_lane * 3
                        single_scale_bytes: pl.Tile[[1, 2048], pl.INT8, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.full([1, 1024], dtype=pl.INT16, value=0), dtype=pl.INT8)
                        for single_scale_page in pl.range(32):
                            single_scale_logical: pl.Scalar[pl.INDEX] = pl.min(single_score_begin // 32 + single_scale_page, pl.max((single_cache_len - 1) // 32, 0))
                            if single_scale_logical >= 0 and single_scale_logical < pl.tensor.dim(idx_block_table, 1):
                                single_scale_physical: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(idx_block_table, [single_batch, single_scale_logical]), pl.INDEX)
                                if single_scale_physical >= 0 and single_scale_physical < pl.tensor.dim(idx_native_kv_cache, 0):
                                    single_scale_bytes: pl.Tile[[1, 2048], pl.INT8, pl.Mem.Vec] = pl.tile.gather_row(
                                        single_scale_bytes, idx_native_kv_cache, [0, single_scale_page * 32 * 2], [single_scale_physical, 4096], [1, 64], transpose=False
                                    )
                        single_scale_half: pl.Tile[[1, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.reinterpret_view(single_scale_bytes, dtype=pl.FP16)
                        single_kv_scale: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.cast(single_scale_half, target_type=pl.FP32, mode="round")
                        pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        single_scores: pl.Tile[[3, 1024], pl.FP32] = pl.tile.load(buf_score_transfer, [single_transfer_row, 0], [3, 1024], [3, 1024])
                        pl.system.sync_set(1, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV)
                        for single_local in pl.unroll(3):
                            single_score_row: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_scores, single_local, 0, [1, 1024], target_memory=pl.Mem.Vec)
                            single_scaled: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.mul(single_score_row, single_kv_scale)
                            single_segments: pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(single_segments, single_scaled, [single_local * 2 + single_step % 2, 0])
                        if single_step % 2 == 1 or single_step + 1 == single_steps:
                            single_segment_begin: pl.Scalar[pl.INDEX] = single_begin + single_step // 2 * 2048
                            single_segment_span: pl.Scalar[pl.INDEX] = (single_step % 2 + 1) * 1024
                            single_segment_rows: pl.Tile[[3, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(single_segments, [3, 2048])
                            for single_local_1 in pl.unroll(3):
                                single_query_lane: pl.Scalar[pl.INDEX] = buf_score_lane * 3 + single_local_1
                                if single_query_lane < buf_live:
                                    _tuple_tmp_3: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.indexer_query_visibility(
                                        bounds, kv_seq_lens, position_ids, single_batch, single_query + single_query_lane
                                    )
                                    single_query_visible: pl.Scalar[pl.INDEX] = _tuple_tmp_3[0]
                                    _causal: pl.Scalar[pl.INT32] = _tuple_tmp_3[1]
                                    single_segment_valid: pl.Scalar[pl.INDEX] = pl.max(pl.min(single_query_visible - single_segment_begin, single_segment_span), 0)
                                    if single_segment_valid > 0:
                                        single_segment_data: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_rows, single_local_1, 0, [1, 2048], target_memory=pl.Mem.Vec)
                                        logical_i32: pl.Scalar[pl.INT32] = pl.cast(single_segment_begin, pl.INT32)
                                        if single_segment_valid <= 512:
                                            tiny_indices: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.adds(
                                                pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False), logical_i32
                                            )
                                            tiny_tile: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_data, 0, 0, [1, 512], target_memory=pl.Mem.Vec)
                                            tiny_raw: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, single_segment_valid])] = pl.tile.set_validshape(
                                                tiny_tile, 1, single_segment_valid
                                            )
                                            tiny_scores: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(
                                                pl.tile.fillpad(tiny_raw, pad_value=pl.PadValue.min), -3.4028234663852886e38
                                            )
                                            tiny_pairs: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(
                                                tiny_scores, pl.tile.reinterpret_view(tiny_indices, dtype=pl.UINT32)
                                            )
                                            tiny_pairs_v1: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(tiny_pairs, pl.const(64, pl.INT32))
                                            tiny_pairs_v2: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(tiny_pairs_v1, pl.const(256, pl.INT32))
                                            single_new_root: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(tiny_pairs_v2, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                        else:
                                            if single_segment_valid <= 1024:
                                                small_indices: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = pl.tile.adds(
                                                    pl.tile.ci(pl.const(0, pl.INT32), [1, 1024], dtype=pl.INT32, descending=False), logical_i32
                                                )
                                                small_tile: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_data, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                                small_raw: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, single_segment_valid])] = pl.tile.set_validshape(
                                                    small_tile, 1, single_segment_valid
                                                )
                                                small_scores: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(
                                                    pl.tile.fillpad(small_raw, pad_value=pl.PadValue.min), -3.4028234663852886e38
                                                )
                                                small_pairs: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(
                                                    small_scores, pl.tile.reinterpret_view(small_indices, dtype=pl.UINT32)
                                                )
                                                small_pairs_v1: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(small_pairs, pl.const(64, pl.INT32))
                                                small_pairs_v2: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(
                                                    small_pairs_v1, pl.const(256, pl.INT32)
                                                )
                                                small_left: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(small_pairs_v2, [1, 1024], [0, 0])
                                                small_right: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(small_pairs_v2, [1, 1024], [0, 1024])
                                                small_tmp: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                                                small_merged: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format2(
                                                    small_left, small_right, small_tmp, exhausted=False
                                                )
                                                single_new_root: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(small_merged, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                            else:
                                                large_indices: pl.Tile[[1, 2048], pl.INT32, pl.Mem.Vec] = pl.tile.adds(
                                                    pl.tile.ci(pl.const(0, pl.INT32), [1, 2048], dtype=pl.INT32, descending=False), logical_i32
                                                )
                                                large_tile: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_segment_data, 0, 0, [1, 2048], target_memory=pl.Mem.Vec)
                                                large_raw: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, single_segment_valid])] = pl.tile.set_validshape(
                                                    large_tile, 1, single_segment_valid
                                                )
                                                large_scores: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(
                                                    pl.tile.fillpad(large_raw, pad_value=pl.PadValue.min), -3.4028234663852886e38
                                                )
                                                large_pairs: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(
                                                    large_scores, pl.tile.reinterpret_view(large_indices, dtype=pl.UINT32)
                                                )
                                                large_pairs_v1: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(large_pairs, pl.const(64, pl.INT32))
                                                large_pairs_v2: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(
                                                    large_pairs_v1, pl.const(256, pl.INT32)
                                                )
                                                large_pairs_v3: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(
                                                    large_pairs_v2, pl.const(1024, pl.INT32)
                                                )
                                                single_new_root: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(large_pairs_v3, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                        if single_step < 2:
                                            single_roots: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(single_roots, single_new_root, [single_local_1, 0])
                                        else:
                                            single_old_root: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_roots, single_local_1, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                            single_merge_tmp: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                                            single_merged: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(single_new_root, single_old_root, single_merge_tmp, exhausted=False)
                                            single_next_root: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_merged, 0, 0, [1, 1024], target_memory=pl.Mem.Vec)
                                            single_roots: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(single_roots, single_next_root, [single_local_1, 0])
                    for single_local_2 in pl.unroll(3):
                        single_query_lane: pl.Scalar[pl.INDEX] = buf_score_lane * 3 + single_local_2
                        if single_query_lane < buf_live:
                            single_root: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.extract(single_roots, single_local_2, 0, [1, 1024], target_memory=pl.Mem.Vec)
                            pl.tile.store(single_root, [(single_query + single_query_lane) * 64 + single_leaf, 0], pair_arena)
                else:
                    buf_valid_count: pl.Scalar[pl.INDEX] = pl.min(buf_leaf_capacity, buf_visible_count - buf_logical_begin)
                    buf_tile_count: pl.Scalar[pl.INDEX] = (buf_valid_count + 1024 - 1) // 1024
                    buf_lane_span: pl.Scalar[pl.INDEX] = pl.min(buf_tile_count * 512, 4096)
                    buf_score_iters: pl.Scalar[pl.INDEX] = (buf_lane_span + 512 - 1) // 512
                    buf_lane_begin: pl.Scalar[pl.INDEX] = buf_score_lane * buf_lane_span
                    for buf_score_step in pl.range(buf_score_iters):
                        buf_score_begin: pl.Scalar[pl.INDEX] = buf_score_step * 512
                        buf_transfer_row: pl.Scalar[pl.INDEX] = buf_worker * 2 * 6 + buf_score_step % 2 * 6
                        buf_scale_bytes: pl.Tile[[1, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.full([1, 512], dtype=pl.INT16, value=0), dtype=pl.INT8)
                        for buf_scale_page in pl.range(16):
                            buf_scale_row: pl.Scalar[pl.INDEX] = buf_logical_begin + buf_score_begin + buf_lane_begin + buf_scale_page * 32
                            buf_scale_safe_page: pl.Scalar[pl.INDEX] = pl.min(buf_scale_row // 32, pl.max((buf_cache_len - 1) // 32, 0))
                            if buf_scale_safe_page >= 0 and buf_scale_safe_page < pl.tensor.dim(idx_block_table, 1):
                                buf_scale_physical_page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(idx_block_table, [buf_batch_idx, buf_scale_safe_page]), pl.INDEX)
                                if buf_scale_physical_page >= 0 and buf_scale_physical_page < pl.tensor.dim(idx_native_kv_cache, 0):
                                    buf_scale_bytes: pl.Tile[[1, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.gather_row(
                                        buf_scale_bytes, idx_native_kv_cache, [0, buf_scale_page * 32 * 2], [buf_scale_physical_page, 4096], [1, 64], transpose=False
                                    )
                        buf_scale_half: pl.Tile[[1, 512], pl.FP16, pl.Mem.Vec] = pl.tile.reinterpret_view(buf_scale_bytes, dtype=pl.FP16)
                        buf_kv_scale: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(buf_scale_half, target_type=pl.FP32, mode="round")
                        pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        buf_score_sums: pl.Tile[[6, 512], pl.FP32] = pl.tile.load(buf_score_transfer, [buf_transfer_row, buf_score_lane * 512], [6, 512], [6, 512])
                        pl.system.sync_set(1, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV)
                        for buf_query_lane in pl.unroll(6):
                            if buf_query_lane < buf_live:
                                _tuple_tmp_4: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.indexer_query_visibility(
                                    bounds, kv_seq_lens, position_ids, buf_batch_idx, buf_query + buf_query_lane
                                )
                                buf_query_visible: pl.Scalar[pl.INDEX] = _tuple_tmp_4[0]
                                _causal: pl.Scalar[pl.INT32] = _tuple_tmp_4[1]
                                buf_lane_valid_rows: pl.Scalar[pl.INDEX] = pl.max(pl.min(buf_query_visible - buf_logical_begin - buf_score_begin - buf_lane_begin, 512), 0)
                                buf_score_sum: pl.Tile[[1, 512], pl.FP32] = pl.tile.slice(buf_score_sums, [1, 512], [buf_query_lane, 0])
                                buf_score_row: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(buf_score_sum, buf_kv_scale)
                                if buf_lane_valid_rows > 0:
                                    pl.tile.store(
                                        pl.tile.set_validshape(buf_score_row, 1, buf_lane_valid_rows), [buf_worker * 2 * 6 + buf_query_lane * 2 + buf_score_lane, buf_score_begin], score_arena
                                    )
                    for buf_query_lane_1 in pl.unroll(6):
                        if buf_query_lane_1 < buf_live:
                            _tuple_tmp_5: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]] = self.indexer_query_visibility(
                                bounds, kv_seq_lens, position_ids, buf_batch_idx, buf_query + buf_query_lane_1
                            )
                            buf_query_visible: pl.Scalar[pl.INDEX] = _tuple_tmp_5[0]
                            _causal: pl.Scalar[pl.INT32] = _tuple_tmp_5[1]
                            buf_half_begin: pl.Scalar[pl.INDEX] = buf_logical_begin + buf_lane_begin
                            buf_half_valid: pl.Scalar[pl.INDEX] = pl.max(pl.min(buf_query_visible - buf_half_begin, buf_lane_span), 0)
                            buf_half_slot: pl.Scalar[pl.INDEX] = (buf_query + buf_query_lane_1) * 64 + buf_leaf * 2 + buf_score_lane
                            if buf_half_valid > 0:
                                self.indexer_topk_half_leaf(score_arena, pair_arena, buf_worker * 2 * 6 + buf_query_lane_1 * 2 + buf_score_lane, buf_half_begin, buf_half_valid, buf_half_slot)
                            else:
                                pl.tile.store(pl.tile.full([1, 1024], dtype=pl.FP32, value=-3.4028234663852886e38), [buf_half_slot, 0], pair_arena)
        return score_arena, pair_arena

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_weights_reduce_stage(
        self,
        weights_partial: pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32],
        weights: pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32],
        tokens: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32]:
        # AIV：等当前M16的四个投影split，原左折叠后乘模型scale，保持FP32给score消费。
        row_blocks: pl.Scalar[pl.INDEX] = (tokens + 16 - 1) // 16
        t_pad: pl.Scalar[pl.INDEX] = row_blocks * 16
        for w_rb in pl.range(worker_id, row_blocks, 48):
            w_r0: pl.Scalar[pl.INDEX] = w_rb * 16
            w_sum: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(weights_partial, [16, 64], [w_r0, 0])
            for kb in pl.unroll(1, 4):
                partial_r0: pl.Scalar[pl.INDEX] = kb * t_pad + w_r0
                w_sum: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(w_sum, pl.tensor.slice(weights_partial, [16, 64], [partial_r0, 0]))
            weights: pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32] = pl.tensor.assemble(weights, pl.tensor.muls(w_sum, 0.011048543456039806), [w_r0, 0])
        return weights

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def kv_norm_rope_stage(
        self,
        kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32],
        gamma_ckv: pl.Tensor[[512], pl.BF16],
        rope_cos_il: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_sin_signed: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_swap_idx: pl.Tensor[[CSA_FULL_T, 64], pl.INT32],
        kv: pl.Tensor[[CSA_FULL_T, 512], pl.BF16],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_T, 512], pl.BF16]:
        # AIV 独占 32 行的全部 512 列；调用前由外层等齐投影和 RoPE 输入。
        #
        #         本 helper 内没有 wait。完整块先对 512 列做同一 RMS，再分别处理 NOPE
        #         和 RoPE；尾块走显式 load/store 的有效行路径，不访问 pad16 之外的 GM。
        #
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv, 0)
        token_tiles: pl.Scalar[pl.INDEX] = (tokens + 32 - 1) // 32
        for tg_idx in pl.range(worker_id, token_tiles, 48):
            tg: pl.Scalar[pl.INDEX] = tg_idx * 32
            valid_rows: pl.Scalar[pl.INDEX] = pl.min(32, tokens - tg)
            if valid_rows == 32:
                kv_sq_lanes: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.full([32, 64], dtype=pl.FP32, value=0.0)
                for kv_sq_col0 in pl.pipeline(0, 512, 64, stage=2):
                    kv_chunk: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(kv_fp32, [32, 64], [tg, kv_sq_col0])
                    kv_sq: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.mul(kv_chunk, kv_chunk)
                    kv_sq_lanes: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.add(kv_sq_lanes, kv_sq)
                kv_sq_sum: pl.Tensor[[1, 32], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(kv_sq_lanes), [1, 32])
                kv_rms: pl.Tensor[[1, 32], pl.FP32] = pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(kv_sq_sum, 0.001953125), 9.9999999999999995e-07))
                kv_inv_rms: pl.Tensor[[1, 32], pl.FP32] = pl.tensor.div(pl.tensor.full([1, 32], dtype=pl.FP32, value=1.0), kv_rms)
                kv_inv_rms_t: pl.Tensor[[32, 1], pl.FP32] = pl.tensor.reshape(kv_inv_rms, [32, 1])
                for n0 in pl.pipeline(0, 448, 64, stage=2):
                    kv_chunk: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(kv_fp32, [32, 64], [tg, n0])
                    gamma_kv_cast: pl.Tensor[[64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_ckv, [64], [n0]), target_type=pl.FP32, mode="round")
                    gamma_kv_chunk: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(gamma_kv_cast, [1, 64])
                    kv_normed: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
                    kv_normed_bf16: pl.Tensor[[32, 64], pl.BF16] = pl.tensor.cast(kv_normed, target_type=pl.BF16, mode="rint")
                    kv: pl.Tensor[[CSA_FULL_T, 512], pl.BF16] = pl.tensor.assemble(kv, kv_normed_bf16, [tg, n0])
                gamma_rope_cast: pl.Tensor[[64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_ckv, [64], [448]), target_type=pl.FP32, mode="round")
                gamma_rope: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(gamma_rope_cast, [1, 64])
                kv_rope_chunk: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(kv_fp32, [32, 64], [tg, 448])
                kv_rope_norm_chunk: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
                kv_cos_il_full: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(rope_cos_il, [32, 64], [tg, 0])
                kv_sin_signed_full: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(rope_sin_signed, [32, 64], [tg, 0])
                kv_swap_idx_full: pl.Tensor[[32, 64], pl.INT32] = pl.tensor.slice(rope_swap_idx, [32, 64], [tg, 0])
                kv_swapped_full: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.gather(kv_rope_norm_chunk, kv_swap_idx_full, dim=-1)
                kv_rope_rot_full: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(kv_rope_norm_chunk, kv_cos_il_full), pl.tensor.mul(kv_swapped_full, kv_sin_signed_full))
                kv_rope_i16_full: pl.Tensor[[32, 64], pl.BF16] = pl.tensor.cast(kv_rope_rot_full, target_type=pl.BF16, mode="rint")
                kv: pl.Tensor[[CSA_FULL_T, 512], pl.BF16] = pl.tensor.assemble(kv, kv_rope_i16_full, [tg, 448])
            else:
                kv_reduce_tmp: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                kv_sq_lanes_tail: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=0.0)
                for kv_sq_col0_tail in pl.pipeline(0, 512, 64, stage=2):
                    kv_chunk_tail: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.load(
                        kv_fp32, [tg, kv_sq_col0_tail], [32, 64], [valid_rows, 64], target_memory=pl.Mem.Vec
                    )
                    kv_sq_tail: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.mul(kv_chunk_tail, kv_chunk_tail)
                    kv_sq_lanes_tail: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(kv_sq_lanes_tail, kv_sq_tail)
                kv_sq_sum_tail: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.row_sum(kv_sq_lanes_tail, kv_reduce_tmp), [1, 32])
                kv_rms_tail: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(pl.tile.adds(pl.tile.muls(kv_sq_sum_tail, 0.001953125), 9.9999999999999995e-07))
                kv_inv_rms_tail: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.div(pl.tile.full([1, 32], dtype=pl.FP32, value=1.0), kv_rms_tail)
                kv_inv_rms_t_tail: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(kv_inv_rms_tail, [32, 1])
                for n0_tail in pl.pipeline(0, 448, 64, stage=2):
                    kv_chunk_tail: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.load(
                        kv_fp32, [tg, n0_tail], [32, 64], [valid_rows, 64], target_memory=pl.Mem.Vec
                    )
                    gamma_kv_input_tail: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(gamma_ckv, [n0_tail], [64], [64], target_memory=pl.Mem.Vec)
                    gamma_kv_cast_tail: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(gamma_kv_input_tail, target_type=pl.FP32, mode="round")
                    gamma_kv_chunk_tail: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(gamma_kv_cast_tail, [1, 64])
                    kv_normed_tail: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.col_expand_mul(
                        pl.tile.row_expand_mul(kv_chunk_tail, kv_inv_rms_t_tail), gamma_kv_chunk_tail
                    )
                    kv_normed_bf16_tail: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.cast(kv_normed_tail, target_type=pl.BF16, mode="rint")
                    kv_normed_valid: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.set_validshape(kv_normed_bf16_tail, valid_rows, 64)
                    pl.tile.store(kv_normed_valid, [tg, n0_tail], kv)
                gamma_rope_input_tail: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(gamma_ckv, [448], [64], [64], target_memory=pl.Mem.Vec)
                gamma_rope_cast_tail: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(gamma_rope_input_tail, target_type=pl.FP32, mode="round")
                gamma_rope_tail: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(gamma_rope_cast_tail, [1, 64])
                kv_rope_chunk_tail: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.load(
                    kv_fp32, [tg, 448], [32, 64], [valid_rows, 64], target_memory=pl.Mem.Vec
                )
                kv_rope_norm_tail: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.col_expand_mul(
                    pl.tile.row_expand_mul(kv_rope_chunk_tail, kv_inv_rms_t_tail), gamma_rope_tail
                )
                kv_cos_il_tail: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.load(
                    rope_cos_il, [tg, 0], [32, 64], [valid_rows, 64], target_memory=pl.Mem.Vec
                )
                kv_sin_signed_tail: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.load(
                    rope_sin_signed, [tg, 0], [32, 64], [valid_rows, 64], target_memory=pl.Mem.Vec
                )
                kv_col: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(
                    pl.tile.full([32, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode="round")
                )
                kv_dup_f: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(kv_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32, mode="round")
                kv_lane: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(kv_col, pl.tile.muls(kv_dup_f, 2.0))
                kv_swap_f: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(kv_col, 1.0), pl.tile.muls(kv_lane, 2.0))
                kv_row_seed: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(
                    pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 32], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode="round"), 64.0
                )
                kv_row_grid: pl.Tile[[64, 32], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 32], dtype=pl.FP32, value=1.0), kv_row_seed)
                kv_row_offset: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(kv_row_grid, 0, 1)
                kv_swap_idx_tail: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(kv_swap_f, kv_row_offset), target_type=pl.INT32, mode="round")
                kv_gather_tmp: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                kv_swapped_tail: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(kv_rope_norm_tail, kv_swap_idx_tail, kv_gather_tmp)
                kv_rope_rot_tail: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.add(
                    pl.tile.mul(kv_rope_norm_tail, kv_cos_il_tail), pl.tile.mul(kv_swapped_tail, kv_sin_signed_tail)
                )
                kv_rope_i16_tail: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.cast(kv_rope_rot_tail, target_type=pl.BF16, mode="rint")
                kv_rope_valid: pl.Tile[[32, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.set_validshape(kv_rope_i16_tail, valid_rows, 64)
                pl.tile.store(kv_rope_valid, [tg, 448], kv)
        return kv

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def hc_post_row_block(
        self,
        x_block: pl.Tensor[[1, 4096], pl.BF16],
        residual_flat: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16],
        post: pl.Tensor[[CSA_FULL_T, 4], pl.FP32],
        comb: pl.Tensor[[CSA_FULL_T, 16], pl.FP32],
        y_flat: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16],
        t0: pl.Scalar[pl.INDEX],
        n0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_T, 16384], pl.BF16]:
        # 严格单行HC post；x_block已经BF16舍入，不把一个token门值广播到多行。
        x_f32: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.cast(x_block, target_type=pl.FP32, mode="round")
        res_0: pl.Tensor[
            [1, 4096],
            pl.FP32,
            pl.TensorView(
                valid_shape=[
                    pl.min(pl.max(CSA_FULL_T - t0, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)),
                    pl.min(pl.max(pl.const(16384, pl.INDEX) - n0, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX)),
                ],
                stride=[],
                layout=pl.TensorLayout.ND,
            ),
        ] = pl.tensor.cast(pl.tensor.slice(residual_flat, [1, 4096], [t0, n0], [1, 4096], clamp=True), target_type=pl.FP32, mode="round")
        res_1: pl.Tensor[
            [1, 4096],
            pl.FP32,
            pl.TensorView(
                valid_shape=[
                    pl.min(pl.max(CSA_FULL_T - t0, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)),
                    pl.min(pl.max(pl.const(12288, pl.INDEX) - n0, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX)),
                ],
                stride=[],
                layout=pl.TensorLayout.ND,
            ),
        ] = pl.tensor.cast(pl.tensor.slice(residual_flat, [1, 4096], [t0, 4096 + n0], [1, 4096], clamp=True), target_type=pl.FP32, mode="round")
        res_2: pl.Tensor[
            [1, 4096],
            pl.FP32,
            pl.TensorView(
                valid_shape=[
                    pl.min(pl.max(CSA_FULL_T - t0, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)),
                    pl.min(pl.max(pl.const(8192, pl.INDEX) - n0, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX)),
                ],
                stride=[],
                layout=pl.TensorLayout.ND,
            ),
        ] = pl.tensor.cast(pl.tensor.slice(residual_flat, [1, 4096], [t0, 8192 + n0], [1, 4096], clamp=True), target_type=pl.FP32, mode="round")
        res_3: pl.Tensor[
            [1, 4096],
            pl.FP32,
            pl.TensorView(
                valid_shape=[
                    pl.min(pl.max(CSA_FULL_T - t0, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)),
                    pl.min(pl.max(pl.const(4096, pl.INDEX) - n0, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX)),
                ],
                stride=[],
                layout=pl.TensorLayout.ND,
            ),
        ] = pl.tensor.cast(pl.tensor.slice(residual_flat, [1, 4096], [t0, 12288 + n0], [1, 4096], clamp=True), target_type=pl.FP32, mode="round")
        for out_h in pl.unroll(4):
            post_scalar: pl.Scalar[pl.FP32] = pl.tensor.read(post, [t0, out_h])
            single_y: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.muls(x_f32, post_scalar)
            comb_scalar_0: pl.Scalar[pl.FP32] = pl.tensor.read(comb, [t0, out_h])
            single_weighted_0: pl.Tensor[
                [1, 4096],
                pl.FP32,
                pl.TensorView(
                    valid_shape=[
                        pl.min(pl.max(CSA_FULL_T - t0, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)),
                        pl.min(pl.max(pl.const(16384, pl.INDEX) - n0, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX)),
                    ],
                    stride=[],
                    layout=pl.TensorLayout.ND,
                ),
            ] = pl.tensor.muls(res_0, comb_scalar_0)
            single_y_v1: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.add(single_y, single_weighted_0)
            comb_scalar_1: pl.Scalar[pl.FP32] = pl.tensor.read(comb, [t0, 4 + out_h])
            single_weighted_1: pl.Tensor[
                [1, 4096],
                pl.FP32,
                pl.TensorView(
                    valid_shape=[
                        pl.min(pl.max(CSA_FULL_T - t0, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)),
                        pl.min(pl.max(pl.const(12288, pl.INDEX) - n0, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX)),
                    ],
                    stride=[],
                    layout=pl.TensorLayout.ND,
                ),
            ] = pl.tensor.muls(res_1, comb_scalar_1)
            single_y_v2: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.add(single_y_v1, single_weighted_1)
            comb_scalar_2: pl.Scalar[pl.FP32] = pl.tensor.read(comb, [t0, 8 + out_h])
            single_weighted_2: pl.Tensor[
                [1, 4096],
                pl.FP32,
                pl.TensorView(
                    valid_shape=[
                        pl.min(pl.max(CSA_FULL_T - t0, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)),
                        pl.min(pl.max(pl.const(8192, pl.INDEX) - n0, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX)),
                    ],
                    stride=[],
                    layout=pl.TensorLayout.ND,
                ),
            ] = pl.tensor.muls(res_2, comb_scalar_2)
            single_y_v3: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.add(single_y_v2, single_weighted_2)
            comb_scalar_3: pl.Scalar[pl.FP32] = pl.tensor.read(comb, [t0, 12 + out_h])
            single_weighted_3: pl.Tensor[
                [1, 4096],
                pl.FP32,
                pl.TensorView(
                    valid_shape=[
                        pl.min(pl.max(CSA_FULL_T - t0, pl.const(0, pl.INDEX)), pl.const(1, pl.INDEX)),
                        pl.min(pl.max(pl.const(4096, pl.INDEX) - n0, pl.const(0, pl.INDEX)), pl.const(4096, pl.INDEX)),
                    ],
                    stride=[],
                    layout=pl.TensorLayout.ND,
                ),
            ] = pl.tensor.muls(res_3, comb_scalar_3)
            single_y_v4: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.add(single_y_v3, single_weighted_3)
            single_result: pl.Tensor[[1, 4096], pl.BF16] = pl.tensor.cast(single_y_v4, target_type=pl.BF16, mode="rint")
            y_flat: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16] = pl.tensor.assemble(y_flat, pl.tensor.set_validshape(single_result, 1, 4096), [t0, out_h * 4096 + n0])
        return y_flat

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def o_dequant_hc_post_stage(
        self,
        partials: pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32],
        act_scale_dq: pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32],
        wo_b_scale: pl.Tensor[[4096], pl.FP32],
        residual_flat: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16],
        post: pl.Tensor[[CSA_FULL_T, 4], pl.FP32],
        comb: pl.Tensor[[CSA_FULL_T, 16], pl.FP32],
        out_flat: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_T, 16384], pl.BF16]:
        # AIV按实际token条带输出；每行先等八组partial及该行post/comb，不含隐式同步。
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(out_flat, 0)
        wb_scale: pl.Tile[[4096], pl.FP32] = pl.tile.load(wo_b_scale, [0], [4096], [4096])
        wb_scale_chunk: pl.Tile[[1, 4096], pl.FP32] = pl.tile.reshape(wb_scale, [1, 4096])
        for b_tb in pl.range(worker_id, tokens, 48):
            acc: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 4096], dtype=pl.FP32, value=0.0)
            for act_g in pl.pipeline(8, stage=2):
                p_col0: pl.Scalar[pl.INDEX] = act_g * 4096
                p_g: pl.Tile[[1, 4096], pl.INT32] = pl.tile.load(partials, [b_tb, p_col0], [1, 4096], [1, 4096])
                g_scale_scalar: pl.Scalar[pl.FP32] = pl.tensor.read(act_scale_dq, [act_g, b_tb])
                p_g_f32: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.cast(p_g, target_type=pl.FP32, mode="none")
                acc: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(acc, pl.tile.muls(p_g_f32, g_scale_scalar))
            out_t: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(acc, wb_scale_chunk)
            out_bf16: pl.Tensor[[1, 4096], pl.BF16] = pl.tensor.create([1, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            out_bf16_v1: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(out_t, target_type=pl.BF16, mode="rint")
            out_flat: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16] = self.hc_post_row_block(out_bf16_v1, residual_flat, post, comb, out_flat, b_tb, 0)
        return out_flat

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def o_projection_groups_quant_stage(
        self,
        o_r_pad: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32],
        o_r_i8_pad: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8],
        act_scale_dq: pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32],
        tokens: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
        group_begin: pl.Scalar[pl.INDEX],
        group_end: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8], pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32]]:
        # 量化动态组区间；尾零属于该组的最后活动块，必须先于本组通知完成。
        task_rows: pl.Scalar[pl.INDEX] = (tokens + 32 - 1) // 32
        padded_rows: pl.Scalar[pl.INDEX] = (tokens + 96 - 1) // 96 * 96
        group_first: pl.Scalar[pl.INDEX] = group_begin * task_rows
        group_limit: pl.Scalar[pl.INDEX] = group_end * task_rows
        first: pl.Scalar[pl.INDEX] = worker_id + (group_first + 48 - 1 - worker_id) // 48 * 48
        for logical in pl.range(first, group_limit, 48):
            g: pl.Scalar[pl.INDEX] = logical // task_rows
            block: pl.Scalar[pl.INDEX] = logical % task_rows
            col_g: pl.Scalar[pl.INDEX] = g * 1024
            quant_start: pl.Scalar[pl.INDEX] = block * 32
            for qt in pl.pipeline(quant_start, pl.min(quant_start + 32, tokens), 8, stage=2):
                rows: pl.Scalar[pl.INDEX] = pl.min(8, tokens - qt)
                oc_amax: pl.Tile[[8, 1024], pl.FP32, pl.TileView(valid_shape=[rows, 1024])] = pl.tile.load(o_r_pad, [qt, col_g], [8, 1024], [rows, 1024])
                reduce_tmp: pl.Tile[[8, 1024], pl.FP32] = pl.tile.create([8, 1024], dtype=pl.FP32)
                g_row_max: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, rows])] = pl.tile.reshape(
                    pl.tile.set_validshape(pl.tile.row_max(pl.tile.abs(oc_amax), reduce_tmp), rows, 1), [1, 8]
                )
                g_amax_floor: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, rows])] = pl.tile.set_validshape(pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001), 1, rows)
                g_amax: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, rows])] = pl.tile.set_validshape(pl.tile.maximum(g_amax_floor, g_row_max), 1, rows)
                g_scale_num: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, rows])] = pl.tile.set_validshape(pl.tile.full([1, 8], dtype=pl.FP32, value=127.0), 1, rows)
                g_sq_row: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, rows])] = pl.tile.set_validshape(pl.tile.div(g_scale_num, g_amax), 1, rows)
                dequant_scale: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, rows])] = pl.tile.set_validshape(pl.tile.muls(g_amax, 0.007874015748031496), 1, rows)
                pl.tile.store(dequant_scale, [g, qt], act_scale_dq)
                g_sq_col: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 1])] = pl.tile.reshape(g_sq_row, [8, 1])
                oc_q: pl.Tile[[8, 1024], pl.FP32, pl.TileView(valid_shape=[rows, 1024])] = pl.tile.load(o_r_pad, [qt, col_g], [8, 1024], [rows, 1024])
                oq_scaled: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 1024])] = pl.tile.set_validshape(pl.tile.row_expand_mul(oc_q, g_sq_col), rows, 1024)
                oq_i32: pl.Tile[[8, 1024], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 1024])] = pl.tile.set_validshape(pl.tile.cast(oq_scaled, target_type=pl.INT32, mode="rint"), rows, 1024)
                oq_half: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 1024])] = pl.tile.set_validshape(pl.tile.cast(oq_i32, target_type=pl.FP16, mode="round"), rows, 1024)
                oq_i8: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[rows, 1024])] = pl.tile.set_validshape(pl.tile.cast(oq_half, target_type=pl.INT8, mode="trunc"), rows, 1024)
                pl.tile.store(oq_i8, [qt, col_g], o_r_i8_pad)
            if quant_start + 32 >= tokens:
                for zt in pl.range(tokens, padded_rows, 8):
                    zero_half: pl.Tile[[8, 1024], pl.FP16, pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
                    zero_i8: pl.Tile[[8, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.cast(zero_half, target_type=pl.INT8, mode="trunc")
                    zero_rows: pl.Scalar[pl.INDEX] = pl.min(8, padded_rows - zt)
                    pl.tile.store(pl.tile.set_validshape(zero_i8, zero_rows, 1024), [zt, col_g], o_r_i8_pad)
        return o_r_i8_pad, act_scale_dq

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def o_projection_group_quant_stage(
        self,
        o_r_pad: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32],
        o_r_i8_pad: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8],
        act_scale_dq: pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32],
        tokens: pl.Scalar[pl.INDEX],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8], pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32]]:
        # 全八组入口，与按组通知分支共享 amax、量化舍入和尾行清零数学。
        _tuple_tmp: pl.Tuple[pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8], pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32]] = self.o_projection_groups_quant_stage(
            o_r_pad, o_r_i8_pad, act_scale_dq, tokens, worker_id, 0, 8
        )
        o_r_i8_pad: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8] = _tuple_tmp[0]
        act_scale_dq: pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32] = _tuple_tmp[1]
        return o_r_i8_pad, act_scale_dq

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def original_kv_writeback_stage(
        self, kv: pl.Tensor[[CSA_FULL_T, 512], pl.BF16], slots: pl.Tensor[[CSA_FULL_T, 2], pl.INT32], cache_rows: pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16], worker_id: pl.Scalar[pl.INDEX]
    ) -> pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16]:
        # AIV 按真实 T 写 Native 原始页；WORKERS>0，worker_id∈[0,WORKERS)。
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv, 0)
        cache_pages: pl.Scalar[pl.INDEX] = pl.tensor.dim(cache_rows, 0) // 32
        blocks: pl.Scalar[pl.INDEX] = (tokens + 8 - 1) // 8
        for block in pl.range(worker_id, blocks, 48):
            first: pl.Scalar[pl.INDEX] = block * 8
            for row_in_block in pl.range(pl.min(8, tokens - first)):
                token: pl.Scalar[pl.INDEX] = first + row_in_block
                page: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(slots, [token, 0]), pl.INDEX)
                offset: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(slots, [token, 1]), pl.INDEX)
                if page >= 0 and page < cache_pages and offset >= 0 and offset < 32:
                    target: pl.Scalar[pl.INDEX] = page * 32 + offset
                    cache_rows: pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16] = pl.tensor.assemble(cache_rows, pl.tensor.slice(kv, [1, 512], [token, 0]), [target, 0])
        return cache_rows

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def q_dequant_rope_stage(
        self,
        wq_b_scale: pl.Tensor[[32768], pl.FP32],
        rope_cos_il: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_sin_signed: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_swap_idx: pl.Tensor[[CSA_FULL_T, 64], pl.INT32],
        q_flat: pl.Tensor[[CSA_FULL_T, 32768], pl.BF16],
        qr_scale_pad_store: pl.Tensor[[CSA_FULL_P8, 1], pl.FP32],
        q_proj_i32: pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[CSA_FULL_T, 32768], pl.BF16]:
        # 显式 AIV 反量化、RMS、NOPE/RoPE；q_flat 必须是外层建立的 GM 二维别名。
        #
        #         完整八行与尾块沿用不同的既有 RMS 算术；每次工作只写八行中的一个 head 组。
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(q_flat, 0)
        dq_token_tiles: pl.Scalar[pl.INDEX] = (tokens + 8 - 1) // 8
        dq_head_tile: pl.Scalar[pl.INDEX] = 4
        if tokens % 8 == 0:
            if dq_token_tiles >= 48 and dq_token_tiles % 48 == 0:
                dq_head_tile: pl.Scalar[pl.INDEX] = 64
            else:
                if dq_token_tiles >= 24 and dq_token_tiles % 24 == 0:
                    dq_head_tile: pl.Scalar[pl.INDEX] = 32
                else:
                    if dq_token_tiles >= 12 and dq_token_tiles % 12 == 0:
                        dq_head_tile: pl.Scalar[pl.INDEX] = 16
                    else:
                        if dq_token_tiles >= 6 and dq_token_tiles % 6 == 0:
                            dq_head_tile: pl.Scalar[pl.INDEX] = 8
        for dq_work in pl.range(worker_id, (tokens + 8 - 1) // 8 * (64 // dq_head_tile), 48):
            hg: pl.Scalar[pl.INDEX] = dq_work % (64 // dq_head_tile) * dq_head_tile
            tg: pl.Scalar[pl.INDEX] = dq_work // (64 // dq_head_tile) * 8
            if tg + 8 <= tokens:
                qr_scale_dq_t: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(qr_scale_pad_store, [tg, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                q_cos_il: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(rope_cos_il, [tg, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                q_sin_signed: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(rope_sin_signed, [tg, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                q_swap_idx: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.load(rope_swap_idx, [tg, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                q_flat_i: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 512], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode="round")
                q_row_i: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.muls(q_flat_i, 0.015625), target_type=pl.INT32, mode="trunc")
                q_row_starts: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(q_row_i, target_type=pl.FP32, mode="round"), 64.0)
                q_abs_swap: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.cast(
                    pl.tile.add(pl.tile.cast(pl.tile.reshape(q_swap_idx, [1, 512]), target_type=pl.FP32, mode="round"), q_row_starts), target_type=pl.INT32, mode="round"
                )
                q_flat_gather_tmp: pl.Tile[[1, 512], pl.INT32] = pl.tile.create([1, 512], dtype=pl.INT32)
                for h_inner in pl.pipeline(dq_head_tile, stage=2):
                    h: pl.Scalar[pl.INDEX] = hg + h_inner
                    h0: pl.Scalar[pl.INDEX] = h * 512
                    q_head_acc: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(q_proj_i32, [tg, h0], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    q_head_scale: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.load(wq_b_scale, [h0], [512], [512], target_memory=pl.Mem.Vec), [1, 512])
                    q_head_acc_fp32: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(q_head_acc, target_type=pl.FP32, mode="none")
                    q_head_row_scaled: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
                    q_head_dq: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_head_row_scaled, q_head_scale)
                    q_head_sq: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_head_dq, q_head_dq)
                    q_head_sq_row: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(q_head_sq, pl.tile.create([8, 512], dtype=pl.FP32))
                    q_head_sq_sum: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(q_head_sq_row, [1, 8])
                    q_head_sq_mean: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(q_head_sq_sum, 0.001953125)
                    q_head_var: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(q_head_sq_mean, 9.9999999999999995e-07)
                    q_head_inv_rms: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(q_head_var, pl.tile.create([1, 8], dtype=pl.FP32))
                    q_head_inv_rms_t: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(q_head_inv_rms, [8, 1])
                    q_nope_normed: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(pl.tile.slice(q_head_dq, [8, 448], [0, 0]), q_head_inv_rms_t)
                    q_nope_bf16: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                    pl.tile.store(q_nope_bf16, [tg, h0], q_flat)
                    q_rope_chunk_raw: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(q_head_dq, [8, 64], [0, 448])
                    q_rope_chunk: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
                    q_rope_flat: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(q_rope_chunk, [1, 512])
                    q_swapped_flat: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.gather(q_rope_flat, q_abs_swap, q_flat_gather_tmp)
                    q_rope_swapped: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(q_swapped_flat, [8, 64])
                    q_rope_base: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_rope_chunk, q_cos_il)
                    q_rope_delta: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_rope_swapped, q_sin_signed)
                    q_rope_rot: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(q_rope_base, q_rope_delta)
                    q_rope_bf16: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
                    pl.tile.store(q_rope_bf16, [tg, h0 + 448], q_flat)
            else:
                valid_tail_rows: pl.Scalar[pl.INDEX] = tokens - tg
                qr_scale_dq_tail: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(qr_scale_pad_store, [tg, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                q_cos_il_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.load(
                    rope_cos_il, [tg, 0], [8, 64], [valid_tail_rows, 64], target_memory=pl.Mem.Vec
                )
                q_sin_signed_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.load(
                    rope_sin_signed, [tg, 0], [8, 64], [valid_tail_rows, 64], target_memory=pl.Mem.Vec
                )
                q_col: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(
                    pl.tile.full([8, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode="round")
                )
                q_dup_f: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(q_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32, mode="round")
                q_lane: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(q_col, pl.tile.muls(q_dup_f, 2.0))
                q_swap_f: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(q_col, 1.0), pl.tile.muls(q_lane, 2.0))
                q_row_seed: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(
                    pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode="round"), 64.0
                )
                q_row_grid: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 8], dtype=pl.FP32, value=1.0), q_row_seed)
                q_row_offset: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(q_row_grid, 0, 1)
                q_swap_idx_tail: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(q_swap_f, q_row_offset), target_type=pl.INT32, mode="round")
                q_head_reduce_tmp: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                q_gather_tmp: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for h_inner_tail in pl.range(4):
                    h_tail: pl.Scalar[pl.INDEX] = hg + h_inner_tail
                    h0_tail: pl.Scalar[pl.INDEX] = h_tail * 512
                    q_head_acc_tail: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(q_proj_i32, [tg, h0_tail], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    q_head_scale_input_tail: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(wq_b_scale, [h0_tail], [512], [512], target_memory=pl.Mem.Vec)
                    q_head_scale_tail: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(q_head_scale_input_tail, [1, 512])
                    q_head_acc_fp32_tail: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(q_head_acc_tail, target_type=pl.FP32, mode="none")
                    q_head_row_scaled_tail: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(q_head_acc_fp32_tail, qr_scale_dq_tail)
                    q_head_dq_tail: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_head_row_scaled_tail, q_head_scale_tail)
                    q_head_sq_tail: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_head_dq_tail, q_head_dq_tail)
                    q_head_sq_sum_tail: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(q_head_sq_tail, q_head_reduce_tmp)
                    q_head_inv_rms_tail: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.recip(pl.tile.sqrt(pl.tile.adds(pl.tile.muls(q_head_sq_sum_tail, 0.001953125), 9.9999999999999995e-07)))
                    q_nope_normed_tail: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(pl.tile.slice(q_head_dq_tail, [8, 448], [0, 0]), q_head_inv_rms_tail)
                    q_nope_bf16_tail: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(q_nope_normed_tail, target_type=pl.BF16, mode="rint")
                    q_nope_valid: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 448])] = pl.tile.set_validshape(q_nope_bf16_tail, valid_tail_rows, 448)
                    pl.tile.store(q_nope_valid, [tg, h0_tail], q_flat)
                    q_rope_chunk_raw_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(q_head_dq_tail, [8, 64], [0, 448])
                    q_rope_chunk_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(q_rope_chunk_raw_tail, q_head_inv_rms_tail)
                    q_rope_swapped_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(q_rope_chunk_tail, q_swap_idx_tail, q_gather_tmp)
                    q_rope_chunk_tail_v1: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.set_validshape(q_rope_chunk_tail, valid_tail_rows, 64)
                    q_rope_swapped_tail_v1: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.set_validshape(q_rope_swapped_tail, valid_tail_rows, 64)
                    q_rope_base_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.mul(q_rope_chunk_tail_v1, q_cos_il_tail)
                    q_rope_delta_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.mul(q_rope_swapped_tail_v1, q_sin_signed_tail)
                    q_rope_rot_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.add(q_rope_base_tail, q_rope_delta_tail)
                    q_rope_bf16_tail: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.cast(q_rope_rot_tail, target_type=pl.BF16, mode="rint")
                    q_rope_valid: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.set_validshape(q_rope_bf16_tail, valid_tail_rows, 64)
                    pl.tile.store(q_rope_valid, [tg, h0_tail + 448], q_flat)
        return q_flat

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def q_r_stage(
        self,
        qa_fp32: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32],
        gamma_cq: pl.Tensor[[1024], pl.BF16],
        qr: pl.Tensor[[CSA_FULL_T, 1024], pl.INT8],
        qr_scale: pl.Tensor[[CSA_FULL_T, 1], pl.FP32],
        qr_i8_matmul: pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8],
        qr_scale_pad: pl.Tensor[[CSA_FULL_P8, 1], pl.FP32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_T, 1024], pl.INT8], pl.Tensor[[CSA_FULL_T, 1], pl.FP32], pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8], pl.Tensor[[CSA_FULL_P8, 1], pl.FP32]]:
        # 由外层 AIV 调用，复用原八行归约与量化算术。
        #
        #         默认外层可沿用 16 个逻辑 AIV worker。传入的编号必须按 AIV 角色编号，
        #         不能直接混用 AIC 核组号；WORKERS 必须为正。未获有效 tile 的 worker
        #         跳过 gamma 搬运，空 T 也不会进入步长为零的循环。
        #
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(qr, 0)
        token_tiles: pl.Scalar[pl.INDEX] = (tokens + 8 - 1) // 8
        workers: pl.Scalar[pl.INDEX] = pl.min(token_tiles, 48)
        if worker_id < workers:
            gamma_cq_row: pl.Tile[[1, 1024], pl.BF16, pl.Mem.Vec] = pl.tile.reshape(pl.tile.load(gamma_cq, [0], [1024], [1024], target_memory=pl.Mem.Vec), [1, 1024])
            qr_gamma_ub: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.cast(gamma_cq_row, target_type=pl.FP32, mode="round")
            for tile_index in pl.range(worker_id, token_tiles, workers):
                tg: pl.Scalar[pl.INDEX] = tile_index * 8
                valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, tokens - tg)
                qr_input_ub: pl.Tile[[8, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.load(qa_fp32, [tg, 0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
                qr_sq_sum: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                qr_amax_g: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                for qr_rms_col0 in pl.pipeline(0, 1024, 256, stage=2):
                    qr_rms_chunk: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(qr_input_ub, 0, qr_rms_col0, [8, 256], target_memory=pl.Mem.Vec)
                    qr_rms_sq: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.mul(qr_rms_chunk, qr_rms_chunk)
                    qr_rms_row_sum: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.row_sum(qr_rms_sq, pl.tile.create([8, 256], dtype=pl.FP32)), [1, 8])
                    qr_sq_sum: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(qr_sq_sum, qr_rms_row_sum)
                    gamma_rms_chunk: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(qr_gamma_ub, 0, qr_rms_col0, [1, 256], target_memory=pl.Mem.Vec)
                    qr_g: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(qr_rms_chunk, gamma_rms_chunk)
                    qr_g_row_max: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.row_max(pl.tile.abs(qr_g), pl.tile.create([8, 256], dtype=pl.FP32)), [1, 8])
                    qr_amax_g: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(qr_amax_g, qr_g_row_max)
                qr_inv_rms: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.rsqrt(pl.tile.adds(pl.tile.muls(qr_sq_sum, 0.0009765625), 9.9999999999999995e-07), pl.tile.create([1, 8], dtype=pl.FP32))
                qr_inv_rms_t: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(qr_inv_rms, [8, 1])
                qr_amax_floor: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
                qr_tile_amax: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(qr_amax_floor, pl.tile.mul(qr_inv_rms, qr_amax_g))
                qr_scale_quant_row: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.div(pl.tile.full([1, 8], dtype=pl.FP32, value=127.0), qr_tile_amax)
                qr_scale_quant_t: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(qr_scale_quant_row, [8, 1])
                qr_tile_scale_dq: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(pl.tile.recip(qr_scale_quant_row), [8, 1])
                pl.tile.store(qr_tile_scale_dq, [tg, 0], qr_scale_pad)
                if valid_rows == 8:
                    pl.tile.store(qr_tile_scale_dq, [tg, 0], qr_scale)
                else:
                    qr_scale_tail: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 1])] = pl.tile.set_validshape(qr_tile_scale_dq, valid_rows, 1)
                    pl.tile.store(qr_scale_tail, [tg, 0], qr_scale)
                for qa in pl.pipeline(0, 1024, 256, stage=2):
                    qr_chunk: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(qr_input_ub, 0, qa, [8, 256], target_memory=pl.Mem.Vec)
                    gamma_q_chunk: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(qr_gamma_ub, 0, qa, [1, 256], target_memory=pl.Mem.Vec)
                    qr_q_normed: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk)
                    qr_q_scaled: pl.Tile[[8, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(qr_q_normed, qr_scale_quant_t)
                    qr_q_i32: pl.Tile[[8, 256], pl.INT32, pl.Mem.Vec] = pl.tile.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
                    qr_q_half: pl.Tile[[8, 256], pl.FP16, pl.Mem.Vec] = pl.tile.cast(qr_q_i32, target_type=pl.FP16, mode="round")
                    qr_q_i8: pl.Tile[[8, 256], pl.INT8, pl.Mem.Vec] = pl.tile.cast(qr_q_half, target_type=pl.INT8, mode="trunc")
                    pl.tile.store(qr_q_i8, [tg, qa], qr_i8_matmul)
                    if valid_rows == 8:
                        pl.tile.store(qr_q_i8, [tg, qa], qr)
                    else:
                        qr_q_tail: pl.Tile[[8, 256], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 256])] = pl.tile.set_validshape(qr_q_i8, valid_rows, 256)
                        pl.tile.store(qr_q_tail, [tg, qa], qr)
        return qr, qr_scale, qr_i8_matmul, qr_scale_pad

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def q_rope_prepare_stage(
        self,
        rope_sin: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_sin_signed: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_swap_idx: pl.Tensor[[CSA_FULL_T, 64], pl.INT32],
        worker_id: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[CSA_FULL_T, 64], pl.FP32], pl.Tensor[[CSA_FULL_T, 64], pl.INT32]]:
        # 显式 AIV 构造 interleaved sin 符号与行内交换索引；只处理有效 token tile。
        #
        #         这段只依赖本轮 RoPE 输入，可与 QR/Q_B 并行；反量化消费前必须等待发布。
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(rope_sin, 0)
        token_tiles: pl.Scalar[pl.INDEX] = (t_dim + 8 - 1) // 8
        if worker_id < pl.min(48, token_tiles):
            qrp_ones: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.full([8, 64], dtype=pl.FP32, value=1.0)
            qrp_idx_i32: pl.Tensor[[1, 64], pl.INT32] = pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            qrp_idx_fp32: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(qrp_idx_i32, target_type=pl.FP32, mode="round")
            qrp_col: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.col_expand_mul(qrp_ones, qrp_idx_fp32)
            qrp_half: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(qrp_col, 0.5)
            qrp_dup_i32: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_half, target_type=pl.INT32, mode="trunc")
            qrp_dup_f: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_dup_i32, target_type=pl.FP32, mode="round")
            qrp_lane: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(qrp_col, pl.tensor.muls(qrp_dup_f, 2.0))
            qrp_next_col: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(qrp_col, 1.0)
            qrp_lane_offset: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(qrp_lane, 2.0)
            qrp_swap_f: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(qrp_next_col, qrp_lane_offset)
            qrp_swap_idx: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_swap_f, target_type=pl.INT32, mode="round")
            qrp_sign: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.subs(pl.tensor.muls(qrp_lane, 2.0), 1.0)
            for qrp_idx in pl.range(worker_id, token_tiles, pl.min(48, token_tiles)):
                qrp_t0: pl.Scalar[pl.INDEX] = qrp_idx * 8
                qrp_valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, t_dim - qrp_t0)
                if qrp_valid_rows == 8:
                    qrp_sin_rows_full: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.slice(rope_sin, [8, 64], [qrp_t0, 0])
                    qrp_sin_signed_full: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.mul(qrp_sin_rows_full, qrp_sign)
                    rope_sin_signed: pl.Tensor[[CSA_FULL_T, 64], pl.FP32] = pl.tensor.assemble(rope_sin_signed, qrp_sin_signed_full, [qrp_t0, 0])
                    rope_swap_idx: pl.Tensor[[CSA_FULL_T, 64], pl.INT32] = pl.tensor.assemble(rope_swap_idx, qrp_swap_idx, [qrp_t0, 0])
                else:
                    qrp_sin_rows_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows, 64])] = pl.tile.load(
                        rope_sin, [qrp_t0, 0], [8, 64], [qrp_valid_rows, 64], target_memory=pl.Mem.Vec
                    )
                    qrp_tail_col: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(
                        pl.tile.full([8, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode="round")
                    )
                    qrp_tail_dup_f: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(
                        pl.tile.cast(pl.tile.muls(qrp_tail_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32, mode="round"
                    )
                    qrp_tail_lane: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(qrp_tail_col, pl.tile.muls(qrp_tail_dup_f, 2.0))
                    qrp_tail_swap_f: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(qrp_tail_col, 1.0), pl.tile.muls(qrp_tail_lane, 2.0))
                    qrp_tail_sign: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(pl.tile.muls(qrp_tail_lane, 2.0), 1.0)
                    qrp_tail_sign_v1: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows, 64])] = pl.tile.set_validshape(qrp_tail_sign, qrp_valid_rows, 64)
                    qrp_sin_signed_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows, 64])] = pl.tile.mul(qrp_sin_rows_tail, qrp_tail_sign_v1)
                    pl.tile.store(pl.tile.set_validshape(qrp_sin_signed_tail, qrp_valid_rows, 64), [qrp_t0, 0], rope_sin_signed)
                    pl.tile.store(pl.tile.set_validshape(pl.tile.cast(qrp_tail_swap_f, target_type=pl.INT32, mode="round"), qrp_valid_rows, 64), [qrp_t0, 0], rope_swap_idx)
        return rope_sin_signed, rope_swap_idx

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def vector_schedule(
        x: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16],
        hc_scale: pl.Tensor[[3], pl.FP32],
        hc_base: pl.Tensor[[24], pl.FP32],
        hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
        norm_w: pl.Tensor[[4096], pl.BF16],
        gamma_cq: pl.Tensor[[1024], pl.BF16],
        x_fp32: pl.InOut[pl.Tensor[[CSA_FULL_T, 16384], pl.FP32]],
        inv: pl.InOut[pl.Tensor[[CSA_FULL_P16, 1], pl.FP32]],
        widen_tail: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]],
        partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32],
        raw: pl.InOut[pl.Tensor[[CSA_FULL_P16, 32], pl.FP32]],
        pre: pl.InOut[pl.Tensor[[CSA_FULL_P16, 8], pl.FP32]],
        post: pl.InOut[pl.Tensor[[CSA_FULL_T, 4], pl.FP32]],
        post_tail: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[CSA_FULL_T, 16], pl.FP32]],
        comb_tail: pl.InOut[pl.Tensor[[8, 32], pl.FP32]],
        mixed: pl.InOut[pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16]],
        normed: pl.InOut[pl.Tensor[[CSA_FULL_T, 4096], pl.BF16]],
        norm_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
        qa: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32],
        qr: pl.InOut[pl.Tensor[[CSA_FULL_T, 1024], pl.INT8]],
        qr_scale: pl.InOut[pl.Tensor[[CSA_FULL_T, 1], pl.FP32]],
        qr_matmul: pl.InOut[pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8]],
        qr_scale_pad: pl.InOut[pl.Tensor[[CSA_FULL_P8, 1], pl.FP32]],
        wq_b_scale: pl.Tensor[[32768], pl.FP32],
        rope_cos: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_sin: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_signed: pl.InOut[pl.Tensor[[CSA_FULL_T, 64], pl.FP32]],
        rope_swap: pl.InOut[pl.Tensor[[CSA_FULL_T, 64], pl.INT32]],
        q_proj: pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32],
        q_flat: pl.InOut[pl.Tensor[[CSA_FULL_T, 32768], pl.BF16]],
        gamma_ckv: pl.Tensor[[512], pl.BF16],
        kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32],
        kv: pl.InOut[pl.Tensor[[CSA_FULL_T, 512], pl.BF16]],
        native_slots: pl.Tensor[[CSA_FULL_T, 2], pl.INT32],
        cache_rows: pl.InOut[pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16]],
        main_value: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        main_score: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        main_state: pl.InOut[pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32]],
        main_state_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_MTABLE_COLS], pl.INT32],
        main_ape: pl.Tensor[[4, 1024], pl.FP32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        main_bounds: pl.Tensor[[CSA_FULL_MQ], pl.INT32],
        main_lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        main_state_slots: pl.Tensor[[CSA_FULL_T, 2], pl.INT32],
        main_pooled: pl.InOut[pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32]],
        main_gamma: pl.Tensor[[512], pl.BF16],
        main_cos: pl.Tensor[[CSA_FULL_MC, 64], pl.FP32],
        main_sin: pl.Tensor[[CSA_FULL_MC, 64], pl.FP32],
        main_offsets: pl.InOut[pl.Tensor[[CSA_FULL_B], pl.INT32]],
        main_count: pl.InOut[pl.Tensor[[1], pl.INT32]],
        main_slots: pl.Tensor[[CSA_FULL_MSLOTS, 2], pl.INT32],
        main_normed: pl.InOut[pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32]],
        main_debug: pl.InOut[pl.Tensor[[CSA_FULL_T, 512], pl.FP32]],
        main_cache: pl.InOut[pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16]],
        idx_value: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        idx_score: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        idx_state: pl.InOut[pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32]],
        idx_state_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_ITABLE_COLS], pl.INT32],
        idx_ape: pl.Tensor[[4, 256], pl.FP32],
        idx_bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        idx_lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        idx_state_slots: pl.Tensor[[CSA_FULL_T, 2], pl.INT32],
        idx_pooled: pl.InOut[pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32]],
        idx_gamma: pl.Tensor[[128], pl.BF16],
        idx_cos: pl.Tensor[[CSA_FULL_IC, 64], pl.FP32],
        idx_sin: pl.Tensor[[CSA_FULL_IC, 64], pl.FP32],
        idx_offsets: pl.InOut[pl.Tensor[[CSA_FULL_B], pl.INT32]],
        idx_count: pl.InOut[pl.Tensor[[1], pl.INT32]],
        idx_slots: pl.Tensor[[CSA_FULL_ISLOTS, 2], pl.INT32],
        idx_normed: pl.InOut[pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16]],
        idx_had: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32],
        idx_scales: pl.InOut[pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32]],
        idx_debug: pl.InOut[pl.Tensor[[CSA_FULL_T, 128], pl.FP32]],
        idx_cache: pl.InOut[pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8]],
        original_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_ORIGINAL_COLS], pl.INT32],
        compressed_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_COMPRESSED_COLS], pl.INT32],
        indexer_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_INDEXER_COLS], pl.INT32],
        idx_wq_b_scale: pl.Tensor[[8192], pl.FP32],
        idx_query_acc: pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32],
        idx_query_rope: pl.InOut[pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16]],
        idx_query_had: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32],
        idx_query_i8: pl.InOut[pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8]],
        idx_query_scale: pl.InOut[pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32]],
        idx_query_scale_heads: pl.Tensor[[CSA_FULL_IDX_SCALE_T_ROWS, 64], pl.FP32],
        weights_partial: pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32],
        head_weights: pl.InOut[pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32]],
        group_offsets: pl.InOut[pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32]],
        group_plan: pl.InOut[pl.Tensor[[8], pl.INT32]],
        coefficients: pl.InOut[pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16]],
        indexer_transfer: pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32],
        score_arena: pl.InOut[pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32]],
        pair_arena: pl.InOut[pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32]],
        topk_scores: pl.InOut[pl.Tensor[[CSA_FULL_T, 512], pl.FP32]],
        topk_indices: pl.InOut[pl.Tensor[[CSA_FULL_T, 512], pl.INT32]],
        attention_indices: pl.InOut[pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32]],
        attention_bias: pl.InOut[pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32]],
        attention_mask: pl.InOut[pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32]],
        inverse_sin: pl.InOut[pl.Tensor[[CSA_FULL_T, 64], pl.FP32]],
        attn_sink: pl.Tensor[[64, 1], pl.FP32],
        kv_ring: pl.InOut[pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16]],
        score_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32],
        probability_ring: pl.InOut[pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16]],
        pv_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32],
        mi_ring: pl.InOut[pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32]],
        li_ring: pl.InOut[pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32]],
        packed: pl.InOut[pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16]],
        ffts: pl.Tensor[[256], pl.INT64],
        o_r: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32],
        o_quant: pl.InOut[pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8]],
        o_scales: pl.InOut[pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32]],
        o_partials: pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32],
        wo_b_scale: pl.Tensor[[4096], pl.FP32],
        output: pl.InOut[pl.Tensor[[CSA_FULL_T, 16384], pl.BF16]],
        signal: pl.InOut[pl.Tensor[[CSA_FULL_WORDS], pl.INT32]],
    ):
        pl.func_attr({"split_aiv": True})
        x_fp32_gm: pl.Tensor[[CSA_FULL_T, 16384], pl.FP32] = x_fp32
        inv_gm: pl.Tensor[[CSA_FULL_P16, 1], pl.FP32] = inv
        widen_tail_gm: pl.Tensor[[8, 16384], pl.FP32] = widen_tail
        raw_gm: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32] = raw
        pre_gm: pl.Tensor[[CSA_FULL_P16, 8], pl.FP32] = pre
        post_gm: pl.Tensor[[CSA_FULL_T, 4], pl.FP32] = post
        post_tail_gm: pl.Tensor[[8, 8], pl.FP32] = post_tail
        comb_gm: pl.Tensor[[CSA_FULL_T, 16], pl.FP32] = comb
        comb_tail_gm: pl.Tensor[[8, 32], pl.FP32] = comb_tail
        mixed_gm: pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16] = mixed
        normed_gm: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16] = normed
        norm_tail_gm: pl.Tensor[[8, 4096], pl.BF16] = norm_tail
        qr_gm: pl.Tensor[[CSA_FULL_T, 1024], pl.INT8] = qr
        qr_scale_gm: pl.Tensor[[CSA_FULL_T, 1], pl.FP32] = qr_scale
        qr_matmul_gm: pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8] = qr_matmul
        qr_scale_pad_gm: pl.Tensor[[CSA_FULL_P8, 1], pl.FP32] = qr_scale_pad
        rope_signed_gm: pl.Tensor[[CSA_FULL_T, 64], pl.FP32] = rope_signed
        rope_swap_gm: pl.Tensor[[CSA_FULL_T, 64], pl.INT32] = rope_swap
        q_flat_gm: pl.Tensor[[CSA_FULL_T, 32768], pl.BF16] = q_flat
        kv_gm: pl.Tensor[[CSA_FULL_T, 512], pl.BF16] = kv
        cache_rows_gm: pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16] = cache_rows
        main_state_gm: pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32] = main_state
        main_pooled_gm: pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32] = main_pooled
        main_offsets_gm: pl.Tensor[[CSA_FULL_B], pl.INT32] = main_offsets
        main_count_gm: pl.Tensor[[1], pl.INT32] = main_count
        main_normed_gm: pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32] = main_normed
        main_debug_gm: pl.Tensor[[CSA_FULL_T, 512], pl.FP32] = main_debug
        main_cache_gm: pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16] = main_cache
        idx_state_gm: pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32] = idx_state
        idx_pooled_gm: pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32] = idx_pooled
        idx_offsets_gm: pl.Tensor[[CSA_FULL_B], pl.INT32] = idx_offsets
        idx_count_gm: pl.Tensor[[1], pl.INT32] = idx_count
        idx_normed_gm: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16] = idx_normed
        idx_scales_gm: pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32] = idx_scales
        idx_debug_gm: pl.Tensor[[CSA_FULL_T, 128], pl.FP32] = idx_debug
        idx_cache_gm: pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8] = idx_cache
        idx_query_rope_gm: pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16] = idx_query_rope
        idx_query_i8_gm: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8] = idx_query_i8
        idx_query_scale_gm: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32] = idx_query_scale
        head_weights_gm: pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32] = head_weights
        group_offsets_gm: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32] = group_offsets
        group_plan_gm: pl.Tensor[[8], pl.INT32] = group_plan
        coefficients_gm: pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16] = coefficients
        score_arena_gm: pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32] = score_arena
        pair_arena_gm: pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32] = pair_arena
        topk_scores_gm: pl.Tensor[[CSA_FULL_T, 512], pl.FP32] = topk_scores
        topk_indices_gm: pl.Tensor[[CSA_FULL_T, 512], pl.INT32] = topk_indices
        attention_indices_gm: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32] = attention_indices
        attention_bias_gm: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32] = attention_bias
        attention_mask_gm: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32] = attention_mask
        inverse_sin_gm: pl.Tensor[[CSA_FULL_T, 64], pl.FP32] = inverse_sin
        kv_ring_gm: pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16] = kv_ring
        probability_ring_gm: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16] = probability_ring
        mi_ring_gm: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32] = mi_ring
        li_ring_gm: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32] = li_ring
        packed_gm: pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16] = packed
        o_quant_gm: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8] = o_quant
        o_scales_gm: pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32] = o_scales
        output_gm: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16] = output
        signal_gm: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = signal
        worker: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
        core: pl.Scalar[pl.INDEX] = 24 + worker
        _tuple_tmp: pl.Tuple[pl.Tensor[[CSA_FULL_WORDS], pl.INT32], pl.Scalar[pl.INT32]] = self.begin_invocation(signal, core)
        signal: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = _tuple_tmp[0]
        generation: pl.Scalar[pl.INT32] = _tuple_tmp[1]
        pl.system.set_ffts(ffts)
        tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        _tuple_tmp_1: pl.Tuple[pl.Tensor[[CSA_FULL_T, 16384], pl.FP32], pl.Tensor[[CSA_FULL_P16, 1], pl.FP32]] = self.hc_widen_rms_range(x, x_fp32, inv, widen_tail, worker, 48)
        x_fp32: pl.Tensor[[CSA_FULL_T, 16384], pl.FP32] = _tuple_tmp_1[0]
        inv: pl.Tensor[[CSA_FULL_P16, 1], pl.FP32] = _tuple_tmp_1[1]
        signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal, 1, core, 72, generation)
        planner: pl.Scalar[pl.INDEX] = pl.min((tokens + 7) // 8, 47)
        if worker == planner:
            _tuple_tmp_2: pl.Tuple[pl.Tensor[[CSA_FULL_B], pl.INT32], pl.Tensor[[1], pl.INT32]] = self.compact_offsets_and_count_stage(main_bounds, main_lengths, main_offsets, main_count, tokens, 0)
            main_offsets: pl.Tensor[[CSA_FULL_B], pl.INT32] = _tuple_tmp_2[0]
            main_count: pl.Tensor[[1], pl.INT32] = _tuple_tmp_2[1]
            _tuple_tmp_3: pl.Tuple[pl.Tensor[[CSA_FULL_B], pl.INT32], pl.Tensor[[1], pl.INT32]] = self.compact_offsets_and_count_stage(idx_bounds, idx_lengths, idx_offsets, idx_count, tokens, 0)
            idx_offsets: pl.Tensor[[CSA_FULL_B], pl.INT32] = _tuple_tmp_3[0]
            idx_count: pl.Tensor[[1], pl.INT32] = _tuple_tmp_3[1]
            _tuple_tmp_4: pl.Tuple[pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32], pl.Tensor[[8], pl.INT32]] = self.indexer_group_plan_stage(
                idx_bounds, idx_lengths, positions, group_offsets, group_plan, tokens, 0
            )
            group_offsets: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32] = _tuple_tmp_4[0]
            group_plan: pl.Tensor[[8], pl.INT32] = _tuple_tmp_4[1]
            group_offsets: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32] = self.indexer_scale_page_plan_stage(
                idx_bounds, idx_lengths, positions, idx_offsets, idx_slots, idx_cache, group_offsets, pl.tensor.dim(idx_scales, 0)
            )
            signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v1, 15, core, 72, generation)
        if False:
            if worker < (tokens + 7) // 8:
                signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v1, 2, 72, generation, worker // 2 * 4, 4, 1, 24, 0, 1)
        else:
            if True:
                if worker < (tokens + 15) // 16:
                    signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v1, 2, 72, generation, worker * 4, 4, 1, 24, 0, 1)
            else:
                projection_tasks: pl.Scalar[pl.INDEX] = (tokens + 15) // 16 * 4
                signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v1, 2, 72, generation, 0, projection_tasks, 1, 24, 0, 1)
            raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32] = self.hc_reduce_range(partials, raw, worker, 48)
            if worker < (tokens + 15) // 16:
                signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v1, 3, core, 72, generation)
            if True:
                hc_gate_tiles: pl.Scalar[pl.INDEX] = (tokens + 7) // 8
                if worker < hc_gate_tiles:
                    signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v1, 3, 72, generation, worker // 2, 1, 1, 48, 24, 1)
                    if worker + 48 < hc_gate_tiles:
                        signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v1, 3, 72, generation, worker // 2 + 24, 1, 1, 48, 24, 1)
            else:
                signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v1, 3, 72, generation, 0, tokens, 16, 48, 24, 1)
                signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v1, 1, 72, generation, 0, tokens, 8, 48, 24, 1)
        _tuple_tmp_5: pl.Tuple[pl.Tensor[[CSA_FULL_P16, 8], pl.FP32], pl.Tensor[[CSA_FULL_T, 4], pl.FP32]] = self.hc_gates_range(
            inv, raw, hc_scale, hc_base, pre, post, post_tail, worker, 48, partials
        )
        pre: pl.Tensor[[CSA_FULL_P16, 8], pl.FP32] = _tuple_tmp_5[0]
        post: pl.Tensor[[CSA_FULL_T, 4], pl.FP32] = _tuple_tmp_5[1]
        if worker < (tokens + 7) // 8:
            signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v1, 4, core, 72, generation)
        _tuple_tmp_6: pl.Tuple[pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16], pl.Tensor[[CSA_FULL_T, 4096], pl.BF16]] = self.hc_mix_norm_range(x_fp32, pre, norm_w, mixed, normed, norm_tail, worker, 48)
        mixed: pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16] = _tuple_tmp_6[0]
        normed: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16] = _tuple_tmp_6[1]
        if worker < (tokens + 7) // 8:
            signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v1, 6, core, 72, generation)
        comb_v1: pl.Tensor[[CSA_FULL_T, 16], pl.FP32] = self.hc_comb_range(inv, raw, hc_scale, hc_base_2d, comb, comb_tail, worker, 48, partials)
        if worker < (tokens + 7) // 8:
            signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v1, 5, core, 72, generation)
        if False:
            qa_tasks: pl.Scalar[pl.INDEX] = pl.min(24, 8 * ((tokens + 15) // 16))
            signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v1, 7, 72, generation, 0, qa_tasks, 1, 24, 0, 1)
        else:
            if tokens < 128:
                qa_tasks: pl.Scalar[pl.INDEX] = 8 * pl.min(3, pl.max(1, tokens // 32))
                signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v1, 7, 72, generation, 0, qa_tasks, 1, 24, 0, 1)
            else:
                qa_tasks: pl.Scalar[pl.INDEX] = 8 * pl.min(3, pl.max(1, tokens // 64))
                signal_v1: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v1, 7, 72, generation, 0, qa_tasks, 1, 24, 0, 1)
        _tuple_tmp_7: pl.Tuple[pl.Tensor[[CSA_FULL_T, 1024], pl.INT8], pl.Tensor[[CSA_FULL_T, 1], pl.FP32], pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8], pl.Tensor[[CSA_FULL_P8, 1], pl.FP32]] = (
            self.q_r_stage(qa, gamma_cq, qr, qr_scale, qr_matmul, qr_scale_pad, worker)
        )
        qr: pl.Tensor[[CSA_FULL_T, 1024], pl.INT8] = _tuple_tmp_7[0]
        qr_scale: pl.Tensor[[CSA_FULL_T, 1], pl.FP32] = _tuple_tmp_7[1]
        qr_matmul: pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8] = _tuple_tmp_7[2]
        qr_scale_pad: pl.Tensor[[CSA_FULL_P8, 1], pl.FP32] = _tuple_tmp_7[3]
        signal_v2: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v1, 8, core, 72, generation)
        _tuple_tmp_8: pl.Tuple[pl.Tensor[[CSA_FULL_T, 64], pl.FP32], pl.Tensor[[CSA_FULL_T, 64], pl.INT32]] = self.q_rope_prepare_stage(rope_sin, rope_signed, rope_swap, worker)
        rope_signed: pl.Tensor[[CSA_FULL_T, 64], pl.FP32] = _tuple_tmp_8[0]
        rope_swap: pl.Tensor[[CSA_FULL_T, 64], pl.INT32] = _tuple_tmp_8[1]
        inverse_sin_v1: pl.Tensor[[CSA_FULL_T, 64], pl.FP32] = self.attention_inverse_rope_sign_stage(rope_sin, inverse_sin, worker)
        signal_v3: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v2, 9, core, 72, generation)
        signal_v4: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v3, 9, 72, generation, 0, tokens, 8, 48, 24, 1)
        signal_v5: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v4, 8, 72, generation, 0, tokens, 8, 48, 24, 1)
        signal_v6: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v5, 16, 72, generation, 0, 32, 1, 24, 0, 1)
        idx_query_rope_v1: pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16] = self.indexer_query_dequant_rope_stage(
            qr_scale, idx_wq_b_scale, rope_cos, rope_signed, idx_query_acc, idx_query_rope, worker
        )
        signal_v7: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v6, 17, core, 72, generation)
        weights_tasks: pl.Scalar[pl.INDEX] = (tokens + 15) // 16 * 4
        signal_v8: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v7, 20, 72, generation, 0, weights_tasks, 1, 24, 0, 1)
        head_weights_v1: pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32] = self.indexer_weights_reduce_stage(weights_partial, head_weights, tokens, worker)
        signal_v9: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v8, 21, core, 72, generation)
        signal_v10: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_producer(signal_v9, 15, 24 + planner, 72, generation)
        idx_requests: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_lengths, 0)
        idx_tasks: pl.Scalar[pl.INDEX] = (tokens + 15) // 16 * 8
        signal_v11: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v10, 23, 72, generation, 0, idx_tasks, 1, 24, 0, 1)
        idx_pooled_v1: pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32] = self.indexer_compressor_pool_stage(
            idx_value, idx_score, idx_state, idx_state_table, idx_ape, positions, idx_bounds, idx_lengths, idx_pooled, worker
        )
        signal_v12: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v11, 27, core, 72, generation)
        signal_v13: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v12, 27, 72, generation, 0, idx_requests, 1, 48, 24, 1)
        idx_state_v1: pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32] = self.indexer_compressor_state_commit_stage(
            idx_value, idx_score, idx_ape, positions, idx_bounds, idx_lengths, idx_state_slots, idx_state, worker
        )
        signal_v14: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v13, 28, core, 72, generation)
        idx_normed_v1: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16] = self.indexer_compressor_norm_rope_stage(
            idx_pooled_v1, idx_gamma, idx_cos, idx_sin, positions, idx_bounds, idx_lengths, idx_offsets, idx_normed, worker
        )
        signal_v15: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v14, 29, core, 72, generation)
        first_segment: pl.Scalar[pl.INDEX] = pl.min(tokens, 512)
        if first_segment < 128:
            kv_tasks: pl.Scalar[pl.INDEX] = 4 * pl.min(3, pl.max(1, first_segment // 32))
            signal_v15: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v15, 12, 72, generation, 0, kv_tasks, 1, 24, 0, 1)
        else:
            kv_tasks: pl.Scalar[pl.INDEX] = 4 * pl.min(3, pl.max(1, first_segment // 64))
            signal_v15: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v15, 12, 72, generation, 0, kv_tasks, 1, 24, 0, 1)
        signal_v16: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v15, 9, 72, generation, 0, tokens, 8, 48, 24, 1)
        kv_v1: pl.Tensor[[CSA_FULL_T, 512], pl.BF16] = self.kv_norm_rope_stage(kv_fp32, gamma_ckv, rope_cos, rope_signed, rope_swap, kv, worker)
        signal_v17: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v16, 13, core, 72, generation)
        signal_v18: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v17, 13, 72, generation, 0, tokens, 32, 48, 24, 1)
        cache_rows_v1: pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16] = self.original_kv_writeback_stage(kv_v1, native_slots, cache_rows, worker)
        signal_v19: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v18, 14, core, 72, generation)
        if worker < tokens:
            signal_v19: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_producer(signal_v19, 18, worker % 24, 72, generation)
        _tuple_tmp_9: pl.Tuple[pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8], pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32]] = self.indexer_query_quant_stage(
            idx_query_had, idx_query_i8, idx_query_scale, tokens, worker
        )
        idx_query_i8: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8] = _tuple_tmp_9[0]
        idx_query_scale: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32] = _tuple_tmp_9[1]
        signal_v20: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v19, 19, core, 72, generation)
        signal_v21: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v20, 21, 72, generation, 0, tokens, 16, 48, 24, 1)
        signal_v22: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v21, 19, 72, generation, 0, tokens, 1, 48, 24, 1)
        coefficients_v1: pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16] = self.indexer_head_coefficients_stage(
            idx_query_scale_heads, head_weights_v1, positions, idx_bounds, idx_lengths, group_offsets, coefficients, worker
        )
        signal_v23: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v22, 33, core, 72, generation)
        compact_count: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(idx_count, [0]), pl.INDEX)
        had_tasks: pl.Scalar[pl.INDEX] = (compact_count + 15) // 16 * 2
        signal_v24: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v23, 30, 72, generation, 0, had_tasks, 1, 24, 0, 1)
        _tuple_tmp_10: pl.Tuple[pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32], pl.Tensor[[CSA_FULL_T, 128], pl.FP32], pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8]] = (
            self.indexer_compact_quant_cache_stage(idx_had, positions, idx_bounds, idx_lengths, idx_offsets, idx_slots, idx_scales, idx_debug, idx_cache, worker)
        )
        idx_scales: pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32] = _tuple_tmp_10[0]
        idx_debug: pl.Tensor[[CSA_FULL_T, 128], pl.FP32] = _tuple_tmp_10[1]
        idx_cache: pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8] = _tuple_tmp_10[2]
        signal_v25: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v24, 31, core, 72, generation)
        signal_v26: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v25, 31, 72, generation, 0, idx_requests, 1, 48, 24, 1)
        idx_cache_v1: pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8] = self.indexer_scale_page_commit_stage(idx_scales, group_offsets, idx_cache, tokens, idx_requests, worker)
        signal_v27: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v26, 32, core, 72, generation)
        signal_v28: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v27, 10, 72, generation, 0, 128, 1, 24, 0, 1)
        q_flat_v1: pl.Tensor[[CSA_FULL_T, 32768], pl.BF16] = self.q_dequant_rope_stage(wq_b_scale, rope_cos, rope_signed, rope_swap, q_flat, qr_scale_pad, q_proj, worker)
        signal_v29: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v28, 11, core, 72, generation)
        main_requests: pl.Scalar[pl.INDEX] = pl.tensor.dim(main_lengths, 0)
        main_tasks: pl.Scalar[pl.INDEX] = (tokens + 63) // 64 * 16
        signal_v30: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v29, 22, 72, generation, 0, main_tasks, 1, 24, 0, 1)
        main_pooled_v1: pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32] = self.compressor_pool_stage(
            main_value, main_score, main_state, main_state_table, main_ape, positions, main_bounds, main_lengths, main_pooled, worker
        )
        signal_v31: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v30, 24, core, 72, generation)
        signal_v32: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v31, 24, 72, generation, 0, main_requests, 1, 48, 24, 1)
        main_state_v1: pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32] = self.compressor_state_commit_stage(
            main_value, main_score, main_ape, positions, main_bounds, main_lengths, main_state_slots, main_state, worker
        )
        signal_v33: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v32, 25, core, 72, generation)
        _tuple_tmp_11: pl.Tuple[pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32], pl.Tensor[[CSA_FULL_T, 512], pl.FP32], pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16]] = (
            self.compressor_norm_rope_cache_stage(
                main_pooled_v1, main_gamma, main_cos, main_sin, main_bounds, main_lengths, positions, main_offsets, main_slots, main_normed, main_debug, main_cache, worker
            )
        )
        main_normed: pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32] = _tuple_tmp_11[0]
        main_debug: pl.Tensor[[CSA_FULL_T, 512], pl.FP32] = _tuple_tmp_11[1]
        main_cache: pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16] = _tuple_tmp_11[2]
        signal_v34: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v33, 26, core, 72, generation)
        signal_v35: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v34, 32, 72, generation, 0, 48, 1, 48, 24, 1)
        _tuple_tmp_12: pl.Tuple[pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32], pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32]] = self.indexer_score_vector_lane_stage(
            idx_cache_v1, indexer_table, positions, idx_lengths, idx_bounds, group_offsets, group_plan, indexer_transfer, score_arena, pair_arena, pl.tile.get_subblock_idx(), pl.tile.get_block_idx()
        )
        score_arena: pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32] = _tuple_tmp_12[0]
        pair_arena: pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32] = _tuple_tmp_12[1]
        signal_v36: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v35, 35, core, 72, generation)
        signal_v37: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v36, 35, 72, generation, 0, 48, 1, 48, 24, 1)
        _tuple_tmp_13: pl.Tuple[pl.Tensor[[CSA_FULL_T, 512], pl.FP32], pl.Tensor[[CSA_FULL_T, 512], pl.INT32]] = self.indexer_pair_publish_stage(
            idx_bounds, idx_lengths, positions, group_plan, pair_arena, topk_scores, topk_indices, worker
        )
        topk_scores: pl.Tensor[[CSA_FULL_T, 512], pl.FP32] = _tuple_tmp_13[0]
        topk_indices: pl.Tensor[[CSA_FULL_T, 512], pl.INT32] = _tuple_tmp_13[1]
        signal_v38: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v37, 36, core, 72, generation)
        signal_v39: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v38, 36, 72, generation, 0, tokens, 1, 48, 24, 1)
        _tuple_tmp_14: pl.Tuple[pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32], pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32], pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32]] = (
            self.attention_slot_plan_stage(
                topk_indices,
                positions,
                idx_bounds,
                idx_lengths,
                original_table,
                compressed_table,
                attention_indices,
                attention_bias,
                attention_mask,
                pl.tensor.dim(cache_rows_v1, 0) // 32,
                pl.tensor.dim(main_cache, 0) // 32,
                worker,
            )
        )
        attention_indices: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32] = _tuple_tmp_14[0]
        attention_bias: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32] = _tuple_tmp_14[1]
        attention_mask: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32] = _tuple_tmp_14[2]
        signal_v40: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v39, 37, core, 72, generation)
        signal_v41: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v40, 37, 72, generation, 0, tokens, 8, 48, 24, 1)
        signal_v42: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v41, 14, 72, generation, 0, tokens, 8, 48, 24, 1)
        signal_v43: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v42, 26, 72, generation, 0, main_requests, 1, 48, 24, 1)
        signal_v44: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v43, 9, 72, generation, 0, tokens, 8, 48, 24, 1)
        _tuple_tmp_15: pl.Tuple[
            pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16],
            pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16],
            pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
            pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
            pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16],
        ] = self.attention_qk_pv_vector_lane_stage(
            cache_rows_v1,
            main_cache,
            original_table,
            compressed_table,
            idx_bounds,
            idx_lengths,
            positions,
            attention_indices,
            attention_bias,
            attention_mask,
            attn_sink,
            rope_cos,
            inverse_sin_v1,
            kv_ring,
            score_ring,
            probability_ring,
            pv_ring,
            mi_ring,
            li_ring,
            packed,
            ffts,
            pl.tile.get_block_idx(),
            pl.tile.get_subblock_idx(),
        )
        kv_ring: pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16] = _tuple_tmp_15[0]
        probability_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16] = _tuple_tmp_15[1]
        mi_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32] = _tuple_tmp_15[2]
        li_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32] = _tuple_tmp_15[3]
        packed: pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16] = _tuple_tmp_15[4]
        signal_v45: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v44, 39, core, 72, generation)
        if True:
            oa_per_group: pl.Scalar[pl.INDEX] = (tokens + 127) // 128 * 8
            quant_per_group: pl.Scalar[pl.INDEX] = (tokens + 31) // 32
            for quant_group in pl.range(8):
                quant_first: pl.Scalar[pl.INDEX] = quant_group * quant_per_group
                quant_owned: pl.Scalar[pl.INDEX] = worker + (quant_first + 48 - 1 - worker) // 48 * 48
                if quant_owned < quant_first + quant_per_group:
                    signal_v45: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v45, 44 + quant_group, 72, generation, quant_group * oa_per_group, oa_per_group, 1, 24, 0, 1)
                    _tuple_tmp_16: pl.Tuple[pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8], pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32]] = self.o_projection_groups_quant_stage(
                        o_r, o_quant, o_scales, tokens, worker, quant_group, quant_group + 1
                    )
                    o_quant: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8] = _tuple_tmp_16[0]
                    o_scales: pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32] = _tuple_tmp_16[1]
                    signal_v45: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v45, 52 + quant_group, core, 72, generation)
        else:
            oa_tasks: pl.Scalar[pl.INDEX] = 8 * ((tokens + 127) // 128) * 8
            signal_v45: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v45, 40, 72, generation, 0, oa_tasks, 1, 24, 0, 1)
            _tuple_tmp_17: pl.Tuple[pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8], pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32]] = self.o_projection_group_quant_stage(
                o_r, o_quant, o_scales, tokens, worker
            )
            o_quant: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8] = _tuple_tmp_17[0]
            o_scales: pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32] = _tuple_tmp_17[1]
            signal_v45: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v45, 41, core, 72, generation)
        signal_v46: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.wait_row_owners(signal_v45, 42, 72, generation, 0, 64, 1, 24, 0, 1)
        output_v1: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16] = self.o_dequant_hc_post_stage(o_partials, o_scales, wo_b_scale, x, post, comb_v1, output, worker)
        signal_v47: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = self.publish_phase(signal_v46, 43, core, 72, generation)
        return (
            x_fp32_gm,
            inv_gm,
            widen_tail_gm,
            raw_gm,
            pre_gm,
            post_gm,
            post_tail_gm,
            comb_gm,
            comb_tail_gm,
            mixed_gm,
            normed_gm,
            norm_tail_gm,
            qr_gm,
            qr_scale_gm,
            qr_matmul_gm,
            qr_scale_pad_gm,
            rope_signed_gm,
            rope_swap_gm,
            q_flat_gm,
            kv_gm,
            cache_rows_gm,
            main_state_gm,
            main_pooled_gm,
            main_offsets_gm,
            main_count_gm,
            main_normed_gm,
            main_debug_gm,
            main_cache_gm,
            idx_state_gm,
            idx_pooled_gm,
            idx_offsets_gm,
            idx_count_gm,
            idx_normed_gm,
            idx_scales_gm,
            idx_debug_gm,
            idx_cache_gm,
            idx_query_rope_gm,
            idx_query_i8_gm,
            idx_query_scale_gm,
            head_weights_gm,
            group_offsets_gm,
            group_plan_gm,
            coefficients_gm,
            score_arena_gm,
            pair_arena_gm,
            topk_scores_gm,
            topk_indices_gm,
            attention_indices_gm,
            attention_bias_gm,
            attention_mask_gm,
            inverse_sin_gm,
            kv_ring_gm,
            probability_ring_gm,
            mi_ring_gm,
            li_ring_gm,
            packed_gm,
            o_quant_gm,
            o_scales_gm,
            output_gm,
            signal_gm,
        )

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def _csa_core_group(
        self,
        x: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16],
        hc_fn: pl.Tensor[[24, 16384], pl.FP32],
        hc_scale: pl.Tensor[[3], pl.FP32],
        hc_base: pl.Tensor[[24], pl.FP32],
        hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
        norm_w: pl.Tensor[[4096], pl.BF16],
        wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        gamma_cq: pl.Tensor[[1024], pl.BF16],
        x_fp32: pl.Tensor[[CSA_FULL_T, 16384], pl.FP32],
        inv: pl.Tensor[[CSA_FULL_P16, 1], pl.FP32],
        widen_tail: pl.Tensor[[8, 16384], pl.FP32],
        partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32],
        raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32],
        pre: pl.Tensor[[CSA_FULL_P16, 8], pl.FP32],
        post: pl.Tensor[[CSA_FULL_T, 4], pl.FP32],
        post_tail: pl.Tensor[[8, 8], pl.FP32],
        comb: pl.Tensor[[CSA_FULL_T, 16], pl.FP32],
        comb_tail: pl.Tensor[[8, 32], pl.FP32],
        mixed: pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16],
        normed: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        norm_tail: pl.Tensor[[8, 4096], pl.BF16],
        qa: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32],
        qr: pl.Tensor[[CSA_FULL_T, 1024], pl.INT8],
        qr_scale: pl.Tensor[[CSA_FULL_T, 1], pl.FP32],
        qr_matmul: pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8],
        qr_scale_pad: pl.Tensor[[CSA_FULL_P8, 1], pl.FP32],
        wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        wq_b_scale: pl.Tensor[[32768], pl.FP32],
        rope_cos: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_sin: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_signed: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_swap: pl.Tensor[[CSA_FULL_T, 64], pl.INT32],
        q_proj: pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32],
        q_flat: pl.Tensor[[CSA_FULL_T, 32768], pl.BF16],
        wkv: pl.Tensor[[4096, 512], pl.BF16],
        gamma_ckv: pl.Tensor[[512], pl.BF16],
        kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32],
        kv: pl.Tensor[[CSA_FULL_T, 512], pl.BF16],
        native_slots: pl.Tensor[[CSA_FULL_T, 2], pl.INT32],
        cache_rows: pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16],
        main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
        main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
        main_value: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        main_score: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        main_state: pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32],
        main_state_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_MTABLE_COLS], pl.INT32],
        main_ape: pl.Tensor[[4, 1024], pl.FP32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        main_bounds: pl.Tensor[[CSA_FULL_MQ], pl.INT32],
        main_lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        main_state_slots: pl.Tensor[[CSA_FULL_T, 2], pl.INT32],
        main_pooled: pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32],
        main_gamma: pl.Tensor[[512], pl.BF16],
        main_cos: pl.Tensor[[CSA_FULL_MC, 64], pl.FP32],
        main_sin: pl.Tensor[[CSA_FULL_MC, 64], pl.FP32],
        main_offsets: pl.Tensor[[CSA_FULL_B], pl.INT32],
        main_count: pl.Tensor[[1], pl.INT32],
        main_slots: pl.Tensor[[CSA_FULL_MSLOTS, 2], pl.INT32],
        main_normed: pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32],
        main_debug: pl.Tensor[[CSA_FULL_T, 512], pl.FP32],
        main_cache: pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16],
        idx_wkv: pl.Tensor[[256, 4096], pl.BF16],
        idx_wgate: pl.Tensor[[256, 4096], pl.BF16],
        idx_value: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        idx_score: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        idx_state: pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32],
        idx_state_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_ITABLE_COLS], pl.INT32],
        idx_ape: pl.Tensor[[4, 256], pl.FP32],
        idx_bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        idx_lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        idx_state_slots: pl.Tensor[[CSA_FULL_T, 2], pl.INT32],
        idx_pooled: pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32],
        idx_gamma: pl.Tensor[[128], pl.BF16],
        idx_cos: pl.Tensor[[CSA_FULL_IC, 64], pl.FP32],
        idx_sin: pl.Tensor[[CSA_FULL_IC, 64], pl.FP32],
        idx_offsets: pl.Tensor[[CSA_FULL_B], pl.INT32],
        idx_count: pl.Tensor[[1], pl.INT32],
        idx_slots: pl.Tensor[[CSA_FULL_ISLOTS, 2], pl.INT32],
        idx_normed: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16],
        hadamard: pl.Tensor[[128, 128], pl.BF16],
        idx_had: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32],
        idx_scales: pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32],
        idx_debug: pl.Tensor[[CSA_FULL_T, 128], pl.FP32],
        idx_cache: pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8],
        q_heads: pl.Tensor[[CSA_FULL_QUERY_HEAD_ROWS, 512], pl.BF16],
        original_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_ORIGINAL_COLS], pl.INT32],
        compressed_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_COMPRESSED_COLS], pl.INT32],
        indexer_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_INDEXER_COLS], pl.INT32],
        idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8],
        idx_wq_b_scale: pl.Tensor[[8192], pl.FP32],
        idx_query_acc: pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32],
        idx_query_rope: pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16],
        idx_query_rope_heads: pl.Tensor[[CSA_FULL_IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
        idx_query_had: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32],
        idx_query_i8: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8],
        idx_query_scale: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32],
        idx_query_scale_heads: pl.Tensor[[CSA_FULL_IDX_SCALE_T_ROWS, 64], pl.FP32],
        weights_proj: pl.Tensor[[4096, 64], pl.BF16],
        weights_partial: pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32],
        head_weights: pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32],
        group_offsets: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32],
        group_plan: pl.Tensor[[8], pl.INT32],
        coefficients: pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16],
        idx_key0: pl.Tensor[[CSA_FULL_KEY0_ROWS, 128], pl.INT8],
        idx_key64: pl.Tensor[[CSA_FULL_KEY64_ROWS, 128], pl.INT8],
        indexer_transfer: pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32],
        score_arena: pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32],
        pair_arena: pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32],
        topk_scores: pl.Tensor[[CSA_FULL_T, 512], pl.FP32],
        topk_indices: pl.Tensor[[CSA_FULL_T, 512], pl.INT32],
        attention_indices: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32],
        attention_bias: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32],
        attention_mask: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32],
        inverse_sin: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        attn_sink: pl.Tensor[[64, 1], pl.FP32],
        kv_ring: pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16],
        score_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32],
        probability_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16],
        pv_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32],
        mi_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
        li_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
        packed: pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16],
        ffts: pl.Tensor[[256], pl.INT64],
        wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        o_r: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32],
        o_quant: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8],
        o_scales: pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32],
        wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        o_partials: pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32],
        wo_b_scale: pl.Tensor[[4096], pl.FP32],
        output: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16],
        signal: pl.Tensor[[CSA_FULL_WORDS], pl.INT32],
    ) -> tuple[
        pl.Tensor[[CSA_FULL_T, 16384], pl.FP32],
        pl.Tensor[[CSA_FULL_P16, 1], pl.FP32],
        pl.Tensor[[8, 16384], pl.FP32],
        pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32],
        pl.Tensor[[CSA_FULL_P16, 32], pl.FP32],
        pl.Tensor[[CSA_FULL_P16, 8], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 4], pl.FP32],
        pl.Tensor[[8, 8], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 16], pl.FP32],
        pl.Tensor[[8, 32], pl.FP32],
        pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16],
        pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        pl.Tensor[[8, 4096], pl.BF16],
        pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 1024], pl.INT8],
        pl.Tensor[[CSA_FULL_T, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8],
        pl.Tensor[[CSA_FULL_P8, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 64], pl.INT32],
        pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32],
        pl.Tensor[[CSA_FULL_T, 32768], pl.BF16],
        pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 512], pl.BF16],
        pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16],
        pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32],
        pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32],
        pl.Tensor[[CSA_FULL_B], pl.INT32],
        pl.Tensor[[1], pl.INT32],
        pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 512], pl.FP32],
        pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16],
        pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32],
        pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32],
        pl.Tensor[[CSA_FULL_B], pl.INT32],
        pl.Tensor[[1], pl.INT32],
        pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16],
        pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32],
        pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 128], pl.FP32],
        pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8],
        pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32],
        pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16],
        pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32],
        pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8],
        pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32],
        pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32],
        pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32],
        pl.Tensor[[8], pl.INT32],
        pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16],
        pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32],
        pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32],
        pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 512], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 512], pl.INT32],
        pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32],
        pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32],
        pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32],
        pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16],
        pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32],
        pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16],
        pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32],
        pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16],
        pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32],
        pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8],
        pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32],
        pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32],
        pl.Tensor[[CSA_FULL_T, 16384], pl.BF16],
        pl.Tensor[[CSA_FULL_WORDS], pl.INT32],
    ]:
        with pl.cluster(name_hint="dynamic_csa_full"):
            _tuple_tmp: pl.Tuple[
                pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32],
                pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32],
                pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32],
                pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32],
                pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
                pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
                pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
                pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
                pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32],
                pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32],
                pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32],
                pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32],
                pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32],
                pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32],
                pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32],
                pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32],
                pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32],
                pl.Tensor[[CSA_FULL_WORDS], pl.INT32],
            ] = self.cube_schedule(
                hc_fn,
                wq_a,
                x_fp32,
                partials,
                normed,
                qa,
                qr,
                qr_matmul,
                wq_b,
                q_proj,
                wkv,
                kv_fp32,
                main_wkv,
                main_wgate,
                main_value,
                main_score,
                positions,
                idx_wkv,
                idx_wgate,
                idx_value,
                idx_score,
                idx_bounds,
                idx_lengths,
                idx_count,
                idx_normed,
                hadamard,
                idx_had,
                idx_cache,
                q_heads,
                indexer_table,
                idx_wq_b,
                idx_query_acc,
                idx_query_rope_heads,
                idx_query_had,
                idx_query_i8,
                weights_proj,
                weights_partial,
                group_offsets,
                group_plan,
                coefficients,
                idx_key0,
                idx_key64,
                indexer_transfer,
                attention_mask,
                kv_ring,
                score_ring,
                probability_ring,
                pv_ring,
                packed,
                ffts,
                wo_a,
                o_r,
                o_quant,
                wo_b,
                o_partials,
                signal,
            )
            partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32] = _tuple_tmp[0]
            qa: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32] = _tuple_tmp[1]
            q_proj: pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32] = _tuple_tmp[2]
            kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32] = _tuple_tmp[3]
            main_value: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32] = _tuple_tmp[4]
            main_score: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32] = _tuple_tmp[5]
            idx_value: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32] = _tuple_tmp[6]
            idx_score: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32] = _tuple_tmp[7]
            idx_had: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32] = _tuple_tmp[8]
            idx_query_acc: pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32] = _tuple_tmp[9]
            idx_query_had: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32] = _tuple_tmp[10]
            weights_partial: pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32] = _tuple_tmp[11]
            indexer_transfer: pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32] = _tuple_tmp[12]
            score_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32] = _tuple_tmp[13]
            pv_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32] = _tuple_tmp[14]
            o_r: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32] = _tuple_tmp[15]
            o_partials: pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32] = _tuple_tmp[16]
            signal: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = _tuple_tmp[17]
            _tuple_tmp_1: pl.Tuple[
                pl.Tensor[[CSA_FULL_T, 16384], pl.FP32],
                pl.Tensor[[CSA_FULL_P16, 1], pl.FP32],
                pl.Tensor[[8, 16384], pl.FP32],
                pl.Tensor[[CSA_FULL_P16, 32], pl.FP32],
                pl.Tensor[[CSA_FULL_P16, 8], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 4], pl.FP32],
                pl.Tensor[[8, 8], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 16], pl.FP32],
                pl.Tensor[[8, 32], pl.FP32],
                pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16],
                pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
                pl.Tensor[[8, 4096], pl.BF16],
                pl.Tensor[[CSA_FULL_T, 1024], pl.INT8],
                pl.Tensor[[CSA_FULL_T, 1], pl.FP32],
                pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8],
                pl.Tensor[[CSA_FULL_P8, 1], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 64], pl.INT32],
                pl.Tensor[[CSA_FULL_T, 32768], pl.BF16],
                pl.Tensor[[CSA_FULL_T, 512], pl.BF16],
                pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16],
                pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32],
                pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32],
                pl.Tensor[[CSA_FULL_B], pl.INT32],
                pl.Tensor[[1], pl.INT32],
                pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 512], pl.FP32],
                pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16],
                pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32],
                pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32],
                pl.Tensor[[CSA_FULL_B], pl.INT32],
                pl.Tensor[[1], pl.INT32],
                pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16],
                pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 128], pl.FP32],
                pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8],
                pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16],
                pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8],
                pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32],
                pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32],
                pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32],
                pl.Tensor[[8], pl.INT32],
                pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16],
                pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32],
                pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 512], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 512], pl.INT32],
                pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32],
                pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32],
                pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32],
                pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
                pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16],
                pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16],
                pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
                pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
                pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16],
                pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8],
                pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 16384], pl.BF16],
                pl.Tensor[[CSA_FULL_WORDS], pl.INT32],
            ] = self.vector_schedule(
                x,
                hc_scale,
                hc_base,
                hc_base_2d,
                norm_w,
                gamma_cq,
                x_fp32,
                inv,
                widen_tail,
                partials,
                raw,
                pre,
                post,
                post_tail,
                comb,
                comb_tail,
                mixed,
                normed,
                norm_tail,
                qa,
                qr,
                qr_scale,
                qr_matmul,
                qr_scale_pad,
                wq_b_scale,
                rope_cos,
                rope_sin,
                rope_signed,
                rope_swap,
                q_proj,
                q_flat,
                gamma_ckv,
                kv_fp32,
                kv,
                native_slots,
                cache_rows,
                main_value,
                main_score,
                main_state,
                main_state_table,
                main_ape,
                positions,
                main_bounds,
                main_lengths,
                main_state_slots,
                main_pooled,
                main_gamma,
                main_cos,
                main_sin,
                main_offsets,
                main_count,
                main_slots,
                main_normed,
                main_debug,
                main_cache,
                idx_value,
                idx_score,
                idx_state,
                idx_state_table,
                idx_ape,
                idx_bounds,
                idx_lengths,
                idx_state_slots,
                idx_pooled,
                idx_gamma,
                idx_cos,
                idx_sin,
                idx_offsets,
                idx_count,
                idx_slots,
                idx_normed,
                idx_had,
                idx_scales,
                idx_debug,
                idx_cache,
                original_table,
                compressed_table,
                indexer_table,
                idx_wq_b_scale,
                idx_query_acc,
                idx_query_rope,
                idx_query_had,
                idx_query_i8,
                idx_query_scale,
                idx_query_scale_heads,
                weights_partial,
                head_weights,
                group_offsets,
                group_plan,
                coefficients,
                indexer_transfer,
                score_arena,
                pair_arena,
                topk_scores,
                topk_indices,
                attention_indices,
                attention_bias,
                attention_mask,
                inverse_sin,
                attn_sink,
                kv_ring,
                score_ring,
                probability_ring,
                pv_ring,
                mi_ring,
                li_ring,
                packed,
                ffts,
                o_r,
                o_quant,
                o_scales,
                o_partials,
                wo_b_scale,
                output,
                signal,
            )
            x_fp32: pl.Tensor[[CSA_FULL_T, 16384], pl.FP32] = _tuple_tmp_1[0]
            inv: pl.Tensor[[CSA_FULL_P16, 1], pl.FP32] = _tuple_tmp_1[1]
            widen_tail: pl.Tensor[[8, 16384], pl.FP32] = _tuple_tmp_1[2]
            raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32] = _tuple_tmp_1[3]
            pre: pl.Tensor[[CSA_FULL_P16, 8], pl.FP32] = _tuple_tmp_1[4]
            post: pl.Tensor[[CSA_FULL_T, 4], pl.FP32] = _tuple_tmp_1[5]
            post_tail: pl.Tensor[[8, 8], pl.FP32] = _tuple_tmp_1[6]
            comb: pl.Tensor[[CSA_FULL_T, 16], pl.FP32] = _tuple_tmp_1[7]
            comb_tail: pl.Tensor[[8, 32], pl.FP32] = _tuple_tmp_1[8]
            mixed: pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16] = _tuple_tmp_1[9]
            normed: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16] = _tuple_tmp_1[10]
            norm_tail: pl.Tensor[[8, 4096], pl.BF16] = _tuple_tmp_1[11]
            qr: pl.Tensor[[CSA_FULL_T, 1024], pl.INT8] = _tuple_tmp_1[12]
            qr_scale: pl.Tensor[[CSA_FULL_T, 1], pl.FP32] = _tuple_tmp_1[13]
            qr_matmul: pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8] = _tuple_tmp_1[14]
            qr_scale_pad: pl.Tensor[[CSA_FULL_P8, 1], pl.FP32] = _tuple_tmp_1[15]
            rope_signed: pl.Tensor[[CSA_FULL_T, 64], pl.FP32] = _tuple_tmp_1[16]
            rope_swap: pl.Tensor[[CSA_FULL_T, 64], pl.INT32] = _tuple_tmp_1[17]
            q_flat: pl.Tensor[[CSA_FULL_T, 32768], pl.BF16] = _tuple_tmp_1[18]
            kv: pl.Tensor[[CSA_FULL_T, 512], pl.BF16] = _tuple_tmp_1[19]
            cache_rows: pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16] = _tuple_tmp_1[20]
            main_state: pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32] = _tuple_tmp_1[21]
            main_pooled: pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32] = _tuple_tmp_1[22]
            main_offsets: pl.Tensor[[CSA_FULL_B], pl.INT32] = _tuple_tmp_1[23]
            main_count: pl.Tensor[[1], pl.INT32] = _tuple_tmp_1[24]
            main_normed: pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32] = _tuple_tmp_1[25]
            main_debug: pl.Tensor[[CSA_FULL_T, 512], pl.FP32] = _tuple_tmp_1[26]
            main_cache: pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16] = _tuple_tmp_1[27]
            idx_state: pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32] = _tuple_tmp_1[28]
            idx_pooled: pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32] = _tuple_tmp_1[29]
            idx_offsets: pl.Tensor[[CSA_FULL_B], pl.INT32] = _tuple_tmp_1[30]
            idx_count: pl.Tensor[[1], pl.INT32] = _tuple_tmp_1[31]
            idx_normed: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16] = _tuple_tmp_1[32]
            idx_scales: pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32] = _tuple_tmp_1[33]
            idx_debug: pl.Tensor[[CSA_FULL_T, 128], pl.FP32] = _tuple_tmp_1[34]
            idx_cache: pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8] = _tuple_tmp_1[35]
            idx_query_rope: pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16] = _tuple_tmp_1[36]
            idx_query_i8: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8] = _tuple_tmp_1[37]
            idx_query_scale: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32] = _tuple_tmp_1[38]
            head_weights: pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32] = _tuple_tmp_1[39]
            group_offsets: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32] = _tuple_tmp_1[40]
            group_plan: pl.Tensor[[8], pl.INT32] = _tuple_tmp_1[41]
            coefficients: pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16] = _tuple_tmp_1[42]
            score_arena: pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32] = _tuple_tmp_1[43]
            pair_arena: pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32] = _tuple_tmp_1[44]
            topk_scores: pl.Tensor[[CSA_FULL_T, 512], pl.FP32] = _tuple_tmp_1[45]
            topk_indices: pl.Tensor[[CSA_FULL_T, 512], pl.INT32] = _tuple_tmp_1[46]
            attention_indices: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32] = _tuple_tmp_1[47]
            attention_bias: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32] = _tuple_tmp_1[48]
            attention_mask: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32] = _tuple_tmp_1[49]
            inverse_sin: pl.Tensor[[CSA_FULL_T, 64], pl.FP32] = _tuple_tmp_1[50]
            kv_ring: pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16] = _tuple_tmp_1[51]
            probability_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16] = _tuple_tmp_1[52]
            mi_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32] = _tuple_tmp_1[53]
            li_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32] = _tuple_tmp_1[54]
            packed: pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16] = _tuple_tmp_1[55]
            o_quant: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8] = _tuple_tmp_1[56]
            o_scales: pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32] = _tuple_tmp_1[57]
            output: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16] = _tuple_tmp_1[58]
            signal: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = _tuple_tmp_1[59]
        return (
            x_fp32,
            inv,
            widen_tail,
            partials,
            raw,
            pre,
            post,
            post_tail,
            comb,
            comb_tail,
            mixed,
            normed,
            norm_tail,
            qa,
            qr,
            qr_scale,
            qr_matmul,
            qr_scale_pad,
            rope_signed,
            rope_swap,
            q_proj,
            q_flat,
            kv_fp32,
            kv,
            cache_rows,
            main_value,
            main_score,
            main_state,
            main_pooled,
            main_offsets,
            main_count,
            main_normed,
            main_debug,
            main_cache,
            idx_value,
            idx_score,
            idx_state,
            idx_pooled,
            idx_offsets,
            idx_count,
            idx_normed,
            idx_had,
            idx_scales,
            idx_debug,
            idx_cache,
            idx_query_acc,
            idx_query_rope,
            idx_query_had,
            idx_query_i8,
            idx_query_scale,
            weights_partial,
            head_weights,
            group_offsets,
            group_plan,
            coefficients,
            indexer_transfer,
            score_arena,
            pair_arena,
            topk_scores,
            topk_indices,
            attention_indices,
            attention_bias,
            attention_mask,
            inverse_sin,
            kv_ring,
            score_ring,
            probability_ring,
            pv_ring,
            mi_ring,
            li_ring,
            packed,
            o_r,
            o_quant,
            o_scales,
            o_partials,
            output,
            signal,
        )

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def decode_csa_spmd(
        self,
        x: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16],
        hc_fn: pl.Tensor[[24, 16384], pl.FP32],
        hc_scale: pl.Tensor[[3], pl.FP32],
        hc_base: pl.Tensor[[24], pl.FP32],
        hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
        norm_w: pl.Tensor[[4096], pl.BF16],
        wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        gamma_cq: pl.Tensor[[1024], pl.BF16],
        x_fp32: pl.InOut[pl.Tensor[[CSA_FULL_T, 16384], pl.FP32]],
        inv: pl.InOut[pl.Tensor[[CSA_FULL_P16, 1], pl.FP32]],
        widen_tail: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]],
        partials: pl.InOut[pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32]],
        raw: pl.InOut[pl.Tensor[[CSA_FULL_P16, 32], pl.FP32]],
        pre: pl.InOut[pl.Tensor[[CSA_FULL_P16, 8], pl.FP32]],
        post: pl.InOut[pl.Tensor[[CSA_FULL_T, 4], pl.FP32]],
        post_tail: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[CSA_FULL_T, 16], pl.FP32]],
        comb_tail: pl.InOut[pl.Tensor[[8, 32], pl.FP32]],
        mixed: pl.InOut[pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16]],
        normed: pl.InOut[pl.Tensor[[CSA_FULL_T, 4096], pl.BF16]],
        norm_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
        qa: pl.InOut[pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32]],
        qr: pl.InOut[pl.Tensor[[CSA_FULL_T, 1024], pl.INT8]],
        qr_scale: pl.InOut[pl.Tensor[[CSA_FULL_T, 1], pl.FP32]],
        qr_matmul: pl.InOut[pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8]],
        qr_scale_pad: pl.InOut[pl.Tensor[[CSA_FULL_P8, 1], pl.FP32]],
        wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        wq_b_scale: pl.Tensor[[32768], pl.FP32],
        rope_cos: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_sin: pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        rope_signed: pl.InOut[pl.Tensor[[CSA_FULL_T, 64], pl.FP32]],
        rope_swap: pl.InOut[pl.Tensor[[CSA_FULL_T, 64], pl.INT32]],
        q_proj: pl.InOut[pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32]],
        q_flat: pl.InOut[pl.Tensor[[CSA_FULL_T, 32768], pl.BF16]],
        wkv: pl.Tensor[[4096, 512], pl.BF16],
        gamma_ckv: pl.Tensor[[512], pl.BF16],
        kv_fp32: pl.InOut[pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32]],
        kv: pl.InOut[pl.Tensor[[CSA_FULL_T, 512], pl.BF16]],
        native_slots: pl.Tensor[[CSA_FULL_T, 2], pl.INT32],
        cache_rows: pl.InOut[pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16]],
        main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
        main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
        main_value: pl.InOut[pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32]],
        main_score: pl.InOut[pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32]],
        main_state: pl.InOut[pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32]],
        main_state_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_MTABLE_COLS], pl.INT32],
        main_ape: pl.Tensor[[4, 1024], pl.FP32],
        positions: pl.Tensor[[CSA_FULL_T], pl.INT64],
        main_bounds: pl.Tensor[[CSA_FULL_MQ], pl.INT32],
        main_lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        main_state_slots: pl.Tensor[[CSA_FULL_T, 2], pl.INT32],
        main_pooled: pl.InOut[pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32]],
        main_gamma: pl.Tensor[[512], pl.BF16],
        main_cos: pl.Tensor[[CSA_FULL_MC, 64], pl.FP32],
        main_sin: pl.Tensor[[CSA_FULL_MC, 64], pl.FP32],
        main_offsets: pl.InOut[pl.Tensor[[CSA_FULL_B], pl.INT32]],
        main_count: pl.InOut[pl.Tensor[[1], pl.INT32]],
        main_slots: pl.Tensor[[CSA_FULL_MSLOTS, 2], pl.INT32],
        main_normed: pl.InOut[pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32]],
        main_debug: pl.InOut[pl.Tensor[[CSA_FULL_T, 512], pl.FP32]],
        main_cache: pl.InOut[pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16]],
        idx_wkv: pl.Tensor[[256, 4096], pl.BF16],
        idx_wgate: pl.Tensor[[256, 4096], pl.BF16],
        idx_value: pl.InOut[pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32]],
        idx_score: pl.InOut[pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32]],
        idx_state: pl.InOut[pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32]],
        idx_state_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_ITABLE_COLS], pl.INT32],
        idx_ape: pl.Tensor[[4, 256], pl.FP32],
        idx_bounds: pl.Tensor[[CSA_FULL_IQ], pl.INT32],
        idx_lengths: pl.Tensor[[CSA_FULL_B], pl.INT32],
        idx_state_slots: pl.Tensor[[CSA_FULL_T, 2], pl.INT32],
        idx_pooled: pl.InOut[pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32]],
        idx_gamma: pl.Tensor[[128], pl.BF16],
        idx_cos: pl.Tensor[[CSA_FULL_IC, 64], pl.FP32],
        idx_sin: pl.Tensor[[CSA_FULL_IC, 64], pl.FP32],
        idx_offsets: pl.InOut[pl.Tensor[[CSA_FULL_B], pl.INT32]],
        idx_count: pl.InOut[pl.Tensor[[1], pl.INT32]],
        idx_slots: pl.Tensor[[CSA_FULL_ISLOTS, 2], pl.INT32],
        idx_normed: pl.InOut[pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16]],
        hadamard: pl.Tensor[[128, 128], pl.BF16],
        idx_had: pl.InOut[pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32]],
        idx_scales: pl.InOut[pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32]],
        idx_debug: pl.InOut[pl.Tensor[[CSA_FULL_T, 128], pl.FP32]],
        idx_cache: pl.InOut[pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8]],
        q_heads: pl.Tensor[[CSA_FULL_QUERY_HEAD_ROWS, 512], pl.BF16],
        original_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_ORIGINAL_COLS], pl.INT32],
        compressed_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_COMPRESSED_COLS], pl.INT32],
        indexer_table: pl.Tensor[[CSA_FULL_B, CSA_FULL_INDEXER_COLS], pl.INT32],
        idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8],
        idx_wq_b_scale: pl.Tensor[[8192], pl.FP32],
        idx_query_acc: pl.InOut[pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32]],
        idx_query_rope: pl.InOut[pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16]],
        idx_query_rope_heads: pl.Tensor[[CSA_FULL_IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
        idx_query_had: pl.InOut[pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32]],
        idx_query_i8: pl.InOut[pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8]],
        idx_query_scale: pl.InOut[pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32]],
        idx_query_scale_heads: pl.Tensor[[CSA_FULL_IDX_SCALE_T_ROWS, 64], pl.FP32],
        weights_proj: pl.Tensor[[4096, 64], pl.BF16],
        weights_partial: pl.InOut[pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32]],
        head_weights: pl.InOut[pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32]],
        group_offsets: pl.InOut[pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32]],
        group_plan: pl.InOut[pl.Tensor[[8], pl.INT32]],
        coefficients: pl.InOut[pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16]],
        indexer_transfer: pl.InOut[pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32]],
        score_arena: pl.InOut[pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32]],
        pair_arena: pl.InOut[pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32]],
        topk_scores: pl.InOut[pl.Tensor[[CSA_FULL_T, 512], pl.FP32]],
        topk_indices: pl.InOut[pl.Tensor[[CSA_FULL_T, 512], pl.INT32]],
        attention_indices: pl.InOut[pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32]],
        attention_bias: pl.InOut[pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32]],
        attention_mask: pl.InOut[pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32]],
        inverse_sin: pl.InOut[pl.Tensor[[CSA_FULL_T, 64], pl.FP32]],
        attn_sink: pl.Tensor[[64, 1], pl.FP32],
        kv_ring: pl.InOut[pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16]],
        score_ring: pl.InOut[pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32]],
        probability_ring: pl.InOut[pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16]],
        pv_ring: pl.InOut[pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32]],
        mi_ring: pl.InOut[pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32]],
        li_ring: pl.InOut[pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32]],
        packed: pl.InOut[pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16]],
        ffts: pl.Tensor[[256], pl.INT64],
        wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        o_r: pl.InOut[pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32]],
        o_quant: pl.InOut[pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8]],
        o_scales: pl.InOut[pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32]],
        wo_b: pl.Tensor[[8192, 4096], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        o_partials: pl.InOut[pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32]],
        wo_b_scale: pl.Tensor[[4096], pl.FP32],
        output: pl.InOut[pl.Tensor[[CSA_FULL_T, 16384], pl.BF16]],
        signal: pl.InOut[pl.Tensor[[CSA_FULL_WORDS], pl.INT32]],
    ) -> tuple[
        pl.Tensor[[CSA_FULL_T, 16384], pl.FP32],
        pl.Tensor[[CSA_FULL_P16, 1], pl.FP32],
        pl.Tensor[[8, 16384], pl.FP32],
        pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32],
        pl.Tensor[[CSA_FULL_P16, 32], pl.FP32],
        pl.Tensor[[CSA_FULL_P16, 8], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 4], pl.FP32],
        pl.Tensor[[8, 8], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 16], pl.FP32],
        pl.Tensor[[8, 32], pl.FP32],
        pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16],
        pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
        pl.Tensor[[8, 4096], pl.BF16],
        pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 1024], pl.INT8],
        pl.Tensor[[CSA_FULL_T, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8],
        pl.Tensor[[CSA_FULL_P8, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 64], pl.INT32],
        pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32],
        pl.Tensor[[CSA_FULL_T, 32768], pl.BF16],
        pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 512], pl.BF16],
        pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16],
        pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
        pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32],
        pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32],
        pl.Tensor[[CSA_FULL_B], pl.INT32],
        pl.Tensor[[1], pl.INT32],
        pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 512], pl.FP32],
        pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16],
        pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
        pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32],
        pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32],
        pl.Tensor[[CSA_FULL_B], pl.INT32],
        pl.Tensor[[1], pl.INT32],
        pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16],
        pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32],
        pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 128], pl.FP32],
        pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8],
        pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32],
        pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16],
        pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32],
        pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8],
        pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32],
        pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32],
        pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32],
        pl.Tensor[[8], pl.INT32],
        pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16],
        pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32],
        pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32],
        pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 512], pl.FP32],
        pl.Tensor[[CSA_FULL_T, 512], pl.INT32],
        pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32],
        pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32],
        pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32],
        pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
        pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16],
        pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32],
        pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16],
        pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32],
        pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
        pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16],
        pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32],
        pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8],
        pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32],
        pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32],
        pl.Tensor[[CSA_FULL_T, 16384], pl.BF16],
        pl.Tensor[[CSA_FULL_WORDS], pl.INT32],
    ]:
        cache_flat: pl.Tensor[[CSA_FULL_ICACHE_PAGES * CSA_FULL_ICACHE_BYTES], pl.INT8] = pl.tensor.reshape(idx_cache, [CSA_FULL_ICACHE_PAGES * CSA_FULL_ICACHE_BYTES])
        prefix: pl.Tensor[[CSA_FULL_ICACHE_PAGES * CSA_FULL_ICACHE_BYTES // pl.const(128, pl.INDEX) * pl.const(128, pl.INDEX)], pl.INT8] = pl.tensor.slice(
            cache_flat, [CSA_FULL_ICACHE_PAGES * CSA_FULL_ICACHE_BYTES // 128 * 128], [0]
        )
        shifted: pl.Tensor[[(CSA_FULL_ICACHE_PAGES * CSA_FULL_ICACHE_BYTES - pl.const(64, pl.INDEX)) // pl.const(128, pl.INDEX) * pl.const(128, pl.INDEX)], pl.INT8] = pl.tensor.slice(
            cache_flat, [(CSA_FULL_ICACHE_PAGES * CSA_FULL_ICACHE_BYTES - 64) // 128 * 128], [64]
        )
        idx_key0: pl.Tensor[[CSA_FULL_ICACHE_PAGES * CSA_FULL_ICACHE_BYTES // pl.const(128, pl.INDEX), 128], pl.INT8] = pl.tensor.reshape(
            prefix, [CSA_FULL_ICACHE_PAGES * CSA_FULL_ICACHE_BYTES // 128, 128]
        )
        idx_key64: pl.Tensor[[(CSA_FULL_ICACHE_PAGES * CSA_FULL_ICACHE_BYTES - pl.const(64, pl.INDEX)) // pl.const(128, pl.INDEX), 128], pl.INT8] = pl.tensor.reshape(
            shifted, [(CSA_FULL_ICACHE_PAGES * CSA_FULL_ICACHE_BYTES - 64) // 128, 128]
        )
        with pl.spmd(24, sync_start=True):
            _tuple_tmp: pl.Tuple[
                pl.Tensor[[CSA_FULL_T, 16384], pl.FP32],
                pl.Tensor[[CSA_FULL_P16, 1], pl.FP32],
                pl.Tensor[[8, 16384], pl.FP32],
                pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32],
                pl.Tensor[[CSA_FULL_P16, 32], pl.FP32],
                pl.Tensor[[CSA_FULL_P16, 8], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 4], pl.FP32],
                pl.Tensor[[8, 8], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 16], pl.FP32],
                pl.Tensor[[8, 32], pl.FP32],
                pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16],
                pl.Tensor[[CSA_FULL_T, 4096], pl.BF16],
                pl.Tensor[[8, 4096], pl.BF16],
                pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 1024], pl.INT8],
                pl.Tensor[[CSA_FULL_T, 1], pl.FP32],
                pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8],
                pl.Tensor[[CSA_FULL_P8, 1], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 64], pl.INT32],
                pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32],
                pl.Tensor[[CSA_FULL_T, 32768], pl.BF16],
                pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 512], pl.BF16],
                pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16],
                pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
                pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32],
                pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32],
                pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32],
                pl.Tensor[[CSA_FULL_B], pl.INT32],
                pl.Tensor[[1], pl.INT32],
                pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 512], pl.FP32],
                pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16],
                pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
                pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32],
                pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32],
                pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32],
                pl.Tensor[[CSA_FULL_B], pl.INT32],
                pl.Tensor[[1], pl.INT32],
                pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16],
                pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32],
                pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 128], pl.FP32],
                pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8],
                pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32],
                pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16],
                pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32],
                pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8],
                pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32],
                pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32],
                pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32],
                pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32],
                pl.Tensor[[8], pl.INT32],
                pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16],
                pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32],
                pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32],
                pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 512], pl.FP32],
                pl.Tensor[[CSA_FULL_T, 512], pl.INT32],
                pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32],
                pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32],
                pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32],
                pl.Tensor[[CSA_FULL_T, 64], pl.FP32],
                pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16],
                pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32],
                pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16],
                pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32],
                pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
                pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32],
                pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16],
                pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32],
                pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8],
                pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32],
                pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32],
                pl.Tensor[[CSA_FULL_T, 16384], pl.BF16],
                pl.Tensor[[CSA_FULL_WORDS], pl.INT32],
            ] = self._csa_core_group(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                hc_base_2d,
                norm_w,
                wq_a,
                gamma_cq,
                x_fp32,
                inv,
                widen_tail,
                partials,
                raw,
                pre,
                post,
                post_tail,
                comb,
                comb_tail,
                mixed,
                normed,
                norm_tail,
                qa,
                qr,
                qr_scale,
                qr_matmul,
                qr_scale_pad,
                wq_b,
                wq_b_scale,
                rope_cos,
                rope_sin,
                rope_signed,
                rope_swap,
                q_proj,
                q_flat,
                wkv,
                gamma_ckv,
                kv_fp32,
                kv,
                native_slots,
                cache_rows,
                main_wkv,
                main_wgate,
                main_value,
                main_score,
                main_state,
                main_state_table,
                main_ape,
                positions,
                main_bounds,
                main_lengths,
                main_state_slots,
                main_pooled,
                main_gamma,
                main_cos,
                main_sin,
                main_offsets,
                main_count,
                main_slots,
                main_normed,
                main_debug,
                main_cache,
                idx_wkv,
                idx_wgate,
                idx_value,
                idx_score,
                idx_state,
                idx_state_table,
                idx_ape,
                idx_bounds,
                idx_lengths,
                idx_state_slots,
                idx_pooled,
                idx_gamma,
                idx_cos,
                idx_sin,
                idx_offsets,
                idx_count,
                idx_slots,
                idx_normed,
                hadamard,
                idx_had,
                idx_scales,
                idx_debug,
                idx_cache,
                q_heads,
                original_table,
                compressed_table,
                indexer_table,
                idx_wq_b,
                idx_wq_b_scale,
                idx_query_acc,
                idx_query_rope,
                idx_query_rope_heads,
                idx_query_had,
                idx_query_i8,
                idx_query_scale,
                idx_query_scale_heads,
                weights_proj,
                weights_partial,
                head_weights,
                group_offsets,
                group_plan,
                coefficients,
                idx_key0,
                idx_key64,
                indexer_transfer,
                score_arena,
                pair_arena,
                topk_scores,
                topk_indices,
                attention_indices,
                attention_bias,
                attention_mask,
                inverse_sin,
                attn_sink,
                kv_ring,
                score_ring,
                probability_ring,
                pv_ring,
                mi_ring,
                li_ring,
                packed,
                ffts,
                wo_a,
                o_r,
                o_quant,
                o_scales,
                wo_b,
                o_partials,
                wo_b_scale,
                output,
                signal,
            )
            x_fp32: pl.Tensor[[CSA_FULL_T, 16384], pl.FP32] = _tuple_tmp[0]
            inv: pl.Tensor[[CSA_FULL_P16, 1], pl.FP32] = _tuple_tmp[1]
            widen_tail: pl.Tensor[[8, 16384], pl.FP32] = _tuple_tmp[2]
            partials: pl.Tensor[[CSA_FULL_PARTIAL, 32], pl.FP32] = _tuple_tmp[3]
            raw: pl.Tensor[[CSA_FULL_P16, 32], pl.FP32] = _tuple_tmp[4]
            pre: pl.Tensor[[CSA_FULL_P16, 8], pl.FP32] = _tuple_tmp[5]
            post: pl.Tensor[[CSA_FULL_T, 4], pl.FP32] = _tuple_tmp[6]
            post_tail: pl.Tensor[[8, 8], pl.FP32] = _tuple_tmp[7]
            comb: pl.Tensor[[CSA_FULL_T, 16], pl.FP32] = _tuple_tmp[8]
            comb_tail: pl.Tensor[[8, 32], pl.FP32] = _tuple_tmp[9]
            mixed: pl.Tensor[[CSA_FULL_P8, 4096], pl.BF16] = _tuple_tmp[10]
            normed: pl.Tensor[[CSA_FULL_T, 4096], pl.BF16] = _tuple_tmp[11]
            norm_tail: pl.Tensor[[8, 4096], pl.BF16] = _tuple_tmp[12]
            qa: pl.Tensor[[CSA_FULL_P16, 1024], pl.FP32] = _tuple_tmp[13]
            qr: pl.Tensor[[CSA_FULL_T, 1024], pl.INT8] = _tuple_tmp[14]
            qr_scale: pl.Tensor[[CSA_FULL_T, 1], pl.FP32] = _tuple_tmp[15]
            qr_matmul: pl.Tensor[[CSA_FULL_QR_ROWS, 1024], pl.INT8] = _tuple_tmp[16]
            qr_scale_pad: pl.Tensor[[CSA_FULL_P8, 1], pl.FP32] = _tuple_tmp[17]
            rope_signed: pl.Tensor[[CSA_FULL_T, 64], pl.FP32] = _tuple_tmp[18]
            rope_swap: pl.Tensor[[CSA_FULL_T, 64], pl.INT32] = _tuple_tmp[19]
            q_proj: pl.Tensor[[CSA_FULL_QPROJ_ROWS, 32768], pl.INT32] = _tuple_tmp[20]
            q_flat: pl.Tensor[[CSA_FULL_T, 32768], pl.BF16] = _tuple_tmp[21]
            kv_fp32: pl.Tensor[[CSA_FULL_KV_ROWS, 512], pl.FP32] = _tuple_tmp[22]
            kv: pl.Tensor[[CSA_FULL_T, 512], pl.BF16] = _tuple_tmp[23]
            cache_rows: pl.Tensor[[CSA_FULL_CACHE_ROWS, 512], pl.BF16] = _tuple_tmp[24]
            main_value: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32] = _tuple_tmp[25]
            main_score: pl.Tensor[[CSA_FULL_MPROJ, 1024], pl.FP32] = _tuple_tmp[26]
            main_state: pl.Tensor[[CSA_FULL_MSTATE_PAGES, CSA_FULL_MSTATE_ELEMS], pl.FP32] = _tuple_tmp[27]
            main_pooled: pl.Tensor[[CSA_FULL_MPOOL, 512], pl.FP32] = _tuple_tmp[28]
            main_offsets: pl.Tensor[[CSA_FULL_B], pl.INT32] = _tuple_tmp[29]
            main_count: pl.Tensor[[1], pl.INT32] = _tuple_tmp[30]
            main_normed: pl.Tensor[[CSA_FULL_MNORM, 512], pl.FP32] = _tuple_tmp[31]
            main_debug: pl.Tensor[[CSA_FULL_T, 512], pl.FP32] = _tuple_tmp[32]
            main_cache: pl.Tensor[[CSA_FULL_MCACHE_ROWS, 512], pl.BF16] = _tuple_tmp[33]
            idx_value: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32] = _tuple_tmp[34]
            idx_score: pl.Tensor[[CSA_FULL_IPROJ, 256], pl.FP32] = _tuple_tmp[35]
            idx_state: pl.Tensor[[CSA_FULL_ISTATE_PAGES, CSA_FULL_ISTATE_ELEMS], pl.FP32] = _tuple_tmp[36]
            idx_pooled: pl.Tensor[[CSA_FULL_IPOOL, 128], pl.FP32] = _tuple_tmp[37]
            idx_offsets: pl.Tensor[[CSA_FULL_B], pl.INT32] = _tuple_tmp[38]
            idx_count: pl.Tensor[[1], pl.INT32] = _tuple_tmp[39]
            idx_normed: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.BF16] = _tuple_tmp[40]
            idx_had: pl.Tensor[[CSA_FULL_ICPAD, 128], pl.FP32] = _tuple_tmp[41]
            idx_scales: pl.Tensor[[CSA_FULL_ISCALES, 1], pl.FP32] = _tuple_tmp[42]
            idx_debug: pl.Tensor[[CSA_FULL_T, 128], pl.FP32] = _tuple_tmp[43]
            idx_cache: pl.Tensor[[CSA_FULL_ICACHE_PAGES, CSA_FULL_ICACHE_BYTES], pl.INT8] = _tuple_tmp[44]
            idx_query_acc: pl.Tensor[[CSA_FULL_IDX_PROJ_ROWS, 8192], pl.INT32] = _tuple_tmp[45]
            idx_query_rope: pl.Tensor[[CSA_FULL_IDX_ROPE_ROWS, 8192], pl.BF16] = _tuple_tmp[46]
            idx_query_had: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.FP32] = _tuple_tmp[47]
            idx_query_i8: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 128], pl.INT8] = _tuple_tmp[48]
            idx_query_scale: pl.Tensor[[CSA_FULL_IDX_HEAD_ROWS, 1], pl.FP32] = _tuple_tmp[49]
            weights_partial: pl.Tensor[[CSA_FULL_WEIGHTS_PARTIAL_ROWS, 64], pl.FP32] = _tuple_tmp[50]
            head_weights: pl.Tensor[[CSA_FULL_WEIGHTS_ROWS, 64], pl.FP32] = _tuple_tmp[51]
            group_offsets: pl.Tensor[[CSA_FULL_GROUP_OFFSET_ROWS], pl.INT32] = _tuple_tmp[52]
            group_plan: pl.Tensor[[8], pl.INT32] = _tuple_tmp[53]
            coefficients: pl.Tensor[[CSA_FULL_COEFF_ROWS, CSA_FULL_COEFF_COLS], pl.FP16] = _tuple_tmp[54]
            indexer_transfer: pl.Tensor[[CSA_FULL_INDEX_TRANSFER_ROWS, CSA_FULL_INDEX_TRANSFER_COLS], pl.FP32] = _tuple_tmp[55]
            score_arena: pl.Tensor[[CSA_FULL_SCORE_ROWS, 8192], pl.FP32] = _tuple_tmp[56]
            pair_arena: pl.Tensor[[CSA_FULL_PAIR_ROWS, 1024], pl.FP32] = _tuple_tmp[57]
            topk_scores: pl.Tensor[[CSA_FULL_T, 512], pl.FP32] = _tuple_tmp[58]
            topk_indices: pl.Tensor[[CSA_FULL_T, 512], pl.INT32] = _tuple_tmp[59]
            attention_indices: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 512], pl.INT32] = _tuple_tmp[60]
            attention_bias: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 640], pl.FP32] = _tuple_tmp[61]
            attention_mask: pl.Tensor[[CSA_FULL_ATTN_PLAN_ROWS, 16], pl.INT32] = _tuple_tmp[62]
            inverse_sin: pl.Tensor[[CSA_FULL_T, 64], pl.FP32] = _tuple_tmp[63]
            kv_ring: pl.Tensor[[CSA_FULL_KV_RING_ROWS, 512], pl.BF16] = _tuple_tmp[64]
            score_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.FP32] = _tuple_tmp[65]
            probability_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 128], pl.BF16] = _tuple_tmp[66]
            pv_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 512], pl.FP32] = _tuple_tmp[67]
            mi_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32] = _tuple_tmp[68]
            li_ring: pl.Tensor[[CSA_FULL_HEAD_RING_ROWS, 1], pl.FP32] = _tuple_tmp[69]
            packed: pl.Tensor[[CSA_FULL_PACK_ROWS, 4096], pl.BF16] = _tuple_tmp[70]
            o_r: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.FP32] = _tuple_tmp[71]
            o_quant: pl.Tensor[[CSA_FULL_O_MM_ROWS, 8192], pl.INT8] = _tuple_tmp[72]
            o_scales: pl.Tensor[[8, CSA_FULL_O_SCALE_ROWS], pl.FP32] = _tuple_tmp[73]
            o_partials: pl.Tensor[[CSA_FULL_O_MM_ROWS, 32768], pl.INT32] = _tuple_tmp[74]
            output: pl.Tensor[[CSA_FULL_T, 16384], pl.BF16] = _tuple_tmp[75]
            signal: pl.Tensor[[CSA_FULL_WORDS], pl.INT32] = _tuple_tmp[76]
        return (
            x_fp32,
            inv,
            widen_tail,
            partials,
            raw,
            pre,
            post,
            post_tail,
            comb,
            comb_tail,
            mixed,
            normed,
            norm_tail,
            qa,
            qr,
            qr_scale,
            qr_matmul,
            qr_scale_pad,
            rope_signed,
            rope_swap,
            q_proj,
            q_flat,
            kv_fp32,
            kv,
            cache_rows,
            main_value,
            main_score,
            main_state,
            main_pooled,
            main_offsets,
            main_count,
            main_normed,
            main_debug,
            main_cache,
            idx_value,
            idx_score,
            idx_state,
            idx_pooled,
            idx_offsets,
            idx_count,
            idx_normed,
            idx_had,
            idx_scales,
            idx_debug,
            idx_cache,
            idx_query_acc,
            idx_query_rope,
            idx_query_had,
            idx_query_i8,
            idx_query_scale,
            weights_partial,
            head_weights,
            group_offsets,
            group_plan,
            coefficients,
            indexer_transfer,
            score_arena,
            pair_arena,
            topk_scores,
            topk_indices,
            attention_indices,
            attention_bias,
            attention_mask,
            inverse_sin,
            kv_ring,
            score_ring,
            probability_ring,
            pv_ring,
            mi_ring,
            li_ring,
            packed,
            o_r,
            o_quant,
            o_scales,
            o_partials,
            output,
            signal,
        )