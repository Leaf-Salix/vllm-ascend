# SPDX-License-Identifier: Apache-2.0
"""生成双query/KV复用实验；完整源码冻结到candidates，正式算子不变。

保留正式K128回退及原始gather；仅替换长历史入口。24个MIX逻辑block组成
12组(2Cube+4AIV)：偶数Cube做QK，奇数Cube做PV，每AIV仍只持有32heads累积。
配对按device请求边界生成，S1/S5奇数尾用members标志，绝不跨请求拼query。
三槽环跨pair连续，四个Vector消费确认后才复用，不能把发布信号误当消费ACK。
"""

import argparse
from pathlib import Path
import re
import textwrap

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


EXTRA = r'''
# 以下为实验长历史入口：不修改默认K128数学，不预记功能或性能通过。
PAIR_CLUSTERS = 12
PAIR_HEADS = H * 2
PAIR_ROWS = PAIR_CLUSTERS * QK_TRANSFER_SLOTS * PAIR_HEADS
PAIR_SIGNAL_STRIDE = 128  # 512B隔离区，复用已有A3协议的保守跨度。
PAIR_SIGNAL_KINDS = 10
PAIR_SIGNAL_ROWS = PAIR_CLUSTERS * PAIR_SIGNAL_KINDS
PAIR_SIGNAL_WORDS = PAIR_SIGNAL_ROWS * PAIR_SIGNAL_STRIDE
PAIR_MAX_TOKENS = 384
PAIR_K = 512
PAIR_PARTS = PAIR_K // ATTN_K_TILE


@pl.jit.inline(auto_scope=False)
def _pair_plan(query_bounds: pl.Tensor, seq_lens: pl.Tensor, positions: pl.Tensor,
               plan: pl.Tensor, header: pl.Tensor, signals: pl.Tensor):
    tokens = pl.tensor.dim(positions, 0)
    requests = pl.tensor.dim(seq_lens, 0)
    header_2d = pl.reshape(header, [1, 16])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_pair_plan") as ready:
        records = pl.tile.full([PAIR_MAX_TOKENS, 16], dtype=pl.INT32, value=0)
        for request, (cursor,) in pl.range(requests, init_values=(pl.cast(0, pl.INDEX),)):
            begin = pl.max(0, pl.cast(pl.read(query_bounds, [request]), pl.INDEX))
            end = pl.min(tokens, pl.cast(pl.read(query_bounds, [request + 1]), pl.INDEX))
            length = pl.max(0, pl.read(seq_lens, [request]))
            # length=0仍发出零输出工作，以便正确覆盖dummy/失活输出；不访问其cache页。
            for local, (write_row,) in pl.range((pl.max(0, end - begin) + 1) // 2, init_values=(cursor,)):
                first = begin + local * 2
                members = pl.min(2, end - first)
                second = first + members - 1
                pl.tile.write(records, [write_row, 0], pl.cast(first, pl.INT32))
                pl.tile.write(records, [write_row, 1], pl.cast(second, pl.INT32))
                pl.tile.write(records, [write_row, 2], pl.cast(members, pl.INT32))
                pl.tile.write(records, [write_row, 3], pl.cast(request, pl.INT32))
                pl.tile.write(records, [write_row, 4], pl.cast(pl.read(positions, [first]), pl.INT32))
                pl.tile.write(records, [write_row, 5], pl.cast(pl.read(positions, [second]), pl.INT32))
                pl.tile.write(records, [write_row, 6], pl.cast(length, pl.INT32))
                pl.tile.write(records, [write_row, 7], pl.cast(begin, pl.INT32))
                pl.tile.write(records, [write_row, 8], pl.cast(pl.read(positions, [begin]), pl.INT32))
                row_done = pl.yield_(write_row + 1)
            total = pl.yield_(row_done)
        if total > 0:
            pl.store(pl.set_validshape(records, total, 16), [0, 0], plan)
        counts = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
        pl.tile.write(counts, [0, 0], pl.cast(total, pl.INT32))
        pl.store(counts, [0, 0], header_2d)
        # 每次调用初始化；整体attention任务依赖ready且禁止提前占满核等待前驱。
        pl.store(pl.tile.full([PAIR_SIGNAL_ROWS, PAIR_SIGNAL_STRIDE], dtype=pl.INT32, value=0),
                 [0, 0], signals)
    return ready


@pl.jit.inline(auto_scope=False)
def _pair_notify(signals: pl.InOut[pl.Tensor], cluster: pl.Scalar[pl.INDEX],
                 kind: pl.Scalar[pl.INDEX], progress: pl.Scalar[pl.INDEX]):
    # 每槽单写者；PIPE_ALL等本核业务MTE3/FIX完成，DCCI/DSB发布信号。
    pl.system.bar_all()
    pl.system.cacheinvalid()
    pl.system.fence()
    pl.system.signal_notify(signals, [(cluster * PAIR_SIGNAL_KINDS + kind) * PAIR_SIGNAL_STRIDE],
                            pl.cast(progress, pl.INT32))
    return signals


@pl.jit.inline(auto_scope=False)
def _pair_wait(signals: pl.InOut[pl.Tensor], cluster: pl.Scalar[pl.INDEX],
               kind: pl.Scalar[pl.INDEX], progress: pl.Scalar[pl.INDEX]):
    pl.system.signal_wait(signals, [(cluster * PAIR_SIGNAL_KINDS + kind) * PAIR_SIGNAL_STRIDE],
                          pl.cast(progress, pl.INT32), backoff_ticks=100)
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    return signals


@pl.jit.incore
def _pair_cube(q_flat: pl.Tensor, raw_kv: pl.Tensor, cmp_work_kv: pl.Tensor,
               plan: pl.Tensor, header: pl.Tensor, scores: pl.InOut[pl.Tensor],
               probs: pl.Tensor, values: pl.InOut[pl.Tensor], signals: pl.InOut[pl.Tensor],
               cmp_work_count: pl.Scalar[pl.INDEX]):
    # plan/header由前置AIV每次重写；读取前显式失效本核Scalar缓存，支持同址metadata变更。
    pl.system.cacheinvalid()
    pl.system.fence()
    worker = pl.tile.get_block_idx()
    cluster = worker // 2
    role = worker % 2
    pair_count = pl.cast(pl.read(header, [0]), pl.INDEX)
    for pair, (offset,) in pl.range(cluster, pair_count, PAIR_CLUSTERS,
                                    init_values=(pl.cast(0, pl.INDEX),)):
        first = pl.cast(pl.read(plan, [pair, 0]), pl.INDEX)
        second = pl.cast(pl.read(plan, [pair, 1]), pl.INDEX)
        request = pl.cast(pl.read(plan, [pair, 3]), pl.INDEX)
        position1 = pl.read(plan, [pair, 5])
        length = pl.read(plan, [pair, 6])
        begin = pl.cast(pl.read(plan, [pair, 7]), pl.INDEX)
        first_position = pl.read(plan, [pair, 8])
        first_length = pl.min(WIN, pl.max(0, pl.min(first_position + 1, length)))
        rows = pl.cast(pl.max(0, pl.min(HCA_MAX_COMPRESSED_ROWS,
                        pl.min((position1 + 1) // COMPRESS_RATIO, length // COMPRESS_RATIO))), pl.INDEX)
        rows = pl.min(rows, cmp_work_count * ATTN_K_TILE)
        blocks = (rows + PAIR_K - 1) // PAIR_K
        work_count = blocks + 1
        if role == 0:
            # 两条query仅在此读取一次；奇数尾重复合法first，Vector禁止发布第二条。
            query = pl.create_tile([PAIR_HEADS, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
            query = pl.gather_row(query, q_flat, [0, 0], [first * H, 0], [H, HEAD_DIM])
            query = pl.gather_row(query, q_flat, [H, 0], [second * H, 0], [H, HEAD_DIM])
            for tick in pl.range(work_count):
                sequence = offset + tick
                if sequence >= QK_TRANSFER_SLOTS:
                    # 四个AIV均已消费旧PV后，才能覆盖同一score/prob/value槽。
                    for consumer in pl.unroll(4):
                        signals = _pair_wait(signals, cluster, 6 + consumer, sequence - QK_TRANSFER_SLOTS + 1)
                row = (cluster * QK_TRANSFER_SLOTS + sequence % QK_TRANSFER_SLOTS) * PAIR_HEADS
                if tick < blocks:
                    parts = pl.min(PAIR_PARTS, (rows - tick * PAIR_K + ATTN_K_TILE - 1) // ATTN_K_TILE)
                    for part in pl.range(parts):
                        seed = pl.create_tile([PAIR_HEADS, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                        for half, (acc_half,) in pl.pipeline(2, stage=2, init_values=(seed,)):
                            dim0 = half * 256
                            piece = pl.load(cmp_work_kv,
                                            [(request * cmp_work_count + tick * PAIR_PARTS + part) * ATTN_K_TILE, dim0],
                                            [ATTN_K_TILE, 256], target_memory=pl.MemorySpace.Mat)
                            transposed = pl.tile.transpose_view(piece)
                            for inner, (acc_inner,) in pl.pipeline(2, stage=2, init_values=(acc_half,)):
                                lhs = pl.tile.extract(query, 0, dim0 + inner * 128, [PAIR_HEADS, 128],
                                                      target_memory=pl.MemorySpace.Left)
                                rhs = pl.tile.extract(transposed, inner * 128, 0, [128, ATTN_K_TILE],
                                                      target_memory=pl.MemorySpace.Right)
                                update = pl.tile.matmul_acc(acc_inner, lhs, rhs, init_cond=(half == 0 and inner == 0))
                                inner_done = pl.yield_(update)
                            half_done = pl.yield_(inner_done)
                        pl.store(half_done, [row, part * ATTN_K_TILE], scores)
                else:
                    # Raw两条query起点不同，各算M64，不能把窗口并集扩成无效MAC。
                    for member in pl.range(2):
                        token = pl.cast(pl.read(plan, [pair, member]), pl.INDEX)
                        drop = pl.min(S - 1, pl.max(first_length + token - begin - WIN, 0))
                        seed = pl.create_tile([H, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                        for half, (acc_half,) in pl.pipeline(2, stage=2, init_values=(seed,)):
                            dim0 = half * 256
                            piece = pl.load(raw_kv, [request * REQUEST_KV_ROWS + drop, dim0],
                                            [ATTN_K_TILE, 256], target_memory=pl.MemorySpace.Mat)
                            transposed = pl.tile.transpose_view(piece)
                            for inner, (acc_inner,) in pl.pipeline(2, stage=2, init_values=(acc_half,)):
                                lhs = pl.tile.extract(query, member * H, dim0 + inner * 128, [H, 128],
                                                      target_memory=pl.MemorySpace.Left)
                                rhs = pl.tile.extract(transposed, inner * 128, 0, [128, ATTN_K_TILE],
                                                      target_memory=pl.MemorySpace.Right)
                                update = pl.tile.matmul_acc(acc_inner, lhs, rhs, init_cond=(half == 0 and inner == 0))
                                inner_done = pl.yield_(update)
                            half_done = pl.yield_(inner_done)
                        pl.store(half_done, [row + member * H, 0], scores)
                signals = _pair_notify(signals, cluster, 0, sequence + 1)
        else:
            for tick in pl.range(work_count):
                sequence = offset + tick
                row = (cluster * QK_TRANSFER_SLOTS + sequence % QK_TRANSFER_SLOTS) * PAIR_HEADS
                for producer in pl.unroll(4):
                    signals = _pair_wait(signals, cluster, 1 + producer, sequence + 1)
                if tick < blocks:
                    parts = pl.min(PAIR_PARTS, (rows - tick * PAIR_K + ATTN_K_TILE - 1) // ATTN_K_TILE)
                    # 每D256只持有两份M128/N128 Acc，共128KB；重复读取P是明确代价。
                    for half in pl.range(2):
                        seed0 = pl.create_tile([PAIR_HEADS, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                        seed1 = pl.create_tile([PAIR_HEADS, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                        for part, (acc0, acc1) in pl.range(parts, init_values=(seed0, seed1)):
                            probability = pl.load(probs, [row, part * ATTN_K_TILE], [PAIR_HEADS, ATTN_K_TILE],
                                                  target_memory=pl.MemorySpace.Mat)
                            lhs = pl.tile.move(probability, target_memory=pl.MemorySpace.Left)
                            value = pl.load(cmp_work_kv,
                                            [(request * cmp_work_count + tick * PAIR_PARTS + part) * ATTN_K_TILE, half * 256],
                                            [ATTN_K_TILE, 256], target_memory=pl.MemorySpace.Mat)
                            rhs0 = pl.tile.extract(value, 0, 0, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                            update0 = pl.tile.matmul_acc(acc0, lhs, rhs0, init_cond=(part == 0))
                            rhs1 = pl.tile.extract(value, 0, PV_N_TILE, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                            update1 = pl.tile.matmul_acc(acc1, lhs, rhs1, init_cond=(part == 0))
                            done0, done1 = pl.yield_(update0, update1)
                        pl.store(done0, [row, half * 256], values)
                        pl.store(done1, [row, half * 256 + PV_N_TILE], values)
                else:
                    for member in pl.range(2):
                        token = pl.cast(pl.read(plan, [pair, member]), pl.INDEX)
                        drop = pl.min(S - 1, pl.max(first_length + token - begin - WIN, 0))
                        probability = pl.load(probs, [row + member * H, 0], [H, ATTN_K_TILE],
                                              target_memory=pl.MemorySpace.Mat)
                        lhs = pl.tile.move(probability, target_memory=pl.MemorySpace.Left)
                        for half in pl.range(2):
                            value = pl.load(raw_kv, [request * REQUEST_KV_ROWS + drop, half * 256],
                                            [ATTN_K_TILE, 256], target_memory=pl.MemorySpace.Mat)
                            rhs0 = pl.tile.extract(value, 0, 0, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                            out0 = pl.tile.matmul(lhs, rhs0)
                            rhs1 = pl.tile.extract(value, 0, PV_N_TILE, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                            out1 = pl.tile.matmul(lhs, rhs1)
                            pl.store(out0, [row + member * H, half * 256], values)
                            pl.store(out1, [row + member * H, half * 256 + PV_N_TILE], values)
                signals = _pair_notify(signals, cluster, 5, sequence + 1)
        offset_done = pl.yield_(offset + work_count)
    return scores, values, signals
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-backoff", type=int, choices=(0, 50, 100), default=100,
                        help="只改变本实验GM等待退避，不改公共SDK或其他算子")
    parser.add_argument("--output", type=Path, default=HERE / "candidates/attention_pair512.py")
    parser.add_argument("--release-score", action="store_true",
                        help="QK仅等待旧score读完，复用概率ready作为ACK，不等待旧PV累加")
    parser.add_argument("--early", action="store_true",
                        help="恢复计划及attention的producer early提示；sync_start仍由运行时依赖/准入门禁保证")
    args = parser.parse_args()
    source = (REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_hca_tnd/attention.py").read_text()
    # PyPTO的分支变量仍共享类型环境，M128压缩体与M64 raw体不可复用同名Tile。
    extra = EXTRA.replace("backoff_ticks=100", f"backoff_ticks={args.signal_backoff}")
    if args.early:
        extra = extra.replace('name_hint="hca_pair_plan") as ready:',
                              'name_hint="hca_pair_plan", allow_early_resolve=True) as ready:')
        extra = extra.replace("# 每次调用初始化；整体attention任务依赖ready且禁止提前占满核等待前驱。",
                              "# 每次调用初始化；early仅允许消费者预派发，实际启动仍等全部前驱发布。")
    if args.release_score:
        extra = extra.replace(
            "# 四个AIV均已消费旧PV后，才能覆盖同一score/prob/value槽。",
            "# QK只写score：四个AIV发布旧概率时已读完旧score，即可覆盖。\n"
            "                    # 概率k+3只能在各AIV消费PV k之后产生；PV k+3又等待该概率，\n"
            "                    # 因此P/V各自的槽也安全，无须额外PV消费ACK阻塞QK。")
        extra = extra.replace("cluster, 6 + consumer, sequence - QK_TRANSFER_SLOTS + 1",
                              "cluster, 1 + consumer, sequence - QK_TRANSFER_SLOTS + 1")
    for begin, end, names, prefix in (
        ("                    # Raw两条query", "                signals = _pair_notify(signals, cluster, 0",
         "seed half acc_half dim0 piece transposed inner acc_inner lhs rhs update inner_done half_done token drop", "rawq_"),
        ("                else:\n                    for member in pl.range(2):",
         "                signals = _pair_notify(signals, cluster, 5",
         "token drop probability lhs half value rhs0 rhs1 out0 out1", "rawp_"),
    ):
        start = extra.index(begin)
        stop = extra.index(end, start)
        body = extra[start:stop]
        for name in names.split():
            body = re.sub(r"\b" + name + r"\b", prefix + name, body)
        extra = extra[:start] + body + extra[stop:]
    # Vector完整数学取自正式K512，替换owner、设备配对几何和GM握手，保留归约顺序。
    vector = source.split("            for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):\n", 1)[1]
    vector = vector.split("        return o_packed_heads, attention_tid", 1)[0]
    vector = textwrap.dedent(vector)
    vector = vector.replace("pl.system.set_ffts(ffts)\n", "")
    vector = vector.replace("worker, t_dim, NUM_QK_CORES,", "cluster, pair_count, PAIR_CLUSTERS,")
    vector = vector.replace("for vec_token, (", "for pair, (")
    start = vector.index("    request = pl.cast(pl.read(token_geometry")
    stop = vector.index("    for vec_tick,", start)
    vector = vector[:start] + textwrap.indent(textwrap.dedent('''
        vec_token = pl.cast(pl.read(plan, [pair, member]), pl.INDEX)
        members = pl.read(plan, [pair, 2])
        request = pl.cast(pl.read(plan, [pair, 3]), pl.INDEX)
        position = pl.read(plan, [pair, 4 + member])
        max_position = pl.read(plan, [pair, 5])
        length = pl.read(plan, [pair, 6])
        cmp_rows = pl.cast(pl.max(0, pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(
            (position + 1) // COMPRESS_RATIO, length // COMPRESS_RATIO))), pl.INDEX)
        cmp_rows = pl.min(cmp_rows, cmp_work_count * ATTN_K_TILE)
        max_rows = pl.cast(pl.max(0, pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(
            (max_position + 1) // COMPRESS_RATIO, length // COMPRESS_RATIO))), pl.INDEX)
        max_rows = pl.min(max_rows, cmp_work_count * ATTN_K_TILE)
        cmp_blocks = (max_rows + MACRO_K - 1) // MACRO_K
        work_count = 1 + cmp_blocks
        if pair + PAIR_CLUSTERS >= pair_count:
            vec_extra = pl.yield_(QK_PRE_LAUNCH)
        else:
            vec_extra = pl.yield_(0)
    ''').strip() + "\n", "    ") + vector[stop:]
    vector = vector.replace("(worker * QK_TRANSFER_SLOTS + vec_global % QK_TRANSFER_SLOTS) * H + head0",
                            "(cluster * QK_TRANSFER_SLOTS + vec_global % QK_TRANSFER_SLOTS) * PAIR_HEADS + member * H + head0")
    vector = vector.replace("(worker * QK_TRANSFER_SLOTS + out_work % QK_TRANSFER_SLOTS) * H + head0",
                            "(cluster * QK_TRANSFER_SLOTS + out_work % QK_TRANSFER_SLOTS) * PAIR_HEADS + member * H + head0")
    vector = vector.replace("pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)",
                            "signals = _pair_wait(signals, cluster, 0, vec_global + 1)")
    vector = vector.replace("pl.system.sync_set(1, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)",
                            "pl.tile.write(query_ring, [vec_global % QK_TRANSFER_SLOTS, 3], pl.cast(members, pl.INT32))\n"
                            "            signals = _pair_notify(signals, cluster, 1 + member * 2 + lane, vec_global + 1)")
    vector = vector.replace("pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)",
                            "signals = _pair_wait(signals, cluster, 5, out_work + 1)")
    # 第一条query可能在C128边界前尚无压缩行，第二条刚好产生第一行；不能做validshape=0读取。
    start = vector.index("\n                        cmp_tmp = pl.create_tile([H_TILE, MACRO_K]") + 1
    stop = vector.index("\n                    row_slot =", start) + 1
    math = vector[start:stop]
    math = math.replace("                        valid_cols = pl.min(MACRO_K, cmp_rows - vec_tick * MACRO_K)\n", "")
    math = math.replace("maximum, total = pl.yield_", "cmp_maximum, cmp_total = pl.yield_")
    replacement = "                        valid_cols = pl.max(0, pl.min(MACRO_K, cmp_rows - vec_tick * MACRO_K))\n"
    replacement += "                        if valid_cols > 0:\n" + textwrap.indent(math, "    ")
    replacement += textwrap.indent(textwrap.dedent('''
        else:
            zero_probability = pl.tile.full([H_TILE, MACRO_K], dtype=pl.BF16, value=0.0)
            pl.store(zero_probability, [vec_row + head_offset, 0], probs)
            no_max = pl.reshape(pl.tile.full([1, H_TILE], dtype=pl.FP32, value=NEG_INF), [H_TILE, 1])
            no_sum = pl.reshape(pl.tile.full([1, H_TILE], dtype=pl.FP32, value=0.0), [H_TILE, 1])
            cmp_maximum, cmp_total = pl.yield_(no_max, no_sum)
    ''').strip() + "\n", "                        ")
    replacement += "                        maximum, total = pl.yield_(cmp_maximum, cmp_total)\n"
    vector = vector[:start] + replacement + vector[stop:]
    vector = vector.replace("if remaining == 0:", "if remaining == 0 and member < pl.tile.read(query_ring, [out_work % QK_TRANSFER_SLOTS, 3]):")
    if not args.release_score:
        vector = vector.replace("            m_after, l_after, left_after, right_after = pl.yield_(",
                                "            signals = _pair_notify(signals, cluster, 6 + member * 2 + lane, out_work + 1)\n"
                                "            m_after, l_after, left_after, right_after = pl.yield_(", 1)
    vector_header = '''
@pl.jit.incore
def _pair_vector(plan: pl.Tensor, header: pl.Tensor, signals: pl.InOut[pl.Tensor],
                 scores: pl.Tensor, probs: pl.InOut[pl.Tensor], values: pl.Tensor,
                 raw_valid: pl.Tensor, cmp_work_valid: pl.Tensor, sink_col: pl.Tensor,
                 rope_cos_il: pl.Tensor, rope_sin_signed: pl.Tensor,
                 o_packed_heads: pl.Out[pl.Tensor], cmp_work_count: pl.Scalar[pl.INDEX]):
    pl.func_attr({"split_aiv": True})
    pl.system.cacheinvalid()
    pl.system.fence()
    cluster = pl.tile.get_block_idx() // 2
    member = pl.tile.get_block_idx() % 2
    lane = pl.tile.get_subblock_idx()
    pair_count = pl.cast(pl.read(header, [0]), pl.INDEX)
'''
    # 原工厂宏参数改为该实验固定计算块；T/B/Q/L仍全为动态设备值。
    vector = vector.replace("MACRO_PARTS", "PAIR_PARTS").replace("MACRO_K", "PAIR_K")
    vector_source = vector_header + textwrap.indent(vector, "    ") + "    return probs, o_packed_heads, signals\n"
    group = '''
@pl.jit.inline(auto_scope=False)
def _pair_group(q_flat: pl.Tensor, raw_kv: pl.Tensor, cmp_work_kv: pl.Tensor,
                plan: pl.Tensor, header: pl.Tensor, scores: pl.Tensor, probs: pl.Tensor,
                values: pl.Tensor, signals: pl.Tensor, raw_valid: pl.Tensor,
                cmp_work_valid: pl.Tensor, sink_col: pl.Tensor, rope_cos_il: pl.Tensor,
                rope_sin_signed: pl.Tensor, o_packed_heads: pl.Tensor, cmp_work_count: pl.Scalar[pl.INDEX]):
    with pl.cluster(name_hint="hca_pair_attention"):
        scores, values, signals = _pair_cube(q_flat, raw_kv, cmp_work_kv, plan, header,
                                            scores, probs, values, signals, cmp_work_count)
        probs, o_packed_heads, signals = _pair_vector(plan, header, signals, scores, probs,
            values, raw_valid, cmp_work_valid, sink_col, rope_cos_il, rope_sin_signed, o_packed_heads, cmp_work_count)
    return scores, probs, values, signals, o_packed_heads
'''
    prefix = source.split("    @pl.jit.inline(auto_scope=False)\n", 1)[1].split(
        "        # 全部历史共用一组 MIX 任务", 1)[0]
    root = "@pl.jit.inline(auto_scope=False)\n" + textwrap.dedent(prefix)
    root = root.replace("def sparse_attn_hca_tp1(", "def sparse_attn_hca_tp1_pair(", 1)
    suffix = '''
    plan = pl.create_tensor([t_dim, 16], dtype=pl.INT32)
    header = pl.create_tensor([16], dtype=pl.INT32)
    signals_2d = pl.create_tensor([PAIR_SIGNAL_ROWS, PAIR_SIGNAL_STRIDE], dtype=pl.INT32)
    plan_ready = _pair_plan(query_bounds, kv_seq_lens, position_ids, plan, header, signals_2d)
    signals = pl.reshape(signals_2d, [PAIR_SIGNAL_WORDS])
    scores = pl.create_tensor([PAIR_ROWS, PAIR_K], dtype=pl.FP32)
    probs = pl.create_tensor([PAIR_ROWS, PAIR_K], dtype=pl.BF16)
    values = pl.create_tensor([PAIR_ROWS, HEAD_DIM], dtype=pl.FP32)
    sink_col = pl.reshape(attn_sink, [H, 1])
    # 跨block通信要求所有角色可驻留。关闭early：全部前驱完成后才准入，
    # 避免占满核后等待尚未派发的QKV/gather生产者；收益须扣除这项启动代价。
    with pl.spmd(NUM_QK_CORES, sync_start=True,
                 deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, rope_cs_tid, q_ready, plan_ready]) as attention_tid:
        scores, probs, values, signals, o_packed_heads = _pair_group(
            q_flat, raw_kv, cmp_work_kv, plan, header, scores, probs, values, signals,
            raw_valid, cmp_work_valid, sink_col, rope_cos_il, rope_sin_signed, o_packed_heads, cmp_work_count)
    return o_packed_heads, attention_tid


sparse_attn_hca_tp1_long = sparse_attn_hca_tp1_pair
'''
    if args.early:
        suffix = suffix.replace("with pl.spmd(NUM_QK_CORES, sync_start=True,",
                                "with pl.spmd(NUM_QK_CORES, sync_start=True, allow_early_resolve=True,")
        suffix = suffix.replace(
            "# 跨block通信要求所有角色可驻留。关闭early：全部前驱完成后才准入，\n"
            "    # 避免占满核后等待尚未派发的QKV/gather生产者；收益须扣除这项启动代价。",
            "# 跨block通信要求所有角色可驻留。producer early允许前后消费者预派发；\n"
            "    # Simpler的sync_start gated cohort在全部核可运行且前驱释放后才统一启动，\n"
            "    # 因此不需要关闭early；数据依赖与输入/输出buffer所有权均保持。")
    candidate = args.output.resolve()
    candidate.write_text(source + extra + vector_source + group + "\n" + root + suffix)
    compile(candidate.read_text(), str(candidate), "exec")
    print(candidate)


if __name__ == "__main__":
    main()
