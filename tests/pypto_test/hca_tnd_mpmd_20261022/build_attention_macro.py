# SPDX-License-Identifier: Apache-2.0
"""从当前TND attention生成大softmax块诊断候选，不修改正式算子。

保留device Q/L、页表、原gather、跨query三槽与两步QK预发。
QK按D256供数、K128点积；PV保留四个L0C部分和，每个宏块仅写回一次。
这会改变浮点舍入，必须独立验证，不能以逐bit调度优化的门禁冒充验收。
"""

import argparse
import ast
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

CUBE = '''        worker = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts)
        # 两级Scalar FIFO保存前两项KV来源；都是同一AIC的局部循环变量。
        # 它们不是GM完成标志，不涉及跨核Scalar缓存一致性；CV交接仍用硬件事件。
        for token, (work_offset, older_src, older_parts, older_raw,
                    recent_src, recent_parts, recent_raw) in pl.range(
            worker, t_dim, NUM_QK_CORES,
            init_values=(pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX),
                         pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX)),
        ):
            request = pl.cast(pl.read(token_geometry, [token, 0]), pl.INDEX)
            query_begin = pl.cast(pl.read(token_geometry, [token, 1]), pl.INDEX)
            length = pl.max(pl.read(token_geometry, [token, 3]), 0)
            position = pl.max(pl.read(position_ids, [token]), -1)
            cmp_rows = pl.cast(pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(
                (position + 1) // COMPRESS_RATIO, length // COMPRESS_RATIO,
            )), pl.INDEX)
            cmp_rows = pl.min(cmp_rows, cmp_work_count * ATTN_K_TILE)
            cmp_blocks = (cmp_rows + MACRO_K - 1) // MACRO_K
            work_count = cmp_blocks + 1
            first_length = pl.min(WIN, pl.max(pl.min(pl.read(position_ids, [pl.min(query_begin, t_dim - 1)]) + 1, length), 0))
            raw_drop = pl.min(S - 1, pl.max(first_length + (token - query_begin) - WIN, 0))
            query = pl.load(q_flat, [token * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            if token + NUM_QK_CORES >= t_dim:
                extra = pl.yield_(QK_PRE_LAUNCH)
            else:
                extra = pl.yield_(0)
            for tick, (old_src, old_parts, old_raw, new_src, new_parts, new_raw) in pl.range(
                work_count + extra,
                init_values=(older_src, older_parts, older_raw, recent_src, recent_parts, recent_raw),
            ):
                global_tick = work_offset + tick
                # FIFO每拍前移，来源用直接Scalar表达式计算，避免条件yield的地址别名。
                # 排空拍将描述符夹到最后raw块；只有已生产的前两拍才会被PV消费。
                is_raw = pl.min(1, pl.max(0, tick - cmp_blocks + 1))
                cmp_source = (request * cmp_work_count + tick * MACRO_PARTS) * ATTN_K_TILE
                raw_source = pl.cast(request * REQUEST_KV_ROWS + raw_drop, pl.INDEX)
                source = cmp_source * (1 - is_raw) + raw_source * is_raw
                compressed_parts = pl.max(0, pl.min(MACRO_PARTS, (cmp_rows - tick * MACRO_K + ATTN_K_TILE - 1) // ATTN_K_TILE))
                parts = compressed_parts * (1 - is_raw) + is_raw
                if tick < work_count:
                    row = (worker * QK_TRANSFER_SLOTS + global_tick % QK_TRANSFER_SLOTS) * H
                    # 宏块内仍是N128微块。D分成两个256列L1块，分别用K128 L0双缓冲。
                    # 不再用一个D512的L1临时块阻塞整段QK供数。
                    for part in pl.range(parts):
                        seed = pl.create_tile([H, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                        for half, (acc_half,) in pl.pipeline(2, stage=2, init_values=(seed,)):
                            dim0 = half * 256
                            if is_raw == 1:
                                raw_piece = pl.load(raw_kv, [request * REQUEST_KV_ROWS + raw_drop + part * ATTN_K_TILE, dim0],
                                                    [ATTN_K_TILE, 256], target_memory=pl.MemorySpace.Mat)
                                piece = pl.yield_(raw_piece)
                            else:
                                cmp_piece = pl.load(cmp_work_kv, [(request * cmp_work_count + tick * MACRO_PARTS + part) * ATTN_K_TILE, dim0],
                                                    [ATTN_K_TILE, 256], target_memory=pl.MemorySpace.Mat)
                                piece = pl.yield_(cmp_piece)
                            transposed = pl.tile.transpose_view(piece)
                            for inner, (acc_inner,) in pl.pipeline(2, stage=2, init_values=(acc_half,)):
                                lhs = pl.tile.extract(query, 0, dim0 + inner * 128, [H, 128],
                                                      target_memory=pl.MemorySpace.Left)
                                rhs = pl.tile.extract(transposed, inner * 128, 0, [128, ATTN_K_TILE],
                                                      target_memory=pl.MemorySpace.Right)
                                updated = pl.tile.matmul_acc(acc_inner, lhs, rhs, init_cond=(half == 0 and inner == 0))
                                inner_done = pl.yield_(updated)
                            half_done = pl.yield_(inner_done)
                        pl.store(half_done, [row, part * ATTN_K_TILE], scores)
                    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                if global_tick >= QK_PRE_LAUNCH:
                    pv_row = (worker * QK_TRANSFER_SLOTS + (global_tick - QK_PRE_LAUNCH) % QK_TRANSFER_SLOTS) * H
                    pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    # 四个N128结果覆盖D512；完整宏块的K累加完成后才写GM。
                    # 每个Acc32KiB，四份合计128KiB，不另保留一份全宽临时结果。
                    seed0 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                    seed1 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                    seed2 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                    seed3 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                    for part, (acc0, acc1, acc2, acc3) in pl.range(
                        old_parts, init_values=(seed0, seed1, seed2, seed3),
                    ):
                        probability = pl.load(probs, [pv_row, part * ATTN_K_TILE], [H, ATTN_K_TILE],
                                              target_memory=pl.MemorySpace.Mat)
                        left = pl.tile.move(probability, target_memory=pl.MemorySpace.Left)
                        # 先提供前半D，让MTE1可以提前开始；后半D使用另一个64KiB L1块。
                        if old_raw == 1:
                            raw0 = pl.load(raw_kv, [old_src + part * ATTN_K_TILE, 0], [ATTN_K_TILE, 256],
                                           target_memory=pl.MemorySpace.Mat)
                            value0 = pl.yield_(raw0)
                        else:
                            cmp0 = pl.load(cmp_work_kv, [old_src + part * ATTN_K_TILE, 0], [ATTN_K_TILE, 256],
                                           target_memory=pl.MemorySpace.Mat)
                            value0 = pl.yield_(cmp0)
                        right0 = pl.tile.extract(value0, 0, 0, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                        if old_raw == 1:
                            raw1 = pl.load(raw_kv, [old_src + part * ATTN_K_TILE, 256], [ATTN_K_TILE, 256],
                                           target_memory=pl.MemorySpace.Mat)
                            value1 = pl.yield_(raw1)
                        else:
                            cmp1 = pl.load(cmp_work_kv, [old_src + part * ATTN_K_TILE, 256], [ATTN_K_TILE, 256],
                                           target_memory=pl.MemorySpace.Mat)
                            value1 = pl.yield_(cmp1)
                        right1 = pl.tile.extract(value0, 0, PV_N_TILE, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                        update0 = pl.tile.matmul_acc(acc0, left, right0, init_cond=(part == 0))
                        right2 = pl.tile.extract(value1, 0, 0, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                        update1 = pl.tile.matmul_acc(acc1, left, right1, init_cond=(part == 0))
                        right3 = pl.tile.extract(value1, 0, PV_N_TILE, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                        update2 = pl.tile.matmul_acc(acc2, left, right2, init_cond=(part == 0))
                        update3 = pl.tile.matmul_acc(acc3, left, right3, init_cond=(part == 0))
                        done0, done1, done2, done3 = pl.yield_(update0, update1, update2, update3)
                    pl.store(done0, [pv_row, 0], values)
                    pl.store(done1, [pv_row, PV_N_TILE], values)
                    pl.store(done2, [pv_row, 2 * PV_N_TILE], values)
                    pl.store(done3, [pv_row, 3 * PV_N_TILE], values)
                    pl.system.sync_set(2, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                old_src_done, old_parts_done, old_raw_done, new_src_done, new_parts_done, new_raw_done = pl.yield_(
                    new_src, new_parts, new_raw, source, parts, is_raw,
                )
            offset_done, older_src_done, older_parts_done, older_raw_done, recent_src_done, recent_parts_done, recent_raw_done = pl.yield_(
                work_offset + work_count, old_src_done, old_parts_done, old_raw_done,
                new_src_done, new_parts_done, new_raw_done,
            )

'''

SOFTMAX = '''                        # 每个AIV逐次处理两个H16片，限制大列块的UB工作集。
                        for head_part, (head_max_ring, head_sum_ring) in pl.range(
                            H // 2 // H_TILE, init_values=(max_ring, sum_ring),
                        ):
                            head_offset = head_part * H_TILE
                            if vec_tick == cmp_blocks:
                                raw_tmp = pl.create_tile([H_TILE, ATTN_K_TILE], dtype=pl.FP32)
                                raw_score = pl.load(scores, [vec_row + head_offset, 0], [H_TILE, ATTN_K_TILE])
                                raw_mask = pl.load(raw_valid, [vec_token, 0], [1, ATTN_K_TILE])
                                raw_scaled = pl.col_expand_add(pl.mul(raw_score, SOFTMAX_SCALE),
                                                              pl.mul(pl.sub(raw_mask, 1.0), -NEG_INF))
                                raw_max = pl.row_max(raw_scaled, raw_tmp)
                                raw_exp = pl.col_expand_mul(pl.exp(pl.row_expand_sub(raw_scaled, raw_max)), raw_mask)
                                raw_sum = pl.row_sum(raw_exp, raw_tmp)
                                pl.store(pl.cast(raw_exp, pl.BF16, mode="rint"), [vec_row + head_offset, 0], probs)
                                maximum, total = pl.yield_(raw_max, raw_sum)
                            else:
                                cmp_tmp = pl.create_tile([H_TILE, MACRO_K], dtype=pl.FP32)
                                valid_cols = pl.min(MACRO_K, cmp_rows - vec_tick * MACRO_K)
                                mask_parts = (valid_cols + ATTN_K_TILE - 1) // ATTN_K_TILE
                                mask_grid = pl.load(cmp_work_valid,
                                                    [request * cmp_work_count + vec_tick * MACRO_PARTS, 0],
                                                    [MACRO_PARTS, ATTN_K_TILE],
                                                    valid_shape=[mask_parts, ATTN_K_TILE])
                                cmp_mask = pl.reshape(pl.fillpad(mask_grid, pad_value=pl.PadValue.zero), [1, MACRO_K])
                                score = pl.load(scores, [vec_row + head_offset, 0], [H_TILE, MACRO_K],
                                                valid_shape=[H_TILE, valid_cols])
                                scaled = pl.col_expand_add(pl.mul(score, SOFTMAX_SCALE),
                                                           pl.mul(pl.sub(cmp_mask, 1.0), -NEG_INF))
                                masked = pl.fillpad(pl.set_validshape(scaled, H_TILE, valid_cols), pad_value=pl.PadValue.min)
                                cmp_max = pl.row_max(masked, cmp_tmp)
                                cmp_exp = pl.col_expand_mul(pl.exp(pl.row_expand_sub(masked, cmp_max)), cmp_mask)
                                cmp_sum = pl.row_sum(cmp_exp, cmp_tmp)
                                pl.store(pl.cast(cmp_exp, pl.BF16, mode="rint"), [vec_row + head_offset, 0], probs)
                                maximum, total = pl.yield_(cmp_max, cmp_sum)
                            row_slot = (vec_global % QK_TRANSFER_SLOTS) * (H // 2 // H_TILE) + head_part
                            next_max_ring = pl.tile.assemble(
                                head_max_ring, pl.reshape(maximum, [1, H_TILE]), [row_slot, 0],
                            )
                            next_sum_ring = pl.tile.assemble(
                                head_sum_ring, pl.reshape(total, [1, H_TILE]), [row_slot, 0],
                            )
                            completed_max, completed_sum = pl.yield_(next_max_ring, next_sum_ring)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, help="未提供时从已记录的基线提交读取，避免后续正式重构改写历史试验")
    parser.add_argument("--macro-k", type=int, choices=(256, 512), default=512)
    parser.add_argument("--softmax-heads", type=int, choices=(16, 32), default=16,
                        help="每个AIV一次处理的head数；只改softmax，输出归约/RoPE仍按原H16")
    parser.add_argument("--qk-part-depth", type=int, choices=(1, 2), default=1,
                        help="QK相邻N128块是否交叠；不改变每个点积的K归约顺序")
    args = parser.parse_args()
    text = args.base.read_text() if args.base else subprocess.check_output([
        "git", "-C", str(REPO), "show",
        "4a65d78cd:vllm_ascend/ops/pypto/deepseek_v4_flash_hca_tnd/attention.py",
    ], text=True)
    text = text.replace("PV_N_TILE = 128", f"PV_N_TILE = 128\nMACRO_K = {args.macro_k}\nMACRO_PARTS = MACRO_K // ATTN_K_TILE")
    start = text.index("        worker = pl.tile.get_block_idx()", text.index('name_hint="hca_unified_attention"'))
    end = text.index("        for lane in pl.split_aiv", start)
    text = text[:start] + CUBE + text[end:]
    text = text.replace("[transfer_rows, ATTN_K_TILE]", "[transfer_rows, MACRO_K]")
    text = text.replace("[QK_TRANSFER_SLOTS, H // 2]", "[QK_TRANSFER_SLOTS * (H // 2 // H_TILE), H_TILE]")
    text = text.replace("            reduce_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32)\n", "")
    old = "cmp_blocks = pl.min(cmp_work_count, (cmp_rows + ATTN_K_TILE - 1) // ATTN_K_TILE)"
    assert text.count(old) == 1
    text = text.replace(old, "cmp_rows = pl.min(cmp_rows, cmp_work_count * ATTN_K_TILE)\n"
                            "                cmp_blocks = (cmp_rows + MACRO_K - 1) // MACRO_K")
    a = text.index("                        score = pl.load(scores, [vec_row, 0]")
    b = text.index("                        # 描述符仅供本AIV消费", a)
    text = text[:a] + SOFTMAX + text[b:]
    text = text.replace("pl.yield_(next_max_ring, next_sum_ring)\n                    else:",
                        "pl.yield_(completed_max, completed_sum)\n                    else:")
    for name in ("updated_max_ring", "updated_sum_ring"):
        text = text.replace(f"{name}, out_work % QK_TRANSFER_SLOTS, 0, [1, H // 2],",
                            f"{name}, (out_work % QK_TRANSFER_SLOTS) * (H // 2 // H_TILE), 0, [H // 2 // H_TILE, H_TILE],")
    a = text.index('"""'); b = text.index('"""', a + 3)
    text = text[:a] + '''"""TND HCA大块attention诊断候选，不进入正式入口。

整体思路：保留原请求gather和设备Q/L；将压缩历史按MACRO_K列做一次softmax，
QK沿N128、D256/L0K128供数；PV在四个L0C内完成宏块K累加后再写GM。
保持三份CV交接槽、两步QK提前执行及跨query连续流水。Cube用本核Scalar FIFO
保存延迟PV的来源，Vector用原UB描述符环；硬件FFTS保证CV数据交接。
S1～S6/失活请求/负页/尾块规则与当前TND一致，没有token//S，也不回读Host数值。
本候选改变浮点舍入，须先独立FP64与Native输出验证；不能当成逐bit优化直接采用。
"""''' + text[b+3:]
    if args.softmax_heads == 32:
        # 独立评估减少Vector的head循环，不能把输出/RoPE的tile一起加倍。
        # 先完整编译检查UB容量，超过容量即停止，不修改共享SDK强行放行。
        text = text.replace("MACRO_PARTS = MACRO_K // ATTN_K_TILE",
                            "MACRO_PARTS = MACRO_K // ATTN_K_TILE\nSOFTMAX_H_TILE = 32")
        a = text.index("                        # 每个AIV逐次处理两个H16片")
        b = text.index("                        # 描述符仅供本AIV消费", a)
        body = text[a:b].replace("H_TILE", "SOFTMAX_H_TILE")
        body = body.replace("每个AIV逐次处理两个H16片，限制大列块的UB工作集。",
                            "每个AIV一次处理H32；保留完整H16输出链，观察UB压力与减少循环的收益。")
        text = text[:a] + body + text[b:]
        text = text.replace("[QK_TRANSFER_SLOTS * (H // 2 // H_TILE), H_TILE]",
                            "[QK_TRANSFER_SLOTS * (H // 2 // SOFTMAX_H_TILE), SOFTMAX_H_TILE]")
        text = text.replace("(out_work % QK_TRANSFER_SLOTS) * (H // 2 // H_TILE), 0, [H // 2 // H_TILE, H_TILE]",
                            "(out_work % QK_TRANSFER_SLOTS) * (H // 2 // SOFTMAX_H_TILE), 0, [H // 2 // SOFTMAX_H_TILE, SOFTMAX_H_TILE]")
    if args.qk_part_depth == 2:
        assert text.count("for part in pl.range(parts):") == 1
        text = text.replace("for part in pl.range(parts):", "for part in pl.pipeline(parts, stage=2):")
    ast.parse(text)
    suffix = "_h32" if args.softmax_heads == 32 else ""
    suffix += "_qk2" if args.qk_part_depth == 2 else ""
    target = HERE / "candidates" / f"attention_macro{args.macro_k}{suffix}_d256.py"
    target.write_text(text)
    print(target)


if __name__ == "__main__":
    main()
