# SPDX-License-Identifier: Apache-2.0
"""由已冻结动态源建立独立 Attention 数值候选，保留原采页及输出收尾。

此生成器只为完整记录候选与参考差异，不生成静态 batch/history 排程。
生成的全部业务轴仍是原 pl.dynamic，真实页表/位置/长度仍从 device 读取。
"""
import ast
import json
import shutil
import textwrap
from pathlib import Path

CUBE_BODY = '''
    """五块 QK 后，以统一尺度的 BF16 P 在四份 L0C 中跨块累加。"""
    pl.system.set_ffts(ffts_workspace)
    t_dim = pl.tensor.dim(q_flat, 0) // H
    # L1 三槽：0/1保留前两块，2在QK阶段流经后面三块；GM保留全部五块。
    # P按K128载入，16KiB即可；不申请全P的80KiB，更不假设640KiB KV能全驻L1。
    for qk_t in pl.range(core_id, t_dim, CORE_GROUPS):
        qk_l1 = pl.create_tile([3 * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16,
                               target_memory=pl.MemorySpace.Mat)
        qk_q = pl.load(q_flat, [qk_t * H, 0], [H, HEAD_DIM],
                      target_memory=pl.MemorySpace.Mat)
        last_stream_block = pl.cast(-1, pl.INDEX)
        for qk_sb in pl.range(SPARSE_BLOCKS):
            if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                qk_slot = core_id * SPARSE_BLOCKS + qk_sb
                qk_l1_slot = pl.min(qk_sb, 2)
                pl.system.sync_wait(EVENT_BASE + QK_KV_READY_EVENT,
                    pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                qk_l1 = pl.gather_row(qk_l1, kv_transfer,
                    [qk_l1_slot * ATTN_K_TILE, 0], [qk_slot * ATTN_K_TILE, 0],
                    [ATTN_K_TILE, HEAD_DIM])
                qk_kv_t = pl.tile.slice(pl.tile.transpose_view(qk_l1),
                    [HEAD_DIM, ATTN_K_TILE], [0, qk_l1_slot * ATTN_K_TILE])
                qk_scores = pl.matmul(qk_q, qk_kv_t, out_dtype=pl.FP32)
                pl.store(qk_scores, [qk_slot * H, 0], score_transfer)
                pl.system.sync_set(EVENT_BASE + QK_SCORE_READY_EVENT,
                    pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                if qk_sb >= 2:
                    last_stream_block = qk_sb
        # 两AIV的五块P都发布后才开始PV。空query也进行这一次交接，
        # 两侧协议一致；空query不读取未定义PV，也不访问非法KV页。
        pl.system.sync_wait(EVENT_BASE + QK_PROB_READY_EVENT,
            pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
        # 四个独立64x128 Acc共128KiB；避免把CZ父Tile的行切片误作连续子Acc。
        acc0 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
        acc1 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
        acc2 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
        acc3 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
        started = pl.cast(0, pl.INT32)
        for pv_sb in pl.range(SPARSE_BLOCKS):
            if pl.read(valid_block_mask, [qk_t, pv_sb]) > 0:
                pv_slot = core_id * SPARSE_BLOCKS + pv_sb
                pv_l1_slot = pv_sb
                if pv_sb >= 2:
                    pv_l1_slot = pl.cast(2, pl.INDEX)
                    if pv_sb != last_stream_block:
                        # 原块0/1已消费，复用其槽重读块2/3。只从已有GM KV读，
                        # 不重复Native页表查找或AIV gather。尾部无效块按真实mask跳过。
                        pv_l1_slot = (pv_sb - 2) % 2
                        qk_l1 = pl.gather_row(qk_l1, kv_transfer,
                            [pv_l1_slot * ATTN_K_TILE, 0], [pv_slot * ATTN_K_TILE, 0],
                            [ATTN_K_TILE, HEAD_DIM])
                pv_p = pl.load(probability_transfer, [pv_slot * H, 0], [H, ATTN_K_TILE],
                               target_memory=pl.MemorySpace.Mat)
                pv_left = pl.tile.move(pv_p, target_memory=pl.MemorySpace.Left)
                # 保留原K128/N128数学tile；首个有效块初始化，随后续加。
                # 真实双缓冲/等待由生成代码审查，不预称与旧流水完全相同。
                right0 = pl.tile.extract(qk_l1, pv_l1_slot * ATTN_K_TILE, 0,
                    [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                acc0 = pl.tile.matmul_acc(acc0, pv_left, right0, init_cond=started == 0)
                right1 = pl.tile.extract(qk_l1, pv_l1_slot * ATTN_K_TILE, PV_N_TILE,
                    [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                acc1 = pl.tile.matmul_acc(acc1, pv_left, right1, init_cond=started == 0)
                right2 = pl.tile.extract(qk_l1, pv_l1_slot * ATTN_K_TILE, 2 * PV_N_TILE,
                    [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                acc2 = pl.tile.matmul_acc(acc2, pv_left, right2, init_cond=started == 0)
                right3 = pl.tile.extract(qk_l1, pv_l1_slot * ATTN_K_TILE, 3 * PV_N_TILE,
                    [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                acc3 = pl.tile.matmul_acc(acc3, pv_left, right3, init_cond=started == 0)
                started = pl.cast(1, pl.INT32)
        if started != 0:
            # 每组只使用自己的第0个PV槽；ABI容量暂与其它head ring保持一致。
            out_row = core_id * SPARSE_BLOCKS * H
            pl.store(acc0, [out_row, 0], pv_transfer)
            pl.store(acc1, [out_row, PV_N_TILE], pv_transfer)
            pl.store(acc2, [out_row, 2 * PV_N_TILE], pv_transfer)
            pl.store(acc3, [out_row, 3 * PV_N_TILE], pv_transfer)
        pl.system.sync_set(EVENT_BASE + QK_PV_READY_EVENT,
            pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
    return score_transfer, pv_transfer
'''

VECTOR_LOOP_HEAD = '''
    # 每个AIV负责32 heads，80KiB score按[5*32,128]放置；整行assemble
    # 避免A3 Vec窄列插入。真正的数学变化只在下面的全局尺度及PV累加。
    for qk_t in pl.range(qk_core, t_dim, CORE_GROUPS):
        score_bank = pl.tile.full([SPARSE_BLOCKS * (H // 2), ATTN_K_TILE],
                                 dtype=pl.FP32, value=FP32_NEG_INF)
        any_valid = pl.cast(0, pl.INT32)
        for qk_tick in pl.range(SPARSE_BLOCKS + 1):
            qk_sb = qk_tick
            if qk_tick < SPARSE_BLOCKS:
'''

VECTOR_LOOP_TAIL = '''
            # 必须先取走上块SCORE事件，才能发布本块KV，避免同一事件
            # 连续set而未wait。五槽免去数据覆盖风险，但不能免去事件协议。
            if qk_tick > 0:
                softmax_sb = qk_tick - 1
                if pl.read(valid_block_mask, [qk_t, softmax_sb]) > 0:
                    pl.system.sync_wait(EVENT_BASE + QK_SCORE_READY_EVENT,
                        pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
            if qk_tick < SPARSE_BLOCKS:
                if pl.read(valid_block_mask, [qk_t, qk_tick]) > 0:
                    pl.system.sync_set(EVENT_BASE + QK_KV_READY_EVENT,
                        pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if qk_tick > 0:
                capture_sb = qk_tick - 1
                if pl.read(valid_block_mask, [qk_t, capture_sb]) > 0:
                    capture_row = (qk_core * SPARSE_BLOCKS + capture_sb) * H + qk_lane_head
                    captured = pl.load(score_transfer, [capture_row, 0], [H // 2, ATTN_K_TILE])
                    bias = pl.load(sparse_bias, [qk_t, capture_sb * ATTN_K_TILE], [1, ATTN_K_TILE])
                    masked = pl.col_expand_add(pl.mul(captured, SOFTMAX_SCALE), bias)
                    score_bank = pl.tile.assemble(score_bank, masked, [capture_sb * (H // 2), 0])
                    any_valid = pl.cast(1, pl.INT32)
        # sink参与max。无有效块时m=sink、l=0，后续分子置零，结果严格为零。
        global_m = pl.load(attn_sink_col, [qk_lane_head, 0], [H // 2, 1])
        for max_sb in pl.unroll(SPARSE_BLOCKS):
            if pl.read(valid_block_mask, [qk_t, max_sb]) > 0:
                max_scores = pl.tile.extract(score_bank, max_sb * (H // 2), 0,
                    [H // 2, ATTN_K_TILE], target_memory=pl.MemorySpace.Vec)
                max_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32)
                block_m = pl.row_max(max_scores, max_tmp)
                global_m = pl.maximum(global_m, block_m)
        global_l = pl.mul(global_m, 0.0)
        for prob_sb in pl.unroll(SPARSE_BLOCKS):
            if pl.read(valid_block_mask, [qk_t, prob_sb]) > 0:
                prob_scores = pl.tile.extract(score_bank, prob_sb * (H // 2), 0,
                    [H // 2, ATTN_K_TILE], target_memory=pl.MemorySpace.Vec)
                block_exp = pl.exp(pl.row_expand_sub(prob_scores, global_m))
                sum_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32)
                block_l = pl.row_sum(block_exp, sum_tmp)
                global_l = pl.add(global_l, block_l)
                block_p = pl.cast(block_exp, target_type=pl.BF16, mode="rint")
                prob_row = (qk_core * SPARSE_BLOCKS + prob_sb) * H + qk_lane_head
                pl.store(block_p, [prob_row, 0], probability_transfer)
        pl.system.sync_set(EVENT_BASE + QK_PROB_READY_EVENT,
            pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
        pl.system.sync_wait(EVENT_BASE + QK_PV_READY_EVENT,
            pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
        if any_valid != 0:
            pv_row = qk_core * SPARSE_BLOCKS * H + qk_lane_head
            left_valid = pl.load(pv_transfer, [pv_row, 0], [H // 2, HEAD_DIM // 2])
            right_valid = pl.load(pv_transfer, [pv_row, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2])
        else:
            left_valid = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
            right_valid = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        m_valid = global_m
        l_valid = global_l
        pv_t = qk_t
'''


def main():
    here = Path(__file__).resolve().parent
    source = here / "source_attention_candidate"
    assert not source.exists(), "不覆盖已生成的候选"
    source.mkdir()
    for path in (here / "source_reference").glob("*.py"):
        shutil.copy2(path, source / path.name)
    path = source / "attention_stream_stages.py"
    original = path.read_text()
    tree = ast.parse(original)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    lines = original.splitlines(keepends=True)
    cube = functions["attention_qk_pv_cube_stage"]
    vector = functions["attention_qk_pv_vector_lane_stage"]
    cube_header = "".join(lines[cube.decorator_list[0].lineno - 1:cube.body[0].lineno - 1])
    vector_header = "".join(lines[vector.decorator_list[0].lineno - 1:vector.body[0].lineno - 1])
    prefix = "".join(lines[:cube.decorator_list[0].lineno - 1])
    prefix = prefix.replace("from ..deepseek_v4_flash_csa.config import FLASH as M",
        "from ..deepseek_v4_flash_csa.config import FLASH as M\nfrom ..deepseek_v4_flash_csa.config import FP32_NEG_INF")
    old_doc = ast.get_docstring(tree)
    first = prefix.index('"""')
    last = prefix.index('"""', first + 3) + 3
    prefix = prefix[:first] + '''"""Attention全局尺度独立数值候选，尚未验收或采用。

整体思路：原device metadata及Native KV采集保持；每核组GM保留五块，
AIC完成五次QK，双AIV各在80KiB UB保存32头的五块score。统一最大值
包含sink；以统一尺度生成BF16 P，在四个独立L0C片段里按原K块顺序
累加，最后仅交接一次完整PV。L1保留三块，两块从已有GM KV重读。
不使用A5的UB到L1直连，不新增Host tiling，不固定batch或历史长度。

正确性边界：概率BF16舍入、分母归约、分子累加位置会改变，用户已允许
独立验证，但不能自动放宽冻结门槛或宣称token一致。原采页、因果与
最终sink/逆RoPE/packed发布代码来自冻结正式源。

同步：每query五槽数据独立，SCORE必须wait后才可继续发布下个KV，
避免同一个FFTS事件重复set。PROB/PV每query各一次，含空query，
两侧一致；空query不读未初始化PV。各核组独立推进，无全核集合。
原跨query QK/PV重叠不保留，损失必须计入完整CSA L1而非只看流量。
"""''' + prefix[last:]
    # 只复用原前置几何与RoPE索引；原online softmax状态不带入新算法。
    pre_begin = original.index("    t_dim = pl.tensor.dim(position_ids, 0)", vector.col_offset)
    pre_end = original.index("    running_m = ", pre_begin)
    prelude = original[pre_begin:pre_end]
    gather_begin = original.index("            if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:", pre_end)
    store_line = "                pl.store(qk_kv_half, [qk_kv_row + qk_lane_kv, 0], kv_transfer)\n"
    gather_end = original.index(store_line, gather_begin) + len(store_line)
    gather = textwrap.indent(original[gather_begin:gather_end], "    ")
    gather = gather.replace("qk_core * QK_TRANSFER_SLOTS + qk_tick % QK_TRANSFER_SLOTS",
                            "qk_core * SPARSE_BLOCKS + qk_sb")
    pub_begin = original.index("                packed_stride = ", gather_end)
    pub_end = original.index("                # 只重置刚完成query", pub_begin)
    publish = textwrap.indent(textwrap.dedent(original[pub_begin:pub_end]), "        ")
    text = prefix + cube_header + CUBE_BODY + "\n\n" + vector_header + prelude
    text += VECTOR_LOOP_HEAD + gather + VECTOR_LOOP_TAIL + publish
    text += "    return kv_transfer, probability_transfer, mi_transfer, li_transfer, o_packed_heads\n"
    ast.parse(text)
    path.write_text(text)
    binding = source / "binding.py"
    text = binding.read_text()
    assert text.count("24 * 3 *") == 6
    binding.write_text(text.replace("24 * 3 *", "24 * 5 *"))
    (here / "ATTENTION_PREPARATION.json").write_text(json.dumps(dict(
        source=str(source), numerical_change_authorized=True,
        formal_changed=False, cpu_compiled=False, device_validated=False,
        files_changed=["attention_stream_stages.py", "binding.py"],
        query_history_shapes="全部保留原pl.dynamic，五块与三L1槽是数学/资源容量",
        allocation_note="暂保持HEAD_RING_ROWS共享轴，六份head/kv workspace按五槽容量分配",
        scope="独立数值原型；正式采用前仍需区间策略、完整精度与七档验证",
    ), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
