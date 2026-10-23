# SPDX-License-Identifier: Apache-2.0
"""K512累计max候选：先压低PV更新的UB活跃区，再评估多query共享KV。

与9月K128试验的实质差异：当前是K512/D256供数、三槽、跨query连续流水，
每query只有少量宏块。只改变K512的softmax/PV数学，K128区间保持原语义。
它改变BF16概率舍入；必须先独立FP64/边界检查，再用原Native输出容差测完整层。
"""

import argparse
import ast
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"生产结构变化，需重新核对替换点：{old[:100]}")
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "candidates/attention_macro512_online.py")
    args = parser.parse_args()
    text = (REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_hca_tnd/attention.py").read_text()
    text = once(text,
        "                    for vec_tick, (m_iter, l_iter, left_iter, right_iter, max_ring, sum_ring) in pl.range(",
        "                    # 概率生产端的累计max每query重置，独立于延迟两拍的PV消费端。\n"
        "                    softmax_seed = pl.reshape(pl.load(sink_col, [head0, 0], [H // 2, 1]),\n"
        "                                              [H // 2 // H_TILE, H_TILE])\n"
        "                    for vec_tick, (m_iter, l_iter, left_iter, right_iter, max_ring, sum_ring, softmax_m_iter) in pl.range(")
    text = once(text,
        "init_values=(m_query, l_query, left_query, right_query, max_query, sum_query),",
        "init_values=(m_query, l_query, left_query, right_query, max_query, sum_query, softmax_seed),")
    text = once(text,
        "ready_max, ready_sum = pl.yield_(next_max_ring, next_sum_ring)",
        "ready_max, ready_sum, ready_softmax = pl.yield_(next_max_ring, next_sum_ring, softmax_m_iter)")
    text = once(text,
        "for head_part, (head_max_ring, head_sum_ring) in pl.range(\n"
        "                                    H // 2 // H_TILE, init_values=(max_ring, sum_ring),",
        "for head_part, (head_max_ring, head_sum_ring, head_softmax_max) in pl.range(\n"
        "                                    H // 2 // H_TILE, init_values=(max_ring, sum_ring, softmax_m_iter),")
    a = text.index("                                # 每个AIV逐次处理两个H16片")
    b = text.index("                            # 描述符仅供本AIV消费", a)
    fragment = text[a:b]
    fragment = once(fragment, "                                    head_offset = head_part * H_TILE",
        "                                    head_offset = head_part * H_TILE\n"
        "                                    # 动态head偏移使用extract整行，避免旧slice别名固定首head的问题。\n"
        "                                    previous_m = pl.reshape(pl.tile.extract(\n"
        "                                        head_softmax_max, head_part, 0, [1, H_TILE],\n"
        "                                        target_memory=pl.MemorySpace.Vec), [H_TILE, 1])")
    for prefix, tile, scratch in (("raw", "raw_scaled", "raw_tmp"), ("cmp", "masked", "cmp_tmp")):
        fragment = once(fragment, f"{prefix}_max = pl.row_max({tile}, {scratch})",
                        f"{prefix}_max = pl.maximum(previous_m, pl.row_max({tile}, {scratch}))")
    fragment = once(fragment,
        "                                    completed_max, completed_sum = pl.yield_(next_max_ring, next_sum_ring)",
        "                                    next_softmax_max = pl.tile.assemble(\n"
        "                                        head_softmax_max, pl.reshape(maximum, [1, H_TILE]), [head_part, 0])\n"
        "                                    completed_max, completed_sum, completed_softmax = pl.yield_(\n"
        "                                        next_max_ring, next_sum_ring, next_softmax_max)")
    fragment = once(fragment, "ready_max, ready_sum = pl.yield_(completed_max, completed_sum)",
                    "ready_max, ready_sum, ready_softmax = pl.yield_(completed_max, completed_sum, completed_softmax)")
    text = text[:a] + fragment + text[b:]
    text = once(text, "updated_max_ring, updated_sum_ring = pl.yield_(ready_max, ready_sum)",
                "updated_max_ring, updated_sum_ring, softmax_m_after = pl.yield_(ready_max, ready_sum, ready_softmax)")
    text = once(text, "updated_max_ring, updated_sum_ring = pl.yield_(max_ring, sum_ring)",
                "updated_max_ring, updated_sum_ring, softmax_m_after = pl.yield_(max_ring, sum_ring, softmax_m_iter)")
    a = text.index("                            next_m = pl.maximum(current_m, out_m)")
    b = text.index("                            remaining = pl.tile.read(query_ring", a)
    old_update = text[a:b].rstrip()
    # 两条分支类型相同；编译期只保留所选策略。K128没有改变乘法或加法顺序。
    base_update = old_update
    for name in ("next_m", "next_l", "next_left", "next_right"):
        base_update = base_update.replace(name, "base_" + name[5:])
    online = '''                            # 概率已经按累计max缩放，PV直接加入；不再计算beta和PV乘beta。
                            online_m = out_m
                            alpha = pl.exp(pl.sub(current_m, online_m))
                            online_l = pl.add(pl.mul(alpha, current_l), out_l)
                            pv_left = pl.load(values, [out_row, 0], [H // 2, HEAD_DIM // 2])
                            online_left = pl.add(pl.row_expand_mul(current_left, alpha), pv_left)
                            pv_right = pl.load(values, [out_row, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2])
                            online_right = pl.add(pl.row_expand_mul(current_right, alpha), pv_right)
'''
    text = text[:a] + (
        "                            if MACRO_K == ATTN_K_TILE:\n" + textwrap.indent(base_update, "    ")
        + "\n                                next_m, next_l, next_left, next_right = pl.yield_(base_m, base_l, base_left, base_right)\n"
        + "                            else:\n" + textwrap.indent(online, "    ")
        + "                                next_m, next_l, next_left, next_right = pl.yield_(online_m, online_l, online_left, online_right)\n"
    ) + text[b:]
    text = once(text,
        "running_m, running_l, running_left, running_right, max_ring_done, sum_ring_done = pl.yield_(\n"
        "                            m_after, l_after, left_after, right_after, updated_max_ring, updated_sum_ring,",
        "running_m, running_l, running_left, running_right, max_ring_done, sum_ring_done, softmax_m_done = pl.yield_(\n"
        "                            m_after, l_after, left_after, right_after, updated_max_ring, updated_sum_ring, softmax_m_after,")
    text = text.replace('"""TND HCA attention：', '"""K512累计max诊断候选：尚未采用。\n\nTND HCA attention：', 1)
    ast.parse(text)
    args.output.write_text(text)
    print(args.output)


if __name__ == "__main__":
    main()
