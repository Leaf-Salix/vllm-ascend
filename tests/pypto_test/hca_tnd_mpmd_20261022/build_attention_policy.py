# SPDX-License-Identifier: Apache-2.0
"""将已验证的K128/K512计算体合为共享准备/输出的源码；仅开发时使用。

输入均为已冻结源码，不从Native复制计算体；生成的是普通可阅读PyPTO模块，
服务运行和JIT都不执行此生成器。编译期选择数学块，运行期Q/L和页表仍动态。
"""

import argparse
import ast
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--macro", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, macro = args.base.read_text(), args.macro.read_text()
    decorator = "@pl.jit.inline(auto_scope=False)\ndef sparse_attn_hca_tp1("
    cut = base.index(decorator)
    header, body = base[:cut], base[cut:]
    a = body.index("        # QK进入下一query时保留上一query的KV槽")
    b = body.index("        for lane in pl.split_aiv", a)
    ma = macro.index("        # 两级Scalar FIFO保存前两项KV来源")
    mb = macro.index("        for lane in pl.split_aiv", ma)
    body = body[:a] + (
        "        # 编译期只保留选中计算体，不在核内增加策略判断或额外任务。\n"
        "        if MACRO_K == ATTN_K_TILE:\n" + textwrap.indent(body[a:b].rstrip(), "    ")
        + "\n        else:\n" + textwrap.indent(macro[ma:mb].rstrip(), "    ") + "\n\n"
    ) + body[b:]
    body = body.replace("[transfer_rows, ATTN_K_TILE]", "[transfer_rows, MACRO_K]")
    # 两档统一用H16列的归约环：K128一次写两行，K512分两次各写一行。
    # 每slot仍是32个head，字节数与次序不变，避免A3列子视图搬运的形状限制。
    body = body.replace("[QK_TRANSFER_SLOTS, H // 2]", "[QK_TRANSFER_SLOTS * (H // 2 // H_TILE), H_TILE]")
    # Cube两支分别沿用旧实现；Vector共用分块计数与输出，只切换softmax。
    vstart = body.index("        for lane in pl.split_aiv")
    vcube, vector = body[:vstart], body[vstart:]
    old = "cmp_blocks = pl.min(cmp_work_count, (cmp_rows + ATTN_K_TILE - 1) // ATTN_K_TILE)"
    assert vector.count(old) == 1
    vector = vector.replace(old, "cmp_rows = pl.min(cmp_rows, cmp_work_count * ATTN_K_TILE)\n"
                                 "                cmp_blocks = (cmp_rows + MACRO_K - 1) // MACRO_K")
    a = vector.index("                        score = pl.load(scores, [vec_row, 0]")
    b = vector.index("                        # 描述符仅供本AIV消费", a)
    ma = macro.index("                        # 每个AIV逐次处理两个H16片")
    mb = macro.index("                        # 描述符仅供本AIV消费", ma)
    macro_softmax = macro[ma:mb]
    base_softmax = vector[a:b].replace("[1, H // 2]", "[H // 2 // H_TILE, H_TILE]")
    base_softmax = base_softmax.replace("[vec_global % QK_TRANSFER_SLOTS, 0]",
                                        "[(vec_global % QK_TRANSFER_SLOTS) * (H // 2 // H_TILE), 0]")
    vector = vector[:a] + (
        "                        if MACRO_K == ATTN_K_TILE:\n"
        + textwrap.indent(base_softmax.rstrip(), "    ")
        + "\n                            ready_max, ready_sum = pl.yield_(next_max_ring, next_sum_ring)\n"
        + "                        else:\n" + textwrap.indent(macro_softmax.rstrip(), "    ")
        + "\n                            ready_max, ready_sum = pl.yield_(completed_max, completed_sum)\n"
    ) + vector[b:]
    vector = vector.replace("updated_max_ring, updated_sum_ring = pl.yield_(next_max_ring, next_sum_ring)",
                            "updated_max_ring, updated_sum_ring = pl.yield_(ready_max, ready_sum)")
    for name in ("updated_max_ring", "updated_sum_ring"):
        vector = vector.replace(f"{name}, out_work % QK_TRANSFER_SLOTS, 0, [1, H // 2],",
                                f"{name}, (out_work % QK_TRANSFER_SLOTS) * (H // 2 // H_TILE), 0, "
                                "[H // 2 // H_TILE, H_TILE],")
    body = vcube + vector
    header = header.replace("当前所有历史长度共用该正确性基线；后续按区间选择更合适的流水粒度。",
                            "短历史/小T采用K128并在L1复用KV；长历史较大T采用K512，减少跨块更新与PV写回。\n"
                            "两档共用请求gather、device Q/L、FFTS及输出/逆RoPE，仅编译期绑定不同计算块。\n"
                            "K512改变浮点舍入，使用原Native输出容差验证；选择区间见native_adapter.py。")
    factory = '''def _make_sparse_attention(MACRO_K):
    """编译期策略工厂，不接收本轮Host数据，不产生TMR运行时策略分支。

    K128保留3份完整KV的L1环以供QK/PV复用；K512按D256供数、PV重载KV，
    用四份L0C完成宏块归约再写GM。两者均保留三槽、两步QK提前执行、跨query流水。
    每AIV的K128 softmax一次H32；K512用两次H16限制UB容量。输出累积与RoPE共用。
    """
    MACRO_PARTS = MACRO_K // ATTN_K_TILE
'''
    text = header + factory + textwrap.indent(body, "    ") + '''
    return sparse_attn_hca_tp1


sparse_attn_hca_tp1 = _make_sparse_attention(ATTN_K_TILE)
sparse_attn_hca_tp1_long = _make_sparse_attention(512)
'''
    ast.parse(text)
    args.output.write_text(text)
    print(args.output)


if __name__ == "__main__":
    main()
