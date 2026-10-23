# SPDX-License-Identifier: Apache-2.0
"""缩小attention的UB更新/发布块，为后续双query复用KV验证容量。

复用旧HCA已验证的八份N64累积、H8输出思路，适配当前TND/K512/三槽跨query链。
此步保持单query分工；不因CPU容量变小就声称双query可执行或有性能收益。
"""

import argparse
import ast
from pathlib import Path

from build_attention_online import once

HERE = Path(__file__).resolve().parent


def names(pattern):
    return ", ".join(pattern.format(i) for i in range(8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "candidates/attention_macro512_online.py")
    parser.add_argument("--output", type=Path, default=HERE / "candidates/attention_macro512_online_n64.py")
    args = parser.parse_args()
    text = args.input.read_text()
    text = once(text, "H_TILE = 16", "H_TILE = 8\nOUTPUT_CHUNK = 64")
    text = once(text, "if PUBLISH_GROUPS != 2:", "if PUBLISH_GROUPS != 1:")
    text = text.replace("HCA merge pack expects two output groups per head tile",
                        "当前H8发布块必须恰好覆盖一个输出组")
    a = text.index("                stream_seed_left =")
    b = text.index("                # 延迟 PV 期间", a)
    text = text[:a] + "".join(
        f"                stream_seed_o{i} = pl.tile.full([H // 2, OUTPUT_CHUNK], dtype=pl.FP32, value=0.0)\n"
        for i in range(8)
    ) + text[b:]
    for old, new in (
        ("stream_seed_left, stream_seed_right", names("stream_seed_o{}")),
        ("left_query, right_query", names("o{}_query")),
        ("left_iter, right_iter", names("o{}_iter")),
        ("current_left, current_right", names("current_o{}")),
        ("fresh_left, fresh_right", names("fresh_o{}")),
        ("left_after, right_after", names("o{}_after")),
        ("running_left, running_right", names("running_o{}")),
        ("left_done, right_done", names("o{}_done")),
    ):
        if old not in text:
            raise ValueError(f"待替换的累积tuple不存在：{old}")
        text = text.replace(old, new)
    a = text.index("                                fresh_left =")
    b = text.index("                                current_m, current_l,", a)
    text = text[:a] + "".join(
        f"                                fresh_o{i} = pl.tile.full([H // 2, OUTPUT_CHUNK], dtype=pl.FP32, value=0.0)\n"
        for i in range(8)
    ) + text[b:]
    a = text.index("                            if MACRO_K == ATTN_K_TILE:", text.index("# 归约状态跟随PV消费的query"))
    b = text.index("                            remaining = pl.tile.read", a)
    update = "                            # 八份N64保存同一完整FP32累积，每次只激活一份PV更新临时块。\n"
    for kind, branch in (("base", "if MACRO_K == ATTN_K_TILE"), ("online", "else")):
        update += f"                            {branch}:\n"
        m = "pl.maximum(current_m, out_m)" if kind == "base" else "out_m"
        update += f"                                {kind}_m = {m}\n"
        update += f"                                alpha = pl.exp(pl.sub(current_m, {kind}_m))\n"
        if kind == "base":
            update += "                                beta = pl.exp(pl.sub(out_m, base_m))\n"
        l = "pl.mul(beta, out_l)" if kind == "base" else "out_l"
        update += f"                                {kind}_l = pl.add(pl.mul(alpha, current_l), {l})\n"
        for i in range(8):
            update += f"                                pv{i} = pl.load(values, [out_row, {i} * OUTPUT_CHUNK], [H // 2, OUTPUT_CHUNK])\n"
            pv = f"pl.row_expand_mul(pv{i}, beta)" if kind == "base" else f"pv{i}"
            update += f"                                {kind}_o{i} = pl.add(pl.row_expand_mul(current_o{i}, alpha), {pv})\n"
        update += f"                                next_m, next_l, {names('next_o{}')} = pl.yield_(\n"
        update += f"                                    {kind}_m, {kind}_l, {names(kind+'_o{}')})\n"
    text = text[:a] + update + text[b:]
    a = text.index("                                    merged = pl.concat(")
    b = text.index("                                    normalized =", a)
    publish = ""
    for j in range(4):
        publish += f"                                    merged_quarter{j} = pl.concat(\n"
        publish += f"                                        pl.tile.slice(next_o{2*j}, [H_TILE, OUTPUT_CHUNK], [offset, 0]),\n"
        publish += f"                                        pl.tile.slice(next_o{2*j+1}, [H_TILE, OUTPUT_CHUNK], [offset, 0]))\n"
    publish += "                                    merged = pl.concat(pl.concat(merged_quarter0, merged_quarter1),\n"
    publish += "                                                       pl.concat(merged_quarter2, merged_quarter3))\n"
    text = text[:a] + publish + text[b:]
    text = text.replace("next_left, next_right", names("next_o{}"))
    text = once(text, "                                    pl.store(groups[1:2, :], [packed_row + T_PAD, 0], o_packed_heads)\n", "")
    text = text.replace("K512累计max诊断候选：尚未采用。", "K512累计max、N64累积与H8发布诊断候选：尚未采用。")
    text = text.replace("每个AIV逐次处理两个H16片，限制大列块的UB工作集。",
                        "每个AIV分H8计算softmax，缩小活跃scratch；完整FP32累积仍保留在UB。")
    ast.parse(text)
    args.output.write_text(text)
    print(args.output)


if __name__ == "__main__":
    main()
