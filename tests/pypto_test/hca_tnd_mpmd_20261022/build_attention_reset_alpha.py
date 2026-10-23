# SPDX-License-Identifier: Apache-2.0
"""以首块alpha=0重置累积，避免额外两份32KiB分支合流Tile。

原首块把零矩阵与旧累积通过if/yield合流，再执行相同乘加，保留了额外64KiB。
此候选只在首块把旧值系数置零，并从sink重置max；不改beta、块归约、PV或RoPE。
前提为累积有限，需在失活/空KV/重复场景实测；不能把容量降低当作性能通过。
可独立把softmax扩至H32，发布仍按H16，保持输出分组及相同三槽流水。
"""

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("正式attention已变化，须重新核对重置变换")
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--softmax-heads", type=int, choices=(16, 32), default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = HERE.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_hca_tnd/attention.py"
    text = source.read_text()
    begin = text.index("                            if out_local == 0:")
    end = text.index("                            # 长历史的 running_m", begin)
    text = text[:begin] + '''                            # 首块只重置标量统计；两份FP32输出累积不再经大Tile的if合流。
                            # 旧累积有限时，alpha=0与先清零再乘alpha等价；不增加GM往返。
                            if out_local == 0:
                                fresh_m = pl.load(sink_col, [head0, 0], [H // 2, 1])
                                current_m = pl.yield_(fresh_m)
                            else:
                                current_m = pl.yield_(m_iter)
''' + text[end:]
    text = once(text,
                "                            alpha = pl.exp(pl.sub(current_m, next_m))",
                "                            if out_local == 0:\n"
                "                                zero_alpha = pl.reshape(pl.tile.full([1, H // 2], dtype=pl.FP32, value=0.0), [H // 2, 1])\n"
                "                                alpha = pl.yield_(zero_alpha)\n"
                "                            else:\n"
                "                                keep_alpha = pl.exp(pl.sub(current_m, next_m))\n"
                "                                alpha = pl.yield_(keep_alpha)")
    for old, new in (("pl.mul(alpha, current_l)", "pl.mul(alpha, l_iter)"),
                     ("pl.row_expand_mul(current_left, alpha)", "pl.row_expand_mul(left_iter, alpha)"),
                     ("pl.row_expand_mul(current_right, alpha)", "pl.row_expand_mul(right_iter, alpha)")):
        text = once(text, old, new)
    text = once(text, "H_TILE = 16", f"H_TILE = 16\nSOFTMAX_H_TILE = {args.softmax_heads}")
    # 只换softmax/ring的head块，不能把最终H16发布块也变大。
    begin = text.index("                max_ring_init =")
    end = text.index("                            out_local =", begin)
    text = text[:begin] + text[begin:end].replace("H_TILE", "SOFTMAX_H_TILE") + text[end:]
    text = text.replace("每个AIV逐次处理两个H16片，限制大列块的UB工作集。",
                        "本实验按SOFTMAX_H_TILE处理softmax；输出发布仍为H16。")
    args.output.write_text(text)
    compile(text, str(args.output), "exec")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
