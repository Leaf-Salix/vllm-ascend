# SPDX-License-Identifier: Apache-2.0
"""最终 PV 按 16 个 head 直接读取，去掉在线合并遗留的左右半宽缓冲。

仅用于已获独立验证授权的全局尺度候选。五块 PV 已在 Cube 中累加为一份，
不再需要为后续在线更新保留两个 32×256 FP32 累加器。每次发布直接读取
16×512；两个轮次合计读取字节不变，分母、逆 RoPE 与 BF16 舍入保持原式。
"""
import ast
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    target = here / "source_attention_candidate/attention_stream_stages.py"
    previous = here / "attention_stream_stages_v4.py"
    assert not previous.exists(), "不覆盖历史尝试"
    text = target.read_text()
    previous.write_text(text)
    start = text.index("        if any_valid != 0:\n            pv_row")
    end = text.index("        m_valid = ", start)
    text = text[:start] + text[end:]
    start = text.index("            m_left = pl.slice(left_valid,")
    end = text.index("            n_sink_bias = ", start)
    text = text[:start] + """            # Cube 已完成跨块累加；直接按本次发布的16个head读最终PV。
            # 不再同时保留左右32头缓冲，更不需要concat构造完整行。
            if any_valid != 0:
                pv_row = qk_core * SPARSE_BLOCKS * H + m_h0
                m_oi = pl.load(pv_transfer, [pv_row, 0], [FINAL_HEAD_TILE, HEAD_DIM])
            else:
                m_oi = pl.tile.full([FINAL_HEAD_TILE, HEAD_DIM],
                                    dtype=pl.FP32, value=0.0)
""" + text[end:]
    ast.parse(text)
    target.write_text(text)


if __name__ == "__main__":
    main()
