# SPDX-License-Identifier: Apache-2.0
"""用两个16头轮次替代80KiB常驻，恢复原64行KV采集，不增加score读取量。"""
import ast
import textwrap
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    target = here / "source_attention_candidate/attention_stream_stages.py"
    saved = here / "attention_stream_stages_v3.py"
    assert not saved.exists(), "不覆盖历史尝试"
    saved.write_text(target.read_text())
    text = (here / "attention_stream_stages_v2.py").read_text()
    text = text.replace("双AIV各在80KiB UB保存32头的五块score",
                        "双AIV各分两轮处理16头，以40KiB UB保存五块score")
    text = text.replace("score_bank = pl.tile.full([SPARSE_BLOCKS * (H // 2), ATTN_K_TILE]",
                        "score_bank = pl.tile.full([SPARSE_BLOCKS * FINAL_HEAD_TILE, ATTN_K_TILE]")
    text = text.replace("captured = pl.load(score_transfer, [capture_row, 0], [H // 2, ATTN_K_TILE])",
                        "captured = pl.load(score_transfer, [capture_row, 0], [FINAL_HEAD_TILE, ATTN_K_TILE])")
    text = text.replace("[capture_sb * (H // 2), 0]", "[capture_sb * FINAL_HEAD_TILE, 0]")
    start = text.index("        # sink参与max。")
    end = text.index("        pl.system.sync_set(EVENT_BASE + QK_PROB_READY_EVENT,", start)
    old_math = text[start:end].replace("H // 2", "FINAL_HEAD_TILE")
    old_math = old_math.replace("[qk_lane_head, 0]", "[qk_lane_head + head_chunk * FINAL_HEAD_TILE, 0]")
    old_math = old_math.replace("* H + qk_lane_head\n",
                                "* H + qk_lane_head + head_chunk * FINAL_HEAD_TILE\n")
    new = '''        # 初轮采QK时只读本lane前16头，另一半留在已完成的五槽GM中。
        # 第二轮读此前未读过的16头；与原32头一次读相比，总字节相同。
        # 保存两轮m/l只需各128B，不能为了40KiB常驻又保留另一份整score。
        m_rows = pl.tile.create([2, FINAL_HEAD_TILE], dtype=pl.FP32)
        l_rows = pl.tile.create([2, FINAL_HEAD_TILE], dtype=pl.FP32)
        for head_chunk in pl.range(2):
            if head_chunk == 1:
                for load_sb in pl.unroll(SPARSE_BLOCKS):
                    if pl.read(valid_block_mask, [qk_t, load_sb]) > 0:
                        late_row = ((qk_core * SPARSE_BLOCKS + load_sb) * H
                                    + qk_lane_head + FINAL_HEAD_TILE)
                        late_scores = pl.load(score_transfer, [late_row, 0],
                                              [FINAL_HEAD_TILE, ATTN_K_TILE])
                        late_bias = pl.load(sparse_bias, [qk_t, load_sb * ATTN_K_TILE],
                                            [1, ATTN_K_TILE])
                        late_masked = pl.col_expand_add(pl.mul(late_scores, SOFTMAX_SCALE), late_bias)
                        score_bank = pl.tile.assemble(score_bank, late_masked,
                                                     [load_sb * FINAL_HEAD_TILE, 0])
'''
    new += textwrap.indent(old_math, "    ")
    new += '''            # 行布局保存，避免对ColMajor列向量做不支持的TEXTRACT。
            m_rows = pl.tile.assemble(m_rows, pl.reshape(global_m, [1, FINAL_HEAD_TILE]),
                                      [head_chunk, 0])
            l_rows = pl.tile.assemble(l_rows, pl.reshape(global_l, [1, FINAL_HEAD_TILE]),
                                      [head_chunk, 0])
'''
    text = text[:start] + new + text[end:]
    text = text.replace("        m_valid = global_m\n        l_valid = global_l\n",
                        "        m_valid = pl.reshape(m_rows, [H // 2, 1])\n"
                        "        l_valid = pl.reshape(l_rows, [H // 2, 1])\n")
    text = text.replace("    # 每个AIV负责32 heads，80KiB score按[5*32,128]放置；整行assemble",
                        "    # 每个AIV负责32 heads，分两轮16头，40KiB score按[5*16,128]放置；整行assemble")
    ast.parse(text)
    target.write_text(text)


if __name__ == "__main__":
    main()
