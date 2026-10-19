# SPDX-License-Identifier: Apache-2.0
"""保留独立64KiB KV区，将逐元素在线oi归并按16头处理，减少瞬时PV临时量。

只拆没有跨头归约的元素运算，不拆row_max/row_sum，不改FP32加乘次序。
这是为上一候选暴露的两个32KiB PV临时区降峰值，不做Tile大小扫描。
"""

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    base = HERE / "source_attention_shared_kv"
    target = HERE / "source_attention_shared_kv_merge16"
    target.mkdir(exist_ok=False)
    for path in base.glob("*.py"):
        (target / path.name).write_bytes(path.read_bytes())
    path = target / "attention_stream_stages.py"
    source = path.read_text()
    begin = source.index("                pv_left = pl.load(")
    end = source.index("                m_valid, l_valid, left_valid, right_valid = pl.yield_", begin)
    original = source[begin:end]
    assert "next_left = pl.add" in original and "next_right = pl.add" in original
    source = source[:begin] + """                # 每个head的oi更新完全独立，不涉及跨head归约。沿既有
                # FINAL_HEAD_TILE=16分两段，避免同时保留两份32KiB的PV/乘积。
                # 每个元素仍是(old*alpha)+(pv*beta)，不合成FMA、不改精度。
                # 用DPS assemble把已算好的半段写回完整32头状态，下一段
                # 读取这个状态的另一半；实际alias与临时峰值由完整编译审计。
                next_left = left_iter
                next_right = right_iter
                for merge_group in pl.unroll((H // 2) // FINAL_HEAD_TILE):
                    merge_head = merge_group * FINAL_HEAD_TILE
                    # 列向量先转行再extract，沿最终发布的已验证表达保留偏移。
                    merge_alpha = pl.reshape(
                        pl.tile.extract(pl.reshape(alpha, [1, H // 2]), 0, merge_head,
                                        [1, FINAL_HEAD_TILE], target_memory=pl.MemorySpace.Vec),
                        [FINAL_HEAD_TILE, 1],
                    )
                    merge_beta = pl.reshape(
                        pl.tile.extract(pl.reshape(beta, [1, H // 2]), 0, merge_head,
                                        [1, FINAL_HEAD_TILE], target_memory=pl.MemorySpace.Vec),
                        [FINAL_HEAD_TILE, 1],
                    )
                    merge_old_left = pl.slice(next_left, [FINAL_HEAD_TILE, HEAD_DIM // 2], [merge_head, 0])
                    merge_pv_left = pl.load(
                        pv_transfer, [pv_transfer_row + qk_lane_head + merge_head, 0],
                        [FINAL_HEAD_TILE, HEAD_DIM // 2],
                    )
                    merge_left = pl.add(pl.row_expand_mul(merge_old_left, merge_alpha),
                                        pl.row_expand_mul(merge_pv_left, merge_beta))
                    next_left = pl.tile.assemble(next_left, merge_left, [merge_head, 0])
                    merge_old_right = pl.slice(next_right, [FINAL_HEAD_TILE, HEAD_DIM // 2], [merge_head, 0])
                    merge_pv_right = pl.load(
                        pv_transfer, [pv_transfer_row + qk_lane_head + merge_head, HEAD_DIM // 2],
                        [FINAL_HEAD_TILE, HEAD_DIM // 2],
                    )
                    merge_right = pl.add(pl.row_expand_mul(merge_old_right, merge_alpha),
                                         pl.row_expand_mul(merge_pv_right, merge_beta))
                    next_right = pl.tile.assemble(next_right, merge_right, [merge_head, 0])
""" + source[end:]
    marker = "    尚须完整CPU确认同址与分离，随后真机冻结检查和同轮完整L1决定采用。\n"
    assert source.count(marker) == 1
    source = source.replace(marker, """    共享KV缓冲已在v1 IR中生效，但两个32KiB PV/乘积临时区成为同样大小
    的新峰值。v2沿既有16头发布宽度拆分独立的oi加乘，不拆任何归约或改变
    元素运算顺序；仍须审计真实copy/alias，不能由源码估算即宣称放得下。
""" + marker)
    ast.parse(source)
    path.write_text(source)
    # 头和列覆盖与两种遍历顺序无关，不能据此替代硬件舍入或同步检查。
    old = {(h, col) for h in range(32) for col in range(512)}
    new = {(group*16+h, half*256+col) for group in range(2)
           for half in range(2) for h in range(16) for col in range(256)}
    assert old == new and len(new) == 16384
    report = dict(changed=["attention_stream_stages.py"], per_head_element_order="old*alpha + pv*beta",
                  element_coverage=len(new), reduction_calls_unchanged=True,
                  merge_head_tile=16, shared_kv_bytes=65536,
                  formal_changed=False, device_validated=False)
    (HERE / "ATTENTION_MERGE16_PREPARE.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(report)


if __name__ == "__main__":
    main()
