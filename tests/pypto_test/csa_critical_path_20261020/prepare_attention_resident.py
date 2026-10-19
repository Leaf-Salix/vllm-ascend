# SPDX-License-Identifier: Apache-2.0
"""保留当前正式数学源，复用已验证的独立 KV 写入表达，降低发布区 UB 峰值。

只生成隔离候选，不写正式 ops 或依赖仓。左右半行逐元素使用同一 FP32 分母，
逆 RoPE 从右半行的原 FP32 归一值读取；softmax/在线归并的树和 BF16 边界不变。
这是让 64KiB 独立 KV 缓冲有机会编译的资源改造，不预先声称性能收益。
"""

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018"
OLD = BASE / "attention_kv_half_reserved_20261019"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"源码上下文不唯一，不能强行替换: {old[:90]}")
    return source.replace(old, new)


def main():
    reference = HERE / "source_reference"
    target = HERE / "source_attention_resident"
    target.mkdir(exist_ok=False)
    old_reference = (OLD / "source_reference/attention_stream_stages.py").read_text()
    current = (reference / "attention_stream_stages.py").read_text()
    assert ast.dump(ast.parse(old_reference)) == ast.dump(ast.parse(current))
    for path in reference.glob("*.py"):
        (target / path.name).write_bytes(path.read_bytes())
    source = (OLD / "source_candidate/attention_stream_stages.py").read_text()
    start = source.index("    本候选在整个Attention期间")
    end = source.index("    PLAN_PHYSICAL_ROWS开启时", start)
    source = source[:start] + """    本候选保留64KiB独立KV搬运区，目标是减少KV与softmax/merge临时量
    同址复用造成的串行。此前直接保留64KiB超出UB容量；本版先把最终发布
    的左右256列分别除以同一FP32分母，删除512列FP32拼接临时量，随后才
    拼接BF16输出。逆RoPE仍读取右半行未cast的FP32值，索引行距改为256。
    softmax、在线归并、逐元素除法与BF16舍入语义保持，不改变三槽通知。
    与旧32KiB候选的区别是一次完成64行，不重复两段Scalar循环及GM写出。
    当前仅做资源可行性；实际地址、同步、完整精度及性能必须另外验证。
""" + source[end:]
    source = source.replace("消费者由MGATHER(row, zero)执行最终行范围保护；一次载入64项物理索引，",
                            "消费者显式执行负值/容量外零填充；一次载入64项物理索引，")
    source = source.replace("替代逐行Scalar读GM，ISA内部仍逐行DMA。SWA/能力外回退及所有事件保持。",
                            "以UB索引驱动逐行DMA。SWA/能力外回退及所有事件保持。")
    source = replace_once(source, "KV_DMA_ROWS = 32  # 32KiB持久UB；每lane分两段覆盖64行，不改变Attention数学块。",
                          "KV_DMA_ROWS = ATTN_K_TILE // 2  # 每lane完整64行，64KiB独立UB；数学块仍为128候选。")
    source = replace_once(source, "NOPE_DIM = 448", "NOPE_DIM = 448\nRIGHT_NOPE_DIM = NOPE_DIM - HEAD_DIM // 2  # 右半行中的非RoPE列数192。")
    source = source.replace("独占的32行缓冲", "独占的64行缓冲")
    source = source.replace("调用方保证32项", "调用方保证64项")
    source = source.replace("# 32KiB缓冲在整个循环存活", "# 64KiB缓冲在整个循环存活")
    source = source.replace("# 物理plan的容量和哨兵语义沿用原逻辑。32行是搬运分段，", "# 物理plan的容量和哨兵语义沿用原逻辑。一次覆盖本lane的64行，")
    source = source.replace("# 只有原SWA/逻辑回退需要先清零；物理分支由MGATHER完整写出，", "# 只有原SWA/逻辑回退需要先清零；物理分支由显式DMA/补零完整写出，")
    source = replace_once(source, "pl.tile.muls(m_lane, 2)), NOPE_DIM + 1)",
                          "pl.tile.muls(m_lane, 2)), RIGHT_NOPE_DIM + 1)")
    source = replace_once(source, "pl.tile.ci(0, [1, FINAL_HEAD_TILE], dtype=pl.INT32), HEAD_DIM)",
                          "pl.tile.ci(0, [1, FINAL_HEAD_TILE], dtype=pl.INT32), HEAD_DIM // 2)")
    source = replace_once(source, "                    m_oi = pl.concat(m_left, m_right)\n", "")
    source = replace_once(source, """                    n_full = pl.row_expand_div(m_oi, n_denom)
                    n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
                    m_rope = n_full[0:FINAL_HEAD_TILE, NOPE_DIM:HEAD_DIM]
""", """                    # 逐元素数学相同，但避免先生成32KiB的512列FP32拼接。
                    # 左右都使用同一分母，未改为reciprocal乘法，也未改归约。
                    n_left = pl.row_expand_div(m_left, n_denom)
                    n_left_bf16 = pl.cast(n_left, target_type=pl.BF16, mode="rint")
                    n_right = pl.row_expand_div(m_right, n_denom)
                    n_right_bf16 = pl.cast(n_right, target_type=pl.BF16, mode="rint")
                    m_rope = n_right[0:FINAL_HEAD_TILE, RIGHT_NOPE_DIM:HEAD_DIM // 2]
""")
    source = replace_once(source, "pl.reshape(n_full, [1, FINAL_HEAD_TILE * HEAD_DIM]),",
                          "pl.reshape(n_right, [1, FINAL_HEAD_TILE * (HEAD_DIM // 2)]),")
    source = replace_once(source, "                    n_full_bf16 = pl.concat(n_bf16[0:FINAL_HEAD_TILE, 0:NOPE_DIM], n_rope_bf16)",
                          """                    n_right_rotated = pl.concat(
                        n_right_bf16[0:FINAL_HEAD_TILE, 0:RIGHT_NOPE_DIM], n_rope_bf16
                    )
                    n_full_bf16 = pl.concat(n_left_bf16, n_right_rotated)""")
    ast.parse(source)
    (target / "attention_stream_stages.py").write_text(source)
    different = [p.name for p in reference.glob("*.py") if p.read_bytes() != (target / p.name).read_bytes()]
    assert different == ["attention_stream_stages.py"]
    # 对每头64个RoPE位置，旧512列拼接和新256列右半行索引须指向同一元素。
    for head in range(16):
        for lane in range(64):
            old_index = head * 512 + 448 + (lane ^ 1)
            new_index = head * 256 + 192 + (lane ^ 1)
            assert (old_index // 512, old_index % 512) == (new_index // 256, new_index % 256 + 256)
    report = dict(changed=different, formal_source_changed=False,
                  base_attention_matches_e46=True, rope_index_checks=16 * 64,
                  reserved_kv_bytes=64 * 512 * 2, device_validated=False,
                  scope="整数地址检查和Python语法通过；不是PyPTO编译、资源分配或真机通过")
    (HERE / "ATTENTION_RESIDENT_PREPARE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
