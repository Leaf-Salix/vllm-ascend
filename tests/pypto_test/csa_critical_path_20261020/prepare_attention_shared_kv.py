# SPDX-License-Identifier: Apache-2.0
"""让互斥KV供数路径共用一份64KiB专用UB，消除旧候选的第二份64KiB。

源级假设必须由完整资源审计核对：两路径的alias地址相同，仍与softmax、
在线累加和最终发布临时量分离。没有修改容量上限，也没有改数学归约。
"""

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    base = HERE / "source_attention_resident"
    target = HERE / "source_attention_shared_kv"
    target.mkdir(exist_ok=False)
    for path in base.glob("*.py"):
        (target / path.name).write_bytes(path.read_bytes())
    path = target / "attention_stream_stages.py"
    source = path.read_text()
    old = "                    qk_kv_half = pl.tile.full([ATTN_K_TILE // 2, HEAD_DIM], dtype=pl.BF16, value=0.0)\n"
    assert source.count(old) == 1
    source = source.replace(old, """                    # 与上面的压缩KV分支互斥。二者都写同一GM半块，故共用
                    # 同一64KiB专用UB；不能另建64KiB full而与永久缓冲叠加。
                    # 将有效行置零后原地填padding，覆盖全部物理元素为+0。
                    # 不读取旧值做乘零（旧值可能NaN），也不分配第二份零Tile。
                    # fillpad_inplace返回同址完整有效Tile，后续页gather和
                    # store沿原SWA/逻辑地址、valid与事件运行。前拍MTE3读完
                    # 才可Vector清零的依赖仍由编译器保留，不能手工省略。
                    kv_empty = pl.set_validshape(kv_dma_buffer, 0, HEAD_DIM)
                    qk_kv_half = pl.tile.fillpad_inplace(kv_empty, pad_value=pl.PadValue.zero)
""")
    start = source.index("    本候选保留64KiB独立KV搬运区")
    end = source.index("    PLAN_PHYSICAL_ROWS开启时", start)
    source = source[:start] + """    本候选的64KiB独立KV搬运区同时供互斥的压缩/原始SWA/逻辑回退路径
    使用。旧版本给SWA额外分配一份64KiB，而两个32KiB在线状态也跨拍
    存活，导致UB峰值201472B超过188416B；不能把这个超限只归因归约scratch。
    原地清零沿现有fillpad_inplace实现，用零有效行表达全部物理元素清零，
    不读取未初始化数据、不改变容量上限。所有地址守卫、页尾和跨拍依赖保留。
    压缩路径仍用已验证的两遍显式DMA/必要时补零，一次64行；与E46两次
    32行写出不同，不重复两段控制。最终左右256列分别除同一分母的数学
    沿前一资源候选，归约树/FP32与BF16舍入边界保持；不是新数学近似。
    尚须完整CPU确认同址与分离，随后真机冻结检查和同轮完整L1决定采用。
""" + source[end:]
    ast.parse(source)
    path.write_text(source)
    different = [p.name for p in base.glob("*.py") if p.read_bytes() != (target / p.name).read_bytes()]
    assert different == ["attention_stream_stages.py"]
    tree = ast.parse(source)
    # 当前helper仍含两次原归约；没有执行曾考虑的scratch拆分或归约树修改。
    row_reduce = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) in ("pl.row_max", "pl.row_sum")]
    assert len(row_reduce) == 2
    report = dict(base=str(base), changed=different,
                  shared_kv_bytes=65536, previous_two_kv_bytes=131072,
                  current_resource_limit_bytes=188416, limit_unchanged=True,
                  scratch_split_attempted=False, formal_changed=False,
                  device_validated=False, note="源码准备，不是资源或性能通过")
    (HERE / "ATTENTION_SHARED_KV_PREPARE.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(report)


if __name__ == "__main__":
    main()
