# SPDX-License-Identifier: Apache-2.0
"""移植27711336的连续scale布局证明和批量读取，证明改为独立多task生产者。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE, section


def proof_module(donor):
    constants = section(donor, "LAYOUT_WORDS_PER_REQUEST = 2", "@pl.jit.inline")
    body = section(donor, "    requests = pl.cast", "    return workspace")
    body = body.replace("    base = indexer_scale_layout_base(requests, tokens, WORKERS)\n", "    base = 0\n")
    body = body.replace(
        "        begin, end, active = request_bounds(bounds, lengths, request, tokens)\n",
        "        # 多task入口原有S6合同；不从capacity猜活动性，dummy长度必须非正。\n"
        "        active = pl.cast(0, pl.INT32)\n"
        "        if request * DECODE_SEQ < tokens:\n"
        "            if pl.read(lengths, [request]) > 0:\n"
        "                active = pl.cast(1, pl.INT32)\n",
    )
    header = '''# SPDX-License-Identifier: Apache-2.0
"""单SPMD已验证连续页证明的S6多task适配；唯一Scalar写者，每次覆盖全部实际请求。

只读Native页表/长度，不读取cache payload或改变cache布局。证明包括全部有效页，
严格升序/降序连续才返回非零方向，碎片/失活/容量外均返回(0,0)。Score显式等待本
task完成后读取；不能仅依赖函数返回别名或把首尾连续当作全表连续。
"""
import pypto.language as pl
from .config import DECODE_BATCH, DECODE_SEQ, TP
from .layout import INDEXER_TABLE_COLUMNS_DYN

B_DYN = pl.dynamic("B_DYN")
SCALE_LAYOUT_CAPACITY = 2 * (DECODE_BATCH // TP)
'''
    signature = """
@pl.jit.inline(auto_scope=False)
def prove_scale_layout(
    lengths: pl.Tensor[[B_DYN], pl.INT32],
    block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
    workspace: pl.InOut[pl.Tensor[[SCALE_LAYOUT_CAPACITY], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    physical_pages: pl.Scalar[pl.INDEX],
):
"""
    result = header + "\n" + constants + signature + body
    result += (
        "    # 对应单SPMD发布前的Scalar缓存维护；跨task依赖另由调用方显式建立。\n"
        "    pl.system.cacheinvalid()\n"
        "    return workspace\n"
    )
    ast.parse(result)
    return result


def migrate(source, donor):
    source = source.replace(
        "import pypto.language as pl\n",
        "import pypto.language as pl\n\nfrom .indexer_scale_layout import SCALE_LAYOUT_CAPACITY, prove_scale_layout\n",
        1,
    )
    start = source.index("def indexer_score_topk_native_cube(")
    end = source.index("def indexer_score_topk_forest(", start)
    kernel = source[start:end]
    marker = "    # 每个worker两个通信槽，每槽容纳整个query组的分数。\n"
    proof = """    # 512B容量沿已有B<=64合同；关闭分支无消费者，由编译器消去。
    scale_layout = pl.create_tensor([SCALE_LAYOUT_CAPACITY], dtype=pl.INT32)
    layout_deps = pl.array.create(1, pl.TASK_ID)
    layout_deps[0] = coefficients_tid
    if balance_leaves:
        # 页表/长度已是Native本轮输入，无需等待Q/权重或cache payload更新。
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="indexer_scale_layout",
                   allow_early_resolve=True) as proof_tid:
            scale_layout = prove_scale_layout(
                kv_seq_lens, idx_block_table, scale_layout,
                pl.tensor.dim(position_ids, 0), pl.tensor.dim(idx_native_kv_cache, 0),
            )
        layout_deps[0] = proof_tid
"""
    if kernel.count(marker) != 1:
        raise AssertionError("Score工作区入口不唯一")
    kernel = kernel.replace(marker, proof + marker, 1)
    kernel = kernel.replace(
        "        deps=[coefficients_tid, cache_write_tid],",
        "        deps=[coefficients_tid, cache_write_tid, layout_deps[0]],",
        1,
    )
    marker = "                        # Native QLI ProcessVec1 assigns complete query rows to each AIV.\n"
    layout = """                        # 同一调用的完整页证明；后续每个leaf/两个AIV复用，不重复扫描。
                        single_page_begin = single_begin // BLOCK_SIZE
                        single_page_count = (single_valid + BLOCK_SIZE - 1) // BLOCK_SIZE
                        layout_row = 2 * single_batch
                        single_contiguous = pl.read(scale_layout, [layout_row + 1])
                        single_first_page = pl.cast(pl.read(scale_layout, [layout_row]), pl.INDEX)
                        single_first_page = (
                            single_first_page + pl.cast(single_contiguous, pl.INDEX) * single_page_begin
                        )
"""
    if kernel.count(marker) != 1:
        raise AssertionError("balanced Vector入口不唯一")
    kernel = kernel.replace(marker, layout + marker, 1)
    old = section(
        kernel,
        "                            single_scale_bytes = pl.tile.create(",
        "                            single_scale_half =",
    )
    # 保留原逐页逻辑作碎片回退，仅采用同一已验证二维读取/逆序页恢复。
    new = section(
        donor,
        "                    single_scale_bytes = pl.tile.reinterpret_view(",
        "                    single_scale_half =",
    )
    new = new.replace("scale_page_coalescing and single_contiguous != 0", "single_contiguous != 0")
    kernel = kernel.replace(old, textwrap.indent(new, " " * 8), 1)
    result = source[:start] + kernel + source[end:]
    ast.parse(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="physical_static")
    parser.add_argument("--candidate", default="scale_pages")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    donor_root = HERE / "sources/spmd"
    (target / "indexer_scale_layout.py").write_text(
        proof_module((donor_root / "indexer_scale_layout_stages.py").read_text())
    )
    path = target / "decode_indexer.py"
    path.write_text(migrate(path.read_text(), (donor_root / "indexer_score_stages.py").read_text()))
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                source_commit="27711336",
                changed_files=[path.name, "indexer_scale_layout.py"],
                strategy="长G6 balanced；布局证明独立提前调度，按完整效果决定是否沿用单SPMD域限制",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
