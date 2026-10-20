# SPDX-License-Identifier: Apache-2.0
"""迁移05d2a3e9/289cc072物理行plan与MGATHER；生成独立完整多task实验包。"""

import argparse
import ast
import json
import re
import shutil
import textwrap

from prepare import HERE, section


def physical_helper(donor):
    body = section(donor, "                    p_raw = pl.slice(", "                    # 逐行写既有GM")
    body = textwrap.dedent(body)
    body = body.replace("p_raw = pl.slice(c_page_indices, [1, TOPK], [row, 0], valid_shape=[1, TOPK])\n", "").replace(
        "c_pages_valid = pl.tile.assemble(c_pages_valid, p_valid, [row, 0])\n", ""
    )
    # 原plan为显式Tile，多task现有plan为Tensor IR；使用对应的同整数/浮点操作。
    body = body.replace("pl.tile.maximums(", "pl.maximum(").replace("pl.tile.minimums(", "pl.minimum(")
    body = body.replace("pl.tile.shrs(", "pl.shrs(").replace("pl.tile.shls(", "pl.shls(")
    body = body.replace("pl.tile.sub(", "pl.sub(").replace("pl.tile.add(", "pl.add(")
    body = body.replace("p_scratch = pl.create_tile([1, TOPK], dtype=pl.INT32)\n", "")
    body = body.replace("pl.tile.gather(p_table, p_safe_columns, p_scratch)", "pl.gather(p_table, p_safe_columns)")
    header = '''# SPDX-License-Identifier: Apache-2.0
"""已验证单SPMD的选中页检查与INT32物理行转换，供多task plan内联使用。

调用方统一证明T>=64、0<页表列<=2048、0<物理页<=2**24；不满足则整体
使用旧逻辑表示。索引转换用INT32位移加法，地址不经FP32；范围比较沿原
FP32精确整数域。外部Top-K保持不变，内部无效物理行写-1，由MGATHER填零。
"""
import pypto.language as pl
from .layout import COMPRESSED_TABLE_COLUMNS_DYN

B_DYN = pl.dynamic("B_DYN")
TOPK = 512
PHYSICAL_PLAN_COLUMNS = 2048
PHYSICAL_PLAN_MAX_PAGES = 1 << 24
PHYSICAL_PLAN_MIN_TOKENS = 64


@pl.jit.inline(auto_scope=False)
def compressed_physical_plan(
    p_raw: pl.Tensor[[1, TOPK], pl.INT32],
    compressed_table: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32],
    request: pl.Scalar[pl.INDEX],
    upper: pl.Scalar[pl.INDEX],
    compressed_pages: pl.Scalar[pl.INDEX],
):
'''
    result = header + textwrap.indent(body, "    ") + "    return p_output, p_valid\n"
    ast.parse(result)
    return result


def migrate(source):
    source = source.replace(
        "import pypto.language as pl\n",
        "import pypto.language as pl\n\n"
        "from .physical_attention_plan import (\n"
        "    PHYSICAL_PLAN_COLUMNS, PHYSICAL_PLAN_MAX_PAGES, PHYSICAL_PLAN_MIN_TOKENS,\n"
        "    compressed_physical_plan,\n"
        ")\n",
        1,
    )
    start = source.index("def sparse_attn_csa(")
    end = source.index("\n)", start)
    source = source[:end] + "\n    PHYSICAL_ROWS: pl.constexpr," + source[end:]
    original_store = section(source, "            cmp_sparse_indices = pl.assemble(", "            sparse_bias =")
    physical_store = """if PHYSICAL_ROWS:
    # 使用完整8行页mask；逐行写入后再限定valid_rows，与单SPMD相同。
    c_pages_valid = pl.full([BIAS_T_TILE, IDX_TOPK], dtype=pl.FP32, value=0.0)
    for plan_row in pl.range(bias_rows):
        plan_token = bias_t0 + plan_row
        plan_request = plan_token // S
        # 与现有HC内联入口相同：声明静态metadata供JIT推导，立即被计算值替换。
        plan_raw = pl.create_tensor([1, IDX_TOPK], dtype=pl.INT32)
        plan_raw = pl.cast(
            pl.slice(c_out_tile, [1, IDX_TOPK], [plan_row, 0], valid_shape=[1, IDX_TOPK]),
            target_type=pl.INT32,
        )
        plan_upper = pl.cast(pl.cast(pl.read(c_pos_q, [plan_row, 0]), pl.INT32), pl.INDEX)
        plan_physical, plan_valid = compressed_physical_plan(
            plan_raw, cmp_block_table, plan_request, plan_upper, pl.tensor.dim(cmp_kv, 0),
        )
        cmp_sparse_indices = pl.assemble(cmp_sparse_indices, plan_physical, [plan_token, 0])
        c_pages_valid = pl.assemble(c_pages_valid, plan_valid, [plan_row, 0])
    c_mask_tile = pl.mul(c_mask_tile, pl.set_validshape(c_pages_valid, bias_rows, IDX_TOPK))
    # 从原始raw重算一次原规范公式；不把物理地址或重复舍入混入bias。
    c_out_tile = pl.sub(pl.mul(c_mask_tile, pl.add(c_raw_tile, 1.0)), 1.0)
else:
"""
    source = source.replace(
        original_store,
        textwrap.indent(physical_store, " " * 12) + textwrap.indent(original_store, " " * 4),
        1,
    )
    original_gather = section(
        source, "                        qk_kv_half = pl.tile.full(", "                        pl.store(qk_kv_half,"
    )
    new_gather = """if PHYSICAL_ROWS and qk_sb > 0:
    # 与本轮plan共享同一表示和容量条件。MGATHER仍逐行DMA，不是并行采64行。
    physical_column = (qk_sb - 1) * ATTN_K_TILE + qk_lane_kv
    physical_rows = pl.load(
        cmp_sparse_indices, [qk_t, physical_column], [1, ATTN_K_TILE // 2],
        target_memory=pl.MemorySpace.Vec,
    )
    qk_kv_half = pl.tile.mgather(cmp_kv_flat, physical_rows, coalesce="row", gather_oob="zero")
else:
"""
    source = source.replace(
        original_gather,
        textwrap.indent(new_gather, " " * 24) + textwrap.indent(original_gather, " " * 4),
        1,
    )
    # orchestration不支持And表达式；用已存在的嵌套if/标志方式选择四种constexpr组合。
    selector = """    use_physical_rows = pl.cast(0, pl.INT32)
    if t_dim >= PHYSICAL_PLAN_MIN_TOKENS:
        if pl.tensor.dim(cmp_block_table, 1) > 0:
            if pl.tensor.dim(cmp_block_table, 1) <= PHYSICAL_PLAN_COLUMNS:
                if pl.tensor.dim(cmp_kv, 0) > 0:
                    if pl.tensor.dim(cmp_kv, 0) <= PHYSICAL_PLAN_MAX_PAGES:
                        use_physical_rows = pl.cast(1, pl.INT32)
"""
    wrapper = source.index("def sparse_attn_csa_tp1(")
    old = section(source[wrapper:], "    if use_local_stats == 1:", "    return packed, ready")
    calls = re.findall(r"        packed, ready = sparse_attn_csa\([\s\S]*?\n        \)\n", old)
    if len(calls) != 2:
        raise AssertionError("需要原local-stats开关的两个调用")
    branches = []
    for index, call in enumerate(calls):
        branches.append("    if use_local_stats == 1:\n" if index == 0 else "    else:\n")
        branches.append("        if use_physical_rows == 1:\n")
        branches.append(textwrap.indent(call.replace(",\n        )", ", True,\n        )"), " " * 4))
        branches.append("        else:\n")
        branches.append(textwrap.indent(call.replace(",\n        )", ", False,\n        )"), " " * 4))
    source = source[:wrapper] + source[wrapper:].replace(old, selector + "".join(branches), 1)
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="score_stats_ob")
    parser.add_argument("--candidate", default="physical_plan")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    source = HERE / "sources" / args.base
    destination = HERE / "sources" / args.candidate
    shutil.copytree(source, destination)
    donor = (HERE / "sources/spmd/attention_plan_stages.py").read_text()
    (destination / "physical_attention_plan.py").write_text(physical_helper(donor))
    target = destination / "decode_sparse_attn_csa.py"
    target.write_text(migrate(target.read_text()))
    (destination / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                source_commits=["05d2a3e9", "289cc072"],
                changed_files=["decode_sparse_attn_csa.py", "physical_attention_plan.py"],
                scope="T>=64且支持的页表/物理页容量；不修改Native cache布局和外部TopK",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(destination)


if __name__ == "__main__":
    main()
