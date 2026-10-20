# SPDX-License-Identifier: Apache-2.0
"""迁移66882634小G6域的M16 QA分核，保留原数学与其它资源域。"""

import argparse
import ast
import json
import re
import shutil
import textwrap

from prepare import HERE
from prepare_fused_topk import add_parameter
from prepare_scale_early import add_calls


def m16_helper(donor):
    tree = ast.parse(donor)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_q_a_m16_parallel_worker")
    lines = donor.splitlines(keepends=True)
    start = next(n.lineno for n in fn.body if isinstance(n, ast.Assign))
    body = "".join(lines[start - 1 : fn.body[-1].lineno - 1])
    body = body.replace("tokens = pl.tensor.dim(x, 0)", "tokens = tile_rows")
    body = re.sub(r"\bqa_fp32\b", "qr_fp32", body)
    body = re.sub(r"\bWORKERS\b", "QA_M16_WORKER_LIMIT", body)
    body = body.replace("[row, k0]", "[tile_base + row, k0]")
    header = '''
QA_M16_WORKER_LIMIT = QR_N_BLOCKS * QR_FIXED_M_GROUPS


@pl.jit.inline(auto_scope=False)
def q_proj_qa_m16_parallel(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    qr_fp32: pl.Tensor[[QPROJ_MM_T_DYN, Q_LORA], pl.FP32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """66882634：每个M16/N128独立输出分核，K次序及实际尾行保持。"""
    launch_workers = pl.min(
        QA_M16_WORKER_LIMIT,
        QR_N_BLOCKS * ((tile_rows + QR_M_TILE - 1) // QR_M_TILE),
    )
    with pl.spmd(launch_workers, name_hint="qr_proj_matmul_m16",
                 allow_early_resolve=True) as qa_tid:
        pl.set_cache_policy(wq_a, pl.CachePolicy.BYPASS)
        worker_id = pl.tile.get_block_idx()
'''
    return header + textwrap.indent(body, "    ") + "    return qa_tid\n\n\n"


def migrate_projection(source, donor):
    tree = ast.parse(source)
    names = ("q_proj_qa", "q_proj_qr", "q_proj_rope")
    old_lines = source.splitlines(keepends=True)
    legacy = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            first_statement = node.body[1] if isinstance(node.body[0], ast.Expr) else node.body[0]
            header = "".join(old_lines[node.decorator_list[0].lineno - 1 : first_statement.lineno - 1])
            legacy[node.name] = (
                header
                + "    return "
                + node.name
                + "_policy(\n"
                + "".join("        " + arg.arg + ",\n" for arg in node.args.args)
                + "        pl.cast(0, pl.INT32),\n    )\n"
            )
    old_qkv = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "qkv_proj_rope")
    old_qkv_text = "".join(old_lines[old_qkv.decorator_list[0].lineno - 1 : old_qkv.end_lineno])
    old_header = old_qkv_text[: old_qkv_text.index("    t_dim = pl.tensor.dim(x, 0)")]
    marker = "@pl.jit.inline(auto_scope=False)\ndef q_proj_qa("
    if source.count(marker) != 1:
        raise AssertionError("需要唯一QA策略入口")
    source = source.replace(marker, m16_helper(donor) + marker, 1)
    argument = "    qa_m16_parallel: pl.Scalar[pl.INT32],\n"
    for name in names:
        source = add_parameter(source, name, argument)
    source = add_calls(source, names, ("qa_m16_parallel",))
    source = source.replace("def qkv_proj_rope(", "def qkv_proj_rope_policy(", 1)
    source = add_parameter(source, "qkv_proj_rope_policy", argument)
    begin = source.index("def q_proj_qa(")
    end = source.index("@pl.jit.inline(auto_scope=False)\ndef q_proj_qr(", begin)
    strategy = source[begin:end]
    marker = "    elif tile_rows < QR_FIXED_SMALL_ROWS:\n"
    strategy = strategy.replace(
        marker,
        (
            "    elif qa_m16_parallel != 0:\n"
            "        qa_tid = q_proj_qa_m16_parallel(x, wq_a, qr_fp32, tile_base, tile_rows)\n"
        )
        + marker,
        1,
    )
    source = source[:begin] + strategy + source[end:]
    # 保留原QKV调用接口；只有完整CSA显式传递已验证的资源域选择。
    wrapper = old_header + (
        "    return qkv_proj_rope_policy(\n"
        + "".join("        " + arg.arg + ",\n" for arg in old_qkv.args.args)
        + "        pl.cast(0, pl.INT32),\n    )\n"
    )
    source += "\n\n" + wrapper
    # HCA仍导入q_proj_qr等辅助入口：旧签名保留，CSA显式走带策略的内部入口。
    for name in names:
        source = re.sub(r"\b" + name + r"(?=\()", name + "_policy", source)
    source += "\n\n" + "\n\n".join(legacy[name] for name in names)
    ast.parse(source)
    return source


def migrate_entry(source):
    source = source.replace(
        "from .qkv_proj_rope import qkv_proj_rope\n",
        "from .qkv_proj_rope import QR_M_TILE, qkv_proj_rope_policy\n",
        1,
    )
    marker = "            q, qa_tid = qkv_proj_rope(\n"
    if source.count(marker) != 1:
        raise AssertionError("需要唯一整层QKV调用")
    selection = (
        "            # 66882634：长G6且16<T<32；按实际T选择，不枚举batch点。\n"
        "            qa_m16_parallel = pl.cast(0, pl.INT32)\n"
        "            if max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:\n"
        "                if t_dim >= LONG_S6_MIN_QUERY_ROWS:\n"
        "                    if t_dim < 2 * QR_M_TILE:\n"
        "                        qa_m16_parallel = pl.cast(1, pl.INT32)\n"
    )
    source = source.replace(marker, selection + marker.replace("qkv_proj_rope(", "qkv_proj_rope_policy("), 1)
    source = add_calls(source, ("qkv_proj_rope_policy",), ("qa_m16_parallel",))
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="core_v2")
    parser.add_argument("--candidate", default="qa_m16")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    path = target / "qkv_proj_rope.py"
    donor = (HERE / "sources/spmd/qkv_stages.py").read_text()
    path.write_text(migrate_projection(path.read_text(), donor))
    path = target / "decode_csa.py"
    path.write_text(migrate_entry(path.read_text()))
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="66882634",
                changed_files=["qkv_proj_rope.py", "decode_csa.py"],
                scope="固定K路径长G6且16<T<32；实际尾行、K顺序、原QA依赖不变",
                preserved="原QKV调用接口；其它工作量及atomic路径不采用新QA分工",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
