# SPDX-License-Identifier: Apache-2.0
"""将已验证scale页证明迁到CSA前段；保留显式生产消费依赖及原数学。"""

import argparse
import ast
import json
import shutil

from prepare import HERE, section

CHAIN = ("indexer_score_topk_native_cube", "indexer_score_topk_forest", "indexer_weights_score", "indexer")


def add_parameters(source):
    for name in CHAIN:
        begin = source.index("def " + name + "(")
        end = source.index("\n)", begin)
        source = (
            source[:end]
            + (
                "\n    scale_layout: pl.Tensor[[SCALE_LAYOUT_CAPACITY], pl.INT32],"
                "\n    layout_tid: pl.Scalar[pl.TASK_ID],"
            )
            + source[end:]
        )
    return source


def add_calls(source, names, arguments):
    tree = ast.parse(source)
    calls = [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names
    ]
    lines = source.splitlines(keepends=True)
    for node in sorted(calls, key=lambda n: n.lineno, reverse=True):
        close = lines[node.end_lineno - 1]
        if node.lineno == node.end_lineno:
            position = node.end_col_offset - 1
            if close[position] != ")":
                raise AssertionError("单行调用结束位置不是右括号")
            lines[node.end_lineno - 1] = close[:position] + ", " + ", ".join(arguments) + close[position:]
            continue
        if close.strip() != ")":
            raise AssertionError("需要独立闭合行以保留调用注释")
        indent = " " * (len(close) - len(close.lstrip()) + 4)
        lines.insert(node.end_lineno - 1, "".join(indent + arg + ",\n" for arg in arguments))
    if not calls:
        raise AssertionError("未定位参数传递链")
    return "".join(lines)


def migrate_indexer(source):
    old = section(source, "    # 512B容量沿已有B<=64合同", "    # 每个worker两个通信槽")
    source = source.replace(old, "", 1)
    source = source.replace(
        "from .indexer_scale_layout import SCALE_LAYOUT_CAPACITY, prove_scale_layout",
        "from .indexer_scale_layout import SCALE_LAYOUT_CAPACITY",
    )
    source = source.replace(
        "deps=[coefficients_tid, cache_write_tid, layout_deps[0]],",
        "deps=[coefficients_tid, cache_write_tid, proof_dependencies[0]],",
        1,
    )
    marker = "    # 每个worker两个通信槽"
    source = source.replace(
        marker,
        "    # 显式值通道，避免多层inline后仅attrs引用的TaskId失去绑定。\n"
        "    proof_dependencies = pl.array.create(1, pl.TASK_ID)\n"
        "    proof_dependencies[0] = layout_tid\n" + marker,
        1,
    )
    source = add_parameters(source)
    source = add_calls(source, CHAIN, ("scale_layout", "layout_tid"))
    ast.parse(source)
    return source


def migrate_entry(source):
    source = source.replace(
        "from .decode_indexer import indexer\n",
        "from .decode_indexer import indexer, LONG_S6_MIN_QUERY_ROWS, TOPK_CANDIDATES_PER_LEAF\n"
        "from .indexer_scale_layout import SCALE_LAYOUT_CAPACITY, prove_scale_layout\n",
        1,
    )
    marker = "        with pl.scope():\n            pre_val_store ="
    insertion = """        # 27711336前段证明的多task适配：尺寸在编排侧解析，证明只读页表/长度。
        # 不让Tensor.dim在证明incore中引入对cache payload写回的伪依赖。
        early_scale_pages = pl.tensor.dim(idx_native_kv_cache, 0)
        early_scale_layout = pl.create_tensor([SCALE_LAYOUT_CAPACITY], dtype=pl.INT32)
        early_scale_deps = pl.array.create(1, pl.TASK_ID)
        early_scale_deps[0] = _widen_tid
        if max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:
            if t_dim >= LONG_S6_MIN_QUERY_ROWS:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="indexer_scale_layout",
                           allow_early_resolve=True) as early_scale_tid:
                    early_scale_layout = prove_scale_layout(
                        kv_seq_lens, idx_block_table, early_scale_layout, t_dim, early_scale_pages,
                    )
                early_scale_deps[0] = early_scale_tid

"""
    if source.count(marker) != 1:
        raise AssertionError("需要唯一HC前段scope")
    source = source.replace(marker, insertion + marker, 1)
    source = add_calls(source, ("indexer",), ("early_scale_layout", "early_scale_deps[0]"))
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="scale_pages")
    parser.add_argument("--candidate", default="scale_early")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    for name, transform in (("decode_indexer.py", migrate_indexer), ("decode_csa.py", migrate_entry)):
        path = target / name
        path.write_text(transform(path.read_text()))
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="27711336",
                changed_files=["decode_csa.py", "decode_indexer.py"],
                reason="将页证明放在HC前段重叠，并把cache尺寸读取移到编排侧；Score仍显式等待证明",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
