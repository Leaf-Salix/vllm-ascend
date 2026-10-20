# SPDX-License-Identifier: Apache-2.0
"""复用13259f92/e91e9f17的共同RoPE表：Query、KV、Indexer使用同一生产者。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE, section


def migrate_projection(source):
    # 原Q/KV已经共用这份八行prepare；只返回真实TaskId供外部接入，不新造dummy。
    begin = source.index("def rope_prepare(")
    end = source.index("\n# NZ 与 ND", begin)
    body = source[begin:end]
    body = body.replace("    for qrp_worker in pl.spmd(\n", "    with pl.spmd(\n", 1)
    body = (
        body.replace(
            "    ):\n        # The pair lane",
            "    ) as shared_rope_tid:\n        qrp_worker = pl.tile.get_block_idx()\n        # The pair lane",
            1,
        ).rstrip()
        + "\n    return shared_rope_tid\n\n\n"
    )
    source = source[:begin] + body + source[end:]
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "qkv_proj_rope")
    lines = source.splitlines(keepends=True)
    original = "".join(lines[function.decorator_list[0].lineno - 1 : function.end_lineno])
    prepared = original.replace("def qkv_proj_rope(", "def qkv_proj_rope_prepared(", 1)
    prepared = prepared.replace("    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],\n", "", 1)
    prepared = prepared.replace(
        "    late_dep: pl.Scalar[pl.TASK_ID],\n",
        "    late_dep: pl.Scalar[pl.TASK_ID],\n"
        "    q_rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],\n"
        "    q_rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],\n",
        1,
    )
    allocation = section(prepared, "    t_dim = pl.tensor.dim(x, 0)\n", "    qa_tid = q_proj_rope(")
    prepared = prepared.replace(allocation, "", 1)
    prepared = prepared.replace(
        '"""Fused q + kv projection: both branches share one token axis and one rope table."""',
        '"""Q/KV消费外层共享RoPE表；Indexer复用同一sin符号，不改变逐元素数学。"""',
        1,
    )
    header = original[: original.index("    t_dim =")]
    arguments = [a.arg for a in function.args.args if a.arg != "rope_sin"]
    wrapper = (
        header
        + allocation
        + (
            "    return qkv_proj_rope_prepared(\n"
            + "".join("        " + name + ",\n" for name in arguments)
            + "        q_rope_sin_signed,\n        q_rope_swap_idx,\n    )\n"
        )
    )
    source = source.replace(original, prepared + "\n\n" + wrapper, 1)
    ast.parse(source)
    return source


def migrate_entry(source):
    source = source.replace(
        "from .qkv_proj_rope import qkv_proj_rope\n",
        "from .qkv_proj_rope import qkv_proj_rope_prepared, rope_prepare\n",
        1,
    )
    original = section(
        source,
        "        # 向上取整分块并保留 valid_shape：",
        "        q = pl.create_tensor(",
    )
    new = (
        "        # 与正式单SPMD相同：Query、KV、Indexer共用sin符号，独立head数学不变。\n"
        "        # 使用原八行prepare的尾块保护；真实TaskId继续保护原late_dep链。\n"
        "        shared_rope_swap = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.INT32)\n"
        "        rope_tid = rope_prepare(freqs_sin, idx_sin_signed, shared_rope_swap)\n\n"
    )
    source = source.replace(original, new, 1)
    tree = ast.parse(source)
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "qkv_proj_rope"
    )
    if not isinstance(call.args[6], ast.Name) or call.args[6].id != "freqs_sin":
        raise AssertionError("QKV调用的sin参数位置已变化")
    call.func.id = "qkv_proj_rope_prepared"
    call.args.pop(6)
    call.args += [ast.Name(id="idx_sin_signed", ctx=ast.Load()), ast.Name(id="shared_rope_swap", ctx=ast.Load())]
    replacement = textwrap.indent("q, qa_tid = " + ast.unparse(call) + "\n", " " * 12)
    lines = source.splitlines(keepends=True)
    lines[call.lineno - 1 : call.end_lineno] = [replacement]
    source = "".join(lines)
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="core_v2")
    parser.add_argument("--candidate", default="shared_rope")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    for name, transform in (("qkv_proj_rope.py", migrate_projection), ("decode_csa.py", migrate_entry)):
        path = target / name
        path.write_text(transform(path.read_text()))
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                source="13259f92的vector_schedule共享rope_signed",
                changed_files=["qkv_proj_rope.py", "decode_csa.py"],
                preserved="QKV独立调用仍可自行prepare；完整CSA仅一次prepare，原数学和显式late_dep保留",
                removed="重复csa_rope_sign任务及重复sin符号工作区",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
