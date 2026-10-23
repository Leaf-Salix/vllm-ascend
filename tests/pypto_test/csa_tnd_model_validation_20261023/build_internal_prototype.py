# SPDX-License-Identifier: Apache-2.0
"""离线生成私有内部工作区原型；数学体取冻结源码，不在执行时生成。"""

import argparse
import ast
import copy
from pathlib import Path


class ShapeExpression(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Attribute) and ast.unparse(node.value) == "self.policy":
            return ast.copy_location(ast.Name(node.attr.upper(), ast.Load()), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "_pad":
            value, size = node.args
            return ast.copy_location(
                ast.BinOp(
                    ast.Call(ast.Attribute(ast.Name("pl", ast.Load()), "cdiv", ast.Load()), [value, size], []),
                    ast.Mult(),
                    copy.deepcopy(size),
                ),
                node,
            )
        return node


def make(source, package, function):
    directory = source / "vllm_ascend/ops/pypto" / package
    module = ast.parse((directory / "entry.py").read_text())
    body = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == function)
    binding = ast.parse((directory / "binding.py").read_text())
    specs = next(
        n.value
        for n in ast.walk(binding)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "specs" for t in n.targets)
    )
    allocations = []
    dtypes = {
        "float32": "FP32",
        "bfloat16": "BF16",
        "float16": "FP16",
        "int8": "INT8",
        "int32": "INT32",
        "int64": "INT64",
    }
    for key, value in zip(specs.keys, specs.values):
        shape, dtype = value.elts
        shape = ShapeExpression().visit(copy.deepcopy(shape))
        # 当前JIT的中间Tensor元数据推导识别list形状；不要沿用Torch规格中的tuple。
        shape = ast.List(shape.elts, ast.Load())
        allocations.append((key.value, ast.unparse(shape), dtypes[dtype.attr]))
    spmd = package.endswith("spmd_tnd")
    if spmd:
        allocations.append(("signal", "[60 * 72 * 128]", "INT32"))
    scratch = {name for name, _, _ in allocations}
    aliases = {"q_heads", "idx_query_rope_heads", "idx_query_scale_heads"}
    remove = scratch | aliases
    arguments = body.args.args
    assert not body.args.kwonlyargs and not body.args.vararg
    assert not (remove - {a.arg for a in arguments}), remove - {a.arg for a in arguments}
    required = len(arguments) - len(body.args.defaults)
    signature = []
    for index, argument in enumerate(arguments):
        if argument.arg in remove:
            continue
        line = f"    {argument.arg}: {ast.unparse(argument.annotation)}"
        if index >= required:
            line += " = " + ast.unparse(body.args.defaults[index - required])
        signature.append(line + ",")
    locals_ = {n.id for n in ast.walk(body) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    private = sorted(
        {
            n.id
            for n in ast.walk(body)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id.startswith("_")
        }
        - locals_
        - {a.arg for a in arguments}
    )
    original = copy.deepcopy(body)
    original.name = "_arithmetic_body"
    original.decorator_list = [ast.parse("pl.jit.inline(auto_scope=False)", mode="eval").body]
    code = [
        "# SPDX-License-Identifier: Apache-2.0",
        '"""私有验证原型：Simpler内部申请所有临时GM；Top-K为显式诊断输出。"""',
        "import pypto.language as pl",
        "from .entry import *  # noqa: F403",
    ]
    if private:
        code.append("from .entry import " + ", ".join(private))
    code += ["", ast.unparse(original), ""]
    if spmd:
        code += [
            "@pl.jit.incore",
            "def _clear_internal_signal(area: pl.Out[pl.Tensor[[48, 11520], pl.INT32]]):",
            "    core = pl.tile.get_block_idx()",
            "    zero = pl.full([1, 11520], dtype=pl.INT32, value=0)",
            "    pl.store(zero, [core, 0], area)",
            "    return area",
            "",
        ]
    code += [
        "@pl.jit",
        f"def {function}_internal_debug(",
        *signature,
        "):",
        "    t = pl.tensor.dim(x, 0)",
        "    b = pl.tensor.dim(idx_lengths, 0)",
        "    p8 = pl.cdiv(t, 8) * 8",
        "    p16 = pl.cdiv(t, 16) * 16",
        "    p64 = pl.cdiv(t, 64) * 64",
        "    p128 = pl.cdiv(t, 128) * 128",
        "    group = QUERY_GROUP",
        "    mm = pl.max(p128, pl.cdiv(t, OB_ROWS) * OB_ROWS)",
    ]
    for name, shape, dtype in allocations:
        code.append(f"    {name} = pl.create_tensor({shape}, dtype=pl.{dtype})")
    code += [
        "    q_heads = pl.reshape(q_flat, [t * 64, 512])",
        "    idx_query_rope_heads = pl.reshape(idx_query_rope, [p16 * 64, 128])",
        "    idx_query_scale_heads = pl.reshape(idx_query_scale, [t, 64])",
    ]
    if spmd:
        code += [
            "    signal_matrix = pl.reshape(signal, [48, 11520])",
            "    with pl.spmd(48, sync_start=True):",
            "        signal_matrix = _clear_internal_signal(signal_matrix)",
            "    signal = pl.reshape(signal_matrix, [60 * 72 * 128])",
        ]
    code += ["    _arithmetic_body("]
    code += [f"        {argument.arg}={argument.arg}," for argument in arguments]
    code += ["    )", ""]
    target = directory / "internal_prototype.py"
    text = "\n".join(code)
    tree = ast.parse(text)
    normal = copy.deepcopy(
        next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function + "_internal_debug")
    )
    normal.name = function + "_internal"
    normal.args.args = [a for a in normal.args.args if a.arg not in ("topk_indices", "topk_scores")]
    normal.body[1:1] = ast.parse(
        "topk_indices = pl.create_tensor([t, 512], dtype=pl.INT32)\n"
        "topk_scores = pl.create_tensor([t, 512], dtype=pl.FP32)\n"
    ).body
    text += "\n" + ast.unparse(normal) + "\n"
    target.write_text(text)
    print(target, "normal_internal_buffers", len(allocations) + 2, "normal_external_arguments", len(signature) - 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    make(args.source, "deepseek_v4_flash_csa_spmd_tnd", "decode_csa_spmd")
    make(args.source, "deepseek_v4_flash_csa_tnd", "decode_csa_tnd")


if __name__ == "__main__":
    main()
