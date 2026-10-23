# SPDX-License-Identifier: Apache-2.0
"""离线生成私有内部工作区原型；数学体取冻结源码，不在执行时生成。"""

import argparse
import ast
import copy
from pathlib import Path


class ShapeExpression(ast.NodeTransformer):
    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node) != "2 * b if SCALE_PAGE_COALESCING else 0":
            raise ValueError(f"未处理的工作区尺寸分支: {ast.unparse(node)}")
        return ast.copy_location(ast.Name("coalescing_rows", ast.Load()), node)

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
                    ast.BinOp(
                        ast.BinOp(
                            ast.BinOp(value, ast.Add(), copy.deepcopy(size)),
                            ast.Sub(),
                            ast.Constant(1),
                        ),
                        ast.FloorDiv(),
                        copy.deepcopy(size),
                    ),
                    ast.Mult(),
                    copy.deepcopy(size),
                ),
                node,
            )
        return node


def external_returns_only(function):
    names = {argument.arg for argument in function.args.args}
    tail = function.body[-1]
    if not isinstance(tail, ast.Return):
        return
    assert isinstance(tail.value, ast.Tuple)
    kept = [value for value in tail.value.elts if isinstance(value, ast.Name) and value.id in names]
    assert any(value.id == "output" for value in kept)
    tail.value = ast.Tuple(kept, ast.Load())


def make(source, package, function, *, direct_body=False, scale_task_boundary=False):
    directory = source / "vllm_ascend/ops/pypto" / package
    module = ast.parse((directory / "entry.py").read_text())
    body = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == function)
    if scale_task_boundary and not package.endswith("spmd_tnd"):
        helper = """\n\n@pl.jit.incore
def indexer_scale_page_commit_task(
    scales: pl.Tensor[[ISCALES, 1], pl.FP32],
    offsets: pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32],
    cache: pl.InOut[pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8]],
    tokens: pl.Scalar[pl.INDEX],
    requests: pl.Scalar[pl.INDEX],
):
    cache = indexer_scale_page_commit_stage(
        scales, offsets, cache, tokens, requests, pl.tile.get_block_idx(), VECTORS
    )
    return cache
"""
        path = directory / "entry.py"
        assert "def indexer_scale_page_commit_task(" not in path.read_text()
        path.write_text(path.read_text() + helper)
        replacements = 0
        for node in ast.walk(body):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "indexer_scale_page_commit_stage"
            ):
                assert ast.unparse(node.args[-2]) == "pl.tile.get_block_idx()"
                assert ast.unparse(node.args[-1]) == "VECTORS"
                node.func.id = "indexer_scale_page_commit_task"
                node.args = node.args[:-2]
                replacements += 1
        assert replacements == 1
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
            "    zero = pl.tile.full([1, 11520], dtype=pl.INT32, value=0)",
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
        "    p8 = (t + 7) // 8 * 8",
        "    p16 = (t + 15) // 16 * 16",
        "    p64 = (t + 63) // 64 * 64",
        "    p128 = (t + 127) // 128 * 128",
        "    group = QUERY_GROUP",
        "    mm = pl.max(p128, (t + OB_ROWS - 1) // OB_ROWS * OB_ROWS)",
    ]
    if spmd:
        code += [
            "    coalescing_rows = 0",
            "    if SCALE_PAGE_COALESCING:",
            "        coalescing_rows = 2 * b",
        ]
    for name, shape, dtype in allocations:
        code.append(f"    {name} = pl.create_tensor({shape}, dtype=pl.{dtype})")
    code += [
        "    q_heads = pl.reshape(q_flat, [t * 64, 512])",
        "    rope_rows = pl.tensor.dim(idx_query_rope, 0)",
        "    idx_query_rope_heads = pl.reshape(idx_query_rope, [rope_rows * 64, 128])",
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
    if direct_body:
        wrapper = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function + "_internal_debug"
        )
        assert isinstance(wrapper.body[-1], ast.Expr)
        wrapper.body[-1:] = copy.deepcopy(body.body)
        external_returns_only(wrapper)
        tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name != "_arithmetic_body"]
        text = ast.unparse(tree) + "\n"
    normal = copy.deepcopy(
        next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function + "_internal_debug")
    )
    normal.name = function + "_internal"
    normal.args.args = [a for a in normal.args.args if a.arg not in ("topk_indices", "topk_scores")]
    normal.body[1:1] = ast.parse(
        "topk_indices = pl.create_tensor([t, 512], dtype=pl.INT32)\n"
        "topk_scores = pl.create_tensor([t, 512], dtype=pl.FP32)\n"
    ).body
    external_returns_only(normal)
    text += "\n" + ast.unparse(normal) + "\n"
    target.write_text(text)
    print(target, "normal_internal_buffers", len(allocations) + 2, "normal_external_arguments", len(signature) - 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--direct-body", action="store_true", help="保持原编排函数体层级，避免额外inline包装")
    parser.add_argument("--scale-task-boundary", action="store_true", help="MPMD的scale写回使用独立incore参数边界")
    args = parser.parse_args()
    for package, function in (
        ("deepseek_v4_flash_csa_spmd_tnd", "decode_csa_spmd"),
        ("deepseek_v4_flash_csa_tnd", "decode_csa_tnd"),
    ):
        make(args.source, package, function, direct_body=args.direct_body, scale_task_boundary=args.scale_task_boundary)


if __name__ == "__main__":
    main()
