# SPDX-License-Identifier: Apache-2.0
"""完整提取已编译 CSA 的核计算体、参数和同步操作；不修改依赖仓库。"""

import argparse
import ast
import collections
import copy
import json
from pathlib import Path


def kernels_from_config(path):
    tree = ast.parse(path.read_text())
    array = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "KERNELS" for t in n.targets))
    kernels = []
    for node in array.elts:
        values = {ast.literal_eval(k): v for k, v in zip(node.keys, node.values)}
        kernels.append({k: ast.literal_eval(values[k]) for k in ("func_id", "name", "core_type")})
    return kernels


def functions_from_ir(path):
    module = ast.parse(path.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef))
    return module, {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}


class ExplicitWorker(ast.NodeTransformer):
    """只替换隐式逻辑核编号。原数值运算、访存、事件和流水均不改写。"""

    REPLACEMENTS = {
        "pl.tile.get_block_idx": "original_block_idx",
        "pl.tile.get_block_num": "original_block_num",
        "pl.tile.get_subblock_idx": "original_subblock_idx",
        "pl.tile.get_subblock_num": "original_subblock_num",
    }

    def __init__(self):
        self.used = collections.Counter()

    def visit_Call(self, node):
        name = ast.unparse(node.func)
        if name in self.REPLACEMENTS:
            if node.args or node.keywords:
                raise ValueError(f"不支持带参数的逻辑核编号：{ast.unparse(node)}")
            target = self.REPLACEMENTS[name]
            self.used[target] += 1
            return ast.copy_location(ast.Name(id=target, ctx=ast.Load()), node)
        return self.generic_visit(node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before_path = next((args.compiled / "passes_dump").glob("*after_OutlineIncoreScopes.py"))
    split_path = next((args.compiled / "passes_dump").glob("*after_ExpandMixedKernel.py"))
    _, logical = functions_from_ir(before_path)
    split_module, split = functions_from_ir(split_path)
    kernels = kernels_from_config(args.compiled / "kernel_config.py")
    missing = {k["name"] for k in kernels} - split.keys()
    if missing:
        raise ValueError(f"生成 kernel 缺少对应计算体：{sorted(missing)}")
    body_nodes, catalog = [], []
    for kernel in kernels:
        original = split[kernel["name"]]
        function = copy.deepcopy(original)
        replacement = ExplicitWorker()
        function = replacement.visit(function)
        function.decorator_list = [ast.parse("pl.function(type=pl.FunctionType.Inline, auto_scope=False)",
                                            mode="eval").body]
        for name in replacement.used:
            function.args.args.append(ast.arg(arg=name, annotation=ast.parse("pl.Scalar[pl.INDEX]",
                                                                             mode="eval").body))
        # 独立解析时，注解所引用的标量必须先声明。记录排列以便调用处按名绑定，
        # 原任务 ABI 的顺序仍保留在 parameters 中，不凭位置混用两个签名。
        function.args.args.sort(key=lambda p: "pl.Scalar[" not in ast.unparse(p.annotation))
        body_nodes.append(function)
        calls = [n for n in ast.walk(original) if isinstance(n, ast.Call)]
        # 首先使用完全同名；MIX 的两个物理核函数回指拆分前的共同逻辑任务。
        parent = kernel["name"]
        if parent not in logical and parent.endswith(("_aic", "_aiv")):
            parent = parent[:-4]
        if parent not in logical:
            raise ValueError(f"找不到逻辑计算体：{kernel['name']}")
        system_calls = [dict(operation=ast.unparse(n.func), arguments=[ast.unparse(v) for v in n.args],
                             keywords={k.arg: ast.unparse(k.value) for k in n.keywords}, source_line=n.lineno)
                        for n in calls if ast.unparse(n.func).startswith("pl.system.")
                        or ast.unparse(n.func) in ("pl.tile.tpush_to_aiv", "pl.tile.tpop_from_aic")]
        catalog.append(dict(kernel, logical_task=parent, original_source_line=original.lineno,
                            original_end_line=original.end_lineno,
                            parameters=[{"name": p.arg, "annotation": ast.unparse(p.annotation)}
                                        for p in original.args.args],
                            return_annotation=ast.unparse(original.returns) if original.returns else None,
                            implicit_worker_replacements=dict(replacement.used),
                            inline_parameter_order=[p.arg for p in function.args.args],
                            system_operations=system_calls,
                            pipelines=[ast.unparse(n) for n in calls if ast.unparse(n.func) == "pl.pipeline"],
                            function_attributes=[ast.unparse(n) for n in calls
                                                 if ast.unparse(n.func) == "pl.func_attr"]))

    declarations = [copy.deepcopy(n) for n in split_module.body if not isinstance(n, ast.ClassDef)]
    declared = {n.id for node in declarations for n in ast.walk(node)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    annotation_symbols = set()
    for function in body_nodes:
        annotations = [p.annotation for p in function.args.args if p.annotation is not None]
        if function.returns is not None:
            annotations.append(function.returns)
        annotation_symbols.update(n.id for a in annotations for n in ast.walk(a)
                                  if isinstance(n, ast.Name))
    implicit_dimensions = sorted(annotation_symbols - declared - {"pl", "tuple", "list", "int", "float", "bool"})
    # 原打印 IR 的注解可引用 orchestration 局部维度。独立计算体文件没有该作用域，
    # 显式声明同名维度占位；后续绑定器必须用原实参形状/标量恢复它们的关系。
    for name in implicit_dimensions:
        declarations.append(ast.parse(f'{name} = pl.dynamic("{name}")').body[0])
    bundle = ast.Module(body=declarations + [ast.ClassDef(name="CSAExtractedCoreBodies", bases=[], keywords=[],
                         body=body_nodes, decorator_list=[ast.parse("pl.program", mode="eval").body])],
                        type_ignores=[])
    ast.fix_missing_locations(bundle)
    args.output.mkdir(parents=True, exist_ok=True)
    generated = args.output / "core_bodies_ir.py"
    generated.write_text(
        "# Copyright (c) PyPTO Contributors.\n"
        "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
        "# 从完整 CSA 编译 IR 提取；供 pl.parse 读取，不是可独立运行的 CSA。\n"
        "# 保留原计算体和事件，仅把隐式逻辑核编号变为参数。\n"
        + ast.unparse(bundle) + "\n")
    generated_functions = functions_from_ir(generated)[1]
    for item, transformed in zip(catalog, body_nodes):
        # 写出后再解析校验，覆盖全部核体，避免代码导出遗漏或顺序截断。
        actual = generated_functions[item["name"]]
        if ast.dump(actual, include_attributes=False) != ast.dump(transformed, include_attributes=False):
            raise ValueError(f"导出的函数不完整：{item['name']}")
        item["generated_line"] = actual.lineno
    result = {"source_ir": str(split_path), "logical_source_ir": str(before_path),
              "body_bundle": str(generated), "physical_kernel_bodies": len(catalog),
              "logical_compute_bodies": len({x["logical_task"] for x in catalog}),
              "all_generated_kernel_bodies_covered": True,
              "implicit_annotation_dimensions": implicit_dimensions,
              "kernel_count_by_role": dict(collections.Counter(x["core_type"] for x in catalog)),
              "status": "EXTRACTED_NOT_YET_PARSED_OR_FUSED", "kernels": catalog}
    (args.output / "core_body_catalog.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "kernels"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
