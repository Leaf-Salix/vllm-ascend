# SPDX-License-Identifier: Apache-2.0
"""诊断实际参数搬运开销：Native 参数保持，临时 GM 视图合并为一个 arena。

复用已经验证的同规格 PTOAS 数学代码，通过公开 jit.extern 注册。不是生产
PyPTO GM 视图实现，也不作为 700 μs 目标的最终验收实现。没有新增数据搬运。
仅当生成源码 AST、工作区布局及控制表逐项相同，且真实设备地址匹配时启用。
"""

import ast
import copy
import json
import re
from pathlib import Path

from workspace_base import aliases, check_addresses


def build_compact_entry(case, prepared, reference, output, *, validate_addresses=True):
    case, reference, output = map(Path, (case, reference, output))
    output.mkdir(parents=True, exist_ok=True)
    source = ast.parse((case / "entry_jit.py").read_text())
    expected = ast.parse((reference / "entry_jit.py").read_text())
    if ast.dump(source) != ast.dump(expected):
        raise ValueError("外部代码必须来自完全相同的数学及调度源码")
    layout = json.loads((case / "layout.json").read_text())
    old_layout = json.loads((reference / "layout.json").read_text())
    if layout["workspace"] != old_layout["workspace"]:
        raise ValueError("外部代码工作区及控制表不匹配")
    if not validate_addresses and prepared.workspace.device.type != "meta":
        raise ValueError("只有 CPU meta 编译可以跳过真实地址核对")
    audit = (check_addresses(case, prepared) if validate_addresses
             else {"actual_addresses_checked": False, "meta_compile_only": True})
    offsets = aliases(case)
    progress_offset = layout["workspace"]["sections"]["progress"]["offset_bytes"]
    offsets = {name: value + progress_offset for name, value in offsets.items()}
    offsets["progress"] = progress_offset
    functions = {n.name: n for n in source.body if isinstance(n, ast.FunctionDef)}
    leaf = functions["csa_static_aiv"]
    retained = [arg.arg for arg in leaf.args.args if arg.arg not in offsets]
    roots = [arg.arg for arg in functions["entry"].args.args if arg.arg not in offsets]
    cpp_paths = {}
    for role in ("aic", "aiv"):
        paths = list((reference / "build_output").glob(f"entry_*/kernels/{role}/csa_static_{role}.cpp"))
        if len(paths) != 1:
            raise ValueError("参考物理核源文件不唯一：" + role)
        cpp = paths[0].read_text()
        if cpp.count('extern "C"') != 1:
            raise ValueError("物理核入口数量改变")
        cpp = cpp[:cpp.index('extern "C"')]
        matches = list(re.finditer(r"^static __aicore__ void csa_static_(aic|aiv)\(([^\n]+)\) \{", cpp, re.M))
        if {m.group(1) for m in matches} != {"aic", "aiv"} or len(matches) != 2:
            raise ValueError("预期包装源码内同一 PTOAS 编译单元的两个物理函数")
        selected_decls = None
        for match in reversed(matches):
            member = match.group(1)
            declarations = [x.strip() for x in match.group(2).split(",")]
            fn = functions["csa_static_" + member]
            if len(declarations) != len(fn.args.args) + (2 if member == "aic" else 3):
                raise ValueError("物理核入参数量改变")
            natives, definitions = [], []
            for arg, declaration in zip(fn.args.args, declarations):
                var = declaration.split()[-1].lstrip("*")
                if arg.arg in offsets:
                    typ = declaration.removesuffix(var).strip()
                    definitions.append(f"  {typ} {var} = ({typ})(csa_arena + {offsets[arg.arg]});")
                else:
                    natives.append(declaration)
            compact = natives + ["__gm__ int8_t* csa_arena"] + declarations[len(fn.args.args):]
            if member == role:
                selected_decls = compact
            header = f"static __aicore__ void csa_static_{member}({', '.join(compact)}) {{\n"
            cpp = cpp[:match.start()] + header + "\n".join(definitions) + cpp[match.end():]
        wrapper = ['extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {',
                   "  set_atomic_none();"]
        values = []
        for index, declaration in enumerate(selected_decls[:len(retained) + 1]):
            var = declaration.split()[-1].lstrip("*")
            typ = declaration.removesuffix(var).strip()
            wrapper += [f"  __gm__ Tensor* t{index} = reinterpret_cast<__gm__ Tensor*>(args[{index}]);",
                        f"  {typ} {var} = ({typ})(t{index}->buffer.addr) + t{index}->start_offset;"]
            values.append(var)
        values += ["get_block_idx(args)", "get_block_num(args)"]
        if role == "aiv":
            values.append("get_sub_block_id(args)")
        wrapper += [f"  csa_static_{role}({', '.join(values)});", "}"]
        path = output / f"compact_{role}.cpp"
        path.write_text(cpp + "\n" + "\n".join(wrapper) + "\n")
        cpp_paths[role] = str(path.resolve())

    arena_arg = ast.parse(
        f"def stub(arena: pl.InOut[pl.Tensor[[{layout['workspace']['bytes']}], pl.INT8]]): pass"
    ).body[0].args.args[0]
    extern = copy.deepcopy(functions["csa_static_group"])
    extern.name = "csa_compact"
    extern.args.args = [a for a in extern.args.args if a.arg not in offsets] + [arena_arg]
    extern.decorator_list = [ast.parse(
        f"pl.jit.extern(core_type='mixed', aic_source={cpp_paths['aic']!r}, "
        f"aiv_source={cpp_paths['aiv']!r}, dual_aiv_dispatch=True)", mode="eval").body]
    outputs = [arg for arg in extern.args.args if ast.unparse(arg.annotation).startswith(("pl.InOut[", "pl.Out["))]
    # mixed extern 的 tuple 返回目前在 Group 包装中丢失类型；使用其正式
    # void/in-place 形式，所有写入仍通过 InOut 参数声明，不隐藏写副作用。
    extern.returns = None
    extern.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
    entry = functions["entry"]
    entry.args.args = [a for a in entry.args.args if a.arg not in offsets] + [copy.deepcopy(arena_arg)]
    context = next(n for n in entry.body if isinstance(n, ast.With))
    names = [arg.arg for arg in extern.args.args]
    context.body = ast.parse(f"csa_compact({', '.join(names)})").body
    root_outputs = [arg for arg in outputs if arg.arg in roots or arg.arg == "arena"]
    entry.returns = ast.Subscript(value=ast.Name(id="tuple", ctx=ast.Load()),
        slice=ast.Tuple(elts=[arg.annotation.slice for arg in root_outputs], ctx=ast.Load()), ctx=ast.Load())
    entry.body[-1] = ast.parse("return (" + ", ".join(a.arg for a in root_outputs) + ")").body[0]
    module = ast.Module(body=[ast.Import(names=[ast.alias(name="pypto.language", asname="pl")]), extern, entry],
                        type_ignores=[])
    ast.fix_missing_locations(module)
    (output / "entry_jit.py").write_text(ast.unparse(module) + "\n")
    bound = dict(zip(prepared.param_names, prepared.core_args))
    arguments = tuple(bound[name] for name in roots) + (prepared.workspace,)
    audit.update(original_root_tensors=len(prepared.core_args), root_tensors=len(arguments),
                 original_leaf_tensors=len(leaf.args.args), leaf_tensors=len(retained) + 1,
                 arena_aliases=len(offsets), math_and_schedule_ast_identical=True,
                 source_reference=str(reference), no_data_copy_added=True,
                 eligible_for_700us_target=False)
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return output / "entry_jit.py", arguments, audit
