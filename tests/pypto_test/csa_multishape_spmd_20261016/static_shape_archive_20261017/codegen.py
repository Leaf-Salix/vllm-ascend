# SPDX-License-Identifier: Apache-2.0
"""从已绑定核体和逐核依赖表生成完整 CSA 单 SPMD。

形状决定本次 JIT 的描述符和分块；实际位置、长度、页表等仍从 device
Tensor 读取。保留原 NZ 逻辑重 lowering、有效视图修正、循环 carry、
分支合流及 Group 返回别名回绑，不加入实验调优开关或测试目录依赖。
"""

import ast
import copy
import math
from pathlib import Path

from .binding import tensor_annotation
from .ir import ExplicitWorker, functions_from_ir


def physical_shape(ref):
    if ref["layout"] != "NZ":
        return ref["shape"]
    c0 = 32 if ref["dtype"] == "INT8" else 16
    m, n = ref["shape"][-2:]
    return [math.prod(ref["shape"][:-2]), n // c0, m // 16, 16, c0]


def annotation(ref, direction=None):
    extra = ", pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)" if ref["layout"] == "NZ" else ""
    result = f"pl.Tensor[{ref['shape']!r}, pl.{ref['dtype']}{extra}]"
    return f"pl.{direction}[{result}]" if direction else result


class Constants(ast.NodeTransformer):
    def __init__(self, values):
        self.values = values

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.values:
            return ast.copy_location(ast.Constant(self.values[node.id]), node)
        return node


class RestoreBranchYields(ast.NodeTransformer):
    """只去掉旧 if 的显式 phi；保留原循环 carry 与 pipeline 结构。"""

    def __init__(self):
        self.parallel_assignments_flattened = 0

    def visit_If(self, node):
        node = self.generic_visit(node)
        for branch in (node.body, node.orelse):
            if not branch:
                continue
            last = branch[-1]
            if (
                isinstance(last, (ast.Assign, ast.AnnAssign))
                and isinstance(last.value, ast.Call)
                and ast.unparse(last.value.func) == "pl.yield_"
            ):
                values = last.value.args
                target = last.target if isinstance(last, ast.AnnAssign) else last.targets[0]
                if len(values) > 1 and isinstance(target, ast.Tuple):
                    written = {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
                    read = {n.id for v in values for n in ast.walk(v) if isinstance(n, ast.Name)}
                    if written & read or len(target.elts) != len(values):
                        raise ValueError("分支合流赋值不能安全按项展开，需保留并行语义")
                    # 原输出 phi 名与 RHS 不相交，因此顺序赋值等价于并行
                    # 赋值；避免创建 Buffer IR 尚不接受的 Tuple[Tile,...]。
                    branch[-1:] = [
                        ast.Assign(targets=[copy.deepcopy(t)], value=copy.deepcopy(v))
                        for t, v in zip(target.elts, values)
                    ]
                    self.parallel_assignments_flattened += 1
                else:
                    last.value = values[0]
        return node


class NormalizeViewValidShape(ast.NodeTransformer):
    """对原 IR 的 reshape/slice 视图，用 slice 的 valid_shape 表达有效范围。

    PTOAS planner 不为 view 分配独立 tile 描述符，不能对 view 发出
    set_validshape。零偏移、相同物理 shape 的 slice 保留原数据及别名，
    仅把相同 valid extent 放在 view 自身；不改变数学或 pipeline。
    """

    def __init__(self, function):
        self.views = set()
        self.rewrites = 0
        for node in ast.walk(function):
            if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
                if ast.unparse(node.value.func) in ("pl.tile.reshape", "pl.tile.slice", "pl.tile.extract"):
                    self.views.add(node.target.id)

    def visit_AnnAssign(self, node):
        call = node.value
        if (
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == "pl.tile.set_validshape"
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id in self.views
        ):
            shape = copy.deepcopy(node.annotation.slice.elts[0])
            if not isinstance(shape, ast.List):
                raise ValueError("view 有效范围转换要求明确的物理 Tile shape")
            node.value = ast.Call(
                func=ast.parse("pl.tile.slice", mode="eval").body,
                args=[call.args[0], shape, ast.List(elts=[ast.Constant(0) for _ in shape.elts], ctx=ast.Load())],
                keywords=[ast.keyword(arg="valid_shape", value=ast.List(elts=call.args[1:], ctx=ast.Load()))],
            )
            self.rewrites += 1
        return self.generic_visit(node)


def dimension_value(expr, known):
    expression = Constants(known).visit(copy.deepcopy(expr))
    for node in ast.walk(expression):
        if isinstance(node, ast.Name) and node.id != "pl":
            raise KeyError(node.id)

    # 输入只来自本地编译器打印的维度表达式，不接受外部任意代码。
    class Dims:
        INDEX = None
        const = staticmethod(lambda value, dtype: value)

    return eval(compile(ast.Expression(expression), "<dimension>", "eval"), {"pl": Dims, "__builtins__": {}})


def bind_dimension(expr, extent, known):
    if isinstance(expr, ast.Name):
        if expr.id in known and known[expr.id] != extent:
            raise ValueError(f"维度绑定矛盾：{expr.id}: {known[expr.id]} != {extent}")
        known[expr.id] = extent
        return
    try:
        value = dimension_value(expr, known)
    except KeyError:
        if not isinstance(expr, ast.BinOp) or not isinstance(expr.op, ast.Mult):
            raise ValueError(f"不能唯一恢复维度：{ast.unparse(expr)} = {extent}")
        for variable, factor in ((expr.left, expr.right), (expr.right, expr.left)):
            try:
                value = dimension_value(factor, known)
            except KeyError:
                continue
            if value and extent % value == 0:
                bind_dimension(variable, extent // value, known)
                return
        raise ValueError(f"不能整除恢复维度：{ast.unparse(expr)} = {extent}")
    if value != extent:
        raise ValueError(f"固定维度不一致：{ast.unparse(expr)} = {value} != {extent}")


def specialize(original, call, member):
    function = copy.deepcopy(original)
    values = {a["parameter"]: a["value"] for a in call["arguments"]}
    known = {name: value for name, value in values.items() if not isinstance(value, dict)}
    known["original_block_num"] = member["logical_block_num"]
    known["original_subblock_num"] = 2
    for parameter in function.args.args:
        tensor = tensor_annotation(parameter.annotation)
        if tensor is None:
            continue
        ref = values[parameter.arg]
        shape = physical_shape(ref) if ref["layout"] == "NZ" and len(tensor[0].elts) == 5 else ref["shape"]
        if len(tensor[0].elts) != len(shape):
            raise ValueError(f"{function.name}/{parameter.arg} Tensor rank 不一致")
        for dim, extent in zip(tensor[0].elts, shape):
            bind_dimension(dim, extent, known)
    function = Constants(known).visit(function)
    function.args.args = [p for p in function.args.args if p.arg not in known]
    function.name = f"call_{call['ordinal']}_{member['core_kind']}_body"
    # 导出前后计算体相等（唯一变换为已核对的维度/标量特化和 worker 参数化）。
    return function, known


def generate(schedule, layout, catalog):
    """按当前形状生成 program 源码和真实叶子 ABI 契约。

    固定采用外部 GM 描述符绑定，并在 orchestration 派生 Native 的部分
    重叠子视图。等价于原生成器 external + native_subviews_in_entry=True。
    不包含 arena 复现路径，也不在核内把 GM 别名伪装成本地 Tile。
    """
    original_module, originals = functions_from_ir(Path(catalog["body_bundle"]))
    pre_nz_path = Path(catalog["logical_source_ir"])
    logical_module, pre_nz = functions_from_ir(pre_nz_path)
    helper_functions = {}
    specialization = {}
    for seq in schedule["per_core_sequences"].values():
        for member in seq:
            key = (member["task"], member["kernel_name"])
            if key in helper_functions:
                continue
            call = schedule["call_bindings"][member["task"]]
            original = originals[member["kernel_name"]]
            source_ir = catalog["source_ir"]
            if any(isinstance(a["value"], dict) and a["value"].get("layout") == "NZ" for a in call["arguments"]):
                # 重新进入编译前端时必须使用原逻辑 NZ load。已阻塞化的 rank-5
                # load 在早期 pass 中会被重新推断为 rank-5 Tile，而其后面的
                # L0 extract 需要 rank-2 Tile。对纯 AIC 任务使用同次
                # 原编译 Outline 后的逻辑核体，让原 TensorToTile/Flatten/
                # BlockNz/AutoTileMatmulL0 passes 再次完成同样的 lowering。
                original = copy.deepcopy(pre_nz[member["kernel_name"]])
                worker = ExplicitWorker()
                original = worker.visit(original)
                for name in worker.used:
                    original.args.args.append(
                        ast.arg(arg=name, annotation=ast.parse("pl.Scalar[pl.INDEX]", mode="eval").body)
                    )
                source_ir = str(pre_nz_path)
            function, known = specialize(original, call, member)
            helper_functions[key] = function
            specialization[function.name] = {
                "kernel": member["kernel_name"],
                "source_ir": source_ir,
                "constants": known,
                "parameter_order": [p.arg for p in function.args.args],
            }
    roots = schedule["root_arguments"]
    names = {"argument:" + name: "arg_" + name for name in roots}
    names.update({name: name for name in layout["workspace"]["temporaries"]})
    _, logical_functions = functions_from_ir(Path(catalog["logical_source_ir"]))
    original_root = logical_functions["_decode_csa_tp1_layer"]
    directions = {}
    for param in original_root.args.args:
        if isinstance(param.annotation, ast.Subscript) and ast.unparse(param.annotation.value) in (
            "pl.Out",
            "pl.InOut",
        ):
            directions[param.arg.removesuffix("__ssa_v0")] = ast.unparse(param.annotation.value).split(".")[-1]

    def inline_body(function, arguments, prefix):
        function = copy.deepcopy(function)
        valid_views = NormalizeViewValidShape(function)
        function = valid_views.visit(function)
        specialization[function.name]["view_validshape_rewrites"] = valid_views.rewrites
        if specialization[function.name]["kernel"] == "qk_pv_aiv":
            branch_yields = RestoreBranchYields()
            function = branch_yields.visit(function)
            specialization[function.name]["parallel_assignments_flattened"] = (
                branch_yields.parallel_assignments_flattened
            )
        replacements = {p.arg: value for p, value in zip(function.args.args, arguments)}
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id not in replacements:
                # 打印 IR 的 __ssa/__tile 是编译器保留后缀。重新进入前端时
                # 使用普通局部名，让新的 SSA/自动命名 passes 自行分配后缀。
                replacements[node.id] = prefix + "v" + str(len(replacements))

        class Rename(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in replacements:
                    return ast.copy_location(ast.Name(id=replacements[node.id], ctx=node.ctx), node)
                return node

        result = []
        for statement in function.body:
            if isinstance(statement, ast.Return):
                continue  # 原返回只传播 GM 别名；数学计算/存储均已在前面的语句中。
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and ast.unparse(statement.value.func) == "pl.func_attr"
            ):
                continue  # 融合函数统一持有 mx_tensor_views_blocked/split_aiv 属性。
            if any(isinstance(node, ast.Return) for node in ast.walk(statement)):
                raise ValueError("不允许忽略嵌套早退")
            result.append(Rename().visit(copy.deepcopy(statement)))
        return result

    # 外部只绑定 GM allocation 的真实视图，NZ 权重保持原始布局。
    # 实际 leaf ABI 必须完整计数，不能用较少的根参数数目替代。
    seen_views, leaf_refs = {}, {}
    for call in schedule["call_bindings"].values():
        for arg in call["arguments"]:
            ref = arg["value"]
            if not isinstance(ref, dict) or "allocation" not in ref:
                continue
            key = (ref["allocation"], tuple(ref["shape"]), tuple(ref["strides"]), ref["offset"])
            if key in seen_views:
                continue
            root_ref = (
                roots[ref["allocation"][9:]]
                if ref["allocation"].startswith("argument:")
                else layout["workspace"]["temporaries"][ref["allocation"]]
            )
            base = names[ref["allocation"]]
            if ref["strides"] != [math.prod(ref["shape"][i + 1 :]) for i in range(len(ref["shape"]))]:
                raise ValueError("当前生成器不猜测非连续 GM 视图")
            if ref["layout"] == "NZ":
                if ref["shape"] != root_ref["shape"] or ref["offset"]:
                    raise ValueError("NZ 只保留原完整 GM 视图")
                view = base
            elif ref["shape"] == root_ref["shape"] and ref["offset"] == 0:
                view = base
            else:
                view = f"view_{len(seen_views)}"
                if ref["offset"] + math.prod(ref["shape"]) > math.prod(root_ref["shape"]):
                    raise ValueError("视图超出原 allocation")
            seen_views[key] = view
            mutable = not ref["allocation"].startswith("argument:") or ref["allocation"][9:] in directions
            leaf_refs[view] = (ref, mutable)
    for name, section in layout["workspace"]["sections"].items():
        leaf_refs[name] = ({**section, "layout": "ND"}, name == "progress")
    leaf_signature = ", ".join(
        f"{name}: {annotation(ref, 'InOut' if mutable else None)}" for name, (ref, mutable) in leaf_refs.items()
    )
    leaf_arguments = ", ".join(leaf_refs)
    leaf_outputs = ", ".join(name for name, (_, mutable) in leaf_refs.items() if mutable)
    leaf_return_type = "tuple[" + ", ".join(annotation(ref) for ref, mutable in leaf_refs.values() if mutable) + "]"
    derived_native_views = []
    # 原 Native 参数以及工作区的具体 GM 视图由调用方一次性绑定。
    # 不在 orchestration 使用尚不支持的 dtype reinterpret，也不在
    # InCore 把 GM 视图伪装成本地 Tile；四份 NZ 权重保持原对象。
    view_definitions = []
    # Native Indexer 的原存储与两个部分重叠子视图不能同时作为可写
    # Torch 入参。沿用动态根图的 reshape/slice，在 orchestration
    # 内只构造描述符；leaf 仍接收完整参数，核体完全不变。
    for name, (ref, mutable) in leaf_refs.items():
        allocation = ref.get("allocation", "")
        if not allocation.startswith("argument:") or not mutable:
            continue
        base = names[allocation]
        if name == base or base not in leaf_refs:
            continue
        base_ref, _ = leaf_refs[base]
        count, base_count = math.prod(ref["shape"]), math.prod(base_ref["shape"])
        if ref["offset"] == 0 and count == base_count:
            continue  # 完整 byte-span 别名已被正式入口支持。
        if ref["layout"] != "ND" or ref["dtype"] != base_ref["dtype"]:
            raise ValueError("不推断 NZ 或跨 dtype 的 Native 子视图")
        if ref["offset"] < 0 or ref["offset"] + count > base_count:
            raise ValueError("Native 子视图超出原存储")
        view_definitions.extend(
            [
                f"{name}_flat = pl.reshape({base}, [{base_count}])",
                f"{name}_span = pl.slice({name}_flat, [{count}], [{ref['offset']}])",
                f"{name} = pl.reshape({name}_span, {ref['shape']!r})",
            ]
        )
        derived_native_views.append({"name": name, "base": base, "reference": ref})
    derived_names = {item["name"] for item in derived_native_views}
    inputs = [
        (name, annotation(ref, "InOut" if mutable else None))
        for name, (ref, mutable) in leaf_refs.items()
        if name not in derived_names
    ]
    signature = ", ".join(f"{name}: {kind}" for name, kind in inputs)
    outputs = [name for name, (_, mutable) in leaf_refs.items() if mutable and name not in derived_names]
    return_names = ", ".join(outputs)
    return_type = "tuple[" + ", ".join(annotation(leaf_refs[name][0]) for name in outputs) + "]"
    root_names = {name for name, _ in inputs}
    specialization["_entry_contract"] = {
        "root_tensor_count": len(inputs),
        "leaf_tensor_count": len(leaf_refs),
        "native_root_tensor_count": len(roots),
        "workspace_binding": "external",
        "ordered_inputs": [
            {"name": name, "reference": ref, "mutable": mutable}
            for name, (ref, mutable) in leaf_refs.items()
            if name in root_names
        ],
        "derived_native_views": derived_native_views,
        "gm_views_bound_outside_incore": True,
        "current_core_limit": None,
        "runtime_supported": None,
        "runtime_validation": "待 compiler 按调用方明确提供的 runtime 容量核对",
    }

    kernels = []
    for role, role_batches in layout["batches"].items():
        lines = [
            "@pl.function(type=pl.FunctionType.InCore)",
            f"def csa_static_{role}(self, {leaf_signature}) -> {leaf_return_type}:",
            "    pl.func_attr({'mx_tensor_views_blocked': True" + (", 'split_aiv': True})" if role == "aiv" else "})"),
            "    group_id = pl.tile.get_block_idx()",
        ]
        if role == "aiv":
            lines.extend(["    lane_id = pl.tile.get_subblock_idx()", "    core_id = 24 + group_id * 2 + lane_id"])
        else:
            lines.append("    core_id = group_id")
        stride = layout["protocol"]["generation_stride"]
        padding = layout["protocol"]["progress_words_per_core"]
        lines.extend(
            [
                f"    own_flag = core_id * {padding}",
                "    pl.system.cacheinvalid(progress, [1], [own_flag])",
                "    prior_progress = pl.tensor.read(progress, [own_flag])",
                f"    generation_base = (prior_progress // {stride} + 1) * {stride}",
                "    pl.tensor.write(progress, [own_flag], generation_base)",
                "    pl.system.cacheinvalid(progress, [1], [own_flag])",
                "    pl.system.fence()",
            ]
        )
        insertions = {}
        for index, batch in enumerate(role_batches):
            member = batch["members"][0]
            call = schedule["call_bindings"][batch["task"]]
            fn = helper_functions[(batch["task"], batch["kernel"])]
            bound = {a["parameter"]: a["value"] for a in call["arguments"]}
            arguments = []
            for p in fn.args.args:
                if p.arg == "original_block_idx":
                    arguments.append(f"block_{index}")
                elif p.arg == "original_subblock_idx":
                    arguments.append(f"subslot_{index}")
                else:
                    ref = bound[p.arg]
                    key = (ref["allocation"], tuple(ref["shape"]), tuple(ref["strides"]), ref["offset"])
                    arguments.append(seen_views[key])
            insertions[index] = (fn, arguments)
            record = batch["record_offset_words"]
            width = layout["protocol"]["record_width"]
            lines.extend(
                [
                    f"    # 原调用 {call['ordinal']} / {batch['kernel']}；各核只执行自己的原 block。",
                    f"    record_{index} = {record} + core_id * {width}",
                    f"    mapped_block_{index} = pl.cast(pl.tensor.read(records, [record_{index}]), pl.INDEX)",
                    # 有效记录本来就非负；max 不改变其值，只把该事实显式提供给
                    # NZ offset 的符号证明（当前证明器不使用外层 if 的范围条件）。
                    f"    block_{index} = pl.max(mapped_block_{index}, 0)",
                    f"    if mapped_block_{index} >= 0:",
                    f"        subslot_{index} = pl.cast(pl.tensor.read(records, [record_{index} + 1]), pl.INDEX)",
                    f"        done_{index} = pl.tensor.read(records, [record_{index} + 2])",
                    f"        wait_base_{index} = pl.cast(pl.tensor.read(records, [record_{index} + 3]), pl.INDEX)",
                    f"        wait_count_{index} = pl.cast(pl.tensor.read(records, [record_{index} + 4]), pl.INDEX)",
                    f"        publish_{index} = pl.tensor.read(records, [record_{index} + 5])",
                    f"        for dependency_{index} in pl.range(wait_count_{index}):",
                    f"            wait_index_{index} = (wait_base_{index} + dependency_{index}) * 2",
                    f"            other_{index} = pl.cast(pl.tensor.read(waits, [wait_index_{index}]), pl.INDEX)",
                    f"            required_{index} = pl.tensor.read(waits, [wait_index_{index} + 1])",
                    f"            target_{index} = generation_base + required_{index}",
                    f"            flag_{index} = other_{index} * {padding}",
                    f"            pl.system.cacheinvalid(progress, [1], [flag_{index}])",
                    f"            initial_{index} = pl.tensor.read(progress, [flag_{index}])",
                    f"            for (ready_{index},) in pl.while_(init_values=(initial_{index},)):",
                    f"                pl.cond(ready_{index} < target_{index})",
                    f"                pl.system.cacheinvalid(progress, [1], [flag_{index}])",
                    f"                after_{index} = pl.yield_(pl.tensor.read(progress, [flag_{index}]))",
                    # 不增加循环后伪使用来躲避 #3009。
                    "        pl.system.fence()",
                    "        pl.system.cacheinvalid()",
                    f"        __INSERT_ORIGINAL_BODY_{index}__()",
                    # 保留原 kernel 完成边界。fence 是本核流水/GM 屏障，不是全卡集合。
                    "        pl.system.fence()",
                    "        pl.system.cacheinvalid()",
                    "        pl.system.fence()",
                    f"        if publish_{index} != 0:",
                    f"            pl.tensor.write(progress, [own_flag], generation_base + done_{index})",
                    "            pl.system.cacheinvalid(progress, [1], [own_flag])",
                    "            pl.system.fence()",
                ]
            )
        lines.append(f"    return ({leaf_outputs})")
        kernel = ast.parse("\n".join(lines)).body[0]

        class InsertBodies(ast.NodeTransformer):
            def visit_Expr(self, node):
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id.startswith("__INSERT_ORIGINAL_BODY_")
                ):
                    number = int(node.value.func.id.removeprefix("__INSERT_ORIGINAL_BODY_").removesuffix("__"))
                    body, bound_arguments = insertions[number]  # noqa: B023 本轮立即展开。
                    return inline_body(body, bound_arguments, f"stage_{number}_")
                return self.generic_visit(node)

        # 在源码生成阶段粘合真实核体，避免把多个带 Out/InOut 的独立调用
        # 留给解析器；每个 batch 的局部 SSA 名称隔离，跨 batch 只共享 GM。
        kernels.append(InsertBodies().visit(kernel))

    aic_returns = ", ".join("aic_return_" + name for name, (_, mutable) in leaf_refs.items() if mutable)
    aiv_returns = ", ".join("aiv_return_" + name for name, (_, mutable) in leaf_refs.items() if mutable)
    group = ast.parse(f"""
@pl.function(type=pl.FunctionType.Group)
def csa_static_group(self, {leaf_signature}) -> {leaf_return_type}:
    pl.func_attr({{'mx_tensor_views_blocked': True}})
    ({aic_returns}) = self.csa_static_aic({leaf_arguments})
    ({aiv_returns}) = self.csa_static_aiv({leaf_arguments})
    return ({leaf_outputs})

@pl.function(type=pl.FunctionType.Orchestration)
def entry(self, {signature}) -> {return_type}:
    pl.func_attr({{'mx_tensor_views_blocked': True}})
{chr(10).join("    " + line for line in view_definitions)}
    with pl.spmd(24, sync_start=True, name_hint='csa_static_full_card'):
        ({leaf_outputs}) = self.csa_static_group({leaf_arguments})
    return ({return_names})
""").body
    cls = ast.ClassDef(
        name="CSAStaticPrepared",
        bases=[],
        keywords=[],
        body=kernels + group,
        decorator_list=[ast.parse("pl.program", mode="eval").body],
    )
    declarations = {}
    for module_source in (original_module, logical_module):
        for node in module_source.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                declarations[node.targets[0].id] = copy.deepcopy(node)
    module = ast.Module(
        body=[ast.parse("import pypto.language as pl").body[0], *declarations.values(), cls], type_ignores=[]
    )
    ast.fix_missing_locations(module)
    source = (
        "# 从已核对的 CSA 核体与当前形状逐核依赖表生成；生成本身不代表设备验证。\n"
        f"# 根入口 {len(inputs)} Tensor，核共享 ABI {len(leaf_refs)} Tensor；需由调用方验证 runtime 参数容量。\n"
        "# AIC/AIV 是同一个 MIX 任务的物理两半；不是两次 kernel 提交。\n" + ast.unparse(module) + "\n"
    )
    ast.parse(source)
    return source, specialization


def jit_source(source):
    """转换成同源 kernel-mode JIT 入口，保持单个 MIX 提交与 GM 返回别名。"""
    module = ast.parse(source)
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef))
    functions = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    group = functions["csa_static_group"]

    class UnbindSelf(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
            return self.generic_visit(node)

    class JitTensorLayout(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.unparse(node.func) == "pl.TensorView":
                layout = next((k.value for k in node.keywords if k.arg == "layout"), None)
                stride = next((k.value for k in node.keywords if k.arg == "stride"), None)
                if (
                    layout is not None
                    and ast.unparse(layout) == "pl.TensorLayout.NZ"
                    and isinstance(stride, ast.List)
                    and not stride.elts
                ):
                    # JIT 签名接受 plain NZ layout，不接受 TensorView 包装。
                    # 原 TensorView 的 stride 为空；这只是同布局的前端记法。
                    return ast.copy_location(ast.parse("pl.NZ", mode="eval").body, node)
            return self.generic_visit(node)

    class InferLocalTensorTypes(ast.NodeTransformer):
        def visit_AnnAssign(self, node):
            if tensor_annotation(node.annotation) is not None:
                # 旧打印 IR 的局部 Tensor annotation 可带未被 JIT 捕获的
                # dynamic 占位名。这里由相同 tensor op 和已绑定实参推断
                # 结果类型，避免把未参与形状绑定的名字带入 JIT 专门化。
                return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)
            return self.generic_visit(node)

    for name in ("csa_static_aic", "csa_static_aiv", "csa_static_group", "entry"):
        function = functions[name]
        function.args.args = [p for p in function.args.args if p.arg != "self"]
        for parameter in function.args.args:
            parameter.annotation = JitTensorLayout().visit(parameter.annotation)
        if function.returns is not None:
            function.returns = JitTensorLayout().visit(function.returns)
        decorator = "pl.jit" if name == "entry" else "pl.jit.inline" if name == "csa_static_group" else "pl.jit.incore"
        function.decorator_list = [ast.parse(decorator, mode="eval").body]
        if name == "csa_static_group":
            # SPMD 在解析时只看到一个 wrapper 调用；InlineFunctions 展开后，
            # OutlineClusterScopes 将两个物理核组合为一个 Group，保持一次 MIX 提交。
            # JIT 前端先检查 InOut 的使用顺序；把每次返回绑定回同名参数，
            # 使下一调用读取 post-call 别名。它们仍是同一 GM 存储，没有数据搬运。
            members = [copy.deepcopy(n) for n in group.body if isinstance(n, ast.Assign)]
            outputs = next(n for n in function.body if isinstance(n, ast.Return)).value
            for member in members:
                target = copy.deepcopy(outputs)
                for node in ast.walk(target):
                    if isinstance(node, (ast.Tuple, ast.Name)):
                        node.ctx = ast.Store()
                member.targets = [target]
            prefix = [n for n in function.body if isinstance(n, ast.Expr)]
            returns = [n for n in function.body if isinstance(n, ast.Return)]
            function.body = (
                prefix
                + [
                    ast.With(
                        items=[
                            ast.withitem(
                                context_expr=ast.parse("pl.cluster(name_hint='csa_static_group')", mode="eval").body
                            )
                        ],
                        body=members,
                    )
                ]
                + returns
            )
        functions[name] = InferLocalTensorTypes().visit(UnbindSelf().visit(function))
    module.body = [n for n in module.body if not isinstance(n, ast.ClassDef)] + [
        functions[n] for n in ("csa_static_aic", "csa_static_aiv", "csa_static_group", "entry")
    ]
    ast.fix_missing_locations(module)
    return "# 同源 kernel-mode 入口；设备执行前须通过编译、ABI 与运行时验证。\n" + ast.unparse(module) + "\n"
