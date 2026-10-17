# SPDX-License-Identifier: Apache-2.0
"""归因实验：数学阶段在同一 PyPTO IR 上独立规划、PTOAS 编译，再作 device 调用。

运行时仍只派发一个 SPMD。保持阶段完整的等待、计算、发布；代际由外层传入。
本实验只帮助定位全函数 PTOAS 自动同步，不作为生产入口或 700 us 达标实现。
"""

import contextlib
import json
import re
from pathlib import Path


def build_helpers(program, layout, selected, reuse, allocate, output, compact_params=False):
    from pypto import ir, passes
    from pypto.pypto_core import codegen
    from pypto.backend.pto_backend import _compile_pto_module

    identities = {}

    def identity(node):
        # name_hint 是显示名字，不是 SSA 身份；TensorToTile 可产生多个 t__tile。
        bucket = identities.setdefault(node.name_hint, [])
        for index, existing in enumerate(bucket):
            if existing.same_as(node):
                return node.name_hint, index
        bucket.append(node)
        return node.name_hint, len(bucket) - 1

    class Names(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.names = set()

        def add(self, node):
            self.names.add(identity(node))
            if isinstance(node.type, ir.TileType) and node.type.memref is not None:
                self.names.add(identity(node.type.memref.base_))

        def visit_var(self, node):
            self.add(node)
            super().visit_var(node)

        def visit_iter_arg(self, node):
            self.add(node)
            super().visit_iter_arg(node)

    def uses(node):
        visitor = Names()
        if isinstance(node, ir.Stmt):
            visitor.visit_stmt(node)
        else:
            visitor.visit_expr(node)
        return visitor.names

    helpers = {}
    for role in ("aic", "aiv"):
        fn = program.get_function("csa_static_" + role)
        statements = list(fn.body.stmts)
        stages = [node for node in statements if isinstance(node, ir.IfStmt)
                  and any(name[0].startswith("mapped_block_") for name in uses(node.condition))]
        if len(stages) != len(layout["batches"][role]):
            raise ValueError("阶段数量不匹配")
        definitions = {}
        for node in statements:
            if isinstance(node, ir.AssignStmt):
                name = identity(node.var)
                if name in definitions:
                    raise ValueError("顶层 SSA 定义不唯一：" + str(name))
                definitions[name] = node
        generation = [node.var for name, node in definitions.items() if name[0].startswith("generation_base")]
        if len(generation) != 1:
            raise ValueError("代际变量不唯一")
        generation = generation[0]
        generation_key = identity(generation)
        for index, stage in enumerate(stages):
            batch = layout["batches"][role][index]
            if batch["kernel"] not in selected:
                continue
            required = uses(stage)
            while True:
                previous = set(required)
                for name in tuple(required):
                    if name in definitions and name != generation_key:
                        required.update(uses(definitions[name]))
                if previous == required:
                    break
            retained = [node for node in statements if isinstance(node, ir.AssignStmt)
                        and identity(node.var) in required and identity(node.var) != generation_key]
            # 必须只通过外层传入代际，不能再读取 progress 重建/发布下一代。
            if any(name[0].startswith("prior_progress") for name in required):
                raise ValueError("阶段还依赖原入口代际读取")
            helper_name = f"csa_phase_{role}_{index}"
            body = ir.SeqStmts([*retained, stage, ir.ReturnStmt([], fn.span)], fn.body.span)
            indices = [i for i, param in enumerate(fn.params)
                       if not compact_params or identity(param) in required]
            parameters = [(fn.params[i], fn.param_directions[i]) for i in indices]
            helper = ir.Function(helper_name, [*parameters, generation], [],
                                 body, fn.span, fn.func_type, fn.level, fn.role, fn.attrs,
                                 fn.requires_runtime_binding, fn.ir_stage)
            one = ir.Program([helper], program.name, program.span)
            destination = output / helper_name
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "before.py").write_text(one.as_python("pl"))
            ctx = passes.PassContext.current()
            with passes.PassContext(
                [], ctx.get_verification_level(), ctx.get_diagnostic_phase(),
                ctx.get_disabled_diagnostics(), ctx.get_memory_planner(),
                ctx.get_enable_pypto_l0c_double_buffer(), ctx.get_runtime(), ctx.get_enable_buffer_ir(),
            ):
                planned = allocate()(reuse()(one))
            (destination / "allocated.py").write_text(planned.as_python("pl"))
            codes = codegen.PTOCodegen().generate(planned)
            pto = codes if isinstance(codes, str) else "\n".join(codes.values())
            cpp = _compile_pto_module(pto, helper_name, str(destination))
            match = re.search(r"^AICORE void " + helper_name + r"\(([^\n]+)\) \{.*?\n}", cpp, re.M | re.S)
            if not match:
                raise ValueError("找不到阶段函数定义")
            definition = match.group().replace("AICORE void ", "static AICORE __attribute__((always_inline)) inline void ", 1)
            (destination / "stage.cpp").write_text(cpp)
            helpers[role, index] = {"name": helper_name, "signature": match.group(1),
                                    "definition": definition, "tensor_count": len(fn.params),
                                    "parameter_indices": indices,
                                    "kernel": batch["kernel"]}
    if {value["kernel"] for value in helpers.values()} != set(selected):
        raise ValueError("选定阶段未全部编译")
    return helpers


def combine(source, helpers):
    functions = list(re.finditer(r"^AICORE void csa_static_(aic|aiv)\(([^\n]+)\) \{", source, re.M))
    audit = []
    for function in reversed(functions):
        role = function.group(1)
        end = source.find("\n}", function.end())
        body = source[function.end():end]
        declaration = function.group(2).split(",")
        arguments = [item.strip().split()[-1].lstrip("*") for item in declaration]
        phases = list(re.finditer(r"^  if \([^\n]+ >= [^\n]+\) \{\n", body, re.M))
        first_notify = re.search(r"pto::comm::TNOTIFY\([^,]+, ([^,]+), [^)]+\);", body[:phases[0].start()])
        if first_notify is None:
            raise ValueError("外层代际发布未找到")
        generation = first_notify.group(1).strip()
        for index in reversed(range(len(phases))):
            helper = helpers.get((role, index))
            if helper is None:
                continue
            count = helper["tensor_count"]
            indices = helper["parameter_indices"]
            expected = len(indices) + len(arguments) - count + 1
            if len(helper["signature"].split(",")) != expected:
                raise ValueError("阶段 helper 物理参数个数改变")
            # helper 与外层保留同一组 GM 参数身份和顺序，只新增代际 Scalar。
            types = lambda x: re.sub(r"\b\w+$", "", x.strip()).strip()
            for a, b in zip([declaration[i] for i in indices], helper["signature"].split(",")[:len(indices)]):
                if types(a) != types(b):
                    raise ValueError("阶段 helper GM 参数类型不一致")
            actual = [*[arguments[i] for i in indices], generation, *arguments[count:]]
            start = phases[index].end()
            stop = body.find("\n  }", start)
            body = body[:start] + f"    {helper['name']}({', '.join(actual)});\n" + body[stop:]
            audit.append({"role": role, "stage": index, "kernel": helper["kernel"], "arguments": actual})
        if all((role, index) in helpers for index in range(len(phases))):
            # 全阶段均独立编译时，外层只保留 Scalar 分发及代际初始化。
            # 外层原数学的循环事件不能带入 helper 的独立事件生命周期。
            # 只移除严格配对、位于外层顶级的 auto-sync 初始化/回收；
            # helper 自身的流水、MIX 跨核事件及全部数据屏障保持原样。
            flags = list(re.finditer(r"^  (set_flag|wait_flag)\(([^\n]+)\);\n", body, re.M))
            set_flags = sorted(match.group(2) for match in flags if match.group(1) == "set_flag")
            wait_flags = sorted(match.group(2) for match in flags if match.group(1) == "wait_flag")
            if set_flags != wait_flags:
                raise ValueError("外层 auto-sync 生命周期未闭合，不能移除")
            for match in reversed(flags):
                body = body[:match.start()] + body[match.end():]
            if re.search(r"\b(?:set_flag|wait_flag)\(", body):
                raise ValueError("外层仍有未识别的流水事件，拒绝混用事件编号")
            audit.append({"role": role, "outer_math_fully_outlined": True,
                          "removed_outer_auto_sync_pairs": set_flags,
                          "helper_events_unchanged": True})
        source = source[:function.end()] + body + source[end:]
    start = source.index("AICORE void csa_static_")
    source = source[:start] + "\n".join(v["definition"] for _, v in sorted(helpers.items())) + "\n" + source[start:]
    return source, audit


@contextlib.contextmanager
def independent_stage_codegen(layout, output, selected=("hc_widen_rms",), compact_params=False,
                              noinline=False):
    from pypto import passes
    from pypto.backend import pto_backend

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reuse, allocate = passes.memory_reuse, passes.allocate_memory_addr
    compiler = pto_backend._compile_pto_module
    helpers = {}
    if noinline and selected != ("*",):
        raise ValueError("noinline 归因只允许全部阶段已分离、外层事件生命周期已清理的情况")
    if selected == ("*",):
        selected = tuple(sorted({batch["kernel"] for batches in layout["batches"].values()
                                 for batch in batches}))

    def capture(program):
        if program.get_function("csa_static_aic") is not None:
            helpers.update(build_helpers(program, layout, selected, reuse, allocate, output, compact_params))
            if noinline:
                for helper in helpers.values():
                    helper["definition"] = helper["definition"].replace(
                        "__attribute__((always_inline)) inline", "__attribute__((noinline))")
        ctx = passes.PassContext.current()
        with passes.PassContext(
            [], ctx.get_verification_level(), ctx.get_diagnostic_phase(),
            ctx.get_disabled_diagnostics(), ctx.get_memory_planner(),
            ctx.get_enable_pypto_l0c_double_buffer(), ctx.get_runtime(), ctx.get_enable_buffer_ir(),
        ):
            return reuse()(program)

    def compiled(pto, name, output_dir, *args, **kwargs):
        source = compiler(pto, name, output_dir, *args, **kwargs)
        if name == "csa_static_group":
            transformed, audit = combine(source, helpers)
            (output / "original.cpp").write_text(source)
            (output / "combined.cpp").write_text(transformed)
            (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
            return transformed
        return source

    passes.memory_reuse = lambda: passes.create_program_pass(capture, "CSAIsolatedStageCodegen")
    pto_backend._compile_pto_module = compiled
    try:
        yield
    finally:
        passes.memory_reuse = reuse
        pto_backend._compile_pto_module = compiler
