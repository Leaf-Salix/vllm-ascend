# SPDX-License-Identifier: Apache-2.0
"""实验性阶段内存规划：在已排空流水的阶段之间复用片上内存。

直接操作 IR 节点，保留 SSA 与 MemRef 身份，不经文本打印/重解析。
每个阶段仍由原 PyPTO MemoryReuse/AllocateMemoryAddr 规划。
只在当前实验进程替换 pass factory，不改共享依赖仓库。
"""

import contextlib
import json
from pathlib import Path


def phase_plan(program, reuse, allocate, output):
    from pypto import ir, passes

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "before.py").write_text(program.as_python("pl"))

    def allocation(node):
        return (isinstance(node, ir.AssignStmt) and isinstance(node.value, ir.Call)
                and node.value.op.name == "tile.alloc")

    class Names(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.vars, self.onchip = set(), set()

        def visit_var(self, node):
            self.vars.add(node.name_hint)
            if isinstance(node.type, ir.TileType) and node.type.memref is not None:
                self.onchip.add(node.type.memref.base_.name_hint)
            super().visit_var(node)

        def visit_iter_arg(self, node):
            self.vars.add(node.name_hint)
            if isinstance(node.type, ir.TileType) and node.type.memref is not None:
                self.onchip.add(node.type.memref.base_.name_hint)
            super().visit_iter_arg(node)

    def stage(node):
        if not isinstance(node, ir.IfStmt):
            return False
        names = Names()
        names.visit_expr(node.condition)
        return any(name.startswith("mapped_block_") for name in names.vars)

    def replace_body(fn, body):
        return ir.Function(
            fn.name, list(zip(fn.params, fn.param_directions)), fn.return_types,
            ir.SeqStmts(body, fn.body.span), fn.span, fn.func_type, fn.level,
            fn.role, fn.attrs, fn.requires_runtime_binding, fn.ir_stage,
        )

    result_functions, audit = [], []
    for fn in program.functions.values():
        if fn.name not in {"csa_static_aic", "csa_static_aiv"}:
            result_functions.append(fn)
            continue
        role = fn.name.removeprefix("csa_static_")
        body = list(fn.body.stmts)
        stages = [(index, node) for index, node in enumerate(body) if stage(node)]
        alloc_names = {node.var.name_hint for node in body if allocation(node)}
        stage_ids = {index for index, _ in stages}
        live, ownership = {}, {}
        for index, node in stages:
            names = Names()
            names.visit_stmt(node)
            live[index] = names.vars | names.onchip
            for name in live[index] & alloc_names:
                if name in ownership and ownership[name] != index:
                    raise ValueError("阶段间共享片上 allocation，不能独立规划：" + name)
                ownership[name] = index
        for index, node in enumerate(body):
            if index in stage_ids or allocation(node):
                continue
            names = Names()
            names.visit_stmt(node)
            if names.onchip:
                raise ValueError("存在阶段外片上值，必须先分析其生命周期")
        allocations, replacements = {}, {}
        for ordinal, (index, node) in enumerate(stages):
            selected = [
                item for pos, item in enumerate(body)
                if (pos not in stage_ids or pos == index)
                and (not allocation(item) or item.var.name_hint in live[index])
            ]
            isolated = ir.Program([replace_body(fn, selected)], program.name, program.span)
            # 内层 pass 不消耗外层 IRDump 的 pass_index，其他设置完整继承。
            ctx = passes.PassContext.current()
            with passes.PassContext(
                [], ctx.get_verification_level(), ctx.get_diagnostic_phase(),
                ctx.get_disabled_diagnostics(), ctx.get_memory_planner(),
                ctx.get_enable_pypto_l0c_double_buffer(), ctx.get_runtime(),
                ctx.get_enable_buffer_ir(),
            ):
                planned = allocate()(reuse()(isolated))
            planned_fn = planned.get_function(fn.name)
            planned_stages = [item for item in planned_fn.body.stmts if stage(item)]
            if len(planned_stages) != 1:
                raise ValueError("规划后的阶段数改变")
            if not ir.structural_equal(planned_stages[0].condition, node.condition):
                raise ValueError("独立内存规划改变了阶段条件")
            replacements[index] = planned_stages[0]
            current = []
            for item in planned_fn.body.stmts:
                if allocation(item):
                    name = item.var.name_hint
                    if name in allocations:
                        raise ValueError("不同阶段产生了重名 allocation：" + name)
                    allocations[name] = item
                    current.append(name)
            audit.append({"role": role, "stage": ordinal, "allocations": current,
                          "cross_stage_onchip_values": False})
        merged = [*allocations.values(), *[
            replacements.get(index, node) for index, node in enumerate(body) if not allocation(node)
        ]]
        result_functions.append(replace_body(fn, merged))
    if not audit:
        raise ValueError("阶段规划只支持当前 CSA 融合入口")
    result = ir.Program(result_functions, program.name, program.span)
    (output / "after.py").write_text(result.as_python("pl"))
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return result


@contextlib.contextmanager
def scoped_memory_planning(output):
    from pypto import passes

    reuse, allocate = passes.memory_reuse, passes.allocate_memory_addr
    transformed = False

    def apply(program):
        nonlocal transformed
        if program.get_function("csa_static_aic") is None:
            return reuse()(program)
        result = phase_plan(program, reuse, allocate, output)
        transformed = True
        return result

    def keep_planned(program):
        return program if transformed else allocate()(program)

    passes.memory_reuse = lambda: passes.create_program_pass(apply, "CSAIndependentPhaseMemory")
    passes.allocate_memory_addr = lambda: passes.create_program_pass(keep_planned, "CSAKeepPhaseAddresses")
    try:
        yield
    finally:
        passes.memory_reuse, passes.allocate_memory_addr = reuse, allocate
