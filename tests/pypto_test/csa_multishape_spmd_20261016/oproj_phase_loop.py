# SPDX-License-Identifier: Apache-2.0
"""将相邻八个O-A阶段共享同一核内循环体，保持原记录表和依赖执行顺序。

原来各阶段只有group、源行基址和目标列基址不同。逐AST核对这些差异后，
把它们变为循环局部Scalar。没有独立device函数或编译器事件重写。
"""

import ast
import copy
import re


CONTROL_NAMES = frozenset(("record", "mapped_block", "block", "subslot", "done", "wait_base",
                           "wait_count", "publish", "dependency", "wait_index", "other",
                           "required", "target", "flag"))


def deduplicate(source, layout):
    tree = ast.parse(source)
    header = "".join(source.splitlines(keepends=True)[:tree.body[0].lineno - 1])
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "csa_static_aic")
    phases = [node for node in fn.body if isinstance(node, ast.If)]
    batches = layout["batches"]["aic"]
    if len(phases) != len(batches):
        raise ValueError("原AIC阶段数与记录表不符")
    selected = [i for i, batch in enumerate(batches)
                if {r["kernel_name"] for r in batch["members"]} == {"proj_a_mm_0"}]
    if len(selected) != 8 or selected != list(range(selected[0], selected[0] + 8)):
        raise ValueError("当前候选要求相邻八个原N128 O-A阶段")
    row_stride = None
    groups, bases, bodies, begin_positions = [], [], [], []
    for index in selected:
        phase = phases[index]
        position = fn.body.index(phase)
        begin_positions.append(position - 3)
        prefix = copy.deepcopy(fn.body[position - 3:position])
        if (len(prefix) != 3 or not isinstance(prefix[0], ast.Assign)
                or ast.unparse(prefix[0].targets[0]) != f"record_{index}"):
            raise ValueError("原记录加载位置改变")
        base = prefix[0].value.left
        if not isinstance(base, ast.Constant) or not isinstance(base.value, int):
            raise ValueError("原阶段record基址不是静态整数")
        bases.append(base.value)
        body = copy.deepcopy(phase)
        group_reads = []
        for node in ast.walk(body):
            if (isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.tensor.slice"
                    and ast.unparse(node.args[0]) == "arg_wo_a"):
                g = node.args[2].elts[0]
                if not isinstance(g, ast.Constant) or not isinstance(g.value, int):
                    raise ValueError("O-A权重group不是静态整数")
                group_reads.append(g.value)
        if len(group_reads) != 2 or len(set(group_reads)) != 1:
            raise ValueError("原O-A权重读次数或group发生改变")
        group = group_reads[0]
        groups.append(group)
        row_node = next(node for node in ast.walk(body) if isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Name) and node.target.id == f"stage_{index}_v9")
        if not isinstance(row_node.value, ast.BinOp) or not isinstance(row_node.value.left, ast.Constant):
            raise ValueError("原O-A源行表达式改变")
        if group:
            stride, remainder = divmod(row_node.value.left.value, group)
            if remainder or row_stride not in (None, stride):
                raise ValueError("O-A源行stride不一致")
            row_stride = stride
        if row_stride is None:
            # 源视图按8个group等分，只读取静态Tensor描述符。
            parameter = next(arg for arg in fn.args.args if arg.arg == "temporary_51")
            tensor_type = next(node for node in ast.walk(parameter.annotation)
                               if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "pl.Tensor")
            shape = tensor_type.slice.elts[0].elts
            row_stride = ast.literal_eval(shape[0]) // 8
        if row_node.value.left.value != group * row_stride:
            raise ValueError("O-A源行基址不是group×stride")
        row_node.value.left = ast.BinOp(ast.Name("oa_group", ast.Load()), ast.Mult(), ast.Constant(row_stride))
        target_writes = 0
        for node in ast.walk(body):
            if not isinstance(node, ast.Call):
                continue
            if ast.unparse(node.func) == "pl.tensor.slice" and ast.unparse(node.args[0]) == "arg_wo_a":
                node.args[2].elts[0] = ast.Name("oa_group", ast.Load())
            if ast.unparse(node.func) == "pl.tensor.assemble":
                col = node.args[2].elts[1]
                if not isinstance(col, ast.BinOp) or ast.unparse(col.left) != str(group * 1024):
                    raise ValueError("O-A目标列偏移发生改变")
                col.left = ast.BinOp(ast.Name("oa_group", ast.Load()), ast.Mult(), ast.Constant(1024))
                target_writes += 1
        if target_writes != 1:
            raise ValueError("O-A输出写点不是唯一的")

        class Rename(ast.NodeTransformer):
            def visit_Name(self, node):
                name = node.id
                if name.startswith(f"stage_{index}_"):
                    name = name.replace(f"stage_{index}_", "stage_oa_", 1)
                else:
                    match = re.fullmatch(r"(.+)_" + str(index), name)
                    if match and match.group(1) in CONTROL_NAMES:
                        name = match.group(1) + "_oa"
                return ast.copy_location(ast.Name(name, node.ctx), node)

        prefix[0].value.left = ast.parse("oa_record_base", mode="eval").body
        canonical = Rename().visit(ast.Module(body=prefix + [body], type_ignores=[]))
        bodies.append(canonical)
    if sorted(groups) != list(range(8)):
        raise ValueError("O-A group没有覆盖0..7各一次")
    if any(ast.dump(body) != ast.dump(bodies[0]) for body in bodies[1:]):
        raise ValueError("八个O-A阶段还存在group/base以外差异，不能合并")
    if any(b - a != 432 for a, b in zip(bases, bases[1:])):
        raise ValueError("阶段record stride不是72核×6字段")
    if any(b - a != 4 for a, b in zip(begin_positions, begin_positions[1:])):
        raise ValueError("O-A阶段之间存在其他工作，不能移动")
    prefix = ast.parse(f"oa_record_base = {bases[0]} + oa_phase * 432\n"
                       f"oa_group = pl.const({groups[-1]}, pl.INDEX)\n").body
    for index, group in enumerate(groups[:-1]):
        prefix += ast.parse(f"if oa_phase == {index}:\n    oa_group = pl.const({group}, pl.INDEX)\n").body
    loop = ast.parse("for oa_phase in pl.range(8):\n    pass\n").body[0]
    loop.body = prefix + bodies[0].body
    start, stop = begin_positions[0], begin_positions[-1] + 4
    fn.body[start:stop] = [loop]
    ast.fix_missing_locations(tree)
    transformed = header + ast.unparse(tree) + "\n"
    audit = {"phases": selected, "group_order": groups, "record_bases": bases,
             "row_stride": row_stride, "all_eight_canonical_bodies_equal": True,
             "execution_order_preserved": True, "record_and_wait_tables_unchanged": True,
             "mathematical_tiles_and_k_order_unchanged": True,
             "source_bytes_before": len(source.encode()), "source_bytes_after": len(transformed.encode()),
             "device_validated": False}
    return transformed, audit
