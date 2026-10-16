# SPDX-License-Identifier: Apache-2.0
"""只在 CPU 执行 CSA 的形状/视图编排，记录全部计算调用；绝不执行核数学。"""

import argparse
import ast
import collections
import contextlib
import copy
import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

from build_ir_catalog import functions_from_ir


DTYPES = {"torch.float32": "FP32", "torch.bfloat16": "BF16", "torch.float16": "FP16",
          "torch.int8": "INT8", "torch.int32": "INT32", "torch.int64": "INT64"}
DTYPE_BYTES = {"FP32": 4, "BF16": 2, "FP16": 2, "INT8": 1, "INT32": 4, "INT64": 8}


def strides(shape):
    return [math.prod(shape[i + 1:]) for i in range(len(shape))]


@dataclasses.dataclass
class TensorRef:
    allocation: str
    shape: list
    dtype: str
    strides: list
    offset: int = 0
    layout: str = "ND"
    metadata: object = None
    label: str = ""
    manual_dep: bool = False


@dataclasses.dataclass(frozen=True)
class TaskRef:
    ordinal: int


def serializable(value):
    if isinstance(value, TensorRef):
        result = dataclasses.asdict(value)
        result.pop("metadata")
        return result
    if isinstance(value, TaskRef):
        return {"task_ordinal": value.ordinal}
    if isinstance(value, (list, tuple)):
        return [serializable(x) for x in value]
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    return value


class PlainOrchestration(ast.NodeTransformer):
    """把 SSA 的 loop-carried 记法还原为普通 Python 的局部变量更新。"""

    def visit_FunctionDef(self, node):
        node.decorator_list = []
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is None:
            return ast.Pass()
        value = self.visit(node.value)
        if isinstance(node.target, ast.Name):
            value = ast.Call(func=ast.Name(id="_label", ctx=ast.Load()), args=[
                value, ast.Constant(node.target.id)], keywords=[])
        return ast.copy_location(ast.Assign(targets=[node.target], value=value), node)

    def visit_Call(self, node):
        name = ast.unparse(node.func)
        expressions = [ast.Constant(ast.unparse(arg)) for arg in node.args]
        if name.startswith("self."):
            return ast.copy_location(ast.Call(func=ast.Name(id="_record_call", ctx=ast.Load()), args=[
                ast.Constant(node.func.attr), ast.List(elts=expressions, ctx=ast.Load()),
                ast.Constant(node.lineno), ast.Tuple(elts=[self.visit(a) for a in node.args], ctx=ast.Load())],
                keywords=[]), node)
        if name == "pl.submit":
            if not isinstance(node.args[0], ast.Attribute) or not ast.unparse(node.args[0]).startswith("self."):
                raise ValueError("只支持编译 IR 中的直接 pl.submit")
            return ast.copy_location(ast.Call(func=ast.Name(id="_record_submit", ctx=ast.Load()), args=[
                ast.Constant(node.args[0].attr), ast.List(elts=expressions[1:], ctx=ast.Load()),
                ast.Constant(node.lineno), ast.Tuple(elts=[self.visit(a) for a in node.args[1:]], ctx=ast.Load()),
                ast.Dict(keys=[ast.Constant(k.arg) for k in node.keywords],
                         values=[self.visit(k.value) for k in node.keywords])], keywords=[]), node)
        return self.generic_visit(node)

    def visit_For(self, node):
        init = next((k.value for k in node.iter.keywords if k.arg == "init_values"), None)
        if init is None:
            return self.generic_visit(node)
        if not isinstance(node.target, ast.Tuple) or len(node.target.elts) != 2:
            raise ValueError("不支持的 SSA 循环 target")
        index, carried = node.target.elts
        last = node.body[-1]
        target = last.target if isinstance(last, ast.AnnAssign) else last.targets[0]
        if not isinstance(last.value, ast.Call) or ast.unparse(last.value.func) != "pl.yield_":
            raise ValueError("SSA 循环缺少末尾 yield")
        initialize_result = copy.deepcopy(init) if isinstance(target, ast.Tuple) else copy.deepcopy(init.elts[0])
        start = [ast.Assign(targets=[copy.deepcopy(carried)], value=copy.deepcopy(init)),
                 ast.Assign(targets=[copy.deepcopy(target)], value=initialize_result)]
        next_values = copy.deepcopy(target)
        for n in ast.walk(next_values):
            if hasattr(n, "ctx"):
                n.ctx = ast.Load()
        if not isinstance(next_values, ast.Tuple):
            next_values = ast.Tuple(elts=[next_values], ctx=ast.Load())
        node.target = index
        node.iter.keywords = [k for k in node.iter.keywords if k.arg != "init_values"]
        node.body.append(ast.Assign(targets=[copy.deepcopy(carried)], value=next_values))
        return [self.visit(n) for n in start] + [self.generic_visit(node)]

    def visit_If(self, node):
        expression = ast.unparse(node.test)
        node = self.generic_visit(node)
        node.test = ast.Call(func=ast.Name(id="_record_condition", ctx=ast.Load()),
                             args=[node.test, ast.Constant(expression), ast.Constant(node.lineno)],
                             keywords=[])
        return node


class Recorder:
    def __init__(self, functions):
        self.functions = functions
        self.tasks = []
        self.allocations = {}
        self.current = None
        self.scope = []
        self.next_scope = 0
        self.scopes = []
        self.conditions = []
        self.return_aliases = {name: self.resolve_returns(fn) for name, fn in functions.items()
                               if name != "_decode_csa_tp1_layer"}
        self.pl = SimpleNamespace(
            tensor=SimpleNamespace(create=self.create, reshape=self.reshape, slice=self.slice,
                                   dim=lambda x, i: x.shape[i], read=self.read),
            array=SimpleNamespace(create=lambda size, **kw: [None] * size,
                                  update_element=self.array_update, get_element=lambda a, i: a[i]),
            system=SimpleNamespace(task_dummy=self.dummy, task_invalid=lambda: TaskRef(-1)),
            min=min, max=max, cast=lambda x, dtype: int(x), const=lambda x, dtype: x,
            range=range, parallel=range, yield_=lambda *v: v[0] if len(v) == 1 else v,
            scope=self.runtime_scope, spmd=self.spmd, TensorLayout=SimpleNamespace(ND="ND", NZ="NZ"),
            ScopeMode=SimpleNamespace(AUTO="auto", MANUAL="manual"),
            **{name: name for name in [*DTYPE_BYTES, "INDEX", "TASK_ID"]})

    @staticmethod
    def resolve_returns(function):
        """核返回只传播 GM Tensor 引用；追踪 SSA 别名，不求值 Tensor 数值。"""
        params = {p.arg for p in function.args.args}
        aliases = collections.defaultdict(set)
        returns = [n for n in function.body if isinstance(n, ast.Return)]
        if not returns or returns[-1].value is None:
            return []

        def bind(target, value):
            if isinstance(target, ast.Tuple):
                if isinstance(value, ast.Tuple):
                    values = value.elts
                elif isinstance(value, ast.Call) and ast.unparse(value.func) == "pl.yield_":
                    values = value.args
                else:
                    return
                for t, v in zip(target.elts, values):
                    bind(t, v)
            elif isinstance(target, ast.Name):
                if isinstance(value, ast.Name):
                    aliases[target.id].add(value.id)
                elif isinstance(value, ast.Call):
                    op = ast.unparse(value.func)
                    if op in ("pl.yield_", "pl.tensor.assemble") and len(value.args) >= 1:
                        bind(target, value.args[0])
                    elif op == "pl.tile.store":
                        bind(target, value.args[2])

        for node in ast.walk(function):
            if isinstance(node, ast.AnnAssign) and node.value is not None:
                bind(node.target, node.value)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    bind(target, node.value)
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
                initial = next((k.value for k in node.iter.keywords if k.arg == "init_values"), None)
                if initial is not None:
                    bind(node.target.elts[1], initial)

        def resolve(name, seen):
            if name in params:
                return {name}
            if name in seen:
                return set()
            return set().union(*(resolve(v, seen | {name}) for v in aliases[name]))

        value = returns[-1].value
        values = value.elts if isinstance(value, ast.Tuple) else [value]
        result = []
        for v in values:
            sources = resolve(v.id, set()) if isinstance(v, ast.Name) else set()
            if len(sources) != 1:
                raise ValueError(f"不能唯一恢复 {function.name} 的返回别名：{ast.unparse(v)} / {sources}")
            result.append(next(iter(sources)))
        return result

    def create(self, shape, dtype, layout="ND", manual_dep=False):
        ref = TensorRef(f"temporary_{len(self.allocations)}", list(shape), dtype, strides(shape),
                        layout=layout, manual_dep=manual_dep)
        self.allocations[ref.allocation] = ref
        return ref

    @staticmethod
    def label(value, name):
        if isinstance(value, TensorRef) and not value.label:
            value.label = name
        return value

    @staticmethod
    def reshape(value, shape):
        if math.prod(value.shape) != math.prod(shape):
            raise ValueError(f"reshape 元素数量不一致：{value.shape} -> {shape}")
        if value.strides != strides(value.shape):
            raise ValueError("暂不允许对非连续 GM 视图猜测 reshape 的 stride")
        return dataclasses.replace(value, shape=list(shape), strides=strides(shape), label="")

    @staticmethod
    def slice(value, shape, offsets, **kwargs):
        if len(shape) != len(value.shape) or len(offsets) != len(shape):
            raise ValueError("slice 维数不一致")
        return dataclasses.replace(value, shape=list(shape),
                                   offset=value.offset + sum(o * s for o, s in zip(offsets, value.strides)),
                                   label="")

    @staticmethod
    def read(value, index):
        if value.metadata is None:
            raise ValueError(f"编排读取需要真实 metadata：{value.label}")
        result = value.metadata
        for i in index:
            result = result[i]
        return result

    @staticmethod
    def array_update(array, index, value):
        result = list(array)
        result[index] = value
        return result

    def condition(self, value, expression, source_line):
        self.conditions.append({"expression": expression, "value": bool(value),
                                "source_line": source_line})
        return value

    @contextlib.contextmanager
    def runtime_scope(self, **options):
        self.next_scope += 1
        self.scope.append(self.next_scope)
        self.scopes.append({"id": self.next_scope, "options": options, "parent_chain": list(self.scope[:-1])})
        try:
            yield
        finally:
            self.scope.pop()

    def new_task(self, blocks, options):
        task = {"ordinal": len(self.tasks), "block_num": blocks,
                "options": serializable(options), "scope": list(self.scope), "logical_function": None}
        self.tasks.append(task)
        return task

    @contextlib.contextmanager
    def spmd(self, blocks, **options):
        if self.current is not None:
            raise ValueError("不支持嵌套提交")
        self.current = self.new_task(blocks, options)
        try:
            yield TaskRef(self.current["ordinal"])
        finally:
            self.current = None

    def dummy(self, **options):
        task = self.new_task(0, options)
        task["kind"] = "dependency_only"
        return TaskRef(task["ordinal"])

    def call(self, name, expressions, line, values):
        if self.current is None:
            raise ValueError(f"计算函数缺少提交范围：{name}")
        function = self.functions[name]
        params = function.args.args
        if len(params) != len(values):
            raise ValueError(f"{name} 参数数量不一致")
        binding = dict(zip((p.arg for p in params), values))
        self.current.update(logical_function=name, source_line=line, kind="compute",
                            arguments=[{"parameter": p.arg, "annotation": ast.unparse(p.annotation),
                                        "expression": expr, "value": serializable(v)}
                                       for p, expr, v in zip(params, expressions, values)])
        outputs = [binding[name] for name in self.return_aliases[name]]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def submit(self, name, expressions, line, values, options):
        with self.spmd(1, **options) as task:
            result = self.call(name, expressions, line, values)
        outputs = list(result) if isinstance(result, tuple) else [result]
        return tuple(outputs + [task])


def tensor_annotation(annotation):
    if isinstance(annotation, ast.Subscript) and ast.unparse(annotation.value) in ("pl.Out", "pl.InOut"):
        annotation = annotation.slice
    if not isinstance(annotation, ast.Subscript) or ast.unparse(annotation.value) != "pl.Tensor":
        return None
    return annotation.slice.elts


def bind(ir_path, descriptors):
    _, functions = functions_from_ir(ir_path)
    root = functions["_decode_csa_tp1_layer"]
    recorder = Recorder(functions)
    env = {"pl": recorder.pl, "_label": recorder.label,
           "_record_call": recorder.call, "_record_submit": recorder.submit,
           "_record_condition": recorder.condition}
    inputs = []
    external = {}
    for arg in root.args.args:
        if arg.arg == "self":
            inputs.append(None)
            continue
        original_name = arg.arg.removesuffix("__ssa_v0")
        descriptor = descriptors[original_name]
        annotation = tensor_annotation(arg.annotation)
        if annotation is None:
            inputs.append(descriptor["scalar"])
            continue
        shape = descriptor["shape"]
        for declared, extent in zip(annotation[0].elts, shape):
            if isinstance(declared, ast.Name):
                if declared.id in env and env[declared.id] != extent:
                    raise ValueError(f"动态维冲突：{declared.id}")
                env[declared.id] = extent
            elif ast.literal_eval(declared) != extent:
                raise ValueError(f"根参数固定维度不一致：{original_name}")
        layout = "NZ" if len(annotation) > 2 and "NZ" in ast.unparse(annotation[2]) else "ND"
        ref = TensorRef("argument:" + original_name, shape, DTYPES[descriptor["dtype"]],
                        descriptor["stride"], layout=layout,
                        metadata=descriptor.get("metadata_values"), label=original_name)
        inputs.append(ref)
        external[original_name] = serializable(ref)
    executable = ast.Module(body=[PlainOrchestration().visit(copy.deepcopy(root))], type_ignores=[])
    ast.fix_missing_locations(executable)
    exec(compile(executable, str(ir_path), "exec"), env)
    env[root.name](*inputs)
    allocations = [{**serializable(ref), "bytes": math.prod(ref.shape) * DTYPE_BYTES[ref.dtype]}
                   for ref in recorder.allocations.values()]
    return {"source_ir": str(ir_path), "scope": "只执行形状和视图编排；未执行任何核数学或 NPU 运算",
            "compute_calls": sum(t["kind"] == "compute" for t in recorder.tasks),
            "dependency_only_calls": sum(t["kind"] == "dependency_only" for t in recorder.tasks),
            "temporary_bytes_without_reuse": sum(a["bytes"] for a in allocations),
            "root_arguments": external, "allocations": allocations, "scopes": recorder.scopes,
            "tasks": recorder.tasks, "selected_control_path": recorder.conditions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--root-descriptors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = bind(args.ir, json.loads(args.root_descriptors.read_text()))
    result["descriptor_source"] = str(args.root_descriptors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("compute_calls", "dependency_only_calls",
                                            "temporary_bytes_without_reuse")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
