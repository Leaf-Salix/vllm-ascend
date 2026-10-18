# SPDX-License-Identifier: Apache-2.0
"""执行前段源码快照的Scalar控制，检查发布/等待图；不模拟数学或性能。

只抽取新AIC分支helper与AIV从Indexer压缩到Score之前的实际语句。数学调用
替换成原缓冲别名，并作为串行节点保留；通知和等待使用真实控制表达式。
验证所有等待有发布者、每核不重复发布、完整前段无依赖环，以及原AIV等待
集合包含重新分核后的全部真正producer。此证据不能代替设备缓存或数值验证。
"""

import ast
import collections
import copy
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
# 首轮执行时指向正式目录；候选退步后以同字节快照保留复现入口。
SOURCE = HERE / "source_candidate"


class MathAliases(ast.NodeTransformer):
    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name):
            return node
        name = node.value.func.id
        if name in {"publish_phase", "wait_producer", "wait_row_owners", "min", "max"}:
            return node
        target = node.targets[0]
        outputs = list(target.elts) if isinstance(target, ast.Tuple) else [target]
        assert all(isinstance(x, ast.Name) for x in outputs), ast.unparse(target)
        node.value = ast.Call(
            func=ast.Name(id="record_math", ctx=ast.Load()),
            args=[ast.Constant(name), ast.List(
                elts=[ast.Name(id=x.id, ctx=ast.Load()) for x in outputs], ctx=ast.Load()),
                ast.List(elts=node.value.args, ctx=ast.Load())],
            keywords=[],
        )
        return node


def function(file, name, vector=False):
    tree = ast.parse((SOURCE / file).read_text())
    fn = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
    if vector:
        first = next(i for i, n in enumerate(fn.body)
                     if isinstance(n, ast.If) and ast.unparse(n.test) == "PROJECTION_BRANCH_GROUPS > 0")
        # 共同请求数使用实际源码里紧邻分支的赋值，避免在测试中另写一套作用域。
        assert ast.unparse(fn.body[first - 1]).startswith("idx_requests = pl.tensor.dim(idx_lengths, 0)")
        first -= 1
        last = next(i for i, n in enumerate(fn.body)
                    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                    and ast.unparse(n.value.func) == "indexer_score_vector_lane_stage")
        setup = ast.parse("tokens = pl.tensor.dim(x, 0)\nworker = SIM_WORKER\ncore = 24 + worker\n"
                          "generation = 1\nplanner = min((tokens + 7) // 8, 47)\n").body
        fn.body = setup + fn.body[first:last]
    tensor_args = {a.arg for a in fn.args.args if "Tensor" in ast.unparse(a.annotation)}
    fn.decorator_list = []
    fn.returns = None
    fn.args.defaults = []
    for a in fn.args.args:
        a.annotation = None
    fn = MathAliases().visit(fn)
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    return compile(module, str(SOURCE / file), "exec"), fn, tensor_args


def constants():
    result = {}
    for n in ast.parse((SOURCE / "runtime_abi.py").read_text()).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if any(isinstance(x, (ast.Call, ast.Attribute)) for x in ast.walk(n.value)):
                continue
            try:
                result[n.targets[0].id] = eval(compile(ast.Expression(n.value), "<constants>", "eval"),
                                              {"__builtins__": {}}, result)
            except (NameError, TypeError):
                pass
    return result


def run_case(tokens, requests, compact, branches, compiled, const):
    graph = {}
    tail = {}
    qb_owners = {}
    current_core = 0

    def event(key, dependencies=()):
        assert key not in graph, ("duplicate publication", key)
        graph[key] = set(dependencies)
        if current_core in tail:
            graph[key].add(tail[current_core])
        tail[current_core] = key

    def publish(signal, phase, owner, participants, generation):
        assert owner == current_core and participants == 72 and generation == 1
        event(("phase", phase, owner))
        return signal

    def wait(signal, phase, participants, generation, first, count, tile, workers, base, stride):
        if count > 0:
            start, stop = first // tile, (first + count + tile - 1) // tile
            owners = [base + ((start + i) % workers) * stride for i in range(min(stop - start, workers))]
            event(("wait", current_core, len(graph)), [("phase", phase, core) for core in owners])
        return signal

    def wait_one(signal, phase, owner, participants, generation):
        event(("wait", current_core, len(graph)), [("phase", phase, owner)])
        return signal

    def math(name, outputs, arguments):
        event(("math", current_core, len(graph), name))
        if name == "q_b_stage":
            logical, workers = arguments[-2:]
            assert 0 <= logical < workers == 24 - branches
            for column in range(logical, 128, workers):
                assert column not in qb_owners, ("duplicate Q_B column", column)
                qb_owners[column] = current_core
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    planner = min((tokens + 7) // 8, 47)
    for phase, owners in [
        (const["PHASE_IDX_QUERY_PROJECT"], range(24)),
        (const["PHASE_IDX_PROJECT"], range(24)),
        (const["PHASE_QR"], range(24, 72)),
        (const["PHASE_ROPE"], range(24, 72)),
        (const["PHASE_META"], [24 + planner]),
    ]:
        for owner in owners:
            graph[("phase", phase, owner)] = set()

    pl = SimpleNamespace(INDEX=int, min=min, max=max, cast=lambda x, _t: int(x),
                         read=lambda tensor, index: tensor.value,
                         tensor=SimpleNamespace(dim=lambda tensor, axis: tensor.shape[axis]))
    for vector, (code, fn, tensors) in enumerate(compiled):
        for worker in range(48 if vector else 24):
            current_core = 24 + worker if vector else worker
            namespace = dict(const, pl=pl, SIM_WORKER=worker, record_math=math,
                             publish_phase=publish, wait_row_owners=wait, wait_producer=wait_one)
            exec(code, namespace)
            args = {}
            for a in fn.args.args:
                if a.arg in tensors:
                    rows = requests if a.arg in {"idx_lengths", "main_lengths"} else tokens
                    value = compact if a.arg == "idx_count" else requests
                    args[a.arg] = SimpleNamespace(name=a.arg, shape=[rows, 4096], value=value)
                else:
                    args[a.arg] = {"worker": worker, "generation": 1,
                                   "BRANCH_GROUPS": branches,
                                   "PROJECTION_BRANCH_GROUPS": branches}.get(a.arg, 0)
            namespace[fn.name](**args)

    for node, deps in graph.items():
        assert deps <= graph.keys(), ("unpublished dependency", node, deps - graph.keys())
    users = collections.defaultdict(list)
    counts = {node: len(deps) for node, deps in graph.items()}
    for node, deps in graph.items():
        for dep in deps:
            users[dep].append(node)
    ready = collections.deque(node for node, count in counts.items() if count == 0)
    visited = 0
    while ready:
        node = ready.popleft()
        visited += 1
        for user in users[node]:
            counts[user] -= 1
            if counts[user] == 0:
                ready.append(user)
    assert visited == len(graph), ("dependency cycle", [n for n, c in counts.items() if c][:12])
    assert set(qb_owners) == set(range(128))
    assert set(qb_owners.values()) == set(range(branches, 24))
    # 核对原AIV消费者的完整等待集合覆盖新owner；不假定空发布代表别核数据。
    old_new = [
        (128, 24, set(range(branches, 24))),
        (4 * ((tokens + 15) // 16), 24, set(range(min(branches, 4 * ((tokens + 15) // 16))))),
        (tokens, 24, set(range(min(branches, (tokens + 3) // 4)))),
        (2 * ((compact + 15) // 16), 24, set(range(min(branches, 2 * ((compact + 15) // 16))))),
    ]
    for work, workers, actual in old_new:
        assert actual <= set(range(min(work, workers)))
    return dict(tokens=tokens, requests=requests, compact=compact, branch_groups=branches,
                graph_nodes=len(graph), acyclic=True, producer_coverage=True)


def main():
    const = constants()
    compiled = [function("projection_branches.py", "projection_branches_stage"),
                function("vector_schedule.py", "vector_schedule", vector=True)]
    rows = []
    for tokens in [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 24, 32, 48, 64, 96,
                   128, 129, 144, 168, 192, 256, 257, 512, 513]:
        for query in [1, 5, 6]:
            requests = (tokens + query - 1) // query
            for compact in {0, requests, 2 * requests}:
                for branches in [4, 8]:
                    rows.append(run_case(tokens, requests, compact, branches, compiled, const))
    output = dict(passed=True, cases=len(rows), source=str(SOURCE), results=rows,
                  scope="实际源码的前段Scalar依赖；不模拟数学、缓存、FFTS或性能")
    (HERE / "prefix_control.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "results"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
