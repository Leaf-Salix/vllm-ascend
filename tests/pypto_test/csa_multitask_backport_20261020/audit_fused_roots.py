# SPDX-License-Identifier: Apache-2.0
"""执行候选实际Scalar控制，独立按Score写入条带核对每条query的根等待集合。"""

import argparse
import ast
import copy
import json
from types import SimpleNamespace

from prepare import HERE


def scalar_function(tree, name):
    function = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
    function.decorator_list = []
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None
    return function


def audit(source):
    tree = ast.parse((source / "decode_indexer.py").read_text())
    config = ast.parse((source / "config.py").read_text())
    definitions = {
        target.id: n.value
        for module in (config, tree)
        for n in module.body
        if isinstance(n, ast.Assign)
        for target in n.targets
        if isinstance(target, ast.Name)
    }
    maximum = ast.literal_eval(
        next(k.value for k in definitions["FLASH"].keywords if k.arg == "max_position_embeddings")
    )

    class Constants(dict):
        def __missing__(self, key):
            expression = definitions[key]
            if any(isinstance(n, ast.Call) for n in ast.walk(expression)):
                raise ValueError(f"Scalar审计不执行配置调用：{key}")
            value = eval(compile(ast.Expression(expression), "<source-constant>", "eval"), {"__builtins__": {}}, self)
            self[key] = value
            return value

    constants = Constants(M=SimpleNamespace(max_position_embeddings=maximum))
    names = (
        "S",
        "TOPK_SCORE_WORKERS",
        "TOPK_QUERY_WORKERS",
        "TOPK_MAX_CANDIDATES",
        "TOPK_CANDIDATES_PER_LEAF",
        "TOPK_MAX_LEAVES",
        "BUFFERED_LONG_SCORE_TILE",
        "COMPRESS_RATIO",
        "SCORE_SIGNAL_SLOT_WORDS",
        "SCORE_SIGNAL_WORDS",
    )
    namespace = {name: constants[name] for name in names}
    events, received = [], []
    pl = SimpleNamespace(
        min=min,
        max=max,
        range=range,
        cast=lambda value, dtype: value,
        INT32="i32",
        INDEX="index",
        read=lambda tensor, offsets: tensor[offsets[0]],
        tensor=SimpleNamespace(dim=lambda tensor, axis: len(tensor)),
        system=SimpleNamespace(
            fence=lambda: None,
            cacheinvalid=lambda: None,
            bar_all=lambda: None,
            signal_notify=lambda flag, offsets, value: events.append(("notify", offsets[0], value)),
            signal_wait=lambda flag, offsets, value: events.append(("wait", offsets[0], value)),
        ),
    )
    namespace["pl"] = pl
    namespace["indexer_topk_query_merge_one"] = lambda query, *args: (
        received.append((query, list(events))),
        events.clear(),
    )
    selected = [scalar_function(tree, name) for name in ("indexer_long_leaf_plan", "indexer_fused_topk_finish")]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source / "decode_indexer.py"), "exec"), namespace)
    rows = []
    group_size, workers = constants["S"], constants["TOPK_SCORE_WORKERS"]
    for batch in (4, 5, 8, 16, 24, 28):
        for pattern in ("uniform", "mixed_dummy", "capacity_edge"):
            lengths = [131078] * batch
            if pattern == "mixed_dummy":
                lengths = [([0, 32774, 65542, 131078][i % 4]) for i in range(batch)]
            elif pattern == "capacity_edge":
                lengths = [maximum] * batch
            positions = [
                max(length - group_size, 131072 if length == 0 else 0) + q
                for length in lengths
                for q in range(group_size)
            ]
            cache_max = min(max(lengths) // constants["COMPRESS_RATIO"], constants["TOPK_MAX_CANDIDATES"])
            leaves, tiles, extra = namespace["indexer_long_leaf_plan"](cache_max, batch)
            writers = {}
            # 从Score真实条带分工枚举写入者，不复用候选等待公式。
            for worker in range(workers):
                for item in range(worker, batch * leaves, workers):
                    if batch < workers:
                        group, leaf = item % batch, item // batch
                    else:
                        group, leaf = item // leaves, item % leaves
                    for lane in range(2):
                        for q in range(group_size // 2):
                            query = group * group_size + lane * (group_size // 2) + q
                            writers[query, leaf] = worker * 2 + lane
            checked, wait_count = set(), 0
            for worker in range(constants["TOPK_QUERY_WORKERS"]):
                events.clear()
                received.clear()
                namespace["indexer_fused_topk_finish"](None, positions, lengths, None, None, None, worker)
                notifications = [e for _, sequence in received for e in sequence if e[0] == "notify"]
                notifications += [e for e in events if e[0] == "notify"]
                assert notifications == [("notify", worker * constants["SCORE_SIGNAL_SLOT_WORDS"], 1)]
                for query, sequence in received:
                    assert query not in checked
                    checked.add(query)
                    visible = min(
                        lengths[query // group_size] // constants["COMPRESS_RATIO"],
                        (positions[query] + 1) // constants["COMPRESS_RATIO"],
                        constants["TOPK_MAX_CANDIDATES"],
                    )
                    # 按每叶的实际起点判断哪些根会被消费，独立于候选的ceil公式。
                    roots = [
                        leaf
                        for leaf in range(leaves)
                        if (leaf * tiles + min(leaf, extra)) * constants["BUFFERED_LONG_SCORE_TILE"] < visible
                    ]
                    expected = [writers[query, leaf] for leaf in roots]
                    actual = [e[1] // constants["SCORE_SIGNAL_SLOT_WORDS"] for e in sequence if e[0] == "wait"]
                    assert actual == expected, (batch, pattern, query, actual, expected)
                    assert all(0 <= e[1] < constants["SCORE_SIGNAL_WORDS"] and e[2] == 1 for e in sequence)
                    wait_count += len(actual)
            assert checked == set(range(batch * group_size))
            rows.append(
                dict(batch=batch, pattern=pattern, queries=len(checked), leaves=leaves, waits=wait_count, passed=True)
            )
    return dict(
        passed=True,
        source=str(source),
        cases=rows,
        constants={k: constants[k] for k in names},
        scope="实际候选Scalar代码的写者/等待者映射及槽界；不代替设备数学、内存可见性或活性验证",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="fused_topk_v2")
    args = parser.parse_args()
    if not args.candidate.isidentifier():
        raise ValueError("候选必须是标识符")
    result = audit(HERE / "sources" / args.candidate)
    destination = HERE / f"ROOT_OWNERS_{args.candidate}.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(destination)
    print(f"{len(result['cases'])} control cases passed")


if __name__ == "__main__":
    main()
