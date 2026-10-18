# SPDX-License-Identifier: Apache-2.0
"""离线检查真实消费者控制函数是否覆盖所有生产者，不模拟设备缓存或数学。

从候选源码提取两个实际等待函数，只替换pl的整数/读取/信号运行环境。
oracle从生产者正向枚举(item, lane, query, root)写入关系，避免把消费者
反解公式复制一遍当成验证。另核对Top-K与plan数学AST未被改动。
"""

import ast
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent


def lifted_functions():
    tree = ast.parse((HERE / "source_candidate/attention_dependencies.py").read_text())
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node = copy.deepcopy(node)
            node.decorator_list = []
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            functions.append(node)
    pl = SimpleNamespace(
        cast=lambda value, dtype: int(value), INDEX=None,
        read=lambda value, indices: value[indices[0]], range=range,
        system=SimpleNamespace(fence=lambda: None, cacheinvalid=lambda: None),
    )
    env = dict(pl=pl, GROUPS=24, VECTORS=48, CORES=72,
               PHASE_SCORE_VECTOR=35, PHASE_ATTN_PLAN=37, SIGNAL_SLOT_WORDS=128)

    def wait(signals, offset, generation):
        assert generation == 7 and offset % 128 == 0
        signals.append((offset // 128 // 72, offset // 128 % 72))
        return signals

    env["csa_signal_wait"] = wait
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 str(HERE / "source_candidate/attention_dependencies.py"), "exec"), env)
    return env


def math_unchanged():
    """只去掉本次明确增加的等待节点，旧数学子树须原样保留。"""
    reports = {}
    for filename, function_name in (
        ("indexer_publish_stages.py", "indexer_pair_publish_stage"),
        ("attention_plan_stages.py", "attention_slot_plan_stage"),
    ):
        trees = []
        for side in ("source_reference", "source_candidate"):
            module = ast.parse((HERE / side / filename).read_text())
            function = next(n for n in module.body
                            if isinstance(n, ast.FunctionDef) and n.name == function_name)
            if side == "source_candidate":
                class RemoveNewControl(ast.NodeTransformer):
                    def visit_Assign(self, node):
                        targets = [n.id for n in node.targets if isinstance(n, ast.Name)]
                        if targets in (["local_query"], ["group"], ["signals"]):
                            return None
                        return self.generic_visit(node)

                    def visit_Return(self, node):
                        if isinstance(node.value, ast.Tuple):
                            node.value.elts = [n for n in node.value.elts
                                               if not isinstance(n, ast.Name) or n.id != "signals"]
                        return node

                function = RemoveNewControl().visit(function)
            trees.append(ast.dump(ast.Module(body=function.body, type_ignores=[])))
        assert trees[0] == trees[1], function_name
        reports[function_name] = True
    # Attention核内流水、Score根生产者均未改动。
    for filename in ("attention_stream_stages.py", "indexer_score_stages.py", "indexer_topk_stages.py"):
        assert (HERE / "source_reference" / filename).read_bytes() == (HERE / "source_candidate" / filename).read_bytes()
        reports[filename] = True
    return reports


def main():
    env = lifted_functions()
    cases = 0
    writer_counts = []
    for groups, leaves, query_group, balanced in itertools.product(
        (1, 2, 3, 4, 7, 8, 15, 16, 23, 24, 25, 28, 32, 48, 65),
        (1, 2, 3, 4, 6, 7, 16, 32), (1, 2, 6), (False, True),
    ):
        if balanced and query_group == 1:
            continue
        # 按生产者的遍历顺序生成每个query的根owner，包含少/多group两种work映射。
        work = [(g, leaf) for leaf in range(leaves) for g in range(groups)] if groups < 24 else [
            (g, leaf) for g in range(groups) for leaf in range(leaves)]
        producers = {}
        for item, (g, leaf) in enumerate(work):
            for lane in (0, 1):
                local_queries = range(lane * (query_group // 2), (lane + 1) * (query_group // 2)) if balanced else range(query_group)
                for local in local_queries:
                    root = leaf if balanced else 2 * leaf + lane
                    producers[g, local, root] = 24 + 2 * (item % 24) + lane
        for g in range(groups):
            for local in range(query_group):
                # 同group内query可因因果位置有更短的可见叶前缀。
                for live_leaves in sorted({1, max(1, leaves - 1), leaves}):
                    roots = live_leaves if balanced else 2 * live_leaves
                    actual = env['wait_indexer_query_roots'](
                        [], [groups, 0, leaves, 0, 0, query_group, int(balanced), 1 if balanced else 2],
                        g, local, roots, 7, query_group, balanced,
                    )
                    expected = {(35, producers[g, local, root]) for root in range(roots)}
                    assert set(actual) == expected, (groups, leaves, query_group, balanced, g, local)
                    cases += 1
                    if (groups, leaves, query_group, balanced, live_leaves) == (16, 6, 6, True, 6):
                        writer_counts.append(len(set(actual)))
    # plan生产者正向遍历八行块，消费者读取query%24；覆盖尾块及多轮strip。
    plan_cases = 0
    for tokens in list(range(0, 200)) + [255, 383, 384, 385, 511, 768, 1025, 4096]:
        writers = {}
        for worker in range(48):
            for first in range(worker * 8, tokens, 48 * 8):
                queries = range(first, min(first + 8, tokens))
                # plan获取的Top-K writer集合必须等于各query实际唯一writer。
                count = min(len(queries), 48)
                waits = {(first + i) % 48 for i in range(count)}
                assert waits == {q % 48 for q in queries}
                for q in queries:
                    writers[q] = 24 + worker
        for core in range(24):
            actual = env['wait_attention_plan_rows']([], tokens, core, 7)
            assert set(actual) == {(37, writers[q]) for q in range(core, tokens, 24)}
            plan_cases += 1
    # request-local分组必须在非统一S时仍与group producer的正向枚举相同。
    mixed_cases = 0
    for sizes in ((6,) * 16, (1,) * 16, (5,) * 7, (1, 5, 6, 0, 9, 2), (0, 0, 1), (3, 17, 2, 0, 7)):
        for query_group in (1, 2, 6):
            token, group = 0, 0
            for size in sizes:
                begin, base = token, group
                for local in range(0, size, query_group):
                    for query in range(begin + local, begin + min(local + query_group, size)):
                        assert base + (query - begin) // query_group == group
                        assert (query - begin) % query_group == query - (begin + local)
                        mixed_cases += 1
                    group += 1
                token += size
    report = dict(root_dependency_cases=cases, plan_dependency_cases=plan_cases,
                  mixed_query_mapping_cases=mixed_cases,
                  b16_g6_six_leaves_unique_writers_per_query=sorted(set(writer_counts)),
                  math_ast_unchanged=math_unchanged(),
                  hardware_sync_validated=False, performance_evidence=False)
    (HERE / 'DEPENDENCY_COVERAGE.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
