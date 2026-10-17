# SPDX-License-Identifier: Apache-2.0
"""只重排593动态源码的两个完整阶段块，不生成静态shape或改数学。

新观测中主QB结束后，Indexer compact norm已就绪，原AIC仍先做独立主压缩
再做compact Hadamard；AIV则先做主Query/主压缩写回再发布Indexer scale。
这里把compact Hadamard放在主QB之后、主压缩投影之前，把compact量化及
scale提交放在系数之后、主Query之前。AIC可提前填两槽Score，双AIV继续
执行自己的主Query/主压缩工作，随后仍按原位置进入Score消费，不改变FFTS。

两个移动块携带原等待、计算、发布；全部顶层AST语句的多重集合保持相同。
这只证明未删改语句，不证明偏序无环、机器数值或性能；后续另行模型/验证。
"""

import ast
import json
from collections import Counter
from pathlib import Path


def statements(text, function):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == function)
    return Counter(ast.dump(n, include_attributes=False) for n in node.body)


def move_block(text, begin, end, before):
    assert all(text.count(anchor) == 1 for anchor in (begin, end, before))
    start = text.index(begin)
    stop = text.index(end, start)
    block = text[start:stop]
    remaining = text[:start] + text[stop:]
    return remaining.replace(before, block + before, 1)


def main():
    evidence = Path(__file__).resolve().parent
    worktree = evidence.parents[2]
    package = worktree / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    baseline = Path(
        "/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/"
        "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/baseline_source"
    )
    sources = {name: (baseline / name).read_text() for name in ("cube_schedule.py", "vector_schedule.py")}
    cube = move_block(
        sources["cube_schedule.py"],
        "    # 到这里才首次读取公共计划。",
        "    # Score消费query INT8、系数和Native key/scale。",
        "    # 先发布Indexer两种投影和query Hadamard，再执行独立的主压缩投影。",
    )
    cube = cube.replace(
        "    # 到这里才首次读取公共计划。未参与HC前段的AIV此前已经可以做metadata；\n"
        "    # 满核时planner先widen后扫描，仍不会阻止HC投影。每次从实际T重新计算同一owner。\n",
        "    # 主QB之后优先完成已就绪的Indexer compact Hadamard，随后做独立主压缩。\n"
        "    # 本轮META早于投影完成；仍显式获取本次计划与所有IDX_NORM生产者。\n"
        "    # 不假定CPU时序已保证可见性。每次从真实T重新计算同一planner owner。\n",
    )
    cube = cube.replace(
        "    # 先发布Indexer两种投影和query Hadamard，再执行独立的主压缩投影。\n"
        "    # 此时AIV可以做Indexer pool/norm及query量化；主投影不等待这些结果，\n"
        "    # 等IDX_NORM放在主投影之后，避免空等，也不改变任一矩阵累加顺序。\n",
        "    # compact Hadamard已发布；AIV可在这段主压缩投影期间量化并提交Indexer\n"
        "    # cache/scale。主压缩的数学、owner和累加顺序均不变。若别的形状下\n"
        "    # IDX_NORM较晚，前移的等待可能抵消收益，因此必须实测，不能假定免费。\n",
    )
    vector = move_block(
        sources["vector_schedule.py"],
        "    compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)\n",
        "    # Score两种角色必须同时可推进。",
        "    # 每个head占两个NZ N256列块。",
    )
    vector = vector.replace(
        "    compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)\n",
        "    # 系数之后先提交Indexer compact key/scale。主Query和主压缩仅供后继\n"
        "    # Attention消费，仍在本AIV进入Score之前完成；这里不先等待整段Score。\n"
        "    # AIC可以用既有两槽提前生产，槽复用继续由原SCORE_READY/ACK保证。\n"
        "    compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)\n",
        1,
    )
    cube = cube.replace(
        "主QB/压缩投影→compact Hadamard→Indexer score→Attention→O_A/O_B。矩阵计算调用各",
        "主QB→compact Hadamard→主压缩投影→Indexer score→Attention→O_A/O_B。矩阵计算调用各",
    )
    vector = vector.replace(
        "Indexer量化/系数和主Query反量化，利用AIC主QB计算时的Vector窗口；主/Indexer",
        "Indexer query量化/系数、compact key量化与scale提交，再主Query反量化及主压缩。\n"
        "让已具备Indexer输入的AIC提前填充Score双槽，本AIV仍先完成主分支再消费Score；\n"
        "没有等整段Score或跳过ACK。前移等待的代价须由整层实测判断。主/Indexer",
    )
    updated = {"cube_schedule.py": cube, "vector_schedule.py": vector}
    audit = {}
    for name, value in updated.items():
        function = name.removesuffix(".py")
        before, after = statements(sources[name], function), statements(value, function)
        assert before == after, f"{name}不只是语句重排"
        assert (package / name).read_text() == sources[name], "只允许在未修改的593基线上生成一次"
        (package / name).write_text(value)
        audit[name] = {"same_statement_multiset": True, "statements": sum(before.values())}
    (evidence / "source_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
