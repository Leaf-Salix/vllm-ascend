#!/usr/bin/env python3
"""执行真实源码的Scalar控制部分，独立核对分段后不漏写、不重写、不换owner。

只替换Tile数学为记录(tg,hg,count)的桩。此检查不能代替设备数学、内存规划或
FFTS验证；不使用调度成本模型，也不据窗口个数估算性能。
"""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def function(path, name):
    module = ast.parse(path.read_text())
    return copy.deepcopy(next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name))


def register(node, env):
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, str(HERE / "source_candidate_v1"), "exec"), env)


def main():
    ref = HERE / "source_reference/query_projection_stages.py"
    cand = HERE / "source_candidate_v1/query_projection_stages.py"
    score = HERE / "source_candidate_v1/indexer_score_stages.py"
    calls = []

    def block(*args):
        tokens, tg, hg, count = args[-4:]
        assert count > 0 and 0 <= hg < 64 and hg + count <= 64
        assert 0 <= tg < tokens and tg % 8 == 0
        if tg + 8 > tokens:
            assert count == 4 and hg % 4 == 0
        calls.extend((tg, h) for h in range(hg, hg + count))
        return args[4]

    pl = SimpleNamespace(tensor=SimpleNamespace(dim=lambda t, axis: t), min=min, max=max, range=range)
    env = dict(pl=pl, H=64, Q_HEADS=64, Q_ROPE_T_TILE=8, Q_ROPE_H_TILE=4,
               Q_DEQUANT_WORKERS=48, TOPK_SCORE_WORKERS=24, q_dequant_rope_block_stage=block)
    old = function(ref, "q_dequant_rope_stage")
    work_loop = next(n for n in old.body if isinstance(n, ast.For))
    work_loop.body = work_loop.body[:2] + ast.parse(
        "q_flat = q_dequant_rope_block_stage(None,None,None,None,q_flat,None,None,tokens,tg,hg,dq_head_tile)"
    ).body
    old.name = "reference"
    register(old, env)
    register(function(cand, "q_dequant_head_group_size"), env)
    register(function(cand, "q_dequant_rope_chunk_stage"), env)
    stage = function(score, "indexer_score_vector_lane_stage")
    tail = stage.body[-2]
    assert isinstance(tail, ast.If) and ast.unparse(tail.test) == "INTERLEAVE_QUERY_DQ"
    tail_fn = ast.parse("def drain(q_flat,q_dq_work,q_dq_head_offset):\n    pass").body[0]
    tail_fn.body = tail.body
    for key in ("q_weight_scale", "q_cos", "q_sin", "q_swap", "q_scale", "q_acc"):
        env[key] = None
    register(tail_fn, env)
    checks = 0
    for tokens in range(1, 513):
        for worker in range(48):
            calls.clear()
            env["reference"](None, None, None, None, tokens, None, None, worker, 48)
            expected = list(calls)
            assert len(expected) == len(set(expected))
            for windows in (0, 1, 2, 3, 8, 32):
                calls.clear()
                work, offset = worker, 0
                for _ in range(windows):
                    _, work, offset = env["q_dequant_rope_chunk_stage"](
                        None, None, None, None, tokens, None, None, work, offset, 48
                    )
                env["drain"](tokens, work, offset)
                assert calls == expected, (tokens, worker, windows)
                checks += 1
    def protocol(path):
        body = function(path, "indexer_score_vector_lane_stage")
        return [ast.unparse(n) for n in ast.walk(body) if isinstance(n, ast.Call)
                and ast.unparse(n.func) in {"pl.system.sync_wait", "pl.system.sync_set"}]
    assert protocol(HERE / "source_reference/indexer_score_stages.py") == protocol(score)
    (HERE / "OWNER_AUDIT.json").write_text(json.dumps(dict(
        passed=True, cases=checks, token_range=[1, 512], workers=48,
        available_windows=[0, 1, 2, 3, 8, 32], exact_order_and_owner=True,
        full_row_and_tail_contract=True, score_ffts_calls_unchanged=True,
        limitation="仅控制/所有权核对，不是设备精度、同步时序或性能结果",
    ), ensure_ascii=False, indent=2) + "\n")
    print(f"{checks} source-control cases passed; original Score READY/ACK calls retained")


if __name__ == "__main__":
    main()
