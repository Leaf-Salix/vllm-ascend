# SPDX-License-Identifier: Apache-2.0
"""CPU执行真实DSL中的地址/发布语句，验证长档缩批不会读取未写根。"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision/decode_indexer.py"


def score_function():
    return next(
        node
        for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "indexer_score_topk_native_cube"
    )


def execute(statements, values):
    module = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), values)


def test_parallel_leaves_use_private_score_rows_and_single_leaf_stays_contiguous():
    function = score_function()
    vector_branch = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "balance_leaves"
        and any(
            isinstance(child, ast.Name) and child.id == "buf_store_row"
            for item in node.orelse
            for child in ast.walk(item)
        )
    )
    statements = []
    for node in ast.walk(ast.Module(body=vector_branch.orelse, type_ignores=[])):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "buf_store_row" for target in node.targets
        ):
            # The first assignment starts the actual row/column selection block.
            parent = next(block for block in ast.walk(function) if isinstance(block, ast.For) and node in block.body)
            index = parent.body.index(node)
            statements = parent.body[index : index + 3]
            break
    assert len(statements) == 3, "缺少多leaf私有score地址分派"
    owners = {}
    for worker in range(24):
        for lane in range(2):
            for aiv in range(2):
                values = dict(
                    buf_query=0,
                    buf_query_lane=lane,
                    buf_worker=worker,
                    query_group_size=2,
                    buf_score_lane=aiv,
                    buf_score_begin=64,
                    buf_lane_begin=aiv * 4096,
                    buf_max_leaves=5,
                )
                execute(statements, values)
                row = values["buf_store_row"]
                assert row not in owners, "并行leaf的score行发生重叠"
                owners[row] = (worker, lane, aiv)
                assert values["buf_store_column"] == 64
                values["buf_max_leaves"] = 1
                execute(statements, values)
                assert values["buf_store_row"] == lane
                assert values["buf_store_column"] == 64 + aiv * 4096


@pytest.mark.parametrize("lengths", [(6,), (6, 6), (6, 6, 6), (1, 5, 2)])
def test_all_consumed_half_roots_are_written_for_ragged_small_batches(lengths):
    function = score_function()
    publishers = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "buf_max_leaves > 1"
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "indexer_topk_half_leaf"
            for child in ast.walk(node)
        )
    ]
    assert len(publishers) == 1, "长档merge读取的pair根必须有本轮publisher"
    positions = [131333 + i for length in lengths for i in range(length)]
    written = {}

    def publish(scores, pairs, row, begin, valid, slot):
        assert valid > 0
        assert slot not in written, "不完整query组越界发布到下一请求"
        written[slot] = True

    def store(value, offset, target):
        assert offset[0] not in written
        written[offset[0]] = False

    pl = SimpleNamespace(
        unroll=range,
        read=lambda tensor, index: tensor[index[0]],
        min=min,
        max=max,
        store=store,
        FP32=None,
        tile=SimpleNamespace(full=lambda *args, **kwargs: None),
    )
    begin = 0
    for length in lengths:
        for group in range(begin, begin + length, 2):
            rows = min(2, begin + length - group)
            for leaf in range(5):
                for aiv in range(2):
                    values = dict(
                        pl=pl,
                        buf_max_leaves=5,
                        query_group_size=2,
                        buf_group_rows=rows,
                        position_ids=positions,
                        buf_query=group,
                        buf_cache_len=40000,
                        COMPRESS_RATIO=4,
                        TOPK_MAX_CANDIDATES=65536,
                        buf_logical_begin=leaf * 8192,
                        buf_lane_begin=aiv * 4096,
                        buf_lane_span=4096,
                        TOPK_ROWS_PER_QUERY=16,
                        buf_leaf=leaf,
                        buf_score_lane=aiv,
                        buf_worker=0,
                        score_arena=None,
                        pair_arena=None,
                        TOPK_PAIR_WIDTH=1024,
                        FP32_NEG_INF=-float("inf"),
                        indexer_topk_half_leaf=publish,
                    )
                    execute(publishers, values)
        begin += length
    expected = {
        query * 16 + leaf * 2 + aiv
        for query, position in enumerate(positions)
        for leaf in range((((position + 1) // 4) + 8191) // 8192)
        for aiv in range(2)
    }
    assert set(written) == expected
    assert any(not live for live in written.values()), "尾leaf的空半区也必须初始化"
