# SPDX-License-Identifier: Apache-2.0
"""复用实际迁移的页证明helper，检查正序/逆序/碎片/失活/容量及同址图更新。"""

import importlib
import re

import pypto.language as pl

ROWS = 6
COLUMNS = 1056
VALID_PAGES = 1025
OUTPUT_WORDS = 128
PHYSICAL_PAGES = 8192


def make_probe(prove_fn):
    @pl.jit
    def scale_layout_probe(
        lengths: pl.Tensor[[ROWS], pl.INT32],
        table: pl.Tensor[[ROWS, COLUMNS], pl.INT32],
        workspace: pl.InOut[pl.Tensor[[OUTPUT_WORDS], pl.INT32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="scale_layout_probe"):
            workspace = prove_fn(lengths, table, workspace, ROWS * 6, PHYSICAL_PAGES)
        return workspace

    return scale_layout_probe


def validate(adapter, device, options):
    import torch

    helper = importlib.import_module(adapter.__package__ + ".indexer_scale_layout").prove_scale_layout
    probe = make_probe(helper)
    data = torch.full((ROWS, COLUMNS), -1, dtype=torch.int32)
    data[0, :VALID_PAGES] = torch.arange(100, 100 + VALID_PAGES)
    data[1, :VALID_PAGES] = torch.arange(4095, 4095 - VALID_PAGES, -1)
    data[2, :VALID_PAGES] = torch.arange(2000, 2000 + VALID_PAGES)
    data[2, 1], data[2, 2] = data[2, 2].item(), data[2, 1].item()
    data[3] = data[0]
    data[4, :VALID_PAGES] = torch.arange(5000, 5000 + VALID_PAGES)
    data[4, 300] = -1
    data[5] = data[0]
    lengths_cpu = torch.full((ROWS,), VALID_PAGES * 32 * 4, dtype=torch.int32)
    lengths_cpu[3] = 0
    lengths_cpu[5] = (COLUMNS + 1) * 32 * 4
    table, lengths = data.to(device), lengths_cpu.to(device)
    output = torch.full((OUTPUT_WORDS,), 37, dtype=torch.int32, device=device)
    probe(lengths, table, output, config=options)
    torch.npu.synchronize()

    def check(expected):
        value = output.cpu()
        actual = value[: 2 * ROWS].reshape(ROWS, 2).tolist()
        passed = actual == expected and bool((value[2 * ROWS :] == 37).all())
        if not passed:
            raise AssertionError(f"布局证明/保护区错误：{actual}，期望{expected}")
        return dict(passed=True, expected=expected, actual=actual, unused_words_unchanged=True)

    first = check([[100, 1], [4095, -1], [0, 0], [0, 0], [0, 0], [0, 0]])
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            probe(lengths, table, output, config=options)
        data[0, 1], data[0, 2] = data[0, 2].item(), data[0, 1].item()
        data[1, :VALID_PAGES] = torch.arange(500, 500 + VALID_PAGES)
        data[2, :VALID_PAGES] = torch.arange(3000, 3000 - VALID_PAGES, -1)
        data[4, 300] = 5300
        lengths_cpu[3] = VALID_PAGES * 32 * 4
        lengths_cpu[5] = VALID_PAGES * 32 * 4
        table.copy_(data)
        lengths.copy_(lengths_cpu)
        graph.replay()
        stream.synchronize()
    second = check([[0, 0], [500, 1], [3000, -1], [100, 1], [5000, 1], [100, 1]])
    return dict(passed=True, source="实际candidate helper", eager=first, same_address_graph_update=second)


def verify_edges(trace):
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    task_pattern = r"(?:r\d+)*t\d+"
    producers = {
        label
        for event in events
        if event.get("name", "").startswith("indexer_scale_layout")
        for label in re.findall(task_pattern, event["name"])
    }
    scores = [e for e in events if e.get("name", "").startswith("indexer_score_topk_native_pair")]
    if not producers:
        return dict(applicable=False, reason="非balanced路径不创建布局证明task")
    if len(producers) != 1 or not scores:
        raise AssertionError("需要唯一布局证明生产者和Score消费者")
    if any(not producers.issubset(set(re.findall(task_pattern, e["args"].get("fanin-hint", "")))) for e in scores):
        raise AssertionError("Score泳道缺少布局证明的直接生产者依赖")
    return dict(applicable=True, passed=True, producer=list(producers), score_core_records=len(scores))
