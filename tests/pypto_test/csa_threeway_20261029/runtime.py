# SPDX-License-Identifier: Apache-2.0
"""实验共用公开ring容量选项；不修改SDK或算子。"""

import json
from functools import wraps

RING_HEAP = (256 << 20, 512 << 20, 256 << 20, 256 << 20)


def check_kv_budget(configs, budgets):
    rows = []
    for rank, (config, budget) in enumerate(zip(configs, budgets, strict=True)):
        planned = sum(((tensor.size + 511) // 512) * 512 for tensor in config.tensors)
        rows.append({"rank": rank, "planned_bytes": planned, "profiled_available_bytes": budget})
        if planned > budget:
            raise RuntimeError(f"EIGHT_CARD_KV_BUDGET_EXCEEDED {rows[-1]}")
    return rows


def install_capacity_guard():
    """测试进程内检查分配计划，不修改共享vLLM文件或容量算法。"""
    import vllm.v1.core.kv_cache_utils as utils
    import vllm.v1.engine.core as core

    original = utils.get_kv_cache_configs
    if getattr(original, "_eight_card_guard", False):
        return

    @wraps(original)
    def guarded(config, specs, available_memory):
        budgets = list(available_memory)
        result = original(config, specs, available_memory)
        print("EIGHT_CARD_KV_BUDGET " + json.dumps(check_kv_budget(result, budgets)), flush=True)
        return result

    guarded._eight_card_guard = True
    utils.get_kv_cache_configs = core.get_kv_cache_configs = guarded


def configure_runtime():
    import pypto.torch

    original = pypto.torch.init

    @wraps(original)
    def init(**kwargs):
        kwargs.setdefault("ring_heap", RING_HEAP)
        if tuple(kwargs["ring_heap"]) != RING_HEAP:
            raise ValueError("Three-way tests must share the recorded ring heap configuration")
        return original(**kwargs)

    pypto.torch.init = init
