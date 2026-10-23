# SPDX-License-Identifier: Apache-2.0
"""通过真实dsv4_hca_forward验证TND接管、KV钩子顺序和按步metadata缓存。"""

import importlib


def validate(*, package, layer, call, config, fixture, output, baseline, restore, collect):
    import torch
    from dsv4_csa_validation import compare_tensor
    from vllm.forward_context import get_forward_context
    from vllm_ascend import memcache_comm_fence
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention import utils

    module = importlib.import_module(package + ".service")
    adapter = importlib.import_module(package + ".native_adapter")
    wrapper = layer.self_attn.dsa_attn
    old_forward = layer.forward
    old_runtime = getattr(wrapper, "_pto_hca_runtime", None)
    events = []
    selected_policies = []

    class CountedOperator:
        def __init__(self, operator, policy):
            self.operator = operator
            self.policy = policy
            self._schema = operator._schema

        def __call__(self, *args):
            events.append("pto")
            selected_policies.append(self.policy)
            self.operator(*args)

    originals = {}
    for name, label in (("wait_for_kv_layer_from_connector", "wait"),
                        ("notify_kv_cache_written", "notify"),
                        ("maybe_save_kv_layer_to_connector", "save")):
        original = getattr(utils, name)
        originals[name] = original

        def wrapped(*args, _original=original, _label=label, **kwargs):
            events.append(_label)
            return _original(*args, **kwargs)

        setattr(utils, name, wrapped)
    original_start = memcache_comm_fence.record_attention_compute_start

    def start(*args, **kwargs):
        events.append("compute")
        return original_start(*args, **kwargs)

    memcache_comm_fence.record_attention_compute_start = start
    impl = wrapper.dsa_attn.impl
    original_compact = impl._compute_compressor_metadata

    def compact(*args, **kwargs):
        events.append("compact")
        return original_compact(*args, **kwargs)

    impl._compute_compressor_metadata = compact
    results = []
    try:
        counted = adapter.HCAOperators(CountedOperator(call.operators.attention, "default"))
        if getattr(call.operators, "attention_tall", None) is not None:
            tall_policy = "tall_projection" if hasattr(adapter, "use_tall_projection") else "tall_qb"
            counted = adapter.HCAOperators(counted.attention, CountedOperator(call.operators.attention_tall, tall_policy))
        if getattr(call.operators, "attention_long", None) is not None:
            counted = adapter.HCAOperators(counted.attention, counted.attention_tall,
                                           CountedOperator(call.operators.attention_long, "long_attention"))
        runtime = module.bind_layer(layer, counted,
                                    max_num_seqs=64, max_num_batched_tokens=384)
        for step, calls in ((0, 2), (1, 1)):
            # 两次同context调用复用compact；下一个context必须重新执行Native producer。
            with set_ascend_forward_context(fixture["metadata"], config, num_tokens=fixture["tokens"],
                                           num_actual_tokens=fixture["actual_tokens"]):
                context = get_forward_context()
                eligible = runtime.eligible(context, fixture["hidden"], fixture["positions"])
                if not eligible:
                    raise AssertionError("有效TND输入未由新服务接管，不能以Native回退冒充PASS")
                for iteration in range(calls):
                    events.clear()
                    restore(fixture)
                    output.fill_(float("nan"))
                    torch.ops.vllm.dsv4_hca_forward(
                        fixture["hidden"], fixture["positions"], output, wrapper.prefix,
                    )
                    torch.npu.synchronize()
                    current = collect()
                    checks = {"output": compare_tensor(current["output"], baseline["output"], 0, 0)}
                    for name in fixture["groups"]:
                        checks[name] = [compare_tensor(a, b, 0, 0)
                                        for a, b in zip(current[name], baseline[name])]
                    flat = [q for value in checks.values() for q in (value if isinstance(value, list) else [value])]
                    expected = ["wait"] + (["compact"] if iteration == 0 else []) + ["compute", "pto", "notify", "save"]
                    results.append({"step": step, "iteration": iteration, "eligible": eligible,
                                    "events": list(events), "expected_events": expected, "exact": checks,
                                    "selected_policy": selected_policies[-1],
                                    "passed": events == expected
                                              and selected_policies[-1] == getattr(call, "policy", "default")
                                              and all(q["status"] == "PASS" for q in flat)})
        return {"passed": all(row["passed"] for row in results), "calls": results,
                "aclgraph_replay": False, "performance_measured": False,
                "scope": "真实attention半层custom op接管及KV钩子；不包含FFN/整模型token验收"}
    finally:
        for name, original in originals.items():
            setattr(utils, name, original)
        memcache_comm_fence.record_attention_compute_start = original_start
        impl._compute_compressor_metadata = original_compact
        layer.forward = old_forward
        wrapper._pto_hca_runtime = old_runtime
