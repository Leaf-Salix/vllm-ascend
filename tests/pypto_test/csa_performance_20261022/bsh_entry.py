# SPDX-License-Identifier: Apache-2.0
"""原BSH多task的真实NativeCSACall入口，仅用于S6的同输入比较。"""

from types import SimpleNamespace


class BSHEntry:
    def __init__(self, adapter, weights, fixture, groups, output, native):
        self.adapter = adapter
        self.call = adapter.NativeCSACall(
            adapter.CSAOperators.register(namespace="csa_performance_bsh"),
            weights,
            fixture["hidden"],
            fixture["positions"],
            groups,
            layer_name=fixture["groups"]["compressed"]["prefix"],
            compact_metadata=fixture["compact"],
            buffers={"idx_topk_scores": native["idx_topk_scores"], "idx_topk": native["idx_topk"], "x_out": output},
        )
        self.buffers = self.call.args
        self.policy = SimpleNamespace(constexprs=lambda: {})

    def __call__(self, *, config=None):
        self.adapter.decode_csa_tp1_layer_test(*self.call.core_args, config=config)
        return self.call.args["x_out"]
