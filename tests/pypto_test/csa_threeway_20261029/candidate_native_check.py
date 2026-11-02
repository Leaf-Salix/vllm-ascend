# SPDX-License-Identifier: Apache-2.0
"""性能胜出后，对拍Native完整HC-pre至HC-post的输出及全部物理cache。"""

import json


def check_native(fixture, layer, config, expected, directory):
    import torch
    from dsv4_csa_single_layer import guard_checks, restore

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.dsv4_csa import _native_attention_half

    output = torch.empty_like(fixture["hidden"])

    def snapshot():
        return {
            "output": output.cpu().clone(),
            **{n: g["allocation"].cpu().clone() for n, g in fixture["groups"].items()},
        }

    def compare(actual, reference):
        return {
            k: torch.equal(v.contiguous().view(torch.uint8), reference[k].contiguous().view(torch.uint8))
            for k, v in actual.items()
        }

    def call():
        with set_ascend_forward_context(
            fixture["metadata"], config, num_tokens=fixture["tokens"], num_actual_tokens=fixture["tokens"]
        ):
            _native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], output)

    restore(fixture)
    call()
    torch.npu.synchronize()
    eager = snapshot()
    assert all(row["status"] == "PASS" for row in guard_checks(fixture).values())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    restore(fixture)
    graph.replay()
    torch.npu.synchronize()
    actual = snapshot()
    result = dict(
        scope="Native FULL complete CSA; output and all physical caches; TopK unobserved",
        native_eager_full=compare(actual, eager),
        candidate_vs_native=compare(actual, expected),
        guards=guard_checks(fixture),
    )
    (directory / "candidate-native.json").write_text(json.dumps(result, indent=2) + "\n")
    assert all(result["native_eager_full"].values()), "Native eager/FULL first difference"
    assert all(row["status"] == "PASS" for row in result["guards"].values())
    assert all(result["candidate_vs_native"].values()), "first candidate/Native output or cache difference"
    return result
