# SPDX-License-Identifier: Apache-2.0
"""Real Leaf native/full-CSA host entry probe using frozen reference fixtures.

The reference supplies only test fixture/measurement helpers. All production
model, metadata, cache, native operators and CSA host code come from this tree.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    """Run common-state numeric, capture/replay, timing and fallback checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-tests", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        VLLM_ASCEND_ENABLE_NZ="2",
        VLLM_ASCEND_PTO_CSA_ATOMIC_ADD="0",
        VLLM_ASCEND_PYPTO_DSV4_CSA="1",
        HCCL_DETERMINISTIC="false",
    )
    sys.path.insert(0, str(args.reference_tests.resolve()))
    import dsv4_csa_single_layer as helpers
    import torch
    import torch_npu
    from dsv4_csa_native_case import native_session
    from dsv4_csa_validation import compare_tensor
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    import vllm_ascend
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention import pto_layer
    from vllm_ascend.utils import enable_custom_op

    expected = Path(__file__).resolve().parents[2] / "vllm_ascend"
    if Path(vllm_ascend.__file__).resolve().parent != expected:
        raise RuntimeError(f"Wrong production source: {vllm_ascend.__file__}")
    current_platform.pre_register_and_update()
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    if not enable_custom_op():
        raise RuntimeError("Native custom operators were not registered")
    torch_npu.npu.set_deterministic_level(0)
    config = EngineArgs(
        model=str(args.checkpoint),
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        quantization="ascend",
        hf_overrides={"sliding_window": 128},
        max_model_len=16384,
        max_num_seqs=40,
        max_num_batched_tokens=256,
        enable_prefix_caching=False,
        enforce_eager=True,
        block_size=32,
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
    ).create_engine_config()
    report = {
        "status": "RUNNING",
        "source": str(expected),
        "batch": 4,
        "history": 8192,
        "sequence": 6,
        "layer": 2,
        "seed": 1024,
        "timing_metadata": "produce",
        "scope": "Leaf真实单层host入口；真实权重/合成hidden和历史；不是完整模型验收",
    }
    try:
        with native_session(config, 0), torch.inference_mode():
            layer, weights = helpers.make_layer(config, args.checkpoint, torch.device("npu:0"), 2)
            report["weights"] = weights
            fixture = helpers.make_fixture(config, layer.self_attn, 4, 8192, 1024, torch.device("npu:0"))
            pto_layer.prepare_model(SimpleNamespace(model=SimpleNamespace(layers=[layer])), config)
            output = torch.empty_like(fixture["hidden"])
            topk = {}
            original_qli = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer

            def record_topk(*inputs, **kwargs):
                value = original_qli(*inputs, **kwargs)
                topk["value"] = value[0]
                return value

            def native():
                with set_ascend_forward_context(fixture["metadata"], config, num_tokens=24, num_actual_tokens=24):
                    pto_layer._native_attention_half(layer, fixture["hidden"], fixture["positions"], output)

            hits = []
            operator = layer._pto_csa_operator

            def counted(*inputs):
                hits.append(1)
                return operator(*inputs)

            layer._pto_csa_operator = counted

            class Probe:
                def __init__(self):
                    self.args = {"x_out": output, "idx_topk": layer._pto_csa_topk[:24]}

                def __call__(self):
                    with set_ascend_forward_context(fixture["metadata"], config, num_tokens=24, num_actual_tokens=24):
                        torch.ops.vllm.full_csa_forward(
                            fixture["hidden"], fixture["positions"], output, layer.self_attn.dsa_attn.prefix
                        )

            probe = Probe()
            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = record_topk
            try:
                helpers.restore(fixture)
                native()
                torch.npu.synchronize()
                native_state = helpers.collect_state(fixture, output, topk["value"])
                report["native_guards"] = helpers.guard_checks(fixture)
                helpers.restore(fixture)
                probe()
                torch.npu.synchronize()
                if not hits:
                    raise RuntimeError("Full CSA entry fell back instead of launching its kernel")
                csa_state = helpers.collect_state(fixture, output, probe.args["idx_topk"])
                report["csa_guards"] = helpers.guard_checks(fixture)
                report["csa_native"] = {
                    key: compare_tensor(value, native_state[key], 0, 0) for key, value in csa_state.items()
                }
                torch.save({"native": native_state, "pto": csa_state}, args.output / "states.pt")
                helpers.check_graph_replay(fixture, probe, csa_state, report)
                report["timing"] = {
                    "native": helpers.measure_graph_interval(
                        fixture,
                        native,
                        output,
                        lambda: topk["value"],
                        native_state,
                        iters=20,
                        warmup=5,
                        require_exact=False,
                    ),
                    "csa": helpers.measure_graph_interval(
                        fixture,
                        probe,
                        output,
                        lambda: probe.args["idx_topk"],
                        csa_state,
                        iters=20,
                        warmup=5,
                        require_exact=True,
                    ),
                }
                # Exercise the production native branch with unchanged valid metadata.
                original_eligible = pto_layer.eligible
                before = len(hits)
                pto_layer.eligible = lambda *unused: False
                try:
                    helpers.restore(fixture)
                    probe()
                    torch.npu.synchronize()
                finally:
                    pto_layer.eligible = original_eligible
                fallback = helpers.collect_state(fixture, output, topk["value"])
                report["forced_native_fallback"] = {
                    key: compare_tensor(value, native_state[key], 0, 0) for key, value in fallback.items()
                }
                if len(hits) != before or any(x["status"] != "PASS" for x in report["forced_native_fallback"].values()):
                    raise RuntimeError("Native fallback changed results or re-entered CSA")
                for name in ("native_guards", "csa_guards"):
                    if any(x["status"] != "PASS" for x in report[name].values()):
                        raise RuntimeError("Unexpected cache/metadata write")
                report.update(status="MEASURED", host_kernel_launches=len(hits))
            finally:
                torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_qli
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
