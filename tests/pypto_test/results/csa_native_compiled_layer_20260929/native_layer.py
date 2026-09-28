"""Compare Native manual graph and the actual vLLM compile path on one CSA half-layer."""
import argparse
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

from dsv4_csa_env import activate

activate()
import torch  # noqa: E402
import torch_npu  # noqa: E402
from dsv4_csa_native_case import native_session  # noqa: E402
from dsv4_csa_single_layer import (  # noqa: E402
    collect_state,
    guard_checks,
    make_fixture,
    make_layer,
    measure_graph_interval,
    restore,
)
from offline_pd.run import decode_additional_config  # noqa: E402
from vllm.compilation.decorators import support_torch_compile  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402


@support_torch_compile(dynamic_arg_dims={"hidden_states": 0, "positions": 0, "output": 0})
class CompiledAttentionHalf(torch.nn.Module):
    def __init__(self, *, vllm_config, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor):
        # Same Native decoder half and dsa_forward boundary as the deployed model.
        layer = self.layer
        residual = hidden_states.clone()
        mixed, post, comb = layer.hc_pre(
            hidden_states, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
        normed = layer.input_layernorm(mixed)
        attended = layer.self_attn(positions=positions, hidden_states=normed, llama_4_scaling=None)
        output.copy_(layer.hc_post(attended, residual, post, comb))
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "history": args.history, "batch": args.batch, "layer_index": 4,
              "device": args.device, "cann": os.environ["ASCEND_HOME_PATH"],
              "opp": os.environ["ASCEND_OPP_PATH"], "custom_opp": os.environ["ASCEND_CUSTOM_OPP_PATH"],
              "scope": "Native attention half with real layer4 weights and synthetic history; no MoE/EP16 acceptance",
              "timing": {}}
    original_qli = None
    try:
        current_platform.pre_register_and_update()
        torch.npu.set_device(args.device)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("Native custom operators unavailable")
        tokens = args.batch * 6
        additional = decode_additional_config(SimpleNamespace(
            graph_mode="full_decode_only", capture_sizes=[tokens], recompute_scheduler=False))
        additional.update(weight_nz_mode=2, enable_kv_nz=False, enable_dsa_cp=False)
        config = EngineArgs(
            model="/data/model/DeepSeek-V4-Flash-0731-w8a8", tokenizer_mode="deepseek_v4",
            trust_remote_code=True, dtype="bfloat16", quantization="ascend", tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128}, max_model_len=max(16384, args.history + 128),
            max_num_seqs=40, max_num_batched_tokens=256, block_size=32, enable_prefix_caching=False,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [tokens],
                                "cache_dir": str(args.output / "compile_cache")},
            additional_config=additional,
        ).create_engine_config()
        report["requested"] = additional
        report["compile_config"] = str(config.compilation_config)
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.ops.dsv4_csa import _native_attention_half

        with native_session(config, args.device), torch.inference_mode():
            device = torch.device(f"npu:{args.device}")
            layer, weights = make_layer(config, Path(config.model_config.model), device, 4)
            report["weights"] = weights
            fixture = make_fixture(config, layer.self_attn, args.batch, args.history, 1024, device)
            # Match the existing paired CSA inputs, including nonuniform page scales.
            group = fixture["groups"]["indexer"]
            scale = group["views"][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            group["initial"] = group["allocation"].cpu()
            output = torch.empty_like(fixture["hidden"])
            topk = {}
            original_qli = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer

            def record_qli(*inputs, **kwargs):
                value = original_qli(*inputs, **kwargs)
                topk["value"] = value[0]
                return value

            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = record_qli

            def context():
                return set_ascend_forward_context(
                    fixture["metadata"], config, num_tokens=tokens, num_actual_tokens=tokens)

            def manual_call():
                with context():
                    _native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], output)

            restore(fixture)
            manual_call()
            torch.npu.synchronize()
            reference = collect_state(fixture, output, topk["value"])
            report["initial_guards"] = guard_checks(fixture)
            report["timing"]["manual"] = measure_graph_interval(
                fixture, manual_call, output, lambda: topk["value"], reference,
                iters=20, warmup=5, require_exact=False, profile_dir=args.output / "profile/manual")

            from npugraph_ex._acl_concrete_graph import static_kernel

            static_results = []
            original_compile = static_kernel.static_compile

            def observe_static(*inputs, **kwargs):
                success = original_compile(*inputs, **kwargs)
                static_results.append(success)
                return success

            static_kernel.static_compile = observe_static
            compiled = CompiledAttentionHalf(vllm_config=config, layer=layer)

            def compiled_call():
                with context():
                    compiled(fixture["hidden"], fixture["positions"], output)

            restore(fixture)
            compiled_call()
            torch.npu.synchronize()
            report["compiler"] = {"wrapper_compiled": compiled.compiled,
                                  "static_compile_results": static_results,
                                  "installed_static_packages": len(static_kernel._installed_run_pkgs)}
            if not compiled.compiled or not static_results or not all(static_results):
                raise RuntimeError(f"Native compilation did not succeed: {report['compiler']}")
            if not static_kernel._installed_run_pkgs:
                raise RuntimeError("No static kernel package was installed")
            report["timing"]["compiled"] = measure_graph_interval(
                fixture, compiled_call, output, lambda: topk["value"], reference,
                iters=20, warmup=5, require_exact=False, profile_dir=args.output / "profile/compiled")
            report["means_us"] = {side: statistics.mean(value["samples_us"])
                                  for side, value in report["timing"].items()}
            report["change_pct"] = 100 * (report["means_us"]["compiled"] / report["means_us"]["manual"] - 1)
            report["status"] = "MEASURED"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        if original_qli is not None:
            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_qli
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        summary = {k: report[k] for k in ("status", "means_us", "change_pct", "compiler", "error") if k in report}
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
