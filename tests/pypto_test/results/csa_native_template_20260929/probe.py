"""Single-card Native dependency/fusion/static-compile probe; no PyPTO or model performance claim."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401 -- registers NPU operators before vLLM imports
from dsv4_csa_native_case import native_session
from offline_pd.run import decode_additional_config
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform

ROOT = Path(__file__).resolve().parent


def main():
    current_platform.pre_register_and_update()
    device = int(os.environ["TASK_DEVICE"])
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(1024)
    from vllm_ascend.utils import enable_custom_op, register_ascend_customop

    if not enable_custom_op():
        raise RuntimeError("Native custom operators unavailable")
    additional = decode_additional_config(SimpleNamespace(
        graph_mode="full_decode_only", capture_sizes=[96], recompute_scheduler=False))
    additional.update(weight_nz_mode=2, enable_kv_nz=False, enable_dsa_cp=False)
    config = EngineArgs(
        model="/data/model/DeepSeek-V4-Flash-0731-w8a8", tokenizer_mode="deepseek_v4",
        trust_remote_code=True, dtype="bfloat16", quantization="ascend",
        max_model_len=131232, max_num_seqs=16, max_num_batched_tokens=256,
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [96]},
        additional_config=additional,
    ).create_engine_config()
    report = {"status": "RUNNING", "device": device, "cann": os.environ["ASCEND_HOME_PATH"],
              "custom_opp_path": os.environ["ASCEND_CUSTOM_OPP_PATH"], "requested": additional,
              "scope": "Native dependency and compiler probe, not CSA/model performance"}
    try:
        with native_session(config, device), torch.inference_mode():
            register_ascend_customop(config)
            x = torch.randn(96, 4096, dtype=torch.bfloat16, device=f"npu:{device}")
            residual, gamma = torch.randn_like(x), torch.ones_like(x[0])
            outputs = torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, gamma, None, 1e-6)
            torch.npu.synchronize()
            assert outputs[0].shape == x.shape and torch.isfinite(outputs[0]).all().item()
            report["add_rms_norm_bias"] = "PASS"
            from vllm_ascend.compilation.passes.norm_quant_fusion_pass import AddRMSNormQuantFusionPass

            fusion = AddRMSNormQuantFusionPass(config)
            report["norm_quant_fusion_registration"] = type(fusion).__name__
            import npugraph_ex as nge
            from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel
            from npugraph_ex.configs.npugraphex_config import _process_kwargs_options

            from vllm_ascend.ascend_config import get_ascend_config
            from vllm_ascend.compilation.compiler_interface import _configure_backend

            backend_config = nge.CompilerConfig()
            _configure_backend(backend_config, get_ascend_config().ascend_compilation_config,
                               config, process_kwargs_options=_process_kwargs_options)
            static_calls = []
            static_results = []
            original_compile = static_kernel.static_compile

            def observe_result(*args, **kwargs):
                success = original_compile(*args, **kwargs)
                static_results.append(success)
                return success

            static_kernel.static_compile = observe_result
            original_static = acl_graph.compile_static_kernel

            def observe_static(*args, **kwargs):
                result = original_static(*args, **kwargs)
                static_calls.append("completed")
                return result

            acl_graph.compile_static_kernel = observe_static

            def forward(value, skip, weight):
                normed, _, updated = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    value, skip, weight, None, 1e-6)
                quantized, scale = torch.ops.npu.npu_dynamic_quant(normed)
                return quantized, scale, updated

            # vLLM compiles a symbolic token dimension. Its sym_range filter
            # skips graphs with no symbolic inputs, even with static_kernel=True.
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(residual, 0)
            compiled = torch.compile(forward, backend=nge.get_npu_backend(compiler_config=backend_config),
                                     fullgraph=True, dynamic=None)
            actual = compiled(x, residual, gamma)
            torch.npu.synchronize()
            assert all(torch.isfinite(value).all().item() for value in actual)
            assert actual[0].shape == x.shape and actual[0].dtype == torch.int8
            report["static_compile_calls"] = static_calls
            report["static_packages"] = sorted(map(str, static_kernel._installed_run_pkgs))
            report["static_compile_results"] = static_results
            if not static_calls or not static_results or not all(static_results):
                raise RuntimeError("Static kernel compiler did not succeed")
            if not report["static_packages"]:
                raise RuntimeError("No static kernel package was installed")
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                replayed = compiled(x, residual, gamma)
            graph.replay()
            torch.npu.synchronize()
            for before, after in zip(actual, replayed):
                torch.testing.assert_close(after, before, rtol=0, atol=0)
            report["graph_replay"] = "PASS"
            report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        (ROOT / "probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
