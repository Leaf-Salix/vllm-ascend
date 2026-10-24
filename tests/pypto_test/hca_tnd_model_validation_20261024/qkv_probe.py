# SPDX-License-Identifier: Apache-2.0
"""真实第4层输入的QKV边界诊断，Native归一化输入固定，排除attention/KV历史的干扰。"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(args.output / "build")
    os.environ["PYPTO_CACHE_DIR"] = str(args.output / "cache")
    report = {"status": "RUNNING", "operator": str(args.operator), "source_input": "真实模型首轮layer3输入",
              "shared_toolchain": os.environ.get("CSA_SHARED_TOOLCHAIN_RELEASE"), "performance_measured": False}
    def save():
        (args.output / "result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    try:
        from dsv4_csa_env import activate
        activate()
        import vllm_ascend.ops.pypto as package
        package.__path__ = [str(args.operator.resolve())]
        import pypto.torch
        import torch
        import torch_npu
        from pypto import passes
        from pypto.runtime import CompileOptions
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import make_layer
        from dsv4_csa_validation import compare_tensor
        from csa_mpmd_tnd_20261021.scenarios import make_fixture
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.utils import enable_custom_op

        device_id = int(os.environ["TASK_DEVICE"])
        assert args.device is None or args.device == device_id
        torch.npu.set_device(device_id)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(1)
        current_platform.pre_register_and_update()
        assert enable_custom_op()
        model = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
        config = EngineArgs(model=str(model),tokenizer_mode="deepseek_v4",trust_remote_code=True,
                           dtype="bfloat16",quantization="ascend",tensor_parallel_size=1,
                           hf_overrides={"sliding_window":128},max_model_len=131570,max_num_seqs=16,
                           max_num_batched_tokens=256,block_size=32,enable_prefix_caching=False,enforce_eager=True,
                           speculative_config={"method":"dspark","num_speculative_tokens":5,"enforce_eager":True},
                           additional_config={"weight_nz_mode":2,"enable_kv_nz":False,"enable_dsa_cp":False}).create_engine_config()
        device = torch.device(f"npu:{device_id}")
        actual_dir = HERE / "runs/first_token_native_v05/precision/rank0"
        summary = json.loads((actual_dir / "summary.json").read_text())
        histories = [length-1 for length in summary["first_forward"]["seq_lens"]]
        with native_session(config,device_id),torch.inference_mode():
            layer, report["weights"] = make_layer(config,model,device,3)
            fixture = make_fixture(config,layer.self_attn,16,131072,1024,device,
                                   query_lengths=[1]*16,history_lengths=histories,table_history=131570)
            x = torch.load(actual_dir / "layer_03.pt",map_location="cpu",weights_only=True)["input"].to(device)
            fixture["hidden"].copy_(x)
            main = fixture["metadata"][fixture["groups"]["compressed"]["prefix"]].decode
            swa_meta = fixture["metadata"][fixture["groups"]["swa"]["prefix"]].decode
            swa = fixture["groups"]["swa"]["views"][0]
            impl = layer.self_attn.dsa_attn.dsa_attn.impl
            native = {}
            original_qa = impl.cv_wq_a.matmul
            def qa(*a, **kw):
                y = original_qa(*a, **kw)
                native["qa"] = y.detach().clone()
                return y
            impl.cv_wq_a.matmul = qa
            with set_ascend_forward_context(fixture["metadata"],config,num_tokens=16,num_actual_tokens=16):
                mixed, post, comb = layer.hc_pre(x,layer.hc_attn_fn,layer.hc_attn_scale,layer.hc_attn_base)
                normalized = layer.input_layernorm(mixed)
                layer_name = layer.self_attn.dsa_attn.dsa_attn.layer_name
                cos, sin = main.cos[layer_name], main.sin[layer_name]
                q, qr, scale = impl._mla_prolog_multistream(normalized,cos,sin,swa,swa_meta.slot_mapping)
                torch.npu.synchronize()
            impl.cv_wq_a.matmul = original_qa
            slots = swa_meta.slot_mapping.cpu().tolist()
            native.update(q=q.reshape(16,32768).cpu(),qr=qr.cpu(),qr_scale=scale.reshape(16,1).cpu(),
                          kv=torch.stack([swa[page,offset,0,:] for page,offset in slots]).cpu(),qa=native["qa"].cpu())
            report["native_qa_dtype"] = str(native["qa"].dtype)
            report["native_normalized_dtype"] = str(normalized.dtype)
            adapter = importlib.import_module("vllm_ascend.ops.pypto.deepseek_v4_flash_hca_tnd.native_adapter")
            weights = adapter.prepare_weights(layer.self_attn,layer)
            pypto.torch.init(device=device_id,platform="a2a3",runtime="tensormap_and_ringbuffer",
                             ring_heap=[64<<20,256<<20,64<<20,64<<20],ring_task_window=[4096]*4)
            entry = importlib.import_module("vllm_ascend.ops.pypto.deepseek_v4_flash_hca_tnd.qkv_probe_entry")
            operation = pypto.torch.register(entry.qkv_probe,"hca_precision_probe::qkv",
                                              config=CompileOptions(platform="a2a3",memory_planner=passes.MemoryPlanner.PYPTO))
            outputs = {"q":torch.empty((16,32768),device=device,dtype=torch.bfloat16),
                       "kv":torch.empty((16,512),device=device,dtype=torch.bfloat16),
                       "qr":torch.empty((16,1024),device=device,dtype=torch.int8),
                       "qr_scale":torch.empty((16,1),device=device,dtype=torch.float32)}
            inputs = dict(x=normalized,wq_a=weights["wq_a"],wq_b=weights["wq_b"],wq_b_scale=weights["wq_b_scale"],
                          wkv=weights["wkv"],rope_cos=cos.reshape(16,64),rope_sin=sin.reshape(16,64),
                          gamma_cq=weights["gamma_cq"],gamma_ckv=weights["gamma_ckv"],**outputs)
            operation(*(inputs[a.name] for a in operation._schema.arguments))
            torch.npu.synchronize()
            actual = {k:v.cpu() for k,v in outputs.items()}
            report["comparisons"] = {k:compare_tensor(actual[k],native[k],0,0) for k in outputs}
            report["q08"] = {k:compare_tensor(actual[k][8:9],native[k][8:9],0,0) for k in outputs}
            report["status"] = "COMPARED"
            torch.save({"native":native,"pto":actual,"normalized":normalized.cpu()},args.output / "values.pt")
            save()
            print(json.dumps({"status":report["status"],"qa_dtype":report["native_qa_dtype"],
                              "metrics":{k:{f:v[f] for f in ("max_abs","rmse","mismatches")} for k,v in report["comparisons"].items()}},ensure_ascii=False),flush=True)
    except BaseException:
        report.update(status="FAIL",traceback=traceback.format_exc());save();raise


if __name__ == "__main__":
    main()
