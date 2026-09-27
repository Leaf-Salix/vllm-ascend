"""固定正式HC权重与相同输入，隔离pre/norm、gates和post的差异来源。"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / "tests/pypto_test"))
    from dsv4_csa_env import activate, write_json
    from dsv4_csa_validation import compare_tensor

    activate()
    import pypto.torch
    import torch
    import torch_npu
    from safetensors import safe_open

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.hc_post import hc_post_test
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.hc_pre import hc_pre_norm_test
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    torch_npu.npu.set_deterministic_level(0)
    if not enable_custom_op():
        raise RuntimeError("Native custom ops unavailable")
    device = f"npu:{args.device}"
    checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    config = json.loads((checkpoint / "config.json").read_text())
    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm.weight"):
        key = f"layers.4.{name}"
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as handle:
            weights[name] = handle.get_tensor(key).to(device).contiguous()
    torch.manual_seed(1024)
    torch.npu.manual_seed(1024)
    x = torch.randn((96, 4, 4096), device=device, dtype=torch.bfloat16)
    attention = torch.randn((96, 4096), device=device, dtype=torch.bfloat16)
    report = {"scope": "单卡layer4正式HC权重，B16/S6；合成相同输入，不加载attention或MoE，不提供模型性能",
              "source": str(args.source), "dtype": {k: str(v.dtype) for k, v in weights.items()}, "checks": {}}

    def compare(name, actual, expected):
        report["checks"][name] = compare_tensor(actual, expected, 0, 0)

    def native_pre():
        return torch.ops._C_ascend.npu_hc_pre_v2(
            x, weights["hc_attn_fn"], weights["hc_attn_scale"], weights["hc_attn_base"],
            config["hc_mult"], config["hc_sinkhorn_iters"], config["rms_norm_eps"], config["hc_eps"])

    def native_post(post, comb):
        return torch.ops._C_ascend.npu_hc_post(
            attention.unsqueeze(0), x.unsqueeze(0), post.unsqueeze(0), comb.reshape(96, 4, 4).unsqueeze(0)
        ).squeeze(0)

    with torch.inference_mode():
        mixed, native_post_gate, native_comb = native_pre()
        native_norm = torch_npu.npu_rms_norm(mixed, weights["attn_norm.weight"], epsilon=config["rms_norm_eps"])[0]
        native_output = native_post(native_post_gate, native_comb)
        mixed_again, post_again, comb_again = native_pre()
        compare("native_pre_repeat.mixed", mixed_again, mixed)
        compare("native_pre_repeat.post", post_again, native_post_gate)
        compare("native_pre_repeat.comb", comb_again, native_comb)

        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        post = torch.empty((96, 4), device=device, dtype=torch.float32)
        comb = torch.empty((96, 16), device=device, dtype=torch.float32)
        normed = torch.empty((96, 4096), device=device, dtype=torch.bfloat16)
        hc_pre_norm_test(x.float(), weights["hc_attn_fn"], weights["hc_attn_scale"], weights["hc_attn_base"],
                         weights["attn_norm.weight"], post, comb, normed)
        compare("pto_pre_norm", normed, native_norm)
        compare("pto_post_gate", post, native_post_gate)
        compare("pto_comb", comb, native_comb.reshape(96, 16))

        output = torch.empty_like(x)
        hc_post_test(attention, x, native_post_gate, native_comb.reshape(96, 16), output)
        compare("pto_post_same_native_inputs", output, native_output)
        native_with_pto_gates = native_post(post, comb)
        compare("native_post_with_pto_gates", native_with_pto_gates, native_output)
        report["status"] = "MEASURED"
        if any(v.get("nonfinite") != 0 for v in report["checks"].values()):
            raise RuntimeError("Nonfinite or incomplete diagnostic output")
    write_json(args.output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
