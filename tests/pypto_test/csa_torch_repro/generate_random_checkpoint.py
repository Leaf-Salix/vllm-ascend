# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""生成仅供 CSA 算子一致性测试的随机 checkpoint；不读取模型文件、不使用 NPU。"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="新建 checkpoint 目录（不能已存在）")
    parser.add_argument("--seed", type=int, default=20261022, help="随机种子")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)  # 防止覆盖已有权重；重复运行时使用新目录。
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    weights = {}

    def put(name, value):
        weights["layers.4." + name] = value.contiguous()

    def normal(shape, std=0.01):
        return torch.randn(shape, generator=generator) * std

    put("hc_attn_fn", normal((24, 16384), 1 / 128))
    put("hc_attn_scale", torch.ones(3))
    put("hc_attn_base", normal((24,)))
    put("attn_norm.weight", torch.ones(4096))
    for name, rows, cols in (
        ("wq_a", 1024, 4096),
        ("wkv", 512, 4096),
        ("wo_a", 8192, 4096),
        ("indexer.weights_proj", 64, 4096),
    ):
        put("attn." + name + ".weight", normal((rows, cols), cols**-0.5).bfloat16())
    for name, rows, cols in (("wq_b", 32768, 1024), ("wo_b", 4096, 8192), ("indexer.wq_b", 8192, 1024)):
        put("attn." + name + ".weight", torch.randint(-8, 8, (rows, cols), generator=generator, dtype=torch.int8))
        scale = (0.75 + torch.rand((rows, 1), generator=generator)) / (8 * cols**0.5)
        put("attn." + name + ".weight_scale", scale)
        put("attn." + name + ".weight_offset", torch.zeros(rows, 1))
    for prefix, width in (("compressor", 512), ("indexer.compressor", 128)):
        for name in ("wkv", "wgate"):
            put(f"attn.{prefix}.{name}.weight", normal((2 * width, 4096), 1 / 64))
        put(f"attn.{prefix}.ape", normal((4, 2 * width)))
        put(f"attn.{prefix}.norm.weight", torch.ones(width))
    put("attn.q_norm.weight", torch.ones(1024))
    put("attn.kv_norm.weight", torch.ones(512))
    put("attn.attn_sink", normal((64,)))

    filename = "random-csa.safetensors"
    save_file(weights, str(out / filename), metadata={"source": "synthetic", "seed": str(args.seed)})
    (out / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: filename for name in weights}}, indent=2) + "\n"
    )
    # 仅包含当前测试入口需要的 RoPE 参数，不是可用于整模型推理的 config。
    config = {
        "compress_rope_theta": 160000,
        "qk_rope_head_dim": 64,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "SYNTHETIC_WEIGHTS.json").write_text(
        json.dumps(
            {
                "source": "synthetic",
                "seed": args.seed,
                "layer": 4,
                "purpose": "CSA implementation parity only; not model accuracy or real-weight performance",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"随机 checkpoint：{out}；{len(weights)} 个张量")


if __name__ == "__main__":
    main()
