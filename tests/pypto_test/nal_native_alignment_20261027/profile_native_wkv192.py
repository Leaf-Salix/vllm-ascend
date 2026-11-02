# SPDX-License-Identifier: Apache-2.0
"""仅观测已复现的Native WKV，保存实际kernel/tiling属性，不改算子。"""

import json
import os
import sys
from pathlib import Path


def main():
    import torch
    import torch_npu
    from safetensors import safe_open

    assert os.environ["TASK_DEVICE"] in ("8", "9", "10", "11", "12", "13", "15")
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(exist_ok=False)
    data = torch.load(
        root / "results/qa192fix-stages-full-20261102-v1/observation.pt", map_location="cpu", weights_only=True
    )
    x = data["input_norm_native"].to("npu")
    assert x.shape == (192, 4096) and x.dtype == torch.bfloat16
    checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    key = "layers.4.attn.wkv.weight"
    shard = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"][key]
    with safe_open(checkpoint / shard, framework="pt", device="cpu") as reader:
        weight_cpu = reader.get_tensor(key)
    weight = torch_npu.npu_format_cast(weight_cpu.to("npu"), 29)
    gold = data["wkv_projection_bf16_native"]

    def exact(value, expected):
        return torch.equal(value.cpu().contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))

    before = torch.nn.functional.linear(x, weight)
    torch.npu.synchronize()
    assert exact(before, gold), "profiling must reproduce the actual Native CV boundary"
    profiler = torch_npu.profiler
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        record_shapes=True,
        on_trace_ready=profiler.tensorboard_trace_handler(str(out / "profile")),
        experimental_config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level1,
            record_op_args=True,
            op_attr=True,
            data_simplification=False,
        ),
    ):
        observed = torch.nn.functional.linear(x, weight)
        torch.npu.synchronize()
    after = torch.nn.functional.linear(x, weight)
    torch.npu.synchronize()
    result = {
        "native_saved_equal": exact(before, gold),
        "profiled_equal": exact(observed, gold),
        "after_equal": exact(after, gold),
        "input_readonly": exact(x, data["input_norm_native"]),
        "weight_readonly": exact(weight, weight_cpu),
        "physical_device": os.environ["TASK_DEVICE"],
        "scope": "isolated Native WKV; profiler observation only; not performance measurement",
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert all(v for v in result.values() if isinstance(v, bool))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
