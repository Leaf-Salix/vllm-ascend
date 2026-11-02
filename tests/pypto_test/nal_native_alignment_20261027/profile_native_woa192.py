# SPDX-License-Identifier: Apache-2.0
"""仅观测已复现的Native WO-A，保存实际kernel/tiling属性，不改算子。"""

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
        root / "results/qa-wkv192fix-heads-full-20261102-v1/observation.pt", map_location="cpu", weights_only=True
    )
    x_cpu = data["native_output_stages"]["heads"].reshape(192, 8, 4096)
    x = x_cpu.to("npu")
    assert x.shape == (192, 8, 4096) and x.dtype == torch.bfloat16
    checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    key = "layers.4.attn.wo_a.weight"
    shard = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"][key]
    with safe_open(checkpoint / shard, framework="pt", device="cpu") as reader:
        weight_cpu = reader.get_tensor(key).reshape(8, 1024, 4096).transpose(1, 2).contiguous()
    weight = torch_npu.npu_format_cast(weight_cpu.to("npu"), 29)
    gold = data["native_output_stages"]["wo_a"].reshape(192, 8192)

    def native():
        return torch_npu.npu_transpose_batchmatmul(
            x,
            weight,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
            batch_split_factor=1,
        ).reshape(192, 8192)

    def exact(value, expected):
        return torch.equal(value.cpu().contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))

    before = native()
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
        observed = native()
        torch.npu.synchronize()
    after = native()
    torch.npu.synchronize()
    result = {
        "native_saved_equal": exact(before, gold),
        "profiled_equal": exact(observed, gold),
        "after_equal": exact(after, gold),
        "input_readonly": exact(x, x_cpu),
        "weight_readonly": exact(weight, weight_cpu),
        "physical_device": os.environ["TASK_DEVICE"],
        "scope": "isolated Native WO-A; profiler observation only; not performance measurement",
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert all(v for v in result.values() if isinstance(v, bool))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
