# SPDX-License-Identifier: Apache-2.0
"""仅读取safetensors头估算八卡常驻权重；不加载tensor、不初始化设备。"""

import argparse
import json
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    index = json.loads((args.model / "quant_model_weights.safetensors.index.json").read_text())
    config = json.loads((args.model / "config.json").read_text())
    experts = config["n_routed_experts"]
    assert experts == 256 and experts % 8 == 0
    shared, expert_bytes = 0, [0] * 8
    tensors = 0
    for shard in sorted(set(index["weight_map"].values())):
        with (args.model / shard).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            assert size < 64 << 20
            header = json.loads(stream.read(size))
        for name, row in header.items():
            if name == "__metadata__":
                continue
            assert index["weight_map"][name] == shard
            tensors += 1
            count = row["data_offsets"][1] - row["data_offsets"][0]
            if ".ffn.experts." in name:
                expert = int(name.split(".ffn.experts.", 1)[1].split(".", 1)[0])
                assert 0 <= expert < experts
                expert_bytes[expert // (experts // 8)] += count
            else:
                shared += count
    assert tensors == len(index["weight_map"])
    result = {
        "model": str(args.model),
        "DP": 8,
        "EP": 8,
        "tensors": tensors,
        "shared_checkpoint_bytes": shared,
        "expert_checkpoint_bytes_per_rank": expert_bytes,
        "estimated_weight_gib_per_rank": [(shared + value) / (1 << 30) for value in expert_bytes],
        "hbm_capacity_gib_per_device": 64,
        "scope": "checkpoint storage estimate only; excludes loader conversions, KV, graphs and runtime workspace",
        "npu_initialized": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
