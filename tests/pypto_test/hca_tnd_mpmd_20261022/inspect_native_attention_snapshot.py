# SPDX-License-Identifier: Apache-2.0
"""只在CPU检查已截取的Native attention输入，不重跑异常kernel。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    data = torch.load(args.snapshot, map_location="cpu", weights_only=True)

    def describe(value, name):
        if isinstance(value, torch.Tensor):
            item = {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype),
                    "contiguous": value.is_contiguous(), "storage_offset": value.storage_offset()}
            if name in ("cu_seqlens_q", "seqused_kv"):
                item["values"] = value.tolist()
            if name == "metadata":
                item["first_64_values"] = value.reshape(-1)[:64].tolist()
            return item
        if isinstance(value, (tuple, list)):
            return [describe(v, name) for v in value]
        if isinstance(value, dict):
            return {k: describe(v, k) for k, v in value.items()}
        return value

    report = describe(data, "root")
    report["snapshot"] = str(args.snapshot.resolve())
    report["npu_execution"] = False
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
