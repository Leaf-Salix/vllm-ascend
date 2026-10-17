# SPDX-License-Identifier: Apache-2.0
"""仅核对新增 device count 的同源控制流，不重复 RoPE/cache 或运行设备。

独立枚举每请求真实关闭位置作为预期；同时核对只允许worker0写以及同一次
前缀扫描的offset。与本轮已执行的19个场景相同，保留此脚本方便后续复现。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from check_oracle import interpreted_source, metadata_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    helpers, _reads = interpreted_source()
    results = []
    for widths, histories in metadata_cases():
        bounds = np.array([0, *np.cumsum(widths)], dtype=np.int32)
        lengths = np.array([w + h if h >= 0 else 0 for w, h in zip(widths, histories)], dtype=np.int32)
        offsets = np.full(len(widths), -123, dtype=np.int32)
        count = np.array([-99], dtype=np.int32)
        helpers["compact_offsets_and_count_stage"](bounds, lengths, offsets, count, int(bounds[-1]), 1)
        assert (offsets == -123).all() and count[0] == -99
        helpers["compact_offsets_and_count_stage"](bounds, lengths, offsets, count, int(bounds[-1]), 0)
        prefix = 0
        for request, (width, history) in enumerate(zip(widths, histories)):
            expected_offset = prefix - history // 4 - 1 if history >= 0 and width > 0 else 0
            assert offsets[request] == expected_offset
            if history >= 0:
                prefix += sum((position + 1) % 4 == 0 for position in range(history, history + width))
        assert count[0] == prefix
        results.append({"query_counts": widths, "histories": histories, "count": int(count[0]), "passed": True})
    args.output.write_text(
        json.dumps({"passed": True, "device_execution": False, "cases": results}, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
