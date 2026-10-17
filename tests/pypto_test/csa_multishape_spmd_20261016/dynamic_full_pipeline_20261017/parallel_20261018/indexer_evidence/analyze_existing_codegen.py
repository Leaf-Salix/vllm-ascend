# SPDX-License-Identifier: Apache-2.0
"""只读既有两档生成物，整理策略、活跃tile地址和UB上界；不运行编译/设备。"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = Path(
    "/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
)
SIZE = {"f32": 4, "i32": 4, "ui32": 4, "f16": 2, "bf16": 2, "i16": 2, "i8": 1}
INTEREST = (
    "buf_half_scores",
    "buf_score_sums",
    "buf_score_row",
    "buf_kv_scale",
    "buf_scale_bytes",
    "buf_half_data",
    "ub_half_leaf_scores_raw",
    "ub_half_leaf_indices",
    "ub_half_pairs",
)


def inspect_pto(path):
    source = path.read_text()
    allocations = []
    for line in source.splitlines():
        match = re.search(
            r"(\S+) = pto.alloc_tile addr = %c(\d+)_i64.*!pto.tile_buf<loc=vec, dtype=(\w+), rows=(\d+), cols=(\d+)",
            line,
        )
        if match:
            name, address, dtype, rows, cols = match.groups()
            size = int(rows) * int(cols) * SIZE.get(dtype, 0)
            allocations.append(
                dict(
                    name=re.sub(r"_inline\d+", "_inlineN", name),
                    address=int(address),
                    rows=int(rows),
                    cols=int(cols),
                    bytes=size,
                    end=int(address) + size,
                )
            )
    return dict(
        pto=str(path),
        max_vec_end=max(x["end"] for x in allocations),
        indexer_tiles=[x for x in allocations if any(name in x["name"] for name in INTEREST)],
    )


def main():
    result = {}
    for batch in (16, 32):
        case = D / "parallel_20261018" / f"indexer_h8192_b{batch}_v1"
        report = json.loads((case / "result.json").read_text())
        matrix = json.loads((case / "source_matrix/result.json").read_text())
        item = dict(
            batch=batch,
            tokens=6 * batch,
            groups=3 * batch,
            group_waves=(3 * batch + 23) // 24,
            policy=report["policy"],
            statistics_us=matrix["statistics_us"],
        )
        for label in ("spmd_candidate", "spmd_reference"):
            kernel = Path(matrix["kernel_configs"][label]).parent
            item[label] = inspect_pto(kernel / "ptoas/dynamic_csa_full.pto")
        result[f"b{batch}"] = item
    (HERE / "existing_codegen_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, item in result.items():
        print(name, "policy", item["policy"], "waves", item["group_waves"])
        for label in ("spmd_candidate", "spmd_reference"):
            detail = item[label]
            print(label, "max_vec_end", detail["max_vec_end"], "selected_tiles", detail["indexer_tiles"][:10])


if __name__ == "__main__":
    main()
