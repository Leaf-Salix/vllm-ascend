# SPDX-License-Identifier: Apache-2.0
"""Remove two HC tail-only GM round trips without changing numerical work."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "physical_capacity_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    hc = source["hc_stages.py"]
    begin = hc.index("                    widen_tail[0:T_TILE, w_d0 : w_d0 + RMS_K_TILE] = w_val\n")
    end = hc.index("                    pl.store(w_out, [w_t0, w_d0], x_hc32_flat)\n", begin)
    end += len("                    pl.store(w_out, [w_t0, w_d0], x_hc32_flat)\n")
    hc = hc[:begin] + "                    pl.store(w_valid, [w_t0, w_d0], x_hc32_flat)\n" + hc[end:]
    begin = hc.index("                    x_normed_tail_store[0:T_TILE, d0 : d0 + D_TILE] = normed_bf16\n")
    end = hc.index("                    pl.store(normed_valid, [t0, d0], x_normed)\n", begin)
    hc = (
        hc[:begin]
        + "                    normed_valid = pl.set_validshape(normed_bf16, valid_rows, D_TILE)\n"
        + hc[end:]
    )
    hc = replace_once(
        hc,
        "                    # 写实际 T 的 x_fp32 时借唯一尾块 scratch 恢复有效 store 边界。\n",
        "                    # 直接按有效行写出，省去尾块GM暂存和重读；平方和原序不变。\n",
    )
    ast.parse(hc)
    source["hc_stages.py"] = hc
    target = here / "runs/hc_direct_tail_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        (target / name).write_text(content)
    report = dict(
        source=str(target.resolve()),
        parent="physical_capacity_production_v1.json",
        adopted=False,
        device_execution=False,
        scope="C01 tail-row pilot; do not combine with head_partition2_v1",
        changed={"hc_stages.py": "Direct valid-row stores for widened FP32 input and normalized BF16 output"},
        unchanged="RMS K512/K256 reductions, fillpad, rounding, helper ownership, "
        "publication order and tensor signatures",
        removed_per_tail=dict(
            gm_store_sites=2,
            gm_load_sites=2,
            expanded_fp32_scratch_bytes=8 * 16384 * 4,
            normalized_bf16_scratch_bytes=8 * 4096 * 2,
        ),
        gate="Check real emitted valid-row stores and resource/ABI behavior, then strict complete L1 and tail guards",
        files={name: hashlib.sha256(content.encode()).hexdigest() for name, content in source.items()},
    )
    with (here / "hc_direct_tail_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
