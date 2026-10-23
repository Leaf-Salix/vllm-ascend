# SPDX-License-Identifier: Apache-2.0
"""Freeze an HC input ownership pilot without changing production or root ABI."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "physical_capacity_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    source["hc_cooperative_stages.py"] = (here / "hc_cooperative_source.py").read_text()
    vector = source["vector_schedule.py"]
    vector = replace_once(
        vector,
        "from .hc_stages import hc_comb_range, hc_gates_range, hc_mix_norm_range, "
        "hc_reduce_range, hc_widen_rms_range\n",
        "from .hc_stages import hc_comb_range, hc_gates_range, hc_mix_norm_range, hc_reduce_range\n"
        "from .hc_cooperative_stages import hc_combine_rms, hc_widen_columns\n",
    )
    vector = replace_once(
        vector,
        "    x_fp32, inv = hc_widen_rms_range(x, x_fp32, inv, widen_tail, worker, VECTORS)\n",
        "    x_fp32, widen_tail = hc_widen_columns(x, x_fp32, widen_tail, worker, VECTORS)\n",
    )
    begin = vector.index("    if BALANCED and QUERY_GROUP == 2:\n", vector.index("    x_fp32, widen_tail"))
    end = vector.index("    # 公共metadata", begin)
    vector = (
        vector[:begin]
        + """    # INPUT now publishes column payloads and individual RMS chunks. All
    # writers finish before any RMS consumer waits; there is no dependency cycle.
    if worker < ((tokens + 7) // 8) * 4:
        signal = publish_phase(signal, PHASE_INPUT, core, CORES, generation)
    elif worker == pl.min((tokens + 7) // 8, VECTORS - 1):
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
    if worker < (tokens + 7) // 8:
        # Row-tile waves advance by 48: all use the same four input owners.
        signal = wait_row_owners(
            signal, PHASE_INPUT, CORES, generation, worker * 4, 4, 1, VECTORS, GROUPS, 1
        )
    inv = hc_combine_rms(widen_tail, inv, tokens, worker, VECTORS)

"""
        + vector[end:]
    )
    source["vector_schedule.py"] = vector
    cube = source["cube_schedule.py"]
    begin = cube.index("    if HC_ROW_DEPENDENCIES:\n")
    end = cube.index("    partials = hc_project_range(", begin)
    cube = (
        cube[:begin]
        + """    # Original AIC task=(M16 row tile, K4096 split). Its two M8 inputs now
    # have separate column owners. Later AIC waves advance both owners by 48,
    # hence the same published whole-owner signals cover every local wave.
    hc_tasks = ((tokens + 15) // 16) * 4
    if worker < hc_tasks:
        hc_row = (worker // 4) * 16
        hc_input_owner = ((worker // 4) * 8 + worker % 4) % VECTORS
        signal = wait_row_owners(
            signal, PHASE_INPUT, CORES, generation, hc_input_owner, 1, 1, VECTORS, GROUPS, 1
        )
        if hc_row + 8 < tokens:
            signal = wait_row_owners(
                signal, PHASE_INPUT, CORES, generation, hc_input_owner + 4, 1, 1, VECTORS, GROUPS, 1
            )
"""
        + cube[end:]
    )
    source["cube_schedule.py"] = cube
    for content in source.values():
        ast.parse(content)
    target = here / "runs/hc_cooperative_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        (target / name).write_text(content)
    report = dict(
        source=str(target.resolve()),
        parent="physical_capacity_production_v1.json",
        adopted=False,
        device_execution=False,
        scope="Isolated pilot for T<=192; RMS scratch capacity supports T<=1024",
        mechanism="Four K4096 AIV owners per M8 input tile; each AIC waits only its column producers",
        math="Unchanged K512 row sums, ordered 32-step +0 left fold, eps/high-precision rsqrt and downstream math",
        scratch="Existing widen_tail repurposed: [split, row_tile*128 + local_chunk*8]; 512B owner slots",
        costs=[
            "32 FP32 partial row sums per token written/read",
            "Four-owner RMS acquire per active M8 consumer",
            "Original metadata planner can now own a K4096 input slice and reach metadata later",
        ],
        changed_modules=["cube_schedule.py", "vector_schedule.py", "hc_cooperative_stages.py"],
        root_abi_changed=False,
        gate="Ownership/math simulation, complete CPU build and resource/store/dependency audit before task-submit",
        files={name: hashlib.sha256(content.encode()).hexdigest() for name, content in source.items()},
    )
    with (here / "hc_cooperative_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
