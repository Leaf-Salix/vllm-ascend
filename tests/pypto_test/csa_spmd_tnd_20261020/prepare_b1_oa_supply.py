# SPDX-License-Identifier: Apache-2.0
"""B1-only O_A trial: M16, two K512 L1 windows, original ordered K128 MMAs.

The accepted B1 O_A still allocates M128 tiles with dynamic valid rows. Its
actual generated L0 chain contains 32 K128 products. This trial preserves that
chain and the eight column owners per output group, but loads K512 per L1
window. Only the accumulator crosses the K1024 loop; each iteration explicitly
loads two independent windows before consuming them. It does not use the
compiler-internal compact allocation flag: every valid M in this B1 range
rounds to the physical 16 rows.
"""

import ast
import difflib
import json
import shutil
from pathlib import Path


def consume(label, previous, first):
    lines = [f"oa_weight_{label} = pl.tile.reshape(oa_w_{label}, [512, A_COL_TILE])"]
    for k in range(4):
        lines.extend([
            f"oa_l_{label}_{k} = pl.tile.extract(oa_x_{label}, 0, {k * 128},",
            "    [16, 128], target_memory=pl.MemorySpace.Left)",
            f"oa_r_{label}_{k} = pl.tile.extract(oa_weight_{label}, {k * 128}, 0,",
            "    [128, A_COL_TILE], target_memory=pl.MemorySpace.Right)",
        ])
        # Prime two windows, issue their MMAs, then refill the released oldest
        # window while the following MMA may still be in flight. Do not put a
        # refill before the already-ready following MMA. K order is unchanged.
        for j in ([0, 1] if k == 1 else [k] if k > 1 else []):
            old = previous if j == 0 else f"oa_acc_{label}_{j - 1}"
            init = "(oa_pair == 0)" if first and j == 0 else "False"
            lines.extend([
                f"oa_acc_{label}_{j} = pl.tile.matmul_acc(",
                f"    {old}, oa_l_{label}_{j}, oa_r_{label}_{j}, init_cond={init})",
            ])
    return "\n".join("            " + line for line in lines) + "\n"


def helper_source():
    source = '''# SPDX-License-Identifier: Apache-2.0
"""Small-M O_A supply trial; selected only for one request and at most16 tokens."""

import pypto.language as pl

from .output_projection_a_stages import MM_ROWS_DYN, PACK_TOTAL_ROWS_DYN


@pl.jit.inline(auto_scope=False)
def o_projection_a_small_supply(
    packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, 4096], pl.BF16],
    weight: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ],
    output: pl.InOut[pl.Tensor[[MM_ROWS_DYN, 8192], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    A_COL_TILE: pl.constexpr,
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    """Same group/N owner and K order; every load/store keeps actual M valid."""
    stride = pl.tensor.dim(packed, 0) // 8
    first_group = group_begin * (1024 // A_COL_TILE)
    limit = group_end * (1024 // A_COL_TILE)
    first = worker + ((first_group + WORKERS - 1 - worker) // WORKERS) * WORKERS
    for work in pl.range(pl.max(0, first), limit, WORKERS):
        group = pl.max(0, work // (1024 // A_COL_TILE))
        column = pl.max(0, work % (1024 // A_COL_TILE)) * A_COL_TILE
        row = group * stride
        oa_seed = pl.tile.create([16, A_COL_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
        oa_valid_seed = pl.set_validshape(oa_seed, tokens, A_COL_TILE)
        # Mat windows are local to each iteration. Only the Acc is loop-carried.
        for oa_pair, (oa_acc_iter,) in pl.range(4, init_values=(oa_valid_seed,)):
            k0 = oa_pair * 1024
            oa_x_first = pl.tile.load(packed, [row, k0], [16, 512], [tokens, 512],
                                      target_memory=pl.MemorySpace.Mat)
            oa_w_first = pl.tile.load(weight, [group, k0, column], [1, 512, A_COL_TILE],
                                      [1, 512, A_COL_TILE], target_memory=pl.MemorySpace.Mat, cache=1)
            oa_x_second = pl.tile.load(packed, [row, k0 + 512], [16, 512], [tokens, 512],
                                       target_memory=pl.MemorySpace.Mat)
            oa_w_second = pl.tile.load(weight, [group, k0 + 512, column], [1, 512, A_COL_TILE],
                                       [1, 512, A_COL_TILE], target_memory=pl.MemorySpace.Mat, cache=1)
'''
    source += consume("first", "oa_acc_iter", True)
    source += consume("second", "oa_acc_first_3", False)
    source += '''            oa_finished = pl.yield_(oa_acc_second_3)
        pl.store(oa_finished, [0, group * 1024 + column], output)
    return output
'''
    return source


def main():
    here = Path(__file__).resolve().parent
    source = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    destination = here / "runs/b1_oa_supply_native_candidate_v1"
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    changes = {}
    text = (destination / "policy.py").read_text()
    marker = "    split_score_rows: bool = False\n"
    assert text.count(marker) == 1
    text = text.replace(marker, marker + "    oa_small_supply: bool = False\n", 1)
    marker = '            "SPLIT_SCORE_ROWS": self.split_score_rows,\n'
    assert text.count(marker) == 1
    text = text.replace(marker, marker + '            "OA_SMALL_SUPPLY": self.oa_small_supply,\n', 1)
    marker = "        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,\n"
    assert text.count(marker) == 1
    changes["policy.py"] = text.replace(marker, marker + "        oa_small_supply=requests == 1 and tokens <= 16,\n", 1)
    text = (destination / "entry.py").read_text()
    marker = "    SPLIT_SCORE_ROWS: pl.constexpr = False,\n"
    assert text.count(marker) == 2
    text = text.replace(marker, marker + "    OA_SMALL_SUPPLY: pl.constexpr = False,\n")
    marker = "            SPLIT_SCORE_ROWS,\n"
    assert text.count(marker) == 2
    changes["entry.py"] = text.replace(marker, marker + "            OA_SMALL_SUPPLY,\n")
    text = (destination / "cube_schedule.py").read_text()
    text = text.replace("import pypto.language as pl\n", "import pypto.language as pl\n\n"
                        "from .oa_small_supply import o_projection_a_small_supply\n", 1)
    marker = "    SPLIT_SCORE_ROWS: pl.constexpr = False,\n"
    assert text.count(marker) == 1
    text = text.replace(marker, marker + "    OA_SMALL_SUPPLY: pl.constexpr = False,\n", 1)
    marker = """                o_r = o_projection_a_groups_stage_nz(
                    packed, wo_a, o_r, tokens, worker, GROUPS, OA_COLS, oa_group, oa_group + 1
                )"""
    assert text.count(marker) == 1
    selected = marker.replace("o_projection_a_groups_stage_nz", "o_projection_a_small_supply")
    indent = lambda value: "\n".join("    " + line for line in value.splitlines())
    changes["cube_schedule.py"] = text.replace(marker,
        "                if OA_SMALL_SUPPLY:\n" + indent(selected)
        + "\n                else:\n" + indent(marker), 1)
    changes["oa_small_supply.py"] = helper_source()
    patches = []
    for name, after in changes.items():
        path = destination / name
        before = path.read_text() if path.exists() else ""
        ast.parse(after)
        path.write_text(after)
        patches.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                           fromfile="a/" + name, tofile="b/" + name))
    (here / "b1_oa_supply_v1.patch").write_text("".join(patches))
    (here / "b1_oa_supply_v1.json").write_text(json.dumps({
        "candidate": str(destination), "adopted": False,
        "scope": "One request, tokens<=16; current decode contract limits query capacity to6",
        "preserved": "32 increasing K128 MMAs per output; original8column owners; all phase/quant/OB/math order",
        "nominal_resources": "Mat2*(16*512+512*128)*2=288KiB; Left2*16*128*2=8KiB;Right64KiB;Acc8KiB",
        "required_before_device": "Full compile, actual memory/events/order and disabled-path audit",
        "no_device_performance_claim": True,
    }, indent=2) + "\n")
    for batch in (1, 16):
        original = here / f"runs/cohort_prefix_a241_h131072_b{batch}_s6_v41/buffer_descriptors.json"
        description = json.loads(original.read_text())
        description["policy"]["OA_SMALL_SUPPLY"] = batch == 1
        (here / f"runs/b1_oa_supply_b{batch}_descriptors.json").write_text(json.dumps(description, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
