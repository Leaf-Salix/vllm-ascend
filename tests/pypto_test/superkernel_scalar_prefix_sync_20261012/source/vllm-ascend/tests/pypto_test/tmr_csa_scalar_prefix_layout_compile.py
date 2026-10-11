# SPDX-License-Identifier: Apache-2.0
"""Compile a frozen CSA with three manually audited Scalar-prefix changes."""

import argparse
import difflib
import json
import shutil
from pathlib import Path

import tmr_csa_full_128k_b16_compile as demo

PACKAGE = "deepseek_v4_flash_csa_scalar_prefix_layout"


def replace_once(text, old, new):
    assert text.count(old) == 1, f"frozen source changed: {old[:100]!r}"
    return text.replace(old, new, 1)


def prepare_package(output):
    """Move only named arithmetic; this is a fixture, not a compiler transform."""
    root = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/pypto"
    source = root / demo.PACKAGES["mixed"]
    target = root / PACKAGE
    replacements = {}

    path = source / "decode_o_proj.py"
    original = path.read_text()
    begin = original.index("def _proj_b_mm_nz_kernel(")
    end = original.index("\n\n@pl.jit.inline", begin)
    body = original[begin:end]
    body = replace_once(
        body,
        "    d0 = pl.tile.get_block_idx() * PROJ_B_D_TILE\n    pl.system.wait_pre_task_end()",
        "    d0 = pl.tile.get_block_idx() * PROJ_B_D_TILE\n"
        "    # Scalar arguments only; no GM read or pipeline state before Wait.\n"
        "    weight_group_k0 = pl.max(g, 0) * O_LORA\n"
        "    partial_group_n0 = g * D\n"
        "    pl.system.wait_pre_task_end()",
    )
    body = replace_once(body, "wk0 = pl.max(g, 0) * O_LORA + k0", "wk0 = weight_group_k0 + k0")
    body = replace_once(body, "weight_k0 = pl.max(g, 0) * O_LORA", "weight_k0 = weight_group_k0")
    assert body.count("g * D + n0") == 3
    body = body.replace("g * D + n0", "partial_group_n0 + n0")
    replacements[path.name] = original[:begin] + body + original[end:]

    path = source / "qkv_proj_rope.py"
    original = path.read_text()
    replacements[path.name] = replace_once(
        original,
        "        pl.system.wait_pre_task_end()\n"
        "        for dq_work in pl.range(\n"
        "            dq_worker,\n"
        "            ((tile_rows + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE) * (H // dq_head_tile),\n"
        "            Q_DEQUANT_WORKERS,\n"
        "        ):\n"
        "            hg = (dq_work % (H // dq_head_tile)) * dq_head_tile\n"
        "            tg = (dq_work // (H // dq_head_tile)) * Q_ROPE_T_TILE",
        "        # These bounds depend only on Scalar task arguments.\n"
        "        dq_head_groups = H // dq_head_tile\n"
        "        dq_work_count = ((tile_rows + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE) * dq_head_groups\n"
        "        pl.system.wait_pre_task_end()\n"
        "        for dq_work in pl.range(dq_worker, dq_work_count, Q_DEQUANT_WORKERS):\n"
        "            hg = (dq_work % dq_head_groups) * dq_head_tile\n"
        "            tg = (dq_work // dq_head_groups) * Q_ROPE_T_TILE",
    )

    path = source / "decode_sparse_attn_csa.py"
    original = path.read_text()
    moved = replace_once(
        original,
        "        qk_core = pl.tile.get_block_idx()\n"
        "        pl.system.wait_pre_task_end()\n"
        "        pl.system.set_ffts(ffts_workspace)\n",
        "        qk_core = pl.tile.get_block_idx()\n",
    )
    replacements[path.name] = replace_once(
        moved,
        "        qk_block_count = qk_query_count * SPARSE_BLOCKS\n",
        "        qk_block_count = qk_query_count * SPARSE_BLOCKS\n"
        "        # Keep MIX FIFO/FFTS initialization after the full local/fanin wait.\n"
        "        pl.system.wait_pre_task_end()\n"
        "        pl.system.set_ffts(ffts_workspace)\n",
    )

    if target.exists():
        for name, expected in replacements.items():
            assert (target / name).read_text() == expected, f"private package changed: {name}"
    else:
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name, updated in replacements.items():
            (target / name).write_text(updated)
    patch = "".join(
        "".join(difflib.unified_diff(
            (source / name).read_text().splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{name}", tofile=f"b/{name}",
        ))
        for name, updated in replacements.items()
    )
    (output / "operator_changes.diff").write_text(patch)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    package = prepare_package(args.output)
    reused = {}
    # Reuse only ordinary PTO. Recompile both boundary paths with the fixed
    # InsertSync pass: old CSA artifacts emitted pipe primes before some Waits.
    for name, source_name in (("baseline", "baseline"),):
        source = (args.reuse / source_name).resolve(strict=True)
        target = args.output / name
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)
        assert target.resolve() == source
        reused[name] = str(source)
    (args.output / "reused_artifacts.json").write_text(json.dumps(reused, indent=2) + "\n")
    (args.output / "package.txt").write_text(str(package) + "\n")
    demo.PACKAGES["current"] = demo.PACKAGES["mixed"]
    demo.PACKAGES["prefix_layout"] = PACKAGE
    demo.compile_variant("current", args.output / "current")
    demo.compile_variant("prefix_layout", args.output / "prefix_layout")


if __name__ == "__main__":
    main()
