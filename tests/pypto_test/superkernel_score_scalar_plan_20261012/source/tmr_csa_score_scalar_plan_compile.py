# SPDX-License-Identifier: Apache-2.0
"""Reuse the existing orchestration maximum and isolate Score's early planning."""

import argparse
import difflib
import json
import shutil
from pathlib import Path

import tmr_csa_full_128k_b16_compile as demo
from tmr_csa_scalar_prefix_layout_compile import replace_once

PACKAGES = {
    "score_scalar": "deepseek_v4_flash_csa_score_scalar_plan",
    "score_early": "deepseek_v4_flash_csa_score_early_plan",
}


def prepare_package(variant, output):
    root = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/pypto"
    source = root / demo.PACKAGES["mixed"]
    target = root / PACKAGES[variant]
    original = (source / "decode_indexer.py").read_text()
    begin = original.index("def indexer_score_topk_native_cube(")
    end = original.index("\n\n@pl.jit.inline", begin)
    body = original[begin:end]
    body = replace_once(
        body,
        "    query_group_size: pl.constexpr,\n",
        "    max_topk_cache_len: pl.Scalar[pl.INDEX],\n    query_group_size: pl.constexpr,\n",
    )
    body = replace_once(
        body,
        "        buf_max_cache_len = 0\n"
        "        for buf_batch in pl.range(buf_query_count // S):\n"
        "            buf_max_cache_len = pl.max(buf_max_cache_len, "
        "pl.read(kv_seq_lens, [buf_batch]) // COMPRESS_RATIO)\n",
        "        # The entry already scans these immutable lengths for shape dispatch.\n"
        "        # Reuse that maximum: max(floor(length / R)) == floor(max(length) / R).\n"
        "        buf_max_cache_len = pl.max(max_topk_cache_len, 0)\n",
    )
    if variant == "score_early":
        body = replace_once(body, "        pl.system.wait_pre_task_end()\n", "")
        body = replace_once(
            body, "        pl.system.set_ffts(buf_score_ffts)\n",
            "        # All preceding planning uses Scalar arguments; GM/FFTS remain after Wait.\n"
            "        pl.system.wait_pre_task_end()\n"
            "        pl.system.set_ffts(buf_score_ffts)\n",
        )
    updated = original[:begin] + body + original[end:]
    forest_begin = updated.index("def indexer_score_topk_forest(")
    forest_end = updated.index("\n\n@pl.jit.inline", forest_begin)
    forest = updated[forest_begin:forest_end]
    old = "                    cache_write_tid,\n"
    assert forest.count(old) == 4, "expected four native-Cube strategy calls"
    forest = forest.replace(old, old + "                    max_topk_cache_len,\n")
    updated = updated[:forest_begin] + forest + updated[forest_end:]
    if target.exists():
        assert (target / "decode_indexer.py").read_text() == updated
    else:
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (target / "decode_indexer.py").write_text(updated)
    patch = "".join(difflib.unified_diff(
        original.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile="a/decode_indexer.py", tofile="b/decode_indexer.py",
    ))
    (output / f"{variant}_operator.diff").write_text(patch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reused = {}
    for name in ("baseline", "current"):
        source = (args.reuse / name).resolve(strict=True)
        target = args.output / name
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)
        assert target.resolve() == source
        reused[name] = str(source)
    (args.output / "reused_artifacts.json").write_text(json.dumps(reused, indent=2) + "\n")
    for variant, package in PACKAGES.items():
        prepare_package(variant, args.output)
        demo.PACKAGES[variant] = package
        demo.compile_variant(variant, args.output / variant)


if __name__ == "__main__":
    main()
