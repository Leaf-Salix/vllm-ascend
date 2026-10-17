# SPDX-License-Identifier: Apache-2.0
"""Attention计划/逆RoPE动态签名完整CPU后端探针，不执行设备或绑定样例shape。"""

import argparse
import json
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[4]
    package = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    report = {"passed": False, "sample_shapes_supplied": False, "device_execution": False}
    try:
        for path in Path(__file__).resolve().parent.glob("*.py"):
            (output / path.name).write_text(path.read_text())
        for name in ("attention_plan_stages.py", "request_geometry.py"):
            (output / name).write_text((package / name).read_text())
        import dsv4_csa_env

        dsv4_csa_env.activate()
        import pypto.language as pl
        from pypto import passes
        from pypto.ir.printer import python_print
        from pypto.runtime import RunConfig

        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.attention_plan_stages import (
            COMPRESSED_COLUMNS_DYN,
            ORIGINAL_COLUMNS_DYN,
            PLAN_CAP_DYN,
            QUERY_BOUNDS_DYN,
            REQUESTS_DYN,
            ROPE_CAP_DYN,
            T_DYN,
            attention_inverse_rope_sign_stage,
            attention_slot_plan_stage,
        )

        assert "vllm_ascend.ops.pypto.deepseek_v4_flash_csa.nz_mode" not in sys.modules
        ORIGINAL_PAGES = pl.dynamic("CSA_ATTN_PROBE_ORIGINAL_PAGES")
        COMPRESSED_PAGES = pl.dynamic("CSA_ATTN_PROBE_COMPRESSED_PAGES")

        @pl.jit(auto_scope=False)
        def entry(
            topk: pl.Tensor[[T_DYN, 512], pl.INT32],
            positions: pl.Tensor[[T_DYN], pl.INT64],
            bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
            lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
            original_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
            compressed_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
            original_cache: pl.Tensor[[ORIGINAL_PAGES, 32, 1, 512], pl.BF16],
            compressed_cache: pl.Tensor[[COMPRESSED_PAGES, 32, 1, 512], pl.BF16],
            indices: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, 512], pl.INT32]],
            bias: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, 640], pl.FP32]],
            mask: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32]],
            sin: pl.Tensor[[T_DYN, 64], pl.FP32],
            signed_sin: pl.InOut[pl.Tensor[[ROPE_CAP_DYN, 64], pl.FP32]],
        ):
            with pl.spmd(24, name_hint="dynamic_attention_plan_and_inverse_rope"):
                worker = pl.tile.get_block_idx()
                original_pages = pl.tensor.dim(original_cache, 0)
                compressed_pages = pl.tensor.dim(compressed_cache, 0)
                indices, bias, mask = attention_slot_plan_stage(
                    topk,
                    positions,
                    bounds,
                    lengths,
                    original_table,
                    compressed_table,
                    indices,
                    bias,
                    mask,
                    original_pages,
                    compressed_pages,
                    worker,
                    24,
                )
                signed_sin = attention_inverse_rope_sign_stage(sin, signed_sin, worker, 24)
            return indices, bias, mask, signed_sin

        (output / "symbolic_entry.py").write_text(python_print(entry.specialize()))
        report["symbolic_parse_passed"] = True
        artifact = entry.compile(
            config=RunConfig(
                platform="a2a3",
                save_kernels=True,
                save_kernels_dir=str(output / "kernel"),
                dump_passes=True,
                memory_planner=passes.MemoryPlanner.PYPTO,
            )
        )
        report["artifact_output"] = str(artifact.output_dir)
        report["compiled_meta"] = json.loads((Path(artifact.output_dir) / "compiled_meta.json").read_text())
        report["passed"] = True
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
