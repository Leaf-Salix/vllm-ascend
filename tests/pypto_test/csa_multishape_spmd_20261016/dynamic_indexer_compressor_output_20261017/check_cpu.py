# SPDX-License-Identifier: Apache-2.0
"""Indexer compact norm/RoPE与Hadamard的动态签名CPU编译；不执行设备。"""

import argparse
import json
import sys
import traceback
from pathlib import Path

from check_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("both", "norm", "hadamard"), default="both")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[4]
    source = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    report = {"device_execution": False, "sample_shapes_supplied": False, "stages": {}}
    try:
        for name in ("indexer_compressor_output_stages.py", "metadata_stages.py", "request_geometry.py"):
            (output / name).write_text((source / name).read_text())
        report["audit"] = audit(repo)
        import dsv4_csa_env

        dsv4_csa_env.activate()
        import pypto.language as pl
        from pypto import passes
        from pypto.ir.printer import python_print
        from pypto.runtime import RunConfig

        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_compressor_output_stages import (
            COMPACT_PAD_ROWS_DYN,
            COMPACT_ROWS_DYN,
            POOL_ROWS_DYN,
            QUERY_BOUNDS_DYN,
            REQUESTS_DYN,
            T_DYN,
            indexer_compressor_hadamard_stage,
            indexer_compressor_norm_rope_stage,
        )

        assert "vllm_ascend.ops.pypto.deepseek_v4_flash_csa.nz_mode" not in sys.modules

        @pl.jit(auto_scope=False)
        def norm_entry(
            pooled: pl.Tensor[[POOL_ROWS_DYN, 128], pl.FP32],
            gamma: pl.Tensor[[128], pl.BF16],
            cos: pl.Tensor[[COMPACT_ROWS_DYN, 64], pl.FP32],
            sin: pl.Tensor[[COMPACT_ROWS_DYN, 64], pl.FP32],
            positions: pl.Tensor[[T_DYN], pl.INT64],
            bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
            lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
            offsets: pl.Tensor[[REQUESTS_DYN], pl.INT32],
            normed: pl.InOut[pl.Tensor[[COMPACT_PAD_ROWS_DYN, 128], pl.BF16]],
        ):
            with pl.spmd(24, name_hint="dynamic_indexer_compact_norm_rope"):
                worker = pl.tile.get_block_idx()
                normed = indexer_compressor_norm_rope_stage(
                    pooled, gamma, cos, sin, positions, bounds, lengths, offsets, normed, worker, 24
                )
            return normed

        @pl.jit(auto_scope=False)
        def hadamard_entry(
            normed: pl.Tensor[[COMPACT_PAD_ROWS_DYN, 128], pl.BF16],
            hadamard: pl.Tensor[[128, 128], pl.BF16],
            valid_count: pl.Tensor[[1], pl.INT32],
            output: pl.Out[pl.Tensor[[COMPACT_PAD_ROWS_DYN, 128], pl.FP32]],
        ):
            with pl.spmd(24, name_hint="dynamic_indexer_compact_hadamard"):
                worker = pl.tile.get_block_idx()
                count = pl.cast(pl.read(valid_count, [0]), pl.INDEX)
                output = indexer_compressor_hadamard_stage(normed, hadamard, output, count, worker, 24)
            return output

        for name, entry in (("norm", norm_entry), ("hadamard", hadamard_entry)):
            if args.stage not in ("both", name):
                continue
            record = {"parse_passed": False, "full_codegen_passed": False}
            report["stages"][name] = record
            try:
                directory = output / name
                directory.mkdir()
                (directory / "specialized.py").write_text(python_print(entry.specialize()))
                record["parse_passed"] = True
                artifact = entry.compile(
                    config=RunConfig(
                        platform="a2a3",
                        save_kernels=True,
                        save_kernels_dir=str(directory / "kernel"),
                        dump_passes=True,
                        memory_planner=passes.MemoryPlanner.PYPTO,
                    )
                )
                meta = json.loads((Path(artifact.output_dir) / "compiled_meta.json").read_text())
                assert any(-1 in p["shape"] for p in meta["params"])
                record.update(full_codegen_passed=True, artifact_output=str(artifact.output_dir), params=meta["params"])
            except BaseException as error:
                record.update(error=str(error), traceback=traceback.format_exc())
        report["all_passed"] = all(r["full_codegen_passed"] for r in report["stages"].values())
    except BaseException as error:
        report.update(all_passed=False, error=str(error), traceback=traceback.format_exc())
    finally:
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report.get("all_passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
