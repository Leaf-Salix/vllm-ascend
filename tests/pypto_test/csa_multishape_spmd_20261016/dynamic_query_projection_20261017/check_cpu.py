# SPDX-License-Identifier: Apache-2.0
"""动态 query 分支的源码、覆盖与 CPU 编译证据；不执行 NPU。

三份独立测试包装只负责建立外层 SPMD，以检查 helper 不依赖冻结形状。
编译产物不能证明三阶段之间已正确同步，也不代表整层完成。
"""

import argparse
import ast
import copy
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

from extract_source import Names, build


def audit_source(repo):
    target_path = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/query_projection_stages.py"
    target = ast.parse(target_path.read_text())
    expected = ast.parse(build(repo))
    actual_funcs = {node.name: node for node in target.body if isinstance(node, ast.FunctionDef)}
    expected_funcs = {node.name: node for node in expected.body if isinstance(node, ast.FunctionDef)}
    for name in actual_funcs:
        actual_body, expected_body = copy.deepcopy(actual_funcs[name]), copy.deepcopy(expected_funcs[name])
        # Ruff 会调整多行 docstring 缩进；注释排版不属于数学/控制 AST。
        actual_body.body = actual_body.body[1:]
        expected_body.body = expected_body.body[1:]
        assert ast.dump(actual_body, include_attributes=False) == ast.dump(expected_body, include_attributes=False), name
    calls = [ast.unparse(node.func) for node in ast.walk(target) if isinstance(node, ast.Call)]
    assert not any(name in ("pl.spmd", "pl.at", "pl.system.submit") for name in calls)
    # 独立检查两个 head 数学循环：包括乘法次序、规约、cast、gather 和尾块公式。
    original = ast.parse((repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/qkv_proj_rope.py").read_text())
    old_dequant = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "q_proj_q_dequant")
    for name in ("h_inner", "h_inner_tail"):
        previous = next(node for node in ast.walk(old_dequant) if isinstance(node, ast.For)
                        and isinstance(node.target, ast.Name) and node.target.id == name)
        current = next(node for node in ast.walk(actual_funcs["q_dequant_rope_stage"]) if isinstance(node, ast.For)
                       and isinstance(node.target, ast.Name) and node.target.id == name)
        previous = Names({"out_tg": "tg"}).visit(copy.deepcopy(previous))
        assert ast.dump(previous, include_attributes=False) == ast.dump(current, include_attributes=False), name

    # 覆盖检查只枚举数学索引；不构造 Tensor，不是按这些 T 生成设备二进制。
    cases = 0
    for tokens in range(385):
        head_tile = 64 if tokens >= 384 and tokens % 384 == 0 else (16 if tokens >= 96 and tokens % 96 == 0 else 4)
        row_tiles = (tokens + 7) // 8
        visits = Counter()
        for worker in range(48):
            for work in range(worker, row_tiles * (64 // head_tile), 48):
                row0 = (work // (64 // head_tile)) * 8
                head0 = (work % (64 // head_tile)) * head_tile
                active_heads = head_tile if row0 + 8 <= tokens else 4
                if row0 + 8 > tokens:
                    assert head_tile == 4
                for row in range(row0, min(row0 + 8, tokens)):
                    visits.update((row, head) for head in range(head0, head0 + active_heads))
        assert len(visits) == tokens * 64 and all(count == 1 for count in visits.values())
        active_rope = min(row_tiles, 48)
        rope_tiles = [tile for worker in range(active_rope) for tile in range(worker, row_tiles, active_rope)]
        assert sorted(rope_tiles) == list(range(row_tiles))
        for n_tile in (256, 512):
            n_blocks = 32768 // n_tile
            columns = [block for worker in range(20) for block in range(worker, n_blocks, 20)]
            assert sorted(columns) == list(range(n_blocks))
            padded = (tokens + 127) // 128 * 128
            for row in range(0, padded, 128):
                assert 0 < min(128, tokens - row) <= 128
        cases += 1
    return {"source_functions_equal_to_high_level_extraction": sorted(actual_funcs),
            "full_and_tail_head_math_ast_equal": True, "no_nested_dispatch": True,
            "token_range_inclusive": [0, 384], "coverage_cases": cases,
            "all_valid_rows_and_heads_covered_once": True, "device_numerical_equivalence_proven": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("qb", "dequant", "rope", "all"), default="all")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[4]
    report = {"device_execution": False, "performance_measured": False,
              "weight_nz_mode": os.environ.get("VLLM_ASCEND_ENABLE_NZ"), "stages": {}}
    try:
        report["source_audit"] = audit_source(repo)
        import dsv4_csa_env
        dsv4_csa_env.activate()
        import pypto.language as pl
        from pypto import passes
        from pypto.ir.printer import python_print
        from pypto.runtime import RunConfig
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.query_projection_stages import (
            PAD_ROWS_DYN, QPROJ_ROWS_DYN, QUANT_WEIGHT_LAYOUT, T_DYN,
            q_b_stage, q_dequant_rope_stage, q_rope_prepare_stage,
        )
        report["original_nz_probe_module_imported"] = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa.nz_mode" in sys.modules
        assert not report["original_nz_probe_module_imported"]

        @pl.jit(auto_scope=False)
        def qb_entry(
            wq_b: pl.Tensor[[1024, 32768], pl.INT8, QUANT_WEIGHT_LAYOUT],
            qr_live: pl.Tensor[[T_DYN, 1024], pl.INT8],
            qr_pad: pl.Tensor[[PAD_ROWS_DYN, 1024], pl.INT8],
            q_i32: pl.Out[pl.Tensor[[QPROJ_ROWS_DYN, 32768], pl.INT32]],
        ):
            tokens = pl.tensor.dim(qr_live, 0)
            with pl.spmd(20, name_hint="dynamic_qb_stage"):
                pl.set_cache_policy(wq_b, pl.CachePolicy.BYPASS)
                worker = pl.tile.get_block_idx()
                q_i32 = q_b_stage(wq_b, qr_pad, q_i32, tokens, worker, 20)
            return q_i32

        @pl.jit(auto_scope=False)
        def dequant_entry(
            scale_w: pl.Tensor[[32768], pl.FP32],
            cos: pl.Tensor[[T_DYN, 64], pl.FP32],
            sin: pl.Tensor[[T_DYN, 64], pl.FP32],
            swap: pl.Tensor[[T_DYN, 64], pl.INT32],
            q_flat: pl.Out[pl.Tensor[[T_DYN, 32768], pl.BF16]],
            scale: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
            q_i32: pl.Tensor[[QPROJ_ROWS_DYN, 32768], pl.INT32],
        ):
            with pl.spmd(48, name_hint="dynamic_q_dequant_rope"):
                worker = pl.tile.get_block_idx()
                q_flat = q_dequant_rope_stage(scale_w, cos, sin, swap, q_flat, scale, q_i32, worker, 48)
            return q_flat

        @pl.jit(auto_scope=False)
        def rope_entry(
            sin: pl.Tensor[[T_DYN, 64], pl.FP32],
            signed: pl.Out[pl.Tensor[[T_DYN, 64], pl.FP32]],
            swap: pl.Out[pl.Tensor[[T_DYN, 64], pl.INT32]],
        ):
            with pl.spmd(48, name_hint="dynamic_q_rope_prepare"):
                worker = pl.tile.get_block_idx()
                signed, swap = q_rope_prepare_stage(sin, signed, swap, worker, 48)
            return signed, swap

        for name, entry in (("qb", qb_entry), ("dequant", dequant_entry), ("rope", rope_entry)):
            if args.stage not in ("all", name):
                continue
            result = {"specialized": False, "full_codegen_passed": False}
            report["stages"][name] = result
            try:
                directory = args.output / name
                directory.mkdir()
                source = python_print(entry.specialize())
                assert 'pl.dynamic("CSA_QUERY_T_DYN")' in source
                (directory / "specialized.py").write_text(source)
                result["specialized"] = True
                artifact = entry.compile(config=RunConfig(
                    platform="a2a3", save_kernels=True, save_kernels_dir=str(directory / "kernel"),
                    dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO,
                ))
                result.update(full_codegen_passed=True, artifact_output=str(artifact.output_dir))
            except BaseException as error:
                result.update(error=str(error), traceback=traceback.format_exc())
        report["all_passed"] = all(stage["full_codegen_passed"] for stage in report["stages"].values())
    except BaseException as error:
        report.update(all_passed=False, error=str(error), traceback=traceback.format_exc())
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report.get("all_passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
