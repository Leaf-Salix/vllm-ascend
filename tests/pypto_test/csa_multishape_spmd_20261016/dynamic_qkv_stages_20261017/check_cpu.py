# SPDX-License-Identifier: Apache-2.0
"""验证源级动态 Q_A/QR helper：数学 AST、行覆盖及独立阶段 CPU 编译。

输入为产物目录和阶段选择，输出 JSON 与编译器 dump。阶段入口只有测试包装
负责 spmd，正式 helper 不派发任务；本脚本不执行设备、不测精度或性能。
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


def source_audit(repo):
    original = ast.parse((repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/qkv_proj_rope.py").read_text())
    target = ast.parse((repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/qkv_stages.py").read_text())
    old = next(node for node in original.body if isinstance(node, ast.FunctionDef)
               and node.name == "q_proj_qr_normalize")
    new = next(node for node in target.body if isinstance(node, ast.FunctionDef) and node.name == "q_r_stage")
    # 从同一份 FLASH 定义读取模型字段，只解释常量 AST，不 import 原 nz_mode 探针。
    config = ast.parse((repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/config.py").read_text())
    flash = next(node.value for node in config.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "FLASH")
    model = {keyword.arg: ast.literal_eval(keyword.value) for keyword in flash.keywords
             if keyword.arg in ("hidden_size", "q_lora_rank", "rms_norm_eps")}

    def resolve_constants(tree):
        assignments = {node.targets[0].id: node.value for node in tree.body if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name)}

        def value(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                return value(assignments[node.id])
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "M":
                return model[node.attr]
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
                return value(node.left) // value(node.right)
            raise AssertionError(ast.dump(node))

        return {name: value(assignments[name]) for name in
                ("D", "Q_LORA", "EPS", "Q_LORA_TILE", "QR_DENSE_M_TILE", "QR_FIXED_SMALL_M_TILE",
                 "QR_FIXED_SMALL_ROWS", "QR_FIXED_M_GROUPS", "QR_M_TILE", "QR_N_TILE", "QR_N_BLOCKS",
                 "QR_K_TILE", "QUANT_TILE", "T_TILE")}

    constants = resolve_constants(original)
    assert constants == resolve_constants(target)
    forbidden = [ast.unparse(node.func) for node in ast.walk(target) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) in ("pl.spmd", "pl.at", "pl.system.submit", "pl.system.task_dummy")]
    assert not forbidden

    class Remap(ast.NodeTransformer):
        def visit_Name(self, node):
            names = {"qr_fp32": "qa_fp32", "qr_scale_pad_store": "qr_scale_pad", "out_tg": "tg",
                     "qr_view": "qr", "qr_scale_view": "qr_scale"}
            return ast.copy_location(ast.Name(id=names.get(node.id, node.id), ctx=node.ctx), node)

    def loop(function, name):
        return next(node for node in ast.walk(function) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == name)

    matching = []
    for name in ("qr_rms_col0", "qa"):
        previous = Remap().visit(copy.deepcopy(loop(old, name)))
        current = loop(new, name)
        assert ast.dump(previous, include_attributes=False) == ast.dump(current, include_attributes=False), name
        matching.append(name)
    names = ("qr_sq_sum", "qr_amax_g", "qr_inv_rms", "qr_inv_rms_t",
             "qr_amax_floor", "qr_tile_amax", "qr_scale_quant_row", "qr_scale_quant_t", "qr_tile_scale_dq")
    for name in names:
        def assignment(function):
            return next(node for node in ast.walk(function) if isinstance(node, ast.Assign)
                        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)
        assert ast.dump(assignment(old), include_attributes=False) == ast.dump(assignment(new), include_attributes=False)
    old_qa = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "_q_proj_qa_nd")
    new_qa = next(node for node in target.body if isinstance(node, ast.FunctionDef) and node.name == "_q_a_worker")

    class QAMap(ast.NodeTransformer):
        def __init__(self, tail):
            self.names = {"QR_SPLIT_K_TILE": "D", "x_view": "x", "q_a_col0": "column"}
            self.names.update({"db": "k_block", "qr_d0": "k0", "qr_rows": "valid_rows",
                               "tile_rows": "tokens", "t0": "row", "x_t0": "row",
                               "q_x_chunk_bf16": "tail_x", "w_chunk": "tail_weight", "q_acc": "tail_acc"}
                              if tail else {"dense_k": "k_block", "dense_d0": "k0", "dense_x0": "row",
                                            "dense_w": "dense_weight"})

        def visit_Name(self, node):
            if node.id == "qr_k_base":
                return ast.copy_location(ast.Constant(value=0), node)
            return ast.copy_location(ast.Name(id=self.names.get(node.id, node.id), ctx=node.ctx), node)

        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.Add) and isinstance(node.left, ast.Constant) and node.left.value == 0:
                return node.right
            return node

    target_qa_loops = [node for node in ast.walk(new_qa) if isinstance(node, ast.For)
                       and isinstance(node.target, ast.Name) and node.target.id == "k_block"]
    for index, name in enumerate(("dense_k", "db")):
        before = copy.deepcopy(loop(old_qa, name))
        # tile_base 固定为本 helper 的整段输入零偏移；删除旧 chunk 地址的别名赋值。
        before.body = [node for node in before.body if not (isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "x_t0")]
        before = QAMap(index == 1).visit(before)
        assert ast.dump(before, include_attributes=False) == ast.dump(target_qa_loops[index], include_attributes=False), (
            name, ast.unparse(before), ast.unparse(target_qa_loops[index]))
    coverage = []
    for workers in (8, 16, 24):
        for tokens in range(385):
            dense = 32 if tokens < 128 else 64
            groups = min(3, max(1, tokens // dense))
            full = tokens // dense * dense
            padded = (tokens + 15) // 16 * 16
            visited = Counter()
            for worker in range(workers):
                for unit in range(worker, 8 * groups, workers):
                    column, group = unit % 8, unit // 8
                    for row in range(group * dense, full, groups * dense):
                        assert row + dense <= tokens
                        visited.update((r, column) for r in range(row, row + dense))
                    for row in range(full + group * 16, padded, groups * 16):
                        assert 0 <= row < tokens and row + 16 <= padded
                        visited.update((r, column) for r in range(row, min(row + 16, tokens)))
            assert len(visited) == tokens * 8 and all(value == 1 for value in visited.values())
        coverage.append({"aic_workers": workers, "token_range_inclusive": [0, 384], "covered_once": True})
    for tokens in range(385):
        tiles = (tokens + 7) // 8
        workers = min(tiles, 16)
        rows = []
        for worker in range(16):
            if worker < workers:
                for tile in range(worker, tiles, workers):
                    rows.extend(range(tile * 8, min(tile * 8 + 8, tokens)))
        assert sorted(rows) == list(range(tokens))
    return {"no_nested_dispatch": True, "qr_mathematical_loops_ast_equal": matching,
            "source_constant_values_equal": constants,
            "qr_arithmetic_assignments_ast_equal": names, "qa_coverage": coverage,
            "qa_fixed_k_dense_and_tail_pipeline_ast_equal": True,
            "qr_tokens_0_through_384_covered_once": True,
            "numerical_device_equivalence_proven": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("qa", "qr", "both"), default="both")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[4]
    report = {"device_execution": False, "performance_measured": False, "stages": {},
              "weight_nz_mode": os.environ.get("VLLM_ASCEND_ENABLE_NZ")}
    try:
        report["source_audit"] = source_audit(repo)
        import dsv4_csa_env
        dsv4_csa_env.activate()
        import pypto.language as pl
        from pypto import passes
        from pypto.ir.printer import python_print
        from pypto.runtime import RunConfig
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.qkv_stages import (
            BF16_WEIGHT_LAYOUT, MATMUL_ROWS_DYN, QA_ROWS_DYN, SCALE_ROWS_DYN, T_DYN, q_a_stage, q_r_stage,
        )
        report["original_nz_probe_module_imported"] = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa.nz_mode" in sys.modules
        assert not report["original_nz_probe_module_imported"]
        report["bf16_weight_layout"] = "NZ" if BF16_WEIGHT_LAYOUT == pl.NZ else "ND"

        @pl.jit(auto_scope=False)
        def qa_entry(
            x: pl.Tensor[[T_DYN, 4096], pl.BF16],
            wq_a: pl.Tensor[[1024, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
            qa: pl.Out[pl.Tensor[[QA_ROWS_DYN, 1024], pl.FP32]],
        ):
            with pl.spmd(24, name_hint="dynamic_qa_stage"):
                pl.set_cache_policy(wq_a, pl.CachePolicy.BYPASS)
                worker = pl.tile.get_block_idx()
                qa = q_a_stage(x, wq_a, qa, worker, 24)
            return qa

        @pl.jit(auto_scope=False)
        def qr_entry(
            qa: pl.Tensor[[QA_ROWS_DYN, 1024], pl.FP32],
            gamma: pl.Tensor[[1024], pl.BF16],
            qr: pl.Out[pl.Tensor[[T_DYN, 1024], pl.INT8]],
            scale: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
            qr_pad: pl.Out[pl.Tensor[[MATMUL_ROWS_DYN, 1024], pl.INT8]],
            scale_pad: pl.Out[pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32]],
        ):
            with pl.spmd(16, name_hint="dynamic_qr_stage"):
                worker = pl.tile.get_block_idx()
                qr, scale, qr_pad, scale_pad = q_r_stage(qa, gamma, qr, scale, qr_pad, scale_pad, worker, 16)
            return qr, scale, qr_pad, scale_pad

        for name, entry in (("qa", qa_entry), ("qr", qr_entry)):
            if args.stage not in ("both", name):
                continue
            record = {"specialized": False, "full_codegen_passed": False}
            report["stages"][name] = record
            try:
                directory = args.output / name
                directory.mkdir()
                source = python_print(entry.specialize())
                (directory / "specialized.py").write_text(source)
                assert 'pl.dynamic("CSA_QKV_T_DYN")' in source
                record["specialized"] = True
                artifact = entry.compile(config=RunConfig(
                    platform="a2a3", save_kernels=True, save_kernels_dir=str(directory / "kernel"),
                    dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO,
                ))
                record.update(full_codegen_passed=True, artifact_output=str(artifact.output_dir))
            except BaseException as error:
                record.update(error=str(error), traceback=traceback.format_exc())
        report["all_passed"] = all(item["full_codegen_passed"] for item in report["stages"].values())
    except BaseException as error:
        report.update(all_passed=False, error=str(error), traceback=traceback.format_exc())
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report.get("all_passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
