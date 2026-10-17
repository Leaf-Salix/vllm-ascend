# SPDX-License-Identifier: Apache-2.0
"""动态 KV 阶段的源式比较、分核覆盖与完整 CPU lowering；不运行设备。

原数学只用AST读取，不导入包含nz_mode设备探针的旧模块。测试包装独立建立
SPMD，仅证明两个正式inline helper可以按动态签名编译，不证明跨阶段同步。
"""

import argparse
import ast
import copy
import json
import sys
import traceback
from collections import Counter
from pathlib import Path


class Names(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "kv_k_base":
            return ast.copy_location(ast.Constant(value=0), node)
        return ast.copy_location(
            ast.Name(
                id={"x_view": "x", "KV_SPLIT_K_TILE": "D", "kv_view": "kv", "out_tg": "tg"}.get(node.id, node.id),
                ctx=node.ctx,
            ),
            node,
        )

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Add) and isinstance(node.left, ast.Constant) and node.left.value == 0:
            return node.right
        return node


def audit_source(repo):
    original_path = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/qkv_proj_rope.py"
    target_path = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/kv_stages.py"
    original, target = [ast.parse(p.read_text()) for p in (original_path, target_path)]
    old_funcs = {n.name: n for n in original.body if isinstance(n, ast.FunctionDef)}
    new_funcs = {n.name: n for n in target.body if isinstance(n, ast.FunctionDef)}

    def loop(function, name):
        return next(
            n
            for n in ast.walk(function)
            if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == name
        )

    for name in ("dense_k", "db"):
        before = Names().visit(copy.deepcopy(loop(old_funcs["_kv_project"], name)))
        after = loop(new_funcs["_kv_project_worker"], name)
        assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False), name

    def branch(function):
        return next(
            n
            for n in ast.walk(function)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "valid_rows == KV_RMS_T_TILE"
        )

    before = Names().visit(copy.deepcopy(branch(old_funcs["kv_proj_rope"])))
    after = branch(new_funcs["kv_norm_rope_stage"])
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)

    # 独立核对所有涉及数学的常量；固定K模式解释旧源码的IfExp为非atomic分支。
    config = ast.parse((repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/config.py").read_text())
    flash = next(
        n.value
        for n in config.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "FLASH"
    )
    model = {
        k.arg: ast.literal_eval(k.value)
        for k in flash.keywords
        if k.arg in ("hidden_size", "head_dim", "qk_rope_head_dim", "rms_norm_eps")
    }
    model["nope_head_dim"] = model["head_dim"] - model["qk_rope_head_dim"]

    def constants(tree):
        values = {
            n.targets[0].id: n.value
            for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        }

        def value(n):
            if isinstance(n, ast.Constant):
                return n.value
            if isinstance(n, ast.Name):
                if n.id == "ATOMIC_ADD":
                    return 0
                return value(values[n.id])
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "M":
                return model[n.attr]
            if isinstance(n, ast.IfExp):
                return value(n.body if value(n.test) else n.orelse)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.FloorDiv):
                return value(n.left) // value(n.right)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "float":
                return float(value(n.args[0]))
            raise AssertionError(ast.dump(n))

        keys = (
            "D",
            "HEAD_DIM",
            "ROPE_DIM",
            "ROPE_DIM_SCALE",
            "NOPE_DIM",
            "EPS",
            "KV_M_TILE",
            "KV_DENSE_M_TILE",
            "KV_FIXED_SMALL_M_TILE",
            "KV_FIXED_SMALL_ROWS",
            "KV_N_TILE",
            "KV_K_TILE",
            "KV_OM",
            "KV_RMS_T_TILE",
            "KV_TILE",
        )
        return {key: value(values[key]) for key in keys}

    expected_constants = constants(original)
    assert constants(target) == expected_constants
    qproj = ast.parse((repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/q_projection.py").read_text())
    for tree in (qproj, target):
        assignment = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "PREFILL_DENSE_TILE"
        )
        assert ast.literal_eval(assignment.value) == 512
    forbidden = {
        "pl.spmd",
        "pl.at",
        "pl.system.submit",
        "pl.system.task_dummy",
        "pl.system.signal_notify",
        "pl.system.signal_wait",
        "pl.system.syncall",
    }
    calls = {ast.unparse(n.func) for n in ast.walk(target) if isinstance(n, ast.Call)}
    assert not forbidden.intersection(calls)
    assert not any(isinstance(n, ast.ImportFrom) and "nz_mode" in (n.module or "") for n in target.body)

    # 从新的工作划分枚举所有真实行/列块，并与完整笛卡尔集合比较；边界不止七点。
    token_cases = list(range(385)) + [511, 512, 513, 543, 544, 575, 576, 639, 640, 767, 768, 769, 1023, 1024, 1025]
    coverage_count = 0
    for workers in (1, 3, 8, 24):
        for tokens in token_cases:
            visited, storage = Counter(), Counter()
            padded = (tokens + 15) // 16 * 16
            for worker in range(workers):
                for base in range(0, tokens, 512):
                    rows = min(512, tokens - base)
                    dense = 32 if rows < 128 else 64
                    groups = min(3, max(1, rows // dense))
                    full, local_pad = rows // dense * dense, (rows + 15) // 16 * 16
                    for logical in range(worker, 4 * groups, workers):
                        col, group = logical // groups, logical % groups
                        for row in range(group * dense, full, groups * dense):
                            assert base + row + dense <= tokens
                            visited.update((r, col) for r in range(base + row, base + row + dense))
                            storage.update((r, col) for r in range(base + row, base + row + dense))
                        for row in range(full + group * 16, local_pad, groups * 16):
                            valid = min(16, rows - row)
                            assert 0 < valid <= 16 and base + row + 16 <= padded
                            visited.update((r, col) for r in range(base + row, base + row + valid))
                            storage.update((r, col) for r in range(base + row, base + row + 16))
            assert len(visited) == tokens * 4 and all(v == 1 for v in visited.values())
            assert len(storage) == padded * 4 and all(v == 1 for v in storage.values())
            coverage_count += 1
    norm_count = 0
    for workers in (1, 3, 24, 48):
        for tokens in token_cases:
            visited = Counter()
            for worker in range(workers):
                for tile in range(worker, (tokens + 31) // 32, workers):
                    row = tile * 32
                    valid = min(32, tokens - row)
                    assert valid > 0 and row + valid <= tokens
                    # 与旧512行分段的full/tail判断逐块相同，不能跨过该边界。
                    base = row // 512 * 512
                    old_valid = min(32, min(512, tokens - base) - (row - base))
                    assert valid == old_valid
                    visited.update(range(row, row + valid))
            assert len(visited) == tokens and all(v == 1 for v in visited.values())
            norm_count += 1
    return {
        "source": str(original_path),
        "fixed_k_pipelines_ast_equal": ["dense_k", "db"],
        "full_tail_norm_rope_entire_branch_ast_equal": True,
        "arithmetic_constant_values_equal": expected_constants,
        "source_prefill_chunk_rows_equal": 512,
        "no_nested_dispatch_or_sync": True,
        "no_nz_mode_import": True,
        "explicit_changes": [
            "固定KV_OK=1，拒绝atomic模式；删除旧SPMD/task依赖和GM reshape。",
            "各逻辑M/N单元映射到显式worker条带循环。",
            "投影输出从chunk局部行改成tile_base+行，使用全T pad16工作区。",
            "norm按全T的32行tile条带分核；512是32的倍数，完整/尾块分支不变。",
        ],
        "projection_coverage_cases": coverage_count,
        "norm_coverage_cases": norm_count,
        "tokens": {"continuous": [0, 384], "additional_chunk_boundaries": token_cases[385:]},
        "all_valid_rows_and_columns_once": True,
        "projection_pad16_storage_once": True,
        "device_numerical_equivalence_proven": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("project", "norm", "both", "audit"), default="both")
    parser.add_argument("--audit-report", type=Path, help="复用同一数学源码已完成的CPU审计，避免重复枚举")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[4]
    report = {"device_execution": False, "performance_measured": False, "stages": {}}
    try:
        (args.output / "kv_stages.py").write_text(
            (repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/kv_stages.py").read_text()
        )
        (args.output / "check_cpu.py").write_text(Path(__file__).read_text())
        if args.audit_report is None:
            report["source_audit"] = audit_source(repo)
        else:
            prior = json.loads(args.audit_report.read_text())
            assert prior["all_passed"]
            previous_source = ast.parse((args.audit_report.parent / "kv_stages.py").read_text())
            current_source = ast.parse((args.output / "kv_stages.py").read_text())
            assert ast.dump(previous_source, include_attributes=False) == ast.dump(
                current_source, include_attributes=False
            )
            report["source_audit"] = prior["source_audit"]
            report["source_audit_reused_from"] = str(args.audit_report.resolve())
        if args.stage == "audit":
            report["all_passed"] = True
            return
        import dsv4_csa_env

        dsv4_csa_env.activate()
        import pypto.language as pl
        from pypto import passes
        from pypto.ir.printer import python_print
        from pypto.runtime import RunConfig

        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.kv_stages import (
            PAD_ROWS_DYN,
            T_DYN,
            kv_norm_rope_stage,
            kv_project_stage,
        )

        report["original_nz_probe_module_imported"] = (
            "vllm_ascend.ops.pypto.deepseek_v4_flash_csa.nz_mode" in sys.modules
        )
        assert not report["original_nz_probe_module_imported"]

        @pl.jit(auto_scope=False)
        def project_entry(
            x: pl.Tensor[[T_DYN, 4096], pl.BF16],
            wkv: pl.Tensor[[4096, 512], pl.BF16],
            kv_fp32: pl.Out[pl.Tensor[[PAD_ROWS_DYN, 512], pl.FP32]],
        ):
            with pl.spmd(24, name_hint="dynamic_kv_project"):
                pl.set_cache_policy(wkv, pl.CachePolicy.BYPASS)
                worker = pl.tile.get_block_idx()
                kv_fp32 = kv_project_stage(x, wkv, kv_fp32, worker, 24)
            return kv_fp32

        @pl.jit(auto_scope=False)
        def norm_entry(
            kv_fp32: pl.Tensor[[PAD_ROWS_DYN, 512], pl.FP32],
            gamma: pl.Tensor[[512], pl.BF16],
            cos: pl.Tensor[[T_DYN, 64], pl.FP32],
            signed_sin: pl.Tensor[[T_DYN, 64], pl.FP32],
            swap: pl.Tensor[[T_DYN, 64], pl.INT32],
            kv: pl.Out[pl.Tensor[[T_DYN, 512], pl.BF16]],
        ):
            with pl.spmd(24, name_hint="dynamic_kv_norm_rope"):
                worker = pl.tile.get_block_idx()
                kv = kv_norm_rope_stage(kv_fp32, gamma, cos, signed_sin, swap, kv, worker, 24)
            return kv

        for name, entry in (("project", project_entry), ("norm", norm_entry)):
            if args.stage not in ("both", name):
                continue
            record = {"specialized": False, "full_codegen_passed": False}
            report["stages"][name] = record
            try:
                directory = args.output / name
                directory.mkdir()
                logical = python_print(entry.specialize())
                (directory / "specialized.py").write_text(logical)
                assert 'pl.dynamic("CSA_KV_T_DYN")' in logical
                assert 'pl.dynamic("CSA_KV_PAD_ROWS_DYN")' in logical
                record["specialized"] = True
                artifact = entry.compile(
                    config=RunConfig(
                        platform="a2a3",
                        save_kernels=True,
                        save_kernels_dir=str(directory / "kernel"),
                        dump_passes=True,
                        memory_planner=passes.MemoryPlanner.PYPTO,
                    )
                )
                output = Path(artifact.output_dir)
                metadata = json.loads((output / "compiled_meta.json").read_text())
                assert any(-1 in param["shape"] for param in metadata["params"])
                record.update(
                    full_codegen_passed=True, artifact_output=str(output), dynamic_signature=metadata["params"]
                )
            except BaseException as error:
                record.update(error=str(error), traceback=traceback.format_exc())
        report["all_passed"] = bool(report["stages"]) and all(
            s["full_codegen_passed"] for s in report["stages"].values()
        )
    except BaseException as error:
        report.update(all_passed=False, error=str(error), traceback=traceback.format_exc())
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report.get("all_passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
