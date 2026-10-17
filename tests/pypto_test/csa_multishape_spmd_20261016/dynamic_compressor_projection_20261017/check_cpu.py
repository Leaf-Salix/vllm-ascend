# SPDX-License-Identifier: Apache-2.0
"""两路动态压缩投影的同源数学、分核覆盖与完整CPU后端证据。

原模块只读取AST，避免导入旧task编排或设备探针。生产文件不含spmd；测试
用一个独立SPMD包装顺序调用两路helper，验证两者可一起完整lowering，不运行设备。
"""

import argparse
import ast
import copy
import json
import operator
import sys
import traceback
from collections import Counter
from pathlib import Path


class Names(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self.mapping.get(node.id, node.id), ctx=node.ctx), node)


def named_functions(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def idx_loop(function):
    return next(n for n in ast.walk(function) if isinstance(n, ast.For) and ast.unparse(n.target) == "idx")


def constants(tree, model):
    assignments = {
        n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }

    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return value(assignments[node.id])
        if isinstance(node, ast.Attribute) and ast.unparse(node.value) == "M":
            return model[node.attr]
        if isinstance(node, ast.BinOp):
            operation = {ast.Add: operator.add, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}[type(node.op)]
            return operation(value(node.left), value(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            return value(node.left) == value(node.comparators[0])
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "int":
            return int(value(node.args[0]))
        raise AssertionError(ast.dump(node))

    return lambda name: value(assignments[name])


def audit(repo, output):
    old_root = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa"
    new_path = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/compressor_projection_stages.py"
    new = ast.parse(new_path.read_text())
    config = ast.parse((old_root / "config.py").read_text())
    flash = next(
        n.value
        for n in config.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "FLASH"
    )
    model = {
        k.arg: ast.literal_eval(k.value)
        for k in flash.keywords
        if k.arg in ("hidden_size", "head_dim", "index_head_dim")
    }
    new_constants = constants(new, model)
    new_functions = named_functions(new)
    records = {}
    cases = list(range(385)) + [511, 512, 513, 543, 544, 575, 576, 639, 640, 767, 768, 769, 1023, 1024, 1025]
    for label, old_file, old_name, new_name, names in (
        (
            "main",
            "decode_compressor_ratio4.py",
            "compressor_ratio4_project",
            "compressor_project_stage",
            {
                "OUT_DIM": "CMP_OUT_DIM",
                "MM_B_TILE": "CMP_M_TILE",
                "OUT_TILE": "CMP_N_TILE",
                "x_flat": "x",
                "cmp4_kv_proj_pad": "kv_proj_pad",
                "cmp4_score_proj_pad": "score_proj_pad",
                "kv_worker": "worker_id",
                "KV_SCORE_WORKERS": "WORKERS",
            },
        ),
        (
            "indexer",
            "decode_indexer_compressor.py",
            "indexer_compressor_project",
            "indexer_compressor_project_stage",
            {
                "OUT_DIM": "INDEXER_OUT_DIM",
                "MM_B_TILE": "INDEXER_M_TILE",
                "PROJ_OUT_TILE": "INDEXER_N_TILE",
                "x_flat": "x",
                "kv_worker": "worker_id",
                "KV_SCORE_WORKERS": "WORKERS",
            },
        ),
    ):
        old_path = old_root / old_file
        old = ast.parse(old_path.read_text())
        (output / old_file).write_text(old_path.read_text())
        before = named_functions(old)[old_name]
        after = new_functions[new_name]
        canonical = Names(names).visit(copy.deepcopy(idx_loop(before)))
        assert ast.dump(canonical, include_attributes=False) == ast.dump(idx_loop(after), include_attributes=False)
        # 行数表达式也独立比对，避免仅数学体相同却改成按pad容量领取工作。
        for variable in ("bs", "t_matmul"):

            def assignment(function, name):
                return next(n for n in function.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name)

            a = Names(names).visit(copy.deepcopy(assignment(before, variable)))
            assert ast.dump(a, include_attributes=False) == ast.dump(
                assignment(after, variable), include_attributes=False
            )
        old_constants = constants(old, model)
        n_name = "OUT_TILE" if label == "main" else "PROJ_OUT_TILE"
        dims = {}
        for key in ("D", "K_TILE", "MM_B_TILE", n_name, "OUT_DIM"):
            assert old_constants(key) == new_constants(names.get(key, key)), key
            dims[key] = old_constants(key)
        tile_m, tile_n, width = dims["MM_B_TILE"], dims[n_name], dims["OUT_DIM"]
        assert dims["D"] == 4096 and dims["K_TILE"] == 512
        coverage_count = 0
        extra_capacity_count = 0
        for workers in (1, 3, 8, 24):
            for tokens in cases:
                padded = (tokens + tile_m - 1) // tile_m * tile_m
                seen, windows = Counter(), Counter()
                max_store_row = 0
                for worker in range(workers):
                    for logical in range(worker, (padded // tile_m) * (width // tile_n), workers):
                        start = logical // (width // tile_n) * tile_m
                        column = logical % (width // tile_n)
                        valid = min(tile_m, tokens - start)
                        assert 0 < valid <= tile_m and start + valid <= tokens
                        assert start + tile_m <= padded
                        max_store_row = max(max_store_row, start + tile_m)
                        windows.update([(start // tile_m, column)])
                        seen.update((row, column) for row in range(start, start + valid))
                expected_windows = (padded // tile_m) * (width // tile_n)
                assert len(windows) == expected_windows and all(v == 1 for v in windows.values())
                assert len(seen) == tokens * (width // tile_n) and all(v == 1 for v in seen.values())
                assert max_store_row == padded
                # 实际pad可更大；枚举结果独立于容量，不制造额外有效token或读取padding。
                for capacity in (padded, padded + tile_m, padded + 7 * tile_m):
                    assert max_store_row <= capacity
                    extra_capacity_count += 1
                coverage_count += 1
        calls = [ast.unparse(n.func) for n in ast.walk(idx_loop(after)) if isinstance(n, ast.Call)]
        if label == "main":
            assert calls.count("pl.matmul") == 2 and calls.count("pl.matmul_acc") == 2
            initialization = "kb==0时两次pl.matmul；其余两次pl.matmul_acc，完整AST相同"
        else:
            assert calls.count("pl.matmul") == 0 and calls.count("pl.matmul_acc") == 2
            initialization = "两次pl.matmul_acc(init_cond=kb==0)，完整AST相同"
        records[label] = {
            "source": str(old_path),
            "source_function": old_name,
            "target_function": new_name,
            "entire_idx_loop_ast_equal_after_names": names,
            "dynamic_T_and_ceil_M_expressions_ast_equal": True,
            "arithmetic_constants_equal": dims,
            "accumulator_initialization": initialization,
            "geometry_cases": coverage_count,
            "capacity_cases": extra_capacity_count,
            "every_valid_row_column_block_exactly_once": True,
            "logical_store_views_nonoverlapping_and_inside_pad": True,
            "padding_values_initialized_or_valid": False,
        }
    forbidden = {"pl.spmd", "pl.at", "pl.system.submit", "pl.system.syncall", "pl.system.task_dummy"}
    calls = {ast.unparse(n.func) for n in ast.walk(new) if isinstance(n, ast.Call)}
    assert not forbidden.intersection(calls)
    assert not any("signal_" in x or "notify" in x or "wait" in x for x in calls)
    assert not any(isinstance(n, ast.ImportFrom) and "nz_mode" in (n.module or "") for n in new.body)
    assert not {"DECODE_BATCH", "DECODE_SEQ", "BS_PAD"}.intersection(
        n.id for n in ast.walk(new) if isinstance(n, ast.Name)
    )
    return {
        "stages": records,
        "tokens": {"continuous": [0, 384], "additional_boundaries": cases[385:]},
        "worker_counts": [1, 3, 8, 24],
        "no_dispatch_sync_or_shape_fixture": True,
        "numerical_device_equivalence_proven": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[4]
    report = {"device_execution": False, "performance_measured": False, "sample_shapes_supplied": False}
    try:
        (output / "compressor_projection_stages.py").write_text(
            (repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/compressor_projection_stages.py").read_text()
        )
        (output / "check_cpu.py").write_text(Path(__file__).read_text())
        report["source_audit"] = audit(repo, output)
        import dsv4_csa_env

        dsv4_csa_env.activate()
        import pypto.language as pl
        from pypto import passes
        from pypto.ir.printer import python_print
        from pypto.runtime import RunConfig

        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.compressor_projection_stages import (
            INDEXER_PAD_ROWS_DYN,
            MAIN_PAD_ROWS_DYN,
            T_DYN,
            compressor_project_stage,
            indexer_compressor_project_stage,
        )

        report["original_nz_probe_module_imported"] = (
            "vllm_ascend.ops.pypto.deepseek_v4_flash_csa.nz_mode" in sys.modules
        )
        assert not report["original_nz_probe_module_imported"]

        @pl.jit(auto_scope=False)
        def combined_entry(
            x: pl.Tensor[[T_DYN, 4096], pl.BF16],
            main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
            main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
            indexer_wkv: pl.Tensor[[256, 4096], pl.BF16],
            indexer_wgate: pl.Tensor[[256, 4096], pl.BF16],
            main_value: pl.Out[pl.Tensor[[MAIN_PAD_ROWS_DYN, 1024], pl.FP32]],
            main_score: pl.Out[pl.Tensor[[MAIN_PAD_ROWS_DYN, 1024], pl.FP32]],
            indexer_value: pl.Out[pl.Tensor[[INDEXER_PAD_ROWS_DYN, 256], pl.FP32]],
            indexer_score: pl.Out[pl.Tensor[[INDEXER_PAD_ROWS_DYN, 256], pl.FP32]],
        ):
            with pl.spmd(24, name_hint="dynamic_compressor_both_projections"):
                worker = pl.tile.get_block_idx()
                main_value, main_score = compressor_project_stage(
                    x, main_wkv, main_wgate, main_value, main_score, worker, 24
                )
                indexer_value, indexer_score = indexer_compressor_project_stage(
                    x, indexer_wkv, indexer_wgate, indexer_value, indexer_score, worker, 24
                )
            return main_value, main_score, indexer_value, indexer_score

        logical = python_print(combined_entry.specialize())
        (output / "specialized.py").write_text(logical)
        for symbol in (
            "CSA_COMPRESSOR_PROJECTION_T",
            "CSA_COMPRESSOR_PROJECTION_MAIN_PAD_ROWS",
            "CSA_COMPRESSOR_PROJECTION_INDEXER_PAD_ROWS",
        ):
            assert f'pl.dynamic("{symbol}")' in logical
        report["parse_passed"] = True
        artifact = combined_entry.compile(
            config=RunConfig(
                platform="a2a3",
                save_kernels=True,
                save_kernels_dir=str(output / "kernel"),
                dump_passes=True,
                memory_planner=passes.MemoryPlanner.PYPTO,
            )
        )
        artifact_output = Path(artifact.output_dir)
        metadata = json.loads((artifact_output / "compiled_meta.json").read_text())
        params = metadata["params"]
        dynamic = [p for p in params if -1 in p["shape"]]
        assert len(dynamic) == 5, params
        assert [p["shape"] for p in dynamic].count([-1, 1024]) == 2
        assert [p["shape"] for p in dynamic].count([-1, 256]) == 2
        assert [p["shape"] for p in dynamic].count([-1, 4096]) == 1
        report.update(
            full_codegen_passed=True,
            artifact_output=str(artifact_output),
            dynamic_signature=params,
            all_passed=True,
        )
    except BaseException as error:
        report.update(all_passed=False, error=str(error), traceback=traceback.format_exc())
    finally:
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report.get("all_passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
