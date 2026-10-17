# SPDX-License-Identifier: Apache-2.0
"""O_A原K数学AST与量化表达式核对；仅整数所有权模型，不执行设备浮点或排序。"""

import ast
import copy
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/output_projection_a_stages.py"


def function(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def assigns(tree):
    return {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }


class MathOnly(ast.NodeTransformer):
    """仅剥离有效形状与Tile接口差异，不交换操作数、cast、加法或除法。"""

    def visit_Call(self, node):
        node = self.generic_visit(node)
        name = ast.unparse(node.func)
        if name == "pl.set_validshape":
            return node.args[0]
        if name == "pl.row_max":
            node.args = node.args[:1]  # Tile显式scratch不参与数学。
        if name == "pl.tile.full":
            node.func = ast.parse("pl.full", mode="eval").body
        return node


def norm(node):
    return ast.dump(MathOnly().visit(copy.deepcopy(node)), include_attributes=False)


def main():
    reference = ast.parse((HERE / "reference_decode_o_proj.py").read_text())
    actual = ast.parse(SOURCE.read_text())
    math = []
    for mode in ["nd", "nz"]:
        old = function(reference, "_proj_a_mm_" + mode)
        new = function(actual, "o_projection_a_stage_" + mode)
        old_assign, new_assign = assigns(old), assigns(new)
        for target in ["xa_first", "wa_first", "acc_a", "xa_k_chunk", "wa_k_chunk"]:
            assert norm(old_assign[target]) == norm(new_assign[target]), (mode, target)
            math.append(mode + ":" + target)
        old_loop = next(n for n in ast.walk(old) if isinstance(n, ast.For) and ast.unparse(n.target) == "kb")
        new_loop = next(n for n in ast.walk(new) if isinstance(n, ast.For) and ast.unparse(n.target) == "kb")
        assert norm(old_loop) == norm(new_loop), mode
        math.append(mode + ":原15次K256累加loop整体AST")
        old_seed = next(n.value for n in ast.walk(old) if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) == "acc_a" and isinstance(n.value, ast.Call)
                        and ast.unparse(n.value.func) == "pl.matmul")
        new_seed = next(n.value for n in ast.walk(new) if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) == "acc_a" and isinstance(n.value, ast.Call)
                        and ast.unparse(n.value.func) == "pl.matmul")
        assert norm(old_seed) == norm(new_seed), mode
        math.append(mode + ":首K256的matmul种子")
    old_quant = function(reference, "_decode_o_proj_tp1_tiled")
    new_quant = function(actual, "o_projection_group_quant_stage")
    old_assign, new_assign = assigns(old_quant), assigns(new_quant)
    for target in ["g_row_max", "g_amax_floor", "g_amax", "g_scale_num", "g_sq_row", "g_sq_col",
                   "oq_scaled", "oq_i32", "oq_half", "oq_i8", "zero_half", "zero_i8"]:
        assert norm(old_assign[target]) == norm(new_assign[target]), target
        math.append("quant:" + target)
    scale_expr = next(n.value for n in ast.walk(old_quant) if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Subscript)
                      and ast.unparse(n.targets[0].value) == "act_scale_dq")
    assert norm(scale_expr) == norm(new_assign["dequant_scale"])
    math.append("quant:dequant_scale仍amax*(1/127)")
    for target in ["g_amax_floor", "g_amax", "g_scale_num", "g_sq_row", "dequant_scale",
                   "oq_scaled", "oq_i32", "oq_half", "oq_i8"]:
        assert ast.unparse(new_assign[target].func) == "pl.set_validshape", target
    for target in ["oc_amax", "oc_q"]:
        assert any(kw.arg == "valid_shape" and ast.unparse(kw.value) == "[rows, O_LORA]"
                   for kw in new_assign[target].keywords)
    assert not any(isinstance(n, ast.Call) and ast.unparse(n.func) in ["pl.spmd", "pl.submit", "pl.system.notify"]
                   for n in ast.walk(actual))

    tokens_set = list(range(0, 34)) + [47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129,
                                      143, 144, 145, 167, 168, 169, 191, 192, 193, 255, 256, 257,
                                      383, 384, 385, 511, 512, 513, 769]
    cases = 0
    for t in tokens_set:
        for workers in [1, 24, 48]:
            for col_tile in [128, 256]:
                row_blocks = (t + 127) // 128
                columns = 1024 // col_tile
                blocks = Counter()
                for worker in range(workers):
                    for logical in range(worker, 8 * row_blocks * columns, workers):
                        group = logical // (row_blocks * columns)
                        row = (logical % (row_blocks * columns)) // columns
                        old_col = (logical % (row_blocks * columns)) % columns
                        col = logical % columns
                        assert old_col == col
                        r0, rows = row * 128, min(128, t - row * 128)
                        assert 0 < rows <= 128 and r0 + rows <= t
                        for pack_stride in [t, ((t + 15) // 16) * 16 + 17]:
                            assert group * pack_stride + r0 + rows <= (group + 1) * pack_stride
                        blocks[(group, row, col)] += 1
                assert len(blocks) == 8 * row_blocks * columns and all(v == 1 for v in blocks.values())
            for rb in [32, 96, 128]:
                task_rows, used = (t + 31) // 32, ((t + rb - 1) // rb) * rb
                writes, scales, reads, tails = Counter(), Counter(), Counter(), Counter()
                for worker in range(workers):
                    for logical in range(worker, 8 * task_rows, workers):
                        group, block = divmod(logical, task_rows)
                        first = block * 32
                        for qt in range(first, min(first + 32, t), 8):
                            rows = min(8, t - qt)
                            assert rows > 0 and qt + rows <= t
                            for row in range(qt, qt + rows):
                                writes[group, row] += 1
                                scales[group, row] += 1
                                reads[group, row] += 1
                        if first + 32 >= t:
                            tails[group] += 1
                            for zt in range(t, used, 8):
                                rows = min(8, used - zt)
                                for row in range(zt, zt + rows):
                                    writes[group, row] += 1
                assert len(reads) == len(scales) == 8 * t
                assert len(writes) == 8 * used
                assert all(v == 1 for v in writes.values()) and all(v == 1 for v in scales.values())
                assert len(tails) == (8 if t else 0) and all(v == 1 for v in tails.values())
                cases += 1
    result = {
        "status": "PASS", "source": str(SOURCE), "reference": str(HERE / "reference_decode_o_proj.py"),
        "math_ast_checks": math, "math_ast_check_count": len(math),
        "tail_structure": "两次FP32 load限制rows；归约输出/常数/div/cast/store有效形状明确；尾零无FP32读。",
        "geometry_case_count": cases, "tokens": tokens_set, "workers": [1, 24, 48],
        "a_column_tiles": [128, 256], "b_row_tiles": [32, 96, 128],
        "geometry_scope": ("OA全组/行/列唯一所有权、动态packed组stride；量化有效行唯一写、"
                           "每组唯一尾owner、INT8完整RB初始化、scale仅T行。"),
        "limits": ["数学AST比较剥离valid_shape与显式Tile scratch接口，不是设备浮点运行。",
                   "整数模型不验证NZ物理寻址或实际MTE/Vector有效掩码，须看后端并后续上卡。"],
        "device_execution": False,
    }
    (HERE / "source_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "math_ast_checks": len(math), "geometry_cases": cases}, ensure_ascii=False))


if __name__ == "__main__":
    main()
