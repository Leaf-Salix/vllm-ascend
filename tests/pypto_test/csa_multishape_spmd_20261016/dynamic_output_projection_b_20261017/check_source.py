# SPDX-License-Identifier: Apache-2.0
"""输出后段的同源数学AST和有界分核/尾块审计，不导入PyPTO或构造设备Tensor。

仅归一化明确的Tile种子、单行常量与等价load语法；不展开或交换乘加。
独立整数模型枚举ND/NZ的group/row/D块所有权和实际M容量，证明索引覆盖，
不把这种CPU审计当作A3数值、内存模型或性能证据。
"""

import ast
import copy
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/output_projection_b_stages.py"
OLD = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa"


def functions(path):
    return {node.name:node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def plain_body(function):
    return [n for n in copy.deepcopy(function.body)
            if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant) or not isinstance(n.value.value, str)]


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id in ("ROWS", "valid_rows", "PROJ_B_ACT_T_TILE"):
            return ast.copy_location(ast.Constant(1), node)
        if node.id == "PROJ_B_ACT_N_TILE":
            return ast.copy_location(ast.Name(id="D", ctx=ast.Load()), node)
        if node.id == "ob_n0":
            return ast.copy_location(ast.Constant(0), node)
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Add) and isinstance(node.right, ast.Constant) and node.right.value == 0:
            return node.left
        return node

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.value, ast.Name) and node.value.id == "partials"
                and isinstance(node.ctx, ast.Load)
                and isinstance(node.slice, ast.Tuple)
                and isinstance(node.slice.elts[0], ast.Slice)
                and ast.unparse(node.slice.elts[0].lower) == "b_tb"):
            return ast.copy_location(ast.parse("pl.load(partials, [b_tb, p_col0], [1, D])", mode="eval").body, node)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.func) in ("pl.create_tile", "pl.create_tensor"):
            node.func = ast.parse("pl.create_tensor", mode="eval").body
            node.keywords = [kw for kw in node.keywords if kw.arg != "target_memory"]
        return node


def normalized(node):
    return ast.dump(Normalize().visit(copy.deepcopy(node)), include_attributes=False)


def main():
    source = functions(SOURCE)
    old = functions(OLD / "decode_o_proj.py")
    hc = functions(OLD / "hc_post.py")
    old_nd = next(n for n in ast.walk(old["_proj_b_mm_nd"]) if isinstance(n, ast.For) and ast.unparse(n.target) == "nf")
    new_nd = next(n for n in ast.walk(source["_o_b_nd_stage"]) if isinstance(n, ast.For) and ast.unparse(n.target) == "nf")
    assert ast.dump(old_nd, include_attributes=False) == ast.dump(new_nd, include_attributes=False), "ND nf/K数学不一致"
    old_nz = next(n for n in old["_proj_b_mm_nz_kernel"].body if isinstance(n, ast.If))
    new_nz = next(n for n in ast.walk(source["_o_b_nz_stage"]) if isinstance(n, ast.If))
    assert ast.dump(old_nz, include_attributes=False) == ast.dump(new_nz, include_attributes=False), "NZ两类循环树不一致"
    old_hc = ast.Module(body=plain_body(hc["hc_post_block"]), type_ignores=[])
    new_hc = ast.Module(body=plain_body(source["hc_post_row_block"]), type_ignores=[])
    assert normalized(old_hc) == normalized(new_hc), "HC单行数学不一致"
    old_dequant = old["_decode_o_proj_tp1_tiled"]
    new_dequant = source["o_dequant_hc_post_stage"]
    old_group = next(n for n in ast.walk(old_dequant) if isinstance(n, ast.For) and ast.unparse(n.target) == "act_g")
    new_group = next(n for n in ast.walk(new_dequant) if isinstance(n, ast.For) and ast.unparse(n.target) == "act_g")
    assert normalized(old_group) == normalized(new_group), "8组左折叠不一致"
    for target in ("out_t", "out_bf16"):
        def select(function):
            return sorted([n for n in ast.walk(function) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == target], key=lambda n:n.lineno)
        assert [normalized(n) for n in select(old_dequant)] == [normalized(n) for n in select(new_dequant)]
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(n, ast.Call)]
    assert not any(name in ("pl.spmd", "pl.at", "pl.tile.get_block_idx", "pl.system.submit") for name in calls)
    cases = []
    for tokens in (0, 1, 31, 32, 33, 95, 96, 97, 127, 128, 129, 169, 384):
        for rb in (32, 96, 128):
            rows = (tokens + rb - 1) // rb
            used = rows * rb
            # P_MM可大于实际所需容量；循环不得扫描额外capacity。
            mm_capacity, scale_capacity = used + 2 * rb, tokens + 3
            for workers in (1, 24):
                nd, nz = Counter(), Counter()
                for worker in range(workers):
                    for work in range(worker, 8 * rows * 8, workers):
                        group, unit = divmod(work, rows * 8)
                        tb, dc = divmod(unit, 8)
                        nd[(group, tb, dc)] += 1
                        assert (tb + 1) * rb <= used <= mm_capacity
                    for work in range(worker, 8 * 8, workers):
                        group, dc = divmod(work, 8)
                        for tb in range(rows):
                            nz[(group, tb, dc)] += 1
                            assert (tb + 1) * rb <= used <= mm_capacity
                expected = {(g,tb,dc) for g in range(8) for tb in range(rows) for dc in range(8)}
                assert set(nd) == set(nz) == expected and all(v == 1 for v in list(nd.values()) + list(nz.values()))
                dequant = [t for w in range(48) for t in range(w, tokens, 48)]
                assert sorted(dequant) == list(range(tokens)) and tokens <= scale_capacity
                # 有效行与pad分区互斥且覆盖每个O_B整M读取；末尾容量不读。
                assert set(range(tokens)).isdisjoint(range(tokens, used))
                assert set(range(tokens)) | set(range(tokens, used)) == set(range(used))
                cases.append(dict(tokens=tokens, row_tile=rb, workers=workers, mm_capacity=mm_capacity,
                                  scale_capacity=scale_capacity, mm_rows_used=used, pad_rows_required=used-tokens,
                                  owner_once=True, extra_capacity_unused=True))
    report = {"passed": True, "source": str(SOURCE.relative_to(ROOT)), "canonical": [str((OLD/name).relative_to(ROOT)) for name in ("decode_o_proj.py", "hc_post.py")],
              "nd_math_ast_exact": True, "nz_small_large_loop_ast_exact": True,
              "hc_single_row_normalized_ast_exact": True, "eight_group_fold_normalized_ast_exact": True,
              "weight_scale_and_bf16_boundary_ast_exact": True, "no_nested_dispatch": True,
              "normalizations": ["ND/NZ数学循环直接AST相等，没有归一化", "HC与dequant的ROWS/valid_rows/ACT_T=1，ACT_N=D、ob_n0=0", "同一partials单行切片改为显式load"],
              "scheduling_changes": ["8组任务合并成显式worker条带；ND保持group/row/D任务顺序，NZ保持group/D任务及内部循环树", "终端从旧task的2/4行连续任务改成单行worker轮转，门值与数学均仍逐token"],
              "coverage_cases": len(cases), "cases": cases,
              "device_execution": False, "cpu_numeric_execution": False,
              "limitations": ["不证明A3数值或设备内存可见性", "不能代替量化正确零填pad的证据", "需要外层按记录owner等待全部真实生产者", "输入输出alias、非法容量和非法ROW_TILE由入口拒绝"]}
    (HERE / "source_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"passed":True,"coverage_cases":len(cases),"device_execution":False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
