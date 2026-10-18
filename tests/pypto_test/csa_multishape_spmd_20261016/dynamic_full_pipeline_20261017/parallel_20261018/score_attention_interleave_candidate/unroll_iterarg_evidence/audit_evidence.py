# SPDX-License-Identifier: Apache-2.0
"""只读既有小pass文件与冻结v2 IR；不重编、不读622MB result、不计算hash。"""

import ast
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = Path("/data/pyptouser/qinchuanyu/pto-eager")
FULL = WORKSPACE / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_interleave_candidate"
DEPENDENCY = WORKSPACE / ".worktrees/pypto-csa-static-fixes-20261016"


def main():
    parsed = ast.parse((HERE / "scope_kernels.py").read_text())
    functions = {node.name: node for node in parsed.body if isinstance(node, ast.FunctionDef)}
    direct = copy.deepcopy(functions["direct_kernel"])
    alias = copy.deepcopy(functions["alias_kernel"])
    alias.name = direct.name

    class RemoveBridge(ast.NodeTransformer):
        def visit_AnnAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id in ("step_arg", "live_arg"):
                assert ast.dump(node.annotation) == ast.dump(ast.parse("pl.Scalar[pl.INDEX]", mode="eval").body)
                assert isinstance(node.value, ast.Name)
                assert node.value.id == {"step_arg": "step_iter", "live_arg": "live_iter"}[node.target.id]
                return None
            return self.generic_visit(node)

        def visit_Name(self, node):
            node.id = {"step_arg": "step_iter", "live_arg": "live_iter"}.get(node.id, node.id)
            return node

    inverse_ast_equal = ast.dump(direct) == ast.dump(RemoveBridge().visit(alias))
    assert inverse_ast_equal
    report = {"inverse_ast_equal": inverse_ast_equal, "fresh_scalar_bridges": 2,
              "whole_csa_compiled": False, "npu_execution": False, "full_v2_passes": []}
    for name in ("00_frontend.py", "01_after_InlineFunctions.py", "02_after_UnrollLoops.py",
                 "03_after_CtrlFlowTransform.py", "04_after_ConvertToSSA.py"):
        path = FULL / "cpu_long_v2/kernel/passes_dump" / name
        text = path.read_text()
        lines = text.splitlines()
        first = next((i for i, line in enumerate(lines) if "__FREE_VAR" in line), None)
        item = {"path": str(path), "bytes": path.stat().st_size, "free_var_count": text.count("__FREE_VAR")}
        if first is not None:
            excerpt = HERE / ("full_v2_" + name.replace(".py", "_first_free.txt"))
            lo, hi = max(0, first - 2), min(len(lines), first + 3)
            excerpt.write_text("\n".join(f"{i+1}: {lines[i]}" for i in range(lo, hi)) + "\n")
            item.update(first_free_line=first + 1, excerpt=str(excerpt))
        report["full_v2_passes"].append(item)
    report["large_files_unmodified"] = [{"path": str(p), "bytes": p.stat().st_size} for p in (
        FULL / "cpu_long_v2/result.json", FULL / "cpu_long_v2_from_repo.log")]
    excerpts = (
        ("src/ir/transforms/utils/deep_clone_utils.cpp", 92, 115),
        ("src/ir/transforms/unroll_loops_pass.cpp", 88, 99),
        ("python/pypto/language/parser/ast_parser.py", 2753, 2761),
    )
    parts = []
    for relative, first, last in excerpts:
        path = DEPENDENCY / relative
        lines = path.read_text().splitlines()
        parts.append(str(path) + f":{first}\n" + "\n".join(f"{i+1}: {lines[i]}" for i in range(first - 1, last)))
    (HERE / "DEPENDENCY_EXCERPTS.txt").write_text("\n\n".join(parts) + "\n")
    report["minimal_result"] = str(HERE / "cpu_v2/result.json")
    report["minimal_hypothesis_confirmed"] = json.loads((HERE / "cpu_v2/result.json").read_text())["hypothesis_confirmed"]
    assert report["minimal_hypothesis_confirmed"]
    (HERE / "AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"inverse_ast_equal": inverse_ast_equal, "minimal_hypothesis_confirmed": True,
                      "full_v2_free_counts": [p["free_var_count"] for p in report["full_v2_passes"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
