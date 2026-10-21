# SPDX-License-Identifier: Apache-2.0
"""将完整七档与受影响边界已通过的精确Top-K迁入正式多task。"""

import ast
import json
import os
import shutil
import subprocess

from adopt_core_v2 import RUFF, TARGET, equivalent, format_part
from prepare import HERE, REPO

FILES = ("topk_threshold_stages.py", "decode_indexer.py")


def algorithm_source(source):
    """允许更新模块验证说明；函数体、常量和导入仍作完整AST比较。"""
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        if isinstance(tree.body[0].value.value, str):
            source = "".join(source.splitlines(keepends=True)[tree.body[0].end_lineno :])
    return source


def main():
    matrix = json.loads((HERE / "MATRIX_topk_prefix_v1_topk_v1.json").read_text())
    if not matrix["passed"] or len(matrix["rows"]) != 7:
        raise ValueError("七档未完整通过")
    for row in matrix["rows"]:
        if row["current"] != "plan_workers_v1":
            raise ValueError("采用参照必须是正式Plan")
        before, after = (row["l1_us"][key] for key in ("multitask_current", "multitask_candidate"))
        if row["history"] == 131072 and after["mean_us"] >= before["mean_us"]:
            raise ValueError("实际采用的长档并非全部受益")
        if after["max_us"] > before["max_us"]:
            raise ValueError("当前七档存在max回退，需重新审查采用决定")
    boundary = json.loads((HERE / "runs/topk_prefix_boundary_v1/h131071_b5_boundary_v1/result.json").read_text())
    if not boundary["passed"] or not boundary["boundary_replay"]["passed"]:
        raise ValueError("碎片页、尾块、同图分支/padding未通过")
    review = json.loads((HERE / "TOPK_SHORT_DISABLED_REVIEW.json").read_text())
    if not all(c["binary_equal"] and c["threshold_calls"] == [0, 0] for c in review["checks"]):
        raise ValueError("短档实际Score关闭路径尚未核对")
    paths = [str((TARGET / name).relative_to(REPO)) for name in FILES]
    if subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *paths], cwd=REPO).strip():
        raise ValueError("目标文件已有未提交修改，拒绝覆盖")
    if (TARGET / FILES[0]).exists():
        raise ValueError("正式Top-K helper已存在，应先核对状态")
    if not equivalent(
        (TARGET / "decode_indexer.py").read_text(),
        (HERE / "sources/plan_workers_v1/decode_indexer.py").read_text(),
    ):
        raise ValueError("正式源码已经偏离本轮Plan参照")
    originals = {name: (TARGET / name).read_bytes() if (TARGET / name).exists() else None for name in FILES}
    stage = HERE / "sources/adoption_topk_prefix_v2"
    stage.mkdir(parents=True, exist_ok=False)
    for name in FILES:
        shutil.copy2(HERE / "sources/topk_prefix_v1" / name, stage / name)
    helper = stage / FILES[0]
    source = helper.read_text()
    old = (
        "实现状态\n"
        "    依赖本轮独立 PyPTO 计数对齐修复及 PTOAS A3 Scalar compare lowering。\n"
        "    首版完整B16/B28精度通过但未达性能门槛；本版缩短gather容量，\n"
        "    需要重新验证。正式算子未改，不能沿用首版PASS作为本版证据。\n"
    )
    new = (
        "实现与验证\n"
        "    c49ee5b2短前缀迁入多task，七档S6及尾块/碎片页/同图分支边界通过。\n"
        "    数学和同分规则不变；当前入口仍限S6，不扩大为所有动态query验证。\n"
        "    需要PyPTO计数对齐修复、PTOAS Scalar GE及布尔分组修复；配套版本\n"
        "    见tests/pypto_test/csa_multitask_backport_20261020/ADOPTION_topk_prefix.json。\n"
    )
    if source.count(old) != 1:
        raise ValueError("来源helper说明与预期不同")
    helper.write_text(source.replace(old, new, 1))
    subprocess.run([str(RUFF), "format", str(helper)], check=True)
    indexer = stage / "decode_indexer.py"
    source = indexer.read_text()
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name in ("indexer_score_topk_native_cube", "indexer_score_topk_forest")
    ]
    for node in sorted(nodes, key=lambda node: node.lineno, reverse=True):
        source = format_part(source, min(d.lineno for d in node.decorator_list), node.end_lineno, TARGET / indexer.name)
    indexer.write_text(source)
    subprocess.run([str(RUFF), "check", "--select", "I", "--fix", *[str(stage / n) for n in FILES]], check=True)
    subprocess.run([str(RUFF), "check", *[str(stage / n) for n in FILES]], check=True)
    for name in FILES:
        if not equivalent(
            algorithm_source((HERE / "sources/topk_prefix_v1" / name).read_text()),
            algorithm_source((stage / name).read_text()),
        ):
            raise ValueError(f"整理改变了算法、常量或导入：{name}")
        current = (TARGET / name).read_bytes() if (TARGET / name).exists() else None
        if current != originals[name]:
            raise ValueError(f"准备期间目标被其它会话修改：{name}")
    for name in FILES:
        temporary = TARGET / ("." + name + ".backport-tmp")
        shutil.copy2(stage / name, temporary)
        os.replace(temporary, TARGET / name)
    (HERE / "ADOPTION_topk_prefix.json").write_text(
        json.dumps(
            {
                "source": "sources/topk_prefix_v1",
                "source_donor": "c49ee5b2",
                "baseline": "plan_workers_v1 / cc07f727",
                "files": list(FILES),
                "algorithm_ast_and_import_symbols_equal": True,
                "validation": "七档S6，B16/B8 metadata/padding，B5/T30碎片页与八次同图长短分支回放",
                "numeric_change": False,
                "adoption_reason": "五个128K档mean全部改善、七档max均降低；保持原长G6选择范围",
                "retained_negative": "8K/B32 mean回退1.305%；实际短Score两核二进制相同，尚未唯一定位原因",
                "scope_limit": "原多task入口限S6；非整模型token/DSpark验收",
                "required_toolchain": matrix["toolchain_identity"],
                "shared_install_changed": False,
                "performance_evidence": "MATRIX_topk_prefix_v1_topk_v1.json",
                "weighted_reduction_vs_plan_pct": matrix["weighted"]["reduction_vs_current_pct"],
                "weighted_reduction_vs_spmd_pct": matrix["weighted"]["reduction_vs_spmd_pct"],
                "goal_complete": False,
                "new_device_tests_for_formatting": False,
                "pushed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("已迁入正式多task两文件；保持原长G6范围，仍未完成全七档超过单SPMD目标。")


if __name__ == "__main__":
    main()
