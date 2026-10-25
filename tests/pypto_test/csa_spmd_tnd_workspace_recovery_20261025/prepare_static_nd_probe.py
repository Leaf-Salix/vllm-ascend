# SPDX-License-Identifier: Apache-2.0
"""仅作可行性探针：ND容量从实参推导特化，保留NZ标注及所有计算/动态metadata读取。"""

import ast
import copy
import json
import shutil
from pathlib import Path

from prepare import hashes


def without_parameter_annotations(tree):
    tree = copy.deepcopy(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            node.annotation = None
    return ast.dump(tree)


def main():
    here = Path(__file__).resolve().parent
    source = here / "candidates/internal_single_f72a"
    target = here / "candidates/static_nd_shapes_probe_v1"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    trees = {p.stem: ast.parse(p.read_text()) for p in source.glob("*.py")}
    dynamic = {name: set() for name in trees}
    for name, tree in trees.items():
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if ast.unparse(node.value.func) == "pl.dynamic":
                    dynamic[name].update(t.id for t in node.targets if isinstance(t, ast.Name))
    changed = True
    while changed:
        changed = False
        for name, tree in trees.items():
            before = set(dynamic[name])
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module in dynamic:
                    for alias in node.names:
                        if alias.name in dynamic[node.module]:
                            dynamic[name].add(alias.asname or alias.name)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                    if node.value.id in dynamic[name]:
                        dynamic[name].update(t.id for t in node.targets if isinstance(t, ast.Name))
            changed |= before != dynamic[name]
    changes = {}
    for name, tree in trees.items():
        original = (source / (name + ".py")).read_bytes()
        lines = original.splitlines(keepends=True)
        offsets, total = [], 0
        for line in lines:
            offsets.append(total)
            total += len(line)
        edits = []
        for argument in (node for node in ast.walk(tree) if isinstance(node, ast.arg) and node.annotation):
            for node in ast.walk(argument.annotation):
                if not isinstance(node, ast.Subscript) or ast.unparse(node.value) != "pl.Tensor":
                    continue
                if not isinstance(node.slice, ast.Tuple) or len(node.slice.elts) != 2:
                    continue  # 非ND布局及带memref的签名保持原样。
                dims = {item.id for item in ast.walk(node.slice.elts[0]) if isinstance(item, ast.Name)}
                if not dims.intersection(dynamic[name]):
                    continue
                edits.append(
                    (
                        offsets[node.lineno - 1] + node.col_offset,
                        offsets[node.end_lineno - 1] + node.end_col_offset,
                        argument.arg,
                        ast.unparse(node),
                    )
                )
        revised = original
        for begin, end, _, _ in sorted(edits, reverse=True):
            revised = revised[:begin] + b"pl.Tensor" + revised[end:]
        assert without_parameter_annotations(ast.parse(revised)) == without_parameter_annotations(tree)
        if edits:
            (target / (name + ".py")).write_bytes(revised)
            changes[name] = [
                {"parameter": arg, "before": annotation, "after": "pl.Tensor"} for _, _, arg, annotation in edits
            ]
    assert changes and "entry" in changes and "vector_schedule" in changes
    report = {
        "candidate": str(target),
        "parent": "internal_single_f72a",
        "CPU_feasibility_only": True,
        "adopted": False,
        "only_parameter_annotations_changed_AST_verified": True,
        "all_computations_and_control_flow_unchanged": True,
        "explicit_non_ND_layout_annotations_unchanged": True,
        "shared_SDK_unchanged": True,
        "scope": "ND Tensor容量由实际绑定形状推导特化；请求Q/L/positions/pages的值仍全部由设备运行时读取",
        "open_requirements": [
            "确认原调用域、dtype与形状合同没有放宽，所有Native绑定校验仍保留",
            "确认实际生成物的参数处理/Scalar指令得到减少，不能仅凭注解变化宣称优化",
            "候选编译数量/缓存占用以及混长、padding、同址metadata和多图服务验证",
            "尚未通过CPU可行性或NPU验证，禁止作为正式版本使用",
        ],
        "changes": changes,
        "source_hashes": hashes(target),
    }
    (here / "static_nd_shapes_probe_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{sum(map(len, changes.values()))} annotations in {len(changes)} modules; all function bodies unchanged")


if __name__ == "__main__":
    main()
