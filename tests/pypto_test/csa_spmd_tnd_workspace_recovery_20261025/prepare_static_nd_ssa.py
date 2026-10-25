# SPDX-License-Identifier: Apache-2.0
"""将MIX返回的静态形状别名绑定到新SSA名，避免与本地维度变量的旧类型重赋值。"""

import ast
import copy
import json
import shutil
from pathlib import Path

from prepare import hashes


def main():
    here = Path(__file__).resolve().parent
    source = here / "candidates/static_nd_shapes_probe_v1"
    target = here / "candidates/static_nd_shapes_probe_v2"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    original = (source / "entry.py").read_bytes()
    tree = ast.parse(original)
    lines = original.splitlines(keepends=True)
    offsets, position = [], 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    edits, renames = [], {}
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name not in (
            "decode_csa_spmd",
            "decode_csa_spmd_debug",
        ):
            continue
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "_csa_core_group"
        ]
        assert len(calls) == 1 and len(calls[0].targets) == 1
        names = [node for node in ast.walk(calls[0].targets[0]) if isinstance(node, ast.Name)]
        mapping = {node.id: node.id + "_after_mix" for node in names}
        assert not set(mapping.values()).intersection(n.id for n in ast.walk(function) if isinstance(n, ast.Name))
        returns = [n for n in function.body if isinstance(n, ast.Return)]
        assert len(returns) == 1 and returns[0] is function.body[-1]
        names += [n for n in ast.walk(returns[0]) if isinstance(n, ast.Name) and n.id in mapping]
        for node in names:
            edits.append(
                (
                    offsets[node.lineno - 1] + node.col_offset,
                    offsets[node.end_lineno - 1] + node.end_col_offset,
                    mapping[node.id].encode(),
                )
            )
        renames[function.name] = mapping
    revised = original
    for begin, end, replacement in sorted(edits, reverse=True):
        revised = revised[:begin] + replacement + revised[end:]
    check = copy.deepcopy(ast.parse(revised))
    for function in check.body:
        if isinstance(function, ast.FunctionDef) and function.name in renames:
            reverse = {v: k for k, v in renames[function.name].items()}
            for node in ast.walk(function):
                if isinstance(node, ast.Name) and node.id in reverse:
                    node.id = reverse[node.id]
    assert ast.dump(check) == ast.dump(tree), "除明确的返回SSA别名之外，禁止任何计算/调度改变"
    (target / "entry.py").write_bytes(revised)
    report = {
        "parent": "static_nd_shapes_probe_v1",
        "candidate": str(target),
        "adopted": False,
        "change": "仅两个最外层入口把MIX返回值接到新的SSA名称，函数return引用相同返回对象",
        "normalizing_new_names_recovers_identical_AST": True,
        "reason": "解析器将本地create_tensor的[t,16384]与callee返回[96,16384]视为不同类型；新名不改变数据或别名",
        "open_requirements": "延续v1全部CPU/NPU/形状dtype合同与服务验证要求，当前仍是可行性探针",
        "renames": renames,
        "source_hashes": hashes(target),
    }
    (here / "static_nd_shapes_probe_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
