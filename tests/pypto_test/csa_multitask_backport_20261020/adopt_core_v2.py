# SPDX-License-Identifier: Apache-2.0
"""核对七档与受影响边界后迁入core_v2；仅修正import排版和超长语句，保持算法AST。"""

import ast
import collections
import json
import os
import shutil
import subprocess
import textwrap

from prepare import HERE, REPO

FILES = (
    "indexer_scale_layout.py",
    "physical_attention_plan.py",
    "decode_indexer.py",
    "decode_sparse_attn_csa.py",
    "decode_o_proj.py",
    "qkv_proj_rope.py",
    "decode_csa.py",
)
RUFF = REPO.parent / ".venv-dsv4-0251rc1/bin/ruff"
TARGET = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa"


def equivalent(before, after):
    def fingerprint(source):
        tree = ast.parse(source)
        imports = []
        body = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.extend(
                    (getattr(node, "module", None), getattr(node, "level", 0), n.name, n.asname) for n in node.names
                )
            else:
                body.append(node)
        return collections.Counter(imports), ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)

    return fingerprint(before) == fingerprint(after)


def format_part(source, start, end, filename):
    lines = source.splitlines(keepends=True)
    segment = "".join(lines[start - 1 : end])
    indentation = " " * (len(lines[start - 1]) - len(lines[start - 1].lstrip()))
    formatted = subprocess.run(
        [str(RUFF), "format", "--line-length", str(max(40, 120 - len(indentation))), "--stdin-filename", str(filename)],
        input=textwrap.dedent(segment),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines[start - 1 : end] = [textwrap.indent(formatted, indentation)]
    return "".join(lines)


def main():
    matrix = json.loads((HERE / "MATRIX_core_v2_matrix_v1.json").read_text())
    if not matrix["passed"] or matrix["weighted"]["reduction_vs_original_pct"] <= 0:
        raise ValueError("七档尚未完整通过或未获得总体收益")
    for case in ("h131072_b16_matrix_v1", "h131072_b5_tail_v1", "h131072_b5_fragmented_v1"):
        record = json.loads((HERE / "runs/core_v2" / case / "result.json").read_text())
        if not record["passed"] or not record.get("boundary_replay", {}).get("passed"):
            raise ValueError(f"边界未通过：{case}")
    paths = [str((TARGET / name).relative_to(REPO)) for name in FILES]
    if subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *paths], cwd=REPO).strip():
        raise ValueError("目标文件存在其他未提交改动，拒绝覆盖")
    for name in FILES[:2]:
        if (TARGET / name).exists():
            raise ValueError(f"新helper已存在，先核对来源：{name}")
    originals = {name: (TARGET / name).read_bytes() if (TARGET / name).exists() else None for name in FILES}
    stage = HERE / "sources/adoption_core_v2"
    stage.mkdir(parents=True, exist_ok=False)
    for name in FILES:
        shutil.copy2(HERE / "sources/core_v2" / name, stage / name)
    # 正式Score此前已排版；只整理两个发生改变的长函数，不重排整份遗留文件。
    indexer = stage / "decode_indexer.py"
    source = indexer.read_text()
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name in ("indexer_score_topk_native_cube", "indexer_score_topk_forest")
    ]
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        source = format_part(source, min(d.lineno for d in node.decorator_list), node.end_lineno, TARGET / indexer.name)
    indexer.write_text(source)
    subprocess.run([str(RUFF), "check", "--select", "I", "--fix", *[str(stage / n) for n in FILES]], check=True)
    for name in FILES:
        path = stage / name
        for _ in range(3):
            result = subprocess.run(
                [str(RUFF), "check", "--select", "E501", "--output-format", "json", str(path)],
                capture_output=True,
                text=True,
            )
            diagnostics = json.loads(result.stdout)
            if not diagnostics:
                break
            source = path.read_text()
            tree = ast.parse(source)
            spans = set()
            for diagnostic in diagnostics:
                row = diagnostic["location"]["row"]
                nodes = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and n.lineno <= row <= n.end_lineno]
                node = min(nodes, key=lambda n: (n.end_lineno - n.lineno, -n.col_offset))
                spans.add((node.lineno, node.end_lineno))
            for start, end in sorted(spans, reverse=True):
                source = format_part(source, start, end, TARGET / name)
            path.write_text(source)
        if not equivalent((HERE / "sources/core_v2" / name).read_text(), path.read_text()):
            raise AssertionError(f"排版改变了算法、常量或接口AST：{name}")
    subprocess.run([str(RUFF), "check", *[str(stage / n) for n in FILES]], check=True)
    for name, old in originals.items():
        current = (TARGET / name).read_bytes() if (TARGET / name).exists() else None
        if current != old:
            raise ValueError(f"准备过程中目标被其他会话修改：{name}")
    # 全部准备和核对结束后才替换；新helper先落地，最外层接线最后落地。
    for name in FILES:
        temporary = TARGET / ("." + name + ".backport-tmp")
        shutil.copy2(stage / name, temporary)
        os.replace(temporary, TARGET / name)
    (HERE / "ADOPTION_core_v2.json").write_text(
        json.dumps(
            dict(
                source="sources/core_v2",
                target=str(TARGET.relative_to(REPO)),
                files=list(FILES),
                algorithm_ast_and_import_symbols_equal=True,
                evidence="MATRIX_core_v2_matrix_v1.json及长B16/T30/碎片页的同图边界",
                source_donors=[
                    "13259f92",
                    "f222a917",
                    "b26e8e86",
                    "05d2a3e9",
                    "289cc072",
                    "27711336",
                    "9d5327e5",
                    "ef608731",
                ],
                weighted_reduction_vs_original_pct=matrix["weighted"]["reduction_vs_original_pct"],
                weighted_reduction_vs_spmd_pct=matrix["weighted"]["reduction_vs_spmd_pct"],
                goal_complete=False,
                new_device_tests_for_formatting=False,
                pushed=False,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("已迁入正式多task；算法AST与七档实测core_v2一致，仍未达到优于单SPMD的目标。")


if __name__ == "__main__":
    main()
