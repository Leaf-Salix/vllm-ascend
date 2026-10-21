# SPDX-License-Identifier: Apache-2.0
"""将成对压缩按Score实际工作波数选择，保留两波区间原短前缀路径。"""

import argparse
import ast
import json
import shutil
import subprocess
import textwrap

from prepare import HERE, REPO
from prepare_query_ready import replace_once


def function_text(source, node):
    start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    return "".join(source.splitlines(keepends=True)[start - 1 : node.end_lineno])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="formal_4d5b2ed9")
    parser.add_argument("--candidate", default="paired_wave_policy_v1")
    args = parser.parse_args()
    if not all(name.isidentifier() for name in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base = HERE / "sources" / args.base
    target = HERE / "sources" / args.candidate
    donor = subprocess.check_output(
        [
            "git",
            "show",
            "c0d27e9a:tests/pypto_test/csa_roofline_optimization_20261020/source_topk_paired/topk_threshold_stages.py",
        ],
        cwd=REPO,
        text=True,
    )
    donor_functions = {n.name: n for n in ast.parse(donor).body if isinstance(n, ast.FunctionDef)}
    base_functions = {
        n.name: n
        for n in ast.parse((base / "topk_threshold_stages.py").read_text()).body
        if isinstance(n, ast.FunctionDef)
    }
    if ast.dump(donor_functions["sort_top512"]) != ast.dump(base_functions["sort_top512"]):
        raise ValueError("共用原完整排序前需确认两份算术相同")
    compact_helper = (
        "# SPDX-License-Identifier: Apache-2.0\n"
        '"""c0d27e9a已验证的成对压缩；与旧短前缀共用完整排序和常量。"""\n\n'
        "import pypto.language as pl\n\n"
        "from .topk_threshold_stages import (\n"
        "    COMPARE_SCRATCH_BYTES, MAX_SEGMENT, NEGATIVE_SENTINEL, PAIR_WIDTH, sort_top512,\n"
        ")\n\n"
        + function_text(donor, donor_functions["sort_compacted_prefix"])
        + "\n\n\n"
        + function_text(donor, donor_functions["threshold_merge_top512"]).replace(
            "def threshold_merge_top512(", "def threshold_merge_paired_top512(", 1
        )
        + "\n"
    )
    source = (base / "decode_indexer.py").read_text()
    source = replace_once(
        source,
        "from .topk_threshold_stages import threshold_merge_top512\n",
        "from .topk_compact_stages import threshold_merge_paired_top512\n"
        "from .topk_threshold_stages import threshold_merge_top512\n",
    )
    source = replace_once(
        source,
        "        pl.system.set_ffts(buf_score_ffts)\n",
        "        # 已测B8两波区间成对压缩回退；按实际分组数×叶数选择，不枚举batch。\n"
        "        buf_use_paired_prefix = pl.cast(1, pl.INT32)\n"
        "        if balance_leaves:\n"
        "            buf_score_items = buf_query_count // query_group_size * buf_max_leaves\n"
        "            if buf_score_items > TOPK_SCORE_WORKERS:\n"
        "                if buf_score_items <= 2 * TOPK_SCORE_WORKERS:\n"
        "                    buf_use_paired_prefix = pl.cast(0, pl.INT32)\n"
        "        pl.system.set_ffts(buf_score_ffts)\n",
    )
    tree = ast.parse(source)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "threshold_merge_top512"
    ]
    if len(calls) != 1:
        raise ValueError("阈值归并调用必须唯一")
    node = calls[0]
    lines = source.splitlines(keepends=True)
    old_call = textwrap.dedent("".join(lines[node.lineno - 1 : node.end_lineno]))
    new_call = old_call.replace("threshold_merge_top512(", "threshold_merge_paired_top512(", 1)
    replacement = (
        "if buf_use_paired_prefix == 0:\n"
        + textwrap.indent(old_call, "    ")
        + "else:\n"
        + textwrap.indent(new_call, "    ")
    )
    lines[node.lineno - 1 : node.end_lineno] = [textwrap.indent(replacement, " " * node.col_offset)]
    source = "".join(lines)
    ast.parse(source)
    ast.parse(compact_helper)
    shutil.copytree(base, target)
    (target / "decode_indexer.py").write_text(source)
    (target / "topk_compact_stages.py").write_text(compact_helper)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="c0d27e9a已提交成对压缩；多task八档证据MATRIX_topk_paired_v1_pair_v1.json",
                changed_files=["decode_indexer.py", "topk_compact_stages.py"],
                policy="24<实际G6分组数*动态叶数<=48时用原短前缀，其余用成对压缩",
                caveat="两波为待验证的资源启发式，不主张已证明所有两波场景必然回退",
                preserved="完整排序/阈值同分规则/数学/任务依赖；已保留B1与HCpost策略",
                shared_math="两路径共用原sort_top512与常量，避免重复完整排序实现",
                status="private_candidate_no_device_evidence",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
