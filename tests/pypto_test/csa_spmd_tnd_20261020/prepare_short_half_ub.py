# SPDX-License-Identifier: Apache-2.0
"""Port the scoped G2 half-leaf helper, retaining the current fallback verbatim."""

import ast
import difflib
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from prepare_score_owner_harness import replace_once


def function(text, name):
    return next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name)


def append_parameter(text, name, parameter="SHORT_HALF_SCORE_UB"):
    node = function(text, name)
    lines = text.splitlines(keepends=True)
    lines.insert(node.args.args[-1].end_lineno, f"    {parameter}: pl.constexpr = False,\n")
    return "".join(lines)


def append_argument(text, name, argument="SHORT_HALF_SCORE_UB"):
    calls = [
        node
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]
    assert len(calls) == 1, (name, len(calls))
    call = calls[0]
    lines = text.splitlines(keepends=True)
    closing = lines[call.end_lineno - 1]
    assert closing.strip() == ")", closing
    indent = closing[: len(closing) - len(closing.lstrip())] + "    "
    lines.insert(call.end_lineno - 1, indent + argument + ",\n")
    return "".join(lines)


def main():
    here = Path(__file__).resolve().parent
    origin = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    external = (
        here.parents[4] / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/"
        "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/"
        "indexer_evidence/cpu_medium_v7/source/indexer_score_stages.py"
    )
    target = here / "runs/short_half_ub_candidate_v95"
    borrowed = external.read_text()
    old_vector = function(borrowed, "indexer_score_vector_lane_stage")
    old_branch = next(
        node for node in ast.walk(old_vector) if isinstance(node, ast.If) and ast.unparse(node.test) == "balance_leaves"
    )
    old_fallback = ast.Module(body=old_branch.orelse[0].orelse, type_ignores=[])
    helper = function(borrowed, "indexer_score_local_half_leaf")
    helper_source = "\n".join(borrowed.splitlines()[helper.decorator_list[0].lineno - 1 : helper.end_lineno])
    originals = {
        name: (origin / name).read_text()
        for name in ("indexer_score_stages.py", "entry.py", "vector_schedule.py", "policy.py")
    }
    revised = dict(originals)
    text = originals["indexer_score_stages.py"]
    vector = function(text, "indexer_score_vector_lane_stage")
    branch = next(
        node for node in ast.walk(vector) if isinstance(node, ast.If) and ast.unparse(node.test) == "balance_leaves"
    )
    fallback = ast.Module(body=branch.orelse, type_ignores=[])
    assert ast.dump(fallback) == ast.dump(old_fallback), "Current non-balanced mathematics changed"
    lines = text.splitlines(keepends=True)
    first, last = branch.orelse[0].lineno - 1, branch.orelse[-1].end_lineno
    original_block = lines[first:last]
    indent = original_block[0][: len(original_block[0]) - len(original_block[0].lstrip())]
    replacement = [indent + "if SHORT_HALF_SCORE_UB:\n", indent + "    indexer_score_local_half_leaf(\n"]
    replacement.extend(indent + "        " + arg.arg + ",\n" for arg in helper.args.args)
    replacement.extend([indent + "    )\n", indent + "else:\n"])
    replacement.extend("    " + line if line.strip() else line for line in original_block)
    lines[first:last] = replacement
    text = append_parameter("".join(lines), "indexer_score_vector_lane_stage")
    text = replace_once(
        text,
        "@pl.jit.inline(auto_scope=False)\ndef indexer_score_vector_lane_stage(",
        helper_source + "\n\n\n@pl.jit.inline(auto_scope=False)\ndef indexer_score_vector_lane_stage(",
    )
    revised["indexer_score_stages.py"] = text
    text = originals["entry.py"]
    for name in ("_csa_core_group", "decode_csa_spmd"):
        text = append_parameter(text, name)
    for name in ("_csa_core_group", "vector_schedule"):
        text = append_argument(text, name)
    revised["entry.py"] = text
    text = append_parameter(originals["vector_schedule.py"], "vector_schedule")
    revised["vector_schedule.py"] = append_argument(text, "indexer_score_vector_lane_stage")
    text = originals["policy.py"]
    text = replace_once(
        text,
        "SCORE_SPLIT_MIN_TOKENS = 32\n",
        "SCORE_SPLIT_MIN_TOKENS = 32\n"
        "# At most two query-group work items per each of the 24 physical groups.\n"
        "SHORT_HALF_SCORE_GROUP_LIMIT = 48\n",
    )
    text = replace_once(
        text,
        "    output_work_balance: bool = False\n",
        "    output_work_balance: bool = False\n    short_half_score_ub: bool = False\n",
    )
    text = replace_once(
        text,
        '            "OUTPUT_WORK_BALANCE": self.output_work_balance,\n',
        '            "OUTPUT_WORK_BALANCE": self.output_work_balance,\n'
        '            "SHORT_HALF_SCORE_UB": self.short_half_score_ub,\n',
    )
    text = replace_once(
        text,
        "        ob_rows=ob_rows,\n",
        "        ob_rows=ob_rows,\n"
        "        # Host upper bound only; real mixed-query membership remains on device.\n"
        "        short_half_score_ub=(not long_history and group == 2\n"
        "                             and requests * ((max_query_tokens + 1) // 2)\n"
        "                             <= SHORT_HALF_SCORE_GROUP_LIMIT),\n",
    )
    revised["policy.py"] = text
    shutil.copytree(origin, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patch = []
    for name, text in revised.items():
        ast.parse(text)
        (target / name).write_text(text)
        patch.extend(
            difflib.unified_diff(
                originals[name].splitlines(True), text.splitlines(True), fromfile="a/" + name, tofile="b/" + name
            )
        )
    spec = importlib.util.spec_from_file_location("short_half_ub_policy", target / "policy.py")
    policy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = policy
    spec.loader.exec_module(policy)
    descriptors = []
    for name, batch, history, template, active in (
        ("short_b16", 16, 8192, "short_history_b16.json", True),
        ("short_b32", 32, 8192, "short_history_b32.json", False),
        ("long_b16", 16, 131072, "single_m16_qa_b16_v83.json", False),
    ):
        desc = json.loads((here / "compile_descriptors" / template).read_text())
        selected = policy.select_policy(tokens=batch * 6, requests=batch, max_query_tokens=6, max_seq_len=history + 6)
        assert selected.short_half_score_ub == active
        desc["policy"] = selected.constexprs()
        path = here / "compile_descriptors" / ("short_half_ub_" + name + "_v95.json")
        path.write_text(json.dumps(desc, indent=2) + "\n")
        descriptors.append({"case": name, "path": str(path), "active": active, "policy": desc["policy"]})
    report = {
        "source": str(target),
        "borrowed_source": str(external),
        "borrowed_source_sha256": hashlib.sha256(external.read_bytes()).hexdigest(),
        "helper_ast_sha256": hashlib.sha256(ast.dump(helper).encode()).hexdigest(),
        "unchanged_fallback_ast_sha256": hashlib.sha256(ast.dump(fallback).encode()).hexdigest(),
        "changed_files": list(revised),
        "descriptors": descriptors,
        "scope": "Short G2/768; Host request/query-cap upper bound <=48 groups; real device Q/L unchanged",
        "preserved": [
            "all balanced/B1 mathematics",
            "current Cube schedule",
            "leaf/tie ordering",
            "adopted output ownership",
            "native150",
            "Native TND/service binding",
        ],
        "adopted": False,
        "device_execution": False,
    }
    (here / "short_half_ub_v95.json").write_text(json.dumps(report, indent=2) + "\n")
    (here / "short_half_ub_v95.patch").write_text("".join(patch))
    print(target)


if __name__ == "__main__":
    main()
