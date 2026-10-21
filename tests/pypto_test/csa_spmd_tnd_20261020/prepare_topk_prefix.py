# SPDX-License-Identifier: Apache-2.0
"""Borrow accepted exact TopK pruning without replacing TND geometry or schedules."""

import ast
import difflib
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from prepare_score_owner_harness import replace_once
from prepare_short_half_ub import append_argument, append_parameter, function


def algorithm(text):
    tree = ast.parse(text)
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        tree.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    origin = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    other = here.parents[4] / "vllm-ascend-dsv4-pto-0251rc1"
    donor = other / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/topk_threshold_stages.py"
    earlier = other / "tests/pypto_test/csa_roofline_optimization_20261020/source_topk_payload/topk_threshold_stages.py"
    assert algorithm(donor.read_text()) == algorithm(earlier.read_text()), "Donor algorithm changed after measurement"
    target = here / "runs/topk_prefix_candidate_v97"
    originals = {
        name: (origin / name).read_text()
        for name in ("indexer_score_stages.py", "entry.py", "vector_schedule.py", "policy.py")
    }
    revised = dict(originals)
    text = originals["indexer_score_stages.py"]
    vector = function(text, "indexer_score_vector_lane_stage")
    segments = [
        node
        for node in ast.walk(vector)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "single_segment_valid > 0"
    ]
    assert len(segments) == 1
    segment = segments[0]
    assert ast.unparse(segment.body[0].targets[0]) == "single_segment_data"
    fallback = segment.body[1:]
    assert ast.unparse(fallback[0].targets[0]) == "logical_i32"
    lines = text.splitlines(keepends=True)
    start, end = fallback[0].lineno - 1, fallback[-1].end_lineno
    old_block = lines[start:end]
    indent = old_block[0][: len(old_block[0]) - len(old_block[0].lstrip())]
    new_block = [
        indent + "if SHORT_TOPK_PREFIX and single_step >= 2 and single_segment_valid > 512:\n",
        indent + "    threshold_old_root = pl.tile.extract(\n",
        indent + "        single_roots, single_local, 0, [1, TOPK_PAIR_WIDTH],\n",
        indent + "        target_memory=pl.MemorySpace.Vec,\n",
        indent + "    )\n",
        indent + "    threshold_next = threshold_merge_top512(\n",
        indent + "        single_segment_data, threshold_old_root,\n",
        indent + "        pl.cast(single_segment_begin, pl.INT32),\n",
        indent + "        pl.cast(single_segment_valid, pl.INT32),\n",
        indent + "    )\n",
        indent + "    single_roots = pl.tile.assemble(single_roots, threshold_next, [single_local, 0])\n",
        indent + "else:\n",
    ]
    new_block.extend("    " + line if line.strip() else line for line in old_block)
    lines[start:end] = new_block
    text = append_parameter("".join(lines), "indexer_score_vector_lane_stage", "SHORT_TOPK_PREFIX")
    text = replace_once(
        text,
        "from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN\n",
        "from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN\n"
        "from .topk_threshold_stages import threshold_merge_top512\n",
    )
    revised["indexer_score_stages.py"] = text
    text = originals["entry.py"]
    for name in ("_csa_core_group", "decode_csa_spmd"):
        text = append_parameter(text, name, "SHORT_TOPK_PREFIX")
    for name in ("_csa_core_group", "vector_schedule"):
        text = append_argument(text, name, "SHORT_TOPK_PREFIX")
    revised["entry.py"] = text
    text = append_parameter(originals["vector_schedule.py"], "vector_schedule", "SHORT_TOPK_PREFIX")
    revised["vector_schedule.py"] = append_argument(text, "indexer_score_vector_lane_stage", "SHORT_TOPK_PREFIX")
    text = replace_once(
        originals["policy.py"],
        "    output_work_balance: bool = False\n",
        "    output_work_balance: bool = False\n    short_topk_prefix: bool = False\n",
    )
    text = replace_once(
        text,
        '            "OUTPUT_WORK_BALANCE": self.output_work_balance,\n',
        '            "OUTPUT_WORK_BALANCE": self.output_work_balance,\n'
        '            "SHORT_TOPK_PREFIX": self.short_topk_prefix,\n',
    )
    text = replace_once(
        text,
        "        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,\n",
        "        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,\n"
        "        # Candidate only: reuse the existing G6 resource boundary; real retention is device-side.\n"
        "        short_topk_prefix=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,\n",
    )
    revised["policy.py"] = text
    helper = donor.read_text()
    doc = ast.parse(helper).body[0]
    helper_lines = helper.splitlines(keepends=True)
    helper_lines[doc.lineno - 1 : doc.end_lineno] = [
        '"""来自已采用多task的精确Top-K短前缀；本TND副本为待验收候选。\n\n'
        "保留旧根阈值、GE相等项、稳定压缩、原归并方向及完整回退。\n"
        "仅复制helper；真实query/causal/页表与同步仍由当前TND调用方管理。\n"
        "算法与6ba55b45/c49ee5b2一致；其旧S6记录不代替本轮设备/混长/同分验收。\n"
        '"""\n'
    ]
    helper = "".join(helper_lines)
    assert algorithm(helper) == algorithm(donor.read_text())
    revised["topk_threshold_stages.py"] = helper
    shutil.copytree(origin, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patch = []
    for name, text in revised.items():
        ast.parse(text)
        (target / name).write_text(text)
        patch.extend(
            difflib.unified_diff(
                originals.get(name, "").splitlines(True),
                text.splitlines(True),
                fromfile=("a/" + name if name in originals else "/dev/null"),
                tofile="b/" + name,
            )
        )
    spec = importlib.util.spec_from_file_location("topk_prefix_policy", target / "policy.py")
    policy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = policy
    spec.loader.exec_module(policy)
    jobs = {row["task_id"]: row for row in json.loads((here / "jobs.json").read_text())}
    cohort = json.loads((here / "cohort_output_balance_v4.json").read_text())
    descriptors = []
    for batch, history, active in ((28, 131072, True), (4, 131072, False), (1, 131072, False), (16, 8192, False)):
        task = next(row["task_id"] for row in cohort["cases"] if (row["history"], row["batch"]) == (history, batch))
        desc = json.loads((Path(jobs[task]["output"]) / "buffer_descriptors.json").read_text())
        selected = policy.select_policy(tokens=batch * 6, requests=batch, max_query_tokens=6, max_seq_len=history + 6)
        assert selected.short_topk_prefix == active
        desc["policy"] = selected.constexprs()
        path = here / "compile_descriptors" / f"topk_prefix_h{history}_b{batch}_v97.json"
        path.write_text(json.dumps(desc, indent=2) + "\n")
        descriptors.append({"path": str(path), "active": active, "batch": batch, "history": history})
    report = {
        "source": str(target),
        "donor": str(donor),
        "donor_algorithm_ast_equal": True,
        "donor_sha256": hashlib.sha256(donor.read_bytes()).hexdigest(),
        "donor_commits": ["6ba55b45", "c49ee5b2"],
        "changed_files": list(revised),
        "fallback_ast_sha256": hashlib.sha256(
            ast.dump(ast.Module(body=fallback, type_ignores=[])).encode()
        ).hexdigest(),
        "descriptors": descriptors,
        "adopted": False,
        "device_execution": False,
        "preserved": [
            "device TND geometry/visibility/live guards",
            "B1 original half-leaf algorithm",
            "Cube schedule and all synchronization",
            "adopted output balance",
            "native150",
        ],
        "next": "Full active/disabled CPU and exact ties; first B28 paired against adopted TND",
    }
    (here / "topk_prefix_v97.json").write_text(json.dumps(report, indent=2) + "\n")
    (here / "topk_prefix_v97.patch").write_text("".join(patch))
    print(target)


if __name__ == "__main__":
    main()
