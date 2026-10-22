# SPDX-License-Identifier: Apache-2.0
"""Freeze a common non-QA projection grid and its matching stage placement."""

import argparse
import ast
import hashlib
import json
import runpy
import subprocess
from pathlib import Path

from joint_prefix_source import SOURCE
from prepare_scale_gather import replace_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="joint_prefix_v2")
    args = parser.parse_args()
    if args.variant != "joint_prefix_v2":
        raise ValueError("The current fixed-width helper template belongs to joint_prefix_v2")
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    parent = subprocess.check_output(["git", "rev-parse", "79fae6db0"], cwd=repo, text=True).strip()
    package = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    paths = subprocess.check_output(["git", "ls-tree", "--name-only", parent, package + "/"], cwd=repo, text=True)
    original = {
        Path(path).name: subprocess.check_output(["git", "show", f"{parent}:{path}"], cwd=repo, text=True)
        for path in paths.splitlines()
        if path.endswith(".py")
    }
    source = dict(original)
    source["joint_prefix_stages.py"] = SOURCE
    name = "policy.py"
    source[name] = replace_once(
        source[name],
        "    prefix_branch_groups: int = 4\n",
        "    prefix_branch_groups: int = 4\n"
        "    joint_bf16_prefix: bool = False\n    prefix_projection_rows: int = 128\n",
    )
    source[name] = replace_once(
        source[name],
        '            "PREFIX_BRANCH_GROUPS": self.prefix_branch_groups,\n',
        '            "PREFIX_BRANCH_GROUPS": self.prefix_branch_groups,\n'
        '            "JOINT_BF16_PREFIX": self.joint_bf16_prefix,\n'
        '            "PREFIX_PROJECTION_ROWS": self.prefix_projection_rows,\n',
    )
    source[name] = replace_once(
        source[name],
        "        prefix_branch_groups=4,\n",
        "        prefix_branch_groups=4,\n"
        "        joint_bf16_prefix=64 < tokens <= 192,\n"
        "        prefix_projection_rows=192 if 128 < tokens <= 192 else 128,\n",
    )
    for name in ["entry.py", "cube_schedule.py"]:
        signature = "    PREFIX_BRANCH_GROUPS: pl.constexpr = 4,\n"
        assert source[name].count(signature) == (2 if name == "entry.py" else 1)
        source[name] = source[name].replace(
            signature,
            signature
            + "    JOINT_BF16_PREFIX: pl.constexpr = False,\n    PREFIX_PROJECTION_ROWS: pl.constexpr = 128,\n",
        )
    name = "entry.py"
    call = "            PREFIX_BRANCH_GROUPS,\n"
    assert source[name].count(call) == 2
    source[name] = source[name].replace(
        call, call + "            JOINT_BF16_PREFIX,\n            PREFIX_PROJECTION_ROWS,\n"
    )
    name = "cube_schedule.py"
    source[name] = replace_once(
        source[name],
        "from .kv_stages import kv_project_stage\n",
        "from .kv_stages import kv_project_stage\nfrom .joint_prefix_stages import joint_bf16_prefix_stage\n",
    )
    start = source[name].index("    idx_value, idx_score = indexer_compressor_project_stage(\n")
    marker = "    signal = publish_phase(signal, PHASE_IDX_PROJECT, worker, CORES, generation)\n"
    end = source[name].index(marker, start) + len(marker)
    previous = source[name][start:end]
    replacement = """    if JOINT_BF16_PREFIX and COOPERATIVE_PREFIX:
        idx_value, idx_score, main_value, main_score, kv_fp32 = joint_bf16_prefix_stage(
            normed, idx_wkv, idx_wgate, idx_value, idx_score,
            main_wkv, main_wgate, main_value, main_score, wkv, kv_fp32,
            worker, GROUPS, PREFIX_PROJECTION_ROWS,
        )
        signal = publish_phase(signal, PHASE_IDX_PROJECT, worker, CORES, generation)
        signal = publish_phase(signal, PHASE_MAIN_PROJECT, worker, CORES, generation)
        signal = publish_phase(signal, PHASE_KV_PROJECT, worker, CORES, generation)
    else:
"""
    replacement += "".join("    " + line for line in previous.splitlines(keepends=True))
    source[name] = source[name][:start] + replacement + source[name][end:]
    source[name] = replace_once(
        source[name], "            QB_WIDE_N,\n        )\n", "            QB_WIDE_N, JOINT_BF16_PREFIX,\n        )\n"
    )
    name = "projection_overlap_stages.py"
    source[name] = replace_once(
        source[name],
        "    QB_WIDE_N: pl.constexpr = False,\n",
        "    QB_WIDE_N: pl.constexpr = False,\n    JOINT_BF16_PREFIX: pl.constexpr = False,\n",
    )
    for first, last in [
        (
            "    main_value, main_score = compressor_project_stage(\n",
            "    signal = publish_phase(signal, PHASE_MAIN_PROJECT, worker, CORES, generation)\n",
        ),
        (
            "    kv_worker = (worker + GROUPS - BRANCH_GROUPS) % GROUPS\n",
            "    signal = publish_phase(signal, PHASE_KV_PROJECT, worker, CORES, generation)\n",
        ),
    ]:
        begin = source[name].index(first)
        end = source[name].index(last, begin) + len(last)
        previous = source[name][begin:end]
        source[name] = (
            source[name][:begin]
            + "    if not JOINT_BF16_PREFIX:\n"
            + "".join("    " + line for line in previous.splitlines(keepends=True))
            + source[name][end:]
        )
    target = here / "runs" / args.variant
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    select = runpy.run_path(str(target / "policy.py"))["select_policy"]
    baseline = json.loads((here / "baseline.json").read_text())
    for case_id in ["C03", "C04", "C05", "C07"]:
        case = baseline["cases"][case_id]["case"]
        descriptor = json.loads((here / f"descriptors/{case_id}_leaf_scale_guard_v1.json").read_text())
        descriptor["policy"] = select(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
        ).constexprs()
        (here / f"descriptors/{case_id}_{args.variant}.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    report = dict(
        source=str(target),
        parent_commit=parent,
        adopted=False,
        previous_attempt="joint_prefix_v1; generic width annotations relaxed two static ABI dimensions",
        changes=[name for name in source if source[name] != original.get(name)],
        files={name: hashlib.sha256(content.encode()).hexdigest() for name, content in source.items()},
        mechanism=(
            "Five non-QA BF16 projections share a 24-job M128/M192 N128 grid; main/KV complete before Indexer query"
        ),
        guard="64 < Host token capacity <= 192, and cooperative prefix is enabled",
        unchanged="All Native tensor roots, weights, QA/head-weight math, downstream math and 512B signal protocol",
        changed_math="Original compressor/KV K512 blocks become ascending K256; exactness requires device validation",
        next_gate="CPU compiler resources and actual writer/consumer contracts before full L1 testing",
    )
    with (here / f"{args.variant}.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
