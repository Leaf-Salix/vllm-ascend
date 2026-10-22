# SPDX-License-Identifier: Apache-2.0
"""One uniform proof in the existing planner; share it through immutable metadata."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

from prepare_scale_gather import replace_once
from uniform_meta_source import HELPER_SOURCE


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    parent = subprocess.check_output(["git", "rev-parse", "79fae6db0"], cwd=repo, text=True).strip()
    package = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    paths = subprocess.check_output(["git", "ls-tree", "--name-only", parent, package + "/"], cwd=repo, text=True)
    original = {Path(p).name: subprocess.check_output(["git", "show", f"{parent}:{p}"], cwd=repo, text=True)
                for p in paths.splitlines() if p.endswith(".py")}
    source = dict(original)
    name = "indexer_group_stages.py"
    source[name] = replace_once(source[name], "    ROOTS_PER_LEAF: pl.constexpr,\n",
                                "    ROOTS_PER_LEAF: pl.constexpr,\n    UNIFORM_QUERY_SIX: pl.constexpr = False,\n")
    marker = "        # 前缀有跨请求依赖，只扫描一次。描述符也在这名早期AIV上生成，\n"
    source[name] = replace_once(source[name], marker,
        "        uniform_six = pl.cast(0, pl.INT32)\n"
        "        uniform_length = pl.cast(0, pl.INDEX)\n"
        "        if UNIFORM_QUERY_SIX:\n"
        "            if requests > 1 and tokens == requests * 6 and descriptor_base > requests + 1:\n"
        "                ub, ue, ua = request_bounds(bounds, lengths, 0, tokens)\n"
        "                if ua != 0 and ub == 0 and ue == 6:\n"
        "                    uniform_length = pl.cast(pl.read(lengths, [0]), pl.INDEX)\n"
        "                    uniform_six = pl.cast(1, pl.INT32)\n" + marker)
    start = source[name].index("def indexer_group_plan_stage(")
    end = source[name].index("def indexer_query_visibility(")
    region = source[name][start:end]
    marker = "            begin, end, active = request_bounds(bounds, lengths, request, tokens)\n"
    region = replace_once(region, marker, marker +
        "            if UNIFORM_QUERY_SIX:\n"
        "                if active == 0 or begin != request * 6 or end != (request + 1) * 6:\n"
        "                    uniform_six = pl.cast(0, pl.INT32)\n")
    marker = "                length = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX)\n"
    region = replace_once(region, marker, marker +
        "                if UNIFORM_QUERY_SIX:\n"
        "                    if length != uniform_length:\n"
        "                        uniform_six = pl.cast(0, pl.INT32)\n")
    marker = "        pl.write(offsets, [requests], pl.cast(groups, pl.INT32))\n"
    region = replace_once(region, marker, marker +
        "        if UNIFORM_QUERY_SIX:\n"
        "            if descriptor_base > requests + 1:\n"
        "                # Same sole META writer, outside all live prefixes/descriptors.\n"
        "                pl.write(offsets, [descriptor_base - 1], uniform_six)\n")
    source[name] = source[name][:start] + region + source[name][end:]
    source["uniform_query_stages.py"] = HELPER_SOURCE
    name = "attention_plan_stages.py"
    source[name] = replace_once(source[name], "import pypto.language as pl\n",
        "import pypto.language as pl\n\nfrom .uniform_query_stages import query_request_uniform_six\n")
    # Add proof only to the early geometry signature, before its default constexpr.
    start = source[name].index("def attention_geometry_prepare_stage(")
    end = source[name].index("def attention_slot_plan_stage(")
    region = source[name][start:end]
    region = replace_once(region, "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n",
                            "    uniform_six: pl.Scalar[pl.INT32],\n    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n")
    region = replace_once(region,
        "            request, begin, end, active = query_request(bounds, lengths, token, tokens)\n",
        "            request, begin, end, active = query_request_uniform_six(\n"
        "                bounds, lengths, token, tokens, uniform_six\n"
        "            )\n")
    source[name] = source[name][:start] + region + source[name][end:]
    source[name] = replace_once(source[name], "    CACHE_QUERY_METADATA: pl.constexpr,\n",
                                "    CACHE_QUERY_METADATA: pl.constexpr,\n    uniform_six: pl.Scalar[pl.INT32],\n")
    source[name] = replace_once(source[name],
        "        owner, begin, end, owner_active = query_request("
        "bounds, lengths, query, pl.tensor.dim(positions, 0))\n",
        "        owner, begin, end, owner_active = query_request_uniform_six(\n"
        "            bounds, lengths, query, pl.tensor.dim(positions, 0), uniform_six\n"
        "        )\n")
    name = "attention_stream_stages.py"
    start = source[name].index("def attention_qk_pv_vector_lane_stage(")
    region = source[name][start:]
    region = replace_once(region, "    CORE_GROUPS: pl.constexpr,\n",
                            "    uniform_six: pl.Scalar[pl.INT32],\n    CORE_GROUPS: pl.constexpr,\n")
    region = replace_once(region,
        "                        position_ids, bounds, lengths, valid_block_mask, qk_t, CACHE_QUERY_METADATA\n",
        "                        position_ids, bounds, lengths, valid_block_mask, qk_t, CACHE_QUERY_METADATA,\n"
        "                        uniform_six,\n")
    source[name] = source[name][:start] + region
    name = "vector_schedule.py"
    source[name] = replace_once(source[name], "import pypto.language as pl\n",
        "import pypto.language as pl\n\nfrom .uniform_query_stages import read_uniform_six\n")
    source[name] = replace_once(source[name],
        "from .phase_sync import begin_invocation, publish_phase, wait_row_owners\n",
        "from .phase_sync import begin_invocation, publish_phase, wait_producer, wait_row_owners\n")
    marker = "            ROOTS,\n        )\n        # scale的地址/页所有权也只依赖本轮metadata"
    source[name] = replace_once(source[name], marker,
        "            ROOTS,\n            PLAN_PHYSICAL_ROWS and QUERY_GROUP == 6,\n        )\n"
        "        # scale的地址/页所有权也只依赖本轮metadata")
    marker = "    if EARLY_ATTENTION_GEOMETRY:\n        attention_indices = attention_geometry_prepare_stage(\n"
    source[name] = replace_once(source[name], marker,
        "    early_uniform_six = pl.cast(0, pl.INT32)\n"
        "    if PLAN_PHYSICAL_ROWS and QUERY_GROUP == 6:\n"
        "        if worker * 8 < tokens:\n"
        "            signal = wait_producer(signal, PHASE_META, GROUPS + planner, CORES, generation)\n"
        "            early_uniform_six = read_uniform_six(group_offsets, pl.tensor.dim(idx_lengths, 0))\n" + marker)
    marker = "            worker,\n            VECTORS,\n            PLAN_PHYSICAL_ROWS,\n        )\n"
    source[name] = replace_once(source[name], marker,
        "            worker,\n            VECTORS,\n            early_uniform_six,\n"
        "            PLAN_PHYSICAL_ROWS,\n        )\n")
    marker = "    kv_ring, probability_ring, mi_ring, li_ring, packed = attention_qk_pv_vector_lane_stage(\n"
    source[name] = replace_once(source[name], marker,
        "    attention_uniform_six = pl.cast(0, pl.INT32)\n"
        "    if PLAN_PHYSICAL_ROWS and QUERY_GROUP == 6:\n"
        "        # Both prefix orders have already acquired META in indexer_compressor_vector_stage.\n"
        "        attention_uniform_six = read_uniform_six(group_offsets, pl.tensor.dim(idx_lengths, 0))\n" + marker)
    marker = "        pl.tile.get_subblock_idx(),\n        GROUPS,\n        2,\n"
    source[name] = replace_once(source[name], marker,
        "        pl.tile.get_subblock_idx(),\n        attention_uniform_six,\n        GROUPS,\n        2,\n")
    target = here / "runs/uniform_meta_v1"
    target.mkdir(exist_ok=False)
    for name, text in source.items():
        ast.parse(text)
        (target / name).write_text(text)
    report = dict(
        source=str(target), parent_commit=parent, baseline="baseline.json", adopted=False,
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        changes=[n for n in source if source[n] != original.get(n)],
        guard="PLAN_PHYSICAL_ROWS and QUERY_GROUP==6; B>1; all Q[r]=6*r, all positive L equal; full T=6*B",
        storage="Last alignment word before descriptors only if outside B+1 prefix; no spare word => general path",
        synchronization=(
            "Existing META publish; early active geometry roles acquire it; late lookup follows existing META wait"
        ),
        differences_from_rejected_v1=(
            "One integrated proof by META, constant //6 consumer, no repeated full Q/L scan per AIV"
        ),
        unchanged="Math, causal position checks, pages, FIFO, 512B signal layout, Host ABI and retained leaf proof",
    )
    with (here / "uniform_meta_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
