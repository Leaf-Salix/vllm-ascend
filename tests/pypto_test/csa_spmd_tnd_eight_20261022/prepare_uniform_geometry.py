# SPDX-License-Identifier: Apache-2.0
"""Freeze an equal-length branch on top of retained production commit 79fae6db0."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

from prepare_scale_gather import replace_once
from uniform_geometry_source import UNIFORM_HELPERS

PARENT = "79fae6db0"
PACKAGE = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    parent = subprocess.check_output(["git", "rev-parse", PARENT], cwd=repo, text=True).strip()
    names = subprocess.check_output(["git", "ls-tree", "--name-only", parent, PACKAGE + "/"], cwd=repo, text=True)
    source = {
        Path(name).name: subprocess.check_output(["git", "show", f"{parent}:{name}"], cwd=repo, text=True)
        for name in names.splitlines()
        if name.endswith(".py")
    }
    previous = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()}
    source["request_geometry.py"] += UNIFORM_HELPERS
    name = "attention_plan_stages.py"
    source[name] = replace_once(
        source[name],
        "from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN, query_request\n",
        "from .request_geometry import (\n"
        "    QUERY_BOUNDS_DYN, REQUESTS_DYN, query_request, query_request_with_uniform, uniform_query_width,\n"
        ")\n",
    )
    marker = "    for first in pl.range(worker * ROWS, tokens, WORKERS * ROWS):\n"
    source[name] = replace_once(
        source[name], marker,
        "    uniform_width = pl.cast(0, pl.INDEX)\n"
        "    if PLAN_PHYSICAL_ROWS:\n"
        "        if worker * ROWS < tokens:\n"
        "            uniform_width = uniform_query_width(bounds, lengths, tokens)\n" + marker,
    )
    marker = "            request, begin, end, active = query_request(bounds, lengths, token, tokens)\n"
    # Restrict to the early geometry function, leaving the late fallback intact.
    start = source[name].index("def attention_geometry_prepare_stage(")
    end = source[name].index("def attention_slot_plan_stage(")
    region = source[name][start:end]
    region = replace_once(
        region, marker,
        "            request, begin, end, active = query_request_with_uniform(\n"
        "                bounds, lengths, token, tokens, uniform_width\n"
        "            )\n",
    )
    source[name] = source[name][:start] + region + source[name][end:]
    source[name] = replace_once(
        source[name], "    CACHE_QUERY_METADATA: pl.constexpr,\n",
        "    CACHE_QUERY_METADATA: pl.constexpr,\n    uniform_width: pl.Scalar[pl.INDEX],\n",
    )
    source[name] = replace_once(
        source[name],
        "        owner, begin, end, owner_active = query_request("
        "bounds, lengths, query, pl.tensor.dim(positions, 0))\n",
        "        owner, begin, end, owner_active = query_request_with_uniform(\n"
        "            bounds, lengths, query, pl.tensor.dim(positions, 0), uniform_width\n"
        "        )\n",
    )
    name = "attention_stream_stages.py"
    source[name] = replace_once(
        source[name], "import pypto.language as pl\n",
        "import pypto.language as pl\n\nfrom .request_geometry import uniform_query_width\n",
    )
    marker = "    qk_block_count = qk_query_count * SPARSE_BLOCKS\n"
    start = source[name].index("def attention_qk_pv_vector_lane_stage(")
    region = replace_once(
        source[name][start:], marker,
        marker + "    uniform_width = pl.cast(0, pl.INDEX)\n"
        "    if PLAN_PHYSICAL_ROWS and not CACHE_QUERY_METADATA:\n"
        "        if qk_query_count > 0:\n"
        "            uniform_width = uniform_query_width(bounds, lengths, t_dim)\n",
    )
    source[name] = source[name][:start] + region
    source[name] = replace_once(
        source[name],
        "                        position_ids, bounds, lengths, valid_block_mask, qk_t, CACHE_QUERY_METADATA\n",
        "                        position_ids, bounds, lengths, valid_block_mask, qk_t, CACHE_QUERY_METADATA,\n"
        "                        uniform_width,\n",
    )
    target = here / "runs/uniform_geometry_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    report = dict(
        source=str(target), parent_commit=parent, baseline="baseline.json", adopted=False,
        parent_files=previous,
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        changes=[n for n, s in source.items() if previous[n] != hashlib.sha256(s.encode()).hexdigest()],
        guard="Existing PLAN_PHYSICAL_ROWS resource tier; each live AIV proves all device Q/L entries anew",
        predicate="B>1; Q[r]=r*S; B*S=T; all L equal and L>=S>0; actual positions/pages still checked",
        fallback="Original query_request scan for mixed/inactive/gapped/truncated/unequal-KV metadata",
        unchanged="All math, task/phase/FIFO, buffers, ABI, Host graph contract and retained C04 leaf proof",
        purpose="Amortize per-token request scans across each AIV's actual query work",
    )
    with (here / "uniform_geometry_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
