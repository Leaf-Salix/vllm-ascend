# SPDX-License-Identifier: Apache-2.0
"""将原量化/系数融合移到本轮冻结的内部工作区入口；不改生产或公共SDK。"""

import argparse
import ast
import difflib
import json
import runpy
import shutil
import sys
import textwrap
from pathlib import Path

from prepare import replace_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    version = parser.parse_args().version
    here = Path(__file__).resolve().parent
    parent = here / "frozen/pypto/deepseek_v4_flash_csa_spmd_tnd"
    previous = here.parent / "csa_spmd_tnd_eight_20261022"
    target = here / f"candidates/fused_coefficients_current_{version}"
    # 只复用原型的纯源码变换函数，不运行其main，不复制旧binding/entry。
    sys.path.insert(0, str(previous))
    try:
        helpers = runpy.run_path(str(previous / "prepare_fused_coefficients.py"))
    finally:
        sys.path.pop(0)
    original = {p.name: p.read_text() for p in parent.glob("*.py")}
    source = dict(original)
    source["fused_coefficients.py"] = (
        (previous / "fused_coefficients_template.py").read_text()
        + "\n\n"
        + helpers["make_fused_quant"](source["indexer_query_stages.py"])
    )
    if version == "v2":
        # 原request_bounds逐请求过滤，不强制Q全局单调。多个有效区间可能
        # 覆盖同一个token：单值映射必须显式回退，不能丢失旧实现的系数片。
        source["fused_coefficients.py"] = replace_once(
            source["fused_coefficients.py"],
            "                pl.write(offsets, [map_base + query], pl.cast(group * GROUP_SIZE + lane, pl.INT32))\n",
            "                if pl.read(offsets, [map_base + query]) == -1:\n"
            "                    pl.write(offsets, [map_base + query], pl.cast(group * GROUP_SIZE + lane, pl.INT32))\n"
            "                else:\n"
            "                    # 同一token可能属于多个有效区间；由唯一token owner处理。\n"
            "                    pl.write(offsets, [map_base + query], pl.cast(-2, pl.INT32))\n",
        )
        tree = ast.parse(source["fused_coefficients.py"])
        quant = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_query_quant_coeff_stage"
        )
        loop = next(n for n in quant.body if isinstance(n, ast.For))
        loop.body.extend(
            ast.parse("""
if encoded == -2:
    requests = pl.tensor.dim(lengths, 0)
    groups = pl.cast(pl.read(offsets, [requests]), pl.INDEX)
    for overlap_group in pl.range(groups):
        overlap_request, overlap_first, overlap_live, _overlap_visible = indexer_decode_group(
            bounds, lengths, positions, offsets, overlap_group, GROUP_SIZE
        )
        if idx >= overlap_first and idx < overlap_first + overlap_live:
            overlap_lane = idx - overlap_first
            _overlap_visible, overlap_causal = indexer_query_visibility(
                bounds, lengths, positions, overlap_request, idx
            )
            if overlap_causal != 0:
                overlap_weights = weights[idx:idx + 1, 0:HEADS]
                overlap_scale_half = pl.cast(qh_scale_recip, pl.FP16, mode="rint")
                overlap_weight_half = pl.cast(overlap_weights, pl.FP16, mode="rint")
                overlap_coefficients = pl.mul(overlap_scale_half, overlap_weight_half)
                coefficients = pl.assemble(
                    coefficients, overlap_coefficients,
                    [overlap_group * COEFFICIENT_ROWS + overlap_lane, overlap_lane * HEADS]
                )
""").body
        )
        source["fused_coefficients.py"] = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    policy = replace_once(
        source["policy.py"],
        "    plan_table_columns: int = DEFAULT_PHYSICAL_PLAN_COLUMNS\n",
        "    plan_table_columns: int = DEFAULT_PHYSICAL_PLAN_COLUMNS\n    fused_indexer_coeff: bool = False\n",
    )
    policy = replace_once(
        policy,
        '            "PLAN_TABLE_COLUMNS": self.plan_table_columns,\n',
        '            "PLAN_TABLE_COLUMNS": self.plan_table_columns,\n'
        '            "FUSED_INDEXER_COEFF": self.fused_indexer_coeff,\n',
    )
    source["policy.py"] = replace_once(
        policy,
        "        plan_table_columns=plan_table_columns,\n",
        "        plan_table_columns=plan_table_columns,\n"
        "        # 只选已有长G6资源域；真实group、causal、尾组仍由device决定。\n"
        "        fused_indexer_coeff=(long_history and group == 6 and 64 <= tokens <= 192),\n",
    )
    for name, count in (("entry.py", 3), ("cube_schedule.py", 1), ("vector_schedule.py", 1)):
        marker = "    SCORE_GROUP_ORDER: pl.constexpr = False,\n"
        assert source[name].count(marker) == count
        source[name] = source[name].replace(marker, marker + "    FUSED_INDEXER_COEFF: pl.constexpr = False,\n")
    source["entry.py"] = helpers["forward_flag"](
        source["entry.py"], {"cube_schedule", "vector_schedule", "_csa_core_group"}
    )
    allocation = (
        "    group_offsets = pl.create_tensor(\n"
        "        [((b + 1 + 16 - 1) // 16 * 16 + 4 * t + 16 - 1) // 16 * 16 + 48 + 4 * t + 2 * b], dtype=pl.INT32\n"
        "    )\n"
    )
    assert source["entry.py"].count(allocation) == 2
    source["entry.py"] = source["entry.py"].replace(
        allocation,
        "    # 原分区及其偏移不变；仅融合域在内部offset尾追加T个INT32映射。\n"
        "    group_offset_words = ((b + 1 + 16 - 1) // 16 * 16 + 4 * t + 16 - 1) // 16 * 16 + 48 + 4 * t + 2 * b\n"
        "    if FUSED_INDEXER_COEFF:\n"
        "        group_offset_words = group_offset_words + t\n"
        "    group_offsets = pl.create_tensor([group_offset_words], dtype=pl.INT32)\n",
    )
    vector = replace_once(
        source["vector_schedule.py"],
        "import pypto.language as pl\n",
        "import pypto.language as pl\n\n"
        "from .fused_coefficients import prepare_fused_query_coefficients, indexer_query_quant_coeff_stage\n",
    )
    meta = "        signal = publish_phase(signal, PHASE_META, core, CORES, generation)\n"
    vector = replace_once(
        vector,
        meta,
        "        if FUSED_INDEXER_COEFF:\n"
        "            group_offsets, coefficients = prepare_fused_query_coefficients(\n"
        "                idx_bounds, idx_lengths, positions, group_offsets, coefficients, tokens, QUERY_GROUP\n"
        "            )\n" + meta,
    )
    begin = vector.index("    idx_query_i8, idx_query_scale = indexer_query_quant_stage(\n")
    end_marker = "    signal = publish_phase(signal, PHASE_COEFF, core, CORES, generation)\n"
    end = vector.index(end_marker, begin) + len(end_marker)
    old_quant_coeff = vector[begin:end]
    fused = """    if FUSED_INDEXER_COEFF:
        # 每个AIV在此之前已发布WEIGHTS并获取META；前移获取不依赖QUANT。
        signal = wait_row_owners(signal, PHASE_WEIGHTS, CORES, generation, 0, tokens, 16, VECTORS, GROUPS, 1)
        idx_query_i8, idx_query_scale, coefficients = indexer_query_quant_coeff_stage(
            idx_query_had, idx_query_i8, idx_query_scale, tokens, worker, VECTORS,
            head_weights, idx_bounds, idx_lengths, positions, group_offsets, coefficients, QUERY_GROUP,
        )
        # 同一业务发布覆盖原INT8/scale及本token的64列对角系数片。
        signal = publish_phase(signal, PHASE_IDX_QUERY_QUANT, core, CORES, generation)
        signal = publish_phase(signal, PHASE_COEFF, core, CORES, generation)
    else:
"""
    source["vector_schedule.py"] = vector[:begin] + fused + textwrap.indent(old_quant_coeff, "    ") + vector[end:]
    old_acquire = (
        "    groups = pl.cast(pl.read(group_plan, [0]), pl.INDEX)\n"
        "    signal = wait_row_owners(signal, PHASE_COEFF, CORES, generation, 0, groups, 1, VECTORS, GROUPS, 1)\n"
    )
    source["cube_schedule.py"] = replace_once(
        source["cube_schedule.py"],
        old_acquire,
        "    if not FUSED_INDEXER_COEFF:\n" + textwrap.indent(old_acquire, "    "),
    )
    assert source["binding.py"] == original["binding.py"]
    assert source["graph_contract.py"] == original["graph_contract.py"]
    for text in source.values():
        ast.parse(text)
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    changed = []
    patch = []
    for name, text in sorted(source.items()):
        if text != original.get(name, ""):
            changed.append(name)
            (target / name).write_text(text)
            patch.extend(
                difflib.unified_diff(
                    original.get(name, "").splitlines(True),
                    text.splitlines(True),
                    fromfile="a/" + name,
                    tofile="b/" + name,
                )
            )
    (here / f"fused_coefficients_current_{version}.patch").write_text("".join(patch))
    report = {
        "source": str(target),
        "parent": "frozen_current_internal_workspace_20261025",
        "reference_prototype": str(previous / "fused_coefficients_v1.json"),
        "changed_files": changed,
        "binding_unchanged": True,
        "graph_contract_unchanged": True,
        "policy": "long_history and query_group == 6 and 64 <= token_capacity <= 192",
        "workspace": "debug/service两个入口各自在原group_offsets尾增加T个INT32；其余分区偏移不变",
        "mechanism": "token%48的量化owner直接写对角系数；QUANT发布覆盖两种payload，AIC不再单独获取COEFF",
        "status": "CPU_gate_pending",
        "device_execution": False,
        "adopted": False,
        "overlapping_query_intervals": "v2标记-2后由同一token owner逐组恢复原causal片；v1未覆盖，不得上卡"
        if version == "v2"
        else "CPU风险已发现：多重token映射未覆盖，不得上卡",
    }
    (here / f"fused_coefficients_current_{version}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
