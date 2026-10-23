# SPDX-License-Identifier: Apache-2.0
"""Fuse each token's query quantization and diagonal head coefficient output."""

import ast
import copy
import hashlib
import json
import runpy
import textwrap
from pathlib import Path

from prepare_scale_gather import replace_once


def forward_flag(source, names):
    lines = source.splitlines(keepends=True)
    calls = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names
    ]
    assert {n.func.id for n in calls} == set(names)
    for call in sorted(calls, key=lambda n: (n.end_lineno, n.end_col_offset), reverse=True):
        end = call.end_lineno - 1
        assert lines[end].strip() == ")"
        spaces = len(lines[end]) - len(lines[end].lstrip()) + 4
        lines.insert(end, " " * spaces + "FUSED_INDEXER_COEFF=FUSED_INDEXER_COEFF,\n")
    return "".join(lines)


def make_fused_quant(source):
    original = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "indexer_query_quant_stage"
    )
    fn = copy.deepcopy(original)
    fn.name = "indexer_query_quant_coeff_stage"
    additions = (
        ast.parse("""
def signature(
    weights: pl.Tensor[[WEIGHT_ROWS_DYN, HEADS], pl.FP32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    coefficients: pl.InOut[pl.Tensor[[COEFF_ROWS_DYN, COEFF_COLS_DYN], pl.FP16]],
    GROUP_SIZE: pl.constexpr,
):
    pass
""")
        .body[0]
        .args.args
    )
    fn.args.args.extend(additions)
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    loop.body.extend(
        ast.parse("""
map_base = pl.tensor.dim(offsets, 0) - tokens
encoded = pl.cast(pl.read(offsets, [map_base + idx]), pl.INDEX)
if encoded >= 0:
    group = encoded // GROUP_SIZE
    lane = encoded % GROUP_SIZE
    request, first, live, _visible = indexer_decode_group(
        bounds, lengths, positions, offsets, group, GROUP_SIZE
    )
    if lane < live and idx == first + lane:
        _visible, causal = indexer_query_visibility(bounds, lengths, positions, request, idx)
        if causal != 0:
            query_weights = weights[idx:idx + 1, 0:HEADS]
            query_scale_half = pl.cast(qh_scale_recip, pl.FP16, mode="rint")
            query_weight_half = pl.cast(query_weights, pl.FP16, mode="rint")
            head_coefficients = pl.mul(query_scale_half, query_weight_half)
            coefficients = pl.assemble(
                coefficients, head_coefficients, [group * COEFFICIENT_ROWS + lane, lane * HEADS]
            )
""").body
    )
    fn.body[-1] = ast.parse("return qr_hadamard_i8, qr_hadamard_scale_dq, coefficients").body[0]
    return ast.unparse(ast.fix_missing_locations(fn)) + "\n"


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "output_rectangle_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    source["fused_coefficients.py"] = (
        (here / "fused_coefficients_template.py").read_text()
        + "\n\n"
        + make_fused_quant(source["indexer_query_stages.py"])
    )
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
    policy = replace_once(
        policy,
        "        plan_table_columns=plan_table_columns,\n",
        "        plan_table_columns=plan_table_columns,\n"
        "        fused_indexer_coeff=(long_history and group == 6 "
        "and EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS),\n",
    )
    source["policy.py"] = policy
    binding = source["binding.py"]
    old = "_pad(_pad(b + 1, 16) + 4 * t, 16) + 48 + 4 * t + (2 * b if self.policy.scale_page_coalescing else 0)"
    source["binding.py"] = replace_once(
        binding, old, old + ' + (t if getattr(self.policy, "fused_indexer_coeff", False) else 0)'
    )
    for name, count in (("entry.py", 2), ("cube_schedule.py", 1), ("vector_schedule.py", 1)):
        marker = "    SCORE_GROUP_ORDER: pl.constexpr = False,\n"
        assert source[name].count(marker) == count
        source[name] = source[name].replace(marker, marker + "    FUSED_INDEXER_COEFF: pl.constexpr = False,\n")
    source["entry.py"] = forward_flag(source["entry.py"], {"cube_schedule", "vector_schedule", "_csa_core_group"})
    vector = source["vector_schedule.py"]
    vector = replace_once(
        vector,
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
    original = vector[begin:end]
    fused = """    if FUSED_INDEXER_COEFF:
        signal = wait_row_owners(signal, PHASE_WEIGHTS, CORES, generation, 0, tokens, 16, VECTORS, GROUPS, 1)
        idx_query_i8, idx_query_scale, coefficients = indexer_query_quant_coeff_stage(
            idx_query_had, idx_query_i8, idx_query_scale, tokens, worker, VECTORS,
            head_weights, idx_bounds, idx_lengths, positions, group_offsets, coefficients, QUERY_GROUP,
        )
        # QUANT now publishes the token's original data and diagonal coefficients.
        signal = publish_phase(signal, PHASE_IDX_QUERY_QUANT, core, CORES, generation)
        signal = publish_phase(signal, PHASE_COEFF, core, CORES, generation)
    else:
"""
    vector = vector[:begin] + fused + textwrap.indent(original, "    ") + vector[end:]
    source["vector_schedule.py"] = vector
    cube = source["cube_schedule.py"]
    old = (
        "    groups = pl.cast(pl.read(group_plan, [0]), pl.INDEX)\n"
        "    signal = wait_row_owners(signal, PHASE_COEFF, CORES, generation, 0, groups, 1, VECTORS, GROUPS, 1)\n"
    )
    source["cube_schedule.py"] = replace_once(
        cube, old, "    if not FUSED_INDEXER_COEFF:\n" + textwrap.indent(old, "    ")
    )
    target = here / "runs/fused_coefficients_v1"
    target.mkdir(exist_ok=False)
    for name, text in source.items():
        ast.parse(text)
        (target / name).write_text(text)
    select = runpy.run_path(str(target / "policy.py"))["select_policy"]
    cases = json.loads((here.parent / "csa_performance_20261022/cases.json").read_text())["cases"]
    selected = {}
    for case in cases:
        desc = json.loads((here / f"descriptors/{case['id']}_output_rectangle_v1.json").read_text())
        desc["policy"] = select(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=6,
            max_seq_len=max(case["seq_lens"]),
            compressed_table_columns=desc["buffers"]["compressed_table"]["shape"][1],
        ).constexprs()
        enabled = desc["policy"]["FUSED_INDEXER_COEFF"]
        selected[case["id"]] = enabled
        if enabled:
            desc["buffers"]["group_offsets"]["shape"][0] += case["token_capacity"]
        with (here / f"descriptors/{case['id']}_fused_coefficients_v1.json").open("x") as out:
            out.write(json.dumps(desc, indent=2) + "\n")
    report = dict(
        source=str(target.resolve()),
        parent="output_rectangle_production_v1.json",
        adopted=False,
        device_execution=False,
        selected=selected,
        mechanism="META zeros coefficient matrices and builds a T-word token map; original token%48 quantization "
        "writes diagonal coefficients from local scales; QUANT publication covers both payloads",
        unchanged=(
            "Quant arithmetic, INT8/FP32 scale outputs, causal guards, group geometry, phase IDs and root arguments"
        ),
        costs="Extra T INT32 workspace words and META initialization; weights acquisition moves before quantization",
        gate="Token-map isolation, exact coefficient/quant values, ownership, acquire/release, CPU resources, full L1",
        files={name: hashlib.sha256(text.encode()).hexdigest() for name, text in source.items()},
    )
    with (here / "fused_coefficients_v1.json").open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
