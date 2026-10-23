# SPDX-License-Identifier: Apache-2.0
"""Freeze descriptor-selected plan capacity while preserving the 2048-column tier."""

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from prepare_scale_gather import replace_once


def forward_capacity(source, names):
    lines = source.splitlines(keepends=True)
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names
    ]
    assert {node.func.id for node in calls} == set(names)
    for node in sorted(calls, key=lambda n: (n.end_lineno, n.end_col_offset), reverse=True):
        line = node.end_lineno - 1
        if lines[line].strip() == ")":
            padding = lines[line][: len(lines[line]) - len(lines[line].lstrip())] + "    "
            lines.insert(line, padding + "PLAN_TABLE_COLUMNS=PLAN_TABLE_COLUMNS,\n")
        else:
            offset = node.end_col_offset - 1
            assert lines[line][offset] == ")"
            lines[line] = lines[line][:offset] + ", PLAN_TABLE_COLUMNS=PLAN_TABLE_COLUMNS" + lines[line][offset:]
    return "".join(lines)


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "merged_schedule_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    plan = source["attention_plan_stages.py"]
    plan = replace_once(
        plan,
        "    ENABLED: pl.constexpr,\n",
        "    ENABLED: pl.constexpr,\n    PLAN_TABLE_COLUMNS: pl.constexpr = PHYSICAL_PLAN_COLUMNS,\n",
    )
    assert plan.count("    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n") == 2
    plan = plan.replace(
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n",
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n"
        "    PLAN_TABLE_COLUMNS: pl.constexpr = PHYSICAL_PLAN_COLUMNS,\n",
    )
    plan = replace_once(plan, "columns <= PHYSICAL_PLAN_COLUMNS", "columns <= PLAN_TABLE_COLUMNS")
    plan = replace_once(plan, "[1, PHYSICAL_PLAN_COLUMNS],", "[1, PLAN_TABLE_COLUMNS],")
    source["attention_plan_stages.py"] = forward_capacity(plan, {"physical_plan_supported"})

    stream = source["attention_stream_stages.py"]
    stream = replace_once(
        stream, "    COMPRESSED_COLUMNS_DYN,\n", "    COMPRESSED_COLUMNS_DYN,\n    PHYSICAL_PLAN_COLUMNS,\n"
    )
    stream = replace_once(
        stream,
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n",
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n"
        "    PLAN_TABLE_COLUMNS: pl.constexpr = PHYSICAL_PLAN_COLUMNS,\n",
    )
    source["attention_stream_stages.py"] = forward_capacity(stream, {"physical_plan_supported"})

    for name, count, calls in (
        ("entry.py", 2, {"vector_schedule", "_csa_core_group"}),
        (
            "vector_schedule.py",
            1,
            {"attention_geometry_prepare_stage", "attention_slot_plan_stage", "attention_qk_pv_vector_lane_stage"},
        ),
    ):
        text = source[name]
        text = replace_once(
            text,
            "import pypto.language as pl\n",
            "import pypto.language as pl\n\nfrom .attention_plan_stages import PHYSICAL_PLAN_COLUMNS\n",
        )
        marker = "    SCORE_GROUP_ORDER: pl.constexpr = False,\n"
        assert text.count(marker) == count
        text = text.replace(marker, marker + "    PLAN_TABLE_COLUMNS: pl.constexpr = PHYSICAL_PLAN_COLUMNS,\n")
        source[name] = forward_capacity(text, calls)

    policy = source["policy.py"]
    policy = replace_once(
        policy,
        "LEAF_LAYOUT_MAX_CACHE_ROWS = 6 * 8192\n",
        "LEAF_LAYOUT_MAX_CACHE_ROWS = 6 * 8192\n"
        "DEFAULT_PHYSICAL_PLAN_COLUMNS = 2048\nEXTENDED_PHYSICAL_PLAN_COLUMNS = 2080\n",
    )
    policy = replace_once(
        policy,
        "    score_group_order: bool = False\n",
        "    score_group_order: bool = False\n    plan_table_columns: int = DEFAULT_PHYSICAL_PLAN_COLUMNS\n",
    )
    policy = replace_once(
        policy,
        '            "SCORE_GROUP_ORDER": self.score_group_order,\n',
        '            "SCORE_GROUP_ORDER": self.score_group_order,\n'
        '            "PLAN_TABLE_COLUMNS": self.plan_table_columns,\n',
    )
    policy = replace_once(
        policy,
        "def select_policy(*, tokens: int, requests: int, max_query_tokens: int, max_seq_len: int) -> CSASPMDPolicy:\n",
        "def select_policy(*, tokens: int, requests: int, max_query_tokens: int, max_seq_len: int,\n"
        "                  compressed_table_columns: int | None = None) -> CSASPMDPolicy:\n",
    )
    policy = replace_once(
        policy,
        "    values = (tokens, requests, max_query_tokens, max_seq_len)\n",
        "    if compressed_table_columns is not None:\n"
        "        if type(compressed_table_columns) is not int or compressed_table_columns <= 0:\n"
        '            raise ValueError("compressed_table_columns must be a positive allocated dimension")\n'
        "    values = (tokens, requests, max_query_tokens, max_seq_len)\n",
    )
    policy = replace_once(
        policy,
        "    return CSASPMDPolicy(\n",
        "    plan_physical_rows = tokens >= EARLY_GEOMETRY_MIN_TOKENS\n"
        "    plan_table_columns = DEFAULT_PHYSICAL_PLAN_COLUMNS\n"
        "    if (plan_physical_rows and compressed_table_columns is not None\n"
        "            and DEFAULT_PHYSICAL_PLAN_COLUMNS < compressed_table_columns <= EXTENDED_PHYSICAL_PLAN_COLUMNS):\n"
        "        plan_table_columns = EXTENDED_PHYSICAL_PLAN_COLUMNS\n"
        "    return CSASPMDPolicy(\n",
    )
    policy = replace_once(
        policy,
        "        plan_physical_rows=tokens >= EARLY_GEOMETRY_MIN_TOKENS,\n",
        "        plan_physical_rows=plan_physical_rows,\n        plan_table_columns=plan_table_columns,\n",
    )
    source["policy.py"] = policy

    binding = source["binding.py"]
    binding = replace_once(
        binding,
        "        self.policy = policy or select_policy(\n",
        '        if native["cmp_block_table"].ndim != 2:\n'
        '            raise ValueError("cmp_block_table must be two-dimensional")\n'
        "        self.policy = policy or select_policy(\n",
    )
    binding = replace_once(
        binding,
        "            tokens=t, requests=b, max_query_tokens=max_query_tokens, max_seq_len=max_seq_len\n",
        "            tokens=t, requests=b, max_query_tokens=max_query_tokens, max_seq_len=max_seq_len,\n"
        '            compressed_table_columns=native["cmp_block_table"].shape[1],\n',
    )
    source["binding.py"] = binding
    service = source["service.py"]
    service = replace_once(
        service,
        "        policy = self._select_policy(\n            tokens=tokens,\n"
        "            requests=indexer.decode.seq_lens.numel(),\n",
        '        compressed_table = metadata[self.prefixes["compressed"]].decode.block_table\n'
        "        if compressed_table is None or compressed_table.ndim != 2:\n            return None\n"
        "        policy = self._select_policy(\n            tokens=tokens,\n"
        "            requests=indexer.decode.seq_lens.numel(),\n",
    )
    service = replace_once(
        service,
        "            max_seq_len=indexer.decode.max_seq_lens,\n",
        "            max_seq_len=indexer.decode.max_seq_lens,\n"
        "            compressed_table_columns=compressed_table.shape[1],\n",
    )
    service = replace_once(
        service,
        "            max_seq_len=maximum,\n        )\n",
        "            max_seq_len=maximum,\n"
        '            compressed_table_columns=native["cmp_block_table"].shape[1],\n        )\n',
    )
    source["service.py"] = service
    for content in source.values():
        ast.parse(content)
    target = here / "runs/physical_capacity_tier_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        (target / name).write_text(content)

    spec = importlib.util.spec_from_file_location("physical_capacity_tier_policy", target / "policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cases = json.loads((here.parent / "csa_performance_20261022/cases.json").read_text())["cases"]
    selected = {}
    for case in cases:
        description = json.loads((here / f"descriptors/{case['id']}_merged_schedule_v1.json").read_text())
        args = dict(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=6,
            max_seq_len=max(case["seq_lens"]),
            compressed_table_columns=description["buffers"]["compressed_table"]["shape"][1],
        )
        constexprs = module.select_policy(**args).constexprs()
        assert {k: v for k, v in constexprs.items() if k != "PLAN_TABLE_COLUMNS"} == description["policy"]
        selected[case["id"]] = constexprs["PLAN_TABLE_COLUMNS"]
        description["policy"] = constexprs
        with (here / f"descriptors/{case['id']}_physical_capacity_tier_v1.json").open("x") as stream:
            stream.write(json.dumps(description, indent=2) + "\n")
    assert selected == {f"C{i:02}": 2080 if i == 2 else 2048 for i in range(1, 9)}
    report = dict(
        source=str(target),
        parent="merged_schedule_v1.json",
        adopted=False,
        device_execution=False,
        mechanism="Select plan table capacity from allocated compressed-table columns; "
        "Q/L and addresses remain device-side",
        selected_capacities=selected,
        gate="C02 emitted code must match the measured 2080 prototype; all other cases must match retained 2048 code",
        files={name: hashlib.sha256(content.encode()).hexdigest() for name, content in source.items()},
    )
    with (here / "physical_capacity_tier_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
