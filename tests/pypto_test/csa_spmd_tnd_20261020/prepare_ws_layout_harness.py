# SPDX-License-Identifier: Apache-2.0
"""Freeze the WS/scale layout replay harness without editing queued probes."""

import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path

from prepare_score_owner_harness import replace_once


def main():
    here = Path(__file__).resolve().parent
    target = here / "runs/ws_layout_harness_v90"
    target.mkdir(exist_ok=False)
    names = ["run_probe.py", "service_probe.py", "service_metadata_replay.py", "tnd_performance.py",
             "bsh_probe.py", "metadata_matrix.py", "submit_probe.py", "run_probe.sh"]
    originals = {name: (here / name).read_text() for name in names}
    revised = dict(originals)
    name = "run_probe.py"
    revised[name] = replace_once(originals[name], "    return result\n", (
        "    if entry.policy.scale_page_coalescing:\n"
        "        from scale_layout_diagnostics import inspect_scale_layout\n\n"
        "        result[\"scale_page_layout\"] = inspect_scale_layout(entry, plan, scale_base + 48 + 4 * tokens)\n"
        "    return result\n"
    ))

    name = "metadata_matrix.py"
    text = originals[name]
    start = text.index("    if b1_only and (capacity")
    end = text.index("    if table_history is None", start)
    text = text[:start] + (
        "    if b1_only or continuous_only:\n"
        "        raise ValueError(\"WS独立矩阵需完整metadata和连续state检查\")\n"
        "    if (capacity, requests, fixture[\"query_lengths\"]) != (65, 16, [6] * 6):\n"
        "        raise ValueError(\"需B6/S6、T容量65、请求容量16的初始输入\")\n"
        "    if not (candidate.policy.scale_page_coalescing and candidate.policy.balanced\n"
        "            and getattr(candidate.policy, \"score_ws_pool_panels\", 1) == 8):\n"
        "        raise ValueError(\"必须实际开启WS8列池和设备连续页证明\")\n"
    ) + text[end:]
    text = replace_once(text, "    def new_case(queries, histories, owners=None, reverse=False):\n",
                        "    def new_case(queries, histories, owners=None, reverse=False,"
                        " page_pattern=\"original\"):\n")
    text = replace_once(text, "            allow_unassigned_tail=b1_only,\n",
                        "            allow_unassigned_tail=False,\n            indexer_page_pattern=page_pattern,\n")
    text = replace_once(text, '            if name not in ("signal", "ffts"):\n', (
        '            # Retain prior layout words to expose missing fallback/inactive rewrites.\n'
        '            if name not in ("signal", "ffts", "group_offsets"):\n'
    ))
    start = text.index("    if b1_only:\n        cases =")
    end = text.index("    for name, queries, histories, reverse in cases:", start)
    text = text[:start] + (
        '    cases = [(f"uniform_s{q}", [q] * 6, [history + r for r in range(6)], False, "original")\n'
        '             for q in range(1, 7)]\n'
        '    cases.extend([\n'
        '        ("mixed_261", [2, 6, 1], [history + 3, history + 7, history + 11], True, "original"),\n'
        '        ("mixed_123456", [1, 2, 3, 4, 5, 6], [history + r * 3 for r in range(6)], False, "original"),\n'
        '        ("layout_ascending_fast", [2] * 6, [history] * 6, False, "ascending_history"),\n'
        '        ("layout_fragmented", [6] * 6, [history] * 6, False, "fragmented_history"),\n'
        '        ("layout_descending_restore", [6] * 6, [history] * 6, False, "original"),\n'
        '        ("layout_ascending_tail_fallback", [6] * 6, [history] * 6, False, "ascending_history"),\n'
        '    ])\n'
        '    expected_directions = {"layout_ascending_fast": 1, "layout_fragmented": 0,\n'
        '                           "layout_descending_restore": -1, "layout_ascending_tail_fallback": 0}\n'
    ) + text[end:]
    text = replace_once(text, "    for name, queries, histories, reverse in cases:\n"
                        "        case = new_case(queries, histories, reverse=reverse)\n", (
        "    for name, queries, histories, reverse, page_pattern in cases:\n"
        "        case = new_case(queries, histories, reverse=reverse, page_pattern=page_pattern)\n"
    ))
    marker = '        checked["passed"] &= all(value["status"] == "PASS" for value in baseline_guards.values())\n'
    text = replace_once(text, marker, marker + (
        '        layout = checked["scale_writer_plan"]["scale_page_layout"]\n'
        '        if name in expected_directions:\n'
        '            checked["passed"] &= all(\n'
        '                row["direction"] == expected_directions[name]\n'
        '                for row in layout["requests"] if row["active"]\n'
        '            )\n'
        '        checked["indexer_page_pattern"] = page_pattern\n'
        '        checked["layout_workspace_preserved_between_replays"] = True\n'
    ))
    start = text.index("    if b1_only:\n        result[\"passed\"]")
    end = text.index("    # Request IDs own stable physical pages", start)
    text = text[:start] + text[end:]
    revised[name] = text

    name = "submit_probe.py"
    text = originals[name]
    start = text.index("    if args.metadata_matrix:\n")
    end = text.index("    if args.service_metadata_replay", start)
    text = text[:start] + (
        "    if args.metadata_matrix:\n"
        "        if args.b1_metadata_matrix or (args.batch, queries, token_capacity, request_capacity) != (\n"
        "            6, [6] * 6, 65, 16\n"
        "        ):\n"
        "            raise ValueError(\"WS矩阵初始输入要求B6/S6、T65、请求容量16\")\n"
    ) + text[end:]
    text = replace_once(text, '    path = here / "jobs.json"\n', f'    path = Path({str(here / "jobs.json")!r})\n')
    revised[name] = text

    name = "run_probe.sh"
    text = replace_once(originals[name], 'mode = sys.argv[4]\n',
                        'mode = sys.argv[4]\n' + f'harness = Path({str(target)!r})\n')
    for filename in names:
        old = f'repo / "tests/pypto_test/csa_spmd_tnd_20261020/{filename}"'
        if old in text:
            text = replace_once(text, old, f'harness / "{filename}"')
    text = replace_once(text, '    repo / "tests/pypto_test/dsv4_csa_single_layer.py",\n', (
        '    harness / "dsv4_csa_single_layer.py",\n'
        '    harness / "scale_layout_diagnostics.py",\n'
    ))
    revised[name] = text
    patch = []
    for name in names:
        if name.endswith(".py"):
            ast.parse(revised[name])
        (target / name).write_text(revised[name])
        if originals[name] != revised[name]:
            patch.extend(difflib.unified_diff(originals[name].splitlines(True), revised[name].splitlines(True),
                                              fromfile="a/" + name, tofile="b/" + name))
    shutil.copyfile(here.parent / "dsv4_csa_single_layer.py", target / "dsv4_csa_single_layer.py")
    shutil.copyfile(here / "scale_layout_diagnostics.py", target / "scale_layout_diagnostics.py")
    (here / "ws_layout_harness_v90.patch").write_text("".join(patch))
    report = {
        "harness": str(target), "device_execution": False,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in target.iterdir()},
        "original_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in originals.items()},
        "initial_matrix": {"batch": 6, "tokens": 65, "request_capacity": 16, "query_tokens": 6},
        "scope": "12metadata+4continuous planned; actual ascending/fragmented/descending/fallback proof transitions",
        "timer": "tnd_performance.py byte-identical; all layout diagnostics outside measured execution",
        "next": "CPU fixture/diagnostic checks; device evidence remains required",
    }
    (here / "ws_layout_harness_v90.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("harness", "scope", "device_execution")}, indent=2))


if __name__ == "__main__":
    main()
