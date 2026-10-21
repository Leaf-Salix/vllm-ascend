# SPDX-License-Identifier: Apache-2.0
"""Freeze a separate B28 replay harness, leaving queued QA probes untouched."""

import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    here = Path(__file__).resolve().parent
    target = here / "runs/score_owner_harness_v89"
    target.mkdir(exist_ok=False)
    names = ["run_probe.py", "service_probe.py", "service_metadata_replay.py", "tnd_performance.py",
             "bsh_probe.py", "metadata_matrix.py", "submit_probe.py", "run_probe.sh"]
    originals = {name: (here / name).read_text() for name in names}
    revised = dict(originals)

    name = "run_probe.py"
    text = replace_once(originals[name], "    if plan.numel() == legacy_words + 64:\n", (
        "    if getattr(entry.policy, \"score_prefix_balance\", False):\n"
        "        from score_owner_diagnostics import inspect_score_owner_plan\n\n"
        "        owner_base = scale_base + 48 + 4 * tokens + 2 * requests\n"
        "        result[\"score_owner_permutation\"] = inspect_score_owner_plan(entry, plan, owner_base)\n"
        "    elif plan.numel() == legacy_words + 64:\n"
    ))
    revised[name] = text

    name = "metadata_matrix.py"
    text = originals[name]
    start = text.index("    if b1_only and (capacity")
    end = text.index("    if table_history is None", start)
    text = text[:start] + (
        "    if b1_only or continuous_only:\n"
        "        raise ValueError(\"本独立矩阵需完整大T检查，不接受B1或跳过metadata模式\")\n"
        "    if (capacity, requests, fixture[\"query_lengths\"]) != (168, 32, [6] * 28):\n"
        "        raise ValueError(\"需B28/S6、T容量168、请求容量32的初始捕获输入\")\n"
        "    if not getattr(candidate.policy, \"score_prefix_balance\", False):\n"
        "        raise ValueError(\"本矩阵必须实际编译Score重分配路径\")\n"
    ) + text[end:]
    text = replace_once(text, "    def new_case(queries, histories, owners=None, reverse=False):\n",
                        "    def new_case(queries, histories, owners=None, reverse=False,"
                        " page_pattern=\"original\"):\n")
    text = replace_once(text, "            allow_unassigned_tail=b1_only,\n", (
        "            # Native capacity may exceed Q[-1]; remaining rows have no request owner.\n"
        "            # Dummy rows still consume at most six tokens and retain seq_len=0.\n"
        "            allow_unassigned_tail=True,\n"
        "            indexer_page_pattern=page_pattern,\n"
    ))
    text = replace_once(text, '            if name not in ("signal", "ffts"):\n', (
        '            # Keep the prior map to test active -> identity -> active rewrites.\n'
        '            if name not in ("signal", "ffts", "group_offsets"):\n'
    ))
    start = text.index("    if b1_only:\n        cases =")
    end = text.index("    for name, queries, histories, reverse in cases:", start)
    text = text[:start] + (
        '    cases = [(f"uniform_s{q}", [q] * 28, [history + r % 4 for r in range(28)], False, "original")\n'
        '             for q in range(1, 7)]\n'
        '    cases.extend([\n'
        '        ("mixed_123456_histories", [1, 2, 3, 4, 5, 6] * 4 + [2, 6, 1, 5],\n'
        '         [8192 if r % 5 == 0 else 65536 if r % 5 == 1 else history + r % 4 for r in range(28)],\n'
        '         True, "fragmented_history"),\n'
        '        ("few_groups_identity", [6] * 23, [history] * 23, False, "original"),\n'
        '        ("active_again_ascending", [6] * 26 + [4, 3], [history + r % 4 for r in range(28)],\n'
        '         False, "ascending_history"),\n'
        '    ])\n'
    ) + text[end:]
    text = replace_once(text, "    for name, queries, histories, reverse in cases:\n"
                        "        case = new_case(queries, histories, reverse=reverse)\n", (
        "    for name, queries, histories, reverse, page_pattern in cases:\n"
        "        case = new_case(queries, histories, reverse=reverse, page_pattern=page_pattern)\n"
    ))
    marker = '        checked["passed"] &= all(value["status"] == "PASS" for value in baseline_guards.values())\n'
    text = replace_once(text, marker, marker + (
        '        mapping = checked["scale_writer_plan"]["score_owner_permutation"]\n'
        '        checked["passed"] &= mapping["groups"] == len(queries)\n'
        '        if name == "few_groups_identity":\n'
        '            checked["passed"] &= mapping["identity"]\n'
        '        if name in ("uniform_s6", "active_again_ascending"):\n'
        '            checked["passed"] &= not mapping["identity"]\n'
        '        checked["indexer_page_pattern"] = page_pattern\n'
        '        checked["mapping_workspace_preserved_between_replays"] = True\n'
    ))
    start = text.index("    if b1_only:\n        result[\"passed\"]")
    end = text.index("    # Request IDs own stable physical pages", start)
    text = text[:start] + text[end:]
    start = text.index("    history_by_id =")
    end = text.index("    original_qli =", start)
    text = text[:start] + (
        "    history_by_id = {request: history + request for request in range(30)}\n"
        "    steps = [\n"
        "        (list(range(28)), [1, 2, 3, 4, 5, 6] * 4 + [2, 6, 1, 5]),\n"
        "        (list(reversed(range(28))), [6] * 26 + [4, 3]),\n"
        "        (list(range(23)), [2, 6, 1, 4, 3, 5] * 3 + [2, 6, 1, 4, 3]),\n"
        "        ([27, 28, 29] + list(range(25)), [6] * 26 + [5, 4]),\n"
        "    ]\n"
    ) + text[end:]
    marker = '            result["continuous_steps"].append(checked)\n'
    text = replace_once(text, marker, (
        '            mapping = checked["scale_writer_plan"]["score_owner_permutation"]\n'
        '            checked["passed"] &= mapping["groups"] == len(ids)\n'
        '            if len(ids) < 24:\n'
        '                checked["passed"] &= mapping["identity"]\n'
        '            else:\n'
        '                checked["passed"] &= not mapping["identity"]\n'
        '            checked["mapping_workspace_preserved_between_replays"] = True\n'
    ) + marker)
    text = replace_once(text, '    result["passed"] = True\n', (
        '    result["passed"] = True\n'
        '    result["scope"] = "T168/Bcap32；28活动组和23组身份路径同址切换；含无请求归属尾部"\n'
    ))
    revised[name] = text

    name = "submit_probe.py"
    text = originals[name]
    start = text.index("    if args.metadata_matrix:\n")
    end = text.index("    if args.service_metadata_replay", start)
    text = text[:start] + (
        "    if args.metadata_matrix:\n"
        "        if args.b1_metadata_matrix or (args.batch, queries, token_capacity, request_capacity) != (\n"
        "            28, [6] * 28, 168, 32\n"
        "        ):\n"
        "            raise ValueError(\"Score矩阵初始输入要求B28/S6、T168、请求容量32，不能使用B1矩阵\")\n"
    ) + text[end:]
    text = replace_once(text, '    path = here / "jobs.json"\n', f'    path = Path({str(here / "jobs.json")!r})\n')
    revised[name] = text

    name = "run_probe.sh"
    text = originals[name]
    marker = 'mode = sys.argv[4]\n'
    text = replace_once(text, marker, marker + f'harness = Path({str(target)!r})\n')
    for filename in names:
        old = f'repo / "tests/pypto_test/csa_spmd_tnd_20261020/{filename}"'
        if old in text:
            text = replace_once(text, old, f'harness / "{filename}"')
    marker = '    repo / "tests/pypto_test/dsv4_csa_single_layer.py",\n'
    text = replace_once(text, marker, (
        '    harness / "dsv4_csa_single_layer.py",\n'
        '    harness / "score_owner_diagnostics.py",\n'
    ))
    revised[name] = text
    patch = []
    for name in names:
        if name.endswith(".py"):
            ast.parse(revised[name])
        (target / name).write_text(revised[name])
        if revised[name] != originals[name]:
            patch.extend(difflib.unified_diff(
                originals[name].splitlines(True), revised[name].splitlines(True),
                fromfile="a/" + name, tofile="b/" + name,
            ))
    shutil.copyfile(here.parent / "dsv4_csa_single_layer.py", target / "dsv4_csa_single_layer.py")
    shutil.copyfile(here / "score_owner_diagnostics.py", target / "score_owner_diagnostics.py")
    (here / "score_owner_harness_v89.patch").write_text("".join(patch))
    report = {
        "harness": str(target), "device_execution": False,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in target.iterdir()},
        "original_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in originals.items()},
        "unchanged_numerical_protocol_checks": (
            "candidate_run/collect/guards/geometry retained except preserving group_offsets"
        ),
        "unchanged_timer": "tnd_performance.py byte-equal; diagnostics outside all L1 windows",
        "initial_matrix": {"batch": 28, "tokens": 168, "request_capacity": 32, "query_tokens": 6},
        "scope": (
            "9 metadata cases plus4continuous steps; S1-S6, mixed histories/pages, active/identity/active ownership"
        ),
        "next": "CPU preflight/diagnostic tests, then first B28 S6 paired L1; device matrix requires actual execution",
    }
    (here / "score_owner_harness_v89.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("harness", "scope", "device_execution")}, indent=2))


if __name__ == "__main__":
    main()
