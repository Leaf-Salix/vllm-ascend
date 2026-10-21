# SPDX-License-Identifier: Apache-2.0
"""Balance existing O_B jobs against O_A ownership, without changing mathematics."""

import ast
import collections
import difflib
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def prove_ownership():
    """Enumerate actual helper loop bounds and compare with the output grid."""
    cases = []
    for tokens in range(1, 97):
        owners = collections.defaultdict(list)
        counts = [0] * 24
        for physical in range(24):
            for group in range(8):
                virtual = physical + 16 if physical < 16 else physical - 16 + (group % 2) * 8
                first = group * 8
                owned = virtual + ((first + 31 - virtual) // 32) * 32
                if owned < first + 8:
                    for wave in range(first // 32, (first + 8 - virtual + 31) // 32):
                        job = virtual + wave * 32
                        if job >= first:
                            assert job // 8 == group
                            owners[job].append(physical)
                            counts[physical] += 1
        assert set(owners) == set(range(64))
        assert all(len(v) == 1 for v in owners.values())
        assert counts == [2] * 16 + [4] * 8
        # The unchanged HC_post waits for all 24 physical writers. Every
        # writer publishes once only after completing all its new O_B jobs.
        assert set(v[0] for v in owners.values()) == set(range(24))
        oa_counts = [len(range(core, 64, 24)) for core in range(24)]
        cases.append({"tokens": tokens, "oa_jobs": oa_counts, "ob_jobs": counts})
    return {"passed": True, "cases": cases, "scope": "Coverage/ownership only, not device or latency proof"}


def main():
    here = Path(__file__).resolve().parent
    origin = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    target = here / "runs/output_work_balance_native_candidate_v1"
    shutil.copytree(origin, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    changes = {}

    path = target / "cube_schedule.py"
    before = path.read_text()
    text = before.replace(
        "    SPLIT_SCORE_ROWS: pl.constexpr = False,\n",
        "    SPLIT_SCORE_ROWS: pl.constexpr = False,\n    OUTPUT_WORK_BALANCE: pl.constexpr = False,\n",
    )
    start = text.index("        for ob_group in pl.range(8):")
    end = text.index("    else:\n        # 保留整阶段路径", start)
    old_loop = text[start:end]
    balanced = '''        if OUTPUT_WORK_BALANCE:
            # In the existing N128, single-M128 domain, O_A gives cores 0..15
            # three jobs and cores 16..23 two. Use 32 logical O_B ranks on the
            # same 24 physical AICs: respectively two/four independent jobs.
            # These are software ranks, not additional SPMD participants.
            for ob_group in pl.range(8):
                ob_worker = worker + 16
                if worker >= 16:
                    ob_worker = worker - 16 + (ob_group % 2) * 8
                ob_first = ob_group * 8
                ob_owned = ob_worker + ((ob_first + 31 - ob_worker) // 32) * 32
                if ob_owned < ob_first + 8:
                    signal = wait_row_owners(
                        signal, PHASE_O_QUANT_GROUP_BASE + ob_group, CORES, generation,
                        ob_group * quant_per_group, quant_per_group, 1, VECTORS, GROUPS, 1
                    )
                    o_partials = o_b_groups_stage(
                        o_quant, wo_b, o_partials, tokens, ob_worker, 32, OB_ROWS, ob_group, ob_group + 1
                    )
        else:
'''
    text = text[:start] + balanced + "".join("    " + line for line in old_loop.splitlines(True)) + text[end:]
    changes[path.name] = (before, text)

    path = target / "entry.py"
    before = path.read_text()
    declaration = "    SPLIT_SCORE_ROWS: pl.constexpr = False,\n"
    argument = "            SPLIT_SCORE_ROWS,\n"
    assert before.count(declaration) == before.count(argument) == 2
    text = before.replace(declaration, declaration + "    OUTPUT_WORK_BALANCE: pl.constexpr = False,\n")
    text = text.replace(argument, argument + "            OUTPUT_WORK_BALANCE,\n")
    changes[path.name] = (before, text)

    path = target / "policy.py"
    before = path.read_text()
    field = "    split_score_rows: bool = False\n"
    setting = '            "SPLIT_SCORE_ROWS": self.split_score_rows,\n'
    assert before.count(field) == before.count(setting) == 1
    text = before.replace(field, field + "    output_work_balance: bool = False\n")
    text = text.replace(setting, setting + '            "OUTPUT_WORK_BALANCE": self.output_work_balance,\n')
    marker = "        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,\n"
    assert text.count(marker) == 1
    text = text.replace(marker, marker + "        output_work_balance=tokens <= LOCAL_STATS_MAX_TOKENS,\n")
    changes[path.name] = (before, text)

    patch = []
    for name, (before, text) in changes.items():
        ast.parse(text)
        (target / name).write_text(text)
        patch.extend(difflib.unified_diff(
            before.splitlines(True), text.splitlines(True), fromfile="a/" + name, tofile="b/" + name,
        ))
    (here / "output_work_balance_v1.patch").write_text("".join(patch))
    proof = prove_ownership()
    spec = importlib.util.spec_from_file_location("output_balance_policy_check", target / "policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for tokens in (1, 6, 16, 24, 48, 64, 96, 97, 128, 144, 168, 192):
        policy = module.select_policy(tokens=tokens, requests=(tokens + 5) // 6,
                                      max_query_tokens=6, max_seq_len=131078)
        assert policy.output_work_balance == (policy.oa_cols == 128)
    descriptions = {}
    for batch, run in ((1, "prefix_policy_b1_s1_s6_a241_v43"), (16, "cohort_prefix_a241_h131072_b16_s6_v41")):
        desc = json.loads((here / "runs" / run / "buffer_descriptors.json").read_text())
        desc["policy"]["OUTPUT_WORK_BALANCE"] = True
        output = here / "compile_descriptors" / f"output_work_balance_b{batch}.json"
        output.write_text(json.dumps(desc, indent=2) + "\n")
        descriptions[str(batch)] = str(output)
    proof.update(candidate=str(target), descriptors=descriptions, changed_files=list(changes),
                 device_execution=False, adopted=False)
    (here / "output_work_balance_v1.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
