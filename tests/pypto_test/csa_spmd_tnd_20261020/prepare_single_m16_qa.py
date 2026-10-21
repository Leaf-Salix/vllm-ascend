# SPDX-License-Identifier: Apache-2.0
"""Port a658188c's independent N64 QA ownership into the native single SPMD."""

import ast
import collections
import difflib
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    here = Path(__file__).resolve().parent
    origin = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    target = here / "runs/single_m16_qa_candidate_v83b"
    shutil.copytree(origin, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    changes = {}
    declaration = "    QA_M16_PARALLEL: pl.constexpr = False,\n"
    new_declaration = declaration + "    QA_SINGLE_M16: pl.constexpr = False,\n"

    path = target / "qkv_stages.py"
    before = path.read_text()
    text = replace_once(before, "QR_N_BLOCKS = Q_LORA // QR_N_TILE\n", (
        "QR_N_BLOCKS = Q_LORA // QR_N_TILE\n"
        "QA_SINGLE_M16_N_TILE = QR_N_TILE // 2\n"
        "QA_SINGLE_M16_OUTPUT_WORKERS = Q_LORA // QA_SINGLE_M16_N_TILE\n"
    ))
    start = text.index("def _q_a_m16_parallel_worker(")
    end = text.index("\n\n@pl.jit.inline", start)
    helper = text[start:end]
    helper = replace_once(helper, "    WORKERS: pl.constexpr,\n", (
        "    WORKERS: pl.constexpr,\n    N_TILE: pl.constexpr = QR_N_TILE,\n"
    ))
    split = helper.index("    tokens = pl.tensor.dim(x, 0)")
    helper = helper[:split] + helper[split:].replace("QR_N_BLOCKS", "(Q_LORA // N_TILE)").replace("QR_N_TILE", "N_TILE")
    text = text[:start] + helper + text[end:]
    text = replace_once(text, declaration, new_declaration)
    text = replace_once(text, "    if QA_M16_PARALLEL:\n", (
        "    if QA_SINGLE_M16:\n"
        "        qa_fp32 = _q_a_m16_parallel_worker(\n"
        "            x, wq_a, qa_fp32, worker_id, WORKERS, QA_SINGLE_M16_N_TILE\n"
        "        )\n"
        "    elif QA_M16_PARALLEL:\n"
    ))
    text = replace_once(
        text,
        "_q_a_m16_parallel_worker(x, wq_a, qa_fp32, worker_id, WORKERS)",
        "_q_a_m16_parallel_worker(x, wq_a, qa_fp32, worker_id, WORKERS, QR_N_TILE)",
    )
    changes[path.name] = (before, text)

    path = target / "cube_schedule.py"
    before = path.read_text()
    text = replace_once(before, declaration, new_declaration)
    text = replace_once(text,
                        "    qa = q_a_stage(normed, wq_a, qa, worker, GROUPS, QA_M16_PARALLEL)\n",
                        "    qa = q_a_stage(normed, wq_a, qa, worker, GROUPS, QA_M16_PARALLEL, QA_SINGLE_M16)\n")
    changes[path.name] = (before, text)

    path = target / "vector_schedule.py"
    before = path.read_text()
    text = replace_once(before, declaration, new_declaration)
    text = replace_once(text, "from .qkv_stages import q_r_stage\n",
                        "from .qkv_stages import QA_SINGLE_M16_OUTPUT_WORKERS, QR_M_TILE, q_r_stage\n")
    text = replace_once(text, "    if QA_M16_PARALLEL:\n", (
        "    if QA_SINGLE_M16:\n"
        "        # Every active row needs all sixteen N64 column owners.\n"
        "        qa_tasks = pl.min(\n"
        "            GROUPS, QA_SINGLE_M16_OUTPUT_WORKERS * ((tokens + QR_M_TILE - 1) // QR_M_TILE)\n"
        "        )\n"
        "        signal = wait_row_owners(signal, PHASE_QA, CORES, generation, 0, qa_tasks, 1, GROUPS, 0, 1)\n"
        "    elif QA_M16_PARALLEL:\n"
    ))
    changes[path.name] = (before, text)

    path = target / "entry.py"
    before = path.read_text()
    argument = "            QA_M16_PARALLEL,\n"
    assert before.count(declaration) == 2 and before.count(argument) == 3
    text = before.replace(declaration, new_declaration)
    text = text.replace(argument, argument + "            QA_SINGLE_M16,\n")
    changes[path.name] = (before, text)

    path = target / "policy.py"
    before = path.read_text()
    text = replace_once(before, "SCORE_SPLIT_MIN_TOKENS = 32\n", (
        "SCORE_SPLIT_MIN_TOKENS = 32\n"
        "# One physical M16 row block; independent N64 output columns fill 16 AICs.\n"
        "SINGLE_M16_QA_MAX_TOKENS = 16\n"
    ))
    text = replace_once(text, "    qa_m16_parallel: bool = False\n",
                        "    qa_m16_parallel: bool = False\n    qa_single_m16: bool = False\n")
    text = replace_once(text, '            "QA_M16_PARALLEL": self.qa_m16_parallel,\n',
                        '            "QA_M16_PARALLEL": self.qa_m16_parallel,\n'
                        '            "QA_SINGLE_M16": self.qa_single_m16,\n')
    text = replace_once(text, "        qa_m16_parallel=parallel_small_qa,\n",
                        "        qa_m16_parallel=parallel_small_qa,\n"
                        "        qa_single_m16=long_history and tokens <= SINGLE_M16_QA_MAX_TOKENS,\n")
    changes[path.name] = (before, text)

    patch = []
    for name, (before, after) in changes.items():
        ast.parse(after)
        (target / name).write_text(after)
        patch.extend(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), fromfile="a/" + name, tofile="b/" + name,
        ))
    (here / "single_m16_qa_v83b.patch").write_text("".join(patch))
    spec = importlib.util.spec_from_file_location("qa_single_m16_policy", target / "policy.py")
    policy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = policy
    spec.loader.exec_module(policy)
    ownership_cases = []
    for tokens in range(1, 17):
        writes = collections.Counter()
        for core in range(24):
            for job in range(core, 16, 24):
                for row in range(tokens):
                    for column in range(job * 64, (job + 1) * 64):
                        writes[(row, column)] += 1
        assert len(writes) == tokens * 1024 and set(writes.values()) == {1}
        ownership_cases.append({"tokens": tokens, "unique_output_elements": len(writes)})
    for tokens in (1, 6, 12, 16, 17, 24, 48, 64, 96, 168, 192):
        for maximum in (8198, 131078):
            selected = policy.select_policy(tokens=tokens, requests=(tokens + 5) // 6,
                                            max_query_tokens=6, max_seq_len=maximum)
            assert selected.qa_single_m16 == (tokens <= 16 and maximum == 131078)
    descriptors = {}
    for batch in (1, 16):
        run = ("output_work_balance_b1_s6_and_s1_s6_v67" if batch == 1
               else "output_work_balance_h131072_b16_s6_v67")
        data = json.loads((here / "runs" / run / "buffer_descriptors.json").read_text())
        data["policy"]["QA_SINGLE_M16"] = batch == 1
        path = here / "compile_descriptors" / f"single_m16_qa_b{batch}_v83.json"
        path.write_text(json.dumps(data, indent=2) + "\n")
        descriptors[str(batch)] = str(path)
    report = {
        "candidate": str(target), "source_commit": "a658188cde9c4d58081feb3a3fbe96307b21ff23",
        "origin_session": "CSA-多task", "changed_files": list(changes), "descriptors": descriptors,
        "ownership_cases": ownership_cases, "device_execution": False, "adopted": False,
        "scope": "Port N64 ownership and matching QR acquisition; keep one MIX and existing NORM/QA protocol",
        "not_ported": "Multi-task sync_start and extra registrations do not apply inside an existing single SPMD",
    }
    (here / "single_m16_qa_v83b.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "ownership_cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
