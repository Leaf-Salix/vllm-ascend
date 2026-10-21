# SPDX-License-Identifier: Apache-2.0
"""Port the reviewed Score ownership permutation without replacing TND changes."""

import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def function_node(text, name):
    return next(node for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def transplant_function(local, base, candidate, name):
    """Reject substantive local changes before borrowing a complete function."""
    current = function_node(local, name)
    previous = function_node(base, name)
    revised = function_node(candidate, name)
    assert ast.dump(current) == ast.dump(previous), name
    start = min([current.lineno] + [node.lineno for node in current.decorator_list]) - 1
    new_start = min([revised.lineno] + [node.lineno for node in revised.decorator_list]) - 1
    lines = local.splitlines(True)
    lines[start:current.end_lineno] = candidate.splitlines(True)[new_start:revised.end_lineno]
    return "".join(lines)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    origin = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    external = here.parents[4] / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_roofline_optimization_20261020"
    base = external / "source_after_b1_half_v2"
    borrowed = external / "source_score_owner_interval"
    target = here / "runs/borrowed_score_owner_candidate_v88"
    originals = {path.name: path.read_text() for path in origin.glob("*.py")}
    revised = dict(originals)
    for name, function, import_line in (
        ("vector_schedule.py", "vector_schedule",
         "from .score_owner_stages import score_balance_plan_stage, score_logical_worker\n"),
        ("indexer_publish_stages.py", "indexer_pair_publish_stage",
         "from .score_owner_stages import wait_balanced_score_roots\n"),
    ):
        revised[name] = transplant_function(
            originals[name], (base / name).read_text(), (borrowed / name).read_text(), function,
        )
        revised[name] = replace_once(revised[name], "import pypto.language as pl\n",
                                     "import pypto.language as pl\n\n" + import_line)

    name = "cube_schedule.py"
    text = replace_once(originals[name], "from .indexer_score_stages import indexer_score_cube_stage\n",
                        "from .indexer_score_stages import indexer_score_cube_stage\n"
                        "from .score_owner_stages import SCORE_BALANCE_BRANCH_GROUPS, score_logical_worker\n")
    text = replace_once(text, "    OUTPUT_WORK_BALANCE: pl.constexpr = False,\n",
                        "    OUTPUT_WORK_BALANCE: pl.constexpr = False,\n"
                        "    SCORE_PREFIX_BALANCE: pl.constexpr = False,\n")
    text = replace_once(text, "idx_normed, idx_had, wkv, kv_fp32, signal, worker, generation, 4,",
                        "idx_normed, idx_had, wkv, kv_fp32, signal, worker, generation, SCORE_BALANCE_BRANCH_GROUPS,")
    text = replace_once(text, "    indexer_transfer = indexer_score_cube_stage(\n", (
        "    score_worker = worker\n"
        "    if SCORE_PREFIX_BALANCE:\n"
        "        score_requests = pl.tensor.dim(idx_lengths, 0)\n"
        "        score_worker = score_logical_worker(group_offsets, score_requests, tokens, worker)\n"
        "    indexer_transfer = indexer_score_cube_stage(\n"
    ))
    text = replace_once(text, "        indexer_transfer,\n        worker,\n",
                        "        indexer_transfer,\n        score_worker,\n")
    revised[name] = text

    name = "entry.py"
    text = originals[name]
    declaration = "    OUTPUT_WORK_BALANCE: pl.constexpr = False,\n"
    forwarding = "            OUTPUT_WORK_BALANCE,\n"
    assert text.count(declaration) == text.count(forwarding) == 2
    text = text.replace(declaration, declaration + "    SCORE_PREFIX_BALANCE: pl.constexpr = False,\n")
    text = text.replace(forwarding, forwarding + "            SCORE_PREFIX_BALANCE,\n")
    text = replace_once(text, "            PLAN_PHYSICAL_ROWS,\n        )",
                        "            PLAN_PHYSICAL_ROWS,\n            SCORE_PREFIX_BALANCE,\n        )")
    text = replace_once(text,
                        "    SCALE_PAGE_COALESCING启用时尾部另加2*B项，关闭时保留原容量。\n",
                        "    连续scale或Score映射启用时尾部先保留2*B项；Score映射再追加96项。\n")
    revised[name] = text

    name = "binding.py"
    text = replace_once(originals[name], "from .policy import CSASPMDPolicy, select_policy\n",
                        "from .policy import CSASPMDPolicy, select_policy\n"
                        "from .score_owner_stages import SCORE_BALANCE_WORDS\n")
    text = replace_once(text,
                        "                 + (2 * b if self.policy.scale_page_coalescing else 0),),",
                        "                 + (2 * b if self.policy.scale_page_coalescing"
                        " or self.policy.score_prefix_balance else 0)\n"
                        "                 + (SCORE_BALANCE_WORDS if self.policy.score_prefix_balance else 0),),")
    text = replace_once(text,
                        "            # 启用连续页策略时尾部2*B项为(first_page,direction)，关闭时维持原容量。\n",
                        "            # 连续页或Score映射保留2*B项布局证明区；映射另追加四份24项表。\n")
    revised[name] = text

    name = "policy.py"
    text = replace_once(originals[name], "    output_work_balance: bool = False\n",
                        "    output_work_balance: bool = False\n    score_prefix_balance: bool = False\n")
    text = replace_once(text, '            "OUTPUT_WORK_BALANCE": self.output_work_balance,\n',
                        '            "OUTPUT_WORK_BALANCE": self.output_work_balance,\n'
                        '            "SCORE_PREFIX_BALANCE": self.score_prefix_balance,\n')
    marker = "        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,\n"
    text = replace_once(text, marker, marker + (
        "        # Two M64 KV bodies plus more than one M16 tail, within two Q_B M128 boxes.\n"
        "        # Actual group/leaf/visible imbalance is checked anew by the META writer.\n"
        "        score_prefix_balance=long_history and group == 6 and (2 * 64 + 16) < tokens <= 256,\n"
    ))
    revised[name] = text
    revised["score_owner_stages.py"] = (borrowed / "score_owner_stages.py").read_text()
    assert "pl.read(plan, [6]) != 0" in revised["score_owner_stages.py"]
    changed = [name for name, text in revised.items() if originals.get(name) != text]
    assert len(changed) == 7, changed
    for text in revised.values():
        ast.parse(text)
    # No existing target is ever overwritten. Device jobs retain immutable sources.
    shutil.copytree(origin, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patch = []
    for name in changed:
        (target / name).write_text(revised[name])
        patch.extend(difflib.unified_diff(
            originals.get(name, "").splitlines(True), revised[name].splitlines(True),
            fromfile="a/" + name if name in originals else "/dev/null", tofile="b/" + name,
        ))
    (here / "borrowed_score_owner_v88.patch").write_text("".join(patch))
    report = {
        "source": str(target), "borrowed": str(borrowed), "external_base": str(base),
        "changed_files": changed,
        "source_sha256": {path.name: sha256(path) for path in target.glob("*.py")},
        "external_sha256": {name: sha256(borrowed / name) for name in changed},
        "function_ast_preconditions": ["vector_schedule", "indexer_pair_publish_stage"],
        "preserved": ["inactive B1 shape fix", "native150 wait", "TND service and binding",
                      "adopted output balance", "original Score math/leaf/tie/GM slots",
                      "physical group FFTS handshake and phase publication"],
        "scope": "long G6, 144 < token capacity <= 256; device proves actual imbalance",
        "device_execution": False, "adopted": False,
        "required": "CPU active/disabled codegen, actual mapping/inverse checks, full TND replay and paired L1",
    }
    (here / "borrowed_score_owner_v88.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("source", "changed_files", "scope", "adopted")}, indent=2))


if __name__ == "__main__":
    main()
