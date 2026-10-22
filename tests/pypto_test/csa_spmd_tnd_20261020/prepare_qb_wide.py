# SPDX-License-Identifier: Apache-2.0
"""Try fewer Q_B column work items in the existing small-token resource domain."""

import ast
import difflib
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from prepare_score_owner_harness import replace_once
from prepare_short_half_ub import append_argument, append_parameter, function


def main():
    here = Path(__file__).resolve().parent
    source = here / "runs/borrowed_combined_candidate_v106"
    target = here / "runs/qb_wide_candidate_v108b"
    assert not target.exists()
    originals = {p.name: p.read_text() for p in source.glob("*.py")}
    revised = {}
    text = originals["query_projection_stages.py"]
    old = function(text, "_q_b_nz_stage")
    lines = text.splitlines(keepends=True)
    helper = "".join(lines[old.decorator_list[0].lineno - 1:old.end_lineno])
    helper = helper.replace("_q_b_nz_stage", "_q_b_nz_wide_stage")
    for before, after in (
        ("QPROJ_PIPE_M_TILE", "QB_WIDE_M_TILE"),
        ("QPROJ_PIPE_K_TILE", "QB_WIDE_K_TILE"),
        ("QPROJ_MM_N_TILE", "QB_WIDE_N_TILE"),
        ("QPROJ_N_BLOCKS", "QB_WIDE_N_BLOCKS"),
    ):
        helper = helper.replace(before, after)
    helper = helper.replace("原 K 流水不变", "小容量 M64/N512/K128 流水")
    helper = helper.replace("M128", "M64").replace("不足128行", "不足64行")
    helper = helper.replace("后三段按 K256 原顺序追加", "后七段按 K128 递增追加，INT32 点积无溢出")
    # Keep the original implementation as the exact compile-time fallback.
    begin, end = old.body[1].lineno - 1, old.body[-1].lineno - 1
    block = lines[begin:end]
    lines[begin:end] = [
        "    if QB_WIDE_N:\n",
        "        q_proj_i32 = _q_b_nz_wide_stage(\n",
        "            wq_b, qr_i8_matmul, q_proj_i32, tokens, worker_id, WORKERS\n",
        "        )\n",
        "    else:\n",
        *("    " + line if line.strip() else line for line in block),
    ]
    text = "".join(lines)
    text = append_parameter(text, "_q_b_nz_stage", "QB_WIDE_N")
    text = append_parameter(text, "_q_b_nd_stage", "QB_WIDE_N")
    text = replace_once(text, "@pl.jit.inline(auto_scope=False)\ndef _q_b_nz_stage(",
                        helper + "\n\n\n@pl.jit.inline(auto_scope=False)\ndef _q_b_nz_stage(")
    text = replace_once(text, "QPROJ_PIPE_K_TILE = 256\n",
                        "QPROJ_PIPE_K_TILE = 256\n"
                        "# Small-token Q_B experiment: one accumulator <=128KiB, each Right tile <=64KiB.\n"
                        "QB_WIDE_M_TILE = 64\nQB_WIDE_N_TILE = 512\nQB_WIDE_K_TILE = 128\n"
                        "QB_WIDE_N_BLOCKS = H * HEAD_DIM // QB_WIDE_N_TILE\n")
    revised["query_projection_stages.py"] = text
    text = append_parameter(originals["cube_schedule.py"], "cube_schedule", "QB_WIDE_N")
    text = replace_once(text,
                        "q_b_stage(wq_b, qr_matmul, q_proj, tokens, worker, GROUPS)",
                        "q_b_stage(wq_b, qr_matmul, q_proj, tokens, worker, GROUPS, QB_WIDE_N)")
    revised["cube_schedule.py"] = append_argument(text, "projection_overlap_stage", "QB_WIDE_N")
    text = append_parameter(originals["projection_overlap_stages.py"], "projection_overlap_stage", "QB_WIDE_N")
    revised["projection_overlap_stages.py"] = replace_once(
        text,
        "q_b_stage(wq_b, qr_matmul, q_proj, tokens, qb_worker, GROUPS - BRANCH_GROUPS)",
        "q_b_stage(wq_b, qr_matmul, q_proj, tokens, qb_worker, GROUPS - BRANCH_GROUPS, QB_WIDE_N)",
    )
    text = originals["entry.py"]
    for name in ("_csa_core_group", "decode_csa_spmd"):
        text = append_parameter(text, name, "QB_WIDE_N")
    for name in ("_csa_core_group", "cube_schedule"):
        text = append_argument(text, name, "QB_WIDE_N")
    revised["entry.py"] = text
    text = replace_once(originals["policy.py"], "    short_half_score_ub: bool = False\n",
                        "    short_half_score_ub: bool = False\n    qb_wide_n: bool = False\n")
    text = replace_once(text, '            "SHORT_HALF_SCORE_UB": self.short_half_score_ub,\n',
                        '            "SHORT_HALF_SCORE_UB": self.short_half_score_ub,\n'
                        '            "QB_WIDE_N": self.qb_wide_n,\n')
    text = replace_once(text, "        ob_rows=ob_rows,\n",
                        "        ob_rows=ob_rows,\n"
                        "        # Does not infer real query lengths; only the existing Host token-capacity tier.\n"
                        "        qb_wide_n=tokens < EARLY_GEOMETRY_MIN_TOKENS,\n")
    revised["policy.py"] = text
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patch = []
    for name, text in revised.items():
        ast.parse(text)
        (target / name).write_text(text)
        patch.extend(difflib.unified_diff(originals[name].splitlines(True), text.splitlines(True),
                                         fromfile="a/" + name, tofile="b/" + name))
    spec = importlib.util.spec_from_file_location("qb_wide_policy", target / "policy.py")
    policy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = policy
    spec.loader.exec_module(policy)
    descriptors = []
    for batch, history in ((1, 131072), (8, 131072), (16, 131072)):
        descriptor = here / "compile_descriptors" / f"borrowed_combined_v105_h{history}_b{batch}.json"
        desc = json.loads(descriptor.read_text())
        selected = policy.select_policy(tokens=batch * 6, requests=batch, max_query_tokens=6, max_seq_len=history + 6)
        assert selected.qb_wide_n == (batch in (1, 8))
        desc["policy"] = selected.constexprs()
        path = here / "compile_descriptors" / f"qb_wide_h{history}_b{batch}_v108b.json"
        path.write_text(json.dumps(desc, indent=2) + "\n")
        descriptors.append(dict(batch=batch, history=history, path=str(path), active=selected.qb_wide_n))
    report = dict(source=str(target), base=str(source), changed_files=list(revised),
                  descriptors=descriptors, adopted=False, device_execution=False,
                  integer_dot_bound=1024 * 128 * 128,
                  source_manifest={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
                  rationale="Q_B work items128->64; M64/K128 preserves INT32 dot products and all publishers",
                  required="Full CPU resource/code audit, exact full chain and paired L1 against current TNDv106")
    (here / "qb_wide_v108b.json").write_text(json.dumps(report, indent=2) + "\n")
    (here / "qb_wide_v108b.patch").write_text("".join(patch))
    print(target)


if __name__ == "__main__":
    main()
