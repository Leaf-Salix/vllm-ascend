# SPDX-License-Identifier: Apache-2.0
"""Port only the reviewed WS/scale delta, preserving current TND fixes and tiers."""

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
    origin = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    external = here.parents[4] / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_roofline_optimization_20261020"
    base = external / "source_after_b1_half_v2"
    borrowed = external / "source_score_stream_guarded"
    target = here / "runs/borrowed_score_stream_candidate_v84"
    shutil.copytree(origin, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    changes = {}
    path = target / "indexer_score_stages.py"
    before = path.read_text()
    marker = "[TOPK_CANDIDATES_PER_LEAF // (2 * score_tile), score_tile]"
    replacement = "[TOPK_CANDIDATES_PER_LEAF // (2 * BUFFERED_LONG_SCORE_TILE), BUFFERED_LONG_SCORE_TILE]"
    external_base = (base / path.name).read_text()
    external_candidate = (borrowed / path.name).read_text()
    assert external_base.count(marker) == external_candidate.count(marker) == 2
    # The only local difference in this file is our inactive-B1 parser fix.
    # Reject the port if another session changes its base or our mathematics.
    assert external_base.replace(marker, replacement) == before
    changes[path.name] = (before, external_candidate.replace(marker, replacement))

    declaration = "    SPLIT_SCORE_ROWS: pl.constexpr = False,\n"
    forwarding = "            SPLIT_SCORE_ROWS,\n"
    path = target / "entry.py"
    before = path.read_text()
    assert before.count(declaration) == before.count(forwarding) == 2
    text = before.replace(declaration, declaration + "    SCORE_WS_POOL_PANELS: pl.constexpr = 1,\n")
    text = text.replace(forwarding, forwarding + "            SCORE_WS_POOL_PANELS,\n")
    changes[path.name] = (before, text)

    path = target / "cube_schedule.py"
    before = path.read_text()
    text = replace_once(before, declaration, declaration + "    SCORE_WS_POOL_PANELS: pl.constexpr = 1,\n")
    text = replace_once(text, "        SPLIT_SCORE_ROWS,\n",
                        "        SPLIT_SCORE_ROWS,\n        SCORE_WS_POOL_PANELS,\n")
    changes[path.name] = (before, text)

    path = target / "policy.py"
    before = path.read_text()
    text = replace_once(before, "    split_score_rows: bool = False\n",
                        "    split_score_rows: bool = False\n    score_ws_pool_panels: int = 1\n")
    text = replace_once(text, '            "SPLIT_SCORE_ROWS": self.split_score_rows,\n',
                        '            "SPLIT_SCORE_ROWS": self.split_score_rows,\n'
                        '            "SCORE_WS_POOL_PANELS": self.score_ws_pool_panels,\n')
    text = replace_once(text, "(parallel_small_qa or tokens > LOCAL_STATS_MAX_TOKENS)",
                        "(parallel_small_qa or tokens >= EARLY_GEOMETRY_MIN_TOKENS)")
    marker = "        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,\n"
    text = replace_once(text, marker, marker + (
        "        # Two 48KiB QK accumulators plus one 32KiB WS pool fill L0C.\n"
        "        score_ws_pool_panels=8 if (long_history and group == 6 and\n"
        "                                  EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS) else 1,\n"
    ))
    changes[path.name] = (before, text)
    patch = []
    for name, (before, after) in changes.items():
        ast.parse(after)
        (target / name).write_text(after)
        patch.extend(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), fromfile="a/" + name, tofile="b/" + name,
        ))
    (here / "borrowed_score_stream_v84.patch").write_text("".join(patch))
    report = {
        "source": str(target), "borrowed": str(borrowed), "changed_files": list(changes),
        "borrowed_score_sha256": hashlib.sha256(external_candidate.encode()).hexdigest(),
        "preserved": ["inactive-B1 typefix", "native150 waits", "B1 original half-leaf math",
                      "adopted output balance", "TND service and metadata binding"],
        "device_execution": False, "adopted": False,
        "required": "Fresh compile, active/disabled resource audit, full TND correctness and paired L1",
    }
    (here / "borrowed_score_stream_v84.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
