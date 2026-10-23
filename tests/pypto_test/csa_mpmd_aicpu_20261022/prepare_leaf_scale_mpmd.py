# SPDX-License-Identifier: Apache-2.0
"""在既有MPMD叶内证明中接入SPMD 79fae6db0的一有效行策略，不增加规划任务。"""

import difflib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NAME = "aicpu_v37_single_row_leaf_scale"


def change(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)


def main():
    subprocess.run(["python3", str(HERE / "prepare.py"), NAME], check=True)
    experiment = HERE / "experiments" / NAME
    operator = experiment / "operator"
    path = operator / "indexer_scale_layout_stages.py"
    text = path.read_text()
    text = change(
        text,
        "    physical_pages: pl.Scalar[pl.INDEX],\n",
        "    physical_pages: pl.Scalar[pl.INDEX],\n    VALID_PROOF_ROWS: pl.constexpr = LAYOUT_PROOF_ROWS,\n",
    )
    text = change(
        text,
        "                                LAYOUT_PROOF_ROWS,\n",
        "                                VALID_PROOF_ROWS,\n",
    )
    text = change(
        text,
        "pl.col_expand_mul(ones, difference), LAYOUT_PROOF_ROWS, pages",
        "pl.col_expand_mul(ones, difference), VALID_PROOF_ROWS, pages",
    )
    text = change(
        text,
        "每个叶的两条AIV各自校验，不复用上一请求、上一叶或上次图重放的结果。",
        "每个叶的两条AIV各自校验，不复用上一请求、上一叶或上次图重放的结果。\n"
        "物理布局保持8×256，策略可只计算一个有效行；仍逐页完整检查，失败回退逐页读取。",
    )
    path.write_text(text)
    path = operator / "indexer_score_stages.py"
    text = path.read_text()
    text = change(
        text,
        "    scale_page_coalescing: pl.constexpr,\n",
        "    scale_page_coalescing: pl.constexpr,\n    SCALE_PROOF_ROWS: pl.constexpr,\n",
    )
    text = change(
        text,
        "                        pl.tensor.dim(idx_native_kv_cache, 0),\n",
        "                        pl.tensor.dim(idx_native_kv_cache, 0),\n                        SCALE_PROOF_ROWS,\n",
    )
    path.write_text(text)
    path = operator / "entry.py"
    text = path.read_text()
    text = change(
        text,
        "    SCALE_PAGE_COALESCING: pl.constexpr,\n",
        "    SCALE_PAGE_COALESCING: pl.constexpr,\n    SCALE_PROOF_ROWS: pl.constexpr,\n",
        2,
    )
    text = change(
        text,
        "    SCALE_PAGE_COALESCING: pl.constexpr = False,\n",
        "    SCALE_PAGE_COALESCING: pl.constexpr = False,\n    SCALE_PROOF_ROWS: pl.constexpr = 8,\n",
    )
    text, count = re.subn(
        r"(?m)^( +)SCALE_PAGE_COALESCING,$", r"\1SCALE_PAGE_COALESCING,\n\1SCALE_PROOF_ROWS,", text
    )
    assert count == 3
    path.write_text(text)
    path = operator / "policy.py"
    text = path.read_text()
    text = change(
        text,
        "NONEMPTY_WORKER_MAX_TOKENS = 64\n",
        "NONEMPTY_WORKER_MAX_TOKENS = 64\n"
        "# SPMD已验证的一有效行叶内证明域，最多六个原始8192候选叶。\n"
        "LEAF_LAYOUT_MAX_CACHE_ROWS = 6 * 8192\n",
    )
    text = change(
        text,
        "    scale_page_coalescing: bool = False\n",
        "    scale_page_coalescing: bool = False\n    scale_proof_rows: int = 8\n",
    )
    text = change(
        text,
        '            "SCALE_PAGE_COALESCING": self.scale_page_coalescing,\n',
        '            "SCALE_PAGE_COALESCING": self.scale_page_coalescing,\n'
        '            "SCALE_PROOF_ROWS": self.scale_proof_rows,\n',
    )
    text = change(
        text,
        "    parallel_small_qa = group == 6 and 16 < tokens < 32\n",
        "    parallel_small_qa = group == 6 and 16 < tokens < 32\n"
        "    single_row_leaf_scale = (\n"
        "        long_history\n        and group == 6\n"
        "        and EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS\n"
        "        and cache_rows <= LEAF_LAYOUT_MAX_CACHE_ROWS\n    )\n",
    )
    text = change(
        text,
        "        scale_page_coalescing=(long_history and group == 6 and "
        "(parallel_small_qa or tokens > LOCAL_STATS_MAX_TOKENS)),",
        "        scale_page_coalescing=(\n"
        "            single_row_leaf_scale\n"
        "            or (long_history and group == 6 and (parallel_small_qa or tokens > LOCAL_STATS_MAX_TOKENS))\n"
        "        ),\n        scale_proof_rows=1 if single_row_leaf_scale else 8,",
    )
    path.write_text(text)
    files = ["indexer_scale_layout_stages.py", "indexer_score_stages.py", "entry.py", "policy.py"]
    patch = []
    for filename in files:
        relative = f"vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd/{filename}"
        patch.extend(
            difflib.unified_diff(
                (REPO / relative).read_text().splitlines(True),
                (operator / filename).read_text().splitlines(True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    (HERE / "scheduling_v37_leaf_scale.patch").write_text("".join(patch))
    manifest = json.loads((experiment / "manifest.json").read_text())
    manifest.update(
        candidate=NAME,
        optimization_reference_commit="79fae6db0",
        changed_operator_files=files,
        scope="Existing leaf-local proof, one valid row only in validated descriptor tier; no GM proof table or task",
        fixed_current_and_legacy_cases_unchanged=True,
        frozen=False,
    )
    (experiment / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(experiment)


if __name__ == "__main__":
    main()
