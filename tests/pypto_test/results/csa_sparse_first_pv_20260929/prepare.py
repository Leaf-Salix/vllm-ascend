"""隔离最新AscendC首PV块跳过零累加项的候选，不改公开算子。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-native-inplace-seven-20260929"
PREFIX = WORKSPACE / ".cache/csa-sparse-first-pv-55b89ee2"
OLD_PACKAGE = "dsv4_csa_native_inplace_seven_20260929"
PACKAGE = "dsv4_csa_sparse_first_pv_55b89ee2"


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("禁止修改已入队候选")
    for side in ("baseline", "candidate"):
        dest = Path(f"{PREFIX}-{side}")
        shutil.copytree(BASE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        packages = dest / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
    relative = Path("vllm_ascend/ops/pypto") / PACKAGE / "decode_sparse_attn_csa.py"
    path = Path(f"{PREFIX}-candidate") / relative
    before = path.read_text()
    start = before.index("                        alpha = pl.exp(pl.sub(m_iter, next_m))")
    stop = before.index("                    else:\n"
                        "                        m_valid, l_valid, left_valid, right_valid", start)
    replacement = '''                        beta = pl.exp(pl.sub(pv_m, next_m))
                        pv_left = pl.load(
                            pv_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, HEAD_DIM // 2]
                        )
                        pv_right = pl.load(
                            pv_transfer, [pv_transfer_row + qk_lane_head, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2]
                        )
                        if pv_sb == 0:
                            # AscendC DealBmm2ResBaseBlock skips the previous-PV
                            # rescale on the first S2 block. PTO still needs beta:
                            # its local maximum may be below the attention sink.
                            first_l = pl.mul(beta, pv_l)
                            first_left = pl.row_expand_mul(pv_left, beta)
                            first_right = pl.row_expand_mul(pv_right, beta)
                            next_l, next_left, next_right = pl.yield_(first_l, first_left, first_right)
                        else:
                            alpha = pl.exp(pl.sub(m_iter, next_m))
                            update_l = pl.add(pl.mul(alpha, l_iter), pl.mul(beta, pv_l))
                            update_left = pl.add(pl.row_expand_mul(left_iter, alpha),
                                                 pl.row_expand_mul(pv_left, beta))
                            update_right = pl.add(pl.row_expand_mul(right_iter, alpha),
                                                  pl.row_expand_mul(pv_right, beta))
                            next_l, next_left, next_right = pl.yield_(update_l, update_left, update_right)
                        m_valid, l_valid, left_valid, right_valid = pl.yield_(
                            next_m, next_l, next_left, next_right
                        )
'''
    after = before[:start] + replacement + before[stop:]
    ast.parse(after)
    path.write_text(after)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile=f"a/{relative}", tofile=f"b/{relative}")))
    source = {
        "baseline": "55b89ee2", "source_prefix": str(PREFIX), "base_source": str(BASE),
        "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
        "native_reference": "ops-transformer 28f40354 SASVectorBlock::DealBmm2ResBaseBlock !isFirstSInnerLoop",
        "arithmetic": "首PV块省略alpha乘零及加零；保留beta、sink、softmax与舍入，需零容差完整状态检查",
        "inplace_pass": True, "scope": "独立PTO核内A/B，不替代同时在跑的Native七档基线",
    }
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    origin = ROOT.parent / "csa_sparse_final_publish_fixed_pair_20260929"
    for name in ("run.sh", "run_side.sh"):
        text = (origin / name).read_text()
        text = text.replace("csa_sparse_final_publish_fixed_pair_20260929", ROOT.name)
        text = text.replace("csa-sparse-final-publish-fix-4ffccb7b", "csa-sparse-first-pv-55b89ee2")
        text = text.replace("dsv4_csa_sparse_final_publish_4ffccb7b", PACKAGE)
        text = text.replace("# Baseline 4ffccb7b already contains grouped coefficient preparation "
                            "and single publication.\n# Compile the fused final-publication candidate "
                            "before submitting the actual compiled pair.",
                            "# Baseline 55b89ee2 already fuses the final Sparse publication.\n"
                            "# Only test the first-PV zero-state specialization here.")
        (ROOT / name).write_text(text)
    compiler = (ROOT.parent / "csa_sparse_final_publish_fix_20260929/compile.py").read_text()
    compiler = compiler.replace("dsv4_csa_sparse_final_publish_4ffccb7b", PACKAGE)
    compiler = compiler.replace("csa-sparse-final-publish-fix-4ffccb7b", "csa-sparse-first-pv-55b89ee2")
    (ROOT / "compile.py").write_text(compiler)


if __name__ == "__main__":
    main()
