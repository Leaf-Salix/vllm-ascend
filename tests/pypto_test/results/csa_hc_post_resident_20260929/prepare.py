"""按AscendC Permanent-X方式复用HC残差，先检查UB及生成代码。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-native-inplace-seven-20260929"
PREFIX = WORKSPACE / ".cache/csa-hc-post-resident-55b89ee2"
OLD_PACKAGE = "dsv4_csa_native_inplace_seven_20260929"
PACKAGE = "dsv4_csa_hc_post_resident_55b89ee2"


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("禁止修改已入队副本")
    for side in ("baseline", "candidate"):
        dest = Path(f"{PREFIX}-{side}")
        shutil.copytree(BASE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        packages = dest / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
    source_file = BASE / "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/hc_post.py"
    before = source_file.read_text()
    anchor = "                for out_h in pl.unroll(HC_MULT):\n"
    if before.count(anchor) != 1:
        raise ValueError("共享HC入口发生改变")
    loads = "                # Native Permanent-X: each residual row is loaded/cast once per token.\n"
    for i in range(4):
        loads += (f"                res_{i} = pl.cast(pl.load(residual_flat, [t, {i} * D], [1, D]), pl.FP32)\n")
    after = before.replace(anchor, loads + anchor)
    begin = after.index("                    for in_h in pl.pipeline(HC_MULT, stage=4):")
    end = after.index("                    # 写回下一层的 hc 残差流", begin)
    compute = ""
    for i in range(4):
        compute += (f"                    comb_{i} = pl.read(comb, [t, {i} * HC_MULT + out_h])\n"
                    f"                    weighted_{i} = pl.mul(res_{i}, comb_{i})\n"
                    f"                    y_row = pl.add(y_row, weighted_{i})\n")
    after = after[:begin] + compute + after[end:]
    after = after.replace("pl.cast(x[t : t + 1, 0:D], target_type=pl.FP32)",
                          "pl.cast(pl.load(x, [t, 0], [1, D]), target_type=pl.FP32)")
    after = after.replace("y_flat[t : t + 1, out_h * D : out_h * D + D] = pl.cast(y_row, pl.BF16, mode=\"rint\")",
                          "pl.store(pl.cast(y_row, pl.BF16, mode=\"rint\"), [t, out_h * D], y_flat)")
    after = after.replace("HC_DIM = M.hc_dim\n", "HC_DIM = M.hc_dim\nassert HC_MULT == 4\n")
    ast.parse(after)
    relative = Path("vllm_ascend/ops/pypto") / PACKAGE / "hc_post.py"
    (Path(f"{PREFIX}-candidate") / relative).write_text(after)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="a/shared/hc_post.py", tofile=f"b/{relative}")))
    source = {
        "baseline": "55b89ee2", "source_prefix": str(PREFIX), "base_source": str(BASE),
        "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
        "native_reference": "ops-transformer 28f40354 mhc_post arch22 ComputeCopyOutAllX / USE_PERMANENT_X",
        "arithmetic": "保持post*x后按in_h=0/1/2/3依次mul/add及BF16 RINT；只复用FP32残差行",
        "inplace_pass": True, "scope": "只改性能私有包；共享精度实现及生产文件保持",
    }
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    origin = ROOT.parent / "csa_sparse_first_pv_20260929"
    for name in ("compile.py", "run.sh", "run_side.sh"):
        text = (origin / name).read_text().replace(origin.name, ROOT.name)
        text = text.replace("csa-sparse-first-pv-55b89ee2", "csa-hc-post-resident-55b89ee2")
        text = text.replace("dsv4_csa_sparse_first_pv_55b89ee2", PACKAGE)
        text = text.replace("# Only test the first-PV zero-state specialization here.",
                            "# Only test HC residual residency here; do not combine the pending first-PV candidate.")
        (ROOT / name).write_text(text)


if __name__ == "__main__":
    main()
