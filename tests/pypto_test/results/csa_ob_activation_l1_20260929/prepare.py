"""冻结单根基底，只在NZ O-B小中档复用完整激活L1。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-score-single-root-d93bba14-candidate"
PREFIX = WORKSPACE / ".cache/csa-ob-activation-l1-2ed8ae2e"
OLD_PACKAGE = "dsv4_csa_score_single_root_d93bba14"
PACKAGE = "dsv4_csa_ob_activation_l1_2ed8ae2e"


def transform(before):
    start = before.index("    d0 = pl.tile.get_block_idx() * PROJ_B_D_TILE\n",
                         before.index("def _proj_b_mm_nz_kernel("))
    stop = before.index("    return partials\n", start)
    original = before[start:stop]
    large = original[:original.index("        else:\n")]
    large = large.replace("        if ROW_TILE > PROJ_B_MEDIUM_T_TILE:\n", "")
    # Preserve the existing large-row full-weight path verbatim, one indent less.
    large = "\n".join(line[4:] if line.startswith("            ") else line for line in large.splitlines()) + "\n"
    large = large.replace("    d0 = pl.tile.get_block_idx() * PROJ_B_D_TILE\n", "")
    new = '''    d0 = pl.tile.get_block_idx() * PROJ_B_D_TILE
    if ROW_TILE <= PROJ_B_MEDIUM_T_TILE:
        for tb in pl.range(proj_b_t_rows):
            t0 = tb * ROW_TILE
            # Native's AL1-full/N-first idea: one activation load serves both
            # N256 outputs. Keep weights streamed at K256; no full B residency.
            activation_l1 = pl.load(
                o_r_i8_pad, [t0, col_g], [ROW_TILE, O_LORA], target_memory=pl.MemorySpace.Mat,
            )
            for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                n0 = d0 + nf * PROJ_B_MM_N_TILE
                acc_b = pl.create_tile([ROW_TILE, PROJ_B_MM_N_TILE], dtype=pl.INT32,
                                       target_memory=pl.MemorySpace.Acc)
                for kb in pl.pipeline(0, O_LORA // B_K_TILE, stage=2):
                    k0 = kb * B_K_TILE
                    wk0 = pl.max(g, 0) * O_LORA + k0
                    b_weight_l1 = pl.load(
                        wo_b, [wk0, n0], [B_K_TILE, PROJ_B_MM_N_TILE], target_memory=pl.MemorySpace.Mat,
                    )
                    b_act_left = pl.tile.extract(
                        activation_l1, 0, k0, [ROW_TILE, B_K_TILE], target_memory=pl.MemorySpace.Left,
                    )
                    b_weight_right = pl.tile.move(b_weight_l1, target_memory=pl.MemorySpace.Right)
                    acc_b = pl.tile.matmul_acc(acc_b, b_act_left, b_weight_right, init_cond=(kb == 0))
                pl.store(acc_b, [t0, g * D + n0], partials)
    else:
'''
    new += "\n".join("    " + line if line else line for line in large.splitlines()) + "\n"
    after = before[:start] + new + before[stop:]
    after = after.replace("    BF16_WEIGHT_LAYOUT,\n", "")  # superseded by WO_A_WEIGHT_LAYOUT in the baseline
    ast.parse(after)
    return after


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("禁止修改已提交任务源码")
    for side in ("baseline", "candidate"):
        dest = Path(f"{PREFIX}-{side}")
        shutil.copytree(BASE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        packages = dest / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
    relative = Path("vllm_ascend/ops/pypto") / PACKAGE / "decode_o_proj.py"
    target = Path(f"{PREFIX}-candidate") / relative
    before = target.read_text()
    after = transform(before)
    target.write_text(after)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="a/decode_o_proj.py", tofile="b/decode_o_proj.py")))
    source = {"baseline": "2ed8ae2e", "base_source": str(BASE), "source_prefix": str(PREFIX),
              "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
              "change": "NZ O-B ROW32/96完整A驻留L1跨两个N256复用；B仍K256流水，ROW128保持原路径",
              "native_reference": "ops-nn19614968 QuantBatchMatmulV3Tiling::GetIteratorOrder AL1-full/N-first",
              "difference_from_rejected": "旧方案为完整K1024权重B驻留；本方案只复用A，B仍分段双缓冲",
              "scope": "单卡长B16/短B24；同卡A/B，ND、Native、精度版、权重布局、量化及调度不变"}
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    origin = ROOT.parent / "csa_score_single_root_20260929"
    for name in ("compile.py", "run.sh", "run_side.sh"):
        text = (origin / name).read_text().replace(origin.name, ROOT.name)
        text = text.replace("csa-score-single-root-d93bba14", "csa-ob-activation-l1-2ed8ae2e")
        text = text.replace(OLD_PACKAGE, PACKAGE)
        text = text.replace("# Baseline includes Sparse final publication and adopted HC_post residency.\n"
                            "# Only test Native-style long-query streaming and single-root publication.",
                            "# Baseline includes the adopted single-root Indexer; only O-B activation reuse changes.")
        (ROOT / name).write_text(text)


if __name__ == "__main__":
    main()
