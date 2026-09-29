"""私有整包构造O-B反量化/HC_post融合，保留BF16边界与共享算术。"""

import ast
import difflib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
PREVIOUS = ROOT.parent / "csa_ob_activation_l1_k128_20260929"
BASE = WORKSPACE / ".cache/csa-ob-activation-l1-k128-2ed8ae2e-candidate"
PREFIX = WORKSPACE / ".cache/csa-ob-hc-fused-9a868d26"
OLD_PACKAGE = "dsv4_csa_ob_activation_l1_k128_2ed8ae2e"
PACKAGE = "dsv4_csa_ob_hc_fused_9a868d26"

HC_HELPER = '''@pl.jit.inline
def hc_post_block(
    x_block: pl.Tensor,
    residual_flat: pl.Tensor,
    post: pl.Tensor,
    comb: pl.Tensor,
    y_flat: pl.Out[pl.Tensor],
    t0: pl.Scalar[pl.INDEX],
    n0: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    ROWS: pl.constexpr,
    COLS: pl.constexpr,
):
    # x_block is already BF16: retain the rounding boundary before HC math.
    x_f32 = pl.cast(x_block, pl.FP32)
    res_0 = pl.cast(pl.slice(residual_flat, [ROWS, COLS], [t0, n0],
                            valid_shape=[valid_rows, COLS], clamp=True), pl.FP32)
    res_1 = pl.cast(pl.slice(residual_flat, [ROWS, COLS], [t0, D + n0],
                            valid_shape=[valid_rows, COLS], clamp=True), pl.FP32)
    res_2 = pl.cast(pl.slice(residual_flat, [ROWS, COLS], [t0, 2 * D + n0],
                            valid_shape=[valid_rows, COLS], clamp=True), pl.FP32)
    res_3 = pl.cast(pl.slice(residual_flat, [ROWS, COLS], [t0, 3 * D + n0],
                            valid_shape=[valid_rows, COLS], clamp=True), pl.FP32)
    if ROWS == 1:
        for out_h in pl.unroll(HC_MULT):
            post_scalar = pl.read(post, [t0, out_h])
            single_y = pl.mul(x_f32, post_scalar)
            comb_scalar_0 = pl.read(comb, [t0, out_h])
            single_weighted_0 = pl.mul(res_0, comb_scalar_0)
            single_y = pl.add(single_y, single_weighted_0)
            comb_scalar_1 = pl.read(comb, [t0, HC_MULT + out_h])
            single_weighted_1 = pl.mul(res_1, comb_scalar_1)
            single_y = pl.add(single_y, single_weighted_1)
            comb_scalar_2 = pl.read(comb, [t0, 2 * HC_MULT + out_h])
            single_weighted_2 = pl.mul(res_2, comb_scalar_2)
            single_y = pl.add(single_y, single_weighted_2)
            comb_scalar_3 = pl.read(comb, [t0, 3 * HC_MULT + out_h])
            single_weighted_3 = pl.mul(res_3, comb_scalar_3)
            single_y = pl.add(single_y, single_weighted_3)
            single_result = pl.cast(single_y, pl.BF16, mode="rint")
            y_flat[t0 : t0 + ROWS, out_h * D + n0 : out_h * D + n0 + COLS] = pl.set_validshape(
                single_result, valid_rows, COLS)
    else:
        # Expose the loaded UB tile's full physical extent before taking
        # views. Padded rows/columns never reach the masked output stores.
        post_loaded = pl.slice(post, [ROWS, 8], [t0, 0],
                               valid_shape=[valid_rows, HC_MULT], clamp=True)
        comb_loaded = pl.slice(comb, [ROWS, HC_MULT * HC_MULT], [t0, 0],
                               valid_shape=[valid_rows, HC_MULT * HC_MULT], clamp=True)
        post_trans = pl.transpose(pl.set_validshape(post_loaded, ROWS, 8), 0, 1)
        comb_trans = pl.transpose(pl.set_validshape(comb_loaded, ROWS, HC_MULT * HC_MULT), 0, 1)
        for out_h in pl.unroll(HC_MULT):
            post_w = pl.reshape(post_trans[out_h : out_h + 1, 0:ROWS], [ROWS, 1])
            y_block = pl.row_expand_mul(x_f32, post_w)
            comb_0 = pl.reshape(comb_trans[out_h : out_h + 1, 0:ROWS], [ROWS, 1])
            weighted_0 = pl.row_expand_mul(res_0, comb_0)
            y_block = pl.add(y_block, weighted_0)
            comb_1 = pl.reshape(comb_trans[HC_MULT + out_h : HC_MULT + out_h + 1, 0:ROWS], [ROWS, 1])
            weighted_1 = pl.row_expand_mul(res_1, comb_1)
            y_block = pl.add(y_block, weighted_1)
            comb_2 = pl.reshape(comb_trans[2 * HC_MULT + out_h : 2 * HC_MULT + out_h + 1, 0:ROWS], [ROWS, 1])
            weighted_2 = pl.row_expand_mul(res_2, comb_2)
            y_block = pl.add(y_block, weighted_2)
            comb_3 = pl.reshape(comb_trans[3 * HC_MULT + out_h : 3 * HC_MULT + out_h + 1, 0:ROWS], [ROWS, 1])
            weighted_3 = pl.row_expand_mul(res_3, comb_3)
            y_block = pl.add(y_block, weighted_3)
            multi_result = pl.cast(y_block, pl.BF16, mode="rint")
            y_flat[t0 : t0 + ROWS, out_h * D + n0 : out_h * D + n0 + COLS] = pl.set_validshape(
                multi_result, valid_rows, COLS)
    return y_flat


'''


def change_shared_hc(before):
    start = before.index("                # One cast per token:")
    end = before.index("    return y\n", start)
    body = '''                x_block = pl.slice(x, [1, D], [t, 0])
                y_flat = hc_post_block(x_block, residual_flat, post, comb, y_flat, t, 0, 1, 1, D)
'''
    after = before[:start] + body + before[end:]
    return after.replace("def _hc_post(\n", HC_HELPER + "def _hc_post(\n", 1)


def change_o_proj(before):
    after = before.replace("from .nz_mode import (", "from .hc_post import hc_post_block\nfrom .nz_mode import (", 1)
    after = after.replace("PROJ_B_ACT_TASK_T_TILE = 32", "PROJ_B_ACT_TASK_T_TILE = 16", 1)
    after = after.replace("PROJ_B_ACT_N_TILE = 512", "HC_MULT = M.hc_mult\n\nPROJ_B_ACT_N_TILE = 512", 1)
    parameter = "    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],\n"
    replacement = '''    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
'''
    if after.count(parameter) != 2:
        raise ValueError("O投影两入口签名发生变化")
    after = after.replace(parameter, replacement)
    after = after.replace("pl.tensor.dim(attn_out, 0)", "pl.tensor.dim(x_out, 0)")
    after = after.replace("    # Dequantize each group with its own scale, then sum in FP32.\n",
                          "    residual_flat = pl.reshape(residual, [t_dim, HC_MULT * D])\n"
                          "    y_flat = pl.reshape(x_out, [t_dim, HC_MULT * D])\n"
                          "    # Keep group dequantization and BF16 rounding before HC_post.\n")
    after = after.replace('name_hint="proj_b_act"', 'name_hint="proj_b_act_hc_post"')
    store = ("            attn_out = pl.assemble(attn_out, pl.set_validshape(out_bf16, output_rows, "
             "PROJ_B_ACT_N_TILE), [b_tb, ob_n0])")
    if after.count(store) != 1:
        raise ValueError("反量化发布锚点发生变化")
    after = after.replace('            out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")',
                          '            out_bf16 = pl.create_tensor('
                          '[PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.BF16)\n'
                          '            out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")')
    after = after.replace(store, '''            y_flat = hc_post_block(
                out_bf16, residual_flat, post, comb, y_flat, b_tb, ob_n0,
                output_rows, PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE,
            )''')
    after = after.replace("attn_out = _decode_o_proj_tp1_tiled(", "x_out = _decode_o_proj_tp1_tiled(")
    after = after.replace("wo_b_scale, attn_out, heads_dep, ",
                          "wo_b_scale, residual, post, comb, x_out, heads_dep,\n            ")
    after = after.replace("    return attn_out\n", "    return x_out\n")
    if "attn_out" in after:
        raise ValueError("融合路径仍有旧中间输出引用")
    return after


def change_csa(before):
    after = before.replace("from .hc_post import hc_post\n", "")
    start = after.index("    # attn_out 在大 scope")
    end = after.index("    with pl.scope():\n", start)
    after = after[:start] + after[end:]
    old = ("            attn_out = decode_o_proj_tp1(o_packed_heads, wo_a, wo_b, wo_b_scale, attn_out, heads_dep)\n"
           "            hc_post(attn_out, x_hc, post_t, comb_t, x_out)")
    new = ("            x_out = decode_o_proj_tp1(\n"
           "                o_packed_heads, wo_a, wo_b, wo_b_scale, x_hc, post_t, comb_t, x_out, heads_dep,\n"
           "            )")
    if after.count(old) != 1:
        raise ValueError("CSA收尾锚点发生变化")
    return after.replace(old, new)


def export_shared_hc(_before):
    # Historical frozen variants materialize this file instead of re-exporting it.
    return '''# SPDX-License-Identifier: Apache-2.0
from ..deepseek_v4_flash_dspark.hc_post import hc_post as hc_post
from ..deepseek_v4_flash_dspark.hc_post import hc_post_block as hc_post_block
from ..deepseek_v4_flash_dspark.hc_post import hc_post_test as hc_post_test
'''


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("不得修改已排队副本")
    for artifact in ("compile_candidate.json", "compile_shared_hc.json", "static_evidence.json"):
        (ROOT / artifact).unlink(missing_ok=True)
    for side in ("baseline", "candidate"):
        destination = Path(str(PREFIX) + "-" + side)
        packages = destination / "vllm_ascend/ops/pypto"
        if not destination.exists():
            shutil.copytree(BASE, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (packages / OLD_PACKAGE).rename(packages / PACKAGE)
        elif not (packages / PACKAGE / "decode_o_proj.py").is_file():
            raise RuntimeError(f"私有副本不完整：{destination}")
    changes = [(f"{PACKAGE}/decode_o_proj.py", change_o_proj),
               (f"{PACKAGE}/decode_csa.py", change_csa),
               (f"{PACKAGE}/hc_post.py", export_shared_hc),
               ("deepseek_v4_flash_dspark/hc_post.py", change_shared_hc)]
    for relative, transform in changes:
        path = Path(str(PREFIX) + "-candidate") / "vllm_ascend/ops/pypto" / relative
        before = (Path(str(PREFIX) + "-baseline") / "vllm_ascend/ops/pypto" / relative).read_text()
        after = transform(before)
        ast.parse(after)
        path.write_text(after)
        subprocess.run(["ruff", "check", "--fix", "--select", "I", str(path)], check=True)
    patch = []
    for relative, _ in changes:
        before = (Path(str(PREFIX) + "-baseline") / "vllm_ascend/ops/pypto" / relative).read_text()
        after = (Path(str(PREFIX) + "-candidate") / "vllm_ascend/ops/pypto" / relative).read_text()
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                           fromfile="a/" + relative, tofile="b/" + relative))
    (ROOT / "candidate.patch").write_text("".join(patch))
    for name in ("compile.py", "run.sh", "run_side.sh"):
        text = (PREVIOUS / name).read_text().replace(PREVIOUS.name, ROOT.name)
        text = text.replace("csa-ob-activation-l1-k128-2ed8ae2e", PREFIX.name).replace(OLD_PACKAGE, PACKAGE)
        text = text.replace("8192:16", "8192:24")
        text = text.replace("# Baseline includes the adopted single-root Indexer; only O-B activation reuse changes.",
                            "# Baseline includes validated O-B reuse; only final AIV task organization changes.")
        (ROOT / name).write_text(text)
    source = {"baseline": "9a868d26", "base_source": str(BASE), "source_prefix": str(PREFIX),
              "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
              "change": "O-B反量化与HC_post共用AIV任务，T16/N512工作块、内部T8；BF16边界和相加顺序保留",
              "expected_workers": {"128K/B16": 48, "8K/B24": 72},
              "scope": "私有CPU候选；共享HC单行路径保留原算术，多行路径按行广播；未修改生产或工具链"}
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
