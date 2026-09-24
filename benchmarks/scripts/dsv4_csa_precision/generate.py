# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Generate private, content-addressed, single-factor CSA diagnostics."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Source contract changed: expected one occurrence of {old[:100]!r}")
    return text.replace(old, new, 1)


def instrument_csa(text):
    outputs = {
        "q": "[t_dim, H, HEAD_DIM], dtype=pl.BF16",
        "kv": "[t_dim, HEAD_DIM], dtype=pl.BF16",
        "qr": "[t_dim, Q_LORA], dtype=pl.INT8",
        "qr_scale": "[t_dim, 1], dtype=pl.FP32",
        "topk_scores": "[t_dim, IDX_TOPK], dtype=pl.FP32",
        "topk_indices": "[t_dim, IDX_TOPK], dtype=pl.INT32",
    }
    extra = ""
    for name, spec in outputs.items():
        shape, dtype = spec.split(", dtype=")
        extra += f"    {name}: pl.Out[pl.Tensor[{shape.replace('t_dim', 'T_DYN')}, {dtype}]],\n"
        text = replace_once(text, f"    {name} = pl.create_tensor({spec})\n", "")
    extra += "    o_packed_heads: pl.Out[pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16]],\n"
    text = replace_once(
        text,
        "    o_packed_heads = pl.create_tensor(\n        [O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.BF16,\n    )\n",
        "",
    )
    anchor = "    attn_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],\n"
    return replace_once(text, anchor, anchor + extra)


def oproj_variant(text, bf16, global_quant, int_sum):
    start = text.index("        with pl.spmd((t_dim + QUANT_TOKEN_TILE")
    end = text.index("        # Zero all padding", start)
    rounded = 'pl.cast(pl.cast(chunk, target_type=pl.BF16, mode="rint"), target_type=pl.FP32)' if bf16 else "chunk"
    # Identical scheduling in all four cells; only the amax domain changes.
    domain = "0, O_GROUPS" if global_quant else "qg, qg + 1"
    quant = f"""        with pl.spmd((t_dim + QUANT_TOKEN_TILE - 1) // QUANT_TOKEN_TILE,
                     name_hint="quant_factor", deps=[proj_a_tids[i] for i in range(O_GROUPS)]) as q_tid:
            qt = pl.tile.get_block_idx() * QUANT_TOKEN_TILE
            qrows = pl.min(QUANT_TOKEN_TILE, t_dim - qt)
            for qg in pl.range(O_GROUPS):
                row_max = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for rg in pl.range({domain}):
                    rc = rg * O_LORA
                    chunk = pl.slice(o_r_pad, [QUANT_TOKEN_TILE, O_LORA], [qt, rc], valid_shape=[qrows, O_LORA])
                    rounded = {rounded}
                    row_max = pl.maximum(row_max, pl.reshape(pl.row_max(pl.abs(rounded)), [1, QUANT_TOKEN_TILE]))
                scale_q = pl.div(pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), row_max)
                scale_dq = pl.mul(row_max, 1.0 / INT8_SCALE_MAX)
                qc = qg * O_LORA
                chunk = pl.slice(o_r_pad, [QUANT_TOKEN_TILE, O_LORA], [qt, qc], valid_shape=[qrows, O_LORA])
                rounded = {rounded}
                scaled = pl.row_expand_mul(rounded, pl.reshape(scale_q, [QUANT_TOKEN_TILE, 1]))
                i32 = pl.cast(scaled, target_type=pl.INT32, mode="rint")
                half = pl.cast(i32, target_type=pl.FP16, mode="round")
                i8 = pl.cast(half, target_type=pl.INT8, mode="trunc")
                o_r_i8_pad[qt:qt+QUANT_TOKEN_TILE, qc:qc+O_LORA] = pl.set_validshape(i8, qrows, O_LORA)
                act_scale_dq[qg:qg+1, qt:qt+QUANT_TOKEN_TILE] = pl.set_validshape(
                    pl.reshape(scale_dq, [1, QUANT_TOKEN_TILE]), 1, qrows
                )

"""
    text = text[:start] + quant + text[end:]
    if not int_sum:
        start = text.index("            acc_i32 = pl.full(", text.index("# Sum INT32 group partials"))
        end = text.index("            out_bf16 = ", start)
        text = (
            text[:start]
            + """            acc_sum = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for act_g in pl.pipeline(O_GROUPS, stage=2):
                p_col0 = act_g * D + ob_n0
                p_g = partials[b_tb:b_tb+PROJ_B_ACT_T_TILE, p_col0:p_col0+PROJ_B_ACT_N_TILE]
                scale_row = act_scale_dq[act_g:act_g+1, b_tb:b_tb+PROJ_B_ACT_T_TILE]
                scale_col = pl.reshape(scale_row, [PROJ_B_ACT_T_TILE, 1])
                acc = pl.row_expand_mul(pl.cast(p_g, target_type=pl.FP32), scale_col)
                acc_sum = pl.add(acc_sum, acc)
            out_t = pl.col_expand_mul(acc_sum, wb_scale_chunk)
"""
            + text[end:]
        )
    # Export diagnostic intermediates; these variants are only called by oproj_bench.
    extra = """    o_r_pad: pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.FP32],
    o_r_i8_pad: pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.INT8],
    act_scale_dq: pl.Tensor[[O_GROUPS, T_PAD], pl.FP32],
"""
    text = replace_once(
        text, "    heads_dep: pl.Scalar[pl.TASK_ID],\n", extra + "    heads_dep: pl.Scalar[pl.TASK_ID],\n"
    )
    for name, spec in [
        ("o_r_pad", "[T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32"),
        ("o_r_i8_pad", "[T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8"),
        ("act_scale_dq", "[O_GROUPS, T_PAD], dtype=pl.FP32"),
    ]:
        text = replace_once(text, f"    {name} = pl.create_tensor({spec})\n", "")
    return text


BENCH = """import pypto.language as pl
from .decode_o_proj import decode_o_proj_tp1, O_GROUPS, T_PAD, O_GROUP_IN, O_LORA, D, T_DYN

@pl.jit
def oproj_bench(
    heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wa: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wb: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    scale: pl.Tensor[[D], pl.FP32],
    out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    oa: pl.Out[pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.FP32]],
    quant: pl.Out[pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.INT8]],
    dq: pl.Out[pl.Tensor[[O_GROUPS, T_PAD], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="bench_start") as dep:
        out[0:1, 0:32] = pl.full([1, 32], dtype=pl.BF16, value=0.0)
    decode_o_proj_tp1(heads, wa, wb, scale, out, oa, quant, dq, dep)
    return out, oa, quant, dq
"""


def generate(repo, output):
    source = repo / "vllm_ascend/attention"
    factors = json.loads(Path(__file__).with_name("factors.json").read_text())
    variants = {"control": None, **{name: edits for name, edits in factors.items()}}
    variants.update(q_pair=factors["q_mm"] + factors["q_rms"], kv_pair=factors["kv_mm"] + factors["kv_rms"])
    variants.update(
        {name: None for name in ("oa32_group", "oa16_group", "oa32_global", "oa16_global", "oa16_global_intsum")}
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, edits in variants.items():
        dst = output / name
        if dst.exists():
            raise FileExistsError(f"Refusing to overwrite existing variant {dst}")
        dst.mkdir()
        shutil.copy2(source / "pto_attn.py", dst / "pto_attn.py")
        shutil.copytree(
            source / "pto_kernels", dst / "pto_kernels", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        ds = dst / "pto_kernels/dspark"
        if name.startswith("oa"):
            p = ds / "decode_o_proj.py"
            p.write_text(oproj_variant(p.read_text(), "16" in name, "global" in name, "intsum" in name))
            (ds / "oproj_bench.py").write_text(BENCH)
        else:
            p = ds / "decode_csa.py"
            p.write_text(instrument_csa(p.read_text()))
            p = ds / "qkv_proj_rope.py"
            s = p.read_text()
            for edit in edits or []:
                s = replace_once(s, edit["anchor"], edit["anchor"] + edit["insert"])
            p.write_text(s)
        manifest[name] = {
            str(p.relative_to(dst)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dst.rglob("*.py"))
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.repo.resolve(), args.output.resolve())
