# SPDX-License-Identifier: Apache-2.0
"""First precision single-SPMD slice: Native-order Norm -> WKV.

This is not the complete HC-pre-to-HC-post replacement. The baseline package
and default service route stay independent. Scratch is caller-owned. The only
permitted alias is accum_read: it must be the exact same device view as accum.
Every other buffer is disjoint, including FFTS. Each call uses the full
24-AIC/48-AIV A3 participant set.
"""

import pypto.language as pl

from .norm_wkv_stages import ACC_ROWS, T, norm_rows, wkv_matmul, wkv_round
from .rendezvous import GROUPS, handoff


@pl.jit.incore
def cube_schedule(
    normalized: pl.Tensor[[T, 4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16],
    order: pl.Tensor[[32, 16], pl.INT32],
    accum: pl.InOut[pl.Tensor[[ACC_ROWS, 512], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
):
    pl.system.set_ffts(ffts)
    handoff()
    accum = wkv_matmul(normalized, weight, order, accum, pl.tile.get_block_idx())
    handoff()
    handoff()
    return accum


@pl.jit.incore
def vector_schedule(
    x: pl.Tensor[[T, 4096], pl.BF16],
    gamma: pl.Tensor[[4096], pl.BF16],
    normalized: pl.Out[pl.Tensor[[T, 4096], pl.BF16]],
    rstd: pl.Out[pl.Tensor[[T, 8], pl.FP32]],
    accum: pl.Tensor[[ACC_ROWS, 512], pl.FP32],
    output: pl.Out[pl.Tensor[[T, 512], pl.BF16]],
    ffts: pl.Tensor[[256], pl.INT64],
):
    pl.func_attr({"split_aiv": True})
    pl.system.set_ffts(ffts)
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    normalized, rstd = norm_rows(x, gamma, normalized, rstd, worker)
    handoff()
    handoff()
    output = wkv_round(accum, output, worker)
    handoff()
    return normalized, rstd, output


@pl.jit.inline(auto_scope=False)
def norm_wkv_group(
    x: pl.Tensor[[T, 4096], pl.BF16],
    gamma: pl.Tensor[[4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16],
    order: pl.Tensor[[32, 16], pl.INT32],
    normalized: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    rstd: pl.Out[pl.Tensor[[T, 8], pl.FP32]],
    accum: pl.InOut[pl.Tensor[[ACC_ROWS, 512], pl.FP32]],
    accum_read: pl.Tensor[[ACC_ROWS, 512], pl.FP32],
    output: pl.Out[pl.Tensor[[T, 512], pl.BF16]],
    ffts: pl.Tensor[[256], pl.INT64],
):
    x.bind_dynamic(0, T)
    normalized.bind_dynamic(0, T)
    rstd.bind_dynamic(0, T)
    output.bind_dynamic(0, T)
    accum.bind_dynamic(0, ACC_ROWS)
    accum_read.bind_dynamic(0, ACC_ROWS)
    with pl.cluster(name_hint="precision_norm_wkv"):
        accum = cube_schedule(normalized, weight, order, accum, ffts)
        # Read-only descriptor of the same GM storage, not a Cube-return SSA
        # alias. H2 publishes all Cube writes before Vector consumes this view.
        normalized, rstd, output = vector_schedule(x, gamma, normalized, rstd, accum_read, output, ffts)
    return normalized, rstd, accum, output


@pl.jit
def precision_norm_wkv(
    x: pl.Tensor[[T, 4096], pl.BF16],
    gamma: pl.Tensor[[4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16],
    order: pl.Tensor[[32, 16], pl.INT32],
    normalized: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    rstd: pl.Out[pl.Tensor[[T, 8], pl.FP32]],
    accum: pl.InOut[pl.Tensor[[ACC_ROWS, 512], pl.FP32]],
    accum_read: pl.Tensor[[ACC_ROWS, 512], pl.FP32],
    output: pl.Out[pl.Tensor[[T, 512], pl.BF16]],
    ffts: pl.Tensor[[256], pl.INT64],
):
    x.bind_dynamic(0, T)
    normalized.bind_dynamic(0, T)
    rstd.bind_dynamic(0, T)
    output.bind_dynamic(0, T)
    accum.bind_dynamic(0, ACC_ROWS)
    accum_read.bind_dynamic(0, ACC_ROWS)
    with pl.spmd(GROUPS, sync_start=True):
        normalized, rstd, accum, output = norm_wkv_group(
            x, gamma, weight, order, normalized, rstd, accum, accum_read, output, ffts
        )
    return normalized, rstd, accum, output
