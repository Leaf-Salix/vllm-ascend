# SPDX-License-Identifier: Apache-2.0
"""Move a cast across pure layout copies; retain every Native arithmetic boundary."""

import pypto.language as pl

from ..deepseek_v4_flash_csa_tnd_precision_spmd.hc_prefix_stages import HC_DIM, P16
from ..deepseek_v4_flash_csa_tnd_precision_spmd.norm_wkv_stages import T


@pl.jit.inline(auto_scope=False)
def hc_pack_input_bf16(
    x_flat: pl.Tensor[[T, HC_DIM], pl.FP32],
    panel64_packed_flat: pl.InOut[pl.Tensor[[P16, HC_DIM * 2], pl.BF16]],
    worker: pl.Scalar[pl.INDEX],
):
    t_dim = pl.tensor.dim(x_flat, 0)
    t_linear = pl.tensor.dim(panel64_packed_flat, 0)
    panel64_inputs_per_task = 1
    if t_dim >= 64:
        panel64_inputs_per_task = 4
    for panel64_input_group in pl.range(worker, t_linear // 16 * (HC_DIM // 256) // panel64_inputs_per_task, 48):
        for panel64_input_local in pl.range(panel64_inputs_per_task):
            panel64_transpose_unit = panel64_input_group * panel64_inputs_per_task + panel64_input_local
            panel64_t0 = panel64_transpose_unit // (HC_DIM // 256) * 16
            panel64_k0 = panel64_transpose_unit % (HC_DIM // 256) * 256
            panel64_values = pl.load(
                x_flat, [panel64_t0, panel64_k0], [16, 256], valid_shape=[pl.min(16, t_dim - panel64_t0), 256]
            )
            for panel64_invalid_row in pl.range(pl.min(16, t_dim - panel64_t0), 16):
                for panel64_invalid_col in pl.range(256):
                    pl.write(panel64_values, [panel64_invalid_row, panel64_invalid_col], 0.0)
            panel64_values = pl.set_validshape(panel64_values, 16, 256)
            panel64_values = pl.tile.cast(panel64_values, pl.BF16)
            panel64_transposed_values = pl.tile.transpose(panel64_values, axis1=0, axis2=1)
            panel64_packed_transposed = pl.tile.full([512, 16], dtype=pl.BF16, value=0.0)
            for panel64_chunk in pl.range(32):
                panel64_input8 = pl.tile.slice(panel64_transposed_values, [8, 16], [panel64_chunk * 8, 0])
                panel64_packed_transposed = pl.tile.assemble(
                    panel64_packed_transposed, panel64_input8, [panel64_chunk * 16, 0]
                )
                panel64_packed_transposed = pl.tile.assemble(
                    panel64_packed_transposed, panel64_input8, [panel64_chunk * 16 + 8, 0]
                )
            panel64_packed_values = pl.tile.transpose(panel64_packed_transposed, axis1=0, axis2=1)
            pl.store(panel64_packed_values, [panel64_t0, panel64_k0 * 2], panel64_packed_flat)
    return panel64_packed_flat
