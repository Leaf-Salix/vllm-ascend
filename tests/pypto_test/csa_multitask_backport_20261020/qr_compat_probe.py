# SPDX-License-Identifier: Apache-2.0
"""只编译HCA仍调用的七参数QR入口；不运行设备或声称HCA性能通过。"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.nz_mode import BF16_WEIGHT_LAYOUT


def make_probe(qr_fn):
    @pl.jit(auto_scope=False)
    def legacy_qr_probe(
        x: pl.Tensor[[24, 4096], pl.BF16],
        weight: pl.Tensor[[1024, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
        gamma: pl.Tensor[[1024], pl.BF16],
        qr: pl.InOut[pl.Tensor[[24, 1024], pl.INT8]],
        scale: pl.InOut[pl.Tensor[[24, 1], pl.FP32]],
        padded: pl.InOut[pl.Tensor[[512, 1024], pl.INT8]],
        padded_scale: pl.InOut[pl.Tensor[[512, 1], pl.FP32]],
    ):
        _qa_tid = qr_fn(x, weight, gamma, qr, scale, padded, padded_scale)
        return qr, scale, padded, padded_scale

    return legacy_qr_probe
