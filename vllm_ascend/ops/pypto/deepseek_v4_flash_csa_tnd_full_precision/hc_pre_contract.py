# SPDX-License-Identifier: Apache-2.0
"""HC BF16分解的加载期权重资格；不修改或替换Native权重。"""

import hashlib

import numpy as np
import torch

HC_WEIGHT_SHAPE = (24, 16384)
HF32_MIN_BITS = 79 << 23  # 2^-48
HF32_MAX_BITS = 167 << 23  # 2^40


def qualify_hc_pre_weight(weight: torch.Tensor, layer_name: str) -> dict:
    """只在加载完成后调用；本检查不证明动态hidden的值域。"""
    if weight.dtype != torch.float32 or tuple(weight.shape) != HC_WEIGHT_SHAPE:
        raise ValueError(f"{layer_name}: HC-pre requires original FP32 {HC_WEIGHT_SHAPE} weight")
    raw = weight.detach().cpu().contiguous().numpy()
    if not np.isfinite(raw).all():
        raise ValueError(f"{layer_name}: HC-pre raw weight contains NaN or Inf")
    rounded_bits = np.add(raw.view(np.uint32), np.uint32(0x800), dtype=np.uint32) & np.uint32(0xFFFFF000)
    rounded = rounded_bits.view(np.float32)
    magnitude = rounded_bits & np.uint32(0x7FFFFFFF)
    nonzero = magnitude != 0
    if np.any(nonzero & ((magnitude < HF32_MIN_BITS) | (magnitude > HF32_MAX_BITS))):
        raise ValueError(f"{layer_name}: HC-pre rounded weight must be zero or have abs in [2^-48, 2^40]")
    high = (rounded_bits & np.uint32(0xFFFF0000)).view(np.float32)
    low = np.subtract(rounded, high, dtype=np.float32)
    tails = (high.view(np.uint32) | low.view(np.uint32)) & np.uint32(0xFFFF)
    if np.any(tails) or not np.isfinite(high).all() or not np.isfinite(low).all():
        raise ValueError(f"{layer_name}: HC-pre weight does not split into exact BF16 operands")
    if not np.array_equal(high.astype(np.float64) + low.astype(np.float64), rounded.astype(np.float64)):
        raise ValueError(f"{layer_name}: HC-pre high/low reconstruction is not exact")
    return {
        "version": 1,
        "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "rounded_sha256": hashlib.sha256(rounded.tobytes()).hexdigest(),
        "min_nonzero_abs": float(np.abs(rounded[nonzero]).min()) if np.any(nonzero) else 0.0,
        "max_abs": float(np.abs(rounded).max()),
    }
