# SPDX-License-Identifier: Apache-2.0
"""Expose the unchanged Q/KV reduction metadata expected by model loading."""

from ..deepseek_v4_flash_csa_tnd_precision_spmd.qkv_proj_rope import KV_OK, QR_OK

__all__ = ["KV_OK", "QR_OK"]
