# SPDX-License-Identifier: Apache-2.0
"""Preserve accepted Native-order binding and immutable packet lifecycle."""

from ..deepseek_v4_flash_csa_tnd_precision_spmd_combined.service import CSAServiceRuntime as CombinedRuntime


class CSAServiceRuntime(CombinedRuntime):
    """Keep qualification isolated while sharing the accepted service contract."""
