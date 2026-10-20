# SPDX-License-Identifier: Apache-2.0
"""Reuse the accepted loading, cache, stream and immutability contracts."""

from ..deepseek_v4_flash_csa_tnd_precision_spmd_prepack.service import CSAServiceRuntime as PrepackedRuntime


class CSAServiceRuntime(PrepackedRuntime):
    """Keep variant qualification hooks isolated while inheriting the lifecycle."""
