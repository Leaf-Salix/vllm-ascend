# SPDX-License-Identifier: Apache-2.0
"""Retain the Native-order service and immutable prepacked-weight lifecycle."""

from ..deepseek_v4_flash_csa_tnd_precision_spmd_prepack.service import CSAServiceRuntime as PrepackedRuntime


class CSAServiceRuntime(PrepackedRuntime):
    """Isolate qualification hooks while retaining the accepted service."""
