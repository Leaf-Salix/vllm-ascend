# SPDX-License-Identifier: Apache-2.0
"""Separate full root, sharing the accepted immutable-weight lifecycle."""

from ..deepseek_v4_flash_csa_tnd_precision_spmd.native_adapter import CSAOperators as PrecisionOperators
from ..deepseek_v4_flash_csa_tnd_precision_spmd_prepack.native_adapter import PrepackedWorkspace
from ..deepseek_v4_flash_csa_tnd_precision_spmd_prepack.prefix import prepack_fixed_hc_weight
from .decode_csa import decode_csa_tp1_layer_test


class CSAOperators(PrecisionOperators):
    @classmethod
    def register(cls):
        import pypto.torch

        from ..deepseek_v4_flash_csa_tnd_precision.reduction import validate_reduction_mode

        validate_reduction_mode()
        attention = pypto.torch.register(decode_csa_tp1_layer_test, "dsv4_csa_tnd_precision_spmd_bf16layout::attention")
        prepare = pypto.torch.register(
            prepack_fixed_hc_weight, "dsv4_csa_tnd_precision_spmd_bf16layout::prepare_hc_weight"
        )
        return cls(attention, PrepackedWorkspace(prepare))
