# SPDX-License-Identifier: Apache-2.0
"""Register two complete roots sharing immutable packets and a workspace."""

from ..deepseek_v4_flash_csa_tnd_precision_spmd.native_adapter import CSAOperators as PrecisionOperators
from ..deepseek_v4_flash_csa_tnd_precision_spmd_overlap.decode_csa import (
    decode_csa_tp1_layer_test as original_attention,
)
from ..deepseek_v4_flash_csa_tnd_precision_spmd_prepack.native_adapter import PrepackedWorkspace
from ..deepseek_v4_flash_csa_tnd_precision_spmd_prepack.prefix import prepack_fixed_hc_weight
from .decode_csa import decode_csa_tp1_layer_test
from .selection import CaptureAttention


class CSAOperators(PrecisionOperators):
    @classmethod
    def register(cls):
        import pypto.torch

        from ..deepseek_v4_flash_csa_tnd_precision.reduction import validate_reduction_mode

        validate_reduction_mode()
        small = pypto.torch.register(original_attention, "dsv4_csa_tnd_precision_spmd_combined::attention_small")
        combined = pypto.torch.register(decode_csa_tp1_layer_test, "dsv4_csa_tnd_precision_spmd_combined::attention")
        assert str(small._schema).partition("(")[2] == str(combined._schema).partition("(")[2]
        assert len(combined._schema.arguments) == 64
        prepare = pypto.torch.register(
            prepack_fixed_hc_weight, "dsv4_csa_tnd_precision_spmd_combined::prepare_hc_weight"
        )
        return cls(CaptureAttention(small, combined), PrepackedWorkspace(prepare))
