# SPDX-License-Identifier: Apache-2.0
"""Reuse accepted packets, preparer and workspace; register one large-T root."""

from ..deepseek_v4_flash_csa_tnd_precision_spmd_combined.native_adapter import CSAOperators as CombinedOperators
from .decode_csa import decode_csa_tp1_layer_test
from .selection import CaptureAttention


class CSAOperators(CombinedOperators):
    @classmethod
    def register(cls):
        import pypto.torch

        baseline = CombinedOperators.register()
        tuned = pypto.torch.register(decode_csa_tp1_layer_test, "dsv4_csa_tnd_precision_spmd_tuned::attention")
        assert str(baseline.attention._schema).partition("(")[2] == str(tuned._schema).partition("(")[2]
        assert len(tuned._schema.arguments) == 64
        return cls(CaptureAttention(baseline.attention, tuned), baseline.workspace)
