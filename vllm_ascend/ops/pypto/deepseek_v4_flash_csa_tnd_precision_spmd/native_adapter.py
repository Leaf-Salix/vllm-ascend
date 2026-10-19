# SPDX-License-Identifier: Apache-2.0
"""Keep Native metadata binding and register a distinct full-SPMD-prefix root."""

from dataclasses import dataclass

from ..deepseek_v4_flash_csa_tnd_precision.native_adapter import (
    CSAOperators as BaselineOperators,
)
from ..deepseek_v4_flash_csa_tnd_precision.native_adapter import (  # noqa: F401
    NativeCSACall,
    prepare_weights,
)
from .decode_csa import decode_csa_tp1_layer_test
from .workspace import PrecisionWorkspace


@dataclass(frozen=True)
class CSAOperators(BaselineOperators):
    workspace: PrecisionWorkspace

    @classmethod
    def register(cls):
        import pypto.torch

        from ..deepseek_v4_flash_csa_tnd_precision.reduction import validate_reduction_mode

        validate_reduction_mode()
        return cls(
            pypto.torch.register(decode_csa_tp1_layer_test, "dsv4_csa_tnd_precision_spmd::attention"),
            PrecisionWorkspace(),
        )
