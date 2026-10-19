# SPDX-License-Identifier: Apache-2.0
"""Register a separate root and a model-owned load-time weight preparer."""

from ..deepseek_v4_flash_csa_tnd_precision_spmd.native_adapter import CSAOperators as PrecisionOperators
from ..deepseek_v4_flash_csa_tnd_precision_spmd.workspace import PrecisionWorkspace
from .decode_csa import decode_csa_tp1_layer_test
from .prefix import prepack_fixed_hc_weight


class PrepackedWorkspace(PrecisionWorkspace):
    """Keep the preparer with the shared pool when qualification wraps the op."""

    def __init__(self, prepare_hc_weight):
        super().__init__()
        self.prepare_hc_weight = prepare_hc_weight


class CSAOperators(PrecisionOperators):
    @classmethod
    def register(cls):
        import pypto.torch

        from ..deepseek_v4_flash_csa_tnd_precision.reduction import validate_reduction_mode

        validate_reduction_mode()
        attention = pypto.torch.register(decode_csa_tp1_layer_test, "dsv4_csa_tnd_precision_spmd_prepack::attention")
        prepare = pypto.torch.register(
            prepack_fixed_hc_weight, "dsv4_csa_tnd_precision_spmd_prepack::prepare_hc_weight"
        )
        return cls(attention, PrepackedWorkspace(prepare))
