# SPDX-License-Identifier: Apache-2.0
"""复用原生metadata绑定，注册HC-pre至HC-post唯一PTO入口。"""

from ..deepseek_v4_flash_csa_tnd_precision.native_adapter import (
    CSAOperators as BaselineOperators,
)
from ..deepseek_v4_flash_csa_tnd_precision.native_adapter import (  # noqa: F401
    NativeCSACall,
    prepare_weights,
)
from .decode_csa import decode_csa_tp1_layer_test


class CSAOperators(BaselineOperators):
    @classmethod
    def register(cls):
        import pypto.torch

        from ..deepseek_v4_flash_csa_tnd_precision.reduction import validate_reduction_mode

        validate_reduction_mode()
        return cls(pypto.torch.register(decode_csa_tp1_layer_test, "dsv4_csa_tnd_full_precision::attention"))
