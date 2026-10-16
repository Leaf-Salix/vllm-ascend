# SPDX-License-Identifier: Apache-2.0
"""HC-post恢复阶段复用不变的TND attention主体。"""

from ..deepseek_v4_flash_csa_tnd_precision.decode_csa import (  # noqa: F401
    _decode_csa_tp1_layer,
    decode_csa_tp1_layer_test,
)
