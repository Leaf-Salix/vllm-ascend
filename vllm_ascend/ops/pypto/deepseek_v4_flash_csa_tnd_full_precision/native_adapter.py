# SPDX-License-Identifier: Apache-2.0
"""共享冻结基准的权重/metadata绑定及attention注册，不改变其算术。"""

from ..deepseek_v4_flash_csa_tnd_precision.native_adapter import (  # noqa: F401
    CSAOperators,
    NativeCSACall,
    prepare_weights,
)
