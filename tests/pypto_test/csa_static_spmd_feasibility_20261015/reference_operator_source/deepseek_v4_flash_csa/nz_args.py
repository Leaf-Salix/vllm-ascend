# SPDX-License-Identifier: Apache-2.0
"""性能版使用自身根签名与共用布局转换，不另维护 NZ 权重名单。"""

from .decode_csa import _decode_csa_tp1_layer
from .native_adapter import repack_weights
from .nz_mode import root_weight_layouts, root_weight_shapes


def pack_args(tensors: dict, *, source_layouts: dict) -> dict:
    return repack_weights(tensors, source_layouts, root_weight_layouts(_decode_csa_tp1_layer),
                          root_weight_shapes(_decode_csa_tp1_layer))
