# SPDX-License-Identifier: Apache-2.0
"""显式来源布局的根权重转换；已是目标布局的入参保持不动。"""

from .decode_csa import _decode_csa_tp1_layer
from .native_adapter import repack_weights
from .nz_mode import root_weight_layouts, root_weight_shapes


def pack_args(tensors: dict, *, source_layouts: dict) -> dict:
    return repack_weights(tensors, source_layouts, root_weight_layouts(_decode_csa_tp1_layer),
                          root_weight_shapes(_decode_csa_tp1_layer))
