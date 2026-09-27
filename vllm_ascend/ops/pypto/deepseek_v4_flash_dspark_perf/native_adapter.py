# SPDX-License-Identifier: Apache-2.0
"""性能版入口；权重准备、Native 存储/metadata 绑定与精度版共用。"""

from ..deepseek_v4_flash_dspark.native_adapter import CSAOperators as _CSAOperators
from ..deepseek_v4_flash_dspark.native_adapter import NativeCSACall as _NativeCSACall
from ..deepseek_v4_flash_dspark.native_adapter import prepare_weights as _prepare_weights
from ..deepseek_v4_flash_dspark.native_storage import indexer_storage
from .decode_csa import _decode_csa_tp1_layer, decode_csa_tp1_layer_test


class CSAOperators(_CSAOperators):
    @classmethod
    def register(cls):
        return super().register(decode_csa_tp1_layer_test)


def prepare_weights(attention, hadamard, layer=None):
    return _prepare_weights(attention, hadamard, layer, root_function=_decode_csa_tp1_layer)


class NativeCSACall(_NativeCSACall):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, kernel=decode_csa_tp1_layer_test, **kwargs)

    def _indexer_cache_arguments(self):
        native_cache = indexer_storage(*self.views["indexer"])
        return {"idx_native_kv_cache": native_cache}
