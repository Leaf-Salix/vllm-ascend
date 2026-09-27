# SPDX-License-Identifier: Apache-2.0
"""性能版入口；权重准备、Native 存储/metadata 绑定与精度版共用。"""

from ..deepseek_v4_flash_dspark.native_adapter import CSAOperators as _CSAOperators
from ..deepseek_v4_flash_dspark.native_adapter import NativeCSACall as _NativeCSACall
from ..deepseek_v4_flash_dspark.native_adapter import prepare_weights as _prepare_weights
from ..deepseek_v4_flash_dspark.native_storage import indexer_storage
from .decode_csa import _decode_csa_tp1_layer, decode_csa_tp1_layer_test
from .indexer_cache import SplitIndexerCache


class CSAOperators(_CSAOperators):
    @classmethod
    def register(cls):
        return super().register(decode_csa_tp1_layer_test)


def prepare_weights(attention, hadamard, layer=None):
    return _prepare_weights(attention, hadamard, layer, root_function=_decode_csa_tp1_layer)


class NativeCSACall(_NativeCSACall):
    def __init__(self, *args, indexer_cache=None, **kwargs):
        self.indexer_cache = indexer_cache
        super().__init__(*args, kernel=decode_csa_tp1_layer_test, **kwargs)

    def _indexer_cache_arguments(self):
        key, scale = self.views["indexer"]
        indexer_storage(key, scale)  # Retain the existing Native layout checks.
        if self.indexer_cache is None:
            self.indexer_cache = SplitIndexerCache(
                key, scale, batch_capacity=self.req["indexer"].seq_lens.numel(),
                table_columns=self.tables["indexer"].shape[1],
            )
        elif not self.indexer_cache.matches(key, scale):
            raise ValueError("Prepared split cache belongs to a different Native allocation")
        key, scale = self.indexer_cache.views(self.req["indexer"].seq_lens.numel())
        return {"idx_kv_cache": key, "idx_kv_scale": scale}

    def prepare_indexer_cache(self):
        self.indexer_cache.load(self.args["idx_block_table"], self.args["kv_seq_lens"])

    def run_kernel(self):
        return super().__call__()

    def commit_indexer_cache(self):
        self.indexer_cache.commit(
            self.args["idx_slot_mapping"], self.args["kv_seq_lens"], self.args["idx_query_start_loc"],
        )

    def __call__(self):
        self.prepare_indexer_cache()
        output = self.run_kernel()
        self.commit_indexer_cache()
        return output
