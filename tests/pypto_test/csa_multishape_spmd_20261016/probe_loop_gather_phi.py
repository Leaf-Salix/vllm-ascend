# SPDX-License-Identifier: Apache-2.0
"""CPU缩减探针：循环携带的Mat经两次条件DPS更新后是否仍保持同一地址。

来自L1环v2的可疑结构，不执行NPU；编译成功后仍需检查物理地址。
完整CSA中的错误证据在indexer_l1_ring_cpu_v2，不能假设缩减例一定复现。
"""

import pypto.language as pl


@pl.jit(auto_scope=False)
def entry(query: pl.Tensor[[384, 128], pl.INT8],
          cache: pl.Tensor[[1024, 128], pl.INT8],
          flags: pl.Tensor[[2], pl.INT32],
          out: pl.Out[pl.Tensor[[768, 64], pl.INT32]]):
    with pl.spmd(1, name_hint="loop_gather_phi_probe"):
        worker = pl.tile.get_block_idx()
        query_mat = pl.tile.load(query, [0, 0], [384, 128], target_memory=pl.Mem.Mat)
        query_left = pl.tile.move(query_mat, target_memory=pl.Mem.Left)
        pool = pl.tile.load(cache, [0, 0], [192, 128], target_memory=pl.Mem.Mat)
        first = pl.tensor.read(flags, [0])
        second = pl.tensor.read(flags, [1])
        for panel in pl.range(1, 15):
            key = pl.tile.extract(pl.tile.transpose_view(pool), 0, panel % 3 * 64,
                                  [128, 64], target_memory=pl.Mem.Right)
            if first != 0:
                pool = pl.tile.gather_row(pool, cache, [(panel + 1) % 3 * 64, 0],
                                          [panel * 64, 0], [32, 128])
            else:
                pool = pl.tile.gather_row(pool, cache, [(panel + 1) % 3 * 64, 0],
                                          [0, 0], [32, 128])
            if second != 0:
                pool = pl.tile.gather_row(pool, cache, [(panel + 1) % 3 * 64 + 32, 0],
                                          [panel * 64 + 32, 0], [32, 128])
            else:
                pool = pl.tile.gather_row(pool, cache, [(panel + 1) % 3 * 64 + 32, 0],
                                          [32, 0], [32, 128])
            acc = pl.tile.matmul(query_left, key)
            out = pl.tile.store(acc, [worker * 768, 0], out)
        final_key = pl.tile.extract(pl.tile.transpose_view(pool), 0, 0,
                                    [128, 64], target_memory=pl.Mem.Right)
        result = pl.tile.matmul(query_left, final_key)
        out = pl.tile.store(result, [worker * 768 + 384, 0], out)
    return out
