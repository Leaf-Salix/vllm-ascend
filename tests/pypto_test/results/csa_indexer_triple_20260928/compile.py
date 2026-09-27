"""只在CPU编译双/三query的Score与Top-K动态分派，检查所有Cube/Vector分支。"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="编译完整CSA，检查与其余算子的集成")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / "tests/pypto_test"))
    from dsv4_csa_env import activate

    os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"
    activate()
    import pypto.language as pl
    from pypto.runtime import RunConfig

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf import decode_indexer as mod

    score = mod.indexer_score_topk_forest
    query_rows = mod.T_PAD * mod.IDX_N_HEADS
    padded_tokens = mod.T_PAD
    tokens, batches = pl.dynamic("TRIPLE_TOKENS"), pl.dynamic("TRIPLE_BATCHES")
    pages, page_bytes, columns = (
        pl.dynamic("TRIPLE_PAGES"),
        pl.dynamic("TRIPLE_PAGE_BYTES"),
        pl.dynamic("TRIPLE_COLUMNS"),
    )

    def root(
        query: pl.Tensor[[query_rows, 128], pl.INT8],
        query_scale: pl.Tensor[[query_rows, 1], pl.FP32],
        weights: pl.Tensor[[padded_tokens, 64], pl.FP32],
        cache: pl.Tensor[[pages, page_bytes], pl.INT8],
        table: pl.Tensor[[batches, columns], pl.INT32],
        positions: pl.Tensor[[tokens], pl.INT64],
        lengths: pl.Tensor[[batches], pl.INT32],
        scores: pl.Out[pl.Tensor[[tokens, 512], pl.FP32]],
        indices: pl.Out[pl.Tensor[[tokens, 512], pl.INT32]],
    ):
        dep = pl.system.task_invalid()
        score(query, query_scale, weights, cache, table, positions, lengths, scores, indices, dep, dep, dep)
        return scores, indices

    if args.full:
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_csa import decode_csa_tp1_layer_test

        kernel = decode_csa_tp1_layer_test
    else:
        kernel = pl.jit(auto_scope=False)(root)
    compiled = kernel.compile(
        config=RunConfig(platform="a2a3", save_kernels=True, save_kernels_dir=str(args.output.resolve()))
    )
    compiled.load()
    print(f"COMPILE_PASS full_csa={args.full}; no device execution", flush=True)


if __name__ == "__main__":
    main()
