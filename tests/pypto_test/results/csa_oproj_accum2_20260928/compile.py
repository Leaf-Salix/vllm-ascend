"""仅CPU编译整个输出投影，检查动态tile分支和L0/L1分配，不占NPU。"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", type=int, choices=(0, 2), default=2)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / "tests/pypto_test"))
    from dsv4_csa_env import activate

    os.environ["VLLM_ASCEND_ENABLE_NZ"] = str(args.mode)
    activate()
    import pypto.language as pl
    from pypto.runtime import RunConfig

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf import decode_o_proj as mod

    projection = mod.decode_o_proj_tp1
    bf16_layout, quant_layout = mod.BF16_WEIGHT_LAYOUT, mod.QUANT_WEIGHT_LAYOUT
    tokens = pl.dynamic("OPROJ_TOKENS")

    def root(
        packed: pl.Tensor[[3072, 4096], pl.BF16],
        wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, bf16_layout],
        wo_b: pl.Tensor[[8192, 4096], pl.INT8, quant_layout],
        scale: pl.Tensor[[4096], pl.FP32],
        out: pl.Out[pl.Tensor[[tokens, 4096], pl.BF16]],
    ):
        ready = pl.create_tensor([16], dtype=pl.INT32)
        with pl.spmd(1, name_hint="input_ready") as ready_tid:
            ready_index = pl.tile.get_block_idx()
            pl.write(ready, [ready_index * 16], pl.cast(1, pl.INT32))
        out = projection(packed, wo_a, wo_b, scale, out, ready_tid)
        return out

    kernel = pl.jit(auto_scope=False)(root)
    compiled = kernel.compile(
        config=RunConfig(platform="a2a3", save_kernels=True, save_kernels_dir=str(args.output.resolve())),
    )
    # Builds CCE binaries and the host orchestration; does not create a
    # device Worker, allocate device buffers, register or execute a program.
    compiled.load()
    print(f"COMPILE_PASS mode={args.mode}; no device execution", flush=True)


if __name__ == "__main__":
    main()
