"""QKV BF16边界候选：CPU lowering、PTOAS、CCE及链接，不执行设备。"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / "tests/pypto_test"))
    from dsv4_csa_env import activate

    os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"
    activate()
    import pypto.language as pl
    from pypto.runtime import RunConfig

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.qkv_proj_rope import qkv_proj_rope

    tokens = pl.dynamic("QKV_BOUNDARY_TOKENS")

    def root(
        x: pl.Tensor[[tokens, 4096], pl.BF16],
        wqa: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ],
        wqb: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ],
        wqb_scale: pl.Tensor[[32768], pl.FP32],
        wkv: pl.Tensor[[4096, 512], pl.BF16],
        cos: pl.Tensor[[tokens, 64], pl.FP32],
        sin: pl.Tensor[[tokens, 64], pl.FP32],
        gamma_q: pl.Tensor[[1024], pl.BF16],
        gamma_kv: pl.Tensor[[512], pl.BF16],
        q: pl.Out[pl.Tensor[[tokens, 64, 512], pl.BF16]],
        kv: pl.Out[pl.Tensor[[tokens, 512], pl.BF16]],
        qr: pl.Out[pl.Tensor[[tokens, 1024], pl.INT8]],
        qr_scale: pl.Out[pl.Tensor[[tokens, 1], pl.FP32]],
    ):
        dep = pl.system.task_invalid()
        q, qa_tid = qkv_proj_rope(x, wqa, wqb, wqb_scale, wkv, cos, sin, gamma_q, gamma_kv, q, kv, qr, qr_scale, dep)
        return q, kv, qr, qr_scale

    kernel = pl.jit(auto_scope=False)(root)
    compiled = kernel.compile(
        config=RunConfig(platform="a2a3", save_kernels=True, save_kernels_dir=str(args.output.resolve()))
    )
    compiled.load()
    print("COMPILE_PASS mode=2; no device execution", flush=True)


if __name__ == "__main__":
    main()
