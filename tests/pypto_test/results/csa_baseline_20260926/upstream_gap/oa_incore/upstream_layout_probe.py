"""仅诊断：复现 pypto-lib O-A 的 [G,N,K] NZ 消费方向，不用于服务。"""
from pathlib import Path
import pypto.language as pl
from pypto.runtime import RunConfig

T_PAD = 384
G = 8
K = 4096
N = 1024

@pl.jit(auto_scope=False)
def oa_upstream_layout(
    x: pl.Tensor[[G * T_PAD, K], pl.BF16],
    w: pl.Tensor[[G, N, K], pl.BF16, pl.NZ],
    out: pl.Out[pl.Tensor[[T_PAD, G * N], pl.FP32]],
    rows: pl.Scalar[pl.INDEX],
    t_dim: pl.Scalar[pl.INDEX],
    row_base: pl.Scalar[pl.INDEX],
    group: pl.Scalar[pl.INDEX],
    out_col: pl.Scalar[pl.INDEX],
):
    # 与上游 TP1 主档的 M128/N128/K256、首块 matmul + 后续 pipeline 相同。
    with pl.spmd(8, name_hint='proj_a_mm'):
        n0 = pl.tile.get_block_idx() * 128
        for rb in pl.range(rows):
            r0 = rb * 128
            valid = pl.min(128, t_dim - r0)
            src = row_base + r0
            xa = pl.slice(x, [128, 256], [src, 0], valid_shape=[valid, 256])
            wa = w[group:group + 1, n0:n0 + 128, 0:256]
            acc = pl.matmul(xa, wa, out_dtype=pl.FP32, b_trans=True)
            for kb in pl.pipeline(1, K // 256, stage=2):
                k0 = kb * 256
                xk = pl.slice(x, [128, 256], [src, k0], valid_shape=[valid, 256])
                wk = w[group:group + 1, n0:n0 + 128, k0:k0 + 256]
                acc = pl.matmul_acc(acc, xk, wk, b_trans=True)
            out = pl.assemble(out, acc, [r0, out_col + n0])
    return out

if __name__ == '__main__':
    target = Path.cwd() / 'build_output/oa_incore_experiment/upstream_layout_build'
    oa_upstream_layout.warmup(config=RunConfig(platform='a2a3',save_kernels=True,save_kernels_dir=str(target),dump_passes=True))
