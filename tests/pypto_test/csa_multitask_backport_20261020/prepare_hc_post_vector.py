# SPDX-License-Identifier: Apache-2.0
"""参考用户指定Native A3实现，私有验证HC系数UB广播与四输出组并行。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE
from prepare_query_ready import replace_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="hc_post_workers_v1")
    parser.add_argument("--candidate", default="hc_post_vector_v5")
    parser.add_argument("--columns", type=int, choices=(1024, 2048), default=1024)
    args = parser.parse_args()
    if not all(value.isidentifier() for value in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    original = HERE / "sources" / args.base
    projection = replace_once(
        (original / "decode_o_proj.py").read_text(),
        "from .hc_post import hc_post_block\n",
        "from .hc_post import hc_post_vector_block as hc_post_block\n",
    )
    projection = replace_once(
        projection,
        "    y_flat = pl.reshape(x_out, [t_dim, HC_MULT * D])\n",
        "    y_flat = pl.reshape(x_out, [t_dim * HC_MULT, D])\n",
    )
    helper = textwrap.dedent(
        '''


        @pl.jit.inline
        def hc_post_vector_block(
            x_block: pl.Tensor,
            residual_flat: pl.Tensor,
            post: pl.Tensor,
            comb: pl.Tensor,
            y_flat: pl.Out[pl.Tensor],
            t0: pl.Scalar[pl.INDEX],
            n0: pl.Scalar[pl.INDEX],
            valid_rows: pl.Scalar[pl.INDEX],
            ROWS: pl.constexpr,
            COLS: pl.constexpr,
        ):
            """单token四HC输出同时运算；每元素仍按post*x及in_h=0..3顺序左折叠。"""
            # 调用仍是ROWS=1、COLS=4096；2048为核内D循环，不增加同步任务。
            # FP32物理8列满足32B搬运；有效四系数，输出也只写四行。
            # 外层循环保证t0<T；有效行固定1，不引入动态clamp妨碍局部reshape。
            post_row = pl.slice(post, [1, 8], [t0, 0], valid_shape=[1, HC_MULT])
            post_col = pl.set_validshape(pl.reshape(post_row, [8, 1]), HC_MULT, 1)
            comb_row0 = pl.slice(comb, [1, 8], [t0, 0], valid_shape=[1, HC_MULT])
            comb_col0 = pl.set_validshape(pl.reshape(comb_row0, [8, 1]), HC_MULT, 1)
            comb_row1 = pl.slice(comb, [1, 8], [t0, HC_MULT], valid_shape=[1, HC_MULT])
            comb_col1 = pl.set_validshape(pl.reshape(comb_row1, [8, 1]), HC_MULT, 1)
            comb_row2 = pl.slice(comb, [1, 8], [t0, 2 * HC_MULT], valid_shape=[1, HC_MULT])
            comb_col2 = pl.set_validshape(pl.reshape(comb_row2, [8, 1]), HC_MULT, 1)
            comb_row3 = pl.slice(comb, [1, 8], [t0, 3 * HC_MULT], valid_shape=[1, HC_MULT])
            comb_col3 = pl.set_validshape(pl.reshape(comb_row3, [8, 1]), HC_MULT, 1)
            x_f32 = pl.cast(x_block, pl.FP32)
            for hc_n in pl.range(0, COLS, 2048):
                shape_full = pl.full([8, 2048], dtype=pl.FP32, value=0.0)
                shape_tile = pl.set_validshape(shape_full, HC_MULT, 2048)
                x_part = pl.slice(x_f32, [1, 2048], [0, hc_n])
                x_group = pl.col_expand(shape_tile, x_part)
                combined = pl.row_expand_mul(x_group, post_col)
                res0 = pl.cast(pl.slice(residual_flat, [1, 2048], [t0, n0 + hc_n]), pl.FP32)
                res_group0 = pl.col_expand(shape_tile, res0)
                weighted0 = pl.row_expand_mul(res_group0, comb_col0)
                combined = pl.add(combined, weighted0)
                res1 = pl.cast(pl.slice(residual_flat, [1, 2048], [t0, D + n0 + hc_n]), pl.FP32)
                res_group1 = pl.col_expand(shape_tile, res1)
                weighted1 = pl.row_expand_mul(res_group1, comb_col1)
                combined = pl.add(combined, weighted1)
                res2 = pl.cast(pl.slice(residual_flat, [1, 2048], [t0, 2 * D + n0 + hc_n]), pl.FP32)
                res_group2 = pl.col_expand(shape_tile, res2)
                weighted2 = pl.row_expand_mul(res_group2, comb_col2)
                combined = pl.add(combined, weighted2)
                res3 = pl.cast(pl.slice(residual_flat, [1, 2048], [t0, 3 * D + n0 + hc_n]), pl.FP32)
                res_group3 = pl.col_expand(shape_tile, res3)
                weighted3 = pl.row_expand_mul(res_group3, comb_col3)
                combined = pl.add(combined, weighted3)
                result = pl.cast(combined, pl.BF16, mode="rint")
                result = pl.set_validshape(result, HC_MULT, 2048)
                y_flat = pl.assemble(y_flat, result, [t0 * HC_MULT, n0 + hc_n])
            return y_flat
        '''
    )
    # 融合反量化保留额外活跃Tile；2048实测占237728B，超过当前188416B的Vec预算。
    helper = helper.replace("2048", str(args.columns))
    hc_post = (original / "hc_post.py").read_text() + helper
    ast.parse(hc_post)
    ast.parse(projection)
    shutil.copytree(original, target)
    (target / "hc_post.py").write_text(hc_post)
    (target / "decode_o_proj.py").write_text(projection)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                changed_files=["hc_post.py", "decode_o_proj.py"],
                donor="用户指定cann-recipes-infer@6d060311 A3 hc_post_d_split.h",
                changes=f"post/comb读入UB后广播，4个有效HC输出并行，D{args.columns}核内循环控制UB占用",
                preserved="原分核/同步/依赖、八组反量化、HC前BF16边界、逐元素mul/add顺序与末尾BF16",
                boundary="仅独立行/固定HC4与D4096；不改变同步block数，无额外GM中间结果",
                status="private_native_adaptation_no_device_evidence",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
