# SPDX-License-Identifier: Apache-2.0
"""N192主体与N64尾块共享一个数学helper，避开NZ动态有效列的既有限制。"""
import ast
import shutil
import subprocess
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repo = Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=here,text=True).strip())
    formal = repo / 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
    saved = {p.name:p.read_bytes() for p in formal.glob('*.py')}
    assert all(data == (here/'source_reference'/name).read_bytes() for name,data in saved.items())
    source = {p.name:p.read_text() for p in (here/'source_candidate_v2').glob('*.py')}
    old = (here/'source_candidate/output_projection_a_stages.py').read_text()
    start = old.index('@pl.jit.inline(auto_scope=False)\ndef o_projection_a_groups_stage_nz(')
    end = old.index('@pl.jit.inline(auto_scope=False)\ndef o_projection_a_stage_nz(',start)
    group_fn = old[start:end]
    body_first = group_fn.index('        xa_first = pl.slice(')
    body_end = group_fn.index('    return o_r_pad',body_first)
    body = group_fn[body_first:body_end]
    body = '\n'.join(s[4:] if s.startswith('    ') else s for s in body.splitlines())+'\n'
    body = body.replace('A_COL_TILE','BLOCK_COLUMNS')
    helper = '''@pl.jit.inline(auto_scope=False)
def o_projection_a_nz_column_block(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    pa_r0: pl.Scalar[pl.INDEX],
    pa_rows: pl.Scalar[pl.INDEX],
    pa_src0: pl.Scalar[pl.INDEX],
    g: pl.Scalar[pl.INDEX],
    out_col_g: pl.Scalar[pl.INDEX],
    n0: pl.Scalar[pl.INDEX],
    BLOCK_COLUMNS: pl.constexpr,
):
    """一个有界N列块的原16段K256累加，主体192列与末64列共用同一数学。

    当前NZ降低不支持动态有效列，因此不构造物理N192、有效N64的动态NZ
    视图。调用方由实际nf分支选择数学常量192或64；T与有效M仍动态。
    必须保证0≤n0且n0+BLOCK_COLUMNS≤1024，不以切片语法自动处理越界。
    """
'''+body+'    return o_r_pad\n\n\n'
    calls='''        # 末块是模型1024列除数学N192后的固定余数64，不固定T/B/S。
        # 两分支调用同一数学helper，避免复制累加链或依赖动态NZ列扩展。
        if A_COL_TILE == 192 and nf == column_blocks - 1:
            o_r_pad = o_projection_a_nz_column_block(
                o_packed, wo_a, o_r_pad, pa_r0, pa_rows, pa_src0, g, out_col_g, n0, 64
            )
        else:
            o_r_pad = o_projection_a_nz_column_block(
                o_packed, wo_a, o_r_pad, pa_r0, pa_rows, pa_src0, g, out_col_g, n0, A_COL_TILE
            )
'''
    group_fn = group_fn[:body_first]+calls+group_fn[body_end:]
    text=source['output_projection_a_stages.py']
    start2=text.index('@pl.jit.inline(auto_scope=False)\ndef o_projection_a_groups_stage_nz(')
    end2=text.index('@pl.jit.inline(auto_scope=False)\ndef o_projection_a_stage_nz(',start2)
    text=text[:start2]+helper+group_fn+text[end2:]
    text=text.replace('    N尾列由显式valid_shape裁剪，不能补读下一group的权重或写越界。',
        '    ND尾列显式valid_shape裁剪；NZ用同一helper的N64数学tile，禁止越界。')
    source['output_projection_a_stages.py']=text
    for text in source.values():ast.parse(text)
    try:
        for name,text in source.items():(formal/name).write_text(text)
        shutil.copytree(formal,here/'source_candidate_v3',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    finally:
        for name,data in saved.items():(formal/name).write_bytes(data)
    print('prepared fixed N192/N64 NZ blocks with common helper')


if __name__=='__main__':
    main()
