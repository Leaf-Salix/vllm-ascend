# SPDX-License-Identifier: Apache-2.0
"""修正N192候选的显式末64列范围；保留v1完整失败源码。"""
import ast
import json
import shutil
import subprocess
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repo = Path(subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel'], cwd=here, text=True,
    ).strip())
    formal = repo / 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
    saved = {p.name: p.read_bytes() for p in formal.glob('*.py')}
    assert all(v == (here / 'source_reference' / n).read_bytes() for n,v in saved.items())
    source = {p.name: p.read_text() for p in (here / 'source_candidate').glob('*.py')}
    text = source['output_projection_a_stages.py']
    text = text.replace('N尾列由GM视图的valid_shape裁剪', 'N尾列由显式valid_shape裁剪')
    text = text.replace('Tensor slice保留有效列', 'Tensor slice必须显式传入有效列')
    for old in ['        n0 = nf * A_COL_TILE\n', '        n0 = pl.max(0, nf) * A_COL_TILE\n']:
        assert text.count(old) == 1
        text = text.replace(old, old +
            '        # 切片语法不会自动钳制越过模型N轴的固定size；末块显式限制64列。\n'
            '        pa_cols = pl.min(A_COL_TILE, O_LORA - n0)\n', 1)
    for old,new in [
        ('        wa_first = wo_a[g : g + 1, 0:A_K_TILE, n0 : n0 + A_COL_TILE]',
         '        wa_first = pl.slice(wo_a, [1, A_K_TILE, A_COL_TILE], [g, 0, n0],\n'
         '                            valid_shape=[1, A_K_TILE, pa_cols])'),
        ('            wa_k_chunk = wo_a[g : g + 1, k0 : k0 + A_K_TILE, n0 : n0 + A_COL_TILE]',
         '            wa_k_chunk = pl.slice(wo_a, [1, A_K_TILE, A_COL_TILE], [g, k0, n0],\n'
         '                                  valid_shape=[1, A_K_TILE, pa_cols])'),
    ]:
        assert text.count(old) == 2
        text = text.replace(old,new)
    source['output_projection_a_stages.py'] = text
    for text in source.values(): ast.parse(text)
    try:
        for name,text in source.items(): (formal / name).write_text(text)
        shutil.copytree(formal, here / 'source_candidate_v2',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    finally:
        for name,data in saved.items(): (formal / name).write_bytes(data)
    print('prepared v2 with explicit valid columns')


if __name__ == '__main__':
    main()
