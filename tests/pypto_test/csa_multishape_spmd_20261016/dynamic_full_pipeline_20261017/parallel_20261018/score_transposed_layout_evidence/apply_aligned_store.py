# SPDX-License-Identifier: Apache-2.0
"""在prepare.py生成的Key-major首版上应用64B整行写回；拒绝重复应用。

仅扩大已经计算出的WS写回列数及内部GM行距。AIV仍只读取前8列，
前6列有效数学、全部矩阵、Vector操作与同步保持。
"""

import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parent/'source_candidate')
    args=parser.parse_args()
    file=args.source/'indexer_score_stages.py'
    text=file.read_text()
    for value in ('buf_reduced_pair','buf_last_scores'):
        before=f'pl.set_validshape({value}, qk_panel_cols, query_group_size)'
        after=f'pl.set_validshape({value}, qk_panel_cols, NATIVE_QLI_WEIGHT_ROWS)'
        if text.count(before)!=1:
            raise ValueError('必须是尚未应用的Key-major首版：'+before)
        text=text.replace(before,after,1)
    text=text.replace('每槽GM布局[score_tile,8]，只写前G列；两AIV读同一槽并恢复原',
        '每槽GM布局[score_tile,16]，整行64B写出全部已计算列；前G列有效。\n    两AIV仅读前8列并恢复原')
    text=text.replace('双槽GM改为[24*2*1024,8]','双槽GM改为[24*2*1024,16]')
    text=text.replace('Key-major GM每槽[1024,8]','Key-major GM每槽[1024,16]，仅读取前8列')
    binding=args.source/'binding.py'
    bound=binding.read_text()
    old='(24 * 2 * self.policy.score_tile, 8)'
    if bound.count(old)!=1:
        raise ValueError('GM交接区不是首版的8列布局')
    bound=bound.replace(old,'(24 * 2 * self.policy.score_tile, 16)',1)
    file.write_text(text)
    binding.write_text(bound)


if __name__=='__main__':
    main()
