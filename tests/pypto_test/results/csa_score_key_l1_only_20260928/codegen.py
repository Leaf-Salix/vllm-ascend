"""复用已验证的地址解析，核查三query仅Key L1预取的实际分配。"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location(
        'key_codegen', ROOT.parent / 'csa_score_key_l1_pair_20260928/codegen.py')
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    reporter.main(root=ROOT, kernel='kernels/aic/indexer_score_topk_native_pair_0_aic.cpp',
                  changed_binary='incore_40_aic_indexer_score_topk_native_pair_0_aic_a2a3.bin',
                  scope='长档三query/M192/N128，只预取下一Key到L1；生成地址/静态等待，不代表设备收益。')


if __name__ == '__main__':
    main()
