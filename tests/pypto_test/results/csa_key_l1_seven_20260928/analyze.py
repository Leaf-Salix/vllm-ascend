"""复用主机入场/builder诊断，保留本轮异常rank的原始十步。"""

from collect_model import ROOT, load_module


def main():
    phases = load_module('seven_host_phases', ROOT.parent / 'csa_kv_k512_ep16_20260928/analyze.py')
    phases.ROOT = ROOT
    phases.main(attribution_note='本轮554b3bca，加入Key L1策略及可选builder子区间；'
                '不将跨轮差额归因于单项，不扣除EP等待，不能以未复现关闭旧尾部。')
    compact = load_module('seven_host_compact', ROOT.parent / 'csa_ub_combined_20260928/analyze.py')
    compact.ROOT = ROOT
    compact.export_tail_evidence()


if __name__ == '__main__':
    main()
