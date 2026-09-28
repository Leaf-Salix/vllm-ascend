"""沿用主机/builder及既有标记间隙，只读本轮正式样本。"""

from collect_model import REVISION, ROOT, load_module


def main():
    phases = load_module('targeted_phases', ROOT.parent / 'csa_kv_k512_ep16_20260928/analyze.py')
    phases.ROOT = ROOT
    phases.PHASES.update({
        'execute_to_input_sync_gap': ('execute_entry', 'input_sync_begin'),
        'inputs_to_coordination_gap': ('inputs_end', 'batch_coordination_begin'),
        'coordination_to_metadata_gap': ('batch_coordination_end', 'attention_metadata_begin'),
        'metadata_to_preprocess_gap': ('attention_metadata_end', 'preprocess_begin'),
    })
    phases.main(attribution_note=f'本轮{REVISION}仅验收长B4/B8新预取及短B16控制；'
                '保留原始尾部，不扣除EP等待，不将跨轮差额归因于单项。')
    compact = load_module('targeted_tail', ROOT.parent / 'csa_ub_combined_20260928/analyze.py')
    compact.ROOT = ROOT
    compact.export_tail_evidence()


if __name__ == '__main__':
    main()
