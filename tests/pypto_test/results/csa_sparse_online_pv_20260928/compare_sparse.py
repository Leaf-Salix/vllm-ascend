"""CPU 核对单侧缩放与原累计算法；固定输入证据不代替整层/模型验收。"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402


def main():
    torch.set_num_threads(4)
    old_root = ROOT.parent / 'csa_softmax_cumulative_20260928/cumulative'
    old_report = json.loads((old_root / 'report.json').read_text())
    report = json.loads((ROOT / 'fixed_native/report.json').read_text())
    tail = json.loads((ROOT / 'tail_b3/report.json').read_text())
    if Path(old_report['input']).resolve() != Path(report['input']).resolve():
        raise ValueError('已有累计算法输出与本候选使用的固定 Native 输入不一致')
    expected = torch.load(old_root / 'output.pt', map_location='cpu', weights_only=True)
    actual = torch.load(ROOT / 'fixed_native/output.pt', map_location='cpu', weights_only=True)
    equal = compare_tensor(actual, expected, 0, 0)
    errors = []
    if equal['status'] != 'PASS':
        errors.append('与原累计最大值算法逐元素零容差不一致，需要先解释，不能进入性能测量')
    if report['comparison']['nonfinite'] != 0:
        errors.append('固定 Native 输入产生非有限值')
    if tail['status'] != 'PASS':
        errors.append('B3 解析尾块失败')
    result = {
        'status': 'FAIL' if errors else 'PASS', 'errors': errors,
        'device': report['device'], 'old_cumulative_source': str(old_root),
        'same_cumulative_algorithm': equal, 'native_comparison': report['comparison'],
        'tail_b3': tail['comparison'],
        'limits': '旧累计最大值输出来自已完成的固定 Native 输入诊断；只核对当前单侧缩放的算术等价性。'
                  '不把对 Native 零容差差异改为 PASS，不覆盖完整 CSA 或 EP16。',
    }
    (ROOT / 'sparse_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(result)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
