"""复用入场/GC诊断，不新增设备执行。"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT.parent / 'csa_kv_k512_ep16_20260928/analyze.py'
    spec = importlib.util.spec_from_file_location('forward_host_analysis', source)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    analyzer.ROOT = ROOT
    analyzer.main(attribution_note=(
        '本轮冻结2d2f9ca0，验证Top-K UB与QR尾修正版组合；两侧预热计时事件，'
        '本会话候选编译在模型任务仍pending时完成，正式窗口未并行编译。'
        '不能把旧版本跨轮差额归因于任一单项。'
    ))


if __name__ == '__main__':
    main()
