"""复用本轮正式样本诊断入场尾部，不增加设备执行。"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT.parent / "csa_kv_k512_ep16_20260928/analyze.py"
    spec = importlib.util.spec_from_file_location("forward_host_analysis", source)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    analyzer.ROOT = ROOT
    analyzer.main(attribution_note=(
        "本轮验证d1f170ff的四路Top-K与HC输入/RMS组合源码，"
        "两侧均预热计时事件；不能把跨轮变化单独归因于某项核内优化。"
    ))


if __name__ == "__main__":
    main()
