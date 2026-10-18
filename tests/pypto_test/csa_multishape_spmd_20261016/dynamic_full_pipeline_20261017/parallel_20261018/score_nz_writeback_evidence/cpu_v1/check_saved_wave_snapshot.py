# SPDX-License-Identifier: Apache-2.0
"""离线重验已保存的设备 wave 信号；不能代替丢失的数值结果或重新上卡。

输入是 run_device 在失败后回读的原始控制区及同轮 result.json。这里只核对
实际 metadata 对应的 ROOT/TOPK/PLAN 写者，并验证 true 的 1/-1 两种表示。
输出保留所有不匹配项；不会改写原测试结果，也不会把原失败轮标为 PASS。
"""

import argparse
import json
from pathlib import Path

from wave_signal_checks import expected_wave_writers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run / "result.json").read_text())
    tensors = json.loads((args.run / "failure_snapshot.json").read_text())["tensors"]
    plan = tensors["group_plan"]["values"]
    offsets = tensors["group_offsets"]["values"]
    signal = tensors["signal"]["word0_by_phase_core"]
    kwargs = dict(tokens=run["batch"] * run["query_tokens"], requests=run["batch"],
                  offsets=offsets, policy=run["policy"])
    waves = expected_wave_writers(plan=plan, **kwargs)
    other_true = plan[:]
    other_true[6] = 1 if plan[6] == -1 else -1
    assert expected_wave_writers(plan=other_true, **kwargs) == waves
    rejected = []
    for bad_flag in (0, 2, -2, 37):
        bad = plan[:]
        bad[6] = bad_flag
        try:
            expected_wave_writers(plan=bad, **kwargs)
        except AssertionError:
            rejected.append(bad_flag)
        else:
            raise AssertionError(f"错误平衡标记未拒绝：{bad_flag}")
    # 该快照只用于首次调用；后续代际要用 WaveSignalVerifier 递推未使用槽。
    assert signal[0] == [1] * 72
    expected = [[0] * 72 for _ in signal[60:]]
    for wave, info in enumerate(waves):
        for stage, owners in enumerate(info["writers"]):
            for owner in owners:
                expected[3 * wave + stage][owner] = 1
    mismatches = [[60 + phase, core, wanted, signal[60 + phase][core]]
                  for phase, row in enumerate(expected) for core, wanted in enumerate(row)
                  if wanted != signal[60 + phase][core]]
    report = dict(source=str(args.run.resolve()), plan=plan, waves=waves,
                  rejected_flags=rejected, equivalent_true_encodings=[-1, 1],
                  mismatches=mismatches, passed=not mismatches,
                  scope="首次调用原始设备控制区离线核对；不改原失败结果，不证明数值精度")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(waves=len(waves), mismatches=len(mismatches), passed=not mismatches)))
    assert not mismatches, mismatches


if __name__ == "__main__":
    main()
