# SPDX-License-Identifier: Apache-2.0
"""核对启用/关闭分支完整PTO IR；只消除来源loc、SSA名称和空白。"""

import json
from pathlib import Path
import re


def normalize(source):
    source = re.sub(r' loc\("(?:[^"\\]|\\.)*":\d+:\d+\)', "", source)
    names = {}
    source = re.sub(r"%[A-Za-z0-9_.$]+", lambda match: names.setdefault(match.group(), f"%ssa{len(names)}"), source)
    return re.sub(r"\s+", "", source)


def compare(before, after):
    old, new = normalize(before.read_text()), normalize(after.read_text())
    report = dict(before=str(before), after=str(after), alpha_equal=old == new,
                  normalized_lengths=[len(old), len(new)])
    if old != new:
        offset = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))
        report["first_difference"] = dict(offset=offset, before=old[max(0, offset - 100):offset + 250],
                                          after=new[max(0, offset - 100):offset + 250])
    return report


def main():
    here = Path(__file__).resolve().parent
    name = "kernel/ptoas/dynamic_csa_full.pto"
    result = json.loads((here / "score_owner_b24_boundary_v1/result.json").read_text())
    reference = Path(result["source_matrix"]["kernel_configs"]["spmd_reference"]).parent
    rows = {
        "stream_enabled": compare(here / "score_stream_b16_cpu_v2" / name,
                                  here / "score_stream_guarded_on_cpu_v1" / name),
        "stream_disabled": compare(here / "b1_guarded_g6_cpu_v1" / name,
                                   here / "score_stream_guarded_off_cpu_v1" / name),
        "owner_interval_b24_disabled": compare(reference / "ptoas/dynamic_csa_full.pto",
                                               here / "score_owner_interval_b24_cpu_v1" / name),
    }
    report = dict(passed=all(row["alpha_equal"] for row in rows.values()), comparisons=rows,
                  normalization="只删除source loc、SSA名称与空白；没有删除运算、常量、地址、类型或分支",
                  new_device_tests=False, hashes_computed=False)
    (here / "FOCUS_INTERVAL_IR_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
