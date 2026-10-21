# SPDX-License-Identifier: Apache-2.0
"""均衡Score的设备端快速证明：省掉没有收益的逐group/leaf成本累加。

证明条件是groups为24的倍数、leaves与24互素、各group visible相同。
此时item=group*leaves+leaf模24，每个worker获得相同数量的每一种叶。
叶边界按1024连续铺开，所以每核步数有精确闭式；不是B24单点例外。
"""

import ast
import json
import math
from pathlib import Path
import shutil


def main():
    here = Path(__file__).resolve().parent
    original = here / "source_score_owner_balance"
    candidate = here / "source_score_owner_uniform"
    shutil.copytree(original, candidate, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = candidate / "score_owner_stages.py"
    text = path.read_text()
    start = text.index("        for group in pl.range(groups):")
    end = text.index("        maximum = pl.cast(0, pl.INT32)", start)
    previous = text[start:end]
    proof = '''        # 可证明均衡的完整轮：每核得到同样的各叶集合，无须逐项累计。
        # 24的质因子仅2、3；不把group数==24写成B24白名单。
        uniform = pl.cast(0, pl.INT32)
        common_visible = pl.cast(0, pl.INDEX)
        if groups % GROUPS == 0 and leaves % 2 != 0 and leaves % 3 != 0:
            uniform = pl.cast(1, pl.INT32)
            common_visible = pl.cast(pl.read(workspace, [descriptor_base + 3]), pl.INDEX)
            for probe_group in pl.range(1, groups):
                probe_visible = pl.cast(pl.read(workspace, [descriptor_base + GROUP_DESCRIPTOR_WORDS * probe_group + 3]), pl.INDEX)
                if probe_visible != common_visible:
                    uniform = pl.cast(0, pl.INT32)
        if uniform != 0:
            covered = pl.max(pl.min(common_visible, (leaves * leaf_tiles + extra) * SCORE_TILE), 0)
            uniform_steps = ((covered + SCORE_TILE - 1) // SCORE_TILE) * (groups // GROUPS)
            for uniform_core in pl.range(GROUPS):
                pl.write(workspace, [base + uniform_core], pl.cast(uniform_steps, pl.INT32))
        else:
'''
    text = text[:start] + proof + "".join("    " + line if line.strip() else line for line in previous.splitlines(True)) + text[end:]
    path.write_text(text)
    ast.parse(text)
    checked = 0
    # 独立逐项枚举验证闭式，不调用待验证的设备helper，也不拟合微秒。
    for groups in (24, 48):
        for leaves in range(1, 33):
            if math.gcd(leaves, 24) != 1:
                continue
            for tiles in (1, 3, 8, 16):
                for extra in (0, leaves - 1):
                    capacity = (leaves * tiles + extra) * 1024
                    for visible in (0, 1, 1023, 1024, 32769, capacity - 1, capacity, capacity + 1):
                        costs = [0] * 24
                        for group in range(groups):
                            for leaf in range(leaves):
                                begin = (leaf * tiles + min(leaf, extra)) * 1024
                                valid = max(min(visible - begin, (tiles + int(leaf < extra)) * 1024), 0)
                                costs[(group * leaves + leaf) % 24] += (valid + 1023) // 1024
                        closed = ((max(min(visible, capacity), 0) + 1023) // 1024) * (groups // 24)
                        assert costs == [closed] * 24, (groups, leaves, tiles, extra, visible, costs)
                        checked += 1
    report = dict(reference=str(original), candidate=str(candidate), changed_files=[path.name],
                  host_enumeration_cases=checked, exact_uniform_cost_proof_passed=True,
                  device_verified=False, adopted=False,
                  reason="B28正常反向净省12.78us/5窗，B24身份回退慢16.16us/5窗；先降低无收益的planner工作")
    (here / "SCORE_OWNER_UNIFORM_SOURCE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
