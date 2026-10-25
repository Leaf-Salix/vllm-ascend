# SPDX-License-Identifier: Apache-2.0
"""仅扩展初始化布局的计时合同，不改写旧冻结脚本和已完成结果。"""

import ast
import json
import shutil
from pathlib import Path

from prepare import replace_once


def main():
    here = Path(__file__).resolve().parent
    old = here / "frozen"
    new = here / "frozen_v2"
    shutil.copytree(old, new, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(here / "complete_l1_v2.py", new / "complete_l1_v2.py")
    text = (old / "run_pto.py").read_text()
    source = replace_once(text, "from complete_l1 import export_l1", "from complete_l1_v2 import export_l1")
    source = replace_once(
        source,
        "    args = parser.parse_args()\n",
        '    parser.add_argument("--candidate-init-layout", choices=("separate", "fused"), '
        'default="separate")\n    args = parser.parse_args()\n',
    )
    source = replace_once(
        source,
        '        "enable_chip_swimlane": 1,\n',
        '        "enable_chip_swimlane": 1,\n        "candidate_init_layout": args.candidate_init_layout,\n',
    )
    source = replace_once(
        source,
        'export_l1(source, args.output / f"l1_{sample:02d}_{side}", side, tables[side])',
        'export_l1(source, args.output / f"l1_{sample:02d}_{side}", side, tables[side], '
        'init_layout=args.candidate_init_layout if side == "candidate" else "separate")',
    )
    ast.parse(source)
    (new / "run_pto.py").write_text(source)
    launcher = (here / "launch.sh").read_text().replace('"$here/frozen/run_pto.py"', '"$here/frozen_v2/run_pto.py"')
    launcher = launcher.replace('--frozen "$here/frozen"', '--frozen "$here/frozen_v2"')
    launcher = replace_once(
        launcher,
        '--uncompared-candidate --device "$TASK_DEVICE"',
        '--uncompared-candidate --candidate-init-layout fused --device "$TASK_DEVICE"',
    )
    (here / "launch_fused_init.sh").write_text(launcher)
    report = {
        "parent": "frozen/run_pto.py",
        "target": "frozen_v2/run_pto.py",
        "reference_unchanged": True,
        "measurement": "对照保留120角色两task；融合候选显式声明72角色单MIX，清零和屏障计入MIX首尾",
        "unchanged": [
            "同一冻结算法与SDK",
            "fixture/restore",
            "逐bit检查",
            "graph replay",
            "3次warmup及两侧各10窗交替次序",
        ],
        "scope_safety": "默认仍拒绝72条记录；只有声明fused且实际metadata证实24AIC+48AIV全设备时接受",
    }
    (here / "fused_harness_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(new)


if __name__ == "__main__":
    main()
