# SPDX-License-Identifier: Apache-2.0
"""HC投影已有第二波时用16组，保留sync_start并为独立metadata留出AIV。"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "policy.py"
    source = path.read_text()
    needle = "    hc_groups = min(24, max(4, 4 * ((tokens + 15) // 16)))\n"
    assert source.count(needle) == 1
    source = source.replace(
        needle,
        "    hc_linear_units = 4 * ((tokens + 15) // 16)\n"
        "    # 只在16/24组最大投影波数相同的范围内收紧资源，不给大容量增加波次。\n"
        "    # 行tile及信号容量随同一资源策略重算，数学/同步协议不改。\n"
        "    hc_overlap = hc_linear_units > 24 and (hc_linear_units + 15) // 16 == (hc_linear_units + 23) // 24\n"
        "    hc_groups = 16 if hc_overlap else min(24, max(4, hc_linear_units))\n",
        1,
    )
    ast.parse(source)
    path.write_text(source)
    print(path)


if __name__ == "__main__":
    main()
