# SPDX-License-Identifier: Apache-2.0
"""重新评估小容量的两条局部前段链；只调整编排分档，不改核内数学。"""

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
    needle = "FUSED_DAG_MIN_TOKENS = 64"
    assert source.count(needle) == 1
    source = source.replace(
        "# 两个M32容量起使用局部融合编排；小容量保留已测更快的独立阶段。\n" + needle,
        "# 实验：前段已重构成两条并行局部MIX，重新评估小容量；不是沿用旧v25融合结论。\n"
        "# 所有正容量启用；正式采用仍须按各容量完整L1验证，空调用由外层处理。\n"
        "FUSED_DAG_MIN_TOKENS = 1",
    )
    assert needle not in source
    ast.parse(source)
    path.write_text(source)
    print(target)


if __name__ == "__main__":
    main()
