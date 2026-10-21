# SPDX-License-Identifier: Apache-2.0
"""单请求小T输出O_B使用M16，减少原M32的算术和量化尾零，不改变数学次序。"""

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def main():
    here = Path(__file__).resolve().parent
    original = here / "source_after_b1_half_v2"
    candidate = here / "source_b1_ob16"
    shutil.copytree(original, candidate, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = candidate / "policy.py"
    text = path.read_text()
    old = "    ob_rows = 32 if tokens <= 32 else 96 if tokens <= 96 else 128"
    new = """    # 用户允许B1独立资源分档。真实T<=16时O_B只需一个M16块；
    # 原M32会计算更多尾零行。K256累加顺序、N256及八组反量化均不变。
    # 工作区仍保留O_A所需pad128容量，只改变O_B实际使用行和量化尾零。
    ob_rows = 16 if requests == 1 and tokens <= 16 else 32 if tokens <= 32 else 96 if tokens <= 96 else 128"""
    assert text.count(old) == 1
    path.write_text(text.replace(old, new))
    for path in candidate.glob("*.py"):
        ast.parse(path.read_text())
    changes = [path.name for path in original.glob("*.py") if path.read_bytes() != (candidate / path.name).read_bytes()]
    assert changes == ["policy.py"]
    spec = importlib.util.spec_from_file_location("b1_ob16_policy", candidate / "policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rows = []
    for tokens, requests, capacity in ((1, 1, 6), (5, 1, 6), (6, 1, 6),
                                         (16, 1, 16), (17, 1, 17), (16, 16, 1),
                                         (24, 4, 6), (96, 16, 6), (168, 28, 6)):
        value = module.select_policy(tokens=tokens, requests=requests,
                                     max_query_tokens=capacity, max_seq_len=131072 + capacity)
        expected = 16 if requests == 1 and tokens <= 16 else 32 if tokens <= 32 else 96 if tokens <= 96 else 128
        assert value.ob_rows == expected
        rows.append(dict(tokens=tokens, requests=requests, capacity=capacity, ob_rows=value.ob_rows))
    (here / "B1_OB16_SOURCE.json").write_text(json.dumps(
        dict(reference=str(original), candidate=str(candidate), changed_files=changes,
             policy_cases=rows, adopted=False, device_verified=False,
             rationale="全核阶段诊断提示输出后段值得检查；M32改M16减尾行，权重字节不变，不能预测耗时减半",
             stop_condition="完整L1无收益或精度失败则停止，不扫描M8或改变整数累加/反量化次序"),
        ensure_ascii=False, indent=2) + "\n")
    print(candidate)


if __name__ == "__main__":
    main()
