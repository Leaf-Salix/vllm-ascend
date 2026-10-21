# SPDX-License-Identifier: Apache-2.0
"""只收窄重排的Host资源区间；启用的SPMD源码保持原实测版本。

T<=128没有第二Q_B M128块；129..144只有两块KV M64后的一个M16尾块。
该区间先保留原分工，重排只用于T>144到256。此条件不枚举B24/B28，
组数/有效长度及实际是否需要置换仍由本轮device计划确定。
"""

import importlib.util
import json
from pathlib import Path
import shutil
import sys


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    here = Path(__file__).resolve().parent
    original = here / "source_score_owner_balance"
    target = here / "source_score_owner_interval"
    shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "policy.py"
    text = path.read_text()
    old = "        score_prefix_balance=long_history and group == 6 and 128 < tokens <= 256,"
    new = """        # 两个KV M64主体后，仅一个M16尾块的区间保持原排程；
        # 更重的尾部区间才尝试把较短Score交给较晚空闲的核。
        # 这是资源边界144..256，非B24排除表；实际不均衡仍在device判断。
        score_prefix_balance=long_history and group == 6 and (2 * 64 + 16) < tokens <= 256,"""
    assert text.count(old) == 1
    path.write_text(text.replace(old, new))
    before = load(original / "policy.py", "score_owner_before_interval")
    after = load(path, "score_owner_after_interval")
    cases = []
    for tokens, requests in ((6, 1), (96, 16), (128, 22), (129, 22), (144, 24), (145, 25), (168, 28), (192, 32), (256, 43), (257, 43)):
        args = dict(tokens=tokens, requests=requests, max_query_tokens=6, max_seq_len=131078)
        a, b = before.select_policy(**args).constexprs(), after.select_policy(**args).constexprs()
        difference = {k: [a[k], b[k]] for k in a if a[k] != b[k]}
        assert not set(difference) - {"SCORE_PREFIX_BALANCE"}
        cases.append(dict(arguments=args, changed=difference))
    unchanged = [p.name for p in original.glob("*.py") if p.name != "policy.py" and p.read_bytes() == (target / p.name).read_bytes()]
    assert len(unchanged) == 37
    controls = after.select_policy(tokens=144, requests=24, max_query_tokens=6, max_seq_len=131078).constexprs()
    (here / "controls_score_owner_interval_b24.json").write_text(json.dumps(controls, indent=2) + "\n")
    report = dict(reference=str(original), candidate=str(target), changed_files=["policy.py"],
                  identical_operator_files=unchanged, policy_cases=cases,
                  b28_enabled_operator_and_controls_identical=True,
                  b24_disabled_codegen_pending=True, adopted=False,
                  performance_evidence=["score_owner_b28_full_v2", "score_owner_b28_reverse_v1"],
                  rejected_proof_candidate="source_score_owner_uniform：B24慢9.34us/1窗快，B28慢6.14us/0窗快")
    (here / "SCORE_OWNER_INTERVAL_SOURCE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
