# SPDX-License-Identifier: Apache-2.0
"""将已验证的 WS 列池迁移到当前 150 tick 参考，保留历史实验。

只替换 Score 文件。旧试验完整精度通过，两重点档曾小幅改善，但因当时
20 μs 门槛停止。用户已撤销该门槛；这里复核当前环境下的小收益，不修改
tile、数学或同步，也不把旧轮次收益与本轮相加。
"""

import ast
import json
from pathlib import Path
import shutil


def main():
    here = Path(__file__).resolve().parent
    reference = here / "source_after_wait150"
    previous = here.parent / "csa_critical_path_20261020/score_split_pool/source_candidate"
    target = here / "source_score_pool_goal10"
    # 先确认旧候选没有其它未解释的算子差异，禁止静默搬入额外实验。
    changed = []
    for path in reference.glob("*.py"):
        old = previous / path.name
        if ast.dump(ast.parse(path.read_text())) != ast.dump(ast.parse(old.read_text())):
            changed.append(path.name)
    if sorted(changed) != ["indexer_score_stages.py", "wait_codegen.py"]:
        raise ValueError(f"历史源与当前参考的差异超出预期：{changed}")
    shutil.copytree(reference, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(previous / "indexer_score_stages.py", target / "indexer_score_stages.py")
    record = {
        "reference": str(reference),
        "previous_candidate": str(previous),
        "candidate": str(target),
        "changed_operator_files": ["indexer_score_stages.py"],
        "wait_ticks": 150,
        "new_math_or_policy": False,
        "previous_precision": "完整 B16/B28 冻结检查通过，见原 score_split_pool/PERFORMANCE.json",
        "previous_saved_us": {"B16": 5.42, "B28": 12.12},
        "adopted": False,
        "new_device_result": "尚未取得；不能复用旧性能作为当前收益",
    }
    (here / "SCORE_POOL_GOAL10_SOURCE.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
