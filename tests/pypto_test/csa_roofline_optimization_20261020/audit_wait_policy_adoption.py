# SPDX-License-Identifier: Apache-2.0
"""核对正式等待入口与已通过设备测试的完整 C++ 产物。

读取原始 PTOAS 输出，再调用正式 rewrite_wait_source；要求全文与已实测
modified.cpp 相同，不只检查一个常量。这样默认值或接口整理不必重复跑
同一个设备验证。该检查不把尚未结束的任务算作通过，也不验证新 shape。
"""

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TASK_FILES = (
    "WAIT_BACKOFF_TASKS.json",
    "WAIT_BACKOFF_EXPANSION_TASKS.json",
    "WAIT_BACKOFF_200_TASKS.json",
    "WAIT_BACKOFF_QUERY_TASKS.json",
    "WAIT_BACKOFF_200_FOLLOWUP_TASKS.json",
    "WAIT_BACKOFF_B1_TASKS.json",
)


def main():
    source = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/wait_codegen.py"
    spec = importlib.util.spec_from_file_location("audited_wait_codegen", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks, pending = [], []
    for manifest in TASK_FILES:
        path = HERE / manifest
        if not path.exists():
            continue
        for task in json.loads(path.read_text()):
            output = REPO / task["output"]
            result_file = output / "result.json"
            if not result_file.exists() or not json.loads(result_file.read_text()).get("passed"):
                pending.append(task["task"])
                continue
            query = (output / "WAIT_BACKOFF_QUERY_AUDIT.json").exists()
            audit_file = output / (
                "WAIT_BACKOFF_QUERY_AUDIT.json" if query else "WAIT_BACKOFF_COMPARISON.json"
            )
            if not audit_file.exists():
                pending.append(task["task"])
                continue
            audit = json.loads(audit_file.read_text())
            for role in audit["roles"]:
                directory = (
                    output / ("compile_" + role["role"]) if query else REPO / role["probe"]
                )
                original = (directory / "original.cpp").read_text()
                expected = (directory / "modified.cpp").read_text()
                actual, info = module.rewrite_wait_source(original, backoff_ticks=role["ticks"])
                assert actual == expected, directory
                assert info["backoff_syscnt_ticks"] == role["ticks"]
                default_matches = None
                if role["ticks"] == module.WAIT_BACKOFF_TICKS:
                    default_actual, _ = module.rewrite_wait_source(original)
                    default_matches = default_actual == expected
                    assert default_matches, directory
                checks.append(dict(
                    task=task["task"], directory=str(directory.relative_to(REPO)),
                    ticks=role["ticks"], exact_cpp_match=True,
                    default_matches_measured_cpp=default_matches,
                ))
    invalid = (True, 0, 1001, 150.0, "150")
    for value in invalid:
        try:
            module.rewrite_wait_source("", backoff_ticks=value)
        except ValueError:
            continue
        raise AssertionError(f"不合法的参数未被拒绝：{value!r}")
    report = dict(
        passed=True, default_ticks=module.WAIT_BACKOFF_TICKS, checks=checks,
        pending_tasks=pending, invalid_parameter_rejections=len(invalid),
        scope="正式显式参数及默认入口与已实测C++全文一致；未重复设备验证",
    )
    (HERE / "WAIT_EXPLICIT_POLICY_CODEGEN_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(dict(
        passed=True, default_ticks=module.WAIT_BACKOFF_TICKS,
        matched_cpp=len(checks), pending_tasks=pending,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
