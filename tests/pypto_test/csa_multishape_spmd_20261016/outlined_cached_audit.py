# SPDX-License-Identifier: Apache-2.0
"""检查全阶段noinline诊断的完整覆盖及实际等待替换，不开放部分阶段混用。"""

import json
import re
from pathlib import Path

from qk_phase_clock import close_brace
from signal_polling import audit_wait_artifact


def audit(artifact, stages, signals):
    changes = json.loads((Path(stages) / "audit.json").read_text())
    boundaries = [row for row in changes if row.get("outer_math_fully_outlined")]
    if {row["role"] for row in boundaries} != {"aic", "aiv"} or len(boundaries) != 2:
        raise ValueError("外层事件只在全部数学均分离时允许清理")
    if not all(row.get("helper_events_unchanged") for row in boundaries):
        raise ValueError("阶段自身的事件被改变")
    phase_rows = [row for row in changes if "stage" in row]
    expected = {f"csa_phase_{row['role']}_{row['stage']}" for row in phase_rows}
    if len(expected) != len(phase_rows):
        raise ValueError("独立阶段不唯一")
    wait_result = audit_wait_artifact(artifact, signals, 100, cached=True)
    roles = {}
    for role in ("aic", "aiv"):
        source = (Path(artifact) / "kernels" / role / f"csa_static_{role}.cpp").read_text()
        names = re.findall(r"static (?:AICORE|__aicore__) __attribute__\(\(noinline\)\) void (csa_phase_\w+)\(", source)
        if set(names) != expected or len(names) != len(expected):
            raise ValueError("最终源码没有完整且唯一的noinline数学函数")
        for outer_role in ("aic", "aiv"):
            function = re.search(r"^(?:static )?(?:AICORE|__aicore__) void csa_static_" + outer_role + r"\([^\n]+\) \{", source, re.M)
            if function is None:
                raise ValueError("最终源码未保留外层入口")
            body = source[function.end():close_brace(source, function.end() - 1)]
            if re.search(r"\b(?:set_flag|wait_flag)\(", body):
                raise ValueError("外层数学事件仍会与helper的事件编号混用")
            calls = re.findall(r"\b(csa_phase_\w+)\(", body)
            required = {name for name in expected if name.startswith(f"csa_phase_{outer_role}_")}
            if set(calls) != required or len(calls) != len(required):
                raise ValueError("外层没有精确覆盖每个数学阶段")
        roles[role] = {"noinline_helper_count": len(names), "outer_events_fully_removed": True}
    return {"passed": True, "roles": roles, "wait_audit": wait_result,
            "full_stage_coverage": len(expected), "device_precision_validated": False,
            "eligible_for_700us_target": False,
            "scope": "C++分函数归因诊断，不是正式PTO接口实现或性能达标证据"}
