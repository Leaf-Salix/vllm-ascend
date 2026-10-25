# SPDX-License-Identifier: Apache-2.0
"""按用户要求删除独立清零与回退路径，所有档位只保留一个MIX SPMD。"""

import ast
import json
import shutil
import textwrap
from pathlib import Path

from prepare import replace_once


def remove_flag(source):
    return "".join(line for line in source.splitlines(True) if "FUSED_SIGNAL_INIT" not in line)


def transform_entry(source):
    tree = ast.parse(source)
    changes = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "_clear_internal_signal":
            changes.append((min(d.lineno for d in node.decorator_list) - 1, node.end_lineno, ""))
        if node.name not in ("decode_csa_spmd", "decode_csa_spmd_debug"):
            continue
        guard = next(
            n
            for n in node.body
            if isinstance(n, ast.If) and ast.unparse(n.test) == "pl.system.available_cluster_count() == GROUPS"
        )
        lines = source.splitlines(True)
        body = "".join(lines[guard.body[0].lineno - 1 : guard.body[-1].end_lineno])
        body = "\n".join(line[4:] if line.startswith("    ") else line for line in body.split("\n"))
        changes.append((guard.lineno - 1, guard.end_lineno, body))
    lines = source.splitlines(True)
    for start, end, replacement in sorted(changes, reverse=True):
        lines[start:end] = [replacement]
    result = remove_flag("".join(lines))
    result = result.replace(
        "    # Fixed CSA ownership is valid for exactly 24 MIX groups. The query\n"
        "    # also proves full occupancy to the hard-syncall compiler verifier.\n"
        "    # Other device widths retain the original separate initialization.\n",
        "    # 所有输入档位统一在主MIX内清零并硬同步；没有独立初始化task。\n"
        "    # binding准备时检查24AIC/48AIV；全设备查询保证硬屏障全占用。\n",
    )
    result = result.replace(
        "    decode_csa_spmd先清零内部信号，再创建一次24组业务SPMD；_csa_core_group把AIC和双AIV角色\n",
        "    decode_csa_spmd只创建一次MIX SPMD，清零、同步与业务均在其中；_csa_core_group把AIC和双AIV角色\n",
    )
    result = result.replace(
        "    24组全占用时每次调用在主MIX内清零并硬屏障，其余设备保留独立清零。图重放包含同样的\n",
        "    所有档位每次调用在同一个主MIX内清零并硬屏障。绑定要求24AIC/48AIV，图重放包含同样的\n",
    )
    return result


def transform_schedule(source):
    tree = ast.parse(source)
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "FUSED_SIGNAL_INIT"]
    assert len(guards) == 1
    guard = guards[0]
    lines = source.splitlines(True)
    body = "".join(lines[guard.lineno : guard.end_lineno])
    lines[guard.lineno - 1 : guard.end_lineno] = [textwrap.indent(textwrap.dedent(body), "    ")]
    return remove_flag("".join(lines))


def transform_binding(source):
    helper = '''
REQUIRED_CUBE_CORES = 24
REQUIRED_VECTOR_CORES = 2 * REQUIRED_CUBE_CORES


def validate_core_geometry(cube_core_num: int, vector_core_num: int) -> None:
    """固定CSA核角色映射要求整卡24C48V；硬件不匹配时在提交前拒绝。"""
    if (cube_core_num, vector_core_num) != (REQUIRED_CUBE_CORES, REQUIRED_VECTOR_CORES):
        raise NotImplementedError(
            "CSA单SPMD的核内清零/硬同步要求24个AIC和48个AIV；"
            f"当前为{cube_core_num}个AIC和{vector_core_num}个AIV，不启动其他task路径"
        )

'''
    source = replace_once(source, "\n\nclass PreparedDynamicCSA:", "\n" + helper + "\nclass PreparedDynamicCSA:")
    return replace_once(
        source,
        "        device = hidden.device\n",
        "        device = hidden.device\n"
        '        if device.type == "npu":\n'
        "            import torch_npu\n\n"
        "            # 只查询硬件属性，不读取metadata或Tensor值；绑定准备时执行一次。\n"
        "            properties = torch_npu.npu.get_device_properties(device)\n"
        "            validate_core_geometry(properties.cube_core_num, properties.vector_core_num)\n",
    )


def main():
    here = Path(__file__).resolve().parent
    parent = here / "candidates/init_coeff_production_v1"
    target = here / "candidates/init_coeff_single_spmd_v2"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, transform in (
        ("entry.py", transform_entry),
        ("cube_schedule.py", transform_schedule),
        ("vector_schedule.py", transform_schedule),
        ("binding.py", transform_binding),
    ):
        path = target / name
        source = transform(path.read_text())
        ast.parse(source)
        path.write_text(source)
    entry = (target / "entry.py").read_text()
    assert "_clear_internal_signal" not in entry and "FUSED_SIGNAL_INIT" not in entry
    assert entry.count("with pl.spmd(") == 2  # 普通/诊断入口各一次。
    for filename in ("cube_schedule.py", "vector_schedule.py"):
        assert "FUSED_SIGNAL_INIT" not in (target / filename).read_text()
    report = {
        "parent": str(parent),
        "candidate": str(target),
        "user_requirement": "需要的清零全部包含在一个SPMD内，不按场景保留两task路径",
        "normal_and_debug_entries_each_one_spmd": True,
        "standalone_clear_and_fallback_removed": True,
        "fixed_ownership_hardware_check": "binding准备时查询torch_npu设备属性，要求24AIC/48AIV",
        "math": "原组合的FUSED_SIGNAL_INIT=True路径无条件执行，系数融合及业务计算不变",
        "status": "CPU_gate_pending",
    }
    (here / "single_spmd_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
