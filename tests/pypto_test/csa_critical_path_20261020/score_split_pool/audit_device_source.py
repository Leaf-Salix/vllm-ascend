# SPDX-License-Identifier: Apache-2.0
"""核对设备PTOAS原始输出及100tick等待变换后的CCEC输入，直接比内容。"""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    compare = module_from_file("pool_cpp_compare", HERE.parent / "wave_current_math/audit_device_source.py")
    wait = module_from_file("pool_wait", HERE / "source_candidate/wait_codegen.py")
    reports = {}
    for batch, job in json.loads((HERE / "TASKS.json").read_text()).items():
        cpu = HERE / f"cpu_b{batch}_v1/kernel/ptoas/dynamic_csa_full.cpp"
        probe = HERE / job["folder"] / "codegen_probe"
        original = probe / "original.cpp"
        modified = probe / "modified.cpp"
        audit = json.loads((probe / "audit.json").read_text())
        assert audit["wait_backoff"] and not audit["phase_clock"]
        source = cpu.read_text()
        assert compare.normalize_cpp(source) == compare.normalize_cpp(original.read_text())
        expected, changes = wait.rewrite_wait_source(source)
        assert changes == audit["wait_change"]
        assert compare.normalize_cpp(expected) == compare.normalize_cpp(modified.read_text())
        reports[batch] = {
            "cpu_raw_cpp": str(cpu), "actual_ptoas_cpp": str(original),
            "actual_cpp_after_wait_rewrite": str(modified), "wait_change": changes,
            "raw_equal_after_comments_whitespace": True,
            "rewritten_equal_after_comments_whitespace": True,
        }
    reference = HERE.parent / "source_after_score_split"
    formal = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    assert sorted(p.name for p in reference.glob("*.py")) == sorted(p.name for p in formal.glob("*.py"))
    assert all(p.read_bytes() == (formal / p.name).read_bytes() for p in reference.glob("*.py"))
    (HERE / "DEVICE_SOURCE_AUDIT.json").write_text(json.dumps({
        "passed": True, "formal_unchanged": True, "comparisons": reports,
        "note": "原始PTOAS及等待变换后的实际后端输入相符，不声明机器指令逐字相同；不做hash",
    }, ensure_ascii=False, indent=2) + "\n")
    print("两档原始PTOAS及100tick变换后源码相符；正式源不变")


if __name__ == "__main__":
    main()
