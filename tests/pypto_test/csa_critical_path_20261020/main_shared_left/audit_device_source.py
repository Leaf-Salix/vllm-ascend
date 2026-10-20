# SPDX-License-Identifier: Apache-2.0
"""确认共享Left设备实际C++匹配已审计CPU产物，直接比内容，不计算hash。"""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location("cpp_compare", HERE.parent / "wave_current_math/audit_device_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reports = {}
    jobs = json.loads((HERE / "TASKS_V2.json").read_text())
    for batch, row in jobs.items():
        cpu = HERE / f"cpu_b{batch}_v2/kernel/ptoas/dynamic_csa_full.cpp"
        invocations = [json.loads(line) for line in
                       (HERE / row["folder"] / "COMPILER_INVOCATIONS.jsonl").read_text().splitlines()]
        device = list(dict.fromkeys(Path(r["pto"]).with_suffix(".cpp") for r in invocations
                                    if Path(r["pto"]).name == "dynamic_csa_full.pto"))
        normalized = module.normalize_cpp(cpu.read_text())
        matches = [str(p) for p in device if p.exists()
                   and module.normalize_cpp(p.read_text()) == normalized]
        assert matches, batch
        reports[batch] = {"cpu_cpp": str(cpu), "actual_matching_device_cpp": matches,
                          "equal_after_comments_whitespace": True}
    formal = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    reference = HERE.parent / "source_after_score_split"
    assert sorted(p.name for p in formal.glob("*.py")) == sorted(p.name for p in reference.glob("*.py"))
    assert all(p.read_bytes() == (formal / p.name).read_bytes() for p in reference.glob("*.py"))
    (HERE / "DEVICE_SOURCE_AUDIT.json").write_text(json.dumps({
        "passed": True, "formal_unchanged": True, "device_codegen": reports,
        "note": "实际编译内容对照，不替代硬件精度或性能",
    }, ensure_ascii=False, indent=2) + "\n")
    print("两档实际C++及正式源核对通过")


if __name__ == "__main__":
    main()
