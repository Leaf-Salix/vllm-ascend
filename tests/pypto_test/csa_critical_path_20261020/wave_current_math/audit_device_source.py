# SPDX-License-Identifier: Apache-2.0
"""核对正式源未变化、发布获取实现未变化及CPU审计覆盖实际设备C++。"""

import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
REPO = CASE.parents[2]


def normalize_cpp(text):
    # 生成文件没有含注释符的业务字符串；忽略路径注释和排版，不修改文件。
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def main():
    reference = CASE / "source_after_score_split"
    formal = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    candidate = HERE / "source_candidate"
    reference_files = sorted(p.name for p in reference.glob("*.py"))
    if reference_files != sorted(p.name for p in formal.glob("*.py")):
        raise ValueError("正式源文件集合与当前基线不一致")
    if any((reference / n).read_bytes() != (formal / n).read_bytes()
           for n in reference_files):
        raise ValueError("正式源发生变动，不能继续称为13259f92基线")
    sync_names = ("signals.py", "phase_sync.py", "wait_codegen.py")
    unchanged_sync = {
        n: (reference / n).read_bytes() == (candidate / n).read_bytes()
        for n in sync_names
    }
    if not all(unchanged_sync.values()):
        raise ValueError("候选修改了原发布获取或退避实现")
    jobs = json.loads((HERE / "DEVICE_TASKS_V1.json").read_text())
    reports = {}
    for batch, task in jobs.items():
        cpu = HERE / f"cpu_b{batch}_v1/kernel/ptoas/dynamic_csa_full.cpp"
        device = HERE / task["folder"]
        invocations = [json.loads(line) for line in
                       (device / "COMPILER_INVOCATIONS.jsonl").read_text().splitlines()]
        candidates = list(dict.fromkeys(
            Path(row["pto"]).with_suffix(".cpp") for row in invocations
            if Path(row["pto"]).name == "dynamic_csa_full.pto"
        ))
        normalized_cpu = normalize_cpp(cpu.read_text())
        matches = [str(path) for path in candidates if path.exists()
                   and normalize_cpp(path.read_text()) == normalized_cpu]
        if not matches:
            raise ValueError(f"B{batch}实际设备C++没有匹配完整CPU审计产物")
        reports[batch] = {"cpu_cpp": str(cpu), "device_cpp_matches": matches,
                          "equal_after_comments_whitespace": True}
    result = {
        "passed": True, "formal_files": len(reference_files),
        "formal_equal_current_reference": True,
        "unchanged_signal_implementations": unchanged_sync,
        "device_codegen": reports,
        "note": "直接比较内容，不计算hash；不将代码相同当成性能或全场景正确性证据",
    }
    (HERE / "DEVICE_SOURCE_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print("正式37源、发布获取及两档实际设备C++核对通过")


if __name__ == "__main__":
    main()
