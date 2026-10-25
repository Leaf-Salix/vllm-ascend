# SPDX-License-Identifier: Apache-2.0
"""核对八档候选生成物中的初始化到达、独立准备、释放等待和首次通知顺序。"""

import hashlib
import json
import re
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    result = {}
    for case in range(1, 9):
        directory = here / f"runs/codegen_init_arrive_overlap_C{case:02}_v1"
        report = json.loads((directory / "result.json").read_text())
        assert report["passed"] and report["stage"] == "cpu_backend_compiled"
        files = list(directory.glob("build_output/*/ptoas/dynamic_csa_full.cpp"))
        assert sorted(f.read_text().count("SYNCALL<") for f in files) == [0, 1, 2]
        candidate = next(f for f in files if f.read_text().count("SYNCALL<") == 1)
        text = candidate.read_text()
        vector = text[text.index("AICORE void vector_schedule(") :]
        arrive = re.search(
            r"(v\d+) = getFFTSMsg\(FFTS_MODE_VAL, (v\d+)\);\n  ffts_cross_core_sync\(PIPE_MTE3, \1\);",
            vector,
        )
        assert arrive and re.search(r"const int64_t " + arrive[2] + r" = 12;", vector)
        release = re.search(r"(v\d+) = \(int32_t\) (v\d+);\n  wait_flag_dev\(\1\);", vector[arrive.end() :])
        assert release and re.search(r"const int64_t " + release[2] + r" = 13;", vector)
        wait_position = arrive.end() + release.start()
        assert wait_position < vector.index("TNOTIFY(")
        assert "TSTORE(" in vector[: arrive.start()]
        independent_work = vector[arrive.end() : wait_position]
        assert "TSTORE(" in independent_work and "signal__ssa" not in independent_work
        result[f"C{case:02}"] = {
            "passed": True,
            "report": str(directory / "result.json"),
            "candidate_cpp": str(candidate),
            "cpp_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "cpp_MIX_barriers": 1,
            "AIV_arrive_event": 12,
            "AIV_release_wait": 13,
            "no_signal_view_access_between_arrive_wait": True,
            "init_clear_before_arrive": True,
            "all_TNOTIFY_after_wait": True,
            "CPU_policies": report["cpu_policies"],
        }
    (here / "init_arrive_overlap_CPU_gate.json").write_text(
        json.dumps(
            {"passed": True, "cases": result, "scope": "CPU编译及事件顺序检查；尚待NPU正确性与性能，不是验收"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("all 8 CPU codegen gates passed")


if __name__ == "__main__":
    main()
