# SPDX-License-Identifier: Apache-2.0
"""审查实际PTOAS代码中的PV加载、softmax及MTE2事件边界。"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def audit(directory):
    result = json.loads((directory / "result.json").read_text())
    assert result["passed"] and result["stage"] == "cpu_backend_compiled"
    (path,) = directory.glob("build_output/*/ptoas/dynamic_csa_full.cpp")
    lines = path.read_text().splitlines()

    def next_op(anchor, operation):
        start = next(i for i, line in enumerate(lines) if anchor in line)
        return next(i for i in range(start, len(lines)) if operation in lines[i])

    score = next_op("%sm_scores_inline", "TLOAD(")
    bias = next_op("%sm_bias_inline", "TLOAD(")
    pv = next_op("%pf_pv_left_inline", "TLOAD(")
    softmax = next(i for i in range(pv, len(lines)) if "TMULS(" in lines[i])
    assert score < bias < pv < softmax
    barriers = [i + 1 for i in range(pv + 1, softmax) if "pipe_barrier(PIPE_ALL)" in lines[i]]
    waits = []
    for i in range(pv + 1, softmax):
        match = re.search(r"wait_flag\(PIPE_MTE2, PIPE_V, (EVENT_ID\d+)\)", lines[i])
        if match:
            event = match.group(1)
            producer = max(j for j in range(i) if f"set_flag(PIPE_MTE2, PIPE_V, {event})" in lines[j])
            waits.append(
                {"event": event, "wait_line": i + 1, "set_line": producer + 1, "set_precedes_pv_load": producer < pv}
            )
    # 这里只审查两块同时有效的稳态路径；无效块和排空依赖由事件模型另验。
    permits = not barriers and all(w["set_precedes_pv_load"] for w in waits)
    return {
        "cpu_codegen_passed": True,
        "compile_result": str(directory / "result.json"),
        "generated_cpp": str(path),
        "generated_cpp_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "score_load_line": score + 1,
        "bias_load_line": bias + 1,
        "pv_load_line": pv + 1,
        "first_softmax_arithmetic_line": softmax + 1,
        "intervening_full_barriers": barriers,
        "mte2_vector_waits": waits,
        "emitted_code_permits_intended_overlap": permits,
        "decision": "允许完整设备验证" if permits else "暂不上卡，所需重叠仍被屏障截断",
        "source_excerpt": [f"{i + 1}: {lines[i]}" for i in range(pv, softmax + 1)],
        "limitations": [
            "生成代码允许重叠不代表真机已发生重叠或完整链路已有收益",
            "本报告不是精度、服务图重放或八档验收结果",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    report = audit(here / "runs" / args.attempt)
    target = here / (args.attempt + "_audit.json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_excerpt"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
