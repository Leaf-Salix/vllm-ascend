# SPDX-License-Identifier: Apache-2.0
"""读取已有QKV诊断快照，验证三组输入/Native参考相同并生成汇总；只用CPU。"""
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def bit_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def main():
    names = ("qkv_base_v2", "qkv_qa_bf16_v2", "qkv_all_bf16_v2")
    values = {name: torch.load(HERE / "runs" / name / "values.pt", map_location="cpu", weights_only=True)
              for name in names}
    base = values[names[0]]
    rows = []
    for name in names:
        actual = values[name]
        report = json.loads((HERE / "runs" / name / "result.json").read_text())
        assert report["status"] == "COMPARED"
        input_equal = bit_equal(actual["normalized"], base["normalized"])
        native_equal = {key: bit_equal(value, base["native"][key]) for key,value in actual["native"].items()}
        assert input_equal and all(native_equal.values())
        rows.append({"name": name, "input_equal": input_equal, "native_equal": native_equal,
                     "comparisons": report["comparisons"], "q08": report["q08"],
                     "bit_equal": {key: bit_equal(value, actual["native"][key]) for key,value in actual["pto"].items()},
                     "q08_bit_equal": {key: bit_equal(value[8:9], actual["native"][key][8:9])
                                       for key,value in actual["pto"].items()},
                     "qa_dtype": report["native_qa_dtype"]})
    # 追加采样复核已恢复边界的同一结果；采样不能改变Native/QKV输出。
    stage = torch.load(HERE / "runs/qkv_all_bf16_stage_v3/values.pt", map_location="cpu", weights_only=True)
    identical = {key: bit_equal(value, values[names[-1]]["pto"][key]) for key,value in stage["pto"].items()}
    assert all(identical.values())
    q = values[names[-1]]["pto"]["q"].reshape(16,64,512)
    reference = values[names[-1]]["native"]["q"].reshape_as(q)
    head_counts = (q != reference).sum(-1)
    report = {"status": "INPUT_AND_REFERENCE_VALIDATED", "rows": rows,
              "stage_probe_output_equal": identical,
              "remaining_q_heads": [{"row": row, "head": head, "different_elements": int(head_counts[row,head])}
                                    for row,head in head_counts.nonzero().tolist()],
              "acceptance": "仅数值来源诊断；未作答案质量结论、未采用候选、不测性能"}
    (HERE / "QKV_BOUNDARY_COMPARISON.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({"status": report["status"], "remaining_q_heads": report["remaining_q_heads"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
