# SPDX-License-Identifier: Apache-2.0
"""纯CPU核对首token诊断的可比性，再比较逐层误差和q08 logits。"""
import argparse
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def diff(a, b):
    delta = a.float()-b.float()
    return {"shape": list(a.shape), "bit_equal": torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)),
            "max_abs": float(delta.abs().max()), "rmse": float(delta.square().mean().sqrt()),
            "nonfinite": int((~torch.isfinite(a)).sum()+(~torch.isfinite(b)).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("pto", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = read(HERE / "B16_COMPARISON.json")
    historical = {(r["rank"], r["key"]): r for r in old["rows"]}
    report = {"status": "ANALYZING", "source": {"native": str(args.native), "pto": str(args.pto)},
              "reference_checks": [], "layers": []}
    data = {}
    for side, root in (("native", args.native), ("pto", args.pto)):
        same = 0
        for rank in range(16):
            result = read(root / f"rank{rank}.mixed.json")
            capture = read(root / f"precision/rank{rank}/summary.json")
            first = capture["first_forward"]
            assert capture["logits_captured"] and first["tokens"] == first["actual_tokens"] == first["requests"] == 16
            assert first["query_start_loc"] == list(range(17)) and first["graph_mode"] == "NONE"
            for request in result["requests"]:
                expected = historical[(rank, request["key"])]["native_tokens" if side == "native" else "hca_tokens"][0]
                if request["output_token_ids"] == [expected]:
                    same += 1
            if rank == 0:
                data[side] = capture
        report["reference_checks"].append({"side": side, "matching_first_tokens": same, "requests": 256,
                                           "passed": same == 256})
    for field in ("input_ids", "positions", "query_start_loc", "seq_lens"):
        assert data["native"]["first_forward"][field] == data["pto"]["first_forward"][field], field
    report["same_model_input"] = True
    report["same_input_layer_checks"] = data["pto"]["layers"]
    assert len(data["native"]["layers"]) == len(data["pto"]["layers"]) == 20
    for row in data["pto"]["layers"]:
        assert row["restored_exact"] and row["input_unchanged"] and row["output_unchanged"]
        index = row["layer"]
        native = load(args.native / f"precision/rank0/layer_{index:02d}.pt")
        pto = load(args.pto / f"precision/rank0/layer_{index:02d}.pt")
        report["layers"].append({"layer": index, "incoming_trajectory": diff(pto["input"], native["input"]),
                                  "outgoing_trajectory": diff(pto["output"], native["output"]),
                                  "same_input_intrinsic": diff(pto["output"], pto["native_same_input_output"]),
                                  "q08_intrinsic": diff(pto["output"][8], pto["native_same_input_output"][8])})
    native_logits = load(args.native / "precision/rank0/logits.pt")
    pto_logits = load(args.pto / "precision/rank0/logits.pt")
    report["logits"] = diff(pto_logits, native_logits)
    report["final_hidden"] = diff(load(args.pto / "precision/rank0/final_hidden.pt"),
                                    load(args.native / "precision/rank0/final_hidden.pt"))
    report["q08"] = {"row": 8, "input_position": data["pto"]["first_forward"]["positions"][8],
                       "native_top10_ids": data["native"]["top10_ids"][8],
                       "native_top10_scores": data["native"]["top10_scores"][8],
                       "pto_top10_ids": data["pto"]["top10_ids"][8],
                       "pto_top10_scores": data["pto"]["top10_scores"][8]}
    # topk对并列项的顺序不等于argmax采样规则，必须包含实际选出的两个token。
    expected_q08 = historical[(0, "h131072_q08")]
    candidate_ids = sorted({expected_q08["native_tokens"][0], expected_q08["hca_tokens"][0]})
    report["q08"]["native_argmax"] = int(native_logits[8].argmax())
    report["q08"]["pto_argmax"] = int(pto_logits[8].argmax())
    report["q08"]["candidate_scores"] = [{"token": i, "native": float(native_logits[8,i]),
                                            "pto": float(pto_logits[8,i])} for i in candidate_ids]
    report["status"] = "COMPARED" if all(r["passed"] for r in report["reference_checks"]) else "DIAGNOSTIC_CHANGED_REFERENCE"
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ("layers","same_input_layer_checks")},ensure_ascii=False))
    for row in report["layers"]:
        print(row["layer"], "input", row["incoming_trajectory"]["rmse"],
              "output", row["outgoing_trajectory"]["rmse"], "intrinsic", row["same_input_intrinsic"]["rmse"])


if __name__ == "__main__":
    main()
