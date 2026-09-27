"""Map existing Native device events and V10 tasks by source call order.

No device execution. Native durations and PTO block means remain separate.
"""

import json
import re
from pathlib import Path

from summarize_v10 import process_events

ROOT = Path(__file__).resolve().parent
FP_MM = "aclnnMatmulWeightNz_MatMulCommon_MatMulV2"
INT_MM = "aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3"
ND_MM = "aclnnMatmul_MatMulCommon_MatMulV2"
BATCH_MM = "aclnnTransposeBatchMatMul_TransposeBatchMatMul_TransposeBatchMatMul"


def main():
    source = json.loads((ROOT / "v10_incore.json").read_text())
    rows = []
    for case in source["cases"]:
        events = sorted(process_events(Path(case["native_trace"]), "Ascend Hardware"),
                        key=lambda event: float(event["ts"]))
        by_name = {}
        for event in events:
            by_name.setdefault(event["name"], []).append(event)
        expected = {FP_MM: 3, INT_MM: 3, ND_MM: 2, BATCH_MM: 1, "Compressor": 2,
                    "CompressorMetadata": 2, "InplacePartialRotaryMul": 4,
                    "RmsNorm": 2, "RmsNormDynamicQuant": 1, "triton_rms_kernel_0": 1,
                    "HcPre": 1, "HcPost": 1}
        for name, count in expected.items():
            if len(by_name.get(name, [])) != count:
                raise ValueError(f"Ambiguous Native call order: {case['history']}/{case['batch']} {name}")
        # Call order is checked against _mla_prolog_multistream,
        # cv_indexer_select_qli, _forward_decode, and _forward_o_proj.
        specs = [
            ("HC_pre", [("HcPre", 0)], r"(?:hc_widen|hc_pre_linear|hc_pre_linear_reduce|hc_pre_rms|comb_sinkhorn|split_pre_post)_spmd"),
            ("input_RMSNorm", [("RmsNorm", 0)], r"mix_x_rms_norm_spmd"),
            ("Q_A", [(FP_MM, 0)], r"qr_proj_(?:seed|matmul_spmd)"),
            ("Q_A_RMS_quant", [("RmsNormDynamicQuant", 0)], r"qr_rms_norm_quant_spmd"),
            ("KV_projection", [(FP_MM, 1)], r"kv_proj_(?:seed|matmul_spmd|native_240_spmd)"),
            ("KV_RMS_RoPE", [("RmsNorm", 1), ("InplacePartialRotaryMul", 0)], r"kv_rms_norm_rope(?:_spmd)?"),
            ("Q_B_dequant_RMS_RoPE", [(INT_MM, 0), ("triton_rms_kernel_0", 0),
                                      ("InplacePartialRotaryMul", 1)],
             r"qproj_(?:matmul|dequant_rms_nope_rope)_spmd"),
            ("indexer_compressor", [("Compressor", 0)],
             r"(?:kv_score_proj_0|scatter_softmax_pool_0|compress_state_commit_0|rmsnorm_rope)_spmd"),
            ("attention_compressor", [("Compressor", 1)],
             r"(?:kv_score_proj|scatter_softmax_pool|compress_state_commit|rmsnorm_rope_cache_write)_spmd"),
            ("indexer_Q_projection_RoPE", [(INT_MM, 1), ("InplacePartialRotaryMul", 2)],
             r"idx_qr_(?:proj_matmul|dequant_rope)_spmd"),
            ("indexer_K_hadamard", [(ND_MM, 0)], r"kv_hadamard"),
            ("indexer_Q_hadamard", [(ND_MM, 1)], r"qr_hadamard_(?:matmul|quant)_spmd"),
            ("indexer_head_weight_projection", [(FP_MM, 2)], r"weights_proj(?:_reduce)?_spmd"),
            ("O_A", [(BATCH_MM, 0)], r"proj_a_mm(?:_\d+)?_spmd"),
            ("O_B", [(INT_MM, 2)], r"(?:(?:_proj_b_mm_nz_kernel(?:__\d+)?|proj_b_mm(?:_\d+)?|proj_b_act(?:_\d+)?)_spmd|quant(?:_\d+_spmd)?)"),
            ("HC_post", [("HcPost", 0)], r"hc_post_spmd"),
        ]
        groups = []
        for label, refs, pattern in specs:
            native = []
            for name, occurrence in refs:
                event = by_name[name][occurrence]
                native.append({"name": name, "occurrence": occurrence, "duration_us": event["dur"],
                               "timestamp": event["ts"], "stream": event["tid"],
                               "task_id": event.get("args", {}).get("Task Id")})
            matched = {}
            for window in case["pto_windows"]:
                tasks = {name: stats for name, stats in window["tasks"].items() if re.fullmatch(pattern, name)}
                if not tasks:
                    raise ValueError(f"Missing PTO mapping: {case['history']}/{case['batch']} {label}")
                for name, stats in tasks.items():
                    matched.setdefault(name, {"blocks": stats["blocks"], "window_block_mean_us": []})["window_block_mean_us"].append(stats["mean_us"])
            groups.append({"stage": label, "native_events": native, "pto_tasks": matched})
        rows.append({"history": case["history"], "batch": case["batch"], "operator_revision": case["operator_revision"],
                     "native_trace": case["native_trace"],
                     "pto_trace_paths": [window["path"] for window in case["pto_windows"]], "groups": groups})
    result = {
        "scope": "Source-based operation correspondence, not identical fusion or arithmetic boundaries; no summed durations or speedup claims",
        "native_order_evidence": "dsa_v1.py: _mla_prolog_multistream, cv_indexer_select_qli, _forward_decode, _forward_o_proj",
        "pto_order_evidence": "decode_csa.py calls attention compressor before indexer compressor; suffix _0 is the latter in these V10 captures",
        "limits": ["Compressor PTO cache-write task includes output scatter, Native Compressor does not include the separate scatter kernel",
                   "Q dequantization and BF16 rounding placement differ",
                   "Native O_B column excludes its separate dynamic quant; PTO group includes quant and dequant",
                   "PTO QR hadamard quant group includes scale/quant, Native entry is matrix multiply only",
                   "Metadata, cache scatter, layout/seed/adapter tasks remain separate; no full critical-path accounting"],
        "cases": rows,
    }
    (ROOT / "v10_other_incore.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Mapped {len(rows)} cases, {len(specs)} operation groups each; no device execution")


if __name__ == "__main__":
    main()
