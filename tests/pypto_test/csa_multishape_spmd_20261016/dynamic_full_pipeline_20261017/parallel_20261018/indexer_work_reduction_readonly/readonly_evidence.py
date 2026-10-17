#!/usr/bin/env python3
"""有限源码直接对照与工作量算术；不加载 PyPTO、不编译、不执行设备。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANONICAL = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1")
PAR = CANONICAL / "tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018"
SOURCE = CANONICAL / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
NAMES = ["indexer_score_stages.py", "indexer_topk_stages.py", "indexer_group_stages.py",
         "indexer_publish_stages.py", "policy.py"]

q, heads, dim, candidates, ws_rows = 6, 64, 128, 1024, 16
qk = q * heads * dim * candidates
ws = ws_rows * q * heads * candidates
useful_ws = q * heads * candidates
steps = 22
one_step_score = q * candidates * 4
one_step_qk = q * heads * candidates * 2
run_size, runs = 512, 4
result = dict(
    decision="本轮不提出实现候选；未找到同时有新证据、明确大额省工与逐bit依据的路径",
    source_comparison=[dict(file=n, current=str(SOURCE/n), reference=str(PAR/"baseline_source"/n),
                            current_bytes=(SOURCE/n).stat().st_size,
                            same_text_as_593=(SOURCE/n).read_text()==(PAR/"baseline_source"/n).read_text())
                       for n in NAMES],
    representative=dict(history=131072, batch=16, query_tokens=6,
                        query_group=q, qk_panel=64, score_tile=candidates,
                        plan_steps_per_core=steps, cores=24,
                        plan_evidence=str(PAR/"indexer_query_group_audit/README.md")),
    work=dict(qk_int8_mac_per_step=qk, ws_fp16_mac_per_step=ws,
              useful_diagonal_ws_products_per_step=useful_ws,
              ws_zero_or_padding_products_per_step=ws-useful_ws,
              ws_fraction_of_combined_mac_count=ws/(qk+ws),
              zero_ws_fraction_of_combined_mac_count=(ws-useful_ws)/(qk+ws),
              qk_int8_mac_per_core=qk*steps, ws_fp16_mac_per_core=ws*steps,
              qk_calls_per_core=16*steps, ws_calls_per_core=16*steps,
              current_score_transfer_bytes_per_step=one_step_score,
              raw_qk_transfer_bytes_per_step=one_step_qk,
              raw_qk_transfer_ratio=one_step_qk/one_step_score,
              extra_qk_handoff_write_plus_read_bytes_per_step=2*(one_step_qk-one_step_score),
              raw_qk_bytes_per_aiv=one_step_qk//2,
              extra_key_staging_bytes_long_b16=steps*24*1024*128*2),
    segment_merge=dict(candidates=2048, merge_passes=3,
                       full_merge_output_pairs=2048*3,
                       exhausted_min_logical_output_pairs=run_size,
                       interleaved_scores_output_pairs=run_size+(runs-1)*(run_size-1),
                       best_reduction_of_three_merge_output_pairs=(2048-512)/(3*2048),
                       guaranteed_reduction_of_sort32_input=0,
                       note="仅理想逐项耗尽的逻辑计数；硬件写出粒度/耗时另论，不是一个已提实现"),
    evidence_limits=["INT8与FP16 MAC不能按次数直接换成时延，原QK/WS已流水重叠",
                     "593旧插桩不是当前9d的性能；前驱compact顺序已改变",
                     "当前helper文本相同仅支持算法工作量沿用，不支持旧等待时间直接相减",
                     "短尾、WS搬迁、GM暂存、G/leaf/panel、根前缀认证均复用已有排重记录"],
    compiler_execution=False, device_execution=False,
)
assert all(x["same_text_as_593"] for x in result["source_comparison"])
(ROOT/"evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(dict(decision=result["decision"], work=result["work"],
                      segment_merge=result["segment_merge"]), ensure_ascii=False, indent=2))
