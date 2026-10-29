# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind private DSL separately from the Native plugin.
"""在实际Native QLI输入上构造精确同分，验证两个原始位置的归并优先级。"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import pypto.torch
import torch
import torch_npu
from bind_package import bind

bind()
from topktie_kernels import topktie_arm

from vllm_ascend.utils import enable_custom_op

out = Path(sys.argv[1])
record = {"complete": False, "pass": False, "scope": "controlled exact tie, not whole-model or full score equality"}
start = time.monotonic()
try:
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true" and enable_custom_op()
    reference = Path(os.environ["TOPK_PROBE_REFERENCE"])
    data = torch.load(reference, map_location="cpu", weights_only=True)
    qli = data["native_qli_inputs"]
    row, candidate_ids = 183, [820, 1967]
    ends = qli["actual_seq_lengths_query"]
    request = int(torch.searchsorted(ends, torch.tensor(row), right=True))
    begin = int(ends[request - 1]) if request else 0
    # Native passes uncompressed key lengths and masks each query before /4.
    visible = (int(qli["actual_seq_lengths_key"][request]) - int(ends[request]) + row + 1) // 4
    record["native_lengths"] = {"q_end": int(ends[request]), "k_end": int(qli["actual_seq_lengths_key"][request])}
    assert visible == 2049, "isolated sort geometry must match the observed row"
    assert begin <= row < int(ends[request]) and qli["query"].shape[0] == 240
    assert all(name in qli for name in ("block_table", "key", "key_dequant_scale"))
    locations = [(int(qli["block_table"][request, index // 32]), index % 32) for index in candidate_ids]
    assert locations[0] != locations[1] and all(0 <= page < qli["key"].shape[0] for page, _ in locations)
    record.update(reference=str(reference), row=row, request=request, candidates=candidate_ids, locations=locations)
    actual_scores = data["observed_topk_scores"][row]
    actual_topk = data["qli_topk_actual"][row]
    ranks = [int((actual_topk == index).nonzero().item()) for index in candidate_ids]
    record["original_scores"] = {
        "ranks": ranks,
        "values": actual_scores[ranks].tolist(),
        "bits": actual_scores.contiguous().view(torch.int32)[ranks].tolist(),
    }
    assert record["original_scores"]["bits"][0] == record["original_scores"]["bits"][1]
    controlled = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in qli.items()}
    for name in ("query", "key", "weights"):
        controlled[name].zero_()
    controlled["query"][row, 0].fill_(1)
    controlled["weights"][row, 0] = 1
    controlled["query_dequant_scale"].fill_(1)
    controlled["key_dequant_scale"].fill_(1)
    for page, slot in locations:
        controlled["key"][page, slot, 0].fill_(1)
    device_inputs = {
        name: value.to("npu") if isinstance(value, torch.Tensor) else value for name, value in controlled.items()
    }
    operator = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer
    first = operator(**device_inputs)[0].cpu()
    second = operator(**device_inputs)[0].cpu()
    record["native_output_shape"] = list(first.shape)
    assert first.shape == second.shape == (240, 1, 512)
    first, second = first.reshape(240, 512), second.reshape(240, 512)
    assert torch.equal(first, second), "Native controlled tie repeats differ"
    record["native_order"] = first[row, :2].tolist()
    assert set(record["native_order"]) == set(candidate_ids), "controlled candidates not top two"
    # The only nonzero head has QK=128; FIX dequant 2^-10 and all scales1
    # make both scores exactly0.125 in FP16 and FP32, without reduction ambiguity.
    pypto.torch.init()
    score_cpu = torch.zeros((2, 8192), dtype=torch.float32)
    score_cpu[0, candidate_ids] = 0.125
    score_cpu[1, : visible - 1152] = score_cpu[0, 1152:visible]
    scores = score_cpu.to("npu")
    record["arms"] = {}
    for mode, label in enumerate(("original_half_tree", "native_2048_tree", "segmented_fix")):
        pairs = torch.full((2, 1024), -37, dtype=torch.float32, device="npu")
        indices = torch.full((1, 512), -37, dtype=torch.int32, device="npu")
        values = torch.full((1, 512), -37, dtype=torch.float32, device="npu")

        def run(mode=mode, pairs=pairs, indices=indices, values=values):
            topktie_arm(scores, pairs, indices, values, VISIBLE=visible, MODE=mode)

        run()
        eager_i, eager_v = indices.cpu().clone(), values.cpu().clone()
        graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, stream=stream):
            run()
        indices.fill_(-37)
        values.fill_(float("nan"))
        graph.replay()
        full_i, full_v = indices.cpu(), values.cpu()
        assert torch.equal(full_i, eager_i) and torch.equal(full_v.view(torch.int32), eager_v.view(torch.int32))
        assert torch.equal(scores.cpu(), score_cpu)
        order = full_i[0, :2].tolist()
        assert set(order) == set(candidate_ids) and full_v[0, :2].tolist() == [0.125, 0.125]
        record["arms"][label] = {"order": order, "scores": full_v[0, :2].tolist(), "graph_eager_equal": True}
    assert record["arms"]["original_half_tree"]["order"] == [1967, 820]
    assert record["arms"]["native_2048_tree"]["order"] == record["native_order"] == [820, 1967]
    assert record["arms"]["segmented_fix"]["order"] == record["native_order"]
    # All valid candidates tie: also exercise the top512 selection cutoff and
    # Native's later-2048-block priority, not merely the first two positions.
    controlled["key"].fill_(1)
    device_inputs["key"] = controlled["key"].to("npu")
    native_all_ties = operator(**device_inputs)[0].cpu().reshape(240, 512)[row]
    assert torch.equal(native_all_ties, operator(**device_inputs)[0].cpu().reshape(240, 512)[row])
    score_cpu.zero_()
    score_cpu[0, :visible] = 0.125
    score_cpu[1, : visible - 1152] = 0.125
    scores.copy_(score_cpu)
    topktie_arm(scores, pairs, indices, values, VISIBLE=visible, MODE=2)
    torch.npu.synchronize()
    all_tie_eager = indices.cpu().clone()
    graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(graph, stream=stream):
        topktie_arm(scores, pairs, indices, values, VISIBLE=visible, MODE=2)
    indices.fill_(-37)
    graph.replay()
    all_tie_full = indices.cpu()
    assert torch.equal(all_tie_full, all_tie_eager)
    record["all_ties"] = {
        "different_elements": int((all_tie_full[0] != native_all_ties).sum()),
        "native_first8": native_all_ties[:8].tolist(),
        "actual_first8": all_tie_full[0, :8].tolist(),
        "graph_eager_equal": True,
    }
    assert record["all_ties"]["different_elements"] == 0
    record["pass"] = True
except Exception:
    record["error"] = traceback.format_exc()
finally:
    record.update(complete=True, elapsed_s=time.monotonic() - start)
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
