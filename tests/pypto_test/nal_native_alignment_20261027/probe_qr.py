# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind the private package before DSL imports.
"""复用已核验完整SPMD的QA，单因素测量QA舍入对QR的影响。"""

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
from qr_kernels import qr_arm

from vllm_ascend.utils import enable_custom_op

out = Path(sys.argv[1])
record = {"complete": False, "scope": "isolated production QR helper; same saved QA/gamma", "arms": {}}
start = time.monotonic()


def metric(actual, expected):
    actual, expected = actual.cpu().contiguous(), expected.cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    kind = torch.int8 if actual.element_size() == 1 else torch.int32
    diff = actual.view(kind) != expected.view(kind)
    return {
        "bit_equal": not bool(diff.any()),
        "different_elements": int(diff.sum()),
        "max_abs": float((actual.double() - expected.double()).abs().max()),
    }


try:
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true" and enable_custom_op()
    reference = Path(os.environ["QR_PROBE_REFERENCE"])
    record["reference"] = str(reference)
    data = torch.load(reference, map_location="cpu", weights_only=True)
    qa = data["qa_accumulator"].to("npu")
    gamma = data["qr_gamma"].to("npu")
    native_qa = data["q_a_projection_bf16_native"].to("npu")
    tokens = native_qa.shape[0]
    native_qr, native_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(native_qa, gamma, epsilon=1e-6)
    assert metric(native_qr, data["q_r_quantized_native"])["bit_equal"]
    assert metric(native_scale.reshape(tokens, 1), data["qr_native_scale"])["bit_equal"]
    pypto.torch.init()
    qr = torch.empty((tokens, 1024), dtype=torch.int8, device="npu")
    scale = torch.empty((tokens, 1), dtype=torch.float32, device="npu")
    padded = torch.empty(((tokens + 127) // 128 * 128, 1024), dtype=torch.int8, device="npu")
    scale_pad = torch.empty(((tokens + 7) // 8 * 8, 1), dtype=torch.float32, device="npu")
    for name, source in (("unrounded_qa", qa), ("bf16_qa", qa.bfloat16().float())):
        qr_arm(source, gamma, qr, scale, padded, scale_pad)
        torch.npu.synchronize()
        eager = (qr.cpu().clone(), scale.cpu().clone())
        graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, stream=stream):
            qr_arm(source, gamma, qr, scale, padded, scale_pad)
        qr.fill_(-37)
        scale.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        row = {
            "qr_native": metric(qr, native_qr),
            "scale_native": metric(scale, native_scale.reshape(tokens, 1)),
            "qr_graph_eager": metric(qr, eager[0]),
            "scale_graph_eager": metric(scale, eager[1]),
        }
        assert row["qr_graph_eager"]["bit_equal"] and row["scale_graph_eager"]["bit_equal"]
        if name == "unrounded_qa":
            row["production_qr"] = metric(qr, data["q_r_quantized_actual"])
            assert row["production_qr"]["bit_equal"]
        record["arms"][name] = row
    record["complete"] = True
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
