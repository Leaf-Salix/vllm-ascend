# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind independent DSL after choosing the frozen Native plugin.
"""同一次真实Sparse消费者输入：先复现Native，再复核旧精度K512数学。"""

import hashlib
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
from attention_native_micro import native_attention_arm
from attention_reference import SOFTMAX_SCALE, T_PAD, attention_reference_arm

from vllm_ascend.utils import enable_custom_op

out = Path(sys.argv[1])
variant = os.environ.get("ATTENTION_PROBE_VARIANT", "reference")
assert variant in ("reference", "native-spmd")
record = {
    "complete": False,
    "pass": False,
    "scope": "saved actual Sparse inputs, 8K/B4/T24 only",
    "variant": variant,
    "precision": {},
}
payload = {}
start = time.monotonic()


class FirstDifference(Exception):
    pass


def metric(actual, expected):
    actual, expected = actual.cpu().contiguous(), expected.cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if actual.is_floating_point():
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    dtype = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[actual.element_size()]
    diff = actual.view(dtype) != expected.view(dtype)
    return {
        "bit_equal": not bool(diff.any()),
        "different_elements": int(diff.sum()),
        "max_abs": float((actual.double() - expected.double()).abs().max()),
        "first_coordinates": diff.nonzero()[:8].tolist(),
    }


def check(name, actual, expected):
    row = metric(actual, expected)
    record["precision"][name] = row
    if not row["bit_equal"]:
        record["first_difference"] = name
        raise FirstDifference(name)


try:
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true" and enable_custom_op()
    reference = Path(os.environ["ATTENTION_PROBE_REFERENCE"])
    fixture = Path(os.environ["ATTENTION_PROBE_FIXTURE"])
    saved = torch.load(reference, map_location="cpu", weights_only=True)
    initial = torch.load(fixture, map_location="cpu", weights_only=True)["initial"]
    observer = json.loads(reference.with_name("result.json").read_text())
    assert observer["complete"] and observer["first_difference"] == "attention_heads_after_inverse_rope"
    for gate in ("observation_gate", "native_qli_hook_control", "input_equivalence"):
        assert all(row["bit_equal"] for row in observer[gate].values())
    assert observer["precision"]["qli_topk"]["bit_equal"]
    stages = saved["native_output_stages"]
    captured = stages["sparse_inputs"]
    check("native_sparse_vs_decode_return", stages["sparse_output"].reshape(24, 64, 512), stages["attention"])
    input_owners, inputs = {}, {}
    for name, value in captured.items():
        if isinstance(value, torch.Tensor):
            owner = torch.full((value.numel() + 256,), -37, dtype=value.dtype, device="npu")
            target = owner[128:-128].reshape(value.shape)
            target.copy_(value.to("npu"))
            input_owners[name], inputs[name] = owner, target
        else:
            inputs[name] = value
    assert inputs["layout_q"] == "TND" and inputs["layout_kv"] == "PA_ND"
    assert inputs["cmp_ratio"] == 4 and inputs["softmax_scale"] == SOFTMAX_SCALE
    assert inputs["ori_win_left"] == 127 and inputs["ori_win_right"] == 0
    assert inputs["ori_mask_mode"] == 4 and inputs["cmp_mask_mode"] == 3
    check("actual_query_bounds_vs_fixture", inputs["cu_seqlens_q"], initial["readonly_swa.query_start_loc"])
    check("actual_seq_lens_vs_fixture", inputs["seqused_kv"], initial["readonly_swa.seq_lens"])
    check("actual_original_table_vs_fixture", inputs["ori_block_table"], initial["readonly_swa.block_table"])
    check("actual_compressed_table_vs_fixture", inputs["cmp_block_table"], initial["readonly_compressed.block_table"])
    check("actual_native_query_vs_prefix", inputs["query"], saved["q_norm_rope_native"].reshape(24, 64, 512))
    check(
        "actual_native_topk_vs_qli",
        inputs["cmp_sparse_indices"].reshape(24, 512),
        saved["native_full_topk"].reshape(24, 512),
    )
    before = {name: value.cpu().clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    query = inputs.pop("query")
    native_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(query, **inputs)[0]
    torch.npu.synchronize()
    check("native_sparse_replay", native_output, stages["sparse_output"])
    cos, sin = (initial["readonly_rope." + name].to("npu") for name in ("cos", "sin"))
    rotated = native_output.clone().reshape(24, 64, 512)
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        rotated.unsqueeze(1), cos, -sin, rotary_mode="interleave", partial_slice=[448, 512]
    )
    torch.npu.synchronize()
    check("native_inverse_replay", rotated, stages["heads"])
    bounds = initial["readonly_swa.query_start_loc"]
    owners_cpu = torch.searchsorted(bounds[1:], torch.arange(24, dtype=bounds.dtype), right=True).long()
    lengths = captured["seqused_kv"]
    expected_positions = (
        lengths[owners_cpu] - (bounds[owners_cpu + 1] - bounds[owners_cpu]) + torch.arange(24) - bounds[owners_cpu]
    ).long()
    check("actual_geometry_vs_positions", initial["readonly_positions"], expected_positions)
    requests = owners_cpu.int().to("npu")
    positions = initial["readonly_positions"].reshape(24, 1).to("npu")
    arguments = (
        query,
        inputs["ori_kv"].reshape(-1, 32, 1, 512),
        inputs["ori_block_table"],
        inputs["cmp_kv"].reshape(-1, 32, 1, 512),
        inputs["cmp_block_table"],
        inputs["cmp_sparse_indices"].reshape(24, 512),
        positions,
        inputs["seqused_kv"],
        requests,
        inputs["sinks"].reshape(64),
        cos.reshape(24, 64),
        sin.reshape(24, 64),
    )
    owners = [torch.full((8 * T_PAD + 16, 4096), -37, dtype=torch.bfloat16, device="npu") for _ in range(2)]
    pre, post = (owner[8:-8] for owner in owners)

    def heads(value):
        return value.reshape(8, T_PAD, 8, 512)[:, :24].permute(1, 0, 2, 3).reshape(24, 64, 512)

    if variant == "native-spmd":
        assert torch.equal(bounds[1:] - bounds[:-1], torch.full((4,), 6, dtype=bounds.dtype))
        assert int(initial["readonly_positions"].min()) == 8192
        for token in range(24):
            chosen = captured["cmp_sparse_indices"].reshape(24, 512)[token]
            limit = (int(initial["readonly_positions"][token]) + 1) // 4
            assert bool(((chosen >= 0) & (chosen < limit)).all())
        assert bool((captured["ori_block_table"] >= 0).all() and (captured["cmp_block_table"] >= 0).all())
        mask = torch.zeros((24, 16), dtype=torch.int32, device="npu")
        mask[:, :5] = 1
        bias = torch.zeros((24, 640), dtype=torch.float32, device="npu")
        bounds_npu = bounds.to("npu")
        side_before = [value.cpu().clone() for value in (mask, bias, bounds_npu)]

    def run():
        if variant == "reference":
            attention_reference_arm(*arguments, post, pre)
        else:
            native_attention_arm(*arguments[:8], *arguments[9:], post, mask, bias, bounds_npu, pre)

    pypto.torch.init()
    run()
    torch.npu.synchronize()
    eager = [heads(value).cpu().clone() for value in (pre, post)]
    graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(graph, stream=stream):
        run()
    for value in (pre, post):
        value.fill_(float("nan"))
    graph.replay()
    torch.npu.synchronize()
    actual = [heads(value).cpu() for value in (pre, post)]
    # Preserve the first failed execution gate; do not lose its inputs/outputs
    # when the graph/eager assertion stops the later Native comparison.
    payload.update(
        eager_preinverse=eager[0],
        eager_postinverse=eager[1],
        preinverse=actual[0],
        postinverse=actual[1],
        native_pre=stages["attention"],
        native_post=stages["heads"],
    )
    record["graph_eager"] = [metric(value, expected) for value, expected in zip(actual, eager)]
    head_differences = (actual[0].contiguous().view(torch.int16) != eager[0].contiguous().view(torch.int16)).sum(-1)
    record["graph_eager_preinverse_heads"] = [
        {"query": int(query), "head": int(head), "different_elements": int(head_differences[query, head])}
        for query, head in head_differences.nonzero().tolist()
    ]
    record["guard_ok"] = all(
        bool((owner[:8].cpu() == -37).all() and (owner[-8:].cpu() == -37).all()) for owner in owners
    )
    record["input_readonly"] = {
        name: metric(value, before[name])
        for name, value in {"query": query, **inputs}.items()
        if isinstance(value, torch.Tensor)
    }
    assert record["guard_ok"] and all(row["bit_equal"] for row in record["graph_eager"])
    record["input_guard_ok"] = all(
        bool((owner[:128].cpu() == -37).all() and (owner[-128:].cpu() == -37).all()) for owner in input_owners.values()
    )
    assert record["input_guard_ok"] and all(row["bit_equal"] for row in record["input_readonly"].values())
    if variant == "native-spmd":
        record["plan_readonly"] = [metric(value, old) for value, old in zip((mask, bias, bounds_npu), side_before)]
        assert all(row["bit_equal"] for row in record["plan_readonly"])
    record["reference"] = str(reference)
    record["fixture"] = str(fixture)
    for name, path in (("reference", reference), ("fixture", fixture)):
        with path.open("rb") as stream:
            record[name + "_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    record["reference_source_sha256"] = hashlib.sha256(
        Path(__file__).with_name("attention_reference.py").read_bytes()
    ).hexdigest()
    if variant == "native-spmd":
        record["native_helper_sha256"] = hashlib.sha256(
            (
                Path(__file__).resolve().parents[3]
                / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/attention_native_stream_stages.py"
            ).read_bytes()
        ).hexdigest()
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    check("reference_preinverse", actual[0], stages["attention"])
    check("reference_postinverse", actual[1], stages["heads"])
    # The previous candidate drifted between executions. Qualify several
    # poisoned FULL replays only after the first Native gates pass.
    for repeat in range(1, 4):
        for value in (pre, post):
            value.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        check(f"repeat_{repeat}_preinverse", heads(pre), stages["attention"])
        check(f"repeat_{repeat}_postinverse", heads(post), stages["heads"])
        assert all(bool((owner[:8].cpu() == -37).all() and (owner[-8:].cpu() == -37).all()) for owner in owners)
    record.update(complete=True, **{"pass": True})
except FirstDifference:
    record["complete"] = True
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    torch.save(payload, out / "attention.pt")
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
