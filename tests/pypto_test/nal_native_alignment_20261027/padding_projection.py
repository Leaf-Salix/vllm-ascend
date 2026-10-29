# SPDX-License-Identifier: Apache-2.0
"""只改变主投影判断看到的尾0跨度；其余计算与state写回不变。"""

import json

import pypto.torch
import torch
from mainproj_kernels import mainproj_arm


def compare_projection(f, layer, normed, native, metadata, out, record, payload, metric, save):
    previous_dir = out.parent / "padding-capacity5-v1"
    previous_record = json.loads((previous_dir / "result.json").read_text())
    previous = torch.load(previous_dir / "failure.pt", map_location="cpu", weights_only=True)
    assert previous_record["first_difference"] == "state.0"
    assert record["stage_source_sha256"] == previous_record["stage_source_sha256"]
    assert metadata.keys() == previous["metadata"].keys()

    # The padding harness now canonicalizes Native's [T,1,1,64] RoPE view to
    # [T,64] capacity descriptors. Compare all original bytes, not a slice.
    def canonical(name, value):
        return value.reshape(-1, 64) if name.endswith((".cos.rows", ".sin.rows")) else value

    inputs = {
        "hidden": metric(f["hidden"], previous["hidden"]),
        "positions": metric(f["positions"], previous["positions"]),
        **{
            f"initial.{name}": metric(group["initial"], previous["initial"][name])
            for name, group in f["groups"].items()
        },
        **{f"native.{name}": metric(value, previous["native_full"][name]) for name, value in native.items()},
        **{
            f"metadata.{name}": metric(canonical(name, value), canonical(name, previous["metadata"][name]))
            for name, value in metadata.items()
        },
    }
    record["projection_same_inputs"] = inputs
    save()
    assert all(row["bit_equal"] for row in inputs.values())
    impl = layer.self_attn.dsa_attn.dsa_attn.impl
    weights = [
        value.cpu().to("npu")
        for value in (impl.compressor_wkv.weight, impl.compressor_wgate.weight, impl.compressor_ape)
    ]
    req = f["metadata"][f["groups"]["compressed"]["prefix"]].decode
    state_req = f["metadata"][f["groups"]["state"]["prefix"]].decode
    bounds = req.query_start_loc
    assert bounds.cpu().tolist() == [0, 6, 12, 18, 24, 24]
    state_view = f["groups"]["state"]["views"][0]
    group = f["groups"]["state"]
    initial_view = (
        group["initial"]
        .view(state_view.dtype)
        .as_strided(state_view.shape, state_view.stride(), state_view.storage_offset())
    )
    initial = initial_view.reshape(state_view.shape[0], -1).clone()
    gold = native["state.0"].reshape_as(initial)
    actual = previous["pto_full_0"]["state.0"].reshape_as(initial)
    owner_initial = torch.cat((torch.full((32,), -37.0), initial.reshape(-1), torch.full((32,), -37.0)))
    owner = owner_initial.to("npu")
    state = owner[32:-32].reshape_as(initial)
    projection_owners = [torch.full((80, 1024), -37.0, device="npu") for _ in range(2)]
    values, scores = (value[8:72] for value in projection_owners)
    x = normed.to("npu")
    assert x.shape == (24, 4096) and x.dtype == torch.bfloat16
    payload["projection_normed_native"] = normed
    record["projection_arms"] = {}
    pypto.torch.init()
    readonly = [x, *weights, f["positions"], bounds, req.seq_lens, state_req.slot_mapping]
    readonly_initial = [value.cpu().clone() for value in readonly]
    for label, q in (("original_bounds6", bounds), ("trim_tail_zero_bounds5", bounds[:5])):

        def run(q=q):
            mainproj_arm(
                x, *weights, f["positions"], q, req.seq_lens, state_req.slot_mapping, state, values, scores, MODE=3
            )

        owner.copy_(owner_initial)
        run()
        torch.npu.synchronize()
        eager = state.cpu().clone()
        payload[label + ".eager"] = eager
        assert all(torch.equal(value.cpu(), initial) for value, initial in zip(readonly, readonly_initial))
        graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, stream=stream):
            run()
        row = {"replays": []}
        record["projection_arms"][label] = row
        for _ in range(2):
            owner.copy_(owner_initial)
            values.fill_(float("nan"))
            scores.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            result = state.cpu().clone()
            payload[label] = result
            checks = {
                "full_eager": metric(result, eager),
                "native": metric(result, gold),
                "production_reproduction": metric(result, actual),
                "readonly": all(
                    torch.equal(value.cpu(), initial) for value, initial in zip(readonly, readonly_initial)
                ),
                "guards": bool(
                    torch.equal(owner[:32].cpu(), owner_initial[:32])
                    and torch.equal(owner[-32:].cpu(), owner_initial[-32:])
                    and all(
                        bool((value[:8].cpu() == -37).all() and (value[72:].cpu() == -37).all())
                        for value in projection_owners
                    )
                ),
            }
            row["replays"].append(checks)
            save()
            assert checks["full_eager"]["bit_equal"] and checks["guards"] and checks["readonly"]
            if label.startswith("original"):
                assert checks["production_reproduction"]["bit_equal"], (
                    "micro must reproduce the actual first difference"
                )
            else:
                assert checks["native"]["bit_equal"], "trim-only candidate has not resolved Native state difference"
