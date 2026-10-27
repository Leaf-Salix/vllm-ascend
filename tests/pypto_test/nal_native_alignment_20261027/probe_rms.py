# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind the private package before importing its DSL modules.
"""Native显式InvRms与两种DSL归约同输入逐位对拍，首差停止。"""

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

source = bind()
from rms_kernels import rms_pair

from vllm_ascend.utils import enable_custom_op

out = Path(sys.argv[1])
record = {
    "complete": False,
    "pass": False,
    "scope": "isolated HC input RMS; not fused HcPre or whole CSA",
    "reference_kind": "standalone_hc_pre_inv_rms",
    "next_gate_requires_fused_hc_pre_reference": True,
    "cases": [],
}
start = time.monotonic()


def save():
    (out / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def metric(actual, expected):
    actual, expected = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    differences = actual.view(torch.int32) != expected.view(torch.int32)
    delta = actual.double() - expected.double()
    return {
        "bit_equal": not bool(differences.any()),
        "different_elements": int(differences.sum()),
        "max_abs": float(delta.abs().max()),
        "max_bit_distance": int((actual.view(torch.int32).long() - expected.view(torch.int32).long()).abs().max()),
        "first_coordinates": differences.nonzero()[:6].tolist(),
    }


try:
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    assert enable_custom_op()
    record["native_schema"] = str(torch.ops._C_ascend.npu_hc_pre_inv_rms.default._schema)
    record["sdk"] = {
        name: os.environ.get(name)
        for name in (
            "CSA_SHARED_TOOLCHAIN_RELEASE",
            "CSA_SHARED_PYPTO_ROOT",
            "PTOAS_ROOT",
            "CSA_SHARED_SIMPLER_ROOT",
            "ASCEND_HOME_PATH",
        )
    }
    record["source_sha256"] = {
        name: hashlib.sha256((source / "deepseek_v4_flash_csa_spmd_tnd" / name).read_bytes()).hexdigest()
        for name in ("hc_stages.py", "hc_input_rms_stages.py")
    }
    record["determinism"] = {
        "torch": torch.are_deterministic_algorithms_enabled(),
        "npu_level": torch_npu._C._npu_get_deterministic_level(),
        "HCCL_DETERMINISTIC": os.environ["HCCL_DETERMINISTIC"],
    }
    pypto.torch.init()
    for tokens in (24, 18, 1, 56, 192):
        item = {"tokens": tokens, "graph_tokens": tokens, "padded_request_rows": 0}
        record["cases"].append(item)
        torch.manual_seed(1024 + tokens)
        hidden_cpu = torch.randn((tokens, 4, 4096), dtype=torch.bfloat16)
        x = hidden_cpu.to("npu").view(tokens, 16384)
        p16 = (tokens + 15) // 16 * 16
        wide = torch.empty((tokens, 16384), dtype=torch.float32, device="npu")
        guards = [torch.full((p16 + 16, 1), -37.0, dtype=torch.float32, device="npu") for _ in range(2)]
        baseline, candidate = [value[8 : 8 + p16] for value in guards]
        squares = torch.full((p16 * 22, 16), float("nan"), dtype=torch.float32, device="npu")
        hidden = x.view(tokens, 4, 4096)
        native = torch.ops._C_ascend.npu_hc_pre_inv_rms(hidden, 1e-6).reshape(tokens, 1)
        native_repeated = torch.ops._C_ascend.npu_hc_pre_inv_rms(hidden, 1e-6).reshape(tokens, 1)
        item["native_repeat"] = metric(native, native_repeated)
        assert item["native_repeat"]["bit_equal"]
        with torch.inference_mode():
            rms_pair(x, wide, baseline, candidate, squares)
            torch.npu.synchronize()
            eager = [value[:tokens].cpu().clone() for value in (baseline, candidate)]
            graph = torch.npu.NPUGraph()
            stream = torch.npu.Stream()
            with torch.npu.graph(graph, stream=stream):
                rms_pair(x, wide, baseline, candidate, squares)
            wide.fill_(float("nan"))
            squares.fill_(float("nan"))
            baseline.fill_(float("nan"))
            candidate.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
        item["replay_eager"] = [
            metric(value[:tokens], expected) for value, expected in zip((baseline, candidate), eager, strict=True)
        ]
        assert all(value["bit_equal"] for value in item["replay_eager"])
        item["wide_replay_exact"] = torch.equal(wide.cpu(), hidden_cpu.view(tokens, 16384).float())
        assert item["wide_replay_exact"], "widen stage missing from FULL replay"
        diagonal = torch.arange(tokens, dtype=torch.long) * 16 + torch.arange(tokens, dtype=torch.long) % 16
        square_cpu = squares.cpu().flatten()
        item["square_diagonal_rewritten"] = all(
            bool(torch.isfinite(square_cpu[diagonal + part * p16 * 16]).all()) for part in range(22)
        )
        assert item["square_diagonal_rewritten"], "square stage missing from FULL replay"
        item["nalinaly_vs_native_inv_rms"] = metric(baseline[:tokens], native)
        item["candidate_vs_native_inv_rms"] = metric(candidate[:tokens], native)
        item["guards"] = all(bool((value[:8] == -37.0).all()) and bool((value[-8:] == -37.0).all()) for value in guards)
        assert item["guards"]
        torch.save(
            {
                "hidden": hidden_cpu,
                "native_inverse": native.cpu(),
                "nalinaly_inverse": baseline[:tokens].cpu(),
                "candidate_inverse": candidate[:tokens].cpu(),
                "square_partials": squares.cpu(),
            },
            out / f"t{tokens}.pt",
        )
        if tokens == 24 and item["candidate_vs_native_inv_rms"]["bit_equal"]:
            item["aba"] = []
            for label, hidden_cpu_second in (("B", hidden_cpu + 0.125), ("A", hidden_cpu)):
                x.copy_(hidden_cpu_second.view(tokens, 16384))
                native_second = torch.ops._C_ascend.npu_hc_pre_inv_rms(x.view(tokens, 4, 4096), 1e-6).reshape(tokens, 1)
                with torch.inference_mode():
                    rms_pair(x, wide, baseline, candidate, squares)
                    torch.npu.synchronize()
                    eager_second = [value[:tokens].cpu().clone() for value in (baseline, candidate)]
                    wide.fill_(float("nan"))
                    squares.fill_(float("nan"))
                    baseline.fill_(float("nan"))
                    candidate.fill_(float("nan"))
                    graph.replay()
                    torch.npu.synchronize()
                replay = [
                    metric(value[:tokens], expected)
                    for value, expected in zip((baseline, candidate), eager_second, strict=True)
                ]
                comparison = metric(candidate[:tokens], native_second)
                item["aba"].append({"input": label, "replay_eager": replay, "candidate_vs_native": comparison})
                assert all(row["bit_equal"] for row in replay), "FULL replay stale intermediate"
                assert torch.equal(wide.cpu(), hidden_cpu_second.view(tokens, 16384).float())
                assert comparison["bit_equal"], "first RMS difference on changed input"
        save()
        print(json.dumps(item), flush=True)
        if not item["candidate_vs_native_inv_rms"]["bit_equal"]:
            record["first_difference_tokens"] = tokens
            break
    record["pass"] = all(item["candidate_vs_native_inv_rms"]["bit_equal"] for item in record["cases"])
    record["complete"] = True
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    save()
