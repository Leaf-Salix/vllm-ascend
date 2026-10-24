# SPDX-License-Identifier: Apache-2.0
"""Check A3 GEMV against CSA's exact diagonal WS before any full candidate.

This is a numerical probe, not a performance benchmark.
The reference executes the retained two K192 FP16 matrix products. Six K64
GEMVs use the six nonzero coefficient rows and store independently. The
column-pool variant is frozen in the failed CPU v2 evidence: its assemble
emitted unsupported Acc-to-Acc TMOVs. Inputs/outputs have fixed addresses
through graph replay; guard checks and bit comparisons include signed zero.
"""

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import pypto.language as pl


@pl.jit.incore
def ws_compare(
    coefficients: pl.Tensor[[16, 384], pl.FP16],
    scores: pl.Tensor[[384, 64], pl.FP16],
    reference: pl.InOut[pl.Tensor[[6, 96], pl.FP32]],
    candidate: pl.InOut[pl.Tensor[[1, 416], pl.FP32]],
):
    coefficients0 = pl.load(coefficients, [0, 0], [16, 192], target_memory=pl.MemorySpace.Mat)
    coefficients1 = pl.load(coefficients, [0, 192], [16, 192], target_memory=pl.MemorySpace.Mat)
    scores0 = pl.load(scores, [0, 0], [192, 64], target_memory=pl.MemorySpace.Mat)
    scores1 = pl.load(scores, [192, 0], [192, 64], target_memory=pl.MemorySpace.Mat)
    left0 = pl.move(coefficients0, target_memory=pl.MemorySpace.Left)
    left1 = pl.move(coefficients1, target_memory=pl.MemorySpace.Left)
    right0 = pl.move(scores0, target_memory=pl.MemorySpace.Right)
    right1 = pl.move(scores1, target_memory=pl.MemorySpace.Right)
    accumulator = pl.tile.create([16, 64], pl.FP32, target_memory=pl.MemorySpace.Acc)
    partial = pl.tile.matmul_acc(accumulator, left0, right0, init_cond=True)
    result = pl.tile.matmul_acc(partial, left1, right1)
    reference = pl.store(pl.set_validshape(result, 6, 64), [0, 16], reference)

    for query in pl.unroll(6):
        # A3's one-row Mat-to-Left transfer requires 512B physical K alignment.
        # Only K64 is valid/read from GM and consumed by GEMV.
        row = pl.load(
            coefficients, [query, query * 64], [1, 256], valid_shape=[1, 64], target_memory=pl.MemorySpace.Mat
        )
        row_left = pl.move(row, target_memory=pl.MemorySpace.Left)
        block = pl.load(scores, [query * 64, 0], [256, 64], valid_shape=[64, 64], target_memory=pl.MemorySpace.Mat)
        block_right = pl.move(block, target_memory=pl.MemorySpace.Right)
        reduced = pl.tile.gemv(row_left, block_right)
        candidate = pl.store(reduced, [0, 16 + query * 64], candidate)
    return reference, candidate


@pl.jit
def entry(
    coefficients: pl.Tensor[[16, 384], pl.FP16],
    scores: pl.Tensor[[384, 64], pl.FP16],
    reference: pl.InOut[pl.Tensor[[6, 96], pl.FP32]],
    candidate: pl.InOut[pl.Tensor[[1, 416], pl.FP32]],
):
    with pl.spmd(1):
        reference, candidate = ws_compare(coefficients, scores, reference, candidate)
    return reference, candidate


def fixture(torch, seed, mode):
    """QK-like positive FP16 values and explicitly diagonal coefficient inputs."""
    generator = torch.Generator().manual_seed(seed)
    query = torch.randint(-127, 128, (384, 128), generator=generator, dtype=torch.int32)
    key = torch.randint(-127, 128, (128, 64), generator=generator, dtype=torch.int32)
    scores = ((query @ key).clamp_min(0).float() / 1024.0).half()
    scale = (torch.rand((6, 64), generator=generator) * 0.025 + 0.001).half()
    weight = torch.randn((6, 64), generator=generator).half()
    rows = (scale * weight).half()
    if mode == "positive":
        rows = rows.abs()
    elif mode == "zeros":
        rows[0].zero_()
        rows[1, ::2] = -0.0
        scores[128:192].zero_()
        scores[:, ::8] = 0.0
    elif mode == "wide":
        exponents = torch.randint(-14, 8, (6, 64), generator=generator)
        rows = (torch.sign(weight) * torch.pow(2.0, exponents.float())).half()
        scores[:, 1::4] *= 0.001
    coefficients = torch.zeros((16, 384), dtype=torch.float16)
    for query_index in range(6):
        coefficients[query_index, query_index * 64 : (query_index + 1) * 64] = rows[query_index]
    assert torch.isfinite(coefficients).all() and torch.isfinite(scores).all()
    return coefficients, scores


def audit_artifact(root):
    """Reject accidental TMATMUL promotion and hidden Acc-to-Acc copies."""
    import re

    cpp_files = list(root.glob("kernels/aic/*.cpp"))
    assert len(cpp_files) == 1, cpp_files
    cpp = cpp_files[0].read_text()
    counts = {
        op: len(re.findall(r"\b" + op + r"(?:<[^;]*?>)?\(", cpp))
        for op in ("TGEMV", "TGEMV_ACC", "TMATMUL", "TMATMUL_ACC")
    }
    assert counts == {"TGEMV": 6, "TGEMV_ACC": 0, "TMATMUL": 1, "TMATMUL_ACC": 1}, counts
    assert "TINSERT" not in cpp, "Acc column assembly must be alias-only"
    return dict(cpp=str(cpp_files[0]), sha256=hashlib.sha256(cpp.encode()).hexdigest(), operations=counts)


def run_device(torch, config, report):
    import pypto.torch
    import torch_npu  # noqa: F401

    device_id = int(os.environ["TASK_DEVICE"])
    torch.npu.set_device(device_id)
    pypto.torch.init(device=device_id, platform="a2a3", runtime="tensormap_and_ringbuffer")
    device = torch.device(f"npu:{device_id}")
    coefficients = torch.zeros((16, 384), dtype=torch.float16, device=device)
    scores = torch.zeros((384, 64), dtype=torch.float16, device=device)
    reference = torch.full((6, 96), -7777777.0, dtype=torch.float32, device=device)
    candidate = torch.full((1, 416), -7777777.0, dtype=torch.float32, device=device)
    entry(coefficients, scores, reference, candidate, config=config)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        entry(coefficients, scores, reference, candidate, config=config)
    report["device"] = device_id
    for seed in (1024, 20261023, 314159):
        for mode in ("signed", "positive", "zeros", "wide"):
            a, b = fixture(torch, seed, mode)
            coefficients.copy_(a)
            scores.copy_(b)
            for invocation in ("eager", "graph"):
                reference.fill_(-7777777.0)
                candidate.fill_(-7777777.0)
                if invocation == "eager":
                    entry(coefficients, scores, reference, candidate, config=config)
                else:
                    graph.replay()
                torch.npu.synchronize()
                ref, got = reference.cpu(), candidate.cpu()
                expected = ref[:, 16:80].contiguous()
                actual = got[:, 16:400].reshape(6, 64).contiguous()
                different = expected.view(torch.int32) != actual.view(torch.int32)
                guards = all(
                    bool(torch.all(part == -7777777.0))
                    for part in (ref[:, :16], ref[:, 80:], got[:, :16], got[:, 400:])
                )
                unchanged = torch.equal(coefficients.cpu().view(torch.int16), a.view(torch.int16)) and torch.equal(
                    scores.cpu().view(torch.int16), b.view(torch.int16)
                )
                finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
                report["cases"].append(
                    dict(
                        seed=seed,
                        mode=mode,
                        invocation=invocation,
                        bitwise=not bool(different.any()),
                        mismatch_per_query=different.sum(dim=1).tolist(),
                        max_abs_difference=float((actual - expected).abs().max()),
                        guards=guards,
                        inputs_unchanged=unchanged,
                        finite=finite,
                    )
                )
    report["passed"] = all(
        row["bitwise"] and row["guards"] and row["inputs_unchanged"] and row["finite"] for row in report["cases"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    if args.device is not None:
        assert args.device == int(os.environ["TASK_DEVICE"])
    args.output.mkdir(parents=True, exist_ok=False)
    from csa_shared_toolchain.selection import selected_toolchain
    from pypto import passes
    from pypto.runtime import CompileOptions

    selected, _, manifest = selected_toolchain()
    report = dict(
        passed=False,
        cpu_only=args.cpu_only,
        full_csa_validated=False,
        performance_measured=False,
        shared_toolchain=selected,
        shared_manifest=str(manifest),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=[],
    )
    config = CompileOptions(
        platform="a2a3",
        output_dir=str(args.output / "kernel"),
        dump_passes=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
    )
    try:
        if args.cpu_only:
            from pypto.jit._artifact_manifest import BuildKind
            from pypto.runtime._prebuilt import prepare_prebuilt

            artifact = entry._resolve_kernel_artifact((), {"config": config}, allow_signature_mode=True)
            report["artifact"] = str(artifact.output_dir)
            prepare_prebuilt(artifact.output_dir, "a2a3", BuildKind.SINGLE_CHIP, kernel_abi=artifact.kernel_abi)
            report["binary_compile_passed"] = True
            report["passed"] = True
        else:
            import torch

            run_device(torch, config, report)
        report["codegen"] = audit_artifact(args.output / "kernel")
    except BaseException as exc:
        report.update(passed=False, error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in report.items() if key != "traceback"}, ensure_ascii=False), flush=True
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
