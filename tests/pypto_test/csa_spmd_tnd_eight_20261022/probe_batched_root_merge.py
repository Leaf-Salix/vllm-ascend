# SPDX-License-Identifier: Apache-2.0
"""Check exact tie semantics of batching three 2-way Top512 root merges.

Each source root contains 512 sorted (FP32 score, raw UINT32 index) pairs.
Splitting each root into two sorted 256-item runs permits format1/block_len=512
to process three independent 1024-item merges in one instruction invocation.
The relative priority of equal scores may differ from the original format2
merge; natural and reversed-half layouts are both checked, without relaxing
score/index bit equality. This probe is not a CSA performance measurement.
"""

import argparse
import hashlib
import json
import os
import re
import traceback
from pathlib import Path

import pypto.language as pl


@pl.jit.incore
def compare_roots(
    new_roots: pl.Tensor[[3, 1024], pl.FP32],
    old_roots: pl.Tensor[[3, 1024], pl.FP32],
    reference: pl.InOut[pl.Tensor[[3, 2080], pl.FP32]],
    natural: pl.InOut[pl.Tensor[[3, 2080], pl.FP32]],
    swapped: pl.InOut[pl.Tensor[[3, 2080], pl.FP32]],
):
    for query in pl.unroll(3):
        new_row = pl.load(new_roots, [query, 0], [1, 1024])
        old_row = pl.load(old_roots, [query, 0], [1, 1024])
        scratch = pl.tile.create([1, 2048], pl.FP32, target_memory=pl.MemorySpace.Vec)
        original = pl.tile.mrgsort(new_row, old_row, tmp=scratch)
        pl.store(original, [query, 16], reference)
    new_rows = pl.load(new_roots, [0, 0], [3, 1024])
    old_rows = pl.load(old_roots, [0, 0], [3, 1024])
    natural_rows = pl.concat(new_rows, old_rows)
    natural_flat = pl.reshape(natural_rows, [1, 6144])
    natural_sorted = pl.mrgsort(natural_flat, block_len=512)
    pl.store(pl.reshape(natural_sorted, [3, 2048]), [0, 16], natural)
    new_swapped = pl.concat(pl.slice(new_rows, [3, 512], [0, 512]), pl.slice(new_rows, [3, 512], [0, 0]))
    old_swapped = pl.concat(pl.slice(old_rows, [3, 512], [0, 512]), pl.slice(old_rows, [3, 512], [0, 0]))
    swapped_rows = pl.concat(new_swapped, old_swapped)
    swapped_flat = pl.reshape(swapped_rows, [1, 6144])
    swapped_sorted = pl.mrgsort(swapped_flat, block_len=512)
    pl.store(pl.reshape(swapped_sorted, [3, 2048]), [0, 16], swapped)
    return reference, natural, swapped


@pl.jit
def entry(
    new_roots: pl.Tensor[[3, 1024], pl.FP32],
    old_roots: pl.Tensor[[3, 1024], pl.FP32],
    reference: pl.InOut[pl.Tensor[[3, 2080], pl.FP32]],
    natural: pl.InOut[pl.Tensor[[3, 2080], pl.FP32]],
    swapped: pl.InOut[pl.Tensor[[3, 2080], pl.FP32]],
):
    with pl.spmd(1):
        reference, natural, swapped = compare_roots(new_roots, old_roots, reference, natural, swapped)
    return reference, natural, swapped


def fixture(torch, seed, mode):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn((2, 3, 512), generator=generator)
    if mode == "quantized":
        scores = torch.round(scores * 3.0)
    elif mode == "all_equal":
        scores.fill_(1.0)
    elif mode == "new_dominates":
        scores[0] += 100.0
    elif mode == "old_dominates":
        scores[1] += 100.0
    elif mode == "signed_zero":
        scores.zero_()
        scores[:, :, ::2] = -0.0
    elif mode == "sentinel":
        scores.fill_(-3.4028234663852886e38)
    elif mode == "padded":
        for side in range(2):
            for row, count in enumerate((1, 255, 511)):
                scores[side, row, count:] = -3.4028234663852886e38
    roots = []
    for side in range(2):
        packed = torch.empty((3, 1024), dtype=torch.float32)
        for row in range(3):
            values, order = torch.sort(scores[side, row], descending=True, stable=True)
            indices = torch.arange(512, dtype=torch.int32) + row * 100000 + side * 10000 + seed % 997
            packed[row, 0::2] = values
            packed.view(torch.int32)[row, 1::2] = indices[order]
        roots.append(packed)
    return roots


def audit(root):
    (cpp_path,) = root.glob("kernels/aiv/*.cpp")
    cpp = cpp_path.read_text()
    calls = [line for line in cpp.splitlines() if re.search(r"\bTMRGSORT(?:<[^;]*?>)?\(", line)]
    assert len(calls) == 5, calls
    pto = (root / "ptoas/compare_roots.pto").read_text()
    instructions = [line.split(" loc(")[0] for line in pto.splitlines() if "pto.tmrgsort" in line]
    assert len(instructions) == 5
    return dict(
        cpp=str(cpp_path),
        sha256=hashlib.sha256(cpp.encode()).hexdigest(),
        merge_calls=5,
        format2_reference_calls=3,
        format1_candidate_calls=2,
    )


def run_device(torch, config, report):
    import pypto.torch
    import torch_npu  # noqa: F401

    device_id = int(os.environ["TASK_DEVICE"])
    torch.npu.set_device(device_id)
    pypto.torch.init(device=device_id, platform="a2a3", runtime="tensormap_and_ringbuffer")
    device = torch.device(f"npu:{device_id}")
    new = torch.zeros((3, 1024), dtype=torch.float32, device=device)
    old = torch.zeros_like(new)
    outputs = [torch.full((3, 2080), -7777777.0, dtype=torch.float32, device=device) for _ in range(3)]
    entry(new, old, *outputs, config=config)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        entry(new, old, *outputs, config=config)
    report["device"] = device_id
    modes = ("random", "quantized", "all_equal", "new_dominates", "old_dominates", "signed_zero", "sentinel", "padded")
    for seed in (1024, 20261023, 314159):
        for mode in modes:
            a, b = fixture(torch, seed, mode)
            new.copy_(a)
            old.copy_(b)
            for invocation in ("eager", "graph"):
                for output in outputs:
                    output.fill_(-7777777.0)
                if invocation == "eager":
                    entry(new, old, *outputs, config=config)
                else:
                    graph.replay()
                torch.npu.synchronize()
                values = [output.cpu() for output in outputs]
                payloads = [v[:, 16:2064].contiguous().view(torch.int32) for v in values]
                expected = payloads[0]
                row = dict(
                    seed=seed,
                    mode=mode,
                    invocation=invocation,
                    inputs_unchanged=torch.equal(new.cpu().view(torch.int32), a.view(torch.int32))
                    and torch.equal(old.cpu().view(torch.int32), b.view(torch.int32)),
                    guards=all(
                        bool(torch.all(v[:, :16] == -7777777.0) and torch.all(v[:, 2064:] == -7777777.0))
                        for v in values
                    ),
                    variants={},
                )
                for name, actual in zip(("natural", "swapped"), payloads[1:]):
                    different = actual != expected
                    row["variants"][name] = dict(
                        prefix_bitwise=not bool(different[:, :1024].any()),
                        full_bitwise=not bool(different.any()),
                        prefix_mismatches=different[:, :1024].sum(dim=1).tolist(),
                        full_mismatches=different.sum(dim=1).tolist(),
                    )
                if mode in ("all_equal", "signed_zero", "sentinel") and invocation == "eager":
                    row["tie_index_chunks"] = {
                        name: [int(payload[0, offset * 2 + 1]) for offset in (0, 255, 256, 511, 512, 767, 768, 1023)]
                        for name, payload in zip(("reference", "natural", "swapped"), payloads)
                    }
                report["cases"].append(row)
    report["qualified_variants"] = [
        name
        for name in ("natural", "swapped")
        if all(row["variants"][name]["prefix_bitwise"] for row in report["cases"])
    ]
    report["execution_checks_passed"] = all(row["inputs_unchanged"] and row["guards"] for row in report["cases"])
    report["passed"] = report["execution_checks_passed"] and bool(report["qualified_variants"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    from csa_shared_toolchain.selection import selected_toolchain
    from pypto import passes
    from pypto.runtime import CompileOptions

    selected, _, manifest = selected_toolchain()
    report = dict(
        passed=False,
        cpu_only=args.cpu_only,
        performance_measured=False,
        full_csa_validated=False,
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
            prepare_prebuilt(artifact.output_dir, "a2a3", BuildKind.SINGLE_CHIP, kernel_abi=artifact.kernel_abi)
            report.update(passed=True, binary_compile_passed=True)
        else:
            import torch

            run_device(torch, config, report)
        report["codegen"] = audit(args.output / "kernel")
    except BaseException as exc:
        report.update(passed=False, error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("traceback", "cases")}, ensure_ascii=False), flush=True
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
