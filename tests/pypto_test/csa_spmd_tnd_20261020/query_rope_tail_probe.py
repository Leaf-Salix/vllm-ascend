# SPDX-License-Identifier: Apache-2.0
"""Exact helper comparison for real T tails and untouched padded rows; no timing."""

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path


def load_entry(directory, output):
    from source_loader import load_package

    reference = load_package(directory / "reference", "rope_tail_reference")
    candidate = load_package(directory / "candidate", "rope_tail_candidate")
    text = f"""import pypto.language as pl
from {reference.name}.indexer_query_stages import (
    T_DYN, PROJ_ROWS_DYN, ROPE_ROWS_DYN, indexer_query_dequant_rope_stage as reference_stage,
)
from {candidate.name}.indexer_query_stages import indexer_query_dequant_rope_stage as candidate_stage

@pl.jit.incore
def compare(qscale: pl.Tensor[[T_DYN, 1], pl.FP32],
            wscale: pl.Tensor[[8192], pl.FP32],
            cos: pl.Tensor[[T_DYN, 64], pl.FP32],
            sin: pl.Tensor[[T_DYN, 64], pl.FP32],
            acc: pl.Tensor[[PROJ_ROWS_DYN, 8192], pl.INT32],
            ref: pl.InOut[pl.Tensor[[ROPE_ROWS_DYN, 8192], pl.BF16]],
            out: pl.InOut[pl.Tensor[[ROPE_ROWS_DYN, 8192], pl.BF16]]):
    worker = pl.tile.get_block_idx()
    ref = reference_stage(qscale, wscale, cos, sin, acc, ref, worker, 48)
    out = candidate_stage(qscale, wscale, cos, sin, acc, out, worker, 48)
    return ref, out

@pl.jit
def entry(qscale: pl.Tensor[[T_DYN, 1], pl.FP32],
          wscale: pl.Tensor[[8192], pl.FP32],
          cos: pl.Tensor[[T_DYN, 64], pl.FP32],
          sin: pl.Tensor[[T_DYN, 64], pl.FP32],
          acc: pl.Tensor[[PROJ_ROWS_DYN, 8192], pl.INT32],
          ref: pl.InOut[pl.Tensor[[ROPE_ROWS_DYN, 8192], pl.BF16]],
          out: pl.InOut[pl.Tensor[[ROPE_ROWS_DYN, 8192], pl.BF16]]):
    with pl.spmd(48):
        ref, out = compare(qscale, wscale, cos, sin, acc, ref, out)
    return ref, out
"""
    path = output / "entry.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location("query_rope_tail_test_entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.entry


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--device", type=int)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, device_execution=False, performance_measured=False, cases=[])
    try:
        import torch
        from pypto import passes
        from pypto.runtime import CompileOptions

        entry = load_entry(Path(__file__).resolve().parent, args.output)
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        if args.cpu_only:
            artifact = entry._resolve_kernel_artifact((), {"config": options}, allow_signature_mode=True)
            report["artifact"] = str(artifact.output_dir)
        else:
            import pypto.torch
            import torch_npu

            from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs

            dev = int(os.environ["TASK_DEVICE"])
            if args.device is not None and args.device != dev:
                raise ValueError("Probe device must match task-submit allocation")
            torch.npu.set_device(dev)
            torch_npu.npu.set_deterministic_level(0)
            pypto.torch.init(device=dev, platform="a2a3", runtime="tensormap_and_ringbuffer", **ring_sizing_kwargs())
            report["device_execution"] = True
            for tokens in (*range(1, 10), 15, 16, 17, 23):
                shapes = ((tokens, 1), (8192,), (tokens, 64), (tokens, 64), (64, 8192))
                inputs, backing = [], []
                for index, shape in enumerate(shapes):
                    dtype = torch.int32 if index == 4 else torch.float32
                    count = 1
                    for dimension in shape:
                        count *= dimension
                    storage = torch.full((count + 256,), 37, dtype=dtype, device=f"npu:{dev}")
                    inputs.append(storage[128:-128].view(shape))
                    backing.append(storage)
                outputs = [torch.full((66, 8192), 37, dtype=torch.bfloat16, device=f"npu:{dev}") for _ in range(2)]
                ref, out = [value[1:-1] for value in outputs]
                tensors = (*inputs, ref, out)
                entry(*tensors, config=options)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    entry(*tensors, config=options)
                addresses = tuple(value.data_ptr() for value in tensors)
                for seed in (20261021, 20261022, 0):
                    generator = torch.Generator().manual_seed(seed)
                    values = [torch.rand(shape, generator=generator) * 0.01 for shape in shapes[:-1]]
                    values[2] = values[2] * 200 - 1
                    values[3] = values[3] * 200 - 1
                    acc = torch.randint(-1000000, 1000001, shapes[-1], dtype=torch.int32, generator=generator)
                    if seed == 0:
                        acc.zero_()
                    values.append(acc)
                    for tensor, value in zip(inputs, values):
                        tensor.copy_(value)
                    for tensor in outputs:
                        tensor.fill_(37)
                    graph.replay()
                    torch.npu.synchronize()
                    actual = [value.cpu() for value in outputs]
                    guards = all(bool((value[0] == 37).all() and (value[tokens + 1 :] == 37).all()) for value in actual)
                    guards = guards and all(
                        bool((value[:128].cpu() == 37).all() and (value[-128:].cpu() == 37).all()) for value in backing
                    )
                    inputs_ok = all(torch.equal(tensor.cpu(), value) for tensor, value in zip(inputs, values))
                    same = torch.equal(actual[0].view(torch.uint8), actual[1].view(torch.uint8))
                    addresses_ok = addresses == tuple(value.data_ptr() for value in tensors)
                    row = dict(
                        tokens=tokens,
                        seed=seed,
                        exact=same,
                        guards=guards,
                        inputs_unchanged=inputs_ok,
                        same_addresses=addresses_ok,
                    )
                    row["passed"] = same and guards and inputs_ok and addresses_ok
                    report["cases"].append(row)
                    if not row["passed"]:
                        torch.save(
                            dict(reference=actual[0], candidate=actual[1], inputs=values), args.output / "failure.pt"
                        )
                        raise AssertionError(row)
        report["passed"] = True
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("traceback", "cases")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
