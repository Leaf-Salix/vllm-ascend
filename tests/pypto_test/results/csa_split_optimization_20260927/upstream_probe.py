"""Current pypto-lib TP1 CSA, S6/H8192/B16, using the same Torch graph timer.

The active direct-score branch is unchanged; the unreachable B>=64 buffered
branch is removed in an isolated copy to reproduce the reference collected
before the local FIXPIPE port. The upstream synthetic fixture and FP32 boundary
remain upstream's own contract; this is a performance reference, not a Native
numerical or model acceptance test.
"""

import argparse
import ast
import inspect
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


def export_window(directory):
    builds = list(Path("build_output").glob("_jit__decode_csa_tp1_*/kernel_config.py"))
    if len(builds) != 1:
        raise RuntimeError(f"Expected one upstream root build, found {len(builds)}")
    config = builds[0]
    shutil.copyfile(config, directory / "kernel_config_source.py")
    tables = [node.value for node in ast.parse(config.read_text()).body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "KERNELS" for t in node.targets)]
    names = {}
    for node in tables[0].elts:
        values = {ast.literal_eval(k): v for k, v in zip(node.keys, node.values)}
        names[str(ast.literal_eval(values["func_id"]))] = ast.literal_eval(values["name"])
    name_map = directory / "name_map.json"
    name_map.write_text(json.dumps({"callable_id_to_name": names}, indent=2))
    target = directory / "merged_swimlane.json"
    result = subprocess.run([sys.executable, "-m", "simpler_setup.tools.swimlane_converter",
                             str(directory / "chip_swimlane_records.json"), "--func-names", str(name_map),
                             "-o", str(target)], capture_output=True, text=True)
    (directory / "converter_output.txt").write_text(result.stdout + result.stderr)
    result.check_returncode()
    return str(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--swimlane", action="store_true")
    parser.add_argument("--history", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output)
    if not 1 <= args.batch < 64:
        raise ValueError("The direct-branch reference is restricted to B<64")
    isolated = args.output / "reference_modules"
    isolated.mkdir(exist_ok=True)
    source = (args.upstream / "models/deepseek_v4_flash_dspark/decode_indexer.py").read_text()
    branches = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.If)
                and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                and n.test.left.id == "buffered_score"]
    if len(branches) != 1:
        raise ValueError("Upstream buffered-score branch changed; inspect before adapting")
    branch = branches[0]
    lines = source.splitlines(keepends=True)
    direct = lines[:branch.lineno - 1] + [line[4:] if line.startswith("    ") else line
        for line in lines[branch.orelse[0].lineno - 1:branch.end_lineno]] + lines[branch.end_lineno:]
    (isolated / "decode_indexer.py").write_text("".join(direct))
    sys.path[:0] = [str(isolated), str(args.upstream / "models/deepseek_v4_flash_dspark"), str(args.upstream)]
    import config

    # Match the integration's compiled 64-request capacity and actual B16/S6.
    config.DECODE_BATCH = 16  # Upstream reference capacity; default 64 exceeds the kernel-mode heap.
    config.DSPARK_SPEC_TOKENS = 5
    config.DECODE_SEQ = 6
    config.DECODE_TOKENS = config.DECODE_BATCH * config.DECODE_SEQ
    config.CSA_STATE_BLOCKS_PER_REQUEST = (8 + 6 + config.C4A_COMPRESSOR_BLOCK_SIZE - 1) // config.C4A_COMPRESSOR_BLOCK_SIZE
    config.CSA_INNER_STATE_BLOCKS_PER_REQUEST = config.CSA_STATE_BLOCKS_PER_REQUEST
    config.CSA_STATE_PHYSICAL_BLOCKS = config.DECODE_BATCH * config.CSA_STATE_BLOCKS_PER_REQUEST
    config.CSA_INNER_STATE_PHYSICAL_BLOCKS = config.CSA_STATE_PHYSICAL_BLOCKS
    sys.argv += ["--tp", "1"]
    import decode_csa
    import torch
    import torch_npu  # noqa: F401
    import pypto.torch

    torch.set_num_threads(8)
    torch.manual_seed(1024)
    torch.npu.set_device(args.device)
    torch.npu.set_deterministic_level(0)
    specs = decode_csa.build_tensor_specs(start_pos=[args.history] * args.batch, batch=args.batch)
    tensors = {s.name: s.create_tensor().reshape(s.shape).to(f"npu:{args.device}") for s in specs}
    mutable = ("compress_state", "inner_compress_state", "kv_cache", "cmp_kv", "idx_kv_cache", "idx_kv_scale")
    initial = {name: tensors[name].clone() for name in mutable}
    op = pypto.torch.register(decode_csa.decode_csa_tp1_test, "csa_upstream_probe::attention")
    ordered = [tensors[name] for name in inspect.signature(decode_csa._decode_csa_tp1).parameters]
    pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer",
                     **({"enable_chip_swimlane": 4, "enable_dep_gen": True,
                         "output_dir": str(args.output / "dfx")} if args.swimlane else {}))

    def reset():
        for name, value in initial.items():
            tensors[name].copy_(value)
        tensors["x_out"].fill_(float("nan"))

    reset()
    op(*ordered)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        op(*ordered)
    for _ in range(5):
        reset()
        graph.replay()
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    samples, windows = [], []
    for index in range(4 if args.swimlane else 20):
        reset()
        if args.swimlane:
            torch.npu.synchronize()
            pypto.torch.begin_dfx()
        start.record()
        graph.replay()
        end.record()
        if args.swimlane:
            pypto.torch.end_dfx()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
        if args.swimlane:
            directory = args.output / "dfx"
            if index:
                directory /= f"window_{index}"
            windows.append({"window": index, "merged_swimlane": export_window(directory),
                            "profiled_event_us": samples[-1]})
    finite = bool(torch.isfinite(tensors["x_out"]).all().item())
    report = {"scope": "upstream active direct-score branch unchanged, synthetic fixture; performance reference only",
              "compatibility": "isolated decode_indexer.py removes unreachable B>=64 buffered branch; local PyPTO lacks store(pre_quant)",
              "source_revision": subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
              "batch": args.batch, "sequence": 6, "history": args.history, "tp": 1, "compiled_batch_capacity": config.DECODE_BATCH,
              "seed": 1024, "device": args.device, "warmup": 5, "profiler": args.swimlane,
              "boundary": "HC_pre to HC_post; upstream FP32 residual and scale, independent contiguous cache",
              "weight_nz": ["wq_a", "wq_b", "wo_a", "wo_b"], "samples_us": samples,
              "mean_us": statistics.mean(samples), "p50_us": statistics.median(samples),
              "output_finite": finite, "numerical_acceptance": "not evaluated", "swimlane_windows": windows}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not finite:
        raise ValueError("Upstream output contains nonfinite values")


if __name__ == "__main__":
    main()
