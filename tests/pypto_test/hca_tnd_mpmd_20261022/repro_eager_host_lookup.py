# SPDX-License-Identifier: Apache-2.0
"""无需模型权重的热JIT查找复现：一条32层inline链，最终只有一个加法kernel。

    python repro_eager_host_lookup.py --output /absolute/new_dir --device 0

普通注册调用与已预热registration使用相同二进制、相同输入和SDK参数校验。
后一条只作诊断，不作为生产公共API；所有源码、constexpr和编译配置保持不变。
"""

import argparse
import cProfile
import importlib.util
import json
import os
import pstats
import statistics
import sys
import time
from pathlib import Path


def write_graph(path, depth):
    source = '''import pypto.language as pl
T = pl.dynamic("T")

@pl.jit.inline
def leaf(x: pl.Tensor[[T, 32], pl.FP32], y: pl.Tensor[[T, 32], pl.FP32], bias: pl.constexpr):
    with pl.spmd(1):
        row0 = pl.tile.get_block_idx() * 32
        rows = pl.tensor.dim(x, 0)
        tile = pl.load(x, [row0, 0], [32, 32], valid_shape=[rows - row0, 32])
        pl.store(pl.add(tile, bias), [row0, 0], y)
    return y
'''
    previous = "leaf"
    for index in range(depth):
        name = f"stage_{index}"
        source += f'''
@pl.jit.inline
def {name}(x: pl.Tensor[[T, 32], pl.FP32], y: pl.Tensor[[T, 32], pl.FP32], bias: pl.constexpr):
    y = {previous}(x, y, bias + 1.0)
    return y
'''
        previous = name
    source += f'''
@pl.jit
def entry(x: pl.Tensor[[T, 32], pl.FP32], y: pl.Out[pl.Tensor[[T, 32], pl.FP32]]):
    y = {previous}(x, y, 0.0)
    return y
'''
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("eager_lookup_graph", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "0").split(",")[0]))
    parser.add_argument("--depth", type=int, default=32)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(root / "build")
    import torch
    import torch_npu
    import pypto.torch
    from pypto.runtime.kernel.context import get_process_kernel_state
    from pypto.torch.launch import enqueue

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
    entry = write_graph(root / "graph.py", args.depth)
    registered = pypto.torch.register(entry, "jit_host_lookup_repro::add")
    x = torch.ones((32, 32), dtype=torch.float32, device=f"npu:{args.device}")
    y = torch.empty_like(x)
    for _ in range(3):
        registered(x, y)
        torch.npu.synchronize()
    state = get_process_kernel_state()
    bound = state.require_config()
    registrations = [state.find_specialization(k, bound) for k in state._specializations if k[0] is entry]
    assert len(registrations) == 1
    registration = registrations[0]

    def ordinary():
        registered(x, y)

    def prepared():
        enqueue(registration, (x, y))

    result = {"depth": args.depth, "warmup_calls": 3, "aclgraph_replay": False,
              "sdk_modified": False, "timing": {}}
    for name, run in (("registered", ordinary), ("prepared_diagnostic", prepared)):
        samples = []
        for _ in range(15):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            run()
            end_submit = time.perf_counter_ns()
            torch.npu.synchronize()
            end = time.perf_counter_ns()
            samples.append({"submit_us": (end_submit - start) / 1000, "completion_us": (end - start) / 1000})
        assert torch.equal(y.cpu(), torch.full((32, 32), 1.0 + args.depth))
        profile = cProfile.Profile()
        for _ in range(3):
            profile.runcall(run)
            torch.npu.synchronize()
        stats = [{"file": file, "line": line, "function": fn, "calls": nc,
                  "own_us": tt * 1e6, "cumulative_us": ct * 1e6}
                 for (file, line, fn), (cc, nc, tt, ct, callers) in pstats.Stats(profile).stats.items()]
        result["timing"][name] = {
            "samples": samples, "submit_median_us": statistics.median(v["submit_us"] for v in samples),
            "completion_median_us": statistics.median(v["completion_us"] for v in samples),
            "cprofile_calls": 3, "cprofile": sorted(stats, key=lambda v: v["cumulative_us"], reverse=True),
        }
    # 相同registration仍然接受动态T变化和新Tensor；不依赖旧地址/旧Tensor对象缓存。
    for rows in [16, 32]:
        fresh = torch.full((rows, 32), 2.0, dtype=torch.float32, device=x.device)
        target = torch.empty_like(fresh)
        enqueue(registration, (fresh, target))
        torch.npu.synchronize()
        assert torch.equal(target.cpu(), torch.full((rows, 32), 2.0 + args.depth))
    try:
        enqueue(registration, (x,))
    except (ValueError, TypeError) as exc:
        result["missing_argument_rejected"] = str(exc)
    else:
        raise AssertionError("参数校验被绕过")
    result["passed"] = True
    result["dynamic_T_new_buffers_pass"] = True
    (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: {a: v[a] for a in ("submit_median_us", "completion_median_us")}
                      for k, v in result["timing"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
