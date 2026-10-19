# SPDX-License-Identifier: Apache-2.0
"""验证迁入的 SDMA 能在一个 MIX 中由一个 AIV 发起，48 条 AIV 角色计算并跨图重放复用。

24 条 AIC 同时计算独立的单位阵乘法；一个 AIV 预取自己的连续区域，其余 AIV 保持普通读取，
等待真实事件后写入结果及完成标记；每条角色独占64字节标记缓存行。这里只验证调用链和不干扰计算，
不把复制正确或事件完成当作 L2 命中率或性能收益的证明。
"""

import argparse
import json
import os
from pathlib import Path

import pypto.language as pl


@pl.jit.incore
def cube_copy(a: pl.Tensor[[384, 16], pl.BF16], b: pl.Tensor[[16, 16], pl.BF16],
              out: pl.InOut[pl.Tensor[[384, 16], pl.FP32]]):
    worker = pl.tile.get_block_idx()
    left = pl.load(a, [worker * 16, 0], [16, 16])
    right = pl.load(b, [0, 0], [16, 16])
    value = pl.matmul(left, right, out_dtype=pl.FP32)
    out = pl.store(value, [worker * 16, 0], out)
    return out


@pl.jit.incore
def vector_prefetch(x: pl.Tensor[[48, 1024], pl.FP32],
                    out: pl.InOut[pl.Tensor[[48, 1024], pl.FP32]],
                    done: pl.InOut[pl.Tensor[[768], pl.INT32]]):
    pl.func_attr({"split_aiv": True})
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    complete = pl.const(True, pl.BOOL)
    if worker == 32:
        region = pl.tensor.slice(x, [1, 1024], [worker, 0], gm_view=True)
        ctx = pl.prefetch.make_context()
        event = pl.prefetch.async_prefetch(region, ctx)
        session = pl.prefetch.session(ctx)
        complete = pl.prefetch.wait(event, session)
    if complete:
        value = pl.load(x, [worker, 0], [1, 1024])
        out = pl.store(pl.add(value, 1.0), [worker, 0], out)
        pl.write(done, [worker * 16], pl.const(1, pl.INT32))
        pl.system.cacheinvalid(done, [1], [worker * 16])
        pl.system.fence()
    return out, done


@pl.jit.inline(auto_scope=False)
def mixed(a: pl.Tensor, b: pl.Tensor, x: pl.Tensor,
          cube_out: pl.InOut[pl.Tensor], vector_out: pl.InOut[pl.Tensor],
          done: pl.InOut[pl.Tensor]):
    with pl.cluster(name_hint="sdma_mix_smoke"):
        cube_out = cube_copy(a, b, cube_out)
        vector_out, done = vector_prefetch(x, vector_out, done)
    return cube_out, vector_out, done


@pl.jit
def entry(a: pl.Tensor[[384, 16], pl.BF16], b: pl.Tensor[[16, 16], pl.BF16],
          x: pl.Tensor[[48, 1024], pl.FP32],
          cube_out: pl.InOut[pl.Tensor[[384, 16], pl.FP32]],
          vector_out: pl.InOut[pl.Tensor[[48, 1024], pl.FP32]],
          done: pl.InOut[pl.Tensor[[768], pl.INT32]]):
    with pl.spmd(24, sync_start=True):
        cube_out, vector_out, done = mixed(a, b, x, cube_out, vector_out, done)
    return cube_out, vector_out, done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", choices=("tensormap_and_ringbuffer", "host_build_graph"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--physical-ids", action="store_true")
    args = p.parse_args()
    if not args.compile_only and "TASK_DEVICE" not in os.environ:
        raise RuntimeError("必须由队列分配 NPU")
    if not args.compile_only and args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("设备参数必须与队列分配一致")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(args.output / "build")
    os.environ["PYPTO_CACHE_DIR"] = str(args.output / "cache")
    if args.physical_ids:
        # 仅诊断产物插入原生硬件编号读回，不用于性能或正式CSA采用。
        from simpler_setup.kernel_compiler import KernelCompiler

        original_compile = KernelCompiler.compile_incore

        def compile_with_ids(self, source_path, *pos, **kw):
            path = Path(source_path)
            if path.name == "vector_prefetch.cpp":
                text = path.read_text()
                marker = "    // Forward to ptoas-generated function"
                probe = """    int __diag_slot = (__pypto_spmd_block_idx * 2 + __pypto_spmd_subblock_idx) * 16;
    done__ssa_v0[__diag_slot + 1] = static_cast<int32_t>(get_block_idx());
    done__ssa_v0[__diag_slot + 2] = static_cast<int32_t>(get_subblockid());
    dcci((__gm__ void*)(done__ssa_v0 + __diag_slot), cache_line_t::SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
"""
                if "__diag_slot" not in text:
                    assert marker in text
                    path.write_text(text.replace(marker, probe + marker))
            return original_compile(self, source_path, *pos, **kw)

        KernelCompiler.compile_incore = compile_with_ids
    if args.compile_only:
        from pypto import passes
        from pypto.runtime import CompileOptions

        runtime = (passes.RuntimeKind.HOST_BUILD_GRAPH if args.runtime == "host_build_graph"
                   else passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER)
        with passes.PassContext([], runtime=runtime, memory_planner=passes.MemoryPlanner.PYPTO):
            artifact = entry._resolve_kernel_artifact(
                (), {"config": CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"))},
                allow_signature_mode=True,
            )
        artifact.load()
        assert artifact.requires_sdma
        orchestration = "\n".join(p.read_text() for p in (args.output / "kernel/orchestration").glob("*.cpp"))
        assert "MixedKernels mixed_0 = {0, 1, 1}" in orchestration
        report = {"passed": True, "scope": "CPU lowering and native assembly only",
                  "runtime": args.runtime, "requires_sdma": True, "device_execution": False}
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    import pypto
    import pypto.torch
    import torch
    import torch_npu
    from pypto.runtime import CompileOptions

    device = int(os.environ["TASK_DEVICE"])
    torch_npu.npu.set_device(device)
    pypto.torch.init(device=device, runtime=args.runtime, enable_sdma=True,
                     ring_task_window=64, ring_heap=1 << 20, ring_dep_pool=1024)
    pypto.configure_cache(pypto.CacheConfig(enabled=False))
    a_cpu = (torch.arange(384 * 16).reshape(384, 16) % 13).to(torch.bfloat16)
    x_cpu = (torch.arange(48 * 1024).reshape(48, 1024) % 251).float()
    a = a_cpu.to(f"npu:{device}")
    b = torch.eye(16, dtype=torch.bfloat16).to(a.device)
    x = x_cpu.to(a.device)
    cube_out = torch.full((384, 16), -999.0, dtype=torch.float32, device=a.device)
    vector_out = torch.full_like(x, -999)
    done = torch.zeros(768, dtype=torch.int32, device=a.device)
    values = (a, b, x, cube_out, vector_out, done)
    config = CompileOptions()
    entry(*values, config=config)
    torch.npu.synchronize()

    def verify():
        actual_cube = cube_out.cpu()
        actual_vector = vector_out.cpu()
        actual_done = done.cpu()
        diagnostic = {
            "cube_equal": torch.equal(actual_cube, a_cpu.float()),
            "vector_row_mismatches": (actual_vector != x_cpu + 1).sum(dim=1).tolist(),
            "vector_row_first": actual_vector[:, :4].tolist(),
            "done": actual_done[::16].tolist(),
            "raw_block": actual_done[1::16].tolist(),
            "raw_subblock": actual_done[2::16].tolist(),
        }
        (args.output / "last_verification.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        assert diagnostic["cube_equal"]
        assert torch.equal(actual_vector, x_cpu + 1)
        expected_done = torch.zeros(768, dtype=torch.int32)
        expected_done[::16] = 1
        if args.physical_ids:
            expected_done[1::16] = actual_done[1::16]
            expected_done[2::16] = actual_done[2::16]
        assert torch.equal(actual_done, expected_done)

    verify()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        entry(*values, config=config)
    checks = []
    for step in range(10):
        a_cpu = ((torch.arange(384 * 16).reshape(384, 16) + step * 3) % 13).to(torch.bfloat16)
        x_cpu = ((torch.arange(48 * 1024).reshape(48, 1024) + step * 17) % 251).float()
        a.copy_(a_cpu)
        x.copy_(x_cpu)
        cube_out.fill_(-999)
        vector_out.fill_(-999)
        done.zero_()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            graph.replay()
        stream.synchronize()
        verify()
        checks.append({"step": step, "aic_roles": 24, "aiv_roles": 48, "passed": True})
    report = {"passed": True, "runtime": args.runtime, "enable_sdma": True,
              "eager": True, "prefetch_roles": 1, "graph_replays": checks,
              "pypto": pypto.__file__, "performance_measured": False}
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
