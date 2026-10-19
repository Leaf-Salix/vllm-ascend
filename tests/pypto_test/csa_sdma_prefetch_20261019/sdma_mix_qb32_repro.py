# SPDX-License-Identifier: Apache-2.0
"""公开 API 的 A3 SDMA MIX 复现，不依赖 GM subview 或 NZ 元数据补丁。

一个 MIX 启动 24 AIC、48 AIV。由设备 mask 选择 SDMA 发起者，其余 AIV
仍计算；预取对象是独立连续 32 MiB INT8 Tensor。显式通道开关仅用于因果
诊断：本例只有一个全卡 MIX，因此逻辑 worker 到通道的一一映射安全，
不能将这种诊断代码当作多 task 并发场景的 runtime 通用修复。
"""

import argparse
import json
import os
import re
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
                    prefetch_src: pl.Tensor[[33554432], pl.INT8],
                    issuer_mask: pl.Tensor[[48], pl.INT32],
                    out: pl.InOut[pl.Tensor[[48, 1024], pl.FP32]],
                    done: pl.InOut[pl.Tensor[[768], pl.INT32]]):
    pl.func_attr({"split_aiv": True})
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    complete = pl.const(1, pl.INT32)
    if pl.read(issuer_mask, [worker]) != 0:
        ctx = pl.prefetch.make_context()
        event = pl.prefetch.async_prefetch(prefetch_src, ctx)
        session = pl.prefetch.session(ctx)
        if pl.prefetch.wait(event, session):
            complete = pl.const(1, pl.INT32)
        else:
            complete = pl.const(0, pl.INT32)
    if complete == 1:
        value = pl.load(x, [worker, 0], [1, 1024])
        out = pl.store(pl.add(value, 1.0), [worker, 0], out)
        pl.write(done, [worker * 16], pl.const(1, pl.INT32))
        pl.system.cacheinvalid(done, [1], [worker * 16])
        pl.system.fence()
    return out, done


@pl.jit.inline(auto_scope=False)
def mixed(a: pl.Tensor, b: pl.Tensor, x: pl.Tensor,
          prefetch_src: pl.Tensor, issuer_mask: pl.Tensor,
          cube_out: pl.InOut[pl.Tensor], vector_out: pl.InOut[pl.Tensor],
          done: pl.InOut[pl.Tensor]):
    with pl.cluster(name_hint="sdma_mix_smoke"):
        cube_out = cube_copy(a, b, cube_out)
        vector_out, done = vector_prefetch(x, prefetch_src, issuer_mask, vector_out, done)
    return cube_out, vector_out, done


@pl.jit
def entry(a: pl.Tensor[[384, 16], pl.BF16], b: pl.Tensor[[16, 16], pl.BF16],
          x: pl.Tensor[[48, 1024], pl.FP32],
          prefetch_src: pl.Tensor[[33554432], pl.INT8],
          issuer_mask: pl.Tensor[[48], pl.INT32],
          cube_out: pl.InOut[pl.Tensor[[384, 16], pl.FP32]],
          vector_out: pl.InOut[pl.Tensor[[48, 1024], pl.FP32]],
          done: pl.InOut[pl.Tensor[[768], pl.INT32]]):
    with pl.spmd(24, sync_start=True):
        cube_out, vector_out, done = mixed(a, b, x, prefetch_src, issuer_mask, cube_out, vector_out, done)
    return cube_out, vector_out, done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", choices=("tensormap_and_ringbuffer", "host_build_graph"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--physical-ids", action="store_true")
    p.add_argument("--eager-only", action="store_true")
    p.add_argument("--issuers", choices=("all", "single", "one-per-group"), default="all")
    p.add_argument("--explicit-channel", action="store_true")
    args = p.parse_args()
    if args.explicit_channel and not args.physical_ids:
        raise ValueError("显式通道诊断必须同时采集事件状态")
    if not args.compile_only and "TASK_DEVICE" not in os.environ:
        raise RuntimeError("必须由队列分配 NPU")
    if not args.compile_only and args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("设备参数必须与队列分配一致")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(args.output / "build")
    os.environ["PYPTO_CACHE_DIR"] = str(args.output / "cache")
    if args.physical_ids:
        # 仅诊断产物插入原生硬件编号读回，不用于性能或正式CSA采用。
        from pypto.runtime.kernel_compiler import KernelCompiler

        original_compile = KernelCompiler.compile_incore

        def compile_with_ids(self, source_path, *pos, **kw):
            path = Path(source_path)
            if path.name == "vector_prefetch.cpp":
                text = path.read_text()
                marker = "    // Forward to ptoas-generated function"
                probe = """    int __diag_slot = (__pypto_spmd_block_idx * 2 + __pypto_spmd_subblock_idx) * 16;
    done__ssa_v0[__diag_slot + 1] = static_cast<int32_t>(get_block_idx());
    done__ssa_v0[__diag_slot + 2] = static_cast<int32_t>(get_subblockid());
    done__ssa_v0[__diag_slot + 7] = __diag_slot / 16 + 1000;
    dcci((__gm__ void*)(done__ssa_v0 + __diag_slot), cache_line_t::SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
"""
                if "__diag_slot" not in text:
                    assert marker in text
                    text = text.replace(marker, probe + marker)
                    signature = re.search(r"void vector_prefetch\([^\n]*__gm__ int32_t\* (\w+),[^\n]*int32_t (\w+), int32_t (\w+), int32_t (\w+)\)", text)
                    assert signature, "诊断必须识别真实AIV函数参数"
                    done_ptr, block, block_count, sub = signature.groups()
                    if args.explicit_channel:
                        def bind_unique_channel(match):
                            ctx, workspace = match.groups()
                            return match.group(0) + f"""
    // 仅单一全卡 MIX 诊断：不能用于一般多任务调度。
    TASSIGN({ctx}.scratchTile, 0);
    pto::comm::sdma::BuildSdmaSession({ctx}.scratchTile, {workspace}, {ctx}.session,
        0U, {{64ULL * 1024ULL * 1024ULL, 0, 1}},
        static_cast<uint32_t>({block} * 2 + {sub}));
"""
                        text, binds = re.subn(r"pto::PrefetchAsyncContext (\w+) = pto::PrefetchAsyncContext\((\w+)\);", bind_unique_channel, text)
                        assert binds == 1, "必须显式绑定唯一的预取context"
                    def add_event_probe(match):
                        complete, event, session = match.groups()
                        return match.group(0) + f"""
+  int __event_slot = ({block} * 2 + {sub}) * 16;
+  {done_ptr}[__event_slot + 3] = static_cast<int32_t>({event}.valid());
+  {done_ptr}[__event_slot + 4] = static_cast<int32_t>({session}.valid);
+  {done_ptr}[__event_slot + 5] = static_cast<int32_t>({complete});
+  {done_ptr}[__event_slot + 6] = static_cast<int32_t>({event}.handle & 0x7fffffffULL);
+  {done_ptr}[__event_slot + 8] = static_cast<int32_t>({session}.channelGroupIdx);
+  dcci((__gm__ void*)({done_ptr} + __event_slot), cache_line_t::SINGLE_CACHE_LINE);
+  dsb(DSB_DDR);
+""".replace("\n+", "\n")
                    text, count = re.subn(r"bool (\w+) = (\w+)\.Wait\((\w+)\);", add_event_probe, text)
                    assert count == 1
                    path.write_text(text)
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
    prefetch_src = torch.zeros(33554432, dtype=torch.int8, device=a.device)
    issuer_mask_cpu = torch.ones(48, dtype=torch.int32)
    if args.issuers == "single":
        issuer_mask_cpu.zero_()
        issuer_mask_cpu[32] = 1
    elif args.issuers == "one-per-group":
        issuer_mask_cpu[1::2] = 0
    issuer_mask = issuer_mask_cpu.to(a.device)
    values = (a, b, x, prefetch_src, issuer_mask, cube_out, vector_out, done)
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
            "instrumentation_marker": actual_done[7::16].tolist(),
            "event_valid": actual_done[3::16].tolist(),
            "session_valid": actual_done[4::16].tolist(),
            "wait_result": actual_done[5::16].tolist(),
            "event_handle_low": actual_done[6::16].tolist(),
            "channel_group": actual_done[8::16].tolist(),
        }
        (args.output / "last_verification.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        if args.physical_ids:
            for index in range(48):
                if issuer_mask_cpu[index] != 0:
                    assert diagnostic["event_valid"][index] == 1
                    assert diagnostic["session_valid"][index] == 1
                    assert diagnostic["wait_result"][index] == 1
        assert diagnostic["cube_equal"]
        assert torch.equal(actual_vector, x_cpu + 1)
        expected_done = torch.zeros(768, dtype=torch.int32)
        expected_done[::16] = 1
        if args.physical_ids:
            expected_done[1::16] = actual_done[1::16]
            expected_done[2::16] = actual_done[2::16]
            expected_done[7::16] = torch.arange(48, dtype=torch.int32) + 1000
            for column in (3, 4, 5, 6, 8):
                expected_done[column::16] = actual_done[column::16]
        assert torch.equal(actual_done, expected_done)

    verify()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    if not args.eager_only:
        with torch.npu.graph(graph, stream=stream):
            entry(*values, config=config)
        stream.synchronize()
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
            if args.eager_only:
                entry(*values, config=config)
            else:
                graph.replay()
        stream.synchronize()
        verify()
        checks.append({"step": step, "aic_roles": 24, "aiv_roles": 48, "passed": True})
    report = {"passed": True, "runtime": args.runtime, "enable_sdma": True, "issuers": args.issuers,
              "explicit_channel_diagnostic": args.explicit_channel,
              "eager": True, "eager_only": args.eager_only, "checked_repetitions": checks,
              "pypto": pypto.__file__, "performance_measured": False}
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
