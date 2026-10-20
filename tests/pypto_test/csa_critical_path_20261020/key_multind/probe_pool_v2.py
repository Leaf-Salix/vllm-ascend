# SPDX-License-Identifier: Apache-2.0
"""连续物理页的零拷贝三维描述符及 ND→NZ 搬运验证。

这里只验证准备接入正式动态 CSA 的搬运能力，不计完整 CSA 性能：Native 页
包含 4096B Key 和至少 64B scale。根入参保持Native连续缓存，核内建立共享storage的[P,32,128]正stride
描述符，不复制Key，不读取Host页表值。incore从device Tensor读取
起始物理页，把两页 Key 一次载入 Mat，保持原顺序，再以 INT8 点积检查所有
元素。两页是 N64 面板的数学容量，不是 batch/history 特化。页号变化通过
同一固定地址 Tensor 更新；图复放检查同一个已编译入口消费最新页号。

CPU 模式核对符号编译与实际生成；设备模式覆盖奇偶物理页、最后合法两页、
不同 Native 页 stride、输入完整保护及输出两端 poison。性能收益必须随后
由完整 CSA 同轮五窗 L1 验证，本探针 PASS 不能代替它。
"""

import argparse
import json
import os
import traceback
from pathlib import Path

import pypto.language as pl

PAGES = pl.dynamic("KEY_PROBE_PAGES")
PAGE_BYTES = pl.dynamic("KEY_PROBE_PAGE_BYTES")


@pl.jit.incore
def page_pair_dot(
    native_keys: pl.Tensor[[PAGES, PAGE_BYTES], pl.INT8],
    query: pl.Tensor[[16, 128], pl.INT8],
    page_id: pl.Tensor[[1], pl.INT32],
    output: pl.InOut[pl.Tensor[[16, 96], pl.INT32]],
):
    page = pl.cast(pl.read(page_id, [0]), pl.INDEX)
    # Native scale 所在的页尾由 descriptor stride 跳过，不能当成 Key 读入。
    # 根入参仍是Native连续缓存。只在核内建立带页尾空隙的GM视图；
    # 页宽/容量由测试fixture或正式binding保证，不做host读取页表或数据复制。
    native_pages = pl.tensor.dim(native_keys, 0)
    native_pitch = pl.tensor.dim(native_keys, 1)
    keys_view = pl.tensor.view(
        native_keys, [native_pages, 32, 128], strides=[native_pitch, 128, 1],
    )
    keys_3d = pl.load(keys_view, [page, 0, 0], [2, 32, 128], target_memory=pl.MemorySpace.Mat)
    keys = pl.tile.reshape(keys_3d, [64, 128])
    # 完整CSA把N64放进两个Mat槽；这里同步验证这个实际目的布局，避免
    # 只验证独立Tile却在接入时才发现Mat→Mat复制或父Tile行距问题。
    key_pool = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.MemorySpace.Mat)
    slot = page % 2 * 64
    key_pool = pl.tile.assemble(key_pool, keys, [slot, 0])
    key_right = pl.tile.extract(
        pl.tile.transpose_view(key_pool), 0, slot, [128, 64], target_memory=pl.MemorySpace.Right,
    )
    q = pl.load(query, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
    result = pl.matmul(q, key_right, out_dtype=pl.INT32)
    pl.store(result, [0, 16], output)
    return output


@pl.jit
def entry(
    native_keys: pl.Tensor[[PAGES, PAGE_BYTES], pl.INT8],
    query: pl.Tensor[[16, 128], pl.INT8],
    page_id: pl.Tensor[[1], pl.INT32],
    output: pl.InOut[pl.Tensor[[16, 96], pl.INT32]],
):
    with pl.spmd(1):
        output = page_pair_dot(native_keys, query, page_id, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--device", type=int, help="由任务队列注入；必须匹配 TASK_DEVICE")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import pypto
    from pypto import passes, pypto_core
    from pypto.runtime import CompileOptions, ensure_pto_isa_root

    report = dict(passed=False, scope="连续两页Key的搬运与descriptor验证", cpu_only=args.cpu_only,
                  pypto=pypto.__file__, core=pypto_core.__file__, full_csa_validated=False,
                  performance_measured=False, cases=[])
    config = CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"),
                            dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO)
    try:
        report["pto_isa_root"] = ensure_pto_isa_root()
        if args.cpu_only:
            artifact = entry._resolve_kernel_artifact((), {"config": config}, allow_signature_mode=True)
            report["artifact"] = str(artifact.output_dir)
        else:
            import pypto.torch
            import torch
            import torch_npu  # noqa: F401

            device_id = int(os.environ["TASK_DEVICE"])
            if args.device is not None:
                assert args.device == device_id, "队列设备与环境变量不一致"
            torch.npu.set_device(device_id)
            pypto.torch.init(device=device_id, platform="a2a3", runtime="tensormap_and_ringbuffer")
            device = torch.device(f"npu:{device_id}")
            gen = torch.Generator().manual_seed(20261018)
            q_cpu = torch.randint(-127, 128, (16, 128), dtype=torch.int8, generator=gen)
            q = q_cpu.to(device)
            page = torch.zeros((1,), dtype=torch.int32, device=device)
            out = torch.full((16, 96), -7777777, dtype=torch.int32, device=device)
            for page_bytes in (4160, 4224, 8192):
                raw_cpu = torch.randint(-127, 128, (9, page_bytes), dtype=torch.int8, generator=gen)
                raw = raw_cpu.to(device)
                keys = raw
                assert keys.data_ptr() == raw.data_ptr() and keys.is_contiguous()
                assert 4160 <= page_bytes <= 65535 and page_bytes % 64 == 0
                entry(keys, q, page, out, config=config)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    entry(keys, q, page, out, config=config)
                for first in (0, 1, 7, 0):
                    page.fill_(first)
                    out.fill_(-7777777)
                    graph.replay()
                    torch.npu.synchronize()
                    expected = q_cpu.int() @ raw_cpu[first:first + 2, :4096].reshape(64, 128).int().T
                    actual = out.cpu()
                    torch.testing.assert_close(actual[:, 16:80], expected, rtol=0, atol=0)
                    assert torch.all(actual[:, :16] == -7777777) and torch.all(actual[:, 80:] == -7777777)
                    assert torch.equal(raw.cpu(), raw_cpu) and torch.equal(q.cpu(), q_cpu)
                    report["cases"].append(dict(page_bytes=page_bytes, first_page=first,
                                                exact=True, guards=True, input_unchanged=True))
        report["passed"] = True
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
