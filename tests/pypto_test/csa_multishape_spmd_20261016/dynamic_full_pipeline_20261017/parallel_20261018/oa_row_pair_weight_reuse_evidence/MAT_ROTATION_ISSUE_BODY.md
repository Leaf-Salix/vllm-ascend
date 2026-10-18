## 问题范围

在实现单SPMD的L1预取时，希望current/next两份Mat数据轮转：先加载下一块，
再计算当前块，然后让current引用下一块。当前PYPTO内存规划路径将if/loop
值合并物化成Mat→Mat TMOV，A2/A3后端正确拒绝这类搬运。

这是PyPTO的值/缓冲轮转表达与lowering限制，不应通过让PTOAS接受不支持的
硬件指令解决。与#3016不同：该问题是原本同一窗口的条件gather；本例的
current与next在预取时必须是**不同且同时存活**的窗口，不能强制合并存储。

## 最小完整CPU复现

无需模型或NPU。在已构建PyPTO环境保存下面文件，运行
`python mat_prefetch_rotation_repro.py --output /tmp/unique_mat_rotation_output`。

```python
# SPDX-License-Identifier: Apache-2.0
"""CPU最小复现：Mat current/next跨if与loop时不应产生目标不支持的Mat复制。

该脚本不使用权重或设备，不作为CSA数值/性能测试。保存全编译结果和失败
日志，用于区分#3016的同窗口phi与这里不同存储的预取轮转需求。
"""

import argparse
import json
import traceback
from pathlib import Path

import pypto.language as pl


@pl.jit.incore
def rotate(a: pl.Tensor[[512, 64], pl.FP16], b: pl.Tensor[[64, 16], pl.FP16],
           out: pl.Out[pl.Tensor[[512, 16], pl.FP32]]):
    current = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
    weight = pl.load(b, [0, 0], [64, 16], target_memory=pl.MemorySpace.Mat)
    right = pl.tile.move(weight, target_memory=pl.MemorySpace.Right)
    for block in pl.range(8):
        if block + 1 < 8:
            following = pl.load(a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        else:
            following = current
        left = pl.tile.move(current, target_memory=pl.MemorySpace.Left)
        result = pl.tile.matmul(left, right)
        pl.store(result, [block * 64, 0], out)
        current = following
    return out


@pl.jit
def entry(a: pl.Tensor[[512, 64], pl.FP16], b: pl.Tensor[[64, 16], pl.FP16],
          out: pl.Out[pl.Tensor[[512, 16], pl.FP32]]):
    with pl.spmd(1):
        out = rotate(a, b, out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from pypto import passes
    from pypto.runtime import CompileOptions

    report = dict(passed=False, device_execution=False)
    try:
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                                 output_dir=str(args.output / "kernel"), dump_passes=True)
        artifact = entry._resolve_kernel_artifact((), {"config": options}, allow_signature_mode=True)
        report.update(passed=True, artifact=str(artifact.output_dir))
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

```

## 实际结果与证据

完整CSA候选和上述最小例均在PYPTO planner上复现，PTOAS错误为：

```text
'pto.tmov' op expects a supported tmov address-space pair for this target
```

最小例只需两份64×64 FP16的L1数据，无资源容量压力。生成IR有：

```text
following__ssa_v1       : Mat[64,64] FP16 @0
following__ssa_v1_mv    : Mat[64,64] FP16 @8192
pto.tmov following__ssa_v1 -> following__ssa_v1_mv
```

循环回边另有current对应的Mat→Mat copy。PTOAS拒绝前已由PyPTO发出这些
不同地址的显式copy，不能仅删指令而让后继读取尚未写过的窗口。

复现环境为A2/A3、PTOAS 0.66，本地独立PyPTO树d01de8523，包含此前CSA
融合所需的本地修复（含别名与内存复用修改）。**尚未宣称干净main也完成了
本例的二进制复现**；源/生成IR与最小例用于组件接手定位。没有设备执行，
也没有把编译失败当数值错误或性能结论。

## 建议方向

1. 保留current/next的真实活跃区间和两份L1容量；考虑将循环值轮转lower为
   buffer handle/address轮转，而不是复制整个Mat内容。
2. 对可证明的二槽顺序循环，可使用固定双bank与`block % 2`选择地址，再
   在消费后复用。仅是建议算法，不是已验证补丁；需保留MTE1读取完成到
   下一次MTE2覆盖的依赖，不能为实现预取删同步。
3. 若当前IR暂不能表达，应在PyPTO明确报告Mat phi/loop需要不可用复制，
   给出支持的双缓冲写法，避免最终才在PTOAS报通用TMOV空间错误。
4. 后续测试应覆盖最后一轮不再预取的else别名、偶/奇轮数、两个Mat参数
   同时轮转、真实设备数值和双槽覆盖保护；CPU不能替代这些验证。

相关：#2693的Buffer IR、#3016的同窗口phi；#2902是AutoTileMatmulL0对
已有Mat的额外搬运，本例使用显式Left/Right，未走该触发方式。

本次只提交issue和建议，不提交PR、不请求标签。完整算子另沿普通range的
合法路径继续性能对照，未修改共享安装或放宽平台容量。
