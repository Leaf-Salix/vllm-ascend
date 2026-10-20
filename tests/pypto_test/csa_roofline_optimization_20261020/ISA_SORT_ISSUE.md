### 问题

CPU-SIM 的 `TSORT32` 在 score 相等时按输入 payload index 的数值升序排序；
A3 Sort32 的规则是保留原输入位置先后。payload index 不一定等于物理输入位置，
因此逆序/任意 payload 时 CPU 模拟与 A3 行为不同。

本问题在精确 Top-K 保序筛选诊断中发现。CSA 当前真实 index 单调，本次并未
由此发现 CSA 正式输出错误；但不能把此 CPU 模型当成任意 payload 的硬件 oracle。

### CPU 最小复现（不需要 NPU）

```cpp
// SPDX-License-Identifier: Apache-2.0
// 与 A3 probe 的逆序 payload 场景对应，核对 CPU-SIM 真实 TSORT32 实现。
#include <cstdint>
#include <pto/common/cpu_stub.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/cpu/TAssign.hpp>
#include <pto/cpu/TSort32.hpp>
#include <cstdio>
#include <cstring>

int main() {
  using namespace pto;
  using Values = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor>;
  using Indices = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor>;
  using Pairs = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor>;
  NPU_MEMORY_INIT();
  Values src;
  Indices idx;
  Pairs dst;
  TASSIGN_IMPL(src, 0);
  TASSIGN_IMPL(idx, 128);
  TASSIGN_IMPL(dst, 256);
  for (int j = 0; j < 32; ++j) {
    src.data()[j] = 1.0f;
    idx.data()[j] = 31 - j;
  }
  TSORT32_IMPL(dst, src, idx);
  std::printf("{\"cpu_indices\":[");
  for (int j = 0; j < 32; ++j) {
    uint32_t payload = 0;
    std::memcpy(&payload, &dst.data()[2 * j + 1], sizeof(payload));
    std::printf("%s%u", j ? "," : "", payload);
  }
  std::printf("]}\n");
}

```

```bash
g++-15 -std=c++20 -D__CPU_SIM -I /path/to/pto-isa/include repro.cpp -o repro
./repro
```

实测输出：

```json
{"cpu_indices":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]}

```

若保留原输入位置，32 个相同 score 配上 payload `[31,30,...,0]` 的输出
也应保持 `[31,30,...,0]`。

### A3 设备证据

同一 ISA 版本，Ascend910_9392 / CANN 9.2.0-beta.2。实际 PyPTO probe
输入 2048 个 score=1、payload `[2047,2046,...,0]`，运行 Sort32 后依次
MrgSort(block_len=64/256/1024)。设备输出的前 32 个 payload 为
`[2047,2046,...,2016]`，保持了首个 32 元素输入块的原位置顺序。
各级 merge 不会重排已经有序的输入队列内部，故不能用 payload 升序解释。
任务 `task_20261020_172756_41138729719` 已 completed/exit0；全部46组
阈值压缩排序语义场景完成，逆序 payload 属于其中一个专用诊断项。
没有宣称已重新运行其它芯片或所有 CPU-SIM 算子。

### 根因与建议

`include/pto/cpu/TSort32.hpp` 使用 `std::stable_sort`，但 comparator 在同分
时返回 `a.index < b.index`，这会主动改变 payload 的顺序，抵消 stable_sort。
有限 FP32 输入的建议为只比较 score：

```cpp
std::stable_sort(segment.begin(), segment.end(),
  [](const ScoreIndexPair<T>& a, const ScoreIndexPair<T>& b) {
    return a.score > b.score;
  });
```

`ScoreIndexPair::operator<`、注释/文档中“索引小优先”的说法也应区分原位置
和用户提供的 payload。建议回归正序、逆序、乱序及重复 payload，在相同
分数跨32元素块的情况下再验证 MrgSort。NaN 等特殊值语义另按 ISA 合同处理，
本文仅实测有限 FP32。

### 版本

PTO-ISA `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`。2026-10-20 只读查看 GitHub main，仍存在上述 comparator。
PyPTO `2dfa7f120ba5e577566046d920845aa241bfa7bb`，PTOAS
`e5c1c432af8249bac2f905c7648ebafefec3c20a`。未修改 ISA、未提 PR、不加标签。
