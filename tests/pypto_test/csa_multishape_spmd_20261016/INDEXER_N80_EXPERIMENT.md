# B16/S6 Indexer N80 候选：精度通过，性能不采用

2026-10-17。文件 `indexer_n80.py`，显式开关 `--indexer-n80`。
只在实验目录支持长历史 B16，与已选的 QK 地址缓存和行依赖细化组合。
没有修改生产入口、冻结数学源文件、PyPTO、Simpler 或 PTO-ISA。

## 改动与预期

原 Indexer 每 1024 个候选分为 16 个 N64 panel。本候选采用 13 个 N80 panel，
末 panel 只存 64 列，QK/WS 矩阵调用总数 32→26。
保持 QK M384/K128、六个 query、WS M16/K384、FIX 量化/ReLU 和原归约顺序。
叶划分、1024 列 GM 分数块、双缓冲、FFTS 发布/确认及排空、Top-K AIV 全部保留。

Native Key 页每 32 行附带 scale。80 列 panel 分成 32/32/16 或 16/32/32 行加载，
不能直接跨页把 scale 字节当作 Key。每 1024 列的加载段从 32 增至 39；
尾部多计算的 16 列不写入 GM，也不改变有效候选数。

N80 的 WS Right 占 60 KiB，Key Right 占 10 KiB，不能同时驻留 64 KiB L0B。
因此仅将下一个 Key 预取到 L1，当前 QK 消费 Key 后，才覆盖 Right 放入 WS 数据。
调用次数减少与预取重叠减少同时发生，不能只凭前者断言提速。

## CPU 与实际生成代码

`indexer_n80_prepare_v1/` 完成分档准备、依赖与协议检查。
`indexer_n80_cpu_v1/` 完成 CPU 全流程代码生成；这不代表设备执行或数值通过。

实际 AIC 代码中：

- 13 个 QK Acc 均在 0 地址，占 122880 B；前 12 个 WS Acc 位于 122880，占 5120 B。
- 最后一个 WS 在 QK FIX 已消费结果后复用 Acc 0 地址，峰值 128000 B，小于 131072 B。
- WS Right 与 Key Right 同在 0 地址，M→MTE1 的通知/等待保护复用。
- CPU 与设备实际生成代码的 52 个 N80 Acc/Right Tile 类型、形状、地址一致。

证据：`indexer_n80_codegen_audit.json`、`indexer_n80_codegen_fragment.cpp`、
`indexer_n80_device_codegen_comparison.json`。没有做 hash 校验。
初版地址提取脚本误合并不同函数的重名常量、误要求末 WS 使用 122880，已修正；
全文阶段提取又遇到被优化掉的 SSA 名称，改为直接读取实际 Tile 与 TASSIGN。
这些是审计脚本错误，不是设备算子失败；没有宣称完整 C++ 阶段逐字一致。

## 设备结果与结论

任务 `task_20261017_050914_1722359619`，结果 `b16_row_deps_indexer_n80_v1/result.json`，exit=0。
正式第 4 层 W8A8 权重，两侧均为 PTO，未执行 Native forward。
输出、Top-K、六份 state/cache、eager/graph/最终结果逐 bit 一致；
30 代信号、只读区、未写区及工作区保护区通过。

| 完整无插桩 L1 incore 首尾 | 三窗（μs） | 中位数（μs） |
| --- | --- | --- |
| 原动态 PTO | 893.40 / 895.08 / 919.26 | 895.08 |
| 行依赖 + N80 单 SPMD | 879.10 / 887.62 / 878.82 | 879.10 |

图外 Event 中位数为动态 956.57 μs、候选 981.13 μs，不混入 L1 目标。
N80 虽比同轮原动态 L1 略快，但明显慢于此前已选行依赖单 SPMD 的两轮 807.16/817.88 μs。
**不采用 N80，不扩测其他档位，不因其减少矩阵调用而放宽验收。**

生成代码证明预取重叠减少和页段增加，尚未单独测出两者各贡献多少微秒；
本轮没有为负候选追加阶段探针或动态长度测试。代码和失败尝试完整保留。
