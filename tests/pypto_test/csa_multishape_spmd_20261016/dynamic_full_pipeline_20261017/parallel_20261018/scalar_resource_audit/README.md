# 现有 d253 产物的 Scalar 寄存器、栈和 spill 证据边界

本轮只读已有 `hc_active_notifications_h131072_b16_v1` 完整作业中的 d253
融合产物 `decode_csa_spmd_3kuwaqat` 与同作业原分 task
`_decode_csa_tp1_layer_uy0bge_t`。未重新编译、上卡、复制大文件或计算 hash，
没有改动任何已冻结候选。

**结论：当前保存的产物不能证明融合引入了 Scalar spill 或栈开销；没有足够
证据提出新的寄存器优化候选。未知不等于没有 spill，更不能把未知值填成零。**

## 可直接取得什么

| 项目 | 融合 | 同作业原分 task |
| --- | --- | --- |
| kernel 数 | 2 | 73 |
| 保存形式 | raw `kernel_*.bin` | raw `kernel_*.bin` |
| 保留 `.o/.elf/.s/.ll/.su` | 无 | 无 |
| Scalar 寄存器计数 | 未提供 | 未提供 |
| stack / spill / local scratch 字节 | 未提供 | 未提供 |
| 函数符号表和逐函数 `.text` 区间 | 未保留 | 未保留 |

两份 `binary_manifest.json` 仅提供 kernel 的名称、func_id、签名和二进制描述；
`compiled_meta.json` 提供运行 ABI、入参、平台等，不含寄存器/栈资源字段。
两边保存的 `report/perf_hints.log` 也没有 stack/spill/register/local scratch
统计。这不是完整 CCEC 寄存器分配报告，不能从 PyPTO 的 UB/L1 Tile 分配量
推算 Scalar spill。

可以记录 raw kernel 文件字节数：融合 AIC 50200 B、AIV 145600 B；原分 task
的 Attention AIC（func_id 56）2272 B、AIV（func_id 57）10548 B。前者是完整
CSA，后者只有 Attention，且 raw image 没有保留可用于拆解融合内各阶段的
符号表。上述文件大小不是单个阶段的精确函数 text 大小，也不能用来证明
I-cache 失速、寄存器溢出或由此产生的时延；本轮不重复历史体积结论。

## 为什么本轮不继续猜机器码

既有 `A3_FFTS_WAIT_TIMING_READONLY.md` 的同日机器码审查已记录：官方工具不能
直接识别 raw `.bin`；仅包 ELF 容器后，A3 目标仍只给出 `<not available>`。
已有真实 A3 对象的对照也未输出指令助记符。该边界仍不足以辨认栈指针调整、
spill load/store 或物理寄存器使用，本轮没有重复包装和反汇编尝试。

C++ 中数千个临时 Scalar 名、宽参数列表或较大的完整函数，都不能替代物理
寄存器和 spill 指令证据。没有现场测量、机器码助记符和编译资源统计，就
不能把它们升格为已发现的性能问题。

## 收口和复现

`read_existing_metadata.py` 仅读现有元数据、日志、二进制头四字节及文件大小；
结果保存于 `evidence.json`，包括所有 75 个 kernel 的路径/大小、元数据键、
资源字段搜索范围和工具边界引用。不保存 manifest 中的 hash 值，不重新计算。

```bash
python read_existing_metadata.py
```

要进一步判断只能在将来另有授权的正常编译中保留编译器资源报告或可解码
原始对象；本次没有为取这些信息增加编译或设备任务，也不建议据现有空缺改代码。
