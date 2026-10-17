# 主Q双N128列块驻留L1

2026-10-17，128K/B16/S6，`qproj_l1_columns.py`；默认关闭。

## 与既有预取候选的区别

原主Q每个worker处理一组N256列，K1024分四次K256累加。
此前显式L0预取候选用N256/K128，完整L1为830.62μs，未采用。
本候选保留原20worker和每worker的原列归属，把每组N256拆成两个N128，
各自仍按0/256/512/768的K顺序累加。两个累加链交错，
第一列的矩阵计算提交后，准备下一K块，再提交第二列的计算。
QR每worker入口加载一次；两份1024×128权重每列组各加载一次到L1。

INT8乘积和INT32累加不溢出，绝对值上界16777216；原输出Tensor、权重NZ布局、
20worker以及所有32768列的唯一归属保持。没有Host读取设备数据，也没有额外GM布局转换。
这里只先验证T96，其他档位需要重新确定M及内存容量，不直接套用固定M96。

## CPU分配证据

`qproj_l1_columns_prepare_v1`与`qproj_l1_columns_cpu_v1`通过。
实际Left在0/24576，Right在0/32768，两个Acc在0/49152，成对不重叠。
L1两份权重在65536/196608，QR在327680，最大端点425984字节。
每个原N256列组为两次matmul初始化及六次matmul_acc。
`qproj_l1_columns_codegen_audit.json`记录实际地址和操作，
`qproj_l1_columns_generated_stage.cpp.txt`保存整个stage5。
源级存活容量与实际地址均不代替流水重叠或性能测量。

## 首轮完整CSA

任务`task_20261017_083715_36883533451` exit0；输出/Top-K/六份cache/state、
eager/graph、30代信号、保护区全PASS。正式权重，未执行Native forward。
完整无插桩L1为809.22/806.46/797.56μs，中位806.46μs；
同轮原动态PTO中位903.86μs。Event中位单SPMD912.24、动态951.82μs，分别报告。

现选历史首轮807.16μs、同进程复核817.88μs。首轮数字接近且各窗有变化，
不能拿不同运行的最小值或中位数直接宣布收益。

## 同进程对照完成：未采用

`--paired-qproj-spmd-only`复用配对测试助手，对照端明确启用现选row_dependencies=True、
QK地址缓存和100tick退避；新候选只有主Q不同。两边相同根Tensor地址、独立工作区和信号，
每侧2次预热、3窗L1，AB/BA交替；不重复Event或动态metadata验证。
任务`task_20261017_084154_377975115763`，目录`b16_qproj_l1_columns_paired_v1`。
该任务exit0，输出/Top-K/六状态及双方工作区、7代信号全PASS。
对照端生成源码AST与现选完全一致，见`qproj_paired_reference_source_audit.json`。

| 版本 | 三窗L1（μs） | 中位数（μs） |
|---|---|---:|
| 现选单SPMD | 805.28 / 808.92 / 810.24 | 808.92 |
| 主Q双列候选 | 806.02 / 804.06 / 810.32 | 806.02 |

中位差-2.90μs（-0.36%），均值差-1.35μs，范围明显重叠，
未证明足以替换现选的稳定收益。未采用、不扩七档、不重复该比较。
`selected_shape_policies.json`保持原选择。结果`qproj_l1_columns_paired_result.json`。

所有重放在调用前恢复fixture，不能据此声称连续decode轨迹已通过。
