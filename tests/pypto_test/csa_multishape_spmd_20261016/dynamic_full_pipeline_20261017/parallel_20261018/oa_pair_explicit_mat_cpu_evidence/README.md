# O_A 显式 Mat/Right 生命周期 CPU 探针（2026-10-18）

本目录独立于正式 `66882634`，只使用隔离 PyPTO/Simpler 的只读安装。**v1 CPU 失败并保留；唯一获准的 v2 修正完整 CPU 通过。没有 NPU、设备精度或性能结论；尚未接入完整 CSA。**

## v1：资源达到设计，但 Mat 回边不支持

`pair_explicit.py` 沿用原八组 BF16 NZ 权重根布局、真实输出 stride8192、动态 `valid_rows: Scalar[INDEX]`（1..128）。24 pair 仅覆盖每组前三对 N128；最后两个 singleton 不在本探针范围。不是完整 CSA，不是以 capacity384 固定实际行数。

与旧 `oa_pair_source_v1` 不同：完全去掉 pipeline，将当前/下一组三份 K256 Mat 显式同时存活；K128 内在 MMA 后再创建下一 Right，使任意时刻只有两个总 Right。两个 Acc 每轮都直线更新，没有 v5 的条件性 Acc if-phi。

唯一 v1 lowering 用时约9.12秒。`cpu_v1/command.json` 保留原命令，`cpu_v1/result.json` 和 `compile.log` 保留原错误。真实 `38_after_AllocateMemoryAddr.py` 分配如下：

| 空间 | 实际峰值 | 关键地址 |
|---|---:|---|
| Mat |393216 B（384 KiB）|0/65536/131072/196608/262144/327680，各64 KiB|
| Left |65536 B|0/32768，各32 KiB|
| Right |65536 B|0/32768，各32 KiB|
| Acc |131072 B|0/65536，各64 KiB|

因此 L0 交错没有重现旧 v3/v7 的128 KiB Right问题。但 `37_after_MemoryReuse.py` 为固定回边位置插入三次 Mat→Mat：

- a_next：262144 → 65536，65536 B；
- w0_next：327680 → 131072，65536 B；
- w1_next：0 → 196608，65536 B。

它们是每轮回边的192 KiB拷贝，源代码没有显式写 move。PTOAS 在 `oa_pair_explicit_mat.pto:113/115/117` 的 `pto.tmov` 拒绝 `mat→mat` address-space pair。**没有 AIC kernel C++，所以不能审计最终 load→MMA 的硬件事件，更不能声称预取已经异步重叠。** 此处是当前循环携带接口/后端表达边界，未修改依赖，没有尝试绕过 ISA 检查。

`audit_failure.py` 只读已有 AST/MLIR，生成 `failure_audit.json`，可复现资源和三次不合法移动审计；不触发编译。

## 获准的 v2 定界修正

外层普通 `range(4)`，每轮是固定数学 K1024（不是 B/S 的静态展开）。内部四个 K256：load0，load1→compute0，load2→compute1，load3→compute2，compute3；每个 compute 仍是原两次 K128×两条独立 N128 Acc。

Mat 全部本轮局部，只有两个 Acc 回边；两个 Mat 三元组交错生存的容量目标仍384 KiB。该表达与旧 v1–v7 的嵌套 pipeline/按奇偶更新Acc不同，也不再尝试 Mat loop-carried。每 K1024 存在一次供数启动/排空边界，不能仅凭源码把它当成性能进步。旧 MASTER 没有找到完全相同的 O_A 四段手工流水；HC_pre 只提供相邻 L0 值交错的表达参考。

v1 完整保留，v2 只允许一次独立 CPU lowering。v2 的一次编译与生成审计已完成，结果如下；没有设备测量。

## 复现

v1 原记录位于 `cpu_v1/command.json`；禁止覆盖历史输出。另给全新输出名时可执行：

```bash
python run_cpu.py pair_explicit.py new_cpu_output
python audit_failure.py
```

本轮不再运行 v1；所有本地路径和大小将列入 `LOCAL_ARTIFACTS.json`，不做 hash。

## v2 最终结果：资源与生成依赖通过，不代表性能通过

`v2/cpu_v2/result.json` 的 `full_codegen_passed=true`，唯一尝试约8.81秒。`v2/generated_audit.json` 与 `event_sequence.txt` 保留实际槽位、每个 SSA 分配、MMA顺序与完整关键事件。

- Mat393216 B、Left65536 B、Right65536 B、Acc131072 B，恰为设计上界。Right只有0/32768两槽；两个Acc为0/65536，compact有效行保留。
- `38_after_AllocateMemoryAddr.py` 不再有任何 `tile.move`；只有两个Acc跨 `range(4)`，每轮四K256是有限数学tile展开。每条Acc沿K0、128、…、3968累加。
- 真实 `kernels/aic/oa_pair_four_k256.cpp`：下一块1三次TLOAD位于375/415/449行，当前块0第一次extract在478；当前块0只等自己的MTE2→MTE1事件0/1/2，而不是未来块1的3/4/5。
- 块2 TLOAD位于612/652/686，块1首extract在712；块3 TLOAD位于834/874/908，块2首extract在934。生成代码保留了相邻Mat供数窗口，没有用未来块完成事件提前阻塞当前块的首extract。
- Right复用保留真实M→MTE1事件，例如MMA0后514行发事件2，533行等待后才537行提取下一Right-a；另一Right由546行发事件3、554行等待后再557行提取。不能按SSA死亡而跳过硬件完成。
- 该C++仍有每K1024的局部启动/收尾、MMA等待与跨轮槽重用依赖。能证明生成依赖允许供数窗口，不能证明设备实际重叠比例，也不能由静态调用数估计完整L1时延。

探针仅24pair，最后16个singleton及动态整层的量化owner/通知仍未接入。v2不是正式可替换helper，没有修改生产或依赖；下一步只读规划完整覆盖/等待合同，未经授权不继续编译或上卡。

已有审计可仅用系统Python重读，不触发编译：

```bash
python audit_failure.py
python v2/audit_generated.py
```
