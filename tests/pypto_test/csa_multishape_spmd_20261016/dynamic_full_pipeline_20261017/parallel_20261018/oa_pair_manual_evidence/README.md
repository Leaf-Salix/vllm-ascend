# O_A 双输出完整动态 CSA 独立候选（2026-10-18）

**一次完整符号 CPU 编译通过；源码已冻结，尚无本目录内的设备精度或性能结果。** 主会话已分别提交 long B4 与 short B16 两个代表，正式源码未修改。本目录不发设备任务，也不重复CPU编译。

- CPU：long G6、N128、OB_ROWS96、QA_M16_PARALLEL=false（T96的当前策略），130 runtime roots、58动态轴、单MIX、sync_start、未提供样例shape。
- 已提交设备任务由主会话管理：`task_20261018_055534_12872405208`（long B4/T24，QA_M16=true）；`task_20261018_055534_128729815924`（short B16/T96，G2）。这两种controls会重新生成代码，不能把本次G6/QAfalse的allocator当成它们已验证的结果。
- 冻结基线：66882634的全部33文件；只改4文件，其余29文件逐字相同。源码与接口AST审计见 `source_audit.json`、`interface_ast_audit.json`。

## 改动与边界

候选仅在 `OUTPUT_GROUP_PIPELINE && OA_COLS==128 && 0<T<=96` 生效，沿用当前N128区间。T97及以上、N256、关闭组流水均保留原路径。没有B/S白名单，没有把实际T固定96；packed组步长与输出容量来自动态Tensor轴。

| 文件 | 本轮改变 |
|---|---|
| output_projection_a_stages.py | 新增NZ pair+single组helper；旧函数AST全部不变。pair用已独立CPU通过的K1024普通range、四K256局部Mat流水，两条独立K128 FP32 Acc；single沿原首K256与后15段数学 |
| cube_schedule.py | 小T区间按新owner执行完整本组工作后发布；仍先完成本核所有OA，之后才等quant做OB |
| vector_schedule.py | 同谓词替换量化的OA等待集合，量化/尾零/OB/HCpost原数学不变 |
| phase_sync.py | 仅新增专用五owner等待；旧信号函数AST全部不变，等完后一次原fence/cacheinvalid/fence |

每组g：3个pair的owner为3g+p（p=0..2），各写N128列块2p和2p+1；两个single owner为(3g+3+s)%24，各写块6+s。五owner互异，24pair+16single覆盖64个N128输出；16核各3个输出、8核各2个。归约/量化没有合并为N256结果。新增工作区/phase/runtime参数均为0。

`owner_evidence/` 保存最初{2g,2g+1}映射、最终回绕映射、96个T的唯一写/等待模型，以及各group就绪的P/S符号表达式。它不假设pair=两倍single，不模拟硬件时延。新方案去掉同组pair+single串行，但一些pair owner会先做前组single，故不预记组流水性能收益。

## 完整 CPU 的真实资源

`cpu_v1/result.json` 为唯一编译结果，`generated_audit.json` 是之后只读已生成物的审计。

| 范围/空间 | 实际占用/地址 |
|---|---|
| pair Mat | 六份64 KiB，共384 KiB；在全函数内地址65536..458752，最高地址448 KiB |
| pair Left | 0/32768两个32 KiB槽，共64 KiB |
| pair Right | 0/32768两个32 KiB槽，共64 KiB |
| pair Acc | 0/65536两个64 KiB槽，共128 KiB，compact实际行保留 |
| 整AIC Mat最高地址 | 475136 B（464 KiB） |
| 整AIC Left/Right/Acc最高地址 | 65536/65536/131072 B |

pair内没有Mat→Mat复制。不能把“pair六份Mat共384 KiB”误写成“全函数Mat上界384 KiB”。完整函数仍有原有pipeline降深提示，本次没有为所有提示重新编译相同控制baseline，不能称所有其它分配/性能均不受影响。

## 生成事件与动态有效写回

完整C++位于 `cpu_v1/kernel/kernels/aic/cube_schedule.cpp`；已提取较小的 `pair_generated_cpp.cpp.txt`、`pair_event_sequence.txt` 和 `pair_allocated_ir.py.txt`。

- 块1三次TLOAD在18940/18976/19006行，当前块0首extract在19032。当前块只等本块MTE2→MTE1事件0/1/2，不提前等未来块1的3/4/5。
- 块2三次TLOAD在19161/19197/19227，块1首extract在19253；块3三次TLOAD在19370/19406/19436，块2首extract在19462。实际依赖保留了相邻Mat供数窗口；不是仅由源代码排序猜测异步。
- Right重用仍有M→MTE1等待；不因前一MMA已发射就覆盖。两个Acc写回前保留M→FIX；后续发布仍含业务fence/cache维护。未删FFTS、READY/ACK或全函数drain。
- wrapper从Tensor shape读取动态 `CSA_FULL_T`，传入v133；双Acc都`SetValidShape(v133,128)`，两次TSTORE目标都是实际v133行、128列。P_MM和packed容量仍是独立动态轴，不要求buffer扩到探针固定384或每组128。
- 生成IR生产/消费两端同为`0<tokens<=96`。AIV等待三pair owner和两个回绕single owner，全部满足后只进行一次原cache维护。`wait_allocated_ir.py.txt` 保留实际代码。

以上只证明CPU生成和可检查的依赖/地址，不证明设备重叠比例、bitwise精度、guard、graph或跨次代际正确；这些由主会话设备任务判定。

## 过程与复现

前置独立探针位于 `../oa_pair_explicit_mat_probe_20261018/`：v1的Mat循环携带生成三次A3不支持的Mat→Mat；唯一v2修正改为四K256局部流水后CPU通过。没有修改依赖。完整候选沿用v2已通过表达，不隐藏v1失败。

本目录 `prepare_candidate.py` 从Git读取66882634并拒绝覆盖冻结源码；`run_cpu.sh` 保存本次唯一命令。CPU入口复用主仓 `dynamic_full_pipeline_20261017/reproduce_cpu.sh`，只用 `--spmd-source` 选择私有包和 `--controls-json` 选择OA128/OB96。源和运行脚本已经冻结，不应在设备任务运行时修改。

只读已有生成审计：

```bash
python audit_generated.py
```

初次该提取脚本误用被CSE消除的SSA注释作为片段结束标记，出现StopIteration；改用真实single分支后通过。修正只涉及审计脚本，记于 `audit_script_correction.json`，不曾修改候选或触发第二次编译。

`LOCAL_ARTIFACTS.json` 仅列本地文件绝对路径及大小，不做hash；大型原始CPU产物/可能二进制留本地，主会话按清单归档必要文本。此目录没有commit/push或主仓编辑。
