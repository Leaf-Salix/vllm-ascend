# 唯一 CPU 产物的只读生成审阅

结论 PASS。此目录只读取主会话已完成的 `PAR/idx_had_owner_evidence/cpu_v1`，没有新编译或 NPU。对应原件和大小见 `INPUT_ARTIFACTS.json`；只保存必要小摘录，没有复制整份约4.8 MB PTO或约2.5 MB C++。

参考明确取 `PAR/oa_pair_manual_h131072_b16_v1/source_matrix/result.json` 的 `spmd_reference`：`build_output/decode_csa_spmd_jp7o4g2r`，不是该任务的 OA pair 候选。其33份源与本目录上级冻结的668参考逐字相同。两侧 controls 完全相同：G6/QK64/Score1024、OA128/OB96、QAfalse及原组流水。CPU结果保留130根、58动态轴、单MIX、sync_start，没有给样本静态shape。

## 真实分支、地址和生命周期

完整行号与原始路径在 `wait_had.pto.txt` / `wait_had.cpp.txt`，常量值在 `cpp_constants.txt`。

- PTO1427是 `worker < arg132`；arg132对应 C++ v133。实际wrapper从 `x_fp32.shape[0]` 读取 `CSA_FULL_T` 并传给v133，v189来自 `__pypto_spmd_block_idx`、再转为worker v926；没有把T改成某个测试常量。
- C++10258是 `if (v926 < v133)`。10259循环从0到16；10262计算 `((worker/8)*16+ordinal)%48+24`，再乘每槽128个INT32，加phase17基址156672。
- C++10281为TWAIT；10283–10291是原MTE2/MTE3/FIX barrier与DSB、DCCI、再次barrier与DSB。全部位于Hadamard常量TLOAD（10307）及业务RoPE TLOAD（10345）之前。不是仅在源码中存在维护，真实C++仍在使用者之前。
- HAD循环仍从worker到 `T*64/64`，步长24；一次只消费一个token的64×128 BF16，计算后写64×128 FP32。PTO1466 matmul、1469 TSTORE后，1471–1473原业务维护保留；1476 PIPE_ALL后在1479向phase18发布。
- C++在最后TSTORE后保留FIX→M完成等待及原pipeline排空，再做业务维护、PIPE_ALL和TNOTIFY。空worker虽不执行数学循环，发布仍在循环外；未删原空HAD通知。AIV侧不变，包括没有RoPE数学工作的AIV也按原调度发布阶段完成，不能误称它们“不发布”。

这里只改变哪些生产者需要等待，未将业务payload的隔离粒度等同于signal的512B隔离槽；未假设Scalar缓存一致。

## 与同控参考的有限等价证据

| 项目 | 实际结果 |
| --- | --- |
| AIC PTO | 仅剔除这一个等待块后，其余完整函数归一化相同；HAD的T×64计算也保留比较 |
| AIV PTO | 不剔除任何内容，完整函数归一化相同 |
| AIC Tile | 558项，逐项地址/内存空间/dtype/rows/cols/字节相同 |
| AIV Tile | 1812项，逐项同上 |
| AIC片上地址上界 | Mat475136、Left65536、Right65536、Acc131072、Vec16384字节，均不变 |
| AIV片上地址上界 | Vec163872字节，不变 |
| AIC局部事件 | 1711条set_flag/wait_flag/pipe_barrier，顺序和event ID相同 |
| AIV局部事件 | 3088条，顺序和event ID相同 |

归一化仅按**各函数自己的作用域**去掉loc/SSA名字并内联同值同类型常量；保留所有数学、地址、数据类型、布局和控制流。四个 `.diff` 均为空。初查若跨整个module收集常量，会让不同函数重名SSA互相污染；审计脚本已纠正这一点。脚本首次读取AIV分配时缺少ui32宽度，发生KeyError；补为4字节后只重新读取原件，未修改源码/产物或触发编译。

局部事件的平铺列表相同，不代表新分支动态执行次数相同：worker<T和等待owner缩减正是候选目标。设备参考用了wait-backoff辅助实现，而CPU为默认TWAIT；不宣称辅助头、完整机器码或实际时序等价。这个结论也不替代NPU精度、图、保护区、同步可见性或性能结果。

复算：`python audit_generated.py`。所有输出只写本审阅目录。
