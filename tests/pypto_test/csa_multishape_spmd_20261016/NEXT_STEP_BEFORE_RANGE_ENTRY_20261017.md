# CSA单SPMD动态档位：当前交接

## 新目标已生效：全七档至少降低 30%

先读 [GOAL_AND_CONSTRAINTS.md](GOAL_AND_CONSTRAINTS.md)。用户已确认 `SPMD ≤ 同轮动态 PTO × 0.70`；原 700 μs 单点门槛已替换。以下旧记录保留历史语境，不再代表当前目标。

当前继续实现按工作量的区间入口、补齐缺少参考的调用分支；并调查七档共同的数学/访存瓶颈。原七档收益约 2.9%～10.8%，没有一档满足新门槛。


**先读 [GOAL_AND_CONSTRAINTS.md](GOAL_AND_CONSTRAINTS.md)。** 用户已明确：分档是按 CSA
工作量划分区间并选择计算/调度策略；七个测试点不是支持白名单，不照搬 HC_pre 的阈值
或 T=96 专用分支。原动态 shape 与 128K/B16 <700μs 目标保持不变。

当前goal active、无预算上限：按batch/历史策略分别编译的动态metadata执行，继续优化128K/B16完整无插桩L1至700μs以下。用户允许像hc_pre按档选择不同分块/分核，不要求单二进制覆盖全部batch。目标未达到，不标complete/blocked/paused。本轮有CPU代码和完整设备验证进展，没有需要用户批准的动作。

## 当前状态

全部CPU和设备任务均已终态，PENDING_JOBS.json已更新，没有待poll任务。
七档仍为S6：128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；
8K B16/B24/B32=621.96/836.74/905.72μs；另一同进程现选B16为807.50μs。
selected_shape_policies.json没有改变，各档用自己的排程和数学策略；长B8/B16用Vector索引，只有长B16用rowdeps，均cached100。
静态a919f119已push保留；动态实验仍未commit。本轮仅修改N实验目录并镜像过程文本，没有改生产/F/依赖仓或共享安装。

## 本轮结果：不重复

1. 固定物理分核MILP：v1全局模型120秒无整数候选；v2约简约束+最多96对顺序变化得到773.41模型值，无改善；v3限定模型≤750后120秒无候选。不是不可行证明或设备下界，没有上卡。
   weighted_precedence约简24825→20402数学边，242701可达对的最长时间差全部保持，误差0。见FIXED_CORE_MILP_EXPERIMENT.md、源码v1/v2快照和三个结果目录。ortools独立安装失败日志保留，使用已有SciPy/HiGHS，没有改共享环境。
2. RoPE去gather探针：v1把FP32 reshape成256×2，PTOAS因8B行不满足32B对齐拒绝；v2先转置、交换行对、再转回完整CPU通过，但新增搬运且没有设备性能。不要据此改掉PTOAS安全分支。现选QK还有任意页表gather，不能宣称只改RoPE就让整个函数无gather。
3. rope_flat_gather.py合并三个逐行gather，原列索引保持，改绝对位置r*64+c。仅3个循环修改，其他AST完全保持；262144地址组合证明；实际1024/1024/2048 Tile与索引链、同时活跃UB不重叠通过。CPU v1只是命令路径static/误用，未进入编译；v2按variants/路径完整PASS，集成源码AST完全相同。
   task_20261017_131052_534942170 / b16_row_deps_rope_flat_v1 exit0：所有eager/graph/final输出/TopK/六state零差异，30代/保护区PASS；完整L1 819.58/815.10/810.50，中位815.10，同轮动态920.58。未优于现选、未采用、不扩档。--rope-flat-gather默认关闭。见ROPE_GATHER_EXPERIMENT.md、rope_flat_gather_result_v1.json。
4. 全阶段noinline+当前cached100组合：早期默认等待时代1044.54不能直接外推现选，因此补做该组合。CPU独立编译74helper，外层事件只有全部分离且set/wait成对时清理，helper自身事件、FFTS及数据维护不变。实际两个CCEC源码各74处cached100，完整覆盖/外层无遗留事件PASS。审计适配backend的static __aicore__拼写，没有修改编译产物。
   v1 task_20261017_131724_17808923318因另一处旧rowdeps参数保护拒绝，exit1，未执行算子。窄范围开放B16/128K/all stages/compact/noinline/现选rowdeps+QK索引/cached100；CPU解析验证原现选与此组合可用，partial或缺noinline仍拒绝。
   v2 task_20261017_131834_1981138300 / b16_row_deps_outlined_cached_v2 exit0，完整精度/图/30代/保护区和实际源码门禁PASS。L1 807.02/805.34/810.66，中位807.02，同轮动态902.84，约与现选持平，不采用、不扩档。eligible_for_700us_target=false，C++诊断不是正式PyPTO接口。40秒观察超时不等于任务故障，后续正常完成。见OUTLINED_CACHED_WAIT_EXPERIMENT.md及outlined_cached_result_v2.json。

## 新方向，仅完成源码分片枚举

qproj_dequant_ownership.py和qproj_dequant_ownership_cpu_v1.json：
B16/T96原head_tile16时，每个后处理worker读32个N256列块，覆盖全部20个投影worker；不能直接减少父核等待。
若仅将该档后处理head_tile改4，每个worker仍执行16个head×8行数学迭代，输入只涉及8个投影worker，12×64行/head块各有唯一写者。tile8需要16父核。tile1/2也需要8父核，但QK依赖的后处理worker增至48/32；tile4和原16的QK读范围都涉及16个后处理worker。

下一步可先补真正的同步合同，再CPU模型筛选：
- 核对本轮绑定tile_rows=96、tile_base=0、q_proj_i32生产/消费视图完全一致，其他共享临时区和根输入的读写性质；不能只凭allocation名称相同。
- 精确核对后处理完整行/head写回与QK的读范围、完整图偏序、原216条FFTS复用边、各核既有顺序及缓存维护。
- 原数学体head_tile16没有直接删投影依赖的空间；如比较4，要把外层循环次数变化和索引准备开销当真实风险，不能按相同数学迭代数保证同耗时。
- 可以先在CPU估计缩父核后的关键路径，再决定是否值得生成/上卡。当前没有新等待表、没有图模型收益、没有设备候选；不得预记PASS或直接删边。
该方向不同于已测的改投影worker数、主Q整体后移、N128/L0预取；避免重复旧方向。

## 保留的重要限制

- 正式权重/data/model/DeepSeek-V4-Flash-0731-w8a8，PTO对PTO，不跑Native forward；只经task-submit --device auto使用NPU。
- 不修改运行中设备任务导入的driver/helper；上下文入栈后才_resolve_kernel_artifact，实际等待门禁必须检查kernels/，ptoas/是替换前源码。
- 完整exact输出/TopK/六state、图、30代、保护区后才采3窗完整无插桩L1，含等待与空隙，排除外围AICPU。不能用短插桩或Event时间替代700验收。
- 七档同地址动态metadata已验证，不重复。固定fixture恢复不是连续decode或整网；生产自动派发尚未接入。
- 现选QK按head分工8/8/8与16/32/8已完整PASS但885.08/830.46，均未采用；见上次NEXT_STEP快照和QK_ROLE_SPLIT_EXPERIMENT.md。
- 固定局部交换120秒773.41→773.41、主Q后移→788.94、成员三启发式和query两组六模型都无收益；不延长相同排程搜索。历史L1没有已验证logical block映射，不能靠结束顺序猜成本。
- 所有负结果见ATTEMPTS.md，不能因为目标困难或本轮无收益自行暂停/标blocked。新候选必须有新依据，避免重复七档测试。
- 原静态和所有失败产物保留。镜像只增量复制过程文本，超过2MiB和二进制留本地列路径，不做hash，不删除目标文件。


## 最新续接：2026-10-17T13:42:01+08:00

上述“主Q生产者范围CPU核对”的待办已完成：qproj_dequant_dependencies.py和qproj_dequant_dependency_model_v1/result.json证明绑定/写者/偏序；tile16/8/4模型都为771.53，没有收益，未改核体/未上卡，不重复。当前没有本会话新设备任务。

用户现在要求采用hc_pre式范围分档：seqlen区间×batch场景（精确特化须有CSA自身依据），七档仅作为性能回归点。已展示逐档同轮分task vs单SPMD表，来源重新检查保存selected_seven_readback_20261017.json。不要把七个测试点当成最终支持范围，也不要把S6冻结源码误说成已支持S5。下一步按GOAL_AND_CONSTRAINTS.md实现范围策略，并继续700目标。
