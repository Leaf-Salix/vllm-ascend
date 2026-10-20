# B1：保留原排序语义，半叶分数留在UB

本候选基于已经通过H139264/S5全同分和128K/B1/S6完整验证的
`source_b1_legacy_half`。原半叶GM版最新同轮完整L1为378.66μs，
正式SPMD399.18μs、新多task381.86μs；相对多task仅提升0.85%。
旧4K流式版13.38%由于同分精度失败不能采用，不作为本候选的性能参考。

## 数据流与预期作用

原半叶版由AIC生成每段1024候选的FP32分数，两个AIV各负责一条query，
乘KV scale后先写GM score scratch，半叶全部完成后又由排序helper读回。
新候选用每AIV的4×1024 FP32 Tile累积这4096项，再reshape成1×4096，
交给保留原六条长度分支的排序helper。原叶起点、动态半跨度、有效候选数、
Sort32/MrgSort次数及输入先后、最终根的左折叠顺序均不改变。

这里4096是原半叶的数学Tile上界，实际B/T、query边界、历史、位置和页表
仍从device Tensor读取。只沿已获准的B1区间，不新增B16/B28单点策略。
每个query最后仍写同一pair根，跨核发布、获取和代际维护不改变。

期望减少Indexer内的中间GM写回/重读及相关MTE同步。代价是16KiB分数
跨循环存活、UB assemble/extract和初始化；不能只用删去字节量推导微秒。
完整生成PTO的Vec声明地址上界两版均163872B（不是生命周期证明）。

## 编写与编译过程

完整记录见`B1_HALF_SCORE_RESIDENT_ATTEMPTS.json`，未删除失败。

1. v1命令路径相对仓根错误，未进入编译。
2. v2参数仅写`pl.Tile`，缺形状/dtype；补成`pl.Tile[[1,4096],pl.FP32]`。
3. v3分支内定义跨循环使用未通过SSA作用域检查；移至共同作用域，
   G6不消费该Tile，保持原路径，未修改编译器。
4. v4按动态列插入1×4096母Tile，触发PTOAS子视图物理列宽恢复问题。
   输入TMOV两端均1×1024，ResolveBufferSelect后目标变1×4096、
   valid=1×1024，A3同形状检查拒绝。纯PTO列/行对照已复现。
5. v5改为4×1024按行插入再reshape；58符号轴、单MIX完整编译通过，
   不传样例shape、不放宽资源预算。

写侧子视图问题已补到
[PTOAS #1649](https://github.com/hw-native-sys/PTOAS/issues/1649#issuecomment-6299434307)，
含最小PTO复现、失败pass、同形状行布局对照和修改建议；没有PR或标签。
`b1_tmov_subview_repro_v1`误用未配置的tools包装器，未进入编译；
v2使用与生产验证相同的build/bin/ptoas，列版失败、行版成功。

## 设备验收

候选`source_b1_half_score_resident`，生成脚本`prepare_b1_half_score_resident.py`。
同分回归与完整S6五窗任务见`B1_HALF_SCORE_RESIDENT_TASKS.json`，
完整性能单列于`B1_HALF_SCORE_RESIDENT_FULL_TASK.json`。
S6对照是原半叶GM版，多task为最新37c47753冻结快照，均150tick，
独立工作区并记录地址。不拿旧轮次或最快窗口替代本轮中位数。

任一精度/状态/保护/图失败都不能采用。若完整L1无收益，停止这一存储
候选，不靠删除同步或改同分规则取巧。完整目标仍未达成，正式数学未改。


## 第一轮设备结果

同分H139264/S5任务335495412701和完整S6任务335557216998均exit0。
同分、精确输出、Top-K/score、完整状态/保护/图检查通过。
S6同轮多task388.40、半叶GM对照378.70、分数留UB候选376.10μs，
中位净省2.60μs，4/5配对窗更快；相对多task提升3.27%，尚不足10%。
此处的小收益仍可能受实例/地址波动影响，不把它与另一轮20.52μs相加。

接着核对实际S1/capacity6，并以正式source_after_wait150直接配对核对
整体候选收益；任务见B1_ADOPTION_CONFIRM_TASKS.json。正式尚未采用。


## 最终采用

S1/capacity6和直接正式源确认均exit0。确认轮多task387.60、正式改前
401.30、候选371.94μs，五窗均更快，净省29.36μs，相对多task4.21%。
只采用B1既有授权区间，10%目标仍未达到，其余八档整轮未重测。

G6 CPU核对发现无用full初始化仍发射16KiB填充，因而没有直接复制初版。
最终source_b1_half_score_resident_guarded在共同作用域create、仅B1分支full。
B1完整PTO IR仅去loc/SSA拼写/空白后与已测v5逐字一致；G6额外Tile与填充
均消失。保留两版源与CPU证据，不为该等价B1代码重复设备测量。
正式源只增加整体中文说明，运算AST与guarded一致；新正式快照
source_after_b1_half_v2。首次source_after_b1_half仅作失败过程参考。
