# Sparse首PV块省略零累加项

任务`task_20260929_100643_249687018410`于10:06正常auto提交，设备1，已完成exit0。
**不采用、不扩测**：八类状态零容差通过，但Sparse核时两档均退化，四窗口范围不重叠。
长B16 AIC/AIV分别+4.938%/+5.075%，短B24为+8.633%/+8.726%；
CSA分别+0.442%/+1.791%，长短8:2为+0.712%，两档P95也回退。
[完整结果](RESULTS.md)、[四窗分布与原始样本](summary.json)。

基线55b89ee2已融合末块归一化/逆RoPE发布，本候选只处理首个PV块的零初态。
参考最新本地ops-transformer 28f40354的
`experimental/attention/sparse_attn_sharedkv/op_kernel/arch22/sparse_attn_sharedkv_scfa_block_vector.h`：
`DealBmm2ResBaseBlock`仅在`!info.isFirstSInnerLoop`时执行旧结果缩放和相加。

当前PTO在每个query首块仍计算alpha×0加beta×PV，以及alpha×0加beta×l。
候选在`pv_sb==0`且block有效时直接取beta×PV与beta×l；后续块不变。
保留beta是必要的：PTO概率按局部最大值生成，初态m含attention sink，
不能照搬Native累计最大值概率的“首块直接取原PV”。未改softmax、BF16舍入或跨块顺序。
无效首块仍走原跳过路径，后续有效块使用原通用更新；保护区和padding需保持。

pypto-lib2164563的局部softmax/后续合并是原实现参考；本次吸收AscendC的首块特化，
没有切换到Native累计最大值算法，也不重试已否定的联合softmax候选。

CPU完整PTOAS/CCE编译、链接及load通过，两根依赖图可解析。
生成`qk_pv_aiv.cpp`首块分支只保留1次统计量TMUL与2次TROWEXPANDMUL；
alpha对应TEXP、旧结果2次TROWEXPANDMUL及加法只出现在else分支。
这证明删除表达已落到设备代码，不是性能收益结论。

两侧均复制私有整包自`.cache/csa-native-inplace-seven-20260929`，测试包装保持inplace_pass=True。
只跑128K/B16、8K/B24，同卡交替次序，CANN9.2/mode2/atomic0/det0，
5次预热20次正式图事件、各四窗DFX及八类完整状态零容差比较。
核内与CSA/P95分开判断，长短8:2；没有真实核内收益则不合入、不扩测。
同一时间进行的Native七档任务与本A/B相互独立，不拼接两轮数据。
生产算子保持不变；此结果不否定Native自身首块特化，只否定本次PTO分支/驻留表达。
Vector侧调整后Cube核时也增加，说明核内等待仍须计入，不能只按删除的算术指令预估收益；
当前证据不把退化精确归因于某条同步或UB分配，不追加无依据的调参。

[准备](prepare.py)、[最小补丁](candidate.patch)、[源码来源](source.json)、
[CPU编译](compile_candidate.json)、[入口](run.sh)、[收集器](collect.py)。
