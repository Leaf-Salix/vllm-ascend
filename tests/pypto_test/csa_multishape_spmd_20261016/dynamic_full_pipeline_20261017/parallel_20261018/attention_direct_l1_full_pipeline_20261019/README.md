# Attention压缩KV直接采集到AIC L1的完整动态候选

参考正式289cc072。新算子先在ops/pypto的动态源码中接线，再冻结到
source_candidate；正式采用版本已恢复，运行任务只读冻结快照。

目标是删除Native主压缩cache→AIV UB→GM kv_ring→AIC L1中的中转写读。
相同物理plan容量谓词成立时，AIC使用设备INT32物理行索引直接MGATHER到
三个L1槽；SWA继续走原AIV采集，容量外仍走原逻辑路径。保留原QK/PV、
softmax、统计/归并、Top-K顺序和全部事件计数。物理plan的符号T、device
Q/L/positions/页表/有效块没有实值化，不为验收点建白名单。

新AIC读主cache前显式获取MAIN_OUTPUT真实生产者及payload缓存语义；
不能把plan完成当cache完成。三个Mat槽各自携带，QK领先PV两拍，消费者
保留在对应槽分支，避免对不同存储做Mat→Mat搬运。三个槽和Query共448KiB，
概率Mat另16KiB，这是源码容量估算，实际规划和编译仍须核对。

依赖独立gm-subview PyPTO的连续GM视图及Flatten修复，PTOAS v3，私有
PTO-ISA头文件的ZERO→ND2NZ屏障修复（#352）。同轮三方共用这些工具链，
共享安装和托管组件仓库不改。Mat条件选择问题已上报PyPTO #3049；本候选
采用分支内消费者，不伪造编译器修复。

入口仍是完整CSA、一个MIX task。运行方式沿既有完整三方harness，先全部
精度/状态/保护/协议/metadata/padding/图门槛，再同轮五窗完整L1。CPU及
设备结果未完成前不记通过、不采用、不更新七档性能表。

完整CPU v1保留58动态轴，但view的shape内嵌tensor.dim乘法被生成成
arith.muli的空SSA操作数，PTOAS语法解析拒绝。v2把dim先绑定成命名Scalar，
不改变shape/地址或动态算法；是否解决待编译。原始PTO和失败保留。

CPU v2完整通过58动态轴/单MIX，五份权重CachePolicy绑定仍对应原参数。
单独最小view复现在未修改72b7ffb9上nested/named均通过，不能把整层v1
失败泛化为上游所有tensor.dim内嵌表达式均不支持；继续保留完整失败证据。
完整设备任务task_20261019_111815_402547725611已提交，实际待测源为
source_candidate_v2，三方均使用GMP/PTOAS v3和私有ISA屏障修正，不预记通过。

补查view嵌套dim问题：未改上游72b7ffb9两写法均通过；NEWP父工具链和GMP
均为nested失败/named通过，故不是新增gm_view本身引入。源码差异显示本地
基础缺少PR #3013的FlattenCallExpr Var/IterArg类型元数据保留处理（两条
VisitExpr直接返回原Var/IterArg）；无该保护时类型中的dim被hoist为另一份
临时值，最终SSA操作数缺失。该上游修正已经存在，不重复提交新issue。
本轮保留命名Scalar写法继续整层验证，没有在运行中的任务上更换二进制。

完整B16设备任务已完成A→B→A、metadata/padding和图门槛，进入参考编译；
在性能窗口完成前仍不记录收益。另准备较大T/连续scale/LOCAL_STATS关闭的
编译分支检查，不凭B16编译结果推断所有资源区间通过。

## 完整真机结论：精确通过、性能大幅回退，不采用

B16任务已completed/exit0，A→B→A输出、Top-K/score、Native全部cache/state、
写保护、真实代际、基础图和metadata/padding图均通过。五窗中位数为：
原多task857.88μs、正式参考814.76μs、直接L1候选1340.56μs，较参考回退
64.5343%。不采用、不扩测其它档位、不更新当前七档最佳性能。

较大T/连续scale开启且LOCAL_STATS关闭的完整CPU编译也通过58轴/单MIX；
它不算B28设备证据。源码每个压缩块由两个AIV各采64行改为一个AIC发128个
单行ND2NZ，加入内部ZERO顺序屏障，三槽改为显式分支携带；既有事件协议
保持。减少了GM中转，但增加/迁移了AIC Scalar与DMA负担。不能仅凭该
时延把回退完全归因于某一个成本，也不能删除必要屏障换取虚假性能。
原始窗口、实际CCEC头文件路径和完整精度结果均保留。

## 复现最新已验证候选

在本目录执行。CPU默认source_candidate_v2/controls_on；较大T编译追加
`--controls-json "$PWD/controls_large.json"`。输出必须使用全新绝对路径。

```bash
bash ./run_cpu.sh /tmp/csa-direct-l1-cpu-NEW

task-submit --device auto --max-time 1800 \
  "bash $PWD/run_device.sh /tmp/csa-direct-l1-device-NEW --batch 16 --history 131072 --l1-windows 5 --metadata-replay --padding-replay --failure-snapshot"
```

依赖工作树及环境仍按现有session路径；私有ISA头文件不入Git，缺失时由
E35/prepare_isa_overlay.py从本地Simpler托管原ISA重建并应用同一处补丁。
原始大构建/二进制本地路径和大小见ARCHIVE_FILES.json，不做hash校验。
