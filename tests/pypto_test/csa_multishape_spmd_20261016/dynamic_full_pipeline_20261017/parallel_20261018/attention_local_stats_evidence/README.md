# Attention同lane统计留在UB：新内存规划下重新评估

2026-10-18。基线25de3f32，源从正式动态目录产生，非静态B16生成体。
旧attention_local_ring_evidence的768B环在旧完整规划下使总Vec达255104B，
未上卡。本次使用后续已有独立内存复用修正，不修改共享依赖。

保留一个[6,32] FP32环，slot 2s/2s+1分别为mi/li。work w在tick w+1
生产，w+3消费，同槽下一生产w+4，跨query沿用连续work并保留尾排空。
旧33825种mask模型可复用，事件与环索引没有变化。区别是取消已无消费者
的两个GM写回；旧候选为诊断仍保留它们。本次全部原softmax和在线m/l/oi
数学、Q/L/position/page、两AIV分工及FFTS保持，不把本地缓存当跨核同步。

先完整无样例shape编译并核对实际UB；通过后再做完整PTO精确/图/保护及
同轮五窗L1。未提前承诺精度或性能；根接口暂保留mi/li workspace避免
把ABI清理混入首轮因果对照。源码与失败/结果持续保存在本目录。

## 编译与组件定位终态

v1在新内存复用树仍为Vec255104B超限。原因细化为：动态slice后reshape
具有非ConstInt MemRef偏移，本地安全门槛关闭整个Vec空间的细分复用。
不是768B环自己被扩张。v2改为显式ND extract两个128B行再reshape，保留
真实本地copy与编译器依赖，完整编译通过；全AST收集的Vec上界163872B，
环为135168..135936，58动态轴、130根、一个MIX。未改组件或关闭检查。

最小独立探针验证当前kernel调试树：row=0正确，row=1/4/5/1全部误读第0行；
extract对照五次均正确，任务exit0表示复现完成，不表示slice数学正确。
`dynamic_view_repro/DEVICE_SUBMISSION.json`和设备结果保留全部实际数组。
CPU最初两种kernel均编译成功，仅报告PosixPath序列化失败，修脚本后直接
登记既有产物，未重新编译；主线对照首次因旧/新JIT私有接口不同而失败，
改为公开IR编译接口后完成，两个脚本问题也留档。

不能把旧分支的地址丢失重复报成最新PyPTO bug：核对main14736f65并在
其派生树72cadb1e5编译相同输入，main已保留index*128动态偏移；相关allocator
和datamove源与GitHub main逐字相同。但当前PTOAS0.66对主线输入漏掉
TLOAD→TSTORE的MTE2→MTE3同步；完整32行PTO去loc后独立重编译仍复现，
24行extract对照有MTE2→V→MTE3保护。主线这个输入未追加NPU数值测试，
不能用旧分支的设备失败混作其证据。

已向现有PTOAS #1620补充精简复现、源码/版本界限及修改建议：
https://github.com/hw-native-sys/PTOAS/issues/1620#issuecomment-6249288578
远端已回读逐字核对，不建重复issue、不提PR、不加标签。

## 七档设备结果

七档完整PTO精确、Top-K/score、cache/state、A→B→A、图、保护区及代际均通过。
S1/B1、S5/B5对旧动态的精确/图回归通过；六次metadata与四次padding同图
回放通过。不扩称所有mixed-query、empty-rank、连续decode或Native/token精度。

同进程五窗完整L1见 `SEVEN_RESULTS_V1.md/json`：四档改善0.92%～4.44%，
128K/B24回退0.10%、B28回退1.38%、8K/B16回退3.28%。不全域采用；在独立
`../attention_local_stats_policy_evidence/`沿既有Score历史/G与O_A的T96
分界选择路径，并另补边界。不删除负窗口，不将跨轮数据冒充同轮收益。
相对原多task逐档降低30%的目标仍0/7。
