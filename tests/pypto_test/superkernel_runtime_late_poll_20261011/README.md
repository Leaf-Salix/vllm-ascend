# 过程058：运行时首次探测后的后到交接与128K/B16 CSA

日期：2026-10-11。仅私有A3/TMR环境、单卡。算子作者仍使用局部Wait/Set；
不做自动前缀识别，chip swimlane、PMU与参数dump继续串行回退。公共CSA默认未改变。

功能进展：正式预留L1/UB scratch和通知事件后，编译器生成的尾部回调在AIC/AIV及MIX上通过。
后到计数发生在**原执行器第一次探测失败之后**，解决过程057首探位置偏早的证据限制。
性能尚未受益：128K/B16新通知路径较同SDK单次探测版mean多4.32μs，不能作为加速结论。

## 实现与资源合同

- PyPTO `fee0196c2`：共享MemoryReuse/地址分配规则在显式FIFO之后、tile之前预留64B。
  AIC使用Mat，AIV使用Vec；不再硬编码本地存储末尾地址。内部属性传递资源能力。
- PTOAS `c63b7d28`：为带内部能力的函数保留MTE2/MTE1/M/FIX/V→MTE3的event7；
  显式pipe-event函数拒绝该实验合同，避免未证明的事件冲突。正反编译检查均完成。
- Simpler `1c258b041`：先执行原有单次handoff探测，失败才调用当前kernel的私有尾部入口。
  参数指针bit1选择这个入口，bit0继续表示原前缀交接；尾部入口在参数解码、atomic初始化和
  算子主体之前返回，**不会重复执行算子主体**。无新FIN/ACK、无派发触发或核分配变化。
- 每物理核三行HBM通知内存（目标、固定+1源、固定−1源），A3合计13.5KiB，放在共享内存
  首个ring之前。目标交替写+1/−1，任务不逐次由Scalar清零。DispatchPayload仍为576B。
  通知初始化与缓存写回目前在运行时冷启动执行，普通路径也使用这份新SDK。
- AIC把MTE2/MTE1/M/FIX汇合到MTE3，AIV把MTE2/V汇合到MTE3，随后DMA写目标。
  轮询正常派发寄存器和通知值；本核流水、完整fanin与cohort等待仍由原Wait/完整PIPE_ALL保证。
- `RuntimePollable=6`保留完整等待回退；新`RuntimeCompiledPoll=7`表示编译尾部回调。
  trace word4低8位仍为7，bit8表示实际进入首次探测后的轮询，bit9表示之后验证成功的handoff。
  尾等待含新增DMA，不能把其时长单独当作原Cube/Vector重叠证据。

私有vLLM驱动提交`60d19a78`。完整版本、增量补丁及提交说明见[versions.json](versions.json)、
各仓`*-incremental.patch`和`*-commits.txt`。补丁从过程057基线生成，使用零上下文，应用时加
`--unidiff-zero`；runtime gitlink按versions中的Simpler版本单独同步。文本归档只清理行尾空白。

## 失败过程与原因边界

| 阶段 | 单卡任务 | 结果 |
| --- | --- | --- |
| runtime直接DMA，通知位于常驻SO | `task_20261011_202646_376678118010` / 0 | once通过；AIV late第一个输入卡住，退出1 |
| 首次增加显式flush | `task_20261011_203237_399280831420` / 0 | SDK未提交，严格版本检查拒绝；未执行设备计算 |
| SO通知加显式flush及阶段记录 | `task_20261011_203514_409195615834` / 1 | AIV late仍卡住，退出1 |
| 通知迁到HBM | `task_20261011_204302_20304531505` / 0 | AIV late仍卡在同一阶段，退出1 |
| 编译器分配scratch，旧kernel内通知探针 | `task_20261011_204722_37767012376` / 0 | 两类核、两种长度、两组输入、三路及重放通过 |
| 编译器尾部回调，运行时首次探测后进入 | `task_20261011_205936_84957416722` / 2 | AIC/AIV输出、完整fanin读取、保护区和重放通过 |
| MIX | `task_20261011_210141_94488019903` / 0 | 2项通过，含跨核FIFO、sync_start和整组完成 |
| CSA | `task_20261011_210335_103255211417` / 2 | 三路径功能及计时完成 |

失败版诊断高位阶段值为4（`late_handoff_scratch=17179869184`），即到达DMA前、未到DMA提交后。
SO与HBM两次源值都正确，但均卡住。因此，**“仅是SO内存位置导致失败”的推断撤回**；
具体到哪条指令、哪项编译上下文差异尚未确定。保留debug/device日志，不把编译回调通过当成精确根因证明。
全部任务结束并释放设备，没有16卡任务。

2026-10-12补充：过程067已找到旧执行器的架构弱符号混用，两种外层入口都引用AIC定义，
编译期Cube分支因此进入AIV路径。改为各架构唯一强符号、按实际worker核型调用后，AIC/AIV/MIX均通过；
核型由实际trace确认。以上“尚未确定”保留为当时状态，最新证据见
[过程067](../superkernel_runtime_dma_entry_20261012/README.md)。直接通知的性能成本仍单独处理。

CPU中还出现：新单测遗漏target_memory、HBM header遗漏platform_config include、
负例先触发另一条验证规则，以及CCEC保存临时文件使Simpler变脏并触发严格SDK拒绝。
均修正后继续：原地址分配22项通过，新2项通过；调度11项、共享内存22项通过；PTOAS正反例通过。
临时预处理文件移出Simpler源码后重新编译通过。失败日志保留，不归为设备数值错误。

## 后到交接证据

以下每项列出两组输入的handoff总数 / 其中原首次探测后新增handoff数 / 进入轮询数。
AIV共48个物理成员，AIC共24个；两种工作长度，额外人工Scalar循环为0。

| 类型 | loops | 输入0 | 输入1 | 图重放 |
| --- | ---: | --- | --- | --- |
| AIV | 1 | 0 / 0 / 48 | 10 / 10 / 48 | 11 / 11 / 48 |
| AIV | 4 | 8 / 0 / 40 | 20 / 18 / 46 | 13 / 7 / 42 |
| AIC | 1 | 7 / 7 / 24 | 17 / 17 / 24 | 6 / 6 / 24 |
| AIC | 4 | 15 / 15 / 24 | 24 / 24 / 24 | 17 / 17 / 24 |

每条bit9都关联到同核、同generation的实际消费者，而非仅看到一个寄存器变化。
不同运行的交接总数不可相减作为固定集合上的因果收益。独立原计算重叠证据仍引用过程057；
本轮着重证明原首次探测后的覆盖与安全性。scratch隔离探针的计时仅保留原始记录，不当作优化成绩。

## CSA同SDK三路径结果

CANN9.2、TMR、NZ2、atomic关闭、deterministic level0、诊断关闭；第2层真实权重，
128K合成历史，B16×verify6。区间为HC_pre+norm+CSA+HC_post单次完整NPUGraph重放，
NPU event计时，状态reset在区间外。每路预热5次，再轮换各30次。

| 路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 普通版 baseline | 967.66 | 989.89 | 1017.96 |
| 同MIX标记，原单次探测 once | 969.76 | 993.18 | 1015.04 |
| 同MIX标记，编译后到通知 mixed | 964.56 | 997.50 | 1022.46 |

once与mixed使用相同的scratch/事件预留；once只在wrapper生成时关闭新Set adapter和回调。
两者差别是本轮通知协议。普通版没有局部标记，且三者都包含新SDK共享通知区的冷初始化；
本表不能证明“新SDK对普通版毫无开销”，也不能和另一张卡的过程054绝对时间相减归因。

mixed较once mean多4.32μs（0.43%），max多7.42μs，30组中10组更快；
较baseline mean多7.61μs（0.77%），max多4.50μs，30组中8组更快。
9项输出/缓存/状态及改变输入重放逐bit一致，每路31项保护区通过。
这是本轮三种PTO实现之间的功能检查，不代表Native数值或整模型token验收。

原始数据：[90条计时样本](csa_samples.csv)、[完整结果](runs/csa_compiled_poll_128k_b16_20261011/results.json)。
本轮保留功能实验，未将它替换公共默认。下一步单独减少每个Set的控制行写入与回调开销，
优先128K/B16；不因扩大覆盖就宣布性能收益，也不改变原派发来追求命中数。

## 复现入口

`source <private>/env.sh`前进入私有Git仓；按versions同步SDK、binding及torch adapter，
构建线程数使用环境中PYPTO_BUILD_JOBS。先CPU预编译，再提交单卡任务。
`repro/`保留本机脚本与原路径，`source/`保留驱动及关键代码快照，`generated/`保留AIC/AIV两个task的生成入口。

```bash
task-submit --timeout 900 --max-time 900 --device auto --device-num 1 \
  --run 'bash /data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/artifacts/run_csa_compiled_poll_128k_b16_20261011.sh'
```

原compiled目录及私有包在`<private>/artifacts/csa_compiled_poll_128k_b16_20261011/compiled/`。
复现旧版本应另用版本对应目录，避免覆盖本记录样本；不复制另一SDK的二进制。
