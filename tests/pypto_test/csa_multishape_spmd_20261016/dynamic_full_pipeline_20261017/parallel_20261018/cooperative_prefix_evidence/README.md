# 动态 CSA 前段生产与消费协同重排

2026-10-18。基于静态/动态配对端点，处理Score消费者被独立主分支挡住的问题。
工作在正式 `ops/pypto/deepseek_v4_flash_csa_spmd`，验证后已由默认策略启用；`source_candidate`
保存被测完整35模块，`source_reference` 保存原正式33模块。

## 与此前失败候选的具体区别

以前4辅助核首版把KV压在少数核上，且AIV在Indexer量化前等KV，完整L1回退；
修正版恢复24核KV，但主压缩仍晚做，完整L1也未改善。此轮不扫描核数，而是
同时改变主压缩生产/消费位置和Score消费者顺序：

1. AIC先用24个原owner生产主压缩，再做head权重；主压缩完整K归约不变。
2. 前4个AIC先做短compact Hadamard，再做query Hadamard；后20个AIC做Q_B。
   query Hadamard仍是已验证的四token/M256块，实际T和尾行继续动态。
3. 每核分支结束后做KV投影，仍使用原24个逻辑worker，采用循环置换让常见
   非空KV块落在Q_B侧；没有把KV压到4核。AIV获取KV时等待全部真实角色。
4. AIV先完成Indexer压缩，再做query RoPE与head权重；随后消费已前移的主
   压缩，在AIC处理Q_B/Hadamard期间完成pool/state/cache。
5. AIV完成Indexer量化/系数/key/scale及主Query反量化后直接进入Score。
   原KV norm/cache改在本核Score完成后执行，无相关行块的AIV继续Top-K。

## 同步、动态性及维护

三个AIV阶段链抽到 `vector_prefix_stages.py`，AIC分支在
`projection_overlap_stages.py`；总览和关键等待均有中文说明。
不新增根Tensor、工作区、Host回读、静态records/waits或第二个task。
Tensor维度和Native Q/L/位置/页表均保持device动态输入。核数参数描述资源
划分，不将七个测点当支持白名单。开关同时传给AIC/AIV，禁止只变其中一侧。

主/Indexer pool仍须全部相关请求读完共享prefix，真实state写者才提交；
没有实际请求的角色不再等待无关投影/pool，但仍按原合同唯一发布空完成。
32行KV norm owner与8行cache writer分开计算；后者等待实际norm生产者。
全部通知仍经原fence/cacheinvalid/notify，未删除A3缓存维护或Score READY/ACK。

独立索引缓存候选已证实无收益，本轮没有叠加；Attention数学与索引路径保持
原版。`before_modules` 仅保存两次编辑衔接时的文件，不是本轮测试入口。

## 当前验证

- 完整 `cpu_v1` 编译/PTOAS通过，58个动态轴、130根Tensor、一个MIX/sync_start，
  没有向编译器传入B16或其它样例shape。
- `check_prefix_control.py` 执行实际前段Scalar源码并展开新增inline阶段链，
  450组组合通过：发布唯一、等待有来源、依赖无环、Q_B128列块覆盖、24个
  KV逻辑worker循环置换覆盖。包含S1/S5/S6对应请求数及T=1..513的代表边界。
- 模型将Score三种角色的进入/完成分开建边，检查生产者与消费者能共同进入；
  不模拟FFTS计数、数学或DCache，不代替真机。
- 已提交128K/B16、128K/B28、8K/B32三个代表规格，任务见 `SUBMISSIONS_V1.json`。
  每轮先原精确/状态/图/保护，再同进程三方五窗完整L1。提交不等于通过或收益。

## 复现

`python check_prefix_control.py` 只执行CPU控制检查。
完整CPU：`bash run_cpu.sh 全新绝对输出目录 --controls-json controls_long.json`，
controls参数需使用绝对路径。设备使用 `run_device.sh`，必须通过队列提交。
大产物本地保留路径和大小，不计算hash，不提交权重或二进制。

## 终态与采纳

七任务均exit0，精确输出/Top-K/score、完整cache/state、A→B→A、图重放、
保护区及代际全部通过。完整同进程五窗中位见
[七档对照](SEVEN_RESULTS_V1.md)，原始样本和来源在JSON。
相对旧动态实现，七档降低1.97%～6.49%；相对原多task的30%目标仍0/7。
不是只统计某个提前的投影，也未用历史静态或最小样本替代本轮完整L1。

新增真实形状回归：

- B1/S1：`task_20261018_161630_168463818084`，exit0。
- B5/S5：`task_20261018_161630_168461724394`，exit0，覆盖四token Hadamard尾块。
- 两者均保持真实S，不补成S6；相对旧正式动态SPMD的输出、Top-K/score、完整
  allocation和保护区精确一致，A→B→A三轮图重放/固定地址/代际通过。
- 此次参考是旧动态SPMD，不新增S1/S5相对Native或整网token的精度结论。

B16/S6专项任务 `task_20261018_161516_163215116226` exit0：同一图改变历史、
位置、页表及compact metadata的六次重放通过；Native builder更新同址padding
输入的四次重放通过。检查有效输出和所有cache/state写保护；dummy输出本身
没有新增数值约定。该专项未覆盖empty rank、mixed-query或连续decode轨迹。

正式源码的唯一执行选择变化为 `select_policy` 默认开启协同前段，保留关闭
入口供对照。`ADOPTION_AUDIT.json` 核对34个非policy模块的函数AST与被测
快照一致，以及默认选择产生的constexpr与七档和S1/S5实际执行值一致；另有
中文说明更新和删除无用import，没有在采纳时混入新的核函数算法。

## 测试入口失败记录

S1/S5首轮在argparse阶段遗漏bootstrap追加的 `--device`，两任务exit2；
第二轮独立测试目录未包含被导入模块，exit1。两轮均未执行算子，属于本轮
测试脚本问题，记录在 `query_cli_v1_failure`、`query_import_v2_failure`。
第三轮保存原纯Native入参绑定函数到 `query_inputs.py` 后完成上述回归。
没有将这些失败登记成PyPTO/Simpler/PTOAS问题，没有修改依赖仓。
