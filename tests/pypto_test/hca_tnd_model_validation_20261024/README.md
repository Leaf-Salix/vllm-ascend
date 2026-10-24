# HCA TND完整模型接入验证，2026-10-24

用户确认：PTO不使用ACLGraph仅针对单算子性能测试；整模型接入允许并要求验证真实图重放。
本轮参考CSA-MPMD-TND，但只替换20层C128 HCA，21层CSA及SWA/FFN仍Native。
模型使用显式新架构 `PyptoHCATNDDeepseekV4ForCausalLM`，不悄悄切换旧HCA入口。

## 当前进度

2026-10-24后续精度诊断已记录于[HCA_NUMERICAL_DIFFERENCES.md](HCA_NUMERICAL_DIFFERENCES.md)。
用户明确：**以decode答案是否满足题意验收，不强制与Native逐bit或逐token一致**。
已定位Q/KV融合省略五处Native BF16舍入边界；恢复边界仅是隔离诊断候选，未采用。
正式算子继续保留FP32融合。已有192token答案的客观核对见[ANSWER_AUDIT.md](ANSWER_AUDIT.md)，
两侧都存在明确错误及未完成答案，不能把Native token一致率解释为答案正确率。
本轮任务均已结束，不继续为逐项对齐追加整模型测试；完整答案质量尚未验收，未扩B24。

单卡v04已通过，任务 `task_20261024_132445_31180867686`，exit=0。
6类场景：128K/B16/S6、混长T56、B24/S6、B16/S1、B16/S5、B1/S6。
共29个真实ACLGraphWrapper重放步骤，覆盖131071/131073/131199历史、C128关闭边界、
7请求尾批、1请求尾批、padding、混长请求重排及A→B→A。
输出及全部cache/state对独立PTO eager参考逐bit一致，保护区通过；Python提交数
在replay时不增加。每类场景的完整Native输出沿用atol/rtol=0.01通过。
这里不代表Native cache/state严格逐bit一致，也不代表整模型token已经通过。

16卡B16任务 `task_20261024_132810_32584376495` 已exit=0，功能与真实图重放通过，使用冻结v04。
正式权重为 `/data/model/DeepSeek-V4-Flash-0731-w8a8`，TP1、DP/EP16、DSpark出5验6，
每rank16个不同问题，每请求192输出token，256请求共49152token，零抢占。
每rank有33个真实满批S6 step，以及18个真实尾批step，其中15步有padding；
全部51个真实长历史FULL step的20层HCA都replay，未发生Python重新派发。
另有1步正常S1 eager提交，及34步等待其他DP时的短dummy图，不能把dummy算进业务吞吐。

**B16初轮观察，未提交B24。** 对Native：64/256请求完整token一致，简单请求32/32一致；
复杂请求192/256存在差异。q08/q12/q14从第0个输出token就有分歧；后续诊断已确认
首个HCA层输入相同、S1 eager阶段即有数值差异，QKV的舍入边界为已证实来源之一。
HCA各DP rank输出完全相同；平均接受草稿数3.753067，Native参考3.806502，DSpark事件并不全相同。
不能仅凭功能和图重放通过，或单层0.01输出容差通过，宣称整模型答案质量已通过。
详细输出见 `B16_COMPARISON.md/json`。初轮曾停下讨论，之后按用户“先查精度问题”
完成首token与QKV诊断；最新验收口径见本节开头，历史数值保留不改写为正确率。

当前适配仍在冻结候选和 `MODEL_ADAPTER.patch`，正式模型入口尚未切换。

## 接入内容和边界

- `integrate.py`仅在冻结包增加模型选择及最小框架适配；当前生产HCA计算代码未改。
- `candidates/graph_service.py`复用Native权重/页表/slot与原HCA MPMD根入口。
  C128 compact producer按图容量执行，device L屏蔽dummy；不回读Q/L，不展开元数据。
- 使用框架已有TND图合同槽位，检查地址、容量及K128/K512、Q_B/O_B策略。
  捕获Native/空路径不能作为PTO图使用；图合同失败按正常服务重新派发并留证据。
- 图预热历史与真实服务范围一致，避免在短dummy上冻结错误策略。实际decode调度不改。
- `observer.py`区分捕获时选择、生成时真正replay和Python重新提交，记录真实请求数、
  token数、padding容量、20层路由、空调用及显存；不读取device输入、不插入forward同步。
- 所有SDK仍使用validated的 `csa-shared-20261024-group-dims`，没有修改共享SDK。
  冻结以已提交HEAD为底，不带入CSA会话正在修改的测试文件。

## 容量调整和失败尝试

1. v01：队列注入`--device`而新测试CLI未接收，入口解析失败；未执行算子。
2. v02：直接沿用CSA的`[16,256,16,16] MiB`不适合HCA；预热ring2申请22044672B，
   大于16MiB，设备日志明确报heap exhausted，尚未进入graph capture。
3. v03：ring2增加至64MiB，C04及C08的11个重放步骤通过；B24预热ring0申请
   20372480B，大于16MiB。原数据保留，不能将部分通过记成整组PASS。
4. v04：HCA使用`[64,256,64,64] MiB`、task_window4096，6类29步全部通过。
   调整的是运行时容量，不是SDK或算法修正；是否适合16卡显存预算由整模型实测决定。

## 整模型口径

复用已存在的P阶段离线KV：
`/data/pyptouser/qinchuanyu/pto-eager/csa-repro-20261021-clean/kv-repro/mixed128k_fixed`。
只读使用既有128K前缀与问题后缀，不重新预填充、不做hash扫描。

功能阶段沿用CSA已经跑通的 `FULL_DECODE_ONLY`、npugraph_ex、NZ2、level1确定性、
HCCL_DETERMINISTIC=true、AIV展开、event_work_mode=1、EPLB关闭。
静态编译及SuperKernel按参考功能流程关闭；本轮不做整模型性能比较，不能与前一轮
开启全部优化的Native单算子性能混算。B16预算0.92/HCCL1800，B24预算0.935/HCCL768。

`check_model.py`要求20层HCA实际为TND、21层CSA保持Native、满批真实B×S6、
完整192token、零抢占；FULL请求的20层必须实际replay，不能以FULL标签或捕获期
命中冒充。`compare_model.py`逐请求比较Native token、首处分歧、全文和DSpark事件；
不会自动把CSA另一会话接受的复杂问题差异当作HCA答案质量通过。
token与接受事件的比较仅作诊断；后续答案验收须使用题意真值并区分输出截断。

命令、任务号和阶段状态见 `jobs.json`、`STATUS.json`，小型结果/日志在 `evidence/`。
大快照、二进制、模型权重及原始cache保留本地，提交路径和版本记录。
后续详细中文commit，仅本地提交；push须用户另外明确指示。
