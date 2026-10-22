# CSA 单 SPMD 原生 TND 实现记录

本文件单独记录 TND 工作，不改写 `DSV4_CSA_SPMD_OPTIMIZATION_LOG.md`。
后续按实际执行顺序追加改动、命令、任务编号、结果与未完成项；只有原始证据
覆盖相应范围时才标记通过，不把旧 S6 结果当成混长、连续状态或服务接入通过。

## 阅读入口与当前状态

2026-10-22 按用户要求，将混长对照、下一轮优化目标及优化过程统一归入本文件，
保留当时的输入、计时口径、全部样本、正负结果和证据索引，不再维护三个独立过程文档。

- [全 S6 与固定混长 B16 对照](#csa-tnd-s6-mixed-b16-20261022)
- [v122 之后的九档优化目标：原始计划与约束](#csa-tnd-next-10pct-goal-20261022)
- [v122 之后的九档优化过程：截至暂停的全部记录](#csa-tnd-next-10pct-log-20261022)
- [后续性能测试规范：八档、六列、至少十个有效窗口](CSA_PERFORMANCE_TEST_SPEC.md)
- [新八档核内与核间优化目标及执行记录](#csa-tnd-eight-case-10pct-20261022)

旧九档优化 Goal 已结束追踪，九档 10% 目标没有达成。用户随后建立了新的八档
核内与核间优化 Goal，目前为 **active**；其冻结基线与进度见上面的新八档入口。
下文的 active、pending、未授权 push 等描述是各条记录形成时的状态，按时间顺序阅读。
此前文档归并及用户授权的累积提交推送未恢复当时的优化，也未将历史五窗口结果改标为
新规范要求的十窗口结果。新八档测试规范与旧九档优化验收分别保留其口径。

## 本轮目标与边界

- 目标：单卡 TP1、现有模型/dtype/量化/分页 KV 布局，单 SPMD 完成
  `HC_pre → input RMSNorm → CSA → HC_post`，活动请求 query 长度为 1～6，
  同 batch 可以不同，直接使用 Native Q/L/positions/页表/slots/紧凑 metadata。
- 入口仍为 `hidden[T_capacity, 4, 4096]`；B、token 容量、有效 token 数独立。
  请求归属从 `query_start_loc` 获得，不以固定 S 或 T/B 反推，不补成 B×6
  再调用旧核。内部 tile 的对齐及尾块处理允许保留。
- 明确历史区间、范围外回退与空请求行为；支持范围内必须证明真实服务命中
  新 SPMD，不能将 Native 回退的成功记为新实现通过。
- 覆盖输出/Top-K/score/cache/state、写入保护、同地址图重放、连续 decode。
  保留原 S6 七档并新增128K/B1，共八档性能回归，计完整 L1 首尾及新增前后处理成本。
- 用户最新修正为相对另一会话最新已采用的单SPMD BSH性能，八档
  `T_TND/T_BSH`几何平均不超过0.90，逐档报告。此前仅做性能回归不再作为
  最终完成条件，不继承旧30%目标。
  不扩展 prefill、混合 prefill/decode、新并行方式、
  ACLNN 或 SuperKernel。
- 设备执行全部经 `task-submit`，不修改共享工具链或其他会话 worktree。

用户目标原文保存在
`tests/pypto_test/csa_spmd_tnd_20261020/GOAL.txt`。

## 基线

- 仓库基线：`99e29cb376c527d8ab625b4b112d77bc6bfbfcfa`。
- 隔离分支：`feat/csa-spmd-tnd-20261020`。
- 隔离 worktree：
  `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-spmd-tnd-20261020`。
- 源包：`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd`。
- 新包：`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd`。
- 37 个 Python 模块从基线原样复制，复制时逐文件字节核对通过。
  原包保留为同轮基线，不在原包内开展本轮改动。
- 源文件 Git blob 和初始工具链位置/revision 记录在
  `tests/pypto_test/csa_spmd_tnd_20261020/baseline_manifest.json`。
  正式运行工具链的revision、二进制SHA256与加载位置另存
  `tests/pypto_test/csa_spmd_tnd_20261020/runtime_manifest.json`。
- 本轮设备证据及失败诊断见下文；历史S6记录不代替本轮新场景验证。

## 验收清单

| 项目 | 当前状态 | 完成证据 |
| --- | --- | --- |
| 新包、独立 worktree、基线固定 | 已建立 | baseline_manifest.json；37 文件字节比较 |
| TND 请求边界及尾组代码审计/修正 | 已实现，已有设备覆盖 | v8八个metadata用例、v9连续状态 |
| 原生绑定与可选择服务路由 | 单层正式路径通过，未运行完整模型服务 | v10～v17实际prepare钩子、custom-op及框架图 |
| 短历史/历史范围/空请求与回退合同 | 已测边界及空/短回退通过 | L8192/32768/32772/1048576；v10混合1～64K历史 |
| S1～S6、混长、非整 tile 尾部 | 已覆盖 | v8八个独立用例通过；Native既有差异单列 |
| C4 边界、物理页、padding、请求变化 | 已覆盖 | v8/v9，T实际21/容量41、B容量12 |
| 同址图重放 | 独立SPMD与框架变metadata均通过 | v8、v13框架内Q/positions/物理页表更改；Python不重提交 |
| 连续多步 decode | 8K及128K四步通过 | v9/v16各分支保留上一步cache/state |
| B1 S1～S6及容量尾部 | 最新a241fa61数学及原生150tick下通过 | v37纯基线、v43等待裁剪；固定T容量6，实际Q为1～6 |
| S6 八档与代表性混长性能 | 最新八档完整；10%目标未达 | v41/v47等价候选GM=0.993346；v18混长、v19适配成本单列 |
| 支持范围、命令、任务号、交付检查 | 待完成 | 最终逐项核对 |

S6 八档：128K/B1、B4、B8、B16、B24、B28；8K/B16、B32。
用户允许B1单独分档；先测默认策略，若调整则额外覆盖B1的S1～S6。
首轮混长样例包括 `[2,6,1]`、`[1,2,3,4,5,6]`、`[6,1,5,2]`；
这些是验证点，不是实现白名单。

## 过程

### 2026-10-20：建立隔离基线，审计服务入口

已读取用户附件与仓库 AGENTS.md，goal 状态为 active。适用目录没有额外
AGENTS.md。已读取 incore-profiling 技能及其 canonical guide：单核模拟器
trace 不代表本目标要求的完整设备 L1，正式性能须复用真实输入的设备计时。

确认的缺口：

1. 旧 `NativeCSACall` 强制 `tokens == batch * 6`，服务 eligibility 与图重放
   也有均匀 S6 判据；不能直接作为 TND 绑定层。
2. `models/pypto_deepseek_v4.py::prepare_csa_model` 当前要求多 task 包的
   `CSAOperators`、`decode_csa` 和 `qkv_proj_rope`，新 SPMD 需独立接入分支。
3. 新包继承的 request_geometry、Indexer group、Compressor pool 已按 Q/L
   计算范围；仍需检查下游消费者、图容量和真实服务元数据的一致性。
4. `policy.select_policy` 对 `max_seq_len//4 < 2048` 尚不支持；
   本轮需要明确支持范围与回退，不能宣称全历史已支持。
5. 旧服务图的 S6 guard 同时用于 CSA/HCA；修改时必须按所选实现区分，
   不能放开旧核或 HCA 的输入合同。

本轮尚未发布功能或精度通过结论，尚未提交 NPU 任务。

### 2026-10-20：新绑定、真实混长 fixture 与首轮设备诊断

- 新增 `deepseek_v4_flash_csa_spmd_tnd/native_adapter.py`：直接绑定 Native
  根 Tensor 和分页存储，不调用带 S6 限制的 `NativeCSACall`，不回读 Q/L。
- `dsv4_csa_single_layer.make_fixture` 增加可选 `query_lengths` /
  `history_lengths`，按真实前缀和、positions 调用 Native builder；默认
  等长输入保持原值。新探针用 `[2,6,1]`，B=3、T=9、history=8192。
- 新探针 `tests/pypto_test/csa_spmd_tnd_20261020/run_probe.py` 复用已有
  Native 数值/状态/保护/图稳定性诊断，并调用新生产绑定函数。这仍是算子
  诊断，不是服务接入或连续 decode 完成。
- 41 个 Python 文件 AST 解析、shell 语法和 `git diff --check` 通过。
- 选用既有复现工具链：PyPTO `pypto-csa-repro-20261019`（本地 revision
  `2dfa7f120ba5e577566046d920845aa241bfa7bb`，检查时 tracked diff 为空）、
  Simpler `simpler-csa-core-args-256-20261016`、PTOAS
  `ptoas-csa-repro-20261019/build`。测试 bootstrap 增加显式 workspace-root，
  修复嵌套 worktree 下把 `.worktrees` 误作总工作区的问题。
- 本地 Native 扩展通过链接复用原 release 构建；`_build_info.py` /
  `_version.py` 复制原构建生成信息。这些是忽略的本地运行文件，不作为源码
  或新的 Native 构建结果提交。

| 任务 | 用例/结果目录 | 实际结果 |
| --- | --- | --- |
| `task_20261020_234434_41806820769` | `runs/mixed_261_h8192_v1` | exit 1；导入缺少 `_build_info.py`，未执行 SPMD |
| `task_20261020_234611_48699728663` | `runs/mixed_261_h8192_v2` | exit 1；Native 权重准备前 CANN 初始化失败，plog 明确为 ASCEND_CACHE_PATH 目录未建立；未执行 SPMD |

修复第二项启动脚本后再提交新任务，原始结果与错误保留。当前不能据这两项
初始化失败评价 TND 算子的正确性或性能。

### 2026-10-21：TND服务候选与CPU合同检查

已实现但尚未设备验收的部分：

1. `PTO_CSA_VARIANT=spmd_tnd` 显式选择新包，默认仍为原多task。
   `prepare_csa_model` 对该实现使用 Native JIT 入口，不再要求多task包的
   `CSAOperators/decode_csa/qkv_proj_rope` 接口。
2. 新 `service.py` 从各组 Native Host 描述符判断资格，允许 `[2,6,1]`
   等混长；绑定 Q/L、页表与缓存，并沿用 connector 等待/发布和 metadata
   跨层复用。按描述符与策略准备工作区，`rebind` 不重新清零信号代际。
3. `policy` 移除 `T_capacity <= B*max_query_tokens` 的错误限制；
   实际有效 query 由 Q/L 定义，图输入尾部容量不能当成有效 token 数。
4. 空 token/全 dummy decode 在 custom-op 边界安全返回输入值，不访问
   Native cache/state，不启动全核等待。设备验证仍待完成。
5. 当前候选服务显式保留 `max_seq_lens`（含本轮 query 的未压缩长度上界）
   为 8192～1048576 的策略范围；更短历史回退 Native，未冒充 Vector Score
   已接通。混合历史、区间中间点和边界仍需设备验收。
6. 完整模型图合同尚未实现，因此配置暂时明确拒绝新实现的模型图模式；
   这是一项待完成工作，不把 eager 接入或独立 kernel 图等同完整图目标完成。
7. 新包移除未被运行入口引用的历史 `query_source.py`（等长多task源码生成
   辅助工具）；原包及原工具保留。它不参与新单SPMD数学和现有优化。

CPU证据：

- 初始绑定测试：6 passed，覆盖真正混长描述符、零复制别名、禁止
  `.item/.cpu/.tolist/.numpy` 回读以及错误描述符。
- 新服务范围/资格判定 + variant选择 + 旧 S6 服务回归：
  **57 passed**，原始结果 `runs/cpu_contract_v2.xml`。
- 真实 `dsv4_csa_forward` Python 路由、Native connector 顺序、warmup到
  capture的同描述符工作区复用，以及范围外回退：
  **1 passed**，原始结果 `runs/cpu_route_v1.xml`。
  此项使用CPU提交桩，不能作为NPU命中证据。
- 复现入口为 `bash tests/pypto_test/csa_spmd_tnd_20261020/run_cpu.sh ...`。

设备任务：

- `task_20261020_234958_67318423765`：
  `runs/mixed_261_h8192_v3`，Q长度 `[2,6,1]`，目前排队。
- 2026-10-21 00:03 左右队列查询显示 16/16 卡由 `ci-runner` 占用，
  本任务 pending；未取消他人任务、未绕过队列占卡、未重复提交同一待运行任务。
- 启动器现在在任务开始时固定算子与fixture源码副本及 SHA256，避免等待期间
  工作区继续修改后无法确定实际运行版本。正式结果以该任务的源码manifest为准。
- 任务清单保存在 `tests/pypto_test/csa_spmd_tnd_20261020/jobs.json`。
  清单中的上次状态只是记录；继续工作时须查询对应 task-id 的实时状态。

下一步：设备上先查混长数值及状态，补齐真实服务调用；继续实现并验证模型图
捕获/重放合同、请求变化和连续状态轨迹，然后再做S6七档与混长L1性能对照。

### 2026-10-21：首个真实混长设备结果，精度未通过

`task_20261020_234958_67318423765` 已执行结束（exit 1），原始结果在
`runs/mixed_261_h8192_v3/result.json`，实际运行源码及 SHA256 在同级
`mixed_261_h8192_v3_source/`。本次 B3、Q=`[2,6,1]`、T9、各请求 history=8192，
使用正式第4层权重和 Native TND builder。不是补成 B×6 的计算。

已观察到：

- 完整SPMD执行结束，输出有限；Native页写入范围、只读metadata和保护检查通过。
- 两次固定输入同地址图重放均与候选eager精确一致，信号代际递增且地址不变。
  这两次没有改变Q/L，不算动态metadata或连续decode验收。
- 对Native的输出逐元素 `atol=0.01, rtol=0.01` 门槛仍失败：
  147456个元素中17个超限，max_abs=0.015625，RMSE=0.0028023490799762288。
- Top-K无非法/重复索引，但9行均有集合差异，共26个替换索引；
  不能把结构合法等同选择规则通过。
- Native cache/state也有非零差异；Indexer scale精确一致，其余逐项数据保留
  在原始JSON，未统一标为精度通过。

下一项诊断先在相同Native输入上比较原单SPMD基线与TND候选，判断遗留数值
差异和新改动的归属。该比较只用于定位回归，不把原多task的固定S6核改作
混长golden，也不替代Native/独立参考的最终精度门槛。本轮未放宽任何阈值。

### 2026-10-21：图重放合同检查与原SPMD归属诊断准备

- 新增每个runtime持有的图合同：捕获时记录根Tensor描述符/地址及策略；
  同地址Q/L数值变化不做device回读，描述符/地址/策略变化则回到正常服务分派。
  捕获为Native回退或空调用的图不直接重放到新工作量。
- ACLGraphWrapper在重放前核对该合同，失配时临时使用NONE模式运行，随后恢复
  context。新TND模型图配置仍暂时拒绝；此CPU合同不是完整模型图设备验收。
- 最新相关测试 **34 passed**（新TND合同和旧HBG回归），结果
  `runs/cpu_graph_contract_v1.xml`。没有NPU分配。
- 诊断探针增加原单SPMD的同输入、同工具链比较，覆盖输出、TopK/score、
  cache/state及保护区。新旧源码都在任务执行时快照，原Native门槛保持独立。

### 2026-10-21：原单SPMD对照确认与服务设备诊断

- `task_20261021_002804_315127125659`，目录
  `runs/mixed_261_h8192_attribution_v4`，completed/exit 1，无执行异常。
  `[2,6,1]`、B3/T9、history8192：候选与原单SPMD的输出、TopK/score、
  全部cache/state逐项零差异，保护区通过。原单SPMD对Native也有同样的
  17个输出超限及26个TopK替换。这只确认差异归属，不撤销Native精度门槛。
- ACLGraph通用CPU回归初次为19 passed/7 failed；失败均因旧测试模拟
  ForwardContext未提供Ascend实际附加的additional_kwargs（原HBG代码已依赖）。
  补齐测试fixture后 **26 passed**，结果 `runs/cpu_acl_graph_v2.xml`。
- 已准备正式模型初始化钩子 → Torch custom-op → 新TND服务的单层NPU测试，
  不替换设备提交。包含与手工算子输出对照、固定输入图重放、空调用、真实
  2048短历史Native回退；它仍不代表完整模型服务或变metadata图已验收。
- 同一任务增加HC_pre/RMSNorm/Indexer输入的Native中间结果采集，只作精度
  归因，不计性能。任务 `task_20261021_003559_34885217752`，目录
  `runs/mixed_261_h8192_service_stages_v5`，结果待收取。
- 已就“是否把原有Native数值差异一并修复”提出范围澄清。在收到明确调整前，
  保持原goal及Native精度要求，不能将原SPMD等价自动认作完整精度通过。

### 2026-10-21：用户明确调整精度验收范围

用户回复“原有差异单列，本轮聚焦 TND 正确性与无新增回归”。该指令覆盖上一段
待澄清状态：本轮不扩展为修复原单SPMD已存在的Native数值差异，原始阈值和
FAIL结果完整保留。TND语义、同输入无新增回归、实际服务、连续状态、图和
七档性能要求仍然保留。详细记录在测试目录 `GOAL_UPDATES.md`。

v5设备任务已完成（exit 1）：HC_pre/RMSNorm/Indexer输入差异证据已保存，
随后真实服务入口报 `RopeDataProxy has no attribute get`。已修复候选服务
误将Native代理视为dict的问题，加入实际RopeDataProxy的CPU测试；
`runs/cpu_rope_proxy_v1.xml` 为 **31 passed**。

另外补齐图合同中Native compressor_metadata实际捕获的start_pos、
full_compress_cos/sin，以及请求数/输出容量/block_size等Host标量，变化时
回到正常分派，避免只核对SPMD根Tensor而漏掉其Native metadata生产者。

### 2026-10-21：实际服务路径及padding容量处理

- `task_20261021_004111_372169825135` / `runs/mixed_261_h8192_service_v6`
  完成无执行异常。真实模型准备钩子和Torch custom-op提交2次（warmup与capture）；
  服务输出/cache/state与手工单SPMD逐项一致，两次图重放数值、空调用与2048短
  历史Native回退均通过。总体仍exit 1，因为测试错误比较了warmup与capture
  的根输入地址；Native compact metadata在capture中重新分配，rebind本来允许
  此变化。已改为检查工作区跨capture复用、根输入在capture之后的replay稳定性。
- Native builder按num_input_tokens初始化完整RoPE/slots，再返回按
  num_decode_tokens切短的视图。新增 `token_capacity_view` 在检查已有底层
  存储容量后返回同地址容量视图，不复制、不创建B×6输入、不回读Tensor内容。
  不足容量仍拒绝。相关CPU测试 **32 passed**，结果
  `runs/cpu_capacity_views_v1.xml`。实际padding设备证据仍待下项任务。
- fixture增加独立token/request容量、失活dummy、容量尾部与稳定物理请求槽；
  默认无padding调用保持原数据值。输出与TopK只比较有效token，所有cache/state
  及保护区仍比较完整分配；dummy输出不作为有效token验收。
- `task_20261021_004754_404474221201`：
  `[1,2,3,4,5,6]`，T实际21/容量41，请求实际6/容量12，history8192，
  验证Native输入、C4关闭组/偏移、新旧SPMD以及真实服务路径，结果待收取。
- `task_20261021_005142_170195321`：
  同一T41/B容量12图覆盖均匀S1～S6、两种混长、C4边界和页映射变化，
  以及4步连续decode（包含请求重排/增减）。连续阶段新旧SPMD和Native各自
  保留上一步cache/state；只在分支间换入各自的前一步状态，不恢复初始状态。
  Native数值差异单列，写界与无新增回归作为本轮检查，结果待收取。

### 2026-10-21：padding服务通过；连续参考路径故障与修复

- v7探针已到 `stage=complete, passed=true`：T实际21/容量41、请求实际6/容量12，
  新旧SPMD零差异，独立C4几何、保护、真实服务、捕获后地址稳定、工作区复用、
  空调用和短历史回退均通过。队列终态却为exit 2：Python正常完成后Bash读取
  了执行期间被编辑的启动脚本余段，报unexpected EOF。该工具错误不能改记exit 0，
  也不改变已落盘的逐项检查结果。
- 已新增 `submit_probe.py`，提交前固定独立launcher副本；launcher还使用exec
  替换为Python，避免结束后继续读取变动文件。实际launcher纳入源码manifest。
- v8的8个独立metadata用例（S1～S6及两种混长）全通过，同一张图、T41/B容量12，
  各次Q/L/positions/页表变化后的输出、TopK/score、cache/state、新旧回归、
  C4几何和保护均通过。
- 连续第0/1/2步通过；第3步进程abort、队列exit 134，不能记整轮完成。
  plog定位故障kernel为Native `SparseAttnSharedkv_..._1026`，MTE非法GM地址；
  不是SPMD kernel。原始故障转储已保存在
  `runs/tnd_matrix_h8192_capacity41_v8/native_exception_dump/`。
- 原测试给Native eager的Q端点仍覆盖dummy容量41，但实际SparseAttn Query只有
  有效15行。参考路径改用有效TND行/活动请求的Native描述符，并保持同样完整
  物理cache及各自上一步state；被测SPMD仍使用T41/B容量12。先修参考输入合同，
  不据此直接声称Native组件有bug，也不把失败步骤计作通过。
- `task_20261021_010447_7230338877` / `runs/tnd_continuous_unpadded_native_v9`
  聚焦恢复连续链路，不重跑已通过的8个独立metadata用例。
- `task_20261021_010640_83025121361` / `runs/mixed_history_1_to_64k_raw_service_v10`
  用query `[1,2,3,4,5,6]`、history `[1,3,128,4096,32766,65536]`，
  继续检查混合历史、策略边界、新旧回归和服务。此项关闭测试工具的等待退避改写，
  验证默认生成代码的实际设备路径，结果待收取。
- 新TND模型现在沿用框架FULL_DECODE_ONLY图选择，并去掉额外的S6判据。
  图包装器按TND合同安全分派；未将混长伪标为uniform。旧CSA/HCA检查保留。
  CPU的服务/runner/图包装器合并回归 **78 passed**，
  `runs/cpu_tnd_runner_graph_v1.xml`；完整模型图设备验收仍未完成。

### 2026-10-21：连续状态、生产等待协议、框架图及新增B1

- `task_20261021_010447_7230338877`（v9）completed/exit 0，四步连续decode
  全通过；各分支使用自身上一轮cache/state，包含请求移除、重排和新增。
  v8已通过的八个独立metadata用例没有重复执行。
- `task_20261021_010640_83025121361`（v10）completed/exit 0：混合历史
  `[1,3,128,4096,32766,65536]`、query `[1,2,3,4,5,6]`、T容量41，
  默认生成的等待协议（无测试退避改写）下，新旧SPMD完全一致；真实服务、
  固定输入图、空调用及2048短历史Native回退通过。
- `task_20261021_011523_125530925293`（v11）completed/exit 0：含query最大
  长度32768，首次实际经过框架ACLGraphWrapper捕获/重放，单层正式custom-op
  与原SPMD结果一致。框架空路径合同失配分派通过。此项不是完整模型端到端测试。
- runner现在对范围外真实调用在图padding之前选择eager，避免将dummy Q端点
  交给Native eager。新TND/旧服务CPU回归56 passed，
  `runs/cpu_native_fallback_before_padding_v1.xml`。
- `task_20261021_011532_126446428080` / `runs/l1_s6_h131072_b4_raw`
  completed/exit 0；完整L1五窗口中位数：新TND 677.8us、旧SPMD 664.8us、
  多task 508.9us。新旧相差+1.96%，数值、图、保护区和通知代际通过。
  本轮不以旧工具的30%目标字段作为验收条件。
- 用户新增128K/B1并允许独立分档，性能矩阵扩为八档，详见GOAL_UPDATES.md。
  `task_20261021_012955_201076229165` / `runs/b1_h131072_service_raw_v12`
  先测默认G2策略的正确性、正式服务和框架图；尚未采用B1特化。
- 其余六个原有性能档位已提交，任务号及完整命令见jobs.json。
- 核查策略实际分界为`max_seq_len // 4 > 8192`，首个长历史长度为32772，
  不是32769。v13补此边界并在同地址更新Q/L、positions、页映射后通过框架图
  重放；另测服务compact生产者及包装器成本。v14采真实混长[2,6,1]的配对L1，
  v15验1M上界/B1，v16补128K连续状态的默认等待协议。结果待收取。

### 2026-10-21：八档回归收齐、混长归因与新10%目标

- 原八档raw TWAIT回归全部exit0，新旧输出/Top-K/score/cache/state完全一致，
  图、保护区、信号代际通过。完整L1逐次样本见RESULTS.md和原始result.json。
  128K/B1新增默认策略为G2：原SPMD690.58us、TND738.44us；其余七档耗时变化
  为-2.82%～+2.43%。这是原复现工具链的回归表，不作为随后新增的BSH目标基线。
- B1 G6试验task_20261021_014721_29211738823通过功能，但同轮原G2为696.26us、
  G6为759.62us，未采用。B1专用分档许可保留，当前生产policy未修改。
- 真实混长[2,6,1]首轮v14为原564.94us、TND634.48us。发现探针两侧工作区分配
  不同；生成的AIC/AIV及完整PTO C++去掉行注释后完全一致，证据
  mixed_l1_codegen_equivalence.json。v18使用同类保护分配，并控制全部GM工作区
  地址后，原592.60us、TND579.36us，全部数值/协议检查通过。首轮差距保留，
  不据此声称发现TND计算逻辑退化；地址布局/运行环境敏感性仍须同轮控制。
- v13（task_20261021_013537_233869216938）在L32772首次长历史策略下通过：
  同一ACLGraphWrapper中Q由[1..6]变成[6..1]，同步更换positions/物理页表，
  根地址保持、Python没有重新提交，结果与原单SPMD逐项一致。
- v15（task_20261021_013537_23389993243）通过L1048576上界/B1；
  v17（task_20261021_013853_252063030343）通过L8192下界/B1/S1。
  v16（task_20261021_013538_233941623733）通过128K四步连续状态，使用默认等待
  协议，包含请求增减/重排且各分支沿用自身上一轮状态。
- v19（task_20261021_014806_295304419774）只补服务成本，复用v13功能证据：
  两侧已处在同一个forward context，排除测试额外eligibility检查。十次稳态
  中位数，直接图/正式服务图的NPU事件为825.28/1109.10us，Host提交为
  27.10/317.93us，同步墙钟为875.95/1163.48us。事件可能含Host提交间隙，
  不能将283.82us事件差解释成纯设备metadata kernel耗时，也不能替代L1。
  当前图合同描述符检查确有Host成本，后续性能报告保留此单列项目。
- 合并CPU合同/旧CSA/HBG/ACLGraph/variant检查99 passed，ruff检查与diff检查通过。
- B1固定T容量6、实际Q长度1～6且没有dummy请求吸收尾部的专用矩阵已写入
  metadata_matrix.py（--b1-metadata-matrix）；尚未提交设备测试，不预记PASS。

用户新目标已经明确：以另一个CSA会话最新已采用的单SPMD BSH性能为基线，
八档S6的TND/BSH耗时比等权几何平均<=0.90，逐档报告；见GOAL_UPDATES.md。
该会话正式SPMD的主要差异是已采用150tick等待，而本会话此前使用默认TWAIT。
已只读核对该会话记录，冻结683a701f2706766bef102a3a7bf259542d6b3cf0对应
源码到bsh_baseline_20261021，并记录bsh_baseline_manifest.json。
BSH参考工具链为PyPTO65477973435f7d717ffaa7f602f4783ad5310417、
PTOAS b48c84ebfea88c58439181e76f5e00fd12950418、原Simpler57e49923。
尚未采用的Top-K成对压缩候选不并入参考。

为保证服务与性能采样使用同一等待机制，已在新的独立PyPTO/PTOAS工作树
实现显式backoff_ticks/backoffTicks传递的候选；默认0保持原TWAIT，正值只作用于
A3单元素INT32/GE信号，DCCI/volatile读取与写侧/payload维护保留。该功能尚在
构建和测试，未接入TND生产signals.py，也没有宣称性能收益。最初复制的PyPTO
二进制未使用，读取本地worktree构建规范后已移除，当前从本工作树重新构建。
PTOAS构建同样从当前源码执行；CANN CMake路径初次定位失败已修正，原错误log保留。

后续性能候选的只读审计：cube_schedule可达32个包内helper，除signal_notify
外没有pl.write，见cube_scalar_write_audit.json。可进一步核对生成代码，研究
纯Tile/FIX生产阶段是否可省略全Scalar DCache发布维护；目前没有删除任何同步，
不能仅凭AST记录宣称该优化安全或有效。新10%目标尚未达成，goal继续active。

### 新BSH基线下的B1尾部验证、等待API和首个性能负例

- `task_20261021_022330_182138327918`，B1固定T容量6、Q实际S1～S6、
  无dummy请求吸收尾部，六例均通过；C4、Top-K合法性、cache/state边界、
  同址图及原单SPMD精确回归通过。使用最新已采用BSH工具链及150tick，实际
  CCEC源审计通过。结果 `runs/b1_s1_to_s6_capacity6_bsh150_v20`。
- PyPTO新worktree已从本源码构建并增量重建。首次CPU测试因共享editable finder
  误加载主checkout，12失败不作为本修改结论；显式startup finder同时作用于
  xdist worker后，**274 passed**，`runs/pypto_native_wait_unit_v2.xml`。
  测试期望没有修改。PTOAS从本工作树构建仍在运行，新原语的实际CCEC/NPU
  证据尚未取得，生产TND signals.py还没有使用新backoff_ticks参数。
- Cube纯Tile发布候选只保留一个GM fence、PIPE_ALL及信号自身维护，省略全
  Scalar cache维护和其后第二个fence。两档完整数值/协议通过，但L1未改善：
  B1参考400.04/候选404.72us，B16参考778.62/候选786.24us；均不采用。
  任务022555_195423132504、022555_19542743018，原始源码和窗口完整保留。
- 下一候选从生产TND重新派生（不叠加上述负例），仅在INPUT、QR、ROPE三个
  阶段省略worker>=ceil(T/8)的无工作核发布。生产者按八行分工，所有消费者
  仍按八行owner等待；保留所有核begin_invocation及活跃核完整发布/获取。
  B1/B16任务023804_33906131467和023804_33959626400正在运行，未预记收益。
- 当前恢复点、构建句柄、任务和后续约束保存在测试目录recovery_state.json。
  目标保持active，未达10%几何平均门槛，没有新增性能方案进入生产目录。

### 空角色通知试验收取与等待依赖试验

- v22 两任务均 completed/exit0，输出、Top-K/score、cache/state、保护区、图与
  代际检查通过。128K/B1 的 BSH150/TND 中位数为 401.12/390.44us（-2.66%）；
  128K/B16 为 787.02/796.76us（+1.24%）。没有作为通用策略采用，五个窗口均保留。
- v23 从生产 TND 重新派生，不叠加 Cube 发布精简或 v22：保留全部通知，仅让
  QR、Indexer query RoPE、head weights、query量化、coefficients、compact量化
  的空循环角色跳过相应输入等待。所有权分别沿原 M8、M8×16、M16、token、
  device group count、request 分工；有工作的角色仍执行完整 acquire。
  B1/B16 任务为 `task_20261021_025144_402125724285` 和
  `task_20261021_025144_402140122430`，结果未收齐前不记收益。
- 新增 `evaluate_s6_cohort.py`，要求显式给出同一冻结候选的八个任务，核验
  S6集合、源码、基线、运行工具二进制、BSH150审计与全72核窗口后计算等权
  几何平均。它只判性能门槛，不替代混长、连续状态和真实服务验收。
- 原生等待候选独立保存在 `runs/native_wait_candidate_v1`，只把
  `signal_wait(..., backoff_ticks=150)` 显式传入编译器；尚未并入生产目录。
  新工具链选择和测前/测后指纹核验已加入测试入口，等待 PTOAS 自身构建完成。
- PyPTO 扩展回归（已通过的274项不重复）为 13890 passed、18 failed、
  4 skipped、1 xfailed。失败明细保存在 `runs/pypto_regression_failures.json`；
  正在针对原冻结源码和本机Node版本归属，不把该次完整测试标成PASS。

### 等待依赖结果、最新阶段时戳和容量特化

- v23 两档均 completed/exit0，完整精确回归、保护区、图和代际检查通过。
  B1 的 BSH150/TND 为 397.90/379.10us（-4.72%），B16 为
  780.02/781.86us（+0.24%）。尚未采用；B1可进一步作为单独策略验证，
  不能凭此宣布八档几何平均达标。
- CPU失败归属完成：Node 20复测原失败的11项全部通过；剩余7项在未加本轮
  backoff参数的冻结PyPTO基线上同样失败（注册清单4项、multi-buffer断言、
  symlink导入、torch版本错误文案各1项）。原FAIL保留，证据
  `pypto_cpu_regression_attribution.json`，没有未归属的新失败。
  PyPTO C++变更行的格式检查通过；只调整缩进后已增量重建。
- 两个v24阶段探针在初始化时失败：新增工具指纹采集过早导入PyPTO Torch扩展，
  触发private backend重复注册，尚未运行候选。已改成测前只读文件指纹、
  测后核对实际已加载模块；不改共享Torch环境。
- v25最新阶段诊断均 completed/exit0：
  `task_20261021_030030_161940329808`（B16）、
  `task_20261021_030030_16197517222`（B28）。完整单次功能和信号检查通过，
  使用已采用150tick；时戳仅用于归因，不作为正式性能、纯计算或引擎独占时间。
  扣除外层GM等待后的阶段中位区间：Score AIC/AIV 为 B16 205.45/189.00us、
  B28 329.88/324.67us；Attention 为 B16 137.70/141.83us、
  B28 234.08/235.65us。仍包含内部FIFO/DMA/Scalar与探针成本。
  逐核原始端点和摘要见 `current_bsh150_phase_summary.json`。
- 容量特化v26只把动态容量Tensor注解交给JIT按已有shape推导，Q/L、positions、
  页表和有效query仍是device数据；固定模型轴和NZ权重注解保持原样。
  代价是每种容量签名单独编译，尚未采用，不能称为仍共享一个动态容量产物。
  B16/B28在候选设备执行前遇到parse shape错误（96/168行与8行的row_expand_mul）。
  原失败任务 `task_20261021_030409_311114030165`、
  `task_20261021_030409_311185832136` 保留；转为CPU描述符编译定位，不反复占卡试错。
- `--prepare-only`只导出容量/dtype供CPU meta Tensor复现；`descriptors_only`不记
  候选功能PASS。`check_capacity_compile.py`按原描述符作specialize和完整编译，
  不执行NPU、不把parse成功当成完整编译或性能成功。

### 容量特化的CPU问题定位与修正

- v1错误定位到共享 `rms_norm_apply`：`x[row:row+8]` 在JIT传参元数据中错误继承
  父Tensor的96行，随后与8行inverse冲突。修正放在隔离PyPTO的通用推导中：
  仅对重复的标量名字和已知非负常量识别固定窗口宽度，不消去函数调用/间接读取。
  两个新回归在原冻结编译器上失败，修正后168项decorator及其余642项JIT测试通过；
  未修改旧测试期望。英文/中文类型文档同步更新，ruff及变更行格式检查通过。
- v1完整编译还遇到原生Indexer cache两个零复制view的维度恢复限制。
  v2保留 `idx_cache` 的动态页数/页字节注解，其余容量仍由输入描述符特化；
  没有调整数学、页表值或请求边界逻辑。CPU完整编译在
  `runs/static_capacity_cpu_b16_v4` 通过，`codegen_audit.json`确认仅一次MIX提交、
  两个角色核、无运行时新增GM分配，剩余动态维度仅为Indexer cache页数/页字节。
- 以上为CPU及代码生成证据，尚无容量特化的设备正确性或性能结果。
  后续组合原生150tick等待的候选单独保存在
  `runs/static_capacity_native_wait_candidate_v1`，生产包仍未采用任何本轮性能试验。

依赖修改的可应用补丁已导出到 `tests/pypto_test/csa_spmd_tnd_20261020/dependency_patches/`，
包含原生等待属性与固定窗口元数据推导修正。两份补丁均在对应冻结基线上只读执行
`git apply --check --whitespace=error-all` 通过；没有修改其他会话源码。PTOAS首次完整
构建已完成，格式调整后的最终增量构建及后续lit仍在进行；尚未生成最终工具链指纹，
也尚未提交原生等待/容量特化的设备性能任务。

### 原生等待的设备证据、计时修正及 B1 分工试验

- 隔离 PTOAS 全量及最终增量构建完成；相关 lit 共15项通过（v3中14项，
  正例CHECK格式修正后的v4中1项）。首次工具缺失与PATH错误未执行断言，
  原日志保留。`native_toolchain_manifest.json`冻结1473份源码/二进制文件；
  所有设备任务前后核验，不修改其他会话工具，不复制其构建产物。
- v28 原生150tick B16任务 `task_20261021_033407_263280125800`通过完整精确、
  状态、保护、图及协议验证，实际CCEC源码确认原生等待、无raw TWAIT。
  相对此前BSH150，中位数785.02→791.92us，不能称为性能优化收益。
- 静态容量+原生等待v28的B16/B28均exit0：
  `task_20261021_033407_263357822636`为780.42→777.20us，
  `task_20261021_033407_263433817616`为1159.08→1147.42us。
  收益仅约0.4%/1.0%，且增加容量签名编译数，暂不采用或扩展八档。
- 混长服务v28 `task_20261021_033407_26353694698`因测试构造器遗漏
  table_history导致changed-metadata fixture物理cache描述符不一致而失败。
  已让克隆fixture沿用原table_history，不修改生产算法或精度阈值。
  v29 `task_20261021_034309_210378126331`完成exit0：Q=[1,2,3,4,5,6]、
  history32766、T容量41/B容量12，正式单层custom-op服务路由、
  ACLGraphWrapper及同地址Q/positions/页表变更重放通过，workspace复用。
  这不等价于完整模型服务端到端推理。
- 发现旧采样器在最后一组交替窗口之间执行大规模CPU精确比较和D2H采集，
  最后BSH窗口反复增加约160～200us。将验证移至全部窗口后：先验证最后一个
  实测输出，再对两侧各做一次不计时的恢复/重放/严格比较，协议计数包含重放。
  保留全部旧窗口，不删慢样本，不放宽检查，不宣称每次计时调用都单独验证。
  v30 `task_20261021_034511_287684532064`验证通过，五个BSH窗口均无该异常：
  BSH777.94/TND790.14us。最终八档必须统一使用修正后的采样器。
- B1 G6+24叶试验v31 `task_20261021_035651_98804820155`完成exit0；
  原5叶扩为24叶以覆盖全部24个AIC，保留动态Q/L与数学。
  此前BSH150的391.76us→359.74us（-8.17%），输出、TopK/score、状态精确一致。
  沿用现有G6工作区种类但部分shape比G2大，不宣称工作区完全不变。
  仍为未采用试验；B1 S1～S6同址图任务`task_20261021_041703_209414811817`
  已提交，且需继续审计不同叶划分的同分TopK行为。
- 相邻阶段等待只保留最后一次payload获取的试验：v1因JIT依赖调用不读取
  constexpr默认值而在CPU解析失败；v2显式绑定全部ACQUIRE参数后完整编译通过。
  与原生等待版本比较，45处等待保持，整DCache维护92→87、PIPE_ALL228→220，
  无新GM分配；不跨越payload读取，最后一次获取保证有正计数。B16/B28
  任务为`task_20261021_041749_213676022003`、
  `task_20261021_041750_213690210682`，结果未确认前不记收益。
- 小T的输出列分工候选在CPU验证：仅把无跨列归约的O_B反量化+HCpost按
  (token,column block)分给空闲AIV，保留八组FP32左折叠、BF16边界及残差顺序。
  v1受constexpr局部Tensor元数据推导限制失败；v2显式同宽视图后完整编译通过。
  尚未上卡，不能记功能或性能通过。

### BSH基线推进至 a241fa61

再次只读核对另一CSA会话，发现其03:53已采用B1动态半叶/UB分数和归并根优化，
提交`a241fa6178ddad24ced30e5a93d109dfb17f5b53`。正式改动仅policy、group、
score、topk、publish五份文件；另一会话的直接配对为401.30→371.94us，
该数字只是其原始证据，不与本会话不同轮窗口相减。

新冻结目录`bsh_baseline_a241fa61`与`bsh_baseline_a241fa61_manifest.json`
保留63份文件和实际配套pair工具链信息。原683a701/7061c112快照及所有已跑
窗口原样保留，v31的8.17%不代表相对最新BSH收益，不能用于最终八档验收。
正在运行的旧基线任务继续按其冻结源码归档；新提交统一采用a241fa61。

配套PyPTO/PTOAS有成对压缩原型及full常量修正，本轮隔离工具仍冻结。
先做最新BSH在两套工具上的CPU完整代码生成比对，确认当前算子实际用到的
路径是否等价；不凭“新API没调用”直接宣布工具链一致。另从TND包建立
`current_bsh_native_tnd_candidate_v1`，只吸纳上述五份已采用数学/策略文件，
保留本轮Native绑定、服务合同和原生150tick等待，尚未进入生产实现。

目标仍active；当前没有最终冻结候选的完整八档结果，也未达到等权几何平均
T_TND/T_BSH<=0.90。最终逐档报告使用同一候选、基线、采样器和审计过的工具链。

### a241fa61工具比对与后续试验收取

- 最新BSH的B16/G6和B1/balanced-G2在本轮原生等待工具、另一会话实际pair工具
  上均完成CPU完整编译。只移除整行`// pto:`调试名称注释后，两组完整C++各自
  逐字一致，未删除运算、地址、数值、屏障或重命名变量。
  证据为`bsh_a241_tool_parity_b16.json`和`bsh_a241_tool_parity_b1.json`；
  B1的CPU描述符还与v34真实设备fixture逐项一致。这是两种已查策略的证据，
  不冒称独立执行了全部八种策略的双工具链对照。
- v32 `task_20261021_041703_209414811817`的B1 G6/24叶S1～S6同址图均通过。
  但v35 `task_20261021_043018_286907020261`用H139264/S6、历史Indexer key全零
  和正常有限scale制造同分，暴露**新增回归**：相对最新BSH有2047个TopK位置
  不同、46161个输出元素不同，max_abs=0.18359375；score/cache/state一致。
  原因与叶划分/归并同分顺序改变一致。故**拒绝该候选**，不能把v31的随机
  输入收益用于验收，不修测试阈值。最新BSH对Native的已有同分差异也原样保留。
- 相邻获取合并v33两档均完整精确/协议通过但未获益：B16为
  780.52→789.94us，B28为1144.86→1155.74us；保持未采用。
- 输出列分工v34使用最新a241fa61数学及基线，两档均exit0且精确一致，
  B1为386.78→393.96us、B4为483.90→493.82us，均不采用。
  对应任务`task_20261021_043018_286871914579`与
  `task_20261021_043018_28688479929`。不因列独立而推定更多AIV必然更快。
- 测试工具增加`--constant-indexer-history`，只改fixture历史key/scale，当前
  token的cache更新和所有保护检查仍由完整CSA执行。此开关禁止混用性能采样。
  汇总工具显示BSH提交版本；正式八档评估器拒绝同分fixture和测试等待改写。
- 初始4次立即重读、之后恢复150tick退避的v36，只在测试回调中修改原生helper，
  保留所有DCCI/volatile读取、spin上限及周期屏障；没有修改冻结编译器或生产包。
  两档均exit0且完整精确/协议通过，但B1为365.82→388.50us，B16为
  782.72→796.76us，未采用，也不为此扩展生产API。
  任务`task_20261021_043407_309493922351`、
  `task_20261021_043407_309517524123`。该试验不具备正式八档验收资格。
- 只读发现另一会话已经完成相同B1连续scale读取试验，365.66→373.14us。
  本轮CPU候选完成编译后停止，不重复上卡；保留其原始证据路径于
  `b1_scale_current_v1.json`。没有拿其不同轮次数据与本轮相减。
- 原生150tick+最新BSH B1数学的纯TND候选，正在验证S1～S6以及正式服务的
  同地址变metadata图重放：`task_20261021_044257_365360219269`、
  `task_20261021_044257_365379017711`。通过前不把旧版本结果扩大到新B1路径。
- 从该纯候选单独移入v23的空消费者等待裁剪，仅改vector_schedule，保留全部
  通知、活跃消费者获取、最新B1 UB数学及同分顺序。CPU完整编译通过。
  后续优先验证B1/B4/B8；B4/B8此前未测，不重复原B16中性试验。
  所有本节候选仍未并入生产包，八档10%几何平均目标尚未达到。

### 最新B1路径、等待裁剪区间与新的Score资源试验

- v37纯a241fa61数学+原生150tick等待的B1 S1～S6同址独立图全部通过，
  `task_20261021_044257_365360219269` completed/exit0。
  另一B1服务任务误用了“批内非均匀query反转”的专用测试开关，按测试合同
  报错`变metadata框架图用例需要非均匀query长度`，没有生产故障结论。
  保留此FAIL与原有guard；B1标准服务/ACLGraphWrapper检查改以v40正确参数
  重跑，`task_20261021_050121_6848358492`。B6混长变metadata框架图证据仍是
  v29；不能把单请求的Host max_query/压缩计数改变伪装成不变的捕获合同。
- v38空消费者等待裁剪均完整精确/状态/图/协议通过：
  B1 `task_20261021_044707_38941343957`，367.74→346.84us（-5.68%）；
  B4 `task_20261021_044707_389431625197`，487.90→491.84us（+0.81%）；
  B8 `task_20261021_044707_38944097777`，627.36→611.48us（-2.53%）。
  不作为通用策略直接采用，不将两档收益相加为几何平均。
- 进一步区间候选在既有`QA_M16_PARALLEL`域恢复原等待并去掉新增group计数
  读取，其余域保留v38。没有新增策略开关，不按B4/B8测试点枚举；这是实测
  取舍，不声称QA的M tile已被证明是收益差异的原因。v1生成脚本错误匹配了
  已有通知guard，断言在修改前阻止操作，随后CPU只编译了未改v38源；该目录
  附`preparation_failure.json`，不作为新策略证据。v2精确定位六个消费者guard。
  B4 v2及纯当前原生等待控制均完整编译；仅去`// pto:`调试注释后完整C++
  逐字一致，见`active_prefix_qa_policy_b4_codegen_audit.json`。尚未正式采用。
- 新Score候选`score_wide_k128_native_candidate_v1`保留a241fa61的B1、AIV
  Score/TopK、叶划分与双槽FFTS协议，仅替换长G6已选split区间的Cube helper。
  旧N128/K192负例的WS Right为48KiB，因此不能同时保留两份16KiB Key Right。
  本轮改为三个K128加权归约块，WS Right降为32KiB，恢复Key到Right的双缓冲。
  QK仍分两份M192，复用一个96KiB Acc；两份QK结果在Mat中组成M384，供中间
  K128块跨两半读取。每query的64个非零head系数不跨K128边界，但FP32精确
  一致仍必须上卡证明，不能仅据稀疏结构宣称成立。
- CPU完整编译与`audit_score_wide_k128.py`通过：Left为24+24+4+4+4KiB；
  Acc为96+8KiB且不重叠；稳态Right为16+16KiB Key及32KiB WS，恰64KiB。
  首panel没有前panel的WS，Key合法复用后来WS的区域；审计按真实生命周期
  区分首panel和稳态，保留两次QK之间的FIX→M释放事件。每1024列为16次QK、
  16次QK FIX、8次WS初始化+16次累计、8次六行STORE，仅一条MIX task提交，
  无新增GM分配。资源/事件检查不代表已获得实际引擎重叠或性能收益。
  B16实际任务`task_20261021_050120_68460921096`正在验证，未预记收益。

### 同步已采用基线并启动同一候选的完整八档

- v39 N128/K128完整精确、TopK/score、状态、保护、图和协议均通过，exit0；
  B16中位数788.26→845.08us（+7.21%）。恢复Right预取仍未带来完整收益，
  不采用、不扩到B28、不把资源审计当作性能结论。原始源码、编译和窗口保留。
- v40 B1纯当前数学/原生等待的正式服务、ACLGraphWrapper、工作区复用、
  空输入及回退检查通过，exit0，`task_20261021_050121_6848358492`。
  连同v37 B1 S1～S6、此前原生等待组件及B6服务/变metadata图证据，
  将a241fa61的五份已采用数学/策略文件和原生signals.py同步进入生产TND包。
  `native_current_bsh_adoption.json/.patch`明确这是继承基线及等待接口对齐，
  不计为本轮相对新BSH的加速，也没有采用失败的性能试验。
- CPU测试入口切换到已冻结的本轮原生工具；TND合同37项通过，日志
  `runs/cpu_tnd_contracts_native_a241_v42.log`。没有重跑未涉及的HBG等旧验证。
  README说明工具依赖、可应用补丁、实际服务范围及尚未达到的性能目标。
- 冻结性能候选`active_prefix_qa_policy_native_candidate_v2`，在QA M16资源域
  保持原路径，其余域采用空消费者等待裁剪。八档同一源码/基线/采样器统一
  提交为`cohort_prefix_a241_*_v41`；五个轮换窗口、完整72核首尾计时。
  必须收齐全部八档后由`evaluate_s6_cohort.py`计算等权几何平均；此前各局部
  结果不能替换其中任一档。该等待裁剪尚未进入生产包。
- 同一候选另补B1 S1～S6与B6混长/连续状态矩阵（v43）。这是新的等待路径
  验证，不用纯基线v37的通过替代。原Native差异仍独立列出，不放宽阈值。

### 完整八档结果：GM=0.993346，尚未达到10%目标

基线为另一CSA会话已采用的`a241fa6178ddad24ced30e5a93d109dfb17f5b53`，
含B1动态半叶/UB优化及150tick等待。候选为原生TND加非QA-M16域的空消费者
等待裁剪；这一裁剪仍是试验，尚未采用到生产包。每档在同进程、同卡、同输入
下交替测量五个窗口，分别取中位数，计全72核完整链路L1首尾，包含等待和空隙。

| 历史长度 | B / S | BSH µs | TND µs | 耗时变化 |
| --- | --- | ---: | ---: | ---: |
| 128K | 1 / 6 | 363.28 | 357.18 | -1.68% |
| 128K | 4 / 6 | 506.12 | 493.52 | -2.49% |
| 128K | 8 / 6 | 617.16 | 620.40 | +0.52% |
| 128K | 16 / 6 | 789.94 | 777.44 | -1.58% |
| 128K | 24 / 6 | 1017.90 | 1013.46 | -0.44% |
| 128K | 28 / 6 | 1153.56 | 1156.34 | +0.24% |
| 8K | 16 / 6 | 635.20 | 634.16 | -0.16% |
| 8K | 32 / 6 | 915.52 | 918.32 | +0.31% |

等权`GM(T_TND/T_BSH)=0.9933462113985101`，平均耗时降低约0.665%，
未达到`<=0.90`。各档功能/精确一致/状态/保护区/图及采样完整性通过，但
性能门槛明确为FAIL；goal保持active。B1五个TND窗口中的504.32us保留，
没有删异常值或取最快窗口，也没有用此前v38更快结果替换。

可审计结果为`tests/pypto_test/csa_spmd_tnd_20261020/`下的
`cohort_prefix_typefix_v47_result.json`；任务选择为`cohort_prefix_typefix_v47.json`。
六个128K使用v41原始测量，两档8K为v47任务
`task_20261021_054041_30074296661`和`task_20261021_054041_3007522242`。
六档复用的代码等价证明如下，并非任意合并不同候选的测量。

#### 8K编译修复及已完成测量的复用

- 原v41两档8K在编译阶段失败，没有性能结果。原因是a241fa61新增的B1专用
  UB临时张量写作`8192//(2*score_tile) × score_tile`，未启用的B1分支在
  short G2的`score_tile=768`下仍被解析，得到3840元素，随后reshape4096失败。
  使用另一会话实际接受的工具链编译原始BSH同样复现，证据为
  `runs/bsh_a241_short_original_cpu_v44/result.json`。原始FAIL和快照保留。
- 只在本轮冻结BSH、TND候选及生产TND中，将上述两处B1专用形状明确为
  `4×1024`；不改短历史计算、归并/同分顺序、循环或同步，也没有写另一会话。
  补丁及修复后基线记录为`short_history_inactive_b1_typefix.patch`、
  `short_history_typefix.json`、`bsh_baseline_a241fa61_short_typefix_manifest.json`。
- 六个128K场景原形状本来就是4×1024。用各档实际保存的描述符重新CPU编译
  候选和基线共12份，移除完整`// pto:`调试名称注释行后，生成C++逐字一致。
  保留变量名、地址、常量、屏障、事件和全部可执行代码，不做语义近似归一化。
  证据为`runs/short_typefix_long_equivalence_v45/result.json`。
- `typefix_cohort_reuse_certificate.json`严格验证源码差异仅上述两处形状、
  launcher的基线manifest文件名及manifest中的修复记录；评估器还逐项核验
  原/新C++哈希、工具指纹与对应场景。篡改C++哈希或换工具哈希的负例均拒绝。
  因此不重复六档已完成的设备测量，只补测失败的8K两档。修复后的8K基线在
  本轮工具与另一会话工具上也生成一致C++，见`short_typefix_baseline_tool_parity.json`。

#### 动态路径与后续方向

- v43 B1 S1～S6同址图六例通过，任务`task_20261021_051507_150340519380`。
- v43混长矩阵入口误给了混长初始输入，违反矩阵要求的六个S6捕获合同，
  原任务`task_20261021_051507_15036488295`失败保留。修正入口参数后，
  v46 `task_20261021_053322_257585813797`完成八个metadata场景及四步连续
  decode，共12次同址图重放，全部通过。覆盖S1～S6、261/123456混长、
  T容量41/B槽12、页映射和连续状态。Native原有差异继续单列。
- 提交器增加CPU入口合同检查，提前拒绝上述错误及B1误用批内反转服务开关；
  不修改设备正确性检查，不为无效参数占卡。
- 下一项试验只将私有`indexer_transfer`每行增加64字节间隔；计算宽度仍为
  1024/768，Native KV布局不变。目的是检查传输行距是否影响整链性能，
  未证明存在内存冲突，也不预记收益。B16 CPU完整生成通过，C++与未加间隔
  候选除调试注释外完全相同，运行时buffer列维变为1040。
  `transfer_stride_pad64_codegen_audit.json`保存证明；新增padding哨兵检查
  位于计时窗口之外，CPU已验证未写入PASS、覆盖一元素FAIL、原宽度不检查。
  先提交一个B16配对设备用例，结果决定是否扩展；试验尚未进入生产包。

### 行间隔试验收取；开始按实际scale写入者等待

- v49 `task_20261021_063940_212922026426` completed/exit0，完整精确、状态、
  图/代际、保护区及新行间隔哨兵均通过。128K/B16五窗BSH/TND中位数为
  790.56/784.44us（-0.77%）。该候选还含先前空消费者等待裁剪，因此不能
  单独将这6.12us归因于行间隔；也不与此前不同轮次相减。收益证据不足，
  暂不扩大此方向或采用。原始五窗及判断保留在`transfer_stride_pad64_v1.json`。
- 从生产TND建立新的独立`scale_owner_wait_native_candidate_v1`，不叠加上述
  行间隔或空消费者裁剪。只在两种角色进入Score前，将48个scale参与者的
  全量等待改为读取本轮已发布META中的链表头，等待非空链表对应的实际写者。
  源码生成器、补丁和逐文件指纹是`prepare_scale_owner_wait.py`、
  `scale_owner_wait_v1.patch/.json`。
- 所有发布、代际、数学、FFTS、页写所有权和量化前驱均不变。每个scale写者
  仍先取得全部真实请求key量化写者，再完成本页scale RMW并发布。任一合法
  key写入的谓词都蕴含META中的对应scale任务；所以等待非空scale写者也覆盖
  当前key写回。若所有链表为空，本轮没有Native Indexer cache写入。
  即使等待集合为空，最终业务`fence/cacheinvalid/fence`仍无条件执行。
- META获取没有省略：AIV先在Indexer compressor helper中无条件取得META；
  AIC此前等待IDX_QUERY_QUANT，其发布者也已取得同代META。链表头由唯一
  planner每轮全部重写，后续只读，不根据上一次调用缓存活跃写者列表。
- B16 CPU完整编译通过。实际C++的AIC非空判断内保留150tick wait；AIV的
  `label846→847/848`分支同样只绕过空writer，回边推进owner，退出后仍执行
  完整业务缓存获取。没有删除原来存在的等待循环或把Scalar通知当作DMA完成。
  CPU证明不是设备正确性；B16完整配对任务为
  `task_20261021_065141_287902612619`，仍待结果。若进入采用范围，须补验
  B1 S1零关闭组及混长/页共享/连续调用的写者变化。

### Scale等待的两种实现：功能通过，收益仍小

- v51直接扫描链表头版completed/exit0，精确输出/TopK/score、状态、保护区、
  图和协议全部通过。实际活跃写者只有8个：2/8/14/20/26/32/38/44。
  但完整L1为790.56→807.12us（+2.09%），五窗全慢，拒绝此实现，
  不继续测其B1。等待数少不等于速度快，新增逐消费者扫描仍有代价。
- v2由已有唯一META核扫描一次，生成`count + owners[]`；消费者只读有效前缀。
  为此在私有`group_offsets`末尾增加64个INT32，共256字节，没有新增Tensor根、
  Host回读、通知阶段或KV搬运。原连续页布局证明分区偏移保持不变。每轮都写
  count，即使为0；旧有效前缀之外的项不读取。全部写者发布和最终缓存获取保留。
- CPU完整编译通过，实际AIV控制流保留active计数回边以及count=0写出，
  count和列表在既有META通知前完成。AIC/AIV消费者均按count控制等待，
  退出后无条件获取业务缓存。审计为`scale_owner_list_codegen_audit.json`。
  测试诊断在计时之外逐项核对列表和48个原链表头，并检查末尾15项保留区。
  CPU正/负例覆盖非空→空、越界count、陈旧owner及保护区覆盖，全部按预期。
- v53 `task_20261021_070255_352921911564` completed/exit0，完整功能/协议及
  8项列表检查通过。五窗BSH/TND中位数786.30/781.54us（-0.61%，4/5窗更快），
  尚不据此采用或扩全八档，也不把v51/v53的跨轮差值宣称为精确净收益。
- 下一项针对B1验证：将S6完整配对计时和S1～S6列表生命周期合为一次编译。
  harness先完成同输入精确检查与固定S6五窗，再运行会修改根metadata的矩阵；
  后续矩阵通过不能覆盖前面失败状态。避免变metadata残留污染固定S6采样，
  也避免为同一个候选重复编译两套基线。矩阵每次重放另核对当前活跃列表；
  B1 S1包括本轮没有C4关闭行的情形。本项尚未报告设备结果。

### B1列表生命周期通过；转向O_A供数的独立CPU候选

- v55 `task_20261021_071520_61535459` completed/exit0。固定S6同输入精确检查
  后，五窗完整L1中位数360.38→358.26us（-0.59%，3/5窗更快）。随后同址图
  S1～S6六例全部通过，S1的列表count明确为0，其余五例为1；旧的非空前缀
  没有被当作新一轮有效列表。输出/TopK/score、cache/state、保护区和代际
  检查保持原标准。一次编译同时完成两类检查，没有把变metadata后的输入
  计入S6窗口。
- 紧凑列表的两个代表档位均不到1%收益，暂不采用、不扩完整八档；生产包
  继续保留原等待。此方向源码、CPU、全部窗口和边界通过证据完整保留。
- 核对a241fa61 B1原始O_A生成代码：物理M128、有效M取实际T；每个输出块
  为32次递增K128累加，16次K256 L1供数。新独立试验只在单请求的小M资源域
  选M16，改为两份K512 L1窗口，仍保持原32次K128的数值顺序、每组八列owner、
  所有发布/量化/O_B依赖。当前decode合同query容量至多6，因此有效M均落在
  物理16行内；不使用编译器内部compact分配标志。
- `prepare_b1_oa_supply.py`生成`b1_oa_supply_native_candidate_v1`。四次K1024
  普通循环各先加载两份K512，只让Acc跨回边；L0先准备两份K128并连续发出
  两次MMA，再填已释放的窗口，全部累加仍按K递增。预计Mat288KiB、Right64KiB，
  只是源级预算，必须以实际编译的地址、同步与分配为准。
- 新`OA_SMALL_SUPPLY`是Host已有单请求/容量信息选的constexpr，其余档位关闭。
  尚未上卡或采用；B1 CPU完整编译进行中，还需资源/事件及B16关闭路径核对。
  不把物理tile缩小或源码预取顺序当作已测性能收益。原生工具链未修改。

### O_A供数候选的实际编译审查与B1设备提交

- v56第一版CPU完整编译通过，但实际Right的K2/K3都复用了地址0，未保留
  连续双槽供数；该版没有上卡。v2调整独立extract的生存期，让下一组K128
  操作数在当前MMA前已存在；逐项MMA调用AST和数值顺序与v1一致。
- v57 B1实际Left地址为0/4096交替、Right为0/32768交替；Acc仅8KiB。
  两份K512权重各128KiB、两份M16输入各16KiB，共288KiB Mat有效分配，
  最高地址末端为360448字节（352KiB），不是把源级容量预算当实际地址。
  只有Acc跨四次K1024循环，没有Mat→Mat拷贝。第一次乘法初始化，后续保持
  32段K128递增累加，实际行数贯穿输入与输出。
- C++确认第二份L1输入/权重TLOAD已发出，第一次MMA只取得当前窗口的MTE2
  事件0/1，不提前等待下一窗口的2/3。L0复用的M→MTE1事件及K链PIPE_M屏障
  保留；不以“存在双槽”直接宣称硬件重叠或收益。复核脚本为
  `audit_b1_oa_supply.py`，证据在`runs/b1_oa_supply_cpu_b1_v57/resource_audit.json`。
- B16关闭新分支的完整C++与当前纯生产TND控制逐字一致，仅去整行`// pto:`
  调试名称注释；见`b1_oa_supply_disabled_b16_audit.json`。没有复测B16设备。
- 测试工具增加O_A未写行保护：`o_r[T_capacity:]`必须保留37哨兵，检查在
  计时窗口之外。CPU正/负例证明写有效行通过、越界写一个元素被拒绝。
- B1设备任务`task_20261021_073455_113981416736`已提交，目录
  `runs/b1_oa_supply_s6_and_s1_s6_v59`：先同输入精确与图检查、再固定S6五窗
  完整L1、最后同址S1～S6矩阵。结果尚待收取，生产实现没有采用此候选。

### O_A收取为负例；三段QK的Score资源路径进入设备验证

- v59 completed/exit0：精确输出/TopK/score、状态、图/协议、O_A未写行哨兵及
  S1～S6六例均通过。但B1完整L1中位数377.48→379.08us（+0.42%），不采用，
  不扫描相邻tile参数。实际供数双槽和较小物理M没有转化为整链收益。
- 当前正式N64 Score**已经**使用两份M192、各48KiB的QK Acc。没有重复尝试
  为它增加第二个Acc。早先v39 N128/M192/K128路径为了容量只保留一份96KiB
  QK Acc，B16虽精确通过但回退7.21%；该负例仍保留。
- 新候选从纯生产TND建立，只替换已有长G6/SPLIT_SCORE_ROWS资源域的Cube
  Score：N128下改三份M128，QK使用两份64KiB Acc；QK0写入L1后，QK2复用
  其地址。QK1完成FIX后，WS0在其已释放区域使用8KiB，QK2保持live直到
  WS0发出后再FIX。后续两次WS仍按K128顺序累加。沿用v39已精确检查过的
  三段K128加权归约，但这一新QK排程仍需独立设备精确验证。
- query/key/系数的数值、FP16 FIX比例/ReLU、Native页表、叶划分、Top-K
  同分顺序、双槽READY/ACK和末尾排空均保留；没有叠加O_A、padding或等待
  裁剪。源码生成器和补丁是`prepare_score_threeway.py`及`score_threeway_v1.patch`。
- v60 CPU完整编译通过，实际Acc地址为QK0/QK2=0、QK1=65536，稳态WS为
  65536～73728，最高末端131072字节，恰在128KiB内；Left合计60KiB。
  `audit_score_threeway.py`逐个核对八个panel中的FIX→M复用依赖；WS Right
  若复用当前Key区域，还核对M→MTE1获取，并确认不覆盖已预取的下一Key。
  所有检查通过，见`runs/score_threeway_cpu_b16_v60/resource_audit.json`。
- 每1024列的生成静态位点为24次QK、24次QK FIX、8次WS初始化+16次WS累计、
  8次Score写回。位点减少不等于已测提速，也不把资源复用当作省略同步。
  代表B16任务`task_20261021_075510_226360516093`已提交，尚待完整精确与
  五窗L1结果。完整八档目标仍未达到，所有本节候选均未进入生产包。

### 三段QK与WS供数调整均为负例，转向核内等待归因

- v61 `task_20261021_075510_226360516093` completed/exit0：同输入精确输出、
  TopK/score、cache/state、保护区、图和协议均通过；B16/S6完整L1中位数
  787.32→814.48us（+3.45%），五组均更慢。不采用。
- v62 CPU完整编译与`audit_score_ws_pipeline.py`检查通过。第二版保持三段
  M128/N128的QK、三段K128的WS及全部数值次序；下一Key只预取到L1，
  为WS腾出Right的0/32768双窗口。QK2的Acc保持live到WS0发出后再FIX，
  复用依赖均保留。该调整仅是实验，不修改生产包。
- v63 `task_20261021_082907_411978514951` completed/exit0：全部上述精确、
  状态及协议检查通过；B16/S6完整L1为786.94→857.00us（+8.90%），五组
  均更慢。终止本轮N128 Cube供数方向，不扩展到其他档位、不采用。
- 较少的静态MMA位点、合法的双缓冲资源分配均不足以证明整链提速。已有
  phase-clock的“other”含FFTS FIFO等待和传输，并非纯计算时间；后续先
  分析Score核间READY/ACK等待及其执行次数，再选择优化方向。
- 最新完整八档仍是`cohort_prefix_typefix_v47_result.json`，等权几何平均
  耗时比0.993346，未达到0.90。未将单档试验拼接为新的八档结果。

### 只读排除重复方向，尝试Score可见范围循环外提

- 复核incore-profiling及CCE设备阶段时钟说明：单核合成仿真不复现本例
  72角色常驻协作；已有Scalar原位READY/ACK记录也不能当作异步引擎纯
  阻塞耗时。没有重复采集无法补充因果证据的全核阶段时钟。
- 另会话记录已覆盖Score循环化：跨回边Right/Mat移动CPU失败，组内双
  panel版本完整精确通过但明显回退；主压缩共享Left也已有完整负例。
  本轮没有重新实现/上卡这些方向。长G6已在UB按2048候选流式归并，
  不把它误作score落GM重读的优化机会。
- `prepare_score_visibility.py`从纯生产TND复制单独候选。长G6每AIV最多
  三个query的可见范围及causal合法性不随同一叶的2048候选段改变；
  将原`indexer_query_visibility`完整调用前移到叶内、step循环外。
  三个Scalar均按`lane*3+local<live`保护，不读取空lane的positions；
  每次调用、每叶重新推导，不缓存上次图重放的Q/L/position。
- 仅balanced G6启用，不改B1 G2及短历史非balanced数学；不改变任何
  tile、叶边界、排序同分规则、store、READY/ACK或外部Tensor根。
  v64完整CPU编译进行中，后续须审计实际循环内metadata读取及AIC
  关闭改动等价性，再决定单档设备试验。尚无精度或性能结论。

- v64 CPU完整编译通过。`audit_score_visibility.py`核对Cube函数全文仅去
  `// pto:`注释后相同；完整PTO的Tile分配地址/布局多重集合相同；
  Score step内710个tile操作静态位点的顺序相同；GM等待/发布45/48、
  FFTS wait/set 8/7静态位点数量保持。step内来自请求几何/可见范围
  helper的180个源位置变为0，三份受live保护的Scalar定义在step外。
  这些是CPU结构证据，不是汇编寄存器压力或已测性能。
- B16/S6任务`task_20261021_085454_157214514444`已提交，输出目录
  `runs/score_visibility_h131072_b16_s6_v65`；冻结纯生产控制与单独候选，
  先精确/保护/图检查，再五窗完整L1。尚未采用，等待终态。

### 可见范围外提收取为负例，本轮候选均未采用

- v65 `task_20261021_085454_157214514444` completed/exit0。候选同输入
  精确匹配冻结BSH的输出、TopK/score与全部cache/state；eager、两次图、
  有效写入/metadata/保护区、代际和采样前后校验均通过。Native原有数值
  差异仍按原阈值保留，没有改标为Native精度PASS。
- 五窗完整L1：BSH `[785.30,783.36,782.78,777.96,780.14]`us；
  候选 `[794.82,800.46,796.64,795.86,793.40]`us。中位782.78→795.86us，
  回退1.67%，0/5窗更快；不采用、不扩其他档位。外提只读推导合法并不
  代表总时延减少，未把回退无证据地归因于寄存器、缓存或某个pipe。
- 顺带只读核对Q_B的K128双Right旧证据：另会话已完整精确通过但回退4.71%，
  不因当前编译提示stage2降为单槽而重新做同一个候选。本轮未改工具链。
- 另会话当前已采用单SPMD仍为`a241fa61`且算子无未提交差异；其未采用
  的WS池/scale/Score分工试验不纳入分母。最新完整八档仍为v47的
  0.993346，约0.665%平均耗时降低，目标0.90未达到。
- 截至本次记录，CPU和设备任务全部终态；生产TND没有采用v61/v63/v65。
  goal继续active，不标完成或阻塞；保留既有代码、所有失败与逐档报告。

### O_A与O_B互补分工：先证明唯一写者，再做完整链路验证

- 前轮v65形成了新的负证据并决定不采用，按goal续跑规则记为有效进展；
  本轮原目标和后续10%八档要求均重读，BSH仍为a241fa61且算子无未提交差异。
- 只读核对TopK的exhausted路径：它在任一输入耗尽时停止，且现有ISA
  包装另有V→S计数获取和UB复制；不能把它当作固定Top512提前结束。
  本轮没有据此修改归并规则或重复旧阈值/前缀实验。
- 发现输出阶段存在相关的负载分配：既有OA_COLS=128、T<=96域共64份
  O_A、64份O_B逻辑工作，原24核条带令0～15核分别承担3+3份，16～23核
  承担2+2份。O_A与O_B数学成本不同，数量不是耗时，但轻核已有较早
  结束的同轮端点；只读B1证据保存为output_work_balance_readonly_evidence.json。
- 新候选只改变O_B owner：仍24个物理AIC，以32个软件rank均分64份O_B；
  原重O_A核各领2份O_B，原轻核各领4份。每组只等待原量化writer，完整
  O_B结束仍由每个物理核发布原PHASE_OB，HCpost仍等全部24核。不存在
  新核组、任务、Tensor根、数值顺序或O_A/量化分工变化。
- `prepare_output_work_balance.py`枚举T1～96，检查完整64格输出各有唯一
  writer、所有24物理核都有工作；启用范围沿已有OA_COLS128区间，不列
  B/S/history白名单。所有其它区间保留旧分支，尚需编译等价证据。
- v66 B1/B16完整CPU编译已启动。源码仅policy/entry/cube_schedule三文件
  变化，所有数学helper保持逐字一致。没有采用，未预记任何设备收益。

- v66 B1/B16完整CPU编译通过。B16整个Vector函数仅去`// pto:`调试注释
  后与纯生产控制相同；全PTO的10462个tile操作静态位点顺序相同，所有
  tile地址/布局多重集合相同。GM wait/notify 45/48与FFTS wait/set 8/7
  静态位点数量相同。审计保存于output_work_balance_codegen_audit.json。
- 已有B1探针中，0～15核的O_A最后端点中位431.10us、O_B结束463.79us；
  16～23核分别395.81us、434.86us。这些只支持进一步试验负载归属，
  含探针、等待与竞争，不能把端点差直接当成可省耗时。
- v67 B1任务`task_20261021_091603_291753515374`，目录
  `runs/output_work_balance_b1_s6_and_s1_s6_v67`；先同输入精确/图，
  再S6五窗完整L1，最后同址S1～S6。B16任务
  `task_20261021_091605_291986121743`，目录
  `runs/output_work_balance_h131072_b16_s6_v67`，同口径五窗L1。
  均为单卡TP1任务，尚待终态，不预记采用或性能收益。

### 输出互补分工：B1/B16完整通过并取得正向结果

- v67两任务均completed/exit0。B1与B16同输入输出、TopK/score、所有
  cache/state完全匹配BSH；保护区、图和72角色协议全部通过。B1随后
  六项S1～S6同址metadata矩阵也全部通过，没有改变固定S6计时输入。
- B1五窗：BSH `[377.58,370.22,371.62,365.46,373.60]`us，
  TND `[351.96,354.52,350.16,352.08,348.00]`us；中位371.62→351.96，
  降低5.290%，五窗均更快。B16五窗：BSH
  `[794.48,791.76,789.32,791.40,787.60]`us，TND
  `[780.18,790.54,778.32,778.00,783.40]`us；中位791.40→780.18，
  降低1.418%，五窗均更快。不能据此声称八档达到10%。
- 实际设备两角色C++与v66 CPU输出在应用原冻结PyPTO包装预处理后相同，
  仅去`// pto:`调试注释并统一函数声明拼写。初次直接比较不同是正常
  wrapper删除`set_ffts_base_addr`，已按真实`_ptoas_preprocess.py`验证，
  没有自行忽略额外同步差异。见output_work_balance_device_codegen_audit.json。
- 暂未采用，继续验证同一冻结候选的其余启用域：B4
  `task_20261021_092602_362071332328`、B8
  `task_20261021_092603_362254932218`已提交五窗L1。随后还需短历史、
  长G6混长/同图变化及关闭域验证，才能决定是否进入生产包。

- B4/B8尚待终态期间，只读复核既有B28阶段记录。当前与该记录的O_A/O_B
  数学文件逐字相同。T168、OA_COLS256时也正好64份O_A：0～15核3份、
  16～23核2份，但每核分别处理完整M128或尾M40。已有带探针端点中，
  同为3份O_A的完整/尾行核组的最后端点中位1116.21/1106.84us；
  2份组为1087.59/1082.63us。这是旧诊断，不能用来预测新L1。
  后续可按“64份O_A工作”这一资源条件评估扩大分工范围，当前v1仍保持
  T<=96、没有修改正在验证的冻结候选。详见
  output_work_balance_b28_readonly_evidence.json。

### B4/B8扩档未支持全区间采用，保留全部窗口并收窄候选

- v68两任务completed/exit0，完整同输入精确/状态/图/保护/协议均通过。
  B4：BSH `[479.46,477.22,467.94,486.28,480.82]`us，
  候选 `[482.58,481.06,467.62,480.22,459.32]`us；中位479.46→480.22
  （+0.159%），3/5窗更快，未取得一致收益。B8：BSH
  `[607.86,605.60,616.02,620.82,611.42]`us，候选
  `[637.06,630.38,611.64,619.80,609.44]`us；中位611.42→619.80
  （+1.371%），3/5窗更快。较慢首窗仍保留，不删除或重定义指标。
- 复核计时器在采样前已有三轮交替预热，未找到足以判无效的计时错误。
  本轮不通过增加预热、删窗口或换统计量来消除负结果。
- `prepare_output_work_balance_guard.py`建立v2，唯一差异是Host策略：
  单请求且query上界<=6可用用户允许的B1分档；多请求仅在已有
  EARLY_GEOMETRY_MIN_TOKENS=64到LOCAL_STATS_MAX_TOKENS=96区间
  开启，其余保持原分工。B4/B8未获收益不代表已证明某个硬件瓶颈；
  64～96是待补边界/混长验证的经验资源区间，尚非全区间收益证明。
- 新旧策略在B1/T6和B16/T96的全部constexpr相同，所有JIT文件除
  Host policy外逐字相同。v69正在对关闭的B8分支与纯生产控制做
  新鲜完整CPU编译，需确认关闭分支无生成代码变化；尚未采用到生产包。

### 关闭分支等价通过，提交短历史与实际开启策略的混长检查

- v69纯生产控制与v2关闭B8分支完整CPU通过，整个PTOAS C++仅去完整
  `// pto:`调试注释后逐字相同。没有删除SSA名、运算、地址或同步来凑
  等价，见output_work_balance_disabled_b8_audit.json。
- 旧混长矩阵Tcap41会落在v2关闭区间，不能拿它验证新分工。测试夹具
  最多36个有效token，原容量为41；现将初始容量合同扩为41～72（12请求槽×6的上界）；
  保留6个初始S6请求和全部既有场景、阈值、保护区、连续状态逻辑。
  B1容量6/请求1的合同不变。八个正/负preflight用例通过，见
  wide_metadata_preflight_checks.json；没有修改L1计时器。
- v70短历史8K/B16五窗L1：`task_20261021_094033_42904922785`，
  目录`runs/output_work_balance_guard_h8192_b16_s6_v70`。长G6混长
  Tcap65/Bcap12、H131072、页表上界131200：
  `task_20261021_094035_4306327707`，目录
  `runs/output_work_balance_guard_long_mixed_t65_v70`，不采性能，只检查
  原八组metadata变化和四步连续状态；需确认新分工确实开启。
- v2的B1/B16开启代码与v1应相同，但在正式八档复用前仍须新鲜编译
  等价证书。新夹具guard仅影响计时之后的metadata矩阵；其源指纹变化
  必须显式说明，不能让聚合器静默忽略harness差异。目标仍未达到。

### 短历史负例与扩容量夹具修正

- v70短历史任务completed/exit0，输出/TopK/score/cache/state、图/保护/
  协议全部通过，但BSH `[633.32,628.56,633.76,632.88,634.06]`us，
  候选 `[637.10,647.80,641.10,645.96,646.46]`us；中位633.32→645.96
  （+1.996%），0/5窗更快，不在短历史开启。v3仅改Host策略，保留既有
  B1 balanced半叶区间，以及长历史G6且64<=T<=96，其它路径关闭。
- v70 T65/B12混长任务completed/exit1：初始S6、图和原SPMD精确比较
  已通过，进入metadata矩阵构造uniform S1时失败，尚无完整混长结果。
  T65的六个S1请求只占6行，余59行要10个dummy请求槽，加活动6个共需
  16槽。此前只按初始S6所需槽数检查，漏掉矩阵中最小query的槽预算。
- 修正仅在测试夹具/提交预检查：请求槽要求为
  `6 + ceil((T_capacity - 6)/6)`，原T41/B12仍合法，新T65需B16。
  原八个metadata和四步连续状态、所有精确/保护/阈值均保留，未开启
  allow_unassigned_tail来绕过dummy检查。容量41～72的64个正/负
  preflight检查、全部11种query向量的槽预算检查通过；旧FAIL原样保留。
- v71正在完整编译短历史关闭分支及纯生产控制；随后用v3、T65/B16
  重新验证真正开启分工的混长/连续状态。尚未采用，不将初始通过冒充
  完整矩阵通过，也不将局部B1/B16收益当作八档10%目标达成。

- v71短历史关闭分支与纯生产控制完整编译通过，整份C++仅去完整
  `// pto:`注释后相同，见output_balance_short_disabled_audit.json。
- v72修正后的长G6混长矩阵：`task_20261021_095605_14307563635`，
  `runs/output_balance_long_mixed_t65_bcap16_v72`，初始6×S6、Tcap65、
  请求容量16、history131072、table_history131200、无L1窗口。
  v72长历史T66/B11/S6边界：`task_20261021_095607_143256629477`，
  `runs/output_balance_long_t66_boundary_v72`，五窗完整L1。两者均冻结
  v3，需确认OUTPUT_WORK_BALANCE开启，尚待结果。
- 本轮生产TND仍未采用新分工，局部收益不替代完整八档目标。全部正负
  结果、夹具失败、修正及关闭分支证据保留；未commit/push。

### 修正后的长G6混长与边界通过，冻结最终候选并建立复用证书

- v72混长任务`task_20261021_095605_14307563635` completed/exit0：
  Tcap65/Bcap16，OUTPUT_WORK_BALANCE=true。uniform S1～S6、mixed261、
  mixed123456共8例同址metadata变化，随后4步请求重排/增减与连续
  cache/state演进全部通过。初始同输入原SPMD精确、图、保护区也通过。
  未用每轮恢复初始状态替代连续轨迹；旧v70夹具FAIL仍保留。
- v72边界`task_20261021_095607_143256629477` completed/exit0：T66/B11/S6
  全部严格检查通过。BSH `[706.42,711.04,700.46,697.16,704.08]`us，
  候选 `[685.02,686.22,695.50,700.86,695.04]`us，中位704.08→695.04
  （-1.284%），4/5窗更快；唯一回退窗保留。这里只支持当前资源区间，
  不宣称所有历史/数据都加速。
- v4只把分工数字16/32/8改为具名常量，仍仅在既有B1半叶域或长G6
  64～96 token域开启；未改变数学helper、Tensor根或物理核数。
  `prepare_output_balance_final.py`保存源码和CPU参考快照，未把快照
  本身当作设备通过。生产包暂未采用。
- v73对B1/B16各重新完整编译candidate与baseline，共4份：全部PTOAS
  C++仅去完整`// pto:`调试注释后逐字相同；4份Host orchestration
  也逐字相同，参数绑定/任务提交没有变化。所有其他源码及冻结工具
  校验，Host策略在两档的全部constexpr相同。
- `certify_output_balance_reuse.py`严格核对仅有Host guard、具名常量、
  计时后metadata夹具guard/注释的变化；不忽略一般harness差异。证书为
  `runs/output_balance_reuse_v73/certificate.json`。2个正例及工具链/
  C++/Host orchestration散列负例全部通过，修改冻结源文件也会被拒绝，
  见output_balance_reuse_checks.json。没有重复B1/B16设备计时。
- 一致八档由cohort_output_balance_v4.json追踪；B1/B16仅按上述证书
  复用v67，其余六档重新同轮成对测量。首先提交关闭策略的B4
  `task_20261021_101618_280762024924`、B8
  `task_20261021_101619_280897124735`，目录
  `runs/cohort_output_balance_h131072_b{4,8}_s6_v74`。其余仍待完成，
  不把目前两个正向点混成八档达标结论。

- v74 B4/B8均completed/exit0，完整严格回归通过，实际policy字段
  OUTPUT_WORK_BALANCE=false。B4中位BSH492.26/TND483.16us，B8为
  630.82/625.78us；全部五窗保留。这些是关闭新分工后的TND/BSH实测，
  不归功于已关闭的O_B分工，也不把与上轮的基线波动当作代码收益。
- v75 B24 `task_20261021_102626_32708358896`、B28
  `task_20261021_102627_32720263427`已提交，冻结同一v4和计时器。
  六档长历史收齐后还需两档8K，不提前计算或宣告完整八档达标。

- v75 B24/B28均completed/exit0，严格数值/状态/图/保护/协议通过，
  OUTPUT_WORK_BALANCE=false。中位分别为1008.46→1019.12us、
  1162.84→1156.10us；正负窗口均保留。v76短历史B16
  `task_20261021_103655_374384014461`、B32
  `task_20261021_103656_37456895465`已提交，同一v4与完整L1口径。
- 用户新指令允许更多分档，要求不要丢失可能的性能优化。已写入
  GOAL_UPDATES.md。当前冻结八档先收齐；之后允许把局部正向结果
  单独纳入更细策略，不要求全域统一启用。数值/状态/同步与原生TND
  合同、无新增D2H及最终八档10%目标均保持；负例继续留证。

## 2026-10-21：v76八档收齐，v77保留已验证输出分档

两项8K任务均completed/exit0，数值/状态/图/保护/协议通过。
八档经evaluate_s6_cohort.py严格核验源码、工具链、计时和等价证书；
GM(TND/BSH)=0.991928345，完整L1耗时降低0.8072%，10%门槛FAIL。

|历史/B|BSH μs|TND μs|耗时变化|
|---|---:|---:|---:|
|128K/B1|371.62|351.96|-5.29%|
|128K/B4|492.26|483.16|-1.85%|
|128K/B8|630.82|625.78|-0.80%|
|128K/B16|791.40|780.18|-1.42%|
|128K/B24|1008.46|1019.12|+1.06%|
|128K/B28|1162.84|1156.10|-0.58%|
|8K/B16|634.46|653.54|+3.01%|
|8K/B32|920.58|917.04|-0.38%|

新输出分工只在B1与长G6/T64～96开启；其余六档关闭，
关闭档的变化不归因于该分工。采用v4的policy/entry/cube_schedule三个文件，
其余源不变，见output_balance_adoption_v77.json。原始全部五窗口及负例
保留；这属于局部优化交付，不宣告整体goal完成。CPU合同检查随后记录。
最新完整报告通过latest_complete_cohort.json显式选择，旧v47报告仍保留。

- v77共51项CPU合同通过（原37项加14项启用范围/边界检查）。
  ruff发现新增Cube两行超长，以及继承的import顺序/未用符号；冻结八档
  不改写，后续只清理正式源并留等价证据。
- v78两个候选仅独立修改Host策略：大容量长G6的128<T<=256启用输出
  互补分工；短历史T64～256启用既有HC消费者归约。前者对129～256
  全部容量的O_A任务数及O_B每行/列唯一覆盖CPU枚举通过，仍保持24个
  物理AIC及全部PHASE_OB发布。后者复用已有四部分左折叠归约；
  两者等待完整CPU编译后单卡独立验证，不预记设备或性能通过。

- v78完整CPU双候选均通过。大输出分工的AIV整函数、10573个tile
  运算序列及全部分配布局不变；HC消费者归约的AIC整函数不变，两者均
  一个MIX/24组，见fine_tier_codegen_audit_v78.json。
- v79上卡：长B28 task_20261021_111201_12821755840；短B32
  task_20261021_111202_128298718985。HC候选新增计时外RAW毒值不变与
  PHASE_RAW零发布检查；原计时器未改。尚待终态。
- 正式Cube文件只整理import/两处断行，全部非import AST逐项相同；
  唯一删除的是未被任何Name引用的PHASE_ATTN_PLAN import。独立证据
  output_balance_production_style_audit_v79.json；修改文件ruff全通过。
  冻结候选/原结果不改写，未把格式整理算成新的设备验收。

### 更细分档的候选保留清单

用户明确允许增加分档后，以下方向继续保留；旧轮次只用于选实验，
不能直接作为当前a241fa61/150tick基线的性能收益，也不保证收益可叠加。

|方向|旧局部正向证据|负向或限制|当前处理|
|---|---|---|---|
|HC消费者归约|8K/B32旧同轮降低2.54%|128K/B16增加1.06%；旧100tick/旧调度|v79按当前基线重测短历史|
|Attention查询几何复用|8K/B32旧同轮降低1.33%|128K/B16增加1.66%；后续physical plan已减少部分重复工作|保留待当前代码审计，不盲目叠加|
|G2短历史score留UB|8K/B16旧同轮降低2.86%|B32增加0.43%；旧关闭分支也改变资源布局|保留B16方向；须当前独立作用域与代码审计|
|O_A两M行块共享权重|8K/B32旧同轮降低1.476%|长B24增加0.883%，H32760/B32增加0.629%|保留短历史较小工作量区间机会|
|Query反量化穿插Score|8K/B32旧同轮降低1.307%|128K/B16增加0.254%，较大代码耦合|保留候选，不作为首个追加方向|
|active-prefix等待裁剪|当前旧v38 B1降低5.68%、B8降低2.50%|v47完整八档GM仅0.993346；其他档有回退|保留与输出分工组合机会，须新同轮验证|

G6/24叶的同分TopK语义FAIL不属于可用局部优化；无论分档数量如何，
都不能以随机输入的收益绕过该失败。完整旧证据保留在各自源目录与日志。

- v79短B32 HC消费者归约completed/exit0，精确旧SPMD对照、状态、
  图、保护/代际通过，RAW整缓冲仍37且PHASE_RAW全零。完整五窗：
  BSH[922.46,927.02,923.64,922.42,918.62]，
  TND[916.66,931.72,928.70,918.18,928.72]μs；
  中位922.46→928.70μs，增加0.6765%，2/5窗口更快，不采用。
  旧8K/B32的2.54%收益未在当前基线上重现，不当成可叠加收益。
- v80准备B1输出分工+active-prefix组合，只改Vector等待，
  编译期仅BALANCED/G2（既有B1域）进入等待裁剪；其余区间仍走原
  等待。所有计算与通知保持，未增加策略参数或device回读。
  CPU完整编译中；组合没有设备证据，不能将两单项收益相加。

- v79长B28输出分工completed/exit0，全严格对照/状态/图/保护/协议
  通过。BSH五窗[1141.52,1147.00,1152.10,1147.02,1150.30]，
  TND[1141.34,1145.54,1148.90,1153.90,1144.48]μs；
  中位1147.02→1145.54μs，仅降低0.1290%，4/5窗口更短。幅度过小，
  尚不视作确定收益，不采用；保留候选并检查尾行工作量不同的B24。
  不追加B28同源重复计时来挑更好的轮次。

- v80 B1组合完整CPU通过；AIC整函数、9274个tile运算序列、分配布局、
  全部48个通知和8/7个FFTS等待/发送静态站点不变，仍一个MIX/24组。
  非B1的长B16/G6路径完整C++仅去pto调试注释后与v4相同，Host
  orchestration逐字相同；分别见b1_output_prefix_codegen_audit_v80.json
  和b1_output_prefix_disabled_codegen_audit_v80.json。
- v81短HC/B16 task_20261021_112147_186635723381；
  B1组合 task_20261021_112239_19267186046。B1先测固定S6五窗，
  再按原次序做同址S1～S6矩阵，不把变长后的工作量混入S6计时。
  两项均待终态；v79单项结果收在fine_tier_results_v79.json，
  不替换最新完整八档报告中的任何一行。

- v81短HC/B16 completed/exit0，全严格检查及RAW毒值/零发布均通过。
  BSH[640.62,621.46,631.92,613.44,632.36]，
  TND[626.52,634.28,640.80,631.52,640.20]μs；
  中位631.92→634.28μs，增加0.3735%，1/5窗口更短。短B16/B32
  两种资源域均没有当前基线收益，不采用HC消费者归约，保留原始结果。
- B1组合S6已测369.48→352.12μs，降低4.6985%，所有五窗比BSH短；
  S1～S6矩阵尚待终态。这与已采用的输出分工收益同量级，不能把两
  不同轮次相减推导等待裁剪的独立收益，暂不替换已采用版本。
- v82开始单独验证长B24输出分工；与B28策略constexpr相同、仅T/B
  描述符不同。沿既有完整编译与数学helper审计上卡，未重复CPU构建。

- v81 B1组合最终completed/exit0，固定S6全部严格对照及随后六项
  S1～S6同址图通过。S6五窗BSH[374.50,359.68,371.04,365.30,369.48]，
  TND[347.44,352.12,358.24,349.20,362.28]μs，5/5比BSH更短。
  尚无相对已采用output-only的独立增益证据；该组合保留为候选，
  不用跨轮次相减或相加来宣称叠加加速，不修改当前正式默认。

- v82长B24输出分工completed/exit0，全严格检查通过。BSH五窗
  [1016.20,1007.46,1014.90,1016.18,1017.72]，TND
  [1017.08,1005.44,1014.46,1017.24,1015.68]μs；
  中位1016.18→1015.68μs，仅降低0.0492%，3/5更快。大T两档
  均未获得有说服力的净收益，不扩入正式策略；原局部B1/T64～96保留。
- 用户新要求读取另外两Codex会话的优化过程/记录，筛选可以迁移的措施。
  当前没有活动NPU或CPU任务；继续原TND/八档10%goal，优先核对两个
  会话的最新未采用候选与证据。旧短G2 score留UB仅做只读审计，
  尚未在本worktree新建候选或编译，不提前记为实现。

## 2026-10-21：按用户要求复核另两个Codex会话，筛选迁移措施

已只读查看session_index及两份实际rollout尾段：CSA-多task
01a0c3f6-3686-7611-95d5-37bed175d574，接入vllm-ascend
01a0c422-2ef1-7f91-a239-96f637f6c740。复核对应源码、提交和原始结果；
未向另一会话发送消息、未改其工作区、未执行其历史push指令。
发现“接入”会话现阶段主要在做单SPMD CSA，不能仅按名字认为仍只改HC_pre。
详细索引及适用边界见cross_session_review_v83.json。

|优先级|可迁移措施|实际证据|TND处理|
|---|---|---|---|
|1|小T QA由N128×8改为N64×16|多task已采用a658188c；B1 mean381.844→371.676μs，八档/边界/图通过|只迁独立列分工及QR等16个实际owner；保持完整K256链、单MIX和原NORM/QA协议|
|2|长G6/T64～96 WS八列窗写回＋连续scale|正常两轮794.50→782.66、792.24→784.34μs，组合确认781.96→780.02μs|另建候选；保留设备逐页连续性证明与碎片回退，不累加跨轮收益|
|3|长G6大T Score逻辑worker换物理核|B28两轮1161.42→1151.32、1162.06→1149.28μs，组合1166.02→1149.78μs|仅144<T<=256，AIC与双AIV整体置换，等待用真实写者逆映射；需TND混长验证|

B24组合反向任务实际已complete/exit0，旧交接仍写“待收”。核对角色后：
spmd_candidate是原版1025.58μs，spmd_reference是组合1022.12μs；组合
此次快3.46μs，首轮组合慢4.68μs。不能把反向字段读反，更不能据此
宣称关闭分支始终零开销。外会话正式单SPMD仍a241fa61，基线不改。

150tick、B1原半叶与分数/根UB、长历史双QK早FIX、物理行MGATHER、
O_B无效尾行裁剪等已经继承，不重复当作新增优化。multi-task的B1整请求
Key复用涉及G6与不同叶/同分布局，不能直接覆盖当前保持原排序的B1 G2。
SDMA预取、multi-task WS局部快但完整慢、已放弃Attention数学和同分FAIL
方案不列入迁移。旧HC_pre的RMS/供数/显式L0流水可借鉴，但其部分和、
RMS/Sinkhorn树与本整链不同，不能整文件替换或直接搬入sync_all。

已建立v83 QA私有候选，新增Host constexpr QA_SINGLE_M16只在长历史
且0<T容量<=16开启；请求真实Q/L仍为device动态数据。N64所有权按T1～16
逐元素枚举唯一覆盖通过，关闭域保留原编译分支。多task的sync_start及
额外注册入口不移入单SPMD内部；新增全局屏障会改变现有依赖图。
当前B1开启与B16关闭完整CPU编译中，尚未设备验证、不提前采用。

## 2026-10-21：拉取最新远端并补齐阶段性提交

此前长期只保留工作区改动、没有阶段性commit，用户明确要求纠正，并要求
先拉取最新远端。已完整备份237项自有修改，tracked改动额外存stash；
快进38个上游提交到65f51175，再应用stash，无冲突、无文件丢失，237项
文件与拉取前逐字节一致。未改另两个会话的工作区，未push。

- af991f59：隔离PyPTO/PTOAS补丁、基线和构建指纹、workspace-root。
- 63533c5d：TND完整实现/服务图合同、已采用输出分工和CPU合同测试。
- 本条及全部实验入口、任务终态、轻量报告作为后续test阶段提交。

依赖patch中的空白上下文行是unified diff格式，必须保留单个空格。
目录.gitattributes只对此类patch文件关闭上下文空白/末行告警，补丁散列未变，
真正被修改源码的git apply --check --whitespace=error-all证据仍保留。
6个继承Python文件清理未用导入与格式，全部非import/文档AST相同；
未修改冻结设备源码。拉取后及整理后均90项相关CPU测试通过，Ruff通过。
原误留仓库根目录的旧审计脚本原样移入测试目录，见legacy_audit_location.json。

QA移植v83首轮两档因旧依赖调用未显式传N_TILE而在解析期失败；已在
v83b显式传原N128参数，两档完整编译通过，B16关闭路径C++与Host编排
仍与v4一致。未提交任何QA设备任务。借入WS/scale的v84只有源码准备，
未编译/上卡，B28核置换仅做审阅。以上均不进入正式实现。

当前104个设备任务全部终态（85 exit0，19非零）；无活动设备或CPU任务。
完整八档仍为v4的GM0.991928345，10%门槛FAIL，原goal保持active。
后续每个可审阅的实现、验收、采用或否决阶段及时commit，不再等总目标
完成后才保存成果。原始大产物和未采用源码留在本机，LOCAL_ARTIFACTS.json
明确索引；Git保存代码、生成器、补丁及轻量证据，不上传权重/二进制。

## 2026-10-21：三阶段commit后继续QA移植验证

上一goal回合为progress：完成八档/局部分档/两会话审阅及三笔提交。
本轮再次读取原goal及后续要求，保持完整TND与八档GM<=0.90目标。
核对另一会话正式单SPMD仍a241fa61、相关目录无改动，不把新多task
QA提交误认成BSH分母更新。v83b私有源码冻结，B1开启/B16关闭CPU已通过。
v86 B1 task_20261021_121214_66656511685：先S6五窗，再S1～S6同址图；
T16边界 task_20261021_121215_667954606：真实Q=[6,6,4]混长、无填充
假BSH，完整严格对照与五窗L1。两任务待终态，不预记采用或收益。

### v87：借入WS/scale的CPU资源与关闭路径通过

QA两项v86任务的权威状态仍pending，尚无输出目录；队列维护关闭，16卡
由其它作业占用。更正此前“已启动”的表述，不把已提交当作已执行，不
杀其它任务、不重提相同任务。等待期间完成不占卡的两个CPU构建。

WS/scale描述符根据实际binding规则同步调整：B16的SCALE_PAGE_COALESCING
从false变true，group_offsets由848字增至880字；新增SCORE_WS_POOL_PANELS=8。
B1只多编译常量1，容量与原策略完全相同。来源见borrowed_score_stream_descriptors_v87.json。

v87完整B16/B1 CPU均通过。实际AllocMemoryAddr中三份Acc分别为
[0,49152)、[49152,98304)、[98304,131072)，两份48KiB QK和32KiB WS
恰好填满128KiB且不重叠。仍一个MIX、24组。B1关闭路径完整C++仅去整行
pto调试注释后与v4相同，Host orchestration逐字相同。独立可复核脚本
为audit_borrowed_score_stream.py，结果borrowed_score_stream_codegen_audit_v87.json。
没有用CPU资源通过替代TND数值、动态页布局或完整L1设备验收。

本CPU阶段及时提交；QA与WS候选仍不进入正式实现，整体10%目标未改变。

### v88：借入B28 Score逻辑worker置换，完成独立CPU阶段

前四个阶段提交已保存为af991f59、63533c5d、87b10fad、e76656c6。
本阶段在正式TND源码之外生成borrowed_score_owner_candidate_v88，
只借入另一会话source_score_owner_interval的七文件增量。Vector调度与
pair发布函数先验证其原版AST和本地完全相同，才移植对应函数；其余
文件作受断言保护的精确修改。保留本地inactive-B1形状修复、native150、
已采用输出分工和TND服务/绑定，不覆盖另一会话或正式包。

Host开启域为长历史G6、144<T容量<=256；真实group/leaf/visible及是否
有至少两个Score步差异仍在本轮META writer判断。AIC与双AIV整组使用
physical→logical；Top-K等根时使用logical→physical，原叶边界、叶内
数学、同分优先级、GM槽位和物理phase发布不变。group_offsets在布局
证明区之后追加96字；B28编译描述符由1480增至1576字，B16仍848字。

B28开启/B16关闭的完整CPU编译通过。B16完整C++仅去整行pto调试注释后
与v73已测v4一致，Host orchestration逐字相同；开启时仍一个MIX、24组。
编译器对helper报ScalarWriteLineShared的保守告警，实际AllocateMemoryAddr
中10处映射表Scalar写均位于worker==planner唯一写者条件内；未通过
关闭诊断、修改编译器或增加不必要全核屏障来处理。

实际Scalar函数体在有边界检查的CPU原语下完成11组检查：B28非身份置换、
非零负数BOOL、active→empty→identity→active、少组/等负载/过多组、
关闭及无效叶计划等。独立枚举每个物理组的producer工作项，核对每条
query/lane的根等待实际writer；置换必须双射且两表互逆。刻意破坏逆表
会被检查器拒绝。另96项Host边界检查和两份描述符/实际policy一致性通过。
这些只是有界CPU语义与生成代码证据，不证明NPU并发或cache可见性。

本次未新增设备任务。既有QA两项v86仍pending，16卡全部被其它作业占用；
等待原任务，不重复提交或干预其它用户。WS与Score候选后续仍需各自
完整L1/精度/动态页及图验证；尤其大T需>=24实际活动组的混长输入，
旧6请求矩阵只会走身份路径，不能算作active置换验证。

本阶段及时commit，正式八档仍为v4的GM0.991928345，10%目标未达到。
同时整理recovery_state中旧v47/未采用/禁止commit等过期下一步描述，
原内容保存在recovery_historical_before_v88.json，不改任何历史测量。

### v89：独立大T映射回放探针，20项CPU检查通过

上一goal回合为progress：移植Score候选、两档CPU编译与归属/等待审计，
已提交7e119834。本回合再次读原goal与后续要求；原两QA任务权威状态
仍pending，未将排队视为运行，也未重复提交。另一会话已采用BSH仍a241fa61。

发现原run_probe只接受legacy或64字活跃writer表，会拒绝Score新96字表。
因此在runs/score_owner_harness_v89冻结独立探针，不改已排队QA使用的
公共脚本。新增计时外诊断逐项核对设备Score步数、前段KV/QB工作排序、
置换和逆置换；用独立枚举与排序检查，未知长度/错误代价/错误逆表均拒绝。
这不是生产D2H，也不用于决定核分工，L1计时文件逐字不变。

大T矩阵捕获B28/S6、T容量168、请求容量32，随后计划9个metadata场景：
28请求的S1～S6、28请求混长/混历史/碎片页、23请求身份路径、28请求恢复
并换升序页；再执行4步连续decode，包含请求反转、删除、增加新ID和
重新激活旧ID。全部沿用真实Native builder和原数值/保护区检查。
小Q下允许Native Q末端小于T容量，其余行没有请求归属；dummy仍每槽
最多6行且seq_len=0，不把实际query补成B×6。该新增大T夹具尚待设备验收。

为验证实际映射重写，回放时保留上一轮group_offsets，仅该表不再填37；
其它原scratch保护、代次信号、精度和cache/state连续性检查保持原样。
每个场景要求实际group数正确，指定场景必须命中非身份/身份路径，不能
以关闭分支通过冒充核置换验收。Native独立连续state参考仍保留。

20项CPU测试通过：合法/负数BOOL映射、7种损坏表拒绝、legacy长度严格性、
5项提交入口、原检查AST/计时文件等价、9+4场景容量与分支前检。日志
runs/score_owner_harness_cpu_v89.log，轻量结果score_owner_harness_cpu_v89.json。
这些不代表13个设备replay已通过，不替代真机精度/同步/图或性能。

pending_device_plan_v89.json保存下一步命令，未提交。先分别评估WS B16和
Score B28的原S6档；Score正式八档仍用请求容量28，独立图矩阵容量32不
混入性能门槛。只有候选首轮通过且值得采用，才执行其新增动态矩阵。
本阶段继续及时提交；原八档GM0.991928345/0.90未达标，goal保持active。

### v90：WS连续页证明探针及25项CPU前检，暂停后恢复归档

暂停前已冻结runs/ws_layout_harness_v90，保留原公共探针字节。新增计时外
诊断逐页核对Native Indexer实际可见页，而不只检查首尾：有效连续页必须
给出正确(first,direction)，碎片/无效/超范围必须为(0,0)，也拒绝错误回退。

专用矩阵初始B6/S6、T65/Rcap16，实际开启长G6的WS八列池和连续scale。
保留原S1～S6、混长和4步连续Native状态验收，再加4个布局转换场景，总计
12个metadata加4步连续decode待设备执行。回放保留上轮group_offsets，
以暴露连续到碎片/失活时漏清理的旧证明；原精度、保护区和协议检查不变。

历史页升序并不保证本轮完整前缀升序：H131072/S2未新增压缩页，可命中
升序路径；同历史S6新增压缩行，原current页后缀可能使完整前缀断开，须
回退。用例要求升序命中、碎片回退、降序恢复、升序尾页断开回退均真实
发生，不以全走fallback冒充优化分支验收。

25项CPU检查通过，含11种合法/回退、6种损坏/残留拒绝、无效请求、5项
提交前检和原检查/计时等价。轻量结果ws_layout_harness_cpu_v90.json；
本地两个探针模块合计45项正常收集。缺冻结本机产物时显式skip，不能
将其视为验收通过。暂停/恢复后不重跑已经通过且未改变的验证。

用户在本阶段要求push并指定远端dsv4-flash-pto-v0.25.1rc1；已推送前六笔
提交至aa2e2a41，设置当前隔离worktree的上游并核对远端。仓库format.sh ci
已调用，但机器缺pre-commit，完整格式pipeline未运行；已报告此限制，
未宣称全量格式通过。随后原goal恢复active，本阶段完成归档和commit。

QA两项v86任务仍pending，未因观察超时重启/重提。pending_device_plan_v90.json
保存WS/Score各自原S6和动态矩阵命令，全部未提交；须先看原S6精度和L1
再决定是否采用、补齐动态矩阵。正式八档GM仍0.991928345，10%目标未达。

### 2026-10-21：推送边界更正

WS阶段已提交136c7af6；推送前远端新增84751e34的Native B1归档，确认
不涉及TND/BSH或本会话探针后合并为25d0b2df。先前发出的推送已完成。
用户随后明确“之后没让你push不要push”：后续仍阶段性本地commit，
但push必须有新的明确指令，不能从上游设置或goal恢复推导持续授权。
本条只保存到本地，不再执行push。

### 2026-10-21：只读汇总八档五组现有性能，未重测

按用户要求展示Native、BSH MPMD含AICPU、BSH MPMD纯incore L1、BSH
SPMD、TND SPMD，以及(Native-TND)/TND百分比。仅读取仓库和另一会话的
已有原始结果，未提交NPU任务、未编译、未执行或重测算子。

Native七档沿用csa_five_way_latest_20261020，B1沿用csa_native_b1_20261021
补充的632.75μs；均取30次Profiler全流设备跨度中位数，不替换成Event。
MPMD使用已采用a658188c的MATRIX_qa_shape_gate_v1_shape_gate_v1完整八档，
两列选择相同multitask_candidate角色：外层5次graph.replay Event含AICPU
与运行时，L1取5次最早incore到最晚结束跨度。外层原表只写mean/min/max，
本次从全部原始样本计算median，统一本表统计量，不挑单次最快值。

BSH与TND SPMD两列取cohort_output_balance_v4_result的八档配对中位数，
未采用QA/WS/Score新候选均不混入。逐档检查24份原始报告、样本数、
S6/H/B、单task/72核与50MHz时钟、功能及无新增回归通过；计算与已存
汇总相互核对。百分比另用Decimal从未舍入值独立计算，CSV与JSON一致。

不同列来自不同测量轮次/工具链；前两列为外层、后三列为L1。末列是
用户指定的跨口径比值，不能当作同口径整服务加速比；外层减L1也不是
单独测得的纯AICPU开销。本报告不改变TND相对BSH八档10%的验收结论。

结果five_way_existing_20261021.json保留每行原样本、来源路径及散列；
复算入口collect_existing_five_way.py只做文件读取与CPU计算。下载CSV与
样本索引位于downloads/csa_eight_cases_five_paths_20261021。仅本地提交，不push。

### v92：设备恢复，QA 首轮设备验收完成，WS/Score 开始执行

用户确认卡可用后继续原目标。原两项 QA v86 任务自动转为 running，未重提，
随后均 exit 0；此前全卡占用的 blocked 计数已清零，不沿用旧阻塞状态。

QA N64 所有权改动在 B1/S6 与 B3/[6,6,4]/T16 两例均通过原单 SPMD 的
输出、Top-K/score、cache/state 精确比较，两个同地址图重放及保护区、协议
检查通过。B1 另外六个 S1～S6 图用例全部通过。实际 CCEC 等待策略仍为
150 tick，1473 项固定工具链文件的运行后散列审计通过。

五个窗口均保留，完整链路 L1 中位数分别为 BSH 367.30 → QA 361.62 µs
（−1.5464%，3/5 窗口更快），混长 499.24 → 498.02 µs（−0.2444%，3/5）。
B1 同时开启了已采用的 output_work_balance，故此结果不能证明 QA 相对
当前 TND 有增量收益；跨轮结果不相减。暂不采用，但保留候选和全部证据；
若继续考虑采用，须与当前 TND 直接配对。T16 混长不是八档 S6 的一档。

Native 原有差异仍为 FAIL：B1 输出 99 个超阈值元素、31 个 Top-K 替换；
T16 输出 783 个、Top-K 替换 97 个，新旧完全相同。未放宽阈值或改标。
qa_device_results_v92.json 保存全部 L1 样本、源码散列、任务终态和检查摘要。

两项 QA 完成后，按 pending_device_plan_v90.json 提交原已准备的独立探针：
WS B16/S6 为 task_20261021_140935_275270122118；Score-owner B28/S6 为
task_20261021_140935_275288020871。各用一张卡，正式 S6 档先验证精度和
五窗口 L1，新增动态矩阵仍待首轮结果后决定。当前完整八档 GM 仍为
0.991928345，0.90 目标未达；本阶段只做本地 commit，不 push。

### v93：WS/Score 首轮无收益；增量探针明确区分 TND 与 BSH 参照

两项 v90 任务均 exit 0，精确比较、图重放、保护区、同步和运行后工具链
审计通过。WS 的 B16 实际 16 请求均命中完整降序连续页证明；Score 的
B28 实际产生非身份核映射，代价、置换及逆表独立检查通过。

完整 L1 五窗口结果：WS 的 BSH 中位数 784.24 → 785.46 µs（+0.1556%，
2/5 配对更快）；Score 为 1142.46 → 1160.02 µs（+1.5370%，1/5 更快）。
全部样本与原有 Native 差异见 ws_score_device_results_v93.json。两者未采用，
新增动态矩阵均未执行；Score 不再扩展当前方案。

QA 和 WS 候选都含已采用的 output_work_balance，之前对 BSH 的结果不能
单独归因给新增优化。为决定是否保留局部收益，冻结增量探针，直接与已
采用 TND v4 比较；不重测用户要求的八档五组历史对照表。两侧算子、完整
L1 计时和原精度/图检查不改，只更换参照并独立保存 comparison_reference。
summarize_results 明示参照种类，evaluate_s6_cohort 显式拒绝此类增量结果。

v93 首次两任务 task_20261021_141858_318904115435 和
task_20261021_141858_318908415657 在算子执行前 exit 1：TND 历史
wait_codegen.py 仍保留未调用的 100 tick 变换，原 BSH 探针初始化要求该
模块常量为 150。实际 TND 使用的是编译器原生 150 tick，二者不是同一路径。
失败记录保留，未修改算子或放松等待检查。v93b 改成只接受 native150 的
源码回调，拒绝 legacy/raw/mixed 等待，实际 CCEC 审计也要求全为 native。

9 项 CPU 检查通过：真实 launcher 的两种快照构建、固定参照散列与计时
字节一致、错误参照拒绝、两种增量标记均拒绝进入 BSH 门槛，以及四种
等待输入的接受/拒绝。v93b 的 B1 QA 和 B16 WS 已分别提交为
task_20261021_142045_325426029053、task_20261021_142045_32543028090。
这些只回答相对当前 TND 的增量效果，不更新正式八档 GM。继续本地提交。

### v94：QA/WS 与已采用 TND 的直接配对完成，均不采用

v93b 两项均 exit 0。输出、Top-K/score、cache/state 精确比较、同地址图、
保护区、代际、72 核单 task 完整 L1 均通过。实际两侧 CCEC 代码全为
native150，没有 legacy rewrite；参照源码快照和 1473 项工具链审计通过。

QA B1/S6：当前 TND 358.08 → 候选 366.26 µs（+2.2844%），五组配对
全部变慢，不采用。B1 的 S1～S6 已在 v86 通过，不重复无关正确性矩阵。
WS B16/S6：当前 TND 785.02 → 候选 788.22 µs（+0.4076%）；4/5 配对
差值为负，但全样本中位数为正，证据弱且方向不一致。不挑选更有利的
统计量、不重测相同代码以等待噪声转正，不采用也不扩动态矩阵。

incremental_results_v94.json 保存两侧全部五窗口、配对差值、原始结果散列、
源码、参照与等待审计。正式八档仍为 output_balance_v4，GM0.991928345，
未达到 0.90。生产算子未因上述三个候选改动，没有新的 push。

下一保留方向为旧短 G2/768 半叶分数驻留 UB。只读核对发现当前非 balanced
分支与旧 scoped_source_v7 的 fallback AST 完全相同，可以只借入局部 helper
而保留当前完整调度与数学。旧 B16 +2.86% 与 B32 −0.43% 是旧基线的选题
依据，不作为本轮收益。拟先按 Host 描述符给出的 query 组数上界不超过
48（24 组的两轮工作）隔离小工作量区间，再做当前工具链的开启/关闭
分配与生成代码审计；此时尚未移植或提交该候选的设备任务。

### v95：短 G2 半叶 UB 候选完成隔离编译，首轮设备任务已提交

在 runs/short_half_ub_candidate_v95 中仅移植旧 scoped v7 helper 和受控调用。
当前 fallback 与旧版逐项 AST 相等，原 Top-K 半叶函数亦相等；不替换整个
旧 Score 文件，保留后来采用的 B1、双 QK、TND 绑定及输出分工。启用条件
为短历史 G2，且 requests×ceil(max_query_tokens/2)≤48；这是已有 Host
容量的 query 组数上界，不用 T/6 推导真实请求，不新增设备回读。

三档候选完整 CPU 编译通过：8K/B16 开启、8K/B32 关闭、128K/B16 关闭。
另新编译当前版本的 8K/B32 对照。两个关闭档的完整 C++（仅去 pto 名字
注释）及 Host 调度均与当前版本相同，旧候选的关闭分支残留分配问题在
这些档位已消除。8K/B16 开启档的 Cube 整函数仍与现有测量版本一致。

实际分配：半叶缓存 22×384 FP32，共33792 B，复用同一物理 UB 区间；
整个 AIV 的 Vec 末端163872 B，小于188416 B。开启档无 score_arena 的
GM store，仍只有一个 MIX/24组。792组 Host 边界组合仅改变新开关，其余
策略逐项不变。详见 short_half_ub_codegen_audit_v95.json；均为 CPU 证据。

独立增量探针仍对照已采用 TND v4，并在计时外核对整个 score_arena 保持
初始毒值37，确认删除的 GM 写回没有残留。新增3项 CPU 检查覆盖正常、
被改写和关闭分支不回读；之前9项无关检查未重复。候选补丁可干净应用，
shell语法和Ruff通过。没有改生产包或其它会话的源码。

首轮8K/B16/S6任务为 task_20261021_144229_86824910253，五窗口完整 L1，
完整精度/状态/图/保护区先行。若值得采用，仍须当前版本的混长、页表变化、
连续状态及同分排序覆盖；旧 helper 的150次回放不代替本轮设备验收。
八档GM与10%目标不变；本阶段继续本地commit，不push。

v95 设备任务已 exit 0：精确输出/Top-K/score/cache/state、图、保护区、
代际及 native150 审计全部通过，score_arena 在图和 L1 前后均保持37。
当前 TND 五窗为652.78/646.50/631.98/647.66/647.54，候选为646.12/
645.08/640.30/643.64/637.44 µs。中位647.54→643.64，降低0.6023%，
4/5配对更快。保留这个小幅局部收益，尚未采用；后续混长、连续状态与
同分排序仍须验收，不把单点增量改写成正式八档收益。

### v96：再次读取两个会话，保留精确 Top-K 短前缀方向

用户要求再看另外两个会话并使用适用措施。本次直接读取两个实际 rollout
的最新记录及源码/采用报告：多task会话已启动独立 TND MPMD 新目标，尚无
该新目标的设备成果；接入vllm会话最新工作是 Native B1 归档，已采用的
单SPMD仍为a241fa61。两个会话的新目标或推送授权不传递到本会话。

发现此前未迁入本会话的措施：6ba55b45 已在多task采用的精确Top-K短前缀，
其算法源为c49ee5b2/source_topk_payload。旧单SPMD七档里，B8为634.96→
617.44、B28为1178.64→1157.64；B4、B24则分别慢3.76、2.14µs。因此
不能整包照搬或宣称收益相加。先以B28为代表，把helper与受控调用移到
当前TND副本，保留真实Q/L、GE相等项、原归并方向和完整回退，另做当前
工具链、同分及整链验证。此项在本记录时尚未完成迁移。

HCpost扩核来自既有SPMD的逐token AIV路径，当前已具备；多task的
sync_start也不能当作单SPMD内部阶段的新原语直接加入。QA/WS/Score已在
v92～v94完成本会话的实测，不重复原样尝试。跨会话证据与选择理由保存于
cross_session_review_v96.json。短G2半叶UB的当前正向候选继续保留。

### v97～v104：借入 Top-K 精确短前缀，按本轮证据收窄采用区间

只迁入已采用多task的阈值helper及受控调用，算法AST与6ba55b45/c49ee5b2
一致；保留当前TND请求映射、causal边界、B1原半叶、Cube和同步。CPU证明
开启档Cube不变，关闭档完整可执行C++及Host调度不变；GE相等项、32B
计数缓冲及Vec容量审计通过。原审计脚本两次结构假设失败均保留，未改内核。

当前TND增量对照：长历史B28中位1151.84→1152.62µs，3/5配对更快，
无净收益，保留原路径。长历史B8中位642.68→622.60µs，降低3.1244%，
5/5配对更快，完整无新增回归、图和状态检查通过；五个窗口全部保留。
采用候选收窄为长历史G6、32≤T容量<64。新开关下B8生成代码与实测版
一致，B28关闭代码与当前版一致，因此不重复该性能测试。

收窄后的Top-K候选在T41/R12下8项元数据+4步连续状态全部通过，任务
task_20261021_160738_306213119230；同地址S1～S6、混261/123456、
页表/请求变化和保护区均纳入现有矩阵。59项Top-K同分及回退helper已CPU
编译通过，上卡任务task_20261021_162057_10868510136尚待回收。

短G2半叶UB：T65/R16的8+4矩阵通过；任务
task_20261021_155419_187365031349的150项同分/可见边界/同址live2→1→2
全部通过，pair根和Cube传输位级一致，缓存不变、score_arena保持37。
注意通用元数据矩阵并未额外回读score_arena；其读回证据来自初始整链图、
L1前后与150项helper，不扩大覆盖声明。短历史G6关闭分支的完整符号签名
CPU编译也证明可执行C++与Host调度相同，未进行设备或性能测量。

证据汇总为borrowed_candidates_results_v104.json。生产包暂未改动，等待
最后Top-K同分结果后组合两个互斥区间，保留各自的设备验证并核对生成
代码。当前正式八档GM仍0.991928345，未达到0.90；本轮不push。

### v105～v106：组合并采用两个已验证的局部优化

Top-K同分任务task_20261021_162057_10868510136已exit0：59种边界输入，
按1024→2048→1024三次同址回放，共177行，根结果、输入和保护区全部
通过。至此两个候选均满足当前设备的数学、图、动态状态验收。

组合后对八档做完整CPU编译，长B8与已测短前缀候选代码相同，短B16与
已测半叶UB候选相同，其余六档与原TND完整可执行C++及Host调度相同。
另对两个混长矩阵的真实描述符编译，生成代码与先前各自12次设备回放
一致；因此不重复已通过的设备矩阵。v106仅整理4处行宽和注释，所有
Python模块的算法AST与v105相同。证据见borrowed_combined_*audit*.json。

现已写入生产包五个文件：Score、entry、Vector调度、policy及Top-K helper。
短前缀仅长历史G6、32≤T容量<64启用；半叶UB仅短G2且Host query组上界
≤48启用。实际query边界、可见历史与Top-K保留数量继续在设备计算。
未改变Cube、同步、Native绑定或服务路由，不新增D2H。原有Native数值
差异仍单列，当前验收表示无新增回归。

新增19个资源边界用例，TND与服务CPU合同合计89项通过；Ruff和差异
空白检查通过。六个未改变档位可复用原完整配对窗口，基线源码及工具链
字节一致；新旧harness只有计时外且关闭的检查及已证实的矩阵容量保护
差异。borrowed_combined_reuse_certificate_v106.json记录完整证明。

两个改变档位正式对照BSH a241fa61+静态形状修复+150tick，任务为
task_20261021_162824_11408309460（128K/B8）和
task_20261021_162824_11409259262（8K/B16），均保留五个完整L1窗口。
尚待回收，不将已有TND增量收益当作正式BSH比例，也不提前报告新八档GM。
本次为阶段性本地采用与提交，没有push。

### v107：正式八档更新与最新跨会话措施复核

两个正式BSH任务均exit0，当前设备C++与v105的CPU完整代码一致；无新增
输出/Top-K/score/cache/state、图和保护区回归。Native原差异不改判：
长B8仍有3080个输出元素超原阈值，短B16仍有1135个，新旧单SPMD完全
一致。完整差异与全部五窗口见borrowed_combined_device_results_v107.json。

| 历史长度 | B / S6 | BSH L1中位µs | 当前TND L1中位µs | 耗时变化 |
|---|---:|---:|---:|---:|
| 131072 | 1 | 371.62 | 351.96 | −5.2904% |
| 131072 | 4 | 492.26 | 483.16 | −1.8486% |
| 131072 | 8 | 622.28 | 617.16 | −0.8228% |
| 131072 | 16 | 791.40 | 780.18 | −1.4177% |
| 131072 | 24 | 1008.46 | 1019.12 | +1.0571% |
| 131072 | 28 | 1162.84 | 1156.10 | −0.5796% |
| 8192 | 16 | 641.12 | 640.38 | −0.1154% |
| 8192 | 32 | 920.58 | 917.04 | −0.3845% |

长B8/短B16为本轮新成对结果，其余六档使用完整可执行代码及Host一致
证明复用原成对数据；没有跨轮选最快窗口。evaluate_s6_cohort完整核验
源码、工具链、150tick、72核记录和计时边界后，等权GM(TND/BSH)=
0.988089028，约降低1.1911%，仍未达到0.90。旧报告与五路线历史表保留。

又读取两会话最新记录：多task会话对T≤64独立任务减少空worker，以及
TaskId依赖、Indexer关键路径顺序实验；接入会话最新为issue状态审计。
前者通过各独立task的launch核数减少运行时开销，不能直接改成常驻
单SPMD的总核数。当前本实现已按行owner等待，并且先发Indexer query
再发Q_B，AIV也先完成Indexer RoPE再做主Q反量化。之前active-prefix
与输出分工组合没有明确增量收益，记录保留，不原样重复。具体代码及
取舍在cross_session_review_v107.json；没有发现新的已采用BSH内核，
基线保持a241fa61。

本轮用户要求的可借用措施已落地，生产采用提交17b8f305；原10%goal
仍active，没有待跑设备任务，也没有push。

### v108b～v109：小容量Q_B改用更宽N块，完成CPU证明后独立上卡

继续原八档10%目标，上一轮为实际采用与验证进展。复查已存阶段诊断：
长B16/B28的Q_B阶段约62.5/88.67µs，Score/Attention仍更长。这些旧
插桩记录只用于选方向，不当作当前完整L1，也不重复采集相同诊断。

新候选在Host token容量<64时采用M64/N512/K128，列块从128降到64。
这里只改NZ INT8 Q_B，输入/输出ABI、动态有效行、其他投影和Vector数学
不变；每个1024项INT8点积绝对值上界16777216，远小于INT32上限。
K128拆分仍按递增K累加，不引入浮点归约变化或新的D2H。

首版准备时错误假设小T关闭cooperative_prefix，预检立即拒绝，未上卡。
真实当前policy始终开启该路径；v108b已显式把开关传过projection_overlap，
保留物理AIC4～23的20个Q_B生产者及worker%20映射，所有原通知不变。
原失败副本与qb_wide_prepare_attempt_v108.json保留。

B1/B8开启、B16关闭的完整CPU编译通过。开启档Vector整函数及Host
字节不变，B16的完整可执行C++也不变。实际生成M64/N512的128KiB
INT32累加器，Right分成两个32KiB槽；整个Cube的Mat峰值475136B、
Acc131072B、Left/Right各65536B，均在原硬件容量内。20/24两种worker
映射枚举均完整且唯一覆盖64个列块。不能由此预判设备性能。

新增量探针以已采用combinedv106为参照，冻结源码并明确禁止用于BSH
目标验收。实际快照与错误参照拒绝两项CPU检查通过，未改计时和精度
门槛，未重复其余11项无关检查。B1任务task_20261021_165204_1559193337
包含固定S6五窗和同址S1～S6；B8任务task_20261021_165204_15601729493
包含完整严格对照与五窗L1。结果待回收；生产仍为combinedv106。

### v110～v113：宽N局部结果、矩阵参数修复与两核Hadamard组合

宽N以当前TNDcombinedv106对照，B1固定S6精确/图/L1通过，中位366.50→
355.42µs（−3.0232%，4/5配对更快）；但同一任务在后续矩阵入口因漏传
table_history而exit1。此为本次提交参数错误，矩阵尚未执行，原任务保持
FAIL。补入至少history+128的固定页表容量，单独任务
task_20261021_170809_210455428383完成同址S1～S6六项，未重复S6计时。
提交前新增对应检查，缺失、差1行、刚好满足、更多容量与普通L1合计
9项CPU回归通过；旧冻结harness不改，新提交使用v110副本。

B8 task_20261021_165204_15601729493 exit0，中位613.56→609.60µs
（−0.6454%，3/5配对更快），保留小幅局部正向结果及全部窗口。其T41/R12
补验任务task_20261021_170809_210475710362的8项metadata+4步连续状态
全部通过。B4另测task_20261021_171014_231115820997 exit0，精确/图/L1
通过，但中位487.38→489.72µs（+0.4801%，2/5配对更快），不采用该区间。
据此在v113仅为长历史T<64且不处于既有G6小M16 QA并行区间启用宽N；
短历史保留原路径。B1/B8所有constexpr值及核内文件与实测版一致，B4
关闭代码另行编译核对。全部数据见qb_wide_results_v113.json。

新的独立组合将T≤8的Hadamard分工由4核改为2核，把Q_B由20核扩为22核。
不是减少常驻MIX总核数，也不删除原阶段通知。该容量域query最多两个
M256块，compact最多两个M16/N64工作项，旧版实际工作者即0/1；
52个token/compact组合证明其tile与owner保持不变，Q_B列块唯一覆盖，
KV工作置换仍为双射。B1开启及B8关闭CPU编译通过，Vector和Host不变，
B8完整可执行代码亦不变。

任务task_20261021_172027_32968638376直接比较“宽N”与“宽N+两核分工”，
含五窗S6和同址S1～S6；参照明确标记为已验证但尚未采用的实验版本，
不是BSH或生产默认。新增快照与非法参照两项CPU检查通过。组合结果
尚待回收；当前生产仍为combinedv106，正式八档GM0.988089028未达0.90。

### 2026-10-21：暂停期间同步上游与公共测试环境

用户要求“拉取一下最新的代码，测试的环境有更新”。本次仅同步代码与测试
环境，goal 状态仍为 paused，未提交新 NPU 任务，也未 push。
从 origin/dsv4-flash-pto-v0.25.1rc1 合入 3c92ca84 的 8 个上游提交；
本地 11 个提交保留，合并提交为 6a86e5e6。23 个未提交文件已先备份并恢复：
22 个字节一致，单测文件自动合并了上游 MPMD 路由覆盖及本地 17 个边界用例。
备份位于工作区 `.cache/csa-sync-backups/20261021_193646`，另保留对应 stash。

公共工具链发布为 `csa-shared-20261021-upstream-fixes`，状态 validated：
PyPTO e891351e、PTOAS 914863ed（0.69）、Simpler 57e49923。
发现上游 selection.py 按固定层数定位工作区，在嵌套 worktree 中产生
`.worktrees/.worktrees` 错误路径。现改为向上定位公共环境文件，再读取主检出
唯一清单；新 CPU 编译与独立探针入口也补接公共 bootstrap。
未修改或重建 SDK。父子进程实际加载路径一致，115 项相关 CPU 测试通过，
Ruff 与 shell 语法检查通过。检查证据见
`tests/pypto_test/csa_spmd_tnd_20261020/shared_toolchain_sync_20261021.json`。
本次没有核内编译或 NPU 性能验收；历史版本结果不得与新工具链拼接计算收益。

暂停现场仍保留 v114 的未提交宽 N/两核前缀组合。此前 v112 和两个 v114
设备任务已完成，旧 recovery_state 的“running”已纠正：B1 的 v114 正式
BSH 对照为 352.66→359.14µs，负向结果保留；B8 为 612.82→603.08µs。
v115 三方诊断未提交设备，已有 CPU 测试为 1 passed / 1 failed（测试 AST
选中了 service_cost 的循环），本次不修复或继续该优化。后续需先明确恢复
goal；冻结旧启动器、原始结果、未完成代码均保留，不能直接作为新环境验收。

### v116～v117：恢复原目标，在公共工具链重建三方对照

用户显式将 goal 恢复 active；继续原 TND 功能、服务、状态与八档 S6 等权
GM(TND/BSH)≤0.90，未改目标或允许 push。上轮环境同步属于有效进展。
核对当前已采用 BSH 与冻结基线，仅存在此前记录的 inactive-B1 静态形状修复。
保留 v114 工作区候选和此前正负结果，归档于 qb_prefix_results_v114.json；
暂不据旧工具链结果认定新公共环境的采用收益。

在公共 PyPTO/PTOAS/Simpler 下，v114 的 B1、B8 完整 CPU 后端编译通过。
Mat 峰值 475136B、Acc 131072B、Left/Right 各 65536B；宽 N 的 Right
两个 32KiB 槽保持独立。这里只证明编译和容量，没有据此宣称设备精度通过。

三方诊断固定为 BSH、已采用 TND v106、v114，六个窗口分别使用三者的全部
六种排列，每个位置各出现两次。每窗原 L1 计时体和严格数值 verifier 的 AST
不变；额外检查三方私有工作区前后保护区。首尾计时覆盖完整链路，恢复输入
仍在计时之外。v115 的失败测试错误地选到了 service_cost 循环，现限定到
profile 后修复，两个实际快照/执行顺序 CPU 检查通过。旧 v115 不改写。

B1 首次 task_20261021_200225_14158972838 因错误混用服务路由与 metadata
矩阵，被入口在初始化 NPU 前拒绝，保留 exit1。新增提交前 fixture 互斥检查，
13 项定向 CPU 检查通过；不放宽原任务限制。改用独立矩阵的 B1 任务为
task_20261021_200334_151444013048，B8 为 task_20261021_200244_14437146776。
本检查点两项均仍运行，结果待收集；三方诊断明确不冒充八档正式性能验收。
真实服务路由后续使用独立 fixture。源代码和已完成证据先阶段提交，不等待
10% 总目标达成才保存，也不覆盖旧结果或重复旧任务。

### v118～v123：三方结论、RoPE尾块局部收益与新公共环境整轮回归

三方任务均 exit0，完整严格/图/保护区/代际检查通过。B8 组合中位616.26µs，
已采用TND v106为640.50µs，BSH为641.85µs，增量改善3.7845%；B1组合
372.14µs、v106为370.30µs、BSH为363.39µs，未见增量收益。B1同址S1～S6
矩阵通过。六种排列及全部原始窗口见 three_way_shared_results_v117.json，
不将三方诊断标成正式八档性能验收，也不拿旧SDK的B1正向结果替代它。

进一步只修改Indexer query反量化/RoPE尾部：原不足8行逐行执行，候选复用
原完整M8分支的逐元素数学。四个输入load按bs-dq_t0裁剪，只在UB内扩展
描述符；各行互不归约，最终两次BF16 store均恢复实际行数。FP32乘法顺序、
RoPE乘乘加和BF16 rint不变。完整块数学AST及其余helper不变；单MIX仍24组，
Vector最高地址163872B，未增加Host metadata读取或设备前后处理。

task_20261021_201840_26967324212完成B1整链严格、图、保护、代际与S1～S6，
直接对照v106：376.22→371.58µs，改善1.2333%，4/5配对更快。此为增量TND
结果，仍不可替代BSH目标。详见 query_rope_tail_results_v118.json。
task_20261021_203211_366084023578另完成39个独立边界检查：T=1～9、15、16、
17、23，每档三个seed（含全零）。完整输出逐bit相等，未写padding、输入只读
及同址图重放均通过；没有借微核耗时宣称整链性能。

独立探针前两次入口失败均保留：v119复制shell后runner误从runs目录解析，
v120未接受队列追加的--device参数，均未执行待测核。修正为仓库绝对runner
入口并校验device与TASK_DEVICE一致；实际迁移后的CPU启动器和两项CLI
回归通过。不能把这些失败记为算子数值失败，也不能丢弃原任务记录。

新v122组合在当前仓库服务代码上冻结：宽N只留已有长G6的32≤T<64区间，
所有档位恢复四个前缀生产者，并组合已验证的RoPE尾块。B1/B8完整CPU编译
通过。候选仍在独立实验包，尚未覆盖生产默认。八档全部重新同任务配对BSH，
使用公共SDK、完整72核五窗L1和同一候选/参照/harness；不复用旧SDK性能数据。
新增公共发布校验要求validated、相同revision清单和匹配的post-run审计；
11项定向CPU检查通过，包括拒绝增量数据冒充BSH。

首两档正式结果：B1 BSH365.92→TND373.88µs（回退2.1753%），B8
618.58→606.54µs（改善1.9464%）。负向B1保留，其余六档已提交并仍在运行。
完整选择见 cohort_shared_combined_v122.json；尚无新版本完整GM结论。

task_20261021_204517_18664022262通过T65/B11真实单层服务检查，query为
十个6加一个5，确有活动token进入多wave尾块。正式模型准备钩子/custom-op
命中SPMD，框架图、反转query/物理页的同址metadata重放、空输入安全处理、
短历史Native回退及保护检查通过。独立事件计时中，直接图中位835.730µs、
服务图1220.050µs，服务增加384.320µs；含Host提交与Native compact生产者，
不能当作incore L1或整模型性能。详见 shared_mixed_service_v123.json。
T41/R12的metadata与连续状态矩阵另行提交，不与服务fixture混用。


### 2026-10-22：公共八档收齐；按用户要求单独验证均匀与混长 B16

v122 八档已经全部 completed/exit0，完整严格回归通过。正式报告为
cohort_shared_combined_v122_result.json，GM(TND/BSH)=1.000719124，未达到
0.90；原始五窗口及负向档全部保留。v122仍是私有候选，未替换默认实现。
T65真实单层服务任务已通过；T41/R12矩阵 task_20261021_205606_81859127914
也已 completed/exit0、stage=complete、passed=true。此前本日志的“运行中”
是提交时状态，不再作为当前状态；恢复文件同步更新，不重做这些验证。

用户随后暂停广泛优化，本次新要求仅验证128K/B16的全S6与S1～S6均匀混长。
16不能被6整除，固定混长为[1,2,3,4,5,6,6,5,4,3,2,1,1,3,4,6]，
每种长度出现2或3次，总T=56、平均S=3.5；全S6总T=96。各请求历史都为
131072，容量分别等于真实T，没有dummy。沿用v122、公共SDK、正式第4层
W8A8/NZ2权重、seed1024及现有精确回归/图/保护/完整72核L1测试器。

新增run_s6_mixed_compare.py仅按顺序启动两份现有探针，不改算子或计时体；
同一task分配一张卡，两组分别在独立进程取5个交替TND/原SPMD窗口。
原SPMD对照也消费本组真实Q/L，作为同输入内核归因，不冒充原BSH服务
已支持混长。结果同时报告总延时、每有效token耗时、工作量变化及同输入
实现差异；不能把96→56个token的耗时下降直接称为等工作量布局加速。

CPU前检8项通过：使用已有真实S6报告验证正例，并拒绝错误query/history、
缺核、统计不一致与新增回归，检查队列追加device参数。Ruff/格式及差异
空白检查通过。单卡任务 task_20261022_005219_338877432506 当前pending；
输出runs/requested_s6_mixed_b16_20261022_v124。源码与启动器已冻结，
该结果独立于八档验收。本轮完成这组比较后停止，不恢复广泛优化，不push。

队列入口说明：首次任务task_20261022_005219_338877432506在`task-submit --timeout 45 --wait`
超时后被CLI自动取消；这是等待命令的副作用，未分卡、未创建结果目录、
未执行算子。CLI明确报告取消，随后status为not_found且find无匹配任务；
以完全相同的冻结命令重新排队为task_20261022_005731_366392031421。旧ID与原因保留，后续只用只读
`--status`轮询，不再使用有取消副作用的超时等待。

排队期间仅作CPU工作量核算，保存s6_mixed_workload_model_v124.json。
两组G6均为16个query组、每组6叶；全S6的24个AIC各22个Score1024块，
混长总块数528→520，但最忙核仍22块。当前尾组保留完整物理M384，
因此T96→T56不意味着Score Cube计算同比减少41.67%。这是源公式模型，
不冒充实测瓶颈或耗时。实际性能待上述唯一任务完成。

任务task_20261022_005731_366392031421已completed/exit0，两组均在0号卡
完成原单SPMD精确回归、自身图/保护区/代际及5个交替完整L1窗口。
全S6：原808.84、TND815.34μs（TND慢0.8036%）；均衡混长T56：原773.80、
TND739.34μs（TND快4.4533%，5/5配对更快）。TND两种工作负载之间
耗时降低9.3213%，token数降低41.6667%；每token耗时8.4931→13.2025μs。
同输入的实现收益与减少token的收益分开报告，保留全部窗口和Native原差异。
两个形状的源码/参照/测试器/SDK快照、实际运行库均一致；结果及参数见
s6_mixed_compare_result_v124.json。完整说明并入本文件
[全 S6 与固定混长对照](#csa-tnd-s6-mixed-b16-20261022)。此结果不替换原八档任一窗口，
没有追加优化或采用v122；完成此次单独验证后停止，原广泛优化goal仍paused。

### 2026-10-22：准备下一轮九档等权10%优化goal prompt

用户要求基于当前版本制定下一步优化计划，形成goal prompt，目标综合再提升10%。
用户最后明确采用八档S6加128K/B16固定混长，共九档等权计入。完整 prompt 已归入本文件
[v122 之后的九档优化目标](#csa-tnd-next-10pct-goal-20261022)。

新计划的分母是已测v122 TND源码，最终同输入成对测量；九档耗时比的几何平均
要求不超过0.90。不会继续拿旧单SPMD/BSH作分母，也不会把T96变T56的工作量
变化重复记作优化收益。计划覆盖B1/B16控制与输出分工、短B32半叶UB、混长
有效计算裁剪与负载均衡、长历史Score生产消费节奏，以及最终九档和动态状态验收。
保留历史负例、严格精度、公共工具链、阶段性本地提交和未经新指令不push等约束。

本次仅编写计划文档，没有修改算子、提交NPU任务或创建/恢复goal；原goal仍paused。

<a id="csa-tnd-s6-mixed-b16-20261022"></a>

## 2026-10-22：全 S6 与固定混长 B16 对照（归档）

本轮 TND 在混长输入上取得了实际收益：相同混长输入下，相对原单 SPMD
内核耗时降低 **4.45%**，5/5 成对窗口更快；全 S6 则慢 **0.80%**。
TND 自身从全 S6 到混长的耗时降低 **9.32%**，包含输入工作量和策略变化。

### 输入与测量口径

- 每个活动请求的已有历史均为131072，B=16。全S6的实际T/容量均为96。
- 混长Q长度为 `[1,2,3,4,5,6,6,5,4,3,2,1,1,3,4,6]`，实际T/容量均为56，平均S=3.5。
- 16不能被6整除，各S分别出现3、2、3、3、2、3次，次数差至多1；没有dummy/padding token。
- 同一队列任务、同一张0号卡；两个形状按顺序在独立进程执行，各取5个交替的TND/原SPMD窗口。
- 完整HC_pre → RMSNorm → CSA → HC_post，全72核最早incore开始至最晚结束，包含等待和阶段间隙。
- TP1、正式第4层W8A8/NZ2权重、BF16、fixture seed1024。形状不同会改变RNG消耗，payload并非逐字节子集。
- 候选固定为私有v122，尚未采用为默认；原内核为a241fa61加inactive-B1静态形状修复，双方150tick。
- 公共工具链csa-shared-20261021-upstream-fixes；两个形状的算子、参照、测试器快照及实际运行库一致。

### 实测结果

| 输入 | 实际token | 原单SPMD内核/μs | TND/μs | TND相对原内核耗时变化 | TND μs/token |
|---|---:|---:|---:|---:|---:|
| 全S6 | 96 | 808.84 | 815.34 | +0.80% | 8.49 |
| S1～S6均衡混长 | 56 | 773.80 | 739.34 | −4.45% | 13.20 |

相同输入的实现差异使用 `(TND/原内核−1)×100%`。混长列的原内核直接消费
本组真实Q/L，是内核层面的归因对照，不代表旧BSH服务入口已支持混长。

全S6原内核五窗：809.34、799.56、801.76、815.04、808.84μs。
全S6 TND五窗：815.02、819.46、815.34、810.12、823.78μs。
混长原内核五窗：772.86、794.42、770.02、776.34、773.80μs。
混长TND五窗：736.46、733.98、749.48、739.82、739.34μs。
全部窗口保留，没有剔除较慢值或拼接历史结果。

### 结果解释与下一步依据

TND总延时从815.34降到739.34μs，节省76.00μs、降低9.32%；实际token减少
41.67%。每token耗时增加55.45%，说明本实现的成本未随有效token数线性缩小；
这是两种工作负载的归一化差异，不能单独解释为TND引入了55%的开销。
同输入的4.45%对照才是本轮观察到的候选实现收益。

源码公式核算：两组仍有16个G6 Score组、每组6叶，尾组仍执行完整物理M384。
Score1024块总数528→520，但最忙AIC仍为22块；Attention最忙组的query数
4→3、稀疏块数20→15。这是离线工作量模型，不是阶段耗时或硬件忙闲实测。
它支持后续检查无效query计算与最忙核分工，不能将总延时差唯一归因到某阶段。

T56还会关闭early geometry、physical-row plan、output work balance，开启
Top-K短前缀与宽N的Q_B。由于同时存在工作量和分档变化，本轮不能单独量化
某个开关的贡献，也不能推断所有混长组合都有同样收益。

两组均通过原单SPMD的输出、Top-K/score、cache/state精确回归，以及自身
图重放、保护区、72核代际和计时前后检查。已有Native数值差异仍单列，
没有放宽阈值或改判为Native精度通过。未做新的整模型或连续decode验收。

### 复现与证据

任务：`task_20261022_005731_366392031421`，completed/exit0。
首次排队尝试被CLI超时等待自动取消，未分卡、未执行；旧ID和恢复原因保留在任务记录。

- [结果、样本与策略差异](../../../tests/pypto_test/csa_spmd_tnd_20261020/s6_mixed_compare_result_v124.json)
- [冻结命令及任务记录](../../../tests/pypto_test/csa_spmd_tnd_20261020/s6_mixed_compare_job_v124.json)
- [CPU工作量模型](../../../tests/pypto_test/csa_spmd_tnd_20261020/s6_mixed_workload_model_v124.json)
- [对照入口](../../../tests/pypto_test/csa_spmd_tnd_20261020/run_s6_mixed_compare.py)

原始逐核记录、完整报告、快照和编译产物保存在本地
`tests/pypto_test/csa_spmd_tnd_20261020/runs/requested_s6_mixed_b16_20261022_v124/`，
大文件未纳入Git。这次仅完成用户指定的两组验证，八档目标仍暂停，未追加优化或push。

<a id="csa-tnd-next-10pct-goal-20261022"></a>

## 2026-10-22：v122 之后的九档优化目标（历史口径）

本节保留当时设置下一轮 goal 的完整 prompt；后续执行及暂停情况见下节过程记录。
用户最终确认的口径是：八档 S6 加 128K/B16 混长，共九档等权计入。

### 目标与范围

基于本会话最新完成设备验证的 CSA 单 SPMD TND v122 候选，继续优化完整
HC_pre → input RMSNorm → CSA → HC_post，在保留原生 TND、动态请求边界、
Native 分页 KV/cache/state、实际服务路由和图重放能力的前提下，达到：

```text
r_i = median_L1(优化后TND，场景i) / median_L1(冻结v122，同一场景i)
G = exp(sum(log(r_i), i=1..9) / 9)
验收条件：G <= 0.90
```

“再提升10%”在本轮明确定义为完整链路 L1 耗时的等权几何平均降低至少10%。
不按 batch、token 数或绝对耗时加权，不要求每档分别降低10%。逐档报告正负结果，
优先通过合理分档隔离回退，不能删掉不利场景或换成更有利的混长分布。

本轮性能分母改为冻结的当前 TND v122，不再以旧单 SPMD/所谓“BSH版”为分母。
此前八档对旧基线的10%目标及历史结果保留，不将两种目标混为一谈。
原有功能、精度、状态和服务验收约束继续有效。

仅包含当前 decode 范围、单卡 TP1、正式第4层 W8A8/NZ2 权重和 BF16 激活。
保持计算主体为一个 MIX 单 SPMD，不扩展 prefill、新并行方式、ACLNN/SuperKernel
或完整模型端到端性能目标，不把增加核外预处理后的内核局部变快当作整链收益。

### 工作区与固定基线

复用本会话隔离 worktree：

```text
/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-spmd-tnd-20261020
branch: feat/csa-spmd-tnd-20261020
upstream: origin/dsv4-flash-pto-v0.25.1rc1
```

以下相对路径均相对于该 worktree。其他会话的工作区与源码不得直接修改。
开始前核对实际 Git、goal、任务队列及交接状态，保留已有提交和本地修改；
需要同步上游时先保护本地成果，再处理合并和测试入口变化。

**本轮唯一性能基线源码：**

```text
tests/pypto_test/csa_spmd_tnd_20261020/runs/shared_combined_candidate_v122
```

相关证据：

- `shared_combined_candidate_v122.json`：候选来源、分档与源码清单。
- `cohort_shared_combined_v122_result.json`：公共环境的完整八档结果。
- `s6_mixed_compare_result_v124.json`：最新同卡全S6/混长结果、全部窗口及策略差异。
- `s6_mixed_workload_model_v124.json`：按源码公式计算的工作量模型。
- [全 S6 与固定混长对照](#csa-tnd-s6-mixed-b16-20261022)：测量解释与复现索引。

前四个文件位于 `tests/pypto_test/csa_spmd_tnd_20261020/`。
本会话历史采用/否决记录在 `vllm_ascend/ops/pypto/DSV4_CSA_SPMD_TND_IMPLEMENTATION_LOG.md`。
另一会话的Score/Attention负例记录在主检出的
`vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md`；混长live-half和叶轮换
记录在主检出 `tests/pypto_test/csa_mpmd_tnd_20261021/` 下的
`score_live_half_results.json`、`score_leaf_rotation_results.json`。这些只作选题证据。
目前 worktree 的默认实现与这个私有候选可能不同；不能未经核对就拿默认包替代基线。
先冻结完整 v122 源码和复现方法，再建立优化工作副本。保存轻量源码、清单与必要
重建脚本，不让唯一基线仅依赖可被清理的临时目录。不要在比较过程中移动分母。

最新同卡参考值是128K/B16全S6的815.34μs、混长T56的739.34μs，二者均为v122。
这些数字只帮助判断量级；最终验收要与同轮实际运行的冻结v122成对比较。
不能把这两个新数字和旧八档中的最快窗口拼成“同轮九档基线”。

### 九个固定业务场景

128K表示每个请求进入本轮前已有历史131072；8K表示8192，另加本轮query。
每个场景的优化前后必须使用完全相同的输入、请求顺序、metadata、历史长度、
物理页映射、状态起点、权重、dtype和容量。

| 场景 | 历史长度 | B | 每请求S | 实际T及token容量 |
|---|---:|---:|---|---:|
| 1 | 131072 | 1 | 全6 | 6 |
| 2 | 131072 | 4 | 全6 | 24 |
| 3 | 131072 | 8 | 全6 | 48 |
| 4 | 131072 | 16 | 全6 | 96 |
| 5 | 131072 | 24 | 全6 | 144 |
| 6 | 131072 | 28 | 全6 | 168 |
| 7 | 8192 | 16 | 全6 | 96 |
| 8 | 8192 | 32 | 全6 | 192 |
| 9 | 131072 | 16 | 下述固定混长 | 56 |

第9档的query长度固定为：

```text
[1,2,3,4,5,6,6,5,4,3,2,1,1,3,4,6]
```

各长度出现2或3次，平均S=3.5，无dummy/padding，请求容量16。
不能把T96变成T56本身的工作量收益记作本轮优化收益：第9档也必须比较
优化后TND/T56与冻结v122/T56。每token耗时和有效token吞吐只作辅助指标。

“单SPMD”是执行方式。历史旧基线的核内已能读取动态Q/L，本轮最新测试中
旧核也通过TND绑定消费56个真实token；旧核773.80→v122的739.34μs不能
解释为单纯BSH→TND布局转换收益，也不能作为本轮新增收益再次计入。

### 执行计划与优先级

#### 1. 对齐基线和验证入口，先建立可归因的比较

1. 阅读当前实现日志、最新结果和此前负例，确认没有未处理的设备任务。
2. 复用现有完整链路L1、精确回归、图重放、保护区与代际检查；建立显式标识
   `reference_kind=frozen_tnd_v122` 的增量比较及九档汇总入口。
3. 不把仍固定比较旧BSH的启动器当作新基线，不放宽或覆盖原八档BSH验收器。
   新报告、任务和候选使用独立名称，防止结果混入历史口径。
4. 复用现有测量选方向；不要为“建立基线”先无意义地单跑整套九档。
   新候选的代表档及最终九档，均在测候选时同时测固定v122。
5. 对不清楚的关键路径只补有针对性的阶段/等待证据。优先关注128K/B16的
   S6与混长、8K/B32；B1用于小工作量控制开销，B28用于检查长尾。
   阶段Scalar时戳、仿真周期、PMU和工作量模型均不能替代完整L1验收。

#### 2. 先处理可独立归因的控制与驻留优化

- **B1与128K/B16输出分工：** 在当前公共环境下，以单变量对照复核
  `output_work_balance`。之前相对旧内核的回退不能直接归因到此开关或原生wait。
- **B1空角色控制工作：** 审计INPUT/QR/ROPE等阶段无工作角色的等待和发布，
  只裁掉确实没有消费者依赖的部分。发布还承担业务缓存维护，metadata planner
  可能位于无输入计算的AIV，必须保留其读取本轮Q/L所需的获取协议。
  每核begin_invocation、代际推进及必要FIFO ACK不得被跳过。
- **8K/B32半叶UB：** 当前G2组数上界96使其走GM scratch，而B16上界48已使用
  半叶驻留UB。48是既有实测分界，不是必然的硬件容量上限。评估多波次下复用
  每核半叶UB，以及按历史上界缩小物理暂存的可行性，保留原排序树、叶边界和同分顺序。
  必须审计完整编译后的生命周期/地址，不能只删掉策略判断就宣布收益。

已有RoPE M8尾块、受控宽N Q_B、部分Top-K短前缀和半叶UB已包含在v122中，
不能将这些已有收益再次算入本轮。

#### 3. 混长的有效计算裁剪与负载均衡联合设计

当前T96与T56都有16个G6组；按源码模型，Score总块数528→520，但最忙AIC
仍为22块，尾组仍使用完整物理M384。以下是优化依据，不是已测瓶颈结论。

1. 优先评估 `live<=3` 时省去无有效query的后半段QK/WS，保留原INT8点积、
   FIX缩放/ReLU、FP16系数与FP32累加语义。必要时使用独立静态物理tile分支。
   不能仅缩小valid_shape而忽略A3紧凑L0C行距与传输布局约束。
2. 同时评估按真实可见历史、有效query数、叶长度和排序成本分配工作，减少
   最忙核而不仅是平均核时间。计划由本轮device metadata推导，不新增D2H。
3. AIC与同组双AIV必须使用一致任务顺序。保留逻辑叶编号、pair根地址和
   Top-K归并顺序；如改变物理owner，所有消费者等待必须使用对应真实写者。
4. 优先用可证明的轻量计划/置换；计划生成、通知和额外搬运都计入整链成本。
   不为追求“动态”直接加入一套重型全局任务队列。
5. 无有效query的lane仍可能需要读FIFO并发ACK；跳过数学不代表可以跳过协议。

另一会话的live-half候选出现过平均核时下降、最慢核不降、完整耗时回退，
叶轮换也只有局部证据。只借鉴机制，不直接搬用其性能数字；其混长输入和
本轮固定第9档不同。保持S6完整路径，并检查新增分支/资源是否影响八档S6。

#### 4. 长历史Score的生产消费节奏与实际搬运

这是覆盖多个S6档位的重要方向；不能只优化第9档就宣称九档综合达标。

- 当前已具有双M192、两份独立48KiB QK Acc、早FIX、双槽交接及流式Top-K。
  先确认真实限制是AIC供数、AIV排序、READY/ACK反压还是核间长尾。
- AIV当前在MTE2读完分数后就发ACK，早于排序；不能把“提前ACK”当作新方案。
- 只有观察到缓冲深度造成的短时反压时，才评估交接粒度/深度；必须同时证明
  环槽容量、事件计数、复用和末尾排空，不假设多一个槽一定更快。
- 若受供数或排序限制，优先减少实际搬运、无效计算和重复工作；保留非连续页、
  尾块及精确Top-K回退。减少源代码行数、静态指令位点或等待数量不等于提速。
- 对多档有效的方案按实际资源域组合；允许更多合理分档，保留小幅但可信的
  局部收益，不要求同一个配置在所有场景更快。

#### 5. 组合、回归与最终验收

每次先建立单一可归因候选，在代表档与固定v122直接成对验证；正确且有整链
增量收益后，再补受影响边界并组合。每个组合必须重新核对实际生效路径和资源，
不能把单项不同轮次收益直接相加。最后以同一份候选源码和冻结v122完成九档报告。

### 已有负例与重试条件

下列方向已有完整负例，不能原样重复：Score扩大到N128的多种Acc/WS方案、
Score/Attention跨波交错、QA N64简单移植、部分WS/scale与核映射组合、
Score可见范围外提，以及改变B1叶划分/同分顺序的方案。

Score/Attention交错曾增加Left重装并暴露波内长尾；更少指令批次也曾以失去
流水重叠为代价。只有新证据明确说明资源、数据路径或机制发生了实质变化，
才重新打开相应方向，说明与旧候选的差别。保留失败记录，不反复测同一代码
等待噪声转正。源码位置和完整证据见两份CSA优化日志。

### 正确性与TND约束

- 直接消费Native hidden、Q/L、positions、block_table、slot_mapping与compact metadata。
  不用token//6、T//B或固定request*6推导真实请求归属；不把整批补成B×6来声称支持TND。
- Host已有容量、query上界、历史上界可用于资源分档；实际请求长度、可见范围、
  有效写入和动态调度仍由本轮device metadata决定，不引入新D2H或Host同步。
- 以冻结v122同输入检查输出、Top-K/score和全部cache/state，保持原严格标准。
  原有Native差异单列并保留原阈值，不将旧FAIL改为PASS，不用固定S6多task作混长golden。
- 调整排序、叶划分或归并时，必须覆盖同分、全零、极端可见范围和回退分支。
  原归并方向、舍入点与状态语义不能以“实数等价”代替设备精确证明。
- 对受影响路径验证B1 S1～S6、[2,6,1]、[1,2,3,4,5,6]、T不整除6/8/16、
  C4边界、物理页变化、dummy、请求增减/重排、同址图Q/L/positions变化和连续多步状态。
- 验证真实vLLM单层路由命中及范围外/空输入回退；不能把测试入口直调成功
  或Native fallback成功当作服务TND路径通过。
- 只重验受影响的功能与边界，复用仍有效的既有证据；编译通过或模型计算不能
  冒充设备正确性。最终九档性能必须对应最终候选；复用需完整可执行代码、ABI、
  计时器和环境等价证据，不能依据“关闭了某开关”就认定不受影响。

### 工具链、测试与交付

三个CSA会话共用主检出的唯一清单：

```text
/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/manifest.json
```

遵守工作区及仓库AGENTS.md，先加载公共环境，再使用公共bootstrap的
`--shared-toolchain`。目前记录为PyPTO e891351e、PTOAS 914863ed、Simpler 57e49923，
实际运行前以唯一清单为准。使用validated发布，不在有相关排队/运行任务时原地
编辑或重建共享SDK，不回到个人旧SDK。若统一发布升级，两侧均在新环境重编和比较，
不能用工具链跨版本差值冒充算子收益；避免反复全量逐文件hash扫描。

全部NPU执行使用task-submit；CPU检查不占卡。用只读`--status`轮询状态。
当前CLI的`--timeout ... --wait`超时会取消pending任务，不用它作为无副作用的等待。
保留准确任务ID，不因暂时排队、日志未出现或观察超时重复提交/重启。

计时沿用完整72核单MIX的最早incore开始至最晚结束，包含等待与阶段间隙。
每档至少5个交替窗口，基线/候选同卡同进程、相同fixture状态和等价工作区分配方式，
报告全部样本与中位数。恢复输入、校验和读回放在计时之外，保持现有测试器的窗口边界。
不以Event、局部stage、最短窗口或不同输入的比例替代验收。

最终交付：

1. 可审阅的PyPTO实现、分档规则、启用边界和关闭/回退路径。
2. 九档逐项的基线/候选中位数、全部窗口、比值、回退和等权几何平均。
3. 混长功能、真实服务、图与连续状态的受影响验证证据，以及原有Native差异。
4. 明确哪些方案采用、否决或仅为候选；说明完整链路收益，不重复计算已有收益。
5. 在`vllm_ascend/ops/pypto/`单独维护本轮优化过程MD；关联源快照、工具链、
   命令、任务ID、原始结果和停止原因。保留此前实现日志与全部失败证据。
6. 每个可审阅阶段及时本地commit，中文说明并Signed-off-by；不等总目标完成
   才提交。不提交权重、二进制和大编译产物，提供本地索引。没有新的明确指令不得push。

只有最终候选的九档`G<=0.90`且全部必要正确性、动态状态和服务验收通过，
才宣布goal完成。未达标必须如实报告，不换分母、缩场景或放宽检查来达标。
完成验收和交付后停止，不自行扩展下一轮优化。

<a id="csa-tnd-next-10pct-log-20261022"></a>

## 2026-10-22：v122 之后的九档优化过程（已暂停）

### 2026-10-22：启动、永久基线与首轮单变量候选

用户已设置新 active goal：实现 [v122 之后的九档优化目标](#csa-tnd-next-10pct-goal-20261022)，
过程中参考 cannbot-knowledge 的 ops。此前暂停状态不再适用。
固定口径为八档 S6 + 128K/B16 固定混长，九档等权 candidate/v122 的 L1
中位耗时比几何平均 <= 0.90。当前尚未有本轮设备结果，不宣称新增收益。

本轮开始于 `a584eded`，worktree 为 `csa-spmd-tnd-20261020`；上游仍是
`origin/dsv4-flash-pto-v0.25.1rc1`。没有执行 push。共享SDK清单仍 validated，
release=`csa-shared-20261021-upstream-fixes`，PyPTO e891351e、PTOAS 914863ed、
Simpler 57e49923。没有修改/重建共享SDK。开始时无本会话未完成设备任务。

#### 固化与验收入口

实验目录：`tests/pypto_test/csa_spmd_tnd_opt_20261022`。
已核对 v122 全部41文件；相对 a584eded 的默认TND包仅 `policy.py` 与
`indexer_query_stages.py` 不同。用固定完整提交、两个原始文件与逐文件SHA256
永久保存基线；`materialize_reference.py` 可在不依赖旧 runs 的情况下重建。
未把当前默认包或旧BSH当作新分母。

增量harness新增只接受永久v122清单的引用，记录 `kind=frozen_tnd_v122`。
沿用原完整链路计时、精确比较、metadata/service检查；额外启用已有
`score_arena_unused_untouched` 断言，确认UB路径不写GM Score scratch。
旧BSH八档验收器未修改。新增独立九档验收器，逐档五组交替72核单MIX窗口；
固定混长 `[1,2,3,4,5,6,6,5,4,3,2,1,1,3,4,6]`，T=56，不得换分布或填到96。

#### cannbot-knowledge 依据与边界

只读使用本地 `cannbot-knowledge/.agents/skills/knowledge-query/SKILL.md`。
首次 discover 返回退出码2：`artifacts/indexes/knowledge.sqlite3` 缺失。
按技能停止索引检索，不重建索引、不扫描全库兜底，不伪造search/ready receipt。
继续阅读此前已明确定位的具体卡片，以下是工程假设依据，不是已完成知识库审计：

- `knowledge/ops/ascendc/guides/synchronization/resident_softmax_state_slot_recurrence_and_reuse_events.md`：
  draft、未定位平台。借鉴同核状态可驻留、复用必须等最后消费者；不直接套其示意代码。
  本轮应用于短历史半叶UB生命周期审计，实际资源和依赖以A3生成代码及设备验证为准。
- `knowledge/ops/ascendc/guides/synchronization/attention_schedule_slots_and_platform_handoffs.md`：
  draft、未定位平台。保留生产/消费身份、READY/ACK与末尾排空；无效数学可裁剪
  不代表协议可以单边跳过。不照搬950的mode4/flag偏移到A3。
- `knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/sort_and_merge_isasi/sort32.md`：
  stable，明确包括A3；32元素块内同分优先原较小位置。半叶UB候选不变更排序树、
  叶地址与归并输入顺序；整树精确性仍由现有TopK/score比较验证。

#### 首轮候选与CPU检查

`output_balance_off` 仅令 `OUTPUT_WORK_BALANCE=False`，分别测128K/B1、B16；
不调整Q/L、tile、其他阶段或wait。代表档descriptor断言只有这一个constexpr不同。

`short_half_ub96` 仅把短G2组数上界48扩至96，测8K/B32；每个AIV顺序处理4波。
源码helper在每叶内创建并消费私有22×384 FP32半叶，33,792字节，未按总组数
复制4份buffer。完整生成AIV代码中只有一个该Tile声明，映射固定UB地址；重用前
有MTE3→Vector等待。保留原六个排序分支、FIFO与全部尾部检查。此项只能说明
候选具备编译与初步生命周期证据，不能证明性能或所有边界已通过。

三个descriptor均已通过CPU parse与完整codegen：
`runs/compile_output_balance_off_h131072_b1/result.json`、
`runs/compile_output_balance_off_h131072_b16/result.json`、
`runs/compile_short_half_ub96_h8192_b32/result.json`，均 passed=true、device_execution=false。

已提交三项独立设备任务，使用同一冻结v122引用及共享SDK：

| 候选/场景 | task-submit ID |
|---|---|
| output_balance_off / 128K B1 S6 | task_20261022_020842_313460812168 |
| output_balance_off / 128K B16 S6 | task_20261022_020844_313577826313 |
| short_half_ub96 / 8K B32 S6 | task_20261022_020845_313708213322 |

输出位于实验目录 `runs/measure_<候选>_h<历史>_b<B>`；完整命令与启动器hash在
旧实验目录 `csa_spmd_tnd_20261020/jobs.json`。只用 `--status` 观察，不用带
timeout的 `--wait`，避免观察超时取消pending任务。候选尚未采用，等待实测。

CPU门禁：新九档/重建/真实启动器快照27项通过，旧增量与共享SDK23项通过；
定向ruff与git diff --check通过。format.sh ci因环境缺少pre-commit退出，
未执行完整仓库格式CI，不把定向检查写成全量CI通过。三项设备任务均已观察为running。

#### 首轮设备结果与取舍

上述三项均 completed(exit=0)，两侧同源SDK审计、严格数值/TopK/score、状态、
图、保护区与代际检查通过。Native原有precision_exact=false继续保留。
全部五窗及源清单见新实验目录 `control_results.json`，不将三个局部对照拼成九档结果。

| 候选/场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 | 取舍 |
|---|---:|---:|---:|---:|---|
| output_balance_off / 128K B1 | 372.26 | 378.94 | +1.7944% | 0/5 | 拒绝，保留原输出分工 |
| output_balance_off / 128K B16 | 803.18 | 801.34 | -0.2291% | 4/5 | 微小正向证据，暂不采用 |
| short_half_ub96 / 8K B32 | 933.52 | 940.12 | +0.7070% | 1/5 | 拒绝，UB驻留不等于整链变快 |

8K/B32候选同时通过score_arena保持37断言，说明设备确实走到了UB路径。
不能将回退解释成开关没有生效。没有修改默认算子，没有重复相同候选等噪声转正。

#### 混长：裁剪与轮换分开建候选

已读另一会话 `prepare_score_live_half.py`、`prepare_score_leaf_rotation.py`
及live-half实测负例。仅借机制，未借用其不同混长fixture的性能数字。
本轮 `live_half` 对实际live<=3的长G6使用固定M192，只删除全零后半QK/WS；
保留六行FIFO、两个AIV ACK、全部READY/消费/drain调用、原live>=4数学AST。
动态valid_shape不改变物理Acc行距，避免已知A3紧凑L0C传输问题。

`live_half_rotate` 在此基础上，仅对少于24个G6组执行
`group=(ordinal+leaf)%groups`；G2和>=24组保持原分工。单SPMD的根消费者同时
用`ordinal=(group+groups-leaf%groups)%groups`反推真实owner；等待原完整阶段
完成信号，不把FIFO ACK作为根完成。逻辑叶/根地址与归并顺序不变。
CPU执行实际生成的映射/等待helper，覆盖空组、1..64组代表边界、1..32叶、
G2/G6和双AIV，验证双射和每一个消费者真实写者；3项检查通过。

固定混长的源码成本模型：最忙AIC的M192半段工作单位44→39，两个AIV lane
各自最大有效query×Score步数(66,55)→(66,36)。这不是耗时预测，也表明
lane0最长工作仍未降低，不能凭AIC平均节省就认定整链提速。
两个候选的完整CPU代码生成均通过，源码清单与借鉴来源hash记录在
`mixed_candidates.json`，各自直接与冻结v122配对五窗：

- `task_20261022_021508_34186829200`：live_half，128K/B16固定混长T56。
- `task_20261022_021510_342018110895`：live_half_rotate，同一固定混长。

两者尚在运行，不能宣布采用或扩大到八档S6通过。

#### 当前版本的定向阶段证据

已准备独立 `phase_harness`，保留shared SDK原生150tick等待和通知，只在每核
每阶段独占512B槽的保留字16..19记录时戳及累计GM wait。原信号行不改；
正式L1 harness不改。诊断允许这四个保留字，其余padding/数值/图检查保留。
阶段区间包含插桩扰动与FFTS/实际流水等待，不将减去GM wait的余数叫纯计算。
新九档门禁显式拒绝带phase_clock或diagnostic_phase_clock标记的结果。
CPU对真实当前SDK整份C++验证：移除插桩后逐字恢复全部原代码；拒绝重复
插桩和非150等待。3项定向检查通过，九档27项仍通过，新增脚本ruff通过。

`task_20261022_021917_359785823204` 已提交128K/B16/S6的v122对自身阶段诊断；
不计入任何性能收益，待核对编译实际插桩、完整数据与阶段端点后再选供数/排序/
等待方向。未用旧v25不同SDK的插桩时间直接判定当前瓶颈。

#### 第二轮结果：保留局部正向证据，九档目标未达成

混长两项均 completed(exit=0)，完整严格回归/图/保护/代际通过：

| 候选/场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 |
|---|---:|---:|---:|---:|
| live_half / 128K B16 混长T56 | 765.98 | 745.48 | -2.6763% | 4/5 |
| live_half_rotate / 同固定混长 | 734.24 | 725.30 | -1.2176% | 5/5 |

两项独立任务/卡，只与各自同轮v122相除；不能比较745.48与725.30来断言
轮换更好，也不能叠加这两项收益。全部窗口及源清单见 `mixed_results.json`。
优先继续单独live_half的回归，轮换保留候选，未正式采用。

live_half追加S6与动态检查：

- `task_20261022_022636_391372328412`，128K/B8/S6：611.96→611.62μs，
  -0.0556%，3/5配对快，基本持平。
- `task_20261022_022637_391508821796`，128K/B16/S6：795.34→785.52μs，
  -1.2347%，5/5配对快。S6都走完整M384分支，不能把这个结果解释为省掉了
  无效query；加入动态分支后的代码生成/布局及波动均需与真实裁剪收益区分。
- `task_20261022_022508_384730226445`，T41/R12：同址S1～S6、混261/123456
  共8场景及4步连续状态全部通过，12次同图重放；见 `live_half_metadata_t41_summary.json`。

两个S6任务也均 completed(exit=0)，全五窗及Native已有差异单列于
`incremental_progress_results.json`。它们不覆盖尚未测的其它档位或真实服务。

#### 阶段诊断与排序合批候选

阶段诊断 `task_20261022_021917_359785823204` completed(exit=0)，每侧1619条
有效发布端点，均为同一个v122源码。数据见 `phase_v122_b16_s6_summary.json`。
两次非计时流程的热身/缓存状态与分配不同，整体端点835.60/785.50μs不能当成
候选收益。较稳定的局部观察是Score AIV区间中位192.38/188.59μs，AIC扣除
累计GM wait后的区间仍约200μs量级；包含FFTS等待与流水，不能称纯计算。
Attention后O_A按列分工仍有约三波，每波约26μs。原归因候选优先研究Score
重复排序发射，未原样重试N128、改数学归约或Score/Attention跨波交错负例。

`batched_sort3_v2`基于冻结v122独立制作，不叠加live_half。仅G6、关闭短前缀，
且本AIV三个query均有完整2048有效分数时合批Sort32及三级MRGSort。先逐query
检查真实Q/L/positions；尾段、失活/陈旧位置、短前缀与其他组型走原完整逐query
分支。每个query仍有各自索引与根，末级合并顺序保持new-before-old。

依据为已读cannbot Sort32卡及当前固定PTO-ISA
`327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`的
`include/pto/npu/a2a3/TMrgSort.hpp`：format1每次归并四个等长列表，允许重复。
6144个分数的Sort32重复192次；12288个FP32 pair元素在blockLen64/256/1024
时重复48/12/3次，边界恰好保持三个query的独立2048段，没有跨query排序。
底层接口事实不代替设备同分检查。

首版CPU被拒：INT32的batch_sort_valid重新赋普通字面量1，后者推导为INDEX。
保留 `runs/compile_batched_sort3_b16/result.json` 与旧候选清单；v2显式INT32赋值，
并用query_group_size表达未启用分支的物理形状。v2完整CPU代码生成通过，
AllocateMemoryAddr的Vec声明地址上界180224B，低于当前188416B预算；
证据 `batched_sort3_v2_resources.json`，未修改共享SDK。

`task_20261022_023226_417823515723` 的128K/B16/S6完整五窗已 completed(exit=0)：
v122 802.56→候选786.78μs，-1.9662%，5/5配对快；严格输出/TopK/score、cache/
state、图、保护区及代际均通过。保留为正向候选，不能与live_half收益直接相加。
已追加以下受影响边界，尚未收齐：

- `task_20261022_024003_29860510152`：B16/S6历史Indexer key置零制造大量同分，
  correctness-only，不混入性能。
- `task_20261022_024004_29935615774`：T65/R16，S1～S6/混长、padding及连续状态矩阵。
- `runs/compile_batched_sort3_v2_b28`：B28完整CPU编译，收取后再决定设备代表档。

固定基线的两个覆盖文件改用`.py.source`扩展名，保留原始字节及41文件清单，
避免格式器修复历史源码的未使用变量而悄悄改变分母。重建与九档门禁27项再次
通过，当前实验脚本ruff通过。没有改默认生产算子，没有push。当前goal保持active；
九档最终几何平均尚未形成，10%目标尚未达成，恢复入口为实验目录 `current_state.json`。

B28完整CPU编译已通过，追加成对五窗任务`task_20261022_024155_4403114082`，
输出`runs/measure_batched_sort3_v2_b28`；不重复已完成的B16测量。

Git状态补记：共享origin引用在02:06:49被外部push更新到059dab4a，导致
ahead/behind计数变为943/3；本会话仅新增三次本地阶段提交，没有fetch、push、
rebase或改写其他会话。保留当前固定基线和工作树，不能把943理解为本轮新增提交数。

### 2026-10-22：继续控制优化、否决O_A重分块、组合回归

上一goal轮属于实际进展：新增三次本地阶段提交、已完成代表档成对测量，
未缩小九档目标。本轮开始核对goal仍active、工作树干净、公共SDK仍validated。
最初16卡由CI占用；保留原三个pending handle，没有重启或取消。随后三项均
completed(exit=0)：排序合批的B16大量同分、T65/R16八项metadata加四步连续状态
全部通过；B28中位1163.44→1154.60μs，-0.7598%，五窗均快。
完整证据 `batched_sort_validation_results.json`；没有把同分fixture计入性能。

#### B1空发布与O_A行tile：CPU模型和实际结果分开

`b1_idle_publish`只在既有BALANCED/G2域裁掉INPUT/QR/ROPE的空角色发布。
INPUT无业务的metadata planner仍执行fence/cacheinvalid/fence，读取当轮Q/L、
positions和页表；其他空角色保留其后原QA获取。所有72角色begin_invocation、
代际推进、活跃发布、FIFO ACK及数学不变。它不同于旧v22直接删INPUT发布的
准备版，明确保留planner获取。关闭的G6/B16完整C++仅去独立pto调试注释后，
与已测v122相同，见 `b1_idle_closed_g6_audit.json`；此证据不扩大成所有关闭档验收。

`oa_rows48/64`改变实际O_A行tile及Cube/AIV两侧的组任务数/等待owner，
保留N256、原K256顺序与有效行；只是独立原型，未作为通用服务策略。
T144/M48的最忙核M16取整行量由384降至192，T168/192的M64降至256，
但总权重tile读取次数由64增加至96。这里是源码工作量模型，不是HBM字节
或耗时预测。与另一会话E48仅轮换满/尾行块的机制不同。

39项CPU合同执行实际Scalar分工，检查空角色metadata获取、每个真实输出
行/列唯一覆盖、K序列和组等待writer集合；全部通过。五个descriptor完整CPU
代码生成通过。设备成对五窗均completed(exit=0)且严格回归/图/状态通过：

| 候选/场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 | 决策 |
|---|---:|---:|---:|---:|---|
| B1空发布 / 128K B1 | 367.54 | 361.26 | -1.7087% | 5/5 | 纳入待验证组合 |
| O_A M48 / 128K B24 | 1018.50 | 1020.84 | +0.2297% | 0/5 | 拒绝 |
| O_A M64 / 128K B28 | 1165.02 | 1169.98 | +0.4257% | 2/5 | 拒绝 |
| O_A M64 / 8K B32 | 929.00 | 939.08 | +1.0850% | 0/5 | 拒绝 |

四项task依次为`task_20261022_025444_123873330198`、
`task_20261022_025446_124036225902`、`task_20261022_025447_124188114719`、
`task_20261022_025448_124304110989`。源/模型/五窗见
`control_tiling_candidates.json`、`control_tiling_results.json`。没有因为纸面
均衡而采用回退，也未原样重测相同候选等待噪声转正。

#### 继续参考ops资料和其它会话

沿此前明确定位路径阅读cannbot的
`knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/l2_cache_mode_config.md`。
卡片为draft、frontmatter未定位平台，正文列出A3；仅用其解释重复读取与L2
替换之间的权衡，不把更多TLOAD等同于更多HBM事务。本轮不改变L2配置。
当前PTOAS914863ed的TLoad.cpp仍要求L2Bypass且存在offset才调整源地址；
虽有get_l2_cache_offset op，本轮未建立PyPTO到实际runtime offset的有效通路。
不运行仅改标签却没有实际指令变化的设备实验，也不硬编码地址偏移。
知识索引缺失状态保持；没有伪造检索receipt或自动重建知识仓。

只读复核另一会话新提交7d8b40d0/d2f35d5c及query_capacity_gate_results：
MPMD的整DAG分档/前段核数联动尚非当前SPMD可直接照搬的已采用收益。
本会话v122诊断中query Hadamard末端177.84μs早于Main输出198.64和query量化
207.92μs，不能凭增加Hadamard核数就断言关键链更短。未机械移植其核数参数。

#### combined_v1：组合实测不支持直接叠加

独立生成`combined_v1`，只组合B1空发布、live_half和batched_sort3_v2。
两个Score变换操作不同函数，交换变换顺序得到逐字相同源码；仍需实际测量。
B1、B16、混长、短B32四个完整CPU descriptor均通过。新成对结果：

| 场景 | v122中位μs | combined_v1中位μs | 耗时变化 | 配对快窗口 |
|---|---:|---:|---:|---:|
| 128K/B1/S6 | 374.44 | 370.10 | -1.1591% | 3/5 |
| 128K/B16/S6 | 789.32 | 781.40 | -1.0034% | 3/5 |
| 128K/B16固定混长T56 | 744.50 | 743.60 | -0.1209% | 5/5 |

任务为`task_20261022_031228_215821124043`、`task_20261022_031229_215896931619`、
`task_20261022_031231_216043424472`，全部completed(exit=0)，五窗完整保留于
`combined_v1_representative_results.json`。B1和混长的CPU完整C++分别与对应单项
相同（仅去独立pto调试注释）；不能把本轮较小收益直接归为组合相互抵消，也不
选回此前更大窗口替代当前组合结果。这个源码审计不是完整跨任务复用证书。

组合功能追加结果均completed(exit=0)：

- `task_20261022_031232_216162928598`：B1同址图S1～S6六场景通过；该工具不含
  四步连续状态，不把它写成12次回放。
- `task_20261022_031233_216343514801`：T65/R16八场景加四步连续状态通过，12次同图重放。
- `task_20261022_031234_216428831819`：真实模型准备钩子、单层Torch custom-op、
  ACLGraphWrapper通过；反转请求/物理页后同址图重放且没有Python重新提交；
  空输入处理、旧图拒绝和短历史fallback无SPMD提交均通过。范围不是整模型端到端。
  详见`combined_v1_service_summary.json`。Native原有差异仍为precision_exact=false。

归档工具首版假设所有metadata矩阵都有same_graph_replays字段，B1六场景报告
没有此字段，因此归档退出；已改为保留未报告值，不伪造次数。只重新读取原始
结果，没有重新执行设备测试。生产算子仍未改动，综合10%仍未达标。

#### 下一候选：Q_B实际列生产者等待

源码审计发现Q_B之后每个AIV仍等待全24个角色，包含辅助分支空发布者，以及
本AIV没有读取的列块。新的`query_exact_waits`只改这一条依赖边，不叠加组合。
从原反量化函数原样抽取head分组选择，展开helper后整函数数学AST与原版相同；
按本AIV的固定head组计算所读N256/N512列，再反推20个Q_B逻辑worker对应的
真实物理编号。48可整除所有可能的head组数，因此跨token波次仍读相同列组。
相邻列仅等待不同owner一次，保持全部发布/代际/FIFO及有效消费者的缓存获取。
未工作AIV不读q_proj，后继原QUERY发布仍保留业务缓存维护。

传递现有PREFIX_BRANCH_GROUPS/QB_WIDE_N constexpr，未增加Tensor ABI或Host
回读。物理反向映射显式保持取模输入非负，覆盖0<R<24；非协同路径保持原24核
列分工。14项CPU验证从生产者正向分配枚举实际读列，覆盖N256/N512、不同R、
T边界及空角色；B8、B28、混长完整CPU代码生成通过。该改动不同于旧v38/v80
只裁掉无工作消费者的等待，本轮没有重复旧QA/weights/coeff等待裁剪。

已提交三个独立代表档，尚待收取，不预记收益：

- `task_20261022_033052_292215211155`：128K/B8/S6。
- `task_20261022_033053_292310924414`：128K/B28/S6。
- `task_20261022_033054_29245487447`：128K/B16固定混长。

当前优先按这些结果决定下一次组合；目标仍为完整九档等权G<=0.90。
尚无最终九档组合报告，不将三个代表档的平均或单项历史收益代替验收。

### 2026-10-22：收取精确等待结果，尝试B1排序重叠

再次核对goal为active，验收仍是八档S6加固定128K/B16混长九档等权G<=0.90。
上一轮三个Q_B实际列writer等待任务均completed(exit=0)，完整精确/状态/图通过：

| 场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 |
|---|---:|---:|---:|---:|
| 128K/B8/S6 | 595.90 | 593.14 | -0.4632% | 4/5 |
| 128K/B28/S6 | 1153.08 | 1149.98 | -0.2688% | 3/5 |
| 128K/B16固定混长 | 754.34 | 753.00 | -0.1776% | 4/5 |

完整五窗及冻结源/SDK指纹见`query_exact_wait_results.json`。只保留小幅正向证据，
尚未采用或与组合叠加。Native原有差异仍单列，未改精度阈值。

重新检查静态容量历史：旧v28已经在B16/B28完整通过，约0.4%/1.0%改善，且
增加容量签名编译数；当前没有新机制证明值得原样重试，未再次上卡。

#### B1的4096半叶排序：保持叶边界与同分树

`b1_sort_stream`只对BALANCED/G2、真实lane、四个1024分数块、完整可见4096
半叶生效。原4096分支的Sort32和block_len=64/256/1024只形成两个独立2048根；
候选在收到前两个1024块后先排序前根，后两个块到达后再排序后根，最后仍按
right→left归并。原六个有效长度分支完整保留作为回退，没有改变逻辑叶划分。
全部READY/ACK、双槽排空、scale和Score数学不变；不是旧B1改G6/24叶方案。

独立`b1_sort_unused_fills`只把G2不使用的single_roots/single_segments改为
create，在G6保留原full初始化。原B1生成物可见这两次TEXPANDS，不能假设DCE
已经删除。两项均从v122独立生成，避免把无用填充收益归给排序重叠。

参考此前定位的cannbot Sort32稳定API卡（支持A3）：32项内部排序、不同repeat
互不归并，同分保留输入先后。CPU模型执行实际候选表达式及原4096分支，覆盖
全同分、随机同分、正反序、负值、无穷、非零全局索引；另外检查激活条件、
原FIFO调用和非目标函数不变。14项通过。首轮仅测试中的目标函数名称写错，
13项已通过；更正为实际indexer_score_vector_lane_stage后全部通过。
模型用于比较操作树，不能替代设备指令的同分精确检查。

两候选完整CPU编译均通过，无SDK修改。设备任务已提交，未预记结果：

- `task_20261022_041606_4679556748`：stream，128K/B1/S6，五窗完整L1。
- `task_20261022_041612_4726514515`：unused_fills，同一业务档五窗。
- `task_20261022_041614_47370630607`：stream，H139264/S5/Tcap6，全同分正确性。

只读参考另一会话42b14d97：MPMD两条前段局部MIX确有并发核集合重叠导致串行
的窗口，但其ready/gated派发修改不适用于当前单SPMD。当前前段已经包含四query
Hadamard，不重复计算迁移收益。未拷贝其不同fixture的绝对性能作本轮基线。
默认生产算子尚未修改，本轮10%目标仍未达成，未push。

#### B1结果与fixture失败纠正

两项完整五窗任务均completed(exit=0)，严格数值/状态/图通过，结果为：

| 候选 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 | 决策 |
|---|---:|---:|---:|---:|---|
| 4096半叶提前排序 | 367.20 | 375.36 | +2.2222% | 1/5 | 拒绝 |
| 去掉G2无用填充 | 373.42 | 363.90 | -2.5494% | 4/5 | 保留供组合 |

见`b1_sort_performance_results.json`。提前排序增加了分支、两次排序启动及循环
携带根的UB搬运；生成物可见根的独立地址与回边TMOV。这是解释方向，不把静态
指令数量认定为已测瓶颈。没有以“理论重叠”采用回退版。无用填充版生成物已无
single_roots/single_segments，也不把完整收益全部归成12KiB写量的理论耗时。

原S5同分任务`041614_47370630607`在fixture初始化失败，未执行候选：S5/Tcap6/R1
缺少吸收多余padding token的dummy请求槽。这不是算法数值失败或runtime加载故障；
后续“extension未加载”来自清理审计阶段。保留原完整错误，改为真实T5/R1后提交
`task_20261022_041846_57849415001`，仍是H139264/S5同分正确性，不计入性能。
没有增加R改变B1分档，也没有删除原失败记录。

#### 复用既有页布局证明，减少AIC页表读取

`key_page_proof`只在原SCALE_PAGE_COALESCING且split-G6路径中读取已有
META的(first_page,direction)，把每个panel内的页号查表改为first+direction*page。
原逻辑页clamp、逻辑/物理容量检查、两种64B偏移Native key view、搬运次序以及
全部QK/WS数学不变。direction=0仍逐页查原表。没有扩张证明启用域、追加Host
metadata、修改页表或增加证明工作；复用前段原有META获取与业务发布协议。

22项CPU检查执行实际地址表达式，覆盖正/逆序、碎片、共享页、页尾clamp、
1..1025页及关闭域不读取陈旧证明。AIC搬运/数学/同步调用AST全部保持原样。
B28启用和B8关闭完整CPU编译通过。B28五窗任务
`task_20261022_042002_63203019442`正在运行，尚无性能结论；B8只是关闭域编译
检查，不为未生效路径安排设备“提速”实验。

#### 前段顺序候选：Main AIV后移至Indexer输入发布之后

当前阶段端点显示Main输出先于Indexer query量化/系数。新的`main_vector_late`
把COOPERATIVE_PREFIX下原main_compressor_vector_stage完整调用移至IDX_SCALE
发布之后、主Q反量化之前，使AIC有机会在主压缩Vector处理期间先做Score。
原pool→等全部读者→state commit→norm/cache整段及参数逐字保留；非协同路径
保持原次序。中间Indexer链不读取main cache/state，Attention仍等MAIN_OUTPUT。
所有调用AST多重集合不变，未减少任何通知、代际或FIFO边。

这是根据本轮阶段证据建立的单SPMD顺序实验，不是直接移植MPMD的ready选核。
潜在代价是AIV较晚消费Score产生双槽反压，必须完整成对测量。B16/S6和固定
混长正在完整CPU编译，未预记设备通过或收益；生产包未采用。

追加完成状态：S5真实T5同分任务`041846_57849415001`已completed(exit=0)，
严格输出/TopK/score/状态/图通过，详见`b1_sort_tie_results.json`。该功能证据
不撤销提前排序版的性能拒绝结论。Main后移的两个CPU完整编译均通过，随后提交
`task_20261022_042503_85147929684`（128K/B16/S6）和
`task_20261022_042512_85871224786`（固定T56混长），均五窗、冻结v122成对。
保留这两个handle继续收取；没有重复已完成的代表档或变更九档权重。

### 2026-10-22：前段分链的单SPMD原型与B1组合

上一goal轮有实际进展：两个本地阶段提交、B1正负设备证据，以及三个明确运行
的后续任务。本轮重新核对goal active、工作树干净、公共SDK仍validated；没有
重启任何在途任务。已收齐三项，均completed(exit=0)，严格输出/状态/图通过：

| 候选/场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 | 决策 |
|---|---:|---:|---:|---:|---|
| 页证明复用 / 128K B28 | 1161.00 | 1164.34 | +0.2877% | 2/5 | 拒绝 |
| Main AIV后移 / 128K B16 S6 | 811.74 | 803.00 | -1.0767% | 4/5 | 保留候选 |
| Main AIV后移 / 固定T56混长 | 753.38 | 746.70 | -0.8867% | 4/5 | 保留候选 |

证据为`key_page_proof_results.json`、`main_vector_late_results.json`。没有把少查
页表当作已获收益，也没有把两项Main后移结果算作最终九档达标。

#### front_partition_i8_v2：Indexer与主QKV使用不重叠核组

参考另一会话MPMD的两条前段链，但本原型仍是原72核单MIX、一个SPMD提交，
没有加入动态task调度器或第二个incore。共享HC/QA/QR/RoPE之后，在原G6且
COOPERATIVE_PREFIX域内划分：AIC0..7及AIV0..15负责Indexer压缩、query、
Hadamard、量化、系数和scale；AIC8..23及AIV16..47负责主压缩、Q_B和反量化。
KV投影仍由全部24个逻辑worker完整分摊，循环置换到以主分支为先；后继
Score、Top-K、Attention、输出仍使用原24组分工和数学。

只克隆前段调度wrapper以改变WORKERS及真实producer编号，数学helper保持
原文件。主反量化仍沿原48波对齐规则选head tile，32个实际worker只改变条带，
不改RMS归约、尾部recip(sqrt)、量化或K累计顺序。所有Tensor签名、GM工作区、
代际和原阶段发布保留；空角色发布先于本分支业务，真实发布仍在写回后。

关键依赖逐项处理：

- 主分支不再经过Indexer wrapper，显式补回PHASE_META获取，不能假定此前
  的QR/RoPE或主投影已经获取本轮compact offsets/counts。
- 主压缩pool等待AIC8..23的真实投影者；state commit等AIV16..47中全部实际
  pool读者，保留共享prefix读完才能覆盖state的生命周期。
- Attention等待主cache的物理角色为24+16起的真实写者，不能被Indexer角色
  提前发出的空MAIN_OUTPUT骗成就绪。
- scale计划仍有48个逻辑页writer。16个Indexer AIV分别领取owner、owner+16、
  owner+32，整页64B RMW始终归一个物理核；不改变META中的页owner链表。
- 原Score的READY/ACK、槽重用及末尾排空完全保留，不通过跳过空lane通信提速。

首版完整CPU编译在NZ非负性证明失败：`worker-8`虽处于正确分支，当前编译器
不把差值视为非负NZ列偏移。未改共享SDK或权重布局；v2仅把Q_B logical worker
改为physical_worker % 16，后16个连续物理编号形成完整双射，消费者仍等待全部
真实16个写者。原失败源及`compile_front_partition_i8_b16/result.json`保留。

CPU调度检查执行实际两条schedule及克隆wrapper，以数学stub记录事件：22项
覆盖4/8/12组方案、T24/41/56/96/168/192/385、请求4..65，检查全部阶段恰好
发布一次、等待图无环、state读者屏障、主META获取、worker唯一覆盖及48页owner。
首版测试漏识别q_b_stage的NZ alias导致前置断言失败；补齐实际_q_b_nz_stage
签名后全部通过，没有放宽依赖断言。该模型不等价于设备数值或流水性能验证。

v2的B16/S6和固定混长完整CPU代码生成通过。设备五窗对照任务：

- `task_20261022_043809_13687342079`：128K/B16/S6。
- `task_20261022_043810_136987923117`：固定128K/B16/T56混长。

这两项尚未收取最终结果，不预记性能通过。默认算子未修改。

#### combined_v2：单独保留B1无用填充收益

在已验证combined_v1上加入B1去掉single_roots/single_segments无用填充，
与live_half/batched_sort变换交换顺序后源码逐字一致。没有加入负性能的提前
排序或页表复用，也没有把Main后移及分链实验混入。B1完整CPU编译通过，提交：

- `task_20261022_044119_150603731813`：B1/S6五窗完整L1。
- `task_20261022_044120_15073038393`：B1同址图S1～S6六场景。

继续参考已明确定位的cannbot `resident_softmax_state_slot_recurrence_and_reuse_events`
卡片；它为draft且平台未定位，只用于检查最后消费者、槽复用和事件方向。
没有把增加槽数或声明常驻资源等同于必然10%收益，也没有因知识索引缺失自行
重建知识仓。九档等权分母和目标保持不变，未push。

补充13项CPU合同已通过：直接执行Attention的MAIN_OUTPUT等待表达式，确认只等
主分支的真实物理写者；关闭分链guard后，完整Cube/Vector schedule AST与原版
完全一致。合计35项CPU合同，不等同于最终设备边界验收。组合生成器新参数的
局部循环变量曾覆盖报告名，已把变量分开并将本轮清单改回combined_v2.json；
候选目录、源码字节、已编译及已提交快照未变化，没有重复提交设备任务。

#### 前段分链与B1组合收取；按实际32个AIV修正head分组

四个任务全部completed(exit=0)，严格数值、状态、保护区及图检查通过：

| 候选/场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对快窗口 |
|---|---:|---:|---:|---:|
| front_partition_i8_v2 / B16 S6 | 801.20 | 810.78 | +1.1957% | 0/5 |
| front_partition_i8_v2 / 固定混长 | 747.72 | 739.36 | -1.1181% | 5/5 |
| combined_v2 / B1 S6 | 370.72 | 368.68 | -0.5503% | 2/5 |

原五窗分别归档在`front_partition_i8_v2_results.json`、`combined_v2_b1_results.json`。
分链v2不作为通用S6改进采用，混长正向证据保留；不把此轮混长候选绝对值与
其它轮基线拼接。B1组合收益很小且窗口方向混合，不能相加此前空发布/填充的
单项百分比。B1同址图S1～S6六项全部通过，仍不把该工具写成有四步连续状态。

发现分链引入的具体负载问题：主Query AIV由48个变为32个，但原helper继续按
固定Q_DEQUANT_WORKERS=48选择head分组。T96的12个M8块会选16 head，48份工作
分给32核后，16核做16个head、另16核做32个head（每head含8行）。v3克隆该
helper，仅把选择器的48改为实际WORKERS；T96改选8 head，96份工作分给32核，
每核均24 head。总量768个head块不变，这只是负载模型，不是25%实测提速。

每个head的完整计算循环AST与原版相同，FP32两次scale乘法、512列RMS顺序、
rsqrt及尾行recip(sqrt)、BF16边界均未修改。T56仍选择4 head；含尾行T65也仍为4。
原helper不改，只有分链启用时调用新helper，关闭域继续保留原完整schedule。
新增五项CPU检查执行实际选择器，覆盖24/32/40/48个worker和T0..385边界，验证
每个真实(token,head)唯一覆盖及尾行分组。全部通过，v3 B16完整编译通过。

已提交`task_20261022_045018_19010782266`（front_partition_i8_v3、B16/S6五窗），
沿用冻结v122、原工作区方法与公共SDK，尚未预记结果。当前不存在最终九档
G<=0.90证据；生产算子未改、未push，goal继续active。

### 2026-10-22：停止分链扩扫，建立M64/N256 Score独立试验

上一goal轮完成两次本地阶段提交及多项正负对照，属于实际进展。本轮核对原
v3任务仍运行，继续原handle；随后completed(exit=0)，严格回归通过。B16/S6
799.34→809.12μs，+1.2235%，仅1/5窗口更快，见`front_partition_i8_v3_results.json`。
按32个实际AIV均衡head工作量没有转化为完整收益。停止分链这条路线的参数
扩扫；保留v2混长正向和Main后移的小幅收益，但不作为全S6采用结论。

#### 新Score资源组织与历史负例的区别

先重新阅读旧v39/v60/v62记录：N128/M192单96KiB Acc与K128 WS回退7.21%；
N128三M128、两个64KiB Acc复用回退3.45%；仅预取到L1的变体回退8.90%。
不重复这些路径。新的`score_query_pipeline`仍独立基于v122，只作用于原
长G6/SPLIT_SCORE_ROWS域：

- QK按六个独立query分别做M64/N256/K128，使用一个64KiB INT32 Acc。
- 加权归约为六段K64，保持全局head顺序0..383，使用独立16KiB FP32 Acc；
  每个query的64个非零系数仍完整处于一段。不能据此预先声称设备逐bit相同。
- 当前query的QK/FIX与前一query的WS交错，最后补WS5；每256列仅写一次六行。
- Left保留六个8KiB query与六个2KiB系数片，共60KiB。Right为32KiB Key及
  32KiB WS；只把下一Key预取到L1，不同时保留第三块Right。
- 每1024列由原32次QK/32次WS变为24次QK/24次WS；这只是操作数模型，
  不能代替完整耗时。原1024列双槽、逻辑叶、Native页检查、FP16 FIX比例/ReLU、
  AIV Scale/Top-K、同分树和READY/ACK/末尾排空均保留。

#### 必须确认实际启用；资源与传递事件审计

首次CPU编译误用了固定v122描述符，仍是QK_PANEL=64/TO_L0=True，所以新wrapper
没有选中N256 helper。虽编译成功，但资源审计未找到query_qk_acc，不能作为新
候选编译证据。已从实际候选policy重建描述符，先确认相同输入的v122 policy
逐项等于原描述符，再只更新QK_PANEL=256、TO_L0=False；所有Tensor buffer
形状/dtype保持相同。文件`descriptors/score_query_pipeline_b16.json`。

实际启用版`compile_score_query_pipeline_active_b16`完整CPU编译通过。
`score_query_pipeline_resources.json`从AllocateMemoryAddr和最终C++确认：
QK Acc 0..65536、WS Acc 65536..81920互不重叠，Left末端61440字节；
四个panel的Key各32KiB，WS0..4 Right不覆盖仍被六次QK消费的Key。
最后WS5可以复用已退役Key地址，下一panel的Key也只能在WS5读完后覆盖。

首版审计要求退役Key必须有直接M→MTE1事件，因此停止。逐段查看生成C++确认
实际存在QK5 M→FIX、其FIX→MTE1两个完整set/wait链：WS5的Right正从QK5的FIX
结果加载，这个传递依赖已保证最后一次Key读取完成。审计改为明确验证这两段
事件，而非忽略检查；其它QK Acc和Right复用仍检查各自set/wait。最终审计通过，
并验证旧的未启用N64编译产物会被拒绝。没有修改生成C++或公共SDK。

四项CPU合同检查通过：原分档范围、全局K/head覆盖、页/FIFO/Vector数学不变、
实际policy与CPU描述符一致。首版枚举包含非法T>R×query上界，测试已改为要求
新旧policy都拒绝该输入，没有放宽生产验证。设备对照仅提交代表B16：
`task_20261022_050510_249159220949`，五窗完整L1和严格同输入比较，尚待最终结果。

#### 不重复另一会话的multi-ND负例

只读核对当前固定ISA的`pto/common/arch/memory/tload_common.hpp`，确有三维
ND2NZ的ndNum通路，可表示4160字节页stride，但多矩阵stride必须正向。
当前fixture为倒序物理页，不能只把二维Mat reshape就当成逻辑顺序正确。
继续查另一会话记录，找到2026-10-20的Key多ND完整试验：54项UT、12项probe
通过，B28全链1163.64→1310.50μs、0/5快；Mat父双槽assemble不支持，最终
直接进两个Right槽，改变了供数生命周期。此前负例及其数据路径限制保留，
本轮没有重复上卡、换成正序fixture或编辑共享SDK。

当前生产默认未变，最终九档G仍未形成；完整10%目标继续active，未push。

#### M64/N256终态与下一步诊断范围

`task_20261022_050510_249159220949`已completed(exit=0)：严格输出/Top-K/score、
cache/state、保护、图和代际均通过；B16完整五窗798.16→848.84μs，+6.3496%，
0/5快。见`score_query_pipeline_results.json`。拒绝采用，不扩其它档，也不将
更少MMA或合法的Acc分配当作性能收益。当前无待收取设备任务。

开始前已复核历史Scalar READY/ACK探针与phase-clock局限：它们包含异步队列
和探针成本，不能把包围intrinsic的Scalar时钟差直接称为引擎阻塞时间。本轮
没有重复已完成的全核阶段端点测量。下一步优先补可解释的生产路径诊断，
再决定新的代码方向，而不是继续扫宽N或前段核数。

已阅读本地`pypto-lib/.claude/skills/incore-profiling/SKILL.md`及其canonical
`docs/debug-and-tune/incore-simulator-profiling.md`，并按技能执行只读函数发现。
构建路径须传绝对路径（相对路径会由脚本按pypto-lib根解析）；正确发现唯一
`dynamic_csa_full`及其配对.cpp/.pto。尚未运行仿真或生成任何trace。

完整CSA的72角色常驻协作不能直接装进该技能的单核合成case。后续若采用
仿真，应单独建立明确标识的Score生产路径诊断，复用原数学和页访问，手工
核对group/leaf/Q/L/页表、系数和Scalar维度，让CUBE/MTE2/FIX均有真实工作。
可评估独立生产者不受FIFO反压时的成本；这种隔离案例不用于解释完整72角色
调度，更不能进入九档性能汇总。不得将原完整kernel预填就绪标志后冒充真实
全核执行，也不得改变正式源、基线或用合成输入的周期替代实机L1。

只读确认正式CANN 9.2.0-beta.2目录已有`tools/msopprof/bin/msopprof`与
`tools/profiler/bin/msprof`；没有使用--msopprof/--auto-msopprof，没有复制worker
或注入库、修改共享CANN/SDK。具体仿真case、输入绑定和有效trace尚待建立。


### 2026-10-22：Score单生产者仿真已跑通，宽N负例得到独立指令证据

独立诊断只复制原Score helper并移除三个FIFO调用及空guard/drain，数学、
页读取、query尾行处理保持。入口一AIC、一G6、一逻辑叶、六个1024列step，
不伪造72角色协作，也不预填完整CSA就绪标志。12个buffer分别按真实维度
分配；从packed wrapper提取14个动态轴及block_idx/block_num，实际绑定0/1。
否则默认block_idx=1会得到空trace。Native页表为192..1倒序，page字节4160，
INT8输入/Key为1，系数仅本query的64个head为1；输出12288个FP32值全部应为8。
两个只读Key别名为合成独立buffer，逻辑字节一致，不声称真实cache别名性能。

初版pl.cluster(1)不合法，保留失败源后改为直接调用单incore入口。两个v2
完整CPU生成通过。首轮task_20261022_054417_411548524006和
054418_411706318223均exit1：技能旧兼容前导与CANN9.2内置FP8/FP4类型冲突。
随后CPU干跑另发现MrgSortExecutedNumList重复定义；仅生成case前导移除已由
当前CANN/PTO-ISA提供的声明。原生成kernel正文、SDK、CANN及技能文件均未改。
v2/v3中间失败目录保留；v4先完成CCEC编译/链接再采集，没有重复正式L1任务。

最终task_20261022_055755_4107924772（v122）与
055756_4117009850（N256）均completed/exit0；均实际执行、输出全部校验通过，
各导出一个Cube核trace/指令CSV。共用已验证工具链，仿真模型Ascend910_9391、
dav-c220-cube；模型不等于对实机具体SoC身份的证明。两项都经task-submit auto，
未复制worker或安装依赖。五项CPU合同通过，相关六个脚本Ruff通过。

轻量结果及输入/代码/trace索引在score_producer_sim_results.json，复现提取器
summarize_score_simulator.py。原始大产物位于runs/score_sim_{v122,n256}_v4，
clean/同时保留raw二进制、clean trace和instr_metrics.json。cleaner分别跳过
10/8个无法重锚定的同步箭头，因此事件细节以raw及生成代码为准，不认为clean
图包含所有依赖。原始trace的时间数值为微秒，displayTimeUnit=ns只是显示设置；
MMAD时间与指令CSV独立核对为1850 cycles/us。非MMAD的CSV周期与trace队列
生命周期不能直接互换，未把WAIT/SET累计周期当作不重叠阻塞时间。

| 单生产者、同一合成输入 | v122 N64 | M64/N256 |
|---|---:|---:|
| 完整X事件首尾跨度（仿真μs） | 46.7697 | 67.0222 |
| INT8 QK MMA次数 | 192 | 144 |
| FP16 WS MMA次数 | 192 | 144 |
| MMAD可见区间并集（μs） | 28.9065 | 28.1773 |
| MTE2搬入字节 | 847872 | 847872 |
| MTE2指令数 | 194 | 194 |
| MMAD之间最大空隙（μs） | 0.4114 | 1.4741 |

此表是无FIFO反压的单生产者诊断，不能进入正式九档、不能外推+43%的硬件
回退；正式B16已测+6.3496%的结论不变。指令区间并集也不是物理引擎利用率。
但它确实否定“少25% MMA即可大幅加速”的假设：Cube区间总量近似不变，
大面板串行依赖/间隙明显增加，GM搬入总字节未减少。停止宽N参数扩扫。

下一步回到N64供数和FIX写回组织。只读核对另一会话日志19849及20571等段：
保留双48KiB QK早FIX的WS八列池方案，TSTORE16→2，旧轮B16/B28省5.42/12.12μs；
换公共环境后B16仅省5.14μs，B28中位略慢。它是可借鉴机制而非本会话已取得
收益，不能整目录覆盖TND或据别的会话采用直接宣称通过。先局部移植、审计
Acc地址和最后消费者，再诊断并仅以同轮完整L1决定是否保留。明确保留旧负例，
不扫池宽、不复跑已被否定的联合短尾。生产默认未改、未push，goal仍active。


#### N64 WS列池：局部迁移、真实资源/复用事件与独立仿真正向

prepare_score_ws_pool.py只变换v122的_indexer_score_cube_split_store，保留
原policy、TND geometry、双M192、页保护和FIFO。两段K192按原顺序累计到
八个16×64列窗，满16×512才写原6×512输出，最后READY仍在完整step后。
没有整包复制另一会话的BSH，也没有叠加live_half/其它候选。三个CPU合同
核对原数学/FIX、page/FIFO和真实源偏移产生的完整1024列覆盖，全部通过。

完整B16与单生产者CPU均通过。score_ws_pool_resources.json审计两份最终
AllocateMemoryAddr/C++：QK0 0..49152、QK1 49152..98304、WS98304..131072，
16列窗按8个4096字节地址两遍复用；无WS Acc复制、每step32次QK/32次FIX、
16次WS init+16次续加，只有2次TSTORE。中间pool覆盖有FIX→M事件；最后
store到下一step的复用通过同PIPE_FIX队列、下一QK0 FIX→M→QK panel1→WS0
传递依赖保护，不能因没有紧贴末store的冗余直接flag就误判缺失同步。
此审计是源码/地址/事件机制证明，不是硬件重叠或性能测量。

单生产者task_20261022_060953_246241820844 completed/exit0，输出12288值
全部等于8。score_ws_pool_sim_results.json复用上轮原始N64仿真作同模型
诊断：跨度46.7697→43.5227μs；QK/WS MMA仍各192次，MTE2仍847872字节，
实际GM FIX由96→12次、L1 FIX192次不变。没有把约6.94%的合成生产端改善
称为完整层收益，也没有重跑已完成的reference采集。

完整代表B16/S6任务task_20261022_061056_25055969684已提交，严格冻结v122
同进程五配对窗口；当前仍在运行，不能使用初始化result里的false当终态
失败，也不能预记通过。公共SDK、计时口径、Native差异与九档目标均未变。

并行只读核对Query访存，发现旧“连续head分工”只改worker区间、不改八token
跨行tile，本轮拟评估真正八个相邻head组成Vec行。增加weight-scale重复读取
是明确代价；原512列规约、两次FP32乘法及rsqrt保留。T不整除8时完整回退
原helper，保留tail recip(sqrt)。前驱等全部Q_B、后继Cube等全部48个QUERY
写者，所以重新分配head不需要更改生产者信号合同。三个源级所有权/数学/
回退检查通过。首版统一pl.reshape在incore被解析为Tile，CPU类型检查拒绝；
v2明确使用pl.tensor.reshape构造三个GM别名，正在CPU编译，尚未上卡。
未改共享依赖、生产默认或push；目标继续active。


#### WS列池完整L1不受益，停止扩档；Query GM视图接口修正

上述061056任务已completed/exit0。严格v122输出/Top-K/score、cache/state、
保护、metadata和图检查通过；原Native precision_exact=false及5850项原有
输出差异保持独立，不改成新精度PASS。五窗789.00→789.70μs，+0.08872%，
仅1/5快，score_ws_pool_results.json保存全部配对窗口和工具链。暂不采用、
不扩B28或扫池宽。独立生产者更快而整链未受益，只能说明此前合成改善没有
转为当前系统级增益，不能直接据此量化AIV反压或认定某一个物理引擎瓶颈。

Query v2的pl.tensor.reshape仍在ConvertTensorToTileOps中变成Tile，完整CPU
同处拒绝。只读检查公共编译器发现tensor.view明确保留GM Tensor，而reshape
是转换运算；v3改三个别名为pl.tensor.view，未改共享依赖，正重新编译。
v1/v2失败源码、描述和日志保留；没有把类型失败当设备性能或组件bug。
目标仍未达10%，生产默认不变，所有改动只做本地阶段commit，未push。


#### Query连续head v4完整编译通过，提交代表档严格对照

v3已经保留三个GM Tensor view，但显式full([8,1], FP32)的行字节4不符合
Vec的32字节对齐而被PTOAS拒绝。v4去掉这块新声明：激活scale使用同轮GM
的device FP32 Scalar read，再做MULS；随后权重scale仍为独立FP32 MUL。
这是设备读数，无Host回读或新预处理。数学乘法次序未合并，但MULS与旧
row_expand_mul逐bit行为仍必须以设备对照确认，CPU数学模型不替代精度。

三项测试改为检查实际v4源码，枚举T0..384的完整行容量、24/32/48 worker，
每个(token,head)唯一覆盖且GM地址与原[T,64*512]一致。原尾行helper整体AST
保持；RMS平方/512列规约/rsqrt、NOPE、RoPE及BF16 rint运算保持。只允许
两次乘法广播方式改变；前后信号合同未改。三个检查与Ruff均通过。

v4完整CPU成功，query_contiguous_heads_v4_resources.json确认：三个view与
原q_proj/q_flat/wq_b_scale根及偏移相同，stride=[512,1]；compiled_meta与
相同描述符、原Query代码编译结果完全相等，没有新增GM根/参数/返回；
Vec地址最大163872B，与原Query相同。实际生成8×512连续读取、FP32 Scalar
read和MULS后MUL。v1..v4全部CPU结果归档compile_history，失败目录不覆盖。
旧连续DQ worker-range候选没有改变8-token跨行tile，本候选改变Vec行的
含义，因此不是原样重跑旧分工负例；增加scale/rope准备读取的代价仍保留。

任务task_20261022_062222_1899330389已提交，128K/B16/S6、冻结v122同进程
五配对窗口，等待严格精度/状态/图/完整L1终态。尚未预记PASS或收益，未扩
其它档。此前WS池完整负例已收取，当前只有这一项待收设备任务，无CPU悬挂。
本轮生产默认保持不变、未push，九档综合10%仍未达成，goal继续active。


#### Query连续head：B16完整对照正向，扩固定混长和短B32

062222任务completed/exit0，严格冻结v122输出/Top-K/score、cache/state、
保护、metadata、padding和图检查通过。B16/S6五窗中位796.44→785.62μs，
降低1.35855%，5/5配对更快；全部窗口在query_contiguous_heads_v4_results.json。
Native旧precision_exact=false仍另列，不将旧差异改成通过。不与main_vector_late
或其它独立候选的百分比相加，生产默认未采用，也不能据一档宣称九档10%。

query_head_read_model.py从原函数实际device head分组条件提取模型：在T96，
权重scale逻辑读取1572864→12582912字节（8倍），scale/RoPE准备296448→592896
字节（2倍），Q投影输入、Q输出、8×512 RMS块数相同。T56/T168的准备数据
反而减半，T192准备变4倍；因此不能把B16结论直接扩到所有档。该模型不含
缓存、对齐、异步重叠、设备指令成本，不能把逻辑读取当实际HBM流量或耗时。

已顺序提交两个同源码、同冻结v122、同公共环境的独立配对任务：
- task_20261022_063040_245239019924：固定128K/B16混长T56，原九档的第9档。
- task_20261022_063041_24714692920：8K/B32/S6，检查较大的scale/准备重读代价。

两任务当前running，无CPU任务；使用原句柄继续收取，不因观察等待重提。
目前新增阶段提交af32fd56、0c52ab13，后续结果继续分阶段提交，未push。
完整九档G尚未形成，goal保持active，后续还需受影响边界、实际服务和组合验收。


#### Query扩展两档均已收取：暂只保留B16方向，准备独立分档

两项任务均completed/exit0，严格冻结v122精度、Top-K/score、cache/state、
保护及图验证通过。保持独立轮次中位数，完整五窗见对应结果JSON：

| v4场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对更快 |
|---|---:|---:|---:|---:|
| 128K/B16/S6（上一轮） | 796.44 | 785.62 | -1.35855% | 5/5 |
| 固定128K/B16混长T56 | 752.40 | 754.70 | +0.30569% | 2/5 |
| 8K/B32/S6 | 926.10 | 930.56 | +0.48159% | 0/5 |

结果文件分别为query_contiguous_heads_v4_results.json、
query_contiguous_heads_v4_mixed_results.json、
query_contiguous_heads_v4_short_results.json。混长方向混合且幅度小，不能
把0.31%认定为稳定损失；但它没有足够正向采用证据。短B32五窗全慢，不
全域开启。三档数据不构成九档G，也不与其它优化不同轮次相加。

下一步优先构建独立constexpr分档：沿已有长G6、64<=token容量<=96资源域，
保留实际device T不整除8时完整旧helper回退。64处Q_B wide-N及前置几何
策略改变，96是既有LocalStats/O_A资源边界；不是按B16名称写白名单。
这个区间只是有依据的待验证范围，尚未证明T64/T72/T80等都有性能收益。
应显式加QUERY_CONTIGUOUS_HEADS，不能隐式捆绑现有Attention开关；源码
启用端与已测v4、关闭端与v122的生成代码分别核对，再补T64/T80与T65尾行
等受影响边界。没有改生产默认，没有提前写采用PASS。

另外只读复审Score消费链：当前B16关闭scale_page_coalescing，逐step逐页
读取FP16 scale；B28原本开启。另会话已有B16 scale/WS组合小幅正向，不能
直接迁移其百分比。若进一步并行化页布局证明，需同时处理group_offsets
尾区的单缓存行写者、各轮覆盖、IDX_SCALE发布/获取、动态页与碎片回退；
现有2字/request不能分给多个Scalar写者而忽略64B行共享。本轮没有生成
这个候选、修改工作区或启动新任务，只把它列为需完整设计的后续方向。

全部当前NPU/仿真/CPU任务已终态，current_state没有pending。本轮实际完成
单生产者诊断、WS整链负例、Query三档正负结果与源码/资源审计，有实质进展，
不存在重复外部阻塞；目标继续active，生产默认未变，未push。


### 2026-10-22：Query独立分档已生成，启用/关闭完整代码与实测产物一致

上一goal轮完成四个阶段commit、单生产者证据、WS完整负例及Query三档正负
实测，属于实质进展。本轮再次核对goal active、Git干净、共享manifest仍为
validated的同一版本，并用task-submit确认上轮两个最后任务均completed/exit0。
继续九档等权G<=0.90目标，不缩为B16单点，不重新测已完成的三个v4试验。

prepare_query_head_guard.py基于实测v4增加独立QUERY_CONTIGUOUS_HEADS，显式
经过policy→decode→core_group→vector_schedule→Query helper。区间为既有
long G6且64<=token容量<=96；实际T不整除8时仍执行整个原helper。未用
LOCAL_ATTENTION_STATS等其它开关隐式控制新选择，修改它们不会改变该flag。
没有新增环境变量、GM buffer、host metadata或公共SDK修改。暂为候选，未采用。

四项CPU合同通过：枚举历史阈值、query上界、请求/容量和63/64/65/95/96/97
等边界，旧policy各项不变；flag独立；实际转发完整；打开的数学与v4相同、
关闭调用整个原helper，其它调度AST不变。相关三个脚本Ruff通过，格式修复
后额外核对生成policy字节未变化，没有因代码格式重编已通过的候选。

完整B16启用与短B32关闭CPU编译均成功。query_head_guard_codegen_audit.json
直接绑定上轮实际设备产物，先核对实测spmd/original_spmd全部源与相应清单，
再比较guard的C++、AIC/AIV packed wrappers、orchestration和compiled_meta。
两侧五份产物均一致：C++只删除// pto:位置注释，运算、常数、地址、分支、
同步保持；host/ABI metadata逐字相等。未把它扩大为ELF相同，未以此直接
标记最终九档性能通过，最终组合仍须完整验收。

本轮重读已定位cannbot ops的attention_schedule_slots_and_platform_handoffs
（draft、未定位platform）：采用其中区分运行期未走分支与编译期实际消除的
核查思路，并以本次生成代码作证；不照搬平台事件编号、mode4或PIPE_S示意。
知识索引缺失的原限制保留，没有重建索引或更换共享工具链。

已提交并确认running：
- task_20261022_064744_56493329836：B16实际S4、query上界6，T64，五窗完整L1。
- task_20261022_064745_5670722673：B16实际S5、query上界6，T80，五窗完整L1。
- task_20261022_064818_59201812816：T72/R17 metadata矩阵，覆盖启用路径。
- task_20261022_064819_5935356678：T65/R16 metadata矩阵，覆盖token尾行回退。

最初尝试T80 metadata矩阵在提交前被原harness合同拒绝：该矩阵只支持
41<=Tcap<=72，且request容量须能接收S1时全部dummy尾部。没有产生任务ID、
没有占卡、没有放宽harness。改用T72/R17（6+ceil((72-6)/6)）覆盖开启路径；
T65/R16合法且保留。T80仍由前述独立完整L1实际S5用例验证。此处只登记
真实提交和已完成CPU证据，尚未预记任何新设备PASS或性能收益。

当前无CPU任务，四项设备任务使用原句柄收取。生产默认未变、未push，
完整九档10%尚未达成，goal继续active。


#### 边界终态与实际分档纠正：S4/S5是关闭对照，不能归因新算法

064744、064745、064818、064819四项均completed/exit0、严格功能/状态检查
通过。T72/R17与T65/R16两个metadata矩阵各8项（S1..6、261、123456）和4步
连续轨迹全部通过，各12次同址图重放。连续轨迹保留前轮状态，包含请求减少、
重排和新增；历史偏移覆盖C4余数。摘要query_head_guard_metadata_results.json，
原metadata_matrix.json保留逐项输出/Top-K/cache/state、写界及同址证据。
T72实际flag=True且容量整除8，覆盖新Query；T65实际flag=True但容量尾行，
覆盖整个原Query回退。没有将这些描述为新增B1服务或完整九档通过。

收取两个L1边界时，原collect_trials正确拒绝S4/S5不是固定九档。未修改或
放宽evaluate_nine_cases。另建collect_query_guard_boundaries.py，显式仅收
四种已定义边界输入，沿用完整72核/单MIX/native150/五交替配对/隔离工作区
验证，强制记录实际QUERY_GROUP和QUERY_CONTIGUOUS_HEADS，并核对实际生成
PTO是否存在head GM-view路径。八项负向/合同测试通过，包括确认原九档gate
仍拒绝边界用例。所有边界报告formal_gate_eligible=False。

更重要的实际分档发现：测试器在显式query-lengths下将query_tokens更新为
真实最大值。原T64全S4记录query_tokens=4/G2/flag=False，T80全S5为5/G2/False；
之前登记“上界6”只是CLI意图，不是实际执行条件，现明确纠正。两个实际候选
与同轮v122的完整C++（仅去pto位置注释）、AIC/AIV wrappers、orchestration、
compiled_meta全部一致，见query_head_guard_disabled_boundary_results.json。
其818.94→829.10μs（+1.24063%、0/5）和978.06→979.52μs（+0.14928%、1/5）
完整保留，但不能归因新Query，也不能作为真正启用区间的性能证据。

没有重测已完成S4/S5，而是采用真实最大S=6且同总T的新输入，保持原harness：
- task_20261022_065845_127797123832：T64，[6,2]+[4]*14。
- task_20261022_065846_127906019085：T80，[6,4]+[5]*14。

两者history131072、B16/Rcap16，无dummy、同卡同进程冻结v122五窗；收取器
将要求实际G6且flag=True，否则拒绝当作启用证据。两项尚待终态，不预记收益。
正式八档S6与固定第9档混长完全不变。当前只剩这两个设备任务，无CPU任务，
生产默认未改、未push，goal继续active。


#### 真正G6启用边界回退；组合v3仅编译，转为同址归因

065845、065846均completed/exit0，严格输出/Top-K/score、状态、保护及图通过。
新边界收取器同时验证实际QUERY_GROUP=6、QUERY_CONTIGUOUS_HEADS=True以及
生成PTO含head view，两个输入没有再落入关闭路径。完整五窗结果：

| 场景 | v122中位μs | guard中位μs | 耗时变化 | 更快窗 |
|---|---:|---:|---:|---:|
| T64，[6,2]+[4]*14 | 723.56 | 733.00 | +1.30466% | 0/5 |
| T80，[6,4]+[5]*14 | 780.70 | 798.96 | +2.33893% | 0/5 |

证据query_head_guard_active_boundary_results.json。拒绝把64..96整个资源区间
作为采用策略；不靠只挑T96或换成B16白名单掩盖两个负例。原T96正向证据保留，
但尚不能证明普遍/稳定收益，尤其代码相同的关闭对照也出现约1%的耗时差。

在边界运行期间已准备combined_v3：基于最充分验证的combined_v1
（B1空发布、live-half、batch-sort3），加入main_vector_late和Query guard。
INPUT/QR/ROPE裁剪与Main后移变换逐字可交换，Score原组合源完整复制，独立
Query flag仍显式传递；未加入无整链收益的WS池或combined_v2方向混合的填充项。
B1/B16完整CPU均通过，源码/描述符/结果已保存。但边界结果否决了包含的
Query区间，因此组合v3不提交设备、不采用、不作任何性能结论。

为区分T96的实际代码收益与工作区地址差异，提交
`task_20261022_070839_165489119202`。使用已有--l1-shared-workspace归因模式，
两版串行交替共享全部GM buffer（含signal），相同初态恢复仍在计时外，
五窗完整72核L1及代际/精确验证保留。候选用v4（与guard启用生成代码已证
一致），因为其policy字典与v122完全一致，满足原工具同资源策略约束；
没有修改或放宽harness。该诊断shared_workspace=True，原九档gate将拒绝，
只解释先前小幅正向，不能替代最终独立工作区验收。当前该任务running。

另做只读排重：已有实现日志1684后的B1四→两Hadamard核/Q_B二十二核及
随后公共工具链复核记录，不能把同一配置重命名为新优化。本轮没有重跑。

Score连续scale后续设计发现一个必要约束：binding允许自定义allocator，
只显式检查signal的512B对齐；group_offsets并无64B首地址合同。因此不能
只按Tensor相对16字分包就保证物理缓存行单写者，也不能直接让每请求写者
并行更新当前2字记录。安全设计需实际物理对齐处理/回退，或把记录间隔扩到
能覆盖任意INT32起始偏移的两个缓存行，并同步调整binding、消费者及独立
harness的布局/保留区验证。原META完成之后的IDX_QUANT/IDX_SCALE获取链和
最后消费者也必须明确核对。此方向本轮只完成约束分析，没有生成候选或
修改任何共享SDK/生产布局。不可将这个设计记录当作已获性能收益。

本轮仍为有证据的实质进展，无外部阻塞。当前只有同址归因一个任务待收，
无CPU任务；生产默认不变、未push，九档G仍未形成，goal继续active。


#### 同址Query诊断与第二个完整S6边界；容量细化只保留为候选

070839同址任务completed/exit0，严格输出/Top-K/score、状态、保护、图和代际
检查通过。790.76→782.88μs，-0.99651%，4/5快。收取器明确验证原九档gate
因shared_workspace=True拒绝；此结果只支持归因，不能替代正式独立工作区
验收。原harness把baseline.buffers设为candidate.buffers，binding每次调用
从该字典传参，未另改计时/恢复位置。结果与完整窗口见
query_contiguous_heads_v4_shared_diagnostic_results.json，旧Native差异另列。

又测非B16的完整S6容量边界：task_20261022_072146_218734617423，B12/S6/T72，
实际G6/flag=True，completed/exit0，全部严格检查通过。706.90→706.22μs，
-0.09619%，3/5快，仅接近零的小幅证据，不据此宣称稳定改善。边界专用收取器
增加这一明确非cohort输入，原九档case集合完全不动；八项拒绝错误分档、
时钟/核数/次序/共享区和phase-clock的测试仍通过。

准备过query_head_capacity_guard，仅在原64..96长G6资源域再要求
T容量=request容量×query上界。只改policy，不从该乘积推断实际Q/L或请求归属，
动态计算与尾行回退完整保留。两个CPU策略/源码合同通过，T80关闭完整编译
及实际v122生成C++/ABI/host逐字核对通过。B16/T96和B12/T72的kernel文件及
所有constexpr值与已测guard一致；没有再重复编译这两个相同输入。该细化
没有采用到生产，B12证据很小，转而优先处理新发现的实际重复读取。

#### 新机制：同一AIV的8-head权重scale只载入一次

连续head映射的work=worker+k*WORKERS。H=64、每work八head，当WORKERS为8的
倍数时，work%8恒等于worker%8；当前48个AIV每轮都读同一组8-head权重scale。
v4却把16KiB重新载入每次迭代。query_resident_weights独立基于v4，只改Query
helper：一次读入readonly Vec tile，循环内复用；WORKERS不是8倍数时保持原
逐work读取，T不整除8时仍走整个原helper。没有改scale乘法顺序、RMS、RoPE、
BF16边界、同步或Tensor ABI。

两个CPU合同通过：实际源表达式枚举多种worker数与T，确认可驻留/必须回退的
所有权差异；还原这一个load移动后，整个旧数学body的AST完全一致。B16
完整CPU编译成功，packed_meta与v4相同。逻辑权重读取量T96由v4的12MiB变为
768KiB，原v122为1.5MiB；这是逻辑元素量，不是HBM流量或耗时预测。

按cannbot resident_softmax_state_slot_recurrence_and_reuse_events的生命周期
核查思路审计最终生成物（该卡仍draft/未定位platform，不搬用硬件结论）：
- resident_head_scale物理UB[65536,81920)，16KiB，整个Query循环前只有一次TLOAD。
- 重复主循环中该物理区间只有两个只读别名，既无输出覆盖也无scratch/tmp占用。
  加强检查按实际UB偏移判断重叠，不能把不同MemRef名当成独立UB地址空间。
- 奇数最后单work尾轮可以在最后一次MUL读取时原地覆盖该区间；它在主循环后，
  之后没有任何原scale消费者。没有把合法的最后一次复用误判成循环内破坏。
- 原行索引准备的V消费者先通过V→MTE2的配对事件退休，之后才加载权重；
  生成的Query输入TLOAD全部晚于权重TLOAD，原MTE2→V获取覆盖前序MTE2工作。
- 整体Vec地址上界仍163872B，没有新增GM根/参数/返回。

审计一度因PTOAS合并了展开的_1/_2名字提示而停止；改为核对真实C++分支中的
INT32 8×512输入Tile/TLOAD，未删除控制或事件检查。逐步审计报告保留，最终
入口query_resident_weights_resources_v4.json；这仍只是静态机制证据，真机
正确性与完整耗时由下列任务另证。Ruff及两个新合同检查通过。

#### 驻留版三档终态：B16正向，混长仍回退，不全域采用

三个任务都completed/exit0，严格冻结v122输出/Top-K/score、所有cache/state、
保护、图及代际检查通过，Native原有差异不改标：

| 任务 / 场景 | v122中位μs | 驻留版中位μs | 耗时变化 | 配对更快 |
|---|---:|---:|---:|---:|
| 075108_326400111708 / 128K B16 S6 | 799.88 | 789.18 | -1.33770% | 5/5 |
| 075109_32650589990 / 固定混长T56 | 735.92 | 747.40 | +1.55995% | 2/5 |
| 075110_326637827 / 8K B32 S6 | 930.10 | 928.36 | -0.18708% | 2/5 |

完整数据query_resident_weights_initial_results.json及
query_resident_weights_mixed_results.json。不能跨轮把驻留版与v4绝对值相减，
也不能将少读权重等同于全域收益。只保留B16方向，混长不采用；短B32幅度很小
且窗口混合，不给强正向结论。此前combined_v3继续隔离，不上卡或采用。

#### 下一实际省工点及平台边界复审

继续追踪发现，连续head helper每work还重新读取/展开rope_swap、加行偏移并
转换为q_abs_swap。该swap是内部q_rope_prepare_stage写出的固定列配对，所有行
相同，不同于随位置变化的外部cos/sin。vector_schedule已在609行附近获取全部
PHASE_ROPE写者，之后才进入Indexer/Main Query。下一候选可在证明生产者行
一致后只准备一次readonly q_abs_swap；仍每query读cos/sin，保留每次gather的
可写scratch，不根据位置相等或T/B推断外部sin/cos数值。此处尚未生成或上卡，
不预记收益，也不更改其余Goal约束。

只读排除了一个较大但无新传输机制的改写：固定PyPTO的
`docs/en/dev/passes/24-expand_mixed_kernel.md`与TPUSH/TPOP契约明确a2a3经GM，
固定PTO-ISA `include/pto/npu/a2a3/TPush.hpp`的producer为TSTORE到GM、consumer
为TLOAD。不能套用a5的消费者SRAM路径，也不能声称把当前手写Group换成
pl.spmd/split_aiv/aic_gather就自动省去GM。本轮没有为该假设改写Attention或
更改共享SDK；这条结论只针对当前固定实现，不扩为所有硬件能力断言。

当前所有任务已终态，无CPU/设备pending。生产默认未变，仍未形成最终九档
G<=0.90及完整新服务验收；goal继续active，未push。


#### 内部RoPE索引常驻：生产者证明与完整生成物审计通过，三档已提交

目标继续为九档等权G<=0.90：八档S6和固定128K/B16混长，分母仍是冻结v122。
新query_resident_swap基于query_resident_weights，只将q_swap_one、
q_swap_repeated和q_abs_swap移到Query工作循环前；读取内部rope_swap的第0行。
新增tokens>0保护，空输入不读不存在的第0行。逐query的cos/sin、逐work的
可写gather scratch、完整原tail helper及所有同步/ABI均保持。

三个CPU合同通过：执行实际q_rope_prepare_stage的数学，覆盖T0/1/6/7/8/9、
T24/48/56/65/72/96/144/168/192/193和1/7/16/48个worker，随机sin不同，
所有有效行仍为(1,0,3,2,...)；恢复三条索引准备语句后整个旧Query函数AST相同；
核对原PHASE_ROPE生产、发布、获取先于消费者，等待范围覆盖row0生产者。
producer用min(workers,token_tiles)分配，row0始终由worker0写；原获取还包含
业务cacheinvalid/fence和本轮generation，没有增加新通知协议。

B16完整CPU编译通过；参考cannbot已读的resident状态复用卡，按真实物理UB
地址核查，不把MemRef名当独立bank。新索引为[51200,53248)，两路展开和最后
单work均无覆盖，包括scratch和别名分配；C++只有一次索引TLOAD，三个TGATHER
均读同一驻留tile。MTE2→V事件成对且在转换前获取。固定a2a3 TGather.hpp的
FP32分支将byte-offset暂存和gather结果写到dst，不修改src1索引。权重驻留
独立复审也通过；整体Vec上界仍163872B，packed_meta与父版完全相同。

复现入口prepare_query_resident_swap.py，CPU合同test_query_resident_swap.py，
审计audit_query_resident_swap.py及原audit_query_resident_weights.py。报告
query_resident_swap_resources.json和query_resident_swap_weight_resources.json。
这些是源码/生成物证据，不是设备正确性或性能通过。Ruff检查通过。

完整72核L1、同轮冻结v122、五个交替窗口任务：
- B16/S6：task_20261022_083320_6419639195
- 固定混长T56：task_20261022_083333_65038322156
- 8K/B32/S6：task_20261022_083346_6590615021

均由冻结harness的submit_probe.py经task-submit自动分卡，不改公共SDK。
当前待终态；无生产采用，不相加父版或独立候选收益。


#### RoPE索引驻留三档收齐：全部严格正确，收益有限，不全域采用

三个任务均completed/exit0，严格冻结v122输出/Top-K/score、cache/state、
保护区、图及代际通过；Native原有差异继续单列。完整L1五个交替窗口：

| 场景 | 耗时变化 | 配对更快 |
|---|---:|---:|
| 128K/B16/S6 | -0.36862% | 3/5 |
| 固定混长T56 | +0.14270% | 2/5 |
| 8K/B32/S6 | -0.57553% | 4/5 |

原始窗口、两侧中位数可由query_resident_swap_b16_results.json及
query_resident_swap_mixed_short_results.json重算。减少逻辑load和索引算术
未给出全域稳定收益，不将跨轮绝对值相减来推断比权重驻留更好或更坏；
短B32保留小幅局部证据，不当成正式采用或九档目标完成。

复核既有O_A生成物：已有两份K256 L1以及双K128 Left/Right，不把简单
新增双槽当作新机制；历史B1显式K512/M16供数已是完整负例。本轮不重试。
新发现输出量化的oc_amax/oc_q是相同GM地址的两次真实TLOAD，源和最终IR
均存在。准备独立基于v122的output_quant_reuse，仅用oc_q=oc_amax延长原值
生存期；所有amax/div/scale乘法/三次cast、valid行、尾零、分工AST保持。
CPU合同和完整B16编译已通过；仍需检查abs/scratch不可覆盖原值到最后的
row_expand_mul，再做完整L1实测，不预记收益。


#### O_A量化原值复用：生成物安全，开始完整L1与B1尾行矩阵

output_quant_reuse完整B16编译和audit_output_quant_reuse.py通过。输入物理
UB在两个展开lane分别为[65536,98304)、[98304,131072)，abs与row_max scratch
均另有存储；到最后一次row_expand_mul前，重叠地址只有oc_q的只读别名。
最终乘法后原值再无消费者，允许结果覆盖原tile。尾轮也逐项同样核查。
展开后原输入load静态位点6→3，整体Vec地址上界163872→155872B，packed ABI
不变。C++确认实际TLOAD、TABS与TROWEXPANDMUL，别名TASSIGN同址。
审计首次假设别名会使用同一个C++变量名而失败；修为核验别名的TASSIGN地址，
保留物理区间覆盖及生命周期检查，未放宽任何数值/保护区条件。

一个CPU源码合同和Ruff通过；本轮只替换第二次相同load，不改运算顺序或
尾部语义，不改变custom allocator合同。B16逻辑输入字节6MiB→3MiB，不声称
这是实际HBM流量或完整收益。此源独立基于v122，不叠加Query或其他优化。

三项冻结harness/公共SDK完整L1五窗任务：
- B16/S6 task_20261022_084339_11452478515
- 8K/B32/S6 task_20261022_084352_115624411592
- B1/S6及同址S1～S6 task_20261022_084401_116502025524；Tcap6/Rcap1，
  table-history131200满足原矩阵动态边界合同，固定S6计时先于metadata回归。

任务尚待终态。生产默认不变、没有push，目标继续active。


#### 量化复用完整结果：B1降低5.06%，两个大档回退，建立独立B1分档

三个任务均completed/exit0，严格v122输出/Top-K/score、状态/保护/图/代际通过：

| 场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对更快 |
|---|---:|---:|---:|---:|
| 128K/B16/S6 | 784.54 | 788.54 | +0.50985% | 0/5 |
| 8K/B32/S6 | 933.44 | 937.86 | +0.47352% | 2/5 |
| 128K/B1/S6 | 384.72 | 365.24 | -5.06342% | 5/5 |

完整样本output_quant_reuse_results.json。B1计时之后同址S1～S6六例也全过，
轻量摘要output_quant_reuse_b1_metadata.json保留原件路径/hash。此矩阵没有
多请求连续步骤，不把六例扩大为新连续状态或服务验证。原Native差异仍单列。
少一次TLOAD和更小UB并未让大batch受益，不能据逻辑字节数预判全域收益。

按用户允许B1单独分档，output_quant_guard新增独立constexpr
OUTPUT_QUANT_REUSE：requests==1且token容量<=8；既有decode query上界最多6。
其余容量恢复原双load。选择只依赖已有Host容量，不用它推断真实Q/L或请求
归属，不引入环境变量、D2H或通知。默认False，通过entry→vector→两种量化
入口完整传递；未捆绑Top-K或Attention开关。

6项策略/数学/参数传递CPU合同通过，B1启用与B16关闭完整CPU编译通过。
audit_output_quant_guard.py对实际已测产物逐项比较：启用与B1复用版、关闭与
B16的v122，完整C++仅去整行pto位置注释后相同；AIC/AIV packed封装、Host
orchestration和compiled_meta逐字一致。证据output_quant_guard_codegen_audit.json。
不把源码开关关闭自动当作等价证明，也不声称ELF完全同一或九档性能已通过。
不为已证明同一代码再重复B1同样的设备窗口。

新组合combined_v4仅包含最充分验证的combined_v1（B1空发布、混长live-half、
三query排序合批）、main_vector_late、B1量化复用分档。先证明空发布修改与
主AIV后移的转换可交换，再传量化flag；Score文件取原v1，其他量化/policy/
entry取已核对guard。所有Query head改写和被隔离的combined_v3均排除，
未加入已有负例、无用fill或WS池。源manifest和prepare_combined_v4.py可重建。
B1/B16完整CPU编译进行中；组合收益必须重新实测，不相加上表与历史百分比。
生产默认仍不变，九档目标未达成，goal active，未push。


#### combined_v4静态检查完成，三项完整对照已提交

B1/B16均完整CPU编译通过；packed_meta与对应quant guard完全相同。真实
物理地址上界：Cube Mat475136、Left/Right各65536、Acc131072B；Vector
B1为155872、B16为180224B，均在各空间容量内。B1的量化源保持三个load
位点，B16关闭复用恢复六个。B1原值生命周期审计在组合生成物上再次通过，
不是直接复用独立候选的地址证明。报告combined_v4_compile_results.json与
combined_v4_b1_quant_resources.json，仍不当作设备性能或组合精度结论。

完整v122同轮72核L1、各五窗任务（源码同一combined_v4）：
- task_20261022_085830_177785012157：128K/B1/S6，计时后同址S1～S6。
- task_20261022_085837_178275330818：128K/B16/S6。
- task_20261022_085847_179054217504：固定第9档混长T56/R16。

全部使用原harness和公共SDK，经task-submit自动分卡；当前pending/running，
无CPU任务。保留原句柄等待终态，不因排队或观察超时重复提交。未改生产默认。


#### combined_v4三档完成：精确通过，但没有保住单项收益

三个任务均completed/exit0，全部严格v122输出/Top-K/score、状态/保护/图/
代际通过；B1计时后的同址S1～S6六例也全过，没有新增连续多请求步骤。

| 场景 | v122中位μs | 组合中位μs | 耗时变化 | 配对更快 |
|---|---:|---:|---:|---:|
| 128K/B1/S6 | 377.80 | 378.06 | +0.06882% | 4/5 |
| 128K/B16/S6 | 781.44 | 791.38 | +1.27201% | 0/5 |
| 固定混长T56 | 748.16 | 747.60 | -0.07485% | 3/5 |

全部样本combined_v4_b1_results.json、combined_v4_b16_mixed_results.json，
矩阵摘要combined_v4_b1_metadata.json。不能把4/5配对较快替代约定的中位数
比值，也不能把独立B1的5.06%带入此组合。v4不采用；原v1仍是最多验证的
组合。下一组合应先去除未经B1单独证明的Main后移等交互，并核对各分档
完整生成物，不假定数学独立就能保留时序收益。原Native差异未改标。

#### 页表证明单有效行：保持物理8行，对同一逻辑行只归约一次

独立page_proof_single_row从冻结v122建立，只在compressed_pages_all_valid
和indexer_scale_layout_plan_stage内把ones/repeated的valid rows从8改成1。
物理8×1024和8×1 tile保持；所有有效列、页表读取/边界、回退、输出及通知
不改。两者原本把同一行复制8次，最终只读row0，此候选没有改变请求布局。

4项CPU合同执行真实helper，覆盖0/1/7/32/33/1023/1024/1025/2049页、短表、
正逆序、碎片、负页/越界、大整数边界、空request、工作区保留区；Attention
另对独立范围oracle，布局阶段与原输出全数组相同。Ruff检查通过。固定ISA
a2a3 TROWMIN/TROWMAX明确把src.GetValidRow()传入TRowReduceInstr；不从通用
API名推断硬件循环，也未改共享SDK。

B28完整CPU编译通过，包含实际启用的两种证明。生成IR的输出物理8×1、
32B，valid_shape=[1,1]；输入物理8×1024、有效行1、原动态有效列。完整C++
确认对应TROWMIN/TROWMAX的Tile构造实际有效行1、只取第0个结果。审计首次
因set_validshape在PTOAS成为原Tile的metadata操作而找不到别名注释；修为
跟随同址且valid rows一致的metadata别名，未删物理布局/范围检查。报告
page_proof_single_row_resources.json；Vec上界163872B，packed ABI不变。

此次只减少广播/归约的有效重复行，full常量仍按物理布局初始化，不能声称
所有相关指令或HBM流量减少8倍，更不能预记完整L1收益。复现prepare、test、
audit_page_proof_single_row.py已落盘。两项完整五窗L1已提交：
- B28/S6 task_20261022_091214_268296623111
- B16/S6 task_20261022_091222_272500228073

仍与同轮冻结v122比较，不与v4组合混算。九档G未形成，goal active，无push。


#### 用户暂停：只收尾已经启动的尝试

用户明确暂停goal，并要求“把本次的性能优化尝试做完之后就停下”。核对工具
状态已为paused，不恢复总目标。本轮收尾仅包括已提交的page_proof_single_row
B28/B16两项设备任务、已启动的combined_v5 CPU编译及结果归档，不提交新任务。

暂停前已生成combined_v5：从output_quant_guard复制，仅换入combined_v1的
Score文件，保留G6半段裁剪/三query排序和B1量化复用；去除Main后移及空发布
裁剪。此举是隔离组合交互，不认定哪一项单独导致v4回退。两项CPU编译均完成。
收尾时比较完整生成C++（只去pto位置注释）、两个packed封装、Host及meta：
B1与已测量化复用版相同，B16与已测combined_v1相同，全部通过。
证据combined_v5_compile_results.json和combined_v5_codegen_audit.json。
combined_v5没有NPU性能验证、没有生产采用，也不是最终九档达标证据。

当前两个旧设备句柄仍live，沿原句柄收取，不因暂停或观察等待重启。完成后
本地commit并停止；未授权push，生产默认不变，目标未达到不标complete。


#### 本次尝试收尾完成，按用户要求停止

两个page_proof_single_row任务均completed/exit0。严格冻结v122输出/Top-K/
score、全部cache/state、保护、图及代际检查通过；Native原有差异保持单列。
五窗完整72核L1结果：

| 场景 | v122中位μs | 候选中位μs | 耗时变化 | 配对更快 |
|---|---:|---:|---:|---:|
| 128K/B28/S6 | 1159.82 | 1158.68 | -0.09829% | 3/5 |
| 128K/B16/S6 | 801.58 | 793.50 | -1.00801% | 5/5 |

结果page_proof_single_row_results.json保留全部窗口及源码/SDK证据。B16保留
局部正向候选；B28接近零，不宣称稳定提速。此轮没有额外提交动态页矩阵、
组合或完整九档，也没有将候选采用到生产。后续验证属于暂停后的未完成工作。

已启动的combined_v5 B1/B16 CPU编译及完整生成物等价检查也已归档；未启动
v5设备任务，不用CPU等价冒充它完成了九档验收。所有本会话设备任务现已终态，
无CPU或设备pending。Goal保持paused，G<=0.90仍未达成；生产默认未变。
按用户“本次尝试做完就停下”的指令结束本轮，不自动恢复或扩大优化。
本地阶段提交后停止，没有push。

<a id="csa-tnd-eight-case-10pct-20261022"></a>

## 2026-10-22：新八档核内与核间优化，固定现有已验证基线

用户清除了旧 Goal，建立新目标：参考 cannbot-knowledge 的 ops 做核内优化，
分析各 task 均衡性做核间调度优化，性能测试按 CSA_PERFORMANCE_TEST_SPEC.md，
在现有基础上提升 10%。随后明确继续优化 **TND-SPMD**，八档等权几何平均耗时
降低至少 10%；这是本轮验收定义，不是每档必须分别降低 10%。

```text
r_i = candidate_TND_SPMD_L1_median_i / frozen_combined_v1_L1_median_i
G = exp(sum(log(r_i), i=1..8) / 8)
目标：G <= 0.90
```

正式性能展示保持规范规定的八行、六种计时和 Native 相对 TND-SPMD 的百分比，
不增加几何平均行；G 作为优化目标的内部验收字段保存。各格至少 10 个有效窗口，
保留全部样本，混长两列 BSH 为 N/A。原九档/v122 分母和已归档结果不改写。
保持正式第 4 层 W8A8/NZ2、BF16、TP1、完整 HC_pre/RMSNorm/CSA/HC_post，
动态 Q/L、Native 分页和 cache/state 合同继续有效，原 Native 差异单列。

### 同步、来源和历史数据复用

本 worktree 从已发布的 `2b670ca8d` 快进到 `67134f28b`，取得另外会话刚完成的
`tests/pypto_test/csa_performance_20261022/` 八档六列数据，没有改写远端成果。
新目标没有授权继续 push；后续按阶段本地提交。公共 SDK 仍为已验证的
`csa-shared-20261021-upstream-fixes`，没有修改或重建它。

本轮实验目录为 `tests/pypto_test/csa_spmd_tnd_eight_20261022/`。
逐格审查现有结果后，选用已完成全部八档测量的 **combined_v1**，不使用更慢的
v122 或仅 CPU 通过的 combined_v5 作性能分母。`freeze_baseline.py` 完成：

- 核对全部六列已填格的样本数和中位数；只对明确不支持的混长 BSH 格接受 N/A。
- 从八份完整结果和 80 份原始 L1 重新计算 TND-SPMD 跨度，逐窗确认 72 核、
  一个 MIX task、incore 首尾和完整记录，并与全部保存样本逐项匹配。
- 验证实际设备测量冻结包的 41 个模块与 combined_v1 清单一致。
- 固定 `baseline.json`、完整 Git revision 和必要的六份 `.py.source` 覆盖文件；
  `materialize_baseline.py` 已从这些永久材料重建出逐字相同的基线。

八档已有 TND-SPMD 中位数依次为 355.01、971.61、617.39、773.83、1012.17、
1158.70、924.00、750.79 μs。原始未舍入值、全部样本和来源保存在 baseline.json。
本次未重测这些合格历史值；后续对照检查可以执行冻结基线获取正确性输出，
但不能用另一次更慢的基线测量替换本轮固定性能分母。

### 核间分析与核内候选的依据

`analyze_balance.py` 只读现有原始 L1，复用另一会话按同次编排提交序号命名 task
的解析器。`baseline_balance.json` 保留十个窗口的 MPMD 原始 L1 索引与核间统计，
以及 SPMD 全核尾部统计；逐核区间仍保存在对应原始记录。B1 的中间链 AIC
最大/平均耗时中位数约 1.89，混长约 1.08；
多数等长大档接近 1.01。它们包含等待和空角色，不是计算利用率，不能据此断言
已经算力均衡，也不能把 MPMD 的阶段耗时直接移植为 SPMD 内部阶段耗时。
继续结合真实 owner 公式、每核工作量及必要的阶段端点定位可改善的尾部。

知识查询 `discover` 确认 `artifacts/indexes/knowledge.sqlite3` 缺失，退出码 2。
依 `knowledge-query/SKILL.md` 停止索引检索，不扫描全库兜底、不修改知识库或
自行伪造 ready/outcome。只读已定位的 ops 卡片：

- `guides/synchronization/resident_softmax_state_slot_recurrence_and_reuse_events.md`：
  draft、未定位平台；作为保持最后消费者与复用顺序的工程提醒，不认证其示例。
- `guides/synchronization/attention_schedule_slots_and_platform_handoffs.md`：
  draft、未定位平台；保持参与者、事件数、空块和排空协议，不移植 950 专用路径。
- `guides/operator_practice/simd_operator_optimization/memory_access/l2_cache_mode_config.md`：
  draft，正文说明 A3 适用；区分带宽和缓存替换，不能无 PMU 证据认定缓存瓶颈。
- `apis/simd_api/basic_api/memory_vector_compute/sort_and_merge_isasi/sort32.md`：
  stable、明确 A3；保留局部同分顺序和既有归并树，不以近似 TopK 改写正确性边界。

以上路径均相对 `cannbot-knowledge/knowledge/ops/ascendc/`。同时读取当前公共
PyPTO 的 mgather、reinterpret_view、cmp/sels 接口实现，作为候选的实际接口依据。
这些阅读不构成已取得设备收益的证据。

首个独立候选针对 Score AIV 的 Native scale 页读取：原通用路径对每个 1024 候选
块执行 32 次 Scalar 页址及边界处理；候选用 Tile 一次计算页号、显式屏蔽非法页，
再读取相同的 64B scale 行。Native cache 仍是唯一外部根，在编排内建立只读别名，
无 Host 数据搬运。连续页快速分支、Score/FIFO/ACK/排空、浮点运算及 TopK 树保持。
非 64B page stride 或 INT32 索引范围外保留原逐页回退。

21 组 CPU 检查执行真实 helper 的算子逻辑，对照原逐字节读取，覆盖正/逆序与碎片页、
负页/越界/INT32 极大值、空表、31/32/33 页边界、重复尾页和三种 page stride，全部通过。
这只证明地址与字节合同，不代替设备代码、缓存协议和性能验证。

首版完整编译在 JIT 的 Tensor dtype/view 元数据推导处失败；第二版改为 INT8
64B 行别名，进入主体解析后发现分支 Tile 的 pad 类型不一致。第三版保持完整有效区，
显式携带 zero-pad 类型后重新编译。三个尝试均单独保存源码和结果，不改公共 SDK，
当前尚无该候选的设备性能结论。结果和后续取舍在此入口继续追加。

### 16:48：Scale 批量读取完成代码生成，首个十窗口任务已排队

第三版已通过主体解析，但 PTOAS 指出 A2/A3 TSELS 的 tmp 必须与 src/dst 同 dtype，
且至少容纳一整行。已读取公共 PTOAS `PTOScatterAndSelect.cpp` 的实际检查，
第四版改用 INT32 的完整索引行 scratch，保持原候选地址逻辑。
`compile_scale_gather_C04_v4/result.json` 确认 parse/full_codegen 均通过。

`scale_gather_v4_codegen.json` 从实际分配后 IR 与 C++ 核对：页表使用
`MGATHER<Coalesce::Elem, GatherOOB::Zero>`，scale 行使用
`MGATHER<Coalesce::Row, GatherOOB::Zero>`，三个 TSELS 屏蔽非法地址；
外部 kernel ABI 与原八档实测基线完全一致。UB 地址上界 180224B，Mat 475136B，
Left/Right 各 65536B，Acc 131072B，均在既有空间容量内。只读别名没有增加外部
cache 根或搬运任务。首次审计误按 `TMGATHER` 拼写查 C++ 而失败，修正为实际
生成的 `MGATHER` 名称；没有放松指令变体、地址类型或空间边界检查。

复用原八档 fixture/scale 构造和 L1 导出代码，生成独立测量器：
冻结参考用于 eager、图重放精确比较及交替重放，但不重新采集其性能；候选采集
10 个完整窗口，固定历史中位数仍是唯一性能分母。最终输出/TopK/score 和
cache/state 对冻结参考逐字节比较，Native 原有差异保持原状。
CPU 执行生成后的实际采样循环，确认十次 reference 重放不进入 DFX，候选有十次
成对 begin/end，每次只测一次图重放，原始记录目录互不覆盖。与地址检查合计
22 项通过，Ruff 与 shell 语法检查通过。

首个任务 `task_20261022_164733_203463730090`，指定历史 C04 使用的同一张 8 号卡，
单卡、最大运行时间 1500 秒，当前观察为 pending。候选源码为
`runs/scale_gather_v4`，输出 `runs/h131072_b16_s6_scale_gather_v4`。
本轮尚无已保留的新设备优化，未把编译、CPU 合同或局部工作量变化当成 10% 达标。

### 16:57：修复启动环境缺项，保留失败任务后重提

`task_20261022_164733_203463730090` 已 completed/exit1。日志停在 Native 模板
OPP 检查，未导入候选、未执行算子。启动器误从本 worktree 定位未构建的 OPP，
而原八档测量使用主检出的已构建 Native 模板依赖。修为 source 主检出的同一
`csa_native_template_20260929/env.sh`，CPU shell 路径前检通过；没有重建或改动 OPP。

同一冻结候选重新提交为 `task_20261022_165739_25359074832`，仍指定 8 号卡。
输出使用新 attempt `runs/h131072_b16_s6_scale_gather_v4_env2`；旧任务及初始化
失败保存在 jobs.json/current_state.json，不归因于候选算法，不计入性能样本。

### 17:39：Scale 设备失败转微型定位；N128 已完成编译和关键依赖审查

`task_20261022_165739_25359074832` 已 completed/exit1：冻结参考 eager 的输出、
状态与保护检查通过，候选首次同步报 507015/VEC instruction parameter invalid，
尚未产生性能窗口。原始 result.json、日志与本任务的 kernel dump 均保留在
`runs/h131072_b16_s6_scale_gather_v4_env2/`，未把该候选列为保留优化。

直接核对公共 Simpler 所绑定的 PTO-ISA `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`
发现 A2/A3 `TCmps.hpp` 的 INT32 分支实际调用相等比较，不能用于本候选的 GE/LT。
这是明确的语义缺陷，但尚不能认定它就是 VEC 异常的唯一原因。`MGather.hpp`
显式支持 INT8，Row/Zero 的非法行以 Scalar 写零；已排除“INT8 清零必然使用
不支持的向量 duplicate”这一未经证实的假设，没有修改公共 SDK。

helper 改用 FP32 做边界比较，同时保留 INT32 地址及选择。调用端限制容量小于
2**24，并保留原索引乘法范围检查；新增六组边界证明，包含 INT32 极值及 FP32
舍入边界，连同地址字节合同和测量循环共 28 项 CPU 检查通过。构造仅含原 helper
的独立设备诊断，比较碎片/非法/尾页的原始字节，不声称 CSA 性能。
首个诊断 `task_20261022_173315_40566603033` 在编译期发现 INDEX→FP32 Scalar
转换不支持，未执行候选 kernel。修为先转 INT32 后转 FP32，CPU meta 全量编译通过，
冻结到 `runs/scale_probe_v6_source` 后提交 `task_20261022_173811_16954222671`。

第二个独立候选 N128 仅扩展原长历史 G6/SPLIT_SCORE_ROWS 区间：Score N64→N128，
一个 QK Acc 依次接收两段 K128 点积，WS 的两段 K192 累加顺序及 FP16 舍入不变；
WS Right 在自身消费前装载，减少共存空间。它与 scale 候选没有叠加。
v1 超出 L0B（81920>65536B）；v2 改将 Key 下一块只预取至 L1，已完成全量编译。
`score_n128_single_acc_v2_codegen.json` 检查 UB 180224B、Mat 475136B、
Left/Right 65536B、Acc 131072B，外部 kernel ABI 与历史 C04 相同；实际 C++
24 次 QK 写回与复用保留 M→FIX/FIX→M 依赖，8 处 WS Right 复用保留 M→MTE1。
这只审查关键依赖，不替代设备正确性，也不保证减少 panel 必然胜过新增等待。

v2 的 Host 策略构造错误（dataclass 没有 to_l0 参数）在排队阶段发现，取消尚未执行
的 `task_20261022_173551_3898926889`。修为在 constexprs 中由已选 split/N128
生成 TO_L0；冻结 v3。八档及 T30/T36 两边界真实 select_policy CPU 前检通过，
40 个非 policy 模块与完整 C04 编译描述均与 v2 相同，因此复用其编译审查。
新任务 `task_20261022_173834_1885489714` 使用历史同卡 8，仅测 C04 的候选十窗口。
截至此记录仍无本轮已保留的新设备优化，完整八档 10% 目标尚未证明。

17:48 补记：scale 微型 v6 的首个普通页案例逐字节通过；重复尾页案例首行
64B 不一致，任务 completed/exit1。未扩大到完整 CSA。设备端 TSELS 的 scratch
含 Scalar 写入，因此另测纯 Vector 的 TSEL 路径，不据此直接宣布根因。
v7 两次 CPU 前检分别修复 full 参数顺序及显式 Tile 类型，v7r2 全量编译通过，
提交 `task_20261022_174755_7079965369`。诊断器现在保留每例失败的 actual/
expected 字节，并汇总五例结果；普通错误不提前丢弃后续案例。N128 v3 当前
任务 running，沿用原句柄，不重复提交。上述候选均未改生产默认或 push。

### 17:57：N128 性能负例归档；scale 指令链通过微型设备验证

N128 v3 的 C04 任务 completed/exit0，eager、图重放及采样后输出/Top-K/score/
cache/state 均与冻结参考逐字节相同，全部保护检查通过。十个有效窗口均为
72 核、一个 MIX task；中位数 915.88 μs，对固定历史 773.83 μs 为
1.18356745，即慢 18.36%。本版不保留，不扩测其他七档；原始十个样本和
校验索引见 `score_n128_single_acc_v3_result.json`。减少 panel 未形成完整收益，
不能仅凭生成代码断言回退完全由单 Acc 串行化导致。

`task_20261022_174755_7079965369`（9 号卡）completed/exit0：纯 Vector TSEL
版本五例 scale 字节检查全部通过，包含重复尾页、逆序、碎片、非法页和负逻辑页。
这仅解除微型正确性阻碍，不是完整 CSA 已通过。冻结完整 `scale_gather_v7`，
CPU 全量编译及 MGATHER/TSEL/ABI/空间审查通过，UB 等空间上界仍在原容量内。
C04 完整十窗口任务为 `task_20261022_175230_90457915022`，指定历史同卡 8。

为定位下一步核间优化，另从本轮冻结基线生成独立阶段诊断器。复用已验证的
native150 插桩方式，只在当前进程编译钩子改写候选 C++，不编辑共享 SDK；
每个阶段私有槽保留字 16..19 记录端点和累计等待。静态审查当前 C04 产物
覆盖 45 个等待位点、48 个通知位点，不修改原等待/通知/FIFO 数量或数学。
运行器保留未插桩参考逐字节对照及保护检查，每档取三个图重放的阶段记录，
禁止产生 statistics_us/ratio_to_frozen 或任何性能验收窗口。插桩自身会干扰调度，
只能作为阶段与尾部定位依据。C04/C08 两项诊断分别为
`task_20261022_175638_115263920914`、`task_20261022_175639_11540146694`。
没有重测合格的 Native/MPMD/BSH 性能，也没有更换本轮固定性能分母。

### 18:20：Scale 完整负例与当前阶段证据；按真实有效行重排 Score 组

`scale_gather_v7` C04 任务 completed/exit0，严格逐字节及保护/图检查全部通过，
十窗口中位数 802.88 μs，对固定 773.83 μs 慢 3.754%。本版不保留、不扩档；
`scale_gather_v7_result.json` 留下全部样本和结果索引。A3 的 MGATHER Row 实现
仍逐行发 DMA；本候选减少源码层页址分支，并未等价减少 DMA 次数，额外向量
寻址和等待是否占主要回退需另测，不能将该机制推断当作已证明的唯一原因。

两个基线阶段诊断均 completed/exit0，插桩结果与未插桩冻结参考逐字节相同，
各取三个图重放。插桩前 C++ 与对应历史产物仅 `// pto:` 位置注释不同；只去掉
这些整行注释后，完整可执行 C++ 相同，比较规则与原 hash 均保留。
`baseline_phase_analysis.json` 显示 Score AIC 完成时刻的最早/最晚差距中位数：
C04 8.28 μs，C08 76.64 μs。C08 三次最晚两核均为 6/7，4/5 随后；这些核
还承接不同前段工作。interval 含前一发布到当前发布之间的所有工作，记录的
GM wait 不含 FFTS/FIFO 等待，不把剩余时间解释为纯算力，也不把插桩值用于 G。

`score_owner_work_model.json` 按真实 Q 和当前 owner/叶公式统计 padded QK 行工作。
C08 最大/均值约 1.3469，假设将重组与轻组配对后约 1.0102；这只是工作模型，
不保证完成时序或全链收益。新候选 `score_group_order_v2` 针对原有 16-group
平衡叶区间，仅在容量允许、叶数为 3 的倍数、有效历史相差最多一行、重组不超过
一半且原来存在重/重配对时重排。原请求归属、叶边界、所有浮点计算及 TopK 归并
树保持；AIC/双 AIV 使用同一正向表，TopK 根等待用逆向表寻找真实生产者。

映射放在既有 4*T 描述符容量内的未用部分，显式检查边界，不侵入后续 scale-plan
分区，不增加 Native 根、工作区大小、task、通知或屏障。每轮重新写启用标志，
条件不满足恢复逻辑恒等，不能消费前轮的旧表。CPU 三项属性检查运行真实 helper，
覆盖 40 组 group/leaf 组合、全部根写者与等待对应、同址启用→关闭及不等历史回退。
C08 全量编译通过；Mat/Left/Right/Acc 上界保持，UB 163872B，packed ABI 与历史
C08 相同；静态 GM notify/wait 位点仍为 48/45，但这不代替设备事件验证。
任务 `task_20261022_181543_203120114774` 指定历史 C08 的 2 号卡，当前 pending。
本轮已测两个性能候选均未保留；生产实现不变，八档 10% 目标继续 active。

### 19:27：混长组重排初步有效，隔离分档并验证服务与状态

组重排 v2 的 C08 任务 completed/exit0，完整精确/保护/eager/图及采样后检查通过，
十个有效窗口中位数 728.33 μs，对固定 750.79 μs 降低 2.9915%。原始十窗和检查
索引见 `score_group_order_v2_result.json`。这是单档初步收益，不是八档综合达标。

v3 增加 `SCORE_GROUP_ORDER` constexpr。Host 仅以既有长 G6、T>=32 且容量
T<requests*6 选择是否编译；实际组数、有效行与历史差异仍由 device 判定。
八档真实 select_policy 前检确认只有 C08 开启，其余原有策略项全部相同。
生成代码比较覆盖全部八档：七个关闭档在源码/两个 packed 封装/Host 编排及
kernel ABI 上与历史产物一致；同类的 C02/C04、C05/C06 共用代表编译，不重复
做完全相同的 CPU 编译。开启的 C08 与实际设备测过的 v2 产物一致。
比较只移除整行 `// pto:` 位置注释，不改变量、指令、地址或事件；无 ELF 同一声明。
证据 `score_group_order_v3_codegen.json` 支持复用七档历史窗口及 C08 的新十窗，
没有重新采集已合格的其他五列性能。

服务集成审查发现 frozen combined_v1 的两个 Host 文件早于仓库已有 Native
`max_seqlen_q_actual` 修复。v4 保留当前仓库 `service.py/service_config.py`，
防止把该修复覆盖回旧版；其余 40 个模块逐字等于 v3，核内和编译策略不变。
这项 Host 同步不算新的核内优化或重新定义基线。完整服务门禁仍需真机确认。

已冻结两个只做功能的工具，均禁止生成性能统计：

- `task_20261022_190734_409850120824`：v3 同址状态矩阵，15 个 metadata 场景及
  4 步连续 decode；新增重排开启、恒等映射、历史不等、重新开启/逆序物理 owner、
  碎片页五例，核对实际 flag/正向表/逆向表及每个根的状态。CPU observer 前检
  验证三种路径和篡改 flag 的拒绝，4 项通过。
- `task_20261022_192200_5031203393`：v4 正式 `spmd_tnd` 模型准备、canonical
  custom-op、ACLGraphWrapper、工作区复用、空输入和 Native 回退。保留现有
  服务验证规则，明确要求实际服务选中本混长分档，不把直接 Prepared 调用
  当作正式服务路径。当前两项都 running，沿用原句柄等待。

### 另两个独立核内尝试的完整结果

局部连续页证明 `local_scale_v1` 仅在当前 32 页全部合法连续时用二维带行间隔
load，否则逐字保留原 Scalar 页循环和重复尾页；原全请求连续证明路径不变。
25 项 CPU 字节/浮点整数界限检查及五例设备微型检查通过。实际生成一个
32×64 INT8 TLOAD，归约保持物理 8 行、有效 1 行，packed ABI 与空间上界不变。
C04 完整十窗为 769.31 μs，对固定 773.83 μs 仅降低 0.5841%；精确/图/保护全过。
当前只视为小幅单档结果，未保留、未扩大为八档收益。记录 `local_scale_v1_result.json`；
微型任务 `task_20261022_183054_268827429855`，完整任务
`task_20261022_184933_341013719076` 均 completed/exit0。

N128 双 Acc 变体把每个 192 行 QK 半块拆成 M128/M64，各用 64/32KiB Acc，
FP16/ReLU 逐元素转换位置、完整 K128 点积及后续两段 K192 WS 顺序不变。
九项 CPU 检查执行生成后的切片/插入，覆盖随机/极值/交替输入；归约与 FIFO
调用保持。全量编译及 48 个 QK 插入/复用、8 个 WS Right 复用依赖审查通过。
任务 `task_20261022_184613_33179372574` completed/exit0，完整精确检查通过，
十窗中位 862.32 μs，比固定基线慢 11.4353%。虽然比单 Acc 版本更短，仍不采用，
不扩档；全部样本见 `score_n128_two_acc_v1_result.json`。两种 N128 负例不再盲目重测。
本轮生产代码尚未改变，没有 push，10% 总目标仍 active。

### 20:40 更正工具链可比性；优先推进 128K/B16、B24

公共 SDK 在 18:00 发布、18:14 验证为 `csa-shared-20261022-logical-codegen`，
PyPTO 从 e891351e 升到 26e94e05；旧冻结基线仍是 20261021 版本。逐项读取
原 task 启动日志后，确认组重排 v2、局部 scale v1、N128 双 Acc 都使用新 SDK。
此前针对这三项计算的跨 SDK 比率全部撤回：C08 的 2.9915%、局部 scale 的
0.5841%、N128 双 Acc 的 11.4353% 以及八档约 0.38% 均不作为验收结论。
这是本会话版本检查遗漏。原始窗口和报告完整保留，不改标工具链或删慢样本；
摘要增加不可比标记，证据见 `sdk_provenance_audit.json`。源码/ABI 一致只能证明
代码等价，不能豁免 SDK 一致要求，19:27 节的七档历史性能复用声明同步撤回。
N128 单 Acc v3 和 scale_gather_v7 的实际启动版本与旧基线一致，原负例仍有效。

组重排的同址 15 场景/连续 4 步、canonical SPMD 服务验证均 completed/exit0；
126 项生产 CPU 测试、C08 完整编译和代码等价审查通过。性能未确认前撤回本会话
临时生产集成，冻结 v5 与 UT 快照保留；没有覆盖其他 session 修改，没有 push。
compact Score pl.range 变体 CPU 代码生成因 Mat 同源同目的 TMOV 失败，未上设备。

用户最新要求优先优化 128K/B16/S6（C04）、128K/B24/S6（C05），最终八档等权
几何平均降低 10% 的口径不变。原 combined_v1 算法分母和旧 SDK 数据保持不变；
另行补齐同一实现的新 SDK 基线。已有新 SDK C05 历史十窗来自更旧 SPMD 源码，
完整代码不匹配；名为 h131072_b16 的记录实际是不同混长、容量和 seed，均不能
代替 C04/C05。本轮先补这两个缺项，不重测 Native/BSH/MPMD 已有列。采样器将
记录实际 SDK 并拒绝跨版本收益计算。C08 候选暂缓扩展。

### 同 SDK 采样门禁与两重点档候选准备

`measurement_provenance.py` 和独立冻结的 `sdk_measurement_harness_v1` 保留
原 fixture、严格逐字节比较、复位、warmup 和完整 72-role 单 MIX L1 边界。
启动时记录 bootstrap 实际 release、三个 SDK 根及 Git HEAD，核对已验证 manifest
和实际 PyPTO 导入路径；不扫描 SDK 文件。校准必须逐文件等于原 combined_v1；
候选只能对同 SDK、同 case、同源码且十个完整窗口一致的校准结果计算比率。
16 项门禁/真实采样循环检查通过。历史 harness、baseline.json、原始报告不变。

两项缺失校准已提交：C04 `task_20261022_204306_347423614603` 指定原候选卡8，
C05 `task_20261022_204306_347428319389` 自动分配。当前均等待他人的16卡任务，
不重复提交、不干预其他任务；每档只采十窗，参考重放仍只用于准确性和交错。

只读复用另会话 d263bc3f2 的 `indexer_leaf_scale_layout`，候选 `leaf_scale_v1`
在每叶实际 AIV 对至多256页完整证明正/逆连续，结果留在本核Scalar，删除单
META写者的整请求布局证明。原连续读取和逐页回退、数学与FIFO保持；B16开启
此读取，B24延续连续读取但改变证明位置。暂保留原可选2*B尾分区容量，无新跨核
共享状态。两档完整CPU编译通过，UB地址上界180224B，其余各空间不超限；
15项属性测试覆盖每个中间页破坏、正逆方向、尾叶和2^24整数边界。
来源是可借鉴的功能机制，不复用MPMD性能收益。完整L1/状态门禁尚未采集。

先前逐块分布式证明 `distributed_scale_v1` 的C05完整CPU也通过；它每32页重复
校验，现优先更少重复的叶内证明，未提交其NPU任务。B16另尝试保持N64数学与
双QK Acc，拆出首尾panel后将中间14块写为pl.range，解决先前条件Mat自拷贝
失败；当前仍是CPU可行性实验。首次生成空if语法失败且编译器入口无源码，未
执行设备；修正生成器后冻结并使用全新目录编译，所有失败记录保留。

21:03补记：两校准已运行，B16卡8、B24卡0，参考eager通过，正在编译候选侧
（同一个未修改基线）。另提交C05阶段诊断 `task_20261022_205927_375116654`，
卡2运行，同公共SDK，三次插桩重放，只用于端点/等待定位，禁止进入性能统计。
B16已有阶段记录不重复采集。Score工作模型显示C04/C05的24核组按padded行
步数都为max/mean=1；不把块数相等称为实际时间相等。

缩循环peeled v1在ConvertToSSA报buf_key_panel作用域错误；v2将已知prefetch
constexpr在分支内展开后通过SSA，但最终Mat仍同址163840自拷贝，未上设备。
停止此方向在当前公共SDK下的进一步试探，不修改共享SDK或删除生成指令绕过。
本阶段已本地提交cd7a96149，没有push。

### 21:06 两重点档同 SDK 基线完成，开始叶内证明完整对照

C04/C05校准均completed/exit0，严格eager、graph、post-L1及所有保护检查通过，
每档十个完整72-role单MIX窗口。原combined_v1源码逐文件相同，新SDK固定为
20261022-logical-codegen。C04卡8中位778.39μs，C05卡0中位1014.46μs；全部
样本、窗口路径、来源和SDK在 `baseline_current_sdk_C04_C05.json`，旧baseline.json
保持原样，不将环境变化列为优化收益。

复用已完成的同新SDK候选十窗，重算C04局部块证明769.31μs相对778.39μs降低
约1.17%，暂不采用；N128双Acc862.32μs相对新分母仍慢约10.78%，停止扩展。
旧跨版本比率仍保留为撤回记录，原报告不修改。C08没有同SDK基线，收益仍未确认。

叶内证明对照已提交：C04 `task_20261022_210538_22017831360` 卡8，
C05 `task_20261022_210539_22025123534` 卡0；同新SDK及同卡校准，完整十窗与
严格回归。当前不能预判性能通过，不改生产默认、不push。

### 21:20 叶内证明首版无收益；根据B24阶段记录调整后续方向

`leaf_scale_v1`两档completed/exit0，完整精确、保护及图检查全部通过。C04
788.33μs，比同SDK778.39慢1.277%；C05 1021.82μs，比1014.46慢0.726%。
本版不采用、不扩档；全部十窗见 `leaf_scale_v1_results.json`。这也说明另一
会话MPMD中消除AICPU扫描的收益不能套用到单SPMD。后续只隔离一个具体开销：
物理8×256证明Tile保留，但有效行设1，删除七行相同向量工作；两档全CPU编译
及最终valid_shape=[1,1]归约/ABI/空间审查通过。与八行版共30项页表属性测试
全部通过。one_row的C04/C05任务分别211934_72225422027、211934_72236616480。

B24三次阶段诊断完成且逐字节相同；插桩前完整可执行C++与同SDK校准相同。
META端点中位66.64μs，Score AIC完成差33.38μs、AIV差39.18μs；这些含探针值
只用于定位，不作为性能窗口或跨SDK比较。三次均是AIC4～7在KV_PROJECT及
SCORE_CUBE最后，前置QB/KV负载叠加比Score块数更值得处理。详细全阶段端点
见 `baseline_C05_phase_sdk_v1.json`，不把GM wait扣除后的余数叫纯计算。

`defer_kv_v1`仅在既有COOPERATIVE_PREFIX、长G6 split、OA_COLS=256区间，
把原KV投影从分支末尾移到本核Score全部FIFO排空之后、Attention之前；数学
helper、logical owner循环置换、KV真实完成通知均保留。AIV仍在自己Score归还
后才等待KV及消费，未加跨核同步或提前通知；可能与Vector收尾/部分TopK重叠，
也可能推迟KV norm，必须用完整L1取舍。两改动文件之外全部等于基线，8种策略
组合证明KV恰好执行/发布一次，另有数学/工作区合同检查，共9项通过。完整C05
编译和ABI/资源审查通过，任务211820_67056725661卡0排队。

### 21:37 B16单有效行取得单档正向，B24 KV后移为负例

`leaf_scale_one_row_v1` C04 completed/exit0，十个完整窗口中位763.65μs，
对同SDK同卡冻结基线778.39μs降低1.89365%；六组严格eager/graph/post检查
及保护全过。这是单档候选，尚未宣称生产采用或八档达标。已提交真实T96/R16
的同址/连续验证213452_137176811781，沿用原10项metadata矩阵并增加满容量S6
及满容量反向物理owner两例，共12例、4步连续decode；没有DFX性能测量。
C05同一证明候选仍在执行。

KV后移C05 completed/exit0，严格正确性通过，十窗1026.00μs，对1014.46μs
慢1.1376%；不采用，不将局部Score提前等同整链收益。后续独立候选
`rotate_kv_v1`保持全部阶段原顺序，只把既有大T长G6资源域的KV循环owner偏移
从4变为16：T144的8个非空逻辑worker由物理4～11移到16～23。24逻辑worker
仍为双射，AIV原先就等待全部24个真实KV生产者，故消费者/通知合同不变。
候选只改变两个调度文件，未改变任何数学、buffer或阶段位置；正在完整CPU编译，
还没有设备性能证据。O_B尾部不均衡另外记录为后续假设，尚未实现，不并入本轮。
本地阶段提交新增0fcdcbf16，仍未push。

### 21:46 收窄页证明采用候选；B16动态状态完成

C04的12项同址metadata和4步连续decode completed/exit0，实际T96/R16，
完整输出/Top-K/score/cache/state、保护区、代际及几何全过。C05单有效行
中位1015.58μs，对1014.46μs为+0.1104%，不将其说成收益。

新增`leaf_scale_guard_v1`：只在既有长G6且64<=T容量<=96区间启用叶内证明，
其他区间恢复原META证明及scale策略；八档仅C02/C04开启，C05/C06保持原策略，
C03/C08也保持原路径。四项核内调用链显式传递constexpr，页表与实际可见范围
仍由device校验；并保留生产中已有service.py/service_config.py的Native
query-bound修复。正在编译C04开启与C05关闭的完整产物，核对同SDK实际代码。
保留C02尚缺性能验证，不能据同T就预记更大历史收益或八档通过。

KV核分工旋转的21项CPU覆盖检查通过：T1..1025及分段/尾块，24逻辑worker
始终双射，输出tile/有效行集合相同，数学与Vector消费者未变。最终IR确认为
`(worker+8)%24`，packed ABI不变；C05同卡任务213951_157406224063正在运行。
其余O_B尾部调度仍只是记录中的后续假设。

### 21:56 B16正式服务通过；B24转向按尾行成本分配O_B

leaf_guard C04开启的完整C++、两个packed wrapper、Host编排与ABI与实际
测过的单有效行版本完全一致（仅去pto位置注释）；C05关闭与同SDK两份基线
产物均一致。48项策略/边界检查通过。正式`spmd_tnd`模型/custom-op、
ACLGraphWrapper两次重放、工作区复用、空输入和Native回退completed/exit0，
实际服务明确LEAF_SCALE_LAYOUT=true，逐bit及所有保护通过。C04无需重复计时，
保留763.65μs及12+4状态证据。仍未验证C02性能和完整八档目标，不改生产默认。

B24 rotate_kv_v1严格对照通过，十窗1010.04μs，对1014.46μs降低0.4357%；
幅度小、样本交叠，不当成稳定可采用收益，不重复同源计时挑轮次。旧v78/v82
大T输出分工也只有微小收益，本轮不重复其按16重核/8轻核简单分配的方案。

新`output_tail_balance_v1`针对一个完整M128加不超过32行尾块的既有资源
区间（长G6、128<T<=160），根据C05真实OB最晚0～3/8～11核，转移8项
原O_B工作到只处理两份O_A尾行块的20～23核：仅group0/1的dc0～3改变owner，
其余56项不变。完整64格输出恰好一次，每组仍等同一quant生产者，数学helper
及Vector全24核最终等待不变。17项CPU覆盖/guard检查与完整C05编译通过；
这是新的按完整/尾行成本分配方案，不能预估端点差就是收益。独立C05同卡
完整十窗任务已提交，结果待收集。

21:58任务补记：C05输出尾行分工任务为215626_232405323561，沿原卡0十窗。
C04收益及动态/服务门禁已全部通过；由于64～96长G6策略同时覆盖256K/B16，
为避免未验证历史扩展而误用，启动C02原实现的新SDK缺项校准
215719_235949521745（卡8），之后仅测guarded候选；不重复已有C04窗口。
仍不补跑Native/BSH/MPMD无关列，未把当前两重点档当完整八档验收。

### 22:09 B24输出分工无收益；补查新归档避免重复C02基线

output_tail_balance_v1 C05 completed/exit0，完整严格检查通过，十窗1018.45μs，
对1014.46μs慢0.3933%。本轮C05叶证明、KV后移、KV置换及按尾行分配O_B均
未形成可采用的净收益，保持原策略；所有窗口和负例保留。C04叶证明的服务、
状态和代码等价证据已本地提交3a0f423c6，未push。

补查之前遗漏的新增归档目录csa_performance_20261022_aicpu，发现20:07提交、
已完成的C02同SDK十窗：task_20261022_200748_181884127036，卡9，中位969.17μs。
该SPMD源码含另一候选的关闭分支，文件hash不同，但此case完整C++、两个
packed封装、编排及ABI与本轮同SDK未改基线完全一致；原case、fixture、72核
L1原始时间戳、SDK启动头和十窗统计均重新核验。证书C02_reused_reference.json
明确保留实际源码hash差异，不把它伪装成相同源码。

本会话C02重复校准215719_235949521745在compile_and_eager_candidate阶段
被主动终止，0个L1窗口，exit130，不作为失败性能/无效慢样本。之前的查找范围
漏了新目录，是本会话检索遗漏。C04/C05已经完成的同卡校准不删除、不择优重置；
新目录另有不同卡的776.29/1007.72μs记录，保留为额外背景。B16方向一致，
B24的微幅正向更不支持采用。

C02只测新guarded候选，task_20261022_220713_27566122223固定同一原记录卡9。
独立sdk_measurement_harness_v2只增加“候选采样、外部证书比较”模式：仍检查
实际SDK、以原41模块基线做严格正确性参考，原warmup和十窗循环逐AST相同；
原始结果不产生ratio或伪造基线hash，待结束后用证书计算比较。所有旧harness
和报告保持原样。

22:23补记：C01/C03/C07三个关闭资源类也完成CPU编译；结合C05代表C06、
C03代表C08，六个关闭档的完整C++/packed/Host编排/ABI均与原基线一致，
只删除pto位置注释做比较，见leaf_scale_guard_disabled_classes.json。
这份证书只证明关闭路径代码不变，不允许把旧SDK计时改标成新SDK。

C02外部复用还核对了scenarios.py及EngineArgs、make_layer、make_fixture、
native_bind、prepare_weights调用AST，均相同；未使用的Native/BSH工具及
L1之外outer profiler差异单列。没有伪造源码hash或替换任何原始报告。
当前C02候选任务仍排队等待CI释放16卡，沿同一task句柄等待。

等待期间只读研究后续信号布局：当前512B槽间距明确是沿用的验证值，并非
硬件最小行声明；公共编译器文档/ISA说明单写者64B隔离及TNOTIFY的DCCI、
DSB和末尾PIPE_ALL。潜在缩距只作signal_layout_research.json假设，未改源码
或提交设备任务。尤其不能删除通知前PIPE_ALL：通知后的barrier不保证信号
发布前业务写回完成；现有阶段插桩使用槽内16～19字，若缩到16字也必须重做。

### 22:47 本轮重点优化收尾：仅保留128K/B16的新页证明

C02 guarded v1在原记录同卡9 completed/exit0，严格逐bit、图及保护全过，
十窗972.05μs对复用969.17μs为+0.2972%。保持原始采样报告ratio=null，由
同SDK/同case/同卡及外部等价证书计算比较，不伪装新的基线测量。

最终`leaf_scale_guard_v2`只进一步收紧Host历史资源范围：压缩长度上界
`cache_rows<=6*8192`，即原8192候选叶不超过6个的区间；当前八档只有C04
开启，C02恢复原策略，其余六个关闭档也不变。v2所有核内模块逐字等于v1，
C04全部constexpr仍相同，原性能、12+4动态状态及服务验证可以复用；C02
关闭完整CPU产物与同SDK历史基线的C++/packed/Host/ABI完全一致。104项
Host分档/历史边界测试通过，未按B16单点枚举，也未宣称区间内每点同等收益。

已将最终版本集成到本会话隔离worktree的正式TND-SPMD目录，共6个算子文件
（含1个新增leaf helper）。同时承接原先已验证、已作为本轮分母的combined_v1
三项逻辑，不把它们重复计成新收益；保留原service/query-bound修复。更新说明
和格式时仅规范import/module文档及未使用变量名，完整非文档AST和导入集合
一致；生产目录C04开启与C02关闭完整编译再次匹配冻结产物，见
leaf_guard_production_style_audit.json / leaf_guard_production_codegen.json。

正式CPU合同、variant及query-bound检查130项通过。原5项G2 QB预期同步到
本轮冻结基线已经采用的窄N/4前缀核策略，新加15项叶证明分档边界；未改阈值
以迁就性能或数值。针对源码的Ruff及git diff检查通过，没有重复NPU计时。
本轮只保留1项新优化：C04完整L1 778.39→763.65μs（-1.89365%），其余
7个业务档保留基线策略/生成代码。B24无可靠收益，未采用弱正向或负例。
所有设备任务与CPU编译已收齐，无活动本会话任务。八档10%总目标仍active且
未达成，正式G不靠跨SDK数据构造；不更新goal为complete，不push。采用证书
为retained_leaf_scale.json，完整原始窗口及过程继续保留。


### 等长TND独立分支：用户授权与首个候选

用户明确允许多batch实际Seqlen一致、等效BSH时采用独立分支；128K/B16和
B24仍优先，八档等权10%及原combined_v1分母不变。此前signal slot footprint
只有只读研究，尚未改码或上卡，先转向这项新授权。

`uniform_geometry_v1`以已采用的生产提交79fae6db0为父版本，保留C04单有效
行叶证明，只改request_geometry、attention_plan和attention_stream三个模块。
沿现有PLAN_PHYSICAL_ROWS资源域，每个实际工作AIV每次调用检查全部Q/L：
B>1，Q从0开始、严格等间距S>0、B*S=T，全部KV length相同且>=S。通过后
请求定位改为query//S；其它输入仍走原query_request扫描。早期Attention
几何可复用一次证明处理八行，后续AIV在多个query之间复用自身的核内Scalar。
不跨replay保存谓词，不回读device内容到Host，不新增GM工作区/任务/同步。
原positions因果校验、页表保护及全部数学保持。此版保守地要求query和KV
两种长度都相等，未把只有shape或query长度相等当成充分条件。

同源helper对独立owner集合oracle的71项CPU测试通过：S1至S16、B1/2/8/16/
24/28/32，逐个中间Q/L篡改、padding/重叠/短存储、同址真/假反复切换及随机
非法边界均覆盖。CPU完整代码生成进行中，尚无设备正确性或新收益结论。
新候选不是已采用版本，不覆盖原始冻结结果；manifest和生成脚本位于已有
csa_spmd_tnd_eight_20261022目录。

B16/B24完整CPU代码生成均通过；C04的AIC函数、两侧packed入口和orchestration
与已采用生产版一致（仅忽略pto位置注释），变化在AIV内部几何分支。B24初次
CPU启动描述符文件名写错，未进入解析/设备，改为经完整policy核对的已有描述符
后成功。设备任务B16=`task_20261022_231311_146232724441`（卡8），
B24=`task_20261022_231741_162492223535`（卡0），各10有效窗及全部样本；
仍对原combined_v1的同SDK基线比较，另观察是否超过已采用C04的763.65μs。
同址验证器已准备16组metadata和4步连续解码，增加等长→不同KV长→等长、
等长→失活padding→等长转换，仅当候选值得继续时执行设备门禁。


### 等长几何候选收齐：两档负例；PyPTO问题已上报

两任务completed/exit0，各10个完整72角色L1窗口，reference/candidate的eager、
graph和post-L1全部逐bit与guards检查通过。C04中位789.36μs，对同SDK原分母
778.39μs为+1.4093%，对已采用763.65μs为+3.3667%；C05为1024.20μs，
对1014.46μs为+0.9601%。完整样本、原始路径和任务信息见
`uniform_geometry_v1_results.json`。两档均不采用，不再为此负例执行16+4状态
矩阵或服务门禁。正式生产代码仍是79fae6db0已验证版本。本结果只否定这版
几何寻址实现，不否定用户新授权的等长专用计算/调度路径；未将控制流或寄存器
变化武断归为已证实瓶颈。

补充关闭档编译发现宽度恒0时，虽然除法受if width>0保护，Simplify仍先折叠
不可达的query//0。增加外层constexpr helper调用开关的v2，C04编译成功但
C03关闭档仍失败，未进行额外NPU计时，未改已排队/运行的v1冻结源码。
最小@pl.program复现只运行InlineFunctions→Simplify：前者成功，后者报
const_fold.h:218；将受保护除法的分母写为pl.max(width,1)的对照可正常化简为0。
此处是CPU编译问题，不归因于PTOAS/Simpler或硬件，不宣称与本次性能回退有关。

用户明确授权发现PyPTO、Simpler、PTOAS的问题可直接提issue。已查重并提交
[PyPTO #3114](https://github.com/hw-native-sys/pypto/issues/3114)，含精确commit、
独立最小复现、预期IR、错误及已验证绕过；提交后再次读取核对正文和OPEN状态。
记录为`upstream_issue_3114.json`，本地复现脚本为repro_unreachable_floor_div.py。
仓库模板所写ir-pass标签实际不存在，首次未创建；核实无同名issue后不带标签
成功提交，没有重复issue。公共SDK未编辑/重建。

本会话本轮两项设备任务及所有CPU编译已结束，无待收结果。八档10%总目标
仍active且未达成；未push。后续SDK问题沿用本次用户授权，确认与查重后直接
上报相应上游，并在本统一记录中保留链接，不另外拆出过程MD。


### 64B与512B核实：撤回缩槽候选，保持已验证协议

用户指出Scalar DCache行与MTE/GM访问粒度不能混为一谈。查阅[A2/A3官方
避免同地址访问说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00013.html)确认：MTE2、MTE3和Scalar的GM请求均按512B
粒度处理，落在同一对齐512B区域的并发访问可能串行。[架构说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)
中的64B指Scalar DCache行；通用L2行大小随硬件变化，不能把该页的范围值
写成已查实的本机A3 L2物理参数。[缓存路径说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cache_control/system_cache_overview.md)也表明Scalar经DCache
后仍访问L2/GM，512B关注点并非MTE独有。

因此，64B私有槽避免多个写者共用Scalar DCache行，与512B分隔减少GM请求
竞争是两个层次。缩槽把8个不同控制槽放进一个512B区间，不能仅靠分配字节
减少推断同步提速。本轮撤回signal_slots16_v1设备测试，生产128个INT32/
512B槽保持不变，尚未提交任何本会话NPU任务。C04完整CPU生成成功仅表示
可编译；辅助72角色探针因自身INT32 errors + INDEX常量类型不匹配停在解析，
不是SDK故障证据。预备源码与两份描述符归档在既有runs目录，未丢弃证据。

用户提供的[A5 atomic使用手册](https://github.com/nalinaly/simpler/blob/3699b7ae4ff47abcfa1e2916881d4309cf39c00f/tests/atomic_probe/ATOMIC_USAGE_GUIDE.md)已按commit与blob固定参考。其64B规则
着重所有权/一致性；DCCI脏行可能覆盖新值，DSB不能代替写回或跨核交权。
A5特定atomic发布省DSB等实测不直接外推A3。CSA实际Set仍是DCCI→volatile
GM store→DCCI→DSB_DDR，wait为DCCI与Scalar读取；原PIPE_ALL、阶段维护及
等待均未删除。后续审查将分别核对控制字、业务payload与整行所有权，再决定
是否有冗余操作；不把手册的保守工程规则当成当前A3已经复现故障。

审计和链接见signal_layout_audit.json。本轮没有新增性能样本、没有共享SDK
修改，也没有改变已采用算子。原八档10%目标继续active且未完成。


### 2026-10-23 输出量化两项独立候选：B16分工，B24未读尾零

本轮先查AIC纯Tile发布维护，发现10月21日已完整验证B1/B16且无收益，故不
重复该候选。只读核对另两个session最新记录：BSH已有WS/B1/归属措施多已
承接或有负例；AICPU-MPMD的sync_start收益属于多task启动，不能直接套单
SPMD。MPMD v17增加量化并行度可缩短核内量化，但完整输出链变慢，因此只
取它作为局部工作分配的线索，不迁移其性能数值或直接采用。

分别从已采用生产79fae6db0冻结两个候选，保留C04叶证明和512B信号槽。
`output_quant_task16_v1`只把量化逻辑任务32行改16行，实际8行数学tile及
stage2流水、全部舍入/归约不变；Cube与Vector组等待同步使用同一常量。
C04的量化有效AIV从24增至48，每核32行降至16行，但通知/等待增多，完整
收益必须实测。先只测C04，不预判其它档或全域采用。

`output_quant_no_unused_pad_v1`独立保持原32行量化分工，仅在NZ且OB_ROWS>96
时把尾零范围终点改为真实T。既有大M O_B早已按active_rows读activation并
只写真实T，其完整AIC C++与候选逐字相同（只忽略pto位置注释），因此没有
将padding删掉却留给完整M读取。小/中NZ和全部ND仍清零；T内失活query行仍
正常计算。C05少写8*112*1024=917504字节INT8尾零及对应生成指令，不把这
个逻辑字节量当作实测带宽或收益。

14项CPU测试对33个T边界运行实际分工循环，独立验证全部有效行/尾零唯一
覆盖、最终尾owner，以及Cube/Vector等待集合确实包含全部实际writer。
两档完整CPU生成成功；quant数学体与O_B消费者不变。新增测量harness只在
非计时snapshot检查候选o_quant[T:]始终保留INT8哨兵37，覆盖eager、graph、
全部L1后；原严格输出/Top-K/score/cache/state和外保护均保留，计时循环
与完整72角色首尾边界不变。

任务C04=`task_20261023_000814_330260230692`（卡8），C05=
`task_20261023_000814_33026777254`（卡0），各10有效窗。原combined_v1同SDK
分母仍778.39/1014.46μs，C04还须超过已采用763.65μs才值得继续。当前尚未
收齐设备结果，生产未改，公共SDK未改，目标active，未push。


### 两项输出量化候选终态：正确但没有完整L1收益

两任务均completed/exit0，各侧eager/graph/post-L1的输出、Top-K/score、
cache/state逐bit与保护区通过。C04 task16中位779.06μs，相对原同SDK
778.39μs为+0.0861%，相对已采用763.65μs为+2.0179%，不采用。
C05省未读尾零中位1017.81μs，对1014.46μs为+0.3302%，不采用。
C05三次snapshot均证实整个o_quant[T:]保留非零哨兵37，说明结果没有依赖
被省去的清零；正确性与性能分别判断，不能把省917504字节称为耗时改善。
全部20窗口、72角色记录、任务/SDK/原路径见output_quant_priority_results.json。

这两条负例不再扩大到状态/服务及剩余档，正式算子继续保持已有采用版本。
旧PipeUtilization记录也已只读检查，但其来源是20261020较早Score实现，
不是本次SDK/算子；不按那些Scalar/MTE累计值定量解释本次耗时或合成收益。

后续等长专用路径仍获用户授权。已排除的uniform_geometry_v1在每个AIV
各做一遍全Q/L证明；下一步研究由已有META单写者在本轮元数据扫描中证明
一次，再由原依赖保护消费者读取。先核对工作区所有权和发布时机，不新增
Host回读、静态测试点任务表或绕过causal/页保护，不把未实现方案当成收益。
本轮无待收设备或CPU任务，八档10%目标active且未完成，没有push。


### META统一证明的等长候选：单写者、常量除法与原通用回退

上一轮已排除每AIV重复扫描Q/L的uniform_geometry_v1。本轮uniform_meta_v1
仍从生产79fae6db0派生，但把证明融合进既有indexer_group_plan_stage请求
扫描：仅PLAN_PHYSICAL_ROWS且G6域，B>1、T=6B、全部Q[r]=6r且正L一致时
写INT32标记1，否则0。前B+1组前缀按16项对齐后的最后一字只有在确实属于
padding时才用；B+1恰对齐的布局不写/不读证明，保持通用查找。全部8项plan、
真实组描述符和scale计划不变，无额外Tensor根或Host回读。

该字由原META唯一写者每轮覆盖，与其它元数据一起发布，之后直到kernel
结束只读。它不是新增同步信号，512B阶段槽及现有DCCI/DSB/PIPE_ALL保持。
早期几何只有实际有行的AIV先获取原META；晚期Attention在已有Indexer压缩
helper无条件获取META之后读取。通过证明的查找用query//6常量算术；请求
位置因果、页表、gather、数学和FIFO全部保留。混长、padding、不同KV长度、
无预留字及关闭策略均走原query_request，不把形状相等当实际等长。

63项同源CPU检查覆盖Q/L中间项改变、失活、B15/B31无padding字、S1～S6及
同址真/假/真切换；完整plan与原组描述符逐项一致，仅允许专用预留字变化。
C04/C05开启档、C03关闭档完整编译通过。C03的完整C++、两packed入口和
orchestration与基线相同（只去pto位置注释）；C08使用同一constexpr集合与
动态代码类。关闭域没有重复上一版恒零分母的问题，快路径除数始终是6。
证据uniform_meta_v1_codegen.json，源码和hash见uniform_meta_v1.json。

测量器仍保留原10窗、交替未计时reference、完整72角色首尾跨度。只在eager/
graph/post-L1的非计时snapshot追加读取实际META标记并要求等于1，防止死分支
或BOOL编码导致未启用却误记收益。C04卡8任务task_20261023_003650_1390799683，
C05卡0任务task_20261023_003836_22329631593，当前队列状态按task-submit跟踪。
CPU通过不代表设备/状态验收，原生产不变，目标仍active，未push。

同址验证器已从既有uniform_state_harness_v1派生，仅新增逐轮真实META标记
与CPU侧独立Q/L谓词比较，继续保留全部16组metadata、4步连续解码以及
输出/Top-K/score/cache/state、地址/代际/保护检查。等长→不同L→等长与
等长→padding→等长都读取实际槽值，未将Host预期代替device结果。准备证据
uniform_meta_state_v1.json；当前尚未提交此功能任务，等待性能是否值得推进。

最新task-submit确认C04/C05原任务均pending，16卡CI任务
`task_20261023_000924_333970730698`仍running；没有取消、重复提交或换源。
此为已核实的队列等待，不作为失败、无进展阻塞或性能结果。源码阶段提交
91173e5cc，验证准备另行本地提交，均不push。


### 2026-10-23：按用户要求转向完整数据流、成本与调度分析

用户明确要求从更宏观的角度决定下一步性能方向。原八档等权几何平均降低
10%的目标、combined_v1分母、TND-SPMD优化对象及C04/C05优先级均保持。
后续候选先说明改变哪条完整依赖链、减少哪类实质工作，再考虑局部参数。
不将“多做并行”“少几条Scalar指令”直接当作整层收益。

#### 收齐当前等长META候选，保持已采用实现

C04/C05两任务均completed/exit0，各10个有效完整72角色L1窗口，实际META
证明在eager、graph、post-L1三次回读均为1。输出、Top-K/score、cache/state
逐bit及全部保护检查通过。C04为783.41μs，对原同SDK778.39μs慢0.6449%，
对已采用763.65μs慢2.5876%；C05为1024.24μs，对1014.46μs慢0.9641%。
两档均不采用，不扩大这个负例的状态/服务测试。全部20个样本、原始72角色
记录路径、SDK与检查结果见`csa_spmd_tnd_eight_20261022/uniform_meta_v1_results.json`。

`uniform_meta_on_interface_audit.json`另确认C04/C05的AIC数学体、两个packed
入口签名/调用包装和orchestration与对应父版本相同，只忽略整行pto位置注释；
AIV主体有意改变。这不证明AIV无额外开销，也不能用接口一致推定性能。
当前本会话无待收CPU/NPU任务；正式算子仍保留79fae6db0的一项叶证明优化。

#### 收益预算与证据边界

如果仅用“每个重点档自身降低10%”举例，C04相对原778.39μs需省77.839μs，
已有14.74μs后还需约63.099μs；C05需省101.446μs。这只是量级预算，不把
八档几何平均目标擅自改成逐档10%。若只有两个档各降低10%、其余六档不变，
八档几何平均只降低约2.60%；若仅靠两个档共同达成总目标，二者都要降低
34.39%。因此重点档是首轮筛选入口，后续必须找到能覆盖更多资源区间的措施。

新增离线`analyze_macro_schedule.py`及`macro_schedule_analysis.json`，复用
现有case、当前policy、Score算量模型和同SDK C05阶段诊断；本轮没有新设备
采样。该分析区分有效算量、逻辑GM→L1请求量、设备插桩端点，禁止混为耗时。

同SDK C05三次插桩的“最后完成位置”中位数：QUERY 297.28、SCORE_VECTOR
584.26、TOPK 597.60、ATTN_PLAN 613.72、ATTN_VECTOR 829.42、OB 968.92、
OUT 1000.70μs。对应Score AIC结束离散度33.38μs、Attention AIV16.02μs、
OB49.00μs。阶段并发且各角色起点不同，这些端点不能相减冒充纯阶段计算，
离散度不能相加作为可省时间；已有GM等待计数也不含全部FIFO/FFTS等待。
本轮没有从旧SDK的PipeUtilization值推断当前瓶颈，更没有把同步等待全部
算成同步指令本身的成本。需要区分“前驱还没完成”和“就绪后的交接开销”。

#### 当前可确认的结构问题

1. **任务数量均衡不等于成本均衡。** C04/C05的Score padded-row×step
   最大/平均均为1.0，分别每核22/33步；继续只换Score owner不能减少这些
   等长档的总算量。实际尾部差异还涉及前段到达时间、内存及流水等待。
   C08该算量比为1.3469，混长需要另按实际live行、可见历史和叶数量安排。
2. **B24输出的满块和尾块成本相差很大。** 当前O_A为M128/N256，T144拆成
   M128+M16；同样3个任务可能是3个满块或3个尾块。按有效M×N统计，最忙核
   是全24核平均的2倍；C06为1.7143、C07为1.5。两个M块各遍历同一组权重，
   逻辑权重载入合计128MiB，唯一权重为64MiB；这不是实测HBM流量。
3. **已有按owner等待，但发布粒度仍限制下游。** `wait_indexer_query_roots`
   等真正root writer的整段SCORE_VECTOR完成；`wait_attention_plan_rows`
   在Attention helper开始前等待本组全部query的plan owner；O_A开始前获取
   全部活跃Attention writer。当前不是每阶段都全核barrier，不能靠机械
   删除wait提速。要提前消费，必须同步改变任务完成单位、发布协议和数据寿命。
4. **融合/交错有实际成本。** 旧wave_current_math严格通过但B16
   807.88→850.52μs、B28 1167.66→1369.50μs；其版本与当前不同，只用作
   设计负例。原Score每叶恢复一次Query/Coeff，交错后每step恢复，逻辑额外
   Left操作25.3125/45.9375MiB；B28各wave最大步数之和56高于整段最大41。
   不能重复该细粒度wave方案，也不能由此宣称所有跨阶段流水必然无效。

#### 下一步执行顺序

**第一，建立两档完整资源与依赖成本图，按整条链选择改动。** 复用现有C05
证据；只为缺失的实际task到达/开始/结束、组内FIFO等待和载入量补针对性
诊断，优先C04保留版本与C05。每个阶段同时记录唯一数据量、重复读写、有效/
padding计算和最晚前驱。核间等待、GM等待及纯算术不能从阶段时长互相相减
推导。先估计依赖最长链与Cube/Vector/搬运资源下界，再决定哪种重构有数十
微秒的可验证空间；不直接将MPMD任务时长搬到SPMD模型中。

**第二，优先联合优化O_A→量化→O_B整段，同时考虑分工与权重供给。**
已有M48/M64只切小行块的版本增加权重载入且回退；只把8个O_B任务搬给轻核
的版本也回退，这两种不重测。新的分析要联合决定M/N格子、每核O_A/O_B成本、
量化依赖与组就绪顺序。离线反事实“单M块覆盖T、N128”在T144模型中可将
O_A最大/平均从2.0变成1.125，逻辑权重载入128→64MiB；它尚无完整编译、
片上资源和性能证明，不能预记收益。若M192/Left双缓冲不满足容量，应首先
否决资源方案，不能为纸面均衡牺牲原K流水或放宽内存上限。C04已有的输出
互补分工要保留；这个大T方向可覆盖C05/C06/C07，但单独不足以保证八档10%。

**第三，前段→Score→Attention考虑粗粒度的联合分工，保留完整叶的局部性。**
关注前段谁先释放Score输入、谁同时承担重Q_B/KV，及root就绪后的真实等待。
先评估静态成本分配和就绪依赖，避免只轮换一个KV/Score角色的旧负例；如有
足够收益空间，再研究单kernel内按完整Score叶/完整Attention query交接的
执行策略。它必须保留Score叶内Query/Coeff驻留和完整范围负载均衡，不能再
在每个N面板/step保存恢复两套状态，也不能仅把整段切成更多wave。

单SPMD允许角色执行不同的控制分支，并不要求所有核同时处于同一业务阶段。
但引入设备侧就绪队列会带来控制、所有权、缓冲复用与调度成本，应先用成本
图确认上界，再决定是否值得实现；不能声称“留在单kernel里就没有调度成本”。
如确实需要新的atomic队列，需先验证A3上所用发布/获取协议；用户提供的A5
atomic手册用于约束设计，不能把A5专项实测当作本机A3已证明的内存模型。

**第四，等长专用路径应选择整套执行策略，混长保留成本感知的通用路径。**
本轮只缩短query_request查找已证明没有收益；后续等长证明只有在能支持
具体的任务格子、生产消费归属或数据复用时才重新引入。等长不推出物理页
连续，不免除causal/页保护，不用Host容量冒充本轮真实Q/L。混长的工作分配
依据实际live与visible成本，不能为了静态均分扩大到B×S_max的假计算。
此前已验证但缺同SDK原C08分母的group_order候选保持待定，不拼接跨SDK收益。

以上是方向调整与可复算的结构分析，不是新性能采用。原10%目标仍active，
完整72角色L1与严格逐bit/状态约束不变；没有修改公共SDK或生产算子，没有push。


### 宏观分析后的首个联合候选：单M输出格子、权重复用及O_B互补分工

本轮核对原goal仍active，上一轮已提交a4817147a的负例和结构分析属于进展。
工作树起点干净，公共SDK仍csa-shared-20261022-logical-codegen/validated。
直接重读已定位cannbot的L2 CacheMode卡片，继续区分逻辑载入与实际HBM流量；
索引缺失未伪造检索结果，本轮也未修改任何缓存提示或公共工具链。

从正式79fae6db0冻结`output_tall_v1`，只在已有Host token容量128<T<=192的
资源域使用OA_ROWS=192、OA_COLS=128；实际有效行仍来自device Tensor维度，
所有Q/L、页表与因果逻辑保持原样，不要求等长，不通过T/6猜请求归属。
原M128+尾块/N256的64个O_A任务改为单M192/N128的64个任务。输出元素与
原K256顺序不变，候选初衷是同时避免第二M块重复加载权重、改善满/尾块成本
差异。它不同于旧M48/M64增加任务/权重次数，也不同于双M累加器共享权重版本。

两侧O_A组任务与量化依赖统一使用OA_ROWS。量化数学、32行任务及8行stage2
保留；O_B改用已存在的32个软件rank互补分工，物理仍24个AIC。当前单M/N128
格子给0..15号核各3个O_A任务、其余各2个，对应O_B各2/4个任务。按源码
语义计，每个O_A任务读1MiB BF16权重，每个O_B任务读0.5MiB INT8权重，
各物理核合计均4MiB；这不是实际HBM事务或实测带宽。全部阶段发布/获取、
512B控制槽、量化尾零和原最终归约顺序保持，没有新增跨核同步协议。

34项CPU检查执行实际O_A几何与Cube/Vector输出调度循环，对16个实际T边界
独立验证每个输出行列块唯一覆盖、全部64个O_B任务唯一覆盖、真实O_A写者
全部被量化等待、O_B获取正确量化组；另验证17个容量边界及S1/S5/S6、四种
history策略只改变预期三项常量。量化与O_B数学源保持相同。完整测试日志
`runs/output_tall_v1_ut_final.log`，生成脚本和描述符均保存在既有实验目录。

C05完整CPU编译通过。实际O_A确为M192×N128 FP32 Acc（96KiB），双Left
各M192×K64（合计48KiB），Right使用K64×N128；K256外层stage2、内部递增
K64链仍存在，未被编译器拆回两个M块。AIC总Mat475136B、Left/Right各65536B、
Acc131072B，AIV Vec180224B，均未扩大原容量。packed ABI/根参数/返回合同
与C05父版本相同。详见`output_tall_v1_codegen.json`及完整AllocateMemoryAddr
IR；资源通过和源级覆盖均不代替设备逐bit正确性。

先只提交C05完整十窗任务`task_20261023_013451_23777723578`，卡0，原同SDK
分母1014.46μs，完整72角色首尾计时及eager/graph/post-L1回归不变。任务已
核实running，源码冻结。并行进行C04关闭域完整编译，与已采用C04生成物比对；
只有C05完整性能值得继续，才扩大状态/服务及C06/C07验证。生产未采用，未push。

C04关闭域完整编译已完成，完整PTO C++、AIC/AIV两个packed文件及orchestration
仅去整行pto位置注释后，与已采用生产版本逐字相同；见
`output_tall_v1_C04_off_codegen.json`。没有为这份未改变的C04重复设备计时。


### 联合输出候选终态与诊断：均衡改善，完整收益仍很小

C05任务task_20261023_013451_23777723578 completed/exit0，完整十窗中位
1010.88μs，对同SDK原分母1014.46μs仅少3.58μs（0.3529%）。所有eager/
graph/post-L1输出、Top-K/score、cache/state逐bit及保护检查通过，实际
OA_ROWS=192、OA_COLS=128、OUTPUT_WORK_BALANCE=True；设备生成CPP与完整
CPU审阅版相同。全部窗口见`output_tall_v1_C05_result.json`。样本分布重叠，
本轮暂存候选，不采用、不为这个小变化扩大到状态/服务及其它档位，不称它为
稳定已兑现的优化。正式仍只有C04叶证明这一项新采用。

为解释纸面成本与完整效果的差距，只追加一次C05候选阶段诊断：
task_20261023_014251_264666425693，卡0，completed/exit0，严格检查及3个
诊断窗口通过。`output_tall_v1_C05_phase.json`绑定未插桩CPP与该候选实际
性能任务中的CPP；同SDK旧C05原版阶段数据直接复用，不重测。对照见
`output_tall_v1_C05_phase_comparison.json`，不能用于性能验收或相减合成收益。

- O_A第3/6/7组的角色区间中位数仍约25～26μs，没有出现算量模型暗示的
  成比例下降；它们从本核上一发布点开始，不是无等待的纯矩阵指令计时。
- O_A第7组完成离散度11.12→1.20μs，O_B为49.00→14.78μs，但O_B最后完成
  位置只从968.92→965.50μs。O_B最早完成位置920.20→950.72μs，说明离散度
  缩小也包含原先早完成的核承担更多工作，不能把34μs离散度缩小认领为净省。
- 前段/Score端点亦有约十μs差异，目前没有证据将其归因于输出源改动；两轮
  非配对插桩不能扣除这段差异，替代正式十窗的3.58μs结果。

补全两侧操作数的逻辑搬运模型后，O_A权重128→64MiB的同时，N块数量增加
使激活36→72MiB，合计164→136MiB，只少约17.07%。原模型明确只统计权重，
不能将权重减半扩大为总搬运减半。新完整操作数模型与各核明细见
`macro_schedule_operand_analysis.json`；仍不是实际HBM事务、缓存命中率或
硬件带宽。该证据要求后续同时计算两侧重复读取和最慢核的实际服务成本，
暂不因算量比漂亮继续扫描相邻M/N参数。

诊断分析器新增显式候选引用模式。负向自检发现同一测量目录同时含reference
和candidate两个build，仅“目录中任一CPP相同”不足以绑定候选。已修正为
候选模式必须指定该测量目录中的确切CPP，并先与候选完整CPU生成物核对；
默认原版calibration模式的3窗内容保持一致，错实现对照已确认被拒。首次
自检产生的错误归属文件已标记invalidated并保留在runs；没有用它作结论，
此为本轮分析器问题，不是SDK故障。验证见phase_analyzer_candidate_reference_checks.json。

另提交C04当前保留版本的同SDK阶段诊断task_20261023_014328_266874830877，
卡8。使用output_tall_v1的C04关闭路径，已与763.65μs已采用版本的实际CPP
唯一匹配，因此不是测一个新的B16实现。仅补缺失的当前版本阶段证据，
不采正式性能分母；当前仍pending，保留原handle，不重复提交或干预其它任务。

#### 依赖粒度的结构模型：B16与B24不应套同一个重叠假设

`analyze_score_readiness.py`复用真实组/叶/owner枚举，对两个S6 fixture做
抽象Score-step模型。全部核同时起跑、一步成本为1，并忽略排序、plan、
同步、前段到达与同核Score占用；所以只说明依赖限制，不代表可执行调度
或任何微秒收益。此处q//6只是离线固定fixture身份，未进入生产代码。

C04每核22步，每个query组的最后叶在22步才完成；将整段发布改成逐叶
发布，在该模型中不提前任何组。C05每核33步，但部分组的全部叶在8/9/16/
17等步已结束，现行整段发布把它们推迟到33步。即使只改逐叶发布，Top-K
worker按q%48完成全部query后才发布、Attention又先等本组全部query的plan，
模型里的Attention仍全部等到33步；再只细化Top-K发布也没有改变该结果。
同时具备逐叶、逐query Top-K以及Attention按当前query获取时，模型的首个
query门槛才可到8步。这仍未解决实际核占用、因果cache依赖或流水存活状态。

因此若研究B24提前消费，应联合修改完成单位、下游依赖和粗粒度任务执行，
并保留完整叶的Query/Coeff驻留，不能恢复已失败的逐step波次交错；B16先看
真实前段到达与Score稳态供数。`score_readiness_model.json`保留完整数组与
假设，不把33→8写成性能加速。原八档10%目标active，生产未改变，未push。
