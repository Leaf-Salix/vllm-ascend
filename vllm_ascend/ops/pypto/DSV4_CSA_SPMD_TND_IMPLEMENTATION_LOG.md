# CSA 单 SPMD 原生 TND 实现记录

本文件单独记录 TND 工作，不改写 `DSV4_CSA_SPMD_OPTIMIZATION_LOG.md`。
后续按实际执行顺序追加改动、命令、任务编号、结果与未完成项；只有原始证据
覆盖相应范围时才标记通过，不把旧 S6 结果当成混长、连续状态或服务接入通过。

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
