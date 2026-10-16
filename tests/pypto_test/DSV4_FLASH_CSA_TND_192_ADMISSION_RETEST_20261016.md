# TND精度版：128K/B24、192 token实际入场对齐复测

日期：2026-10-16。分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
数学源码为`cbc810c8afab47c2425068049440424f014dcde3`，含长档缩批Indexer根发布修复；性能版未改。
复用已冻结的远端诊断overlay、SDK及官方静态包，不重装框架。CSA首次构建了一份kernel，详见下文缓存资格。
远端清单中的516个`vllm_ascend/`生产文件与该commit逐文件SHA一致；诊断harness是已记录的移植版，不能把整个目录称为本地checkout的完整复制。

## 配置

| 项目 | 口径 |
| --- | --- |
| 模型 | DeepSeek-V4-Flash-0731-w8a8，完整75分片权重 |
| 并行与请求 | TP1、DP=EP16，每rank提交24个不同问题，每请求生成192 token |
| 输入 | 冻结fixed131072 bank，实际history131104–131346；与前轮相同 |
| 共同容量 | 固定24000 KV blocks，计划分配上界21,955,846,144 bytes |
| 解码 | greedy、DSpark提出5验6、ignore EOS |
| 图与数值 | FULL_DECODE_ONLY `[6,144]`；event1、确定性level1、HCCL true、atomic0、NZ2、EPLB关闭 |
| 环境 | CANN9.2 beta2、ATB9.2 ABI1、Torch2.10.0、Torch-NPU2.10.0.post2、vLLM0.25.1 |
| PyPTO/Simpler | 沿用64 token回归的冻结私有运行包及工具链，启动前逐文件清单和实际导入路径核验 |
| 任务限制 | 每项独立提交，max-time780秒；Native重复资格通过后才交CSA；首差即停 |

B24指提交24条，允许等待入场。与此前192 token自动KV容量的结果相比，本轮固定共同实际分配，
不能把该配置变化省略。沿用相同bank，但未核验与nalinaly原始完整bank逐字节一致。
本轮只验证精度；观测包含设备到CPU复制和原值落盘，不产出性能结论。

## Native1及观测结束问题

Native1完成384条副本请求、73,728个token，零抢占，实际活跃请求峰值21。
平均接受步长3.8264033264，接受0/1/2/3/4/5次数为1200/1392/864/1216/1472/9248。

初次联合审计拒绝了原始记录轮数：rank2为79轮，其余15个rank为80轮。
定位发现全部16rank都有同一段78轮完整生成窗口，rank0其中73轮FULL144；
差异仅在请求完成后的1/2次无活请求dummy。客户端先等待全部生成完成，再分别发结束观测RPC，
因此关闭观测时机不同。该故障属于观测边界，不能据此归为CSA精度差异。

修复诊断边界时保留了所有原始文件：

- 从全部16rank的最后活跃轮推导全局生成窗口终点，不手写78。
- 窗口内所有idle仍保留，逐轮核对联合DP向量、实际请求、token、position、seq_lens和padding输入。
- 只对终点后的严格空尾单独审计：无实际token、无请求及新入场/完成/抢占事件、query bounds为`[0]`、FULL6、DP向量全6，完整padding输入与本rank原型相同。
- 派生参考使用独立目录，保存原始trace、outer、audit、完整输出及派生文件SHA；不把原始80轮complete改写成原始78轮。
- 新运行在线逐轮核对生成窗口；超出参考后仍调用原模型并保留raw tail，只跳过诊断回调。出现活跃token、新请求或异常padding立即拒绝执行并锁定首差。
- 原版和新版入场helper使用精确SHA修订清单；其余观测探针不变。原有10项CPU回归及新增6项窗口正/负控制通过。

派生Native1参考包含78轮，原始空尾计数为`[2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2]`。
派生清单SHA256：`347b452c4eed32897b54bd05305a317708fbea987ffeab1ad1ac1461f9153dac`。
因此后续“轨迹一致”指完整生成窗口，并不表示完成后的原始空尾数量也相同。

## Native独立重复结果

两次Native的完整生成窗口78轮在全部16rank逐字段一致，73,728个token、完整接受事件及统计也一致。
rank0全部78轮target hidden、aux hidden和MTP residual逐bit一致，说明新增图内观测通过保真验证。
Native2保存43层、其中21层CSA，共16,614份边界摘要和78份有效行原值快照，供CSA对拍。
Native1没有图内stage观测，因此没有声称两次Native的每一个内部tensor都做过比较。

Native2各rank原始记录79轮，其中78轮为生成窗口、1轮为独立审计的完成后空跑。
原始空尾数量可变的事实继续保留。

## 最终结果

| 阶段 | 任务 | 状态 |
| --- | --- | --- |
| Native1 control | `task_20261016_091954_247756916995` | exit0；完整输出与生成窗口资格通过 |
| Native2 witness | `task_20261016_093935_324666725855` | exit0；完整生成输入/token/事件重复及outer保真资格通过 |
| CSA witness | `task_20261016_095631_393947426780` | exit0；数值资格通过；严格READY复用资格未通过 |

| 对拍项目 | 本轮结果 |
| --- | --- |
| Native1 → Native2 | 全部16rank的78轮生成输入、73,728个输出token及接受事件一致；rank0 outer逐bit一致 |
| Native2 → CSA精度版 | 全部16rank的78轮生成输入、73,728个输出token及接受事件一致；无首差、零抢占 |
| rank0模块边界 | 43层、其中21层CSA；78轮共16,614条边界摘要逐bit一致 |
| rank0 outer | target hidden、aux hidden、MTP residual逐bit一致 |
| 替换范围 | 全部16rank核实：21层CSA为TND精度版，20层HCA仍为Native |
| 实际入场 | 每卡峰值21；q00–q20在第0轮入场，q21/q22在第35轮、q23在第36轮入场 |
| 接受步长 | 3.8264033264，0–5各长度均覆盖，三臂完整接受事件一致 |

rank0生成窗口的图形态是1轮NONE/21 token、73轮FULL/144 token、4轮FULL/6 token。
完整窗口、请求身份/顺序、实际token/position、query bounds/seq lens、padding及DP向量均对齐，
没有仅比较最终文本。CSA各rank保留79轮raw，其中78轮生成窗口及1轮严格验证的完成后空尾。

三项任务均设置780秒执行上限。实际执行约6分38秒、8分钟、8分17秒；
Native2和CSA曾排队，CLI的提交至完成汇总时长包含排队，不能解释为占卡时间。
所有本线程任务已结束。本轮没有扩大配置或改动数学实现，性能版保持原样。

## 缓存资格与数值资格分别记录

默认严格资格审计在CSA的READY复用检查失败：所有rank的ready_hits为0，
仅rank9发生一次生成和一次binary编译，用时16.533138189秒；其余rank等待该产物。
全部16rank原始observation中的bypass、invalid、storage error及forced rebuild计数均为0。
官方13份静态包复用检查仍通过；没有重装/编译PyPTO、Simpler或vLLM框架。

保留原始严格失败记录，不将其改写为通过。随后独立CPU审计显式排除“必须已有暖READY命中”这一项，
继续核验全部输入、输出、模块、确定性、实际替换层及源码身份，数值资格通过。
该结果只能用于本轮精度结论，不构成暖缓存复用或性能结论；没有为获取一次cache hit额外交NPU任务。

CPU审计另行重算冻结精度入口的persistent source和工具链指纹，与READY manifest一致，
并核验全部206份产物文件SHA；本任务实际构建目录中的同206份文件与该产物逐一一致。specialization保留实际manifest及ABI记录，未独立重新specialize，
不推测首次cache miss的具体原因。

## 结论范围

本次固定bank、128K/B24提交、实际峰值21、EPLB关闭的192 token整模型精度对照通过。
逐token结果覆盖全部16rank；逐bit浮点边界仅覆盖rank0已记录的位置，不能扩展为全部rank的所有中间tensor。
本次生成窗口未出现同轮请求query长度不等的情况，不能替代另行的不等长TND测试矩阵。
尚未核验与nalinaly原始完整bank逐字节一致，也没有测性能或宣称任意输入均能逐bit一致。

[结构化记录](evidence_tnd_precision/low_acceptance_192_admission_retest_20261016.json)保留逐rank计数、SHA身份和审计结果。
Native小型原始证据包SHA256：`75ccda6a4115052977ebbc5a68034b5d1bc885207dbbf9dd38541bb5cf3fe756`。

最终小型原始证据包包含391份文件，已逐文件SHA校验；大tensor原值保留在227。
证据包SHA256：`5cff9b86f9e5ec2b1777525b457a65d5ee8ca570f128064a87e86147cbeeeae0`。

## 相关记录

- [原始192 token自动容量结果](DSV4_FLASH_CSA_TND_128K_B24_20261015.md)：旧Indexer版本失败，实际入场未对齐。
- [64 token入场对齐及Indexer修复](DSV4_FLASH_CSA_TND_ADMISSION_ALIGNMENT_20261016.md)：修复后完整短窗口与Native一致，未覆盖192 token。
