# TND精度版：128K/B24、192 token实际入场对齐复测

日期：2026-10-16。分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
数学源码为`cbc810c8afab47c2425068049440424f014dcde3`，含长档缩批Indexer根发布修复；性能版未改。
复用已冻结的远端诊断overlay、SDK、READY及官方静态包，不重装框架。
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

## 当前执行状态

| 阶段 | 任务 | 状态 |
| --- | --- | --- |
| Native1 control | `task_20261016_091954_247756916995` | exit0；完整输出与生成窗口资格通过 |
| Native2 witness | `task_20261016_093935_324666725855` | exit0；完整生成输入/token/事件重复及outer保真资格通过 |
| CSA witness | `task_20261016_095631_393947426780` | Native重复资格通过后提交，等待16卡 |

192 token的Native独立重复及观测保真通过；CSA尚未执行完成，不能宣称替换精度通过。

[结构化记录](evidence_tnd_precision/low_acceptance_192_admission_retest_20261016.json)保留逐rank计数、SHA身份和审计结果。
Native小型原始证据包SHA256：`75ccda6a4115052977ebbc5a68034b5d1bc885207dbbf9dd38541bb5cf3fe756`。

## 相关记录

- [原始192 token自动容量结果](DSV4_FLASH_CSA_TND_128K_B24_20261015.md)：旧Indexer版本失败，实际入场未对齐。
- [64 token入场对齐及Indexer修复](DSV4_FLASH_CSA_TND_ADMISSION_ALIGNMENT_20261016.md)：修复后完整短窗口与Native一致，未覆盖192 token。
