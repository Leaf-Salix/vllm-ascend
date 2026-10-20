# Overlap 128K整模型原始证据独立复核

2026-10-20。仅本机CPU读取归档与源码；未提交设备任务、重跑Native或修改正式实现。

## 结论

**128K/B24独立原始载荷复核通过。**
16rank、每rank24请求、warmup及三轮各192 token，**294,912个输出token和61,568个接受事件逐项等于Nativegold，全部零抢占**。
每个rank的gold token文件恰有24个不同请求key，事件文件也恰有24条不同请求记录，未靠字典去重掩盖重复／遗漏。

测量窗口实际每卡21活跃请求、126真实query行、144图行。18行是全图bucket padding；
此场景不等同“整个图零padding”。不等长请求无需补S6的直接验收保留在先行混合T126 FULL单层用例。

## 源码、编译及工作量身份

- 归档SHA256：`8a8f5bbf7fd3dde27630c7ffd1ceadc2b33b567fc07031ad88b8ffa56a3a6cb9`，230个文件载荷，成员名无重复。
- snapshot manifest包含3399文件，SHA为 `b229b29c69dc4aec637f99d586c1b68bafc9cc28cec9ebae2f54483baf1cdd4c`，与本次配置和8K相同。
- 实际归档77份运行Python源逐个重哈，与当前checkout、manifest一致；新六源同时匹配正式源码清单与编译收据。
- 保留父71源再逐项与此前独立验收的QA126 8K原始归档比较，全部字节SHA相同。
- 实际助手载荷逐SHA对helper manifest及运行配置；编译收据保留真实program、kernel、Torch64、prepare及operators门禁。
- TP=1、DP=EP=16；21个CSA实际使用overlap service、20个HCA为Native。逐层两档6/144的warmup、capture及实际选择、64 tensor namespace、静态metadata、唯一reselection及复用资格均核验。
- 缓存generation与binary build前后稳定，bypass、invalid entry、storage error、forced rebuild均为0。
- 配置保持warm192＋三轮192、确定性level1／HCCL开启、EPLB关闭，无计时profile或stage hook。

本机实际重新哈希的是77份载荷及完整manifest；没有把manifest列出3399文件误说成本地读取3399份源码。
权重没有导出到该包，HC权重资格由构造器／contract源码及实际runtime/capture证据覆盖，不声称本机重新计算每层权重资格。

## 精度与相位轨迹

独立脚本从candidate rank JSON与gold mixed／request_stats JSONL读取原始数组，未依赖审计pass字段：

- warmup和全部三轮每条请求输出token、接受事件、零抢占逐项比较。
- 固定C1→Native N1、C2/C3→Native N2，共960组metadata全部字段仅除 `call` 严格一致，**包含完整input IDs**；没有对固定配对输入作忽略、归一化或0/-1替换。
- 另对所有Native参考轮做1920组几何检查，非input IDs字段完全相同；跨阶段观察缓冲1248个0/-1差异单独计数，仅限这项跨阶段记录。
- 每步21请求、126真实行、144图行、FULL模式、16rank DP tokens均一致；不是仅比较名义B24。

此结论证明本协议下输出token和接受事件一致；CPU观察槽证据不能扩展为GPU padding内容或整模型所有中间张量逐bit证明。

## 性能独立复算

每一步先取16rank最大耗时，再跨轮平均。forward为60点，cycle为57点；历史Native仅两测量轮，分别40／38点。下表标准差为总体标准差，单位ms。

| 实现 | Forward均值／总体标准差 | Cycle均值／总体标准差 |
| --- | ---: | ---: |
| 本次Overlap | **90.712908／0.629769** | **103.079798／0.724302** |
| 历史串行QA126父版 | 92.214152／5.071593 | 107.409141／9.665214 |
| 历史Native | 79.729852／0.805280 | 92.292926／0.999937 |

原始微秒先求均值再转ms，得到 `90.71290766398113`／`103.07979824668483`，与原汇总完全相同。
逐点先转ms再平均，forward为 `90.71290766398111`，仅相差1个FP64 ULP；该CPU统计单位转换差异单独记录，不涉及任何token、event或模型张量精度容差。

相对历史父版forward均值下降约1.628%，相对历史Native仍慢约13.775%。
由于来自不同任务，且父版分布有更大离散度，不能把整模型差值全部归因于本次Sinkhorn移动，也不能称已对齐Native／nalinaly速度。

## 最终中文报告与复现材料复核

项目 `DSV4_FLASH_CSA_TND_PRECISION_SINKHORN_OVERLAP_20261020.md` 的两档token/event计数、计时、并发、FULL/static范围和历史比较限制与原始载荷一致。
报告准确区分前缀3%～6%、最终FULL约0.1760%／0.2175%、历史整模型均值变化；没有把单层Native SK0口径冒称为整模型受控Native性能。
说明保留默认路由／71源、候选opt-in、非全链单SPMD、固定bank非nalinaly bank字节证明、未重跑Native2及不覆盖EPLB开启，范围合理。

已复核报告rstd说明改为对冻结PTO控制值，未把它误称Native直接导出。
新packer已按job内 `FINAL_SOURCE_BOUND.single_layer_gate_result_sha256` 读取case并核SHA，复测不再附旧固定case记录。
复现工具补齐实际 `prepack_candidate.py`、`BASELINE.json`、父编译收据及环境引用脚本，README明确旧71源、原delivery fixtures、父快照和SDK/bank/gold依赖。
本机独立审计器依赖原workspace／旧QA126归档，README已与远端正式审计路径区分，未假称独立便携包。

没有发现阻止本轮结果归档的文档或源码身份问题。精度范围限于实际单层FULL边界和这两档192 token整模型协议。

## 独立产物

- `recheck_overlap_whole_raw.py`：逐载荷比较与样本聚合助手。
- `WHOLE_128K_INDEPENDENT_RECHECK.json`：完整PASS、源身份、960配对、跨阶段记录及统计重算。
- `WHOLE_8K_INDEPENDENT_REVIEW.md`／对应JSON：先行8K独立复核。
- `FULL_MICRO_INDEPENDENT_RECHECK.json`：最终单层FULL身份、门禁记录及12×100原始计时重算。
