# Overlap 8K整模型原始证据独立复核

2026-10-20。仅本机CPU读取原始文本归档和当前源码；未提交NPU任务、重跑Native或修改实现。

## 结论

**8K/B40原始证据复核通过，可以进入128K下一档验证。**

直接逐项比较16个rank、每rank40请求、warmup及三轮各192 token：
**491,520个输出token、105,152个接受事件与Nativegold完全一致，全部请求零抢占。**
21个CSA层实际使用overlap候选，20个HCA层保留Native；实际64 tensor根和FULL静态capture资格符合记录。

128K尚未由本报告验收。整模型输出token/event一致，不等同所有内部张量或logits逐bit相同，也不能扩写为任意请求或环境的精度保证。

## 归档与源码身份

| 项目 | 独立核验 |
| --- | --- |
| 8K归档SHA256 | `d9bb1f8e57adcd821e25e854e8e8abda8efd2abdb8954c1239a96fbed1989f4b` |
| 实际文本载荷 | 230个，成员名无重复 |
| snapshot manifest | 3399文件，原始SHA为 `b229b29c69dc4aec637f99d586c1b68bafc9cc28cec9ebae2f54483baf1cdd4c`，与配置一致 |
| 实际运行源 | 77个归档Python载荷逐个重哈，均匹配manifest及当前正式checkout |
| 保留父源 | 旧71源再与先前独立验收的QA126 8K原始归档逐个SHA比较，全部一致 |
| 新六源 | 与 `FORMAL_CANDIDATE_SOURCES.json`、正式编译收据逐项一致 |
| helper | 归档实际助手逐SHA匹配helper manifest及配置中的助手身份 |

编译收据中的program、kernel-mode、Torch64注册、实际prepare program/kernel及正式operators门禁均完整。
prepare callable来源明确为实际注册使用的父prepack模块，parent_commit为7371，未把旧身份重标为新数学。

本地归档只包含77个实际运行源及3399文件的完整manifest，未声称本机重新读取了3399份源码载荷。
服务端完整source重哈记录是另一项证据；本地独立核对的是manifest身份及77份实际载荷。

## 原始数值与工作量门禁

- 不读取 `audit-8k.json` 的pass来代替判断。直接读取各rank保存的warmup、三轮requests，以及Nativegold的mixed输出token和request_stats事件JSONL。
- 所有请求key、192 token长度、接受事件及零抢占逐项核对。没有只比较接受总数或最终文本。
- 实际TP=1、DP=EP=16；每worker一份实现记录，21个CSA为overlap service、20个HCA为Native。
- level1/HCCL确定性、EPLB关闭、无profile/stage hook、warm192＋三轮192配置保持。
- 16rank×3轮×20步，共960组固定Native phase配对的metadata，全部字段仅除 `call` 严格相同，包含input IDs。
- 进一步对三轮Native参考分别比较，共2880组几何严格相同，跨phase input IDs差异为0。
- 每步40活跃请求、240真实行、240图行、FULL、16rank DP工作量一致。
- 每层两档6/240 warmup及capture、实际PTO选择、静态metadata SHA、64 tensor namespace、唯一reselection和复用资格均核对。
- 缓存generation/binary-build稳定，bypass、invalid、storage error、forced rebuild为0；没有静默Native回退记录。

权重资格覆盖构造器／contract实际源码和21层runtime/capture证据；归档没有导出模型权重，本报告没有重新在CPU评估每层权重数值资格。

## 性能独立重算

每一步先取16rank最大值，再在所有轮次求平均。forward为60点，cycle为57点；下表标准差为总体标准差，单位ms。

| 实现 | Forward均值／总体标准差 | Cycle均值／总体标准差 |
| --- | ---: | ---: |
| 本次Overlap | **106.466899／0.587324** | **122.047167／0.503248** |
| 历史串行QA126父版 | 107.090207／2.616661 | 123.770945／5.357994 |
| 历史Native | 89.482168／1.716194 | 105.574509／3.310200 |

原始微秒数组先平均再换算、逐点换算ms后平均两种方式均与记录值完全相等：
forward `106.46689949035644 ms`，cycle `122.04716732627467 ms`。

本次比历史父版均值略小，但父版与本次属于不同任务，不是受控同时同卡AB；不能仅由这一次历史对照宣称稳定整模型收益。
本次forward仍比历史Native约高18.98%。模型总差异不能单独归因于Sinkhorn阶段。

## 最终六源FULL微基准包交叉核验

`micro-full-evidence.tar.gz` 原始SHA为 `6cf8a17c1e7aded970128bdd069e6873f696afcbd45c6081f20e1ca9b5e0682c`，20份文本载荷。
其中实际完整编译收据与本模型包的收据字节相同，最终FULL结果也与本模型包的single-layer载荷字节相同。
重新核对记录中的逐bit指标、A/B/A、guards、packet只读和80项候选跨bucket检查，未发现失败或来源不符。
这是对保存的数值门禁记录进行独立检查，没有声称从未保存的大tensor重新执行逐bit比较。

| 最终FULL任务 | Native均值μs | 串行均值μs | Overlap均值μs | 对串行平均改善 | 配对窗口更快 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full-t126-v3 | 1090.4975 | 1584.1993 | 1581.4107 | 0.1760% | 11/12 |
| full-t240-v2 | 1378.8172 | 2211.6999 | 2206.8895 | 0.2175% | 12/12 |

两项均为12窗口×100次原始FULL回放，未删除离群值；实际八边界A/B共32项记录及候选80项跨bucket检查均通过。
这些最终完整CSA收益显著小于prefix微基准的3%～6%，不能混用或沿用早期8窗口结果。
单次12窗结果也不是重复多轮稳定性证明，且两项完整CSA均慢于同组Native。

## 独立产物

- `recheck_overlap_whole_raw.py`：逐原始rank/gold/source审计脚本。
- `WHOLE_8K_INDEPENDENT_RECHECK.json`：完整重算数据。
- `FULL_MICRO_INDEPENDENT_RECHECK.json`：最终FULL两项来源、门禁与原始窗口重算。

以上脚本和记录只服务本轮证据复核，没有改动正式运行器或放宽其门禁。
