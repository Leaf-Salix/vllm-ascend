# TND 精度版整模型低接受率验证（2026-10-10）

本轮源码冻结于 `ff2bb7b1690e503ce4a4aa38a984916a12ed0d9d`，显式选择
`deepseek_v4_flash_csa_tnd_precision`。性能包未修改；HCA始终使用Native。

参考用户指定的 nalinaly `low_acceptance_20261001/RESULTS.md`，先做8K/B40，
再做128K/B24。每条输出192 token，TP1/DP=EP16，DSpark提出5验6，greedy、ignoreEOS。
40条不同问题复用共同Native历史前缀，后缀真实prefill；固定完整target/draft cache和state。
旧前缀来源为CANN9.0.1，后缀本次由CANN9.2生成；各臂使用同一冻结状态。
不能把该口径写成完整历史都由CANN9.2重算。

## 判据与次序

1. Native1、Native2分别独立运行，确定性level1、HCCL true、atomic0、NZ2、EPLB关闭。
2. 先核验每worker实际环境、41层选择、静态包安装、FULL捕获和重放、192 token完整性。
3. 检查Native两次逐token和逐请求DSpark proposed/accepted事件序列，不只比较总直方图。
4. Native重复通过后，只替换21个CSA层；20个HCA层继续Native，再比较首个分歧。
5. accepted0…5必须全部覆盖，主批平均接受3…4；未满足时不宣称低接受率测试通过。
6. 首个token或接受事件失配立即停止后续配置，定位同输入阶段。首分歧后的轨迹差异不当作同输入浮点误差。
7. CPU query bounds记录真实请求长度，总token不是graph档位；不强制每请求padding到S6。

整模型通过仍不能推出全部中间tensor逐bit一致；中间逐bit判据由单层阶段测试另行验证。

## 时间与静态编译

每个CPU/NPU任务独立提交，max-time780秒，预留两分钟清理裕量；约五分钟查询一次，
不以持续等待替代小任务拆分。没有创建定时任务。

第一轮Native因1410项静态编译超时，无输出token结果，不计精度失败或通过。
后续先分8个不占卡任务运行官方op_compiler，再通过测试私有桥复用官方包。
当前捕获的完整JSON、CANN/SOC、实际确定性、框架/编译器SHA、输入到成功产物关联均硬核验，
不完整时导出缺失项并立即退出；原安装、barrier、reselect流程保留。

特别记录：op_compiler可能部分失败却返回0。chunk00实际198/200，不能宣称200项全覆盖。
两项CompressorMetadata的const指针兼容错误在独立OPP副本修复，仅增加只读参数const；
补编2/2成功。旧失败两项从旧包覆盖剥离，由新成功包覆盖。未改共享OPP或HCA计算。
首阶段75项由原框架成功编译/安装；合计1485个唯一描述符、10个官方包。
原始包、成功/失败日志、关联清单和header前后SHA均保留。

## 2026-10-10 状态（历史记录）

- 40条固定Native请求状态准备完成，审计PASS。
- 官方离线静态编译、两项兼容补编完成；CPU复用门禁6项通过，独立复核通过。
- Native1重试在安装官方包时退出：测试复制丢失执行位，尚未进入生成。
- 本地改为保留权限的copy2，CPU门禁再次6项通过；远端上传自动审批连续超时，未执行。
- 所有本会话任务均已结束。尚无Native重复一致性或Native/PTO token误差结论。
- 下一步上传权限修复，先完成Native1、Native2并比较，再执行仅CSA替换的精度臂。

## 2026-10-12：恢复运行与固定容量控制

用户批准继续，上传复制权限修复后，Native自动容量重试在约5分钟退出：
严格静态JSON门禁发现1179项未覆盖，生成未开始。
最近差异主要来自不同进程的自动KV页数和零拷贝view偏移，少量为dtype别名。
此结果是测试静态覆盖问题，不能记录为Native精度失配；没有放宽JSON门禁。

只在私有测试入口给8K/B40三臂共同设置`num_gpu_blocks_override=16384`，
使用原生allocator；不修改Native/CSA/HCA算术，不把此容量套用到128K。
全部16个rank确认实际override16384，逻辑容量386893 token，最大并发44.52，
超过本轮40条请求。完整数据见
[容量审计](evidence_tnd_precision/low_acceptance_capacity_20261012.json)。
后续验收还要求零抢占，无OOM；不能仅凭请求参数推断容量足够。

固定容量捕获发现340项未覆盖，硬件任务退出，未生成token。
按150/150/40分成三项CPU任务，均逐项成功；每批均检查唯一n/n记录、
无错误日志、输入JSON精确集合及SHA、非空官方包、编译器与header来源。
340项成功产物已追加到原清单，共13组；旧清单保留，不用工具exit0替代逐项成功。

CPU querybounds观察上限从32扩大到256，覆盖192 token生成的收尾变长边界，
只读取已有CPU metadata，不加入NPU同步；后续三臂共享同一观察版本和源码SHA。
正式精度代码仍冻结于本页列出的ff2bb7b，性能包23文件与备份逐字节一致。

Native固定容量第一轮已重新提交，正在排队；每项max-time780秒，
继续每约5分钟监控。当前仍无Native重复一致性或Native/PTO token误差结论。

## 2026-10-12：官方静态缓存入口与第一轮完成

固定容量后的再次独立启动仍新增385个静态描述符，尚未生成token便退出。
只读追踪确认，Native `_align_memory` 根据真实tensor地址对齐到2 MiB，
因此即使KV页数相同，cache view的storage offset仍会随进程分配地址改变。
逐次扩大完整JSON白名单不能形成稳定的复测入口。

后续三臂共同使用当前Torch-NPU已有的官方缓存分支：
`use_cache_compile=1`，实际CANN版本`9.2.0.beta2`，确定性level1，
同一个私有缓存目录和冻结的13个官方包。测试入口检查包与框架SHA、SOC、
目录所有权、执行位及缓存元数据，保留官方安装、状态广播、重选与barrier。
安装后核验13个包全部登记；直接包裹真实CPP重选调用，仅在其成功返回后记成功。
禁止占卡过程中再执行op_compiler。CPU七项门禁测试和独立复核通过。

该口径证明共同官方包已加载且重选成功，不证明每个算子都命中静态kernel，
也不宣称与此前逐JSON新编译路径逐bit等价。三臂必须共享这一基线，
不把旧入口结果混入新的Native重复或Native/PTO对拍。

8K/B40 Native第一轮已exit0完成，核验通过：

| 项目 | 结果 |
| --- | --- |
| DP rank / 不同问题 | 16 / 40 |
| 合计请求 / 输出token | 640 / 122880（每请求192） |
| 同次跨DP输出token差异 | 0 |
| 同次跨DP接收事件序列差异 | 0 |
| 平均接受draft token | 3.7035909921 |
| 接受长度0/1/2/3/4/5次数 | 1696 / 2432 / 2816 / 2592 / 2240 / 14512 |
| 抢占 | 0 |
| FULL 240-token图重放计数 | 每rank 50 |
| 官方包 / 编译错误 | 13 / 0 |
| 已观察不等长query的rank | 0 / 16 |

这是第一轮运行资格与同次跨DP检查，**不是独立Native重复验证**，
也尚无本轮Native/PTO精度结论。真实CPU query记录未观察到不等长请求，
不把本轮宣称为变长TND整模型覆盖。
独立Native第二轮已提交，保持共同源码、固定状态、静态缓存和运行参数；
通过后才执行CSA精度版，HCA继续Native。

## 2026-10-12：独立Native重复通过

第二个独立Native进程也exit0完成。冻结源码、固定bank、容量参数、静态缓存SHA、
实际worker配置及启动参数核验相同；仅排除通信运行ID和观察输出目录。
两轮各640条请求、122880输出token，逐token差异0，逐请求逐轮接收事件差异0；
接收统计同样一致，全部0…5类别覆盖，平均接受3.7035909921，零抢占。
逐请求证据与运行条件见
[独立Native重复结果](evidence_tnd_precision/low_acceptance_native_repeat_20261012.json)。

两轮的原始rank结果、request_stats JSONL和日志均已另行保存。
第一轮独立复核也确认逐轮事件与Prometheus汇总对账一致。
第一轮每rank观察84或87个query样本，包含收尾请求数减少及graph padding；
有效请求query长度仍全为6，没有不等长请求覆盖。

在此Native基线通过后，已提交TND精度版CSA单独替换任务，HCA仍Native，
复用相同固定状态和官方缓存。任务max-time780秒，按约五分钟间隔监控。
本段只证明Native自身重复稳定，尚无本轮Native/PTO误差结论。

## 2026-10-12：CSA初始化超时与共享JIT缓存修复

旧环境CSA任务触发780秒上限，exit130，未生成任何rank结果。
第一个Graph档位完成，第二个仍在PyPTO orchestration二进制编译等待；
13个官方静态包均已成功安装/重选。此次不是数值失配，精度判定为未测试。

只读核验发现共享缓存已开启，但官方工具链身份检查拒绝
`pypto.pypto_core`的外部导入重定向：Python包来自源码树，原扩展来自editable安装目录。
持久缓存的identity-unusable分支直接build，不检查对象缓存；本轮产生463个构建目录，
451个compiled metadata具有同一ABI，448份binary manifest。
ABI相同不能证明生成代码相同，没有把其中任意产物强行替换到其他编译请求。

修复仅在私有测试目录复制完整PyPTO Python包及原来已编译的两扩展，206个文件逐项SHA一致。
使用限定于pypto命名空间的普通PathFinder，保留Torch/Simpler既有选路；
两扩展仍选中原依赖库路径。原官方工具链身份检查未修改，重新核验usable=true、failures为空。
没有重装或重编译vLLM/PyPTO/Simpler，也未修改原安装目录、性能包或精度算术。

官方无设备kernel artifact接口完成signature-mode预编译及READY发布。
第一次诊断误查不变的GENERATED output_dir而退出；随后改查官方runtime.handle的BINARY_READY状态，
不弱化二进制SHA、ABI与加载检查。独立新CPU进程约3.1秒命中READY，
miss/bypass/generation/binary/storage计数全0。证据见
[运行包身份与READY命中](evidence_tnd_precision/low_acceptance_runtime_package_20261012.json)。
CPU命中仍不能证明NPU真实特殊化全部命中；后续逐rank核查cache_stats，拒绝bypass、
强制重建、身份/存储错误及重复构建，并记录实际模块路径。

这是新的环境基线，三臂必须共用同一运行包manifest SHA和同一观察源码SHA。
旧Native重复PASS保留为历史证据，不与新环境CSA混作主对照。
已提交新环境Native第一轮，之后独立Native第二轮，再仅替换CSA；HCA继续Native。
每项任务仍max-time780秒，继续约五分钟监控。
