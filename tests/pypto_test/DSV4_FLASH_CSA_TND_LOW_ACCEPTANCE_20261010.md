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

## 当前状态

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
