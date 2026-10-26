# Native精度TND：直接发布heads优化（2026-10-26）

## 当前资格

本候选建立在已验收lookahead（79840c776）上；正式基准保留。仍为纯DSL、完整HC-pre→HC-post一个顶层入口、真实TND与64Tensor ABI。单层四组已验收；真实整模型8K/B40已通过，warm192＋测量192共245760个rank/token位置及接受事件与Native一致，0抢占。Forward98.409996ms、cycle113.986288ms。128K/B24也已通过，147456个rank/token位置及接受事件与Native一致，0抢占。两档设备任务均exit0并释放。

## 单层性能与精度

真实L4权重、合成hidden/history，确定性Native，raw FULL graph，12轮×100event和独立9L1窗，不删除样本。先性能至少0.5%且至少10/12成对领先及L1降低，再Native A/B八边界逐位及ABA。控制为lookahead，不能把两步百分比相加。

|配置|lookahead us|直接发布 us|降低|Native精度|
|---|---:|---:|---:|---|
|8K/B32/T192|1635.094|1590.889|2.704%|A/B八边界＋ABA逐bit一致|
|8K/B32反向捕获|1642.737|1599.292|2.645%|A/B八边界＋ABA逐bit一致|
|128K/B16/T96|1398.979|1353.908|3.222%|A/B八边界＋ABA逐bit一致|
|8K/B36混合1～6行/T126|1377.363|1343.273|2.475%|A/B八边界＋ABA逐bit一致|

## 模块归因

独立依赖采集与计时分开，18timing窗全部task/core覆盖，4个dep映射稳定，CPU/JIT Core SHA、编排alpha及metadata门禁通过。TopK→heads512.10→475.68us；heads→输出347.64→339.52us；完整L11546.98→1502.88us。QK AIV包络378.78→414.14us，因为归一化/逆RoPE/发布移入，不能据此认定总体回退。函数跨度包含等待、搬运和计算且可重叠，不相加。

## Native数值合同

所属AIV保留DIV→BF16 rint→FP32→原inverseRoPE两乘加→BF16 rint，直接写原root heads[T,32768]。NoPE来自第一次BF16；RoPE显式8头extract/concat/gather，避免非零偏移alias丢失。读取同lane最终li_transfer；原TopK/cache和O-proj全8192共同量化不变。删除attnOi GM、rope_cs、merge_norm和native_heads任务。root不跨scope重绑Out heads，后续WO-A依赖实际heads写入。无效query显式零，不读RoPE。

CPU独立审查UB147584B，carry左右各32KiB在8头循环全程保留，scratch不覆盖；数据flow qk output(heads)→WO-A input(heads)成立。静态审查不代替以上设备精度。

## 负面候选与限制

H16将独立softmax头分组从8改16，CPU164.25KiB并静态等价，但设备1576.55→1594.08us慢1.11%，故未精度扩测/未合入。其v1仅runner namespace期待错误，保留失败；v2修复后才有有效计时。物理MGATHER/TLOAD否决详见lookahead报告。

最新nalinaly c346573ab源码已同步，其新SDK公开946.46us不与本轮直接作比；固定SDK兼容尝试被compound dim后端限制阻止，未产生新Nal NPU计时。当前仍没有追平nalinaly的受控结论。

## 新增局部优化复测

将fold64所需64列复制从GM往返改为UB TEXTRACT，首轮8K/B32 1609.042→1581.969us，快1.683%，Native A/B八边界及ABA一致；独立反向捕获却1587.185→1601.454us，慢0.899%，0/12领先，L1也回退。因此拒绝合入。后者性能门禁失败未执行Native精度，顶层pass只表示脚本按性能优先策略正常完成。不能将其写成精度验收通过。

## 同输入三方重新计时

本轮同一设备、同一真实第4层权重和合成hidden/history、8K/B32/S6/T192、NZ2、Native SK0、确定性level1/HCCL。三臂均为完整HC-pre→HC-post raw FULL graph，12轮×100回放，不开DFX、不删除样本。Native和我方使用npugraph_ex捕获，nalinaly直接JIT预热后raw capture。nalinaly使用cafcadec96固定SDK兼容封存版，仅删不支持的backoff_ticks=150；不是最新c346573ab的新工具链结果。

| 实现 | min μs | mean μs | median μs | max μs | 对Native精度 |
| --- | ---: | ---: | ---: | ---: | --- |
| Native SK0 | 1251.086 | 1251.967 | 1251.903 | 1252.750 | 基准 |
| nalinaly cafc兼容单SPMD | 1213.203 | 1216.794 | 1216.181 | 1224.889 | 最终输出relative L2=0.332971%；A/B均非逐bit |
| 我方Direct精度TND | 1582.483 | 1587.462 | 1587.778 | 1592.330 | 原输入A和扰动B的八边界全部逐bit，恢复A和guard通过 |

我方本次均值比nalinaly慢30.463%，比Native慢26.797%。历史我方1816us是旧组合实现，当前已经1587us；两次不同批次，不用历史均值相除声称受控加速。这次只验证单层，不将nalinaly输出误差推断为整模型token错误，也不把全部速度差归因于数值误差。

## 下一步优先级

非重叠L1参考区间：当前Direct入口→TopK692.22us、TopK→heads475.68us、heads→结束339.52us；历史同输入nalinaly多task伴随诊断为565.50/280.29/218.97us。后者不是nalinaly单SPMD内部计时，也不是最新SDK计时。attention仍是最大的剩余参考差距。

逐128panel READY＋ACK尝试1597.398→1651.248us慢3.371%，WO-A24worker重分配1577.976→1596.628us慢1.182%，均未精度扩测、不合入。独立READY事件0/4/5/6且无ACK的后续候选1582.214→1588.655us，慢0.407%；L11504.86→1501.36虽略低，但event没有收益，不验精度、不合入。静态模型和生成物通过不能替代性能实测。

继续优化只迁移数据布局、搬运和编排；保留Native sink初态、K512更新、actualK求和树、概率round、逆RoPE前BF16和全8192列共同量化。最新nalinaly的K128局部softmax、量化边界和尾部sink不能直接套入已验收精度版。

## 复现与回退

固定Python3.12.14，PyPTO2dfa7f120、Simpler57e499238、PTOAS0.66、PTO-ISA327cd586、Torch2.10.0、Torch-NPU2.10.0.post2、vLLM0.25.1、CANN9.2.0-beta.2，保持独立SDK不升级。完整身份和二进制SHA在编译收据与各result.json中。

受测SOURCE_ROOT的3399文件manifest SHA为48ee0351ebd554c2d52c93d28ff7aa4be357cf6636a4325d247f64a1f2671539；13个官方静态包metadata SHA为863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602。权重使用DeepSeek-V4-Flash-0731-w8a8。bank、Native黄金、SDK角色沿用上轮lookahead证据，复现应校验真实路径和hash，不只根据目录名称判断版本。

源码封存、manifest、原始JSON/L1/dep与whole审计见[本次证据目录](evidence_tnd_precision/direct_heads_20261026/README.md)。复现依次在新的自有目录恢复封存源码、逐项核对manifest、source_identity和sdk_bind；用compile.sh完成两臂CPU资格，task-submit外780秒、runner内750秒。单层run.sh接收新label、batch、history、明确的1～6行lengths；反向重复加--reverse-capture。每个case顺序执行，先性能再Native A/B八边界、ABA、guard和只读metadata检查。DFX计时与dep单独任务，不同时打开。

整模型使用whole/run.sh 8k，通过后whole/run.sh 128k，16卡、TP1/DP=EP16、DSpark出5验6、FULL、NZ2、确定性开启、EPLB关闭。warm192＋测量192，复用固定Native gold，不重跑Native2。8K40活跃/T240/22000块；128K24提交21活跃、真实T126/物理桶144、24000块、[56,96,176,1]MiB ring/window2048、2GiB预留。每步取16rank最大耗时，20个forward、19个cycle；cycle包含draft/采样/调度和CPU观察，forward仍有Native HCA及FFN。本轮没有新paired whole Native或旧实现臂，不用whole历史差值声称受控加速。

显式选择tnd_precision_tuned；默认和旧combined alias不变，T<96仍走原路径。本轮正式文件只改tuned sparse两个入口及root内部heads传递，保留已有QA126导入，其函数AST与实验QA95相同。保留79840c776作为直接回退点；该点已有lookahead两档token验收。逐层capture/replay和计时期Native计数/JIT不增加均须通过，不能只凭环境变量或token判断真实替换。

## 真实整模型最终验收

| 配置 | 实际入场 | warm192＋测量192 vs Native | Forward均值 ms | Cycle均值 ms | 模型阶段 |
| --- | --- | --- | ---: | ---: | --- |
| 8K/B40 | 40活跃，T240/桶240 | 245760位置token及事件完全一致，零抢占 | 98.410 | 113.986 | 13:02:49～13:11:29，520秒 |
| 128K/B24 | 24提交21活跃，真实T126/桶144 | 147456位置token及事件完全一致，零抢占 | 85.429 | 98.283 | 13:45:01～13:53:07，486秒 |

128K独立复核16rank实际identity、21CSA各T6和T144 capture1/FULL replay、20HCA Native、Native分发与JIT计数不增加；24000块、2GiB余量与原ring配置均成立。此表是相同Native bank/gold的整模型token验收，不宣称与nalinaly原bank逐字节相同，不把单层逐bit扩展成所有隐藏tensor整模型逐bit。单层八边界及整模型逐token是分别验证的范围。

本次正式迁移仅tuned/decode_csa.py与sparse_attention.py；原QA126完整模块AST（除模块文档）与实验QA95相同，正式不复制QA95文件。所有函数格式化后与真实whole受测候选AST一致。CPU路由测试、独立生成物/数值及原始whole审计均完成后才提交。
