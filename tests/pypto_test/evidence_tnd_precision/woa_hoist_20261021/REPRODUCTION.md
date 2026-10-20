# 纯DSL独立计算与依赖等待对拍复现

## 固定范围

正式精度源为48c58ce7c467dfce63d5cb5c41afd225488d6012；文档起点8fe011517。
候选不替换C++、不改SDK、未写入正式算子路由；Math来自已冻结的PTO O-proj与HC-post。
这里仅测单层真实第4层权重的后缀（WO-A、BF16边界、全8192动态量化、WO-B、反量化、HC-post）。
输入heads/residual/gates为固定seed合成值；不是整模型192-token精度或端到端性能，也不是完整非等长请求metadata覆盖。

## 候选与单因素父链

1. channelpreload：原owner2的通道scale载入与16×512系数展开移到phase4等待前，并复用于该worker后续输出块。
2. obowners：独立从owner2派生，将WO-B反量化提前到所有无关生产者完成之前；每16×512块只等待其4个N128列生产者，每块保留payload acquire。
3. postowners：独立从owner2派生，将HC-post的4token块提前到所有无关反量化生产者完成之前；每块只等待其8个512列生产者，每块保留payload acquire。

前3个候选均单独对拍owner2与冻结后缀，不把三种修改叠加，不将生成物只读审查误写为手工C++实现。
第4个候选coefearly以obowners为父版：先等覆盖有效scale行的两个phase3槽并获取payload，提前完成原反量化系数计算，再等phase4所需4个WO-B槽并获取integer；尾部第二scale槽可能为无业务的发布者，不称两个槽都必要。仅移动原系数计算并补充其就绪依赖，不含channelpreload或postowners。

各候选JSON保存源SHA、逆变换AST一致性及索引/生产者覆盖证明。

## 环境与无设备检查

使用已保留的csa-spmd-sdk-20261020隔离环境与toolchain-wheel PTOAS0.66，调度器选择参数--ptoas0.63不代表实际编译器版本。
PyPTO2dfa7f12、Simpler57e49923、PTO-ISA327cd586；CANN9.2beta2；具体导入、版本、路径在原始result的sdk_preflight中。
本目录check.sh、check-obowners.sh、check-postowners.sh、check-coefearly.sh各执行sdk-preflight、完整PTOAS/CCEC、ABI16与KernelArtifact.load，并断言未初始化NPU。
生成产物的kernel_config、PTO、原生生成C++、prebuilt及binary_manifest保存原样供审查。
所有编译源均为DSL，经原工具链自动生成C++，不进行生成后替换。

## 设备执行

各run脚本显式source原环境，设置任务卡号、PYTHONNOUSERSITE=1、HCCL_DETERMINISTIC=true、PTOAS_ROOT、源码目录及独立Ascend/vLLM缓存。
Python先设置确定性算法再set_deterministic_level(1)。
模型权重从/data/model/DeepSeek-V4-Flash-0731-w8a8读取；WO-A BF16 NZ、WO-B INT8 NZ，检查量化offset为0。
任务完整命令和task id在RUNS.json中；必须通过task-submit分配一张卡，内层750秒、外层780秒。
勿在登录shell裸跑NPU。每项任务均独立短测，首差保存并立即失败，不继续大配置。

每任务T126、T144或T240，三臂独立workspace；先eager，再FULL A/B/A输入扰动与恢复，五边界projected/quantized/token_scale/attended/output逐字节比较。
NaN/整数反码毒化、外层guard、七类只读SHA、最终代际1204检查均保留。
然后12窗×100次同stream event计时，轮转三臂与反转顺序，不采PMU/泳道、不删离群值；每窗后复查五边界。
每个成功任务共160个逐位指标，与冻结后缀为0差异才通过；这不能替代新整模型Native token资格。

下载原始JSON后执行analyze.py重算中位数、对父版/冻结比值和12窗方向。
只有稳定快于冻结后缀后，才接完整CSA，并按8K/B40→128K/B24的顺序执行真实整模型token验收。
本轮未通过此门槛的候选不合入正式路由，不为其扩大整模型。
