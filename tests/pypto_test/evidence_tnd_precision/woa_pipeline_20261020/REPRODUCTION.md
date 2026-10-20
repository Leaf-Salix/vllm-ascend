# WO-A搬运/流水与owner增量对拍复现

## 固定身份和范围

正式源48c58ce7c、文档起点a3b9d0dd8保留。本轮是独立后缀，不接正式CSA；金值为冻结PTO数学。
脚本中的绝对路径用于227自有环境；迁移时须同时重建环境与来源，不能只改一条PYTHONPATH。
运行前核对sdk_preflight打印的PyPTO/Simpler/PTOAS/ISA/DSO，任一不符先停。
完整SHA及实际源来源见raw中的proof、SDK_SOURCES、result.sdk_preflight。

## 恢复材料

1. 解包reproduction-tools.tar.gz到独立目录；tools保留所有最终及历史probe，reference含旧a31候选和proof，nalinaly-reference为固定beaefa6f源码。
2. parent-phase1.py/json来自前轮；各candidate-*.json给出直接父版与完整SHA。不要格式化被测源码或修改raw字段。
3. `raw-results.tar.gz`含RUNS.json、AB_VALIDATION、L1_DEVICE_VALIDATION、PMU_DEVICE_VALIDATION和原始result。
4. `woa-pipeline-run-receipts-v2.tar.gz`含16项task最终状态、task/run日志、CPU v1..v11结果和日志。
5. 四个CPU generated归档含实际PTO、C++及report，不含SDK或二进制。

本机离线核验：把tools中的analyze_results.py、validate_l1.py、validate_pmu.py放到解包raw所在目录，
并把L1/PMU两个原始归档放在同级，然后执行：

```bash
python3 analyze_results.py
python3 validate_l1.py
python3 validate_pmu.py
```

校验RAW_SHA256.json与SHA256.json。PMU任务应保持exit1；只独立确认其9份原始计数。

## 227精确环境

自有SDK为`/data/pyptouser/yejia/csa-spmd-sdk-20261020`，本轮脚本位于其`woa-pipeline-20261020`。
解释器为SDK/sources/pypto/.venv/bin/python，SDKFinder设置PyPTO/Simpler精确源。
PTOAS_ROOT为SDK/toolchain-wheel实际0.66；不要用调度的0.63标签代替实际编译器。
SOURCE_ROOT由run脚本source已有独立env.sh设置，指向冻结源码snapshot。
模型路径`/data/model/DeepSeek-V4-Flash-0731-w8a8`，只加载目标层真实权重。
每次有自己的Ascend缓存/日志目录，PYTHONNOUSERSITE=1，确定性level1与HCCL true。
不重装vLLM/SDK，不修改原venv，不使用其它用户目录。

## 无设备编译资格

使用tools/check-candidates-v5.sh（nopipe、k128narrow）、v7（matbase、tailpreload）、
v8（owner2）、v9（tailboth、tailowner2、k128owner2）、v11（m128range、m256range）。
脚本清空可见卡并确认torch_npu未初始化，运行完整PTOAS/CCEC、ABI和artifact.load；每份结果存在独立输出目录。
已经存在的输出不覆盖；复测需给脚本和输出另取明确新名，并重新保存SHA。

v1..v4、v6及v10有历史失败，相关源码快照和日志仅用于解释，不具备上卡资格。
实际生成布局审查见independent-reviews归档，不能只凭specialize或skip_ptoas算通过。

## NPU任务矩阵与脚本选择

只允许task-submit。完整原命令、task ID、tokens、parent、mode在RUNS.json，按该文件复现，勿从文件名猜父版。

| mode | 运行脚本 | Python harness | 适用候选 |
| --- | --- | --- | --- |
| ab | run-ab.sh | ab_probe.py | nopipe、k128narrow；父phase1 |
| v2 | run-v2.sh | ab_probe_v2.py | matbase/phase1、tailpreload/matbase |
| owner2 | run-owner2.sh | ab_probe_owner2.py | owner2/phase1 |
| v3 | run-v3.sh | ab_probe_v3.py | tailboth/tailpreload；tailowner2或k128owner2/owner2 |
| m256range | run-m256range.sh | ab_probe_m256range.py | m256range/m128range，T240；两臂均K128单缓冲 |
| dfx成功v3 | run-dfx.sh | ab_probe_dfx.py | k128narrow/phase1，T240 |
| pmu历史失败 | run-pmu.sh | 当时ab_probe_pmu_v1.py | k128narrow/phase1，T240；保留原footer失败，不用于正式延迟 |

DFX v1/v2失败原脚本分别为ab_probe_dfx_v1.py/v2.py；当前dfx是修正后的第三版。
当前ab_probe_pmu.py已修复footer，但本轮没有使用它重跑；不能用当前源码SHA冒充失败任务的被测版本。

示例（必须换新的case_name，禁止覆盖）：

```bash
ssh pto227 'task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 --env PYTHONNOUSERSITE=1 "bash /data/pyptouser/yejia/csa-spmd-sdk-20261020/woa-pipeline-20261020/run-v3.sh woa-repro-k128owner2-t240-r2 240 k128owner2 owner2"'
```

提交后每次不超过5分钟检查task状态及最新日志；首差停止定位；自身进程超时750秒/任务780秒，始终小于15分钟。
本轮任务已全部结束，复测时不等待旧task再次运行。

## 资格与结果检查

三臂冻结/parent/new，检查实际proof和输入SHA，不能把不同parent名合并统计。
先五边界eager/FULL A/B/A、毒化/guard/独立地址/只读SHA；通过后12窗×100无插桩回放。
正式AB应160项bit_equal、final_generation双1204、guards/readonly/complete/pass为true。
低T尾heads名义K240改为一次K256搬运，不改变15次K16数学；K128只改aligned高T快路径。
量化owner修改覆盖T1..256依赖，但硬件只代表矩阵中已测点，不扩大资格。

DFX第三版为三窗/arm单回放，generation双7，70项bit_equal。L1关dep-gen。
PMU单独采样，不能把其计数或L1跨度与正式12×100直接相减为等待开销。
不删离群值，不把跨任务父基线漂移当候选优势，不相加单因素百分比。

仅当对冻结后缀有稳定净收益、完整CSA精度/性能门禁通过，才做8K/B40后128K/B24真实整模型token复验。
本轮没有达到该条件，未启动整模型。
