# 复现：后缀计算、搬运与同步诊断

## 环境与范围

沿用上一轮独立SDK和精度基准：正式运行源48c58ce7c；原常驻后缀SHA为
`a31a78f89e1ed71c007de2b692434cc0a0f1bc35626b22188418b3e715edd022`。
分支为`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，本轮起始HEAD为93f21dc7f。
参考nalinaly源码固定beaefa6f，未将其不同数值链当作我们的Native金值。

PyPTO 2dfa7f1、Simpler 57e4992、实际PTOAS隔离wheel0.66、Python3.12.14、
Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、CANN9.2beta2。
精确提交、wheel和DSO SHA由原工具链报告和每个设备结果收据核对。
模型为`/data/model/DeepSeek-V4-Flash-0731-w8a8`，真实第4层WO-A/B权重与加载scale；其余输入合成。
不修改SDK、默认路由或旧环境；所有算子候选仍为DSL。

## 文件准备

本轮工具和候选放在自有SDK的`suffix-diagnosis-20261020`，SDK根仍保留原候选及proof、sdk-preflight。
源码/脚本归档均按原样保留，RAW_SHA256逐文件核对；不得重新格式化被测源码后沿用资格。
`prepare*.py`为历史派生脚本，后续审查补强以实际probe文件为准，复现直接使用冻结probe。
特别是早期四项AB任务使用`ab_probe_v1.py`的字节，之后phase1仅扩展CLI choices，使用新`ab_probe.py`；
按各结果的harness_sha256选择正确版本，复制为运行脚本所引用的ab_probe.py再提交。

先检查CPU门禁：

```bash
sdk=/data/pyptouser/yejia/csa-spmd-sdk-20261020
bash "$sdk/suffix-diagnosis-20261020/check-stages.sh"
bash "$sdk/suffix-diagnosis-20261020/check-waits.sh"
bash "$sdk/suffix-diagnosis-20261020/check-candidates.sh"
bash "$sdk/suffix-diagnosis-20261020/check-phase1.sh"
bash "$sdk/suffix-diagnosis-20261020/check-combo.sh"
```

每个CPU脚本写入独立、尚不存在的结果目录；复跑须换case输出名，保留旧证据。
CPU检查包括PTOAS/CCEC及Artifact加载，要求NPU未初始化，不是硬件精度资格。
16参数ABI避免Torch64参数限制。复用ccache，不重新编译vLLM或旧环境。

## 设备命令

各步骤分成独立短任务，进程750秒/调度780秒上限；每5分钟内观察一次并收集最终状态。
每个case名称必须从未使用，首差停止，不越过失败继续扩大。

```bash
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash $sdk/suffix-diagnosis-20261020/run-stages.sh stages-repro-t126 126 stages"
```

完整旧后缀A/B/A和五边界先过，再运行6阶段。工作区由完整旧后缀填充；逐阶段毒化实际写入前T行，
浮点NaN、整数为正确值的逐位反码，防止正确值恰等于固定poison的漏写假通过。
每阶段eager与FULL门禁，随后轮换8个角色12窗×100回放，阶段检查计数应14。
阶段数学节点逐AST证明来自父源，隔离阶段没有跨阶段信号，时间不能相加/相减作为纯同步耗时。

```bash
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash $sdk/suffix-diagnosis-20261020/run-pmu.sh pmu-repro-t126 126 pmu"
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash $sdk/suffix-diagnosis-20261020/run-dfx.sh dfx-repro-t126 126 dfx"
```

PMU与DFX及正式延迟独立任务，不能合并。PMU每角色3回放，PipeUtilization/Level1；
CSV须有三个aicore kernel记录。管线比例可重叠，不是可加总的串行耗时。
部分CANN host-flow关联报错保留；设备CSV与精度检查独立验收，不冒称完整host-flow无告警。
DFX为L1，dep-gen关闭；3窗，每角色每窗1回放，阶段检查计数应5。
核对core_types、实际core集合、6列时戳idx3/4和clock_freq_hz。
run_boundaries每条是一个root round的开始标记；frozen两个root为2条，其余单root为1条，不是起止对。

```bash
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash $sdk/suffix-diagnosis-20261020/run-waits.sh waits-repro-t126 126 stages"
```

wait0/24/48只测前序完整MIX已发布就绪槽的循环和缓存维护；读取当前phase0代际，不能把数据预填未来代际。
全部信号/输出保持不变，三个arm各14次检查；其目标为空，不能称为新增数学精度验收。
wait0保留同样读取代际和获取维护，仅循环次数0。差值仅说明就绪轮询成本，不代表真实producer stall。

```bash
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash $sdk/suffix-diagnosis-20261020/run-ab.sh ab-repro-phase5-t126 126 phase5"
```

分别将variant换为m128、phase1，各自与原resident和冻结PTO同卡三臂，不能叠加因素。
每项五边界、eager/FULL A/B/A、输出毒化、只读SHA、全部工作区/信号guard及双generation通过才计时。
12窗×100回放，rotate+reverse位置平衡；中位数为窗口平均耗时的中位数，不是单次回放p50。
成功后双generation均为1204。m128只在T>=96测试，新integer工作区T+127，其余维持T+63。
通过T126后单独测T240；不稳定收益不作通用保留。短T正式路径不改。

`run-ab-dfx.sh`使用独立L1源，3窗每arm1回放，成功双generation=7；逐arm验证+1/+0，整轮再比较双代际。
它与100连回放正式延迟负载不同，不能用其微秒差替代正式收益或推算精确内部阶段开销。

## 精度与推广边界

所有本轮AB金值是冻结PTO后缀，不是直接Native后缀/完整CSA或整模型。
只有稳定改善完整CSA、且不劣于已验收冻结实现时，才进入8K/B40及128K/B24真实整模型192-token复验。
局部候选即便胜原resident，仍劣于冻结后缀时，不切换默认路由，也不冒称新整模型token验收通过。

## 组合候选的单独门禁

`candidate-combo.py`只组合phase1所需owner与WO-B M128，不包含phase5。
组合CPU预检和Artifact加载通过，不等于继承设备资格。只有phase1 T240通过且收益稳定，才提交组合：

```bash
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash $sdk/suffix-diagnosis-20261020/run-combo.sh combo-repro-t126 126 combo"
```

通过T126再单独提交T240。combo使用独立的`combo_probe.py`，new_integer容量T+127。
三臂是冻结PTO/原resident/组合，检验组合净效果；不是四格factorial，不能由跨任务结果估计交互项或假设收益相加。

## 本轮停止条件及直接增量复现

初次组合T126逐位通过，但不能跨任务排名。随后`combo_incremental_probe.py`实际父源为phase1，
结果键resident代表phase1，不再代表a31；comparison_parent_proof记录真实SHA。
三臂为冻结PTO、phase1、phase1+M128，父integer容量T+63、新T+127，位门禁不变。

```bash
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash $sdk/suffix-diagnosis-20261020/run-combo-incremental.sh incremental-repro-t126 126 combo"
```

正式12窗中增量回退2.4959%，12窗全部更慢，因此不再扩大组合T240，亦不切正式路径或启动整模型复验。
所有本轮设备任务最终exit0，实际探针约80–130秒，调度准备时间另见task收据；排队时间不算占卡。

## 运行时间收据

以下为启动脚本START至END标记的秒数；包含Python导入及探针，不含标记前环境source和排队时间，不能混作纯kernel耗时。
任务均受调度780秒上限约束；原始task/run日志随归档保留。

| case | 启动标记区间秒 | 最终状态 |
| --- | ---: | --- |
| stages-t126-v1 | 129 | exit0 |
| pmu-t126-v1 | 149 | exit0 |
| dfx-t126-v1 | 112 | exit0 |
| waits-t126-v1 | 120 | exit0 |
| ab-phase5-t126-v1 | 115 | exit0 |
| ab-m128-t126-v1 | 107 | exit0 |
| ab-phase5-t240-v1 | 104 | exit0 |
| ab-m128-t240-v1 | 101 | exit0 |
| ab-dfx-phase5-t126-v1 | 96 | exit0 |
| ab-phase1-t126-v1 | 111 | exit0 |
| ab-phase1-t240-v1 | 114 | exit0 |
| ab-dfx-phase1-t126-v1 | 112 | exit0 |
| ab-combo-t126-v1 | 121 | exit0 |
| ab-incremental-t126-v1 | 104 | exit0 |
