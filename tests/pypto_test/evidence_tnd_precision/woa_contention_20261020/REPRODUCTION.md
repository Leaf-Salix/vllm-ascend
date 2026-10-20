# WO-A轮询干扰及后缀切点复现

## 固定身份与目录

报告起点ae9d9cc47，正式精度运行源48c58ce7c保留。测试只在227自有独立目录执行：
`/data/pyptouser/yejia/csa-spmd-sdk-20261020/woa-contention-20261020`。
SDK为同一隔离PyPTO2dfa7f1/Simpler57e4992/PTO-ISA327cd58/实际PTOAS wheel0.66。
原venv、系统CANN和正式源码均不改；路径/版本必须与各result.sdk_preflight一致。

解包reproduction-tools.tar.gz：tools含所有精确脚本、候选及父版；reference含旧a31与来源证明、SDK预检、环境身份。
raw-results.tar.gz包含所有原始JSON及源SHA、RUNS清单、离线复核。RAW_SHA256保存原始字节，勿格式化raw。
候选源为DSL；归档C++是编译器生成物或nalinaly只读参考，没有安装手写C++ helper。

## 本机离线复核

将tools/analyze.py、analyze_instruments.py及verify_prefix_codegen.py置于解包raw同级。
仪器原始包woa-contention-instruments-v1.tar.gz放同级；解包CPU生成码到generated对应目录，路径按包内保留。
安全解包前拒绝绝对路径、..、链接及特殊文件；只接收普通文件/目录。
推荐workspace Python3.11。执行：

```bash
python3 analyze.py
python3 analyze_instruments.py
python3 verify_prefix_codegen.py
```

检查8项结果complete/pass/readonly/guard均true；711项逐位：前缀/仪器231，完整后缀480。
纯前缀计时generation三臂1216，仪器三臂7；完整后缀parent/new均1204。
L1共18窗，每窗72条、24AIC/48AIV、50MHz、一个开始marker、无dropped；PMU9份CSV，每份一个目标MIX设备行。
不要把不同任务时间相减为纯等待，不要把MAC/MTE1/MTE2相加。

## 环境与CPU前置

每个启动脚本从既有独立env.sh source冻结SOURCE_ROOT和CANN/ATB，再指定SDK Python、PTOAS_ROOT、
PYTHONNOUSERSITE=1与自己的缓存/日志。sdk-preflight确认导入、DSO与编译器身份，发现混入先停。
ccache显式SDK/sources/pypto/.ccache，构建并行2；不重装vLLM/SDK，不下载新依赖。
模型为`/data/model/DeepSeek-V4-Flash-0731-w8a8`，make_layer仅加载第4层真实权重，其余输入合成。

| CPU检查 | 内容 | 设备资格 |
| --- | --- | --- |
| check-prefix.sh，cpu-v1 | idle/wait8/ascending24完整PTOAS/CCEC/ABI/artifact.load | idle/wait8采用；原ascending24只归档 |
| check-prefix-v2.sh，cpu-v2 | wait24aligned，首8地址/顺序同wait8 | 采用 |
| check-phasecut.sh，phasecut-cpu-v1 | 两段初稿缺原payload acquire | 即使CPU成功也不占卡 |
| check-phasecut-v2.sh，cpu-v2 | acquire版，但checker注册名含连字符 | checker失败不占卡 |
| check-phasecut-v3.sh，cpu-v3 | 同acquire源码、checker名转下划线，完整编译 | 采用 |
| check-quantcut.sh，cpu-v1 | 初稿中段双AIV，实际只注册空闲入口 | 设备前阻断，不占卡 |
| check-quantcut-v2.sh，cpu-v2 | 单vector_quant、显式48flat AIV | 采用 |

所有输出目录已存在则不覆盖。复测CPU应将checker和输出另取新名，保存新源码SHA；不能覆盖旧回执。
CPU代码生成不是设备精度或性能通过；实际kernel_config、orchestration、资源映射和业务kernel都要审查。

## NPU运行矩阵

只用task-submit，每个case用新名称。完整原命令、task ID和实际harness SHA在RUNS.json。
每任务外780秒、内750秒；提交后每次不超过5分钟检查状态/日志，首差停止定位，不开自动化。
原8项任务全部结束，复现应新提交，不等待旧task再次启动。

| 任务类 | launcher | 实际harness | 形状与直接父版 |
| --- | --- | --- | --- |
| prefix timing | run-prefix.sh，mode=timing | prefix_probe.py | T126/T240；idle/wait8/wait24aligned；gold完整phase1 FP32 accum |
| prefix DFX | run-prefix.sh，mode=dfx | prefix_probe.py | T126/T240；独立单回放采样 |
| prefix PMU | run-prefix-pmu.sh，mode=pmu | prefix_probe_pmu.py | T126；三臂各3次、21逐位指标 |
| WO-A后切点 | run-phasecut-acquire.sh | phasecut_acquire_probe.py | T126/T240；frozen/owner2/phasecut-acquire |
| 量化后增量切点 | run-quantcut-aiv.sh | quantcut_aiv_probe.py | T126；frozen/phasecut-acquire/quantcut-aiv |

示例（换独立case_name）：

```bash
ssh pto227 'task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 --env PYTHONNOUSERSITE=1 "bash /data/pyptouser/yejia/csa-spmd-sdk-20261020/woa-contention-20261020/run-prefix.sh woa-repro-prefix-t126-r2 126 timing"'
```

祖先链均最终到phase1 f22d7d8d及a31来源proof。harness检查实际冻结模块、父源码和候选SHA，独立workspace/信号。
纯等待只测FP32 accum，不把未执行的其它阶段称为通过；完整后缀才测五个业务边界。
A/B同输入：heads改变后重算gold，恢复A再逐位，毒化有效输出；真实T之外padding和所有guard不得改写。

## 推广条件

两段候选T126局部快、T240回退；三段候选T126仅0.38%且9/12窗，仍比冻结慢7.56%。
本轮不推广、不扩大128K、不新增整模型资格。不能用弱父版或跨任务绝对数漂移夸大净收益。
只有候选稳定胜冻结后缀，完整CSA精度/性能过关，才做8K/B40→128K/B24的192-token Native一致性复验。

## nalinaly参考

固定Git提交beaefa6f，NALINALY_WAIT_REFERENCE.json列出原路径和SHA。
其wait_codegen只有显式回调才插入100 syscnt tick退避；归档代码生成audit声明选定reference启用且device_validated=false，不是本轮设备启用证明，不能推广到图片测点。
本轮未采用该C++变换、未扩展SDK，保持纯DSL。不可把100 tick写成100微秒。
