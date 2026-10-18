# 8K单层FULL图：BSH准入与TND对拍（2026-10-18）

## 当前结论与停止位置

本轮按“首个bit差异即停止矩阵和计时”执行。在首项全S6、B4/T24中，
当前单入口纯DSL TND完整精度版与Native的8类输出/cache边界全部逐bit一致；
随后BSH归档精度版与Native不一致，因此未继续短query、混合长度NPU矩阵或性能测量。
不能将本轮写成“不等长FULL图矩阵全部通过”，也不能给出减少padding的实测加速比。

BSH原入口在CPU结构测试中对短query、混合长度返回不符合资格，外层dispatcher回退Native。
它不会自动为每条请求补到S6。NPU全S6一项确认BSH实际准入并执行PTO，未回退。
CPU结构测试与NPU执行证据分开报告。

## 源码与环境

- 工作分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
- TND数学：HEAD `2fc05d81e2ff4f425223ab97eccf75bf19c56b60`所含完整精度包，
  数学实现提交`a20bcdb76`；冻结运行树manifest为
  `7b0650b926a557241b622eba4f032731e7b21d75d807c3b6baccfe689b29c960`。
- BSH：nalinaly `84dc9a3f7ccab364843940be0ac43986df2950bc`的完整
  `vllm_ascend/ops/pypto`子树，含精度包依赖的性能包公共函数。
  其中22份精度源码与nalinaly `a919f11942d93627441c8fa694e310dcbd1c671b`
  的归档目录逐字节一致。BSH通过独立Python包路径加载，没有覆盖当前TND包。
- 本轮比较的是归档BSH精度实现，不是nalinaly最新持续维护的性能算术。
- Python3.12、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、
  vLLM-Ascend0.25.1rc1 overlay、CANN/ATB9.2.0-beta.2、PTOAS0.66；
  原私有PyPTO运行包及Simpler路径与前轮相同，没有重装或修改SDK。
- 启动无设备预检核验3362份源码、测试助手SHA、PyPTO/Simpler精确路径及工具链。
  原静态桥仍逐项校验13包、SDK、SOC、确定性及安装身份；实际reselect一次，后续验证复用两次。

## 测试设计

真实模型`DeepSeek-V4-Flash-0731-w8a8`的L4注意力半层，包含HC-pre至HC-post完整CSA；
不加载MoE，不代表整模型/DP16验证。历史位置8191，hidden及历史cache为固定种子1024的合成数据。
Native/TND/BSH使用逐bit相同输入和初始cache、相互独立的cache分配。

Native和PTO均开启Torch确定性、level1和HCCL确定性，NZ2，静态包开启。
使用`torch.compile(backend="npugraph_ex", fullgraph=True, dynamic=False)`，
`force_eager=False`、`static_kernel_compile=True`、`super_kernel_optimize=False`。
实际捕获且获取完整ACLGraph，要求无Host updated nodes，直接raw replay核验。
本轮是SK关闭的单层图口径，不能和历史SK开启的图表直接混合。

图容量按本例真实T设置，不额外逐请求补S6；这用于验证kernel布局，
不模拟整模型默认graph桶选择。kernel内部tile物理对齐与逐请求补S6是不同概念。

逐arm检查：

1. 通过原Native builder生成五组metadata，保存实际query bounds、请求数及max query。
2. 分别完成暖身、FULL捕获；比较compiled调用与raw replay。
3. output先填NaN、TopK填`-12345`，raw replay必须恢复有效结果，排除陈旧hook引用。
4. 比较最终输出、TopK，以及SWA、compressed、state、indexer key/scale、indexer state。
5. 检查全部cache的允许slot写区、首尾保护区和metadata只读性。
6. 对已通过arm执行同地址hidden A/B/A，确认图实际读取新输入并能恢复结果。
7. 任一对Native bit差异则保存原值并停止后续arm/配置和计时。
8. 原计划只有全部精度通过后，才交错次序测6组、每组10次raw replay；restore在计时外，
   计时后再核验输出/state及保护区。本轮没有进入该阶段。

## BSH入口：结构测试

在CPU上提取并执行固定提交的原`eligible`方法及原dispatcher函数AST。
只用结构桩替代tensor属性与Native/PTO调用，未模拟任何NPU算术。
dispatcher仅移除函数内import及类型注解，分支逻辑不改。

| 请求query长度 | 真实T | 假设逐请求S6所需行数 | BSH准入 | 原dispatcher路径 | NPU是否执行本例 |
| --- | ---: | ---: | --- | --- | --- |
| 6、6、6、6 | 24 | 24 | 是 | BSH PTO | 是 |
| 1、1、1、1 | 4 | 24 | 否 | Native回退 | 否，首差停止 |
| 3、3、3、3 | 12 | 24 | 否 | Native回退 | 否，首差停止 |
| 1、2、3、6 | 12 | 24 | 否 | Native回退 | 否，首差停止 |
| 1、2、3、4、5、6 | 21 | 36 | 否 | Native回退 | 否，首差停止 |

T能被6整除并不足以准入：入口同时检查`num_actual_tokens`、`num_decodes`、
最大query长度、请求边界等条件。表中的S6行数只是理论对照，不是本轮真的构造了补位BSH输入。
未强行绕过入口，未把Native回退包装成BSH kernel的支持或性能。

## 全S6 FULL图结果

实际query bounds为`[0,6,12,18,24]`，三方真实T/图容量均24；请求补位和图桶补位均为0。

| 实现 | 真实PTO准入 | 最终输出不同元素/393216 | 输出relative L2 | 八类输出/cache边界逐bit一致 | 保护区/metadata | 性能 |
| --- | --- | ---: | ---: | --- | --- | --- |
| Native | 不适用 | 基线 | 基线 | 基线 | 通过 | 未计时 |
| TND完整精度版 | 是 | 0 | 0 | 8/8 | 通过 | 未计时 |
| BSH归档精度版 | 是 | 130095 | 0.0027760288 | 2/8 | 通过 | 首差后停止 |

BSH输出max_abs为0.0234375。两类完全一致的是indexer key与scale；
最终输出、TopK、SWA/raw KV、compressed KV、main state和indexer state有差异。
不能将TopK索引整数的relative L2当作精度指标：原始快照另行按集合检查，
24行中4行选点集合相同，其余20行也存在集合差别，不只是返回顺序差别。

Native/TND的compiled/raw、TopK毒化写回与输入ABA均通过。
BSH在raw replay及保护检查后立即首差退出，因此没有继续其输入ABA，也没有计时。
当前证据不能把所有BSH输出误差归因于一个模块，更不能归因于padding：本例根本没有补位。

## 首差定位

复用失败快照的同一份L4/T24 hidden，按AST逐语句提取BSH根中从`t_dim`到
`wb_blocks`之前的HC前缀。运算、常量、原scope均保留，仅将post、comb、normed和inv_rms
的内部缓冲改为显式Out，以便观测；没有修改或替换原BSH根算子。
原生gold直接调用实际安装的`npu_hc_pre_v2`及对应无residual/bias的`npu_rms_norm`。
该诊断使用真实手动ACLGraph replay，输出先填NaN；不冒充额外的整层FULL或性能验证。

| 前缀输出 | 不同元素 | max_abs | 结论 |
| --- | ---: | ---: | --- |
| HC-pre post门控 | 72/96 | 0.0000191182 | 已不一致 |
| HC-pre comb门控 | 383/384 | 0.00000721216 | 已不一致 |
| 输入Norm后BF16 hidden | 215/98304 | 0.001953125 | 进入Q/KV前已不一致 |

因此，当前可观测差异已提前到HC-pre门控输出，不必等到Indexer或attention才出现。
结合源码，BSH与Native在HC线性分K、统计归约及混合求和顺序上有差别，
但本次没有逐一替换这些因素，不能认定其中某一项是本例或最终整层误差的唯一原因。
也没有把已在后续算子发现的差异全部归入HC-pre。
短query/混合长度矩阵维持停止，不更改BSH或TND正式数学实现。

## 失败记录与任务

所有任务均为单卡、通过task-submit提交，max-time780秒，没有自动调度。

| 任务 | 结果 | 说明 |
| --- | --- | --- |
| `task_20261018_190021_374849510199` | exit1，约2分11秒 | Native通过；助手wrap的compile入口被框架from-import别名绕过，在TND前资格断言失败 |
| `task_20261018_190447_391062013211` | exit1，约2分16秒 | 修正到真实reselect API边界后资格通过；助手漏导入custom op注册，在TND算术前退出 |
| `task_20261018_190900_40661794616` | exit1，约3分28秒 | 正式三方图验证；Native/TND通过，BSH首个bit差异，按要求停止 |
| `task_20261018_191851_264183838` | exit0，约1分4秒 | 仅定位同输入HC前缀，确认HC-pre门控已不同；没有继续矩阵或计时 |

前两项是助手失败，不作数学失败或性能样本。原脚本与manifest的v1/v2/v3全部保留。
修正未放宽bit判据；没有因BSH失败而改gold、输入或切换另一套BSH算法。

## 复现与证据

实验助手位于227自有目录
`/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/tnd-bsh-ragged-full-20261018`。
保持前述source、私有runtime package、13包和真实权重不变，使用全新输出标签：

```bash
task-submit --device auto --device-num 1 --max-time 780 --ptoas 0.63 \
  --env PYTHONNOUSERSITE=1 \
  --env ASCEND_HOME_PATH=/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2 \
  --env PTOAS_ROOT=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/experiments/nalinaly-71153bb3/reference-env/bin \
  --env CCACHE_DIR=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/src/pypto/.ccache \
  --env CMAKE_BUILD_PARALLEL_LEVEL=4 \
  "bash /data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/tnd-bsh-ragged-full-20261018/run-case.sh <新标签> 6,6,6,6"
```

本地完整原件保存在workspace的`reports/dsv4-tnd-bsh-ragged-full-20261018/`。
项目内保留[三方S6结果](evidence_tnd_precision/ragged_bsh_full_20261018/s6-summary.json)、
[BSH准入结构测试](evidence_tnd_precision/ragged_bsh_full_20261018/entry-result.json)、
[HC前缀定位](evidence_tnd_precision/ragged_bsh_full_20261018/hc-prefix-result.json)、
[证据索引](evidence_tnd_precision/ragged_bsh_full_20261018/index.json)和
[复现助手及原始日志归档](evidence_tnd_precision/ragged_bsh_full_20261018/evidence.tar.gz)。
大体积失败tensor快照保留本地及227原目录，索引记录SHA/字节数，不把它们混入小体积归档。
未将大模型和整个cache bank纳入仓库。本轮全部四项任务已终态，没有本轮残余占卡任务。
后续继续短query时必须明确沿用首差停止约束或单独处理BSH已知差异，不能默认将本轮未测项视为通过。
