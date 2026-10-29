# 全局bucket与dummy请求FULL精度验证（2026-10-29）

## 范围和基准

- 当前分支：`dev/pypto-dsv4-csa-nal-tnd-native-align-20261027-codex`。
- 保留正式算子基准`5c480a8a`。本轮先增加独立诊断，不改SDK、runtime或生成C++。
- 真实第4层权重、8K合成hidden/history、TP1、seed1024，Native确定性level1，
  `HCCL_DETERMINISTIC=true`。公共工具链和冻结Native沿用[主记录](README.md)。
- 测试生产55参数TND服务，经实际自定义op提交，检查资格和图合同；不外显TopK，
  比较最终输出和六份cache/state。首差后停止，不以未测后续模块作通过结论。
- 两边从相同cache初态做两次FULL回放；检查全cache allocation允许写区、metadata只读
  快照和输出首尾保护区。FULL为单层raw NPUGraph加服务合同验证，不是整模型框架派发验收。
- 仅物理15卡，外层780秒、内层750秒加15秒清理。8K整模型尚未启动，16卡未获本轮授权。

## 首个控制组

物理token容量24、请求容量5，四个真实S6请求加一个零跨度dummy：

```text
Q = [0, 6, 12, 18, 24, 24]
L = [8198, 8198, 8198, 8198, 0]
num_reqs_actual = 4
```

| 门禁 | 结果 |
| --- | --- |
| 不初始化NPU的导入预检 | v2通过；只加载当前容量fixture，Native包仍冻结 |
| 不初始化NPU的生产根编译/load | 55参数、双AIV派发检查通过 |
| Native FULL重复、FULL/eager | 七项逐bit一致 |
| TND FULL/eager | 七项逐bit一致 |
| cache/metadata/输出保护区 | 通过 |
| Native vs TND SWA cache | 逐bit一致 |
| Native vs TND主Compressor state | **首差，41088个FP32元素，最大绝对差1.1920928955078125e-6** |
| 后续cache、最终输出、mixed、整模型 | 按首差停止要求，未验收 |

任务`task_20261029_141052_37178438595`，诊断计时210.03秒，已退出并释放设备。
进程exit0仅表示探针正常保存结果；JSON的`pass=false`才是数值验收结论。

## 首差源码线索与单因素验证

Native [Compressor初始化](../../../csrc/attention/compressor/op_kernel/arch32/compressor_kernel_perf.h)
在`SetBaseSize`前，从尾部剔除Q跨度为0的请求，再判断是否等长。
当前PTO主投影和Indexer投影的等长判断仍遍历整个Q容量，因尾0误判为不等长。
这会改变列分组和K方向累加起点，可能解释已观察的FP32 state差异。

该规则只剔尾部Q跨度0，不能按L=0过滤，也不能跳过中间零跨度。上述为源码线索，
仍须实际单因素对拍证明，不能直接当作冻结Native二进制的行为结论。

独立`--projection`诊断先验证与失败任务的hidden、positions、cache初态及Native金值
逐bit相同，再使用实际Native输入Norm hook的输出，运行相同`mainproj_arm MODE=3`：

1. 原Q容量6，必须完整复现生产首差state。
2. 仅把传入bounds改为原存储的前5个元素，剔除最后零跨度；要求state对Native逐bit一致。
3. 两臂均检查FULL/eager、两次回放及保护区；没有证明因果前不改正式投影。

### 已完成因果验证

| 微任务 | 输入身份与保护 | 原bounds6对Native | 仅剔尾0 bounds5对Native |
| --- | --- | --- | --- |
| v1，32.59秒 | 初始hidden/cache及Native金值一致，state保护通过 | 精确复现41088差异 | state逐bit一致 |
| v2，20.37秒 | RoPE视图形状检查停止，未运行微根 | 未验收 | 未验收 |
| v3，32.85秒 | 143项身份检查全部一致，只读及保护通过 | 两次FULL精确复现41088差异 | 两次FULL state逐bit一致 |

v2是探针把`[24,1,1,64]`和正规化的`[24,64]`视图当作不同形状；v3只将完整RoPE
视图统一为二维，逐bit比较全部原始元素，不截取或放宽阈值。129项捕获输入字段包含
SAS/QLI、Q/L、start、table、slots和RoPE，其余14项为hidden/positions、五组cache
初态及七份Native金值。Native inputNorm来自实际warmup hook，hook仅调用一次并在
capture前移除。原臂逐bit复现整份实际生产state，排除了微根布局替换导致假因果。

据此仅修改主/Indexer投影的等长判定：在device扫描最后一个正Q跨度请求，判断
`[0,live_requests)`是否等长。中间0和非零跨度的L0 dummy仍保留。正式入口ABI不变，
不补每请求S6、不修改Native。

## 修正后完整FULL结果

在**同一张Native图、同一张生产TND图**以及同一组输入地址上，依次执行：

| 状态 | 真实请求长度 | 实际/物理行数 | Native自重复 | TND对Native七项 | guards/地址/合同 |
| --- | --- | --- | --- | --- | --- |
| A满档，尾dummy跨度0 | 6,6,6,6 | 24/24 | 逐bit一致 | 逐bit一致 | 通过 |
| B混合，尾dummy跨度6 | 1,5,6,6 | 18/24 | 逐bit一致 | 逐bit一致 | 通过 |
| 返回A | 6,6,6,6 | 24/24 | 逐bit一致，且回到原A金值 | 逐bit一致 | 通过 |

每个状态两次FULL回放。B的五组metadata真实记录均为：

```text
Q = [0, 1, 6, 12, 18, 24]
L = [8193, 8197, 8198, 8198, 0]
num_decode_tokens = 18
num_reqs_actual = 4
```

**没有把每请求补成S6**；仅在batch末尾增加6行dummy。Native metadata仍保留真实18行，
没有手改为24；旧图内部捕获容量24，当前短view通过原存储正规化后核验相同地址。
同时检查SAS/QLI、完整compressor RoPE、Q/L/start、table/slots等捕获地址，
并按B的实际Native slots重新计算cache允许写区。B的hidden也改变，Native有效输出与
原A不同，返回A再次一致，排除了图始终读旧输入的假通过。

实际生产自定义op计数在capture后不再增加；服务仍eligible，捕获合同允许重放，
实际ABI55且未启用debug/TopK输出。14组保护检查全部通过。

任务`task_20261029_144130_688207118`，诊断计时224.80秒，设备锁持有约242秒，
已退出并释放物理15卡。该耗时包含编译和检查，**不是算子性能数字**。
本地7项合同测试通过；独立源审检查尾零规则、内部零跨度、同址更新与首差门禁。

### 结论边界

本轮通过上述单层、固定请求容量5、真实请求数4、T24→T18→T24的FULL测试。
不代表空rank、动态改变真实请求数、其它层/seed、128K或整模型逐token已通过。
8K整模型尚未运行；现有CSA整模型启动器要求DP=EP=16，已向用户申请16卡授权，
未获批准前只保留8–15卡规则，不提交16卡任务。

## 待完成门禁

- 获批后做8K真实整模型token对拍；保留当前及旧基准，192 token，不跑Native2/128K。
- 每任务仍不超过15分钟；发现首个token差异后停止扩大，先做同输入定位。

## 复现命令

在已有独立实验目录加载`env.sh`，使用公共bootstrap和该环境记录的冻结Native：

```bash
source src/tests/pypto_test/nal_native_alignment_20261027/env.sh
python src/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py \
  --shared-toolchain --native-plugin-root "$SOURCE_ROOT" \
  src/tests/pypto_test/nal_native_alignment_20261027/compile_rms.py \
  results/NEW-cpu spmd 6,6,6,5,1

task-submit --device 15 --max-time 780 \
  --env PYTHONNOUSERSITE=1 --env HCCL_DETERMINISTIC=true \
  --env PTOAS_ROOT --env ASCEND_HOME_PATH \
  "bash $align_root/src/tests/pypto_test/nal_native_alignment_20261027/run_rms.sh NEW-control padding"
```

`6,6,6,5,1`仅给无设备编译选择T24/请求容量5策略，不是数值fixture。
实际控制组由`probe_padding.py`构造；`--projection`只做独立首差微根。
修正后完整测试在任务命令的最后增加`--mixed`。所有label须全新，避免覆盖证据。

## 原始证据

- [导入预检](evidence/padding-preflight-v2/receipt.json)、
  [初始生产根CPU55](evidence/padding-capacity5-compile-v1/receipt.json)、
  [修正后CPU55](evidence/padding-trim-compile-v1/receipt.json)。
- [原控制组首差](evidence/padding-capacity5-v1/result.json)、
  [微根CPU11](evidence/padding-mainproj-cpu-v1/receipt.json)。
- [因果微根v1](evidence/padding-mainproj-v1/result.json)、
  [输入形状检查v2负结果](evidence/padding-mainproj-v2/result.json)、
  [完整身份/只读复核v3](evidence/padding-mainproj-v3/result.json)。
- [修正后A→B→A完整FULL](evidence/padding-trim-mixed-v1/result.json)。

原始JSON保留27份正式源码SHA、SDK版本、实际metadata和图描述符。
大型failure tensor现场保存在独立实验目录，不将真实权重和大Tensor推送GitHub。
