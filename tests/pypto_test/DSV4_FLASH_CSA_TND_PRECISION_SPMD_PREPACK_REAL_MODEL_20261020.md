# SPMD预打包精度版：真实整模型结果（2026-10-20）

## 结论和版本

继续使用`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，未新建或切换分支。
当前验收代码为`ee819cf4a`：独立预打包包六个文件，原43个完整精度源码及父SPMD的14源均未改动。
新增固定HC权重预打包在两档真实整模型中保持全部保存token和接受事件与Native一致。

完整CSA仍在一个PTO根内执行；其中HC-pre、输入Norm、WKV、Indexer权重投影融合为单SPMD前缀，
其余原PTO后缀未全部融合。这不是nalinaly的完整单MIX实现，性能仍未追平Native。

## 精度和性能同表

所有延迟单位ms。每步先取16rank最大值，每轮20个forward和19个cycle间隔；不删除离群样本。
候选、冻结精度及父SPMD各3轮；历史Native的8K为3轮、128K实际为2轮（40/38个样本），按原轮数汇总。
Native、冻结精度版及父SPMD均为历史实测；没有新跑Native2。

| 配置 | 每卡提交/计时活跃 | token比较数/差异 | 接受事件/差异 | Native forward | 冻结精度 forward | 父SPMD forward | 预打包 forward | 相对父SPMD降低 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8K/B40 | 40/40 | 491,520/0 | 105,152/0 | 89.482 | 108.405 | 107.231 | 106.729 | 0.47% |
| 128K/B24 | 24/21 | 294,912/0 | 61,568/0 | 79.730 | 91.434 | 91.153 | 90.761 | 0.43% |

| 配置 | Native cycle | 父SPMD cycle | 预打包 cycle | 预打包forward标准差 | 比Native forward慢 |
| --- | --- | --- | --- | --- | --- |
| 8K/B40 | 105.575 | 122.860 | 122.618 | 0.802 | 19.27% |
| 128K/B24 | 92.293 | 103.513 | 103.122 | 0.903 | 13.84% |

两个配置原始均值均较父版小，但约0.4%～0.5%的跨任务差值不能证明稳定整模型收益。
可确认的稳定收益目前限于[单因素前缀A/B](DSV4_FLASH_CSA_TND_PRECISION_SPMD_PREPACK_20261020.md)：
T21降低12.28%，T240降低9.53%，各12个配对窗口均变快，所有491项精度检查通过。
例如T240每层前缀节约约38μs，21层简单相加约0.8ms，仅为约107ms整模型forward的不到1%；
这只是量级估算，不能按9.53%直接推导整模型加速比例，也不能替代真实流水计时。

预打包增加每层2MiB、整模型42MiB常驻只读权重；保持独立opt-in variant，不修改原默认精度路由。

## 测试方案及验收范围

与[父SPMD整模型报告](DSV4_FLASH_CSA_TND_PRECISION_SPMD_REAL_MODEL_20261020.md)保持相同方案：

- 真实`DeepSeek-V4-Flash-0731-w8a8`，TP1，DP=EP16，DSpark出5验6。
- 21个CSA层完整HC-pre→HC-post使用PTO，20个HCA层Native；FULL graph、NZ2、SK0、EPLB关闭。
- Native确定性level1及HCCL确定性开启；接收步长由模型实际生成，不强制3.8。
- 每rank warm192＋3轮各192 token；token/事件总数包括warm，不仅是计时轮次。
- 8K/B40实际40并发，buckets `[6,240]`；128K/B24提交24、实际21并发，3等待，buckets `[6,144]`。
- 128K计时真实126行、全局graph容量144行。每请求无需补S6由另外的混合1～6行FULL证据证明，
  不能仅凭本次整模型S6窗口宣称已覆盖所有不等长输入。
- 零抢占；8K每个候选窗口与历史Native三轮全部CPU字段（除call）一致。
- 128K固定C1→N1、C2/C3→N2；960条同阶段CPU记录全字段严格一致，含input_ids，未归一化哨兵。
- 题目、顺序及聊天后缀沿用nalinaly对齐方案，使用本项目自己的Native前缀/cache bank；
  未证明与nalinaly原bank全字节相同。本次不是nalinaly/本实现同场硬件速度对比。

这证明已测两档的输出token和接受事件一致，不是任意模型/输入下所有内部Tensor逐bit一致的证明。

## 实验顺序与失败保留

1. 固定权重预打包前缀对拍、配对性能通过，再接入完整根。
2. 混合T21/T6 FULL A/B/A、只读输入、poison、guard及多bucket交替全部逐bit通过。
3. 首次整模型初始化发现缺少loader读取归约日志常量的模块；新增纯reexport兼容文件，无token结果。
4. 第二次初始化发现loader启用梯度；准备函数显式局部no_grad，保留SDK推理专用检查。
5. 单卡真实NPU重现grad enabled＋requires_grad Parameter，准备成功且外部grad恢复，原权重SHA相同，
   完整FULL精度再次通过。
6. 新快照运行8K，独立审核通过后才运行128K；两任务均exit0，调度780秒、内部timeout750秒。

失败日志、旧五源及修复前六源身份未改写，保留在前缀/FULL报告及各历史证据目录。
整模型任务分别为`task_20261020_032047_260695025236`及`task_20261020_033206_30734885368`。

## 复现及原始证据

环境与父报告一致：CANN9.2 beta2、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、
vLLM-Ascend0.25.1rc1、PyPTO `3e87a843`、Simpler `a54c0509`；原SDK及PTOAS保持不变。
设置`PYTHONNOUSERSITE=1`及显式`ASCEND_HOME_PATH/PTOAS_ROOT`；同checkout ccache、四路编译。

variant为`pkg:deepseek_v4_flash_csa_tnd_precision_spmd_prepack`。
源快照3385文件，实际63运行源；manifest SHA256为
`5b1645c5d12475957f3e26b4060157f722188e83a0aa13ff2f542786a77735fd`。
两个原始包各228个文本文件，包含全部rank报告、Native gold、严格phase审核、全部助手、
实际63源码、正式prepare编译receipt、加载回归FULL结果，以及直接父SPMD和冻结精度控制报告。
所有计时样本和逐轮均值见`summary.json`，归档哈希见`manifest.json`。

证据目录为`evidence_tnd_precision/spmd_prepack_real_model_20261020/`。
同模型、原bank、Native gold和静态包就绪后，恢复其中tools并保持记录的源/助手身份，
由task-submit运行`run-case.sh 8k pto`；独立审核通过再运行`run-case.sh 128k pto`。
每次使用全新的输出目录。完整模型、bank张量和二进制cache没有纳入文本归档，需要预先准备；
工具内原实验根需映射为复现者的独立目录，不能直接覆盖原已验收实验。

离线复算：

```bash
python -m pytest tests/pypto_test/test_precision_spmd_prepack_real_model_evidence_cpu.py -q
```

该测试直接读取归档重新比较全部token/events、同阶段CPU轨迹、源码SHA和全部原始延迟，
并绑定同一六源的正式编译receipt与FULL加载回归，不依赖摘要里的pass标记。

权重不可变，重载必须重建runtime与graphs；工作区只支持设备执行顺序串行复用，仍不支持并发stream。
下一布局候选未包含在本验收版本中，必须独立对拍，不将该版精度结果转授给尚未实测的改动。
