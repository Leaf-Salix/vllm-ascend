# HCA TND完整模型接入验证

2026-10-24，参考CSA-MPMD-TND的真实图重放验收方法。
本轮使用正式FP32融合版本，未采用BF16舍入对齐候选。

## 已完成的接入功能

正式模型入口：`vllm_ascend.models.pypto_deepseek_v4_hca.PyptoHCATNDDeepseekV4ForCausalLM`。
通过`hf_overrides={"architectures":["PyptoHCATNDDeepseekV4ForCausalLM"]}`显式选择。
仅20层C128 HCA接管为TND MPMD；21层CSA、SWA和FFN仍为Native。
旧HCA/CSA联合模型不被这个新入口悄悄切换。

服务入口为`vllm_ascend/ops/pypto/deepseek_v4_flash_hca_tnd/graph_service.py`。
Native权重、Q/L、位置、slot、页表及state直接绑定。C128 compact metadata仍调用
Native producer，图内按设备L屏蔽失活请求；Python只使用已有Host计数及描述符。
没有新增Host逐请求计划、cache复制窗口或外部scratch。

初始化选择正确的长历史捕获范围；重放前检查设备地址、容量及核内区间策略。
Native/空路径捕获不会被当成PTO图使用，超出合同范围回正常派发。
6个正式接入文件与已完成设备验证的冻结包文本相同，核对记录为
`PRODUCTION_BINDING_AUDIT.json`；没有用测试目录中的替代数学函数冒充正式入口。

## 设备证据

先单卡再16卡，已通过的检查没有重复跑：

- 单卡：S1/S5/S6、B1/B16/B24、固定混长T56、历史边界、尾批、padding、请求重排，
  共6类29步真实ACLGraphWrapper重放。输出及cache/state对相同PTO eager逐bit一致，保护区通过。
- B16：复用已完成`task_20261024_132810_32584376495`的完整模型结果。
- B24：新任务`task_20261024_151031_276989322516`，exit=0。

| 每rank真实业务步骤 | 128K/B16 | 128K/B24 |
|---|---:|---:|
| 满批S6 | 33 | 33 |
| 尾批 | 18 | 17 |
| 尾批中有padding | 15 | 15 |
| 尾批中单请求6行小图 | 3 | 2 |
| 真实长历史FULL重放总步数 | 51 | 50 |
| 首轮S1 eager | 1 | 1 |

尾批包含padding和小图，不能重复相加。以上各步20层HCA全部实际replay，无Python重新
派发。等待其他DP的短dummy图单列保存，不计入业务步骤。每个DP的实际检查保留在
`MODEL_INTEGRATION_SUMMARY.json`和原始`functional_check.json`。

B16每侧256请求/49152输出token，B24每侧384请求/73728输出token，均零抢占。
两档HCA的16个DP副本输出各自相同。模型退出前rank0空闲显存分别3291.75/2284.45 MiB，
仅是当前功能配置的观测，未做显存归因或性能比较。

## 运行配置和复现

正式权重`/data/model/DeepSeek-V4-Flash-0731-w8a8`，复用已有P阶段KV：
`/data/pyptouser/qinchuanyu/pto-eager/csa-repro-20261021-clean/kv-repro/mixed128k_fixed`。
没有重新prefill或做hash扫描。本轮是D侧TP1、DP/EP16的完整模型验证。

- DSpark出5验6、NZ2、FULL_DECODE_ONLY、npugraph_ex、event mode1、确定性level1、EPLB关闭。
- 功能配置与CSA参考相同：静态编译及SuperKernel关闭。本轮不是单算子性能比较。
- B16预算0.92/HCCL1800；B24预算0.935/HCCL768。
- HCA的Simpler容量必须使用已验证的`PTO_CSA_RING_HEAP_MB=64,256,64,64`及
  `PTO_CSA_RING_TASK_WINDOW=4096`；不能直接沿用CSA更小的heap设置。
- `PTO_CSA_VARIANT=performance`、`VLLM_ASCEND_ENABLE_NZ=2`、
  `VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0`。这些配置不改变Native权重或算子输入布局。
- 使用公共已验收`csa-shared-20261024-group-dims`；先source工作区环境，再source公共env。

已执行命令见`jobs.json`。`run_model.sh`检查共享工具链和单卡门禁，再启动16卡并
检查逐层实际路由；只看到FULL标签或捕获成功不算通过。
后续冻结正式源码时`integrate.py`识别完整接入，避免重复追加模型类或注册。

## 答案质量独立验收

192token功能任务不保证答案写完。B16/B24与Native分别64/256、112/384请求token相同，
这不是正确率；两个实现可能措辞不同，也可能共同答错。详细问答导出到工作区
`downloads/HCA_TND_MODEL_INTEGRATION_20261024/`，不截断已记录的输出，不补写答案。

用户已确认增加B16同一批16问题的完整答案对照：两侧正常EOS，上限2048token，
按题意标准结果评分，不要求Native token一致。`prepare_quality.py`只改变采样和
结果记录，使用相同正式FP32算子和已验证的模型接入；保留终止原因、输出token及全文。
功能通过、答案完成、内容正确、格式符合要求分别报告。两侧任务均已exit=0，
256请求、零抢占，16个DP副本各自相同。Native共297520输出token，HCA共386304。

| 16个独立问题 | Native | HCA MPMD |
|---|---:|---:|
| 正常EOS/停止 | 11 | 7 |
| 到达2048上限 | 5 | 9 |
| 已核验完整内容符合题意 | 0 | 0 |

这不是通用模型准确率：部分答案在上限处截断，已结束答案也有明确错误。
q02正确首行[1,9,5,24]，两侧均错误；重复题要求80/81次，Native输出85/83次，
HCA输出82/82次。不能因为Native自身有错误，就把HCA答案质量自动判通过。
也不能把这些共同错误直接归因为HCA数值实现。

逐题全文、标准结果与核对证据在`QUALITY_EOS2048.md/json`。按CSA的对照展示方式
额外导出并排HTML、Markdown、原始token JSON和包含128K背景的完整输入；
`QUALITY_EXPORT.json`记录目录和下载zip。功能接入已完成，答案质量未验收通过，
按用户要求停下来讨论，不自动增加生成长度、换题或修改算子数学。
