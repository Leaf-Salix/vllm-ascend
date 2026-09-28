# 完整 nalinaly 参考复现：2026-09-28

## 范围与当前结论

本轮先复现完整参考实现及其原始测试，再按完整模块迁移到 Leaf BSH 分支。
此前局部候选的结果不能代表完整参考实现。当前已完成独立环境构建、完整 kernel CPU 编译及原版单层 graph 预检。
单层输出对Native有误差，该档没有提速；七档整模型复现和模块迁移尚未完成。

参考完整性能 kernel 融合 HC pre、input norm、CSA 与 HC post，输入输出包含 mHC 残差流。
现有 Leaf DSA 入口处理已经归一化的 attention 输入，融合边界不同。
后续完整模块迁移需要保留原生 DecoderLayer 的外部签名和返回值，并明确内部融合边界；
不能只移植若干 tile 常量后宣称对齐。

## 固定版本

| 部件 | 本轮参考版本 |
| --- | --- |
| vLLM-Ascend 完整参考源码 | nalinaly `71153bb3be9faa6c79afecd92e229e765cef37c5`，原样归档，逐文件 SHA256 |
| 七档运行方法及报告 | nalinaly `e58ddc94d77c93a0a8a85ab2db4bd773da9dff84` 的 `csa_atomic_matrix_20260928`；运行源码仍为上一行 |
| PyPTO | nalinaly `3e87a843619aca13af39755700513d26b402e924` |
| Simpler | PyPTO 的原始 gitlink `a54c0509552b01e13fb0960e23ce409a01b5024f` |
| PTO-ISA | 官方 `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d` |
| PTOAS | 官方 0.66，CPython 3.12 aarch64 wheel |
| CANN / vLLM / Torch / torch_npu | 9.0.1 / 0.25.1 / 2.10.0+cpu / 2.10.0.post2 |
| Leaf 工作分支 | `dev/pypto-dsv4-csa-v0.25.1rc1-cann9.0.1`，本轮前 HEAD `ee5d9947b`；未修改其 kernel |

早期参考日志的 PyPTO5495749/Simpler166852bf 已过时，不能作为此性能版的依赖。
新建独立 reference-env，构建时启用 `PYPTO_BUILD_TORCH_NPU=ON`、四路并行及已有 ccache。
原有环境保留。预检记录 Python 来源、PyPTO/Simpler/Torch NPU 扩展路径与哈希、Git HEAD/dirty 状态、
ISA HEAD及运行库元数据、PTOAS 版本及 launcher 哈希，并检查 CANN 编译模块 可导入。

复用现有 vLLM-Ascend 原生扩展和 custom OPP，尚未证明其二进制与作者环境完全一致；
不得将当前环境称为逐字节相同的作者环境。

## 验收计划

1. 原版单层脚本：真实第2层权重、合成历史，B4/S6/H8192、performance、NZ2、atomic0、det0、graph。
   检查 Native/PTO 输出与 cache、保护区、A/B/A graph 重放及20次计时。单层通过不代表整模型通过。
2. 原版 prefill 重建真实模型的8K/128K离线 bank，TP4/DP4/EP16；每档四种输入。
   Native 和 PTO decode 使用同一 bank。已有 plan 和75分片检查通过，缓存尚未生成。
3. 原版七档 TP1/DP=EP16、DSpark 出5验6、EPLB关、full_decode_only：128K B4/8/16，8K B16/24/32/40。
   容量40，预算256/400，capture24/48/96/144/192/240；8步warmup、10步独立forward计时、另轮3步profile。
   原 `performance` 入口内部已分开计时与profile，不将profile耗时当正式速度。
4. 核对全部rank配置、位置、token/DSpark、均值/P95/最慢rank及数值精度，再逐模块迁移。
   token一致不替代逐层或共同历史下的数值误差验证。

## 已发生的检查与失败

| 检查/任务 | 结果 | 首因与处理 |
| --- | --- | --- |
| `task_20260928_172049_358065516214` | exit1，未执行算子 | 缺PTO_EAGER_ROOT；补独立原版目录约定 |
| 旧依赖完整CPU编译 | exit1 | assemble不支持pre_quant；改为精确参考工具链，不改kernel |
| 独立依赖构建 | exit0 | PyPTO/Simpler精确提交安装完成 |
| PTOAS启动检查 | 先失败后通过 | 通用tar要求Python3.11；改用官方cp312 wheel，显式使用现有Conda C++库 |
| 完整参考CPU编译 | exit0，COMPILE_PASS | 完整CSA lowering、工具链编译及load；无设备执行 |
| `task_20260928_175830_25582707293` | exit1 | 包装脚本覆盖PYTHONPATH遗漏CANN 编译模块；保留CANN路径并补预检 |
| `task_20260928_180432_275645018241` | exit1 | ASCEND_CACHE_PATH目录未创建；补mkdir |
| `task_20260928_180718_284496627321` | exit1 | 未开启可选Torch NPU扩展；原源码已有模块，补构建开关及预检 |
| `task_20260928_181150_304968913742` | exit1 | 构建进程误用旧editable启动模块，运行库ISA不匹配；修正路径优先级并重建 |
| `task_20260928_181941_208666318027` | exit0 | 原版单层graph/计时完成，具体数值见下节 |

以上启动失败属于环境准备错误，不能计为参考kernel精度失败或性能退化。
曾将未构建扩展误判为未公开文件，已纠正；没有因此新增替代实现。

本地证据目录：工作区 `reports/dsv4-nalinaly-full-repro-20260928/`。
包含原始参考、逐文件manifest、dependency-manifest、环境/构建/运行脚本、checklist及取回的preflight。
远端独立实验目录保留每次失败日志，结果目录不覆盖。后续结果继续追加，不替换失败记录。

## 原版单层预检结果

任务 `task_20260928_181941_208666318027`，原版脚本B4/S6/H8192、正式第2层权重、合成历史，
performance/NZ2/atomic0/det0，四张kernel根权重布局均为NZ。Native与PTO重复运行分别自洽；
graph A/B/A重放与各自eager输出/cache逐位一致，保护区检查通过。

| 指标 | 结果 |
| --- | ---: |
| Native graph中位耗时 | 560.580 μs |
| PTO graph中位耗时 | 568.140 μs |
| PTO相对Native耗时变化 | +1.349% |
| Native/PTO P95 | 576.800 / 581.680 μs |
| 融合层输出相对Native L2 | 0.19502785% |
| 输出不同元素 | 96,880 / 393,216 |
| 输出最大绝对误差 | 0.03125 |
| Top-K集合替换元素数 | 117（24行）；无越界/重复/缺失结构错误 |

输出形状为[24,4,4096]，包含HC post，不能与旧DSA-only输出[24,4096]的误差百分比直接比较。
原比较器的零容差FAIL仅表示有数值差异；进程exit0与graph通过不等于Native数值精度通过。
cache/state相对Native也有差异，全部逐项结果保存在report.json与states.pt；无NaN/Inf。
本档计时20次，compact metadata采用原脚本reuse策略；属于单层预检，不否定或确认七档整模型收益。

单卡原始报告已取回工作区 `single-env6/`；远端保留states.pt和全部原始日志、前后环境指纹。
18:23:08提交 `task_20260928_182308_268934625417`，按原脚本TP4/DP4/EP16依次生成8K/128K真实离线bank。
该任务最终 exit 1；此阶段不测CSA性能。详细原因及后续重测如下。

## 原生 prefill 初始化失败与配置恢复

任务 `task_20260928_182308_268934625417` 在19:06启动、约19:10失败，
日志已备份到本地 `prefill-failed/`。19:48才观察到终态，未做到连续每五分钟监控。
75分片已加载，但首次原生 dummy run 的 MoE hash 路由报 `input_ids=4, rows=2`，
部分TP rank为 `3, rows=2`；真实prefill未开始，两套bank只有plan/token文件，没有KV payload。

根因由源码及独立review确认：

- 历史 `e048502b1` 和 `a28328967` 父提交的离线脚本使用默认auto worker。
- 官方平台只在 `worker_cls == "auto"` 且未启用SP时，将 `all2all_backend` 设置为
  `flashinfer_all2allv`，以关闭框架层 `use_sequence_parallel_moe`。
- `a28328967`（9月26日）新增显式 `OfflineNPUWorker`，绕过该配置分支。
- TP4/DP4/EP16下，模型先拆hidden，MC2随后再次拆hidden；token ID没有第一次拆分。
  8行输入先拆成2行，hidden补6再拆得到2行；ID从8补6再拆得到[4,4,3,3]，吻合错误。
- 官方配套vLLM `752a3a504485790a2e8491cacbb35c137339ad34` 本身也默认
  `allgather_reducescatter`；不能将此错误归因于CSA kernel或仅凭版本字符串归因于依赖漂移。

新增独立 `run_prefill_native_config.py` 副本，仅prefill显式恢复
`all2all_backend="flashinfer_all2allv"`；原始文件、模型、kernel、decode脚本均未修改。
这应称为“恢复原始bank生成的有效配置”，不能称执行脚本完全未修改。
`prefill-native-config.patch` 和manifest保存唯一差异及原/新SHA256。
CPU门禁确认差异恰为该参数、decode kwargs为空、TP4/DP4下的
`use_sequence_parallel_moe` 从True恢复False；这尚不能替代真实NPU验证。

20:00:45提交 `task_20260928_200045_19802682608`，16卡、执行上限7200秒，
仍为原模型、TP4/DP4/EP16、原缓存布局，顺序8K/128K。
输出使用新目录 `results/prefill-native-config-h8192` 与 `...-h131072`，保留失败日志。
状态待核验；尚无完整参考性能结果，也没有将该配置修改用于Leaf生产代码。

## Bank验收与完整decode任务

`task_20260928_200045_19802682608` 已完成，exit0。8K与128K各4组输入、16份TP
manifest、每份191个tensor；原始audit全部PASS，各TP副本有效前缀无差异。
四个DP rank均生成1 token并保存请求结果；prefill耗时含IO，8K约15–18秒、128K约39–41秒，
不能用这些数值声称decode性能。
原始prefill日志及audit已取回本地，远端bank保留完整payload。生产kernel没有改动。

20:12:47提交 `task_20260928_201247_22649656531`，16卡max7200秒，
原始七档矩阵：H131072/B4,8,16和H8192/B16,24,32,40；PTO、Native分开运行，
NZ2/performance/atomic0/det0、DSpark5、EPLB关闭、FULL_DECODE_ONLY。
每档先独立计时再独立profile，不用profile扰动的耗时作主结果；结果尚待核验。
仍使用冻结71153原始decode脚本，prefill配置恢复不传播到decode。
