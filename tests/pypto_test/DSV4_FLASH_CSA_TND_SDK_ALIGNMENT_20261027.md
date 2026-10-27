# 精度TND与nalinaly完整SPMD：同工具链观测（2026-10-27）

## 正式基准

当前分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。正式计算提交11b380e3f，文档提交123c12c63；本轮未改正式计算代码。已验收版本保留，8K/B40、128K/B24 warm192＋测量192的token和接受事件与各自Native完全一致；这是旧验收SDK下的独立资格，不自动扩展到新工具链。

nalinaly公开分支`dsv4-flash-pto-v0.25.1rc1`经GitHub ls-remote核对仍c346573ab3313fcc71cd58756acd129f34755439。使用逐文件SHA封存的该提交源码；整库fetch曾因网络early EOF失败，不宣称完整历史同步成功。

## 差距证据与优化顺序

先前相同旧SDK、8K/B32/S6的三方均值：Native1251.967μs、nalinaly cafc兼容版1216.794μs、我方Direct1587.462μs。我方Native A/B八外边界及ABA逐bit一致；nalinaly兼容版最终输出relative L2约0.333%。此差异不能直接推出整模型token不同，也不能把全部速度差归因于精度。

参考阶段跨度：我方入口→TopK692.22μs、TopK→heads475.68μs、heads→结束339.52μs；历史nalinaly多task伴随诊断为565.50/280.29/218.97μs。attention区间是最大的参考差距，但后者并非最新单SPMD的内部阶段测量。跨度含计算、搬运和等待，不相加归因为纯计算。

独立PMU观测每臂3次，完整PTO worker的AIV Vector中位活跃时间我方313.359μs、nalinaly旧兼容版187.756μs，差125.603μs；AIC MAC209.003/180.316μs，AIC MTE1：261.334/209.499μs。pipe互相重叠；Scalar计数不能当成纯等待，计数差不能当成可直接节省的关键路径时间。原始Profiler的acl-flow/schedule警告保留，不能用Profiler跨度替代普通计时。

优先方向为在相同工具链确认最新单SPMD的真实差距，再迁移数据搬运、临时缓冲和按实际owner等待的调度。Native K512在线更新、actualK归约树、概率BF16 round、inverse RoPE边界和全8192量化保持。nalinaly的K128局部softmax/重标定及部分量化边界不能直接复制并声称精度保持。

## SWA单因素复测：未采用

在正式Direct实现上复测历史SWA Vector128思路；不是首次提出该优化。两组均旧SDK、8K/B32/T192/history8191，12×100普通事件与独立9窗L1；不删除样本。控制及候选分配一致。先性能改善至少0.5%、≥10/12成对领先、L1降低，才执行Native精度扩测。

| 单因素 | 控制event中位μs | 候选event中位μs | 候选耗时变化 | 成对领先 | L1控制→候选μs | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SWA完整128key：softmax512列缩128列，保留GM64暂存 | 1627.572403 | 1619.740295 | −0.481214% | 12/12 | 1523.32→1499.84 | 未过0.5%门槛 |
| 同上，仅以UB view替代GM64暂存 | 1606.897583 | 1609.716721 | ＋0.175440% | 0/12 | 1487.34→1506.92 | 回退 |

两组均exit0；顶层pass仅代表脚本按性能优先流程完成，accuracy_executed=false，不代表精度验收通过。静态源/实际生成物审查通过、AIC与控制一致，仍不能替代NPU精度；本轮不合入、不扩测整模型。

## 新隔离工具链与测试入口

新工具链在yejia自有新目录构建，不修改旧验收SDK，不安装/覆盖Torch、Torch-NPU、Simpler或vLLM。PyPTO d44925c17d9ff7d7fa8e5067e41c4b4f80bf3063、PTOAS914863ed3d0eb3e92a4433a4a706a86cabf4c91f，沿用Simpler57e499238fd4ee3548947ad860be456aaeeb16b5、Python3.12.14、CANN9.2.0-beta.2、Torch2.10/NPU2.10post2、vLLM0.25.1。

公开源码归档SHA及每文件manifest、实际导入路径、DSO和PTOAS二进制SHA均封存；源码归档不含.git，不能伪称本地git checkout HEAD。PyPTO最小CPU UT107项通过，未初始化NPU。PTOAS默认Python模式缺少现有LLVM Python来源，使用公开914提交已有的PTO_ENABLE_PYTHON_BINDING=OFF原生CLI模式复用LLVM19.1.7，不改源码。独立CMake构建显式ccache、四并行。LLVM19.1.7、ISA327cd586及其余依赖复用自有产物；这不是对方私有构建二进制的逐字节复刻，三方是在本轮共同的新SDK内比较。

nalinaly c346改为内部scratch和每call清零同步，没有旧allocator参数。为了观察TopK，采用其原始decode_csa_spmd_debug入口（57Tensor），不是普通decode_csa_spmd入口（55Tensor）；不能称与其普通入口截图完全相同。保留原始backoff_ticks=150，不修改其DSL计算或生成C++。我方完整根64Tensor。最终两臂CPU compile/load/NPUfalse通过，绑定同一harness manifest a63f21b39153ac30656da21acafe10a65539246524226b8238f1c1c6d7171ff9。

实际nalinaly编排70Tensor/33alloc/1MIX，同一scope内按1024B对齐约216.133789MiB，256MiB剩39.866211MiB；这是静态容量资格。内部scratch无法由外部guard直接观察，报告明确为不可检查，不将空allocations解释为零内存或guard通过。Native/cache guard、只读metadata、同臂eager/raw/ABA保持。

## 三方硬件测试方案

单卡、真实第4层权重、合成hidden/history，同一原始输入与cache起点、独立缓存分配。8K/B32/S6/T192/history8192，NZ2，TP1，确定性level1、Torch deterministic、HCCL=true。完整HC-pre→HC-post，FULL graph，不是整模型token测试。

Native与我方用npugraph_ex静态FULL捕获，Native SK0；nalinaly原始JIT预热后direct FULL NPUGraph，compact metadata producer在图内。三方普通event回放同范围、12×100，轮转＋反序均衡，无DFX、无dep-gen、不删样本。先Native/我方A/B八外边界强制逐bit、三方各自ABA和cache guard；我方首差停止矩阵和计时，nalinaly对Native误差只记录，不用作我方精度gold。单任务外780s/内750s。

新SDK三方任务081907已exit0。普通事件计时结果：

| 实现 | min μs | mean μs | median μs | max μs | 单层精度 |
| --- | ---: | ---: | ---: | ---: | --- |
| Native SK0 | 1232.598 | 1233.611 | 1233.516 | 1235.106 | Gold |
| 我方Direct精度TND | 1584.195 | 1592.640 | 1593.825 | 1598.367 | A/B八边界逐bit；ABA和guard通过 |
| nalinaly c346原始DSL debug | 1077.460 | 1081.373 | 1081.329 | 1086.396 | A输出relL2=0.332971%；B=0.334736%；自身ABA通过 |

我方均值慢511.266931μs、47.279438%。新工具链没有消除差距；不能把旧/新SDK不同批次均值之差作为受控SDK提速。该任务仅单层，未重测新SDK的整模型token，旧验收环境保持。

## 后续采用门槛

1. 同SDK普通性能确认后，DFX/依赖与PMU分开采集；不把单SPMD整核计数说成某模块纯计算。
2. 只针对最大可证实模块做单因素候选；允许直接迁移nalinaly源码，数值差异部分保留Native算法。
3. 改善至少0.5%、≥10/12领先及L1降低，再做Native八边界/ABA及反向捕获重复，之后128K/B16与混合T126。
4. 稳定且精度通过后，才测真实整模型8K/B40，再128K/B24 warm192＋测量192 token；每项任务不超过15分钟，发现首差停止定位。

## 原始证据

完整输入身份、源码及SDK SHA、CPU编排、普通计时、PMU、SWA负面实验和复现步骤见[证据目录](evidence_tnd_precision/sdk_alignment_20261027/README.md)。本轮没有新的运行时代码合入。
