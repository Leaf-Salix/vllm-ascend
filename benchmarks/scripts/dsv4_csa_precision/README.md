# DSV4 CSA 单因素精度测试

目标是确定误差在哪个计算边界引入，不将多个数值修改一次合并后归因。
测试加载完整 checkpoint 的一个 C4 注意力层真实权重，使用固定 seed 的合成 hidden/history。
默认 TP1、B4、S6、block128；结束长度8K或128K。不是整模型生成或生产动态 padding 测试。

## 前提

- 私有运行环境已经修复负 scatter slot，能够运行当前CSA；保持已有vLLM/PyPTO/Simpler二进制。
- `HCCL_DETERMINISTIC=true`，初始化后设置并回读NPU确定性level1。
- NPU命令通过目标机器任务队列提交。激活匹配的CANN/ATB，设置正确的OPP路径与PTOAS_ROOT。
- `--sdk-root` 指向与已安装Simpler二进制匹配的干净SDK checkout。
- Torch-NPU2.10.0.post4沿用已有实验的shutdown兼容检查，仅放宽版本门禁，真实shutdown函数仍执行。

## 生成变体

```bash
python generate.py --repo /path/to/vllm-ascend --output /path/to/private-variants
```

输出目录必须不存在。生成器只写私有变体，保存逐文件SHA256清单；源代码不符合预期时直接报错。
所有QKV变体共用相同的诊断输出签名和控制基线，每次只插入一种BF16 round-trip：

|变体|唯一改变|
|---|---|
|control|无额外数值修改；O-proj固定为正式版本|
|qa|Q-A投影结果先舍入BF16，再进入QR RMS与量化|
|q_mm|Q-B反量化结果先舍入BF16，再进入逐head Q RMS|
|q_rms|Q RMS后的旋转部分先舍入BF16，再进入RoPE|
|kv_mm|KV投影结果先舍入BF16，再进入KV RMS|
|kv_rms|KV RMS后的旋转部分先舍入BF16，再进入RoPE|

五个独立因子都直接从control派生，**不是累积链**。因此单因素的最终输出误差可能不单调。
另有两个收尾对照：`q_pair` 相对 `q_mm` 只增加Q RMS到RoPE舍入；
`kv_pair` 相对 `kv_mm` 只增加KV RMS到RoPE舍入。这两条边分别验证投影与RMS边界的组合，
不能将pair直接相对control的改善归因于某一个因素。
同时运行原生Q-B→Q RMS→RoPE链，以CSA的QR及scale作为输入，隔离上游QR误差。

## 运行一个QKV变体

```bash
python run.py \
  --model /path/to/DeepSeek-V4-Flash-0731-w8a8 \
  --extension-dir /path/to/runtime/vllm_ascend \
  --sdk-root /path/to/simpler-sdk \
  --variant-dir /path/to/private-variants/qa \
  --out-dir /path/to/results/8186/qa --start-pos 8186
```

128K使用 `--start-pos 131066`。先运行control，再运行其他因子。
每个进程加载同一checkpoint、设置相同seed，保存hidden、初始cache和全部权重指纹。
Graph warmup、capture后均恢复初始cache，再replay取数。所有权重做CPU快照，变化即失败。

`stages.pt` 保存native/CSA的QR、QR scale、Q、raw KV、heads、最终输出，以及六类cache的有效写入槽、
原生与CSA值；`result.json` 保存误差、精确不同元素数、输入与原生输出指纹、运行版本、源文件hash。
cache误差只统计有效写槽，初始backing另存hash；不以大段未变化history稀释误差。

## O-proj：固定同一组heads

使用同一上下文control输出中的 **native heads**。所有实验和原生参考都消费完全相同的heads。
QR/Q/raw KV/cache作为冻结上游证据引用，相关零差异表示“未重新计算”，不能解释为CSA上游精度通过。

|变体|O-A量化前舍入|量化amax范围|O-B归约|
|---|---|---|---|
|oa32_group|FP32|每组1024|FP32逐组反量化求和|
|oa16_group|BF16|每组1024|同上|
|oa32_global|FP32|整行8192|同上|
|oa16_global|BF16|整行8192|同上|
|oa16_global_intsum|BF16|整行8192|INT32求和后一次反量化|

前四格权重scale统一在group求和后相乘，避免混入另一处乘法舍入。
A→B、C→D只改变O-A舍入；A→C、B→D只改量化范围；D→E只改变归约方式。

```bash
python run.py \
  --model /path/to/DeepSeek-V4-Flash-0731-w8a8 \
  --extension-dir /path/to/runtime/vllm_ascend \
  --sdk-root /path/to/simpler-sdk \
  --variant-dir /path/to/private-variants/oa16_global \
  --out-dir /path/to/results/8186/oa16_global --start-pos 8186 \
  --mode oproj --frozen-stages /path/to/results/8186/control/stages.pt
```

另存O-A原始FP32输出、量化INT8、各group scale、最终输出及原生参考。
这组为了控制数值因素统一调度方式，global量化会重复扫描；**不得用诊断kernel耗时代表正式实现性能**。

## 结果判定

- 首先核验同一上下文的hidden/cache/weight和native输出指纹一致。
- 对照对应模块误差、same-input结果和下游变化，不仅看最终输出。
- allclose使用 `rtol=atol=1e-2`，另外保留relative L2、max abs、不同元素数；通过阈值不等于逐元素一致。
- 所有中间输出均来自Graph replay，但metadata在捕获外固定，不能验证生产metadata刷新、变长请求或空rank。
- 本测试不测延迟；正式性能需另跑去插桩的同配置Graph对照。

## 汇总与独立性核验

```bash
python summarize.py /path/to/results
```

汇总器检查同一上下文的输入、权重及原生输出逐项hash一致，检查所有权重不变、确定性与guard。
还断言Q-only因子的六类cache逐元素不变，KV-only因子的QR/Q与其他五类cache逐元素不变。
O-proj比较heads与权重hash，并计算INT8×各自scale后的activation差异；
不同量化范围下直接比较INT8编码不代表反量化值的误差。结果写入summary.json。

## 合并五个已验证边界后的精度与Graph性能

`generate_graph.py` 从正式源码生成无诊断Out参数的两份kernel；`baseline`逐字节等于正式源码，
`aligned`仅将上述五类QKV舍入合并。O-proj、Indexer/Hadamard和参数适配器均不变。

```bash
python generate_graph.py --repo /path/to/vllm-ascend --output /path/to/graph-variants
python graph.py \
  --model /path/to/DeepSeek-V4-Flash-0731-w8a8 \
  --extension-dir /path/to/runtime/vllm_ascend \
  --sdk-root /path/to/simpler-sdk \
  --variants /path/to/graph-variants \
  --out-dir /path/to/graph-results/8186 --start-pos 8186
```

输出目录必须不存在；128K另以`--start-pos 131066`运行。沿用上述环境和确定性前提。
同一进程内捕获原生、baseline、aligned三个Graph，CSA不允许静默回退。
精度取warmup/capture后重置cache的单次replay，保存输出和六类cache有效写槽。
所有模型参数做不变性检查，输出guard检查在计时前后执行。

计时默认12轮×100次，轮换三个路径的次序，以NPU Event测每次replay均摊毫秒数。
每轮先重置cache；编译、重置和CPU取数均在计时外，没有诊断hook或额外输出。
这是固定metadata重复replay，**不能解释为真实请求连续推进100步或整模型吞吐**。
保存每轮时间、输出/输入hash和所有kernel源码hash；性能与单因素插桩结果分开报告。
