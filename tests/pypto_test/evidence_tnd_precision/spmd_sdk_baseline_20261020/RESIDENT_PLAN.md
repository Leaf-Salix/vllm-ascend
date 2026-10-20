# 对齐10月20日动态SPMD效率：实现与验收方案

## 已知事实

- 精度基准固定为当前分支 `48c58ce7c467dfce63d5cb5c41afd225488d6012`。本轮未修改其77份运行源。
- 用户截图目标已逐项保存在 `USER_BASELINE.json`；它是单层真实第4层权重、合成hidden/KV、均匀S6的完整HC-pre到HC-post测试，不能与整模型forward直接比较。
- nalinaly远端最新固定提交 `beaefa6f6788a9448d49293238f6818c246f24f5`；尚未确认截图是否恰好对应此提交。其最新动态入口也能表达真实请求边界。
- 已读源码表明：参考是完整CSA的24组常驻MIX，当前精度版只是前缀MIX加任务后缀。这是结构差别，尚不等于所有耗时差的因果证明。
- 已验收SDK的 `signal_notify/signal_wait` 实测缺失；参考完整根130Tensor、内核helper132Tensor。现SDK不能直接承载同一入口，见 `CAPABILITIES.json`。

## 迁移原则

借鉴调度、owner与供数方式，保留当前Native精确数学；不整体复制参考的数学helper。

| 模块 | 应迁移的调度 | 必须保留的精度合同 |
| --- | --- | --- |
| HC→Norm→投影 | 按真实行owner发布，不等待无关全核；Sinkhorn与QA/WKV并行 | 既有HF32等价分解、Native K与split顺序、left-fold混合、BF16落点、4096 fold树及精确sqrt/标量除法 |
| 主Q/KV与Indexer | 独立链分配AIC/AIV，及时发布本行结果 | QA126及WKV/Indexer原生K顺序、所有BF16/FP16转换，不隐式重算/转换权重 |
| Metadata与cache | 同resident任务内由独占AIV产生，消费者只等实际owner | Native Q/L/positions/slot/compact RoPE，真实1～6行与padding sentinel语义 |
| Score/TopK | 按请求复用Key、保持环槽READY/ACK；按实际producer发布TopK | 原score计算、系数舍入、稳定tie与TopK索引，不以当前参考PTO结果代替Native金值 |
| Attention | 共用KV供数与双角色流水，布局/预取独立验证 | Native K512、概率舍入与sum树、归一化先BF16再inverse RoPE；不能改回K128或FP32直接逆RoPE |
| O-proj→HC-post | 在同任务按输出组调度，反量化与post共用行owner | WO-A已标定K起点；先拼接8192列统一量化，不能每组1024独立amax；保留attention_out BF16边界和post顺序 |

## 工具链方案（用户已批准独立兼容工具链）

只在227自有独立目录准备兼容的PyPTO/Simpler/PTOAS，优先使用明确提交与可复现构建。旧venv、SDK、源码、编译缓存及金值保持。
先证明：

1. phase notify/wait的发布、获取、DCache与FIX/MTE3语义成立，不能用普通read/write轮询冒充。
2. 132Tensor resident helper可编译，根ABI与当前Torch接入明确；若根仍受64上限约束，先完成typed workspace容量/别名设计，不能绕过检查。
3. 保存依赖提交、实际import、二进制SHA、PTOAS生成器与CCEC输入；算子数学保持DSL。不默认启用参考中可选C++等待重写。
4. 无设备lower/ABI/UB-L1资源检查后才上最小硬件任务；不重编整个vLLM。

用户于本轮明确批准：允许准备独立兼容工具链，保留原环境。新工具链仅放在 `/data/pyptouser/yejia/csa-spmd-sdk-20261020`；当前验收环境和源码不变。17899代理未监听，采用本地下载与rsync传输。

## 逐步落地与停止条件

- 保留当前精度版作为control；同当前分支增加独立文件/variant，不切换分支。
- 先移一个阶段与其producer/consumer信号，检查新旧同输入逐bit、poison、guards、A/B/A、跨桶/跨流生命周期。任何首差先定位，不扩大配置。
- 优先检查本轮DFX确实较重的QK/PV、O-proj以及HC前缀；阶段跨度包含内部等待，且不同阶段可重叠，不相加或误称纯计算。
- 参考与候选必须统一输入、权重、metadata、历史cache、NZ2、确定性与计时跨度；截图原样单独列，不能靠换SK0/均值口径宣称追平。
- L1计时、dep-gen和PMU分任务；无插桩完整graph测median及全部窗口分布，保留离群值。
- 有明确完整CSA收益后，按已验收固定bank先8K/B40，再128K/B24真实整模型192token；逐请求token和接受事件必须与既有Native金值一致。
- 128K每卡提交24、测量实际并发21的口径保持，不因吞吐对比改变入场轨迹。
- 每个NPU任务内750秒、外780秒，拆小提交，不提交连续长占卡任务，不跑Native2。

当前没有新的数值实现，也没有达到截图效率的结论。若只允许保留现SDK，则可验证小范围mix/Norm或O-proj尾部融合，但不能预先承诺它们能弥合约一倍的目标差距。
