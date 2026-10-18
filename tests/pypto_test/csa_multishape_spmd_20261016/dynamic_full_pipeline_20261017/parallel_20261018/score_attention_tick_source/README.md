# Attention typed-Tile 完整 tick 源码准备

2026-10-18。`source_candidate/attention_tick_stages.py`已从冻结的“两L1、PV先行、GM三槽”源码抽取。**本模块尚未PyPTO解析/lowering、设备验证或性能测试**；不修改正式ops/依赖，不接Score，不写wave控制。root已报告独立普通Tile接口CPU通过，但动态valid返回仍有边界；这不等于本模块联合编译已通过。

| helper | 初始化/执行内容 | 返回顺序 |
| --- | --- | --- |
| `attention_cube_init` | 原AIC初始化、set_ffts、真实T/本核块数、Q和KV两槽Mat | T, core, block_count, Q Tile, KV Tile |
| `attention_cube_tick` | 原完整AIC tick，先PV(tick−2)再QK(tick) | score GM, PV GM, Q Tile, KV Tile |
| `attention_vector_init` | 原AIV初始化、lane偏移、只读swap、m/l/oi | T, core, block_count, lane_head, lane_kv, swap, m, l, left, right |
| `attention_vector_tick` | 原gather/softmax/在线归并、末块归一化/RoPE/发布和reset | KV GM, probability GM, mi GM, li GM, packed GM, m_after, l_after, left_after, right_after |

GM参数和显式返回沿原Tensor/InOut合同，未加入新的GM或runtime根。AIC传递Q BF16[64,512]Mat、KV BF16[256,512]Mat；AIV传递m/l FP32[32,1]Vec、left/right FP32[32,256]Vec、只读swap INT32[1,1024]Vec。**仅限固定完整valid形状**，不把动态tail Tile当状态传入；full-valid不意味着KV槽内容已经初始化。真实T、Q/L、页表、positions、有效mask继续动态，EARLY_KV只使用完整T，不使用wave长或padding容量。

调用方必须保留原推进范围：Cube tick∈[0,block_count+2)，Vector tick∈[0,block_count+3)。query公式仍是`qk_core+(work//5)*CORE_GROUPS`，没有局部重编号。两L1槽只用于AIC Mat%2，所有GM地址仍%3；覆盖依赖须由新联合lowering证明，不能按Scalar函数返回认定DMA完成。两个AIV与AIC要按同一mask/FFTS域推进，普通phase信号不能替代槽READY/消费协议。

`source_audit.json`的纯标准库AST还原检查全部通过：

- 两init的数学/分支/调用序列分别等于原外层loop前缀。
- Cube tick正文逐AST相同；Vector tick正文只将原最外loop最后一个`pl.yield_`移为显式返回，所有内部if的yield及数学顺序不改。
- 模块导入/常量、每块分支/同步调用、原GM返回保持；四Tile显式返回、swap不赋值；未留下依赖原loop局部变量的悬空名称。
- 参数/返回新增及外层range移出是明确结构变化，单列于JSON接口清单。未声称函数签名或整个原stage AST没有变化。

独立可复现脚本：

```bash
python /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/score_attention_tick_source_20261018/prepare_and_audit.py
```

参考原文已保存到`source_reference/attention_stream_stages.py`，`candidate.patch`列完整差异。脚本不导入PyPTO，不覆盖不同的冻结候选。首次源码准备脚本曾因嵌套三引号触发Python SyntaxError，见`source_preparation_failure_v1.txt`；修正的是脚本字符串表示，未发生后端编译，也没有掩盖或丢弃设备结果。

下一阶段仍须验证typed-Tile多返回/布局、Mat回边、UB/Left/Mat联合生命周期以及原同步依赖；本轮不自行扩大验证范围。
