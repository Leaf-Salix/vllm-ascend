# 本轮报告独立复核（2026-10-20）

## 结论

审查版本：`REPORT_DRAFT.md` SHA256 `fb417911714a0168b4de29196793199cd6917a9a42f978e427dad635c1a2dcbb`。

**未发现需要阻止发布的身份、精度或性能过度结论。** 当前草稿已明确区分旧SDK设备结果、新SDK的symbolic lowering、完整无设备编译及尚待执行的设备门禁。尤其没有将PTOAS wheel的0.66版本等同于参考源码提交，也没有把SK1安装成功写成数值一致或已取得性能结果。

本复核只读取本地证据、执行CPU身份与统计检查，未修改报告/实现/SDK、访问远端或提交任务。新SDK信号任务的pending状态来自主线程提供的截止信息，本审查未查询队列，也不授予任何硬件通过结论。

## 已核实的证据关系

| 项目 | 独立核实内容与结论 |
| --- | --- |
| 正式版本 | 当前HEAD为`48c58ce7c467dfce63d5cb5c41afd225488d6012`，工作树clean。冻结完整编译收据的43份基础源与6份overlap源逐文件重哈均匹配当前仓库；不将这49份直接检查误称为重新独立检查全部77份。 |
| wheel身份 | `toolchain-wheel.json`记录实际wheel SHA为`d121b5b5db9a23a98eb3c0acc74d0abaef8aaa90e4df0385e79feb67636ef997`，独立venv、实际PTOAS_ROOT、二进制SHA和动态库解析路径均有记录；明确Git提交未知。报告没有将其归到`e5c1c432…`。本地未重新读取远端wheel或二进制，结论是收据之间身份一致。 |
| 三项完整编译 | signal、suffix、frozen三份`*-wheel-full-compile-v1.json`均为`skip_ptoas=false`、`ptoas_completed=true`、CCEC/编排完成、`device_execution=false`、`npu_initialized=false`；参数分别132/16/64。signal收据源码SHA与当前宽信号脚本重哈相同；suffix源码SHA与已审v8生成物记录相同。 |
| 旧lowering记录 | `frozen-overlap-lowering-v1.json`和`nal-resident-lowering-v3.json`均为`skip_ptoas=true`，前者64参，后者130参/单提交/sync_start。不能单独据它们宣称完整编译；草稿新增完整编译表实际由三份wheel收据支持。 |
| CPU门禁 | XML记录7项、0失败/错误/跳过。该范围是接口/ABI，不是设备同步或精度。 |
| 停止重复源码构建 | `ptoas-source-build-superseded.json`明确记录自身ninja PID、SIGINT、`source_build_completed=false`及改用wheel原因。报告保留“未完成”，没有把已完成LLVM等同于已完成PTOAS源码构建。 |
| SK0性能 | 既有独立原始复算为Native 940.8264923 μs、冻结PTO 1333.7268066 μs；比值复算慢41.7611874%。CANN trace分别924.898/1352.887 μs为独立三回放中位跨度，草稿没有替换正式12×100窗口计时。 |
| DFX范围 | 既有独立映射记录三窗各1219条、353 AIC/866 AIV、72核；1244.14/255.72/253.90/177.22/256.36/110.12 μs与草稿一致。跨度含等待与重叠，草稿没有相加或将空隙认定为纯调度。 |
| PMU范围 | 既有原始CSV独立复算的Scalar/MTE2/Vector中位计数与表一致。草稿明确executor聚合、计数重叠与host-flow关联失败，未将其解释成可独立消除的耗时或内部函数归因。 |
| SK1首差 | `sk1-8k-b16-v7.json`为complete=false/pass=false；实际安装194文件、reselect=1，输入与初始cache对SK0匹配但边界身份不同，计时前停止。后续SK0导出complete=true；其`SK1_vs_SK0`原始记录为raw KV 13、TopK 375、输出11755个差异，数值与表一致；其余列出的五组边界完全相同。TopK没有误用relative L2。 |
| 后缀候选限制 | v8实际PTO审查绑定相同源码SHA：单MIX、无额外GM分配、6处GM store、片上地址上限与草稿一致。range替换pipeline被明确披露。CPU/CCEC不证明设备计算、缓存可见性或无死锁，草稿已保留该限制。 |

## 发布时宜保留的两点

1. 证据索引应显式列出`frozen-wheel-full-compile-v1.json`。`frozen-overlap-lowering-v1.json`本身只证明symbolic lowering，不能作为“冻结完整入口已通过PTOAS/CCEC”的唯一链接。
2. 信号门禁即使随后设备通过，也只覆盖该132参通信/矩阵测试；不自动授予参考130参CSA、常驻后缀或新SDK整模型精度。报告当前“设备资格仍待验证”的文字应按实际新结果逐范围更新，不整体改成“新SDK全部通过”。

以上为证据使用边界和索引建议，不是当前草稿发现了新的数值缺陷。截图源提交、SK1实际每算子的命中情况、新wheel源码提交、新SDK设备数值与完整resident性能均继续保持未证实状态。

## 本次读取身份

| 文件 | SHA256 |
| --- | --- |
| `sdk-results/toolchain-wheel.json` | `4eaea51984f29be5d7cce81e849e4091db981d7b1c4237ee53ae44239f84b5e2` |
| `sdk-results/signal-wheel-full-compile-v1.json` | `a479489fbc0dd9dd97c30817d57e73c47a7b776b29dd36458b490f971b56656b` |
| `sdk-results/suffix-wheel-full-compile-v1.json` | `0744cb00c86a8f98ec46820e16883aee733c4c671832103a6ad2bbeb3a3266c4` |
| `sdk-results/frozen-wheel-full-compile-v1.json` | `1f360db423a272e151ef56d055ad15668e6c6dbb1d621119c60a277910f4ae64` |
| `sdk-results/frozen-overlap-lowering-v1.json` | `03fb68d6d1bfa818b31674a27abb16be7d3d3f2bfcbdc01482add34d012618d9` |
