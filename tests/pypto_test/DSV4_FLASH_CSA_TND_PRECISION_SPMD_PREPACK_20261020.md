# TND精度＋SPMD：固定HC权重预打包（2026-10-20）

## 改动

保留原完整精度版及e978的SPMD实现，当前分支只新增
`deepseek_v4_flash_csa_tnd_precision_spmd_prepack/`最初五个数学与绑定文件，并补充一个仅导出归约常量的模型加载兼容文件。
复用已验收的数学与归约、舍入顺序，把每次forward中对固定HC权重的DSL打包移到加载时执行一次。
使用独立namespace `dsv4_csa_tnd_precision_spmd_prepack`，64参数根的第二参数改为只读BF16打包权重；
其余计算顺序与原完整根一致。每层常驻2MiB，21个CSA层共增加42MiB常量；
每次完整根不再申请这个2MiB临时权重缓冲。

数学仍全部使用现有DSL，没有手写C++、SDK修改、依赖重装或隐式权重再量化。
前缀沿用原Cube，Vector只删除固定pack及对应返回；MIX描述符31个，24组，双方各十次handoff。

## 单因素前缀性能

真实C4权重、合成hidden；直接ACLGraph replay；不等同整模型或完整CSA延迟。
每组12个平衡配对窗口，每窗口25次replay；保留全部样本，原完整2MiB打包字节也逐bit比较。

| 前缀行数 | 原SPMD中位数μs | 预打包中位数μs | 延迟减少 | 精度验证 |
| --- | --- | --- | --- | --- |
| T21 | 255.299 | 223.958 | 12.28% | 491项全部逐bit一致，12窗口全部变快 |
| T240 | 399.242 | 361.178 | 9.53% | 491项全部逐bit一致，12窗口全部变快 |

两任务分别为`task_20261020_021518_41710515950`及`task_20261020_021748_8268124546`，exit0。
测量对象是当时独立诊断前缀；正式五文件保留同一数学，完整入口另做下述门禁，不混淆源身份。

## 完整根FULL graph逐bit验证

初版五源任务`task_20261020_024550_121354818222`和当前六源任务
`task_20261020_030304_19005342939`均exit0；加载上下文修复后当前六源任务
`task_20261020_031623_243684218183`也完成、exit0。Native开启确定性，使用真实L4权重、
history8191和合成hidden/cache；请求长度`[1,2,3,4,5,6]`，真实21行，无每请求补S6。

- A/B两个输入，各8个输出/cache/TopK边界全部逐bit一致。
- 原始输入只读检查、输出和工作区poison、guard、A/B/A后复核通过。
- T21捕获后在另一计算stream暖身T6，8边界一致；随后捕获T6。
- 三轮T21→T6→T21交替replay，72项边界检查全部逐bit一致。
- 完整2MiB只读权重与Norm、order输入保持原始字节，未把只读常量列为poison工作区。

正式根和准备函数分别通过program、kernel ABI/CCEC及真实Torch注册；
同时验证64参数attention和2参数prepare schema。编译时发现格式化工具会把嵌套DSL上下文合并，
当前通过显式保留嵌套和对应CPU门禁修复，没有跳过解析检查。

证据在`evidence_tnd_precision/spmd_prepack_20261020/`，包含原前缀归档、完整根结果、
最终编译receipt及实际helper/manifest。最终源码哈希绑定六文件及原57运行文件；五源历史结果单独保留于`five-source-history/`。

## 复现和使用限制

与[SPMD整模型基准](DSV4_FLASH_CSA_TND_PRECISION_SPMD_REAL_MODEL_20261020.md)使用相同环境。
独立variant为`pkg:deepseek_v4_flash_csa_tnd_precision_spmd_prepack`，不改变原默认精度路由。
使用证据内的`run-full-csa.sh <新输出名> 1,2,3,4,5,6 4 8191`，
需要原模型、静态包和独立实验目录；环境及依赖身份先通过receipt检查。
任务调度上限780秒，内部timeout750秒。

加载之后权重不可变。普通Tensor的对象/地址/形状/版本变化在Python入口拒绝；
inference Tensor原地写入、`.data`写入及直接graph replay不能提供完整突变检测。
重载权重必须重建runtime和graphs；不支持动态热更新或并发stream使用同一工作区。

首次8K/B40整模型任务`task_20261020_025114_14369392090`在模型初始化退出，未生成token：
模型加载器按variant导入`qkv_proj_rope`读取`QR_OK/KV_OK`日志常量。
当前补充的兼容文件只重导出父模块两常量，没有复制或修改Q/KV数学；
新增CPU及无设备实际导入门禁，六源完整FULL已复测通过。
第二次8K任务`task_20261020_030701_20494019475`同样在加载期退出：
加载器启用梯度且Native HC Parameter带`requires_grad=True`，推理专用准备函数正确拒绝。
当前在单次准备调用外显式使用`torch.no_grad()`，不改权重和外部grad状态。
新增CPU及真实NPU强grad加载回归，源权重FP32 SHA与修复前相同，后续FULL逐bit门禁全部通过。
第二次失败没有token结果，原六源记录保留于`six-source-before-nograd/`。
当时整模型尚未完成；后续新快照的验收结果见末节。
只有8K token/接受事件一致后才执行128K/B24；任何首差都停止扩展并保留原精度基准。

## 后续整模型验收

当前六源已完成8K/B40及128K/B24，两档所有保存token、接受事件均与Native一致，零抢占。
性能均值仅较父SPMD小约0.4%～0.5%，不认定稳定整模型加速。完整配置与原始证据见
[真实整模型报告](DSV4_FLASH_CSA_TND_PRECISION_SPMD_PREPACK_REAL_MODEL_20261020.md)。
