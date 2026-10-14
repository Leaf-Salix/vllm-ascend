# TND精度版：不等长CSA整层FULL ACLGraph验证（2026-10-14）

分支 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，精度源码 `a9e0ae0d9`。
只选择 `deepseek_v4_flash_csa_tnd_precision`；性能包、框架环境与HCA计算不改。

## 测试范围与结果

这次验证真实CSA整层图，包含HC pre、输入Norm、CSA注意力、O-proj及HC post。
使用DeepSeek-V4-Flash-0731-w8a8正式模型第4层权重，hidden和历史cache为同种子合成数据。
TP1单卡，历史8191，B4请求长度 `[1,2,3,5]`，packed T11。
请求的query bounds为 `[0,1,3,6,11]`；输入只有11行，没有补成B4×S6的24行。

| 检查 | 结果 |
| --- | --- |
| Native两次独立图、独立cache | 9类观测边界逐bit一致 |
| TND精度版对Native | 同9类边界逐bit一致，0个差异元素 |
| 实际图capture/replay | 三路均持有独立AclGraph，raw replay与compiled调用逐bit一致 |
| 固定地址hidden A→B→A | B使输出改变，恢复A后所有观测状态逐bit恢复 |
| 合法cache写区与metadata只读检查 | 全部通过 |
| PyPTO执行与缓存 | 禁止Native回退；READY hit=1，生成/编译次数均0 |

9类边界为最终output、TopK、heads，以及SWA cache、压缩KV、压缩state、
Indexer key、Indexer scale、Indexer state。比较完整cache视图，另外检查allocation合法写入区域。
Native heads的 `[T,64,512]` 与PTO的 `[T,32768]` 只统一元素视图，不进行数值转换。

[完整原始JSON](evidence_tnd_precision/ragged_full_b4_t11_layer4_20261014.json)。

## 图与环境门禁

- Native和PTO均使用 `torch.compile(backend="npugraph_ex", fullgraph=True, dynamic=False)`。
- `force_eager=False`、`static_kernel_compile=True`、`super_kernel_optimize=False`。
- 实际记录AclGraph实例与底层graph，直接执行raw replay；不以context的FULL标签代替图证据。
- 禁止host updated nodes，保持graph输入和cache的生命周期，禁止精度路径回退Native。
- Native开启确定性level1、HCCL确定性；不启用atomic add。
- 沿用CANN9.2 beta2、Torch2.10.0、Torch-NPU2.10.0.post2、vLLM0.25.1及既有私有PyPTO/Simpler。
- 每个NPU任务上限780秒，本次单项约3分钟；未运行128K。

## 观测脚本修正

第一次任务在三路图均完成后，因为heads形状不同而在统计函数退出。
已保存快照按相同元素顺序展开后，9类边界全部逐bit一致。
随后只修正比较视图，重跑同一配置，正式任务exit0且完整PASS。
未将首次异常任务记为完整PASS，也没有修改精度算子来绕过检查。

## FULL含义与限制

这里的FULL是上述CSA整层真实完整ACLGraph，不是整模型vLLM调度器的ragged FULL验收。
当前DSA builder在 `dsa_v1.py` 的类属性及 `get_cudagraph_support` 中仍声明
`AttentionCGSupport.UNIFORM_BATCH`。因此不能据本次单层结果声称现有整模型调度器
已支持不等长FULL；没有为测试盲目修改后端声明。

本轮不测性能，不推导整模型token精度或全部中间tensor逐bit一致。
此前192token验证未观测到同一步不等长请求，本页补充的是实际packed TND的图内证据。

## 同地址query bounds更新：首差定位

A=[1,2,3,5]，B=[5,3,2,1]，均T11。额外捕获独立NativeB图作为B的参考，
初始hidden/cache与A逐bit相同，cache分配独立。只复制捕获输入tensor内容，
保留图的tensor地址和Python scalar/list捕获值；后者的A/B差异单独列入JSON。

首轮Native1原图切换B后，9类边界与独立NativeB全部逐bit相同，保护区通过。
恢复A时出现首差，按要求立即停止，尚未执行Native2/PTO的bounds更新。
差异仅出现在SWA的RoPE部分、TopK、heads与output；压缩KV/state及Indexer缓存/state仍一致。
[首差原始JSON](evidence_tnd_precision/ragged_bounds_rope_restore_failure_20261014.json)。

源码定位：`rope_dsv4.py::get_cos_and_sin_dsa(use_cache=True)`在图外更新共享
`runtime_buffer`，metadata.cos/sin通过`RopeDataProxy`读取它。诊断脚本只遍历普通tensor、
dataclass和dict，没有恢复该代理背后的共享RoPE行；独立NativeB的构造已覆盖它。
恢复positions不等于恢复当前RoPE行，不能把这次观测差异归因于TND算子或Native数学错误。

诊断修正按Native流程在每次B/A重放前刷新当前RoPE行：用`use_cache=False`得到独立期望，
调用官方`use_cache=True`更新缓冲，硬检查cos/sin逐bit一致且原地址未变，
同时把cos/sin加入只读保护。正式算子源码未改。复测结果在后文追加。

## 同地址bounds复测结果：PASS

补齐Native输入准备后，Native1、Native2、PTO三张原图都完成A→B→A：

| 图 | B对独立NativeB | 恢复A对原A | B/A保护区及只读输入 |
| --- | --- | --- | --- |
| Native1 | 9类边界逐bit一致 | 9类边界逐bit一致 | 全部通过 |
| Native2 | 同上 | 同上 | 全部通过 |
| TND精度版 | 同上 | 同上 | 全部通过 |

每次更新均保持捕获tensor地址不变，cos/sin由Native官方流程准备，不重新capture原三张图。
请求长度变为B后，输出确实不同于A，排除了无辨识力输入或陈旧输出。
Python `seq_lens_list` 等捕获值保持A不变，差异在证据中明确记录；不声称这些Python字段已更新。

直接观测也证实首轮根因：三路恢复A前，cached cos/sin均不匹配A的位置；
调用官方刷新后全字节匹配，随后所有cache/heads/output恢复逐bit一致。
正式精度kernel没有改动；这次修复的是测试输入恢复遗漏。

[同地址bounds图重放原始JSON](evidence_tnd_precision/ragged_bounds_full_b4_t11_layer4_20261014.json)。

## 第26层不同尾块复测：PASS

history8189，B4 query lengths=[1,2,4,6]，query bounds=[0,1,3,7,13]，packed T13。
同样不对每请求补S6，使用真实第26层权重和独立cache，Native开启相同确定性。
三路实际完整图capture/replay、hidden A/B/A、合法写区/readonly门禁全部通过；
Native两次与PTO对Native的上述9类边界全部逐bit一致。
PyPTO READY hit=1、generation_builds=0、binary_builds=0。

[第26层T13原始JSON](evidence_tnd_precision/ragged_full_b4_t13_layer26_20261014.json)。

本轮NPU任务全部结束，每项约2–3分钟，上限780秒，未运行128K。
中间一次RoPE恢复修正版因诊断脚本引用Native层名的层级错误在fixture准备时退出，
未进入capture；改用原生cache group提供的compressed prefix后复测通过。
所有失败及修正均未放宽逐bit或图门禁，正式精度和性能源码均未修改。

## 本轮结论

| 层/历史 | 请求长度 | packed T | 结果 |
| --- | --- | --- | --- |
| layer4 / 8191 | [1,2,3,5] | 11 | 9类边界逐bit一致 |
| 同一layer4原图切B后再恢复A | [5,3,2,1] → [1,2,3,5] | 11 | Native两图与PTO图均逐bit一致 |
| layer26 / 8189 | [1,2,4,6] | 13 | 9类边界逐bit一致 |

已实测证明：在这两层、这些不等长输入下，TND精度版的真实完整CSA图不需要
每请求padding到S6，即可与确定性Native对齐。覆盖请求长度1–6、不同T尾块及同址bounds更新。
没有据此外推所有层/形状，也没有宣称整模型调度器的ragged FULL或不等长192token已验收。
