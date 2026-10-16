# TND完整PTO精度恢复：第一步HC-post（2026-10-16）

## 目标和当前状态

分支 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
起点commit `5d5ab1c6c06c4d23c08a01c85d7c87ef97b64bc1`，原精度计算基线
`cbc810c8afab47c2425068049440424f014dcde3`。

替换范围以nalinaly实现为参考：HC-pre到HC-post整个attention半层。
计算、归约、量化及舍入顺序以实际Native vLLM-Ascend为基准；
不把nalinaly归档精度版作为数值黄金。保留真正TND，不要求每个请求padding到S6。

本检查点新增显式候选 `PTO_CSA_VARIANT=tnd_full_precision`，**只恢复HC-post为PTO**。
原 `tnd_precision` 包23个文件逐SHA保持不变，默认variant与原性能实现保持不变。
候选继承原精度版的TMR准入，HBG仍明确拒绝；FULL图准入包含变长query。

| 模块 | 原精度基准 | 本检查点候选 |
| --- | --- | --- |
| HC-pre、输入Norm | Native | Native，尚未恢复 |
| Q路径 | PTO | 复用原精度PTO |
| WKV投影 | Native | Native，尚未恢复 |
| KV后处理、compressor、Indexer主体、attention | PTO | 复用原精度PTO |
| Indexer权重投影 | Native | Native，尚未恢复 |
| O-proj | Native | Native，尚未恢复 |
| HC-post | Native | **PTO，直接写调用方output** |

不能将新包名字解读为完整PTO恢复已经完成，也不能把历史混合实现的整模型精度/性能
直接移作新候选的验收结果。当前没有新整模型或性能测量，不运行128K。

## HC-post算术依据

实际Native模型调用 `torch.ops._C_ascend.npu_hc_post`。
当前A3源码 `csrc/moe/hc_post/op_kernel/hc_post_d_split.h` 的DSplit路径：

1. BF16 attention output与四路residual扩展FP32。
2. 单独计算 `post * attention_output`。
3. 按residual lane 0、1、2、3分别乘comb并顺序相加。
4. 最后RINT到BF16。

复用原精度包中此前未接入的 `hc_post_test`；没有改动原包。
生成PTO代码核对为TMULS、四次顺序TADD和最终TCVT RINT，没有FMA。
另一个架构C310的计算路径不用于本次结论。
本地源码对照不单独证明已安装Native二进制身份，因此还以实际Native调用做设备黄金。

无设备签名/lowering及CCEC编译通过，ELF 252,928字节，SHA256：
`efe77c37088ff418a8d857b807c14ea8dc0989f5c8dd7c0743e04397ddf18ee8`。
9项CPU variant、atomic0及变长准入检查通过。

## 实测方案与结果

所有任务单卡、确定性level1、HCCL确定性开启，780秒任务上限。
沿用精度复现手册的Python3.12、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、
vLLM0.25.1、CANN9.2.0-beta.2、ATB9.2.0-beta.2、PTOAS0.66、
私有PyPTO3e87a843基线包及Simpler a54c0509，没有重新安装框架。

### 独立HC-post

T11、seed17，合成BF16 attention/residual及FP32 post/comb。
先比较实际Native两遍，再比较PTO eager。
分别捕获Native/PTO原始NPU图，各5次replay：A、仅post变化、仅comb变化、
全部输入B、恢复A。每次与同输入fresh Native比较；输出先毒化NaN，
检查全部输入未被改写和输出前后保护行，变化必须实际影响输出。

**14项逐bit检查全部通过，差异元素0、max_abs=0；保护区及A/B/A通过。**
此项只证明本次模块输入，不是模型层或整模型结论。

### 真实第4层权重、完整attention半层FULL图

- 正式DeepSeek-V4-Flash-0731-w8a8第4层权重；hidden/history为合成输入。
- history8191，B4，请求真实长度 `[1,2,3,5]`，T11，原生query bounds `[0,1,3,6,11]`。
- 四臂：Native1、Native2、原精度基准、HC-post PTO候选；独立cache allocation、相同初态。
- `torch.compile`的npugraph_ex后端、`fullgraph=True`、真实AclGraph，static编译，SK关闭。
- 实际持有四张独立图，无host updated nodes；compiled调用与原始graph replay逐bit一致。
- A、hidden+0.125的B、恢复A，各次replay后立即检查保护区，再恢复状态。
- A/B分别比较output、TopK、六类cache/state和attention heads，共9类张量。
- 候选运行时将Native `layer.hc_post`替换为抛错函数；禁止fallback，确认实际执行PTO HC-post。
- 每项首差保存原值并停止后续臂；warmup后generation/binary build及cache错误增量必须为0。

**A/B两组×三臂×9类张量=54项全部逐bit一致，差异元素0、max_abs=0。**
四图各3次原始replay、A/B/A恢复、即时保护区检查全部通过。
本项没有更新metadata做A/B/A，没有整模型调度器、MoE、DP16或所有CSA层结论。

## 保留失败历史

真实层v1在候选执行前失败，不能用于候选验收：

- 旧门禁要求disk `ready_hits>0`，但首次warmup编译后capture合法复用了内存artifact，
  `object_hits`增加且没有新增编译。按实际persistent缓存实现，改查object/ready hit之和及build增量。
- 长输出路径导致CANN静态编译shape_info文件名超过文件系统限制。
  v2使用短输出根和短TMPDIR，并拒绝日志中的 `execute op_compiler error`。
- Native1/2和原基准虽已通过A/B比较，v1整体仍记 `pass=false, complete=false`。
- CPU第一次收集缺相邻测试模块搜索路径，修正后通过；不记作算术失败。

| 任务 | 结果 |
| --- | --- |
| `task_20261016_152931_19731526512` | 独立模块exit0 |
| `task_20261016_153836_44644518673` | 真实层v1 exit1，候选未执行 |
| `task_20261016_154556_7312905013` | 真实层v2 exit0，54项逐bit一致 |

三项均已完成，没有本检查点遗留任务。

## 可复现材料

[原始55文件归档](evidence_tnd_precision/full_pto_20261016/hcpost-evidence.tar.gz)，SHA256：
`35d13bca61505216a5f788d4d434f45a5a8058925b3477d295eb464333eec01b`。
内含每个成员SHA、原始脚本、成功/失败JSON与日志、编译信息、生成CPP、
基准23文件SHA、3353文件新源清单和相对旧冻结源的18项overlay（含5项pytest缓存）。
不含权重、cache张量、venv或大失败快照。

1. 按[精度复现手册](DSV4_FLASH_CSA_TND_PRECISION_REPRO_20261016.md)恢复环境和原3337文件冻结源。
   旧源清单SHA为 `e4de4927d2e5758df65784bddc412d1d90227b56ed978ad2218df50c2bd540ca`。
2. 核对归档SHA，解压到新的证据目录，逐项校验 `evidence-manifest.json`。
3. 将旧冻结源复制到新的实验目录，覆盖归档 `source-overlay`，按
   `step01/frozen-manifest.json` 核对声明的源及helpers。运行期间禁止改源。
   原始新清单SHA为 `0d89617efd50061759963947aef3a2a435644ce2393fab93a7d62d5613aee56a`。
4. 参考归档 `step01/env.sh`、`compile_hcpost.py` 和 `check_frozen.py` 做无设备导入、
   基准SHA、toolchain和lowering检查。路径必须对应新的隔离目录；更新helpers后另存新身份清单。
5. 单独运行模块probe，再单独运行 `step01-layer-v2/run-layer.sh`。
   输出必须不存在，使用短路径；模型/配置/编译选项/数学代码不变。
6. 使用 `task-submit --ptoas 0.63 --device auto --max-time 780 --env PYTHONNOUSERSITE=1
   "bash <run-script> <new-output-label>"`；env.sh显式选择实际PTOAS0.66。
   两项任务分开提交，先确认前项通过。约5分钟查status，不使用wait或自动调度。
7. 任何首差停止，归档原值和失败报告；不继续更大配置。这里只复现HC-post检查点。

独立审核确认55项SHA、14项模块比较及54项层比较；核对默认variant和23文件原包未变。

## 后续：HC-pre独立定位

暂不改输入Norm。实际Native `npu_hc_pre_v2` 的A3路径包含HF32向零转换、
形状相关的split-K、K32 Cube累加、顺序partials归约、Sqrt后Div、
sigmoid的Div和四路顺序加。旧PTO相应分组/倒数写法存在差别。
先建立实际Native三输出黄金及中间量观测，再做单因素修正；
不能仅凭截断FP32权重就假定普通FP32 MAD与Native HF32 MAD逐bit等价。
