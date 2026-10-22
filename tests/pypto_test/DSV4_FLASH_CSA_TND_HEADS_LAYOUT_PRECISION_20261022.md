# TND Heads布局优化：128K/B24精度补验（2026-10-22）

## 结论

**Direct和Packed均通过128K/B24单层FULL图精度补验，与同输入Native逐位一致。**
上轮因预设1%性能筛选门槛跳过精度；用户认可约0.6%的改善，本次独立补测，没有改写上轮历史结果。

| 候选 | 既有8K/B16完整图耗时降低（两轮） | 既有128K/B24耗时降低 | 8K精度 | 本次128K精度 |
| --- | ---: | ---: | --- | --- |
| Direct：直接写Native heads | 2.02%～2.07% | 0.62% | 上轮两次A/B八边界逐位通过 | A/B八边界逐位、ABA及三次独立重放通过 |
| Packed：WO-A直接读分组heads | 2.24%～2.35% | 0.60% | 上轮两次A/B八边界逐位通过 | A/B八边界逐位、ABA及三次独立重放通过 |

性能沿用[上轮实测报告](DSV4_FLASH_CSA_TND_HEADS_LAYOUT_PERFORMANCE_20261022.md)，本次不重新计时，也没有以性能门槛筛掉任一候选。
两候选依然互斥，不能相加收益。默认精度基准与SDK保留，本轮未将布局候选合入默认算子。

## 验收范围

真实DeepSeek-V4-Flash-0731-w8a8第4层权重、合成hidden/history，单层完整HC-pre→HC-post。
history131072、B24、全S6、真实及物理T144，FULL raw graph，TP1、page32、NZ2、seed1024。
Native确定性Torch/NPU level1/HCCL开启；SK0、13个官方静态包实际安装/reselect，DSA多流。

Native、冻结、Direct、Packed逐arm执行；前一arm出现首差立即停止，不继续后续候选。

1. 检查独立cache/state的初始值、hidden/positions、metadata与compact RoPE同输入。
2. 比较编译图与raw graph的八边界；输出NaN、TopK毒化后必须由重放恢复。
3. A输入对同一次Native比较输出、TopK及六类cache/state，共八边界。
4. hidden加0.125形成B输入，确认输出随输入改变，再做八边界比较。
5. 恢复A、恢复cache并重放，与本arm先前A逐位比较。
6. 每arm再做三次独立reset、输出/TopK毒化、raw replay；八边界均与已验A逐位相同。
7. 检查33项guard/metadata及候选readonly权重包，实际PTO路由禁止回退，外部根仍为64Tensor。

结果：冻结、Direct、Packed的A/B对Native共48项边界比较全部逐位一致；四arm的ABA、compiled/raw及毒化恢复通过。
四arm额外12次重放共96项状态一致检查通过，重复重放的保护区与readonly检查通过。
Native自身的重复一致性也通过，未发现数值首差。

**这证明的是上述单层、输入及graph配置，不是新的整模型192-token或任意请求精度证明。**
新候选的短query、混合1～6行以及真实整模型8K/B40、128K/B24 token一致性尚未新增验收。
两候选的布局计算仍使用真实T及原生请求边界，测试全S6不等于实现要求每请求补S6。

## 源码与环境身份

测试延续分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，补验开始HEAD为`0e6fc13cc09569c4dd248bd213bb115fffe9b45b`。
冻结数值源码为`ed2b2e5c3`，QA168修正`ad06a423b`；canonical3399清单`48ee0351…`、77运行源`e0b25682…`、QA`ea30b2de…`保持不变。
四份候选源、SOURCE绑定、原两份无设备编译收据及环境与上轮逐字节相同，本次只新增精度助手。

| 候选源 | SHA256 |
| --- | --- |
| heads_direct_sparse.py | dc98a527e81929c36e6613f04d39df27ba212adcb64004873a83051b05013155 |
| heads_direct_root.py | f432a79f334724547fe22e3ae91f2f3a5811e8b7a72c2a26c82c8697dbbf74e5 |
| heads_packed_oproj.py | 6d41aaad3c9ab5283c394dfe78c200090f9e682726c4e3658d5d2b5b9b207bd6 |
| heads_packed_root.py | 1d1204c3e949a48a1293a1187966a3682178bd5a21fc8b9b06442bfcbe4ce1b7 |

复用原Python3.12、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、CANN9.2beta2及原PyPTO/Simpler/PTOAS环境。
精确导入路径与编译收据由无设备预检、实际任务preflight和封存manifest校验；未重装SDK或vLLM，未修改runtime或编译生成的C++。
正常kernel JIT/cache使用保留，DFX和dep-gen均关闭；本次不是性能采样。

## 任务、复现与证据

任务`task_20261022_115146_399542431884`，单卡设备8，completed exit0，锁已释放。
入口2026-10-22 11:52:05至11:56:22，257秒；外限780秒、内限750秒。
提交后持续观察，没有后台watch或定时自动化。

[补验证据及CPU重算](evidence_tnd_precision/heads_layout_precision_20261022/README.md)包含实际输入配置、完整助手、源码SHA、15文件manifest、实际preflight、逐边界结果、重放检查和终态日志。
归档不含权重、cache payload、失败Tensor或构建二进制。

在既有验收环境中，将工具还原到独立目录，先无设备运行preflight；使用新case标签，避免覆盖历史输出：

```bash
python task.py submit heads-precision-128k-b24-v2 24 131072
python task.py observe heads-precision-128k-b24-v2
python collect.py heads-precision-128k-b24-v2
```

实际命令的ASCEND_HOME_PATH、PTOAS_ROOT、PYTHONNOUSERSITE、CCACHE_DIR和四路编译配置见submission及env.sh。
环境、3399/77源码、原编译收据不符就停止；不得通过重新安装或覆盖原SDK来绕过身份检查。

## 取舍

用户认可小幅但一致的性能改善后，两方案均完成本轮单层8K/128K精度补齐，可以继续作为优化候选保留。
Direct改动更少，8K收益与Packed接近，128K略高；后续整模型候选可优先Direct。
当前尚不能将单层逐位结果写成“整模型token验收通过”，也不把历史门槛未达解释成精度失败。
