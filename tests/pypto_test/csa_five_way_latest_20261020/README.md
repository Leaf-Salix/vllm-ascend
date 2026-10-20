# 七档 CSA 五口径最新实测

用户要求重新提供 Native、原 PTO 多 task 含 AICPU、多 task L1 incore 首尾、
静态单 SPMD、当前正式动态单 SPMD 的七档对比。此处只做测量和记录；
既有性能优化目标保持暂停，不更改正式算法，不自动 push。

规格：128K/B4、B8、B16、B24、B28；8K/B16、B32，均 S6。正式 W8A8
第 4 层权重、TP1、seed=1024 的合成 hidden/KV；范围为 HC_pre、输入
RMSNorm、CSA attention、HCpost，无 MoE/整模型/DP16 验收。

Native 必须实际完成 npugraph_ex fullgraph、dynamic=False、静态 kernel
编译/安装、SuperKernel 优化和 ACLGraph 重放，并保持 Native DSA 多流。
报告记录成功返回值和真实图实例，不能只根据环境开关宣称生效。
所有路径使用 NZ2；Native 和 PTO 四张核心投影权重实际格式必须为29，
PTO根签名和绑定另检查NZ存储。NZ指支持该布局的矩阵权重，并非将所有
activation、metadata或KV Tensor都改成NZ。

每档由一个队列任务在同一卡上依次测量：Native、PTO外层、当前动态与
多task配对L1、静态与多task配对L1。前两项各30次PyTorch Profiler全流
设备首尾跨度，包含图内复制与AICPU，排除图外reset；20次非profile
Event另存，不混作主结果。后两项各5个AB/BA轮换窗口，L1从最早incore
开始至最晚结束，保留中间等待，不累加相互重叠的任务。

列之间仍有外层/核内口径差异；用户要求的两项比例统一为
`(baseline_us - dynamic_l1_us) / dynamic_l1_us * 100`，不能称为同口径
整服务加速比，不能将外层减L1当作精确纯AICPU成本。

当前正式动态源码复制到source_dynamic以避免测量中途改变；数学版本为
13259f92，当前仓库HEAD见PROVENANCE。旧多task、Native与静态源来自
当前ops包，独立工具链固定为10/19复现树，未加载Key多ND负例。
128K/B28没有已验收静态实现，保持空值，不修复旧静态代码，不以邻档替代。

测试脚本只参数化原Native七档、增加NZ实际格式断言，并允许原L1测试壳
使用已有双角色配对路径。Native forward AST和计时器与原已验证代码一致。
基础精度、状态、保护、图和代际检查保持；不重复全套动态边界测试。
任务ID及终态保存TASKS.json；失败阶段另用新attempt目录，禁止覆盖证据。

所有NPU工作均通过task-submit --device auto。此目录不包含权重/二进制，
大编译与trace原始产物本地保留，后续归档只登记路径和字节，不做hash。

## 终态（2026-10-20）

七个正式测量任务全部completed/exit0，Native、PTO外层、多task/动态L1七档
及静态L1六档均通过各自检查。完整表与原始样本见PERFORMANCE.md/json/csv。
Native各档有真实sk_*static_kernel、MatmulWeightNz trace、静态包安装、
SuperKernel成功返回、唯一ACLGraph及实际格式29证据。

初次两个提交因队列追加--device被脚本误作阶段而退出，未执行算子；
修正参数解析后重提。结果目录未覆盖，失败任务/日志完整保留。
另在L1采样前发现原通用计量器的两角色轮转和反转互相抵消，补入与静态
参考相同的两角色分支；三方顺序不变，不改任何计时边界或算子。
七档实际原始窗口均证明AB/BA交替，修改前后源码与当时阶段记录已保存。

动态正式37源与本次快照逐字相同，未增加优化候选或修改依赖。
大快照/二进制/权重不入Git，LOCAL_ARTIFACTS登记路径/数量/字节，无hash。
旧优化目标继续暂停，本次只完成用户要求的最新性能测量；不自动push。
