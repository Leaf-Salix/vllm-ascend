# nalinaly CSA 的 TND 适配与性能对拍（2026-10-08）

分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
起点：`1c3fb7da1`；移植参考：`8e83adda1f5959da0d407f2d27d56f52a16466e5`。
独立工作目录，未切换或修改原有 TND 工作分支。

## 实现与支持范围

采用最新 `deepseek_v4_flash_csa` 实现，保留参考的 Cube tile、请求内 Key 共享、
FIXPIPE/归约和 O-proj 流水。原生 token 级 `query_start_loc` 新增为根 ABI 入参，
与 compact/indexer 的请求边界分别保留。设备端一次生成 token_request，
Indexer 分组按各请求边界切分，末组有效行独立记录，禁止跨请求共享 Key。

本轮修复了原型未验证的边界：

- 请求映射全档初始化，未覆盖尾部及 seq_lens=0 的补位行标记无效。
- Indexer 建组/计数忽略 inactive dummy，补位 TopK 写入 -1。
- metadata 访问前检查有效请求，compact 表为空时不读取负下标。
- attention 流水 drain 阶段不读取超出 token 范围的请求号。
- compressor 的 dummy pooling/compact 发布不消费陈旧 position。
- 宿主按真实请求检查长度1～6，大补位区间不受真实请求上限误拒。
- HCA 保留固定 S6；CSA Graph 判定接入真实最大 query 长度及非6倍数容量。
- 空请求分支标量类型统一；计数 inline 调用先绑定标量再用于条件。

当前只验证 `PTO_CSA_RUNTIME=tensormap_and_ringbuffer`。
TND 的 device 分组尚未迁移到 Host 建图，服务明确拒绝该运行方式。
请求轴总容量上限64；64个真实请求另追加一个dummy的65请求档位尚不支持。
当前支持 target DSpark、TP1，真实请求最多6行；不是任意长度 prefill。
继承参考性能版的量化、舍入和规约，**不宣称已与 Native 数值对齐**。

## 当前验证

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| 本地 scalar metadata/Graph 回归 | 12项通过 | 执行实际 producer，覆盖不等长、空请求、大dummy和旧映射清除 |
| 227 宿主入口/ABI UT | 11项通过 | 新入口、真实长度限制、大dummy、HCA gate、原生bounds绑定 |
| 完整 CSA lower | 通过 | 当前工具链生成完整编排与kernel |
| 关键核 CCEC | 16个二进制通过 | 分组、请求映射、系数、共享Key score、compressor pooling |
| S6 B4/8K 同场 Graph | 已提交，待结果 | 先严格输出/TopK/cache逐位对拍，再测延迟 |
| S6 B16/128K、混合TND及连续回放 | 待验证 | 不以静态检查代替设备结果 |

lower 前三轮发现类型不一致、Tile/Tensor混用、条件内inline展开失败；修复后v4通过。
前两者为源码问题，后者通过等价的标量绑定方式适配现有编译器。
NPU结果完成后在本文追加，不把上述检查当成精度或性能通过。

## 对拍环境与协议

227，CANN9.2.0-beta.2，vLLM0.25.1，Torch2.10.0+cpu、Torch-NPU2.10.0.post2。
复用已构建 PyPTO `3e87a843619aca13af39755700513d26b402e924`、Simpler
`a54c0509552b01e13fb0960e23ce409a01b5024f`、PTOAS0.66，以及原有 release Native 扩展。
两侧同环境、同工具链，未安装或替换共享包；不宣称用了原型文档建议的另一条未构建工具链。
确定性 level1、HCCL_DETERMINISTIC=true、atomic_add=0、weight_nz_mode=2。

真实 checkpoint 的 C4 第2层权重；合成 hidden/history，seed631337，32槽 Native 页布局。
入口包含 HC pre、norm、CSA、HC post，手工 NPUGraph；不代表整模型激活、
服务吞吐或 decode 模板编译优化后的延迟。
BSH 固定为参考 SHA，新 TND 单独注册；两侧独立且相同初态的 cache。
先比较输出、TopK、cache/state 和保护区，再预热50次，24轮×100次，交替顺序，
报告配对差值和全部样本。精度断言失败会停止，不能将改变数值记成性能收益。

## 格式检查状态

本轮修改文件的 ruff 与定向检查通过；codespell 的公共 `pl.InOut` API 误报
通过词表增加准确标识符 inout 处理。运行了完整 format.sh ci，但参考分支中
已有归档/实验脚本的长行、原始日志拼写、禁用import及无注释长函数导致失败。
没有改写历史证据或顺带整理这些文件；格式工具对无关文件的自动改动已恢复。
完整CI不能记为通过，此限制与本轮CPU/lower/CCEC结果分别记录。

整合范围仅包含维护中的CSA包、入口及必要测试helper；未合并参考中的HCA实现、
归档或历史实验产物。默认precision入口保留，TND对拍显式使用performance。

系数 producer 的 worker 上界改为 `min(48, B*ceil(6/group_size))`，仍以设备端
实际组数决定循环范围；S6时恢复参考的worker数量，避免按总token数派发空worker。
独立审查确认该形状上界覆盖所有活跃请求的组，不漏组；性能影响仍待同场测量。

## 第一项设备任务与编排修复

实现提交 `916b39396` 的 B4/S6、history8192 对拍未产出性能结果：
参考 BSH 已执行，随后 TND 编排 C++ 编译失败，Indexer 返回的 TopK SSA
别名声明在子 scope 内，attention 在外部消费，导致未声明标识符。
完整 lower 和关键核 CCEC 未覆盖此编排编译错误。

修复在父 scope 保存根 TopK 描述符，Indexer 继续写入同一 Out 缓冲，
attention 读取父级描述符。保留 Indexer 子 scope 的临时内存生命周期。
修复后的 v7 完整 lower 与编排共享库编译通过；生成代码中 merge 使用
`add_inout(ext_idx_topk)`，attention plan 使用 `add_input(ext_idx_topk)`。
TensorMap 按缓冲地址与重叠区间建立 RAW 依赖；不把 scope 退出当成同步屏障。
设备端正确性与延迟仍需重跑验证，首次任务失败不记为精度失败。
