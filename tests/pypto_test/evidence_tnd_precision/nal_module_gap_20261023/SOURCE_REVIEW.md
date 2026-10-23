# 最新 nalinaly 对照：可借鉴编排与必须保留的数值边界

本轮固定我方 `aad137911`，nalinaly `cafcadec96`。这是源码审查，不把候选结构直接当成性能收益。

| 部分 | nalinaly 当前路径 | 我方 Native 精度约束 | 本轮取舍 |
| --- | --- | --- | --- |
| Attention 更新 | K128局部softmax、末尾加入sink，概率rint | K512在线更新、sink初态、实际K求和树、概率round | 保留我方数学；可借搬运和跨query流水 |
| inverse RoPE | 未BF16的归一化heads直接旋转 | 先BF16、转回FP32、旋转、再BF16 | 不删除中间舍入 |
| WO-A/O-B | WO-A FP32、八组各自1024尺度 | WO-A BF16、全8192列共同尺度、Native packet start及累加顺序 | 可借任务矩形/owner；不借分组量化 |
| heads 布局 | WO-A直接消费分组连续heads | 当前组合版仍有packed→Native纯复制 | Direct/Packed已在10月22日有小幅收益，避免当作新发现；是否纳入组合需重测 |
| Attention统计 | 部分策略mi/li留本核UB | 我方mi/li/alpha经GM发布再回读 | 在同核所有权与有效块分支内尝试等价存储优化 |
| HC/Norm前缀 | 按行/owner发信号；较少全核屏障 | 我方Native精确HF32等价、sqrt/reciprocal及归约 | 先测前缀跨度，再按生产者消费者检查屏障；不能照抄不同数学 |

历史负结果：地址预解码仅约0.1%且7/12窗领先、mgather约慢2.18%、仅等owner未稳定快于冻结后缀。不再以相同理由重复这些方向。

nalinaly最新验证的“无新增回归”是相对其旧PTO结果，不能替换对Native精度验收。新报告须同时注明：单SPMD完整MIX在现有L1仅显示整体包络，内部阶段没有现成时间戳；不能将PMU Scalar占比当纯轮询开销，也不能把跨窗阶段中位数相加当完整差距。
