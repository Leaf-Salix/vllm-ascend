# 下一项较大的借用方案：独立Indexer生产链

2026-10-27；不改正式11b精度基准，不与本轮Had/H4候选组合。当前仅设计审查，不宣称已编译或实测。

## 为什么优先此处

此前同SDK8K/B32完整event，我方约1586us、nalinaly c346约1088us。累计cut参考中入口至TopK+plan差约293us；独立编译及旧执行身份观测不足，不能把293us当可直接节省的纯Indexer成本。最新nalinaly1685源码的cube_schedule/vector_schedule明确按Indexer实际依赖推进，值得做新的单因素资格。

此前Query岛把主Q_B和IndexQuery共同放入一个MIX，Score必须等待整岛返回，完整回放反而慢约4%。本方案不再让mainQ_B成为IndexQuery/Score的额外前驱，也不因名字为qa_tid就认定存在真正QA完成依赖：现有qkv_proj_rope返回的projection_marker只是依赖late_dep的dummy。

## 可直接借用与必须保留

| 部分 | 做法 |
| --- | --- |
| 相位编排 | 允许复制nalinaly1685的Indexer相位、实际owner发布/等待及空角色完成协议；使用当前SDK已支持的纯DSL信号API |
| 主Q_B、main compressor | 保持独立，不作为新Indexer MIX输出，不挡Score |
| IndexKey投影/池化 | 填入我方已验数学体，保留按native_column_group旋转的K256顺序及cache/state所有权 |
| IndexQuery投影/量化 | 保留acc*(actScale*weightScale)及所有BF16/FP16物化边界，不复制nalinaly的(acc*actScale)*weightScale |
| Score/TopK | 先完整保留我方FP16系数、物理pair重排、排序和tie规则，只改生产消费衔接 |
| TND | 显式query bounds/真实行数，短query和混长不补S6 |

## 小步切分

1. 先做仅IndexQuery producer→quant/coeff→现有Score MIX的独立原型；Key必须在MIX发射前由真实任务依赖保证完整ready。不得让满24个Cube的MIX轮询等待一个尚未执行的外部Key Cube任务。
2. 只有前一步生成和性能资格通过，再考虑把IndexKey投影/池化/量化也纳入同一持久MIX，复制nalinaly独立Indexer整体生产链。
3. 不同时改tile、worker数量、K分块或舍入；尽可能直接复用双方原函数，逐块记录来源与保留的Native数值边界。

## 验证门槛

- 全部64参数CPU编译、load且NPUfalse，实际root/leaf变化、scope/ring/UB/L0/L1及信号协议经过独立审查。
- 先同卡8K/B32完整FULL graph12×100普通event及独立L1，计时前建立READY注册到CPU完整子项字节见证；dep另采，不同时打开。
- >=0.5% event收益、>=10/12胜且L1下降后，Native A/B八边界及ABA逐位对拍；首差停止，按最早模块修正，不降门槛。
- 反向重复稳定后，短T、混长、formal95/T126/T168路由；最后真实8K/B40再128K/B24逐token对齐。
- 每个设备任务outer780/inner750秒。没有稳定性能或精度资格，不合入、不外推整模型。
