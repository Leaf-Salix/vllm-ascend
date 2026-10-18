# 最小混类型 Mat 槽验证的前置接口限制

仅查现有隔离PyPTO源码与已有单测；没有新PyPTO/CCEC编译、没有上卡。本轮不为得到成功改成flat Mat或编造冻结IR。

直接路径“BF16 NZ KV槽 → slice → FP16/INT8 reinterpret”已有明确拒绝条件：
`src/ir/op/tile_ops/transform.cpp:374–390`要求`slayout=none_box`，且不接受非零/动态`start_offset`（MX-scale字节别名例外与本算子无关）。已有`tests/ut/ir/operators/test_tile_ops.py:3902–3905`专门验证boxed tile被拒绝。当前实际KV/FP16 panel/INT8 key均为NZ boxed，不能为通过检查换成flat_layout而声称布局不变。

现有`pl.MemRef(slots=N)`提供动态槽，但PTOAS planner拒绝不同shape/valid形状和loop内多槽同活，见`tests/ut/codegen/test_multi_buffer_codegen.py:401–428`。**这项限制不适用于所有planner**：同文件`:434–454`明确PYPTO planner接受这些形状，`:368–382`说明其level3使用baked地址。不能仅凭PTOAS限制否定当前PYPTO路径。

同样，`memory_reuse_pass.cpp:2700–2705`明确不同shape/dtype可共享物理MemRef；boxed reinterpret不支持不等于异dtype物理重用不可能。三参数`pl.MemRef(base, offset, size)`解析器支持表达字节区间，但它描述已有Ptr/打印IR，而用户级命名声明`pl.MemRef()`的子槽接口没有直接提供“本128KiB槽中的偏移48KiB/96KiB”。这里尚未验证以该低层已解析MemRef构造运行时子区、保持三槽及其生命周期能否通过完整后端；没有把这条未验证路径写成失败。

因此本轮可确认的是：**直接view方法被已有接口条件排除；通用异dtype子区别名仍是待证明的源级表达缺口**。按“明确不支持则不盲试”的边界，没有新增一次注定被boxed检查拒绝的编译。若后续采用不同的显式MemRef入口，应先单独验证该入口的真实地址/布局/槽隔离，不能把本资源表当CSA正确性或容量PASS。
