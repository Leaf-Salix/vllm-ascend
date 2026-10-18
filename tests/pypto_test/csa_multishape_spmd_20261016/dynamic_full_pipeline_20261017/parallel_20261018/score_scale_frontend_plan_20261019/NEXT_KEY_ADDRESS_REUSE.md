# 后续候选：Cube key 供数复用已有整请求页布局证明

这是待实现设计，不代表优化或收益已经完成。先等待本轮 scale 前置证明的
全部代表档位终态，再决定适用区间和是否推进。

## 与已有试验的区别

旧连续页试验只在 AIV 合并 Native scale 小 DMA；本轮已将证明放入 META，
同一 first_page/direction 也可以供 Cube 使用。当前 Cube 每叶的每个 key panel
仍 Scalar 读 idx_block_table。128K/B16 目前每核22个Score step，每step32页，
因此存在每核约704次页号读（只是指令/工作量，不是可节省微秒的估算）。

后续可让 Cube 每叶只读两项证明，非零方向时用 first+direction*logical_page
计算物理页；碎片页仍按原表读取。沿用原 META 获取、query live、原预取和
Mat/Right ping-pong，不新增阶段、布局或通知，也不改变数据搬运顺序。

## 必须保留的约束

- logical_page 必须先沿原实现 min 到最后真实页。最后Score块仍可能读
  尾部重复页，不能直接对未钳制逻辑页做线性外推。
- Native page_bytes 与 key0/key64 的128B对齐选择保持。证明连续页不代表
  页间没有64B scale间隔，也不能把整个cache按紧凑key矩阵重新解释。
- 整请求所有页都已证明在物理容量内；只有该方向非零才能替代查表。
- G2/非balanced保持原路径，碎片路径不要为快路删除有效性检查。
- 不改变叶边界、TopK归并顺序、QK/WS FP16边界、双槽ACK或等待协议。
- 对照应为同轮前置证明版本与仅增加Cube地址复用的版本，不能用跨轮
  838/832等绝对数拼出收益。原分task继续作为总目标基线。

若以后考虑合并 key DMA，需要处理 Native 页内 key 与scale间隔及L1 NZ
布局；不能把二维[pages,4096]的Mat加载简单reshape成[pages*32,128]，
否则会改变分形布局。当前设计仅复用Scalar地址，不假称已经合并key搬运。
