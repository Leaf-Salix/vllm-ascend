# 2026-10-24：直接读取压缩KV候选收口

本轮未采用任何新候选，正式HCA仍为`37bbd0f3c`的已验证区间策略。用户再次要求
“本轮优化之后停下”，设备任务已全部结束；完成归档提交后暂停，不push，不发新任务。
完整目标“八档HCA超过Native30%”以及已有精度遗留均未完成。

## 先关闭核外gather分档的尝试

`build_attention_gather_dispatch.py`把选择移到TMR，仅用输入shape推得工作量；
每次提交独立Scalar或Vector核。v1因Tile在分支内定义、分支外使用产生SSA作用域错误；
完整重复IR诊断约42MB，仅本地保留，Git中保存错误摘要、路径和大小。
v2完整CPU通过，但Scalar核仍有无消费者的Vector初始化。v3分开源码体后完整CPU通过，
Scalar生成代码为TCI=0、TEXPANDS=2，不再包含Vector网格及列号；未改共享SDK。

| 场景 | 正式控制μs | 核外分档μs |
|---|---:|---:|
| 128K/B28 | 757.08 | 764.48 |
| 8K/B32 | 584.24 | 597.88 |
| 128K/B24 | 673.54 | 681.60 |
| 256K/B16 | 664.48 | 663.82 |

输出/cache/state对控制精确，Native输出、重复、保护及独立混长FP64边界通过；
但没有稳定完整层收益。L1核对显示gather/attention的依赖数与控制相同，不能声称是
新增依赖导致回退，也不能将不同任务中值相减拼接收益。该方向本轮不继续展开。

## 直接KV的计算与安全边界

生成器`build_attention_direct_kv.py`输出完整可运行的
`candidates/attention_direct_cmp_kv.py`。小档K128保持原函数；长档K512保留D256供数、
两步QK预发、三槽、跨query、原FFTS及全部浮点顺序，只改变压缩KV来源。

- 原每128行的128KiB聚合数据窗口，替换为一条64B页记录；另保留全层共享64KiB零模板。
  这是软件层GM存储/搬运量变化，不等于已测得的HBM带宽收益。
- 页记录由AIV读取device长度和Native页表，保存四页物理起点、有效行数及连续标记。
  一行64B单writer整体MTE3发布；不使用Host地址列表或Scalar GM短写。
- 页计划不读cache内容，可与QKV/C128生产重叠；attention显式同时等待计划及真实cache写回。
  读取计划前维护本核Scalar缓存，不能以读到标记代替业务数据同步。
- 完整连续四页直接GM→L1；非连续、无效页或部分尾页先加载零模板，再只覆盖合法有效行。
  概率为0也不能读取NaN尾区，因此不能省掉这个保护。
- 参考Native arch32 `sparse_attn_sharedkv_scfa_block_cube.h`的Raw `DataCopyPA`路径。
  Native compressed路径实际使用`kvMergeGm_`，本候选不是其逐行移植或既有直接compressed实现。

## 编译及功能证据

v1修正本地INDEX字面量写INT32记录的错误；v2在Mat if合流处生成A3不支持的Mat→Mat
TMOV。v3改为两分支向同一份预分配L1写入，避免这次复制，三个根完整CPU后端通过。
没有修改编译器、PTOAS或Simpler。

v3独立S6、Q=[1,5,6,1,2]混长，在128K和16507边界对密集FP64的误差与正式K512相同，
候选对控制max_abs/RMSE均为0；负页、部分尾页、失活、重复和输出保护通过。
另将物理压缩页随机重排并对页表应用逆置换，逻辑KV及独立FP64参考不变；该检查也通过，
覆盖全有效但物理不连续的页，未将输入强制改成连续布局。

## 完整层性能与决定

同任务、同SDK、正式W8A8、NZ2，全部eager直接执行；各七窗完整HCA L1 incore首尾。

| 场景 | 正式控制μs | 直接KVμs | 增加μs |
|---|---:|---:|---:|
| 128K/B16/S6 | 504.08 | 641.22 | 137.14 |
| 128K/B28/S6 | 745.84 | 961.44 | 215.60 |

两档输出及三组cache/state相对控制精确，Native输出通过原容差，重复与写保护通过；
但性能明显退步，**不采用、不扩档、不继续调参**。这些结果仅证明当前候选未获益，
不能推出直接读取在所有布局上无效，也不能把减少的GM字节直接当作性能提升。
未对各项Scalar、MTE和缓存成本完成独立归因，不将猜测写成确定原因。

## 组件报告与产物

无循环、无phi、无别名逃逸的CPU最小对照确认：未使用的tile.full及arange→cast仍保留
TCI/TCVT/TEXPANDS；clean各0次，unused各1次，真实使用的反例各1次。
已提交[PyPTO #3136](https://github.com/hw-native-sys/pypto/issues/3136)，含完整复现及修改建议，
明确只证明PTO/设备C++，未宣称实测微秒或最终机器码收益。没有PR、没有标签、没有SDK改动。

任务身份：`ATTENTION_GATHER_DISPATCH_TASKS.json`、`ATTENTION_DIRECT_KV_TASKS.json`。
完整样本、CPU失败和本地产物清单：`ATTENTION_NEXT_RESULTS.json`；可读汇总为同名Markdown。
大日志、快照、原始泳道、对象文件和二进制本地保留，不做hash扫描。
