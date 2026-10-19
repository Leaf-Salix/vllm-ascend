# 两行带步长DMA的独立可行性

只读A3 PTO-ISA确认GM->UB的TLoadInstrGm2ubNative接收uint32字节gap，
两条物理行可按地址升序使用nBurst=2、每段1024B、gap=(delta-1)*1024。
与Mat/L1的16位gap限制不同。超过32位gap时公共实现会逐burst退回，
本候选显式保留逐行路径，不把它计为合并搬运。

PyPTO view无任意strides参数。小例尝试先切真实存在的2*delta行span，
再做等元素数的[2,delta*512] ND view，读取每行前512元素。必须额外
证明整个span仍在cache allocation内；不扩大真实存储或依赖校验漏洞。
下降顺序在UB恢复，重复/无效/跨度过大均回退。

这是后续减少compressed gather小DMA数量的独立接口检查，没有改
当前CSA候选，没有真机结果或性能结论，不能据命令数宣称加速。

## 结果与上游反馈

动态两行小例在ConvertTensorToTileOps被显式拒绝：slice先变为Tile，
后续GM view不接受Tile。这是能力限制，不冒称错误输出或现有CSA功能bug。
没有改依赖、没有上NPU，也没有性能收益数据。

最小例见minimal_repro.py；根GM view完整CPU生成成功，slice→view失败
复现同一保护检查（minimal_v2）。v1对照在函数体误用ROWS动态符号导致
Undefined variable，已保留原稿与失败；v2改从Tensor读取dim。两个对照
不是数值等价：前者读取0/1行，后者目标0/2行，只用来区分编译能力。

已提交 https://github.com/hw-native-sys/pypto/issues/3042 ，正文逐字保存在
PYPTO_ISSUE_BODY.md，无label、无PR。附完整复现、成功对照、main源码只读
核对及两条建议：显式GM subview/strided-load，或仅保留纯descriptor链。
不能删除现有Tile→GM拒绝，也不能把整个大span隐式搬进UB。

本地基点d01de8523含既有CSA修正，未声称干净main实测；核对main
2964333d45a2b861b78977101278099c8812fc8b仍有相同拒绝逻辑，未重建该main。
该能力仅服务下一项可选DMA合并优化，不阻塞本轮已验证的物理行plan。

CPU复现：bash run_minimal.sh <绝对新目录>；业务动态尝试：bash run_cpu.sh
<绝对新目录>。两者均不占NPU。编译产物本地保留；失败和源代码入Git。
