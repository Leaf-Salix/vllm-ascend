# 四token Hadamard 借用验证清单

开始：2026-10-27；基准11b380e3f，nalinaly1685。

- [x] 只读复核最新GitHub，无更新。
- [x] 独立helper直接移植行合并，原精度源码未改。
- [x] 双臂CPU64/load/NPUfalse；仅Had matmul变化，62叶SHA相同；尾行、96KiB L1及独立审查通过。
- [x] 8K/B32同卡event12x100与独立L1，不开启dep；实际注册见证。
- [x] 反向显式Native A/B8、ABA、guard/readonly通过；性能未稳定，不采用。
- [x] 双向性能门槛未通过，不扩短T/混长/128K/整模型；不外推精度资格。
- [x] 性能不稳定，不采用；原始证据、结果及复现方法已完整记录。
