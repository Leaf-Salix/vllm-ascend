# HC共享实参观测清单

2026-10-27；HC正反方向反转后，仅诊断physical参数控制因素，不新增优化或改SDK。

- [x] 保留原HC v2两向结果，不改完整性能门槛。
- [x] parent完整CPU63/load/noNPU和实际注册见证资格保留。
- [x] 新shared forward确实替换0/40/53，逐Tensor描述包含pointer/storage pointer/offset/stride/dtype/shape。
- [x] CPU导入通过NPUfalse（首个命令缺parent sys.path，修正命令后通过；非算子失败）。
- [x] 新图计时前与各自原初始化图same-arm八边界、poison/guard检查。
- [x] v2新增真实capture CallFrame64实参观察，原样转发且finally恢复；要求capture_id非0。
- [x] 独立源审v2观察补充。
- [x] 单项8K/B32 --qualify-boundary设备诊断，outer780/inner750。
- [x] 原始性能与NativeA/B8+ABA单独记录，直接root FULL不覆盖服务图口径。
- [x] 无稳定收益不采用，不扩128K或整模型。
