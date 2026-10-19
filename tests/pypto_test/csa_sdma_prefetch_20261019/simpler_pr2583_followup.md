已补齐不依赖本地 GM subview/NZ 修改的完整可运行复现，以及显式通道的因果对照：https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6269309892 。

同一完整 MIX 中，48 AIV 默认通道仍 FAIL；仅将 session 显式绑定到 0..47 的独立通道后，初次 eager + 10 次变输入 eager 全 PASS。单 AIV 发起、其余 47 AIV 正常计算的默认通道版本，也通过初次 eager + 10 次 ACLGraph replay。检查包括真实有效 event/session、Wait 返回值、计算结果和标记保护区。

显式逻辑编号仅用于这个唯一全卡 MIX 的诊断，未作为通用 runtime 修复；正式通道映射仍建议取物理 AIV 唯一编号，覆盖多 task 并发。该补充更支持默认 channel 映射冲突的定位，完整脚本及命令已附在上述 comment 中。
