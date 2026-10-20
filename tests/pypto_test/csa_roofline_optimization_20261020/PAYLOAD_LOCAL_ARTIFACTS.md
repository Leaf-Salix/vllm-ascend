# 短前缀 v5 本地大产物位置

以下路径相对本目录。只记录位置和用途，不计算hash。

- `payload_full_b16_v5/build_output/`、`payload_full_b28_v5/build_output/`：
  两重点档的三方动态编译产物、PTO IR、完整C++及预编译kernel。
- `payload_full_h131072_b4_v5/build_output/`、
  `payload_full_h131072_b8_v5/build_output/`、
  `payload_full_h131072_b24_v5/build_output/`、
  `payload_full_h8192_b16_v5/build_output/`、
  `payload_full_h8192_b32_v5/build_output/`：扩档编译产物。
- 各任务下 `dfx/`：真机调试数据；测试结果及L1 JSON另行入Git。
- `payload_device_v5/`：布尔条件分组错误时的失败记录与大编译产物。
- `payload_device_v5_precedence_fixed/kernel/`：修复后59组×3图回放的产物。
- `payload_cpu_v1/` 至 `payload_cpu_v5/`、`payload_full_cpu_b28_v5/`：
  CPU lowering尝试及完整动态B28生成物。
- `scalar_bool_precedence_repro/` 和其 `after/` 中无扩展名的host可执行文件：
  CPU真值表用程序。对应小型源码与JSON报告入Git，可重新编译。

权重继续使用 `/data/model/DeepSeek-V4-Flash-0731-w8a8`，不进入Git。
未提交 `.bin`、`.so`、`.pt`、可执行文件及大编译快照。
