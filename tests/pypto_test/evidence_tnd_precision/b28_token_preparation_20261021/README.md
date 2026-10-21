# B28逐token补验准备证据

本目录不包含新的NPU或token验收通过结果，正式数学没有改变。

- `cpu-whole-preparation-evidence.tar.gz`：227实际CPU日志、7模块环境及未初始化NPU收据、28请求bank plan/header audit、helper封存清单。
- `REMOTE_CPU_PREPARATION_RECHECK.json`：本机独立重算结果，明确bank阶段任务整体exit1与最终preflight exit0。
- `HISTORICAL_INPUT_CPU.json`：自身保存的T168与旧T126 hidden前缀逐byte比较，不是新的T168 QA金值。
- `whole-b28-tools.tar.gz`：待硬件补验助手和CPU验证方法；其中README保留封存准备时说明。
- `stage-diagnostic-prepared-tools.tar.gz`：13阶段观察器，仅获无设备编译资格；`compile-receipt.json`和`STATIC_REVIEW.json`绑定身份。
- `BASELINE_ENVIRONMENT_BOUND.json`：历史token验收时的原模块路径/version和CANN/ATB/PTOAS环境。

完整当前状态与复现方案见[逐token准备报告](../../DSV4_FLASH_CSA_TND_TOKEN_ACCEPTANCE_PREPARATION_20261021.md)。
归档不包含weights、cache payload、SSH密钥或代理凭据；SHA清单不自引用。
