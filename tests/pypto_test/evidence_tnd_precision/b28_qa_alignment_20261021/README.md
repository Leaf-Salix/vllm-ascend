# T168 QA精度定位证据

完整说明见上级目录的`DSV4_FLASH_CSA_TND_T168_QA_NATIVE_ALIGNMENT_20261021.md`。

- 单层硬件对拍：QA packet三组输入逐位一致；完整CSA的A/B各8个外部边界逐位一致，原Native基准未改变。
- 新canonical根/preparer实际编译、bank及环境预检通过。这里的CPU收据不是整模型token结果。
- 原始JSON的SHA见`SHA256.json`。运行`python verify.py`只重算保存的收据，不占设备，不证明新的模型或性能资格。
- `reproduction-tools.tar.gz`包含诊断源码、准备/编译/运行脚本和独立审查。历史阶段的“准备/待运行”文字是当时记录，最终状态以本报告和实际收据为准。
- 重放需要227既有封存snapshot、原Python/SDK/工具链、完整模型及固定bank。没有打包权重、cache、Tensor payload或编译二进制。
- 新整模型候选在`whole-b28-qa168-v2`，旧`whole-b28-next`和初版`whole-b28-qa168`均保留，不混用manifest或gold。

整模型必须先Native1 READY，再PTO；单任务不超过15分钟。token、接受事件和输入轨迹分别审核，首个token差异即停止。
