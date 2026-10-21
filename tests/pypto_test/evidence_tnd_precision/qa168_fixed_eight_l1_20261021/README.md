# QA168修复版八档L1重测证据

## CPU复核

Python3.10以上，在仓库根运行：

```bash
python tests/pypto_test/evidence_tnd_precision/qa168_fixed_eight_l1_20261021/verify.py
```

仅重算保存的原始L1并核验门禁，不导入Torch/vLLM、不运行NPU。
八档(case,history,batch)须唯一且等于预定集合；每档5窗、72核、A/B八边界、实际64Tensor根、
FULL/raw/毒化恢复/ABA/33guard/只读检查以及完整canonical运行身份均须通过。
此复核不等同于重新执行硬件或整模型token测试。

## 文件

- 八个`fixed-l1-*-evidence.tar.gz`：实际运行JSON、preflight、日志和所有原始六列L1记录。
- `reproduction-tools.tar.gz`：封存4文件助手及manifest、shell/preflight/收集脚本、CPU重算器、
  77源预期清单、43基础源/候选SHA、实际无设备预检、任务提交与终态、独立审查。
- `RECHECK.json`：原始记录逐窗重算结果。
- `CPU_VERIFICATION.json`：正式封包后实际执行CPU重算的成功收据。
- `SHA256.json`：原始包、助手和摘要的SHA绑定。

归档不含模型权重、cache payload、失败Tensor、生成C++或构建二进制。

## 227上重新执行

依赖已存在且验收过的隔离环境及资产：

1. 完整DeepSeek-V4-Flash-0731-w8a8的75分片权重。
2. 基目录`/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex`。
3. `vector-mte2-20261019-codex/full-csa-live-staging-v2/env.sh`及其原SDK桥接、缓存、静态13包。
4. `b28-precision-20261021-codex/whole-b28-qa168-v2`的3399源快照、77运行清单及实际编译收据。
   全部SHA及QA ea30修复身份见助手`tools/binding.json`；只校验引用，不修改原资产。

将助手还原到原测试目录`eight-l1-fixed-20261021-codex`，用**新的case标签及result目录**避免覆盖旧输出；
直接使用`run.sh <case> <batch> <history>`，其case只要求小写字母/数字/横杠。
若复用归档`submit_case.py`和`recheck.py`，它们固定v1标签集合，需在独立副本中一致更新新试验标签、
新目录和期望集合，保留原助手manifest/收据；记录新助手身份，不能冒用本轮SHA。
若改测试根目录，须同步`run.sh/preflight.sh/collect_case.py`中的exp以及本地提交/收集脚本的REMOTE值。

先运行`preflight.sh`，实际模块路径、CANN/PTOAS、原Python环境、3399/77源与编译收据必须匹配，NPU未初始化。
静态检查shell/Python及manifest；NativeHalf之后的数学、fixture、图和计时流程保持封存实现。
不重新pip install SDK或vLLM，不更改生成C++等待函数。

每档通过task-submit单独分配一张卡：device auto、device-num1、ptoas0.63、max-time780，
显式传入PYTHONNOUSERSITE=1、ASCEND_HOME_PATH、PTOAS_ROOT、CCACHE_DIR、CMAKE_BUILD_PARALLEL_LEVEL=4；
完整值和实际命令见每档submission.json，入口内限750秒。
顺序为128K/B28、8K/B16、8K/B32、128K/B1/B4/B8/B16/B24，前档通过并终态退出后再提交下一档。
任何精度首差立即停止矩阵；不要将8档合成一个长占卡任务。
只读status/log监控，提交后每5分钟检查，不使用会超时取消排队的wait/run，不创建定时自动化。

运行环境：原Python3.12、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、
CANN9.2beta2、原PyPTO0.1.0/Simpler/PTOAS；实际import路径由preflight逐项记录。
新环境不符时应报告阻塞，不能跳过门禁。

## 口径与边界

只测S6、TP1、page32、NZ2、seed1024、layer4，8K/128K为history8192/131072。
Native确定性level1/HCCL，SK0且13静态包资格验证，DFX level1、dep_gen关闭、无C++退避。
warm3后5窗，每窗1次FULL raw graph，取所有72核最早incore开始至最晚结束，包含内部等待。
restore与同步在窗外，保留全部窗口，以中位数比较，不剔除离群值。

Nal参考是用户截图，未在本轮同卡同工具链实测；本轮不会证明与其性能差距的唯一原因。
单层逐位与整模型逐token是不同验收。本轮不新增整模型8K/128K性能、短query FULL或任意混合1～6资格。
