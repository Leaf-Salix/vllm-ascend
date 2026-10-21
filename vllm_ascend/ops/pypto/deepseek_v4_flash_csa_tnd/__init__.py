# SPDX-License-Identifier: Apache-2.0
"""原生 TND 的 CSA 多 task 实现。

编排以 deepseek_v4_flash_csa 的多 task DAG 为骨架；真实 Q/L、混长、padding
和核内数学借鉴冻结的 spmd_tnd。完整 CSA 不合并为一个 SPMD。
当前设备验证与接入进度见 tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md。
"""
