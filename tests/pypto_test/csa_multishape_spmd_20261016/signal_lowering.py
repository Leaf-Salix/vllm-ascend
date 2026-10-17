# SPDX-License-Identifier: Apache-2.0
"""历史静态验证的兼容入口；冻结生成器不属于正式动态算子。"""

from historical_sources import load_historical_module

use_local_signals = load_historical_module("signal_codegen").use_local_signals
