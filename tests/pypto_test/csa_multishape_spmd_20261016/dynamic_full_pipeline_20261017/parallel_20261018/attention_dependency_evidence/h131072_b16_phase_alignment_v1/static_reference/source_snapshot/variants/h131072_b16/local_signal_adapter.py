# SPDX-License-Identifier: Apache-2.0
"""CSA 临时核间信号适配；主线 #3020 可用后只替换这里的调用。"""

import pypto.language as pl


@pl.jit.inline
def csa_signal_notify(
    signal: pl.InOut[pl.Tensor], offset: pl.Scalar[pl.INDEX], value: pl.Scalar[pl.INT32]
) -> pl.Tensor:
    # Scalar 发布前排空本核流水。AIC 的 gm fence 仅生成分流水 barrier，
    # 不能据此认定 Scalar 已等待 FIX 写回；这里复用核内 PIPE_ALL。
    # 这不是 syncall，不等待其他核集合；调用方仍负责 payload 的缓存维护。
    pl.system.bar_all()
    pl.system.signal_notify(signal, [offset], value)
    return signal


@pl.jit.inline
def csa_signal_wait(
    signal: pl.InOut[pl.Tensor], offset: pl.Scalar[pl.INDEX], expected: pl.Scalar[pl.INT32]
) -> pl.Tensor:
    # 等待不清除信号，多个消费者可以独立等待同一个生产者。
    pl.system.signal_wait(signal, [offset], expected)
    return signal
