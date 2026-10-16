# SPDX-License-Identifier: Apache-2.0
"""CSA 临时核间信号适配；主线 #3020 可用后只替换这里的调用。"""

import pypto.language as pl


@pl.jit.inline
def csa_signal_notify(
    signal: pl.InOut[pl.Tensor], offset: pl.Scalar[pl.INDEX], value: pl.Scalar[pl.INT32]
) -> pl.Tensor:
    # 只发布单写者信号；调用方先完成该阶段 payload 的流水和缓存发布。
    pl.system.signal_notify(signal, [offset], value)
    return signal


@pl.jit.inline
def csa_signal_wait(
    signal: pl.InOut[pl.Tensor], offset: pl.Scalar[pl.INDEX], expected: pl.Scalar[pl.INT32]
) -> pl.Tensor:
    # 等待不清除信号，多个消费者可以独立等待同一个生产者。
    pl.system.signal_wait(signal, [offset], expected)
    return signal
