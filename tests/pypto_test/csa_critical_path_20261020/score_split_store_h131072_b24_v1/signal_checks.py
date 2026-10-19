# SPDX-License-Identifier: Apache-2.0
"""候选专用信号回读：原60阶段语义不变，所有wave保留位仍须为零。"""

LEGACY_PHASES = 60
CORES = 72
SLOT_WORDS = 128
WAVE_PHASES = 3
WAVE_ROWS = 8


def signal_view(prepared):
    """逐实例读取容量；reference仍可为60阶段，不能借用candidate容量reshape。"""
    signal = prepared.buffers["signal"]
    words = signal.numel()
    words_per_phase = CORES * SLOT_WORDS
    if words % words_per_phase:
        raise AssertionError("signal容量不是完整phase")
    phases = words // words_per_phase
    expected = getattr(prepared, "signal_phase_count", LEGACY_PHASES)
    if phases != expected or phases < LEGACY_PHASES:
        raise AssertionError(("signal容量与绑定不一致", phases, expected))
    if phases != LEGACY_PHASES:
        expected_wave_capacity = LEGACY_PHASES + WAVE_PHASES * ((prepared.tokens + WAVE_ROWS - 1) // WAVE_ROWS)
        if phases != expected_wave_capacity:
            raise AssertionError(("wave容量必须仅由T的shape确定", phases, expected_wave_capacity))
    return signal.cpu().view(phases, CORES, SLOT_WORDS)


def signal_padding_status(signal):
    """本runner不做clock插桩：新旧phase的所有1..127保留位均须为零。"""
    return {
        "legacy_padding_zero": bool((signal[:LEGACY_PHASES, :, 1:] == 0).all()),
        "wave_padding_zero": bool((signal[LEGACY_PHASES:, :, 1:] == 0).all()),
    }


def signal_padding_ok(signal):
    return all(signal_padding_status(signal).values())
