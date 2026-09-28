"""Materialize timing events before generation; keep allocation out of EP entry."""

import torch

TIMING_EVENT_SETUP = "prewarm_before_generation"


def prewarm_timing_events(steps):
    if steps <= 0:
        raise ValueError("timing steps must be positive")
    pairs = [tuple(torch.npu.Event(enable_timing=True) for _ in range(2)) for _ in range(steps)]
    # Constructors are lazy. Recording is necessary to create the device
    # handles, and one setup wait completes these records before reuse.
    for pair in pairs:
        for event in pair:
            event.record()
    pairs[-1][1].synchronize()
    return pairs
