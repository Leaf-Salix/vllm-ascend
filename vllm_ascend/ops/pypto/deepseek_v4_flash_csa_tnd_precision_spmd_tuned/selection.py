# SPDX-License-Identifier: Apache-2.0
"""Select from physical graph rows before capture, preserving short-T kernels."""

from dataclasses import dataclass

PREFETCH_MIN_TOKENS = 96
MAX_TOKENS = 256


@dataclass(frozen=True)
class CaptureAttention:
    baseline: object
    tuned: object

    @property
    def _schema(self):
        return self.tuned._schema

    def for_tokens(self, tokens):
        if not 0 < tokens <= MAX_TOKENS:
            raise ValueError(f"Tuned CSA requires 1..{MAX_TOKENS} tensor rows, got {tokens}")
        if tokens < PREFETCH_MIN_TOKENS:
            return self.baseline.for_tokens(tokens)
        return self.tuned

    def __call__(self, *args, **kwargs):
        return self.for_tokens(args[0].shape[0])(*args, **kwargs)
