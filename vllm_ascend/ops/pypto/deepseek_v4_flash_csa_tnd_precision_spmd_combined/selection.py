# SPDX-License-Identifier: Apache-2.0
"""Choose one complete root before capture; keep short-T kernel unchanged."""

from dataclasses import dataclass

SMALL_TOKEN_LIMIT = 16
MAX_TOKENS = 256


@dataclass(frozen=True)
class CaptureAttention:
    small: object
    combined: object

    @property
    def _schema(self):
        return self.combined._schema

    def for_tokens(self, tokens):
        if not 0 < tokens <= MAX_TOKENS:
            raise ValueError(f"Combined CSA requires 1..{MAX_TOKENS} tensor rows, got {tokens}")
        return self.small if tokens <= SMALL_TOKEN_LIMIT else self.combined

    def __call__(self, *args, **kwargs):
        return self.for_tokens(args[0].shape[0])(*args, **kwargs)
