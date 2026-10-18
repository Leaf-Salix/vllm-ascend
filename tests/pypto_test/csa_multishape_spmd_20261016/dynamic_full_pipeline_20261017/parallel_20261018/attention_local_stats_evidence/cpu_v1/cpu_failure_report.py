# SPDX-License-Identifier: Apache-2.0
"""保留完整本地编译诊断，限制result/stdout大小；不截断原始错误文件。"""

import re
import traceback
from collections import Counter

ERROR_PREVIEW_CHARS = 4096
MAX_SUMMARY_KINDS = 64
SUMMARY_KEY_CHARS = 512


def _bounded_counts(values):
    counts = Counter(value[:SUMMARY_KEY_CHARS] for value in values)
    top = counts.most_common(MAX_SUMMARY_KINDS)
    return dict(top), sum(counts.values()), len(counts)


def record_compilation_failure(output, error):
    """大IR在独立文件只保存一次；traceback仅保存栈，不重复附加完整exception。"""
    message = str(error)
    detail = output / "failure_error_full.txt"
    detail.write_text(message + "\n")
    stack = output / "failure_traceback_stack.txt"
    stack.write_text("".join(traceback.format_tb(error.__traceback__)) + type(error).__name__ + "\n")
    messages, message_total, message_kinds = _bounded_counts(
        match.group(1) for match in re.finditer(r"(?m)^\s*Message: ([^\n]*)", message)
    )
    locations, location_total, location_kinds = _bounded_counts(
        match.group(1) for match in re.finditer(r"(?m)^\s*Location: ([^\n]*)", message)
    )
    return {
        "error_type": type(error).__name__,
        "error": message[:ERROR_PREVIEW_CHARS],
        "error_preview_truncated": len(message) > ERROR_PREVIEW_CHARS,
        "error_full": {"path": str(detail), "bytes": detail.stat().st_size},
        "traceback_stack": {"path": str(stack), "bytes": stack.stat().st_size},
        "diagnostics_summary": {
            "message_counts": messages, "message_total": message_total, "message_kinds": message_kinds,
            "location_counts": locations, "location_total": location_total, "location_kinds": location_kinds,
            "summary_kind_limit": MAX_SUMMARY_KINDS,
            "summary_truncated": message_kinds > MAX_SUMMARY_KINDS or location_kinds > MAX_SUMMARY_KINDS,
        },
    }
