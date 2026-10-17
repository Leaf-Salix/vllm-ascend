# SPDX-License-Identifier: Apache-2.0
"""每层静态 WKV 资格；由完成包校验和真实 reselection 的加载者显式通知。"""

QUALIFIED_STATIC_METADATA_SHA256 = "863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602"
STATIC_WKV_TOKENS = 6
STATIC_WKV_MODE = 1


class WKVProfile:
    """保留原非静态标定；首次捕获后禁止更改执行资格。"""

    def __init__(self):
        self.metadata_sha256 = None
        self.captured = False

    def qualify_static(self, *, metadata_sha256, reselected):
        # SHA 标识已验证的 13 包/SDK/SOC/确定性配置，不替代加载者的完整校验。
        if metadata_sha256 != QUALIFIED_STATIC_METADATA_SHA256 or reselected is not True:
            raise ValueError("WKV requires the qualified static packages and successful actual reselection")
        if self.captured and self.metadata_sha256 != metadata_sha256:
            raise RuntimeError("WKV qualification cannot change after graph capture; recreate the runtime and graphs")
        self.metadata_sha256 = metadata_sha256

    def select(self, tokens, baseline_modes, *, capturing, static_requested):
        if not 1 <= tokens <= len(baseline_modes):
            raise ValueError("WKV token count is outside the calibrated range")
        if capturing:
            if static_requested and self.metadata_sha256 is None:
                raise RuntimeError("Static WKV capture requires verified installation and actual reselection")
            self.captured = True
        if self.metadata_sha256 is not None and tokens == STATIC_WKV_TOKENS:
            return STATIC_WKV_MODE
        return baseline_modes[tokens - 1]
