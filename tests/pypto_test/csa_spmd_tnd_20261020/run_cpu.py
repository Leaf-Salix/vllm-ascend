# SPDX-License-Identifier: Apache-2.0
"""在显式绑定的工具链中运行CPU合同测试，不占用NPU。"""

import sys

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))
