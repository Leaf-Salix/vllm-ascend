# SPDX-License-Identifier: Apache-2.0
"""在公共bootstrap绑定的私有源码下运行TND服务CPU合同检查。"""

import sys

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))
