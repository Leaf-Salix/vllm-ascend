# SPDX-License-Identifier: Apache-2.0
"""独立ISA头文件增加ZERO→ND2NZ顺序，仅验证假设，不改变正式算子。"""
import sys
from pathlib import Path

import dsv4_csa_env
dsv4_csa_env.activate()
from private_isa_scope import install

# 主验证器要求全新结果目录；编译路径审计使用旁边独立目录。
install(Path(sys.argv[1]).with_name(Path(sys.argv[1]).name + "_isa_audit"))
from device_probe import main

if __name__ == "__main__":
    raise SystemExit(main())
