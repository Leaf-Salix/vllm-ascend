# SPDX-License-Identifier: Apache-2.0
"""三个 CSA session 的唯一工具链选择；只读路径，不扫描或哈希源码。"""

import json
from pathlib import Path


def selected_toolchain():
    here = Path(__file__).resolve().parent
    workspace = here.parents[3]
    manifest = here / "manifest.json"
    data = json.loads(manifest.read_text())
    paths = {
        "pypto": (workspace / data["pypto"]["source"]).resolve(),
        "ptoas": (workspace / data["ptoas"]["build"]).resolve(),
        "simpler": (workspace / data["simpler"]["source"]).resolve(),
    }
    for name, path in paths.items():
        if not path.is_dir():
            raise RuntimeError(f"公共 CSA 工具链缺少 {name} 目录：{path}")
    return data, paths, manifest


if __name__ == "__main__":
    data, paths, manifest = selected_toolchain()
    for value in (data["release"], paths["pypto"], paths["ptoas"], paths["simpler"], manifest):
        print(value)
