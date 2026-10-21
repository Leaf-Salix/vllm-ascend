# SPDX-License-Identifier: Apache-2.0
"""三个 CSA session 的唯一工具链选择；只读路径，不扫描或哈希源码。"""

import json
from pathlib import Path


def selected_toolchain():
    here = Path(__file__).resolve().parent
    workspace = next((parent for parent in here.parents if (parent / "env-dsv4-0251rc1.sh").is_file()), None)
    if workspace is None:
        raise RuntimeError(f"无法从 {here} 定位公共 CSA 工作区")
    # 嵌套 worktree 同样读取主检出的唯一清单，避免固定目录层数或旧清单副本。
    manifest = workspace / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/manifest.json"
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
