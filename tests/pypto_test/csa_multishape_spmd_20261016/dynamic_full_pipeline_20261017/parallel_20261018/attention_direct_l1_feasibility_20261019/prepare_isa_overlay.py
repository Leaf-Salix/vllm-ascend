# SPDX-License-Identifier: Apache-2.0
"""从本地Simpler托管的PTO-ISA重建私有头文件副本，不修改原仓库、不计算hash。"""
import os
import shutil
from pathlib import Path


def prepare():
    here = Path(__file__).resolve().parent
    destination = here / "isa_zero_barrier"
    if (destination / "include/pto/npu/a2a3/MGather.hpp").is_file() and (destination / "pkg_inc").is_dir():
        return destination
    if destination.exists():
        raise RuntimeError(f"私有副本不完整，保留现场，请指定新的目录重建：{destination}")
    simpler = Path(os.environ["HCA_SIMPLER_ROOT"])
    source = simpler / "build/pto-isa"
    relative = Path("include/pto/npu/a2a3/MGather.hpp")
    original = (source / relative).read_text()
    old = "        pto_create_cbuf_matrix((__cbuf__ uint16_t*)dstPtr, repeatConfig, 0);"
    assert original.count(old) == 1, "ISA实现已变化，不能盲目套用本次补丁"
    tail = original.split(old, 1)[1]
    assert tail.startswith("\n    }\n"), "原ISA可能已经修正，需重新核对基线"
    shutil.copytree(source / "include", destination / "include")
    shutil.copytree(source / "pkg_inc", destination / "pkg_inc")
    replacement = old + "\n        // ZERO initialization and ND2NZ write overlapping L1 rows; complete fill first.\n        pipe_barrier(PIPE_MTE2);"
    (destination / relative).write_text(original.replace(old, replacement))
    return destination


if __name__ == "__main__":
    print(prepare())
