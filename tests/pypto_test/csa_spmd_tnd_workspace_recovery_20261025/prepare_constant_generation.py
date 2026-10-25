# SPDX-License-Identifier: Apache-2.0
"""在每调用全量清零且硬MIX屏障之后，移除重复的代际读取/发布。"""

import ast
import shutil
from pathlib import Path

from prepare import replace_once


def transformed(source, core):
    source = replace_once(
        source,
        f"    signal, generation = begin_invocation(signal, {core})\n",
        "    # 本入口每次已清零所有阶段槽并完成硬MIX屏障，当前调用代际固定为1。\n"
        "    generation = pl.cast(1, pl.INT32)\n",
    )
    source = replace_once(source, "begin_invocation, ", "")
    ast.parse(source)
    return source


def main():
    here = Path(__file__).resolve().parent
    original = here / "candidates/internal_single_f72a"
    target = here / "candidates/constant_generation_v1"
    # 已用于CPU/NPU的目录不可覆写；再次执行只核对两处变换，保持证据不可变。
    exists = target.exists()
    if not exists:
        shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, core in (("cube_schedule.py", "worker"), ("vector_schedule.py", "core")):
        expected = transformed((original / name).read_text(), core)
        if exists:
            assert ast.dump(ast.parse((target / name).read_text())) == ast.dump(ast.parse(expected)), name
        else:
            (target / name).write_text(expected)


if __name__ == "__main__":
    main()
