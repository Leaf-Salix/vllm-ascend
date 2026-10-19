# SPDX-License-Identifier: Apache-2.0
"""只用现有 tensor.view 复现 FlattenTileNdTo2D 的定义/使用引用失配。

不使用新 gm_view 标志，不初始化 NPU。pitch 来自 Tile 的 Scalar 读取，
执行期只在 pitch==1024 时建立 [4,512]→[2,1024] 的等元素数 GM 别名。
"""
import json
import sys
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()
import pypto.language as pl
from pypto import ir, passes
from pypto.ir.printer import python_print


@pl.program
class Repro:
    @pl.function(type=pl.FunctionType.InCore)
    def core(self, x: pl.Tensor[[4, 512], pl.BF16], indices: pl.Tensor[[1, 16], pl.INT32],
             out: pl.Tensor[[2, 512], pl.BF16]) -> pl.Tensor[[2, 512], pl.BF16]:
        ids = pl.tile.load(indices, [0, 0], [1, 16])
        pitch = pl.cast(pl.tile.read(ids, [0, 0]), pl.INDEX)
        if pitch == 1024:
            alias = pl.tensor.view(x, shape=[2, pitch])
            value = pl.tile.load(alias, [0, 0], [2, 512])
            out = pl.tile.store(value, [0, 0], out)
        return out


def main():
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=False)
    before = python_print(Repro)
    (folder / "before.py").write_text(before)
    after = passes.flatten_tile_nd_to_2d()(Repro)
    text = python_print(after)
    (folder / "after.py").write_text(text)
    result = dict(free_vars_before="__FREE_VAR" in before,
                  free_vars_after="__FREE_VAR" in text,
                  uses_new_gm_view=False, device_execution=False)
    (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
