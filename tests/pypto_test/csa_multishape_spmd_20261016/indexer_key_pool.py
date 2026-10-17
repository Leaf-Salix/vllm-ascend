# SPDX-License-Identifier: Apache-2.0
"""只做CPU候选：扩大Key的L1行池，保持16个N64面板的提交次序。

原128行池每两个面板复用一次。本候选用1024行，使一个分数块的32个
Native页各有独立L1位置，检查生成器能否减少MTE1→MTE2复用等待。
这不是预取/并发度的证明；实际代码没有减少等待时不提交设备。
"""

import argparse
import ast
import json
from pathlib import Path


def transform(loop):
    maths = [ast.dump(n) for n in ast.walk(loop) if isinstance(n, ast.Call)
             and ast.unparse(n.func) in ("pl.tile.matmul", "pl.tile.assemble",
                                        "pl.tile.store", "pl.system.sync_wait",
                                        "pl.system.sync_set")]
    pages = []
    for node in ast.walk(loop):
        if not isinstance(node, ast.If) or not ast.unparse(node.test).endswith("% 128 == 0"):
            continue
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == "pl.tile.gather_row"]
        if len(calls) != 2 or any(ast.unparse(n.args[4]) != "[32, 128]" for n in calls):
            raise ValueError("Key页的奇偶加载合同改变")
        row = len(pages) * 32
        sources = [ast.unparse(n.args[1]) + ":" + ast.unparse(n.args[3]) for n in calls]
        for call in calls:
            call.args[2].elts[0] = ast.Constant(value=row)
        pages.append({"l1_row": row, "sources": sources})
    if len(pages) != 32:
        raise ValueError("需原16个N64面板的32个Key页")
    extracts = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "pl.tile.extract"
                and ast.unparse(n.args[3]) == "[128, 64]"]
    if len(extracts) != 16:
        raise ValueError("原Key Right提取不是16处")
    for panel, node in enumerate(extracts):
        node.args[2] = ast.Constant(value=panel * 64)
    creates = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
               and ast.unparse(n.func) == "pl.tile.create"
               and ast.unparse(n.args[0]) == "[128, 128]"]
    if len(creates) != 1:
        raise ValueError("Key L1池不是唯一创建")
    creates[0].args[0] = ast.parse("[1024, 128]", mode="eval").body
    for node in ast.walk(loop):
        if not isinstance(node, ast.AnnAssign):
            continue
        annotation = ast.unparse(node.annotation)
        if annotation.startswith("pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat"):
            shape = "[128, 1024]" if "pl.TileView" in annotation else "[1024, 128]"
            node.annotation = ast.parse(annotation.replace("[128, 128]", shape), mode="eval").body
    after = [ast.dump(n) for n in ast.walk(loop) if isinstance(n, ast.Call)
             and ast.unparse(n.func) in ("pl.tile.matmul", "pl.tile.assemble",
                                        "pl.tile.store", "pl.system.sync_wait",
                                        "pl.system.sync_set")]
    assert after == maths
    return {"key_pool_bytes_before": 16384, "key_pool_bytes_after": 131072,
            "pages": pages, "key_right_offsets": list(range(0, 1024, 64)),
            "math_and_store_and_sync_calls_unchanged": True,
            "prefetch_depth_changed": False, "device_validated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = ast.parse(args.source.read_text())
    fn = next(n for n in ast.walk(module)
              if isinstance(n, ast.FunctionDef) and n.name == "csa_static_aic")
    loop = next(n for n in ast.walk(fn)
                if isinstance(n, ast.For) and ast.unparse(n.target) == "stage_16_v60")
    audit = transform(loop)
    ast.fix_missing_locations(module)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "entry_jit.py").write_text(ast.unparse(module) + "\n")
    (args.output / "source_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
