# SPDX-License-Identifier: Apache-2.0
"""Port the N64 WS column pool without replacing TND geometry or scheduling."""

import ast
import hashlib
import json
import shutil
from pathlib import Path


def statements(text):
    return ast.parse(text).body


class PoolWrites(ast.NodeTransformer):
    def __init__(self):
        self.changed = []

    def visit_Assign(self, node):
        name = ast.unparse(node.targets[0])
        if name == "split_ws_acc":
            self.changed.append(name)
            return statements("""
split_ws_pool = pl.tile.create(
    [NATIVE_QLI_WEIGHT_ROWS, SPLIT_WS_POOL_PANELS * qk_panel_cols],
    dtype=pl.FP32, target_memory=pl.MemorySpace.Acc,
)
""")
        if name == "split_ws_first":
            self.changed.append(name)
            return statements("""
split_ws_col = pl.max(buf_qk_panel - 1, 0) % SPLIT_WS_POOL_PANELS * qk_panel_cols
split_ws_slot = pl.tile.slice(
    split_ws_pool, [NATIVE_QLI_WEIGHT_ROWS, qk_panel_cols], [0, split_ws_col],
)
split_ws_first = pl.tile.matmul_acc(
    split_ws_slot, split_coefficient0, split_right0, init_cond=True,
)
""")
        if name == "split_ws_first_last":
            self.changed.append(name)
            return statements("""
split_ws_last_col = (SPLIT_WS_POOL_PANELS - 1) * qk_panel_cols
split_ws_last_slot = pl.tile.slice(
    split_ws_pool, [NATIVE_QLI_WEIGHT_ROWS, qk_panel_cols], [0, split_ws_last_col],
)
split_ws_first_last = pl.tile.matmul_acc(
    split_ws_last_slot, split_coefficient0, split_right0_last, init_cond=True,
)
""")
        return self.generic_visit(node)

    def visit_Expr(self, node):
        call = node.value
        if not isinstance(call, ast.Call) or ast.unparse(call.func) != "pl.store":
            return self.generic_visit(node)
        value = ast.unparse(call.args[0])
        if value == "pl.set_validshape(split_ws_done, query_group_size, qk_panel_cols)":
            self.changed.append("store_body")
            return statements("""
split_ws_pool = pl.tile.assemble(split_ws_pool, split_ws_done, [0, split_ws_col])
if buf_qk_panel % SPLIT_WS_POOL_PANELS == 0:
    pl.store(
        pl.set_validshape(split_ws_pool, query_group_size, SPLIT_WS_POOL_PANELS * qk_panel_cols),
        [buf_transfer_row, pl.max(buf_panel_col - SPLIT_WS_POOL_PANELS * qk_panel_cols, 0)],
        buf_score_transfer,
    )
""")
        assert value == "pl.set_validshape(split_ws_done_last, query_group_size, qk_panel_cols)"
        self.changed.append("store_last")
        return statements("""
split_ws_pool = pl.tile.assemble(split_ws_pool, split_ws_done_last, [0, split_ws_last_col])
pl.store(
    pl.set_validshape(split_ws_pool, query_group_size, SPLIT_WS_POOL_PANELS * qk_panel_cols),
    [buf_transfer_row, score_tile - SPLIT_WS_POOL_PANELS * qk_panel_cols], buf_score_transfer,
)
""")


def transform(source):
    tree = ast.parse(source)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_indexer_score_cube_split_store"
    )
    original = ast.get_source_segment(source, fn)
    change = PoolWrites()
    fn = change.visit(fn)
    assert change.changed == ["split_ws_acc", "split_ws_first", "store_body", "split_ws_first_last", "store_last"]
    fn.body[0] = ast.Expr(
        value=ast.Constant(
            value="""长G6/N64：双48KiB QK早FIX与32KiB WS列池。

    每个列窗保留K192+192次序、FP16边界和init；八个列窗合并写原FIFO。
    重用池必须等待此前FIX读取完成，编译后验证真实地址和事件。
    TND group/leaf/page/tail与READY/ACK协议保持，只有现有SPLIT域进入。
    """
        )
    )
    fn.decorator_list = []
    replacement = ast.unparse(ast.fix_missing_locations(fn))
    result = source.replace(original, replacement, 1)
    marker = "@pl.jit.inline(auto_scope=False)\ndef _indexer_score_cube_split_store"
    assert result.count(marker) == 1
    result = result.replace(marker, "SPLIT_WS_POOL_PANELS = 8\n\n\n" + marker, 1)
    ast.parse(result)
    return result


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.glob("*.py")} == expected
    target = here / "runs/score_ws_pool"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    name = "indexer_score_stages.py"
    (target / name).write_text(transform((parent / name).read_text()))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")}
    assert [n for n in hashes if hashes[n] != expected[n]] == [name]
    report = {
        "baseline": "frozen_tnd_v122",
        "source": str(target),
        "files": hashes,
        "modified_files": [name],
        "reference_mechanism": "csa_critical_path_20261020/prepare_split_pool.py in the other BSH SPMD session",
        "enabled_domain": "existing SPLIT_SCORE_ROWS, balanced G6/N64/1024; no policy change",
        "expected_acc_bytes": {"qk0": 49152, "qk1": 49152, "ws_pool": 32768},
        "expected_ws_stores_per_1024": 2,
        "formal_device_validation": "not run",
    }
    with (here / "score_ws_pool.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(target)


if __name__ == "__main__":
    main()
