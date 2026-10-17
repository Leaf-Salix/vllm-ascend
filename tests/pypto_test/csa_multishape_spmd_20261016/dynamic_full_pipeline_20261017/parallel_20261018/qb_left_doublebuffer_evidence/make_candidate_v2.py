#!/usr/bin/env python3
"""仅重排 v1 的现有 load/move，保持 Q_B 原有 Mat 活跃窗口。

输入是已冻结 v1；输出为新目录，拒绝覆盖。没有新 Tensor、地址约束或数学
操作。K256/K512 的四份 Mat 先共同存活，避免 v1 将全部 Mat 串行复用。
Left1 在 MMA0 前生成，但 Right0 先于 Left1，供后端在 Right0 后发 MTE1→M
事件；这只是待编译核查的表达，不能以源码顺序声称实现了硬件重叠。
"""

import ast
import collections
import difflib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
BEFORE = OUT / "source_candidate"
AFTER = OUT / "source_candidate_v2"
NAME = "query_projection_stages.py"


def main():
    files = sorted(BEFORE.glob("*.py"))
    assert len(files) == 33
    AFTER.mkdir(exist_ok=False)
    for path in files:
        shutil.copy2(path, AFTER / path.name)
    original = (BEFORE / NAME).read_text()
    source = original
    # K0/768 各自两份 Mat 同时活跃，保持原 QR 与权重 L1 窗口分离。
    for phase in ("first", "tail"):
        move = f"            qr_{phase}_left = pl.tile.move(qr_{phase}_mat, target_memory=pl.MemorySpace.Left)\n"
        right = f"            wq_{phase}_right = pl.tile.move(wq_{phase}_mat, target_memory=pl.MemorySpace.Right)\n"
        assert source.count(move) == source.count(right) == 1
        source = source.replace(move, "").replace(right, move + right)
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_q_b_nz_stage")
    pair = next(n for n in ast.walk(function) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "k0")
    assert len(pair.body) == 10
    names = [n.targets[0].id for n in pair.body]
    assert names == ["qr_pair0_mat", "qr_pair0_left", "qr_pair1_mat", "qr_pair1_left",
                     "wq_pair0_mat", "wq_pair0_right", "col_acc", "wq_pair1_mat", "wq_pair1_right", "col_acc"]
    lines = source.splitlines(keepends=True)
    blocks = ["".join(lines[n.lineno-1:n.end_lineno]) for n in pair.body]
    order = [0, 4, 2, 7, 1, 5, 3, 6, 8, 9]
    annotations = {
        0: "                # 两份 QR Mat 与两份 weight Mat 共 192 KiB；四次 load 先完成源码\n                # 建立，再消费任何一份，恢复原 pair 的 L1 双槽活跃窗口。\n",
        1: "                # 先提交 Left0/Right0，再建立 Left1。两个 Left 在 MMA0 前共同\n                # 活跃；Right0→M 的事件是否可早于 Left1，必须核对生成 C++。\n",
        8: "                # Right 仍只有 64 KiB 单槽；原 M→MTE1 覆盖等待不得删除。\n",
    }
    body = "".join(annotations.get(i, "") + blocks[i] for i in order)
    source = "".join(lines[:pair.body[0].lineno-1]) + body + "".join(lines[pair.body[-1].end_lineno:])
    source = source.replace("NZ Q_B：仅后续K256 pair显式保留两份Left", "NZ Q_B：保留原 Mat 活跃窗口并显式保留两份 Left")
    # 对比除顺序以外的所有 AST 操作；只允许 docstring 内容不同。
    def leaves(text):
        root = ast.parse(text)
        fn = next(n for n in root.body if isinstance(n, ast.FunctionDef) and n.name == "_q_b_nz_stage")
        return collections.Counter(ast.dump(n, include_attributes=False) for n in ast.walk(fn)
                                   if isinstance(n, (ast.Assign, ast.AnnAssign, ast.Return)))
    assert leaves(original) == leaves(source)
    (AFTER / NAME).write_text(source)
    unchanged = [p.name for p in files if p.name != NAME and p.read_bytes() == (AFTER / p.name).read_bytes()]
    assert len(unchanged) == 32
    (OUT / "candidate_v2.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(True), source.splitlines(True), fromfile="v1/"+NAME, tofile="v2/"+NAME)))
    result = dict(source=str(AFTER), frozen_utc=datetime.now(timezone.utc).isoformat(),
                  only_reordered_load_move=True, assignment_multiset_equal=True,
                  unchanged_files=unchanged, pair_live_mat_bytes=196608,
                  first_tail_live_mat_bytes=98304, device_executed=False)
    (OUT / "source_v2_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
