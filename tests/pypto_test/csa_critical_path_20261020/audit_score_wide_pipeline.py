# SPDX-License-Identifier: Apache-2.0
"""核对宽面板的真实内存区间、调用数和FIX/WS/同Acc复用顺序。

这是生成代码机制审计，既不把set/wait个数当耗时，也不替代真机精度或
完整L1验收。QK0与QK1必须共用96KiB Acc；前面板WS用独立8KiB Acc，
必须能在当前QK0的FIX完成等待之前提交。
"""

import argparse
import ast
import json
from pathlib import Path
import re

from audit_score_split_store import mem, overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.compile_dir.resolve()
    result = json.loads((root / "result.json").read_text())
    assert result["passed"] and result["single_mix"] and len(result["symbolic_dimensions"]) == 58
    path, = (root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
    tree = ast.parse(path.read_text())
    cube = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cube_schedule")
    records = {}
    for node in ast.walk(cube):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            region = mem(node.annotation)
            if region:
                records[node.target.id] = dict(memory=region, operation=ast.unparse(node.value), line=node.lineno)
    def one(prefix):
        values = [v for n,v in records.items() if n.startswith(prefix) and n.endswith("__ssa_v0")]
        assert len(values) == 1, (prefix, len(values))
        return values[0]
    qk = one("split_qk_acc_")["memory"]
    ws = one("split_ws_acc_")["memory"]
    assert qk[2] == 98304 and ws[2] == 8192 and not overlap(qk, ws)
    left = [one(f"split_{kind}{half}_")["memory"] for kind in ("query", "coefficient") for half in (0,1)]
    assert sorted(v[2] for v in left) == [6144,6144,24576,24576]
    assert not any(overlap(a,b) for i,a in enumerate(left) for b in left[i+1:])
    right = {n:v["memory"] for n,v in records.items() if re.match(r"split_right[01](?:_last)?_",n)}
    assert len(right) == 16 and all(v[2] == 49152 for v in right.values())
    right_regions = sorted({tuple(v) for v in right.values()})
    assert len(right_regions) == 1, right_regions
    keys = {n:v["memory"] for n,v in sorted(records.items(),key=lambda nv:nv[1]["line"])
            if n.startswith("buf_current_key_") and "pl.tile.extract" in v["operation"]}
    key_sequence = list(keys.values())
    assert len(keys) == 8 and all(v[2] == 16384 for v in key_sequence)
    # 首面板没有上一面板WS，可使用后续WS区域；只对真正同时存活的7个
    # 稳态面板要求分离。后面的C++顺序检查保留首面板两个QK均先于首WS。
    assert all(not overlap(v,right_regions[0]) for v in key_sequence[1:])
    step, = [n for n in ast.walk(cube) if isinstance(n,ast.For) and "buf_score_step" in ast.unparse(n.target)]
    calls = [n for n in ast.walk(step) if isinstance(n,ast.Call)]
    def matches(function, prefix, argument=0):
        return [n for n in calls if ast.unparse(n.func) == function and ast.unparse(n.args[argument]).startswith(prefix)]
    assert len(matches("pl.tile.matmul_acc","split_qk_acc")) == 16
    assert len(matches("pl.tile.matmul_acc","split_ws_acc")) == 8
    assert len(matches("pl.tile.matmul_acc","split_ws_first")) == 8
    assert len(matches("pl.tile.store","indexer_transfer",2)) == 8
    cpp_path = root / "kernel/ptoas/dynamic_csa_full.cpp"
    cpp = cpp_path.read_text()
    def tile(prefix):
        start = cpp.index("// pto: %" + prefix)
        return re.search(r"\b(v\d+) = Tile<TileType::Acc",cpp[start:]).group(1)
    qvar,wvar = tile("split_qk_acc_"),tile("split_ws_acc_")
    def mma(var,suffix=""):
        return list(re.finditer(r"\bTMATMUL"+suffix+r"\("+var+r", [^;]+;",cpp))
    qmm,wfirst,wsecond = mma(qvar),mma(wvar),mma(wvar,"_ACC")
    qfix = list(re.finditer(r"\bTINSERT[^\n]+\(v\d+, "+qvar+r", [^;]+;",cpp))
    assert len(qmm) == len(qfix) == 16 and len(wfirst) == len(wsecond) == 8
    assert len(re.findall(r"\bTSTORE\(v\d+, "+wvar+r"\);",cpp)) == 8
    def events(start,end):
        fragment = cpp[start:end]
        return {
            "set":re.findall(r"set_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)",fragment),
            "wait":re.findall(r"wait_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)",fragment),
        }
    panels = []
    for i in range(8):
        q0,q1,f0,f1 = qmm[2*i],qmm[2*i+1],qfix[2*i],qfix[2*i+1]
        assert q0.start() < f0.start() < q1.start() < f1.start()
        reuse = events(f0.end(),q1.start())
        assert set(reuse["set"]) & set(reuse["wait"]), (i,"同QK Acc复用必须等待FIX0完成",reuse)
        item = dict(panel=i, qk_same_acc_reuse=reuse)
        if i:
            first,second = wfirst[i-1],wsecond[i-1]
            assert f0.start() < first.start() < q1.start() < f1.start() < second.start()
            before_ws0 = events(f0.end(),first.start())
            after_ws0 = events(first.end(),q1.start())
            assert not set(before_ws0["set"]) & set(before_ws0["wait"]), (i,"WS0被当前QK FIX阻塞",before_ws0)
            assert set(reuse["set"]) & set(after_ws0["wait"]), (i,"QK复用等待未放在WS0之后")
            before_ws1 = events(f1.end(),second.start())
            assert not set(before_ws1["set"]) & set(before_ws1["wait"]), (i,"WS1被当前QK FIX阻塞",before_ws1)
            item.update(ws0_before_qk_reuse_wait=True, before_ws0=before_ws0,
                        after_ws0=after_ws0, before_ws1=before_ws1)
        panels.append(item)
    report = dict(passed=True, compile_dir=str(root), ir=str(path), cpp=str(cpp_path),
                  dynamic_axes=58, single_mix=True, actual_cube_panel_cols=128,
                  qk_acc=qk, ws_acc=ws, left=left, ws_right_regions=right_regions,
                  key_right_regions=sorted({tuple(v) for v in keys.values()}),
                  counts=dict(qk=16,qk_fix=16,ws_init=8,ws_acc=8,store=8),
                  acc_variables=dict(qk=qvar,ws=wvar), panel_events=panels,
                  note="内存及事件提交机制证据，不代表真实重叠时长、精度或完整L1收益")
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print({"passed":True,"actual_panel":128,"qk_acc":98304,"ws_acc":8192,"stores":8})


if __name__ == "__main__":
    main()
