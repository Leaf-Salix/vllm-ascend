# SPDX-License-Identifier: Apache-2.0
"""检查完整编译产物的权重跨M存活、容量、TLOAD与MMA顺序；不代替真机验证。"""

import argparse
import ast
import json
from pathlib import Path
import re


def region(annotation):
    refs = [n for n in ast.walk(annotation) if isinstance(n, ast.Call)
            and ast.unparse(n.func) == "pl.MemRef"]
    assert len(refs) == 1
    ref = refs[0]
    return (ast.unparse(ref.args[0]), ast.literal_eval(ref.args[1].args[0]), ast.literal_eval(ref.args[2]))


def overlap(a, b):
    return a[0] == b[0] and a[1] < b[1] + b[2] and b[1] < a[1] + a[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.compile_dir.resolve()
    result = json.loads((root / "result.json").read_text())
    assert result["passed"] and result["single_mix"] and len(result["symbolic_dimensions"]) == 58
    ir_path, = (root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
    tree = ast.parse(ir_path.read_text())
    weights, right, reuse_qr = {}, {}, {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        name = node.target.id
        if re.fullmatch(r"stream_w[0-3]_inline\d+__ssa_v0", name):
            weights[name] = region(node.annotation)
        if re.fullmatch(r"(?:stream|reuse)_right[0-3]_inline\d+__ssa_v0", name):
            right[name] = region(node.annotation)
        if re.fullmatch(r"reuse_qr[0-3]_inline\d+__ssa_v0", name):
            reuse_qr[name] = region(node.annotation)
    assert len(weights) == 4 and len(right) == 8 and len(reuse_qr) == 4
    assert all(r[2] == 65536 for r in weights.values())
    assert len({(r[0], r[1], r[2]) for r in right.values()}) == 1
    assert next(iter(right.values()))[1:] == (0, 65536)
    values = list(weights.values())
    assert not any(overlap(a, b) for i, a in enumerate(values) for b in values[i+1:])
    assert not any(overlap(a, b) for a in weights.values() for b in reuse_qr.values())
    tail_loop, = [n for n in ast.walk(tree) if isinstance(n, ast.For)
                  and "stream_t0" in ast.unparse(n.target)]
    tail_loads = [n for n in ast.walk(tail_loop) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) == "pl.tile.load"]
    assert len(tail_loads) == 4
    assert all(ast.unparse(n.args[0]).startswith("qr_matmul") for n in tail_loads)
    tail_extracts = [n for n in ast.walk(tail_loop) if isinstance(n, ast.Call)
                     and ast.unparse(n.func) == "pl.tile.extract"]
    assert {ast.unparse(n.args[0]) for n in tail_extracts} == set(weights)

    cpp_path = root / "kernel/ptoas/dynamic_csa_full.cpp"
    cpp = cpp_path.read_text()
    start = cpp.index("// pto: %stream_w0_")
    loop_start = cpp.rfind("for (", 0, start)
    brace_start = cpp.index("{", loop_start)
    depth, end = 1, brace_start + 1
    while depth:
        depth += (cpp[end] == "{") - (cpp[end] == "}")
        end += 1
    part = cpp[loop_start:end]
    math = list(re.finditer(r"\bTMATMUL(?:_ACC)?\([^;]+;", part))
    assert len(math) == 8
    cpp_weights = {}
    for index in range(4):
        marker = part.index(f"// pto: %stream_w{index}_")
        tile = re.search(r"\b(v\d+) = Tile<TileType::Mat", part[marker:]).group(1)
        loads = list(re.finditer(r"\bTLOAD\(" + tile + r", ([^)]+)\);", part))
        assert len(loads) == 1
        load = loads[0]
        cpp_weights[str(index)] = dict(tile=tile, load_offset=load.start(), load=load.group())
    assert cpp_weights['2']['load_offset'] > math[0].end()
    assert cpp_weights['3']['load_offset'] > math[1].end()
    assert cpp_weights['3']['load_offset'] < math[2].start()
    # 被覆盖的是已移到Left的首M QR；同步是MTE1释放Mat后MTE2写入，
    # 而不是等MMA完成。这个审计只检查实际存在的事件，不宣称硬件已充分重叠。
    between = [part[math[i].end():cpp_weights[str(i+2)]['load_offset']] for i in (0,1)]
    assert all("wait_flag(PIPE_MTE1, PIPE_MTE2," in s for s in between)
    assert all("wait_flag(PIPE_M, PIPE_MTE2," not in s for s in between)
    report = dict(passed=True, cpu_compilation=str(root), allocated_ir=str(ir_path), cpp=str(cpp_path),
                  symbolic_dimensions=58, single_mix=True, weights=weights, right=right, reuse_qr=reuse_qr,
                  total_weight_bytes=262144, tail_gm_loads_only_qr=True,
                  tail_weight_handles_identical=True, weight_cpp_loads=cpp_weights,
                  k2_after_first_mma=True, k3_after_second_mma=True,
                  extra_mma_completion_wait_before_k2_k3=False,
                  note="生成代码机制与资源审计；不代替真机精度、流水重叠或完整L1收益")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print({"passed":True,"weight_bytes":262144,"Right_bytes":65536,"tail_weight_gm_loads":0})


if __name__ == "__main__":
    main()
