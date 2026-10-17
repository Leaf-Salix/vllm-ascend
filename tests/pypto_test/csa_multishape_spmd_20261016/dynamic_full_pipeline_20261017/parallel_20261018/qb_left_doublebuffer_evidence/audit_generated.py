#!/usr/bin/env python3
"""只读提取已生成 PTO/C++：地址、有效尾行、四段点积与生成事件。

不编译、不导入 PyPTO、不访问设备。基线取 d56 前移 compact 的真实设备
编译物；候选取完整 symbolic CPU 编译。二者区间策略相同，动态轴未冻结。
地址上界是 allocation 的最大末端，不是把各个生命周期的字节量相加。
"""

import argparse
import ast
import collections
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path("/data/pyptouser/qinchuanyu/pto-eager")
PAR = ROOT / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018"
BASE = PAR / "compact_priority_h131072_b16_v1/build_output/decode_csa_spmd_ad5etrbf"
INPUTS = set()


def read(path):
    INPUTS.add(path)
    return path.read_text()


def functions(source):
    result = {}
    for p in source.glob("*.py"):
        result[p.name] = [(n.lineno, n.end_lineno, n.name) for n in ast.parse(read(p)).body
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return result


def allocations(path, source):
    bounds = functions(source)
    rows = []
    function = ""
    for line_no, line in enumerate(read(path).splitlines(), 1):
        fn = re.search(r"func.func @(\w+)", line)
        if fn:
            function = fn[1]
        if "pto.alloc_tile" not in line:
            continue
        m = re.search(r"(%\w+) = pto.alloc_tile addr = %c(\d+)_i64.*?loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", line)
        assert m, (path, line_no, line)
        loc = re.search(r' loc\("([^"]+)":(\d+):(\d+)\)', line)
        assert loc
        name, source_line = Path(loc[1]).name, int(loc[2])
        owners = [fn for lo, hi, fn in bounds.get(name, []) if lo <= source_line <= hi]
        assert len(owners) <= 1
        bits = int(re.search(r"\d+", m[4])[0])
        size = int(m[5]) * int(m[6]) * bits // 8
        rows.append(dict(function=function, source_file=name, source_function=owners[0] if owners else None,
                         source_line=source_line, ir_line=line_no, variable=m[1],
                         address=int(m[2]), space=m[3], dtype=m[4],
                         shape=[int(m[5]), int(m[6])], size_bytes=size, end=int(m[2])+size))
    return rows


def max_ends(rows):
    result = {}
    for row in rows:
        key = row["function"] + "/" + row["space"]
        result[key] = max(result.get(key, 0), row["end"])
    return result


def key(row):
    return (row["function"], row["source_file"], row["source_function"], row["space"],
            row["dtype"], *row["shape"], row["address"], row["size_bytes"])


def is_qb(row):
    return row["source_file"] == "query_projection_stages.py" and row["source_function"] == "_q_b_nz_stage"


def qbody(pto, source):
    bounds = functions(source)["query_projection_stages.py"]
    lo, hi, _ = next(row for row in bounds if row[2] == "_q_b_nz_stage")
    result = []
    for no, line in enumerate(read(pto).splitlines(), 1):
        m = re.search(r'query_projection_stages.py":(\d+):', line)
        if m and lo <= int(m[1]) <= hi:
            result.append((no, line))
    return result


def cpp_excerpt(path, label, output):
    lines = read(path).splitlines()
    start = next(i for i, line in enumerate(lines) if "// pto: %qr_first" in line)
    store = next(i for i in range(start, len(lines)) if "TSTORE(" in lines[i])
    rows = lines[start:store+5]
    (output / f"{label}_qb.cpp.txt").write_text("\n".join(f"{start+i+1}: {line}" for i, line in enumerate(rows))+"\n")
    operations = []
    for i, line in enumerate(rows, start+1):
        if re.search(r"\b(TLOAD|TMOV|TMATMUL|TMATMUL_ACC|TSTORE|wait_flag|set_flag|pipe_barrier)\(", line):
            operations.append(dict(line=i, text=line.strip()))
    return dict(path=str(path), first_line=start+1, last_line=store+5, operations=operations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=("v1", "v2"))
    args = parser.parse_args()
    candidate = OUT / f"cpu_{args.version}/kernel"
    source = OUT / ("source_candidate" if args.version == "v1" else "source_candidate_v2")
    evidence = OUT / f"generated_{args.version}"
    evidence.mkdir(exist_ok=True)
    pto_before, pto_after = BASE / "ptoas/dynamic_csa_full.pto", candidate / "ptoas/dynamic_csa_full.pto"
    before = allocations(pto_before, OUT / "source_before")
    after = allocations(pto_after, source)
    old_other = collections.Counter(key(r) for r in before if not is_qb(r))
    new_other = collections.Counter(key(r) for r in after if not is_qb(r))
    qb_before, qb_after = [r for r in before if is_qb(r)], [r for r in after if is_qb(r)]
    text_before, text_after = qbody(pto_before, OUT / "source_before"), qbody(pto_after, source)
    for label, rows in (("baseline", text_before), ("candidate", text_after)):
        (evidence / f"{label}_qb.pto.txt").write_text("\n".join(f"{n}: {line}" for n,line in rows)+"\n")
    operations = {}
    for label, rows in (("baseline", text_before), ("candidate", text_after)):
        tally = collections.Counter(re.search(r"\b(pto\.\w+(?:\.\w+)?)\b", line)[1]
                                    for _,line in rows if "pto." in line and " = pto.alloc_tile" not in line)
        operations[label] = dict(tally)
        assert tally["pto.tload"] == 8
        assert tally["pto.tmatmul"] == 1 and tally["pto.tmatmul.acc"] == 3
        assert tally["pto.tstore"] == 1
    valid_rows = {}
    for label, rows in (("baseline", text_before), ("candidate", text_after)):
        text = "\n".join(line for _,line in rows)
        matches = re.findall(r"pto.alloc_tile.*?valid_row = (%\w+).*?loc=(left|acc),", text)
        variables = sorted({var for var,space in matches})
        assert len(variables) == 1
        var = variables[0]
        assert re.search(re.escape(var)+r" = arith.minsi %\w+, %c128_index", text)
        assert re.search(r"sizes = \["+re.escape(var)+r", %c256_index\].*?partition_tensor_view<\?x256xi32>", text)
        valid_rows[label] = dict(shared_left_acc_valid_row=var, min_tokens_minus_t0_128=True,
                                 store_rows_same_symbol=True, store_columns=256)
    result = dict(scope="完整动态单 SPMD 的一次已有 CPU 生成物只读审计；无设备执行/性能结论",
                  version=args.version, baseline=str(BASE), candidate=str(candidate),
                  address_upper_bounds_before=max_ends(before), address_upper_bounds_after=max_ends(after),
                  other_helpers=dict(equal=old_other==new_other, groups_before=len(old_other), groups_after=len(new_other),
                                     removed=[dict(key=list(k), count=n) for k,n in (old_other-new_other).items()],
                                     added=[dict(key=list(k), count=n) for k,n in (new_other-old_other).items()]),
                  qb_allocations_before=qb_before, qb_allocations_after=qb_after,
                  operation_counts=operations, valid_shape=valid_rows,
                  cpp_before=cpp_excerpt(BASE / "kernels/aic/cube_schedule.cpp", "baseline", evidence),
                  cpp_after=cpp_excerpt(candidate / "kernels/aic/cube_schedule.cpp", "candidate", evidence),
                  source_archive_commit="d56c9eb6", no_fixed_address=True, no_new_phase_or_root=True,
                  limitations=["Mat/Left/Right地址本身不能证明流水有收益，必须同时读事件及实际时间。",
                               "有效形状源码及生成物吻合不代替设备保护区/精度验证。",
                               "地址上界按物理Tile尺寸计算；不是累加各阶段的瞬时活跃量。"])
    result["inputs"] = [dict(path=str(p), bytes=p.stat().st_size) for p in sorted(INPUTS)]
    (evidence / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(version=args.version, other_helpers_equal=old_other==new_other,
                         bounds=max_ends(after), qb_left_addresses=sorted({r["address"] for r in qb_after if r["space"]=="left"}),
                         output=str(evidence)),ensure_ascii=False))


if __name__ == "__main__":
    main()
