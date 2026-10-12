"""只审阅源码分工与编译产物；不加载 NPU、不生成性能结论。"""

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def variants(name):
    tree = ast.parse((ROOT / f"{name}.py").read_text())
    entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hc_pre")
    calls = [n for n in ast.walk(entry) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_hc_pre_variant"]
    result = {}
    for call in calls:
        args = [ast.literal_eval(n) for n in call.args[7:]]
        m, k, splits, _, slots, tail_rows, tag, _, rms_splits, scalar, *stream = args
        result[tag] = dict(m=m, k=k, splits=splits, slots=slots, tail_rows=tail_rows,
                           rms_splits=rms_splits, scalar_mix=scalar,
                           stream_k=stream[0] if stream else (1024 if "exact96" in tag else k))
    return result


def tag_for(tokens):
    name = ("small" if tokens <= 32 else "medium48" if tokens <= 48 else
            "medium96" if tokens < 96 else "exact96" if tokens == 96 else "large")
    return f"hc_{name}_spmd"


def schedule(tokens, variant):
    m, splits = variant["m"], variant["splits"]
    blocks = (tokens + m - 1) // m
    units = blocks * splits
    return {
        **variant,
        "tokens_padded": blocks * m,
        "linear_units": units,
        "per_cluster_tasks": [len(range(c, units, 24)) for c in range(24)],
        "partial_bytes": blocks * m * (splits * 32 + variant["rms_splits"]) * 4,
        # 按有效权重元素计的逻辑读取量；不是实测 HBM/L2 流量。
        "logical_weight_read_bytes": blocks * 24 * 16384 * 4,
    }


def tail_schedule(tokens):
    pairs = (tokens + 1) // 2
    sinks = min(tokens, 24)
    mixers = min(48 - sinks, pairs * 8)
    sink_rows = [tokens // sinks + (i < tokens % sinks) for i in range(sinks)]
    mix_work = []
    coverage = [[0] * 8 for _ in range(tokens)]
    for i in range(mixers):
        if mixers <= pairs:
            begin = 2 * (i * (pairs // mixers) + min(i, pairs % mixers))
            end = min(tokens, begin + 2 * (pairs // mixers + (i < pairs % mixers)))
            d0, d1 = 0, 8
        else:
            width, extra = divmod(mixers, pairs)
            prefix = extra * (width + 1)
            if i < prefix:
                pair, lane, lanes = i // (width + 1), i % (width + 1), width + 1
            else:
                pair, lane, lanes = extra + (i - prefix) // width, (i - prefix) % width, width
            begin, end = 2 * pair, min(tokens, 2 * pair + 2)
            d0 = lane * (8 // lanes) + min(lane, 8 % lanes)
            d1 = d0 + 8 // lanes + (lane < 8 % lanes)
        for row in range(begin, end):
            for d in range(d0, d1):
                coverage[row][d] += 1
        mix_work.append(dict(begin=begin, end=end, d_begin=d0, d_end=d1))
    return dict(sink_workers=sinks, sink_rows=sink_rows, mix_workers=mixers,
                mix_work=mix_work, all_vectors_have_work=sinks + mixers == 48,
                mixed_single_writer=all(n == 1 for row in coverage for n in row))


def signatures(candidate):
    build = ROOT / f"build_{candidate}"
    result = {"status": json.loads((build / "report.json").read_text())["status"]}
    for path in sorted((build / "build/ptoas").glob("hc_*.cpp")):
        text = path.read_text()
        matches = re.findall(r"AICORE void (hc_\w+)\(([^\n]+)\) \{", text)
        result[path.stem] = {name: dict(total_arguments=len(args.split(",")),
                                       gm_pointer_arguments=args.count("__gm__"))
                             for name, args in matches}
    return result


def main():
    old = variants("adopted_scalar_medium")
    new = variants("stage1_fewer_partials_db_v1")
    cases = []
    for b in [2, 4, 8, 16, 20, 24, 28]:
        for s in [5, 6]:
            tokens = b * s
            tag = tag_for(tokens)
            cases.append(dict(b=b, s=s, tokens=tokens, baseline=schedule(tokens, old[tag]),
                              candidate=schedule(tokens, new[tag]), tail=tail_schedule(tokens)))
    result = dict(device_execution=False, performance_measured=False, production_adopted=False,
                  cases=cases, compiled={name: signatures(name) for name in [
                      "stage1_fewer_partials_v1", "stage1_fewer_partials_db_v1", "tail_schedule_incore_v1"]})
    (ROOT / "fewer_partials_static_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("已保存14档静态分工及三个候选的生成函数参数表；未执行设备测试。")


if __name__ == "__main__":
    main()
