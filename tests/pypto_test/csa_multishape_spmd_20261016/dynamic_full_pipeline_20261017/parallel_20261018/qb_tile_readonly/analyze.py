"""Q_B M128/N256 与待讨论 M64/N512 的有界只读模型，非性能模拟器。

依据现有动态T循环和24核所有权，统计逻辑输入载荷、有效元素、每核工作量。
同时读取既有phase端点与真实PTO。只写本目录小文本，不编译、不上卡、不改源。
字节数不是HBM实测流量，端点区间不是纯MMA耗时，不据此换算预测加速比。
"""

import ast
import json
import re
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path("/data/pyptouser/qinchuanyu/pto-eager")
CANONICAL = ROOT / "vllm-ascend-dsv4-pto-0251rc1"
PACKAGE = CANONICAL / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
PAR = CANONICAL / "tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018"
JOB = PAR / "attention_tick_h131072_b16_v1"
ACTUAL = PAR / "pv_n256_h131072_b16_v1/build_output"
INPUTS = set()


def read(path):
    INPUTS.add(path)
    return path.read_text()


def tile_model(tokens, m, n, owner_mode):
    nblocks = 32768 // n
    mblocks = (tokens + m - 1) // m
    owners = [dict(valid_elements=0, tile_calls=0, weight_bytes=0, qr_bytes=0) for _ in range(24)]
    for ni in range(nblocks):
        for mi in range(mblocks):
            rows = min(m, tokens - mi * m)
            if owner_mode == "column":
                owner = ni % 24
            elif owner_mode == "n_major_flat":
                owner = (ni * mblocks + mi) % 24
            else:
                assert owner_mode == "m_major_flat"
                owner = (mi * nblocks + ni) % 24
            item = owners[owner]
            item["valid_elements"] += rows * n
            item["tile_calls"] += 1
            item["weight_bytes"] += 1024 * n
            item["qr_bytes"] += rows * 1024
    assert sum(v["valid_elements"] for v in owners) == tokens * 32768
    return dict(m=m, n=n, owner_mode=owner_mode, mblocks=mblocks, nblocks=nblocks,
        owner_load=owners, max_valid_elements=max(v["valid_elements"] for v in owners),
        min_valid_elements=min(v["valid_elements"] for v in owners),
        total_weight_bytes=sum(v["weight_bytes"] for v in owners),
        total_qr_bytes=sum(v["qr_bytes"] for v in owners),
        single_tile_left_bytes=m * 256, single_tile_right_bytes=256 * n,
        single_tile_acc_bytes=m * n * 4)


def phase_summary():
    rounds = []
    for index in range(3):
        rows = json.loads(read(JOB / f"clock_round_{index}_spmd_reference/phase_clock.json"))
        prior = {r["core"]: r for r in rows if r["name"] == "IDX_QUERY_HAD"}
        qb = {r["core"]: r for r in rows if r["name"] == "QB"}
        assert len(prior) == len(qb) == 24
        per_core = [{"core": core, "nblocks": 6 if core < 8 else 5,
            "prior_notify_end_us": prior[core]["notify_end_us"],
            "qb_notify_start_us": qb[core]["notify_start_us"],
            "interval_us": round(qb[core]["notify_start_us"] - prior[core]["notify_end_us"], 3)} for core in range(24)]
        groups = {}
        for count in (5, 6):
            values = [r["interval_us"] for r in per_core if r["nblocks"] == count]
            groups[str(count)] = {"min": min(values), "median": statistics.median(values), "max": max(values)}
        rounds.append(dict(round=index, cores=per_core, by_column_block_count=groups))
    return rounds


def actual_qb():
    tree = ast.parse(read(PACKAGE / "query_projection_stages.py"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_q_b_nz_stage")
    fused = ACTUAL / "decode_csa_spmd_q1m9wnpj/ptoas/dynamic_csa_full.pto"
    original = ACTUAL / "_decode_csa_tp1_layer_22vi3w5d/ptoas/qproj_matmul.pto"
    outputs = {}
    for label, path in (("fused", fused), ("original", original)):
        records, allocation = [], []
        for line_number, line in enumerate(read(path).splitlines(), 1):
            if label == "fused":
                location = re.search(r'query_projection_stages.py":(\d+):', line)
                if not location or not (fn.lineno <= int(location[1]) <= fn.end_lineno):
                    continue
            if "pto.alloc_tile" in line:
                found = re.search(r"addr = %c(\d+)_i64.*?loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", line)
                if found:
                    address, space, dtype, m, n = found.groups()
                    allocation.append(dict(line=line_number,address=int(address),space=space,dtype=dtype,rows=int(m),cols=int(n)))
            if any(key in line for key in ("pto.alloc_tile", "pto.tmatmul", "pto.tstore", "arith.minsi", "scf.for")):
                records.append(f"{line_number}: {line}")
        (HERE / f"{label}_qb.pto.txt").write_text("\n".join(records) + "\n")
        outputs[label] = dict(path=str(path), allocation=allocation)
    return outputs


def main():
    cases = []
    for tokens in (24, 48, 96, 144, 168, 192):
        cases.append(dict(tokens=tokens, current=tile_model(tokens,128,256,"column"),
            candidate=[tile_model(tokens,64,512,mode) for mode in ("column","n_major_flat","m_major_flat")]))
    for name in ("cube_schedule.py", "vector_schedule.py", "binding.py", "qkv_stages.py"):
        read(PACKAGE / name)
    backend = ROOT / ".worktrees/pypto-csa-static-fixes-20261016/include/pypto/backend/910B/backend_910b_handler.h"
    backend_source = read(backend)
    assert "GetL0bCapacityBytes() const override { return 64ULL * 1024; }" in backend_source
    assert "GetL0cCapacityBytes() const override { return 128ULL * 1024; }" in backend_source
    phases = phase_summary()
    artifacts = actual_qb()
    report = dict(scope="有界只读，不编译/设备/预测时延", conclusion="不建议据当前证据启动M64/N512/K256 CPU候选",
        math=["同一INT8 1024项点积最大绝对和16777216，不发生INT32溢出；仅分割K不改数学值",
            "硬件实际Left/Right/Acc与Tile声明/源码列块不能混同；Mat调用可由AutoTile进一步分割"],
        resource_capacity_bytes=dict(left=65536,right=65536,acc=131072,mat=524288),
        cases=cases, phase_rounds=phases, actual_artifacts=artifacts,
        inputs=[dict(path=str(path),bytes=path.stat().st_size) for path in sorted(INPUTS)])
    (HERE / "evidence.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    for row in cases:
        old=row["current"];new=row["candidate"][0]
        mib=lambda n:round(n/(1024*1024),3)
        print(row["tokens"],"weights",mib(old["total_weight_bytes"]),mib(new["total_weight_bytes"]),
            "QR",mib(old["total_qr_bytes"]),mib(new["total_qr_bytes"]),
            "max valid elements",old["max_valid_elements"],[v["max_valid_elements"] for v in row["candidate"]])


if __name__ == "__main__":
    main()
