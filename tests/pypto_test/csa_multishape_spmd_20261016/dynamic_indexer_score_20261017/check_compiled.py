# SPDX-License-Identifier: Apache-2.0
"""只审计已编译的score MIX CPP：动态有效M/K、FIFO事件次序及实际地址高水位。"""

import ast
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROD = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_score_stages.py"
CONFIGS = {"g2_768": (2, 128, 768), "g6_768": (6, 64, 768), "g2_1024": (2, 128, 1024), "g6_balanced": (6, 64, 1024)}


def body(text, name):
    start = text.index("static __aicore__ void " + name + "(")
    end = text.find("static __aicore__ void ", start + 10)
    return text[start : end if end >= 0 else None]


def constant_values(text):
    result = {name: int(value) for name, value in re.findall(r"const (?:u?int\d+_t) (v\d+) = (-?\d+);", text)}
    for _ in range(3):
        for dest, src in re.findall(r"(?:u?int\d+_t) (v\d+) = \((?:u?int\d+_t)\) (v\d+);", text):
            if src in result:
                result[dest] = result[src]
    return result


def tile(text, source_name):
    pattern = (
        r"// pto: %" + source_name + r"[^\n]*\n\s*;\s*\n"
        r"\s*Tile<([^>]+)> (v\d+) = Tile<[^>]+>\(([^)]+)\);"
    )
    matches = re.findall(pattern, text)
    assert matches, source_name
    return matches


def memory(text):
    constants = constant_values(text)
    tiles = {
        name: (space, dtype, int(rows), int(cols))
        for space, dtype, rows, cols, name in re.findall(
            r"Tile<TileType::(\w+), (\w+), (\d+), (\d+), [^>]+> (v\d+)", text
        )
    }
    sizes = {"int8_t": 1, "uint8_t": 1, "half": 2, "int16_t": 2, "uint16_t": 2, "float": 4, "int32_t": 4, "uint32_t": 4}
    peaks = {}
    dynamic_views = []
    for name, address in re.findall(r"TASSIGN\((v\d+), (v\d+)\);", text):
        assert name in tiles, name
        if address not in constants:
            # 动态gather目的切片用无构造器Tile视图，地址是已有buffer+偏移，不是另一次分配。
            assert re.search(r"Tile<[^>]+> " + name + r";", text), (name, address)
            dynamic_views.append(name)
            continue
        space, dtype, rows, cols = tiles[name]
        end = constants[address] + rows * cols * sizes[dtype]
        peaks[space] = max(peaks.get(space, 0), end)
    limits = {"Mat": 512 * 1024, "Left": 64 * 1024, "Right": 64 * 1024, "Acc": 128 * 1024, "Vec": 192 * 1024}
    assert all(value <= limits[space] for space, value in peaks.items()), peaks
    return {"allocation_high_water_bytes": peaks, "dynamic_alias_views_not_double_counted": dynamic_views}


def main():
    snapshot = HERE / "full_v2/indexer_score_stages.py"

    def executable_ast(text):
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef)) and node.body:
                if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                    if isinstance(node.body[0].value.value, str):
                        node.body.pop(0)
        return ast.dump(tree)

    assert executable_ast(snapshot.read_text()) == executable_ast(PROD.read_text())
    result = {
        "status": "PASS",
        "device_execution": False,
        "variants": [],
        "method": "直接读取现有完整CPU后端产物，不重编译；分配高水位=CPP常量TASSIGN地址+物理Tile尺寸，非性能测量",
        "ffts_base": {
            "pypto_preprocess": (
                ".worktrees/pypto-csa-static-fixes-20261016/python/pypto/backend/_ptoas_preprocess.py:123"
            ),
            "behavior": (
                "set_ffts_base_addr被预处理剔除，最终由Simpler A3 kernel/kernel_mode入口设硬件FFTS基地址；"
                "参数INT64[256]不是最终独立硬件队列证明"
            ),
            "runtime_sources": [
                ".worktrees/simpler-csa-core-args-256-20261016/src/a2a3/platform/onboard/aicore/kernel.cpp:99",
                ".worktrees/simpler-csa-core-args-256-20261016/src/a2a3/platform/onboard/aicore/kernel_mode_kernel.cpp:79",
            ],
        },
        "limits": [
            "设备执行/数值/缓存可见性尚未验证",
            "正式完整CSA须分配互不冲突FFTS事件并保留所有阶段排空",
            "同一套runtime dynamic维度不等于支持任意策略；G/panel/tile/balanced是显式编译配置",
        ],
    }
    for name, (group, panel, size) in CONFIGS.items():
        base = HERE / "full_v2" / name / "kernel"
        cpp = (base / "kernels/aic/cube.cpp").read_text()
        cube, vector = body(cpp, "cube"), body(cpp, "vector")
        qc = tile(cube, "buf_query_left_")[0]
        cc = tile(cube, "buf_coefficient_pair_")[0]
        q_dims, c_dims = qc[2].split(", "), cc[2].split(", ")
        valid = q_dims[0]
        assert valid == c_dims[1], (q_dims, c_dims)
        # 实际live来自request内末组，不是query容量；valid由live乘64生成。
        assert re.search(r"int64_t " + valid + r" = [^;]*\*[^;]*;", cube)
        q_load = tile(cube, "buf_query_vector_")[0]
        c_load = tile(cube, "buf_coefficients_l1_")[0]
        assert q_load[2].split(", ")[0] == valid
        assert c_load[2].split(", ")[1] == valid
        right_tiles = tile(cube, "buf_previous_pair_") + tile(cube, "buf_last_pair_")
        right_names = set()
        for _, var, _ in right_tiles:
            right_names.add(var)
            assert var + ".SetValidShape(" + valid + ", " in cube
        matmuls = re.findall(r"TMATMUL\((v\d+), (v\d+), (v\d+)\);", cube)
        sequence = []
        for _, left, right in matmuls:
            if left == qc[1]:
                sequence.append("QK")
            elif left == cc[1]:
                assert right in right_names
                sequence.append("WS")
            else:
                raise AssertionError(left)
        panels = size // panel
        assert sequence == ["QK"] + ["QK", "WS"] * (panels - 1) + ["WS"]
        cconst, vconst = constant_values(cube), constant_values(vector)
        cwait = re.findall(r"wait_flag_dev\((v\d+)\)", cube)
        vwait = re.findall(r"wait_flag_dev\((v\d+)\)", vector)
        assert len(cwait) == 2 and all(cconst[v] == 1 for v in cwait)
        assert len(vwait) == 1 and vconst[vwait[0]] == 0
        ready = re.findall(r"getFFTSMsg\(FFTS_MODE_VAL, (v\d+)\);\s*ffts_cross_core_sync\(PIPE_FIX,", cube)
        ack = re.findall(r"getFFTSMsg\(FFTS_MODE_VAL, (v\d+)\);\s*ffts_cross_core_sync\(PIPE_MTE2,", vector)
        assert len(ready) == 1 and cconst[ready[0]] == 0
        assert len(ack) == 1 and vconst[ack[0]] == 1
        ready_pos = cube.index("ffts_cross_core_sync(PIPE_FIX")
        assert cube.rfind("TSTORE(", 0, ready_pos) >= 0
        assert len(re.findall(r"\bTSTORE\(", cube)) == panels
        ack_pos = vector.index("ffts_cross_core_sync(PIPE_MTE2")
        wait_pos = vector.index("wait_flag_dev(")
        load_pos = vector.rfind("TLOAD(", 0, ack_pos)
        assert wait_pos < load_pos < ack_pos < vector.index("TMUL(", ack_pos)
        assert cube.index("wait_flag_dev(") < cube.index("TMATMUL(")
        assert cube.rindex("wait_flag_dev(") > ready_pos
        orch = next((base / "orchestration").glob("*.cpp")).read_text()
        assert orch.count("rt_submit_task(") == 1 and "MixedKernels" in orch
        assert ".set_block_num(24)" in orch
        dims = re.findall(r"int64_t (CSA_\w+) = .*?->shapes\[\d+\]\);", cpp)
        assert len(set(dims)) == 16
        result["variants"].append(
            {
                "name": name,
                "source": str(base.relative_to(HERE)),
                "G": group,
                "panel": panel,
                "score_tile": size,
                "runtime_dimensions": sorted(set(dims)),
                "live_times_64_cpp": valid,
                "query_left": qc[1],
                "coefficient_left": cc[1],
                "QK_WS_order": sequence,
                "dynamic_WS_right_count": len(right_names),
                "AIC_store_before_FIX_READY": True,
                "AIV_load_before_MTE2_ACK": True,
                "reuse_wait_and_final_drain": True,
                "task_submissions": 1,
                "aic_address_high_water_bytes": memory(cube),
                "aiv_address_high_water_bytes": memory(vector),
            }
        )
    (HERE / "compiled_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "variants": len(result["variants"]),
                "memory": {
                    row["name"]: [row["aic_address_high_water_bytes"], row["aiv_address_high_water_bytes"]]
                    for row in result["variants"]
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
