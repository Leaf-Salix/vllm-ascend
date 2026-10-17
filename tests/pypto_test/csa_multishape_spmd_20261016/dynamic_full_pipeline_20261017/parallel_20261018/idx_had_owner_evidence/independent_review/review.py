#!/usr/bin/env python3
"""只读有限 owner 核对；不导入 PyPTO，不编译，不执行设备测试。"""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
SOURCE = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd")
REFERENCE = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/oa_pair_manual_evidence/source_reference")
BOUNDARIES = [0, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33,
              47, 48, 49, 95, 96, 97, 144, 192, 257, 385]


def main():
    files = sorted(p.name for p in REFERENCE.glob("*.py"))
    assert len(files) == 33
    assert all((SOURCE/name).read_bytes() == (REFERENCE/name).read_bytes() for name in files)
    results = []
    checks = 0
    for tokens in BOUNDARIES:
        units = ((tokens + 7) // 8) * 16
        active = min(units, 48)
        coverage = {unit: worker for worker in range(active) for unit in range(worker, units, active)}
        assert len(coverage) == units
        needed_union = set()
        active_cubes = min(tokens, 24)
        for cube in range(24):
            token_rows = list(range(cube, tokens, 24))
            needed = {coverage[(token // 8) * 16 + hg] for token in token_rows for hg in range(16)}
            proposed = set(range(16 * (cube // 8), 16 * (cube // 8) + 16)) if cube < tokens else set()
            assert needed == proposed, (tokens, cube, sorted(needed), sorted(proposed))
            assert proposed <= set(range(active))
            needed_union |= proposed
            checks += 1
            for token in token_rows:
                for head in (0, 3, 4, 31, 32, 63):
                    for col in (0, 63, 64, 127):
                        assert 2 * (token * 8192 + head * 128 + col) == 2 * ((64 * token + head) * 128 + col)
        assert needed_union == set(range(active))
        results.append({
            "T": tokens, "rope_units": units, "actual_rope_math_workers": active,
            "active_hadamard_workers": active_cubes, "max_hadamard_waves": (tokens + 23) // 24,
            "all_actual_owners_exactly_covered": True,
            "old_signal_checks_per_invocation": 24 * active,
            "proposed_signal_checks_per_invocation": 16 * active_cubes,
        })
    # 反例只限定证明前提：若错误地将 Hadamard stride 改成16，固定16-owner不再成立。
    wrong_stride_tokens = [0, 16, 32]
    wrong_stride_needed = {(token // 8) * 16 % 48 + hg for token in wrong_stride_tokens for hg in range(16)}
    assert wrong_stride_needed == set(range(48))
    result = {
        "status": "PASS",
        "reference": "66882634归档参考；正式33文件直接字节相同",
        "scope": "有限纯整数owner及别名地址核对，通用正确性依靠README代数证明；非CPU lowering或设备测试",
        "T_values": BOUNDARIES, "cube_cases": checks, "cases": results,
        "fixed_contracts": {"AIC": 24, "AIV": 48, "heads_per_token": 64,
                            "head_width": 128, "rope_tokens_per_unit": 8,
                            "heads_per_rope_unit": 4},
        "counterexample_outside_contract": {"hadamard_stride": 16, "T": 33, "cube": 0,
                                           "tokens": wrong_stride_tokens,
                                           "needed_rope_workers": list(range(48)),
                                           "fixed16_is_wrong": True},
        "signal_count_limit": "只统计源级调用次数，不代表执行时间或整层收益",
    }
    (HERE/"review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status":"PASS", "formal_files_equal_reference":33,
                      "T_boundaries":len(BOUNDARIES), "cube_cases":checks,
                      "runtime_or_codegen_test":False},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

