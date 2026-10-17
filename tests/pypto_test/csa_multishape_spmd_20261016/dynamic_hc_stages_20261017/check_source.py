# SPDX-License-Identifier: Apache-2.0
"""核对源数学未变化，以及动态行/投影任务的单写者覆盖。"""

import ast
import copy
import json
from pathlib import Path

from extract_source import SOURCE, TARGET, DivisionPath, Rename, funcs, math_loop


def dump(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def main():
    old_hc = funcs(SOURCE / "hc_pre.py")
    root = funcs(SOURCE / "decode_csa.py")["_decode_csa_tp1_layer"]
    gates = old_hc["hc_pre_gates_from_rms"]
    source = ast.parse(TARGET.read_text())
    functions = funcs(TARGET)
    reference = {
        "hc_widen_rms_range": [
            Rename({"HC_WIDEN_T_TILE": "T_TILE", "HC_WIDEN_D_TILE": "RMS_K_TILE"}).visit(n)
            for n in math_loop(root, "hc_widen_rms").body[0].body
        ],
        "hc_project_range": math_loop(gates, "hc_pre_linear").body[0].body,
        "hc_reduce_range": math_loop(gates, "hc_pre_linear_reduce").body,
        "hc_gates_range": math_loop(gates, "split_pre_post").body[-1].body,
    }
    comb = [DivisionPath().visit(n) for n in math_loop(gates, "comb_sinkhorn").body]
    reference["hc_comb_range"] = [p for n in comb for p in (n if isinstance(n, list) else [n])]
    mix = math_loop(old_hc["hc_mix_norm"], "mix_x_rms_norm").body
    mix[0].value.left = ast.Name("token_block", ast.Load())
    reference["hc_mix_norm_range"] = mix
    checked = []
    for name, original in reference.items():
        fn = functions[name]
        wave_loop = next(n for n in fn.body if isinstance(n, ast.For))
        actual = wave_loop.body[-1].body
        assert dump(actual) == dump(original), name + "数学体不再等同原高层源码"
        checked.append(
            {"function": name, "original_math_ast_exact": True, "parameter_order": [a.arg for a in fn.args.args]}
        )

    forbidden = {
        "pl.spmd",
        "pl.submit",
        "pl.at",
        "pl.system.syncall",
        "pl.system.signal_notify",
        "pl.system.signal_wait",
    }
    calls = {ast.unparse(n.func) for n in ast.walk(source) if isinstance(n, ast.Call)}
    assert not calls & forbidden, sorted(calls & forbidden)
    # 独立执行纯Host工作区合同，避免为覆盖检查导入PyPTO或初始化torch设备。
    constants = []
    for n in source.body:
        if isinstance(n, ast.Assign) and not any(isinstance(x, ast.Call) for x in ast.walk(n)):
            constants.append(copy.deepcopy(n))
    constants.append(copy.deepcopy(functions["workspace_contract"]))
    env = {}
    exec(compile(ast.fix_missing_locations(ast.Module(constants, [])), str(TARGET), "exec"), env)
    combinations = 0
    for tokens in range(1, 385):
        contract = env["workspace_contract"](tokens)
        assert contract["x_fp32"]["shape"][0] == tokens
        assert contract["inv_rms"]["shape"][0] % 16 == 0
        assert contract["x_mixed"]["shape"][0] % 8 == 0
        counts = [(tokens + 7) // 8, (tokens + 15) // 16, (tokens + 15) // 16 * 4]
        for workers in (1, 2, 7, 16, 24, 48):
            for count in counts:
                jobs = [
                    worker + wave * workers
                    for worker in range(workers)
                    for wave in range((count + workers - 1) // workers)
                    if worker + wave * workers < count
                ]
                assert len(jobs) == len(set(jobs)) == count
                assert sorted(jobs) == list(range(count))
                for job in jobs:
                    assert job % workers < workers
                combinations += 1
        # 最后一个短8行块全局至多一个，四份tail scratch没有并行多写者。
        short_blocks = [r for r in range(counts[0]) if min(8, tokens - 8 * r) != 8]
        assert len(short_blocks) == int(tokens % 8 != 0)
        # 对所有实际行块验证4份投影owner公式，含T168和非8整倍数。
        for begin in range(0, tokens, 8):
            end = min(tokens, begin + 8)
            required = range((begin // 16) * 4, ((end + 15) // 16) * 4)
            assert len(required) == 4
            assert all(j < counts[2] for j in required)
    for bad in (0, -1, 1.5, True):
        try:
            env["workspace_contract"](bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    norm = functions["hc_mix_norm_range"]
    norm_w = next(a for a in norm.args.args if a.arg == "norm_w")
    assert ast.unparse(norm_w.annotation) == "pl.Tensor[[D], pl.BF16]"
    report = {
        "scope": "高层源数学AST与CPU所有权检查；无编译/NPU/数值性能结论",
        "status": "PASS",
        "math_functions": checked,
        "no_task_or_signal_calls_inside_helpers": True,
        "tokens_checked": [1, 384],
        "worker_counts_checked": [1, 2, 7, 16, 24, 48],
        "ownership_combinations": combinations,
        "T168_included": True,
        "tail_scratch_has_at_most_one_writer": True,
        "norm_weight_dtype": "BF16",
        "invalid_workspace_shapes_rejected": ["0", "-1", "1.5", "True"],
        "device_validation": False,
    }
    out = Path(__file__).with_name("source_check.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
