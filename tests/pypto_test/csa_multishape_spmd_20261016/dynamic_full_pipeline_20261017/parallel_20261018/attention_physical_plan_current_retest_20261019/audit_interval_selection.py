# SPDX-License-Identifier: Apache-2.0
"""核对当前正式源、区间constexpr及已测结果的对应；不运行设备或估算时间。"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip())
FORMAL = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


def policy_at(folder, name):
    spec = importlib.util.spec_from_file_location(name, folder / "policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = policy_at(HERE / "source_reference", "physical_plan_reference_policy")
new = policy_at(FORMAL, "physical_plan_final_policy")
same = all(p.read_bytes() == (FORMAL / p.name).read_bytes()
           for p in (HERE / "source_interval_final").glob("*.py"))
assert same
changed = sorted(p.name for p in (HERE / "source_reference").glob("*.py")
                 if p.read_bytes() != (FORMAL / p.name).read_bytes())
assert changed == ["attention_plan_stages.py", "attention_stream_stages.py", "entry.py", "policy.py", "vector_schedule.py"]
checks = 0
for history in (8192, 131072):
    for tokens in range(1, 513):
        for maximum in (1, 5, 6):
            args = dict(tokens=tokens, requests=(tokens + maximum - 1) // maximum,
                        max_query_tokens=maximum, max_seq_len=history)
            before, after = old.select_policy(**args).constexprs(), new.select_policy(**args).constexprs()
            enabled = after.pop("PLAN_PHYSICAL_ROWS")
            assert enabled == (tokens >= old.EARLY_GEOMETRY_MIN_TOKENS)
            assert before == after
            checks += 1
selected = []
for result in json.loads((HERE / "COMPLETED_RESULTS.json").read_text()):
    batch, history = result["batch"], result["history"]
    chosen = new.select_policy(tokens=batch * 6, requests=batch, max_query_tokens=6,
                               max_seq_len=history).constexprs()
    flag = chosen.pop("PLAN_PHYSICAL_ROWS")
    assert chosen == result["policies"]["spmd_candidate"] == result["policies"]["spmd_reference"]
    role = "spmd_candidate" if flag else "spmd_reference"
    stats = result["statistics_us"]
    baseline, latency = stats["dynamic_pto"]["median_us"], stats[role]["median_us"]
    selected.append(dict(batch=batch, history=history, physical_plan=flag, measured_role=role,
                         selected_us=latency, multitask_us=baseline, task=result["task"],
                         reduction_vs_multitask_pct=(baseline - latency) / baseline * 100,
                         goal_30pct_met=latency <= baseline * .7))
compiler = (HERE.parent / "score_key_l1_region_sync_20261019/toolchain_v3/bin/ptoas").resolve()
compiler_checks = []
for registry in ("DEVICE_SUBMISSIONS.json", "QUERY_SUBMISSIONS.json", "BOUNDARY_SUBMISSIONS.json"):
    for entry in json.loads((HERE / registry).read_text()):
        if entry.get("terminal_status") != "completed" or entry.get("exit_code") != 0:
            continue
        log = Path(entry["output"]) / "COMPILER_INVOCATIONS.jsonl"
        if not log.exists():
            raise AssertionError(f"missing compiler record: {log}")
        invocations = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        assert invocations and all(Path(row["compiler"]).resolve() == compiler for row in invocations)
        compiler_checks.append(dict(task=entry["task"], records=len(invocations), all_selected_v3=True))
audit = dict(formal_equal_final_snapshot=same, changed_files=changed, policy_checks=checks,
             exact_single_mix_codegen="ON_FINAL_CODEGEN.json / OFF_INTERVAL_V2_CODEGEN.json",
             source_interval=old.EARLY_GEOMETRY_MIN_TOKENS, selected_results=selected,
             compiler_checks=compiler_checks, goal_30pct_met=sum(x["goal_30pct_met"] for x in selected),
             note="七档数值取已完成的同轮候选/参考；区间接线另做开关代码等价与边界，不冒称重跑最终七档。")
(HERE / "ADOPTION_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in audit.items() if k not in ("compiler_checks", "selected_results")}, ensure_ascii=False))
