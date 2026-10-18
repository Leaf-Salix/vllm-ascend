#!/usr/bin/env python3
"""只读有限pass AST统计；不导入PyPTO，不读取result/log，不复制pass全文。"""
import ast
import collections
import json
from pathlib import Path

F = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_interleave_candidate/source_candidate").parent
OUT = Path(__file__).resolve().parent
KEYS = (
    "indexer_transfer", "score_ring", "pv_ring", "kv_ring", "probability_ring",
    "mi_ring", "li_ring", "packed", "pair_arena", "topk_scores", "topk_indices",
    "attention_indices", "attention_bias", "attention_mask",
)
OPS = {
    "pl.tile.store", "pl.tensor.write", "pl.tensor.assemble", "pl.tile.assemble",
    "pl.system.signal_wait", "pl.system.signal_notify", "pl.system.sync_wait",
    "pl.system.sync_set", "pl.system.fence", "pl.system.cacheinvalid", "pl.system.bar_all",
}
FILES = [
    "cpu_long_v2/kernel/passes_dump/01_after_InlineFunctions.py",
    "cpu_long_v3/kernel/passes_dump/01_after_InlineFunctions.py",
    "cpu_long_v3/kernel/passes_dump/05_after_Simplify.py",
    "cpu_long_v3/kernel/passes_dump/29_after_NormalizeReturnOrder.py",
]
records = []
for rel in FILES:
    path = F / rel
    tree = ast.parse(path.read_text())
    totals = collections.Counter()
    critical = collections.Counter()
    forms = collections.Counter()
    examples = {}
    assemble_targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            op = ast.unparse(node.func)
            if op not in OPS:
                continue
            totals[op] += 1
            arg = 2 if op == "pl.tile.store" else 0
            if len(node.args) > arg:
                dst = ast.unparse(node.args[arg])
                if op == "pl.tensor.assemble":
                    assemble_targets.add(dst)
                key = next((k for k in KEYS if dst == k or dst.startswith(k + "_")), None)
                if key:
                    critical[op + ":" + key] += 1
                    examples.setdefault(op + ":" + key, {
                        "line": node.lineno,
                        "operation": op,
                        "target_argument": dst,
                    })
        if isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, ast.Call):
                op = ast.unparse(value.func)
                if op in OPS:
                    forms[op + ":" + type(node).__name__] += 1
    assert not any(name.startswith("pl.tensor.assemble:") for name in critical)
    records.append({
        "path": str(path), "bytes": path.stat().st_size,
        "operation_totals": dict(sorted(totals.items())),
        "critical_targets": dict(sorted(critical.items())),
        "statement_forms": dict(sorted(forms.items())),
        "critical_examples": dict(sorted(examples.items())),
        "tensor_assemble_targets": sorted(assemble_targets),
    })
assert records[0]["operation_totals"] == records[1]["operation_totals"]
assert records[0]["critical_targets"] == records[1]["critical_targets"]
assert records[2]["critical_targets"] == records[3]["critical_targets"]
for op in ("pl.system.signal_wait", "pl.system.signal_notify", "pl.system.sync_wait",
           "pl.system.sync_set", "pl.system.fence", "pl.system.cacheinvalid", "pl.system.bar_all"):
    assert records[2]["operation_totals"][op] == records[3]["operation_totals"][op]
    for record in records:
        assert record["statement_forms"].get(op + ":Expr", 0) == record["operation_totals"][op]
result = {
    "passed": True,
    "scope": "v2/v3现成IR的有限语句/目标分类；非执行次数、非性能、非最终C++证明",
    "v2_v3_inline_operation_totals_equal": True,
    "v2_v3_inline_critical_target_counts_equal": True,
    "v3_simplify_to_normalize_return_critical_targets_equal": True,
    "v3_simplify_to_normalize_return_sync_counts_equal": True,
    "synchronization_always_standalone_eval": True,
    "critical_outputs_do_not_use_tensor_assemble": True,
    "records": records,
    "limits": [
        "静态出现次数包含runtime分支和fallback，不能用数量代替逐路径数据/地址正确性。",
        "Simplify前后会因unroll/constexpr折叠改变计数，因此没有跨该边界强行要求计数相同。",
        "v3当前在AllocateMemoryAddr报Left容量超限，没有最终设备binary，不能宣称最终流水正确。",
    ],
}
(OUT / "pass_evidence.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("records","limits")},ensure_ascii=False))

