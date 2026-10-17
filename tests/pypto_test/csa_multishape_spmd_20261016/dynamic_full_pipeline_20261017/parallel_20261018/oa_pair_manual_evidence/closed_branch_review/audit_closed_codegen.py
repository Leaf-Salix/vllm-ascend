"""只读OA256既有CPU与同控设备reference；不编译，不测设备，不使用B-only数学。"""

import ast
import difflib
import importlib.util
import json
import re
from pathlib import Path


here = Path(__file__).resolve().parent
root = Path("/data/pyptouser/qinchuanyu/pto-eager")
tools_path = root / ".worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_qb_weight_reuse_design_20261018/b_resident_cpu/audit_cpu.py"
spec = importlib.util.spec_from_file_location("existing_qb_audit_helpers_readonly", tools_path)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)  # main有保护；这里只复用纯文本归一化函数，不执行它的审计写文件。
baseline = helpers.BASE
kernel = here / "cpu_oa256/kernel"
result = json.loads((here / "cpu_oa256/result.json").read_text())
matrix_path = baseline.parents[1] / "source_matrix/result.json"
matrix = json.loads(matrix_path.read_text())
assert matrix["kernel_configs"]["spmd_reference"] == str(baseline / "kernel_config.py")
controls = dict(result["controls"])
assert controls.pop("QA_M16_PARALLEL") is False
assert controls == matrix["policies"]["spmd_reference"]
assert result["passed"] and result["single_mix"] and result["sync_start"]
assert len(result["compiled_abi"]) == 130 and len(result["symbolic_dimensions"]) == 58

# source_reference是668原版。已有设备reference来自9d；QA=false下新helper不可达，
# 先证明实际可达源AST的桥接，不能仅因controls相同就宣称跨commit等价。
source_bridge = {}
for path in sorted((here.parent / "source_reference").glob("*.py")):
    if path.name == "policy.py":
        continue  # 旧Host QA规则不同，但这里已显式固定QA=false，并核对同controls。
    old = helpers.FoldQA().visit(ast.parse((helpers.OLD_SOURCE / path.name).read_text()))
    new = helpers.FoldQA().visit(ast.parse(path.read_text()))
    source_bridge[path.name] = ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
assert len(source_bridge) == 32 and all(source_bridge.values())

old_pto_path = baseline / "ptoas/dynamic_csa_full.pto"
new_pto_path = kernel / "ptoas/dynamic_csa_full.pto"
old_pto, new_pto = old_pto_path.read_text(), new_pto_path.read_text()
old_normal, new_normal = helpers.normalize_pto(old_pto), helpers.normalize_pto(new_pto)
pto_equal = old_normal == new_normal
(here / "whole_normalized_pto.diff").write_text("".join(difflib.unified_diff(
    old_normal.splitlines(True), new_normal.splitlines(True), fromfile="existing_device_reference", tofile="oa256_closed_candidate")))
assert pto_equal
old_allocs = helpers.allocs(helpers.cube_pto(old_pto_path))
new_allocs = helpers.allocs(helpers.cube_pto(new_pto_path))
fields = ("address", "space", "dtype", "rows", "cols", "size")
assert len(old_allocs) == len(new_allocs) == 553
assert all(tuple(a[f] for f in fields) == tuple(b[f] for f in fields) for a,b in zip(old_allocs,new_allocs))
maximum = {}
for v in new_allocs:
    maximum[v["space"]] = max(maximum.get(v["space"], 0), v["address"]+v["size"])


def function_body(text, name):
    start = text.index("static __aicore__ void " + name + "(")
    brace = text.index("{", start)
    level = 1
    for i in range(brace+1, len(text)):
        if text[i] == "{": level += 1
        elif text[i] == "}": level -= 1
        if level == 0: return text[start:i+1]
    raise ValueError("无法匹配真实生成函数边界")


events = {}
for role, function in [("aic","cube_schedule"),("aiv","vector_schedule")]:
    cpp_before, cpp_after = [(p / f"kernels/{role}/{function}.cpp").read_text() for p in (baseline,kernel)]
    body_before, body_after = function_body(cpp_before,function), function_body(cpp_after,function)
    pattern = r"^\s*(?:set_flag|wait_flag|pipe_barrier)\([^;]+;"
    first = [line.strip() for line in re.findall(pattern,body_before,re.M)]
    second = [line.strip() for line in re.findall(pattern,body_after,re.M)]
    equal = first == second
    (here/f"{role}_local_events.diff").write_text("".join(difflib.unified_diff(
        [s+"\n" for s in first],[s+"\n" for s in second],fromfile="device_reference",tofile="closed_cpu")))
    events[role] = {"old_count":len(first),"new_count":len(second),"same_sequence_and_event_ids":equal}
    assert equal

assert "pair_first_inline" not in new_pto
assert "wait_oa_pair_group" not in new_pto
report = {
    "scope":"一次OA256/OB128/QAfalse CPU；只读同控已有设备reference，没有新NPU",
    "cpu_passed":True,"roots":130,"dynamic_axes":58,"single_mix":True,"sync_start":True,
    "controls":result["controls"],"baseline_artifact":str(baseline),
    "baseline_role_and_controls_evidence":str(matrix_path),
    "baseline_source":str(helpers.OLD_SOURCE),"qa_false_reachable_source_bridge":source_bridge,
    "host_policy_note":"旧9d规则后来改变，但此处同controls固定QAfalse；候选对668自身policy字节不变另有source_closure.json证明",
    "whole_pto_equal_including_qb":pto_equal,
    "normalization":"仅去loc和SSA名，并把常量声明内联为相同值/类型；保留运算、控制流、类型、cache选项和所有物理地址",
    "aic_allocation_entries":553,"allocation_fields_equal":list(fields),"full_aic_upper_bounds":maximum,
    "cpp_local_events":events,"new_pair_unreachable_in_pto":True,
    "not_claimed":["未比较机器码等价：设备reference开启wait-backoff，CPU为默认wait实现",
                   "OUTPUT_GROUP_PIPELINE=false只有源级折叠证明，未另加CPU",
                   "此结果不替代T1..96打开候选的设备正确性/性能验证"],
    "helper_module_used_readonly":str(tools_path),
}
(here/"closed_codegen_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"whole_pto_equal":pto_equal,"aic_allocs":553,"events":events},ensure_ascii=False))
