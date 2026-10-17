#!/usr/bin/env python3
"""只读现成两次CPU；比较实际文本/分配，不编译、不运行设备。"""
import ast
import importlib.util
import json
import pathlib
import re
import sys

E = pathlib.Path(__file__).resolve().parent
B = pathlib.Path("/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_indexer_20261018/query_dequant_group_20261018/cpu_v1")
F, T = E/"cpu_false_v1", E/"cpu_true_v1"
br, fr, tr = [json.loads((p/"result.json").read_text()) for p in (B,F,T)]
assert all(r["passed"] and r["single_mix"] and not r["sample_shapes_supplied"] for r in (br,fr,tr))
assert br["compiled_abi"] == fr["compiled_abi"] == tr["compiled_abi"]
assert len(fr["compiled_abi"]) == 130
assert fr["symbolic_dimensions"] == tr["symbolic_dimensions"] and len(fr["symbolic_dimensions"]) == 58
assert br["controls"] == fr["controls"]
assert tr["controls"] == dict(br["controls"], QA_M16_PARALLEL=True)


def no_line_comments(text):
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//"))


cpp_checks = []
for relative in ["ptoas/dynamic_csa_full.cpp", "kernels/aic/cube_schedule.cpp",
                 "kernels/aiv/vector_schedule.cpp", "orchestration/decode_csa_spmd.cpp"]:
    old, new = B/"kernel"/relative, F/"kernel"/relative
    a, b = old.read_text(), new.read_text()
    assert no_line_comments(a) == no_line_comments(b)
    assert len(a.splitlines()) == len(b.splitlines())
    changed = [(n+1,x,y) for n,(x,y) in enumerate(zip(a.splitlines(),b.splitlines())) if x != y]
    assert all(x.lstrip().startswith("//") and y.lstrip().startswith("//") for _,x,y in changed)
    cpp_checks.append(dict(relative=relative, reference=str(old), candidate_false=str(new),
                           reference_bytes=old.stat().st_size, false_bytes=new.stat().st_size,
                           lines=len(a.splitlines()), changed_comment_lines=len(changed),
                           noncomment_text_byte_equal=True))


def alloc(path):
    rows, envelope = [], {}
    for n,l in enumerate(path.read_text().splitlines(),1):
        if "pto.alloc_tile" not in l:
            continue
        loc = re.search(r'loc\("([^"]+)":(\d+):(\d+)\)',l)
        s = l.split(" = ",1)[1].split(" loc(",1)[0]
        s = re.sub(r"_inline\d+", "_inline", s)
        s = re.sub(r"%\d+\b", "%ssa", s)
        s = re.sub(r"__(iter|ssa)_v\d+", r"__\1", s)
        m = re.search(r"addr = %c(\d+)_i64.*loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", l)
        assert m
        addr,space,dtype,h,w = m.groups()
        end = int(addr)+int(h)*int(w)*int(re.search(r"\d+",dtype)[0])//8
        envelope[space] = max(envelope.get(space,0),end)
        rows.append(dict(source=pathlib.Path(loc[1]).name, source_line=int(loc[2]), pto_line=n,
                         signature=s, address=int(addr), space=space, dtype=dtype, rows=int(h), cols=int(w)))
    return rows,envelope


def keys(rows):
    return [(r["source"],r["signature"]) for r in rows]


ba,be = alloc(B/"kernel/ptoas/dynamic_csa_full.pto")
fa,fe = alloc(F/"kernel/ptoas/dynamic_csa_full.pto")
ta,te = alloc(T/"kernel/ptoas/dynamic_csa_full.pto")
assert keys(ba) == keys(fa) and be == fe == te
assert keys([r for r in ba if r["source"] != "qkv_stages.py"]) == keys([r for r in ta if r["source"] != "qkv_stages.py"])
qt = ast.parse((E/"source_candidate/qkv_stages.py").read_text())
old_helper = next(n for n in qt.body if isinstance(n,ast.FunctionDef) and n.name == "_q_a_worker")
new_helper = next(n for n in qt.body if isinstance(n,ast.FunctionDef) and n.name == "_q_a_m16_parallel_worker")
small = [r for r in ta if r["source"] == "qkv_stages.py" and new_helper.lineno <= r["source_line"] <= new_helper.end_lineno]
assert len(small) == 15
assert not any(r["source"] == "qkv_stages.py" and old_helper.lineno <= r["source_line"] <= old_helper.end_lineno for r in ta)
assert not any(r["source"] == "qkv_stages.py" and new_helper.lineno <= r["source_line"] <= new_helper.end_lineno for r in fa)
spec = importlib.util.spec_from_file_location("qa2_codegen_policy",E/"source_candidate/policy.py")
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)
selected = {}
for t in (16,17,18,24,30,31,32,48):
    p = policy.select_policy(tokens=t,requests=(t+5)//6,max_query_tokens=6,max_seq_len=131072)
    assert p.qa_m16_parallel == (16<t<32)
    selected[str(t)] = p.constexprs()

result = dict(passed=True, scope="仅两次现成CPU产物只读比较，无设备/性能结论",
    source_candidate=str(E/"source_candidate"), source_reference=str(E/"source_reference_9d"),
    baseline_cpu=str(B), false_cpu=str(F), true_cpu=str(T),
    abi_entries=130, symbolic_dimensions=58, false_controls_omit_flag=True,
    false_cpp=cpp_checks, false_actual_code_allocation_event_order_unchanged=True,
    allocation_comparison="保留alloc地址/类型/形状/静态valid；只归一source路径行号、inline和SSA编号/版本。False的C++另严格去整行注释逐字比较，不归一变量或event ID。",
    allocation_counts=dict(reference=len(ba),false=len(fa),true=len(ta)),
    true_outside_qkv_allocation_records_equal=True,
    false_excludes_m16_parallel_helper=True, true_excludes_original_q_a_worker=True,
    true_small_allocations=small, address_envelopes_bytes=be,
    address_envelope_limit="每空间max(addr+声明形状字节数)，不是相加占用或运行时live峰值。",
    automatic_policy=selected,
    limits=["True同控完整CPU验证130ABI/58动态轴，不代替设备精度、图复用或完整L1", "实际NPU按Host policy自动选择；这里无batch白名单，无样例shape"])
(E/"codegen_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print("PASS: false four C++ noncomment texts byte equal; 2370 retained allocations; true 2325 allocations, outside QA unchanged; 130 ABI/58 dynamic axes; automatic T18/T24/T30.")
