#!/usr/bin/env python3
"""只读既有CPU产物；不启动PyPTO后端/NPU，不改原件。"""
import ast
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
PAR=Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018")
REF=PAR/"oa_pair_manual_h131072_b16_v1/build_output/decode_csa_spmd_jp7o4g2r"
CAND=PAR/"attention_scalar_geometry_evidence/cpu_v1/kernel"
# 只取既有审计的纯文本解析函数；不导入模块、不执行main、不生成其__pycache__。
helper=PAR/"idx_had_owner_evidence/generated_review/audit_generated.py"
wanted={"pto_function","normalize_pto","allocations","cpp_function"}
module=ast.parse(helper.read_text())
exec(compile(ast.Module(body=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in wanted],type_ignores=[]),str(helper),"exec"))

def normcpp(t):
    t=re.sub(r"//[^\n]*","",t).replace("CSA_WAIT_LOCAL_GM(","pto::comm::TWAIT(")
    names={}
    def rename(m):
        names.setdefault(m.group(),"v"+str(len(names)))
        return names[m.group()]
    return re.sub(r"\s+"," ",re.sub(r"\bv\d+\b",rename,t)).strip()

result=json.loads((CAND.parent/"result.json").read_text())
matrix_path=REF.parents[1]/"source_matrix/result.json"
matrix=json.loads(matrix_path.read_text())
assert result["passed"] and result["full_codegen_passed"] and result["single_mix"] and result["sync_start"]
assert not result["sample_shapes_supplied"] and not result["device_execution"]
assert len(result["compiled_abi"])==130 and len(result["symbolic_dimensions"])==58
assert result["controls"]==matrix["policies"]["spmd_reference"]
src=PAR/"attention_scalar_geometry_evidence/source_reference"
fs=list(src.glob("*.py"))
assert len(fs)==33 and all(f.read_bytes()==(Path(matrix["reference_source"])/f.name).read_bytes() for f in fs)
pto=[(p/"ptoas/dynamic_csa_full.pto").read_text() for p in (REF,CAND)]
roles={}
for role,name in [("aic","cube_schedule"),("aiv","vector_schedule")]:
    bodies=[pto_function(t,name) for t in pto]
    alloc=[allocations(t) for t in bodies]
    fields=("address","space","dtype","rows","cols","size")
    assert [[tuple(x[k] for k in fields) for x in a] for a in alloc][0]==[[tuple(x[k] for k in fields) for x in a] for a in alloc][1]
    cpp=[cpp_function((p/f"kernels/{role}/{name}.cpp").read_text(),name) for p in (REF,CAND)]
    ev=[[x.strip() for x in re.findall(r"^\s*(?:set_flag|wait_flag|pipe_barrier)\([^;]+;",t,re.M)] for t in cpp]
    assert ev[0]==ev[1]
    roles[role]={"allocations":len(alloc[1]),"same_addresses_types_shapes":True,
        "upper_bytes":{sp:max(x["address"]+x["size"] for x in alloc[1] if x["space"]==sp) for sp in sorted({x["space"] for x in alloc[1]})},
        "local_events":len(ev[1]),"identical_flat_local_events":True,
        "whole_function_normalized_pto_equal":normalize_pto(bodies[0])==normalize_pto(bodies[1]),
        "whole_function_normalized_cpp_equal":normcpp(cpp[0])==normcpp(cpp[1])}
    if role=="aic":
        assert roles[role]["whole_function_normalized_pto_equal"] and roles[role]["whole_function_normalized_cpp_equal"]
cpp_path=CAND/"kernels/aiv/vector_schedule.cpp"
cpp=cpp_path.read_text()
for line in ["v10874 = v371;","v10875 = v371;","v10876 = v357;","v10877 = v393;",
             "v9066 = v9057 != v10874;","v10910 = v10874;","v10911 = v10875;","v10912 = v10876;","v10913 = v10877;",
             "v9265 = v9062 ? v9057 : v10874;","v10874 = v10910;","v10875 = v10911;","v10876 = v10912;","v10877 = v10913;"]:
    assert line in cpp,line
for name,ty in [("v10874","int64_t"),("v10875","int64_t"),("v10876","int64_t"),("v10877","int32_t")]:
    assert f"{ty} {name};" in cpp
    assert len(re.findall(r"\b"+name+r"\s*=",cpp))==2
for name,value in [("v371","-1"),("v357","0"),("v393","0")]:
    assert re.search(r"const \w+ "+name+r" = "+value+";",cpp)
snippets=[]
for path,spans in [(CAND/"ptoas/dynamic_csa_full.pto",[(12632,12655),(12753,12764),(13005,13018),(13427,13440)]),
                   (cpp_path,[(57563,57604),(57629,57647),(57895,57918),(58437,58456),(59962,60016)])]:
    ls=path.read_text().splitlines()
    for lo,hi in spans:
        snippets.append(f"\n{path}\n{lo}–{hi}\n")
        snippets.extend(str(i)+": "+ls[i-1]+"\n" for i in range(lo,hi+1))
(HERE/"scalar_guard_excerpt.txt").write_text("".join(snippets))
report={"status":"READY_FOR_PARENT_DEVICE_VALIDATION","scope":"既有CPU产物只读，无后端/NPU",
    "roots":130,"dynamic_axes":58,"single_mix":True,"sync_start":True,"controls_equal":True,
    "reference_source_33_byte_equal":True,"controls":result["controls"],"roles":roles,
    "loop_values":"源码原4数学Tile+4新Scalar；物理PTO原数学Tile沿原分配，新增4Scalar iter_args",
    "new_scalar_types":["index","index","index","i32"],"initial_values":[-1,-1,0,0],
    "guards":["tick<block_count","mask>0","current_query!=cached_tag"],
    "masked_and_drain_passthrough":True,"inactive_updates_tag":True,"PV_reset_does_not_write_cache":True,
    "cpp_cache_vars":["v10874","v10875","v10876","v10877"],"cache_var_assignments_each":2,
    "limitations":["AIC全文指整数学函数，归一SSA/去注释并将已有wait-backoff调用还原为TWAIT；不是整个文件/机器码逐字相同。",
                   "AIV全文因cache控制不同；数学/地址保持依据源级逆AST、实际同分配/局部events与guard/phi审阅，不称全文等价。",
                   "事件平列表不能独立证明分支路径；新增tag分支只包原Scalar几何，原mask和事件谓词保留。",
                   "CPU不证明CCEC寄存器/栈spill、设备精度或完整性能。"]}
(HERE/"review.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
originals=[CAND.parent/"result.json",matrix_path,helper]+[p/f for p in (REF,CAND) for f in ["ptoas/dynamic_csa_full.pto","kernels/aic/cube_schedule.cpp","kernels/aiv/vector_schedule.cpp","kernel_config.py"]]
(HERE/"INPUT_ARTIFACTS.json").write_text(json.dumps([{"path":str(p),"bytes":p.stat().st_size,"copied":False} for p in originals],ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"status":report["status"],"roles":roles},ensure_ascii=False,indent=2))

