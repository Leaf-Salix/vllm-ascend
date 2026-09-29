"""代表档和边界通过后冻结同一包、Native显式后端及PTO入口。"""

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
REPO = ROOT.parents[3]
BASE = WORKSPACE / ".cache/csa-sparse-final-publish-fix-4ffccb7b-candidate"
DEST = WORKSPACE / ".cache/csa-sparse-publish-seven-20260929"
OLD_PACKAGE = "dsv4_csa_sparse_final_publish_4ffccb7b"
PACKAGE = "dsv4_csa_sparse_publish_seven_20260929"


def main():
    pair = ROOT.parent / "csa_sparse_final_publish_fixed_pair_20260929"
    if json.loads((pair / "summary.json").read_text())["state_status"] != "PASS":
        raise RuntimeError("完整两档状态尚未通过")
    if json.loads((pair / "boundary/summary.json").read_text())["status"] != "PASS":
        raise RuntimeError("受影响边界尚未通过")
    relative = Path("vllm_ascend/ops/pypto")
    filename = "decode_sparse_attn_csa.py"
    measured_source = (BASE / relative / OLD_PACKAGE / filename).read_text()
    production_source = (REPO / relative / "deepseek_v4_flash_dspark_perf" / filename).read_text()
    if ast.dump(ast.parse(measured_source)) != ast.dump(ast.parse(production_source)):
        raise RuntimeError("当前生产Sparse尚未采用通过验证的候选，先完成独立提交")
    if (ROOT / "task.txt").exists():
        raise RuntimeError("禁止编辑已入队矩阵")
    shutil.copytree(BASE, DEST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (DEST / relative / OLD_PACKAGE).rename(DEST / relative / PACKAGE)
    (DEST / relative / PACKAGE / filename).write_text(production_source)
    runner = DEST / "tests/pypto_test/coefficients_seven_experiment"
    native = WORKSPACE / ".cache/csa-native-graph-replay-4ffccb7b" / runner.relative_to(DEST) / "compiled_case.py"
    native_text = native.read_text()
    # The calibrated Native helper replays the backend-owned graph directly.
    # Copy its actual implementation, not the older outer-capture helper used
    # by the PTO runner in BASE.
    helper_path = native.parents[1] / "dsv4_csa_single_layer.py"
    helper_text = helper_path.read_text()
    functions = [node for node in ast.parse(helper_text).body
                 if isinstance(node, ast.FunctionDef) and node.name == "measure_graph_interval"]
    if len(functions) != 1:
        raise ValueError("Native校准辅助函数不唯一")
    function = ast.get_source_segment(helper_text, functions[0])
    if "torch.npu.graph(" in function or "graph.replay()" in function:
        raise ValueError("不能把旧手工外层capture带回Native")
    (runner / "native_measure.py").write_text(
        '"""复用已校准的后端图直接重放；不另捕获外图。"""\n'
        "import math\nimport statistics\n\n"
        "from dsv4_csa_single_layer import collect_state, guard_checks\n"
        "from dsv4_csa_validation import compare_tensor, compare_topk\n\n\n" + function + "\n")
    native_text = native_text.replace("    measure_graph_interval,\n", "")
    native_text = native_text.replace(
        "from npugraph_ex._acl_concrete_graph import static_kernel\n\n",
        "from npugraph_ex._acl_concrete_graph import static_kernel\n")
    native_text = native_text.replace("from offline_pd.run import decode_additional_config",
                                      "from native_measure import measure_graph_interval  # noqa: E402\n"
                                      "from offline_pd.run import decode_additional_config")
    native_text = native_text.replace('    parser.add_argument("--save-state", action="store_true")',
                                      '    parser.add_argument("--save-state", action="store_true")\n'
                                      '    parser.add_argument("--profile-only", action="store_true")')
    native_text = native_text.replace('        "super_kernel": bool(args.super_kernel),',
                                      '        "super_kernel": bool(args.super_kernel),\n'
                                      '        "diagnostic_only": args.profile_only,')
    native_text = native_text.replace('iters=20, warmup=5, require_exact=',
                                      'iters=1 if args.profile_only else 20,\n'
                                      '                warmup=1 if args.profile_only else 5, require_exact=')
    native_text = native_text.replace('require_exact=runtime is not None, profile_dir=',
                                      'require_exact=runtime is not None,\n                profile_dir=')
    native_text = native_text.replace('report["mean_us"] = statistics.mean(report["timing"]["samples_us"])',
                                      'report["mean_us"] = (None if args.profile_only else\n'
                                      '                                 statistics.mean('
                                      'report["timing"]["samples_us"]))')
    native_text = native_text.replace('report["status"] = "MEASURED"',
                                      'report["status"] = "PROFILED" if args.profile_only else "MEASURED"')
    compile(native_text, str(runner / "native_case.py"), "exec")
    (runner / "native_case.py").write_text(native_text)
    commit = subprocess.check_output(["git", "--work-tree=.", "-c", "core.bare=false",
                                      "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    source = {"source": str(DEST), "base_source": str(BASE), "operator_commit": commit,
              "variant": "pkg:" + PACKAGE, "native_runner_source": str(native),
              "native_measure_source": str(helper_path),
              "cann": "9.2.0-beta.2", "native_super_kernel": True,
              "native_incore": {"super_kernel": False, "static_compile": True, "profile_only": True},
              "cases": [[131072, 4], [131072, 8], [131072, 16], [131072, 24],
                        [8192, 16], [8192, 24], [8192, 32]],
              "scope": "阶段七档出口；新Native自管图设备重放，不拼接旧基线或整模型数据"}
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    code = '''
import importlib,json,os,sys
sys.path.insert(0,sys.argv[1]+"/tests/pypto_test")
os.environ.update(PTO_CSA_VARIANT=sys.argv[2],VLLM_ASCEND_ENABLE_NZ="2",VLLM_ASCEND_PTO_CSA_ATOMIC_ADD="0")
from dsv4_csa_env import activate
activate()
from vllm_ascend.ops.pypto.variant import variant_package
module=importlib.import_module(variant_package()+".decode_csa")
print(json.dumps({"source":sys.argv[1],"variant":sys.argv[2],"graphs":{
 n:type(getattr(module,n)._get_dep_graph()).__name__
 for n in ("decode_csa_tp1_layer","decode_csa_tp1_layer_test")},"status":"PARSE_PASS"}))
'''
    result = subprocess.check_output([sys.executable, "-c", code, str(DEST), source["variant"]], text=True)
    (ROOT / "parse.log").write_text(result)
    records = [json.loads(line) for line in result.splitlines() if line.startswith('{"source":')]
    if len(records) != 1 or records[0]["status"] != "PARSE_PASS":
        raise RuntimeError("依赖图解析结果不完整")
    (ROOT / "parse.json").write_text(json.dumps(records[0], ensure_ascii=False, indent=2) + "\n")
    print(source)


if __name__ == "__main__":
    main()
