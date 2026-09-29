"""按新口径冻结七档：dynamic=False、inplace_pass=True，算子不改。"""

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
PREVIOUS = ROOT.parent / "csa_sparse_publish_seven_20260929"
BASE = WORKSPACE / ".cache/csa-sparse-publish-seven-20260929"
DEST = WORKSPACE / ".cache/csa-native-inplace-seven-20260929"
OLD_PACKAGE = "dsv4_csa_sparse_publish_seven_20260929"
PACKAGE = "dsv4_csa_native_inplace_seven_20260929"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"冻结入口发生改变：{old}")
    return text.replace(old, new)


def main():
    if (ROOT / "task.txt").exists() or DEST.exists():
        raise RuntimeError("不能覆盖已冻结/入队的源码")
    shutil.copytree(BASE, DEST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    package_root = DEST / "vllm_ascend/ops/pypto"
    (package_root / OLD_PACKAGE).rename(package_root / PACKAGE)
    runner = DEST / "tests/pypto_test/coefficients_seven_experiment"
    native_path = runner / "native_case.py"
    native = replace_once(native_path.read_text(), '"inplace_pass": False', '"inplace_pass": True')
    native = replace_once(native, 'report["backend_options"] = options',
                          'report["backend_options"] = options\n'
                          '            report["torch_compile_options"] = {"dynamic": False, "fullgraph": True}')
    native_path.write_text(native)
    # PTO retains its production custom-op boundary. Enable the same pass in
    # this private wrapper too, without modifying shared Native/vLLM code.
    compiler_path = DEST / "vllm_ascend/compilation/compiler_interface.py"
    compiler = replace_once(compiler_path.read_text(), '"inplace_pass": False', '"inplace_pass": True')
    compiler = replace_once(compiler,
                            '# inplace_pass=False: disable reinplace pass to avoid gelu fallback to CPU.',
                            '# Single-CSA comparison: use the requested reinplace pass (no MoE/gelu).')
    compiler_path.write_text(compiler)
    pto_path = runner / "compiled_case.py"
    instrumentation = '''            from vllm_ascend.compilation import compiler_interface

            original_configure = compiler_interface._configure_backend

            def observe_backend(*inputs, **kwargs):
                process = kwargs.get("process_kwargs_options")
                if process is not None:
                    def record_options(backend_config, values):
                        report["backend_options"] = dict(values["options"])
                        if not values["options"]["inplace_pass"]:
                            raise RuntimeError("Requested inplace_pass=True did not reach the backend")
                        return process(backend_config, values)
                    kwargs["process_kwargs_options"] = record_options
                return original_configure(*inputs, **kwargs)

            compiler_interface._configure_backend = observe_backend
'''
    pto = replace_once(pto_path.read_text(), '            static_kernel.static_compile = observe_static\n',
                       '            static_kernel.static_compile = observe_static\n' + instrumentation)
    pto_path.write_text(pto)
    for path in (native_path, compiler_path, pto_path):
        ast.parse(path.read_text())
    source = json.loads((PREVIOUS / "source.json").read_text())
    source.update(source=str(DEST), base_source=str(BASE), variant="pkg:" + PACKAGE,
                  native_runner_source=str(BASE / native_path.relative_to(DEST)),
                  native_dynamic=False, native_fullgraph=True, inplace_pass=True,
                  previous_task=(PREVIOUS / "task.txt").read_text().strip(),
                  scope="按用户新口径完整重取七档；不复用inplace关闭的正式样本")
    source["native_incore"].update(dynamic=False, inplace_pass=True)
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
    output = subprocess.check_output([sys.executable, "-c", code, str(DEST), source["variant"]], text=True)
    (ROOT / "parse.log").write_text(output)
    rows = [json.loads(line) for line in output.splitlines() if line.startswith('{"source":')]
    if len(rows) != 1 or rows[0]["status"] != "PARSE_PASS":
        raise RuntimeError("私有包依赖图解析失败")
    (ROOT / "parse.json").write_text(json.dumps(rows[0], ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(source, ensure_ascii=False))


if __name__ == "__main__":
    main()
