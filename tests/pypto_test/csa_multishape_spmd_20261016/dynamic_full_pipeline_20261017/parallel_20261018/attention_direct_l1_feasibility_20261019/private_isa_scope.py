# SPDX-License-Identifier: Apache-2.0
"""仅此诊断进程更换 CCEC 的 ISA 头文件根；共享安装和托管源码保持不变。"""
import inspect
import json
from pathlib import Path


def install(output):
    import pypto.runtime.kernel_compiler as pypto_compiler
    import simpler_setup.kernel_compiler as simpler_compiler

    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    from prepare_isa_overlay import prepare
    overlay = prepare()
    assert (overlay / "include/pto/npu/a2a3/MGather.hpp").is_file()
    for module in (pypto_compiler, simpler_compiler):
        owner = module.KernelCompiler
        original = owner.compile_incore
        signature = inspect.signature(original)

        def wrapped(self, *args, _original=original, _signature=signature, _module=module.__name__, **kwargs):
            bound = _signature.bind(self, *args, **kwargs)
            prior = bound.arguments.get("pto_isa_root")
            bound.arguments["pto_isa_root"] = str(overlay)
            # 私有编译进程可能并行调用；每条短JSON独立追加，核对实际使用路径。
            with (folder / "isa_compile_calls.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(module=_module, source=bound.arguments.get("source_path"),
                                             original_isa=prior, actual_isa=str(overlay))) + "\n")
            return _original(*bound.args, **bound.kwargs)

        owner.compile_incore = wrapped
