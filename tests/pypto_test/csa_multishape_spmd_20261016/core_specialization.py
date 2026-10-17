# SPDX-License-Identifier: Apache-2.0
"""诊断：每个物理核身份在 C++ 模板中固化，允许控制条件常量折叠。

必须结合已逐项核对的 compiled_control。仍只有一个运行时 MIX SPMD；
模板仅特化 group/lane/24 组核，业务长度、位置、页表仍从 GM 动态读取。
不修改数学源、等待偏序、缓存维护或共享组件；此编译后诊断不是生产接口。
"""

import contextlib
import json
import re
from pathlib import Path


def specialize_cpp(source, leaf_count):
    matches = list(re.finditer(r"^AICORE void csa_static_(aic|aiv)\(([^\n]+)\) \{", source, re.M))
    if {m.group(1) for m in matches} != {"aic", "aiv"} or len(matches) != 2:
        raise ValueError("无法唯一定位两类物理函数")
    audit = []
    for match in reversed(matches):
        role = match.group(1)
        end = source.find("\n}", match.end())
        if end < 0:
            raise ValueError("无法定位物理函数末尾")
        declarations = [x.strip() for x in match.group(2).split(",")]
        extra = 2 if role == "aic" else 3
        if len(declarations) != leaf_count + extra:
            raise ValueError("物理核编号 ABI 与预期不符")
        names = [x.split()[-1].lstrip("*") for x in declarations]
        body = source[match.end():end]
        constants = []
        for declaration, value in zip(declarations[leaf_count:], ("CSA_GROUP", "24", "CSA_LANE")):
            if "__gm__" in declaration:
                raise ValueError("核编号应为标量，不能特化业务 Tensor")
            constants.append("  constexpr " + declaration + " = " + value + ";")
        helper = ("template<int CSA_GROUP, int CSA_LANE = 0>\n"
                  f"AICORE void csa_core_{role}({', '.join(declarations[:leaf_count])}) {{\n"
                  + "\n".join(constants) + body + "\n}\n")
        outer = [f"AICORE void csa_static_{role}({match.group(2)}) {{"]
        group, lane = names[leaf_count], names[-1]
        selector = group if role == "aic" else f"({group} * 2 + {lane})"
        outer.append(f"  switch ({selector}) {{")
        for identity in range(24 if role == "aic" else 48):
            g, v = (identity, 0) if role == "aic" else divmod(identity, 2)
            outer.append(f"    case {identity}: csa_core_{role}<{g}, {v}>({', '.join(names[:leaf_count])}); break;")
        outer += ["  }", "}"]
        source = source[:match.start()] + helper + "\n" + "\n".join(outer) + source[end + 2:]
        audit.append({"role": role, "specializations": 24 if role == "aic" else 48,
                      "math_and_signal_body_unchanged": True,
                      "device_business_metadata_not_specialized": True})
    return source, audit


@contextlib.contextmanager
def specialize_core_identity(leaf_count, output):
    from pypto.backend import pto_backend

    original = pto_backend._compile_pto_module
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def compile_specialized(pto, name, output_dir, *args, **kwargs):
        source = original(pto, name, output_dir, *args, **kwargs)
        if name != "csa_static_group":
            return source
        transformed, audit = specialize_cpp(source, leaf_count)
        (output / "original.cpp").write_text(source)
        (output / "specialized.cpp").write_text(transformed)
        (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        return transformed

    pto_backend._compile_pto_module = compile_specialized
    try:
        yield
    finally:
        pto_backend._compile_pto_module = original


@contextlib.contextmanager
def reuse_identical_core_binaries(reference, output):
    """仅源码逐字相同且已有编译上下文匹配时复用 CPU 预编译，避免占卡重编。

    保留正常运行时的注册、调度及二进制解析，不覆盖缓存保护。没有额外 hash。
    描述符引起的源代码差异会回退正常编译，并在记录中明确标注。
    """
    from pypto.runtime.kernel_compiler import KernelCompiler
    from pypto.runtime.device_runner import _current_binary_context

    reference, output = Path(reference), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    context = json.loads((reference / "cache/binary_context.json").read_text())
    # 当前缓存文档外层同时包含 context 和产物清单；只比较编译上下文。
    context = context.get("context", context)
    original = KernelCompiler.compile_incore
    audit = []

    def compile_or_reuse(compiler, source_path, core_type="aiv", pto_isa_root=None,
                         runtime_name=None, extra_include_dirs=None, build_dir=None):
        source = Path(source_path)
        before = reference / "kernels" / core_type / source.name
        binary = before.with_suffix(".o")
        same_source = before.is_file() and source.read_bytes() == before.read_bytes()
        current = _current_binary_context(compiler, platform=compiler.platform,
                                         runtime_name=runtime_name, pto_isa_root=pto_isa_root)
        same_context = current is not None and current.to_dict() == context
        reused = same_source and same_context and binary.is_file()
        audit.append({"source": str(source), "reference": str(before),
                      "same_source_bytes": same_source, "same_binary_context": same_context,
                      "reused": reused})
        if reused:
            return binary.read_bytes()
        return original(compiler, source_path, core_type=core_type, pto_isa_root=pto_isa_root,
                        runtime_name=runtime_name, extra_include_dirs=extra_include_dirs, build_dir=build_dir)

    KernelCompiler.compile_incore = compile_or_reuse
    try:
        yield
    finally:
        KernelCompiler.compile_incore = original
        (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
