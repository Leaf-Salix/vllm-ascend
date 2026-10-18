# SPDX-License-Identifier: Apache-2.0
"""历史静态CSA的独立调用入口；算子、参数和调度合同均在本包内。

先用prepare_static_csa绑定Native根Tensor及独占工作区，再调用返回对象。
首次调用在既有PyPTO kernel环境中编译完整MIX；后续直接复用同一入口和工作区。
所有shape/stride/NZ格式检查发生在准备阶段，positions/lengths/page tables仍在
device。只支持manifest中明确归档的规格，不用邻档猜测布局。

首次编译为该静态MIX安装历史cached100等待源码回调，并核对真正交给CCEC的
AIC/AIV源码。回调仅在编译期间生效并在finally恢复；应在单线程warmup阶段使用。
这份参考接口不做测量、精度比较或读回设备值，测试逻辑位于tests/pypto_test。
"""

import importlib
import json
from pathlib import Path

from .prepared_arguments import PreparedStaticArguments
from .wait_codegen import rewrite_wait_source


REFERENCE_ROOT = Path(__file__).resolve().parent


def select_reference(*, history: int, batch: int, query_tokens: int = 6) -> dict:
    """按存档标识查找完整静态实现；这是历史枚举，不是正式动态分档策略。"""
    if any(type(value) is not int or value <= 0 for value in (history, batch, query_tokens)):
        raise ValueError("history/batch/query_tokens必须为正整数")
    manifest = json.loads((REFERENCE_ROOT / "reference_manifest.json").read_text())
    matches = [row for row in manifest["entries"]
               if (row["history"], row["batch"], row["query_tokens"]) == (history, batch, query_tokens)]
    if len(matches) != 1:
        raise ValueError(f"没有已验收的静态参考：H{history}/B{batch}/S{query_tokens}")
    return matches[0]


class PreparedStaticCSA(PreparedStaticArguments):
    """一次准备、重复调用；不同并发使用者分别创建实例，不能共享进度区。"""

    def __init__(self, native_args, *, history, batch, query_tokens=6, workspace_guard_bytes=0):
        self.reference = select_reference(history=history, batch=batch, query_tokens=query_tokens)
        case = f"h{history}_b{batch}"
        self.case_directory = REFERENCE_ROOT / "variants" / case
        super().__init__(self.case_directory, native_args, workspace_guard_bytes=workspace_guard_bytes)
        self.require_core_capacity(256)
        module = importlib.import_module(f"{__package__}.variants.{case}.entry_jit")
        self.entry = module.entry
        self.layout = json.loads((self.case_directory / "layout.json").read_text())
        self.artifact = None
        self.compile_options = None
        self.compile_report = None

    def compile(self, *, config=None):
        """完整编译并验证cached100；不读取tests或历史编译产物。"""
        from pypto import passes
        from pypto.backend import pto_backend
        from pypto.runtime import CompileOptions

        if self.artifact is not None:
            if config is not None and config is not self.compile_options:
                raise ValueError("同一静态实例的CompileOptions不能在重放中途更换")
            return self.artifact
        if config is None:
            config = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        original = pto_backend._compile_pto_module
        changes = []

        def compile_with_wait(pto_code, unit_name, output_dir, *args, **kwargs):
            source = original(pto_code, unit_name, output_dir, *args, **kwargs)
            if unit_name != "csa_static_group":
                return source
            transformed, change = rewrite_wait_source(source)
            changes.append(change)
            return transformed

        pto_backend._compile_pto_module = compile_with_wait
        try:
            artifact = self.entry._resolve_kernel_artifact(tuple(self.core_args), {"config": config})
        finally:
            pto_backend._compile_pto_module = original
        directory = Path(artifact.output_dir)
        kernels = {}
        for role in ("aic", "aiv"):
            path = directory / "kernels" / role / f"csa_static_{role}.cpp"
            source = path.read_text()
            calls = source.count("CSA_WAIT_LOCAL_GM(") - 1
            if (calls <= 0 or "pto::comm::TWAIT(" in source
                    or "dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);" not in source
                    or "if (*ptr >= expected) break;" not in source
                    or "get_sys_cnt() - csa_wait_start) < 100)" not in source):
                raise RuntimeError(f"{role}实际生成源码不满足历史cached100等待合同")
            kernels[role] = dict(source=str(path), wait_sites=calls)
        self.artifact = artifact
        self.compile_options = config
        self.compile_report = dict(artifact=str(directory), kernels=kernels, cached100_verified=True,
                                   compilation_callbacks=len(changes), kernel_config=str(directory / "kernel_config.py"))
        return artifact

    def __call__(self, *, config=None):
        self.compile(config=config)
        return self.entry(*self.core_args, config=self.compile_options)


def prepare_static_csa(native_args, *, history, batch, query_tokens=6, workspace_guard_bytes=0):
    """公开准备入口；Native实参名称及布局见各档entry.bindings.json。"""
    return PreparedStaticCSA(native_args, history=history, batch=batch, query_tokens=query_tokens,
                             workspace_guard_bytes=workspace_guard_bytes)
