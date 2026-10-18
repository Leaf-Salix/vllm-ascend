# SPDX-License-Identifier: Apache-2.0
"""CSA 动态源码准备模块：整体思路与边界。

输入与输出
    输入为具名 ``query_tokens``、``token_capacity`` 及调用者的缓存目录；
    输出为隔离的高层数学源码包，可进一步取得原动态入口或逻辑 IR。
    不接收本轮 Tensor、页表内容、实际 batch 或静态 task 排程。

动态变量与容量策略
    ``query_tokens`` 是一个请求本轮统一的 query 数 S，是数学分组配置；
    ``token_capacity`` 是工作区容量上界。配置常量不等于固定本轮 B/T：
    实际请求数和 token 数仍分别由原源码 ``B_DYN/T_DYN = pl.dynamic(...)``
    承载。容量同时满足现有 O projection 的 128 行对齐和可容纳整数个 S 行
    请求的约束，不能按传入 B/T 把整个执行图展开。此处不选择性能区间；
    未来融合策略必须另外保留动态范围。

准备流程与执行职责
    从同仓原 CSA 的相对 import 闭包复制源码，只调整四项配置赋值，再通过
    独立 Python 命名空间导入；所有消费者获得同一个 S。随后可做符号解析，
    或运行到 OutlineIncoreScopes 并将动态逻辑 IR 写入调用者的产物目录。
    原数学中的 HC、投影、压缩、Indexer、Attention、输出阶段及各自 AIC/AIV
    分工和 task 依赖保持原样。本模块不添加核间同步、不分配设备工作区，
    也不绑定业务 Tensor；后续动态融合必须实现自身阶段数据流与通知协议。

与完整 CSA 的关系及当前边界
    这是原多 task 高层源码的准备接口，尚未实现动态单 SPMD，不能作为融合
    完成的证据。仅 S4/S6 有既往源码编译证据；其余满足结构约束的配置只是
    编译候选。奇数 query 和一个 batch 内不同 query 数尚不支持。
    不提供冻结调度回退，不依赖历史核体，不修改共享 config。源码物化阶段
    只有 AST 和文件操作；后续 import_entry/specialize/outline 会导入原数学
    的 nz_mode。如果 NZ=2 且 torch.npu 已初始化，原模块可能执行权重格式
    探针。因此不能把整个工具链描述成绝对无设备操作；隔离 CPU 验证使用
    未初始化的设备环境。仅导入本文件本身不加载 torch/PyPTO、不初始化 NPU。

阅读顺序
    QuerySourceConfig 定义结构边界；materialize_query_source 准备独立包；
    QuerySourceBundle.import_entry 核对配置传播；specialize/outline 再进行
    符号编译。这里的 specialize 是 PyPTO 的签名解析步骤，不是将实际 B/T
    代入历史静态展开器。输出仍必须核对动态轴，不能仅凭方法名判断动态性。
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

_IMPORT_LOCK = threading.RLock()


@dataclass(frozen=True)
class QuerySourceConfig:
    """原数学模板的统一 query 配置与独立容量，未绑定实际 B/T。"""

    query_tokens: int
    token_capacity: int = 384

    def __post_init__(self) -> None:
        # pair 两行不能跨请求；双 AIV 各处理请求内 S/2 行。
        if type(self.query_tokens) is not int or not 2 <= self.query_tokens <= 16:
            raise ValueError("query_tokens 必须是 2..16 的整数；当前 Native decode 边界为 16")
        if self.query_tokens % 2:
            raise ValueError("现有 pair/双 AIV 数学分组不支持奇数 query_tokens")
        # 384 是现有整层工作区已采用的容量边界，不将扩大容量冒充动态形状支持。
        if type(self.token_capacity) is not int or not 0 < self.token_capacity <= 384:
            raise ValueError("token_capacity 必须是 1..384 的整数")
        if self.token_capacity % 128:
            raise ValueError("现有 O projection 容量必须按 128 行对齐")
        # 原源码同时用 B_capacity*S 和 DECODE_TOKENS 表达容量；两者必须相等。
        if self.token_capacity % self.query_tokens:
            raise ValueError("容量必须包含完整 uniform 请求，不能静默截断尾请求")

    @property
    def request_capacity(self) -> int:
        return self.token_capacity // self.query_tokens


@dataclass(frozen=True)
class OutlinedQuerySource:
    """当前编译策略下的动态逻辑 IR；不是静态排程或设备核体。"""

    program: Any
    config: QuerySourceConfig
    pass_names: tuple[str, ...]
    logical_source_ir: Path | None


@dataclass(frozen=True)
class QuerySourceBundle:
    """一个不可变配置的源码副本；缓存目录由调用者管理。"""

    config: QuerySourceConfig
    package: Path
    namespace: str
    modules: tuple[str, ...]
    query_consumers: tuple[str, ...]

    @property
    def manifest_path(self) -> Path:
        return self.package.parent / "source_manifest.json"

    def import_entry(self) -> ModuleType:
        """导入高层动态数学入口，逐模块核对 S 传播；不修改 sys.path。"""
        with _IMPORT_LOCK:
            existing = sys.modules.get(self.namespace)
            if existing is not None:
                if Path(existing.__file__).resolve() != self.package / "__init__.py":
                    raise RuntimeError("隔离模块命名空间被其他源码占用")
            else:
                spec = importlib.util.spec_from_file_location(
                    self.namespace, self.package / "__init__.py",
                    submodule_search_locations=[str(self.package)],
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"不能加载 CSA 源码包：{self.package}")
                package = importlib.util.module_from_spec(spec)
                sys.modules[self.namespace] = package
                try:
                    spec.loader.exec_module(package)
                except BaseException:
                    self._discard_import()
                    raise
            try:
                entry = importlib.import_module(f"{self.namespace}.decode_csa")
                config = importlib.import_module(f"{self.namespace}.config")
                expected = {
                    "DECODE_SEQ": self.config.query_tokens,
                    "DSPARK_SPEC_TOKENS": self.config.query_tokens - 1,
                    "DECODE_TOKENS": self.config.token_capacity,
                    "DECODE_BATCH": self.config.request_capacity,
                }
                for name, value in expected.items():
                    if getattr(config, name) != value:
                        raise RuntimeError(f"隔离配置不匹配：{name}")
                for name in self.query_consumers:
                    module = importlib.import_module(f"{self.namespace}.{name}")
                    if Path(module.__file__).resolve().parent != self.package:
                        raise RuntimeError(f"消费者并非来自同一源码包：{name}")
                    if self.config.query_tokens != module.DECODE_SEQ:
                        raise RuntimeError(f"query 配置传播不完整：{name}")
                    if getattr(module, "S", self.config.query_tokens) != self.config.query_tokens:
                        raise RuntimeError(f"请求内行数与 query 配置不一致：{name}")
                if self.config.token_capacity != entry.T or self.config.request_capacity != entry.B:
                    raise RuntimeError("CSA 工作区容量与显式配置不一致")
                return entry
            except BaseException:
                self._discard_import()
                raise

    def _discard_import(self) -> None:
        # 只撤回本工厂自己的失败导入，不影响 canonical 数学模块或其它配置。
        for name in tuple(sys.modules):
            if name == self.namespace or name.startswith(self.namespace + "."):
                del sys.modules[name]

    def specialize(self) -> Any:
        """仅做高层符号解析；调用者负责选择 PyPTO backend 和编译策略。"""
        import pypto.language as pl

        entry = self.import_entry()
        # 原函数保留 B_DYN/T_DYN；不传 descriptors，不按实际 batch 展开控制流。
        return pl.jit(auto_scope=False)(entry._decode_csa_tp1_layer).specialize()

    def outline(self, *, output_directory: str | Path | None = None) -> OutlinedQuerySource:
        """运行到逻辑 incore 划分，可选把动态 IR 写到调用者的产物目录。

        这只是原多 task 高层编排的编译中间态，尚不是融合入口。绝不调用旧的
        descriptor 绑定、静态 task 展开或冻结核体抽取链。
        """
        from pypto.ir.pass_manager import PassManager
        from pypto.ir.printer import python_print

        program = self.specialize()
        names = []
        for transform in PassManager.get_strategy().passes:
            name = transform.get_name()
            program = transform(program)
            names.append(name)
            if name == "OutlineIncoreScopes":
                break
        else:
            raise RuntimeError("当前 PyPTO pass 策略没有 OutlineIncoreScopes")
        source = python_print(program)
        # 出口仍必须包含请求轴与 token 轴。此检查不能代替未来融合核的动态审计。
        dynamic = _dynamic_declarations(ast.parse(source))
        if not {"B_DYN", "T_DYN"}.issubset(dynamic):
            raise RuntimeError("逻辑 IR 丢失 B_DYN/T_DYN；拒绝当作动态源码导出")
        path = None
        if output_directory is not None:
            directory = Path(output_directory).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "logical_source_ir.py"
            with path.open("x", encoding="utf-8") as stream:
                stream.write(source)
        return OutlinedQuerySource(program, self.config, tuple(names), path)


def _dynamic_declarations(tree: ast.AST) -> dict[str, str]:
    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        call = node.value
        if not isinstance(node.targets[0], ast.Name) or not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "dynamic":
            continue
        result[node.targets[0].id] = ast.unparse(call)
    return result


def _source_closure(source: Path) -> dict[str, str]:
    """追踪高层数学入口的相对 import；不纳入 Native adapter/service。"""
    pending, modules = ["decode_csa", "__init__"], {}
    while pending:
        name = pending.pop()
        if name in modules:
            continue
        text = (source / f"{name}.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        modules[name] = text
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            if node.level != 1 or node.module is None or "." in node.module:
                raise ValueError(f"源码相对导入发生结构变化，需显式适配：{name}:{node.lineno}")
            pending.append(node.module)
    return dict(sorted(modules.items()))


def _configure_source(text: str, config: QuerySourceConfig) -> tuple[str, list[dict]]:
    values = {
        "DECODE_BATCH": config.request_capacity,
        "DSPARK_SPEC_TOKENS": config.query_tokens - 1,
        "DECODE_SEQ": config.query_tokens,
        "DECODE_TOKENS": config.token_capacity,
    }
    lines, edits = text.splitlines(keepends=True), []
    for node in ast.parse(text).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not isinstance(node.targets[0], ast.Name) or node.targets[0].id not in values:
            continue
        name = node.targets[0].id
        if node.lineno != node.end_lineno or any(edit["name"] == name for edit in edits):
            raise ValueError(f"配置赋值结构发生变化，需审查：{name}")
        edits.append({"name": name, "original": ast.unparse(node.value), "value": values[name]})
        lines[node.lineno - 1] = f"{name} = {values[name]}  # 当前隔离源码包的显式配置\n"
    if {edit["name"] for edit in edits} != set(values):
        raise ValueError("原数学配置缺少预期赋值，拒绝部分配置传播")
    return "".join(lines), edits


def materialize_query_source(
    cache_directory: str | Path, *, query_tokens: int, token_capacity: int = 384,
) -> QuerySourceBundle:
    """复制当前同仓数学源码，只改四项配置，不改函数或动态轴。

    ``cache_directory`` 必须尚不存在；不会复用历史冻结 IR。产物目录可以清理，
    但已 import 的包源码在进程结束前应保留，以供 PyPTO inspect 原函数。
    """
    config = QuerySourceConfig(query_tokens=query_tokens, token_capacity=token_capacity)
    source = Path(__file__).resolve().parent.parent / "deepseek_v4_flash_csa"
    modules = _source_closure(source)
    root = ast.parse(modules["decode_csa"])
    dynamic = _dynamic_declarations(root)
    if not {"B_DYN", "T_DYN"}.issubset(dynamic):
        raise ValueError("原 CSA 入口必须保留 B_DYN/T_DYN 动态声明")
    functions = [node for node in ast.walk(root)
                 if isinstance(node, ast.FunctionDef) and node.name == "_decode_csa_tp1_layer"]
    if len(functions) != 1:
        raise ValueError("原 CSA 入口定义发生变化，需要显式适配")
    arguments = {argument.arg: argument.annotation for argument in functions[0].args.args}
    for name, axis in (("x_hc", "T_DYN"), ("kv_seq_lens", "B_DYN"), ("x_out", "T_DYN")):
        annotation = arguments.get(name)
        if annotation is None or not any(isinstance(node, ast.Name) and node.id == axis
                                         for node in ast.walk(annotation)):
            raise ValueError(f"原入口 {name} 未使用预期动态轴 {axis}")
    configured, edits = _configure_source(modules["config"], config)
    consumers, inventory = [], []
    for name, text in modules.items():
        tree = ast.parse(text)
        if any(isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "config"
               and any(alias.name == "DECODE_SEQ" for alias in node.names) for node in ast.walk(tree)):
            consumers.append(name)
        inventory.append({"module": name, "source": str(source / f"{name}.py"),
                          "source_bytes": len(text.encode()), "copied_without_changes": name != "config",
                          "dynamic_declarations": _dynamic_declarations(tree)})
    directory = Path(cache_directory).resolve()
    # 随机命名空间只用于 Python import 隔离，不是数据或源码校验。
    namespace = f"_csa_query_s{config.query_tokens}_t{config.token_capacity}_{uuid.uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    package = directory / namespace
    package.mkdir()
    for name, text in modules.items():
        (package / f"{name}.py").write_text(configured if name == "config" else text, encoding="utf-8")
    manifest = {
        "config": asdict(config), "request_capacity": config.request_capacity,
        "namespace": namespace, "package": str(package), "modules": inventory,
        "decode_seq_consumers": consumers, "config_edits": edits,
        "root_dynamic_declarations": dynamic, "actual_batch_bound": False,
        "actual_tokens_bound": False, "single_spmd_implemented": False,
        "shared_source_modified": False, "frozen_ir_used": False,
        "compiler_configuration": "由调用者选择 backend/NZ/pass 策略；本工厂不修改进程编译配置",
        "limitations": ["仅 uniform 偶数 query；不支持 odd/mixed",
                        "S4/S6 之外仅为满足结构约束的编译候选，未证明设备支持",
                        "这是原多 task 动态源码准备，完整动态单 SPMD 尚待实现"],
    }
    (directory / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return QuerySourceBundle(config, package, namespace, tuple(modules), tuple(consumers))
