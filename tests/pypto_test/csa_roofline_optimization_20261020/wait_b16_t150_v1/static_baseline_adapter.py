# SPDX-License-Identifier: Apache-2.0
"""测试侧历史 selected 静态 SPMD 角色；不替换正式动态包。

factory 只接受 registry 中精确 H/B/S 已验收的 case；B28 当前明确不支持。
绑定调用者现有 Native roots，历史 entry/调度保持原样，仅把 signal helper
的 import 放入独立包命名空间。静态 arena/progress 独占且跨重放保留代际。
compile() 在 cached100 hook 内解析真实 artifact，__call__ 供 eager/graph；
readback(generation) 在调用者 synchronize 后检查进度、全部 padding 和保护区。
本模块顶层只依赖标准库；离线审计不得初始化 torch/PyPTO 或编译内核。
"""

import ast
import contextlib
import importlib.util
import itertools
import json
import shutil
import sys
from pathlib import Path

_NAMES = itertools.count()
_SIGNAL_IMPORT = "from local_signal_adapter import csa_signal_notify, csa_signal_wait"


class UnsupportedStaticCase(ValueError):
    """没有本规格已验收静态 entry；禁止静默借邻档或现场换算法。"""


def _json(path):
    return json.loads(Path(path).read_text())


def resolve_selected_case(registry, *, history, batch, query_tokens=6):
    """标准库只读入口：锁定真实 selected artifact，不读取设备 metadata。"""
    if any(type(v) is not int or v <= 0 for v in (history, batch, query_tokens)):
        raise ValueError("history/batch/query_tokens 必须为已有 Host 正整数描述符")
    registry = Path(registry).resolve()
    rows = [row for row in _json(registry)["rows"]
            if (row["history"], row["batch"], row["query_tokens"]) == (history, batch, query_tokens)]
    if len(rows) != 1:
        raise UnsupportedStaticCase(
            f"静态 selected registry 无唯一已验收 H{history}/B{batch}/S{query_tokens}；"
            "不能复用邻档。128K/B28 须另行生成并完整验证，当前 adapter 不生成。")
    row = rows[0]
    if not row.get("cached_wait") or row.get("wait_backoff_ticks") != 100 or row.get("reuse_gm"):
        raise ValueError("本适配器只接受既有 selected cached100/独立暂存合同")
    result_path = (registry.parent / row["source"]).resolve()
    if not result_path.is_relative_to(registry.parent) or result_path.name != "result.json":
        raise ValueError("selected source 必须指向 registry 内的原始 result.json")
    result = _json(result_path)
    if not result.get("passed") or (result.get("history"), result.get("batch")) != (history, batch):
        raise ValueError("selected 原始结果未通过或规格不匹配")
    name = f"h{history}_b{batch}"
    directory = result_path.parent
    case = directory / "specialization/variants" / name
    files = {key: directory / key for key in (
        "entry_jit.py", "layout.json", "entry.bindings.json", "local_signal_adapter.py")}
    files["schedule.json"] = directory / "specialization/bindings" / f"{name}_prepared.json"
    files["wait_codegen.py"] = registry.parent / "static_shape_archive_20261017/wait_codegen.py"
    files["prepared_arguments.py"] = registry.parent.parent / "csa_static_spmd_feasibility_20261015/prepared_arguments.py"
    files["signal_polling.py"] = registry.parent / "signal_polling.py"
    for key, path in files.items():
        if not path.is_file():
            raise ValueError(f"历史静态源缺失：{key}: {path}")
    # 实际执行 entry 必须与 registry case 的生成 entry 完全相同；不使用文件hash。
    if files["entry_jit.py"].read_bytes() != (case / "entry_jit.py").read_bytes():
        raise ValueError("结果目录 entry 与 selected 生成 case 不同，须先明确实际版本")
    layout = _json(files["layout.json"])
    bindings = _json(files["entry.bindings.json"])
    schedule = _json(files["schedule.json"])
    p = layout["protocol"]
    if (p["core_count"], p["progress_words_per_core"], p["generation_stride"]) != (72, 128, 1024):
        raise ValueError("历史静态 progress 合同改变，不能套用当前 checker")
    progress = layout["workspace"]["sections"]["progress"]
    if progress["dtype"] != "INT32" or progress["shape"] != [72 * 128]:
        raise ValueError("历史静态 progress 不是72核各128字的独占槽")
    if set(layout["workspace"]["sections"]) != {"progress", "records", "waits"}:
        raise ValueError("历史case含未支持的额外控制/诊断区")
    if "task_completion" in layout:
        raise ValueError("本适配器不接受额外task计数器实验")
    roots = schedule["root_arguments"]
    if roots["x_hc"]["shape"][0] != batch * query_tokens or roots["kv_seq_lens"]["shape"] != [batch]:
        raise ValueError("selected schedule 根描述符与规格不一致")
    source = files["entry_jit.py"].read_text()
    rewrite_signal_import(source)  # 标准库审计，拒绝额外隐式导入。
    return dict(registry=str(registry), row=row, case_name=name, files=files,
                layout=layout, bindings=bindings, schedule=schedule,
                historical_result=str(result_path), source=source)


def rewrite_signal_import(source):
    """仅隔离 import；恢复后整文件直接文本相同，不重新生成数学 AST。"""
    imports = [ast.unparse(n) for n in ast.parse(source).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if imports != ["import pypto.language as pl", _SIGNAL_IMPORT] or source.count(_SIGNAL_IMPORT) != 1:
        raise ValueError("历史 entry import 合同改变；禁止隐式切换算子包")
    updated = source.replace(_SIGNAL_IMPORT, _SIGNAL_IMPORT.replace("from ", "from .", 1), 1)
    if updated.replace("from .local_signal_adapter", "from local_signal_adapter", 1) != source:
        raise AssertionError("命名空间变换越过唯一 import")
    return updated


def expected_progress(layout, generation):
    """纯整数重现历史 progress_expectation；word0不是统一generation值。"""
    p = layout["protocol"]
    if type(generation) is not int or not 0 <= generation <= p["max_generation"]:
        raise ValueError("静态信号代际超出协议范围")
    if generation == 0:
        return [0] * p["core_count"]
    base = generation * p["generation_stride"]
    values = [base] * p["core_count"]
    for batches in layout["batches"].values():
        for batch in batches:
            for member in batch["members"]:
                if member["publish_completion"]:
                    core = member["core"]
                    values[core] = max(values[core], base + member["dispatch_seq"])
    return values


def _load(name, path, *, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(Path(path).parent)] if package else None)
    if spec is None or spec.loader is None or name in sys.modules:
        raise RuntimeError(f"无法唯一隔离加载：{name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class PreparedStaticBaseline:
    """同根输入、独占arena，调用数由runner按实际eager/replay维护。"""

    def __init__(self, native_args, *, registry, history, batch, output, query_tokens=6):
        import torch

        spec = resolve_selected_case(registry, history=history, batch=batch, query_tokens=query_tokens)
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.layout = spec["layout"]
        self.selected = dict(spec["row"])
        snapshot = self.output / "source_snapshot"
        case = snapshot / "variants" / spec["case_name"]
        case.mkdir(parents=True)
        (snapshot / "bindings").mkdir()
        # PreparedStaticArguments 的原相对目录合同保持不变。
        for name in ("layout.json", "entry.bindings.json", "local_signal_adapter.py"):
            shutil.copy2(spec["files"][name], case / name)
        shutil.copy2(spec["files"]["schedule.json"], snapshot / "bindings" / (spec["case_name"] + "_prepared.json"))
        for name in ("prepared_arguments.py", "wait_codegen.py", "signal_polling.py"):
            shutil.copy2(spec["files"][name], snapshot / name)
        (case / "entry_jit.original.py").write_text(spec["source"])
        (case / "entry_jit.py").write_text(rewrite_signal_import(spec["source"]))
        (case / "__init__.py").write_text("# 历史静态测试角色的独立加载包。\n")
        name = f"_csa_historical_static_{next(_NAMES)}"
        arguments = _load(name + "_arguments", snapshot / "prepared_arguments.py")
        self.prepared = arguments.PreparedStaticArguments(case, native_args, workspace_guard_bytes=512)
        self.prepared.require_core_capacity(256)
        self.native_args = self.prepared.native_args
        for key, tensor in self.native_args.items():
            if tensor is not native_args[key] or tensor.data_ptr() != native_args[key].data_ptr():
                raise AssertionError(f"历史静态根不再是调用者原Tensor：{key}")
        alias_count = 0
        for item, tensor in zip(self.prepared.contract["ordered_inputs"], self.prepared.core_args):
            ref = item["reference"]
            allocation = ref.get("allocation", "")
            if not allocation.startswith("argument:"):
                continue
            root = native_args[allocation.removeprefix("argument:")]
            if tensor.untyped_storage().data_ptr() != root.untyped_storage().data_ptr():
                raise AssertionError(f"静态derived view不再别名原root：{item['name']}")
            if tensor.storage_offset() != root.storage_offset() + ref["offset"]:
                raise AssertionError(f"静态derived view偏移改变：{item['name']}")
            if ref["layout"] == "NZ" and tensor is not root:
                raise AssertionError("NZ权重必须保留原Tensor对象")
            alias_count += 1
        _load(name, case / "__init__.py", package=True)
        self.module = _load(name + ".entry_jit", case / "entry_jit.py")
        self.wait_module = _load(name + "_wait", snapshot / "wait_codegen.py")
        self.wait_audit = _load(name + "_audit", snapshot / "signal_polling.py")
        self._controls = {
            key: torch.tensor(self.layout["workspace"][key], dtype=torch.int64).reshape(self.prepared.sections[key].shape)
            for key in ("records", "waits")
        }
        self.kernel_config = None
        self._options = None
        self.source_report = dict(
            role="static_spmd", implementation_kind="historical_shape_specialized_spmd",
            target_dynamic_shape=False, registry=spec["registry"], selected=self.selected,
            historical_result=spec["historical_result"], source_snapshot=str(snapshot),
            entry=str(case / "entry_jit.py"), only_entry_change="隔离local_signal_adapter相对import",
            source_files={key: dict(path=str(path), bytes=path.stat().st_size) for key, path in spec["files"].items()},
            same_native_roots=True, native_roots=len(self.native_args), native_aliases_checked=alias_count,
            workspace_bytes=self.prepared.workspace.numel(), workspace_guard_bytes=512,
            root_tensor_count=self.prepared.contract["root_tensor_count"],
            leaf_tensor_count=self.prepared.contract["leaf_tensor_count"],
            new_device_validation=False,
        )
        (self.output / "source_report.json").write_text(json.dumps(self.source_report, ensure_ascii=False, indent=2) + "\n")

    @contextlib.contextmanager
    def _cached_wait(self):
        from pypto.backend import pto_backend

        original = pto_backend._compile_pto_module
        directory = self.output / "signal_polling"
        directory.mkdir(exist_ok=False)
        rewritten = []

        def compile_with_wait(pto_code, unit_name, output_dir, *args, **kwargs):
            source = original(pto_code, unit_name, output_dir, *args, **kwargs)
            if unit_name != "csa_static_group":
                return source
            if rewritten:
                raise RuntimeError("历史静态 MIX 重复编译回调，需检查入口唯一性")
            transformed, change = self.wait_module.rewrite_wait_source(source)
            (directory / "original.cpp").write_text(source)
            (directory / "bypass_wait.cpp").write_text(transformed)
            (directory / "change.json").write_text(json.dumps(change, ensure_ascii=False, indent=2) + "\n")
            rewritten.append(change)
            return transformed

        pto_backend._compile_pto_module = compile_with_wait
        try:
            yield
        finally:
            pto_backend._compile_pto_module = original

    def compile(self, *, config):
        """只在未来设备作业内调用；在首次launch前确认真实cached100 C++。"""
        if self.kernel_config is not None:
            if config is not self._options:
                raise ValueError("static角色的CompileOptions不得在同一轮中途更换")
            return self.compile_report
        with self._cached_wait():
            artifact = self.module.entry._resolve_kernel_artifact(tuple(self.prepared.core_args), {"config": config})
        directory = Path(artifact.output_dir)
        audit = self.wait_audit.audit_wait_artifact(directory, self.output / "signal_polling", 100, cached=True)
        kernel_config = directory / "kernel_config.py"
        if not kernel_config.is_file():
            raise ValueError(f"静态artifact缺少唯一kernel_config：{kernel_config}")
        self.kernel_config = kernel_config
        self._options = config
        self.compile_report = dict(kernel_config=str(kernel_config), wait=audit, artifact=str(directory))
        (self.output / "compile_report.json").write_text(json.dumps(self.compile_report, ensure_ascii=False, indent=2) + "\n")
        return self.compile_report

    def __call__(self, *, config):
        self.compile(config=config)
        return self.module.entry(*self.prepared.core_args, config=config)

    def poison_temporaries(self):
        # 只填业务暂存；records/waits/progress都不能清零或填毒。
        for value in self.prepared.temporaries.values():
            value.fill_(37)

    def readback(self, generation):
        """调用者先synchronize；任何失败都阻止继续采性能。"""
        import torch

        progress = self.prepared.sections["progress"].cpu().view(72, 128)
        expected = torch.tensor(expected_progress(self.layout, generation), dtype=progress.dtype)
        guard = self.prepared.workspace_guard_bytes
        storage = self.prepared.workspace_allocation
        report = dict(
            expected_generation=generation, observed_progress=progress[:, 0].tolist(),
            progress_passed=torch.equal(progress[:, 0], expected),
            all_padding_zero=bool((progress[:, 1:] == 0).all()),
            workspace_guards_passed=bool((storage[:guard].cpu() == 37).all() and (storage[-guard:].cpu() == 37).all()),
            control_tables_unchanged={key: torch.equal(self.prepared.sections[key].cpu(), value)
                                      for key, value in self._controls.items()},
        )
        report["passed"] = all(report[key] for key in (
            "progress_passed", "all_padding_zero", "workspace_guards_passed")) and all(report["control_tables_unchanged"].values())
        if not report["passed"]:
            (self.output / "failed_readback.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            raise AssertionError("静态SPMD进度/全部padding/保护区或只读控制表检查失败")
        return report


def prepare_static_baseline(native_args, *, registry, history, batch, output, query_tokens=6):
    """factory；调用者提供同一Native根字典，独立工作区在capture前一次构造。"""
    return PreparedStaticBaseline(native_args, registry=registry, history=history, batch=batch,
                                  output=output, query_tokens=query_tokens)
