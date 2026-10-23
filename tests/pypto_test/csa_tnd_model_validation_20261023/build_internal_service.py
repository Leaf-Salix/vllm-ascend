# SPDX-License-Identifier: Apache-2.0
"""将内部工作区原型接到私有服务副本；设备验收前不修改生产目录。"""

import argparse
import ast
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"迁移锚点不唯一: {old[:80]}")
    return text.replace(old, new)


def migrate_entry(directory, symbol):
    path = directory / "entry.py"
    text = path.read_text()
    original = ast.parse(text)
    root = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == symbol)
    lines = text.splitlines(keepends=True)
    start = min([root.lineno, *(node.lineno for node in root.decorator_list)])
    del lines[start - 1 : root.end_lineno]
    text = "".join(lines)
    prototype = ast.parse((directory / "internal_prototype.py").read_text())
    if any(isinstance(node, ast.FunctionDef) and node.name == "_arithmetic_body" for node in prototype.body):
        raise ValueError("服务迁移仅接受direct-body原型，不能重新引入额外inline包装")
    for node in prototype.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == symbol + "_internal":
            node.name = symbol
        elif node.name == symbol + "_internal_debug":
            node.name = symbol + "_debug"
        text += "\n\n" + ast.unparse(node) + "\n"
    path.write_text(text)


def migrate_binding(directory, symbol):
    path = directory / "binding.py"
    text = path.read_text()
    module = ast.parse(text)
    doc = module.body[0]
    lines = text.splitlines(keepends=True)
    lines[doc.lineno - 1 : doc.end_lineno] = [
        '"""Native TND零拷贝绑定；临时Tensor由PTO编排经Simpler内部申请。\n\n',
        "本层只保存Native输入/输出、描述符和策略；不读回metadata值，\n",
        "不分配临时工作区或信号。显式提供Top-K输出时使用诊断入口。\n",
        '"""\n',
    ]
    text = "".join(lines).replace(
        "from collections.abc import Callable, Mapping", "from collections.abc import Mapping"
    )
    text = replace_once(
        text, "def _pad(value: int, block: int) -> int:\n    return (value + block - 1) // block * block\n\n\n", ""
    )
    text = replace_once(
        text,
        '    """捕获前准备；allocator可供测试加入保护区，生产默认直接分配独立Tensor。"""',
        '    """准备Native根参数；无外部scratch，调用内由Simpler管理临时内存。"""',
    )
    text = replace_once(text, "        allocator: Callable | None = None,\n", "")
    text = replace_once(
        text,
        "        self._direct_names = tuple(direct.items())",
        """        self._diagnostic = "idx_topk" in native
        if self._diagnostic != ("idx_topk_scores" in native):
            raise ValueError("诊断Top-K索引与分数必须同时提供")
        if not self._diagnostic:
            direct.pop("topk_indices")
            direct.pop("topk_scores")
        self._direct_names = tuple(direct.items())""",
    )
    start = text.index("        p8, p16, p64, p128 = ")
    end = text.index('        cache = self.buffers["idx_cache"]', start)
    text = (
        text[:start] + "        self.allocations: dict[str, torch.Tensor] = {}  # 外部工作区计数为零。\n" + text[end:]
    )
    if directory.name.endswith("spmd_tnd"):
        start = text.index('        signal = self.buffers["signal"]')
        end = text.index("\n    def rebind", start)
        text = text[:start] + text[end:]
    tree = ast.parse(text)
    rebind = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "rebind")
    doc = rebind.body[0]
    lines = text.splitlines(keepends=True)
    lines[doc.lineno - 1 : doc.end_lineno] = [
        '        """替换同描述符的Native根参数；内部临时内存由每次PTO调用管理。"""\n'
    ]
    text = "".join(lines).replace(
        "的Native描述符变化，需要另行准备TND工作区", "的Native描述符变化，需要重新准备TND绑定"
    )
    text = replace_once(
        text, f"        from .entry import {symbol}\n", f"        from .entry import {symbol}, {symbol}_debug\n"
    )
    text = replace_once(
        text,
        f"        {symbol}(**self.buffers, **self.policy.constexprs(), **options)",
        f"        entry = {symbol}_debug if self._diagnostic else {symbol}\n"
        "        entry(**self.buffers, **self.policy.constexprs(), **options)",
    )
    path.write_text(text)


def migrate_service(root):
    directory = root / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    path = directory / "native_adapter.py"
    text = path.read_text()
    text = replace_once(
        text,
        "    topk: torch.Tensor,\n    topk_scores: torch.Tensor,",
        "    topk: torch.Tensor | None = None,\n    topk_scores: torch.Tensor | None = None,",
    )
    start = text.index("    if topk.shape != ")
    end = text.index("\n    req = ", start)
    text = (
        text[:start]
        + """    if (topk is None) != (topk_scores is None):
        raise ValueError("诊断Top-K索引与分数必须同时提供")
    outputs = [positions, output]
    if topk is not None:
        if topk.shape != (tokens, 512) or topk.dtype != torch.int32:
            raise ValueError("诊断Top-K要求INT32[T_capacity,512]")
        if topk_scores.shape != (tokens, 512) or topk_scores.dtype != torch.float32:
            raise ValueError("诊断scores要求FP32[T_capacity,512]")
        outputs.extend((topk, topk_scores))
    if any(value.device != hidden.device for value in outputs):
        raise ValueError("TND CSA根输入输出须处于同一设备")
"""
        + text[end:]
    )
    text = replace_once(text, "        idx_topk=topk,\n        idx_topk_scores=topk_scores,\n", "")
    text = replace_once(
        text,
        "    for name, prefix in (\n",
        "    if topk is not None:\n"
        "        native.update(idx_topk=topk, idx_topk_scores=topk_scores)\n"
        "    for name, prefix in (\n",
    )
    text = text.replace(
        "权重和 Top-K/output 工作区由服务持有并在捕获前准备。", "权重及正式output由Native提供；Top-K仅为可选诊断输出。"
    )
    path.write_text(text)

    path = directory / "service.py"
    text = path.read_text()
    text = replace_once(
        text,
        "def __init__(self, attention, operators, max_num_seqs, layer=None, *, workspace_pool=None):",
        "def __init__(self, attention, operators, max_num_seqs, layer=None):",
    )
    text = replace_once(text, "        self._outputs = {}\n", "")
    text = replace_once(text, "        self._workspace_pool = workspace_pool\n", "")
    start = text.index("        output_key = (tokens, hidden.device)")
    end = text.index("        native = bind_native_inputs(", start)
    text = text[:start] + text[end:]
    text = replace_once(text, "            topk=topk,\n            topk_scores=scores,\n", "")
    text = replace_once(
        text, "                allocator=None if self._workspace_pool is None else self._workspace_pool.allocate,\n", ""
    )
    text = text.replace("按容量准备和复用工作区", "按描述符准备绑定、调用内申请工作区")
    text = text.replace(
        "准备对象按根Tensor描述符和策略缓存，warmup完成分配；capture可替换同描述符\n的输入/输出地址，保留工作区与信号代际。",
        "准备对象按根Tensor描述符和策略缓存，不持有外部工作区；capture可替换同描述符\n的输入/输出地址。临时Tensor及信号的初始化和生命周期均在PTO调用内部。",
    )
    path.write_text(text)
    (directory / "workspace.py").unlink()

    path = root / "vllm_ascend/models/pypto_deepseek_v4.py"
    text = path.read_text()
    start = text.index("    runtime_options = {}\n")
    end = text.index("    count = 0\n", start)
    text = text[:start] + text[end:]
    text = replace_once(
        text,
        "attention, operators, max_num_seqs, layer, **runtime_options",
        "attention, operators, max_num_seqs, layer",
    )
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    root = args.source.resolve()
    if "experiments" not in root.parts:
        raise ValueError("仅允许生成私有实验目录，禁止直接改写生产入口")
    for package, symbol in (
        ("deepseek_v4_flash_csa_tnd", "decode_csa_tnd"),
        ("deepseek_v4_flash_csa_spmd_tnd", "decode_csa_spmd"),
    ):
        directory = root / "vllm_ascend/ops/pypto" / package
        migrate_entry(directory, symbol)
        migrate_binding(directory, symbol)
    migrate_service(root)
    print(root)


if __name__ == "__main__":
    main()
