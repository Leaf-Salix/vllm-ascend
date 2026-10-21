# SPDX-License-Identifier: Apache-2.0
"""按Tensor逻辑行数选择单M16 QA编译分支，关闭分支须单独验证完整编排。"""

import argparse
import ast
import copy
import json
import shutil
import textwrap

from prepare import HERE
from prepare_query_ready import replace_once

FLAG = "SINGLE_M16_QA"
POLICIES = {"q_proj_qa_policy", "q_proj_qr_policy", "q_proj_rope_policy", "qkv_proj_rope_policy"}


def append_flag_parameter(source, name):
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    close = next(i for i in range(node.lineno, node.body[0].lineno) if lines[i].strip() == "):")
    lines.insert(close, " " * (node.col_offset + 4) + f"{FLAG}: pl.constexpr = False,\n")
    return "".join(lines)


def append_policy_flag(source, value):
    """仅改调用实参，兼容原有单行与多行调用；AST位置按UTF-8字节处理。"""
    data = source.encode()
    offsets = [0]
    for line in data.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    calls = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in POLICIES
    ]
    for node in sorted(calls, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        start, end = offsets[node.lineno - 1] + node.col_offset, offsets[node.end_lineno - 1] + node.end_col_offset
        node.keywords.append(ast.keyword(arg=FLAG, value=ast.parse(value, mode="eval").body))
        data = data[:start] + ast.unparse(node).encode() + data[end:]
    return data.decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="qa_single_m16_v2")
    parser.add_argument("--candidate", default="qa_shape_gate_v1")
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / n for n in (args.base, args.candidate))
    source = (base / "qkv_proj_rope.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "q_proj_qa_policy")
    selection = next(n for n in node.body if isinstance(n, ast.If))
    old = copy.deepcopy(selection)
    if len(old.orelse) != 1 or ast.unparse(old.orelse[0].test) != "qa_m16_parallel == 2":
        raise ValueError("需要唯一的新增策略2分支")
    old.orelse = old.orelse[0].orelse
    lines = source.splitlines(keepends=True)
    selected = textwrap.dedent("".join(lines[selection.lineno - 1 : selection.end_lineno]))
    original = ast.unparse(old) + "\n"
    body = f"if {FLAG}:\n" + textwrap.indent(selected, "    ") + "else:\n" + textwrap.indent(original, "    ")
    lines[selection.lineno - 1 : selection.end_lineno] = [textwrap.indent(body, " " * selection.col_offset)]
    source = "".join(lines)
    for name in POLICIES:
        source = append_flag_parameter(source, name)
    # 只在显式policy链上传递开关；HCA旧辅助入口等其它调用固定关闭。
    lines = source.splitlines(keepends=True)
    for function in sorted(
        (n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)), key=lambda n: n.lineno, reverse=True
    ):
        text = "".join(lines[function.lineno - 1 : function.end_lineno])
        value = FLAG if function.name in POLICIES else "False"
        text = append_policy_flag(text, value)
        lines[function.lineno - 1 : function.end_lineno] = [text]
    source = "".join(lines)

    root = (base / "decode_csa.py").read_text()
    root = append_flag_parameter(root, "_decode_csa_tp1_layer")
    root = append_policy_flag(root, FLAG)
    old_selection = (
        "            if max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:\n"
        "                if t_dim <= QR_M_TILE:\n"
        "                    qa_m16_parallel = pl.cast(2, pl.INT32)\n"
        "                elif t_dim >= LONG_S6_MIN_QUERY_ROWS:\n"
        "                    if t_dim < 2 * QR_M_TILE:\n"
        "                        qa_m16_parallel = pl.cast(1, pl.INT32)\n"
    )
    fallback = (
        "if max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:\n"
        "    if t_dim >= LONG_S6_MIN_QUERY_ROWS:\n"
        "        if t_dim < 2 * QR_M_TILE:\n"
        "            qa_m16_parallel = pl.cast(1, pl.INT32)\n"
    )
    root = replace_once(
        root,
        old_selection,
        textwrap.indent(
            f"if {FLAG}:\n"
            + textwrap.indent(textwrap.dedent(old_selection), "    ")
            + "else:\n"
            + textwrap.indent(fallback, "    "),
            " " * 12,
        ),
    )
    # HBG仍只保留其host_max_seq_len运行时参数必填；新constexpr默认为False。
    root = replace_once(
        root,
        "        _decode_csa_tp1_layer.__defaults__ = None\n",
        "        _decode_csa_tp1_layer.__defaults__ = (False,)\n",
    )
    root += (
        "\n\ndef csa_compile_constants(tokens: int) -> dict[str, bool]:\n"
        '    """只看Tensor逻辑行数，不读取NPU内容；长度/页表/padding仍由设备处理。"""\n'
        f'    return {{"{FLAG}": 0 < tokens <= QR_M_TILE}}\n'
    )
    adapter = (base / "native_adapter.py").read_text()
    adapter = replace_once(
        adapter,
        "from .decode_csa import _decode_csa_tp1_layer, decode_csa_tp1_layer_hbg, decode_csa_tp1_layer_test\n",
        "from .decode_csa import (\n"
        "    _decode_csa_tp1_layer, csa_compile_constants, decode_csa_tp1_layer_hbg, decode_csa_tp1_layer_test,\n"
        ")\n",
    )
    adapter = replace_once(
        adapter, "    attention: Any\n", "    attention: Any\n    attention_single_m16: Any = None\n"
    )
    adapter = replace_once(
        adapter,
        '    def register(cls, kernel=decode_csa_tp1_layer_test) -> "CSAOperators":\n',
        '    def register(cls, kernel=decode_csa_tp1_layer_test, *, namespace="dsv4_csa", '
        'config=None) -> "CSAOperators":\n',
    )
    adapter = replace_once(
        adapter,
        "        def register(kernel, name):\n"
        '            return pypto.torch.register(kernel, f"dsv4_csa::{name}")\n'
        "\n"
        "        return cls(\n"
        '            register(kernel, "attention"),\n'
        "        )\n",
        "        def register(name, constants=None):\n"
        '            return pypto.torch.register(kernel, f"{namespace}::{name}", constexpr=constants, config=config)\n'
        "\n"
        "        if kernel is decode_csa_tp1_layer_test:\n"
        "            return cls(\n"
        f'                register("attention", {{"{FLAG}": False}}),\n'
        f'                register("attention_single_m16", {{"{FLAG}": True}}),\n'
        "            )\n"
        '        return cls(register("attention"))\n'
        "\n"
        "    def for_tokens(self, tokens):\n"
        f'        if self.attention_single_m16 is not None and csa_compile_constants(tokens)["{FLAG}"]:\n'
        "            return self.attention_single_m16\n"
        "        return self.attention\n",
    )
    adapter = replace_once(
        adapter,
        "    def __call__(self):\n        self.ops.attention(*self.core_args)\n",
        "    def __call__(self):\n"
        '        selector = getattr(self.ops, "for_tokens", None)\n'
        '        entry = self.ops.attention if selector is None else selector(self.args["x_hc"].shape[0])\n'
        "        entry(*self.core_args)\n",
    )
    for code in (source, root, adapter):
        ast.parse(code)
    shutil.copytree(base, target)
    for name, content in (("qkv_proj_rope.py", source), ("decode_csa.py", root), ("native_adapter.py", adapter)):
        (target / name).write_text(content)
    manifest = json.loads((base / "candidate.json").read_text())
    manifest.update(
        candidate=args.candidate,
        base=args.base,
        shape_specialization="根据Tensor形状选择SINGLE_M16_QA constexpr；每个程序仍支持动态长度/页表/位置/padding",
        false_path_goal="消除新增QA策略2的运行时分支及N64 kernel，必须核对整个编排与核列表，不仅旧核数学",
        integration="CSAOperators注册两个固定constexpr的torch operator，NativeCSACall按逻辑行数选择；HBG保留默认关闭",
        changed_files=["qkv_proj_rope.py", "decode_csa.py", "native_adapter.py"],
        status="private_candidate_no_device_evidence",
    )
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
