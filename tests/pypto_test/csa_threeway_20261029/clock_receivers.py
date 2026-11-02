# SPDX-License-Identifier: Apache-2.0
"""保守追溯本版PTOAS的GM信号receiver；不认识的表达式直接拒绝。"""

import ast

import regex as re


def verify(source, signal_parameter=56):
    results = {}
    for role in ("cube_schedule", "vector_schedule"):
        # MIX两种核共享统一形参表，不能使用Python角色函数中signal的位置。
        # 调用方必须在编译完成后再用host orchestration核验这个位置。
        expected = "v" + str(signal_parameter)
        start = source.index("AICORE void " + role + "(")
        end = source.find("AICORE void ", start + 12)
        body = source[start : end if end >= 0 else len(source)]
        assert re.search(r"__gm__ int32_t\* " + expected + r"\b", body.split("{", 1)[0])
        calls = list(re.finditer(r"pto::comm::TNOTIFY\((v\d+),", body))
        assert calls
        for call in calls:
            prefix = body[: call.start()]

            def rhs(name, prefix=prefix):
                matches = list(re.finditer(r"\b" + re.escape(name) + r"\s*=\s*([^;\n]+);", prefix))
                if not matches:
                    raise ValueError("missing SSA definition: " + name)
                return matches[-1][1].strip()

            def clean(expr):
                return re.sub(r"\((?:u?int(?:32|64)_t)\)", "", expr).strip()

            def zero(expr, depth=0):
                if depth > 40:
                    return False
                expr = clean(expr)
                if "?" in expr:
                    branches = expr.split("?", 1)[1].rsplit(":", 1)
                    return len(branches) == 2 and all(zero(x, depth + 1) for x in branches)
                try:
                    node = ast.parse(expr, mode="eval").body
                except SyntaxError:
                    return False
                if isinstance(node, ast.Constant) and isinstance(node.value, int):
                    return node.value % 128 == 0
                if isinstance(node, ast.Name):
                    try:
                        return zero(rhs(node.id), depth + 1)
                    except ValueError:
                        return False
                if isinstance(node, ast.BinOp):
                    left, right = ast.unparse(node.left), ast.unparse(node.right)
                    if isinstance(node.op, ast.Mult):
                        return zero(left, depth + 1) or zero(right, depth + 1)
                    if isinstance(node.op, (ast.Add, ast.Sub)):
                        return zero(left, depth + 1) and zero(right, depth + 1)
                return False

            def pointer(expr, depth=0, expected=expected):
                assert depth < 40
                expr = clean(expr)
                if expr == expected:
                    return True
                if re.fullmatch(r"v\d+", expr):
                    return pointer(rhs(expr), depth + 1)
                match = re.fullmatch(r"PTOAS__GLOBAL_TENSOR_DATA\((v\d+)\)", expr)
                if match:
                    return pointer(match[1], depth + 1)
                match = re.fullmatch(r"GlobalTensor<.*>\((.*)\)", expr)
                if match:
                    # 第一参数是ptr，后续shape/stride构造不含地址偏移。
                    return pointer(match[1].split(",", 1)[0], depth + 1)
                node = ast.parse(expr, mode="eval").body
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    for base, offset in ((node.left, node.right), (node.right, node.left)):
                        if zero(ast.unparse(offset)) and pointer(ast.unparse(base), depth + 1):
                            return True
                raise ValueError("cannot prove signal pointer: " + expr)

            assert pointer(call[1]), (role, call[1])
        results[role] = dict(receivers=len(calls), signal_parameter=expected, slot_alignment_words=128)
    return results


def verify_host(source, expected=56):
    assert "const Tensor& ext_clock_signal = orch_args.tensor(55).ref();" in source
    block = source.split("CoreTaskArgs params_t0;", 1)[1]
    parameters = re.findall(r"params_t0\.add_(?:input|output|inout)\((\w+)\);", block)  # codespell:ignore inout
    assert parameters.count("ext_clock_signal") == 1
    assert parameters.index("ext_clock_signal") + 1 == expected
    return dict(root_tensor_index=55, mixed_signal_parameter=expected)
