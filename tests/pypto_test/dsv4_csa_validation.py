# SPDX-License-Identifier: Apache-2.0
"""CPU 逐元素门禁；不推断容差，不把性能完成当作正确性通过。"""

import math


def compare_tensor(actual, expected, atol, rtol):
    import torch

    result = {"status": "FAIL", "shape": list(actual.shape), "dtype": str(actual.dtype),
              "expected_shape": list(expected.shape), "expected_dtype": str(expected.dtype),
              "atol": atol, "rtol": rtol, "elements": actual.numel()}
    if not all(math.isfinite(value) and value >= 0 for value in (atol, rtol)):
        return {**result, "reason": "容差必须是非负有限值"}
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {**result, "reason": "shape/dtype 不一致，禁止广播或隐式转换"}
    if not actual.numel():
        return {**result, "reason": "没有可比较的元素"}
    a, b = actual.detach().cpu(), expected.detach().cpu()
    if a.is_complex():
        return {**result, "reason": "CSA 比较器不支持复数"}
    if not a.is_floating_point():
        if atol or rtol:
            return {**result, "reason": "整数默认精确比较；量化/Top-K 差异需专用规则，不能套浮点容差"}
        # INT64 的有效差异不能在转成浮点之后再判断。
        mismatch = a != b
        result.update(comparison="exact", nonfinite=0, max_abs=None, rmse=None)
    else:
        # FP64 统计避免 FP32 大哨兵值的差与平方溢出；不影响原 dtype 的合同检查。
        af, bf = a.double(), b.double()
        finite = torch.isfinite(af) & torch.isfinite(bf)
        difference = (af - bf).abs()
        mismatch = ~finite | (difference > atol + rtol * bf.abs())
        nonfinite = int((~finite).sum())
        result.update(comparison="tolerance", nonfinite=nonfinite,
                      max_abs=None if nonfinite else float(difference.max()),
                      rmse=None if nonfinite else float(difference.square().mean().sqrt()))
    count = int(mismatch.sum())
    result.update(status="FAIL" if count else "PASS", mismatches=count,
                  first_mismatches=mismatch.nonzero()[:8].tolist() if count else [])
    return result


def validate_outputs(actual, required_names, reference=None, tolerances=None):
    """只对明确列出的输出/状态判定；任何缺项都失败，无参考只能 MEASURED。"""
    import torch

    names = tuple(required_names)
    errors, checks = {}, {}
    if not names:
        errors["required_names"] = "未声明必需输出"
    for name in names:
        if name not in actual:
            errors[name] = "缺少实际输出"
            continue
        value = actual[name]
        if not isinstance(value, torch.Tensor) or not value.numel():
            errors[name] = "输出必须是非空张量"
            continue
        if reference is None:
            finite = bool(torch.isfinite(value).all())
            checks[name] = {"finite": finite, "shape": list(value.shape), "dtype": str(value.dtype)}
            if not finite:
                errors[name] = "输出包含 NaN/Inf"
            continue
        if name not in reference or not isinstance(reference[name], torch.Tensor):
            errors[name] = "缺少参考张量"
            continue
        limits = (tolerances or {}).get(name)
        if value.is_floating_point() and (not isinstance(limits, dict) or set(limits) != {"atol", "rtol"}):
            errors[name] = "浮点输出必须逐项声明 atol/rtol"
            continue
        limits = limits if limits is not None else {"atol": 0, "rtol": 0}
        checks[name] = compare_tensor(value, reference[name], **limits)
        if checks[name]["status"] != "PASS":
            errors[name] = checks[name].get("reason", "逐元素比较失败")
    return {"status": "FAIL" if errors else ("MEASURED" if reference is None else "PASS"),
            "scope": "声明的输出与状态逐元素检查；不代表整模型、token 或保护区验收",
            "required_names": list(names), "reference_provided": reference is not None,
            "checks": checks, "errors": errors}
