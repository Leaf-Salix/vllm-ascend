# SPDX-License-Identifier: Apache-2.0
"""CPU 逐元素门禁；不推断容差，不把性能完成当作正确性通过。"""

import math


def compare_topk(actual, expected, visible_counts):
    """检查索引有效性并区分排列/集合差异；不把集合差异自动认作允许的精度权衡。"""
    import torch

    result = {"status": "FAIL", "scope": "Top-K 结构与集合诊断，不代表选择规则或整模型验收"}
    if (actual.ndim != 2 or actual.shape != expected.shape or actual.dtype != expected.dtype
            or actual.dtype not in (torch.int32, torch.int64) or not actual.numel()):
        return {**result, "reason": "Top-K 必须是同 shape/dtype 的非空二维整数张量"}
    if (visible_counts.shape != (actual.shape[0],) or visible_counts.dtype not in (torch.int32, torch.int64)
            or bool((visible_counts < 0).any())):
        return {**result, "reason": "每行可见候选数必须是非负整数"}
    a, b = actual.detach().cpu(), expected.detach().cpu()
    failures, examples = [], []
    order_only, different_sets, replaced = 0, 0, 0
    for row, (left, right, limit) in enumerate(zip(a.tolist(), b.tolist(), visible_counts.cpu().tolist())):
        selections = []
        for name, indices in (("actual", left), ("expected", right)):
            chosen = [index for index in indices if index >= 0]
            invalid = [index for index in indices if index < -1 or index >= limit]
            if invalid or len(set(chosen)) != len(chosen) or len(chosen) != min(a.shape[1], limit):
                failures.append({"row": row, "side": name, "visible_count": limit,
                                 "invalid": invalid[:8], "selected": len(chosen),
                                 "unique": len(set(chosen))})
            selections.append(set(chosen))
        only_actual = sorted(selections[0] - selections[1])
        only_expected = sorted(selections[1] - selections[0])
        if only_actual or only_expected:
            different_sets += 1
            replaced += max(len(only_actual), len(only_expected))
            if len(examples) < 8:
                examples.append({"row": row, "only_actual_count": len(only_actual),
                                 "only_expected_count": len(only_expected),
                                 "only_actual": only_actual[:8], "only_expected": only_expected[:8]})
        elif left != right:
            order_only += 1
    result.update(status="FAIL" if failures else "MEASURED", rows=a.shape[0],
                  position_mismatches=int((a != b).sum()), order_only_rows=order_only,
                  different_set_rows=different_sets, replaced_indices=replaced,
                  invalid_rows=len({entry["row"] for entry in failures}),
                  structural_errors=failures[:8], set_examples=examples)
    return result


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
