"""BF16 bit and ordered-ULP metrics; tolerate neither NaN nor Inf silently."""


def bit_metrics(actual, expected):
    import torch

    a, e = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
    if a.shape != e.shape or a.dtype != e.dtype:
        return {"comparable_dtype": False, "actual_dtype": str(a.dtype), "expected_dtype": str(e.dtype)}
    result = {
        "comparable_dtype": True,
        "dtype": str(a.dtype),
        "numel": a.numel(),
        "equal_fraction": (a == e).double().mean().item(),
    }
    if a.dtype != torch.bfloat16 or not a.numel():
        return result
    aa = a.view(torch.int16).to(torch.int32).reshape(-1) & 65535
    ee = e.view(torch.int16).to(torch.int32).reshape(-1) & 65535

    def ordered(bits):
        return torch.where((bits & 32768) != 0, 32768 - (bits & 32767), 32768 + bits)

    distance = (ordered(aa) - ordered(ee)).abs()
    finite = torch.isfinite(a.float()).reshape(-1) & torch.isfinite(e.float()).reshape(-1)
    result.update({"bit_equal_fraction": (aa == ee).double().mean().item(), "nonfinite_count": (~finite).sum().item()})
    for name, mask in (
        ("all_finite", finite),
        ("reference_abs_ge_1e-2", finite & (e.float().reshape(-1).abs() >= 1e-2)),
    ):
        d = distance[mask].double()
        result[name] = (
            {
                "count": d.numel(),
                "max_ulp": int(d.max()),
                "p99_ulp": torch.quantile(d, 0.99).item(),
                "fraction_le_1_ulp": (d <= 1).double().mean().item(),
                "fraction_le_2_ulp": (d <= 2).double().mean().item(),
            }
            if d.numel()
            else {"count": 0}
        )
    return result
