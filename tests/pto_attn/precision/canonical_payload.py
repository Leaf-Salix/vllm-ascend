"""Deterministic logical histories in the native 128-token cache layout."""

import hashlib

import torch

from .fixture import cache_views


def canonicalize(payload, args, page_size):
    for root in payload["initial"]:
        root.zero_()
    # Recreate CPU views while preserving native shared-allocation aliasing.
    views = cache_views(payload["initial"])
    digest = hashlib.sha256()
    for request in range(args.batch):
        for group, width in ((0, 512), (1, 512), (2, 2048), (3, 512), (4, 128), (5, 1)):
            if group in (0, 4, 5):
                positions = torch.arange(args.start_pos // 4)
            else:
                positions = torch.arange(max(0, args.start_pos - (127 if group == 1 else 7)), args.start_pos)
            generator = torch.Generator().manual_seed(620000 + group * 100 + request)
            shape = (positions.numel(), 1, width)
            if group == 4:
                values = torch.randint(-100, 101, shape, dtype=torch.int8, generator=generator)
            elif group == 5:
                values = torch.full(shape, 0.01, dtype=torch.float16)
            else:
                values = torch.randn(shape, dtype=views[group].dtype, generator=generator) * 0.1
            md_group = {0: 0, 1: 4, 2: 1, 3: 2, 4: 3, 5: 3}[group]
            rows = page_size // 16 if group in (2, 3) else page_size
            ids = payload["tables_cpu"][md_group][request, positions // rows].long()
            views[group][ids, positions % rows] = values
            digest.update(values.contiguous().view(torch.uint8).numpy().tobytes())
    x = torch.randn(
        args.batch * args.seq,
        4096,
        dtype=torch.bfloat16,
        generator=torch.Generator().manual_seed(629999),
    )
    digest.update(x.view(torch.uint8).numpy().tobytes())
    payload["x"].copy_(x)
    for path in ("native", "pto"):
        for dst, src in zip(payload[path + "_roots"], payload["initial"]):
            dst.copy_(src)
    torch.npu.synchronize()
    return digest.hexdigest()
