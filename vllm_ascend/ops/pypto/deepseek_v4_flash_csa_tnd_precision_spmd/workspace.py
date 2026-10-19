# SPDX-License-Identifier: Apache-2.0
"""Small model-owned alias pool for serial full-CSA roots and graph replay.

Captured graphs keep these addresses. They must be replayed serially on the
capture stream, even if Python is not entered. Concurrent model execution or
replay on another stream is outside this experiment's contract.
"""

MAX_TOKENS = 256
HC_SPLITS = 22
NORM_PARAMETERS = (
    ("attn_norm_w", 4096),
    ("gamma_cq", 1024),
    ("gamma_ckv", 512),
    ("cmp_norm_w", 512),
    ("inner_norm_w", 128),
)


def pack_norm_weights(weights):
    """Copy original BF16 bytes once; require a new runtime after weight reload."""
    import torch

    values = [weights[name] for name, _ in NORM_PARAMETERS]
    for (_, size), value in zip(NORM_PARAMETERS, values):
        if value.dtype != torch.bfloat16 or tuple(value.shape) != (size,) or not value.is_contiguous():
            raise ValueError("SPMD Norm packet requires the original contiguous BF16 weight vectors")
    return torch.cat(values)


class PrecisionWorkspace:
    """Share only explicit read aliases; large scratch stays in the root scope."""

    def __init__(self):
        self.device = None
        self.capture_stream = None
        self.views = {}
        self.buffers = {}
        self.order_packets = {}

    def prepare(self, device):
        import torch

        device = torch.device(device)
        if self.device is not None:
            if device != self.device:
                raise ValueError("One SPMD workspace cannot span devices")
            return
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("SPMD workspace must be prepared before capture")
        self.device = device
        self.buffers = {
            "spmd_square": torch.empty((HC_SPLITS * MAX_TOKENS, 16), dtype=torch.float32, device=device),
            "spmd_linear": torch.empty((HC_SPLITS * MAX_TOKENS, 32), dtype=torch.float32, device=device),
            "spmd_accum": torch.empty((MAX_TOKENS + 63, 512), dtype=torch.float32, device=device),
            "spmd_idx_accum": torch.empty((MAX_TOKENS, 64), dtype=torch.float32, device=device),
            "vec_ws": torch.empty((MAX_TOKENS + 8, 64), dtype=torch.float32, device=device),
            "spmd_ffts": torch.zeros(256, dtype=torch.int64, device=device),
        }
        # Finish FFTS initialization before any capture stream can borrow it.
        torch.npu.synchronize(device)
        for tokens in range(1, MAX_TOKENS + 1):
            p16 = (tokens + 15) // 16 * 16
            values = {
                "spmd_square": self.buffers["spmd_square"][: HC_SPLITS * p16],
                "spmd_linear": self.buffers["spmd_linear"][: HC_SPLITS * p16],
                "spmd_accum": self.buffers["spmd_accum"][: tokens + 63],
                "spmd_idx_accum": self.buffers["spmd_idx_accum"][:p16],
                "vec_ws": self.buffers["vec_ws"][: p16 + 8],
                "spmd_ffts": self.buffers["spmd_ffts"],
            }
            for name in ("spmd_square", "spmd_linear", "spmd_accum", "spmd_idx_accum"):
                values[name + "_read"] = values[name]
            self.views[tokens] = values

    def prepare_orders(self, wkv_orders, indexer_orders, oproj_starts, keys):
        """Share immutable qualification tables across layers of this model."""
        import torch

        if self.order_packets:
            if set(keys) != set(self.order_packets):
                raise ValueError("SPMD order profiles changed; recreate the model operators")
            return
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("SPMD order packets must be prepared before capture")
        for key in sorted(set(keys)):
            w, i, o = key
            values = (wkv_orders[w].reshape(-1), indexer_orders[i].reshape(-1), oproj_starts[o].reshape(-1))
            if any(value.dtype != torch.int32 or value.device != self.device for value in values):
                raise ValueError("SPMD order packet requires the original INT32 qualification tables")
            if tuple(value.numel() for value in values) != (512, 1024, 1024):
                raise ValueError("SPMD order packet has an unexpected shape")
            self.order_packets[key] = torch.cat(values)

    def arguments(self, hidden):
        import torch

        if self.device != hidden.device or not self.views:
            raise RuntimeError("SPMD workspace was not prepared for this device")
        tokens = hidden.shape[0]
        if not 1 <= tokens <= MAX_TOKENS:
            raise ValueError("SPMD prefix requires 1..256 real token rows")
        stream = torch.npu.current_stream(hidden.device).npu_stream
        if self.capture_stream is not None and stream != self.capture_stream:
            raise RuntimeError("SPMD captured workspace requires serial execution on its capture stream")
        if torch.npu.is_current_stream_capturing():
            self.capture_stream = stream
        return self.views[tokens]
