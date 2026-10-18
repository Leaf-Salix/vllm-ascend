#!/usr/bin/env python3
"""采集完整动态CSA一次真实调用的物理GM输入与输出，供72核模拟回放。

只包装本测试进程的PreparedDynamicCSA首个eager调用，生产源码不变。保存
完整storage而非tensor.cpu()逻辑值，以保留四份NZ权重、保护区和重叠别名。
每个storage只写一份；根tensor用字节偏移指回它。所有数据仅本地保留。
正常精度检查仍由既有原多task对照驱动执行，capture_complete不代表PASS。
"""

import ctypes
import json
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
DRIVER = HERE.parent / "attention_local_stats_policy_evidence/run_device.py"
CAPTURED = False


def dump_storages(prepared, destination):
    import numpy as np
    import torch
    import torch_npu

    torch.npu.synchronize()
    contract = json.loads((BASE / "source_interface.json").read_text())["runtime_tensors"]
    rows, storages, identities = {}, {}, {}
    for name, tensor in prepared.buffers.items():
        storage = tensor.untyped_storage()
        address, nbytes = int(storage.data_ptr()), storage.nbytes()
        identity = (address, nbytes)
        if identity not in identities:
            key = f"storage_{len(identities):03d}"
            identities[identity] = key
            storages[key] = dict(address=address, bytes=nbytes, names=[], written=False)
        key = identities[identity]
        offset = int(tensor.data_ptr()) - address
        logical_bytes = tensor.numel() * tensor.element_size()
        assert offset >= 0 and offset + logical_bytes <= nbytes, name
        fmt = torch_npu.get_npu_format(tensor)
        rows[name] = dict(storage=key, offset_bytes=offset, shape=list(tensor.shape),
                          strides=list(tensor.stride()), dtype=str(tensor.dtype), format=fmt,
                          logical_bytes=logical_bytes, element_bytes=tensor.element_size())
        storages[key]["names"].append(name)
        storages[key]["written"] |= "pl.InOut[" in contract[name] or "pl.Out[" in contract[name]
    assert len(rows) == 130
    assert sum(s["bytes"] for s in storages.values()) < 16 * 1024**3
    acl = ctypes.CDLL("libascendcl.so")
    memcpy = acl.aclrtMemcpy
    memcpy.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    memcpy.restype = ctypes.c_int

    def write(storage, path):
        host = np.empty(storage["bytes"], dtype=np.uint8)
        error = memcpy(host.ctypes.data, host.nbytes, storage["address"], host.nbytes, 2)
        if error:
            raise RuntimeError(f"aclrtMemcpy D2H失败: {error}, {path.name}")
        host.tofile(path)

    (destination / "before").mkdir(parents=True, exist_ok=False)
    (destination / "after").mkdir()
    for key, storage in storages.items():
        write(storage, destination / "before" / (key + ".bin"))
    report = dict(capture_complete=False, validation_source="../result.json", roots=rows, storages=storages,
                  policy=prepared.policy.constexprs(), tokens=prepared.tokens, requests=prepared.requests,
                  checkpoint="/data/model/DeepSeek-V4-Flash-0731-w8a8", layer=4,
                  capture="首个eager调用，完整storage字节；NZ不转ND，保留别名和保护区",
                  input_scope="既有测试的正式权重与合成历史，B16/S6/H131072",
                  performance_measured=False)
    (destination / "capture.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def finish():
        torch.npu.synchronize()
        for key, storage in storages.items():
            if storage["written"]:
                write(storage, destination / "after" / (key + ".bin"))
        report["capture_complete"] = True
        (destination / "capture.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    return finish


def main():
    sys.path.insert(0, str(BASE))
    import source_loader

    original_loader = source_loader.load_binding
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()

    def capture_binding(source, label):
        binding = original_loader(source, label)
        original_class = binding.prepared

        class CapturePrepared(original_class):
            def __call__(self, *, config=None):
                global CAPTURED
                if CAPTURED:
                    return super().__call__(config=config)
                CAPTURED = True
                finish = dump_storages(self, output / "replay")
                result = super().__call__(config=config)
                finish()
                return result

        binding.prepared = CapturePrepared
        return binding

    source_loader.load_binding = capture_binding
    sys.path.insert(0, str(DRIVER.parent))
    runpy.run_path(str(DRIVER), run_name="__main__")


if __name__ == "__main__":
    main()
