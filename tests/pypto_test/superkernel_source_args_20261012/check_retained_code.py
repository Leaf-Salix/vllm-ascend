"""Check that retention did not change measured device instructions."""
import json
import struct
from pathlib import Path

root = Path(__file__).resolve().parent
private = root.parent.parent

def executable_sections(path):
    data = path.read_bytes()
    assert data[:6] == b"\x7fELF\x02\x01", path
    offset = struct.unpack_from("<Q", data, 40)[0]
    stride, count, strings = struct.unpack_from("<HHH", data, 58)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * stride) for i in range(count)]
    table = headers[strings]
    names = data[table[4]:table[4] + table[5]]
    return {
        names[h[0]:names.index(b"\x00", h[0])].decode(): data[h[4]:h[4] + h[5]]
        for h in headers if h[2] & 4 and h[5]
    }

candidate = root / "compiled/early_score/build/prebuilt"
retained = root / "retained/compiled/early_score/build/prebuilt"
files = sorted(candidate.glob("kernel_*.bin"))
changed = [f.name for f in files if f.read_bytes() != (retained / f.name).read_bytes()]
old_runtime = root / "runtime/lib/a2a3/onboard/tensormap_and_ringbuffer/aicore_kernel.o"
new_runtime = private / "simpler/build/lib/a2a3/onboard/tensormap_and_ringbuffer/aicore_kernel.o"
report = {
    "kernel_count": len(files),
    "changed_kernel_executable_sections": changed,
    "runtime_executable_sections_equal": executable_sections(old_runtime) == executable_sections(new_runtime),
    "comparison": "Raw kernel instructions and runtime ELF executable sections; no hashes",
}
(root / "retained/code_equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
