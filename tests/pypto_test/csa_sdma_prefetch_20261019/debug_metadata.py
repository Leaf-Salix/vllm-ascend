"""只读记录新增权重视图经过 JIT helper 时的元数据，不改变推导结果。"""
import importlib
from pathlib import Path
import runpy
import sys

decorator = importlib.import_module("pypto.jit.decorator")
original = decorator._resolve_dep_call_metadata


def trace(*args, **kwargs):
    result = original(*args, **kwargs)
    print("META", [getattr(a, "__name__", type(a).__name__) for a in args[:2]],
          {k: str(v) for k, v in result[0].items() if "prefetch" in k or k == "wq_b"}, flush=True)
    return result


decorator._resolve_dep_call_metadata = trace
target = Path(sys.argv[1]).resolve()
sys.argv = [str(target), *sys.argv[2:]]
sys.path.insert(0, str(target.parent))
runpy.run_path(str(target), run_name="__main__")
