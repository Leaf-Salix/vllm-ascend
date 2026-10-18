"""Keep this isolated regression run, including xdist workers, on this checkout."""

import importlib.abc
import importlib.machinery
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


class _WorktreeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pypto":
            return importlib.machinery.PathFinder.find_spec(fullname, [str(_ROOT / "python")])
        if fullname.startswith("pypto."):
            return importlib.machinery.PathFinder.find_spec(fullname, path)
        return None


for _NAME in ("pypto", "pypto.pypto_core"):
    if _NAME in sys.modules:
        assert _ROOT in Path(sys.modules[_NAME].__file__).resolve().parents
sys.meta_path.insert(0, _WorktreeFinder())
