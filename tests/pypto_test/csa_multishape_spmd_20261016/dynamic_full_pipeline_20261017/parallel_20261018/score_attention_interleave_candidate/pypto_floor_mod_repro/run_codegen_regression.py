# SPDX-License-Identifier: Apache-2.0
"""隔离导入指定PyPTO，执行FloorMod生成C++的CPU数值回归。"""
import argparse
import json
import sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument("--pypto-root",type=Path,required=True);p.add_argument("--label",required=True)
a=p.parse_args();out=Path(__file__).resolve().parent/a.label;out.mkdir(exist_ok=False)
sys.meta_path[:]=[f for f in sys.meta_path if "pypto.pypto_core" not in getattr(f,"known_wheel_files",{})]
sys.path.insert(0,str(a.pypto_root/"python"))
import pypto
from pypto import pypto_core
import pytest
assert a.pypto_root.resolve() in Path(pypto_core.__file__).resolve().parents
from pypto.backend._ptoas_locate import find_ptoas_binary
source=Path(__file__).resolve().parent/"test_integer_floor_mod_codegen.py"
(out/"provenance.json").write_text(json.dumps(dict(pypto=pypto.__file__,core=str(Path(pypto_core.__file__).resolve()),test=str(source),ptoas=str(find_ptoas_binary())),indent=2)+"\n")
code=pytest.main(["-n","0","-vv","--tb=short","--basetemp="+str(out/"pytest_tmp"),"--junitxml="+str(out/"junit.xml"),str(source)])
(out/"exit.json").write_text(json.dumps(dict(pytest_exit=int(code)))+"\n")
sys.exit(int(code))
