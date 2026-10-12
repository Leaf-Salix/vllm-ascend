"""仅用于逐核诊断：在本进程修正生成时钟适配器的64位汇编寄存器约束。

依赖仓库文件不改。生成C++使用+l，原来的+r在当前CCE上编译失败。
辅助函数另加inline，防止ELF text起点落在辅助函数而非kernel_entry。
正式候选、正式性能轮不经过此入口；带时间戳结果仅用于定位。
"""
import runpy
import sys
from pathlib import Path

ROOT = Path('/data/pyptouser/qinchuanyu/pto-eager')
TESTS = ROOT / 'vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test'
sys.path.insert(0, str(TESTS))
from dsv4_csa_env import activate
activate()
finders = [f for f in sys.meta_path if 'pypto.pypto_core' in getattr(f, 'known_wheel_files', {})]
assert len(finders) == 1
finders[0].known_wheel_files['pypto.pypto_core'] = str(ROOT/'pypto/build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so')
from pypto.backend import pto_backend
old = pto_backend._DEVICE_DIAGNOSTIC_ADAPTERS
assert old.count('"+r"') == 4
pto_backend._DEVICE_DIAGNOSTIC_ADAPTERS = old.replace('"+r"', '"+l"').replace('extern "C" __aicore__ __attribute__', 'extern "C" __aicore__ inline __attribute__')
driver = sys.argv.pop(1)
if driver == 'build':
    runpy.run_path(str(TESTS/'dsv4_hc_pre_bench.py'), run_name='__main__')
elif driver == 'run':
    runpy.run_path(str(Path(__file__).with_name('clock_diagnostic_bench.py')), run_name='__main__')
else:
    raise ValueError(driver)
