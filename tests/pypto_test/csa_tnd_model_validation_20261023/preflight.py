# SPDX-License-Identifier: Apache-2.0
"""单卡核对环境/Native依赖及TND依赖图；不加载整模型或历史KV。"""

import json
import os
import sys
from pathlib import Path

import torch
import torch_npu


def main():
    torch.npu.set_device(0)
    from vllm.platforms import current_platform

    current_platform.pre_register_and_update()
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    residual = torch.full_like(x, 0.5)
    y, rstd, summed = torch.ops._C_ascend.npu_add_rms_norm_bias(
        x.npu(), residual.npu(), torch.ones_like(x[0]).npu(), None, 1e-6
    )
    expected_sum = x + residual
    expected_rstd = (expected_sum.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    torch.testing.assert_close(summed.cpu(), expected_sum, rtol=0, atol=0)
    torch.testing.assert_close(rstd.cpu(), expected_rstd, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        y.cpu(), (expected_sum.float() * expected_rstd).to(torch.bfloat16), rtol=0.008, atol=0.008
    )
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.entry import decode_csa_spmd as spmd
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd.entry import decode_csa_tnd as mpmd

    for fn in (spmd, mpmd):
        fn._get_dep_graph()
    import pypto
    import simpler
    import vllm

    import vllm_ascend

    report = {
        "status": "PASS",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "vllm": vllm.__file__,
        "vllm_ascend": vllm_ascend.__file__,
        "pypto": pypto.__file__,
        "simpler": simpler.__file__,
        "ptoas": os.environ["PTOAS_ROOT"],
        "shared_release": os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"],
        "cann": os.environ["ASCEND_HOME_PATH"],
        "custom_opp": os.environ["ASCEND_CUSTOM_OPP_PATH"],
        "native_add_rms_norm_bias": "PASS",
        "tnd_dep_graphs": "PASS",
    }
    Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
