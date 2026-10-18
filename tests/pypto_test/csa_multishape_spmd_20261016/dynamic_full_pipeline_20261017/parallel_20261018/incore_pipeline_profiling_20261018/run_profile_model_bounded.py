#!/usr/bin/env python3
"""仅模拟诊断：保留全72核及真实同步条件，将失效时钟退避改成有界重读。

正式ops源码和原始二进制都保留。与真实算子的数学、metadata、分工及GM
依赖条件一致，但轮询退避的时间语义不同，因此只用于定位计算/MTE流水
及依赖关系，不能把模型等待时长当成正式实现的定量性能证据。
"""

import run_profile
from prepare_standalone import HERE, prepare_source


if __name__ == "__main__":
    prepare_source(bounded_model_backoff=True)
    run_profile.BUILD = HERE / "build_output/source_build_model_bounded"
    run_profile.BINDINGS = HERE / "WORKLOAD_BINDINGS_MODEL_BOUNDED.json"
    run_profile.RUN_NAME = "full_csa_all72_model_bounded"
    raise SystemExit(run_profile.main())
