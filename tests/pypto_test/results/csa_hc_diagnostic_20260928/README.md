# HC同输入隔离诊断

正式layer4 HC权重、B16/S6合成BF16输入，Native确定性0；不加载attention和MoE，不提供模型性能。
主算子源码2a740c1f。[程序](diagnose.py)、[命令](run.sh)、[原始结果](report.json)。

初次task_20260928_012144_372486520594因诊断脚本直接使用checkpoint FP32 norm权重，
触发PyPTO同dtype cast错误；实际模型创建BF16 RMSNorm参数后copy_，会把该权重转BF16。
修正诊断加载，task_20260928_012327_376775325293退出0，没有修改生产路径。

| 同输入比较 | 不一致元素 | RMSE | max_abs |
| --- | ---: | ---: | ---: |
| Native pre两次执行，mixed/post/comb | 0 | 0 | 0 |
| PTO pre+norm 对 Native | 995/393216 | 1.1585e-5 | 0.001953125 |
| PTO post gate 对 Native | 288/384 | 6.8401e-6 | 4.2617e-5 |
| PTO comb 对 Native | 1533/1536 | 2.1839e-6 | 1.7434e-5 |
| PTO post 使用相同Native输入/gate | 0/1572864 | 0 | 0 |
| Native post 仅换PTO gate | 7623/1572864 | 0.000157192 | 0.015625 |

同输入post本身一致；gate差异直接传到残差的量级较小。独立post使用同一合成attention输出，
不能把其RMSE与完整CSA RMSE直接相减归因；pre/norm的小误差也可能被后续attention/量化放大。
因此不改HC；下一项先隔离已知更大的WO-B量化合同差异：上游式分组标度与Native整token标度。
