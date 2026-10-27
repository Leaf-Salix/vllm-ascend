# 完整 Indexer 链路实验原始证据

主报告：`../../DSV4_FLASH_CSA_TND_INDEXER_PIPELINE_20261027.md`。
本轮候选未采用，正式精度运行基准11b380e3f不变。

## 无设备复算

在本目录使用Python3.11：

```bash
python analyze_indexer_pipeline.py > /tmp/indexer-stage-recheck.txt
```

脚本直接读取两份原始tar，不解包、不导入Torch，不访问NPU。核对源快照、SDK、
实际READY注册的全部CPU子项字节见证及编排身份一致后，用两次独立dependency映射
关联18个L1窗口；要求每条记录都有精确task token映射，不能凭旧func编号猜测。
复算12个event窗口中位、9个L1窗口中位及主Q_B相对于完整MIX的逐窗调度分布。
输出`INDEXER_PIPELINE_STAGE_RECHECK.json`；执行前可复制旧文件至`/tmp`后用`cmp`核对。

`INDEXER_FIRST_RECHECK.json`是第一项计时完成后的独立简要复算，主报告以完整两项
关联后的`INDEXER_PIPELINE_STAGE_RECHECK.json`为准。未执行Native数值验收，不能把
两个任务的exit0或dependency自身重复一致误读成与Native逐位/逐token一致。

## 硬件复现

源快照、SDK、static13包及模型依赖沿用
`../../DSV4_FLASH_CSA_TND_HAD_OWNER_CHAIN_20261027.md`与其引用的阶段定位报告。
本轮Python3.12.14、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、
CANN9.2.0-beta.2；SDK为PyPTO d44925c1、PTOAS914863ed、Simpler57e49923。
完整导入路径、二进制SHA和原源快照文件清单见每份原始result与preflight。

解包`executed_helpers.tar.gz`到独立实验目录。v1～v5只在CPU失败，保留供追溯；
实际上卡的是`indexer-pipeline-v6`及只读派生`indexer-dependency-v2`。
先分别运行v6的`compile.sh baseline`、`compile.sh candidate`，要求CPU完整64参数、
load通过且NPU未初始化；必须重新核对生成依赖、别名、scope容量。
`indexer-cpu-proof.tgz`保留本轮两臂收据、编排全文、meta/config及binary SHA清单，
不发布二进制。

已有目录不覆盖。若更换实验位置，须更新env及派生parent路径，重新生成SOURCE/
manifest与派生SHA，并重新做CPU和实际注册资格，不能直接挪用旧hash。
运行脚本会拒绝复用已经存在的结果label。

```bash
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  'bash <indexer-pipeline-v6>/run.sh <new-label> 32 8192 <32个逗号分隔的6>'

task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  'bash <indexer-dependency-v2>/run.sh <new-dep-label> 32 8192 dep'
```

内层750秒。`--ptoas 0.63`是队列兼容选项，运行env显式设置并预检PTOAS_ROOT为本轮
native0.69独立工具链，不能把实际编译器记成0.63。
计时任务先平衡12×100完整event及独立L1，不同时打开dep-gen；只在预设性能门槛
通过后执行Native A/B八边界与ABA。本轮门槛失败，因此未执行Native精度或反向采用确认。
依赖任务只采两臂各两窗，验证映射重复、自身输出重复、guard及固定packet，只用于定位。

## 来源与限制

允许借用nalinaly的整体持久MIX设计，但本轮保留我方Score等原计算体与舍入。
并未把nalinaly不同的量化分组或softmax算法复制进来。源来自本分支的
`deepseek_v4_flash_csa_tnd_precision/decode_indexer.py`与`precision_spmd/indexer.py`，
来源SHA、候选精确源码及完整helper闭包见归档中的SOURCE/manifest。
归档根LICENSE保留项目Apache-2.0许可。

真实第4层权重、合成hidden/history，完整HC-pre至HC-post FULL graph；这不是整模型
token实验。只对8K/B32本配置有效，不具备128K小ring或其它动态路由资格。
`GENERATED_REVIEW.json`是独立源级/生成审查摘要，不代表设备精度证明。
两个任务均已退出并释放设备，217秒和132秒，详见TASKS及锁释放日志。
