# 八档L1测试：证据与复现

## 当前状态

主报告：[八档完整CSA L1实测](../../DSV4_FLASH_CSA_TND_EIGHT_CASE_L1_20261021.md)。
八项任务均已结束：七档通过精度和L1门禁；128K/B28首次Native对比失败，未计时。
八份原始包已取回：七份成功包共35个L1窗口，B28失败包没有L1；大型首差tensor留在远端，经只读CPU脚本分析后取回摘要。
不能把七包CPU核验的PASS解释为八档精度全部通过。

## 文件

- `eight-l1-*-evidence.tar.gz`：七个成功任务的result、运行日志、环境预检及各5份原始L1，另有B28失败包。
- `RECHECK.json`：主线程直接重读成功包、重比源码身份和逐位门禁、重算全部L1，并单列B28失败；明确矩阵尚未通过。
- `INDEPENDENT_*_RECHECK.json`：另一审查者提供的独立核验器逐项CPU复核结果。
- `RUNS.json`：八项任务ID、case、最终状态、已完成档位中位数及B28失败边界。
- `FINAL_TASK_STATUS.json`：最终CLI状态和含START/END、设备释放的日志。
- 工具包内`OBSERVED_B28_FAILURE.txt`：最初SSH返回的状态/日志摘录及后续连接恢复记录；不是完整raw。
- `B28_CPU_DIFFERENCE.json`：只读保存首差tensor，确认TopK第86行选点集合相同但排列不同，最终输出第86/103行不同；六组cache/state逐byte一致。
- `reproduction-tools.tar.gz`：本次固定helper、run/preflight脚本、manifest、参考源码和SHA、GitHub检索记录、独立审查及CPU核验器。
- `SHA256.json`：本目录文件的SHA和长度，不包含其自身。

不打包模型、编译树、大型cache/失败tensor或SDK。对应原环境和模型仍是复现前提。
原始包中preflight文件带有前置vLLM日志，CPU核验器解析其完整JSON对象，不直接对整文件`json.load`。

## 固定参考与实现

公开Nal参考pin `91c4400e19ec10bd96e8de6314376f29229f6e0e`，源码SHA在`REFERENCE.json`。
截图BSH `a241fa61`及其TND具体版本未定位，不将91c重标为截图版本。
我方数值源保持`48c58ce7c467dfce63d5cb5c41afd225488d6012`，候选为
`pkg:deepseek_v4_flash_csa_tnd_precision_spmd_overlap`，64Tensor完整根。
原SDK的六个实际import路径、PTOAS_ROOT、CANN路径等在每包preflight中；没有混用新宽参数SDK。

真实模型目录为`/data/model/DeepSeek-V4-Flash-0731-w8a8`，只加载正式第4层权重。
实验目录为`/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/eight-l1-20261021-codex`。
环境由原`vector-mte2-20261019-codex/full-csa-live-staging-v2/env.sh`加载，
额外绑定`precision-overlap-20261020`的已验收源码及收据；43基础源、6候选源和父包均校验。

## 恢复工具与无设备检查

在新的临时目录解包工具；不要覆盖已有结果。依赖和冻结源码缺失时应恢复其正确版本，不改断言或换共享安装。

```bash
mkdir -p /private/tmp/eight-l1-repro
COPYFILE_DISABLE=1 tar -xzf reproduction-tools.tar.gz -C /private/tmp/eight-l1-repro
/opt/homebrew/bin/rsync -av --exclude='._*' --exclude=.DS_Store --exclude=__pycache__ \
  /private/tmp/eight-l1-repro/ \
  pto227:/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/eight-l1-20261021-codex/
```

`preflight.py`只读实际导入及全冻结manifest，不初始化NPU；解释器版本和六个import路径必须精确匹配。
任务内会再运行预检。`tools/manifest.json`绑定四份实际helper，复制后不应手动改变源。
没有重新安装PyPTO、Simpler、vLLM或修改正式算子的步骤。

## 逐档硬件复现

必须通过task-submit，每次只提交下一档。前一档最终exit0且result.pass及所有精度门禁通过，才允许继续。
当前B28是失败项：应先定位首差并对照Native修正，不直接补性能数据。

```bash
ssh pto227 'task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash /data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/eight-l1-20261021-codex/run.sh repro-eight-8k-b16-v1 16 8192"'
```

`run.sh`参数为`<全新case名> <batch> <history>`；8K用8192、128K用131072。
协议顺序为8K/B16、B32，再128K/B1、B4、B8、B16、B24、B28；均S6、seed1024、page32/NZ2和非均匀Indexer scale。
task内层timeout750秒、外层780秒。观察`--status`与`--log`，每项结束立即收集；不要提交一个覆盖全矩阵的长占卡任务。

Native先跑并捕获静态SK0/DSA多流；PTO首个byte差即抛异常、保存失败证据，不进入性能计时。
通过后warm3，5个L1窗各一次回放，DFX=1/dep_gen=false，输入恢复与同步在窗外。
原始记录检查72核、24AIC/48AIV、无drop、每窗一个根提交，按[3]/[4]下标计算完整incore跨度。
公开Nal脚本使用的level0、同时dep_gen及100tick C++hook均未启用；这些是本报告已披露的协议差异。

## 归档与CPU复核

`collect_case.py <case>`只打包本目录结果和L1，不递归编译树或大型首差tensor。
先确认归档命令正常结束，再用Mac rsync拉回；不要提前拉正在生成的tar。

将七份成功包与工具解包到同一临时目录后：

```bash
python3 recheck.py
python3 independent_recheck.py eight-l1-8k-b16-v1-evidence.tar.gz \
  --batch 16 --history 8192 --out independent-8k16-new.json
```

核验器直接读取raw记录、完整根/精度/保护区门禁、preflight路径及源码身份；重算35个窗口，保留全部样本。
核验器不导入NPU，不运行新的token生成；B28是已确认失败，不能用成功包的核验来替它授予精度资格，也不能代替整模型验收。
