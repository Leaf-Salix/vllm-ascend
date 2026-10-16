# TND精度版：完整测试方案与复现步骤（2026-10-16）

本页是已完成测试的复现手册。结果见[完整报告](DSV4_FLASH_CSA_TND_NALINALY_LOW_ACCEPTANCE_20261016.md)，
机器可读结果见[结构化记录](evidence_tnd_precision/nalinaly_protocol_alignment_20261016.json)。
适用分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。

## 1. 已验证结果和边界

| 项目 | 128K/B24 ×192 | 8K/B40 ×192 |
| --- | --- | --- |
| 不同问题数 | 24 | 40 |
| DP副本请求数 | 384 | 640 |
| Native独立重复 | 全部输入、token、事件一致 | 全部输入、token、事件一致 |
| CSA对Native token差异 | 0/73,728 | 0/122,880 |
| 接受统计/完整事件差异 | 0/384请求 | 0/640请求 |
| rank0模块边界逐bit比较 | 16,614条全部一致 | 10,863条全部一致 |
| 实际并发峰值/卡 | 21；提交24，允许等待 | 40 |
| 加权平均接受草稿数 | 3.8264033264 | 3.7035909921 |
| 接受0/1/2/3/4/5次数 | 1200/1392/864/1216/1472/9248 | 1696/2432/2816/2592/2240/14512 |
| 抢占 | 0 | 0 |
| 默认strict READY复用资格 | 失败；首次JIT，数值资格单独通过 | 失败；首次JIT，数值资格单独通过 |

只替换21层CSA，20层HCA保持Native。精度实现保留Native HC/norm、WKV、Indexer权重投影和O-proj边界，
CSA主体运行PTO TND；不能称所有计算都由PTO重写。
最终token/事件与实际输入覆盖全部16rank；逐bit浮点比较仅覆盖rank0记录的位置。
Native1仅观测outer，Native2保存内部黄金；没有声称Native重复的全部内部tensor均对拍。
本轮未测性能，也未覆盖同轮请求query长度不等的独立FULL图矩阵。

使用nalinaly固定公开题目和相同chat编码后缀，复用自有Native target/draft bank；
未获其原始bank读取权限，不声称完整prefix/cache与其逐字节相同。
这是基于227现有、经过身份核验资产的复现方案；不提供全新机器的一键安装包。

## 2. 已随Git归档的材料

目录：[repro_20261016](evidence_tnd_precision/repro_20261016/)。
[归档索引](evidence_tnd_precision/repro_20261016/index.json)固定文件SHA256、体积和基准commit。

| 文件 | 内容 |
| --- | --- |
| `repro-capsule-20261016.tar.gz` | 77份文件：58个精确测试overlay、3337文件源清单、启动/bootstrap链、两档bank计划/题目、运行包/static清单及外部资产指纹 |
| `evidence-8k-b40.tar.gz` | 396份原始小证据：三臂实际token、完整事件、逐rank轨迹、模块摘要、资格结果、实际诊断脚本与配置 |
| `evidence-128k-b24.tar.gz` | 391份同类长档证据，保留观测版本转换及严格READY失败 |

基准Git commit为`7e856c40eb5aca4bdef16dacf26b137387189e4e`；
生产数学源码为`cbc810c8afab47c2425068049440424f014dcde3`。
源清单SHA为`e4de4927d2e5758df65784bddc412d1d90227b56ed978ad2218df50c2bd540ca`。
3337份冻结源文件中516份生产文件与数学commit相同；58份overlay属于测试及其参考资料。
不要仅checkout一个分支便假定测试harness也完全相同。

在仓库根目录，用Python 3.11或更新版本执行（不导入Torch、不占卡）：

```bash
python3 tests/pypto_test/verify_tnd_precision_repro.py \
  --extract /tmp/tnd-precision-record-20261016 \
  --restore-source /tmp/tnd-precision-source-20261016
```

两个目标目录必须不存在。工具验证三个归档、全部成员SHA、路径安全及任务exit0，
再由固定Git对象加overlay还原源树，核对3337份文件SHA且拒绝额外文件。
实际已运行通过；这些命令不会安装软件、修改现有源码或提交NPU任务。
校验工具依赖索引中的基准Git对象，使用保留该对象的正常clone，不能缺省为任意浅克隆HEAD。

## 3. 227运行环境和未随Git交付的资产

下列路径是本次227的真实位置，不能替换成同名但不同内容的环境。

```bash
B=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex
S="$B/root-fix-20261016/source"
R=/data/pyptouser/yejia/lc128
```

| 组件 | 本次实际值 |
| --- | --- |
| 服务器/卡 | 227；16卡Ascend A3；所有NPU工作通过task-submit |
| Python | 3.12；`/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/env` |
| Torch / Torch-NPU | 2.10.0+cpu / 2.10.0.post2；不是旧实验的post4 |
| vLLM | 0.25.1；`/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/src/vllm/vllm` |
| vLLM core来源 | 实际为无Git overlay；新增1967个Python/so文件指纹，不虚构commit |
| CANN | 9.2.0-beta.2；`/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2` |
| ATB | 9.2.0-beta.2、ABI1；`/data/pyptouser/yejia/vllm-cann92-dsv4-tnd-main-20260924/Ascend/nnal/atb/9.2.0-beta.2/atb/cxx_abi_1` |
| PyPTO / Simpler | PyPTO基线3e87a843的私有运行包；Simpler a54c0509552b01e13fb0960e23ce409a01b5024f；以文件清单为准 |
| 私有PyPTO | `$B/low-acceptance-20261010/runtime-package`，206文件清单；实际import必须落入该包 |
| PTOAS | 最终实际0.66；`/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/experiments/nalinaly-71153bb3/reference-env/bin` |
| 官方static包 | `$B/low-acceptance-20261010/official-static-cache`；13份，metadata SHA `863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602` |
| 模型 | `/data/model/DeepSeek-V4-Flash-0731-w8a8`，75分片，约313.36GB |
| 8K bank | `$B/low-acceptance-20261010/banks/fixed8192`，主档40题，history约8224–8466 |
| 128K bank | `$B/low-acceptance-128k-b24-20261014/banks/fixed131072`，主档前24题，history131104–131346 |

`--ptoas 0.63`是队列入口参数；脚本随后明确设置PTOAS_ROOT为0.66私有工具链。
环境名称带cann901不代表最后加载CANN9.0：按归档source链叠加后，必须核对实际ASCEND_HOME_PATH和worker配置。
不要复用skill中旧0.29/post4的表来代替本次实际环境。

未打包模型权重、bank大tensor、venv、PyPTO/Simpler扩展、官方static二进制及rank0原值`.pt`。
它们需从227现存位置或另行备份恢复；缺失时停止，不随机生成输入冒充同一bank。
Bootstrap及每层来源在capsule的`experiment/`和`external/bootstrap/activate.sh`中逐字节保存；
其中继承环境会先source旧CANN，之后再source9.2，最终生效值以任务preflight和worker输出为准。

在227的相同仓库副本中执行：

```bash
python3 tests/pypto_test/verify_tnd_precision_repro.py --verify-assets
```

该命令无设备核验bank全部实际payload、tokens/manifest，vLLM core完整模块集合及SHA，
模型config/tokenizer/index SHA和75分片size/mtime。
Bank payload SHA是测试后本次归档新采集的指纹，不能改称历史任务内已计算过；历史任务验证了header及冻结stat/manifest。
模型分片未做完整payload SHA；迁移后mtime改变也会被严格拒绝，应另建资产身份，不能静默放宽历史身份。
私有SDK206文件、static包及工具链的检查还需下文真实环境preflight，不能仅凭本工具通过就占卡。

## 4. 固定测试参数和验收规则

- TP=1、DP=EP=16；不同问题按DP复制，不将副本计为独立问题。
- 每请求生成192 token，greedy、temperature0、ignore EOS；DSpark提出5，实际接受0–5，不强制接受。
- 三臂：Native控制、Native模块见证、TND精度CSA见证。每臂从同一bank恢复target/draft cache/state，不重新prefill。
- FULL_DECODE_ONLY；长档capture `[6,144]`，短档`[6,240]`；max-num-batched-tokens400。
- 长档24000 KV blocks、utilization0.97；短档22000 blocks、0.95；同档三臂完全相同。
- NZ2、atomic0、torch与Torch-NPU确定性level1、HCCL_DETERMINISTIC=true、event1、EPLB关闭。
- enable_npugraph_ex/static_kernel/fuse_norm_quant=true；官方缓存metadata的super_kernel=false，不能套用其他性能实验的SK1口径。
- async scheduling、CPU binding、共享专家多流开启；prefix caching关闭；重计算调度关闭。
- OMP_NUM_THREADS10、OMP_PROC_BIND=false、HCCL_OP_EXPANSION_MODE=AIV、HCCL_BUFFSIZE1800、VLLM_BATCH_INVARIANT0。
- KV分配上界不得超过21GiB，分配后预留至少2GiB；资源失败与数值失败分别记录。
- 所有16rank逐轮核对请求身份/顺序、input IDs、positions、query bounds/lengths、seq lens、padding、DP向量及入场/完成/抢占事件。
- 全部实际输出token IDs、每请求完整接受事件/统计、跨DP副本一致；零抢占；0–5全覆盖、加权均值3–4，目标约3.8。
- rank0观测覆盖43层，其中21层CSA，输入/输出/positions及CSA normed/heads/oproj/attention_half；outer为hidden/aux/MTP residual。
- 图内观测验证新鲜度，按真实有效行比较；不是只比最终文本，也不以一份未更新的观测缓存充当逐bit结果。
- 首差立即锁定并停止扩大测试，保存STOP/第一差异及原值；不继续跑大配置。
- 每任务max-time780秒；约5分钟查看状态，不创建schedule；排队时间不算占卡时间，不合并为长任务。

## 5. 新建独立复测目录

以下为**未来重跑**，本次归档未再次提交NPU。先在227取得上述已校验的解包目录，
设置`E`为其中根目录（包含`8k/`、`128k/`、`capsule/`）。新RUN只用字母、数字、下划线。
从固定脚本成套替换job和三个输出标签，不覆盖历史结果。

```bash
export E=/tmp/tnd-precision-record-20261016
export B=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex
export R=/data/pyptouser/yejia/lc128
export CASE=8k
export RUN=r20261016a
python3 - <<'PY'
import os, re, shutil
from pathlib import Path
case, run = os.environ['CASE'], os.environ['RUN']
assert case in ('8k', '128k') and re.fullmatch(r'[A-Za-z0-9_]+', run)
oldjob = 'plan-align8k-b22000-20261016' if case == '8k' else 'retest192-20261016'
oldprefix = 'plan8k' if case == '8k' else 'retest192'
blocks = 22000 if case == '8k' else 24000
source = Path(os.environ['E'])/case/'helpers'
job = Path(os.environ['B'])/f'repro-{case}-{run}'
assert not job.exists()
for arm in ('native1','native2','precision'):
    assert not (Path(os.environ['R'])/f'{run}-{case}-{arm}').exists()
job.mkdir()
replacements = {oldjob: job.name}
for arm in ('native1','native2','precision'):
    replacements[f'{oldprefix}-{arm}-b{blocks}'] = f'{run}-{case}-{arm}'
for p in source.iterdir():
    if p.suffix in ('.py','.sh'):
        text = p.read_text()
        for before, after in replacements.items():
            text = text.replace(before, after)
        (job/p.name).write_text(text)
for name in ('reference','observation-v1'):
    origin = source/name
    if origin.is_dir():
        shutil.copytree(origin, job/name, ignore=shutil.ignore_patterns('__pycache__'))
if case == '128k':
    shutil.copy2(source/'diagnostic-window-transition.json', job/'diagnostic-window-transition.json')
    shutil.copytree(Path(os.environ['E'])/'8k/helpers/reference', job/'reference')
    shutil.copy2(Path(os.environ['E'])/'8k/helpers/check_bank128.py', job/'check_bank.py')
for p in job.iterdir():
    if p.suffix in ('.py','.sh'):
        assert not any(before in p.read_text() for before in replacements)
print(job)
PY
export J="$B/repro-$CASE-$RUN"
export N1="$R/$RUN-$CASE-native1"
export N2="$R/$RUN-$CASE-native2"
export C="$R/$RUN-$CASE-precision"
export S="$B/root-fix-20261016/source"
```

原始source/root-fix、bank和SDK保持冻结；新目录只含诊断脚本。
若需从Git恢复遗失的source，使用第2节还原并完整核验后，另建source_job再成套更新引用；不要覆盖仍被其他任务使用的源目录。

先执行无设备检查，不能在登录shell调用模型生成：

```bash
bash "$J/preflight.sh"
bash -n "$J/run-case.sh"
(
  source "$J/env.sh"
  export ASCEND_RT_VISIBLE_DEVICES=""
  python "$J/check_bank.py"
)
```

路径、版本、CANN/ATB/PTOAS、206文件runtime包、static包、bank题目/编码/header、CPU正负控制必须全部通过。
新job的共同helper源码和bank在三臂执行期间冻结；128K下文唯一明确版本切换例外按固定SHA验证。

## 6. 依次提交三项短任务

### 6.1 Native控制

128K历史控制组使用旧版观测helper；其资格器明确验证before→after的SHA转换。
要复现本次方案，应按该已审顺序切换，不能直接删掉版本门禁或谎称两版相同。
8K从头至尾使用同一版helper，无此步骤。

```bash
if [ "$CASE" = 128k ]; then
  cp "$J/admission_trace.py" "$J/admission_trace.after.py"
  cp "$J/observation-v1/admission_trace.py" "$J/admission_trace.py"
fi
TASK_N1=$(task-submit --ptoas 0.63 --device auto --device-num 16 \
  --max-time 780 --env PYTHONNOUSERSITE=1 \
  "bash $J/run-case.sh native-control $RUN-$CASE-native1")
printf '%s\n' "$TASK_N1"
task-submit --status "$TASK_N1"
```

每约5分钟再查询`--status`及`--log`；只有completed exit0且无STOP才能继续。若超时/异常，先定位，不能预提交后续任务。
控制结束后，恢复128K已审新版helper，再派生参考：

```bash
if [ "$CASE" = 128k ]; then
  cp "$J/admission_trace.after.py" "$J/admission_trace.py"
fi
PYTHONPATH="$S/tests/pypto_test" python3 "$J/prepare_generation_reference.py"
```

必须完整640/384请求、192 token、零抢占且资格通过。
参考窗口从全部rank最后实际生成轮推导，不硬编码51/78；窗口内idle保留。
只对请求结束后的严格空尾独立审核，保存原始raw以及派生SHA链，不篡改原始complete轮数。

### 6.2 Native模块见证和重复资格

```bash
TASK_N2=$(task-submit --ptoas 0.63 --device auto --device-num 16 \
  --max-time 780 --env PYTHONNOUSERSITE=1 \
  "bash $J/run-case.sh native-witness $RUN-$CASE-native2")
printf '%s\n' "$TASK_N2"
task-submit --status "$TASK_N2"
```

等待exit0后，在仓库根目录运行：

```bash
PYTHONPATH="$S/tests/pypto_test" python3 tests/pypto_test/audit_tnd_precision_cohort.py \
  --job "$J" --phase native --left "$N1" --right "$N2" \
  --output "$J/native-repeat-reproduction.json"
```

该命令已经在两档历史数据上验证。必须numeric_pass=true，完整输入/token/事件重复一致且rank0 outer逐bit一致。
Native2必须保存完整有效行`.pt`与stage摘要；小证据包不包含这些大文件，不能在本机用空文件冒充通过。

### 6.3 CSA见证及数值/缓存两种资格

```bash
TASK_C=$(task-submit --ptoas 0.63 --device auto --device-num 16 \
  --max-time 780 --env PYTHONNOUSERSITE=1 \
  "bash $J/run-case.sh precision-witness $RUN-$CASE-precision")
printf '%s\n' "$TASK_C"
task-submit --status "$TASK_C"
```

等待exit0后：

```bash
PYTHONPATH="$S/tests/pypto_test" python3 tests/pypto_test/audit_tnd_precision_cohort.py \
  --job "$J" --phase csa --left "$N2" --right "$C" \
  --output "$J/csa-reproduction.json"
```

只允许把精确匹配的`rankN: READY reuse failed`单列为strict失败；其他错误立即中止。
数值资格仍核验源身份、全输入、输出、事件、模块、实际变体及图；全部rank缓存bypass/invalid/storage/forced必须0。
新统一入口已在两档历史CSA数据验证numeric_pass=true、strict_pass=false；
不能直接使用128K历史`qualify_numerics.py`作为未来入口，因为它写死了“预期READY失败”，暖缓存成功会被拒绝。

另核验全部rank的CSA选择和实际模型FULL replay（在仓库根目录）：

```bash
BUCKET=240
if [ "$CASE" = 128k ]; then BUCKET=144; fi
FIRST=$(python3 - "$N2" <<'PY'
import json,sys
from pathlib import Path
print(json.loads((Path(sys.argv[1])/'input-trace/rank0.trace.jsonl').read_text().splitlines()[0])['actual_tokens'])
PY
)
python3 "$E/8k/helpers/audit_selection.py" "$C" --bucket "$BUCKET" \
  --first-tokens "$FIRST" --output "$J/selection-reproduction.json"
```

选择计数是warmup/capture/Python资格入口加模型replay证据，不是逐层运行时replay hook。
未来不要求构建rank、空尾数量、总轮数、耗时与本次完全一样；要求完整生成窗口三臂对齐、数值和所有门禁成立。
若新运行首次JIT，保留strict失败并另核验本次实际binary manifest对应的产物身份；
归档中的`audit_artifact.py`绑定历史随机build目录，不能原样用它给新的产物背书。

## 7. 首差、失败历史及后续记录习惯

首差时保留`full-probe/STOP.json`、各rank input STOP、首差step原值及日志；
先确认相同输入/入场，再按最早模块对照Native修正，不继续其他大配置。
本次两档均没有数值首差；旧128K观测空尾修复及Indexer数学修复分属不同问题，见详细历史报告。

短档最初24000 blocks组在生成前仅剩1.756GiB，被2GiB预留门禁拒绝；
后来三臂共同改22000并全部重跑，未复用不同容量的Native黄金，未弱化内存门禁。
128K与8K首次JIT分别约16.53/15.75秒，strict READY失败永久保留；不能宣称本次全暖缓存或性能通过。

每次后续测试都增加独立记录，至少保存：生产commit、诊断脚本SHA、完整环境/输入身份、任务命令与ID、
实际占卡时间、提交/实际并发、完整token/events/轨迹及首差、模块摘要范围、strict/numeric分开结果、原始证据SHA。
写清与上一commit的变化；原始目录和失败证据不覆盖，临时大tensor不入Git。
将新报告、必要复现脚本及索引正常commit/push；不得把这份历史PASS自动归给以后改动的代码。
