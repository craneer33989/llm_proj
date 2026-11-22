# DeepSeek 教师轨迹采集

`finance_agent.deepseek_trajectories` 批量调用 DeepSeek API，让教师在项目的真实财报检索／计算环境中自主执行。输出每次尝试的完整轨迹、结构化最终答案、原始 API 响应、评分和可训练样本。无需新增依赖，使用项目的 `.venv` 即可；无需启动本地 Qwen 服务。

## 正确答案何时使用

第一版采用 **盲跑后筛选**：生成时只提供问题、公司、截止日期、工具说明，以及实际执行产生的观察。`expected`、`oracle_actions`、参考计算和缺失证据提示均不发送给教师。生成后由本地 `score_episode` 使用标准答案和证据校验。

已知正确答案再生成过程，可以用于另一批“答案条件化／纠错”数据，但它测量的是已知目标下的解题能力，容易出现倒推、编造路径。若以后采用，应单独保存并标记 `gold_visible=true`，工具仍然真实执行，并与盲跑数据、盲跑成功率分开。本脚本固定 `gold_visible=false`，不提供注入答案选项。

## 运行

在项目根目录 PowerShell 执行。先验证数据并查看任务数，不需要密钥、不联网，也不创建输出目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.deepseek_trajectories --limit 20 --samples 3 --dry-run
```

在当前终端设置密钥，输入不会进入命令历史或回显：

```powershell
$env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'DeepSeek API Key' -AsSecureString)).Password
```

先用 20 道训练题，每题独立采样 3 次，并发 2 条轨迹：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.deepseek_trajectories --model deepseek-v4-pro --limit 20 --samples 3 --workers 2 --output data/runtime/deepseek-pilot
```

全训练集采集，使用新的输出目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.deepseek_trajectories --model deepseek-v4-pro --samples 3 --workers 2 --output data/runtime/deepseek-train
```

`--limit` 取筛选后的前 N 道题，未指定时使用全部题；当前训练池默认是 210 题，3 次采样对应 630 条候选轨迹。一条轨迹会发起多次 API 请求。每轮默认 `--max-tokens 8192`，包含模型思考输出；出现 `finish_reason=length` 会保留原始响应并拒收该轨迹，调整预算后换新目录运行。

其他常用参数：

| 参数 | 用法 |
|---|---|
| `--groups ratio conditional_bridge` | 只跑指定题型，未知题型会报错 |
| `--thinking disabled` | 关闭教师思考；默认 enabled |
| `--reasoning-effort high` | 思考力度，可选 low/high/max |
| `--timeout 180 --retries 3` | 每次 HTTPS 请求超时；网络错误、408、429 和 5xx 退避重试 |
| `--split validation` | 生成验证轨迹，只保存结果，不导出 SFT |
| `--split test` | 运行封存测试题；查看后须保持其评测用途，避免用于选训练样本或调整提示 |
| `--questions path/to/questions.jsonl` | 自定义同协议题池；需含题目、公司、截止日期、预算、split、expected 等字段，具体格式参考现有题池；`oracle_actions` 不必提供 |
| `--data data/trajectories/v1` | 题池对应的语料、来源和清单目录 |

## 输出与训练

输出目录包含：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 模型配置、题目 ID、输入及代码 SHA-256、盲跑标记；不含密钥 |
| `events/<任务哈希>-<采样序号>.jsonl` | 每次请求正文、原始响应、错误和最终结果，逐条刷盘；保留历次恢复的日志 |
| `episodes/<任务哈希>-<采样序号>.json` | 每次尝试的最新完整结果，原子保存；包含逐轮提示、响应、真实工具观察、最终答案、标准答案、评分和 usage |
| `trajectories.jsonl` | 已落盘尝试的汇总，成功和失败均保留 |
| `answers.jsonl` | 每次尝试的问题、结构化最终答案、标准答案与各项检查；未结束时 answer 为 null |
| `sft-actions.jsonl` | 仅训练集、通过评分且动作轨迹去重后的逐步 prompt/completion |
| `sft-trajectories.jsonl` | 同一批通过筛选的完整 messages，用于支持多轮监督的训练器 |
| `summary.json` | 各题型通过数量、首次采样成绩、多次尝试解出题数、SFT 数量和报告的 token 用量 |

训练时优先使用 `sft-actions.jsonl`，只对 `completion` 和结束 token 计算损失。`prompt` 全部遮罩，包括之前的 assistant 动作和工具观察。不要直接把带 `expected` 的审计结果喂给训练器。

教师的 `reasoning_content` 保存在原始响应中，但不会进入默认 SFT 导出；导出只监督搜索／计算／停止动作，与目前关闭思考的小模型协议一致。完整 messages 的训练需要训练器正确遮罩 user／system，避免与逐步样本同时混用而重复计权。

按同一问题的完整动作序列去重，不同有效路径可以同时保留。模型错误、格式错误、截断和预算耗尽不会被规则修补，也不会导出为成功 SFT；失败轨迹仍保留供分析。

评分沿用现有项目的严格标准，包括精确数值／单位、指定证据集合、已检索证据、参考计算和及时停止。因此 `passed=false` 不总等于财务结论错误，也可能是多余检索、引用集合或等价计算路径未通过；可结合子项人工复核。当前工具只覆盖已核对的结构化指标，本脚本不会自动扩充财报、事实字段或出新题。

## 恢复与费用统计

重复执行完全相同的命令会跳过已经完成的成功／模型失败尝试；API 错误尝试和未落盘尝试会重跑。一次未完成的轨迹从第一轮重新开始，不会从中间步骤继续，也不保证重跑请求不重复计费。历史请求／响应保留在 events 中。

恢复要求模型、采样次数、题目选择、生成设置、输入和代码指纹一致；改变配置或扩大题池时使用新目录。可调整 `--workers`。鉴权、余额及其他非重试 HTTP 错误会停止派发后续任务，正在运行的轨迹可能仍完成。

只重建汇总／SFT 文件、不访问 API 时，在原命令上添加 `--export-only`，保留其他配置；不需要密钥。

目录锁阻止两个进程同时写同一批结果。正常退出会移除锁；进程被强制结束时，确认旧进程已停止后，再手动删除输出目录中的 `run.lock`。Ctrl+C 会停止后续派发，并等待已在执行的任务退出后汇总；网络超时由 `--timeout` 控制。

`summary.json` 的 token 统计包含历史恢复日志中所有响应报告的用量；网络超时或未收到响应的请求可能已计费，无法计入已报告 token。`--dry-run` 输出的是含重试的请求／输出 token 上界，不是费用预测；实际费用使用服务商账单核对。

## API 协议

固定连接 `https://api.deepseek.com/chat/completions`。沿用 `filing-tools-v1` 的 JSON 动作协议，由本机执行工具并把真实结果以 user 消息回传，未使用原生 `tools` 字段。这能保持教师与学生可见输入和动作格式一致。

按 DeepSeek 文档，未携带 `tools` 时无须回传历史 `reasoning_content`；若未来改成原生函数调用，则需另行处理其思考内容回传规则。思考模式的 temperature 不生效，脚本没有设置它。模型别名可能随服务商更新，原始响应保留返回的 model 字段，但无法替服务商保证底层版本固定。

官方参考：[模型说明](https://api-docs.deepseek.com/quick_start/pricing/)、[JSON 输出](https://api-docs.deepseek.com/guides/json_mode/)、[思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)、[错误码](https://api-docs.deepseek.com/quick_start/error_codes/)。

## 有时间与费用边界的无人值守采集

`finance_agent.collect_sft` 在独立输出目录中完成：扩展并回放训练候选 → 45 道按题型平衡的 Pro 小试 → 其余题 Flash 单次生成 → Pro 补救模型失败题 → 合并合格教师 SFT。已有验证／测试文件的哈希会保留并检查，不会重建或覆盖原始题库。

```powershell
.\.venv\Scripts\python.exe -u -X utf8 -m finance_agent.collect_sft --api-key-file D:/work/api_key.txt --output data/runtime/my-new-collection --minutes 80 --max-usd 9.4 --workers 6
```

也可以继续使用进程环境中的 `DEEPSEEK_API_KEY`，省略 `--api-key-file`。文件模式只在内存中解析用户指定的文件，不把密钥复制到输出或打印出来。`Start-DeepSeekCollection.ps1 -ApiKeyFile D:/work/api_key.txt -OutputDirectory data/runtime/my-new-collection` 可从 PowerShell 隐藏启动采集进程。

输出目录必须没有既往教师尝试，避免重置历史费用。这个分阶段调度器目前要求新目录；单批采集的断点重跑仍使用前文 `deepseek_trajectories` 入口及其完全相同的配置。

- `task-pool/`：通过实际回放的候选题，以及明确标记为 `programmatic_reference` 的参考动作；参考动作不会混入教师 SFT。
- `01-pro-pilot/`、`02-flash-bulk/`、`03-pro-rescue/`：各阶段的原始事件、尝试、答案及评分。
- `sft/`：每个任务最多选择一条通过评分的教师轨迹，附带教师模型与阶段来源。
- `budget.json`：每次请求的保守费用估算与未报告用量的预留；不是服务商账单。
- `status.json`、`summary.json`：执行状态与合并统计。

费用守卫按已核实的峰时、缓存未命中单价估算；每次请求（包括重试）先预留最大输出及保守输入成本，再按响应报告的 token 释放预留。没有返回可用 usage 的请求继续占用预留。价格变化、未报告用量和服务商计费规则意味着它不是账单金额的保证；实际金额以服务商记录为准。

请求前检查截止时间，另有绝对时间看门狗，避免服务商持续发送 keep-alive 字节而使 socket 空闲超时一直不触发。到时保留已落盘结果并导出汇总。在输出目录创建空的 `stop.requested` 文件可让采集停止继续发起请求，保留正在收尾的尝试。

## 独立回放与发布训练文件

批量结束后再对合格轨迹重新执行环境，检查真实工具观察、最终答案、条件搜索对 argmax 的因果依赖，以及训练上下文没有读入留出的报告：

```powershell
.\.venv\Scripts\python.exe -u -X utf8 -m finance_agent.audit_teacher_sft --run data/runtime/my-new-collection --output data/sft/my-verified-teacher-release
```

输出 `train-actions.jsonl`、`train-trajectories.jsonl`、`manifest.json` 和 `quarantine.json`。没有通过独立回放的轨迹只记录在隔离文件中，不进入发布训练文件。动作行额外提供 `task_weight=1/该任务动作数`，供需要按任务平衡的采样器使用；训练器是否采用该字段需要明确配置。

如果较短的候选轨迹未通过独立回放，审计器会检查同题其他阶段的合格候选，选择通过真实回放的替代轨迹。`quarantined_candidates` 统计被隔离的候选数，`excluded_tasks` 统计最终没有合格轨迹的任务数。

使用本地学生分词器检查实际上下文和目标长度，无需加载模型权重：

```powershell
.\.model-venv\Scripts\python.exe -X utf8 -m finance_agent.inspect_teacher_sft --release data/sft/my-verified-teacher-release
```

## 获取公开训练种子

```powershell
.\.venv\Scripts\python.exe -u -X utf8 -m finance_agent.download_public_finance --output data/external/public-finance
```

该入口从 FinQA、TAT-QA 作者官方仓库固定 Git 提交，下载 **train 划分**、许可证、问题索引和 SHA-256 清单。不下载测试划分，也不向 GitHub 发送本地任务或密钥。原始问题／答案尚未适配当前工具协议，明确标记 `ready_for_current_sft=false`，不能当作已验证教师轨迹计数。

添加 `--toolace-only` 可单独取得作者官方 Hugging Face 仓库中的 ToolACE，固定提交并保留数据卡和 Apache-2.0 许可信息。这些是上游合成的跨领域工具对话，本机未重放，不并入当前财报教师 SFT。

## 教师 SFT 显存与速度试跑

新增入口 `finance_agent.sft_teacher`，直接读取已发布的 `train-actions.jsonl` 并核对哈希。它使用固定本地 Qwen3-1.7B 检查点新建 LoRA，支持最长 8,192 的输入；不截断样本，损失只计算本步 completion 和 EOS。此前的 48-token／短上下文脚本保持原样。

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.model-venv\Scripts\python.exe -u -X utf8 -m finance_agent.sft_teacher --steps 20 --accumulation 4 --selection mixed --output data/runtime/my-new-sft-benchmark
```

20 步指 20 次优化器更新，每步累积 4 个单样本微批，共 80 条动作。mixed 选择首先测试最长上下文，再覆盖题型、最长目标与随机样本，不重复抽取动作。速度统计默认排除前两步预热，并在计时前后同步 CUDA。`--selection random` 使用纯随机顺序；超过单次数据遍历的更新数会报错，不会偷偷重复样本。

输出配置、完整抽样清单、逐步 loss／时间／长度／显存、LoRA 和保存重载一致性校验。PyTorch 进程峰值与 CUDA mem_get_info 的数值分开标记；Windows WDDM 下不把后者当作 nvidia-smi 的整卡占用。

2026-11-13 的 [20 步实测报告](../data/runtime/sft-teacher-20step-20261113/README.md)：80 条动作，最长 7,017 token，训练 184.04 秒；稳态 7.37 秒／更新、1,292.7 总序列 token／秒；进程峰值分配 5.16 GiB、预留 5.97 GiB。当时按吞吐投影全量 1 epoch 的纯训练约 110 分钟。随后[完整一轮实训](../data/runtime/sft-teacher-epoch1-20261113/README.md)已完成：789 更新、3,156 条动作，有效更新累计约 87 分钟，全程约 99 分钟，峰值分配 5.23 GiB／预留 7.75 GiB。2026-11-14 又完成[测试集前后对照](../evaluation/teacher-sft-epoch1-test-20261114/README.md)：基座 2/120、最终 SFT 118/120；常用指标、测试 Loss/PPL 与曲线图均已保存，入口见[评测说明](teacher-evaluation.md)。

## 完整一轮训练与暂停恢复

本批 3,156 条动作按微批 1、累积 4，对应一轮 **789 次优化器更新**。正式运行使用纯随机顺序，从原始基座新建 LoRA；20 步测试适配器不作为训练起点。

```powershell
.\.model-venv\Scripts\python.exe -u -X utf8 -m finance_agent.sft_teacher --steps 789 --accumulation 4 --selection random --checkpoint-every 50 --output data/runtime/my-new-teacher-epoch
```

每 50 步及结束时保存 checkpoint，包括 LoRA、AdamW 优化器、CPU／CUDA／Python 随机状态、完成步数、数据顺序及逐步记录。checkpoint 先写临时目录，全部文件哈希写完后再原子提交并更新 `latest-checkpoint.json`。中断的 `.saving-*` 目录不会被恢复器选中。

在运行目录创建 `pause.requested`，会在当前完整优化器更新结束后保存断点并退出，释放本训练进程显存。恢复时移除标记，重复原命令并增加 `--resume`；数据哈希、配置、参数顺序和抽样顺序必须一致。

针对本次正式运行，可在项目根目录使用：

```powershell
.\Control-TeacherTraining.ps1 -Action Status
.\Control-TeacherTraining.ps1 -Action Pause
.\Control-TeacherTraining.ps1 -Action Resume
```

每行分别是独立操作，不要连续执行三行。Resume 会从 `config.json` 读取原参数，用隐藏进程恢复并另存 stdout／stderr；运行锁存在时拒绝重复启动。强制关闭进程只保证能恢复最近已提交的 checkpoint，可能重跑之后未保存的更新。恢复时将未提交的旧日志尾部归档，再按已提交断点重建 `steps.jsonl` 后继续追加；最终 `report.json` 的 steps 为权威训练记录。上述正式运行现已完成，再次训练须指定新的输出目录。
