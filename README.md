# 财报证据台

独立新增的 HotpotQA 数据与本地搜索环境见 [HotpotQA 搜索说明](docs/hotpot-search.md)：800 道训练/留出题、20,000 个共享检索块，支持 BM25、Qdrant 向量与混合检索，启动命令为 `.\Start-HotpotSearch.ps1`。尚未生成教师轨迹或进行该任务的 SFT/RL。

训练用 reset/step/finish、预算控制、终止奖励和轨迹日志见 [训练搜索环境](docs/hotpot-training-environment.md)，启动 `.\Start-HotpotSearch.ps1 -Training`，本机接口端口 8021。

定向 ReOPD 已完成本地四组双种子实验和冻结留出评测；在等 token 预算下，未证明定向 ReOPD 优于定向 SFT。见 [结果与复现](docs/reopd-local.md)。

一期面向美的集团、格力电器、海尔智家，收录三家公司 2024、2025 年的年报与半年报全文。可以查询营业收入、归母净利润、经营活动现金流量净额及分产品收入。跨报告问题由搜索规划器选择查询，金额、同比及引用由程序根据查到的证据生成。

## 本机启动

需要 Windows、PowerShell、uv 和 Python 3.13。首次运行需要联网下载依赖、公开财报和模型。

```powershell
.\setup.ps1
.\start.ps1
```

- [问答页面](http://127.0.0.1:8501)
- [FastAPI 接口说明](http://127.0.0.1:8010/docs)
- [模型状态](http://127.0.0.1:8012/health)

服务仅绑定 127.0.0.1。停止时运行 `.\start.ps1 -Stop`。重启沿用已经下载的财报、索引和模型。`data/runtime` 保存日志、SQLite 索引及逐题评测轨迹。

模型是 `Qwen/Qwen3-1.7B` 后训练版，revision 为 `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。在 CUDA 可用时用 BF16 推理，关闭思考模式。`.\start.ps1 -RulesOnly` 启动可复现的规则演示，用于验证数据、页面和接口。页面侧栏标注当前规划方式。

## 问一个问题

页面默认打开「训练与评测」，可选择最新的无下一步提示实验或历史 10 步试跑。可查看逐步 loss、同题训练前后通过率、主动发起第二跳和命中目标页的次数、停止和证据缺失场景、训练显存及适配器重载校验。下方案例回放展示模型原话、搜索词和命中文档，可导出前后记录。`GET /experiments/sft-autonomous` 返回最新实验，`GET /experiments/sft-pilot` 返回历史实验。

任务指标按各自题型显示分母，7 道单跳与 10 道跨报告题分别统计。loss 来自每步不同的 4 条训练动作，用于观察训练拟合；跨报告题共享海尔的两张产品表，且程序提供了检索线索，后续需要在更广的任务上继续验证。

在左侧切换「财报问答」，选择美的集团，查询截止日期设为 2025-08-30 或之后，输入「2024 年年报中收入最高的业务板块，在 2025 年半年报的营业收入和同比变化是多少？」。页面显示答案和计算过程，并分别链接到确定板块的年报及用于计算的半年报。点击「导出查询和检索轨迹」可保存逐轮查询、命中来源和模型动作。

## 重建与评测

```powershell
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.filings
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.corpus
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.evaluate --planner model
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.evaluate --planner rules
```

`data/filings.json` 记录 12 份官方 PDF 的链接、披露日和 SHA-256。`evaluation/questions.jsonl` 包含 30 道基线诊断题和预期出处。评测报告在 `data/runtime/baseline-model.json` 或 `data/runtime/baseline-rules.json`。单次基线固定用原问题检索，多轮模式允许规划器发起最多两次检索，两者共用同一套确定性回答逻辑。规则演示用于检查链路，模型报告单独统计。

`/ask` 接收 `question`、`company_code`、`as_of` 和 `mode`，返回 `status`、`citations`、`values` 和 `trace`。`/retrieve` 接收 `queries`、`topk` 和 `return_scores`，可附加公司、截止日期、报告年度和类型，返回与 Search-R1 兼容的批量结果。

当前问答默认使用模型自主检索，模型自行生成查询词、报告范围和结束状态，最多搜索两次。工具仅按用户指定的公司和截至披露日过滤，提示中没有下一步查询线索。历史实验保存了此前的强提示与规则补查成绩。

本机首次完整评测中，单次检索通过 25/30，Qwen 原始多轮通过 15/30，证据补查模式通过 30/30。原始多轮在 12 道桥接题上均于第一跳后提前停止。补查成绩属于系统整体，逐题报告保留模型原始动作和补查原因。

结构化数值仅覆盖已经核对的字段与表格。同一期同比优先使用该报告披露的同期列，包括明确标注的调整后数据。分产品表里没有同名项目时返回口径不可比。报告语料固定为清单中的版本，新披露的报告或更正公告需要更新清单并重建索引。

## 二期数据准备

已从一期原始多轮日志重放 15 条失败，导出 13 条 SFT 种子和 2 条分公司隔离的诊断样本。重建命令、文件用途与使用边界见[二期 SFT 种子说明](data/phase2/README.md)。

## 扩充工具轨迹与独立评测

新增 [财报工具轨迹 v1](data/trajectories/v1/README.md)：54 份官方 PDF、9 家公司、2023—2025 年年报与半年报，提供恰好 700 条训练动作（210 条完整任务轨迹）、80 道验证题和 120 道测试题。覆盖单跳停止、数值计算、跨文档／跨公司检索、三份报告汇总、比较结果驱动的后续检索、披露日期限制和不可比口径。

新轨迹使用独立的 `filing-tools-v1` 检索／计算／停止协议与评测入口；按公司和报告隔离数据，不给模型输入下一步关键词提示。详细数据格式、重建与完整 rollout 评测命令见目录说明。参考轨迹回放结果只验证数据与工具，不代表新模型训练成绩。

## DeepSeek 教师蒸馏数据

运行 `python -m finance_agent.deepseek_trajectories --limit 20 --samples 3 --dry-run` 可查看采集计划。设置 `DEEPSEEK_API_KEY` 后去掉 `--dry-run`，即可批量让教师使用真实检索／计算工具，逐题保存原始响应、完整轨迹、答案及校验结果。默认生成时不提供正确答案，生成后再评分；仅训练集的合格轨迹导出 SFT。支持并发、多次采样、去重与断点重跑，完整命令和文件格式见 [DeepSeek 轨迹采集说明](docs/deepseek-trajectories.md)。

无人值守入口 `finance_agent.collect_sft` 支持扩展并验证全部训练候选、Pro 小试、Flash 批量、Pro 补救、时间及保守费用边界；`finance_agent.audit_teacher_sft` 会在发布训练文件前独立重放，并检查条件依赖和留出报告隔离。公开 FinQA／TAT-QA 训练种子的下载入口是 `finance_agent.download_public_finance`，原始种子与教师 SFT 分开保存。

2026-11-13 已产出并独立回放验证 **863 条完整 DeepSeek 教师轨迹、3,156 条 SFT 动作样本**，交付文件、使用方式与边界见 [第一轮数据说明](data/sft/deepseek-20261113/README.md)。另有 1,020 道可复用的 RL 训练任务，以及分开保存的 FinQA、TAT-QA、ToolACE 公开种子。采集因接口返回 402（余额不足）停止，157 道待补采题已保存；未自动充值。

教师 SFT 新入口 `finance_agent.sft_teacher` 已完成 [20 步显存与速度测试](data/runtime/sft-teacher-20step-20261113/README.md)：实训 80 条动作、覆盖最长 7,017 token，进程峰值分配 5.16 GiB／预留 5.97 GiB，稳态 7.37 秒／更新。试跑适配器单独保存，尚未做任务效果评测。

随后已完成 [教师数据完整一轮 SFT](data/runtime/sft-teacher-epoch1-20261113/README.md)：3,156 条动作各训练一次，共 789 次更新；有效训练约 87 分钟、含暂停和恢复的全程约 99 分钟，训练进程峰值分配 5.23 GiB／预留 7.75 GiB。最终 LoRA、优化器断点及审计报告均已保存，保存重载校验完全一致。

2026-11-14 已完成 [120 题测试集前后对照与曲线图](evaluation/teacher-sft-epoch1-test-20261114/README.md)：基座完整任务成功率 2/120（1.67%），最终 SFT 为 118/120（98.33%），116 题改善、0 题退步。另提供 Accuracy、Precision／Recall／F1、答案 EM、引用及工具指标，以及相同 400 条测试参考动作的 Loss、PPL 和 Token Accuracy。该成绩衡量固定财报工具协议，训练与测试共享任务模板；基座大量失败与协议格式有关。复现入口见 [评测说明](docs/teacher-evaluation.md)。

训练流程现已整理为 [统一训练流水线](docs/training-pipeline.md)：`Run-Training.ps1` 按 JSON 配置分别执行数据校验与快照、BF16 LoRA SFT、分段保存／恢复、前后评测和图表报告。默认使用验证集；`All` 可自动完成全部阶段，人工 Pause 会在完整更新后保存退出。配置示例为 [teacher-sft.example.json](configs/teacher-sft.example.json)，历史实验可用 [completed 配置](configs/teacher-sft.completed-20261113.json) 只读重建报告。
