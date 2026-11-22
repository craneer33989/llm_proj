# DeepSeek 第一轮训练数据交付（2026-11-13）

已生成并独立回放 **863 条完整教师轨迹、3,156 条动作监督样本**。每条发布轨迹都在真实财报检索／计算环境中重新执行，并检查条件检索顺序、最终答案、引用、预算和留出报告隔离。生成时未向教师提供标准答案或参考路径。

## 直接用于 SFT 的文件

- `train-actions.jsonl`：每行是一个决策时点，包含完整 `prompt` 和单步 `completion`，优先使用此文件。只对 completion 和结束 token 计算损失，prompt 全部遮罩。
- `train-trajectories.jsonl`：完整 messages 与最终答案，适合支持多轮监督的训练器。不要与逐步版本同时混入训练而重复计权。
- `manifest.json`：数据量、题型、教师来源、事实覆盖、哈希与验证边界。
- `quarantine.json`：独立回放额外隔离的候选，不在发布训练文件中。
- `collection-summary.json`：采集终止原因、公开种子数量与费用估算。
- `token-statistics.json`：本地 Qwen3-1.7B 分词器实测：最长 7,017 token，P95 为 5,234；最大 completion 为 281 token，没有超过 8,192 的样本。只检查了分词长度，未测试新训练显存或模型效果。

训练动作包含 `task_weight=1/该任务动作数`，供需要按任务平衡的采样器使用；训练器是否采用该字段要明确配置。此交付未启动模型训练。旧 `sft_pilot.py` 的 48-token 查询格式不能直接使用本批完整工具轨迹。

## 题型覆盖

| 题型 | 完整教师轨迹 |
|---|---:|
| 条件依赖多跳 | 39 |
| 跨公司比较 | 298 |
| 跨报告变化 | 148 |
| 披露截止日／不可回答 | 90 |
| 单跳查询 | 60 |
| 口径不可比 | 41 |
| 比例分析 | 59 |
| 三份报告汇总 | 60 |
| 同比计算 | 68 |

教师来源：Pro 177 条，Flash 686 条。覆盖 5 家训练公司、30 份财报、90 个已核对的结构化事实，指标为营业收入、归母净利润、经营现金流。1,020 个任务是同一资料池的不同任务组合，不等于 1,020 份独立事实。

## 原始轨迹、答案和失败记录

原始数据位于 `../../runtime/deepseek-auto-20261113-v2/`：

- `01-pro-pilot/`：45 条 Pro 小试，43 条通过原评分器。
- `02-flash-bulk/`：975 条 Flash 尝试，687 条通过原评分器。
- `03-pro-rescue/`：已保存的 Pro 补救轨迹与接口错误。
- `answers-all-attempts.jsonl`：主采集批次 1,198 次尝试的答案、标准答案与评分，包含失败和接口错误；这是审计文件，不是 SFT 输入。
- 各阶段的 `events/`、`episodes/`：真实请求正文、原始响应、思考内容、工具观察和用量。不包含 Authorization 请求头。

此前的协议试跑单独保留在 `../../runtime/deepseek-auto-20261113-live/`，没有混入本次正式 SFT。程序参考动作也没有混入教师 SFT。

## 补采与接口状态

接口最终返回 [HTTP 402：余额不足](https://api-docs.deepseek.com/quick_start/error_codes/)，采集进程已停止，未自动充值。全部 1,020 个训练任务均已至少尝试一次，后续 Pro 补救未全部完成。

`pending-questions.jsonl` 保存尚无可发布教师轨迹的 **157 道题**；`pending-reasons.jsonl` 区分模型失败、接口错误与独立审计不通过。两者只用于补采／诊断，不能作为 SFT 输入。

充值后可在项目根目录启动一个新的补采目录，本次没有执行下面的命令：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.deepseek_trajectories --questions data/sft/deepseek-20261113/pending-questions.jsonl --model deepseek-v4-pro --samples 1 --workers 2 --output data/runtime/deepseek-pending-next
```

费用字段是按峰时、缓存未命中价格计算的保守估算，且对未报告用量的失败请求保留预留值，不等于实际账单。主要批次已报告 token 的峰时估算为 $5.2696；连同之前试跑及未报告预留的保守合计为 $9.1641，未超出本次声明的 $10 估算边界。

## 后续 RL 可复用的任务

`../../rl/financial-tasks-20261113/train-tasks.jsonl` 提供 **1,020 道已回放验证的训练任务**，含评判所需标准答案、计算依据和预算，移除了 oracle_actions。调用 `initial_messages(case)` 构造模型输入，expected／calculation 仅交给评判器。这是现有环境的任务格式，尚未提供某个 PPO／GRPO 训练框架的专用适配器，也没有生成学生在线 rollout。

原有验证集 80 题、测试集 120 题保留在 `../../trajectories/v1/`，全部受保护文件哈希保持不变；未混入训练。

## 额外取得的公开种子

原始文件、固定上游提交、许可证和 SHA-256 位于 `../../external/public-finance-20261113/`：

| 来源 | 数量 | 本地用途状态 |
|---|---:|---|
| [FinQA](https://github.com/czyssrs/FinQA) train | 6,251 道题 | 有原始答案与计算程序，尚未适配本地工具协议 |
| [TAT-QA](https://github.com/NExTplusplus/TAT-QA) train | 13,215 道题 | 表文问答种子，尚未适配本地工具协议 |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 11,300 条上游合成对话 | 跨领域工具种子，本机尚未执行验证 |

这些种子没有并入上述 863 条已验证教师轨迹，数量不能混为一谈。FinQA／TAT-QA 仅下载了 train 划分。

## 验证边界

已完成真实 API 生成、工具独立回放、条件依赖检查、数值与引用核对、留出集哈希检查和 93 项代码测试。没有进行全量人工标注审查，也未证明模型训练后的泛化效果。当前仍是三项财务指标、九类共同任务模板上的第一轮数据；后续应扩指标、报告和表达方式，并通过独立评测验证收益。

密钥检查：已扫描本任务所有采集与发布产物，未发现原始 API Key；请求日志不保存 Authorization 请求头。详细记录见 `privacy-check.json`。
