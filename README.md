# Wiki 多跳检索 Agent

用公开 HotpotQA 的英文 Wikipedia 段落搭建本地搜索环境，研究小模型能否根据已找到的证据决定下一步查什么、何时停止。当前主线提供 Wiki 检索 API、可回放的模型评测和 ReOPD 后训练实验。Wiki 问答前端尚未接入。

## 数据与评测

| 数据 | 在项目中的用途 |
| --- | --- |
| HotpotQA `distractor/train` | 从公开题目抽取 **800 道**，分为 SFT 种子 300、策略训练与轨迹采集 300、dev 100、本地留出 test 100。题目、答案和支持句来自原数据。 |
| 英文 Wikipedia 候选段落 | 将候选段落和背景材料整理为 **20,000 个共享检索块**。所有题搜索同一语料库。正式实验使用 SQLite BM25，每次最多返回 3 块，每题最多搜索 6 次。 |
| HotpotQA 官方 `distractor/dev` 和 ReOPD `baohao/search-test` 的 2WikiMultiHopQA 来源题目 | 额外外部评测，用两批各 100 道 HotpotQA 题和 100 道 2Wiki 题，对比原 SFT 学生、定向 SFT 和定向 ReOPD。已完成题目准备与索引构建，整套评测尚未完成。 |

本地留出分数对应项目划分和固定语料库。公开数据扩展评测同样使用按来源构建的共享索引。数据版本、抽样和评分方式见 [HotpotQA 数据与检索](docs/hotpot-search.md)、[公开评测实现](hotpot_search/reopd_public_eval.py)和 [ReOPD 实验记录](docs/reopd-local.md)。

## 本地实验结果

从公开的 **Qwen3-4B 搜索 SFT** 权重初始化学生，使用公开的 **Qwen3-4B 搜索 SFT-RL** 权重作为冻结教师。ReOPD 回放已有的搜索历史，让学生生成下一步动作，由教师逐 token 提供蒸馏信号。定向组额外使用经教师续跑和人工复核的失败状态，目标是改善「未检索就答错」。四组各跑两个训练种子，每份 LoRA 更新 **12,000 个学生 token**。 [ReOPD 论文](https://arxiv.org/abs/2607.04763) · [实验设计](docs/reopd-execution-plan.md)

| 模型或训练方法 | 端到端有证答对 | 固定故障恢复 |
| --- | ---: | ---: |
| 原搜索 SFT 学生 | 27/100 | 0/24 |
| A · 定向继续 SFT | 27/100 | 2/24 |
| B · 教师前缀回放 ReOPD | 23/100 | 1/24 |
| C · 非定向混合 ReOPD | 25/100 | 1/24 |
| D · 定向恢复 ReOPD | 23/100 | 2/24 |

表中为冻结后的本地留出测试，主训练种子为 42。「有证答对」要求答案匹配且检索结果覆盖所需支持证据。D 和 A 都恢复了 2 道固定故障题，同预算下训练耗时分别约 5,317 秒和 92 秒。D 的端到端通过率比 A 低 4 个百分点，配对 95% 区间为 −12 至 +3 个百分点。这次实验没有证据表明定向 ReOPD 优于定向 SFT 或教师前缀回放。 [最终结果与复现](docs/reopd-local.md#最终结果2026-11-22)

## 在本机启动 Wiki 检索

以下命令适用于首次在 Windows PowerShell 配置环境。需要 `uv` 和 Python 3.13，首次下载 HotpotQA 约 316 MiB。已有 `.model-venv` 或本地数据时，跳过相应创建和准备步骤。

```powershell
uv venv .model-venv --python 3.13
uv pip install --python .model-venv/Scripts/python.exe -r requirements-hotpot.txt
$py = '.\.model-venv\Scripts\python.exe'

& $py -X utf8 -m hotpot_search.download
& $py -X utf8 -m hotpot_search.prepare
& $py -X utf8 -m hotpot_search.search build --bm25-only
.\Start-HotpotSearch.ps1
```

打开 [Wiki 检索 API](http://127.0.0.1:8020/docs)，以下命令会返回前三个检索结果。

```powershell
$body = @{query = 'Who directed the film Jaws?'; top_k = 3; mode = 'bm25'} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8020/search -Method Post -ContentType 'application/json' -Body $body
```

- `.\Start-HotpotSearch.ps1 -Training` 启动 [工具交互 API](http://127.0.0.1:8021/docs)，支持搜索、读文档、提交答案与评分。 [训练环境说明](docs/hotpot-training-environment.md)
- `& $py -m hotpot_search.reopd_run reference --run runs/reopd/reference-demo` 运行一条脚本化检索和回放验证。模型基线、LoRA 训练和留出评测的 CUDA 环境、固定权重与命令见 [实验复现](docs/reopd-local.md)。

下载的语料、构建的索引、模型权重和逐题运行产物在本地生成。评测脚本将模型可见的题目和用于评分的答案、支持句分开保存。

## 财报原型

仓库也保留独立的「财报证据台」。`setup.ps1` 和 `start.ps1` 用于财报原型，本地 Streamlit 页面在 [8501 端口](http://127.0.0.1:8501)。它支持上市公司报告问答、原文引用和轨迹展示。 [财报项目导读](项目通俗导读.md) · [财报 SFT 评测](evaluation/teacher-sft-epoch1-test-20261114/README.md)
