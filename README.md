# Wiki 多跳检索 Agent

用公开 HotpotQA 的英文 Wikipedia 段落搭建本地搜索环境，研究小模型能否根据已找到的证据决定下一步查什么、何时停止。当前主线提供 Wiki 检索 API、可回放的模型评测、ReOPD 后训练实验和本地展示页面。

## 数据与评测

| 数据 | 在项目中的用途 |
| --- | --- |
| HotpotQA `distractor/train` | 从公开题目抽取 **800 道**，分为 SFT 种子 300、策略训练与轨迹采集 300、dev 100、本地留出 test 100。题目、答案和支持句来自原数据。 |
| 英文 Wikipedia 候选段落 | 将候选段落和背景材料整理为 **20,000 个共享检索块**。所有题搜索同一语料库。正式实验使用 SQLite BM25，每次最多返回 3 块，每题最多搜索 6 次。 |
| HotpotQA 官方 `distractor/dev` 和 ReOPD `baohao/search-test` 的 2WikiMultiHopQA 来源题目 | 已完成首批 HotpotQA 和 2Wiki 各 100 道题的配对评测。另备有第二批 100 道 HotpotQA 题。 |

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

同一批本地 100 题还用于通用基座、混合检索及早答复核的事后分析。完成的 2Wiki 批次上，A 的严格有证答对从原搜索 SFT 的 26/100 提升到 37/100。 [课程实验结果](docs/course-results.md)

![本地 100 题通过与失败分布](docs/figures/failure-attribution.png)

## 在本机启动 Wiki 检索

### 展示页面

已有语料索引、模型权重和 `.model-venv` 时，在项目根目录运行：

```powershell
.\.model-venv\Scripts\python.exe -m streamlit run hotpot_search/demo_ui.py --server.address 127.0.0.1 --server.port 8502
```

打开 [Wiki 多跳检索台](http://127.0.0.1:8502)。“模型表现对比”读取已保存的结果，可筛选同题得失、并排回放轨迹和核对标注支持句。“输入问题”可用 4B 模型实时检索，展示实际生成的搜索动作与答案。加载权重前要求至少 12 GiB 空闲显存和 8 GiB 空闲系统内存；A–D 还需要本地 LoRA 适配器。

“早答复核路由”先运行原搜索 SFT。若它零检索就正常提交答案，再启用 A 适配器从原题复核。点击“预载原搜索 SFT 与 A”可在本机常驻一份底座，复核仍要进行第二轮推理。页面显示实际耗时。四批同题轨迹可用 `.\.model-venv\Scripts\python.exe -m hotpot_search.routed_eval` 复算。 [路由效果报告](docs/routed-demo.md)

### 检索 API 与首次配置

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

## 课程展示补充评测

课程评测复用已冻结的 100 道本地题，补跑通用 Qwen3-4B 的 BM25 对照，以及定向 SFT A 的 Hybrid RRF 检索对照。命令需在仓库根目录执行，依赖已完成的冻结评测、本地权重和包含 Qdrant 的索引。已有混合索引可直接使用。

```powershell
$py = '.\.model-venv\Scripts\python.exe'
& $py -X utf8 -m scripts.course_eval base-bm25
& $py -X utf8 -m scripts.course_eval a-hybrid
& $py -X utf8 -m scripts.course_eval compare
& $py -X utf8 -m scripts.analyze_course_metrics --require-controls
& $py -X utf8 -m scripts.plot_course_figures
& $py -X utf8 -m pytest -q
```

补充运行写入 `runs/reopd/course-extra/`。按相同命令可继续未跑完的批次。并行的本地服务需要释放 Qdrant 索引后才能运行混合组。冻结成绩及展示解释见 [课程实验结果](docs/course-results.md)。
