# HotpotQA 本地数据与搜索环境

这是独立于 `finance_agent` 的英文多跳搜索环境。当前完成的是原始数据下载、训练种子筛选、共享语料、持久化检索和工具环境，**尚未生成 DeepSeek 示范，尚未执行 SFT 或 RL**。

已补充可供采集器/训练器调用的 reset/step/finish 闭环与独立终止评分，启动 `.\Start-HotpotSearch.ps1 -Training`，详见 [训练搜索环境](hotpot-training-environment.md)。它复用本页的检索服务，模型通过文本/JSON 交互。

## 数据与边界

- 下载 Hugging Face `hotpotqa/hotpot_qa` 的 `distractor/train` 两个完整 Parquet 分片，共 90,447 道原始训练题，约 316 MiB；固定 revision 和 SHA-256 见 `data/hotpotqa/raw/download_manifest.json`。
- 过滤 22 道支持句缺失或越界的记录；按类型和原始难度分层、固定随机种子 `20261116`。
- `sft_seed.jsonl` 300 题：80% bridge，20% comparison；easy/medium/hard 各 100 题。
- `rl.jsonl` 300 题：同上；`dev.jsonl` 100 题；`test.jsonl` 100 题。
- 四个文件仅包含 `id` 和 `question`。答案和证据在 `private/*.gold.jsonl`，搜索服务不读取这些文件。`private` 是程序角色隔离，不是操作系统安全边界；以后不要把整个数据目录挂载给有文件工具的解题 Agent。
- 所有题均来自官方 train；dev/test 是本项目的留出划分，不是官方 dev/test，不提交为官方成绩。
- 划分之间问题去重，核心支持标题不重叠；其他提及实体和普通候选材料可能重叠，不能声称完全实体隔离。
- 汇集所选题的**全部候选上下文**，再加入按固定种子选出的背景材料。最终为 17,585 个原始段落、20,000 个检索块。每道题搜索同一个库，不按题目 ID 缩小候选范围。
- 1,895 条标注支持事实全部保留，长句跨块时保留句子索引映射。完整性、文件校验值和划分分布见 `manifest.json`。
- 当前是原始英文 Wikipedia **段落**，并非完整文章、完整 Wikipedia、实时互联网或旅游知识库；部分原始题可以走捷径，bridge 标签不保证一定需要两次调用。
- 完整性检查不等于所有标准答案已经人工复核，也不证明必须多步搜索。生成老师轨迹前仍需小批量验证答案、证据语义和搜索必要性。

## 搜索实现

- BM25：SQLite FTS5，标题权重 3、正文权重 1。
- Dense：`sentence-transformers/all-MiniLM-L6-v2`，固定 revision，384 维，attention-mask mean pooling、L2 归一化，Qdrant local 持久化；元数据保存在 SQLite。
- Hybrid：两路各取前 50 名，用 RRF（k=60，等权）融合。
- 默认 BM25：在本项目 100 道 dev 上，一次原问题检索的支持文档 Recall@10 为 BM25 87.5%、dense 75.0%、hybrid 84.0%；找到全部支持文档的题目比例分别为 75%、53%、69%。这是检索诊断，不是模型任务准确率。当前混合配置并未优于 BM25。
- 20,000 段适合单机起步。Qdrant local 在这个客户端设置下是本地持久化检索，不应宣传为分布式服务或生产吞吐架构。索引构建、离线审计和服务不要同时打开同一 Qdrant 目录。
- 编码模型为轻量英文起点，没有选用更大的 BGE-M3；后续中文旅游可比较多语言模型，必须重建索引并记录新版本。
- `index/manifest.json` 记录语料哈希、向量模型及权重哈希；修改语料后拒绝沿用旧索引。

## 当前机器启动

在 `D:\work\PostTraining` 执行：

```powershell
.\Start-HotpotSearch.ps1
# 查看 http://127.0.0.1:8020/docs
Invoke-RestMethod http://127.0.0.1:8020/health

$body = @{query='Moonshine Music'; top_k=5; mode='bm25'} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8020/search -ContentType 'application/json' -Body $body

.\Start-HotpotSearch.ps1 -Stop
```

前台运行、查看详细日志：

```powershell
.\Start-HotpotSearch.ps1 -Foreground
```

只绑定 `127.0.0.1`，默认端口 8020，与已有财报服务分开。后台日志位于 `data/hotpotqa/runtime/`。单进程加载模型，以免重复占显存。首次 dense 请求会加载编码器；第一次延迟不代表稳态延迟。

## API 和未来老师模型接入

- `POST /search`：`{query, top_k, mode}`，mode 为 `bm25/dense/hybrid`，返回块 ID、文档 ID、标题、正文、原始百科 URL、分数。
- `GET /documents/{document_id}`：返回原始来源段落全文，不会访问实时网页。
- `GET /tools`：老师模型可使用的 `local_search`、`read_document` 函数定义。
- `POST /retrieve`：批量检索，接收 `{queries, topk, return_scores, mode}`，提供 Search-R1 风格的结果包装；未来接具体框架时仍需核对版本与适配。
- `hotpot_search.environment.SearchSession`：每题默认 6 次工具调用；非法调用也消耗预算；只允许读取已搜索到的文档，可保存 JSONL 工具日志。
- 搜索默认 BM25。可显式选择 dense/hybrid 比较，老师与学生对照实验必须固定同一模式。
- 服务没有答案或打分接口。未来答案评测器必须单独读取 gold，不能把 gold 传给老师或学生模型。

## 重建与复核

当前复用已有环境：`.venv` 有 PyArrow 和 pytest；`.model-venv` 有 PyTorch、Transformers、FastAPI、Qdrant。新环境可安装 `requirements-hotpot.txt`，其中 CUDA PyTorch 需按机器选择正确构建。

```powershell
# 首次需联网；完整训练集约 316 MiB，编码模型约 87 MiB。
python -X utf8 -m hotpot_search.download
.\.model-venv\Scripts\python.exe -X utf8 -m hotpot_search.download_encoder

# 已有 v1 不会被静默覆盖；重建使用新输出目录。
.\.venv\Scripts\python.exe -X utf8 -m hotpot_search.prepare --output data/hotpotqa/v2
.\.model-venv\Scripts\python.exe -X utf8 -m hotpot_search.search build --data data/hotpotqa/v2

# 默认审计 v1；先停止服务，避免 Qdrant local 文件锁冲突。
.\.model-venv\Scripts\python.exe -X utf8 -m hotpot_search.audit --limit 100
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_hotpot_search.py -q -p no:cacheprovider
```

原始下载与编码模型下载均记录版本并校验权重/数据，不关闭 HTTPS 证书校验。普通重复运行会复用校验正确的原始分片。

## 怎样解读检索报告

`retrieval_audit.json` 只用本地 dev，不用 test 来调参：

- `integrity`：逐题核对证据、原始句子、块映射、文件哈希与划分隔离。
- `original_question_support_document_recall`：只拿原问题查一次，在 Top 10 找到了多少必要来源段落。
- `original_question_all_support_documents_rate`：一次检索是否找到全部必要来源。
- `oracle_title_probe_recall_at_10`：用标准支持标题作查询，诊断语料与检索器。这使用了标注，**不是自主多跳搜索，也不是模型成绩**。
- 延迟是本机、预热后、串行诊断条件的结果，不能当作生产并发 SLA。
- `service_smoke.json` 保存 HTTP 三模式检索、文档读取、批量检索和非法输入检查结果；`two_step_smoke.json` 是人工阅读第一步结果后选择第二次查询的真实接口演示，**不是老师模型轨迹或模型准确率**。

后续先用 DeepSeek 在这套真实工具上跑 20 题，检查证据和搜索行为，再扩充示范、SFT、RL。本地训练与真实联网迁移应分开测试，不能据此宣称旅游能力已经提高。

## 来源与许可

HotpotQA 及其派生语料：CC BY-SA 4.0，保留同许可与署名要求。

- https://hotpotqa.github.io/
- https://huggingface.co/datasets/hotpotqa/hotpot_qa
- Zhilin Yang et al. **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering**, EMNLP 2018.
- https://creativecommons.org/licenses/by-sa/4.0/

MiniLM 编码模型：Apache-2.0，来源和固定版本见 `.cache/hotpot-encoder/encoder_manifest.json`。
