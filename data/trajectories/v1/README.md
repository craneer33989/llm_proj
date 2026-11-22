# 财报工具轨迹数据 v1

本目录提供 **700 条训练动作，来自 210 条完整任务轨迹**；另有 **80 道验证题、120 道测试题**。700 指下一步动作的监督样本数，不是 700 条完整多轮任务。所有任务从空证据开始，经过真实检索、必要计算，最后显式停止。

## 来源与划分

来源为巨潮资讯公开披露的 **54 份官方 PDF**：9 家公司各 2023、2024、2025 年的年报与半年报。相较一期新增 42 份 PDF、6 家公司，覆盖家电、动力电池、通信设备、汽车、电力、白酒、乳制品。`sources.json` 保存官方链接、公告编号、实际披露日期、文件大小和 SHA-256；PDF 存放于 `data/reports`。

| 集合 | 公司 | PDF 数 | 题数 | 动作数 |
|---|---|---:|---:|---:|
| train | 美的、格力、海尔、宁德时代、中兴通讯 | 30 | 210 | 700 |
| validation | 比亚迪、长江电力 | 12 | 80 | 270 |
| test | 贵州茅台、伊利股份 | 12 | 120 | 400 |

按公司先划分，之后生成题目；公司、报告、语义任务键和完整 prompt/completion 均检查交集。三个集合使用共同的推理任务族、不同的提示措辞，**不宣称题型模板完全隔离**。海尔曾用于二期诊断，本版纳入训练，因此旧海尔诊断成绩不能作为本版训练后的独立评测成绩。最终测试公司没有出现在旧版三家公司诊断中。

## 训练覆盖

| 任务 | 训练题数 | 训练动作数 | 参考轨迹 |
|---|---:|---:|---|
| 单跳查询并停止 | 50 | 100 | 检索 → 停止 |
| 同报告比例分析 | 30 | 90 | 检索 → 计算 → 停止 |
| 同报告可比列同比 | 20 | 60 | 检索 → 计算 → 停止 |
| 跨年度／跨报告差额及变化率 | 40 | 160 | 检索 A → 检索 B → 计算 → 停止 |
| 三份报告合计／均值 | 20 | 100 | 检索 A → B → C → 计算 → 停止 |
| 比较结果决定第三次检索 | 15 | 75 | 检索 A → B → argmax → 依结果检索 C → 停止 |
| 跨公司相同期间比较 | 20 | 80 | 检索 A → B → 计算 → 停止 |
| 截至日早于披露日 | 10 | 20 | 空检索 → 证据不足停止 |
| 全年／半年不能直接作同比 | 5 | 15 | 检索 A → B → 口径不可比停止 |
| 合计 | **210** | **700** | **345 次检索、145 次计算、210 次停止** |

依赖式多跳例子：比较某公司 2023 与 2024 年年报的收入，先通过计算工具确定较高的年度，再检索该年度半年报的现金流。第三次检索的年度必须和实际 argmax 观察一致。

## 文件与训练格式

- `train-actions.jsonl`：700 行，每行是一个模型决策时点，包含 `case_id`、`turn`、`group`、`prompt`、`completion`。监督损失只计算本行 `completion`，完整历史（包括此前 assistant 动作）都是输入。
- `train-questions.jsonl`：210 道问题、截至披露日、公司、标准结果、必要证据、计算依据、动作预算与参考动作。
- `train-trajectories.jsonl`：210 条完整的动作／工具观察／核验结果。
- `validation-*`、`test-*`：同样三种文件，分别用于模型选择与最终评测，不能并入训练。
- `fact-catalog.jsonl`：从 PDF 抽取的指标及分产品证据，保留原单位、原文片段、PDF 页码及校验信息。
- `corpus.sqlite3`：独立全文 FTS 索引；保留原文干扰项，不是只返回答案的模拟工具。文件超过 100 MB，不纳入 Git；可根据固定来源重新构建。
- `manifest.json`、`corpus-manifest.json`：规模、划分、文件与代码哈希、索引统计。
- `evaluation/`：完整参考轨迹回放结果与提前停止负对照结果。

协议名为 **`filing-tools-v1`**：模型自行给出检索公司、年度、报告期和关键词，计算工具只接受已经观察到的证据引用。最终动作必须声明 `answered`、`insufficient` 或 `incomparable`；可回答的计算题还须引用正确的计算结果。

输入只包含工具协议、问题、公司标识、截至日和已经执行的工具观察。**不会输入标准答案、参考动作、未来检索结果或“仍需查找”的关键词提示。** 问题文件保留 gold 字段供离线评分，`initial_messages()` 会显式筛选允许输入的字段。

工具观察的原文摘录最多 320 字符，完整正文仍在 SQLite 中；结构化数值字段全部保留，官方 URL 按报告去重后单独列出。该约定同时用于训练回放和模型评测，避免因重复长摘录挤掉早期证据。`token-statistics.json` 用固定 Qwen 分词器记录实际序列长度。

这批数据使用更完整的新协议。旧 `sft_pilot.py` 的两跳、48-token 查询格式不能直接用于本数据；下一轮 SFT 应读取本目录的动作文件、保留足够上下文，并对 completion 单独计算损失。此交付没有启动新模型训练，也没有替换页面正在使用的规划器。

## 生成与验证

从项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.trajectory_sources
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.trajectory_corpus
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.build_trajectories
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.evaluate_trajectories --split validation --policy reference
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.evaluate_trajectories --split test --policy reference
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.evaluate_trajectories --split test --policy early-stop
.\.venv\Scripts\python.exe -X utf8 -m pytest -q -p no:cacheprovider
# 使用已安装 Transformers 的模型环境测量真实长度；不加载模型权重、不训练。
.\.model-venv\Scripts\python.exe -X utf8 -m finance_agent.inspect_trajectories
```

来源获取需要网络；后续步骤使用本地固定文件。索引重建会独立核对 PDF SHA-256；一期 36 条指标的当期、同期和披露同比均与旧索引交叉核对。新增指标核对原文单位、调整后列与披露同比；提取不明确、同比不一致或检索无法命中的样本会使构建失败，不会用重复问法补足配额。均值与比率用 Decimal 计算、按两位小数四舍五入。非正同期基数保留为原始事实，不生成常规同比训练题。

## 完整模型评测

`reference` 是数据与工具回放验证，**不是模型成绩**。`early-stop` 在首次检索后直接停止，用于检查评测是否能检出缺失证据。两者均不进行模型推理。

实际基础模型／适配器评测可使用本地模型环境运行：

```powershell
$snapshot = '.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
.\.model-venv\Scripts\python.exe -X utf8 -m finance_agent.evaluate_trajectories --policy model --split validation --snapshot $snapshot --output data/runtime/trajectory-base-validation.json
# 适配器训练完成、确定超参后，再比较最终测试集；将下面路径换成新协议适配器实际目录。
.\.model-venv\Scripts\python.exe -X utf8 -m finance_agent.evaluate_trajectories --policy model --split test --snapshot $snapshot --adapter <adapter-directory> --output data/runtime/trajectory-sft-test.json
```

评测按模型自己的动作进行完整 rollout：不补写后续查询、不规则兜底、不把生成失败转成参考答案。严格检查动作 JSON、数值和单位、全部必要引用、计算来源、是否提前停止、证据足够后的额外检索及动作预算；另外输出各任务通过率和平均检索次数。最大生成长度为 384 token，超长上下文会报错，不会默默截掉证据。

## 适用范围

数据由确定性程序和专家规则构造，未调用大模型生成 gold，也不是独立人工标注集。数值任务目前围绕营业收入、归母净利润、经营现金流三个指标；索引里的分产品表尚未生成本版监督题。跨报告原披露值可能受重述影响，问题明确要求比较原披露值，并不把结果冒称同口径披露同比。54 份 PDF 和更多动作不能替代新任务与人工抽查；结果只应解释为这些任务族、来源和工具协议上的能力。
