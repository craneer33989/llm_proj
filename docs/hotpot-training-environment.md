# HotpotQA 训练搜索环境

这个环境服务于后续教师轨迹采集和学生模型的 SFT/RL：从问题开始，允许模型自主调用真实检索工具，最后提交短答案与引用，由独立评分模块计算终止奖励。

当前没有运行 DeepSeek、学生模型、参数训练或自动难题筛选。原始 800 题和共享语料未修改；环境不保证每题必须多次搜索。

## 启动与结构

```powershell
cd D:\work\PostTraining
.\Start-HotpotSearch.ps1 -Training
```

- 训练服务：[http://127.0.0.1:8021/docs](http://127.0.0.1:8021/docs)
- 检索服务：[http://127.0.0.1:8020/docs](http://127.0.0.1:8020/docs)
- 训练启动脚本在 8020 不可用时会尝试启动检索服务。两者只绑定 loopback。
- 训练服务通过文本/JSON 调用现有 8020 服务，复用它的 SQLite/Qdrant 索引和编码器，避免两个进程同时打开 Qdrant local 目录。
- 停止训练服务：`.\Start-HotpotSearch.ps1 -Training -Stop`。检索服务单独停止：`.\Start-HotpotSearch.ps1 -Stop`。

## 一个完整 episode

```text
训练程序 POST /reset
  → 获取问题、system/user 消息和工具定义（没有参考答案）
模型输出 local_search(query)
训练程序 POST /episodes/{id}/step
  → 返回文本 observation，reward=0
模型根据 observation 继续搜索或 read_document
模型输出 finish(answer, citations)
  → terminated=true，评分器计算终止 reward 和诊断指标
训练程序保存轨迹并 DELETE /episodes/{id} 释放内存
```

模型可在任意时刻 finish，不强制最少搜索次数。预算用完后仍允许下一动作 finish；继续请求工具则截断 episode。

### Reset

```powershell
$base = 'http://127.0.0.1:8021'
$body = @{split='rl'; seed=42; mode='bm25'; max_calls=6; max_top_k=3; run_label='interactive'} | ConvertTo-Json
$episode = Invoke-RestMethod -Method Post -Uri "$base/reset" -ContentType 'application/json' -Body $body
$episode.messages
$episode.tools
```

可选 `task_id`，必须属于指定 split。不传时按 `seed` 确定性选择；相同 seed 会选相同题，采样不同题时要改变 seed。

可用划分为 `sft_seed`（300）、`rl`（300）、`dev`（100）；不暴露 `test`。前两者用于采集或训练，dev 仅做开发评测。此限制是本项目实验隔离，不是外部鉴权系统。

默认预算：6 次搜索/读文档、每次最多 3 条搜索结果、单步最多 5,000 字符工具输出、累计最多 24,000 字符。字符预算按实际序列化 observation 计算，不等同于 tokenizer token 数；未来 RL 框架还需设置模型上下文和生成 token 预算。

### Step：模型动作

```powershell
$action = @{name='local_search'; arguments=@{query='use keywords from the question'; top_k=3}} | ConvertTo-Json -Depth 6
$step = Invoke-RestMethod -Method Post -Uri "$base/episodes/$($episode.episode_id)/step" -ContentType 'application/json' -Body $action
$step.observation

# document_id 使用刚才 results 中返回的 doc_id。
# @{name='read_document'; arguments=@{document_id='d_...'}}

# 最终提交：
# @{name='finish'; arguments=@{answer='short answer'; citations=@('d_...','d_...')}}
```

动作只有 `local_search`、`read_document`、`finish`。工具内部支持 BM25、dense、hybrid；默认 BM25。模式由训练程序在 reset 时固定，不由模型逐次切换，以便对照实验。

只允许读本 episode **实际返回过**的文档。越权读取、非法参数、未知工具均返回可见错误并消耗调用预算。后端断连等基础设施故障返回 `retryable=true`，不算模型错误；调用方须设置整体超时/重试上限，必要时丢弃该 rollout。

### 模型输入与训练反馈分开

- reset 的 `messages` 和 `tools` 可以发给老师/学生模型。
- 每次 step 的 **`observation`** 才是模型应看到的工具回复。
- `reward`、`info.metrics` 是训练程序反馈，不要拼回解题提示。
- 评分模块单独读取 `private/*.gold.jsonl`，不会把参考答案或支持事实标签放入工具结果。
- 不要把整个 data 目录挂载给可执行任意文件读取的 actor。当前是本机研究服务，不提供恶意进程隔离或多用户认证。

## 终止奖励

中间步骤 reward 为 0；只有 finish/truncation 后评分。没有搜索次数或长思考奖励，也暂未加入逐次调用费用惩罚。

```text
终止 reward = 答案 token-F1 × (0.8 + 0.2 × 证据 F1) - 错误惩罚
```

- 答案按 HotpotQA 风格小写、去英文标点/冠词、空白规范化后计算 EM/F1；yes/no/noanswer 特殊处理。没有额外同义答案库，合理别名可能被低估。
- 证据召回：标注支持句的完整规范化文本是否真的出现在模型看到的 observation 中，并且它所在文档被引用。
- 引用精确率：引用文档中真正支持标注证据的文档比例；与上述召回合成证据 F1，避免堆引用刷分。
- 未见文档引用每个扣 0.05，非法工具调用每次扣 0.01，惩罚上限 0.2。
- `supported_success` 要求答案 EM=1、证据召回=1、引用精确率=1、无未见引用。
- 预算耗尽、主动放弃造成 truncation 时 reward=0。

这是可复现的**标注证据匹配评分**，不是语义蕴含裁判。可替代的正确证据、跨版本句子或只看部分句子即可正确回答的情况可能被低估。正式批量训练前应抽样校准，不把该分数直接当真实世界可靠性。

## 轨迹保存与释放

`GET /episodes/{id}/trajectory` 返回实际对话、工具定义、配置和终止指标；其中 messages 不含参考答案标签。`run_label` 用于标记来源，例如 `teacher-deepseek-pilot` 或 `student-baseline`，需由实际调用方如实填写。

JSONL 日志自动保存到 `data/hotpotqa/runtime/episodes/{episode_id}.jsonl`，含 reset、实际可见的 step 观察、终止指标。不是自动生成的合格 SFT 文件；仍需验证正确性、去重并筛选。

每个 episode 的状态和轨迹在服务内存中；`DELETE /episodes/{id}` 释放，磁盘日志保留。未完成时释放会记为 truncation。默认同时最多保留 256 个 episode。进程重启后内存 episode 不恢复，但日志可供审计；RL checkpoint 由训练器负责。

## 验证

```powershell
# 真实 HTTP / 真实语料库的脚本化验收，不调用模型：
python -X utf8 -m hotpot_search.training_smoke

# 单元与接口测试；临时目录应使用项目内全新路径。
$testRoot = Join-Path (Get-Location).Path ('tmp\hotpot-training-' + [Guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_hotpot_training.py tests/test_hotpot_search.py -q -p no:cacheprovider --basetemp $testRoot
```

HTTP TestClient 需要 `httpx`；Windows 受限沙箱的内部事件循环可能停滞，应在本机正常 PowerShell 执行接口测试。当前已实测 15 项通过。

`data/hotpotqa/v1/training_environment_smoke.json` 保存真实验收结果：两次检索、读取文档、finish、终止评分、预算截断、跨 episode 读取拒绝、虚构引用检查、测试集与参考答案注入拒绝。示例动作和答案由脚本预设，并标记 `scripted_smoke_not_model`，**不是模型解题成绩，也不能混入教师训练数据**。

## 后续接模型

老师采集和学生 rollout 复用 reset/step，替换的只是“谁来输出下一次动作”：

1. reset 获取 messages/tools。
2. 将消息交给 DeepSeek 或本地模型，取得一个工具动作。
3. step 执行动作，将 observation 作为 tool 消息追加。
4. finish 后只把 reward/metrics 交给采集器或 RL 训练器。
5. 保存模型名称、版本、采样参数、run_label 和完整工具日志。

目前未实现 DeepSeek 自动采集器、学生模型策略适配器、并行 GPU rollout 或梯度更新。下一步可先采集 20 道教师示范，检查实际模型的多轮行为，再决定是否需要筛出更难的子集。
