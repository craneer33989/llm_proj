# 精选轨迹 SFT 开发实验

实验使用原搜索 Qwen3-4B 的本地检查点。教师、原学生和两批纠错续跑提供训练轨迹。数据仅来自训练侧。筛选器回放搜索并确认补查新增了标注支持句，同时剔除重复查询。

v2 以 40 道零检索答错题和 12 道首跳后早答题为优先样本。每完整轮有 48 条首查、48 条补查、24 条提交。实际处理 249 条样本，更新 29,970 个目标 token，执行 32 次优化器更新。每 8 个样本累积一次梯度，提交动作包含 `<|im_end|>`。训练和重载耗时 186 秒。

## 开发集结果

| 模型 | Answer EM | Answer-Grounded | Strict |
| --- | ---: | ---: | ---: |
| 原搜索学生 | 43/100 | 31/100 | 20/100 |
| 旧 A | 50/100 | 39/100 | 24/100 |
| 精选 SFT v2 | 44/100 | 31/100 | 18/100 |
| 精选 SFT v2.1 | 57/100 | 50/100 | 34/100 |

Answer-Grounded 要求正常提交的答案与参考答案或别名匹配，并且已看到包含该答案的标注支持句，或已看到全部标注支持句。标注支持句必须出现在相应文档的可见片段内。是非题按全证据判定。这是基于标注和可见文本的自动指标。

原学生首跳后提交且标注证据未齐的 27 道开发题中，17 道已有直接答案依据，10 道仍缺直接答案依据。固定首跳后，v2 在前一组保住相同的 10 道有据答对，在后一组发起了 0 次不同查询。单独提供全部标注支持句后，原学生与 v2 分别答对 20/27 和 19/27。

v2 相对原学生有 2 道 Strict 改善和 4 道回退。终局门槛为 Strict 大于 24/100 且 Answer-Grounded 大于 41/100。v2 未达到门槛。

v2.1 相对原学生有 17 道 Strict 改善和 3 道回退。搜索次数从 113 次增至 160 次。v2.1 达到两项终局门槛。

固定首跳后，v2.1 在 17 道已有直接答案依据的题中有据答对从 10 道升至 12 道，原有 10 道均保留。在 10 道真缺证早停题中，3 道改用不同查询，0 道补到了新支持句。给齐证据时，v2.1 答对 18/27，原学生答对 20/27。

## 留出集事后对照

| 模型 | Answer EM | Answer-Grounded | Strict |
| --- | ---: | ---: | ---: |
| 原搜索学生 | 53/100 | 40/100 | 27/100 |
| 旧 A | 50/100 | 40/100 | 27/100 |
| 精选 SFT v2.1 | 64/100 | 57/100 | 46/100 |

相对原学生，v2.1 有 23 道 Strict 改善、4 道回退，平均搜索次数从 1.03 增至 1.52。相对旧 A 有 22 道改善、3 道回退。留出结果来自冻结后的同题对照。 [留出汇总](../runs/reopd/course-extra/selected-sft/summary.json) · [展示图](figures/selected-sft-progress.svg)

## 复现

```powershell
.model-venv/Scripts/python.exe -m hotpot_search.selected_sft train --run runs/reopd/formal-v2-expanded --output runs/reopd/selected-sft-v2-repro
.model-venv/Scripts/python.exe -m hotpot_search.selected_sft eval --run runs/reopd/formal-v2-expanded --output runs/reopd/selected-sft-v2-repro
.model-venv/Scripts/python.exe -m hotpot_search.selected_sft report --run runs/reopd/formal-v2-expanded --output runs/reopd/selected-sft-v2-repro
```

训练入口加 `--mixed` 可复现 v2.1。它保留 v2 的训练动作和重复次数，重新混排各轮样本，并将学习率提高到 `8e-5`。已有轨迹可用 `report` 重新评分。

```powershell
.model-venv/Scripts/python.exe -m scripts.course_eval selected-sft --selected-run runs/reopd/selected-sft-v2.1
.model-venv/Scripts/python.exe -m scripts.analyze_course_metrics --require-controls --selected-report runs/reopd/course-extra/selected-sft/summary.json
.model-venv/Scripts/python.exe -m scripts.plot_course_figures
```
