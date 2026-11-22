# 教师 SFT：前后评测、常用指标与曲线图

实际交付见 [完整报告](../evaluation/teacher-sft-epoch1-test-20261114/README.md)、[全部指标与逐题配对](../evaluation/teacher-sft-epoch1-test-20261114/metrics.json)、[图表](../evaluation/teacher-sft-epoch1-test-20261114/figures/overview.png)。全部使用本地模型与固定测试集，不调用教师 API。

## 评测入口

完整自主 rollout 用 `.model-venv` 执行 `python -u -X utf8 -m finance_agent.evaluate_teacher`，传入：

- `--snapshot`：固定 Qwen3-1.7B 本地快照。
- `--output`：新的运行目录；已有结果默认拒绝覆盖。
- `--adapter`：训练后的 LoRA；省略时评测原始基座。
- `--empty-cache`：生成后释放闲置分配器缓存，可缓解长上下文下的缓存压力，不改变模型与解码设置。
- `--resume`：中断后核验配置、评分函数与已保存轨迹，从未完成题继续。运行锁存在时拒绝重复启动。

默认使用测试集全部 120 题。初始模型输入仅含系统协议和用户题目，不含标准答案、标准动作或下一步提示。贪心生成，每次最多 384 token；不截断历史证据，不修复模型失败。逐题保存原始生成文本、实际工具观察、动作与评分。

完整任务通过需要状态、数值、单位、引用、必要计算、预算与停止行为都正确；条件检索还要求先观察 argmax，再据其结果选择年度。最终状态的 Accuracy、Macro Precision／Recall／F1 属于 answered／insufficient／incomparable 三分类。答案 EM 只核验状态和适用的数值／单位，完整任务成功率另要求来源与工具执行。

## 测试集交叉熵与 PPL

用 `.model-venv` 执行 `python -u -X utf8 -m finance_agent.evaluate_likelihood --snapshot <本地快照> --adapter <最终适配器> --output <新目录>`。

同一个模型分别启用、禁用 LoRA，计算全部 400 条测试参考动作的条件概率，不更新任何参数。标准轨迹前缀与标准前序 token 仅用于这项 teacher-forcing 指标，与盲生成 rollout 分开。prompt 不计 loss，只统计 completion 和 EOS：

`CE = 所有目标 token 的负对数似然总和 / 目标 token 总数`，`PPL = exp(CE)`。

Token Accuracy 同样在标准前序 token 下计算，不能当作自由生成答案正确率。参考措辞和 JSON 排序会影响 PPL；语义正确的替代表达仍可能有较高 NLL。

## 汇总与绘图

使用 `.venv` 执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.report_teacher_eval --sft data/runtime/sft-teacher-test-20261113 --base data/runtime/base-teacher-test-20261114 --likelihood data/runtime/sft-teacher-likelihood-test-20261114 --training data/runtime/sft-teacher-epoch1-20261113 --output evaluation/teacher-sft-epoch1-test-20261114
.\.venv\Scripts\python.exe -X utf8 -m finance_agent.plot_teacher_eval --comparison evaluation/teacher-sft-epoch1-test-20261114 --training data/runtime/sft-teacher-epoch1-20261113 --likelihood data/runtime/sft-teacher-likelihood-test-20261114
```

绘图依赖为项目 `.venv` 中的 Matplotlib 3.11.2，可用 `uv pip install --python .venv/Scripts/python.exe matplotlib==3.11.2` 安装。图表导出 PNG 和可编辑 SVG，原始曲线数值为 `figures/chart-data.json`，文件哈希为 `figures/manifest.json`。

训练 loss 曲线使用最终训练报告的 789 条唯一更新记录，含原始值和 20 步移动平均。测试累计成功率、累计 CE/PPL 的横轴是测试样本数，不能解释为每个训练 checkpoint 的验证曲线。目前只对原始基座与最终适配器作完整评测。

120 题按公司与报告隔离，但共享报告、任务模板与程序构题逻辑；结果用于当前财报工具协议。SFT 两例失败均为全年／半年口径判断错误。根据这些测试错误调整后续训练时，应另留新的最终测试集。
