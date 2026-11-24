# HotpotQA 定向 ReOPD：本地实施与复现

按[执行草案](reopd-execution-plan.md)实施本地实验。沿用现有 HotpotQA 划分；训练和 dev 入口不读取 test，正式成绩通过冻结后的独立 P5 入口评估。上游锁定 `BaohaoLiao/ReOPD@7a499d29ab07ff50dfde039f9d7a44dfe313b95e`；本地用 SQLite BM25 替换其外部检索，保留 Search-R1 单 action 的 JSON `search` 与 `\boxed{}` 短答。16 GiB 本机采用 Transformers/PEFT LoRA（学生 CUDA，冻结教师 CPU），**不是**上游 slime/Megatron 复跑。

## 最终结果（2026-11-22）

P0–P5 的本地实验已完成。P4 对「未检索就答错」跑了四组 × 两种子，共 8 份 LoRA，全部重载验证，每份恰好更新 12,000 个学生 token。种子 42 / 43 的 100 题 dev 严格证据支持成功分别为：未适配学生 20 / 20，A 24 / 24，B 23 / 21，C 22 / 25，D 22 / 22。固定故障队列为同一批 28 题，两种子的 A/B/C/D 恢复分别为 4/4/3/4 和 3/4/4/3。

P5 按冻结规则采用主种子 42。100 道本地 test 的严格证据支持分别为：未适配学生 **27**，A **27**，B **23**，C **25**，D **23**；24 条固定故障中分别恢复 **0、2、1、1、2** 条。D 相对 A 的端到端差为 −4 个百分点（配对 95% 区间 −12 至 +3），固定故障恢复相同。A/D 的训练墙钟约 92 / 5,317 秒；没有证据支持为 D 承担额外教师打分成本，也不据此断言方法等效或外推到其他索引。

以 `runs/reopd/formal-v2-expanded/` 为根，证据见 `test-freeze.json`，`final-test/report.md`、`final-test/complete.json`、配对 CSV 和改善/退化案例，以及两个 `ablation-dev-12000tokens-*-wrong-without-search/` 目录。未本地重训上游 SFT/GRPO；使用已发布权重作为初始化与教师。

## 公开数据前后对比

`reopd_public_eval` 单独运行，不重新训练，也不改动 P5 成绩。顺序固定为 HotpotQA 官方 distractor dev 前 100 题 → ReOPD `baohao/search-test` 的 2WikiMultiHopQA 来源 100 题 → HotpotQA 后 100 题。种子 42，按题型比例抽样；两批 HotpotQA 不重复，且排除与本地划分规范化题干完全相同的题。ReOPD 行必须唯一匹配 2Wiki 官方 dev 的镜像题干、答案与可追溯的支持句；结果只代表这个子集，不代表全部七个来源，也不能排除上游权重的其他重合。

同一数据源的候选段落构成共享 BM25 库，HotpotQA 的 200 道共用一套库；不按题限定候选文档，不把参考答案写进提示。每批依次用未挂 LoRA 的 SFT、A、D，同一角色提示、搜索预算与生成上限，A/D 使用 P5 冻结的种子 42 权重。报告包括答案 EM、严格证据支持、搜索调用、逐题改善/退化及配对区间。这不是官方 fullwiki 或作者默认索引成绩。输入版本与 SHA-256 见代码中的 `SOURCES`，运行目录为 `runs/reopd/public-eval-v1/`；批次可在相同配置下续跑。

## 早期验收记录（2026-11-21）

- 真实 BM25 单题脚本跑两次搜索，reset 后逐步核对 observation hash 和同一排序，答案 EM、支持文档覆盖、支持句覆盖均为 1。该脚本知道查询和答案，不是模型成绩：`runs/reopd/local-20261121/reference-smoke.json`。
- v2 另外在两个临时 loopback 端口实际启动检索与训练 HTTP 服务。搜索、读文档、提交和释放会话通过（终止奖励 1.0）；预算截断、拒绝跨会话读与伪造引用、拒绝 test split 和参考答案注入也通过。服务验证不是模型成绩，报告：`runs/reopd/formal-format-smoke-v2/http-training-smoke.json`。
- 缓存 Qwen3-1.7B GPU 诊断：10/10 题保存，0 后端错误、实际 2 次搜索、0 条按协议完成；`rl-diagnostic/metrics.json`。协议未在正式 4B 权重上校准，1.7B 的结果不能记为正式 Base/SFT/教师成绩，也不能用来筛恢复池。
- v2 提示在同一批 10 道 RL 诊断题上，256 和 512 token 各执行一轮，均为 0/10 按协议完成、0 后端错误。512 token 产物中 4 道生成了闭合的工具调用，但把 `arguments` 当成 JSON 字符串而不是对象；上游同样按对象取 `query`，不因此放宽正式解析。轨迹分别在 `runs/reopd/protocol-diagnostic-v2/rl-diagnostic/` 和 `runs/reopd/protocol-diagnostic-v2-512/rl-diagnostic/`。这是 1.7B 格式诊断，不能推断正式 4B 的表现。
- P0 正式 4B SFT 学生已通过 v2 协议验收：10/10 按协议提交且轨迹和评分逐条复放一致，实际检索命中 15 次，7 题在检索后提交，答案 EM 5/10，严格支持成功 1/10，后端错误 0。见 `runs/reopd/formal-format-smoke-v2/p0-acceptance.json` 和同目录的 `dev-sft/metrics.json`；这是协议冒烟，不是三模型对比。
- P1 首批三模型各 100/100 可评分、后端错误均为 0。基座 / 学生 / 教师的答案 EM 为 42 / 35 / 54，严格支持成功为 22 / 15 / 34，搜索次数为 93 / 94 / 185。逐题对照及固定种子抽出的 30 条待审错误见 `runs/reopd/formal-v2/rl-comparison/`。学生仅有 16 条自动早停候选，尚未经教师恢复或人工审核；其上界低于 20 题门槛，扩展到 300 道 RL 题，不从 dev/test 补数。未启动长训或打开本地 test。

## 固定环境与 P0

在项目根目录用已配置 CUDA Torch、Transformers 4.57 和 PEFT 的 `.model-venv`。本实验直接访问同一份只读 SQLite 索引，不需要启动 8020/8021 服务。每个运行目录锁定数据清单、索引清单/文件、协议和预算；BM25、top_k=3、最多 6 次真实搜索。Reference 重开时用新目录保留旧证据。

```powershell
$py = '.\.model-venv\Scripts\python.exe'
$run = 'runs/reopd/formal-v1'
& $py -m hotpot_search.reopd_run reference --run $run
$env:HF_HOME = "$PWD/.cache/huggingface"
$env:HF_HUB_DOWNLOAD_TIMEOUT = '120'
$hf = '.\.model-venv\Scripts\hf.exe'
& $hf download baohao/Search_Qwen3-4B-Instruct-2507_SFT --revision 19926d55956eaeb04775c89637a8948d750c3c80 --include '*.json' '*.txt' '*.jinja' '*.safetensors'
& $hf download baohao/Search_Qwen3-4B-Instruct-2507_SFT-RL --revision 041b174e3a9e5f37e85a4ec2426b2fd9830ac13e --include '*.json' '*.txt' '*.jinja' '*.safetensors'
& $hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --include '*.json' '*.txt' '*.jinja' '*.safetensors'
& $py -m hotpot_search.reopd_run baseline --model sft --split dev --limit 10 --run runs/reopd/formal-format-smoke-v2
& $py -m hotpot_search.reopd_preflight --run runs/reopd/formal-format-smoke-v2
```

`p0-acceptance.json` 固定输入文件哈希并记录终止原因。验收要求 10 条都可复放且评分一致、至少一条真实检索命中后按协议提交，不能用仅生成文本或纯脚本轨迹替代。随后抽看模型原话和返回文本；通过仅代表协议能贯通，不是准确率门槛。

v2 对所有模型在用户消息中统一增加搜索提示和 `Answer: \boxed{...}` 末行要求。固定版本的[上游评测器](https://github.com/BaohaoLiao/ReOPD/blob/7a499d29ab07ff50dfde039f9d7a44dfe313b95e/eval/search/eval.py)从公开数据行直接读取 `prompt`，其中已包含终答格式；本地题干只有问题，不自带说明。v2 不照搬无限次检索措辞，也不读取本地 test 来调提示。旧目录不续写，旧诊断不可与 v2 合并。

若常规下载反复断流，可用 `& $py -m hotpot_search.reopd_download sft`，其余两套分别传 `teacher`、`base`。备用入口在独立缓存中持久保留已验证范围来源的 `.part` 字节，重跑时从上次写入的位置继续；只在整片 SHA-256 与固定 revision 的 LFS 指针一致时发布分片，所有文件齐备才标记 checkpoint 可用。也可在网络更好的机器上下载上述固定 revision，再复制完整 HF 缓存。本地加载缺分片时直接失败，不会换其他模型。正式 10 题检查 raw action、search/finish、失败原因和实际返回文本，不能只看请求数。

## P1 基线和 P2 恢复池

也可由 `& $py -m hotpot_search.reopd_baseline_pipeline --smoke runs/reopd/formal-format-smoke-v2 --run $run` 串行运行 P0 和 P1。P0 未通过或任一权重缺失时会停；已保存的批次按原配置续跑。完成 P1 后仍须核对抽审案例，不会自动进入 P2/P3。

续跑时保留已评分轨迹（包括模型答错），只按固定重试种子重新执行记录为 `backend_error` 的轨迹。确定固定早停前缀前，未适配 SFT 的完整参考队列必须全部可评分；后端未恢复时先续跑同一队列，不缩小分母。

```powershell
foreach ($model in 'base','sft','teacher') {
    & $py -m hotpot_search.reopd_run baseline --model $model --split rl --limit 100 --run $run
    if ($LASTEXITCODE) { throw "Baseline failed: $model" }
}
& $py -m hotpot_search.reopd_compare baseline --run $run
& $py -m hotpot_search.reopd_run recover --limit 100 --run $run
& $py -m hotpot_search.reopd_compare controls --run $run
```

`episodes.jsonl` 是可见历史、原始动作、真实文档 ID、终止原因和调用数；`scores.jsonl` 是评分侧。后端错误不填失败分数，保留单独分母。`paired_results.csv` 按同一 qid 对齐，并记录首次搜索后采取的动作。P1 用 `failure-cases.private.jsonl` 查看固定种子轮流抽出的至多 30 条错误，把分类与复核理由填入 `failure-audit.csv`；分类包括 `early_stop`、`wrong_entity`、`misread`、`corpus_missing`、`format_error` 和 `other`。这是私有评分侧产物，不发给模型，不用测试集指导修改。公开权重与本地题潜在重合没有排除。

恢复候选只从 RL 基线截取：第一次搜索后直接提交且尚未见到全部支持句。用真实查询复放固定前缀，教师独立采样三次；三次都需可评分，其中至少两次必须下一步搜索、最后答对且显示全部支持句。后端异常保留为未判定，不算模型失败。模型没有 citations，因此只报支持文档/句的可见性，不报引用正确率，也不把看到证据等同于推理使用了证据。审核原始轨迹，确认确有过早答、可见事实支持答案，再将 `audit.csv` 的对应 `human_review` 标为 `approved`；只看自动指标不算审核。审核与学生前缀及教师试验哈希一起绑定。普通对照按第 1 轮和 observation 字符长度就近匹配。少于 20 个不同的审核通过 RL 题，不启动 P3；优先换新运行目录把 RL 扩到 300，再按草案决定是否停止或调整故障类型。

## P3 与继续/停止条件

首轮早停结论：300 道 RL 的 69 个候选均完成三次教师续跑，5 个通过自动检查，3 个经人工批准（门槛 20）；该故障未开训。备选「未检索直接答错」使用 `reopd_run recover --limit 300 --failure-type wrong_without_search --run runs/reopd/formal-v2-expanded`，84 题共 252 次续跑全可评分；28 个通过自动检查，排除 7 个别名、歧义或证据不足的案例后批准 21 个。独立产物位于 `rl-recovery-wrong-without-search`，不混用早停池。

备选故障已完成 A/D 各 1 步和 10 步训练：训练命令加 `--failure-type wrong_without_search`；10 步更新分别消耗 1,342 / 1,590 个学生 token，D 另消耗 4,012 个教师打分输入 token，训练墙钟 26 / 447 秒（包含保存重载），峰值分配 8.17 / 8.05 GiB。重载后，未适配 SFT / A / D 在相同 10 道 dev 题上的严格支持成功均为 1/10，答案 EM 为 5 / 6 / 5；两条固定故障状态均是空历史，对应同一端到端输入，因此复用这些题的原始轨迹，三组均为 0/2。见 `runs/reopd/formal-v2-expanded/smoke-dev-wrong-without-search/`。这验证了链路，没有显示 D 的效果优势，也不是等 token 预算对照。四组继续前需要资源确认；未打开 test。

如果首批 100 题不足以形成恢复池，开一个新目录，以相同 seed、batch 和生成预算对三组分别执行 `reopd_run baseline --model <base|sft|teacher> --limit 300 --run <新目录> --reuse-baseline-run $run`。复用只接受完全一致的环境、推理配置和题序前缀，只搬运完整批次；然后在新目录重新生成配对表和恢复审核材料。

审核时用 `rl-recovery/recovery-cases.private.jsonl` 对照原始学生轨迹、三次教师续跑、真实返回和私有参考证据。只在 `audit.csv` 写入 `approved` 且填写 `review_note`（说明缺失证据及教师在哪里找到）才计入 20 题门槛。审核仅在评分侧进行；换了教师轨迹的旧批准不会继承。

```powershell
foreach ($group in 'A','D') {
    & $py -m hotpot_search.reopd_train $group --steps 1 --run $run
    if ($LASTEXITCODE) { throw "Training probe failed: $group" }
}
```

A 忽略历史与 observation 的损失，只拟合恢复池中教师成功的下一动作；D 用当前学生采样的实际 token IDs，由冻结教师在相同历史和目标上打分，计算采样 reverse-KL 策略梯度损失。本地 probe 为动作后离线评分，不执行新采样工具调用；但恢复前缀实际重新查索引，分开记录两种调用数。首步输出掩码与训练/教师打分 token，验证更新、教师不变、保存且重载 LoRA。单步完成后核对显存、耗时和 tokenizer 审计，再按草案决定是否运行 10–30 步；更新步数相等不等于 token 预算相等。

```powershell
& $py -m hotpot_search.reopd_compare dev --run $run --limit 10 `
    --adapter-a "$run/smoke-A-10step-42/adapter" --adapter-d "$run/smoke-D-10step-42/adapter"
```

Dev 在同一批完整题和由 SFT 先定义的固定早停前缀上评测 A/D，不按教师能否修复筛留出样本；按题聚类给配对区间。10 题 smoke 只验闭环。P4 需按草案另行通过资源、病例和控制组设计门槛；P5 此前不打开 test，不能把 dev 结果称为最终对照。

P4 离线前缀构池入口：`& $py -m hotpot_search.reopd_ablation --run $run --seed 42`。只读取 RL 基线和通过 P2 人工门槛的前缀；A/D 共用 3:1 前缀和顺序，B 对本地成功教师轨迹按 `0.6^t` 保留，C 共用普通教师前缀并以同轮次、最接近的提示 token 数替换定向前缀。产物记录前缀哈希和实际配对长度差，配额不足会失败，不从 dev/test 补数。此入口只构池，不训练；构池完成也不等于通过 P4。

备选故障的四组双种子预检：`& $py -m hotpot_search.reopd_ablation_pipeline --run runs/reopd/formal-v2-expanded --failure-type wrong_without_search --update-tokens 12000 --seeds 42 43`。默认只打印计划并核验权重、P3 与训练池，不执行训练。资源预算确认后追加 `--confirm-p4-budget`，串行训练并运行两次 100 题 dev 对照，不读取 test。重跑时只跳过已完整校验的训练；如发现部分产物则停止并提示，不覆盖原目录。

P3 验收并确认资源预算后，四组分别调用 `reopd_train <A|B|C|D> --run $run --seed 42 --update-tokens <共同预算> --confirm-p4-budget`；本次另加 `--failure-type wrong_without_search`。此模式要求已完成 A/D 单步探针、至少 10 道相同 dev 配对评测以及固定前缀评测产物，并核验构池来源。四组用相同的学生更新 token 上限，最后一步只对剩余额度计损失；另外报告生成 token、教师打分 token、实际更新步数及前缀复放调用。预算确认不是跳过 P2/P3 的开关。第二种子使用独立构池和训练产物。

训练结束后，`reopd_compare ablation-dev --run $run --limit 100 --adapter-a <路径> --adapter-b <路径> --adapter-c <路径> --adapter-d <路径>` 校验四组 adapter 内容哈希、共享训练设置和实际 token 预算，在相同 dev 题及未适配 SFT 定义的固定早停状态上评测。`cohort_metrics.json` 分开记录固定分母、可评分数、后端未判定数及简单题回退；`changed_cases.jsonl` 保存相对未适配 SFT 的改善和退化。以训练种子区分产物，保持评测种子不变；这仍是 dev，不是最终 test 验收。

P5 入口独立于训练。两套种子完成相同 dev 评测后先运行 `reopd_final freeze --run $run --update-tokens <共同预算> --seeds 42 43`，本次另加 `--failure-type wrong_without_search`，不打开 test；规则预先固定为相同 token 预算的终点 checkpoint，较小训练种子为主、另一组用于 dev 复核，不按 test 挑模型。核对 `test-freeze.json` 后才运行 `reopd_final evaluate --run $run --confirm-held-out`。冻结校验通过才加载 100 道 test；中断可续同一冻结配置，产出 `final-test/complete.json` 后拒绝重复评测。若固定前缀为空或后端未判定导致分母不足，报告不能支持恢复效果结论，不回看 test 调参。

最终 `report.md` 和 `conclusion.json` 将随完成凭证保存。预先固定：少于 20 个不同留出故障题或存在未判定前缀，不作恢复效果结论；优效要求按题聚类的配对 95% 区间下界严格大于 0。区间跨零不视作等效。报告列实际分母、简单题回退、训练 token 成本，并区分本地交付与公开 SFT/教师初始化。
