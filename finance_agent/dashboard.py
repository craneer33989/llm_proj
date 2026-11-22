"""Readable training and task-performance charts from saved experiment logs."""
import json

import altair as alt
import pandas as pd
import streamlit as st

STAGES = ["训练前", "训练后（适配器重载）"]
COLORS = ["#94a3b8", "#2563eb"]


def comparison_rows(groups):
    rows = []
    for stage, metrics in groups.items():
        definitions = [
            ("单跳题通过", "direct", "passed"), ("跨报告题通过", "bridge", "passed"),
            ("主动发起第二跳", "bridge", "second_search"), ("第二跳命中目标页", "bridge", "second_hop_hit"),
            ("第一跳后提前停止", "bridge", "stopped_after_first_hop"),
        ]
        if "stop" in metrics:
            definitions = definitions[:-1] + [("证据齐全正确停止", "stop", "passed"), ("缺失证据正确结束", "missing", "passed")]
        for label, group, key in definitions:
            count, total = metrics[group].get(key, 0), metrics[group]["cases"]
            rows.append({"阶段": stage, "指标": label, "数量": count, "题数": total,
                         "比例（%）": count / total * 100 if total else None, "计数": f"{count}/{total}"})
    return rows


def render_trace(row, api):
    answer = row["answer"]
    st.write(answer["answer"])
    if "terminal_status" in row:
        st.caption(f"模型结束判断 · {row['terminal_status'] or '未有效结束'}")
    if not answer["trace"]:
        st.caption("该题未产生搜索动作。")
    for number, step in enumerate(answer["trace"], 1):
        if step["action"] == "search":
            st.markdown(f"**第 {number} 步 · 搜索**")
            st.code(step["query"], language=None)
            st.caption(f"范围 · {step['scope'].get('year')} 年 {step['scope'].get('period')}")
            seen = set()
            for hit in step.get("hits", []):
                key = hit["report_id"], hit["page"]
                if key in seen:
                    continue
                seen.add(key)
                link = f"{api}/filings/{hit['report_id']}.pdf#page={hit['page']}"
                st.markdown(f"[{hit['year']} 年 {hit['period']} · PDF 第 {hit['page']} 页]({link})")
        else:
            st.markdown(f"**第 {number} 步 · {'停止' if step['action'] == 'finish' else '无效动作'}**")
        with st.expander(f"第 {number} 步的模型原话"):
            st.code(step.get("model_output") or "无记录", language="json")
    st.caption(f"实际搜索 {answer['search_count']} 次 · {'本题通过' if row['checks']['passed'] else '本题未通过'}")


def render_dashboard(payload, api):
    report, groups = payload["report"], payload["groups"]
    training = report["training"]
    before, after = groups[STAGES[0]], groups[STAGES[1]]
    old_total = sum(row["cases"] for row in before.values())
    old_pass = sum(row["passed"] for row in before.values())
    new_pass = sum(row["passed"] for row in after.values())
    autonomous = report.get("next_step_hints") is False
    st.subheader(f"{training['updates']} 步 LoRA · 模型表现")
    st.caption(f"{report['model']} · SFT + LoRA · {report['device']} · 评测关闭规则补查 · 同一批海尔留出题")
    if report.get("parent_run"):
        st.caption(f"接上一轮适配器继续训练 1 个 epoch，累计 {report['total_epochs']} 个 epoch。前后沿用相同数据、工具与评分规则。")
    cards = st.columns(3)
    cards[0].metric("全部诊断题通过", f"{new_pass}/{old_total}", f"训练前 {old_pass}/{old_total}", delta_color="off")
    for column, label, key in (
        (cards[1], "主动发起第二跳", "second_search"), (cards[2], "第二跳命中目标页", "second_hop_hit"),
    ):
        column.metric(label, f"{after['bridge'][key]}/{after['bridge']['cases']}",
                      f"训练前 {before['bridge'][key]}/{before['bridge']['cases']}", delta_color="off")
    st.write("先看模型会不会继续查，再看第二次搜索是否找对了报告，最后看答案和出处是否一起通过。")

    data = pd.DataFrame(comparison_rows(groups))
    st.subheader("训练前后 · 同题比较")
    base = alt.Chart(data).encode(
        x=alt.X("指标:N", title=None, axis=alt.Axis(labelAngle=-15), sort=list(dict.fromkeys(data["指标"]))),
        xOffset=alt.XOffset("阶段:N", sort=STAGES),
        y=alt.Y("比例（%）:Q", title="占相应题型的比例（%）", scale=alt.Scale(domain=[0, 112])),
        color=alt.Color("阶段:N", sort=STAGES, scale=alt.Scale(domain=STAGES, range=COLORS), legend=alt.Legend(title=None)),
        tooltip=["指标:N", "阶段:N", "计数:N", alt.Tooltip("比例（%）:Q", format=".1f")],
    )
    chart = base.mark_bar() + base.mark_text(dy=-8, color="#334155").encode(text="计数:N")
    st.altair_chart(chart.properties(height=310), width="stretch")
    st.caption("柱顶标出通过数量与各自题型的分母。" + (
        "单跳、多跳、应当停止、证据缺失各 10 道；评测要求显式给出正确结束状态。" if autonomous else
        "单跳题 7 道，跨报告题 10 道。提前停止越少越好，其余指标越高越好。"
    ))
    changes = payload["transitions"]
    st.write(" · ".join(f"{label} {changes.get(label, 0)} 题" for label in ("改善", "退步", "保持通过", "仍未通过")))

    st.subheader("训练过程 · 每次更新的 loss")
    losses = pd.DataFrame([
        {"更新步": row["update"], "loss": row["loss"], "裁剪前梯度范数": row["grad_norm"]}
        for row in training["losses"]
    ])
    log_axis = st.checkbox("用对数坐标放大后几步的变化", value=False)
    curve = alt.Chart(losses).mark_line(point=True, color="#2563eb").encode(
        x=alt.X("更新步:Q", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("loss:Q", scale=alt.Scale(type="log" if log_axis else "linear", zero=False)),
        tooltip=["更新步:Q", alt.Tooltip("loss:Q", format=".6g"),
                 alt.Tooltip("裁剪前梯度范数:Q", format=".4f")],
    )
    st.altair_chart(curve.properties(height=230), width="stretch")
    st.caption("loss 衡量模型对示范动作的拟合程度。不同步骤使用不同样本，因此会有波动。任务能力要结合留出题表现判断。")
    stats = st.columns(3)
    stats[0].metric("实际参数更新", str(training["updates"]), f"每步至多累积 {training['microbatches_per_update']} 条动作", delta_color="off")
    stats[1].metric("训练进程峰值显存", f"{training['peak_allocated_mib'] / 1024:.2f} GiB")
    stats[2].metric("适配器重载验证", "一致" if report["reload_logits_max_abs_difference"] == 0 else "存在差异")
    st.caption(f"显存为 PyTorch 记录的训练进程峰值分配量。重载一致是指同一固定输入的 logits 最大差异为 {report['reload_logits_max_abs_difference']}。")
    with st.expander("训练参数与逐步记录"):
        st.write(f"LoRA rank {report['rank']} · 学习率 {training['learning_rate']} · "
                 f"可训练参数 {training['trainable_parameters']:,} · 训练更新耗时 {training['training_seconds']} 秒")
        st.dataframe(losses, hide_index=True, width="stretch")

    st.subheader("案例回放 · 模型具体改了哪一步")
    categories = {"跨报告题": "bridge", "单跳题": "direct"}
    if autonomous:
        categories.update({"应当停止": "stop", "证据缺失": "missing"})
    selected_group = st.radio("案例类型", list(categories), horizontal=True)
    group = categories[selected_group]
    candidates = [row for row in payload["cases"] if row["case"]["group"] == group]
    by_id = {row["case"]["id"]: row for row in candidates}
    case_id = st.selectbox("选择一道题", list(by_id), format_func=lambda key: by_id[key]["case"]["question"])
    pair = by_id[case_id]
    st.caption(f"案例编号 · {case_id}")
    left, right = st.columns(2)
    with left:
        st.markdown("**继续训练前 · 上轮 LoRA**" if report.get("parent_run") else "**训练前 · 官方模型**")
        render_trace(pair["before"], api)
    with right:
        st.markdown("**训练后 · 重载 LoRA**")
        render_trace(pair["after"], api)
    st.download_button("导出本题训练前后记录", json.dumps(pair, ensure_ascii=False, indent=2),
                       file_name=f"{case_id}-comparison.json", mime="application/json")
    st.info(
        "本轮使用 126 条训练动作和 40 道评测题。模型自行决定查询词、报告范围和结束状态，停止题预先提供实际检索证据。"
        "评测仅覆盖海尔的现有财报，部分题共享表格，仍需在更多来源上验证。"
        if autonomous else
        "这是一轮 10 步小样本试跑。跨报告诊断题共享海尔的两张产品表；报告范围和检索线索由程序给出。"
        "当前结果说明模型在这些线索下更会继续查，后续还需增加不同表格、缺失证据和应当停止的题目。"
    )
