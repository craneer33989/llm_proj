"""Local classroom viewer for Wiki search and saved model comparisons."""
from contextlib import closing
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
from time import perf_counter

import altair as alt
import pandas as pd
import streamlit as st

from hotpot_search.course_view import (
    ARMS, CATEGORIES, COLORS, CONTROLS, failure_counts, load_tables, route_source, search_calls, support_hits,
)
from hotpot_search.demo_run import RESIDENT_MODELS, ResidentSearch, validate_question
from hotpot_search.reopd_final import HeldOutStore, read_json
from hotpot_search.search import DEFAULT_DATA, SearchEngine

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/reopd/formal-v2-expanded/final-test"
EXTRA = ROOT / "runs/reopd/course-extra"
PYTHON = ROOT / ".model-venv/Scripts/python.exe"
LIVE = {
    "sft": "原搜索 SFT", "teacher": "搜索 RL 教师", "base": "通用 Qwen3-4B",
    "route": "早答复核路由", **{key: ARMS[key] for key in "ABCD"},
}
ALL = "全部题目"
ROUTE_GAIN = "路由新增通过"
A_GAIN = "A 改善"
A_LOSS = "A 回退"
HONOUR = "5a7dbe325542995ed0d1666d"
RONDA = "5a73c3b555429905862fe0b6"


@st.cache_resource
def live_resources():
    return Lock(), ResidentSearch()


def child_run(model, question):
    completed = subprocess.run(
        [str(PYTHON), "-m", "hotpot_search.demo_run", "--model", model],
        input=json.dumps({"question": question}, ensure_ascii=True).encode("utf-8"),
        capture_output=True, cwd=ROOT, timeout=600, check=True,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    if not completed.stdout:
        raise ValueError("模型进程没有返回结果。 " + completed.stderr.decode("utf-8", errors="replace")[-800:])
    return json.loads(completed.stdout.decode("utf-8"))


def show_error(error):
    stderr = getattr(error, "stderr", b"") or b""
    details = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr
    st.error("运行失败。 " + (details or str(error))[-1200:])


def show_steps(steps, expanded=False):
    if not steps:
        st.info("没有产生搜索或有效结束动作。")
    for number, step in enumerate(steps, 1):
        kind = step.get("kind", "unknown")
        label = "搜索" if kind == "search" else "提交答案" if kind == "finish" else "格式错误"
        with st.expander(f"第 {number} 步 · {label}", expanded=expanded and number <= 2):
            if kind == "search":
                st.write("**工具调用** `search`")
                st.code(json.dumps({"query": step.get("value")}, ensure_ascii=False), language="json")
                st.write("**工具返回**")
                st.text(step.get("observation", "该步没有可显示的结果。"))
                if step.get("doc_ids"):
                    st.caption("文档 ID：" + "、".join(step["doc_ids"]))
            elif kind == "finish":
                st.write(step.get("value", "未记录答案"))
            with st.popover("查看模型原始输出"):
                st.code(step.get("raw", ""), language="text")


def show_live():
    lock, resident = live_resources()
    st.caption("输入英文问题查看模型的搜索过程。首次加载需要 CUDA 和本地模型权重。")
    if PYTHON.is_file() and Path(sys.executable).samefile(PYTHON):
        preload, release = st.columns(2)
        if preload.button("预载原搜索 SFT 与 A"):
            try:
                with lock, st.spinner("正在加载权重到显卡…"):
                    started = perf_counter()
                    resident.load()
                st.success(f"模型已就绪 · {perf_counter() - started:.1f} 秒")
            except (RuntimeError, OSError, ValueError) as error:
                show_error(error)
        if release.button("释放常驻显存"):
            with lock:
                resident.release()
        if resident.actor is not None:
            st.caption("已常驻原搜索 SFT 与 A。选择其他模型时会先释放显存。")
    with st.form("question"):
        question = st.text_area("问题", value="Which band was founded first, Oingo Boingo or Sleater-Kinney?", max_chars=500)
        model = st.selectbox("模型", list(LIVE), format_func=LIVE.get)
        submitted = st.form_submit_button("运行检索", type="primary")
    if submitted:
        st.session_state.pop("wiki_run", None)
        try:
            validate_question(question)
            if not PYTHON.is_file():
                raise FileNotFoundError("缺少 .model-venv，请先按 README 配置模型环境。")
            with lock, st.spinner("模型正在生成动作并检索本地语料…"):
                started = perf_counter()
                if resident.actor is not None and model in RESIDENT_MODELS:
                    with closing(SearchEngine()) as engine:
                        result = resident.run(engine, model, question)
                else:
                    resident.release()
                    result = child_run(model, question)
                st.session_state["wiki_run"] = (model, result, perf_counter() - started)
        except (subprocess.SubprocessError, OSError, RuntimeError, ValueError) as error:
            show_error(error)
    if latest := st.session_state.get("wiki_run"):
        model, result, seconds = latest
        st.subheader("本次结果")
        st.caption(f"{LIVE[model]} · 选用轨迹搜索 {result['calls']} 次 · {seconds:.1f} 秒")
        if result.get("route") == "directed_sft_after_no_search":
            st.info("原 SFT 零检索提交，随后运行 A。首轮答案为 " + (result.get("first_answer") or "空"))
        elif result.get("route") == "original_sft":
            st.info("保留原搜索 SFT 的结果。")
        st.write(result["answer"] or "模型没有提交有效答案。")
        show_steps(result["steps"], expanded=True)


@st.cache_data
def saved_cohort(artifacts):
    size = read_json(RUN / "complete.json")["test_questions"]
    store = HeldOutStore(DEFAULT_DATA, size)
    expected = {(qid, 0) for qid in store.tasks["test"]}
    tables, pending = load_tables(RUN, EXTRA, expected)
    return store.tasks["test"], store.gold["test"], tables, pending


def artifact_stamp():
    files = [RUN / "complete.json", DEFAULT_DATA / "manifest.json", DEFAULT_DATA / "test.jsonl",
             DEFAULT_DATA / "private/test.gold.jsonl"]
    for arm in ARMS.keys() - {"route"}:
        path = (EXTRA if arm in CONTROLS else RUN) / arm
        files.extend(path / name for name in ("episodes.jsonl", "scores.jsonl"))
    return tuple((str(path), path.stat().st_mtime_ns if path.is_file() else None) for path in sorted(files))


def show_summary(tables):
    summaries, chart_rows = [], []
    for arm, table in tables.items():
        n = len(table)
        counts = failure_counts(table)
        summaries.append({"模型": ARMS[arm], "有证答对": f"{counts['pass']}/{n}",
                          "答案 EM": f"{sum(row['answer_em'] for row in table.values()):.0f}/{n}",
                          "搜索次数/题": round(sum(search_calls(tables, arm, key) for key in table) / n, 2)})
        for index, (kind, label) in enumerate(CATEGORIES.items()):
            chart_rows.append({"模型": ARMS[arm], "类别": label, "题数": counts[kind], "顺序": index})
    st.subheader(f"本地留出 · 同一批 {len(tables['base'])} 题")
    st.dataframe(pd.DataFrame(summaries), hide_index=True)
    st.caption("五组为冻结成绩。通用基座、混合检索和路由为事后分析。路由的平均搜索次数包含两轮。")
    with st.expander("查看失败分布"):
        category = alt.selection_point(fields=["类别"], bind="legend")
        bars = alt.Chart(pd.DataFrame(chart_rows)).mark_bar().encode(
            x=alt.X("题数:Q", stack="zero"),
            y=alt.Y("模型:N", sort=[ARMS[arm] for arm in tables], axis=alt.Axis(labelLimit=240, title=None)),
            color=alt.Color("类别:N", scale=alt.Scale(domain=list(CATEGORIES.values()), range=list(COLORS)),
                            legend=alt.Legend(orient="bottom")),
            order=alt.Order("顺序:Q"),
            opacity=alt.condition(category, alt.value(1), alt.value(0.22)),
            tooltip=["模型:N", "类别:N", "题数:Q"],
        ).add_params(category)
        st.altair_chart(bars)


def jump_to(qid):
    st.session_state["case_filter"] = ALL
    st.session_state["case_qid"] = qid
    st.session_state["case_arms"] = ["base", "A"]


def show_case(qid, selected, tables, gold):
    st.markdown(f"**标注答案**　{gold['answer']}")
    key = (qid, 0)
    facts = gold["supporting_facts"]
    if not selected:
        st.info("选择至少一个模型查看轨迹。")
        return
    for column, arm in zip(st.columns(len(selected)), selected, strict=True):
        with column:
            episode = tables[arm][key]
            st.subheader(ARMS[arm])
            status = "是" if episode["supported_success"] else "否"
            st.caption(f"搜索 {search_calls(tables, arm, key)} 次 · 有证答对 {status} · {episode['reason']}")
            if arm == "route":
                picked = route_source(tables["base"][key])
                st.caption("选用 A 的复核轨迹" if picked == "A" else "保留原搜索 SFT 的轨迹")
            answer = next((step["value"] for step in episode["steps"] if step["kind"] == "finish"), "未提交")
            st.write(f"**提交答案**　{answer}")
            hits = support_hits(episode, facts)
            st.markdown(f"**标注支持句 · {episode['support_sentence_recall']:.0%} 命中**")
            if hits is None:
                st.caption("日志可确认总覆盖率，逐句匹配无法复核。")
            for index, fact in enumerate(facts):
                status = "待核实" if hits is None else (":green[已覆盖]" if hits[index] else ":red[未覆盖]")
                st.markdown(f"**{status} · {fact['title']}**")
                st.write(fact["text"])
            show_steps(episode["steps"])


def show_comparison():
    if not (RUN / "complete.json").is_file():
        st.info("此工作区还没有冻结结果。")
        return
    try:
        tasks, gold, tables, pending = saved_cohort(artifact_stamp())
    except (OSError, ValueError, KeyError) as error:
        show_error(error)
        return
    show_summary(tables)
    if pending:
        st.caption("此工作区缺少对照产物。 " + "、".join(pending))

    st.subheader("同题轨迹对比")
    first, second = st.columns(2)
    if HONOUR in tasks:
        first.button("修复案例 · Honour Killing", on_click=jump_to, args=(HONOUR,))
    if RONDA in tasks:
        second.button("早停回退 · Ronda Rousey", on_click=jump_to, args=(RONDA,))
    keys = sorted(tables["base"])
    base_pass = {qid for qid, attempt in keys if tables["base"][qid, attempt]["supported_success"]}
    a_pass = {qid for qid, attempt in keys if tables["A"][qid, attempt]["supported_success"]}
    route_pass = {qid for qid, attempt in keys if tables["route"][qid, attempt]["supported_success"]}
    choices = {ALL: set(tasks), ROUTE_GAIN: route_pass - base_pass, A_GAIN: a_pass - base_pass, A_LOSS: base_pass - a_pass}
    case_filter = st.radio("题目筛选", choices, key="case_filter", horizontal=True,
                           format_func=lambda key: f"{key}（{len(choices[key])}）")
    options = sorted(choices[case_filter])
    if not options:
        st.info("该筛选下没有匹配题目。")
        return
    if st.session_state.get("case_qid") not in options:
        st.session_state["case_qid"] = HONOUR if HONOUR in options else options[0]
    qid = st.selectbox("题目", options, key="case_qid", format_func=lambda key: tasks[key]["question"])
    st.session_state.setdefault("case_arms", ["base", "A"])
    selected = st.multiselect("对比模型", list(tables), key="case_arms",
                              max_selections=3, format_func=ARMS.get)
    if qid == RONDA:
        st.caption("原搜索 SFT 已经检索，路由保留它的轨迹。A 在第一次搜索后结束。")
    show_case(qid, selected, tables, gold[qid])


st.set_page_config(page_title="Wiki 多跳检索台", page_icon="🔎", layout="wide")
st.title("Wiki 多跳检索台")
page = st.segmented_control("展示模式", ["模型表现对比", "输入问题"], default="模型表现对比", label_visibility="collapsed")
if page == "输入问题":
    show_live()
else:
    show_comparison()
