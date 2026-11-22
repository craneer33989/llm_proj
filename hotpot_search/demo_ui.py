"""Local Streamlit viewer for live HotpotQA search and frozen model comparisons."""
import json
from pathlib import Path
import subprocess

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "reopd" / "formal-v2-expanded" / "final-test"
PYTHON = ROOT / ".model-venv" / "Scripts" / "python.exe"
NAMES = {"base": "原搜索 SFT", "A": "A · 定向 SFT", "B": "B · 教师前缀回放",
         "C": "C · 普通混合 ReOPD", "D": "D · 定向恢复 ReOPD"}
LIVE = {"sft": "原搜索 SFT", "teacher": "搜索 RL 教师", "base": "基础模型",
        "route": "早答复核路由（SFT → 定向 SFT）",
        **{key: value for key, value in NAMES.items() if key != "base"}}


@st.cache_data
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def show_steps(steps):
    if not steps:
        st.info("没有产生搜索或有效结束动作。")
    for number, step in enumerate(steps, 1):
        kind = step.get("kind", "unknown")
        label = "搜索" if kind == "search" else "提交答案" if kind == "finish" else "格式错误"
        with st.expander(f"第 {number} 步 · {label}", expanded=number <= 2):
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


st.set_page_config(page_title="Wiki 多跳检索台", page_icon="🔎", layout="wide")
st.title("Wiki 多跳检索台")
st.caption("本地 HotpotQA 共享语料 · BM25 · 每次返回最多 3 个片段 · 最多 6 次搜索")
live_tab, compare_tab = st.tabs(["输入问题", "模型表现对比"])

with live_tab:
    st.write("输入英文问题，让选定模型自主决定搜索词、是否继续搜索，以及何时提交答案。")
    st.caption("实时选项均为 4B 权重：基础模型是未做搜索 SFT 的 Qwen；原搜索 SFT 和搜索 RL 教师是公开权重；A–D 在原搜索 SFT 上加载本地 LoRA。每次运行只加载所选的一套。")
    with st.expander("这些模型分别是什么"):
        st.markdown("""
        - **基础模型**：Qwen3-4B-Instruct-2507，未经过本项目的搜索训练。
        - **原搜索 SFT**：公开搜索监督微调权重，也是 A–D 的共同起点。
        - **搜索 RL 教师**：公开的搜索强化学习权重，用于训练时提供指导。
        - **A**：用定向失败样本继续做 SFT。
        - **B**：用教师前缀回放做 ReOPD。
        - **C**：用普通混合前缀做 ReOPD。
        - **D**：用定向恢复前缀做 ReOPD。

        对比页的“原搜索 SFT”是未加载 A–D 适配器的参考组；基础模型和教师不在该次冻结测试表中。
        """)
    st.info("实时运行需要 CUDA 显卡；启动前会检查至少 12 GiB 空闲显存和 8 GiB 空闲系统内存。只想看结果可直接使用“模型表现对比”，无需加载权重。")
    with st.form("question"):
        question = st.text_area("问题", value="Which band was founded first, Oingo Boingo or Sleater-Kinney?", max_chars=500)
        model = st.selectbox("模型", list(LIVE), format_func=LIVE.get, index=0)
        submitted = st.form_submit_button("运行检索", type="primary")
    if submitted:
        if not question.strip():
            st.error("请输入问题。")
        elif not PYTHON.is_file():
            st.error("缺少 .model-venv，请先按 README 配置模型环境。")
        else:
            with st.spinner("模型正在生成动作并检索本地语料，首次加载可能需要数分钟…"):
                try:
                    completed = subprocess.run(
                        [str(PYTHON), "-m", "hotpot_search.demo_run", "--model", model],
                        input=json.dumps({"question": question}), text=True, encoding="utf-8",
                        capture_output=True, cwd=ROOT, timeout=600, check=True,
                    )
                    st.session_state["wiki_run"] = json.loads(completed.stdout)
                    st.session_state["wiki_model"] = model
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as error:
                    details = getattr(error, "stderr", "") or str(error)
                    st.error("运行失败：" + details[-1200:])
    if result := st.session_state.get("wiki_run"):
        st.subheader("本次结果")
        st.caption(f"{LIVE[st.session_state['wiki_model']]} · 搜索 {result['calls']} 次 · {result['reason']}")
        if result.get("route") == "directed_sft_after_no_search":
            st.info("原 SFT 未检索就作答，已用定向 SFT 重新检索和作答。原答案：" + (result.get("first_answer") or "无"))
        elif result.get("route") == "original_sft":
            st.info("原 SFT 已检索或未正常作答，保留其结果。")
        st.write(result["answer"] or "模型没有提交有效答案。")
        show_steps(result["steps"])
        st.caption("自由输入问题没有参考答案或评分；这里显示模型实际动作，不代表答案已被验证。")

with compare_tab:
    if not (RUN / "complete.json").is_file():
        st.warning("冻结测试结果尚未完成，模型对比暂不可用。")
    else:
        cohort = read_json(RUN / "cohort_metrics.json")["arms"]
        rows = [{"模型": NAMES[key], "有证答对": f"{cohort[key]['supported_successes']}/{cohort[key]['scorable']}",
                 "答案准确": f"{cohort[key]['answer_em_correct']:.0f}/{cohort[key]['scorable']}",
                 "平均搜索次数": round(cohort[key]["mean_search_calls"], 2)} for key in NAMES]
        routed_report = ROOT / "runs" / "reopd" / "routed-eval" / "report.json"
        if routed_report.is_file():
            routed = read_json(routed_report)["cohorts"]["local_test"]
            rows.append({"模型": "早答复核路由（回放）",
                         "有证答对": f"{routed['routed']['supported_successes']}/{routed['questions']}",
                         "答案准确": f"{routed['routed']['answer_em_correct']:.0f}/{routed['questions']}",
                         "平均搜索次数": "两模型条件复核"})
        st.subheader("冻结留出测试 · 同一批 100 题")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("“有证答对”要求答案正确且搜索结果覆盖支持证据。各组使用相同语料与预算；此处只读取已保存结果。")
        if routed_report.is_file():
            st.caption("路由行按同题既有轨迹回放，先看原 SFT 是否零次检索，再选择 A；本地 test 已打开，属于事后分析。")
        tasks = {row["id"]: row["question"] for row in read_jsonl(ROOT / "data" / "hotpotqa" / "v1" / "test.jsonl")}
        qid = st.selectbox("查看同一道题的模型轨迹", sorted(tasks), format_func=lambda key: tasks[key])
        selected = st.multiselect("对比模型", list(NAMES), default=["base", "A", "D"], format_func=NAMES.get)
        for key in selected:
            episodes = {row["qid"]: row for row in read_jsonl(RUN / key / "episodes.jsonl")}
            scores = {row["qid"]: row for row in read_jsonl(RUN / key / "scores.jsonl")}
            episode, score = episodes[qid], scores.get(qid, {})
            st.divider()
            st.subheader(NAMES[key])
            st.caption(f"搜索 {episode['calls']} 次 · 有证答对 {'是' if score.get('supported_success') else '否'} · 结束状态 {episode['reason']}")
            show_steps(episode["steps"])
