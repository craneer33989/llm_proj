"""Local Streamlit UI."""
from datetime import date
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st
from finance_agent.dashboard import render_dashboard

API = os.getenv("FINANCE_API_URL", "http://127.0.0.1:8010")
COMPANIES = {"美的集团 000333": "000333", "格力电器 000651": "000651", "海尔智家 600690": "600690"}
EXAMPLES = {
    "跨报告查询": "2024 年年报中收入最高的业务板块，在 2025 年半年报的营业收入和同比变化是多少？",
    "直接查询": "2025 年半年报的归母净利润和同比变化是多少？",
    "口径检查": "请将 2025 年半年报的营业收入和 2024 年年报的营业收入直接算同比。",
}
MODES = {"模型自主检索": "agent", "单次检索基线": "single"}


def request(path, payload=None):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = Request(API + path, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=100) as response:
        return json.load(response)


st.set_page_config(page_title="财报证据台", page_icon="📑", layout="wide")
st.caption("本地财报研究工具 · A 股家电")
st.title("财报证据台")
st.write("看训练过程、比较任务表现，逐题回放模型的查询和停止动作。")
with st.sidebar:
    view = st.radio("查看内容", ["训练与评测", "财报问答"])
if view == "财报问答":
    sidebar = st.sidebar
else:
    sidebar = st.expander("问答资料库", expanded=False)
with sidebar:
    st.header("检索范围")
    company = st.selectbox("公司", COMPANIES)
    as_of = st.date_input("截至披露日", date.today())
    st.caption("当前资料库包含 2024、2025 年报与半年报，问答在本地检索。")
    if view == "财报问答":
        mode = st.radio("检索方式", MODES)
        st.caption("模型自行选择查询、报告范围和结束状态，最多搜索两次。")
    try:
        health = request("/health")
        st.caption("规划方式 · " + ("Qwen3-1.7B" if health["planner"] == "model" else "规则演示"))
        with st.expander("已收录财报"):
            for report in request("/reports"):
                if report["company_code"] == COMPANIES[company]:
                    st.markdown(f"- [{report['title']}]({report['url']}) · {report['published_at']}")
    except (URLError, TimeoutError):
        st.warning("问答 API 尚未就绪。")

if view == "训练与评测":
    experiments = {
        "无下一步提示 · 第二轮 SFT": "sft-autonomous-epoch2",
        "无下一步提示 · 第一轮 SFT": "sft-autonomous",
        "历史记录 · 10 步强提示": "sft-pilot",
    }
    experiment = st.selectbox("实验版本", list(experiments))
    endpoint = experiments[experiment]
    try:
        payload = request(f"/experiments/{endpoint}")
    except HTTPError as error:
        st.error(f"训练记录加载失败。{json.load(error).get('detail')}")
    except (URLError, TimeoutError):
        st.error("训练记录 API 无法连接。请检查启动日志。")
    else:
        render_dashboard(payload, API)
    st.stop()

example = st.selectbox("示例问题", EXAMPLES)
if st.session_state.get("example_selected") != example:
    st.session_state["question"] = EXAMPLES[example]
    st.session_state["example_selected"] = example
with st.form("ask"):
    question = st.text_area("问题", key="question", height=86, max_chars=500)
    submitted = st.form_submit_button("查询并核对出处", type="primary")
if submitted:
    st.session_state.pop("response", None)
    with st.spinner("正在查询财报…"):
        try:
            st.session_state["response"] = request("/ask", {
                "question": question, "company_code": COMPANIES[company], "as_of": as_of.isoformat(),
                "mode": MODES[mode],
            })
        except HTTPError as error:
            st.error(f"查询失败。{json.load(error).get('detail')}")
        except (URLError, TimeoutError):
            st.error("问答 API 无法连接或请求超时。请检查启动日志。")

response = st.session_state.get("response")
if response:
    st.subheader("查询结果")
    st.caption(f"{response['question']} · {response['as_of']}")
    (st.success if response["status"] == "answered" else st.warning)(response["answer"])
    if response.get("calculation"):
        st.caption("计算过程")
        st.code(response["calculation"], language=None)
    st.caption(f"搜索 {response['search_count']} 次 · 耗时 {response['latency_ms'] / 1000:.2f} 秒")
    if any(step.get("fallback_reason") for step in response["trace"]):
        st.info("模型提早结束或未生成有效查询，系统已按当前缺失的证据补查。")
    for cite in response["citations"]:
        local = f"{API}/filings/{cite['report_id']}.pdf#page={cite['page']}"
        with st.expander(f"{cite['company']} · {cite['year']} {cite['period']} · PDF 第 {cite['page']} 页", expanded=True):
            st.write(cite["excerpt"])
            st.markdown(f"[打开本地原文]({local}) · [官方来源]({cite['page_url']}) · 披露于 {cite['published_at']}")
    with st.expander("检索轨迹"):
        st.json(response["trace"])
    st.download_button("导出查询和检索轨迹", json.dumps(response, ensure_ascii=False, indent=2),
                       file_name="filing-trace.json", mime="application/json")
