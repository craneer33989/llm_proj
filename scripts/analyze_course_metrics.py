"""Generate course tables from complete, same-question episode and score logs."""
import argparse
from collections import Counter

from transformers import AutoTokenizer

from hotpot_search.demo_run import should_retry_with_directed_sft
from hotpot_search.reopd_compare import episode_table
from hotpot_search.reopd_final import arm_paths, read_json
from hotpot_search.reopd_model import checkpoint

from .course_eval import EXTRA, FINAL, RUN, frozen_cohort

ARMS = {
    "base-bm25": "通用 Qwen3-4B（BM25）",
    "base": "原搜索 SFT（BM25）",
    "A": "A · 定向 SFT（BM25）",
    "a-hybrid": "A · 定向 SFT（Hybrid RRF）",
    "B": "B · 教师回放 ReOPD",
    "C": "C · 非定向混合 ReOPD",
    "D": "D · 定向恢复 ReOPD",
    "route": "早答复核路由（轨迹回放）",
}
CATEGORIES = {
    "pass": "严格有证答对", "zero_wrong": "零检索答错", "zero_right": "零检索答对",
    "incomplete": "支持证据未齐", "full_wrong": "证据齐但答错",
    "format": "格式错误", "budget": "预算耗尽", "other": "其他终止",
}


def attribution(episode):
    if episode["supported_success"]:
        return "pass"
    match episode["reason"]:
        case "format_error":
            return "format"
        case "tool_budget_exhausted" | "observation_budget_exhausted":
            return "budget"
        case "finished":
            if episode["calls"] == 0:
                return "zero_right" if episode["answer_em"] else "zero_wrong"
            if episode["support_doc_coverage"] < 1 or episode["support_sentence_recall"] < 1:
                return "incomplete"
            if not episode["answer_em"]:
                return "full_wrong"
            raise ValueError(f"Unexpected strict score for episode {episode['qid']}")
        case _:
            return "other"


def cost(episode, tokenizer):
    raw = [step["raw"] for step in episode["steps"]]
    tokens = tokenizer(raw, add_special_tokens=False)["input_ids"] if raw else []
    return episode["calls"], sum(map(len, tokens))


def summary(episodes, costs):
    n = len(episodes)
    counts = Counter(map(attribution, episodes.values()))
    em = sum(episode["answer_em"] for episode in episodes.values())
    calls = sum(call for call, _ in costs.values()) / n
    tokens = sum(token for _, token in costs.values()) / n
    return counts, em, calls, tokens


def markdown_row(columns):
    return "| " + " | ".join(map(str, columns)) + " |"


def markdown_table(headers, records):
    return "\n".join([markdown_row(headers), markdown_row(["---"] + ["---:"] * (len(headers) - 1)),
                      *map(markdown_row, records)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-controls", action="store_true")
    args = parser.parse_args()
    plan, _, specs = frozen_cohort()
    expected = [(qid, attempt) for qid, attempt, _ in specs]
    adapters = arm_paths(RUN, plan["update_tokens"], plan["primary"]["seed"], plan["failure_type"])
    tables, costs, seconds, pending = {}, {}, {}, []
    tokenizer = AutoTokenizer.from_pretrained(checkpoint("sft"), local_files_only=True)
    for name in ARMS:
        if name == "route":
            continue
        directory = EXTRA if name in {"base-bm25", "a-hybrid"} else FINAL
        path = directory / name
        if directory == EXTRA and not (path / "scores.jsonl").is_file() and not args.require_controls:
            pending.append(ARMS[name])
            continue
        table = episode_table(path)
        if table.keys() != set(expected) or any("supported_success" not in row for row in table.values()):
            raise ValueError(f"{name} does not contain the complete scored test cohort")
        tables[name] = table
        arm_tokenizer = (AutoTokenizer.from_pretrained(checkpoint("base"), local_files_only=True)
                         if name == "base-bm25" else tokenizer)
        costs[name] = {key: cost(table[key], arm_tokenizer) for key in expected}
        train_arm = "A" if name == "a-hybrid" else name
        if train_arm in adapters:
            seconds[name] = read_json(adapters[train_arm].parent / "metrics.json")["seconds"]

    base, directed = tables["base"], tables["A"]
    tables["route"], costs["route"] = {}, {}
    routed = 0
    for key in expected:
        retry = should_retry_with_directed_sft(base[key])
        tables["route"][key] = directed[key] if retry else base[key]
        calls, tokens = costs["base"][key]
        if retry:
            routed += 1
            next_calls, next_tokens = costs["A"][key]
            calls, tokens = calls + next_calls, tokens + next_tokens
        costs["route"][key] = calls, tokens

    print("# 课程展示 · 本地留出集诊断\n")
    if pending:
        print("待完成的对照：" + "、".join(pending) + "。\n")
    print("各行均为相同 100 题。失败类别按实际终止状态和已看到的证据归类，证据齐但答错仍需逐题核查原因。")
    print("正常提交可发生在第 6 次搜索之后。零检索答对只表示轨迹里没有搜索证据。\n")
    failure_rows, efficiency_rows = [], []
    n = len(expected)
    for name, table in tables.items():
        counts, em, calls, tokens = summary(table, costs[name])
        train = f"{seconds[name]:.2f}" if name in seconds else "—"
        if name in {"a-hybrid", "route"}:
            train = f"复用 A（{seconds['A']:.2f}）"
        failure_rows.append([ARMS[name], *(f"{counts[category]}/{n}" for category in CATEGORIES)])
        efficiency_rows.append([ARMS[name], f"{em:.0f}/{n}", f"{counts['pass']}/{n}",
                                f"{calls:.2f}", f"{tokens:.1f}", train])
    print("## 逐题结果与失败分布\n")
    print(markdown_table(["配置", *CATEGORIES.values()], failure_rows))
    print("\n## 准确率与开销\n")
    print(markdown_table(["配置", "答案 EM", "严格有证答对", "搜索次数/题", "估计输出 token/题", "训练墙钟秒"],
                         efficiency_rows))
    print()
    print(f"早答复核触发 {routed}/{n} 题，开销包含原 SFT 首轮和触发后的 A 复核。")
    print("输出 token 由日志中的解码文本重新分词估算。原日志缺少生成 token ID、延迟与输入 token。")
    print("新增对照和路由均在正式评测冻结后分析，模型选择沿用冻结结果。")


if __name__ == "__main__":
    main()
