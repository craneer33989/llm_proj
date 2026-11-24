"""Generate course tables from complete, same-question episode and score logs."""
import argparse
from pathlib import Path

from transformers import AutoTokenizer

from hotpot_search.course_view import ARMS, CATEGORIES, answer_grounded, failure_counts, load_tables, route_source, search_calls
from hotpot_search.reopd_final import arm_paths, read_json
from hotpot_search.reopd_model import checkpoint
from hotpot_search.reopd_run import write_json

from .course_eval import EXTRA, FINAL, RUN, frozen_cohort

def output_tokens(episode, tokenizer):
    raw = [step["raw"] for step in episode["steps"]]
    tokens = tokenizer(raw, add_special_tokens=False)["input_ids"] if raw else []
    return sum(map(len, tokens))


def markdown_row(columns):
    return "| " + " | ".join(map(str, columns)) + " |"


def markdown_table(headers, records):
    return "\n".join([markdown_row(headers), markdown_row(["---"] + ["---:"] * (len(headers) - 1)),
                      *map(markdown_row, records)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-controls", action="store_true")
    parser.add_argument("--selected-report", type=Path)
    args = parser.parse_args()
    plan, store, specs = frozen_cohort()
    expected = [(qid, attempt) for qid, attempt, _ in specs]
    adapters = arm_paths(RUN, plan["update_tokens"], plan["primary"]["seed"], plan["failure_type"])
    tables, pending = load_tables(FINAL, EXTRA, set(expected), args.require_controls)
    if args.selected_report and "selected-sft" not in tables:
        parser.error("--selected-report requires a complete selected-sft holdout run")
    tokens, seconds = {}, {}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint("sft"), local_files_only=True)
    for name, table in tables.items():
        if name == "route":
            continue
        arm_tokenizer = (AutoTokenizer.from_pretrained(checkpoint("base"), local_files_only=True)
                         if name == "base-bm25" else tokenizer)
        tokens[name] = {key: output_tokens(table[key], arm_tokenizer) for key in expected}
        if name in adapters:
            seconds[name] = read_json(adapters[name].parent / "metrics.json")["seconds"]
        elif name == "selected-sft":
            seconds[name] = read_json(EXTRA / "selected-sft/manifest.json")["selected_training_seconds"]

    retried = {key for key in expected if route_source(tables["base"][key]) == "A"}
    tokens["route"] = {key: tokens["base"][key] + (tokens["A"][key] if key in retried else 0) for key in expected}

    print("# 课程展示 · 本地留出集诊断\n")
    if pending:
        print("待完成的对照：" + "、".join(pending) + "。\n")
    print("各行均为相同 100 题。失败类别按实际终止状态和已看到的证据归类，证据齐但答错仍需逐题核查原因。")
    print("正常提交可发生在第 6 次搜索之后。零检索答对只表示轨迹里没有搜索证据。\n")
    failure_rows, efficiency_rows = [], []
    report = {}
    n = len(expected)
    for name, table in tables.items():
        counts = failure_counts(table)
        em = sum(row["answer_em"] for row in table.values())
        grounded = sum(answer_grounded(row, store.gold["test"][row["qid"]]) for row in table.values())
        report[name] = {"answer_em": int(em), "answer_grounded": grounded, "strict": counts["pass"]}
        calls = sum(search_calls(tables, name, key) for key in expected) / n
        mean_tokens = sum(tokens[name].values()) / n
        train = f"{seconds[name]:.2f}" if name in seconds else "—"
        if name in {"a-hybrid", "route"}:
            train = f"复用 A（{seconds['A']:.2f}）"
        failure_rows.append([ARMS[name], *(f"{counts[category]}/{n}" for category in CATEGORIES)])
        efficiency_rows.append([ARMS[name], f"{em:.0f}/{n}", f"{grounded}/{n}", f"{counts['pass']}/{n}",
                                f"{calls:.2f}", f"{mean_tokens:.1f}", train])
    print("## 逐题结果与失败分布\n")
    print(markdown_table(["配置", *CATEGORIES.values()], failure_rows))
    print("\n## 准确率与开销\n")
    print(markdown_table(["配置", "答案 EM", "Answer-Grounded", "Strict", "搜索次数/题", "估计输出 token/题", "训练墙钟秒"],
                         efficiency_rows))
    print("\nAnswer-Grounded 要求答案匹配，且可见片段包含带答案或别名的标注支持句，或已看到全部支持句。是非题按全证据判定。")
    print()
    print(f"早答复核触发 {len(retried)}/{n} 题，开销包含原 SFT 首轮和触发后的 A 复核。")
    print("输出 token 由日志中的解码文本重新分词估算。原日志缺少生成 token ID、延迟与输入 token。")
    print("新增对照和路由均在正式评测冻结后分析。")
    if "selected-sft" in tables:
        print("精选 SFT 按开发集门槛入选，留出分数作为事后对照。")
    if args.selected_report:
        selected_table = tables["selected-sft"]
        paired = {}
        for arm in ("base", "A"):
            prior = tables[arm]
            paired[arm] = {"strict_gains": sum(selected_table[key]["supported_success"] and
                                              not prior[key]["supported_success"] for key in expected),
                           "strict_losses": sum(prior[key]["supported_success"] and
                                                not selected_table[key]["supported_success"] for key in expected)}
        write_json(args.selected_report, {"n": n, "post_freeze": True, "full": report, "paired": paired})


if __name__ == "__main__":
    main()
