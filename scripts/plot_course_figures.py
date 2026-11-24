"""Export the classroom figures from saved, validated evaluation artifacts."""
import argparse
import os
from pathlib import Path

from hotpot_search.course_view import ARMS, CATEGORIES, COLORS, failure_counts, load_tables
from hotpot_search.reopd_final import arm_paths, read_json
from hotpot_search.routed_eval import COHORTS, evaluate_cohort

from .course_eval import EXTRA, FINAL, ROOT, RUN, frozen_cohort

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt


def configure_font(font):
    if font is None:
        windows_font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"
        if windows_font.is_file():
            font = windows_font
    if font:
        font_manager.fontManager.addfont(font)
        family = font_manager.FontProperties(fname=font).get_name()
    else:
        installed = {entry.name for entry in font_manager.fontManager.ttflist}
        family = next((name for name in ("Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei") if name in installed), None)
        if family is None:
            raise ValueError("A Chinese font is required. Pass its local path with --font.")
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})


def failure_figure(tables):
    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
    counts = [failure_counts(table) for table in tables.values()]
    starts = [0] * len(counts)
    dark_labels = {"zero_right", "incomplete", "other"}
    for (kind, label), color in zip(CATEGORIES.items(), COLORS, strict=True):
        widths = [count[kind] for count in counts]
        bars = ax.barh(list(map(ARMS.get, tables)), widths, left=starts, label=label, color=color, height=0.67)
        ax.bar_label(bars, labels=[str(n) if n >= 5 else "" for n in widths], label_type="center",
                     color="#182531" if kind in dark_labels else "white", fontsize=11)
        starts = [left + width for left, width in zip(starts, widths, strict=True)]
    ax.invert_yaxis()
    ax.set(xlim=(0, starts[0]), xlabel="题数", title="本地 100 题 · 通过与失败分布")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.18)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.09), ncols=4, frameon=False)
    fig.supxlabel("通用基座、混合检索和路由回放均为事后分析。失败类别描述观察到的结果。", color="#435366", fontsize=10)
    return fig


def transfer_figure(plan):
    fig, (cost, transfer) = plt.subplots(1, 2, figsize=(13, 5.5), width_ratios=(1, 1.65), layout="constrained")
    adapters = arm_paths(RUN, plan["update_tokens"], plan["primary"]["seed"], plan["failure_type"])
    arms = {"A": "A · 定向 SFT", "D": "D · 定向 ReOPD"}
    seconds = [read_json(adapters[arm].parent / "metrics.json")["seconds"] for arm in arms]
    bars = cost.barh(list(arms.values()), seconds, color=("#2a8a74", "#5a668e"), height=0.46)
    cost.bar_label(bars, labels=[f"{value:,.0f} 秒" for value in seconds], padding=8)
    cost.invert_yaxis()
    cost.set(xlim=(0, max(seconds) * 1.29), xlabel="训练墙钟秒", title="相同学生更新 token 预算")
    cost.set_axisbelow(True)
    cost.grid(axis="x", alpha=0.18)

    cohorts = {"local_test": "本地留出", "external_hotpot": "外部 HotpotQA", "external_2wiki": "外部 2Wiki"}
    rows = [evaluate_cohort(COHORTS[name]) for name in cohorts]
    for index, (key, label, color) in enumerate([
        ("base", "原搜索 SFT", "#566c80"), ("directed_sft", "定向 SFT A", "#2a8a74"), ("routed", "路由回放", "#5370ad"),
    ]):
        heights = [100 * row[key]["supported_successes"] / row["questions"] for row in rows]
        bars = transfer.bar([x + (index - 1) * 0.24 for x in range(len(rows))], heights,
                            width=0.22, label=label, color=color)
        transfer.bar_label(bars, fmt="%.0f", padding=3)
    transfer.set(ylim=(0, 52), ylabel="严格有证答对（%）", title="三个同题对照队列")
    transfer.set_xticks(range(len(rows)), list(cohorts.values()))
    transfer.set_axisbelow(True)
    transfer.grid(axis="y", alpha=0.18)
    transfer.legend(loc="upper left", ncols=3, frameon=False, fontsize=9)
    fig.supxlabel("每个队列各 100 题。路由结果来自已有轨迹回放。", color="#435366", fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    configure_font(args.font)
    plan, _, specs = frozen_cohort()
    tables, _ = load_tables(FINAL, EXTRA, {(qid, attempt) for qid, attempt, _ in specs}, require_controls=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, figure in (("failure-attribution", failure_figure(tables)), ("cost-and-transfer", transfer_figure(plan))):
        figure.savefig(args.output / f"{name}.png", dpi=300)
        figure.savefig(args.output / f"{name}.svg")
        plt.close(figure)


if __name__ == "__main__":
    main()
