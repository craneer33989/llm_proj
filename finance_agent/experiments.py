"""Read a saved LoRA pilot and derive task metrics from its per-case traces."""
from collections import Counter
import json

from .filings import RUNTIME

PILOT = RUNTIME / "sft-pilot"


def task_metrics(rows):
    summary = {}
    for group in ("direct", "bridge"):
        subset = [row for row in rows if row["case"]["group"] == group]
        summary[group] = {"cases": len(subset)} | {
            key: sum(bool(row["checks"].get(key)) for row in subset)
            for key in ("passed", "second_search", "second_hop_hit", "stopped_after_first_hop")
        }
    return summary


def training_experiment(path=PILOT):
    def read(name):
        return json.loads((path / name).read_text(encoding="utf-8"))

    report, before, after = read("report.json"), read("before.json"), read("after.json")
    previous = {row["case"]["id"]: row for row in before["rows"]}
    current = {row["case"]["id"]: row for row in after["rows"]}
    if len(previous) != len(before["rows"]) or len(current) != len(after["rows"]) or previous.keys() != current.keys():
        raise ValueError("Before and after evaluation must contain the same unique cases")
    if report["rule_fallback"] or before["rule_fallback"] or after["rule_fallback"]:
        raise ValueError("The pilot dashboard requires unassisted evaluation records")
    for key, row in previous.items():
        if row["case"] != current[key]["case"]:
            raise ValueError(f"{key}: before and after tasks differ")
        if any(step.get("fallback_reason") for stage in (row, current[key]) for step in stage["answer"]["trace"]):
            raise ValueError("A trace contains rule-assisted search")
    training = report["training"]
    if training["updates"] != len(training["losses"]):
        raise ValueError("The training summary and update log disagree")
    group_results = {
        "训练前": task_metrics(before["rows"]),
        "训练后（适配器重载）": task_metrics(after["rows"]),
    }
    transitions = Counter(
        "改善" if not row["checks"]["passed"] and current[key]["checks"]["passed"]
        else "退步" if row["checks"]["passed"] and not current[key]["checks"]["passed"]
        else "保持通过" if current[key]["checks"]["passed"] else "仍未通过"
        for key, row in previous.items()
    )
    report.pop("adapter_file", None)
    return {
        "report": report, "groups": group_results, "transitions": dict(transitions),
        "cases": [{"case": row["case"], "before": row, "after": current[key]} for key, row in previous.items()],
    }


def autonomous_experiment(run="sft-autonomous"):
    if run not in ("sft-autonomous", "sft-autonomous-epoch2"):
        raise ValueError("Unknown autonomous experiment")
    path = RUNTIME / run
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    stages = [json.loads((path / f"{stage}.json").read_text(encoding="utf-8")) for stage in ("before", "after")]
    previous, current = [{row["case"]["id"]: row for row in stage["rows"]} for stage in stages]
    if previous.keys() != current.keys():
        raise ValueError("Unhinted evaluation sets differ")
    pairs, transitions = [], Counter()
    for key, row in previous.items():
        other = current[key]
        if row["case"] != other["case"]:
            raise ValueError("Unhinted evaluation cases differ")
        label = ("改善" if not row["checks"]["passed"] and other["checks"]["passed"]
                 else "退步" if row["checks"]["passed"] and not other["checks"]["passed"]
                 else "保持通过" if other["checks"]["passed"] else "仍未通过")
        transitions[label] += 1
        def display(record):
            return dict(record, answer=dict(record["answer"], trace=record["trace"], search_count=record["search_count"]))
        pairs.append({"case": dict(row["case"], group=row["case"]["category"]),
                      "before": display(row), "after": display(other)})
    return {"report": report, "groups": dict(zip(("训练前", "训练后（适配器重载）"), (stage["summary"] for stage in stages))),
            "transitions": dict(transitions), "cases": pairs}
