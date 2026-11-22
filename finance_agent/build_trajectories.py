"""Create exactly 700 grounded training actions and issuer-held-out evaluation episodes."""
from collections import Counter, defaultdict
from datetime import date, timedelta
from decimal import Decimal
from itertools import combinations
import argparse
import hashlib
import json
from pathlib import Path
from random import Random

from .expand_sft import read_jsonl
from .prepare_sft import sha256, write_jsonl
from .trajectory_corpus import NAMES
from .trajectory_runtime import Environment, PROTOCOL, dumps, initial_messages, rounded, score_episode
from .trajectory_sources import ISSUERS, OUTPUT

QUOTAS = {
    "train": {"direct": 50, "ratio": 30, "yoy": 20, "cross_document": 40, "three_document": 20,
              "conditional_bridge": 15, "cross_company": 20, "cutoff": 10, "incomparable": 5},
    "validation": {"direct": 16, "ratio": 10, "yoy": 8, "cross_document": 12, "three_document": 8,
                   "conditional_bridge": 8, "cross_company": 8, "cutoff": 6, "incomparable": 4},
    "test": {"direct": 24, "ratio": 12, "yoy": 10, "cross_document": 20, "three_document": 12,
             "conditional_bridge": 10, "cross_company": 12, "cutoff": 12, "incomparable": 8},
}
YEARS = (2023, 2024, 2025)
METRICS = ("revenue", "profit", "cash")


def period(year, kind):
    return f"{year}年{'年报' if kind == 'FY' else '半年报'}"


def search_action(row, metrics=None):
    return {"tool": "search", "query": " ".join(NAMES[m] for m in (metrics or [row["fact"]["metric"]])),
            "company_code": row["company_code"], "year": row["year"], "period": row["period"]}


def operand(row, field="current"):
    return {"id": row["id"], "field": field}


def compute_expected(operation, rows, fields=None):
    # Independent oracle: raw disclosed numbers and explicit arithmetic, not model answers.
    values = [Decimal(row["fact"][field]) for row, field in zip(rows, fields or ["current"] * len(rows))]
    if operation == "difference":
        result = values[0] - values[1]
    elif operation == "growth_pct":
        result = (values[0] / values[1] - 1) * 100
    elif operation == "ratio_pct":
        result = values[0] * 100 / values[1]
    elif operation == "sum":
        result = sum(values)
    elif operation == "mean":
        result = sum(values) / Decimal(len(values))
    elif operation == "argmax":
        result = max(values)
    else:
        raise ValueError(operation)
    return rounded(result)


def make_case(kind, rows, question, searches, *, operation=None, operands_rows=None, fields=None,
              answer_row=None, status="answered", as_of=None):
    rows = list({row["id"]: row for row in rows}.values())
    split_set = {row["split"] for row in rows}
    if len(split_set) != 1:
        raise ValueError("An episode crossed issuer splits")
    split = next(iter(split_set))
    citations = [r["id"] for r in rows] if status != "insufficient" else []
    actions = list(searches)
    expected = {"status": status, "citations": citations}
    calculation = None
    if operation:
        op_rows = operands_rows or rows
        refs = [operand(r, f) for r, f in zip(op_rows, fields or ["current"] * len(op_rows))]
        calculation = {"operation": operation, "operands": refs, "value": compute_expected(operation, op_rows, fields)}
        actions.append({"tool": "calculate", "operation": operation, "operands": refs})
    if status == "answered":
        expected.update(value=calculation["value"] if calculation else rounded(Decimal((answer_row or rows[0])["fact"]["current"])),
                        unit="%" if operation in ("ratio_pct", "growth_pct") else "元")
    final = {"done": True, **expected}
    if calculation:
        final["calculation_id"] = "calc-1"
    actions.append(final)
    key = {"kind": kind, "rows": [r["id"] for r in rows], "operation": operation, "fields": fields,
           "as_of": as_of, "answer_row": (answer_row or {}).get("id")}
    task_hash = hashlib.sha256(dumps(key).encode()).hexdigest()[:16]
    # Separate wording families for held-out evaluation; semantic cases are issuer-disjoint.
    suffix = {"train": "请提供必要的原文引用。", "validation": "请核对来源，并在证据齐全后结束。",
              "test": "请逐步取证，只在资料足够时给出数值和出处。"}[split]
    code_names = {r["company_code"]: r["company"] for r in rows}
    case = {"id": f"{split}-{kind}-{task_hash}", "semantic_key": dumps(key), "protocol": PROTOCOL,
            "split": split, "group": kind, "question": question + suffix,
            "companies": [{"code": c, "name": n} for c, n in sorted(code_names.items())],
            "as_of": as_of or max(r["published_at"] for r in rows), "expected": expected,
            "required_reports": sorted({r["report_id"] for r in rows}),
            "minimum_searches": len(searches), "max_searches": min(5, len(searches) + 1),
            "max_actions": len(actions) + 2, "oracle_actions": actions}
    if calculation:
        case["calculation"] = calculation
    return case


def candidates(facts):
    metrics = [row for row in facts if row["fact"]["kind"] == "metric"]
    by_key = {(r["company_code"], r["year"], r["period"], r["fact"]["metric"]): r for r in metrics}
    for row in metrics:
        name, when, label = row["company"], period(row["year"], row["period"]), NAMES[row["fact"]["metric"]]
        search = search_action(row)
        yield make_case("direct", [row], f"{name}{when}中，当期{label}是多少元？", [search])
        if Decimal(row["fact"]["prior"]) > 0:
            yield make_case("yoy", [row], f"按{name}{when}披露的当期和同期可比列，计算{label}同比变化百分比，四舍五入至两位小数。",
                            [search], operation="growth_pct", operands_rows=[row, row], fields=["current", "prior"])
        cutoff = (date.fromisoformat(row["published_at"]) - timedelta(days=1)).isoformat()
        yield make_case("cutoff", [row], f"截至{cutoff}，能否从{name}{when}查到当期{label}？若报告尚未披露，请停止并说明证据不足。",
                        [search], status="insufficient", as_of=cutoff)
    codes = sorted({r["company_code"] for r in metrics})
    for code in codes:
        name = ISSUERS[code][0]
        for year in YEARS:
            for kind in ("FY", "H1"):
                revenue = by_key[code, year, kind, "revenue"]
                for numerator in ("profit", "cash"):
                    row = by_key[code, year, kind, numerator]
                    yield make_case("ratio", [row, revenue],
                        f"根据{name}{period(year, kind)}，{NAMES[numerator]}占营业收入的百分比是多少？金额先统一成元，结果保留两位小数。",
                        [search_action(revenue, [numerator, "revenue"])], operation="ratio_pct", operands_rows=[row, revenue])
            for metric in METRICS:
                annual, half = by_key[code, year, "FY", metric], by_key[code, year, "H1", metric]
                yield make_case("incomparable", [annual, half],
                    f"请查阅{name}{period(year, 'FY')}与{period(year, 'H1')}，把两份报告各自的当期{NAMES[metric]}直接作同比。"
                    "如果全年与半年口径无法直接作为同比，请取证后停止，不要输出增长率。",
                    [search_action(annual), search_action(half)], status="incomparable")
        for older, newer in combinations(YEARS, 2):
            for kind in ("FY", "H1"):
                for metric in METRICS:
                    first, second = by_key[code, older, kind, metric], by_key[code, newer, kind, metric]
                    for operation in ("difference", "growth_pct"):
                        description = "差额（后者减前者），单位为元" if operation == "difference" else "原披露值变化百分比（后者相对前者），保留两位小数"
                        yield make_case("cross_document", [second, first],
                            f"分别查阅{name}{period(older, kind)}与{period(newer, kind)}，计算两份报告当期{NAMES[metric]}的{description}。"
                            "使用各自当时的原披露值，不把结果称为后份报告披露的同比。",
                            [search_action(first), search_action(second)], operation=operation, operands_rows=[second, first])
            first, second = by_key[code, older, "FY", "revenue"], by_key[code, newer, "FY", "revenue"]
            winner = max([first, second], key=lambda r: Decimal(r["fact"]["current"]))
            for metric in METRICS:
                target = by_key[code, winner["year"], "H1", metric]
                case = make_case("conditional_bridge", [first, second, target],
                    f"比较{name}{older}年和{newer}年年报的当期营业收入；选择营业收入较高的那个年度，"
                    f"再查阅该年度的半年报，给出当期{NAMES[metric]}（元）。请引用决定年度和给出结果的全部证据。",
                    [search_action(first), search_action(second)], operation="argmax", operands_rows=[first, second])
                case["expected"].update(value=rounded(Decimal(target["fact"]["current"])), unit="元")
                case["oracle_actions"][-1].update(case["expected"])
                case["oracle_actions"].insert(-1, search_action(target))
                case.update(minimum_searches=3, max_searches=4, max_actions=7)
                # The selected year is observed from argmax before the dependent search executes.
                yield case
        for kind in ("FY", "H1"):
            for metric in METRICS:
                rows = [by_key[code, year, kind, metric] for year in YEARS]
                for operation in ("sum", "mean"):
                    description = "合计" if operation == "sum" else "算术平均值"
                    yield make_case("three_document", rows,
                        f"分别查阅{name}2023、2024、2025年的{'年报' if kind == 'FY' else '半年报'}，"
                        f"计算三份报告当期{NAMES[metric]}原披露值的{description}，单位为元，保留两位小数。",
                        [search_action(row) for row in rows], operation=operation)
    for left, right in combinations(codes, 2):
        if ISSUERS[left][1] != ISSUERS[right][1]:
            continue
        for year in YEARS:
            for kind in ("FY", "H1"):
                for metric in METRICS:
                    a, b = by_key[left, year, kind, metric], by_key[right, year, kind, metric]
                    for operation in ("difference", "ratio_pct"):
                        description = "差额（前者减后者），单位为元" if operation == "difference" else "比值（前者除以后者）百分比，保留两位小数"
                        yield make_case("cross_company", [a, b],
                            f"分别查阅{a['company']}、{b['company']}的{period(year, kind)}，比较相同报告期的当期{NAMES[metric]}，计算{description}。",
                            [search_action(a), search_action(b)], operation=operation)


def replay(case, output=OUTPUT):
    messages, env, records, observations = initial_messages(case), Environment(case, output), [], []
    for turn, action in enumerate(case["oracle_actions"], 1):
        records.append({"case_id": case["id"], "split": case["split"], "group": case["group"], "turn": turn,
                        "action_type": action.get("tool", "finish"), "protocol": PROTOCOL,
                        "prompt": list(messages), "completion": [{"role": "assistant", "content": dumps(action)}]})
        try:
            observation = env.execute(action)
        except (ValueError, KeyError) as error:
            raise ValueError(f"{case['id']} turn {turn}: {error}; observed IDs: {list(env.evidence)}; action: {action}") from error
        messages.append({"role": "assistant", "content": dumps(action)})
        if not action.get("done"):
            observations.append(observation)
            messages.append({"role": "user", "content": "工具观察：" + dumps(observation)})
    final = case["oracle_actions"][-1]
    checks = score_episode(case, final, case["oracle_actions"], observations)
    if not checks["passed"] or checks["extra_searches"]:
        raise ValueError(f"{case['id']}: replay verification failed: {checks}")
    if case["group"] == "conditional_bridge":
        selected = next(o["selected_year"] for o in observations if o.get("operation") == "argmax")
        if selected != case["oracle_actions"][-2]["year"]:
            raise ValueError("Dependent search ignored the observed comparison")
    audit = {"case_id": case["id"], "split": case["split"], "group": case["group"],
             "actions": case["oracle_actions"], "observations": observations, "checks": checks}
    return records, audit


def build(output=OUTPUT):
    facts = read_jsonl(output / "fact-catalog.jsonl")
    pools = defaultdict(list)
    seen = set()
    for case in candidates(facts):
        if case["semantic_key"] in seen:
            raise ValueError("Duplicate semantic task")
        seen.add(case["semantic_key"])
        pools[case["split"], case["group"]].append(case)
    rng = Random(20261113)
    splits, manifest, fingerprints = {}, {}, set()
    for split, quotas in QUOTAS.items():
        cases, actions, trajectories = [], [], []
        for group, count in quotas.items():
            pool = pools[split, group]
            rng.shuffle(pool)
            if len(pool) < count:
                raise ValueError(f"Not enough distinct {split}/{group}: {len(pool)} < {count}")
            # Replay first; low-quality candidates are errors rather than silently padding with paraphrases.
            for case in pool[:count]:
                records, audit = replay(case, output)
                for record in records:
                    fingerprint = hashlib.sha256(dumps([record["prompt"], record["completion"]]).encode()).hexdigest()
                    if fingerprint in fingerprints:
                        raise ValueError(f"Duplicate prompt and completion: {case['id']} / {record['turn']}")
                    fingerprints.add(fingerprint)
                cases.append(case)
                actions.extend(records)
                trajectories.append(audit)
        rng.shuffle(cases)
        by_id = {case["id"]: i for i, case in enumerate(cases)}
        actions.sort(key=lambda r: (by_id[r["case_id"]], r["turn"]))
        trajectories.sort(key=lambda r: by_id[r["case_id"]])
        splits[split] = cases
        write_jsonl(output / f"{split}-questions.jsonl", cases)
        write_jsonl(output / f"{split}-actions.jsonl", actions)
        write_jsonl(output / f"{split}-trajectories.jsonl", trajectories)
        manifest[split] = {"questions": len(cases), "actions": len(actions),
            "by_group": dict(Counter(c["group"] for c in cases)),
            "by_action_type": dict(Counter(r["action_type"] for r in actions)),
            "companies": sorted({company["code"] for c in cases for company in c["companies"]}),
            "reports": sorted({report for c in cases for report in c["required_reports"]}),
            "all_reference_replays_passed": all(r["checks"]["passed"] for r in trajectories)}
        print(split, manifest[split], flush=True)
    if manifest["train"]["actions"] != 700:
        raise ValueError(f"Expected exactly 700 training actions, got {manifest['train']['actions']}")
    for a, b in combinations(splits, 2):
        for field in ("companies", "reports"):
            if set(manifest[a][field]) & set(manifest[b][field]):
                raise ValueError(f"Split leakage: {a}/{b}/{field}")
    files = [p for p in output.iterdir() if p.name.endswith((".jsonl", ".sqlite3")) or p.name in ("sources.json", "corpus-manifest.json")]
    result = {"version": 1, "protocol": PROTOCOL, "seed": 20261113,
        "origin": "Programmatically constructed questions, deterministic expert actions, real PDF retrieval replay; no LLM-generated gold answers.",
        "splits": manifest, "files_sha256": {p.name: sha256(p) for p in sorted(files)},
        "code_sha256": {name: sha256(Path(__file__).parent / name) for name in (
            "trajectory_sources.py", "trajectory_corpus.py", "trajectory_runtime.py", "build_trajectories.py", "evaluate_trajectories.py", "inspect_trajectories.py")},
        "limitations": ["Template-generated, not independent human annotations.", "Only three audited summary metrics; product evidence is indexed but not yet supervised.",
            "Issuer/report-disjoint evaluation uses shared reasoning families, not a fully template-disjoint benchmark.",
            "Reference replay success validates data and tools, not a trained model.",
            "The 700 rows use filing-tools-v1 and are incompatible with the old 48-token query-only pilot without its runner being updated."]}
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    build(parser.parse_args().output)
