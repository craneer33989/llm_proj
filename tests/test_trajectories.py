"""Guard data isolation, financial column semantics, and rollout grading."""
from copy import deepcopy
from decimal import Decimal
from itertools import combinations
import json

import pytest

from finance_agent.expand_sft import read_jsonl
from finance_agent.trajectory_corpus import extract_metric
from finance_agent.trajectory_runtime import Environment, initial_messages, score_episode
from finance_agent.trajectory_sources import OUTPUT
from finance_agent.evaluate_trajectories import rollout


def report():
    return {"id": "sample-2024-FY"}


def test_units_and_adjusted_column_are_read_from_table_not_company():
    pages = ["主要会计数据 单位：百万元\n2024年 2023年 本期比上年增减(%)\n调整前 调整后\n"
             "营业收入 120 90 100 20.00\n"]
    row = extract_metric(report(), pages, "revenue")
    assert Decimal(row["fact"]["current"]) == 120_000_000
    assert Decimal(row["fact"]["prior"]) == 100_000_000
    assert row["extraction_check"]["prior_column"] == 2


def test_parentheses_mean_negative_growth_and_wrapped_units_are_supported():
    pages = ["主要会计数据\n营业收入（万\n元） 90 100 (10.00%) 95\n"]
    row = extract_metric(report(), pages, "revenue")
    assert row["fact"]["yoy"] == "-10.00"
    assert Decimal(row["fact"]["current"]) == 900_000


def test_nonpositive_prior_is_retained_without_fabricating_yoy():
    pages = ["主要会计数据 单位：元\n上年同期增减(%)\n经营活动产生的现金流量净额 100 -20 不适用\n"]
    row = extract_metric(report(), pages, "cash")
    assert row["fact"]["prior"] == "-20"
    assert row["fact"]["yoy"] is None


def test_inconsistent_percentage_fails_closed():
    with pytest.raises(ValueError, match="differ"):
        extract_metric(report(), ["主要会计数据\n营业收入（元） 120 100 30.00% 90\n"], "revenue")


@pytest.fixture(scope="module")
def dataset():
    if not (OUTPUT / "manifest.json").exists():
        pytest.skip("Build the trajectory dataset before integration checks")
    return {split: read_jsonl(OUTPUT / f"{split}-questions.jsonl") for split in ("train", "validation", "test")}


def test_exact_action_budget_and_issuer_report_isolation(dataset):
    assert len(read_jsonl(OUTPUT / "train-actions.jsonl")) == 700
    assert len(dataset["validation"]) == 80
    assert len(dataset["test"]) == 120
    for a, b in combinations(dataset, 2):
        assert {c["semantic_key"] for c in dataset[a]}.isdisjoint(c["semantic_key"] for c in dataset[b])
        assert {r for c in dataset[a] for r in c["required_reports"]}.isdisjoint(r for c in dataset[b] for r in c["required_reports"])
        assert {i["code"] for c in dataset[a] for i in c["companies"]}.isdisjoint(i["code"] for c in dataset[b] for i in c["companies"])
    assert all(len({c["group"] for c in cases}) == 9 for cases in dataset.values())


def test_gold_never_enters_initial_model_messages(dataset):
    case = deepcopy(dataset["test"][0])
    before = initial_messages(case)
    case["expected"] = {"value": "GOLD_SENTINEL"}
    case["oracle_actions"] = [{"query": "ORACLE_SENTINEL"}]
    case["calculation"] = {"value": "ANSWER_SENTINEL"}
    assert initial_messages(case) == before
    assert "仍需查找" not in json.dumps(before, ensure_ascii=False)


def test_full_reference_replay_and_deliberate_early_stop_are_distinguished(dataset):
    case = next(c for c in dataset["test"] if c["group"] == "conditional_bridge")
    assert rollout(case, "reference")["checks"]["passed"]
    bad = rollout(case, "early-stop")
    assert not bad["checks"]["passed"]
    assert bad["checks"]["premature_stop"]


def test_unobserved_citations_and_calculator_operands_are_rejected(dataset):
    case = dataset["test"][0]
    env = Environment(case)
    with pytest.raises(ValueError, match="observed"):
        env.execute({"done": True, "status": "answered", "value": "1", "unit": "元", "citations": ["invented"]})
    with pytest.raises(ValueError, match="observed"):
        env.execute({"tool": "calculate", "operation": "sum", "operands": [
            {"id": "invented", "field": "current"}, {"id": "invented", "field": "current"}]})


@pytest.mark.parametrize("raw", ['not JSON', '[]', '{"tool":"unknown"}', '{"done":true,"status":"answered","citations":[]}'])
def test_bad_model_actions_are_counted_as_failures(dataset, raw):
    row = rollout(dataset["test"][0], "model", generate=lambda _: raw)
    assert not row["checks"]["passed"]
    assert row["invalid"]


def test_wrong_numeric_answer_and_missing_calculation_are_rejected(dataset):
    case = next(c for c in dataset["test"] if c["group"] == "ratio")
    correct = rollout(case, "reference")
    final = deepcopy(correct["actions"][-1])
    final["value"] = "-123.00"
    assert not score_episode(case, final, correct["actions"][:-1] + [final], correct["observations"])["passed"]
    final = deepcopy(correct["actions"][-1])
    final.pop("calculation_id")
    assert not score_episode(case, final, correct["actions"][:-1] + [final], correct["observations"])["passed"]


def test_unnecessary_search_after_evidence_is_complete_fails_stop_check(dataset):
    case = deepcopy(next(c for c in dataset["test"] if c["group"] == "direct"))
    case["oracle_actions"].insert(1, deepcopy(case["oracle_actions"][0]))
    row = rollout(case, "reference")
    assert row["checks"]["extra_searches"] == 1
    assert not row["checks"]["passed"]


def test_commutative_calculator_operands_accept_equivalent_order(dataset):
    case = deepcopy(next(c for c in dataset["test"] if c["group"] == "three_document"))
    action = next(a for a in case["oracle_actions"] if a.get("tool") == "calculate")
    action["operands"].reverse()
    assert rollout(case, "reference")["checks"]["passed"]


def test_final_test_issuers_were_not_used_in_previous_diagnostic(dataset):
    codes = {r["code"] for c in dataset["test"] for r in c["companies"]}
    assert codes.isdisjoint({"000333", "000651", "600690"})


def test_bounded_tool_excerpts_keep_numeric_facts_and_official_source(dataset):
    case = next(c for c in dataset["test"] if c["group"] == "ratio")
    row = rollout(case, "reference")
    observation = row["observations"][0]
    assert len(observation["sources"]) == 1
    assert observation["sources"][0]["url"].startswith("https://static.cninfo.com.cn/")
    assert all(len(hit["contents"]) <= 320 for hit in observation["hits"])
    fact_rows = [r for r in observation["hits"] if r["fact"] and r["fact"]["kind"] == "metric"]
    assert len(fact_rows) >= 2
    assert all(r["fact"]["unit"] == "元" and "prior" in r["fact"] for r in fact_rows)
