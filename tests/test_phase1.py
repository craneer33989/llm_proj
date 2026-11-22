from datetime import date
from decimal import Decimal
import json

import pytest

from finance_agent.agent import ask
from finance_agent.corpus import DB, Scope, retrieve
from finance_agent.evaluate import score
from finance_agent.filings import MANIFEST, ROOT

pytestmark = pytest.mark.skipif(not DB.exists(), reason="Build the official filings index before integration tests.")
CASES = [json.loads(line) for line in (ROOT / "evaluation/questions.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_rules_verify_evidence_and_calculations(case):
    answer = ask(case["question"], case["company_code"], case.get("as_of", "2026-11-10"), planner="rules")
    assert score(case, answer)["passed"], answer
    assert answer["search_count"] <= 2


def test_twelve_official_reports():
    reports = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(reports) == 12
    assert all(len(row["sha256"]) == 64 for row in reports)
    assert all(date.fromisoformat(row["published_at"]) for row in reports)


def test_explicit_question_cutoff_can_only_narrow_request():
    answer = ask("截至2025年8月29日，美的集团2025年半年报的营业收入是多少？",
                 "000333", "2026-11-10", planner="rules")
    assert answer["as_of"] == "2025-08-29"
    assert answer["status"] == "insufficient"
    assert not answer["citations"]


def test_cross_company_is_rejected():
    answer = ask("格力电器2025年半年报的营业收入是多少？", "000333", "2026-11-10", planner="rules")
    assert answer["status"] == "insufficient"
    assert not answer["trace"]


def test_retrieval_uses_actual_disclosure_date():
    hits = retrieve("营业收入", scope=Scope("000333", "2025-08-30", 2025, "H1"))
    assert hits and {row["report_id"] for row in hits} == {"000333-2025-H1"}
    assert not retrieve("营业收入", scope=Scope("000333", "2025-08-29", 2025, "H1"))


def test_yoy_uses_restated_prior_period():
    answer = ask("2025年半年报营业收入同比是多少？", "600690", "2026-11-10", planner="rules")
    assert Decimal(answer["values"]["yoy_percent"]) == Decimal("10.22")
    assert "141,982,482,605.75" in answer["calculation"]


def test_assisted_mode_can_complete_early_stopped_search(monkeypatch):
    from finance_agent import agent

    def premature_finish(question, company, task, hits, generate=None):
        if hits:
            return {"done": True}, '{"done":true}'
        return agent.rule_next(task, hits), '{"query":"分产品 营业收入构成"}'

    monkeypatch.setattr(agent, "model_next", premature_finish)
    case = next(case for case in CASES if case["id"] == "b01")
    raw = ask(case["question"], case["company_code"], "2026-11-10", mode="agent")
    guarded = ask(case["question"], case["company_code"], "2026-11-10", mode="assisted")
    assert raw["failure"] == "second_hop"
    assert score(case, guarded)["passed"]
    assert guarded["search_count"] == 2
    assert guarded["trace"][1]["fallback_reason"] == "premature_finish"
