import json

import pytest

from finance_agent.autonomous import execute, messages
from finance_agent.prepare_autonomous import DATASET, teacher_path
from finance_agent.expand_sft import read_jsonl
from finance_agent.corpus import DB

pytestmark = pytest.mark.skipif(not DB.exists() or not (DATASET / "manifest.json").exists(),
                                reason="Prepare the unhinted dataset first.")


def test_unhinted_dataset_has_search_stop_and_issuer_disjoint_evaluation():
    actions = read_jsonl(DATASET / "train-actions.jsonl")
    cases = read_jsonl(DATASET / "eval-episodes.jsonl")
    assert 100 <= len(actions) + len(cases) <= 200
    assert {row["company_code"] for row in actions}.isdisjoint(row["company_code"] for row in cases)
    assert {row["category"] for row in cases} == {"direct", "bridge", "stop", "missing"}
    assert {row["action_type"] for row in actions} == {"stop", "search"}
    assert not any("仍需查找" in json.dumps(row["prompt"], ensure_ascii=False) or
                   "本轮范围" in json.dumps(row["prompt"], ensure_ascii=False) for row in actions)


def test_complete_evidence_requires_model_to_stop_without_extra_search():
    case = next(row for row in read_jsonl(DATASET / "eval-episodes.jsonl") if row["category"] == "stop")
    stopped = execute(case, lambda _: '{"done":true,"status":"answered"}')
    assert stopped["checks"]["passed"] and stopped["search_count"] == 0
    extra = [case["initial_searches"][0], {"done": True, "status": "answered"}]
    unnecessary = execute(case, lambda _: json.dumps(extra.pop(0)))
    assert unnecessary["checks"]["unnecessary_search"] and not unnecessary["checks"]["passed"]


def test_empty_evidence_cannot_be_marked_as_answered():
    case = next(row for row in read_jsonl(DATASET / "eval-episodes.jsonl") if row["category"] == "missing")
    fabricated = execute(case, lambda _: '{"done":true,"status":"answered"}')
    assert not fabricated["checks"]["passed"]
    assert fabricated["checks"]["premature_stop"]
    path = teacher_path(case)
    verified = execute(case, lambda _: json.dumps(path.pop(0)))
    assert verified["checks"]["passed"]
    assert verified["terminal_status"] == "insufficient"


def test_search_budget_does_not_automatically_finish_when_evidence_is_found():
    case = next(row for row in read_jsonl(DATASET / "eval-episodes.jsonl") if row["category"] == "direct")
    query = teacher_path(case)[0]
    result = execute(case, lambda _: json.dumps(query))
    assert result["search_count"] == 2 and result["terminal_status"] is None
    assert result["trace"][-1]["action"] == "budget_exceeded"
    assert not result["checks"]["passed"]
