import json

import pytest

from finance_agent.agent import ask
from finance_agent.corpus import DB
from finance_agent.expand_sft import expand, task_key
from finance_agent.filings import ROOT
from finance_agent.prepare_sft import BASELINE, prepare, seed_records

pytestmark = pytest.mark.skipif(
    not BASELINE.is_file() or not DB.is_file(),
    reason="The verified index and first-phase model baseline are required.",
)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_seed_export_replays_failures_and_keeps_issuer_holdout(tmp_path):
    manifest = prepare(tmp_path)
    train = load_jsonl(tmp_path / "sft-seed.jsonl")
    diagnostic = load_jsonl(tmp_path / "diagnostic-holdout.jsonl")
    audit = load_jsonl(tmp_path / "failure-replays.jsonl")
    assert manifest["replayed_cases"] == len(train) + len(diagnostic) == len(audit) == 15
    assert manifest["failures"] == {"premature_finish": 14, "invalid_action": 1}
    assert manifest["train"]["companies"] == ["000333", "000651"]
    assert {row["company_code"] for row in audit if row["split"] == "diagnostic"} == {"600690"}
    assert {row["case_id"] for row in train}.isdisjoint(row["case_id"] for row in diagnostic)
    example = next(row for row in train if row["case_id"] == "b01")
    assert "智能家居业务" in json.loads(example["completion"][0]["content"])["query"]
    assert "已有证据" in example["prompt"][1]["content"]


def test_seed_export_rejects_changed_evidence():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assisted = baseline["runs"]["assisted"]["rows"]
    example = next(row for row in assisted if row["case"]["id"] == "b01")
    example["answer"]["trace"][-1]["hits"][0]["id"] = "changed-evidence"
    with pytest.raises(ValueError, match="b01: Recorded evidence differs"):
        list(seed_records(baseline))


def test_expanded_data_replays_and_excludes_prior_question_semantics(tmp_path):
    manifest = expand(tmp_path)
    train = load_jsonl(tmp_path / "train-questions.jsonl")
    diagnostic = load_jsonl(tmp_path / "diagnostic-questions.jsonl")
    actions = load_jsonl(tmp_path / "train-actions.jsonl")
    old = {task_key(case) for case in load_jsonl(ROOT / "evaluation/questions.jsonl")}
    new = {task_key(case) for case in train + diagnostic}
    assert len(new) == len(train) + len(diagnostic) == manifest["replayed_questions"]
    assert new.isdisjoint(old)
    assert {case["company_code"] for case in train} == {"000333", "000651"}
    assert {case["company_code"] for case in diagnostic} == {"600690"}
    assert {row["case_id"] for row in actions} == {case["id"] for case in train}
    assert {case["group"] for case in diagnostic} == {"direct", "bridge"}
    assert len(actions) == sum(2 if case["group"] == "bridge" else 1 for case in train)


def test_unavailable_product_rank_is_reported_without_inventing_evidence():
    answer = ask("2024年年报中收入第六高的板块，在2025年年报的营业收入是多少？",
                 "000333", "2026-11-11", planner="rules")
    assert answer["status"] == "insufficient"
    assert answer["failure"] == "rank_unavailable"
