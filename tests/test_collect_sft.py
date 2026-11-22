"""Guard unattended run costs, credentials, holdouts and teacher provenance."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from finance_agent import collect_sft as collection
from finance_agent import deepseek_trajectories as teacher
from finance_agent.trajectory_sources import OUTPUT


def request(key="one", maximum=100):
    return {"event": "request", "request_id": key,
            "payload": {"messages": [{"role": "user", "content": "test"}], "max_tokens": maximum}}


def test_budget_reserves_and_counts_reported_usage(tmp_path):
    budget = collection.Budget(1, datetime.now(timezone.utc) + timedelta(hours=1), tmp_path / "budget.json")
    budget.observe(request(), "deepseek-v4-pro")
    assert budget.snapshot()["reserved_or_unreported_usd"] > 0
    budget.observe({"event": "response", "request_id": "one", "body": json.dumps({
        "usage": {"prompt_tokens": 20, "completion_tokens": 10}})}, "deepseek-v4-pro")
    assert budget.snapshot()["reserved_or_unreported_usd"] == 0
    assert budget.reported_cost == pytest.approx((20 * 1.32 + 10 * 3.96) / 1_000_000)


def test_no_request_is_dispatched_when_budget_or_deadline_exceeded(tmp_path):
    budget = collection.Budget(.000001, datetime.now(timezone.utc) + timedelta(hours=1), tmp_path / "budget.json")
    with pytest.raises(collection.CollectionStopped, match="cost_limit"):
        budget.observe(request(), "deepseek-v4-pro")
    assert budget.requests == 0
    budget = collection.Budget(1, datetime.now(timezone.utc), tmp_path / "time.json")
    with pytest.raises(collection.CollectionStopped, match="deadline"):
        budget.observe(request(), "deepseek-flash")
    assert budget.requests == 0


def test_unknown_usage_keeps_reservation_and_auth_stops_collection(tmp_path):
    budget = collection.Budget(1, datetime.now(timezone.utc) + timedelta(hours=1), tmp_path / "budget.json")
    budget.observe(request(), "deepseek-v4-pro")
    budget.observe({"event": "request_error", "request_id": "one", "http_status": 401}, "deepseek-v4-pro")
    assert budget.snapshot()["reserved_or_unreported_usd"] > 0
    with pytest.raises(collection.CollectionStopped, match="http_401"):
        budget.observe(request("two"), "deepseek-v4-pro")


def test_key_file_is_parsed_without_echoing_contents(tmp_path, capsys):
    path = tmp_path / "key.txt"
    value = "sk-" + "x" * 30
    path.write_text("DEEPSEEK_API_KEY=" + value, encoding="utf-8-sig")
    assert collection.load_api_key(path) == value
    assert not capsys.readouterr().out
    path.write_text(value + "\nsk-" + "y" * 30, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        collection.load_api_key(path)
    assert value not in str(error.value)


def test_balanced_sampling_is_deterministic_and_keeps_all_cases():
    rows = [{"id": str(i), "group": "a" if i < 8 else "b"} for i in range(10)]
    first = collection.balanced(rows)
    assert first == collection.balanced(rows)
    assert [r["group"] for r in first[:4]] == ["a", "b", "a", "b"]
    assert {r["id"] for r in first} == {r["id"] for r in rows}


def test_merged_sft_uses_only_successful_teacher_actions(tmp_path):
    if not (OUTPUT / "manifest.json").exists():
        pytest.skip("Build trajectory dataset first")
    case = next(c for c in teacher.read_jsonl(OUTPUT / "train-questions.jsonl") if c["group"] == "direct")

    class FakeClient:
        def __init__(self):
            self.actions = iter(case["oracle_actions"])

        def complete(self, messages, record):
            return {"choices": [{"finish_reason": "stop", "message": {"content": teacher.dumps(next(self.actions))}}]}

    phase = tmp_path / "01-pro-pilot"
    teacher.generate_episode(case, 0, OUTPUT, phase, FakeClient(), teacher.Settings())
    summary, solved = collection.merge_phases(tmp_path, [phase], 1)
    assert summary["accepted_teacher_trajectories"] == 1 and case["id"] in solved
    rows = teacher.read_jsonl(tmp_path / "sft" / "train-actions.jsonl")
    assert all(row["provenance"] == "teacher_api_rollout" and row["gold_visible"] is False for row in rows)
    assert all("expected" not in row for row in rows)
    assert all(row["teacher_model"] == "deepseek-v4-pro" for row in rows)
