"""Teacher data must be blind, replayable, resumable and isolated from holdouts."""
from copy import deepcopy
from io import BytesIO
import json
from urllib.error import HTTPError

import pytest

from finance_agent import deepseek_trajectories as teacher
from finance_agent.trajectory_runtime import initial_messages
from finance_agent.trajectory_sources import OUTPUT


@pytest.fixture
def cases():
    if not (OUTPUT / "manifest.json").exists():
        pytest.skip("Build the trajectory corpus for integration checks")
    rows = teacher.read_jsonl(OUTPUT / "train-questions.jsonl")
    return {group: deepcopy(next(c for c in rows if c["group"] == group))
            for group in ("direct", "ratio", "conditional_bridge", "cutoff")}


class FakeClient:
    def __init__(self, actions, *, fail_first=False, finish_reason="stop"):
        self.actions = deepcopy(actions)
        self.calls = []
        self.fail_first = fail_first
        self.finish_reason = finish_reason

    def complete(self, messages, record):
        self.calls.append(deepcopy(messages))
        record({"event": "request", "payload": {"messages": messages}})
        if self.fail_first:
            self.fail_first = False
            raise teacher.APIError("fake network failure")
        action = self.actions.pop(0)
        body = {"id": "fake", "model": "fake-teacher", "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                "choices": [{"finish_reason": self.finish_reason, "message": {
                    "role": "assistant", "content": action if isinstance(action, str) else teacher.dumps(action),
                    "reasoning_content": "PRIVATE_REASONING_SENTINEL"}}]}
        record({"event": "response", "body": json.dumps(body)})
        return body


@pytest.mark.parametrize("group", ["direct", "ratio", "conditional_bridge", "cutoff"])
def test_real_tool_replay_saves_teacher_answers_and_masked_sft(tmp_path, cases, group):
    case = cases[group]
    client = FakeClient(case["oracle_actions"])
    row = teacher.generate_episode(case, 0, OUTPUT, tmp_path, client, teacher.Settings())
    assert row["checks"]["passed"]
    assert row["answer"] == case["oracle_actions"][-1]
    assert row["steps"][0]["response"]["model"] == "fake-teacher"
    assert row["usage"]["completion_tokens"] == 3 * len(row["steps"])
    assert client.calls[0] == initial_messages(case)
    # Changing all privileged labels cannot change any inputs observed by the teacher.
    mutated = deepcopy(case)
    mutated["expected"] = {"status": "insufficient", "citations": [], "secret": "GOLD_SENTINEL"}
    mutated["calculation"] = None
    mutated["oracle_actions"] = [{"secret": "ORACLE_SENTINEL"}]
    second = FakeClient(case["oracle_actions"])
    teacher.generate_episode(mutated, 1, OUTPUT, tmp_path, second, teacher.Settings())
    assert client.calls == second.calls
    summary = teacher.export_results(tmp_path, [case], 1)
    assert summary["sft_trajectories"] == 1
    actions = teacher.read_jsonl(tmp_path / "sft-actions.jsonl")
    assert len(actions) == len(case["oracle_actions"])
    assert all("PRIVATE_REASONING_SENTINEL" not in json.dumps(a) for a in actions)
    assert all("expected" not in a and "oracle_actions" not in a for a in actions)
    assert actions[-1]["completion"] == [{"role": "assistant", "content": teacher.dumps(row["answer"])}]


@pytest.mark.parametrize("kind", ["wrong_answer", "invalid_json", "length", "extra_search", "premature_stop"])
def test_failed_teacher_data_is_kept_but_not_exported_for_sft(tmp_path, cases, kind):
    case = cases["ratio"]
    actions = deepcopy(case["oracle_actions"])
    reason = "stop"
    if kind == "wrong_answer":
        actions[-1]["value"] = "-123456.00"
    elif kind == "invalid_json":
        actions = ["not JSON"]
    elif kind == "length":
        reason = "length"
    elif kind == "extra_search":
        actions.insert(1, deepcopy(actions[0]))
    else:
        actions = [{"done": True, "status": "insufficient", "citations": []}]
    row = teacher.generate_episode(case, 0, OUTPUT, tmp_path, FakeClient(actions, finish_reason=reason), teacher.Settings())
    assert row["outcome"] == "rejected"
    summary = teacher.export_results(tmp_path, [case], 1)
    assert summary["sft_actions"] == 0
    assert len(teacher.read_jsonl(tmp_path / "trajectories.jsonl")) == 1
    assert teacher.read_jsonl(tmp_path / "answers.jsonl")[0]["checks"]["passed"] is False


def test_holdout_never_exported_and_duplicate_trajectories_deduplicated(tmp_path, cases):
    case = cases["direct"]
    settings = teacher.Settings()
    for sample in range(2):
        teacher.generate_episode(case, sample, OUTPUT, tmp_path, FakeClient(case["oracle_actions"]), settings)
    assert teacher.export_results(tmp_path, [case], 2)["sft_trajectories"] == 1
    case["id"] = "holdout-case"
    case["split"] = "test"
    teacher.generate_episode(case, 0, OUTPUT, tmp_path, FakeClient(case["oracle_actions"]), settings)
    assert teacher.export_results(tmp_path, [case], 1)["sft_actions"] == 0


def test_resume_retries_api_errors_and_retains_all_event_usage(tmp_path, cases):
    case = cases["direct"]
    settings = teacher.Settings()
    config = {"samples": 1, "identity": "test"}
    broken = FakeClient(case["oracle_actions"], fail_first=True)
    summary, _ = teacher.run_batch([case], config, OUTPUT, tmp_path, broken, settings, workers=1)
    assert summary["api_error_episodes"] == 1
    client = FakeClient(case["oracle_actions"])
    summary, _ = teacher.run_batch([case], config, OUTPUT, tmp_path, client, settings, workers=1)
    assert summary["passed_episodes"] == 1
    count = len(client.calls)
    assert summary["api_requests_including_retries"] == count + 1
    teacher.run_batch([case], config, OUTPUT, tmp_path, client, settings, workers=1)
    assert len(client.calls) == count
    with pytest.raises(ValueError, match="changed"):
        teacher.run_batch([case], {**config, "identity": "changed"}, OUTPUT, tmp_path, client, settings)
    assert not (tmp_path / "run.lock").exists()


def test_fatal_api_error_stops_queued_episodes(tmp_path, cases):
    class FatalClient:
        calls = 0

        def complete(self, messages, record):
            self.calls += 1
            raise teacher.APIError("DeepSeek HTTP 401", fatal=True)

    client = FatalClient()
    summary, _ = teacher.run_batch([cases["direct"]], {"samples": 4}, OUTPUT, tmp_path, client, teacher.Settings(), workers=1)
    assert client.calls == 1
    assert summary["saved_episodes"] == 1


def test_http_payload_retry_and_credentials_are_not_logged(monkeypatch):
    client = teacher.DeepSeekClient("SECRET_API_KEY", teacher.Settings(retries=1))
    payloads, logs, waits = [], [], []

    def open_request(request):
        payloads.append(json.loads(request.data))
        assert request.get_header("Authorization") == "Bearer SECRET_API_KEY"
        if len(payloads) == 1:
            raise HTTPError(teacher.ENDPOINT, 429, "limited", {"Retry-After": "0"}, BytesIO())
        return BytesIO(json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}).encode())

    monkeypatch.setattr(client, "open", open_request)
    monkeypatch.setattr(teacher.time, "sleep", waits.append)
    client.complete([{"role": "user", "content": "Output JSON"}], logs.append)
    assert len(payloads) == 2 and waits == [0]
    assert payloads[0]["thinking"] == {"type": "enabled"}
    assert payloads[0]["reasoning_effort"] == "high"
    assert "tools" not in payloads[0] and "temperature" not in payloads[0]
    assert "SECRET_API_KEY" not in json.dumps(logs)


def test_http_auth_error_does_not_retry_or_log_server_body(monkeypatch):
    client = teacher.DeepSeekClient("SECRET_API_KEY", teacher.Settings())
    calls, logs = [], []

    def open_request(request):
        calls.append(request)
        raise HTTPError(teacher.ENDPOINT, 401, "bad key", {}, BytesIO(b"SECRET_API_KEY"))

    monkeypatch.setattr(client, "open", open_request)
    with pytest.raises(teacher.APIError, match="401") as error:
        client.complete([], logs.append)
    assert error.value.fatal and len(calls) == 1
    assert "SECRET_API_KEY" not in json.dumps(logs)


def test_dry_run_needs_no_key_and_writes_nothing(tmp_path, monkeypatch, cases):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    destination = tmp_path / "never-created"
    assert teacher.main(["--dry-run", "--limit", "2", "--output", str(destination)]) == 0
    assert not destination.exists()


def test_truncated_event_tail_does_not_eat_next_request(tmp_path, cases):
    case = cases["direct"]
    path = tmp_path / "events" / (teacher.job_key(case["id"], 0) + ".jsonl")
    path.parent.mkdir()
    path.write_text('{"event":"partial', encoding="utf-8")
    teacher.generate_episode(case, 0, OUTPUT, tmp_path, FakeClient(case["oracle_actions"]), teacher.Settings())
    summary = teacher.export_results(tmp_path, [case], 1)
    assert summary["api_requests_including_retries"] == len(case["oracle_actions"])


def test_parallel_workers_keep_episode_histories_separate(tmp_path, cases):
    class ContextClient(FakeClient):
        def complete(self, messages, record):
            # Select the next action from the actual number of prior assistant turns.
            index = sum(m["role"] == "assistant" for m in messages)
            return FakeClient([cases["direct"]["oracle_actions"][index]]).complete(messages, record)

    summary, _ = teacher.run_batch([cases["direct"]], {"samples": 4}, OUTPUT, tmp_path,
                                   ContextClient([]), teacher.Settings(), workers=2)
    assert summary["passed_episodes"] == 4
    assert summary["sft_trajectories"] == 1


def test_export_only_does_not_make_requests(tmp_path, cases):
    case = cases["direct"]
    client = FakeClient(case["oracle_actions"])
    config = {"samples": 1}
    settings = teacher.Settings()
    teacher.run_batch([case], config, OUTPUT, tmp_path, client, settings, workers=1)
    count = len(client.calls)
    summary, _ = teacher.run_batch([case], config, OUTPUT, tmp_path, client, settings, export_only=True)
    assert len(client.calls) == count and summary["sft_trajectories"] == 1


def test_custom_questions_validate_split_and_gold_before_api_use(tmp_path, cases):
    case = cases["direct"]
    path = tmp_path / "custom.jsonl"
    case["expected"]["value"] = "NaN"
    path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="numeric answer label"):
        teacher.prepare(OUTPUT, path, "train", None, None, 1, teacher.Settings())
    case["split"] = "test"
    path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requested split"):
        teacher.prepare(OUTPUT, path, "train", None, None, 1, teacher.Settings())
