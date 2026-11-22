"""Independently check multi-hop causality and actual environment observations."""
from copy import deepcopy
import json

import pytest

from finance_agent.audit_teacher_sft import audit, replay_teacher
from finance_agent.deepseek_trajectories import Settings, atomic_json, atomic_text, generate_episode, read_jsonl
from finance_agent.prepare_sft import sha256
from finance_agent.trajectory_runtime import Environment, dumps, initial_messages, score_episode
from finance_agent.trajectory_sources import OUTPUT


@pytest.fixture
def case():
    if not (OUTPUT / "manifest.json").exists():
        pytest.skip("Build trajectory data first")
    return deepcopy(next(c for c in read_jsonl(OUTPUT / "train-questions.jsonl") if c["group"] == "conditional_bridge"))


def execute(case, actions):
    messages, observations, env = initial_messages(case), [], Environment(case, OUTPUT)
    for action in actions:
        observation = env.execute(action)
        messages.append({"role": "assistant", "content": dumps(action)})
        if not action.get("done"):
            messages.append({"role": "user", "content": "工具观察：" + dumps(observation)})
            observations.append(observation)
    return {"messages": messages, "answer": actions[-1]}, observations


def test_correct_conditional_trajectory_reexecutes(case):
    trajectory, _ = execute(case, case["oracle_actions"])
    assert len(replay_teacher(case, trajectory, OUTPUT)) == len(case["oracle_actions"])


def test_guessing_dependent_year_before_comparison_is_rejected(case):
    actions = deepcopy(case["oracle_actions"])
    actions[2], actions[3] = actions[3], actions[2]
    trajectory, observations = execute(case, actions)
    assert score_episode(case, actions[-1], actions, observations)["passed"]
    with pytest.raises(ValueError, match="preceded"):
        replay_teacher(case, trajectory, OUTPUT)


def test_tampered_tool_observation_is_rejected(case):
    trajectory, _ = execute(case, case["oracle_actions"])
    trajectory["messages"][3]["content"] += "invented observation"
    with pytest.raises(ValueError, match="differs"):
        replay_teacher(case, trajectory, OUTPUT)


def test_heldout_evidence_is_never_promoted(case):
    trajectory, _ = execute(case, case["oracle_actions"])
    with pytest.raises(ValueError, match="held-out"):
        replay_teacher(case, trajectory, OUTPUT, train_reports=set())


def test_valid_resampled_alternative_replaces_a_causally_invalid_shorter_path(tmp_path, case):
    run = tmp_path / "run"
    atomic_text(run / "task-pool/questions.jsonl", dumps(case) + "\n")
    atomic_json(run / "task-pool/manifest.json", {
        "source_manifest_sha256": sha256(OUTPUT / "manifest.json"),
        "source_corpus_sha256": sha256(OUTPUT / "corpus.sqlite3"), "heldout_sha256": {}})
    bad_actions = deepcopy(case["oracle_actions"])
    bad_actions[2], bad_actions[3] = bad_actions[3], bad_actions[2]
    bad, _ = execute(case, bad_actions)
    primary = {**bad, "case_id": case["id"], "group": case["group"], "split": "train", "protocol": case["protocol"],
               "teacher_model": "deepseek-flash", "provenance": "teacher_api_rollout", "gold_visible": False,
               "source_phase": "02-flash-bulk"}
    atomic_text(run / "sft/train-trajectories.jsonl", dumps(primary) + "\n")

    class FakeClient:
        def __init__(self):
            self.actions = iter(case["oracle_actions"])

        def complete(self, messages, record):
            return {"choices": [{"finish_reason": "stop", "message": {"content": dumps(next(self.actions))}}]}

    generate_episode(case, 0, OUTPUT, run / "04-pro-retry", FakeClient(), Settings())
    result = audit(run, OUTPUT, tmp_path / "release")
    assert result["teacher_trajectories"] == 1 and result["excluded_tasks"] == 0
    assert result["quarantined_candidates"] == 1
    rows = read_jsonl(tmp_path / "release/train-trajectories.jsonl")
    assert rows[0]["source_phase"] == "04-pro-retry"
