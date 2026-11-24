import csv
import json
from types import SimpleNamespace

import pytest

from hotpot_search.reopd import CONFIG, Rollout, parse, render
from hotpot_search.course_view import answer_evidence, answer_grounded
from hotpot_search.reopd_ablation import build_pools, trajectory_prefixes
from hotpot_search import reopd_baseline_pipeline, reopd_final
from hotpot_search.reopd_compare import cohort_summary, paired_interval, review_baseline, validate_adapters
from hotpot_search.reopd_model import MODELS
from hotpot_search.download import sha256
from hotpot_search.reopd_preflight import verify_episodes
from hotpot_search.reopd_run import (
    export_recovery, recovery_review_cases, recovery_rows, recovery_states, resume_batch, reuse_baseline_batches, run_batch, verify_lock,
    write_json, write_rows,
)
from hotpot_search.reopd_train import action_logprobs, approved_prefixes, experiment_schedule, sampled_reverse_kl, update_action
from hotpot_search.teacher_recovery import continue_episode


TASK = {"id": "q", "question": "Where was the director of Film Atlas born?"}
GOLD = {"answer": "Cedar City", "supporting_facts": [
    {"doc_id": "film", "text": "Film Atlas was directed by Mira Finch."},
    {"doc_id": "director", "text": "Mira Finch was born in Cedar City."},
]}


def search(query):
    return "<tool_call>" + json.dumps({"name": "search", "arguments": {"query": query}}) + "</tool_call>"


class Engine:
    meta = {"corpus_sha256": "test"}

    def search(self, query, top_k, mode):
        assert (top_k, mode) == (3, "bm25")
        if query == "down":
            raise RuntimeError("search outage")
        fact = GOLD["supporting_facts"][int(query == "Mira Finch")]
        return {"results": [{"doc_id": fact["doc_id"], "title": query, "text": fact["text"]}]}


def test_protocol_strict_tool_schema_and_roles():
    assert parse(search("Film Atlas")) == ("search", "Film Atlas")
    assert parse(r"Reasoning. \boxed{Cedar City}") == ("finish", "Cedar City")
    for bad in (search(" "), search("x") * 2, search("x") + r"\boxed{x}", r"\boxed{ }",
                "<|im_end|>" + search("x"), '<tool_call>{"name":"read_document"}</tool_call>'):
        assert parse(bad)[0] == "format_error"
    env = Rollout(Engine(), TASK, "rl")
    env.advance(search("Film Atlas"))
    prompt = render(env.messages)
    assert "<tool_response>\nDoc 1(Title: Film Atlas)" in prompt
    assert r"Answer: \boxed{your answer}" in prompt and "Question: Where was the director of Film Atlas born?" in prompt
    assert "Cedar City" not in prompt and "supporting_facts" not in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_flash_recovery_sends_search_history_and_scores_only_locally():
    env = Rollout(Engine(), TASK, "rl")
    env.advance(search("Film Atlas"))
    outputs = iter([search("Mira Finch"), r"\boxed{Cedar City}"])
    requests = []

    def complete(messages, tools, record):
        assert tools is None
        requests.append(messages)
        response = {"model": "deepseek-flash", "choices": [
            {"finish_reason": "stop", "message": {"content": next(outputs)}},
        ]}
        record({"event": "response", "body": response})
        return response

    result = continue_episode(env, SimpleNamespace(complete=complete), GOLD, force_first_search=True)
    assert requests[0][-1] == {"role": "user", "content": f"<tool_response>\n{env.steps[0]['observation']}\n</tool_response>"}
    assert "Your next action must be one search" in requests[0][0]["content"]
    assert "Your next action must be one search" not in requests[1][0]["content"]
    assert "Cedar City" not in json.dumps(requests[0])
    assert "Mira Finch was born in Cedar City." in requests[1][-1]["content"]
    assert result["score"]["supported_success"] and result["review_candidate"]
    assert result["episode"]["calls"] == 2


def test_grounded_answer_requires_labeled_evidence_in_its_source():
    env = Rollout(Engine(), TASK, "rl")
    env.advance(search("Mira Finch"))
    env.advance(r"\boxed{Cedar City}")
    episode, score = env.record(GOLD, 0)
    logged = {**episode, **score}
    assert answer_grounded(logged, GOLD) and not score["supported_success"]
    assert answer_evidence(logged, GOLD, prefix=1)
    unlinked = {**logged, "steps": [{**episode["steps"][0], "doc_ids": ["distractor"]}]}
    assert not answer_evidence(unlinked, GOLD, prefix=1)
    yes_no = {**GOLD, "answer": "yes"}
    assert not answer_evidence(logged, yes_no, prefix=1)


def test_scoring_and_replay_distinguish_guess_from_supported_answer(tmp_path):
    env = Rollout(Engine(), TASK, "rl")
    env.advance(search("Film Atlas"))
    prefix = list(env.steps)
    replay = Rollout(Engine(), TASK, "rl", prefix)
    assert replay.messages == env.messages
    with pytest.raises(ValueError, match="replay mismatch"):
        Rollout(Engine(), TASK, "rl", [{**prefix[0], "observation_hash": "tamper"}])
    env.advance(r"\boxed{Cedar City}")
    episode, score = env.record(GOLD, 0)
    assert score["answer_em"] == 1 and score["early_stop"] and not score["supported_success"]
    assert "early_stop" not in episode and "citations" not in json.dumps(episode)
    replay.advance(search("Mira Finch"))
    replay.advance(r"\boxed{Cedar City}")
    record, supported = replay.record(GOLD, 1)
    assert supported["supported_success"] and supported["support_doc_coverage"] == 1
    injured = {**record, "attempt": 0, "reason": "backend_error", "steps": []}
    attempts = [injured, record, {**record, "attempt": 2}]
    scores = [supported, {**supported, "attempt": 2}]
    [unresolved] = recovery_rows([episode], [score], attempts, scores)
    assert not unresolved["eligible"] and unresolved["teacher_scorable"] == 2
    attempts[0] = {**record, "attempt": 0}
    scores.append({**supported, "attempt": 0, "supported_success": False})
    [target] = recovery_rows([episode], [score], attempts, scores)
    assert target["eligible"] and target["prefix"] == prefix and target["teacher_target"] == search("Mira Finch")
    assert target["recovery_evidence_id"] != unresolved["recovery_evidence_id"]
    store = SimpleNamespace(gold={"rl": {"q": GOLD}})
    [case] = recovery_review_cases(store, [target], [episode], [score], attempts, scores)
    assert case["gold"] == GOLD and case["student_episode"] == episode
    assert [trial["status"] for trial in case["teacher_trials"]] == ["failed", "success", "success"]
    export_recovery(tmp_path, [target])
    audit = tmp_path / "audit.csv"
    pending = audit.read_text(encoding="utf-8")
    audit.write_text(pending.replace("pending,", "approved,"), encoding="utf-8")
    export_recovery(tmp_path, [target])
    metrics = tmp_path / "pool-metrics.json"
    assert json.loads(metrics.read_text(encoding="utf-8"))["reviewed_approved"] == 0
    audit.write_text(pending.replace("pending,", "approved,teacher searched Mira Finch"), encoding="utf-8")
    export_recovery(tmp_path, [target])
    assert json.loads(metrics.read_text(encoding="utf-8"))["reviewed_approved"] == 1
    export_recovery(tmp_path, [unresolved])
    assert "pending," in audit.read_text(encoding="utf-8")


def test_budget_and_backend_failure_are_separate():
    env = Rollout(Engine(), TASK, "rl")
    for _ in range(CONFIG.max_calls + 1):
        env.advance(search("Film Atlas"))
    assert env.reason == "tool_budget_exhausted" and len(env.env.session.trace) == CONFIG.max_calls
    store = SimpleNamespace(tasks={"rl": {"q": TASK}}, gold={"rl": {"q": GOLD}})
    actor = SimpleNamespace(generate=lambda histories, temperature: [search("down")] * len(histories))
    [(episode, score)] = run_batch(Engine(), store, actor, "rl", [("q", 0, [])], 0)
    assert episode["reason"] == "backend_error" and score is None


def test_wrong_without_search_requires_wrong_direct_answer_and_first_step_repair():
    failed = Rollout(Engine(), TASK, "rl")
    failed.advance(r"\boxed{Elsewhere}")
    episode, score = failed.record(GOLD, 0)
    assert list(recovery_states([episode], [score], "wrong_without_search")) == [(episode, [])]
    assert not list(recovery_states([episode], [{**score, "answer_em": 1}], "wrong_without_search"))
    assert not list(recovery_states([episode], [score], "early_stop"))
    repaired = Rollout(Engine(), TASK, "rl")
    repaired.advance(search("Film Atlas"))
    repaired.advance(search("Mira Finch"))
    repaired.advance(r"\boxed{Cedar City}")
    attempts = [repaired.record(GOLD, attempt) for attempt in range(3)]
    teacher, scores = map(list, zip(*attempts, strict=True))
    [candidate] = recovery_rows([episode], [score], teacher, scores, "wrong_without_search")
    assert candidate["eligible"] and candidate["turn"] == 0 and candidate["prefix"] == []
    assert candidate["teacher_target"] == search("Film Atlas")


def test_resume_retries_backend_only_with_stable_seed():
    complete = [{"qid": "done", "attempt": 0, "split": "rl", "reason": "format_error"}, {"qid": "done", "attempt": 0}]
    failed = [{"qid": "retry", "attempt": 1, "split": "rl", "reason": "backend_error"}, None]
    specs = [("done", 0, []), ("retry", 1, [{"raw": "original action"}])]
    observed = []

    def retry(subset, seed):
        observed.append((subset, seed))
        return [[{**failed[0], "reason": "finished"}, {"qid": "retry", "attempt": 1}]]

    for _ in range(2):
        resumed = resume_batch([complete, failed], specs, "rl", 42, retry)
        assert resumed[0] is complete and resumed[1][0]["reason"] == "finished"
    assert observed[0] == observed[1] and observed[0][0] == [specs[1]]


def test_expand_baseline_requires_identical_environment_and_copies_only_full_batches(tmp_path):
    source = tmp_path / "original" / "rl-sft"
    destination = tmp_path / "expanded" / "rl-sft"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    lock = {"protocol_sha256": "v2"}
    write_json(source.parent / "environment.json", lock)
    specs = [(f"q{index}", 0, []) for index in range(5)]
    target = {"specs": specs, "split": "rl", "batch": 2, "seed": 42, "limit": 5}
    write_json(source / "manifest.json", {**target, "specs": specs[:3], "limit": 3})
    batch = [[{"qid": qid, "attempt": attempt, "split": "rl", "reason": "finished"}, {"qid": qid, "attempt": attempt}]
             for qid, attempt, _ in specs[:3]]
    write_json(source / "batch-00000.json", batch[:2])
    write_json(source / "batch-00002.json", batch[2:])
    assert reuse_baseline_batches(source, destination, target, lock) == 2
    assert not (destination / "batch-00002.json").exists()
    assert reuse_baseline_batches(source, destination, target, lock) == 0
    with pytest.raises(ValueError, match="identical inference settings"):
        reuse_baseline_batches(source, destination, target, {"protocol_sha256": "other"})


def test_formal_preflight_replays_full_history_and_rejects_gold_injection():
    tasks = {str(number): {**TASK, "id": str(number)} for number in range(10)}
    store = SimpleNamespace(tasks={"dev": tasks}, gold={"dev": {qid: GOLD for qid in tasks}})
    records = []
    for task in tasks.values():
        env = Rollout(Engine(), task, "dev")
        env.advance(search("Film Atlas"))
        env.advance(search("Mira Finch"))
        env.advance(r"\boxed{Cedar City}")
        records.append(env.record(GOLD, 0))
    episodes, scores = map(list, zip(*records, strict=True))
    specs = [(qid, 0) for qid in tasks]
    assert verify_episodes(Engine(), store, specs, episodes, scores)["passed"]
    report = verify_episodes(Engine(), store, specs, [{**episodes[0], "reason": "backend_error"}, *episodes[1:]], scores[1:])
    assert not report["passed"] and report["replayed"] == 9
    episodes[0]["messages"][1]["content"] += " Reference answer: Cedar City."
    with pytest.raises(ValueError, match="differs from replay"):
        verify_episodes(Engine(), store, specs, episodes, scores)


def test_four_arm_pools_share_ordinary_states_and_match_without_replacement():
    env = Rollout(Engine(), TASK, "rl")
    env.advance(search("Film Atlas"))
    env.advance(search("Mira Finch"))
    env.advance(r"\boxed{Cedar City}")
    episode, score = env.record(GOLD, 0)
    records = {("q", 0): {**episode, **score}}
    assert [row["turn"] for row in trajectory_prefixes(records, "teacher")] == [0, 1, 2]
    assert [row["turn"] for row in trajectory_prefixes(records, "student")] == [1, 2]
    teacher = [{"qid": f"t{number}", "turn": 0, "source": "teacher", "teacher_target": search("x")}
               for number in range(7)]
    recovery = [{"qid": f"r{number}", "turn": 1, "source": "student", "prompt_tokens": length,
                 "teacher_target": search("Mira Finch")} for number, length in enumerate((100, 104))]
    student = [*recovery, *[{"qid": f"u{number}", "turn": turn, "source": "student", "prompt_tokens": length}
                            for number, (turn, length) in enumerate(((1, 101), (1, 108), (2, 104)))]]
    pools = build_pools(teacher, student, recovery, seed=42, ordinary_per_recovery=3)
    assert pools == build_pools(teacher, student, recovery, seed=42, ordinary_per_recovery=3)
    assert pools["A"] == pools["D"] and len(pools["B"]) == 7
    assert [(left, right) for left, right in zip(pools["C"], pools["D"]) if left != right] == [
        (left, right) for left, right in zip(pools["C"], pools["D"]) if right["source"] == "student"]
    assert {row["qid"]: row["prompt_token_gap"] for row in pools["C"] if row["source"] == "student"} == {"u0": 1, "u1": 4}
    with pytest.raises(ValueError, match="Insufficient ordinary student prefixes"):
        build_pools(teacher, student[:-2], recovery, seed=42, ordinary_per_recovery=3)


def test_ablation_cannot_start_without_p3_and_caps_student_updates(tmp_path):
    with pytest.raises(ValueError, match="P4 requires completed P3"):
        experiment_schedule(tmp_path, "C", 42, {})
    used = 0
    budget = 7
    for action in ([1, 2, 3, 4], [5, 6, 7, 8, 9]):
        supervised = update_action(action, used, budget)
        used += len(supervised)
    assert used == budget and supervised == [5, 6, 7]
    assert update_action([1, 2, 3], 0, None) == [1, 2, 3]


def test_ablation_eval_checks_exact_token_budget_and_adapter_digest(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (adapter / name).write_bytes(b"fixture")
        weights[name] = sha256(adapter / name)
    manifest = {"group": "C", "student": MODELS["sft"], "environment": {}, "update_token_budget": 18,
                "ablation_pool_manifest_sha256": "pool", "seed": 42, "max_tokens": 10, "lr": 0.001,
                "rank": 8, "alpha": 16, "pool_sha256": "recovery", "audit_sha256": "audit"}
    metrics = {"adapter_reloaded": True, "adapter_changed": True, "teacher_unchanged": True,
               "adapter_sha256": weights, "student_update_tokens": 18}
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "metrics.json", metrics)
    assert validate_adapters({"C": adapter}, {}, True)[1] == {"C": weights}
    write_json(tmp_path / "metrics.json", {**metrics, "student_update_tokens": 17})
    with pytest.raises(ValueError, match="frozen student token budget"):
        validate_adapters({"C": adapter}, {}, True)
    (adapter / "adapter_model.safetensors").write_bytes(b"different")
    with pytest.raises(ValueError, match="not a verified"):
        validate_adapters({"C": adapter}, {}, True)


def test_cohort_report_keeps_fixed_denominator_with_unresolved_episodes(tmp_path):
    paths = {name: tmp_path / name for name in ("base", "D")}
    for name, path in paths.items():
        path.mkdir()
        write_rows(path / "episodes.jsonl", [{"qid": "q", "attempt": 0, "reason": "finished", "calls": 1},
                   {"qid": "other", "attempt": 0, "reason": "backend_error", "calls": 0}])
        write_rows(path / "scores.jsonl", [{"qid": "q", "attempt": 0, "supported_success": name == "base",
                   "answer_em": 1, "early_stop": False, "support_doc_coverage": 1}])
    cohort_summary(paths, tmp_path, fixed_prefixes=True)
    summary = json.loads((tmp_path / "cohort_metrics.json").read_text(encoding="utf-8"))
    assert "natural_early_stop_count" not in summary
    assert summary["arms"]["D"]["requested"] == 2 and summary["arms"]["D"]["easy_regressions"] == 1
    assert json.loads((tmp_path / "changed_cases.jsonl").read_text(encoding="utf-8"))["change"] == "regressed"


def test_final_freeze_uses_two_matched_dev_seeds_without_opening_test(tmp_path, monkeypatch):
    write_json(tmp_path / "environment.json", {"index": "locked"})
    monkeypatch.setattr(reopd_final, "environment_lock", lambda engine: {"index": "locked"})
    monkeypatch.setattr(reopd_final, "code_identity", lambda: {"code": "locked"})
    monkeypatch.setattr(reopd_final, "base_identity", lambda: {"weights": "locked"})
    dev = {"specs": [], "batch": 1, "seed": 20, "max_tokens": 10, "student": list(MODELS["sft"])}
    recipe = {key: 1 for key in ("max_tokens", "lr", "rank", "alpha", "pool_sha256", "audit_sha256")}

    def validated(run, budget, seed, lock, failure_type):
        assert failure_type == "wrong_without_search"
        return dev, {"seed": seed, "recipe": recipe, "dev_sha256": {"fixed-cohort.json": "same"}}

    monkeypatch.setattr(reopd_final, "validated_dev", validated)
    plan = reopd_final.frozen_plan(None, tmp_path, 100, [43, 42], "wrong_without_search")
    assert plan["primary"]["seed"] == 42 and plan["replicate"]["seed"] == 43
    assert plan["failure_type"] == "wrong_without_search"
    assert reopd_final.arm_paths(tmp_path, 100, 42, plan["failure_type"])["D"].parent.name.endswith("-wrong-without-search")
    assert not (tmp_path / "test.jsonl").exists()
    with pytest.raises(ValueError, match="distinct training seeds"):
        reopd_final.frozen_plan(None, tmp_path, 100, [42, 42])


def test_final_rejects_reopening_completed_test_before_loading_questions(tmp_path):
    out = tmp_path / "final-test"
    out.mkdir()
    write_json(tmp_path / "test-freeze.json", {"plan": "frozen"})
    write_json(out / "complete.json", {"complete": True})
    with pytest.raises(ValueError, match="already completed"):
        reopd_final.evaluate(None, tmp_path, tmp_path, {})


def test_final_interpretation_requires_paired_intervals_and_full_cohorts():
    def cohorts(size):
        return {"arms": {name: {"requested": size, "distinct_questions": size, "scorable": size}
                         for name in ("base", *reopd_final.ARMS)}}

    def contrasts(size):
        return {"comparisons": {f"D_minus_{arm}": {"supported_success": {"paired_prefixes": size, "interval_95": [.1, .4]}}
                                for arm in ("A", "B", "C")}}

    fixed, full = cohorts(20), cohorts(100)
    pairs, end = contrasts(20), contrasts(100)
    policy = reopd_final.REPORT_POLICY
    assert reopd_final.interpret_results(fixed, pairs, full, end, policy) == "directed_reopd_supported_on_frozen_cohort"
    end["comparisons"]["D_minus_A"]["supported_success"]["interval_95"] = [-.02, .08]
    assert reopd_final.interpret_results(fixed, pairs, full, end, policy) == "fixed_prefix_gain_without_clear_end_to_end_gain"
    pairs["comparisons"]["D_minus_A"]["supported_success"]["interval_95"] = [0, .1]
    assert reopd_final.interpret_results(fixed, pairs, full, end, policy) == "prefer_directed_sft_pending_evidence_over_a"
    fixed["arms"]["D"]["scorable"] = 19
    assert reopd_final.interpret_results(fixed, pairs, full, end, policy) == "insufficient_fixed_evidence"


def test_baseline_pipeline_requires_p0_before_formal_baselines(tmp_path, monkeypatch):
    actions = []
    monkeypatch.setattr(reopd_baseline_pipeline, "checkpoint", lambda name: actions.append(("checkpoint", name)))

    def execute(name, *args):
        actions.append((name, *args))
        if name == "reopd_preflight":
            raise RuntimeError("P0 not accepted")

    monkeypatch.setattr(reopd_baseline_pipeline, "execute", execute)
    with pytest.raises(RuntimeError, match="P0 not accepted"):
        reopd_baseline_pipeline.pipeline(tmp_path, tmp_path / "smoke", tmp_path / "run", 4, 512, 42)
    assert [action[0] for action in actions] == ["checkpoint", "reopd_run", "reopd_run", "reopd_preflight"]


def test_run_resume_lock_and_review_gate(tmp_path):
    lock = tmp_path / "manifest.json"
    verify_lock(lock, {"specs": [("q", 0, [])]})
    verify_lock(lock, {"specs": [("q", 0, [])]})
    with pytest.raises(ValueError, match="changed"):
        verify_lock(lock, {"specs": [("q", 1, [])]})
    with (tmp_path / "audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerows([["qid", "source_trajectory_id", "human_review"], ["q", "source", "approved"]])
    (tmp_path / "prefixes.jsonl").write_text(json.dumps({"qid": "q", "source_trajectory_id": "stale"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0/20"):
        approved_prefixes(tmp_path)


def test_action_only_causal_alignment_and_reverse_kl():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(vocab_size=24, hidden_size=16, intermediate_size=24,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    student = transformers.Qwen3ForCausalLM(config)
    teacher = transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False)
    prompt, action = torch.tensor([1, 2, 3]), torch.tensor([4, 5])
    before = student.lm_head.weight.detach().clone()
    teacher_before = teacher.lm_head.weight.detach().clone()
    student_lp = action_logprobs(student, prompt, action)
    full = student(input_ids=torch.cat((prompt, action)).unsqueeze(0)).logits[0].log_softmax(dim=-1)
    assert torch.allclose(student_lp, full[[2, 3], action], atol=1e-6)
    teacher_lp = action_logprobs(teacher, prompt, action)
    sampled_reverse_kl(student_lp, teacher_lp).backward()
    torch.optim.SGD(student.parameters(), lr=0.01).step()
    assert not torch.equal(before, student.lm_head.weight)
    assert torch.equal(teacher_before, teacher.lm_head.weight)
    assert teacher.lm_head.weight.grad is None


def test_bootstrap_clusters_by_question_and_keeps_missing_scores_separate():
    pairs = [({"qid": "x", "attempt": attempt, "success": 0}, {"qid": "x", "attempt": attempt, "success": 1})
             for attempt in range(2)]
    pairs.extend([({"qid": "y", "attempt": 0, "success": 1}, {"qid": "y", "attempt": 0, "success": 0}),
                  ({"qid": "z", "attempt": 0}, {"qid": "z", "attempt": 0, "success": 1})])
    result = paired_interval(pairs, "success", seed=1, replicates=200)
    assert result == {"paired_prefixes": 3, "distinct_questions": 2, "delta": 1 / 3, "interval_95": [-1, 1]}


def test_manual_review_sample_excludes_backend_errors_and_keeps_existing_review(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    write_rows(baseline / "episodes.jsonl", [{"qid": "q", "attempt": 0, "split": "rl", "reason": "format_error"},
                                            {"qid": "down", "attempt": 0, "split": "rl", "reason": "backend_error"}])
    write_rows(baseline / "scores.jsonl", [{"qid": "q", "attempt": 0, "supported_success": False}])
    paths = {"sft": baseline}
    store = SimpleNamespace(gold={"rl": {"q": GOLD}})
    review_baseline(paths, tmp_path, store, "rl", 42)
    [case] = (tmp_path / "failure-cases.private.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(case)["gold"] == GOLD
    audit = tmp_path / "failure-audit.csv"
    audit.write_text(audit.read_text(encoding="utf-8").replace("pending,", "format_error,reviewed action"), encoding="utf-8")
    review_baseline(paths, tmp_path, store, "rl", 42)
    assert "format_error,reviewed action" in audit.read_text(encoding="utf-8")
    assert json.loads((tmp_path / "failure-audit-metrics.json").read_text(encoding="utf-8"))["reviewed"] == 1
