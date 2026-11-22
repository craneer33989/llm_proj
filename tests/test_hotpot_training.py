import json

import pytest

from hotpot_search.download import sha256
from hotpot_search.prepare import make_document, write_jsonl
from hotpot_search.scoring import answer_metrics, score_episode
from hotpot_search.search import SearchEngine, build_index
from hotpot_search.training import EpisodeConfig, TaskStore, TrainingEpisode, bounded_observation


@pytest.fixture
def training_data(tmp_path):
    docs, chunks = [], []
    for title, text in [("Film Atlas", "Film Atlas was directed by Mira Finch."),
                        ("Mira Finch", "Mira Finch was born in Cedar City."),
                        ("Noise", "An unrelated magazine about bicycles.")]:
        doc, parts = make_document(title, [text])
        docs.append(doc)
        chunks.extend(parts)
    write_jsonl(tmp_path / "documents.jsonl", docs)
    write_jsonl(tmp_path / "corpus.jsonl", chunks)
    (tmp_path / "private").mkdir()
    gold = {"answer": "Cedar City", "supporting_facts": [{"doc_id": d["id"], "text": d["sentences"][0]} for d in docs[:2]]}
    for split in ("sft_seed", "rl", "dev"):
        task = {"id": split + "-id", "question": "Where was the director of Film Atlas born?"}
        write_jsonl(tmp_path / f"{split}.jsonl", [task])
        write_jsonl(tmp_path / "private" / f"{split}.gold.jsonl", [{"id": task["id"], **gold}])
    hashes = {str(p.relative_to(tmp_path)).replace("\\", "/"): sha256(p) for p in tmp_path.rglob("*.jsonl")}
    (tmp_path / "manifest.json").write_text(json.dumps({"file_sha256": hashes}), encoding="utf-8")
    build_index(tmp_path, dense=False)
    return tmp_path, docs, gold


@pytest.fixture
def episode(training_data):
    path, docs, gold = training_data
    engine = SearchEngine(path)
    store = TaskStore(path)
    env = TrainingEpisode(engine, store.task("rl"), "rl", log_dir=path / "logs")
    yield env, docs, gold
    engine.close()


def test_multistep_finish_reward_and_no_reference_in_actor_messages(episode):
    env, docs, gold = episode
    reset = env.reset_response()
    assert "Cedar City" not in json.dumps(reset)
    assert "supporting_facts" not in json.dumps(reset)
    first = env.step("local_search", {"query": "Film Atlas", "top_k": 1})
    assert not first["terminated"] and first["reward"] == 0
    assert "Mira Finch" in first["observation"]["results"][0]["text"]
    env.step("local_search", {"query": "Mira Finch born", "top_k": 1})
    result = env.step("finish", {"answer": "Cedar City", "citations": [d["id"] for d in docs[:2]]})
    assert result["terminated"] and not result["truncated"]
    score = env.evaluate(gold)
    assert score["reward"] == 1 and score["supported_success"]
    assert len(env.session.trace) == 2
    assert "supporting_facts" not in json.dumps(env.messages)
    events = [json.loads(line) for line in env.log_path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event"] == "terminal_score"
    count = len(events)
    assert env.evaluate(gold) == score
    assert len(env.log_path.read_text(encoding="utf-8").splitlines()) == count
    with pytest.raises(ValueError, match="already finished"):
        env.step("local_search", {"query": "x"})


def test_no_forced_minimum_hops_and_unseen_citations_not_rewarded(episode):
    env, docs, gold = episode
    result = env.step("finish", {"answer": "Cedar City", "citations": [docs[0]["id"]]})
    assert result["terminated"]
    metrics = env.evaluate(gold)
    assert metrics["answer_em"] == 1
    assert metrics["cited_support_sentence_recall"] == 0
    assert metrics["unseen_citations"] == 1
    assert not metrics["supported_success"]


def test_invalid_calls_count_and_budget_exhaustion(episode):
    env, docs, gold = episode
    env.config = EpisodeConfig(max_calls=2)
    env.session.max_calls = 2
    assert "error" in env.step("read_document", {"document_id": docs[0]["id"]})["observation"]
    assert "error" in env.step("local_search", {"query": "Film", "top_k": 10})["observation"]
    result = env.step("local_search", {"query": "Film"})
    assert result["truncated"] and env.invalid_calls == 2
    assert env.evaluate(gold)["reward"] == 0


def test_can_finish_after_exact_tool_budget(episode):
    env, _, _ = episode
    env.config = EpisodeConfig(max_calls=1)
    env.session.max_calls = 1
    env.step("local_search", {"query": "Film Atlas", "top_k": 1})
    assert env.step("finish", {"answer": "Unknown", "citations": []})["terminated"]


def test_actual_observations_are_bounded_and_do_not_expose_hidden_text(episode):
    env, _, gold = episode
    env.config = EpisodeConfig(max_observation_chars=512, max_total_observation_chars=512)
    env.session.engine.search = lambda *args: {"results": [{"doc_id": "hidden", "text": "x"*20000}], "query": "q"}
    response = env.step("local_search", {"query": "q"})
    shown = response["observation"]
    assert len(json.dumps(shown, ensure_ascii=False)) <= 512
    assert shown["observation_truncated"]
    assert len(env.exposed_documents["hidden"][0]) < 512
    assert env.step("local_search", {"query": "again"})["truncated"]
    assert env.evaluate(gold)["reward"] == 0


def test_infrastructure_errors_do_not_consume_model_budget(episode):
    env, _, _ = episode
    def unavailable(*args):
        raise RuntimeError("offline")
    env.session.engine.search = unavailable
    result = env.step("local_search", {"query": "q"})
    assert result["observation"]["retryable"]
    assert not env.session.trace and env.invalid_calls == 0


def test_store_rejects_test_split_and_mismatched_question_id(training_data):
    path, _, _ = training_data
    store = TaskStore(path)
    assert store.task("rl", seed=42) == store.task("rl", seed=42)
    with pytest.raises(ValueError, match="held out"):
        store.task("test")
    with pytest.raises(KeyError):
        store.task("rl", "sft_seed-id")
    (path / "rl.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        TaskStore(path)


def test_scoring_and_citation_stuffing():
    assert answer_metrics("The Cedar City", "Cedar City") == (1., 1.)
    assert answer_metrics("yes", "no") == (0., 0.)
    gold = {"answer": "Cedar", "supporting_facts": [{"doc_id": "a", "text": "a fact"}]}
    exposed = {"a": ["a fact"], "noise": ["other"]}
    clean = score_episode(gold, "Cedar", ["a"], exposed)
    stuffed = score_episode(gold, "Cedar", ["a", "noise"], exposed)
    assert clean["reward"] == 1 and stuffed["reward"] < 1
    assert not stuffed["supported_success"]


def test_training_http_api(training_data, tmp_path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from hotpot_search.training_api import create_training_app
    path, _, _ = training_data
    app = create_training_app(path, tmp_path / "episodes", engine_factory=lambda: SearchEngine(path))
    with TestClient(app) as client:
        assert client.get("/health").json()["ready"]
        assert client.post("/reset", json={"split": "test"}).status_code == 422
        response = client.post("/reset", json={"split": "rl"})
        assert response.status_code == 200
        reset = response.json()
        assert "Cedar City" not in response.text
        eid = reset["episode_id"]
        response = client.post(f"/episodes/{eid}/step", json={"name": "finish", "arguments": {"answer": "Cedar City", "citations": []}})
        assert response.status_code == 200 and response.json()["terminated"]
        assert response.json()["info"]["metrics"]["answer_em"] == 1
        assert client.post(f"/episodes/{eid}/step", json={"name": "finish", "arguments": {}}).status_code == 409
        assert client.get(f"/episodes/{eid}/trajectory").json()["terminated"]
        assert client.delete(f"/episodes/{eid}").json()["released"]
        assert client.get(f"/episodes/{eid}/trajectory").status_code == 404
