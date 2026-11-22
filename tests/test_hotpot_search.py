import json

import pytest

from hotpot_search.environment import SearchSession
from hotpot_search.prepare import make_document, select, validate_row, write_jsonl
from hotpot_search.search import SearchEngine, build_index


def test_support_title_and_question_split_isolation():
    candidates = []
    for i in range(90):
        for kind in ("bridge", "comparison"):
            for level in ("easy", "medium", "hard"):
                candidates.append({"id": f"{i}-{kind}-{level}", "question": f"Q {i} {kind} {level}",
                                   "type": kind, "level": level, "support_titles": [f"shared-{i}", f"other-{i}"]})
    chosen = select(candidates, {"sft": 30, "rl": 30, "dev": 10}, 123)
    assert chosen == select(candidates, {"sft": 30, "rl": 30, "dev": 10}, 123)
    titles = {s: {t for row in rs for t in row["support_titles"]} for s, rs in chosen.items()}
    assert not titles["sft"] & titles["rl"]
    assert not titles["dev"] & (titles["sft"] | titles["rl"])
    assert sum(r["type"] == "comparison" for r in chosen["sft"]) == 6


def test_chunking_preserves_long_support_sentence():
    sentences = ["short opening", " ".join(f"w{i}" for i in range(350)), "ending"]
    doc, chunks = make_document("T", sentences, max_words=100)
    support = " ".join(c["text"] for c in chunks if 1 in c["sentence_ids"])
    assert sentences[1] in support
    assert all(c["doc_id"] == doc["id"] for c in chunks)
    assert max(len(c["text"].split()) for c in chunks) <= 100


def test_invalid_support_is_rejected():
    row = {"question": "Q", "answer": "A", "type": "bridge", "level": "hard",
           "context": {"title": ["A", "B"], "sentences": [["x"], ["y"]]},
           "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 3]}}
    assert validate_row(row) == "missing_support_sentence"


@pytest.fixture
def engine(tmp_path):
    documents, chunks = [], []
    for title, text in [("Film Atlas", "Film Atlas was directed by Mira Finch."),
                        ("Mira Finch", "Mira Finch was born in Cedar City."),
                        ("Unrelated", "A bicycle has two wheels.")]:
        document, parts = make_document(title, [text])
        documents.append(document)
        chunks.extend(parts)
    write_jsonl(tmp_path / "documents.jsonl", documents)
    write_jsonl(tmp_path / "corpus.jsonl", chunks)
    build_index(tmp_path, dense=False)
    instance = SearchEngine(tmp_path)
    yield instance
    instance.close()


def test_real_two_step_tool_flow_and_budget(engine):
    session = SearchSession(engine, "Where was the director of Film Atlas born?", max_calls=3, mode="bm25")
    first = session.call("local_search", {"query": "Film Atlas", "top_k": 1})
    assert "Mira Finch" in first["results"][0]["text"]
    second = session.call("local_search", {"query": "Mira Finch born", "top_k": 1})
    assert "Cedar City" in second["results"][0]["text"]
    doc_id = second["results"][0]["doc_id"]
    assert "Cedar City" in session.call("read_document", {"document_id": doc_id})["text"]
    assert session.call("local_search", {"query": "x"})["error"] == "tool_budget_exhausted"
    assert "answer" not in json.dumps(first)


def test_unknown_documents_and_fts_special_characters(engine):
    session = SearchSession(engine, "Q", mode="bm25")
    assert "error" in session.call("read_document", {"document_id": "private/dev.gold.jsonl"})
    assert engine.search('" OR * NEAR(foo) --', mode="bm25")["query"]
    with pytest.raises(ValueError):
        engine.search("   ", mode="bm25")
    with pytest.raises(ValueError):
        engine.search("Mira", mode="invalid")
    with pytest.raises(ValueError):
        engine.search("Mira", mode="dense")


def test_corpus_mutation_invalidates_index(engine):
    with (engine.data / "corpus.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        SearchEngine(engine.data)
