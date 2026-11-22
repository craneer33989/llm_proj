"""Audit evidence integrity and retrieval, not model/agent task success."""
import argparse
import json
from pathlib import Path
import time

from .download import sha256
from .search import DEFAULT_DATA, SearchEngine, rows


def integrity(data):
    data = Path(data)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["file_sha256"].items():
        if sha256(data / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    documents = {d["id"]: d for d in rows(data / "documents.jsonl")}
    chunks = {c["id"]: c for c in rows(data / "corpus.jsonl")}
    ids, questions, titles = set(), set(), {}
    count, evidence_count = 0, 0
    from .prepare import normalized
    for split, size in manifest["splits"].items():
        public = list(rows(data / f"{split}.jsonl"))
        gold = {g["id"]: g for g in rows(data / "private" / f"{split}.gold.jsonl")}
        assert len(public) == len(gold) == size
        titles[split] = set()
        for question in public:
            assert set(question) == {"id", "question"}, "Unexpected public labels"
            assert question["id"] not in ids
            assert normalized(question["question"]) not in questions
            ids.add(question["id"])
            questions.add(normalized(question["question"]))
            label = gold[question["id"]]
            for evidence in label["supporting_facts"]:
                document = documents[evidence["doc_id"]]
                assert document["sentences"][evidence["sentence_id"]] == evidence["text"]
                parts = [chunks[cid] for cid in evidence["chunk_ids"]]
                assert parts
                assert all(p["doc_id"] == document["id"] and evidence["sentence_id"] in p["sentence_ids"] for p in parts)
                reconstructed = " ".join(p["text"] for p in parts)
                assert " ".join(evidence["text"].split()) in " ".join(reconstructed.split()), "Evidence text lost in chunking"
                titles[split].add(normalized(evidence["title"]))
                evidence_count += 1
            count += 1
    for i, a in enumerate(titles):
        for b in list(titles)[i+1:]:
            assert not titles[a] & titles[b], f"Core support title overlap: {a}/{b}"
    assert all(set(c) == {"id", "doc_id", "title", "text", "sentence_ids", "url", "source"} for c in chunks.values())
    return {"passed": True, "questions": count, "verified_supporting_facts": evidence_count,
            "documents": len(documents), "chunks": len(chunks), "all_file_hashes_verified": True,
            "core_support_titles_disjoint": True, "public_labels_absent": True}


def audit(data=DEFAULT_DATA, limit=30, modes=("bm25", "dense", "hybrid"), split="dev"):
    data = Path(data)
    report = {"integrity": integrity(data), "retrieval": {},
              "note": "Retrieval diagnostics only. Title probes use gold titles and are NOT autonomous agent/model scores."}
    questions = list(rows(data / f"{split}.jsonl"))[:limit]
    gold = {g["id"]: g for g in rows(data / "private" / f"{split}.gold.jsonl")}
    engine = SearchEngine(data)
    try:
        for mode in modes:
            entries, title_hits, title_total = [], 0, 0
            engine.search("warmup", 1, mode)
            for question in questions:
                label = gold[question["id"]]
                targets = {e["doc_id"] for e in label["supporting_facts"]}
                started = time.perf_counter()
                result = engine.search(question["question"], 10, mode)
                elapsed = time.perf_counter() - started
                found = {r["doc_id"] for r in result["results"]}
                for doc_id in targets:
                    title = next(e["title"] for e in label["supporting_facts"] if e["doc_id"] == doc_id)
                    probe = engine.search(title, 10, mode)
                    title_hits += doc_id in {r["doc_id"] for r in probe["results"]}
                    title_total += 1
                entries.append({"id": question["id"], "support_documents": len(targets),
                                "hits_at_10": len(targets & found), "all_support_documents_at_10": targets <= found,
                                "latency_seconds": round(elapsed, 4)})
            latencies = sorted(e["latency_seconds"] for e in entries)
            report["retrieval"][mode] = {
                "split": split, "questions": len(entries), "top_k": 10,
                "original_question_support_document_recall": sum(e["hits_at_10"] for e in entries) / sum(e["support_documents"] for e in entries),
                "original_question_all_support_documents_rate": sum(e["all_support_documents_at_10"] for e in entries) / len(entries),
                "oracle_title_probe_recall_at_10": title_hits / title_total,
                "latency_p50_seconds": latencies[len(latencies)//2],
                "latency_p95_seconds": latencies[min(len(latencies)-1, int(.95*len(latencies)))],
                "entries": entries,
            }
    finally:
        engine.close()
    (data / "retrieval_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"integrity": report["integrity"], "retrieval": {m: {k: v for k, v in r.items() if k != "entries"} for m, r in report["retrieval"].items()}}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--modes", default="bm25,dense,hybrid")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    audit(args.data, args.limit, tuple(args.modes.split(",")))
