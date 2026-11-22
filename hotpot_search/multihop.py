"""Label-assisted screening and conservative checks of observed search dependencies.

All gold-based decisions happen outside the actor. Passing does not prove that
every possible one-shot query fails; it documents a dependency under this setup.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

from .client import SearchClient
from .search import DEFAULT_DATA, rows
from .training import TaskStore


def key(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def entity_names(title):
    names = [title, re.sub(r"\s*\([^)]*\)", "", title)]
    return {key(name) for name in names if len(key(name)) >= 4}


def contains(text, phrase):
    return " " + phrase + " " in " " + key(text) + " "


def coverage(gold, hits):
    targets = {e["doc_id"] for e in gold["supporting_facts"]}
    found = {h["doc_id"] for h in hits}
    return {"target_documents": len(targets), "found_support_documents": len(targets & found),
            "all_support_documents": targets <= found}


def dependency_links(question, gold, events):
    """Find later support retrieval using an entity seen earlier but absent in Q."""
    targets = {e["doc_id"]: e["title"] for e in gold["supporting_facts"]}
    seen_docs, prior_text, links = set(), [], []
    for step, event in enumerate(events, 1):
        provider_turn = event.get("provider_turn", step)
        if event.get("name") != "local_search":
            # A read can introduce a useful entity just as a search snippet can.
            obs = event.get("observation", {})
            if event.get("name") == "read_document" and obs.get("text"):
                prior_text.append((step, obs["text"], provider_turn))
            continue
        query = event["arguments"].get("query", "")
        obs = event.get("observation", {})
        for hit in obs.get("results", []):
            doc_id = hit["doc_id"]
            if doc_id not in targets or doc_id in seen_docs:
                continue
            for name in entity_names(targets[doc_id]):
                if contains(question, name) or not contains(query, name):
                    continue
                source = next(((s, text) for s, text, turn in prior_text if turn < provider_turn and contains(text, name)), None)
                if source:
                    links.append({"entity": targets[doc_id], "matched_name": name,
                                  "introduced_at_step": source[0], "queried_at_step": step,
                                  "query": query, "new_support_doc_id": doc_id})
                    break
        for hit in obs.get("results", []):
            seen_docs.add(hit["doc_id"])
            prior_text.append((step, hit["title"] + "\n" + hit.get("text", ""), provider_turn))
    return links


def screen(data=DEFAULT_DATA, output=None, search_url="http://127.0.0.1:8020"):
    data = Path(data)
    output = Path(output or data / "multihop")
    output.mkdir(parents=True, exist_ok=True)
    store, search = TaskStore(data), SearchClient(search_url)
    diagnostics, candidates = [], []
    for task in store.tasks["sft_seed"].values():
        gold = store.gold["sft_seed"][task["id"]]
        if gold["type"] != "bridge":
            continue
        hits = search.search(task["question"], top_k=10, mode="bm25")["results"]
        c3, c10 = coverage(gold, hits[:3]), coverage(gold, hits)
        visible = "\n".join(h["text"] for h in hits[:3])
        answer_visible = contains(visible, key(gold["answer"]))
        row = {**task, "original_query_top3": c3, "original_query_top10": c10,
               "answer_string_in_top3": answer_visible,
               "scope": "sft_seed only; gold used for screening, never sent to teacher"}
        row["candidate"] = not c10["all_support_documents"] and not answer_visible
        diagnostics.append(row)
        if row["candidate"]:
            candidates.append(row)
    # Favor a visible first hop over a completely missing starting point.
    candidates.sort(key=lambda r: (-r["original_query_top3"]["found_support_documents"], r["id"]))
    from .prepare import write_jsonl
    write_jsonl(output / "screening.jsonl", diagnostics)
    write_jsonl(output / "candidates.jsonl", candidates)
    summary = {"split": "sft_seed", "bridge_questions_screened": len(diagnostics), "candidates": len(candidates),
               "all_support_in_original_top10": sum(r["original_query_top10"]["all_support_documents"] for r in diagnostics),
               "candidate_top3_support_counts": dict(Counter(r["original_query_top3"]["found_support_documents"] for r in candidates)),
               "criteria": "bridge; full original-question BM25 top10 misses support; gold answer string absent in top3",
               "warning": "Candidates only. Does not prove necessity of multiple searches."}
    (output / "screening_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    screen(args.data)
