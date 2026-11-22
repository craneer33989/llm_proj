"""Frozen external before/after evaluation on three ordered public cohorts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from types import SimpleNamespace

from .download import sha256
from .prepare import contexts, make_document, normalized, supporting
from .reopd import environment_lock
from .reopd_compare import cohort_summary, compare, validate_adapters
from .reopd_final import arm_paths, base_identity
from .reopd_model import Actor, MODELS
from .reopd_run import collect, verify_lock, write_json, write_rows
from .scoring import answer_metrics
from .search import SearchEngine, build_index, rows

SOURCES = {
    "hotpot-dev": ("hotpotqa/hotpot_qa", "1908d6afbbead072334abe2965f91bd2709910ab", "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"),
    "reopd-test": ("baohao/search-test", "a3b9cc30c6e617e8653858d0e699ec4995f9b4bc", "4df6405a40b53778c704517e89bb88be65f9f13909e02be329b00f25f92eced0"),
    "2wiki-dev": ("framolfese/2WikiMultihopQA", "fe713bfbd1afbca1a65246741a75890405d56a3a", "408e2dbb28edc6c8b9ca3ba0c94d4fc7bf17ffb923766593a3a7f546ab4cba59"),
}
ORDER = ("hotpot-first", "reopd-2wiki", "hotpot-second")


def allocate(groups, count):
    total = sum(map(len, groups.values()))
    if total < count:
        raise ValueError("Not enough unique eligible benchmark questions")
    quotas = {key: count * len(group) // total for key, group in groups.items()}
    remainder = sorted(groups, key=lambda key: (-(count * len(groups[key]) % total), key))
    for key in remainder[:count - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def draw(records, size, seed):
    groups = defaultdict(list)
    for record in sorted(records, key=lambda row: row["qid"]):
        groups[record["kind"]].append(record)
    rng = random.Random(seed)
    selected = []
    for key, quota in sorted(allocate(groups, size).items()):
        selected.extend(rng.sample(groups[key], quota))
    rng.shuffle(selected)
    return selected


def evidence(row):
    paragraphs = dict(contexts(row))
    facts = []
    for title, index in supporting(row):
        if title not in paragraphs or not 0 <= index < len(paragraphs[title]) or not paragraphs[title][index].strip():
            return None
        facts.append((title, index))
    return (paragraphs, facts) if len({title for title, _ in facts}) >= 2 else None


def unique(records, excluded):
    grouped = defaultdict(list)
    for record in records:
        grouped[normalized(record["question"])].append(record)
    return {key: group[0] for key, group in grouped.items() if key and key not in excluded and len(group) == 1}


def prepare(out, local_data, seed):
    import pyarrow.parquet as pq

    source = out / "source"
    for name, (_, _, checksum) in SOURCES.items():
        if sha256(source / f"{name}.parquet") != checksum:
            raise ValueError(f"Unexpected benchmark source checksum: {name}")
    excluded = {normalized(row["question"]) for split in ("sft_seed", "rl", "dev", "test")
                for row in rows(local_data / f"{split}.jsonl")}
    hotpot = unique(pq.read_table(source / "hotpot-dev.parquet").to_pylist(), excluded)
    wiki = unique(pq.read_table(source / "2wiki-dev.parquet").to_pylist(), excluded | hotpot.keys())
    reopd = pq.read_table(source / "reopd-test.parquet", columns=["id", "question", "data_source", "golden_answers"]).to_pylist()
    hotpot_rows = [{"qid": "hotpot:" + row["id"], "kind": row["type"], "question": row["question"],
                    "answer": row["answer"], "aliases": [], "context": info} for row in hotpot.values()
                   if row["answer"].strip() and (info := evidence(row)) is not None]
    first = draw(hotpot_rows, 100, seed)
    used = {row["qid"] for row in first}
    second = draw([row for row in hotpot_rows if row["qid"] not in used], 100, seed + 1)
    wiki_rows = []
    for key, row in unique((row for row in reopd if row["data_source"] == "2wikimultihopqa"), excluded).items():
        aligned = wiki.get(key)
        aliases = [answer for answer in row["golden_answers"] if isinstance(answer, str) and answer.strip()]
        if (aligned and aliases and any(answer_metrics(alias, aligned["answer"])[0] for alias in aliases)
                and (info := evidence(aligned)) is not None):
            wiki_rows.append({"qid": "reopd-2wiki:" + row["id"], "kind": aligned["type"], "question": row["question"],
                              "answer": aliases[0], "aliases": aliases[1:], "context": info, "context_id": aligned["id"]})
    cohorts = dict(zip(ORDER, (first, draw(wiki_rows, 100, seed + 2), second), strict=True))
    directory = out / "benchmark"
    directory.mkdir(exist_ok=True)
    verify_lock(directory / "manifest.json", {"source": SOURCES, "seed": seed, "order": ORDER,
                "local_questions_excluded_sha256": {split: sha256(local_data / f"{split}.jsonl")
                                                     for split in ("sft_seed", "rl", "dev", "test")},
                "groups": {name: {"qid": [row["qid"] for row in group], "strata": dict(Counter(row["kind"] for row in group))}
                           for name, group in cohorts.items()},
                "index": "shared per benchmark source; all selected candidate paragraphs; no per-question filter"})
    for names in ((ORDER[0], ORDER[2]), (ORDER[1],)):
        data = directory / ("hotpot" if len(names) == 2 else "reopd-2wiki")
        data.mkdir(exist_ok=True)
        documents, chunks = {}, {}
        for name in names:
            public, gold = [], []
            for row in cohorts[name]:
                paragraphs, facts = row["context"]
                titles = {}
                for title, sentences in paragraphs.items():
                    doc, parts = make_document(title, sentences)
                    titles[title] = doc["id"]
                    documents[doc["id"]] = doc
                    chunks.update({part["id"]: part for part in parts})
                public.append({"id": row["qid"], "question": row["question"]})
                gold.append({"id": row["qid"], "answer": row["answer"], "aliases": row["aliases"],
                             "kind": row["kind"], "context_id": row.get("context_id"),
                             "supporting_facts": [{"doc_id": titles[title], "title": title, "text": paragraphs[title][index]}
                                                  for title, index in facts]})
            write_rows(data / f"{name}.jsonl", public)
            write_rows(data / f"{name}.gold.jsonl", gold)
        write_rows(data / "documents.jsonl", documents.values())
        write_rows(data / "corpus.jsonl", chunks.values())
        artifacts = ["documents.jsonl", "corpus.jsonl", *(f"{name}{suffix}.jsonl" for name in names for suffix in ("", ".gold"))]
        verify_lock(data / "manifest.json", {"benchmark_manifest": sha256(directory / "manifest.json"),
                                             "file_sha256": {filename: sha256(data / filename) for filename in artifacts}})
        if not (data / "index" / "manifest.json").exists():
            build_index(data, dense=False)


def evaluate(out, training_run, batch):
    freeze = json.loads((training_run / "test-freeze.json").read_text(encoding="utf-8"))
    selected = arm_paths(training_run, freeze["update_tokens"], freeze["primary"]["seed"], freeze["failure_type"])
    _, hashes = validate_adapters(selected, freeze["environment"], ablation=True)
    if hashes != freeze["primary"]["adapters"] or base_identity() != freeze["base_sha256"]:
        raise ValueError("Public evaluation must use the same frozen before/after weights")
    benchmark = out / "benchmark"
    verify_lock(out / "evaluation-manifest.json", {"benchmark": sha256(benchmark / "manifest.json"),
                "freeze": sha256(training_run / "test-freeze.json"), "order": ORDER, "batch": batch,
                "student": MODELS["sft"], "max_tokens": 512, "temperature": 0,
                "adapters": {name: hashes[name] for name in ("A", "D")}})
    for name in ORDER:
        directory = out / name
        directory.mkdir(exist_ok=True)
        data = benchmark / ("hotpot" if name.startswith("hotpot") else "reopd-2wiki")
        manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        for filename, checksum in manifest["file_sha256"].items():
            if sha256(data / filename) != checksum:
                raise ValueError(f"Prepared benchmark changed: {filename}")
        engine = SearchEngine(data)
        try:
            verify_lock(directory / "environment.json", environment_lock(engine))
            tasks = {row["id"]: row for row in rows(data / f"{name}.jsonl")}
            gold = {row["id"]: row for row in rows(data / f"{name}.gold.jsonl")}
            if len(tasks) != 100 or tasks.keys() != gold.keys():
                raise ValueError("Expected 100 aligned public questions and private scoring rows")
            specs = [(qid, 0, []) for qid in tasks]
            store = SimpleNamespace(tasks={name: tasks}, gold={name: gold})
            paths = {arm: directory / arm for arm in ("base", "A", "D")}
            print(f"Starting {name}", flush=True)
            for arm, path in paths.items():
                actor = Actor("sft", 512, adapter=selected.get(arm))
                _, scores = collect(engine, store, actor, name, specs, path, batch, 0, 42)
                del actor
                if len(scores) != len(specs):
                    raise RuntimeError(f"{name}/{arm} has infrastructure failures; resume before advancing")
            compare(paths, directory)
            cohort_summary(paths, directory, failure_type="wrong_without_search")
            print(f"Finished {name}", flush=True)
        finally:
            engine.close()
    write_json(out / "complete.json", {name: sha256(out / name / "cohort_metrics.json") for name in ORDER})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "evaluate"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=Path("runs/reopd/formal-v2-expanded"))
    parser.add_argument("--local-data", type=Path, default=Path("data/hotpotqa/v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.stage == "prepare":
        prepare(args.out, args.local_data, args.seed)
    else:
        from filelock import FileLock

        with FileLock(str(args.out / ".evaluate.lock"), timeout=0):
            evaluate(args.out, args.run, args.batch)


if __name__ == "__main__":
    main()
