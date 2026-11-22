"""Build a reproducible, evidence-audited subset from HotpotQA train only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import quote

from .download import ROOT, sha256

SPLITS = {"sft_seed": 300, "rl": 300, "dev": 100, "test": 100}


def normalized(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def stable_id(prefix, *values):
    return prefix + hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()[:24]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def iter_rows(raw):
    import pyarrow.parquet as pq
    for path in sorted(Path(raw).glob("train-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()


def contexts(row):
    value = row["context"]
    if isinstance(value, dict):
        return list(zip(value["title"], value["sentences"]))
    return value


def supporting(row):
    value = row["supporting_facts"]
    if isinstance(value, dict):
        return list(zip(value["title"], value["sent_id"]))
    return value


def validate_row(row):
    docs = dict(contexts(row))
    facts = supporting(row)
    if not row.get("question", "").strip() or not row.get("answer", "").strip():
        return "empty_question_or_answer"
    if len({normalized(title) for title, _ in facts}) < 2:
        return "fewer_than_two_support_titles"
    for title, index in facts:
        if title not in docs or not 0 <= index < len(docs[title]) or not docs[title][index].strip():
            return "missing_support_sentence"
    if row["type"] not in ("bridge", "comparison") or row["level"] not in ("easy", "medium", "hard"):
        return "unsupported_stratum"
    return None


def select(candidates, sizes, seed):
    """Stratify by type/level; core supporting titles never cross splits."""
    ranked = sorted(candidates, key=lambda r: stable_id("", seed, r["id"]))
    chosen, forbidden, seen_questions = {}, set(), set()
    for split, size in sizes.items():
        bridge = round(size * .8)
        targets = {}
        for kind, total in (("bridge", bridge), ("comparison", size - bridge)):
            for i, level in enumerate(("easy", "medium", "hard")):
                targets[(kind, level)] = total // 3 + (i < total % 3)
        counts, rows = Counter(), []
        for row in ranked:
            key = (row["type"], row["level"])
            question_key = normalized(row["question"])
            titles = set(row["support_titles"])
            if counts[key] >= targets[key] or question_key in seen_questions or titles & forbidden:
                continue
            rows.append(row)
            seen_questions.add(question_key)
            counts[key] += 1
            if len(rows) == size:
                break
        if len(rows) != size:
            raise ValueError(f"Cannot fill {split}: got {len(rows)} of {size}, strata={dict(counts)}")
        forbidden.update(t for row in rows for t in row["support_titles"])
        chosen[split] = rows
    return chosen


def make_document(title, sentences, max_words=140):
    """Preserve original sentence offsets; long sentences may span chunks."""
    doc_id = stable_id("d_", title, sentences)
    document = {"id": doc_id, "title": title, "sentences": sentences,
                "url": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe=""),
                "source": "hotpotqa_original_context", "language": "en"}
    chunks, pending, sent_ids, length = [], [], [], 0

    def flush():
        nonlocal pending, sent_ids, length
        if pending:
            text = " ".join(pending)
            chunks.append({"id": stable_id("c_", doc_id, len(chunks), text), "doc_id": doc_id,
                           "title": title, "text": text, "sentence_ids": sorted(set(sent_ids)),
                           "url": document["url"], "source": document["source"]})
            pending, sent_ids, length = [], [], 0

    for index, sentence in enumerate(sentences):
        words = sentence.split()
        if not words:
            continue
        if length + len(words) > max_words:
            flush()
        if len(words) > max_words:
            for start in range(0, len(words), max_words):
                pending, sent_ids = [" ".join(words[start:start + max_words])], [index]
                flush()
        else:
            pending.append(sentence.strip())
            sent_ids.append(index)
            length += len(words)
    flush()
    return document, chunks


def build(root=ROOT, output=None, seed=20261116, sizes=None, corpus_target=20000):
    root = Path(root)
    output = Path(output or root / "v1")
    if (output / "manifest.json").exists():
        raise FileExistsError("Dataset already exists; choose a new --output for a new version")
    raw_manifest = json.loads((root / "raw/download_manifest.json").read_text(encoding="utf-8"))
    for item in raw_manifest["files"]:
        if sha256(root / "raw" / item["file"]) != item["sha256"]:
            raise ValueError(f"Raw checksum mismatch: {item['file']}")
    candidates, rejected, raw_count = [], Counter(), 0
    for row in iter_rows(root / "raw"):
        raw_count += 1
        reason = validate_row(row)
        if reason:
            rejected[reason] += 1
            continue
        candidates.append({"id": row.get("id", row.get("_id")), "question": row["question"],
                           "type": row["type"], "level": row["level"],
                           "support_titles": sorted({normalized(t) for t, _ in supporting(row)})})
    chosen = select(candidates, sizes or SPLITS, seed)
    lookup = {row["id"]: split for split, rows in chosen.items() for row in rows}
    documents, chunks, gold = {}, {}, defaultdict(list)
    selected_rows = {}

    def add(title, sentences):
        document, parts = make_document(title, sentences)
        documents[document["id"]] = document
        for part in parts:
            chunks[part["id"]] = part
        return document, parts

    for row in iter_rows(root / "raw"):
        rid = row.get("id", row.get("_id"))
        if rid not in lookup:
            continue
        selected_rows[rid] = row
        mappings = {}
        for title, sentences in contexts(row):
            document, parts = add(title, sentences)
            mappings[title] = (document, parts)
        evidence = []
        for title, index in supporting(row):
            document, parts = mappings[title]
            evidence.append({"doc_id": document["id"], "title": title, "sentence_id": index,
                             "text": document["sentences"][index],
                             "chunk_ids": [p["id"] for p in parts if index in p["sentence_ids"]]})
        gold[lookup[rid]].append({"id": rid, "answer": row["answer"], "supporting_facts": evidence,
                                 "type": row["type"], "level": row["level"]})
    # Background selection is deterministic, independent of labels, and pooled globally.
    background_ids = {r["id"] for r in sorted(candidates, key=lambda r: stable_id("", seed, "background", r["id"]))[:10000]}
    for row in iter_rows(root / "raw"):
        if len(chunks) >= corpus_target:
            break
        if row.get("id", row.get("_id")) not in background_ids:
            continue
        for title, sentences in contexts(row):
            add(title, sentences)
    output.mkdir(parents=True, exist_ok=True)
    (output / "private").mkdir(exist_ok=True)
    for split, rows in chosen.items():
        write_jsonl(output / f"{split}.jsonl", [{"id": r["id"], "question": r["question"]} for r in rows])
        by_id = {g["id"]: g for g in gold[split]}
        write_jsonl(output / "private" / f"{split}.gold.jsonl", [by_id[r["id"]] for r in rows])
    write_jsonl(output / "documents.jsonl", sorted(documents.values(), key=lambda r: r["id"]))
    write_jsonl(output / "corpus.jsonl", sorted(chunks.values(), key=lambda r: r["id"]))
    evidence_count = sum(len(g["supporting_facts"]) for gs in gold.values() for g in gs)
    covered = sum(bool(e["chunk_ids"]) and all(cid in chunks for cid in e["chunk_ids"])
                  for gs in gold.values() for g in gs for e in g["supporting_facts"])
    if covered != evidence_count:
        raise RuntimeError("Evidence coverage audit failed")
    split_titles = {s: {t for r in rs for t in r["support_titles"]} for s, rs in chosen.items()}
    overlaps = {f"{a}/{b}": len(split_titles[a] & split_titles[b])
                for i, a in enumerate(chosen) for b in list(chosen)[i+1:]}
    files = {str(p.relative_to(output)).replace("\\", "/"): sha256(p)
             for p in sorted(output.rglob("*.jsonl"))}
    manifest = {"version": "hotpot-local-v1", "seed": seed, "source": raw_manifest,
                "raw_train_questions": raw_count, "eligible_questions": len(candidates),
                "rejected": dict(rejected), "splits": {s: len(rs) for s, rs in chosen.items()},
                "strata": {s: dict(Counter(r["type"] + "/" + r["level"] for r in rs)) for s, rs in chosen.items()},
                "documents": len(documents), "chunks": len(chunks),
                "supporting_facts": evidence_count, "covered_supporting_facts": covered,
                "core_support_title_overlap": overlaps, "file_sha256": files,
                "setting": "Custom restricted shared corpus; NOT official fullwiki evaluation",
                "leakage_boundary": "Question IDs/text and core support titles disjoint across splits. Raw source documents are shared open-book knowledge. Other mentioned entities may overlap.",
                "limitations": ["English static Wikipedia context paragraphs, not full articles or live web",
                                "No teacher trajectories or model training yet",
                                "Original HotpotQA labels do not guarantee every question requires two actual searches"],
                "license": "CC-BY-SA-4.0", "attribution": "Yang et al., HotpotQA, EMNLP 2018"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("splits", "documents", "chunks", "supporting_facts", "covered_supporting_facts", "core_support_title_overlap")}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--corpus-target", type=int, default=20000)
    args = parser.parse_args()
    build(args.root, args.output, args.seed, corpus_target=args.corpus_target)
