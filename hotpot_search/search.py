"""Shared-corpus BM25, dense Qdrant, and reciprocal-rank hybrid retrieval."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import uuid

from .download import ROOT, sha256

DEFAULT_DATA = ROOT / "v1"
DEFAULT_ENCODER = Path(__file__).resolve().parents[1] / ".cache" / "hotpot-encoder"
COLLECTION = "hotpot_chunks"


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def point_id(chunk_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "hotpot-local:" + chunk_id))


class Encoder:
    def __init__(self, path=DEFAULT_ENCODER, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(path), local_files_only=True).to(self.device).eval()
        self.truncated = 0

    def encode(self, texts):
        t = self.torch
        lengths = self.tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]
        self.truncated += sum(len(value) > 512 for value in lengths)
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with t.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors = t.nn.functional.normalize(vectors, dim=1)
        return vectors.cpu().float().numpy()


def build_index(data=DEFAULT_DATA, encoder_path=DEFAULT_ENCODER, batch_size=64, dense=True):
    data, encoder_path = Path(data), Path(encoder_path).resolve()
    index = data / "index"
    if (index / "manifest.json").exists():
        raise FileExistsError("Index already built; use a new dataset/index version")
    index.mkdir(parents=True, exist_ok=True)
    chunks = list(rows(data / "corpus.jsonl"))
    with closing(sqlite3.connect(index / "documents.sqlite3")) as db:
        db.executescript("""
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS documents;
            DROP TABLE IF EXISTS chunks_fts;
            CREATE TABLE chunks(id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, title TEXT, text TEXT, url TEXT);
            CREATE INDEX chunk_doc ON chunks(doc_id);
            CREATE TABLE documents(id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE VIRTUAL TABLE chunks_fts USING fts5(id UNINDEXED, title, text, tokenize='unicode61 remove_diacritics 2');
        """)
        db.executemany("INSERT INTO chunks VALUES(?,?,?,?,?)", [(c["id"], c["doc_id"], c["title"], c["text"], c["url"]) for c in chunks])
        db.executemany("INSERT INTO chunks_fts VALUES(?,?,?)", [(c["id"], c["title"], c["text"]) for c in chunks])
        db.executemany("INSERT INTO documents VALUES(?,?)", [(d["id"], json.dumps(d, ensure_ascii=False)) for d in rows(data / "documents.jsonl")])
        db.commit()
    meta = {"corpus_sha256": sha256(data / "corpus.jsonl"), "chunks": len(chunks),
            "lexical": "SQLite FTS5 BM25 (title weight=3, text weight=1)",
            "dense": False, "fusion": "RRF k=60, equal weight", "candidate_count": 50}
    if dense:
        from qdrant_client import QdrantClient, models
        encoder = Encoder(encoder_path)
        client = QdrantClient(path=str(index / "qdrant"))
        try:
            if client.collection_exists(COLLECTION):
                raise FileExistsError("Partial Qdrant index exists; select a new index version")
            client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start+batch_size]
                vectors = encoder.encode([c["title"] + ". " + c["text"] for c in batch])
                points = [models.PointStruct(id=point_id(c["id"]), vector=v.tolist(), payload={"chunk_id": c["id"]})
                          for c, v in zip(batch, vectors)]
                client.upsert(COLLECTION, points=points, wait=True)
                if start % (batch_size * 20) == 0:
                    print(f"Embedded {min(start+batch_size, len(chunks))}/{len(chunks)}", flush=True)
            meta.update({"dense": True, "encoder_path": str(encoder_path),
                         "encoder_model": "sentence-transformers/all-MiniLM-L6-v2", "dimensions": 384,
                         "encoder_weights_sha256": sha256(encoder_path / "model.safetensors"),
                         "pooling": "attention-masked mean, L2 normalized", "max_tokens": 512,
                         "truncated_documents": encoder.truncated,
                         "qdrant_points": client.count(COLLECTION, exact=True).count})
            provenance = encoder_path / "encoder_manifest.json"
            if provenance.exists():
                meta["encoder_revision"] = json.loads(provenance.read_text(encoding="utf-8"))["revision"]
        finally:
            client.close()
    (index / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return meta


class SearchEngine:
    def __init__(self, data=DEFAULT_DATA, verify=True):
        self.data = Path(data)
        self.index = self.data / "index"
        self.meta = json.loads((self.index / "manifest.json").read_text(encoding="utf-8"))
        if verify and sha256(self.data / "corpus.jsonl") != self.meta["corpus_sha256"]:
            raise ValueError("Corpus changed since indexing")
        self.encoder = self.client = None

    def _db(self):
        path = str((self.index / "documents.sqlite3").resolve())
        # pytest/Windows may supply extended-length paths; '?' is URI syntax.
        if path.startswith("\\\\?\\"):
            path = path[4:]
        db = sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        return db

    def lexical_ids(self, query, limit):
        terms = list(dict.fromkeys(re.findall(r"\w+", query.casefold())))[:64]
        if not terms:
            return []
        match = " OR ".join('"' + term + '"' for term in terms)
        with closing(self._db()) as db:
            found = db.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts,0,3,1), id LIMIT ?", (match, limit))
            return [r[0] for r in found]

    def dense_ids(self, query, limit):
        if not self.meta["dense"]:
            raise ValueError("Dense index is not built; use bm25")
        if self.encoder is None:
            from qdrant_client import QdrantClient
            self.encoder = Encoder(self.meta["encoder_path"])
            # API serializes queries with a lock; workers can still be different threads.
            self.client = QdrantClient(path=str(self.index / "qdrant"), force_disable_check_same_thread=True)
        vector = self.encoder.encode([query])[0].tolist()
        result = self.client.query_points(COLLECTION, query=vector, limit=limit)
        return [p.payload["chunk_id"] for p in result.points]

    def search(self, query, top_k=5, mode="bm25"):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query must contain 1..2000 characters")
        if mode not in ("bm25", "dense", "hybrid") or not 1 <= top_k <= 20:
            raise ValueError("mode must be bm25/dense/hybrid, top_k must be 1..20")
        limit = max(50, top_k)
        ranked = []
        if mode in ("bm25", "hybrid"):
            ranked.append(self.lexical_ids(query, limit))
        if mode in ("dense", "hybrid"):
            ranked.append(self.dense_ids(query, limit))
        scores = {}
        for ids in ranked:
            for rank, cid in enumerate(ids, 1):
                scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
        best = sorted(scores, key=lambda cid: (-scores[cid], cid))[:top_k]
        with closing(self._db()) as db:
            results = []
            for cid in best:
                row = dict(db.execute("SELECT * FROM chunks WHERE id=?", (cid,)).fetchone())
                row["score"] = scores[cid]
                results.append(row)
        return {"query": query, "mode": mode, "results": results, "corpus_version": self.meta["corpus_sha256"]}

    def read(self, document_id):
        with closing(self._db()) as db:
            row = db.execute("SELECT body FROM documents WHERE id=?", (document_id,)).fetchone()
        if row is None:
            raise KeyError(document_id)
        document = json.loads(row[0])
        document["text"] = " ".join(document.pop("sentences"))
        return document

    def close(self):
        if self.client is not None:
            self.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "query"])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument("--query", default="Who directed the film Jaws?")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    args = parser.parse_args()
    if args.command == "build":
        build_index(args.data, args.encoder, dense=not args.bm25_only)
    else:
        engine = SearchEngine(args.data)
        try:
            print(json.dumps(engine.search(args.query, mode=args.mode), ensure_ascii=False, indent=2))
        finally:
            engine.close()
