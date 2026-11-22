"""Loopback-only search API. Never loads question labels or private gold files."""
import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from .environment import TOOLS
from .search import DEFAULT_DATA, SearchEngine


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["bm25", "dense", "hybrid"] = "bm25"


class RetrieveRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=16)
    topk: int = Field(default=5, ge=1, le=20)
    return_scores: bool = False
    mode: Literal["bm25", "dense", "hybrid"] = "bm25"


def create_app(data=DEFAULT_DATA):
    lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = SearchEngine(data)
        yield
        app.state.engine.close()

    app = FastAPI(title="HotpotQA shared-corpus search", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        meta = app.state.engine.meta
        return {"ready": True, "chunks": meta["chunks"], "dense": meta["dense"],
                "corpus_sha256": meta["corpus_sha256"], "setting": "custom restricted shared corpus"}

    @app.get("/tools")
    def tools():
        return TOOLS

    @app.post("/search")
    def search(request: SearchRequest):
        with lock:
            try:
                return app.state.engine.search(request.query, request.top_k, request.mode)
            except ValueError as error:
                raise HTTPException(400, str(error)) from error

    @app.get("/documents/{document_id}")
    def read(document_id: str):
        try:
            return app.state.engine.read(document_id)
        except KeyError as error:
            raise HTTPException(404, "Unknown document") from error

    @app.post("/retrieve")
    def retrieve(request: RetrieveRequest):
        result = []
        with lock:
            try:
                for query in request.queries:
                    hits = app.state.engine.search(query, request.topk, request.mode)["results"]
                    documents = [{"id": h["id"], "title": h["title"], "contents": h["title"] + "\n" + h["text"]} for h in hits]
                    result.append([{"document": d, "score": h["score"]} for d, h in zip(documents, hits)] if request.return_scores else documents)
            except ValueError as error:
                raise HTTPException(400, str(error)) from error
        return {"result": result}

    return app


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    uvicorn.run(create_app(args.data), host="127.0.0.1", port=args.port, workers=1)
