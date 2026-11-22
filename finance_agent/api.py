"""Loopback API for the local demo and Search-R1-compatible retrieval."""
from datetime import date
import json
import os
from typing import Literal
from urllib.error import URLError

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import MODEL_ID, REVISION, ask
from .corpus import DB, Scope, retrieve
from .filings import MANIFEST, REPORTS
from .experiments import autonomous_experiment, training_experiment
from .autonomous import ask as ask_autonomous

Company = Literal["000333", "000651", "600690"]
PLANNER = os.getenv("FINANCE_PLANNER", "model")
app = FastAPI(title="A 股财报证据问答", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    company_code: Company
    as_of: date
    mode: Literal["single", "agent", "assisted"] = "agent"


class RetrieveRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=32)
    topk: int = Field(default=3, ge=1, le=10)
    return_scores: bool = False
    company_code: Company | None = None
    as_of: date | None = None
    year: int | None = None
    period: Literal["FY", "H1"] | None = None


@app.get("/health")
def health():
    return {"index_ready": DB.is_file(), "planner": PLANNER, "model_id": MODEL_ID, "revision": REVISION}


@app.get("/reports")
def reports():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@app.get("/experiments/sft-pilot")
def experiment_api():
    try:
        return training_experiment()
    except FileNotFoundError as error:
        raise HTTPException(503, "尚无完整的 SFT 试跑记录。") from error


@app.get("/experiments/sft-autonomous")
def autonomous_experiment_api():
    try:
        return autonomous_experiment()
    except FileNotFoundError as error:
        raise HTTPException(503, "无下一步提示的 SFT 实验尚未完成。") from error


@app.get("/experiments/sft-autonomous-epoch2")
def autonomous_epoch2_api():
    try:
        return autonomous_experiment("sft-autonomous-epoch2")
    except FileNotFoundError as error:
        raise HTTPException(503, "第二轮 SFT 评测尚未完成。") from error


@app.get("/filings/{report_id}.pdf")
def filing(report_id: str):
    if report_id not in {row["id"] for row in reports()}:
        raise HTTPException(404, "报告不存在")
    path = REPORTS / f"{report_id}.pdf"
    if not path.is_file():
        raise HTTPException(404, "本地报告尚未下载")
    return FileResponse(path, media_type="application/pdf")


@app.post("/retrieve")
def retrieve_api(payload: RetrieveRequest):
    scope = Scope(payload.company_code, payload.as_of.isoformat() if payload.as_of else None, payload.year, payload.period)
    output = []
    for query in payload.queries:
        if len(query) > 500:
            raise HTTPException(422, "每个查询最多 500 字符")
        documents = []
        for row in retrieve(query, payload.topk, scope):
            document = {field: row[field] for field in ("id", "report_id", "page", "url", "published_at")}
            document["contents"] = f'"{row["company"]} {row["year"]} {row["section"]}"\n{row["contents"]}'
            documents.append({"document": document, "score": row["score"]} if payload.return_scores else document)
        output.append(documents)
    return {"result": output}


@app.post("/ask")
def ask_api(payload: AskRequest):
    if not DB.is_file():
        raise HTTPException(503, "财报索引尚未构建")
    try:
        if payload.mode != "single" and PLANNER == "model":
            return ask_autonomous(payload.question, payload.company_code, payload.as_of.isoformat())
        return ask(payload.question, payload.company_code, payload.as_of.isoformat(), payload.mode, PLANNER)
    except (URLError, TimeoutError) as error:
        raise HTTPException(503, "本机模型服务不可用，请检查模型启动日志。") from error
