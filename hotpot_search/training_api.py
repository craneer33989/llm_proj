"""Trainer-facing reset/step API. Forward only observations, not trainer info, to the model."""
import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from .search import DEFAULT_DATA
from .client import SearchClient
from .training import EpisodeConfig, TaskStore, TrainingEpisode


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    split: Literal["sft_seed", "rl", "dev"] = "rl"
    task_id: str | None = Field(default=None, max_length=100)
    seed: int = 0
    run_label: str = Field(default="interactive", min_length=1, max_length=100)
    mode: Literal["bm25", "dense", "hybrid"] = "bm25"
    max_calls: int = Field(default=6, ge=1, le=20)
    max_top_k: int = Field(default=3, ge=1, le=10)
    max_observation_chars: int = Field(default=5000, ge=512, le=20000)
    max_total_observation_chars: int = Field(default=24000, ge=512, le=100000)


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    arguments: dict


def create_training_app(data=DEFAULT_DATA, log_dir=None, search_url="http://127.0.0.1:8020", engine_factory=None):
    lock = threading.RLock()
    log_dir = Path(log_dir or Path(data).parent / "runtime" / "episodes")

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = engine_factory() if engine_factory else SearchClient(search_url)
        app.state.store = TaskStore(data)
        import json
        manifest = json.loads((Path(data) / "manifest.json").read_text(encoding="utf-8"))
        if app.state.engine.meta["corpus_sha256"] != manifest["file_sha256"]["corpus.jsonl"]:
            raise ValueError("Task data and search backend have different corpus versions")
        app.state.episodes = {}
        yield
        app.state.engine.close()

    app = FastAPI(title="HotpotQA training search environment", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ready": True, "environment": "hotpot-training-v1", "default_mode": "bm25",
                "task_counts": {s: len(ts) for s, ts in app.state.store.tasks.items()},
                "corpus_sha256": app.state.engine.meta["corpus_sha256"], "active_episodes": len(app.state.episodes),
                "note": "Trainer API; send only episode messages/tools and tool observations to the actor"}

    @app.post("/reset")
    def reset(request: ResetRequest):
        with lock:
            if len(app.state.episodes) >= 256:
                raise HTTPException(409, "Episode capacity reached; DELETE completed episodes")
            try:
                task = app.state.store.task(request.split, request.task_id, request.seed)
                config = EpisodeConfig(**request.model_dump(exclude={"split", "task_id", "seed", "run_label"}))
                episode = TrainingEpisode(app.state.engine, task, request.split, config, log_dir, request.run_label)
            except (ValueError, KeyError) as error:
                raise HTTPException(400, str(error)) from error
            app.state.episodes[episode.id] = episode
            return episode.reset_response()

    def get_episode(episode_id):
        if episode_id not in app.state.episodes:
            raise HTTPException(404, "Unknown episode")
        return app.state.episodes[episode_id]

    @app.post("/episodes/{episode_id}/step")
    def step(episode_id: str, request: StepRequest):
        with lock:
            episode = get_episode(episode_id)
            try:
                response = episode.step(request.name, request.arguments)
                if episode.done:
                    metrics = episode.evaluate(app.state.store.gold[episode.split][episode.task["id"]])
                    response["reward"] = metrics["reward"]
                    response["info"]["metrics"] = metrics
                return response
            except ValueError as error:
                raise HTTPException(409, str(error)) from error

    @app.get("/episodes/{episode_id}/trajectory")
    def trajectory(episode_id: str):
        with lock:
            episode = get_episode(episode_id)
            return {"episode_id": episode.id, "task_id": episode.task["id"], "split": episode.split,
                    "run_label": episode.run_label,
                    "messages": episode.messages, "tools": episode.tools, "config": episode.reset_response()["config"],
                    "terminated": episode.terminated, "truncated": episode.truncated, "metrics": episode.score}

    @app.delete("/episodes/{episode_id}")
    def release(episode_id: str):
        with lock:
            episode = get_episode(episode_id)
            if not episode.done:
                episode.truncated = True
                episode._log({"event": "truncated", "reason": "released_by_trainer"})
                episode.evaluate(app.state.store.gold[episode.split][episode.task["id"]])
            del app.state.episodes[episode_id]
            return {"released": True, "log_preserved": True}

    return app


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--search-url", default="http://127.0.0.1:8020")
    args = parser.parse_args()
    uvicorn.run(create_training_app(args.data, search_url=args.search_url), host="127.0.0.1", port=args.port, workers=1)
