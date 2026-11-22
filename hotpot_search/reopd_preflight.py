"""Verify the formal 10-question SFT protocol smoke against the local index."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from .download import sha256
from .reopd import Rollout, environment_lock
from .reopd_model import MODELS
from .reopd_run import write_json
from .search import DEFAULT_DATA, SearchEngine, rows
from .training import TaskStore


def verify_episodes(engine, store, expected, episodes, scores):
    keys = [(row["qid"], row["attempt"]) for row in episodes]
    if keys != expected or len(set(keys)) != 10:
        raise ValueError("Expected the ten distinct manifest episodes in order")
    by_key = {(score["qid"], score["attempt"]): score for score in scores}
    if len(by_key) != len(scores):
        raise ValueError("Duplicate smoke score")
    searches = finished_after_search = replayed = 0
    for key, episode in zip(keys, episodes, strict=True):
        if episode["split"] != "dev":
            raise ValueError(f"Smoke episode is not in dev: {key}")
        if episode["reason"] == "backend_error":
            continue
        qid, attempt = key
        replay = Rollout(engine, store.tasks["dev"][qid], "dev")
        for step in episode["steps"]:
            replay.advance(step["raw"])
        if replay.reason is None:
            raise ValueError(f"Unterminated smoke episode: {key}")
        actual, score = replay.record(store.gold["dev"][qid], attempt)
        if actual != episode or by_key.pop(key, None) != score:
            raise ValueError(f"Smoke trajectory or score differs from replay: {key}")
        replayed += 1
        searches += sum(bool(step.get("doc_ids")) for step in replay.steps if step["kind"] == "search")
        finished_after_search += replay.reason == "finished" and any(step.get("doc_ids") for step in replay.steps)
    if by_key:
        raise ValueError("Unexpected smoke scores, including possible infrastructure failures")
    return {"passed": replayed == 10 and finished_after_search > 0, "requested": 10,
            "replayed": replayed, "searches_with_hits": searches, "finished_after_search": finished_after_search,
            "terminal_reasons": dict(Counter(episode["reason"] for episode in episodes))}


def audit(data, run):
    engine = SearchEngine(data)
    try:
        store = TaskStore(data)
        lock = json.loads((run / "environment.json").read_text(encoding="utf-8"))
        if lock != environment_lock(engine):
            raise ValueError("Smoke environment no longer matches the pinned local index")
        stage = run / "dev-sft"
        manifest_path = stage / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        qids = sorted(store.tasks["dev"])
        random.Random(manifest["seed"]).shuffle(qids)
        specs = [[qid, 0, []] for qid in qids[:10]]
        if (manifest["model"] != list(MODELS["sft"]) or manifest["split"] != "dev" or manifest["limit"] != 10
                or manifest["temperature"] != 0 or manifest["specs"] != specs):
            raise ValueError("Smoke manifest is not the formal SFT ten-question dev cohort")
        episode_path, score_path = stage / "episodes.jsonl", stage / "scores.jsonl"
        report = verify_episodes(engine, store, [(qid, attempt) for qid, attempt, _ in specs],
                                 list(rows(episode_path)), list(rows(score_path)))
        report.update(model=manifest["model"], sha256={path.name: sha256(path)
                      for path in (manifest_path, episode_path, score_path, run / "environment.json")})
        write_json(run / "p0-acceptance.json", report)
        return report
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.data, args.run)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
