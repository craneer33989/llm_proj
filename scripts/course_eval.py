"""Run post-freeze, zero-training controls on the 100 local test questions."""
import argparse
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock

from hotpot_search import reopd
from hotpot_search.download import sha256
from hotpot_search.reopd_compare import cohort_summary, compare, episode_table
from hotpot_search.reopd_final import HeldOutStore, arm_paths, read_json
from hotpot_search.reopd_model import Actor, MODELS, checkpoint
from hotpot_search.reopd_run import collect, verify_lock, write_json
from hotpot_search.search import DEFAULT_DATA, SearchEngine
from hotpot_search.training import EpisodeConfig

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/reopd/formal-v2-expanded"
FINAL = RUN / "final-test"
EXTRA = RUN.parent / "course-extra"
EXPERIMENTS = {"base-bm25": ("base", "bm25"), "a-hybrid": ("sft", "hybrid")}


def frozen_cohort():
    path = RUN / "test-freeze.json"
    receipt = read_json(FINAL / "complete.json")
    if sha256(path) != receipt["freeze_sha256"]:
        raise ValueError("Frozen plan and completion receipt do not match")
    plan = read_json(path)
    store = HeldOutStore(DEFAULT_DATA, plan["test_size"])
    specs = [(qid, 0, []) for qid in sorted(store.tasks["test"])]
    return plan, store, specs


def run_control(name, plan, store, specs):
    model, mode = EXPERIMENTS[name]
    options = plan["evaluation"]
    adapter = None
    adapter_hashes = None
    with closing(SearchEngine(DEFAULT_DATA)) as engine:
        if reopd.environment_lock(engine) != plan["environment"]:
            raise ValueError("Current BM25 environment does not match the frozen test")
        if mode == "hybrid":
            adapter = arm_paths(RUN, plan["update_tokens"], plan["primary"]["seed"], plan["failure_type"])["A"]
            adapter_hashes = {file: sha256(adapter / file) for file in plan["primary"]["adapters"]["A"]}
            if adapter_hashes != plan["primary"]["adapters"]["A"]:
                raise ValueError("Adapter A does not match the frozen checkpoint")
            if not engine.meta["dense"]:
                raise ValueError("The frozen index must include its dense collection for this control")
            engine.search("Wikipedia", top_k=1, mode=mode)

        source = checkpoint(model)
        model_hashes = {path.name: sha256(path) for path in sorted(source.iterdir())
                        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".jinja"}}
        if model == "sft" and model_hashes != plan["base_sha256"]:
            raise ValueError("Search SFT checkpoint does not match the frozen weights")
        original = reopd.CONFIG
        reopd.CONFIG = EpisodeConfig(**{**plan["environment"]["config"], "mode": mode})
        try:
            out = EXTRA / name
            out.mkdir(exist_ok=True)
            verify_lock(out / "manifest.json", {
                "split": "test", "specs": specs, "model": MODELS[model], "model_sha256": model_hashes,
                "adapter_sha256": adapter_hashes, "environment": reopd.environment_lock(engine),
                "script_sha256": sha256(Path(__file__)),
                "runtime_sha256": {file: sha256(ROOT / "hotpot_search" / file) for file in plan["code_sha256"]},
                "config": asdict(reopd.CONFIG), "temperature": 0.0, **options,
            })
            actor = Actor(model, max_tokens=options["max_tokens"], adapter=adapter)
            _, scores = collect(engine, store, actor, "test", specs, out, options["batch"], 0.0, options["seed"])
            if len(scores) != len(specs):
                raise RuntimeError(f"{name} has unresolved backend errors. Rerun to retry those questions.")
        finally:
            reopd.CONFIG = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=(*EXPERIMENTS, "compare"))
    args = parser.parse_args()
    EXTRA.mkdir(parents=True, exist_ok=True)
    with FileLock(EXTRA / ".lock", timeout=0):
        plan, store, specs = frozen_cohort()
        if args.experiment != "compare":
            run_control(args.experiment, plan, store, specs)
        else:
            paths = {"base": FINAL / "base", "base_bm25": EXTRA / "base-bm25",
                     "A_bm25": FINAL / "A", "A_hybrid": EXTRA / "a-hybrid"}
            expected = {(qid, attempt) for qid, attempt, _ in specs}
            for name, path in paths.items():
                table = episode_table(path)
                if table.keys() != expected or any("supported_success" not in row for row in table.values()):
                    raise ValueError(f"{name} does not contain the complete scored test cohort")
            compare(paths, EXTRA)
            cohort_summary(paths, EXTRA, failure_type=plan["failure_type"])
            write_json(EXTRA / "note.json", {"note": "Post-freeze controls. Neither arm influenced model selection."})


if __name__ == "__main__":
    main()
