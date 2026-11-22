"""Scripted HTTP checks of real training episodes. Not model performance or teacher data."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request

from .search import DEFAULT_DATA

DEMO_TASK = "5a838d735542992ef85e22f5"


def smoke(base="http://127.0.0.1:8021", output=None):
    def request(path, body=None, method=None):
        content = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=content, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)

    report = {"kind": "scripted_integration_checks_not_model_performance", "health": request("/health"), "checks": {}}
    reset = request("/reset", {"split": "sft_seed", "task_id": DEMO_TASK, "run_label": "scripted_smoke_not_model"})
    assert "Hawaii" not in json.dumps(reset) and "supporting_facts" not in json.dumps(reset)
    eid = reset["episode_id"]
    citations = []
    for query, title in (("Moonshine Music", "Moonshine Music"), ("DJ Keoki raised", "DJ Keoki")):
        response = request(f"/episodes/{eid}/step", {"name": "local_search", "arguments": {"query": query}})
        assert response["reward"] == 0 and not response["terminated"]
        hit = next(r for r in response["observation"]["results"] if r["title"] == title)
        citations.append(hit["doc_id"])
    response = request(f"/episodes/{eid}/step", {"name": "read_document", "arguments": {"document_id": citations[-1]}})
    assert "Hawaii" in response["observation"]["text"]
    response = request(f"/episodes/{eid}/step", {"name": "finish", "arguments": {"answer": "Hawaii", "citations": citations}})
    assert response["terminated"] and response["info"]["metrics"]["supported_success"]
    assert response["reward"] == 1
    trajectory = request(f"/episodes/{eid}/trajectory")
    assert "supporting_facts" not in json.dumps(trajectory["messages"])
    report["checks"]["scripted_multistep_search_read_finish"] = response
    report["demo_trajectory"] = trajectory
    request(f"/episodes/{eid}", method="DELETE")
    reset = request("/reset", {"split": "rl", "seed": 11, "max_calls": 1, "run_label": "scripted_smoke_not_model"})
    eid = reset["episode_id"]
    request(f"/episodes/{eid}/step", {"name": "local_search", "arguments": {"query": "history"}})
    response = request(f"/episodes/{eid}/step", {"name": "local_search", "arguments": {"query": "history again"}})
    assert response["truncated"] and response["reward"] == 0
    report["checks"]["budget_exhaustion"] = True
    request(f"/episodes/{eid}", method="DELETE")
    reset = request("/reset", {"split": "rl", "seed": 12, "run_label": "scripted_smoke_not_model"})
    eid = reset["episode_id"]
    response = request(f"/episodes/{eid}/step", {"name": "read_document", "arguments": {"document_id": citations[-1]}})
    assert "error" in response["observation"]
    response = request(f"/episodes/{eid}/step", {"name": "finish", "arguments": {"answer": "I cannot confirm", "citations": ["invented-document"]}})
    assert response["info"]["metrics"]["unseen_citations"] == 1
    assert response["info"]["metrics"]["invalid_calls"] == 1
    report["checks"]["cross_episode_read_and_fabricated_citation_rejected"] = True
    request(f"/episodes/{eid}", method="DELETE")
    for body in ({"split": "test"}, {"split": "rl", "gold_answer": "leak"}):
        try:
            request("/reset", body)
            raise AssertionError("Expected schema rejection")
        except urllib.error.HTTPError as error:
            assert error.code == 422
    report["checks"]["test_split_and_reference_injection_rejected"] = True
    output = Path(output or DEFAULT_DATA / "training_environment_smoke.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kind": report["kind"], "checks": {key: value if isinstance(value, bool) else {"passed": True, "reward": value["reward"]} for key, value in report["checks"].items()}, "report": str(output)}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8021")
    args = parser.parse_args()
    smoke(args.base)
