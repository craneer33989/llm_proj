"""Verify a running local HTTP server without exposing any answer labels."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request

from .search import DEFAULT_DATA


def smoke(base="http://127.0.0.1:8020", data=DEFAULT_DATA):
    def request(path, body=None):
        content = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=content, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)

    report = {"health": request("/health"), "checks": {}}
    for mode in ("bm25", "dense", "hybrid"):
        response = request("/search", {"query": "Moonshine Music", "top_k": 3, "mode": mode})
        assert len(response["results"]) == 3
        assert all(not ({"answer", "supporting_facts", "is_supporting"} & set(hit)) for hit in response["results"])
        report["checks"][mode] = {"passed": True, "titles": [h["title"] for h in response["results"]]}
        if mode == "bm25":
            document = request("/documents/" + response["results"][0]["doc_id"])
            assert document["text"]
            report["checks"]["read_document"] = True
    response = request("/retrieve", {"queries": ["Moonshine Music", "Echo Lake Country Club"], "topk": 2, "return_scores": True})
    assert len(response["result"]) == 2 and len(response["result"][0]) == 2
    report["checks"]["retrieve_batch"] = True
    assert len(request("/tools")) == 2
    for path, body, expected in [("/search", {"query": "", "top_k": 3}, 422),
                                 ("/search", {"query": "a", "mode": "bad"}, 422),
                                 ("/documents/private.gold.jsonl", None, 404)]:
        try:
            request(path, body)
            raise AssertionError("Expected HTTP rejection")
        except urllib.error.HTTPError as error:
            assert error.code == expected
    report["checks"]["invalid_inputs_and_private_path_rejected"] = True
    Path(data, "service_smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8020")
    args = parser.parse_args()
    smoke(args.base)
