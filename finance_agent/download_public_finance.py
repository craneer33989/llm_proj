"""Pin and download official training splits as raw seeds, never as teacher SFT."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import Request, urlopen

from .deepseek_trajectories import atomic_json, atomic_text, dumps, now

SOURCES = {
    "finqa": ("czyssrs/FinQA", "main", "dataset/train.json"),
    "tatqa": ("NExTplusplus/TAT-QA", "master", "dataset_raw/tatqa_dataset_train.json"),
}


def fetch(url):
    req = Request(url, headers={"User-Agent": "PublicFinanceDatasetAcquisition/1.0"})
    with urlopen(req, timeout=40) as response:
        body = response.read(200_000_001)
    if len(body) > 200_000_000:
        raise ValueError("Public dataset file exceeds the 200 MB download limit")
    return body


def download(output):
    output.mkdir(parents=True, exist_ok=True)
    result = {"downloaded_at": now(), "source_split": "train", "teacher_generated": False,
              "ready_for_current_sft": False, "requires_protocol_adapter_and_validation": True, "datasets": {}}
    for name, (repo, branch, train_path) in SOURCES.items():
        metadata = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/{branch}"))
        revision = metadata["sha"]
        tree = json.loads(fetch(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1"))
        paths = {item["path"] for item in tree["tree"] if item["type"] == "blob"}
        if train_path not in paths:
            raise ValueError(f"Official training split path changed: {name}")
        entry = next(item for item in tree["tree"] if item["path"] == train_path)
        print(f"Downloading {name} official train split ({entry.get('size', 'unknown')} bytes)", flush=True)
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{train_path}"
        body = fetch(url)
        rows = json.loads(body)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Unexpected training split schema: {name}")
        folder = output / name
        atomic_text(folder / "train.json", body.decode("utf-8-sig"))
        index = []
        for context_index, row in enumerate(rows):
            questions = [row["qa"]] if name == "finqa" else row["questions"]
            for question_index, question in enumerate(questions):
                index.append({"source": name, "source_revision": revision, "source_split": "train",
                              "source_id": row.get("id", question.get("uid")), "question": question["question"],
                              "answer": question.get("exe_ans", question.get("answer")),
                              "answer_type": question.get("answer_type"), "scale": question.get("scale"),
                              "context_index": context_index, "question_index": question_index,
                              "raw_file": f"{name}/train.json", "teacher_generated": False,
                              "ready_for_current_sft": False})
        atomic_text(folder / "question-index.jsonl", "".join(dumps(row) + "\n" for row in index))
        license_paths = sorted(path for path in paths if "/" not in path and path.lower().startswith("license"))
        licenses = []
        for path in license_paths:
            license_url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
            license_body = fetch(license_url)
            atomic_text(folder / path, license_body.decode("utf-8-sig"))
            licenses.append({"path": path, "url": license_url, "sha256": sha256(license_body).hexdigest()})
        result["datasets"][name] = {"repository": f"https://github.com/{repo}", "revision": revision,
                                    "url": url, "sha256": sha256(body).hexdigest(), "bytes": len(body),
                                    "contexts": len(rows), "questions": len(index), "licenses": licenses,
                                    "note": "Public source training seeds; not executed or adapted to filing-tools-v1."}
        atomic_json(output / "manifest.json", result)
        print(f"{name}: {len(index)} training questions, revision {revision}", flush=True)
    return result


def download_toolace(output):
    """Keep upstream synthetic tool dialogues separate from locally verified SFT."""
    metadata = json.loads(fetch("https://huggingface.co/api/datasets/Team-ACE/ToolACE"))
    if metadata.get("gated"):
        raise ValueError("ToolACE unexpectedly requires authentication; no credentials will be sent")
    revision = metadata["sha"]
    if "data.json" not in {item["rfilename"] for item in metadata["siblings"]}:
        raise ValueError("Official ToolACE file layout changed")
    base = f"https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/{revision}"
    url = base + "/data.json"
    body = fetch(url)
    rows = json.loads(body)
    if not isinstance(rows, list) or not rows or any(not isinstance(row.get("conversations"), list) for row in rows):
        raise ValueError("Unexpected ToolACE dialogue schema")
    folder = output / "toolace"
    atomic_text(folder / "data.json", body.decode("utf-8-sig"))
    card = fetch(base + "/README.md")
    atomic_text(folder / "README.upstream.md", card.decode("utf-8-sig"))
    index = [{"source": "ToolACE", "source_revision": revision, "source_id": f"toolace-{i:06d}",
              "conversation_turns": len(row["conversations"]), "upstream_synthetic": True,
              "collected_by_local_teacher": False, "ready_for_current_sft": False,
              "raw_file": "toolace/data.json", "row_index": i} for i, row in enumerate(rows)]
    atomic_text(folder / "dialogue-index.jsonl", "".join(dumps(row) + "\n" for row in index))
    result = {"downloaded_at": now(), "repository": "https://huggingface.co/datasets/Team-ACE/ToolACE",
              "revision": revision, "url": url, "sha256": sha256(body).hexdigest(), "bytes": len(body),
              "dialogues": len(rows), "license": metadata.get("cardData", {}).get("license"),
              "upstream_card_sha256": sha256(card).hexdigest(), "upstream_synthetic": True,
              "ready_for_current_sft": False, "locally_replayed": False,
              "note": "Cross-domain tool-use seeds. Upstream tool schemas/results require adaptation and verification."}
    atomic_json(folder / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/external/public-finance-20261113"))
    parser.add_argument("--toolace-only", action="store_true")
    args = parser.parse_args()
    result = download_toolace(args.output.resolve()) if args.toolace_only else download(args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
