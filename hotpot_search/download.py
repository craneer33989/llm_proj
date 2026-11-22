"""Download the public HotpotQA train split, with pinned revision and SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPO = "hotpotqa/hotpot_qa"
ROOT = Path(__file__).resolve().parents[1] / "data" / "hotpotqa"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_json(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


def download(root=ROOT):
    raw = Path(root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        info = get_json(f"https://huggingface.co/api/datasets/{REPO}")
        revision = info["sha"]
        tree = get_json(f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/distractor")
        files = []
        for item in tree:
            if item["path"].split("/")[-1].startswith("train-"):
                files.append({"file": item["path"].split("/")[-1],
                              "url": f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{item['path']}",
                              "size": item["size"], "sha256": item["lfs"]["oid"]})
        if len(files) != 2:
            raise RuntimeError("Expected two HotpotQA training parquet shards")
        manifest = {"dataset": REPO, "revision": revision, "config": "distractor", "split": "train",
                    "original_source": "https://hotpotqa.github.io/", "license": "CC-BY-SA-4.0",
                    "files": sorted(files, key=lambda x: x["file"])}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for item in manifest["files"]:
        target = raw / item["file"]
        if target.exists() and sha256(target) == item["sha256"]:
            print(f"Verified existing {target.name}", flush=True)
            continue
        for attempt in range(3):
            part = target.with_suffix(".part")
            try:
                print(f"Downloading {target.name} ({item['size'] / 2**20:.1f} MiB)", flush=True)
                with urllib.request.urlopen(item["url"], timeout=120) as response, part.open("wb") as stream:
                    last = time.monotonic()
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        if time.monotonic() - last > 10:
                            print(f"  {stream.tell() / 2**20:.1f} MiB", flush=True)
                            last = time.monotonic()
                if part.stat().st_size != item["size"] or sha256(part) != item["sha256"]:
                    raise RuntimeError(f"Size/checksum mismatch for {target.name}")
                part.replace(target)
                break
            except Exception:
                if attempt == 2:
                    raise
        print(f"Verified {target.name}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    download(parser.parse_args().root)
