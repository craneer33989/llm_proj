"""Fetch exact ReOPD checkpoints with checked, persistent HTTP byte ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import re
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

from .download import get_json, sha256
from .reopd_model import MODELS, verified_dir

ORIGIN = "https://huggingface.co"
ASSETS = re.compile(r"[A-Za-z0-9_.-]+\.(?:json|jinja|txt|safetensors)\Z")


def check_range(response, start, end, size):
    content_range = response.headers.get("Content-Range")
    if response.status != 206 or content_range != f"bytes {start}-{end}/{size}":
        raise OSError(f"Expected HTTP 206 bytes {start}-{end}/{size}; got {response.status} {content_range}")


def transfer(url, target, size, expected_sha256, request_bytes, timeout, idle_retries):
    if target.is_file() and target.stat().st_size == size and sha256(target) == expected_sha256:
        return
    partial = target.with_suffix(target.suffix + ".part")
    stalled = 0
    while (start := partial.stat().st_size if partial.exists() else 0) < size:
        end = min(start + request_bytes, size) - 1
        request = Request(f"{url}?download=true&reopd_range={start}-{end}",
                          headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        failure = None
        try:
            with urlopen(request, timeout=timeout) as response:
                check_range(response, start, end, size)
                with partial.open("ab") as stream:
                    while remaining := (end + 1 - stream.tell()):
                        block = response.read(min(remaining, 1024 * 1024))
                        if not block:
                            raise OSError("Response ended before requested range")
                        stream.write(block)
        except (OSError, HTTPException) as error:
            failure = error
        current = partial.stat().st_size if partial.exists() else 0
        if current == start:
            stalled += 1
            if stalled >= idle_retries:
                raise OSError(f"{target.name}: no progress after {stalled} attempts at byte {start}") from failure
            time.sleep(min(2 ** stalled, 15))
        else:
            stalled = 0
            print(f"{target.parent.name}/{target.name}: {current / 2**20:.0f}/{size / 2**20:.0f} MiB", flush=True)
    if partial.stat().st_size != size or sha256(partial) != expected_sha256:
        raise ValueError(f"Weight checksum mismatch: {partial}; inspect before retrying")
    partial.replace(target)


def describe(model_id, revision):
    tree = get_json(f"{ORIGIN}/api/models/{model_id}/tree/{revision}")
    files = {}
    for asset in tree:
        name = asset["path"]
        if asset["type"] != "file" or not ASSETS.fullmatch(name):
            continue
        entry = {"size": asset["size"], "git_blob": asset["oid"]}
        if "lfs" in asset:
            entry["sha256"] = asset["lfs"]["oid"]
        if name.endswith(".safetensors") and not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            raise ValueError(f"Pinned shard lacks LFS SHA-256: {name}")
        files[name] = entry
    if not {"config.json", "tokenizer_config.json", "tokenizer.json"} <= files.keys() or not any(
            filename.endswith(".safetensors") for filename in files):
        raise ValueError(f"Incomplete pinned model file list: {model_id}@{revision}")
    return {"model": model_id, "revision": revision, "files": files}


def fetch_model(name, workers, request_bytes, timeout, idle_retries):
    from filelock import FileLock

    model_id, revision = MODELS[name]
    root = verified_dir(name)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".download.lock"), timeout=0):
        manifest = describe(model_id, revision)
        tasks = []
        for filename, item in manifest["files"].items():
            url = f"{ORIGIN}/{model_id}/resolve/{revision}/{quote(filename)}"
            target = root / filename
            if "sha256" in item:
                tasks.append((url, target, item["size"], item["sha256"], request_bytes, timeout, idle_retries))
                continue
            if target.is_file():
                body = target.read_bytes()
            else:
                with urlopen(url, timeout=timeout) as response:
                    body = response.read(item["size"] + 1)
            if (len(body) != item["size"] or hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
                    != item["git_blob"]):
                raise ValueError(f"Git blob checksum mismatch: {filename}")
            if not target.is_file():
                target.write_bytes(body)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for task in [pool.submit(transfer, *job) for job in tasks]:
                task.result()
        marker = root / "verified.json"
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(marker)
    print(f"Verified {model_id}@{revision}: {root}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["sft", "teacher", "base"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--range-mib", type=int, default=32)
    parser.add_argument("--read-timeout", type=int, default=45)
    parser.add_argument("--idle-retries", type=int, default=8)
    args = parser.parse_args()
    if min(args.workers, args.range_mib, args.read_timeout, args.idle_retries) < 1:
        parser.error("Download limits must be positive")
    fetch_model(args.model, args.workers, args.range_mib * 2**20, args.read_timeout, args.idle_retries)


if __name__ == "__main__":
    main()
