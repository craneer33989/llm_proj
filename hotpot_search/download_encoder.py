"""Download a pinned public MiniLM encoder, verify weights, and record provenance."""
import json
from pathlib import Path

from .download import sha256
from .search import DEFAULT_ENCODER

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
WEIGHTS_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"


def download(path=DEFAULT_ENCODER):
    from huggingface_hub import snapshot_download
    path = Path(path)
    snapshot_download(MODEL, revision=REVISION, local_dir=path,
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                                      "vocab.txt", "special_tokens_map.json", "1_Pooling/config.json",
                                      "modules.json", "sentence_bert_config.json"])
    actual = sha256(path / "model.safetensors")
    if actual != WEIGHTS_SHA256:
        raise ValueError("Encoder weights checksum mismatch")
    manifest = {"model": MODEL, "revision": REVISION, "weights_sha256": actual,
                "license": "Apache-2.0", "source": "https://huggingface.co/" + MODEL}
    (path / "encoder_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    download()
