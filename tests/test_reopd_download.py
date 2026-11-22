import hashlib
import io

import pytest

from hotpot_search import reopd_download


class Response(io.BytesIO):
    status = 206

    def __init__(self, payload, byte_range):
        super().__init__(payload)
        self.headers = {"Content-Range": byte_range}


def test_range_transfer_resumes_after_short_reply_and_verifies_hash(tmp_path, monkeypatch):
    requested = []
    asset = b"abcdefgh"
    target = tmp_path / "model.safetensors"

    def get_range(request, timeout):
        requested.append(request.headers["Range"])
        if len(requested) == 1:
            return Response(asset[:2], "bytes 0-3/8")
        start, end = map(int, requested[-1].split("=")[1].split("-"))
        return Response(asset[start:end + 1], f"bytes {start}-{end}/8")

    monkeypatch.setattr(reopd_download, "urlopen", get_range)
    reopd_download.transfer("https://example.test/model", target, len(asset), hashlib.sha256(asset).hexdigest(), 4, 1, 1)
    assert requested == ["bytes=0-3", "bytes=2-5", "bytes=6-7"]
    assert target.read_bytes() == asset
    assert not target.with_suffix(".safetensors.part").exists()


@pytest.mark.parametrize("mode", ["range", "hash"])
def test_invalid_asset_never_becomes_ready(tmp_path, monkeypatch, mode):
    target = tmp_path / "model.safetensors"
    wrong_range = "bytes 2-5/4" if mode == "range" else "bytes 0-3/4"
    monkeypatch.setattr(reopd_download, "urlopen", lambda request, timeout: Response(b"xxxx", wrong_range))
    with pytest.raises(OSError if mode == "range" else ValueError):
        reopd_download.transfer("https://example.test/model", target, 4, hashlib.sha256(b"abcd").hexdigest(), 4, 1, 1)
    assert not target.exists()
