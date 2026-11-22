"""Text/JSON client for the existing search server; shares its Qdrant owner."""
import json
import urllib.error
import urllib.request
from urllib.parse import quote


class SearchClient:
    def __init__(self, base_url="http://127.0.0.1:8020"):
        self.base_url = base_url.rstrip("/")
        self.meta = self._request("/health")
        if not self.meta.get("ready"):
            raise RuntimeError("Search backend is not ready")

    def _request(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            try:
                detail = json.load(error).get("detail", str(error))
            except (ValueError, AttributeError):
                detail = str(error)
            if error.code == 404:
                raise KeyError(detail) from error
            raise ValueError(str(detail)) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RuntimeError("Search backend unavailable") from error

    def search(self, query, top_k=5, mode="bm25"):
        return self._request("/search", {"query": query, "top_k": top_k, "mode": mode})

    def read(self, document_id):
        return self._request("/documents/" + quote(document_id, safe=""))

    def close(self):
        pass
