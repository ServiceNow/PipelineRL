"""HTTP client for the external privacy helper service."""

from typing import Any

import requests


class PrivacyHelperClient:
    """Thin synchronous client for the remote privacy helper service."""

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        if not base_url:
            raise ValueError("helper service URL must be provided when remote helper mode is enabled")

        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._session.request(
            method=method,
            url=f"{self.base_url}{path}",
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object from helper service for {path}")
        return data

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        payload: dict[str, Any] = {"texts": texts}
        if model is not None:
            payload["model"] = model
        data = self._request("POST", "/embed", payload)
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError("Helper /embed response did not include embeddings")
        return embeddings

    def search_local(
        self,
        task_id: str,
        query: str,
        k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        data = self._request(
            "POST",
            "/local/search",
            {
                "task_id": task_id,
                "query": query,
                "k": int(k),
                "threshold": float(threshold),
            },
        )
        hits = data.get("hits")
        if not isinstance(hits, list):
            raise ValueError("Helper /local/search response did not include hits")
        return hits

    def search_browsecomp(
        self,
        query: str,
        task_id: str | None,
        k: int,
        max_chars: int | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"query": query, "k": int(k)}
        if task_id is not None:
            payload["task_id"] = task_id
        if max_chars is not None:
            payload["max_chars"] = int(max_chars)
        data = self._request("POST", "/browsecomp/search", payload)
        hits = data.get("hits")
        if not isinstance(hits, list):
            raise ValueError("Helper /browsecomp/search response did not include hits")
        return hits

    def close(self) -> None:
        self._session.close()
