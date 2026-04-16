"""Async HTTP client for the privacy_hopqa helper service."""

from typing import Any

import aiohttp


class AsyncPrivacyHopQAHelperClient:
    """Thin async client for the remote helper service."""

    def __init__(self, base_url: str, session: aiohttp.ClientSession, timeout_s: float = 30.0) -> None:
        if not base_url:
            raise ValueError("helper service URL must be provided when remote helper mode is enabled")
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.timeout_s = float(timeout_s)

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        async with self.session.request(
            method=method,
            url=f"{self.base_url}{path}",
            json=payload,
            timeout=self.timeout_s,
            ssl=False,
        ) as response:
            response.raise_for_status()
            data = await response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object from helper service for {path}")
        return data

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health")

    async def search_local(
        self,
        task_id: str,
        query: str,
        k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        data = await self._request(
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

    async def search_browsecomp(
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
        data = await self._request("POST", "/browsecomp/search", payload)
        hits = data.get("hits")
        if not isinstance(hits, list):
            raise ValueError("Helper /browsecomp/search response did not include hits")
        return hits
