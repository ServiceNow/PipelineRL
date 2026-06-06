"""Embedding helpers for the privacy_hopqa domain.

Two backends: in-process ``huggingface`` (SentenceTransformer) and a remote
OpenAI-compatible ``vllm`` endpoint. Both default to ``Qwen/Qwen3-Embedding-4B``.
"""

import functools
import logging

import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"


@functools.cache
def _load_hf_model(model: str, device: str | None) -> SentenceTransformer:
    # Memoize the loaded weights (loading is expensive). functools.cache is
    # thread-safe and replaces a hand-rolled global + lock; there are only ever a
    # couple of (model, device) keys, so no eviction is needed.
    if device is None and torch.cuda.is_available():
        # Default to the last visible GPU so the embedder doesn't crowd the
        # trainer/inference shards that sit on cuda:0.
        num_gpus = torch.cuda.device_count()
        device = f"cuda:{num_gpus - 1}" if num_gpus > 1 else "cuda:0"
    logger.info("Loading embedding model %s on %s", model, device or "default device")
    return SentenceTransformer(model, device=device, tokenizer_kwargs={"padding_side": "left"})


def get_embeddings(
    texts: list[str],
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    device: str | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    model = model or DEFAULT_EMBEDDING_MODEL
    provider = (provider or "huggingface").lower()

    if provider == "huggingface":
        embeddings = _load_hf_model(model, device).encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings.tolist()

    if provider == "vllm":
        if not base_url:
            raise ValueError("embedding_base_url is required when embedding_provider=vllm.")
        client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "not-needed")
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]

    raise ValueError(f"Unknown embedding provider: {provider!r}. Use 'huggingface' or 'vllm'.")
