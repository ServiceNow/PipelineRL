"""Embedding helpers for the privacy_hopqa domain."""

import logging
import threading

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "openai": "text-embedding-ada-002",
    "openrouter": "openai/text-embedding-ada-002",
    "huggingface": "Qwen/Qwen3-Embedding-4B",
    "vllm": "Qwen/Qwen3-Embedding-4B",
}

_HF_MODEL = None
_HF_MODEL_NAME = None
_HF_LOCK = threading.Lock()


def _openai_client(base_url: str, api_key: str | None) -> OpenAI:
    if not api_key:
        raise ValueError("embedding_api_key is required for OpenAI-compatible embeddings.")
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)


def _get_openai_embeddings(texts: list[str], model: str, api_key: str | None) -> list[list[float]]:
    client = _openai_client("https://api.openai.com/v1", api_key)
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def _get_openrouter_embeddings(
    texts: list[str],
    model: str,
    base_url: str | None,
    api_key: str | None,
) -> list[list[float]]:
    client = _openai_client(base_url or "https://openrouter.ai/api/v1", api_key)
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def _get_huggingface_embeddings(texts: list[str], model: str, device: str | None) -> list[list[float]]:
    global _HF_MODEL, _HF_MODEL_NAME

    if _HF_MODEL is None or _HF_MODEL_NAME != model:
        with _HF_LOCK:
            if _HF_MODEL is None or _HF_MODEL_NAME != model:
                embedding_device = device
                force_cpu = bool(embedding_device and embedding_device.lower().startswith("cpu"))
                if not embedding_device and torch.cuda.is_available():
                    num_gpus = torch.cuda.device_count()
                    embedding_device = f"cuda:{num_gpus - 1}" if num_gpus > 1 else "cuda:0"
                _HF_MODEL = SentenceTransformer(
                    model,
                    device="cpu" if force_cpu else embedding_device or None,
                    tokenizer_kwargs={"padding_side": "left"},
                )
                _HF_MODEL_NAME = model
                logger.info("Loaded embedding model %s on %s", model, embedding_device or "default device")

    embeddings = _HF_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()


def _get_vllm_embeddings(texts: list[str], model: str, base_url: str | None, api_key: str | None) -> list[list[float]]:
    if not base_url:
        raise ValueError("embedding_base_url is required when embedding_provider=vllm.")
    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "not-needed")
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


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

    provider_name = (provider or "huggingface").lower()
    model_name = model or DEFAULT_MODELS[provider_name]

    if model_name.startswith("text-embedding-") and provider_name not in {"openai", "openrouter"}:
        raise ValueError(
            f"Model '{model_name}' requires an OpenAI-compatible embedding provider, not '{provider_name}'."
        )

    if provider_name == "openai":
        return _get_openai_embeddings(texts, model_name, api_key)
    if provider_name == "openrouter":
        return _get_openrouter_embeddings(texts, model_name, base_url, api_key)
    if provider_name == "huggingface":
        return _get_huggingface_embeddings(texts, model_name, device)
    if provider_name == "vllm":
        return _get_vllm_embeddings(texts, model_name, base_url, api_key)

    raise ValueError(f"Unknown embedding provider: {provider_name}")
