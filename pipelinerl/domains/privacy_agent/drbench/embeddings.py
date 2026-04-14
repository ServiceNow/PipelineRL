"""Embedding helpers for the vendored privacy_agent DRBench slice."""


import logging
import os
import threading
from typing import List, Optional

from openai import OpenAI

from .config import OPENAI_API_KEY, OPENROUTER_API_KEY, OPENROUTER_API_URL, VLLM_EMBEDDING_URL

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


def _get_openai_embeddings(texts: List[str], model: str) -> List[List[float]]:
    client = OpenAI(base_url="https://api.openai.com/v1", api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def _get_openrouter_embeddings(texts: List[str], model: str) -> List[List[float]]:
    client = OpenAI(base_url=OPENROUTER_API_URL, api_key=OPENROUTER_API_KEY)
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def _get_huggingface_embeddings(texts: List[str], model: str) -> List[List[float]]:
    global _HF_MODEL, _HF_MODEL_NAME

    from sentence_transformers import SentenceTransformer

    if _HF_MODEL is None or _HF_MODEL_NAME != model:
        with _HF_LOCK:
            if _HF_MODEL is None or _HF_MODEL_NAME != model:
                import torch

                embedding_device = os.getenv("DRBENCH_EMBEDDING_DEVICE")
                force_cpu = bool(embedding_device and embedding_device.lower().startswith("cpu"))

                if torch.cuda.is_available() and not force_cpu:
                    num_gpus = torch.cuda.device_count()
                    default_device = f"cuda:{num_gpus - 1}" if num_gpus > 1 else "cuda:0"
                    embedding_device = embedding_device or default_device
                    try:
                        _HF_MODEL = SentenceTransformer(
                            model,
                            device=embedding_device,
                            model_kwargs={
                                "attn_implementation": "flash_attention_2",
                                "torch_dtype": torch.float16,
                            },
                            tokenizer_kwargs={"padding_side": "left"},
                        )
                    except Exception:
                        _HF_MODEL = SentenceTransformer(
                            model,
                            device=embedding_device,
                            tokenizer_kwargs={"padding_side": "left"},
                        )
                else:
                    _HF_MODEL = SentenceTransformer(
                        model,
                        device="cpu" if force_cpu else None,
                        tokenizer_kwargs={"padding_side": "left"},
                    )

                _HF_MODEL_NAME = model
                logger.info("Loaded embedding model %s", model)

    embeddings = _HF_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()


def _get_vllm_embeddings(texts: List[str], model: str) -> List[List[float]]:
    if not VLLM_EMBEDDING_URL:
        raise ValueError(
            "VLLM_EMBEDDING_URL must be set when using DRBENCH_EMBEDDING_PROVIDER=vllm"
        )
    client = OpenAI(base_url=f"{VLLM_EMBEDDING_URL}/v1", api_key="not-needed")
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def get_embeddings(
    texts: List[str],
    model: Optional[str] = None,
    provider: Optional[str] = None,
    helper_client=None,
) -> List[List[float]]:
    if not texts:
        return []

    provider_name = (provider or "openai").lower()
    model_name = model or DEFAULT_MODELS[provider_name]

    if helper_client is not None:
        return helper_client.embed(texts, model=model_name)

    if model_name.startswith("text-embedding-") and provider_name not in {"openai", "openrouter"}:
        raise ValueError(
            f"Model '{model_name}' requires an OpenAI-compatible embedding provider, not '{provider_name}'."
        )

    if provider_name == "openai":
        return _get_openai_embeddings(texts, model_name)
    if provider_name == "openrouter":
        return _get_openrouter_embeddings(texts, model_name)
    if provider_name == "huggingface":
        return _get_huggingface_embeddings(texts, model_name)
    if provider_name == "vllm":
        return _get_vllm_embeddings(texts, model_name)

    raise ValueError(f"Unknown embedding provider: {provider_name}")
