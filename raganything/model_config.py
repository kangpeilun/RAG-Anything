"""
Model configuration for RAGAnything.

Provides factory functions to create LLM, VLM, embedding, and rerank callables
from environment variables, supporting **different OpenAI-compatible API providers**
for each model type independently.

Usage:
    from raganything.model_config import create_llm_model_func, create_vlm_model_func, create_embedding_func

    # Auto-create from environment variables
    rag = RAGAnything(
        config=config,
        llm_model_func=create_llm_model_func(),
        vision_model_func=create_vlm_model_func(),
        embedding_func=create_embedding_func(),
    )

    # Or configure each model type separately
    from raganything.model_config import OpenAIModelConfig

    llm_cfg = OpenAIModelConfig(api_key="...", base_url="...", model="deepseek-chat")
    rag = RAGAnything(
        config=config,
        llm_model_func=create_llm_model_func(llm_cfg),
        ...
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Optional

from lightrag.utils import EmbeddingFunc

# ---------------------------------------------------------------------------
# Locate .env relative to the project root (two levels up from this file,
# i.e. raganything/model_config.py -> raganything/ -> project root).
# This ensures env vars are available at module-import time regardless of
# the current working directory.
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.is_file():
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=str(_ENV_PATH), override=False)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Ensure LightRAG-compatible env vars are set from our prefixed vars.
# LightRAG's internal code (e.g. create_openai_async_client) reads the
# standard OPENAI_API_KEY, AZURE_OPENAI_API_KEY, etc. directly from the
# environment as a fallback.  We mirror our prefixed vars into those so
# that even if a code path loses the explicitly passed api_key, the
# fallback still works.
# ---------------------------------------------------------------------------
_FALLBACK_ENV_MAP = {
    "OPENAI_API_KEY": "LLM_API_KEY",
    "OPENAI_API_BASE": "LLM_BASE_URL",
}
for target, source in _FALLBACK_ENV_MAP.items():
    if target not in os.environ and source in os.environ:
        os.environ[target] = os.environ[source]


@dataclass
class OpenAIModelConfig:
    """Configuration for an OpenAI-compatible model endpoint.

    Each field maps to an environment variable prefixed by *prefix*:

    ================ ====================== =========
    Field            Env var                Default
    ================ ====================== =========
    ``api_key``      ``{PREFIX}_API_KEY``   ``""``
    ``base_url``     ``{PREFIX}_BASE_URL``  ``""``
    ``model``        ``{PREFIX}_MODEL``     ``""``
    ================ ====================== =========
    """

    api_key: str = ""
    """API key for the provider."""

    base_url: str = ""
    """Base URL of the OpenAI-compatible endpoint (e.g. ``https://api.deepseek.com/v1``)."""

    model: str = ""
    """Model name (e.g. ``deepseek-chat``, ``gpt-4o``)."""

    @classmethod
    def from_env(cls, prefix: str) -> "OpenAIModelConfig":
        """Read config from environment variables with the given *prefix*.

        Args:
            prefix: Uppercase prefix such as ``"LLM"``, ``"VLM"``, ``"EMBEDDING"``.

        Returns:
            OpenAIModelConfig populated from the matching env vars.
        """
        return cls(
            api_key=os.getenv(f"{prefix}_API_KEY", ""),
            base_url=os.getenv(f"{prefix}_BASE_URL", ""),
            model=os.getenv(f"{prefix}_MODEL", ""),
        )

    @property
    def is_configured(self) -> bool:
        """Whether enough config is present to create a working model function."""
        return bool(self.api_key and self.model)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_llm_model_func(config: OpenAIModelConfig | None = None) -> Callable:
    """Create an LLM model function.

    The returned async function has the signature expected by LightRAG /
    RAG-Anything for ``llm_model_func``.

    Args:
        config: Model configuration.  If ``None``, read from ``LLM_*`` env vars.

    Returns:
        An async callable ``(prompt, system_prompt, history_messages, **kwargs) -> str``.
    """
    if config is None:
        config = OpenAIModelConfig.from_env("LLM")

    from lightrag.llm.openai import openai_complete_if_cache

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        return await openai_complete_if_cache(
            config.model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=config.api_key,
            base_url=config.base_url,
            **kwargs,
        )

    return llm_model_func


def create_vlm_model_func(config: OpenAIModelConfig | None = None) -> Callable:
    """Create a VLM (vision) model function.

    Supports three calling modes:
    * **messages format** – for multimodal VLM-enhanced queries.
    * **single image** – ``image_data`` (base64) + ``prompt``.
    * **pure text** – falls back to text-only completion.

    Args:
        config: Model configuration.  If ``None``, read from ``VLM_*`` env vars.

    Returns:
        An async callable with the signature expected by ``vision_model_func``.
    """
    if config is None:
        config = OpenAIModelConfig.from_env("VLM")

    from lightrag.llm.openai import openai_complete_if_cache

    async def vision_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        image_data: str | None = None,
        messages: list[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        # 1. Pre-formatted messages (multimodal VLM query)
        if messages:
            return await openai_complete_if_cache(
                config.model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=config.api_key,
                base_url=config.base_url,
                **kwargs,
            )

        # 2. Single image (traditional image processing)
        if image_data:
            user_content: list[dict] = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    },
                },
            ]
            if prompt:
                user_content.insert(0, {"type": "text", "text": prompt})

            built_messages = [{"role": "user", "content": user_content}]
            if system_prompt:
                built_messages.insert(
                    0, {"role": "system", "content": system_prompt}
                )

            return await openai_complete_if_cache(
                config.model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=built_messages,
                api_key=config.api_key,
                base_url=config.base_url,
                **kwargs,
            )

        # 3. Pure text (fallback)
        return await openai_complete_if_cache(
            config.model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=config.api_key,
            base_url=config.base_url,
            **kwargs,
        )

    return vision_model_func


def create_embedding_func(
    config: OpenAIModelConfig | None = None,
    embedding_dim: int | None = None,
    max_token_size: int = 8192,
) -> EmbeddingFunc:
    """Create an :class:`~lightrag.utils.EmbeddingFunc`.

    Args:
        config: Model configuration.  If ``None``, read from ``EMBEDDING_*`` env vars.
        embedding_dim: Output dimension.  Falls back to ``EMBEDDING_DIM`` env var,
            then to 3072.
        max_token_size: Maximum tokens per chunk.  Falls back to
            ``MAX_EMBED_TOKENS`` env var, then to 8192.

    Returns:
        A fully configured :class:`EmbeddingFunc`.
    """
    if config is None:
        config = OpenAIModelConfig.from_env("EMBEDDING")
    if embedding_dim is None:
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
    if max_token_size is None:
        max_token_size = int(os.getenv("MAX_EMBED_TOKENS", "8192"))

    from lightrag.llm.openai import openai_embed

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=partial(
            openai_embed.func,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        ),
    )


def create_rerank_func(config: OpenAIModelConfig | None = None) -> Callable | None:
    """Create an optional rerank function.

    .. note::
       Rerank APIs vary widely across providers and are not standardised under
       an OpenAI-compatible protocol.  This function currently returns ``None``
       (no-op) and is provided as a placeholder for users who want to wire up
       a custom rerank function via ``lightrag_kwargs["rerank_model_func"]``.

    Args:
        config: Model configuration.  If ``None``, read from ``RERANK_*`` env vars.

    Returns:
        ``None`` (rerank is not auto-created).
    """
    return None