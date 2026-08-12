from __future__ import annotations

import os

from openai import OpenAI

from marxos.config import get_settings


def create_deepseek_client(max_retries: int = 2, timeout: float = 30.0) -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=settings.models.deepseek_base_url,
        max_retries=max_retries,
        timeout=timeout,
    )


def deepseek_model() -> str:
    return get_settings().models.deepseek_model


__all__ = ["create_deepseek_client", "deepseek_model"]

