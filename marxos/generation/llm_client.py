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
    """Primary answer model (profile-driven; pro tier for deep mode)."""
    return get_settings().models.deepseek_model


def deepseek_flash_model() -> str:
    """Cheap tier for auxiliary and everyday tasks (fast/standard modes,
    book location, citation verification, out-of-domain answers)."""
    return get_settings().models.deepseek_flash_model


def generation_model(mode: str = "deep") -> str:
    """Model for main answer generation given a performance mode.

    Only deep mode uses the primary (pro) model; fast and standard modes run
    on the flash tier so everyday answers stay cheap.
    """
    if str(mode or "deep").lower() in {"fast", "standard"}:
        return deepseek_flash_model()
    return deepseek_model()


def deepseek_extra_body() -> dict:
    """Extra request body for all DeepSeek V4 calls.

    V4 defaults to thinking mode; MarxOS answers are retrieval-grounded and
    citation-audited, so reasoning tokens only add latency and cost. Set
    MARXOS_DEEPSEEK_THINKING=1 to opt back in (high intensity).
    """
    if os.getenv("MARXOS_DEEPSEEK_THINKING", "0").lower() in {"1", "true", "yes", "on"}:
        return {"thinking": {"type": "enabled", "reasoning_effort": "high"}}
    return {"thinking": {"type": "disabled"}}


__all__ = [
    "create_deepseek_client",
    "deepseek_extra_body",
    "deepseek_flash_model",
    "deepseek_model",
    "generation_model",
]
