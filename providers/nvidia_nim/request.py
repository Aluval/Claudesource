"""Request builder for NVIDIA NIM provider."""

import re
from copy import deepcopy
from typing import Any

from loguru import logger

from config.nim import NimSettings
from core.anthropic import build_base_request_body, set_if_not_none

_IMMUTABLE_SAMPLING_RE = re.compile(
    r"(?P<name>top_p|temperature|top_k|presence_penalty|frequency_penalty)"
    r" is immutable for this model and must be (?P<value>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _set_extra(
    extra_body: dict[str, Any], key: str, value: Any, ignore_value: Any = None
) -> None:
    if key in extra_body:
        return
    if value is None:
        return
    if ignore_value is not None and value == ignore_value:
        return
    extra_body[key] = value


def clone_body_without_reasoning_budget(body: dict[str, Any]) -> dict[str, Any] | None:
    """Clone a request body and strip only reasoning_budget fields."""
    cloned_body = deepcopy(body)
    extra_body = cloned_body.get("extra_body")
    if not isinstance(extra_body, dict):
        return None

    removed = extra_body.pop("reasoning_budget", None) is not None

    chat_template_kwargs = extra_body.get("chat_template_kwargs")
    if (
        isinstance(chat_template_kwargs, dict)
        and chat_template_kwargs.pop("reasoning_budget", None) is not None
    ):
        removed = True

    if not extra_body:
        cloned_body.pop("extra_body", None)

    if not removed:
        return None

    return cloned_body


def clone_body_without_chat_template(body: dict[str, Any]) -> dict[str, Any] | None:
    """Clone a request body and strip only chat_template."""
    cloned_body = deepcopy(body)
    extra_body = cloned_body.get("extra_body")
    if not isinstance(extra_body, dict):
        return None

    if extra_body.pop("chat_template", None) is None:
        return None

    if not extra_body:
        cloned_body.pop("extra_body", None)

    return cloned_body


def apply_known_model_sampling_overrides(body: dict[str, Any]) -> None:
    """Force sampling values NVIDIA pins on hosted models.

    Claude Code often sends ``top_p=1``. Kimi rejects that with HTTP 400 and
    requires ``0.95``. GLM accepts ``0.95``, so it is safe for all NIM models.
    """
    body["top_p"] = 0.95
    extra_body = body.get("extra_body")
    if isinstance(extra_body, dict) and "top_p" in extra_body:
        extra_body["top_p"] = 0.95


def clone_body_with_immutable_sampling(
    body: dict[str, Any], error_text: str
) -> dict[str, Any] | None:
    """Clone a request body with the sampling value NVIDIA says is required."""
    match = _IMMUTABLE_SAMPLING_RE.search(error_text)
    if match is None:
        return None

    name = match.group("name").lower()
    raw_value = match.group("value")
    required: int | float = int(raw_value) if name == "top_k" else float(raw_value)
    current = body.get(name)
    if current is not None:
        try:
            if float(current) == float(required):
                return None
        except (TypeError, ValueError):
            pass

    cloned_body = deepcopy(body)
    cloned_body[name] = required
    extra_body = cloned_body.get("extra_body")
    if isinstance(extra_body, dict) and name in extra_body:
        extra_body[name] = required
    return cloned_body


def build_request_body(
    request_data: Any, nim: NimSettings, *, thinking_enabled: bool
) -> dict:
    """Build OpenAI-format request body from Anthropic request."""
    logger.debug(
        "NIM_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    body = build_base_request_body(
        request_data,
        include_thinking=thinking_enabled,
    )

    # NIM-specific max_tokens: cap against nim.max_tokens
    max_tokens = body.get("max_tokens") or getattr(request_data, "max_tokens", None)
    if max_tokens is None:
        max_tokens = nim.max_tokens
    elif nim.max_tokens:
        max_tokens = min(max_tokens, nim.max_tokens)
    set_if_not_none(body, "max_tokens", max_tokens)

    # NIM-specific temperature/top_p: fall back to NIM defaults if request didn't set
    if body.get("temperature") is None and nim.temperature is not None:
        body["temperature"] = nim.temperature
    if body.get("top_p") is None and nim.top_p is not None:
        body["top_p"] = nim.top_p

    # NIM-specific stop sequences fallback
    if "stop" not in body and nim.stop:
        body["stop"] = nim.stop

    if nim.presence_penalty != 0.0:
        body["presence_penalty"] = nim.presence_penalty
    if nim.frequency_penalty != 0.0:
        body["frequency_penalty"] = nim.frequency_penalty
    if nim.seed is not None:
        body["seed"] = nim.seed

    body["parallel_tool_calls"] = nim.parallel_tool_calls

    # Handle non-standard parameters via extra_body
    extra_body: dict[str, Any] = {}
    request_extra = getattr(request_data, "extra_body", None)
    if request_extra:
        extra_body.update(request_extra)

    if thinking_enabled:
        chat_template_kwargs = extra_body.setdefault(
            "chat_template_kwargs", {"thinking": True, "enable_thinking": True}
        )
        if isinstance(chat_template_kwargs, dict):
            chat_template_kwargs.setdefault("reasoning_budget", max_tokens)

    req_top_k = getattr(request_data, "top_k", None)
    top_k = req_top_k if req_top_k is not None else nim.top_k
    _set_extra(extra_body, "top_k", top_k, ignore_value=-1)
    _set_extra(extra_body, "min_p", nim.min_p, ignore_value=0.0)
    _set_extra(
        extra_body, "repetition_penalty", nim.repetition_penalty, ignore_value=1.0
    )
    _set_extra(extra_body, "min_tokens", nim.min_tokens, ignore_value=0)
    _set_extra(extra_body, "chat_template", nim.chat_template)
    _set_extra(extra_body, "request_id", nim.request_id)
    _set_extra(extra_body, "ignore_eos", nim.ignore_eos)

    if extra_body:
        body["extra_body"] = extra_body

    apply_known_model_sampling_overrides(body)

    logger.debug(
        "NIM_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
