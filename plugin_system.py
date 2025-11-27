from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from nonebot.log import logger

from .models import HistoryEntry


@dataclass
class LLMRequestPayload:
    """Payload passed to plugins before calling the language model."""

    prompt: str
    history: Sequence[HistoryEntry]
    sender: str
    latest_message: str
    images: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponsePayload:
    """Payload passed to plugins after receiving the language model output."""

    content: str
    request: LLMRequestPayload
    extra: Dict[str, Any] = field(default_factory=dict)


class SimpleGPTPlugin:
    """Base class for pipeline plugins."""

    priority: int = 0

    async def before_llm_request(self, payload: LLMRequestPayload) -> LLMRequestPayload:
        """Modify the payload before the OpenAI request."""
        return payload

    async def after_llm_response(
        self, payload: LLMResponsePayload
    ) -> LLMResponsePayload:
        """Modify the payload after receiving the OpenAI response."""
        return payload


class PluginManager:
    def __init__(self) -> None:
        self._plugins: List[Tuple[int, SimpleGPTPlugin]] = []

    def register(self, plugin: SimpleGPTPlugin, *, priority: int | None = None) -> None:
        plugin_priority = (
            priority if priority is not None else getattr(plugin, "priority", 0)
        )
        self._plugins.append((plugin_priority, plugin))
        self._plugins.sort(key=lambda item: item[0], reverse=True)
        self._log_plugin_order()

    def _log_plugin_order(self) -> None:
        if not self._plugins:
            logger.info("simple-gpt: 当前未加载任何插件。")
            return
        order = ", ".join(
            f"{plugin.__class__.__name__}(priority={priority})"
            for priority, plugin in self._plugins
        )
        logger.info("simple-gpt: 插件加载顺序 -> %s", order)

    async def run_before_llm_request(
        self, payload: LLMRequestPayload
    ) -> LLMRequestPayload:
        for _, plugin in self._plugins:
            payload = await plugin.before_llm_request(payload)
        return payload

    async def run_after_llm_response(
        self, payload: LLMResponsePayload
    ) -> LLMResponsePayload:
        for _, plugin in self._plugins:
            payload = await plugin.after_llm_response(payload)
        return payload


plugin_manager = PluginManager()


def register_simple_gpt_plugin(
    plugin: SimpleGPTPlugin, *, priority: int | None = None
) -> None:
    plugin_manager.register(plugin, priority=priority)


async def emit_before_llm_request(
    payload: LLMRequestPayload,
) -> LLMRequestPayload:
    return await plugin_manager.run_before_llm_request(payload)


async def emit_after_llm_response(
    payload: LLMResponsePayload,
) -> LLMResponsePayload:
    return await plugin_manager.run_after_llm_response(payload)
