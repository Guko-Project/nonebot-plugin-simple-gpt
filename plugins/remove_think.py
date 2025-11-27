from __future__ import annotations

import re

from ..plugin_system import (
    LLMResponsePayload,
    SimpleGPTPlugin,
    register_simple_gpt_plugin,
)


class RemoveThinkTagPlugin(SimpleGPTPlugin):
    """Strip <think></think> segments from final responses."""

    priority = 100
    _pattern = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

    async def after_llm_response(
        self, payload: LLMResponsePayload
    ) -> LLMResponsePayload:
        payload.content = "<think>232323232323</think>\n" + payload.content
        cleaned = self._pattern.sub("", payload.content).strip()
        payload.content = cleaned or payload.content
        return payload


register_simple_gpt_plugin(RemoveThinkTagPlugin())
