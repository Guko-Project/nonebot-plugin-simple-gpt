from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from nonebot.log import logger
from openai import AsyncOpenAI

from ..models import HistoryEntry
from ..plugin_config_inject import register_plugin_config_field
from ..plugin_system import (
    LLMRequestPayload,
    SimpleGPTPlugin,
    register_simple_gpt_plugin,
)

# ---------- 注册配置字段 ----------

register_plugin_config_field(
    "simple_gpt_proactive_filter_enabled",
    bool,
    default=True,
    description="是否启用主动回复过滤（仅对主动发言生效，@触发不受影响）",
)
register_plugin_config_field(
    "simple_gpt_proactive_filter_api_key",
    str,
    default="",
    description="过滤模型 API Key（留空则使用主 API Key）",
)
register_plugin_config_field(
    "simple_gpt_proactive_filter_api_base",
    str,
    default="",
    description="过滤模型 API Base URL（留空则使用主 API Base URL）",
)
register_plugin_config_field(
    "simple_gpt_proactive_filter_model",
    str,
    default="gemini-3.1-flash-lite-preview",
    description="用于判断是否值得回复的模型",
)

if TYPE_CHECKING:
    from .. import Config


FILTER_SYSTEM_PROMPT = """你是一个群聊旁观者，判断以下对话的最新消息是否值得插嘴回复。

判断标准：
- 值得回复（true）：有趣的话题、开放性问题、值得讨论的事件、能引发互动的内容
- 不值得回复（false）：两人之间的私聊、纯粹的日常闲聊（如"好的""哦""嗯"）、与你无关的内部交流、重复无意义的内容

只返回 true 或 false，不要有其他内容。"""


def _get_plugin_config() -> "Config":
    from .. import plugin_config
    return plugin_config


class ProactiveFilterPlugin(SimpleGPTPlugin):
    """主动回复过滤插件：判断是否值得插嘴，避免在无意义对话中发言。"""

    priority = 300  # 最先运行，过滤后其他插件无需执行

    def __init__(self) -> None:
        self._config_loaded = False
        self._enabled = False
        self._api_key = ""
        self._api_base = ""
        self._model = ""

    def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return
        config = _get_plugin_config()
        self._enabled = config.simple_gpt_proactive_filter_enabled
        self._api_key = (
            config.simple_gpt_proactive_filter_api_key or config.simple_gpt_api_key
        )
        self._api_base = (
            config.simple_gpt_proactive_filter_api_base or config.simple_gpt_api_base
        )
        self._model = config.simple_gpt_proactive_filter_model
        self._config_loaded = True
        logger.info(
            f"simple-gpt: proactive_filter 插件已加载 "
            f"(状态: {'已启用' if self._enabled else '未启用'}, model={self._model})"
        )

    async def before_llm_request(
        self, payload: LLMRequestPayload
    ) -> LLMRequestPayload:
        self._ensure_config_loaded()

        # 仅对主动发言生效
        if not self._enabled or not payload.extra.get("is_proactive"):
            return payload

        try:
            worth_replying = await self._judge(payload.history, payload.latest_message)
            if not worth_replying:
                logger.info(
                    f"simple-gpt: proactive_filter 判断无需回复: {payload.latest_message[:30]!r}"
                )
                payload.extra["skip_llm"] = True
        except Exception as exc:
            logger.warning(f"simple-gpt: proactive_filter 判断失败，默认放行 - {exc}")

        return payload

    async def _judge(
        self, history: Sequence[HistoryEntry], latest_message: str
    ) -> bool:
        """调用 LLM 判断是否值得回复，返回 True 表示值得。"""
        context_lines = []
        for entry in history[-6:]:
            role = "助手" if entry.is_bot else entry.speaker
            context_lines.append(f"{role}: {entry.content}")
        context_lines.append(f"最新消息: {latest_message}")
        context = "\n".join(context_lines)

        async with AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._api_base,
        ) as client:
            completion = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                max_tokens=5,
                temperature=0.0,
            )
        answer = (completion.choices[0].message.content or "").strip().lower()
        return answer.startswith("true")


register_simple_gpt_plugin(ProactiveFilterPlugin())
