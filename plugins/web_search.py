from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

from nonebot.log import logger
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..models import HistoryEntry
from ..plugin_config_inject import register_plugin_config_field
from ..plugin_system import (
    LLMRequestPayload,
    SimpleGPTPlugin,
    register_simple_gpt_plugin,
)

# 注册插件配置字段
register_plugin_config_field(
    "simple_gpt_search_check_api_key",
    str,
    default="",
    description="用于判断是否需要搜索的 API Key",
)
register_plugin_config_field(
    "simple_gpt_search_check_base_url",
    str,
    default="",
    description="用于判断是否需要搜索的 API Base URL（留空则使用主系统配置）",
)
register_plugin_config_field(
    "simple_gpt_search_check_model",
    str,
    default="gemini-3.1-flash-lite-preview",
    description="用于判断是否需要搜索的模型",
)
register_plugin_config_field(
    "simple_gpt_search_api_key",
    str,
    default="",
    description="豆包搜索 API Key (doubao-seed)",
)
register_plugin_config_field(
    "simple_gpt_search_base_url",
    str,
    default="https://ark.cn-beijing.volces.com/api/v3",
    description="豆包搜索 API Base URL",
)
register_plugin_config_field(
    "simple_gpt_search_model",
    str,
    default="doubao-seed-1-6-lite-251015",
    description="豆包搜索模型",
)
register_plugin_config_field(
    "simple_gpt_search_enabled",
    bool,
    default=False,
    description="是否启用网络搜索功能",
)

if TYPE_CHECKING:
    from .. import Config


SEARCH_SYSTEM_PROMPT = """
你是一个专业的搜索助手，现在的目标是为下一步的正式回答提供帮助。
由于之后无法再次进行搜索，如果认为需要搜索你必须提供确切的信息。
请在网络上搜索相关信息并准确提供用户问题所需要的信息，如果认为不需要搜索则什么也不返回。
"""


class SearchDecision(BaseModel):
    """搜索决策数据结构"""

    is_search_needed: bool
    search_query: str  # 需要搜索的具体内容


def _get_plugin_config() -> "Config":
    """延迟导入主配置，避免循环依赖。"""
    from .. import plugin_config

    return plugin_config


class WebSearchPlugin(SimpleGPTPlugin):
    """Add web search results to the prompt using doubao-seed model."""

    priority = 190  # 在 datetime_weather (200) 之后，确保搜索结果在最底部

    def __init__(self):
        # 延迟配置加载，避免循环导入
        self._config_loaded = False
        self.enabled = False
        self.check_api_key = ""
        self.check_base_url = ""
        self.check_model = ""
        self.search_api_key = ""
        self.search_base_url = ""
        self.search_model = ""

    def _ensure_config_loaded(self) -> None:
        """延迟加载配置，在第一次使用时执行。"""
        if self._config_loaded:
            return

        config = _get_plugin_config()
        self.enabled = config.simple_gpt_search_enabled

        # 如果判断搜索的 API Key 和 Base URL 未配置，则使用主系统配置
        self.check_api_key = config.simple_gpt_search_check_api_key or config.simple_gpt_api_key
        self.check_base_url = config.simple_gpt_search_check_base_url or config.simple_gpt_api_base
        self.check_model = config.simple_gpt_search_check_model

        self.search_api_key = config.simple_gpt_search_api_key
        self.search_base_url = config.simple_gpt_search_base_url
        self.search_model = config.simple_gpt_search_model
        self._config_loaded = True

        logger.info(
            f"simple-gpt: web_search 插件已加载 "
            f"(状态: {'已启用' if self.enabled else '未启用'}, "
            f"判断API: {'独立配置' if config.simple_gpt_search_check_api_key else '使用主配置'})"
        )

    async def before_llm_request(
        self, payload: LLMRequestPayload
    ) -> LLMRequestPayload:
        """Check if search is needed and add search results to the prompt."""

        # 确保配置已加载
        self._ensure_config_loaded()

        # 如果未启用或配置不完整，跳过
        if not self.enabled:
            logger.debug("simple-gpt: 网络搜索功能未启用，跳过")
            return payload

        if not self.check_api_key or not self.search_api_key:
            logger.warning(
                "simple-gpt: 网络搜索功能已启用，但 API Key 未配置，跳过"
            )
            return payload

        try:
            # 提取用户的问题（从 latest_message 中获取）
            question = payload.latest_message

            # 从 history 和 extra 构造干净的上下文（不包含人设等系统提示）
            context = self._build_clean_context(payload.history, question, payload.extra)

            # 判断是否需要搜索，并获取搜索查询
            search_decision = await self._check_search_needed(context, question)

            if not search_decision["is_needed"]:
                logger.debug("simple-gpt: 判断不需要进行网络搜索")
                return payload

            # 执行搜索
            search_query = search_decision["query"]
            logger.info(f"simple-gpt: 检测到需要搜索，搜索内容：{search_query}")
            search_result = await self._search_answer(search_query)

            if search_result:
                # 将搜索结果添加到 prompt 底部
                search_context = f"\n\n这是用来参考的搜索结果：{search_result}"
                payload.prompt = payload.prompt + search_context

                # 将搜索结果也保存到 extra 中供其他插件使用
                payload.extra["web_search_result"] = search_result
                payload.extra["web_search_query"] = search_query

                logger.info("simple-gpt: 已添加网络搜索结果到 prompt")
            else:
                logger.warning("simple-gpt: 搜索执行完成但未获取到有效结果")

        except Exception as exc:
            logger.exception(f"simple-gpt: 网络搜索插件执行失败 - {exc}")
            # 失败时不影响正常流程，继续返回原始 payload

        return payload

    def _build_clean_context(
        self,
        history: Sequence[HistoryEntry],
        current_question: str,
        extra: Dict[str, Any],
    ) -> str:
        """从历史记录和额外信息构造干净的上下文，不包含系统提示词等干扰内容

        Args:
            history: 对话历史记录
            current_question: 当前用户问题
            extra: payload 中的额外信息（如时间、天气等）

        Returns:
            str: 干净的对话上下文
        """
        context_parts = []

        # 添加已知的上下文信息（来自其他插件）
        if extra:
            extra_info_parts = []
            for key, value in extra.items():
                if key not in ["web_search_result", "web_search_query"]:
                    extra_info_parts.append(f"{key}: {value}")

            # 如果有其他已知信息，可以在这里添加
            # 注意：不要添加 web_search_result，避免循环

            if extra_info_parts:
                context_parts.append("[已知信息]")
                context_parts.extend(extra_info_parts)
                context_parts.append("")  # 空行分隔

        # 构造对话历史
        if history:
            for entry in history:
                role = "助手" if entry.is_bot else "用户"
                context_parts.append(f"{role}: {entry.content}")

        # 添加当前问题
        context_parts.append(f"用户: {current_question}")

        # 限制上下文长度，只保留最近的对话
        # 注意：已知信息部分始终保留
        max_history_entries = 10

        # 找到已知信息的结束位置
        extra_info_end = 0
        for i, part in enumerate(context_parts):
            if part == "":  # 找到分隔空行
                extra_info_end = i + 1
                break

        # 如果对话历史过长，只保留最近的条目
        if len(context_parts) > extra_info_end + max_history_entries:
            context_parts = (
                context_parts[:extra_info_end]
                + context_parts[-(max_history_entries):]
            )

        return "\n".join(context_parts)

    async def _check_search_needed(self, context: str, question: str) -> dict:
        """判断是否需要搜索，并返回搜索内容

        Args:
            context: 对话上下文
            question: 当前用户问题

        Returns:
            dict: {"is_needed": bool, "query": str}
        """
        try:
            async with AsyncOpenAI(
                api_key=self.check_api_key,
                base_url=self.check_base_url,
            ) as client:
                # 构建判断搜索的提示词
                system_prompt = """你是一个搜索决策助手。
根据用户的对话上下文和当前问题，判断是否需要进行网络搜索。
如果需要搜索，请提供具体的搜索查询内容（用简洁的关键词或问题描述）。
如果不需要搜索，search_query 字段请留空。"""

                user_prompt = f"""对话上下文：
{context}

当前问题：
{question}

请判断是否需要网络搜索，并提供搜索内容。"""

                completion = await client.beta.chat.completions.parse(
                    model=self.check_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=SearchDecision,
                )

                result = completion.choices[0].message.parsed
                return {
                    "is_needed": bool(result.is_search_needed),
                    "query": result.search_query or question,  # 如果 search_query 为空，使用原问题
                }

        except Exception as exc:
            logger.warning(f"simple-gpt: 搜索判断失败 - {exc}")
            return {"is_needed": False, "query": ""}

    async def _search_answer(self, question: str) -> str:
        """使用 doubao-seed 模型执行搜索"""
        logger.info(f"simple-gpt: 正在执行网络搜索，问题：{question}")
        try:
            async with AsyncOpenAI(
                api_key=self.search_api_key,
                base_url=self.search_base_url,
            ) as client:
                response = await client.responses.create(
                    model=self.search_model,
                    instructions=SEARCH_SYSTEM_PROMPT,
                    input=question,
                    tools=[{"type": "web_search"}],
                )
                return response.output_text

        except Exception as exc:
            logger.warning(f"simple-gpt: 网络搜索执行失败 - {exc}")
            return ""


register_simple_gpt_plugin(WebSearchPlugin())
