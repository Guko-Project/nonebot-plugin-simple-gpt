from __future__ import annotations

import asyncio
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence

from nonebot import get_driver, get_plugin_config, on_message
from nonebot.adapters import Bot
from nonebot.adapters.onebot.v11 import (
    GroupMessageEvent,
    MessageEvent,
    MessageSegment,
)
from nonebot.log import logger
from nonebot.matcher import Matcher
from nonebot.plugin import PluginMetadata

from .chat import close_chat_client, generate_chat_reply
from .image_utils import extract_image_data_urls
from .models import HistoryEntry
from .plugin_system import (
    LLMRequestPayload,
    LLMResponsePayload,
    emit_after_llm_response,
    emit_before_llm_request,
)

# 先导入插件（让插件注册配置字段）
from . import plugins as _simple_gpt_plugins  # noqa: F401
# 导入配置注入系统
from .plugin_config_inject import inject_plugin_fields_to_config
# 导入配置类
from .config import Config


# 注入插件配置字段
inject_plugin_fields_to_config(Config)

plugin_config = get_plugin_config(Config)


__plugin_meta__ = PluginMetadata(
    name="simple-gpt",
    description="基于大模型的群聊对话插件，支持@触发与随机回复。",
    usage="在群里@鸽子姬即可触发回复，或根据设定的概率自动回复。",
    config=Config,
)


class HistoryManager:
    def __init__(self, limit: int):
        self._limit = limit
        self._store: Dict[str, Deque[HistoryEntry]] = {}

    def snapshot(self, session_id: str) -> List[HistoryEntry]:
        history = self._store.get(session_id)
        if not history:
            return []
        return list(history)

    def append(self, session_id: str, entry: HistoryEntry) -> None:
        history = self._store.get(session_id)
        if history is None:
            history = deque(maxlen=self._limit)
            self._store[session_id] = history
        history.append(entry)


history_manager = HistoryManager(limit=plugin_config.simple_gpt_history_limit)
driver = get_driver()


@driver.on_shutdown
async def _close_client() -> None:
    await close_chat_client()


def generate_prompt(
    *,
    history: List[HistoryEntry],
    sender: str,
    latest_message: str,
    latest_images: Optional[Sequence[str]] = None,
) -> str:
    if history:
        history_lines = "\n".join(_format_history_entry(entry) for entry in history)
    else:
        history_lines = "（暂无聊天记录）"
    latest_section = latest_message
    if latest_images:
        latest_section = _append_image_hint(latest_section, len(latest_images))
    prompt = plugin_config.simple_gpt_prompt_template.format(
        history=history_lines, sender=sender, latest_message=latest_section
    )
    return prompt


def _format_history_entry(entry: HistoryEntry) -> str:
    content = entry.content
    if entry.images:
        content = _append_image_hint(content, len(entry.images))
    return f"{entry.speaker}：{content}"


def _append_image_hint(content: str, count: int) -> str:
    return f"{content}\n（附带 {count} 张图片）"


def should_reply(event: MessageEvent) -> bool:
    if event.is_tome():
        return True
    if plugin_config.simple_gpt_reply_probability <= 0:
        return False
    return random.random() < plugin_config.simple_gpt_reply_probability


def _is_group_allowed_for_proactive(group_id: int) -> bool:
    whitelist = plugin_config.simple_gpt_proactive_group_whitelist
    if not whitelist:
        return False
    return group_id in whitelist


message_matcher = on_message(priority=5, block=False)

IGNORED_PREFIXES = ("/", ".", "!")


@message_matcher.handle()
async def _(matcher: Matcher, bot: Bot, event: MessageEvent) -> None:
    if not isinstance(event, GroupMessageEvent):
        return
    user_id = event.get_user_id()
    if user_id == str(bot.self_id):
        return

    raw_text = event.get_plaintext().strip()
    if raw_text.startswith(IGNORED_PREFIXES):
        logger.info(f"simple-gpt: 忽略前缀消息：{raw_text}")
        return

    if not raw_text:
        plain_text = "（无文字内容）"
    else:
        plain_text = raw_text

    image_contexts = await extract_image_data_urls(event.message)

    session_id = f"group_{event.group_id}"
    display_name = event.sender.card or event.sender.nickname or f"用户{user_id}"

    history_before = history_manager.snapshot(session_id)
    is_tome_event = event.is_tome()
    reply_needed = should_reply(event)
    if (
        reply_needed
        and not is_tome_event
        and not _is_group_allowed_for_proactive(event.group_id)
    ):
        logger.debug(
            f"simple-gpt: 群 {event.group_id} 不在主动发言白名单，跳过主动回复"
        )
        reply_needed = False

    if reply_needed and not is_tome_event and not plugin_config.simple_gpt_api_key:
        reply_needed = False
    reply_text: Optional[str] = None

    if reply_needed:
        prompt = generate_prompt(
            history=history_before,
            sender=display_name,
            latest_message=plain_text,
            latest_images=image_contexts,
        )
        history_images: List[str] = [
            image for entry in history_before for image in entry.images
        ]
        combined_images = [*history_images, *image_contexts]
        llm_request = LLMRequestPayload(
            prompt=prompt,
            history=history_before,
            sender=display_name,
            latest_message=plain_text,
            images=combined_images,
        )
        llm_request = await emit_before_llm_request(llm_request)
        # 被 @ 时提高重试次数，主动发言保持默认重试次数
        max_retries = 5 if is_tome_event else 3
        generated = await generate_chat_reply(
            prompt=llm_request.prompt,
            api_key=plugin_config.simple_gpt_api_key,
            base_url=plugin_config.simple_gpt_api_base,
            model=plugin_config.simple_gpt_model,
            temperature=plugin_config.simple_gpt_temperature,
            max_tokens=plugin_config.simple_gpt_max_tokens,
            timeout=plugin_config.simple_gpt_timeout,
            images=llm_request.images,
            debug=plugin_config.simple_gpt_prompt_debug,
            max_retries=max_retries,
        )
        # 主动发言时，如果服务器错误则不回复
        if not generated and not is_tome_event:
            logger.error(
                "simple-gpt: 主动发言时，服务器返回错误，忽略该次主动发言"
            )
            reply_needed = False
        else:
            reply_text = generated or plugin_config.simple_gpt_failure_reply
            response_payload = LLMResponsePayload(
                content=reply_text,
                request=llm_request,
            )
            response_payload = await emit_after_llm_response(response_payload)
            reply_text = response_payload.content
            lines = [line.strip() for line in reply_text.split("///") if line.strip()]
            for idx, line in enumerate(lines):
                await asyncio.sleep(random.uniform(1.0, 3.0))
                if idx == 0:
                    message = MessageSegment.reply(event.message_id) + line
                else:
                    message = line
                await matcher.send(message)

    history_manager.append(
        session_id,
        HistoryEntry(
            speaker=display_name,
            content=plain_text,
            is_bot=False,
            images=image_contexts,
        ),
    )

    if reply_needed and reply_text:
        history_manager.append(
            session_id,
            HistoryEntry(speaker="鸽子姬", content=reply_text, is_bot=True, images=[]),
        )
