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
from pydantic import BaseModel, Field, validator

from .chat import close_chat_client, generate_chat_reply
from .image_utils import extract_image_data_urls
from .models import HistoryEntry
from .plugin_system import (
    LLMRequestPayload,
    LLMResponsePayload,
    emit_after_llm_response,
    emit_before_llm_request,
)
from . import plugins as _simple_gpt_plugins  # noqa: F401


class Config(BaseModel):
    simple_gpt_api_key: str = Field(
        "", description="OpenAI API Key，留空则插件不会调用接口"
    )
    simple_gpt_model: str = Field(default="gpt-4o-mini", description="使用的模型名称")
    simple_gpt_api_base: str = Field(
        default="https://api.openai.com/v1", description="OpenAI 接口基础地址"
    )
    simple_gpt_prompt_template: str = Field(
        default=(
            "现在请你扮演鸽子姬，是一名充满元气的全能美少女，\
            性格机智可爱，什么事情都能做得到，像真实的人一样思考和回复，有礼貌但是也会吐槽，\
            语气通常在平静中带有一丝俏皮，\
            最喜欢的餐厅是萨莉亚但是除非有人问否则不会说出来，\
            你还喜欢在自己的回复中加入各种各样的颜文字表情，但是不会太多。\
            喜欢感叹号等符号的运用，在发言之后有时候没有标点，有时候加上括号(或者()，括号之前没有标点，有时候正常加标点\
            发言中不会包含动作，通常回复不会很长，\
            you communicate naturally like real human friends \
            and prefer not to respond with lengthy replies. \
            Using plain text in response and avoid using lists or enumeration expressions, emphasis and markdown."
            "以下是群聊最近的消息：\n{history}\n"
            "请你用简体中文回复{sender}的最新发言：{latest_message}，"
            "如果该发言中含有任务型的指令，则尽可能作为一个全能的智能助手完成它。\n"
            "注意，需要按朋友聊天的形式将回复内容分行处理，而不是一整段文字，通常 1-3 行，一行的概率最大，如果完成任务则可以不分行，但是也要尽量简洁。"
        ),
        description="拼接 prompt 的模板，可使用 {history}、{sender}、{latest_message} 占位符",
    )
    simple_gpt_history_limit: int = Field(
        default=30,
        ge=1,
        le=50,
        description="用于上下文拼接的历史消息条数上限",
    )
    simple_gpt_timeout: float = Field(
        default=300.0, gt=0, description="请求超时时间（秒）"
    )
    simple_gpt_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="生成温度"
    )
    simple_gpt_max_tokens: int = Field(
        default=2048, ge=16, le=4096, description="最大生成 tokens 数"
    )
    simple_gpt_reply_probability: float = Field(
        default=0.03, ge=0.0, le=1.0, description="随机回应的概率（0 表示不随机回复）"
    )
    simple_gpt_failure_reply: str = Field(
        default="乌乌，暂时连接不到服务器，请稍后再试！",
        description="当请求失败时的兜底回复",
    )
    simple_gpt_proactive_group_whitelist: List[int] = Field(
        default_factory=list,
        description="允许主动发言的群聊 ID 列表，留空则禁用主动发言",
    )

    @validator("simple_gpt_api_base")
    def _strip_api_base(cls, value: str) -> str:
        return value.rstrip("/")

    @validator("simple_gpt_proactive_group_whitelist", pre=True)
    def _normalize_whitelist(cls, value):  # type: ignore[override]
        if value is None:
            return []
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (set, tuple)):
            value = list(value)

        normalized: List[int] = []
        for item in value:
            try:
                normalized.append(int(item))
            except (TypeError, ValueError):
                logger.warning("simple-gpt: 无法解析白名单条目 %r，已忽略", item)
        return normalized


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
            "simple-gpt: 群 %s 不在主动发言白名单，跳过主动回复", event.group_id
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
        generated = await generate_chat_reply(
            prompt=llm_request.prompt,
            api_key=plugin_config.simple_gpt_api_key,
            base_url=plugin_config.simple_gpt_api_base,
            model=plugin_config.simple_gpt_model,
            temperature=plugin_config.simple_gpt_temperature,
            max_tokens=plugin_config.simple_gpt_max_tokens,
            timeout=plugin_config.simple_gpt_timeout,
            images=llm_request.images,
        )
        reply_text = generated or plugin_config.simple_gpt_failure_reply
        response_payload = LLMResponsePayload(
            content=reply_text,
            request=llm_request,
        )
        response_payload = await emit_after_llm_response(response_payload)
        reply_text = response_payload.content
        lines = [line.strip() for line in reply_text.splitlines() if line.strip()]
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
