from __future__ import annotations

import asyncio
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import httpx
from nonebot import get_driver, get_plugin_config, on_message
from nonebot.adapters import Bot
from nonebot.adapters.onebot.v11 import GroupMessageEvent, MessageEvent
from nonebot.log import logger
from nonebot.matcher import Matcher
from nonebot.plugin import PluginMetadata
from pydantic import BaseModel, Field, validator


class Config(BaseModel):
    simple_gpt_api_key: str = Field("", description="OpenAI API Key，留空则插件不会调用接口")
    simple_gpt_model: str = Field(default="gpt-4o-mini", description="使用的模型名称")
    simple_gpt_api_base: str = Field(
        default="https://api.openai.com/v1", description="OpenAI 接口基础地址"
    )
    simple_gpt_prompt_template: str = Field(
        default=(
            "你是一个友善的中文群聊助手，需要结合最近的聊天记录进行自然对话。"
            "以下是群聊最近的消息：\n{history}\n"
            "请你用简体中文回复{sender}的最新发言：{latest_message}"
        ),
        description="拼接 prompt 的模板，可使用 {history}、{sender}、{latest_message} 占位符",
    )
    simple_gpt_history_limit: int = Field(
        default=20,
        ge=1,
        le=50,
        description="用于上下文拼接的历史消息条数上限",
    )
    simple_gpt_timeout: float = Field(default=15.0, gt=0, description="请求超时时间（秒）")
    simple_gpt_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="生成温度"
    )
    simple_gpt_max_tokens: int = Field(
        default=512, ge=16, le=4096, description="最大生成 tokens 数"
    )
    simple_gpt_reply_probability: float = Field(
        default=0.0, ge=0.0, le=1.0, description="随机回应的概率（0 表示不随机回复）"
    )
    simple_gpt_failure_reply: str = Field(
        default="呜呜，暂时无法连接到大模型，请稍后再试呀。",
        description="当请求失败时的兜底回复",
    )

    @validator("simple_gpt_api_base")
    def _strip_api_base(cls, value: str) -> str:
        return value.rstrip("/")


plugin_config = get_plugin_config(Config)


__plugin_meta__ = PluginMetadata(
    name="simple-gpt",
    description="基于大模型的群聊对话插件，支持@触发与随机回复。",
    usage="在群里@机器人即可触发回复，或根据设定的概率自动回复。",
    config=Config,
)


@dataclass
class HistoryEntry:
    speaker: str
    content: str
    is_bot: bool = False


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


class OpenAIClient:
    def __init__(self, *, timeout: float) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        self._lock = asyncio.Lock()

    async def post_chat_completion(self, payload: dict, *, url: str, api_key: str) -> dict:
        headers = {"Authorization": f"Bearer {api_key}"}
        async with self._lock:
            response = await self._client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self._client.aclose()


openai_client = OpenAIClient(timeout=plugin_config.simple_gpt_timeout)


@driver.on_shutdown
async def _close_client() -> None:
    await openai_client.close()


def generate_prompt(
    *, history: List[HistoryEntry], sender: str, latest_message: str
) -> str:
    if history:
        history_lines = "\n".join(
            f"{entry.speaker}：{entry.content}" for entry in history
        )
    else:
        history_lines = "（暂无聊天记录）"
    prompt = plugin_config.simple_gpt_prompt_template.format(
        history=history_lines, sender=sender, latest_message=latest_message
    )
    return prompt


async def call_large_language_model(
    *, prompt: str, api_key: str
) -> Optional[str]:
    if not api_key:
        logger.warning("simple-gpt: 未配置 API Key，跳过调用。")
        return None
    url = f"{plugin_config.simple_gpt_api_base}/chat/completions"
    payload = {
        "model": plugin_config.simple_gpt_model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": plugin_config.simple_gpt_temperature,
        "max_tokens": plugin_config.simple_gpt_max_tokens,
    }
    try:
        data = await openai_client.post_chat_completion(
            payload, url=url, api_key=api_key
        )
    except httpx.HTTPStatusError as exc:
        if exc.response is not None:
            logger.exception(
                "simple-gpt: OpenAI 返回错误 %s，响应体：%s",
                exc.response.status_code,
                exc.response.text,
            )
        else:
            logger.exception("simple-gpt: OpenAI 请求失败：%s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception("simple-gpt: 无法访问 OpenAI 接口：%s", exc)
        return None

    choices = data.get("choices")
    if not choices:
        logger.warning("simple-gpt: OpenAI 响应未包含 choices 字段：%s", data)
        return None
    message = choices[0].get("message", {})
    content = message.get("content")
    if not content:
        logger.warning("simple-gpt: OpenAI 响应未包含文本内容：%s", data)
        return None
    return content.strip()


def should_reply(event: MessageEvent) -> bool:
    if event.is_tome():
        return True
    if plugin_config.simple_gpt_reply_probability <= 0:
        return False
    return random.random() < plugin_config.simple_gpt_reply_probability


message_matcher = on_message(priority=50, block=False)


@message_matcher.handle()
async def _(matcher: Matcher, bot: Bot, event: MessageEvent) -> None:
    if not isinstance(event, GroupMessageEvent):
        return
    user_id = event.get_user_id()
    if user_id == str(bot.self_id):
        return

    plain_text = event.get_plaintext().strip()
    if not plain_text:
        plain_text = "（无文字内容）"

    session_id = f"group_{event.group_id}"
    display_name = event.sender.card or event.sender.nickname or f"用户{user_id}"

    history_before = history_manager.snapshot(session_id)
    reply_needed = should_reply(event)
    if reply_needed and not event.is_tome() and not plugin_config.simple_gpt_api_key:
        reply_needed = False
    reply_text: Optional[str] = None

    if reply_needed:
        prompt = generate_prompt(
            history=history_before,
            sender=display_name,
            latest_message=plain_text,
        )
        generated = await call_large_language_model(
            prompt=prompt, api_key=plugin_config.simple_gpt_api_key
        )
        reply_text = generated or plugin_config.simple_gpt_failure_reply
        await matcher.send(reply_text)

    history_manager.append(
        session_id,
        HistoryEntry(speaker=display_name, content=plain_text, is_bot=False),
    )

    if reply_needed and reply_text:
        history_manager.append(
            session_id,
            HistoryEntry(speaker="机器人", content=reply_text, is_bot=True),
        )
