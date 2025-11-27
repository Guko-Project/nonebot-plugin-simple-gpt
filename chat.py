from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nonebot.log import logger
from openai import AsyncOpenAI, OpenAIError

_client_lock = asyncio.Lock()
_request_lock = asyncio.Lock()
_client: Optional[AsyncOpenAI] = None
_client_config: Optional[Tuple[str, str, float]] = None


async def _get_client(
    *, api_key: str, base_url: str, timeout: float
) -> Optional[AsyncOpenAI]:
    """Get or create a shared AsyncOpenAI client instance."""

    global _client, _client_config

    if not api_key:
        return None

    config = (api_key, base_url, timeout)
    async with _client_lock:
        if _client is not None and _client_config == config:
            return _client

        if _client is not None:
            try:
                await _client.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"simple-gpt: 关闭旧的 OpenAI 客户端时出错：{exc}")

        _client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        _client_config = config
        return _client


def _extract_text_content(message: object) -> Optional[str]:
    """Extract plain text from a ChatCompletionMessage."""

    content = getattr(message, "content", None)

    if isinstance(content, str):
        stripped = content.strip()
        return stripped or None

    if isinstance(content, Sequence):
        parts: List[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text)
        if parts:
            combined = "".join(parts).strip()
            return combined or None

    return None


async def generate_chat_reply(
    *,
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    images: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Call OpenAI's chat completion API through the official SDK."""
    logger.info(f"simple-gpt: 生成聊天回复，prompt={prompt}")
    client = await _get_client(api_key=api_key, base_url=base_url, timeout=timeout)
    if client is None:
        logger.warning("simple-gpt: 未配置 API Key，跳过调用。")
        return None

    logger.info(
        f"simple-gpt: 准备调用 OpenAI，model={model}, base_url={base_url}, "
        f"temperature={temperature}, max_tokens={max_tokens}, images={len(images or [])}"
    )

    content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    if images:
        for data_url in images:
            if not data_url:
                continue
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )
    user_message_content: Any
    if len(content_parts) == 1:
        user_message_content = prompt
    else:
        user_message_content = content_parts

    print(user_message_content)
    try:
        async with _request_lock:
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_message_content}],
                temperature=temperature,
                # 暂时不添加最大 token 数量
                # max_tokens=max_tokens,
            )
    except OpenAIError as exc:
        logger.exception(f"simple-gpt: OpenAI 请求失败：{exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"simple-gpt: 调用 OpenAI SDK 时出错：{exc}")
        return None

    if not completion.choices:
        logger.warning(f"simple-gpt: OpenAI 响应未包含 choices：{completion}")
        return None

    message = completion.choices[0].message
    content = _extract_text_content(message)
    if not content:
        logger.warning(f"simple-gpt: OpenAI 响应未包含文本内容：{completion}")
        return None

    return content


async def close_chat_client() -> None:
    """Release the shared AsyncOpenAI client, if any."""

    global _client, _client_config

    async with _client_lock:
        if _client is not None:
            try:
                await _client.close()
            finally:
                _client = None
                _client_config = None
