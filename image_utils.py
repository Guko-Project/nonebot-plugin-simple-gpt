from __future__ import annotations

import asyncio
import base64
import imghdr
import mimetypes
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from nonebot.adapters.onebot.v11 import Message, MessageSegment
from nonebot.log import logger

IMAGE_FETCH_TIMEOUT = 15.0
MAX_IMAGE_BYTES = 5 * 1024 * 1024


async def extract_image_data_urls(message: Message) -> List[str]:
    """Convert image segments in a message into data URLs."""

    data_urls: List[str] = []
    for segment in message:
        if segment.type != "image":
            continue
        data_url = await _segment_to_data_url(segment)
        if data_url:
            data_urls.append(data_url)
    return data_urls


async def _segment_to_data_url(segment: MessageSegment) -> Optional[str]:
    data = segment.data or {}
    url = data.get("url")
    file_value = data.get("file")
    path_value = data.get("path")

    content: Optional[bytes] = None
    content_type: Optional[str] = None
    filename: Optional[str] = None

    if file_value:
        filename = file_value
        if file_value.startswith("base64://"):
            content = _decode_inline_base64(file_value)
        elif _looks_like_url(file_value) and not url:
            url = file_value
        elif _is_local_path(file_value):
            content = await _load_local_file(file_value)

    if content is None and path_value:
        filename = path_value
        content = await _load_local_file(path_value)

    if content is None and url:
        content, content_type = await _download_image(url)

    if content is None:
        logger.debug(
            "simple-gpt: 无法解析图片片段，data=%s",
            data,
        )
        return None

    data_url = _build_data_url(content, content_type=content_type, filename=filename)
    return data_url


def _decode_inline_base64(value: str) -> Optional[bytes]:
    payload = value.replace("base64://", "", 1)
    try:
        return base64.b64decode(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("simple-gpt: 无法解码 base64 图片：%s", exc)
        return None


def _looks_like_url(text: str) -> bool:
    return text.startswith("http://") or text.startswith("https://")


def _is_local_path(text: str) -> bool:
    parsed = urlparse(text)
    if parsed.scheme == "file":
        return True
    path = Path(text)
    return path.exists()


async def _load_local_file(path_str: str) -> Optional[bytes]:
    path = _resolve_path(path_str)
    if path is None:
        return None
    try:
        data = await asyncio.to_thread(path.read_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.warning("simple-gpt: 读取本地图片失败：%s", exc)
        return None
    if len(data) > MAX_IMAGE_BYTES:
        logger.warning("simple-gpt: 本地图片过大（%s bytes），已忽略", len(data))
        return None
    return data


def _resolve_path(path_str: str) -> Optional[Path]:
    parsed = urlparse(path_str)
    if parsed.scheme == "file":
        return Path(parsed.path)
    path = Path(path_str)
    if path.exists():
        return path
    return None


async def _download_image(url: str) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        async with httpx.AsyncClient(timeout=IMAGE_FETCH_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("simple-gpt: 下载图片失败 %s：%s", url, exc)
        return None, None

    data = response.content
    if len(data) > MAX_IMAGE_BYTES:
        logger.warning("simple-gpt: 图片过大（%s bytes），已忽略", len(data))
        return None, None

    content_type = response.headers.get("content-type")
    return data, content_type


def _build_data_url(
    content: bytes, *, content_type: Optional[str], filename: Optional[str]
) -> str:
    mime = _resolve_mime(content, content_type=content_type, filename=filename)
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _resolve_mime(
    content: bytes, *, content_type: Optional[str], filename: Optional[str]
) -> str:
    if content_type:
        return content_type.split(";")[0]
    if filename:
        guessed = mimetypes.guess_type(filename)[0]
        if guessed:
            return guessed
    detected = imghdr.what(None, h=content)
    if detected:
        return f"image/{detected}"
    return "image/png"
