from __future__ import annotations

import asyncio
import base64
import imghdr
import io
import mimetypes
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from nonebot.adapters.onebot.v11 import Message, MessageSegment
from nonebot.log import logger

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("simple-gpt: 未安装 Pillow，图片压缩功能将不可用")

IMAGE_FETCH_TIMEOUT = 15.0
MAX_IMAGE_BYTES = 5 * 1024 * 1024


async def extract_image_data_urls(message: Message) -> List[str]:
    """Convert image segments in a message into data URLs."""
    from nonebot import get_plugin_config
    from .config import Config

    plugin_config = get_plugin_config(Config)

    data_urls: List[str] = []
    for segment in message:
        if segment.type != "image":
            continue
        data_url = await _segment_to_data_url(segment, plugin_config)
        if data_url:
            data_urls.append(data_url)
    return data_urls


async def _segment_to_data_url(segment: MessageSegment, plugin_config) -> Optional[str]:
    data = segment.data or {}
    url = data.get("url")
    file_value = data.get("file")
    path_value = data.get("path")
    content: Optional[bytes] = None
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
        content, _ = await _download_image(url)

    if content is None:
        logger.debug(
            f"simple-gpt: 无法解析图片片段，data={data}"
        )
        return None

    # 压缩图片
    if plugin_config.simple_gpt_image_compression_enabled:
        compressed_content = await compress_image(
            content,
            max_size=plugin_config.simple_gpt_image_max_size,
            quality=plugin_config.simple_gpt_image_quality,
            max_bytes=plugin_config.simple_gpt_image_max_bytes_after_compression,
        )
        if compressed_content is None:
            logger.warning("simple-gpt: 图片压缩失败，忽略该图片")
            return None
        content = compressed_content

    content_type = _resolve_mime(content, content_type=None, filename=None)
    data_url = _build_data_url(content, content_type=content_type, filename=filename)
    return data_url


def _decode_inline_base64(value: str) -> Optional[bytes]:
    payload = value.replace("base64://", "", 1)
    try:
        return base64.b64decode(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"simple-gpt: 无法解码 base64 图片：{exc}")
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
        logger.warning(f"simple-gpt: 读取本地图片失败：{exc}")
        return None
    if len(data) > MAX_IMAGE_BYTES:
        logger.warning(f"simple-gpt: 本地图片过大（{len(data)} bytes），已忽略")
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
        logger.warning(f"simple-gpt: 下载图片失败 {url}：{exc}")
        return None, None

    data = response.content
    if len(data) > MAX_IMAGE_BYTES:
        logger.warning(f"simple-gpt: 图片过大（{len(data)} bytes），已忽略")
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


async def compress_image(
    image_bytes: bytes,
    *,
    max_size: int = 1024,
    quality: int = 85,
    max_bytes: int = 512 * 1024,
) -> Optional[bytes]:
    """异步压缩图片。

    Args:
        image_bytes: 原始图片字节数据
        max_size: 压缩后最大边长（像素）
        quality: JPEG 压缩质量（1-100）
        max_bytes: 压缩后最大字节数

    Returns:
        压缩后的图片字节数据，如果压缩失败则返回 None
    """
    if not PIL_AVAILABLE:
        logger.warning("simple-gpt: Pillow 未安装，跳过图片压缩")
        return image_bytes if len(image_bytes) <= max_bytes else None

    try:
        return await asyncio.to_thread(
            _compress_image_sync,
            image_bytes,
            max_size=max_size,
            quality=quality,
            max_bytes=max_bytes,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"simple-gpt: 图片压缩失败：{exc}")
        return None


def _compress_image_sync(
    image_bytes: bytes,
    *,
    max_size: int,
    quality: int,
    max_bytes: int,
) -> Optional[bytes]:
    """同步压缩图片（在线程中执行）。"""
    try:
        # 打开图片
        image = Image.open(io.BytesIO(image_bytes))

        # 转换为 RGB 模式（JPEG 不支持透明度）
        if image.mode in ("RGBA", "LA", "P"):
            # 创建白色背景
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            background.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # 调整尺寸
        original_width, original_height = image.size
        if original_width > max_size or original_height > max_size:
            if original_width > original_height:
                new_width = max_size
                new_height = int(original_height * max_size / original_width)
            else:
                new_height = max_size
                new_width = int(original_width * max_size / original_height)

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(
                f"simple-gpt: 图片尺寸调整：{original_width}x{original_height} -> {new_width}x{new_height}"
            )

        # 压缩为 JPEG
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        compressed_bytes = output.getvalue()

        # 检查压缩后大小
        if len(compressed_bytes) > max_bytes:
            logger.warning(
                f"simple-gpt: 压缩后图片仍然过大（{len(compressed_bytes)} bytes），已忽略"
            )
            return None

        logger.debug(
            f"simple-gpt: 图片压缩成功：{len(image_bytes)} bytes -> {len(compressed_bytes)} bytes "
            f"({len(compressed_bytes) * 100 / len(image_bytes):.1f}%)"
        )

        return compressed_bytes

    except Exception as exc:  # noqa: BLE001
        logger.warning(f"simple-gpt: 图片压缩失败：{exc}")
        return None

