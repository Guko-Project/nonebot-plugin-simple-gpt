from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import random
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
import os

import httpx
from nonebot import get_driver, on_command
from nonebot.adapters.onebot.v11 import (
    ActionFailed,
    Bot,
    GroupMessageEvent,
    Message,
    MessageEvent,
    MessageSegment as OBMessageSegment,
    MessageSegment,
)
from nonebot.log import logger
from nonebot.matcher import Matcher
from nonebot.params import CommandArg

from ...image_utils import detect_image_mime
from ...plugin_config_inject import register_plugin_config_field
from ...plugin_system import (
    LLMResponsePayload,
    SimpleGPTPlugin,
    register_simple_gpt_plugin,
)
from .extractor import (
    EMOTION_TAG_ENUM,
    INTENT_TAG_ENUM,
    SCENE_TAG_ENUM,
    StickerExtractor,
)
from .store import StickerStore
from .vector_store import StickerVectorStore

try:
    from PIL import Image
except ImportError:  # pragma: no cover - pillow is already a runtime dep
    Image = None

if TYPE_CHECKING:
    from ... import Config

register_plugin_config_field(
    "simple_gpt_sticker_db_path",
    str,
    default="data/simple_gpt_sticker",
    description="表情包数据存储目录（SQLite + LanceDB + 图片文件）",
)
register_plugin_config_field(
    "simple_gpt_sticker_extract_api_key",
    str,
    default="",
    description="表情包识别 LLM API Key（留空则回退到 memory，再回退到主 API Key）",
)
register_plugin_config_field(
    "simple_gpt_sticker_extract_api_base",
    str,
    default="https://api.openai.com/v1",
    description="表情包识别 LLM API Base URL（留空则回退到 memory，再回退到主 API Base URL）",
)
register_plugin_config_field(
    "simple_gpt_sticker_extract_model",
    str,
    default="gemini-3.1-flash-lite-preview",
    description="表情包识别模型（留空则回退到 memory，再回退到主模型）",
)
register_plugin_config_field(
    "simple_gpt_sticker_embedding_api_key",
    str,
    default="",
    description="表情包 Embedding API Key（留空则回退到 memory，再回退到主 API Key）",
)
register_plugin_config_field(
    "simple_gpt_sticker_embedding_api_base",
    str,
    default="https://api.openai.com/v1",
    description="表情包 Embedding API Base URL（留空则回退到 memory，再回退到主 API Base URL）",
)
register_plugin_config_field(
    "simple_gpt_sticker_embedding_model",
    str,
    default="text-embedding-3-small",
    description="表情包 Embedding 模型（留空则回退到 memory 默认值）",
)
register_plugin_config_field(
    "simple_gpt_sticker_embedding_dimensions",
    int,
    default=512,
    description="表情包 Embedding 维度（<=0 时回退到 memory 默认值）",
    ge=0,
    le=4096,
)

IMAGE_DIR = "images"
SAVE_COMMAND = "记忆表情"
SAVE_COMMAND_ALIASES = {"memo_sticker"}
SEND_PROBABILITY = 0.4
SEMANTIC_TOP_K = 8
TAG_TOP_K = 8
MAX_CANDIDATES = 10
SEND_THRESHOLD = 0.62
COOLDOWN_SECONDS = 90
RECENT_STICKER_WINDOW = 5
IMAGE_FETCH_TIMEOUT = 30.0
MAX_IMAGE_BYTES = 5 * 1024 * 1024

_TAG_SYNONYMS = {
    "开心": ["高兴", "愉快", "快乐", "乐", "喜滋滋"],
    "无语": ["社死", "不知道说啥", "沉默", "汗颜"],
    "无奈": ["没办法", "算了", "认了"],
    "嘲讽": ["挖苦", "讽刺"],
    "震惊": ["惊讶", "吓到", "卧槽", "惊了"],
    "附和": ["同意", "认同", "点头"],
    "安慰": ["抱抱", "心疼", "安抚"],
    "催促": ["快点", "赶紧", "催一下"],
    "庆祝": ["恭喜", "撒花", "祝贺"],
    "吃瓜": ["围观", "看戏", "看热闹"],
    "敷衍": ["哦", "嗯嗯", "行吧", "随便"],
    "阴阳": ["阴阳怪气"],
    "吐槽": ["开喷", "碎碎念"],
    "尴尬": ["好尬", "脚趾抠地"],
    "委屈": ["想哭", "小委屈"],
    "群聊插话": ["插楼", "冒泡"],
    "回复别人": ["接话", "回一句"],
    "看戏时": ["吃瓜时"],
}

_SYNONYM_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in _TAG_SYNONYMS.items()
    for alias in aliases
}

_ALLOWED_CANONICAL_TAGS = (
    set(EMOTION_TAG_ENUM)
    | set(INTENT_TAG_ENUM)
    | set(SCENE_TAG_ENUM)
    | set(_TAG_SYNONYMS.keys())
)


def _get_plugin_config() -> "Config":
    from ... import plugin_config

    return plugin_config


def _resolve_sticker_setting(
    *,
    env_name: str,
    sticker_value: Any,
    sticker_default: Any,
    memory_value: Any,
    main_value: Any,
) -> Any:
    """Prefer explicitly configured sticker values; otherwise fall back to memory/main."""

    if os.getenv(env_name) not in (None, ""):
        return sticker_value
    if sticker_value != sticker_default and sticker_value not in ("", 0, None):
        return sticker_value
    if memory_value not in ("", 0, None):
        return memory_value
    return main_value


class StickerPlugin(SimpleGPTPlugin):
    priority = 120

    def __init__(self) -> None:
        self._initialized = False
        self._store: StickerStore | None = None
        self._vector_store: StickerVectorStore | None = None
        self._extractor: StickerExtractor | None = None
        self._recent_sent: Dict[str, Deque[Tuple[str, float]]] = defaultdict(
            lambda: deque(maxlen=RECENT_STICKER_WINDOW)
        )

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        config = _get_plugin_config()
        db_path = config.simple_gpt_sticker_db_path
        image_dir = os.path.join(db_path, IMAGE_DIR)
        os.makedirs(image_dir, exist_ok=True)

        extract_api_key = _resolve_sticker_setting(
            env_name="SIMPLE_GPT_STICKER_EXTRACT_API_KEY",
            sticker_value=config.simple_gpt_sticker_extract_api_key,
            sticker_default="",
            memory_value=config.simple_gpt_memory_extract_api_key,
            main_value=config.simple_gpt_api_key,
        )
        extract_api_base = _resolve_sticker_setting(
            env_name="SIMPLE_GPT_STICKER_EXTRACT_API_BASE",
            sticker_value=config.simple_gpt_sticker_extract_api_base,
            sticker_default="https://api.openai.com/v1",
            memory_value=config.simple_gpt_memory_extract_api_base,
            main_value=config.simple_gpt_api_base,
        )
        extract_model = _resolve_sticker_setting(
            env_name="SIMPLE_GPT_STICKER_EXTRACT_MODEL",
            sticker_value=config.simple_gpt_sticker_extract_model,
            sticker_default="gemini-3.1-flash-lite-preview",
            memory_value=config.simple_gpt_memory_extract_model,
            main_value=config.simple_gpt_model,
        )
        embedding_api_key = _resolve_sticker_setting(
            env_name="SIMPLE_GPT_STICKER_EMBEDDING_API_KEY",
            sticker_value=config.simple_gpt_sticker_embedding_api_key,
            sticker_default="",
            memory_value=config.simple_gpt_memory_embedding_api_key,
            main_value=config.simple_gpt_api_key,
        )
        embedding_api_base = _resolve_sticker_setting(
            env_name="SIMPLE_GPT_STICKER_EMBEDDING_API_BASE",
            sticker_value=config.simple_gpt_sticker_embedding_api_base,
            sticker_default="https://api.openai.com/v1",
            memory_value=config.simple_gpt_memory_embedding_api_base,
            main_value=config.simple_gpt_api_base,
        )
        embedding_model = _resolve_sticker_setting(
            env_name="SIMPLE_GPT_STICKER_EMBEDDING_MODEL",
            sticker_value=config.simple_gpt_sticker_embedding_model,
            sticker_default="text-embedding-3-small",
            memory_value=config.simple_gpt_memory_embedding_model,
            main_value="text-embedding-3-small",
        )
        embedding_dimensions = _resolve_sticker_setting(
            env_name="SIMPLE_GPT_STICKER_EMBEDDING_DIMENSIONS",
            sticker_value=config.simple_gpt_sticker_embedding_dimensions,
            sticker_default=512,
            memory_value=config.simple_gpt_memory_embedding_dimensions,
            main_value=512,
        )

        self._store = StickerStore(db_path)
        self._vector_store = StickerVectorStore(db_path, embedding_dimensions)
        self._extractor = StickerExtractor(
            api_key=extract_api_key,
            api_base=extract_api_base,
            model=extract_model,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            timeout=config.simple_gpt_timeout,
        )
        self._initialized = True
        logger.info(
            "simple-gpt: sticker 插件已加载 "
            f"(save_command={SAVE_COMMAND}, aliases={sorted(SAVE_COMMAND_ALIASES)}, "
            f"send_probability={SEND_PROBABILITY})"
        )

    async def after_llm_response(
        self, payload: LLMResponsePayload
    ) -> LLMResponsePayload:
        self._ensure_initialized()
        session_id = payload.request.extra.get("session_id", "")
        if not session_id or random.random() >= SEND_PROBABILITY:
            return payload

        assert self._store is not None
        assert self._vector_store is not None
        assert self._extractor is not None

        sticker_count = await self._store.count_enabled(session_id)
        if sticker_count <= 0:
            return payload

        decision = await self._extractor.decide_sticker(
            history=payload.request.history,
            latest_message=payload.request.latest_message,
            bot_reply=payload.content,
        )
        if not decision.get("should_send"):
            return payload

        candidate = await self._pick_best_sticker(session_id, decision)
        if candidate is None:
            return payload

        sticker_id = candidate["id"]
        now = asyncio.get_running_loop().time()
        self._recent_sent[session_id].append((sticker_id, now))
        payload.post_messages.append(
            MessageSegment.image(Path(candidate["file_path"]).resolve().as_uri())
        )
        logger.info(
            "simple-gpt: sticker 命中 "
            f"(session={session_id}, sticker_id={sticker_id}, score={candidate['final_score']:.3f})"
        )
        return payload

    async def _pick_best_sticker(
        self, session_id: str, decision: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        assert self._vector_store is not None
        assert self._extractor is not None

        query_text = str(decision.get("query_text", "")).strip()
        emotion_tags = _normalize_tags(decision.get("emotion_tags", []))
        intent_tags = _normalize_tags(decision.get("intent_tags", []))
        scene_tags = _normalize_tags(decision.get("scene_tags", []))
        negative_tags = set(_expand_tags(decision.get("negative_tags", [])))

        semantic_query = query_text or " ".join(emotion_tags + intent_tags + scene_tags)
        tag_query = " ".join(emotion_tags + intent_tags + scene_tags)
        semantic_vector = await self._extractor.generate_embedding(semantic_query)
        tag_vector = await self._extractor.generate_embedding(tag_query)

        semantic_rows: List[Dict[str, Any]] = []
        tag_rows: List[Dict[str, Any]] = []
        if semantic_vector:
            semantic_rows = await self._vector_store.search(
                session_id=session_id,
                query_vector=semantic_vector,
                vector_column="semantic_vector",
                top_k=SEMANTIC_TOP_K,
            )
        if tag_vector:
            tag_rows = await self._vector_store.search(
                session_id=session_id,
                query_vector=tag_vector,
                vector_column="tag_vector",
                top_k=TAG_TOP_K,
            )

        merged: Dict[str, Dict[str, Any]] = {}
        for row in semantic_rows:
            merged.setdefault(row["id"], row)["semantic_score"] = row["similarity"]
        for row in tag_rows:
            merged.setdefault(row["id"], row)["tag_vector_score"] = row["similarity"]

        expanded_tags = set(_expand_tags(emotion_tags + intent_tags + scene_tags))
        recent_sent_ids = self._recent_sent[session_id]
        now = asyncio.get_running_loop().time()
        recent_block = {
            sticker_id
            for sticker_id, sent_at in recent_sent_ids
            if now - sent_at < COOLDOWN_SECONDS
        }

        best_candidate: Optional[Dict[str, Any]] = None
        for row in list(merged.values())[:MAX_CANDIDATES]:
            if row["id"] in recent_block:
                continue

            sticker_tags = _normalize_tags(
                _split_csv(row.get("emotion_tags", ""))
                + _split_csv(row.get("intent_tags", ""))
                + _split_csv(row.get("scene_tags", ""))
                + _split_csv(row.get("aliases", ""))
            )
            expanded_sticker_tags = set(_expand_tags(sticker_tags))
            if negative_tags and negative_tags.intersection(expanded_sticker_tags):
                continue
            overlap = expanded_tags.intersection(expanded_sticker_tags)
            tag_match_score = (
                len(overlap) / max(len(expanded_tags), 1)
                if expanded_tags
                else 0.0
            )
            semantic_score = float(row.get("semantic_score", 0.0))
            tag_vector_score = float(row.get("tag_vector_score", 0.0))
            final_score = (
                semantic_score * 0.5
                + tag_vector_score * 0.3
                + tag_match_score * 0.2
            )
            row["final_score"] = final_score
            if final_score < SEND_THRESHOLD:
                continue
            if best_candidate is None or final_score > best_candidate["final_score"]:
                best_candidate = row

        return best_candidate

    def close(self) -> None:
        if self._store is not None:
            self._store.close()


_plugin_instance = StickerPlugin()
register_simple_gpt_plugin(_plugin_instance)

driver = get_driver()


@driver.on_shutdown
async def _close_sticker_plugin() -> None:
    _plugin_instance.close()


_save_sticker_matcher = on_command(
    SAVE_COMMAND,
    aliases=SAVE_COMMAND_ALIASES,
    priority=1,
    block=True,
)


@_save_sticker_matcher.handle()
async def _handle_save_sticker(
    matcher: Matcher,
    bot: Bot,
    event: GroupMessageEvent,
    args: Message = CommandArg(),
) -> None:
    _plugin_instance._ensure_initialized()
    assert _plugin_instance._store is not None
    assert _plugin_instance._vector_store is not None
    assert _plugin_instance._extractor is not None

    if not isinstance(event, GroupMessageEvent):
        await matcher.finish()

    session_id = f"group_{event.group_id}"
    reply_message = _extract_reply_message(event)
    if reply_message is None:
        await matcher.finish(
            f"请回复一张表情图后再发送“{SAVE_COMMAND}”"
            f"或“{next(iter(SAVE_COMMAND_ALIASES))}”。"
        )

    try:
        data_urls = await _extract_image_data_urls_from_message(
            reply_message,
            bot=bot,
            timeout=min(max(_get_plugin_config().simple_gpt_timeout, 15.0), 120.0),
        )
    except Exception as exc:
        logger.warning(f"simple-gpt: sticker 解析回复图片失败: {exc}")
        await matcher.finish("回复里的图片解析失败，稍后再试。")
    if not data_urls:
        await matcher.finish("被回复消息里没有可识别的图片。")

    image_data_url = data_urls[0]
    image_bytes, ext = _decode_data_url(image_data_url)
    if not image_bytes:
        await matcher.finish("图片解析失败，暂时无法保存。")

    sha256 = hashlib.sha256(image_bytes).hexdigest()
    phash = _compute_perceptual_hash(image_bytes)
    duplicate = await _plugin_instance._store.find_duplicate(session_id, sha256, phash)
    if duplicate:
        await matcher.finish("这张表情已经记住了。")

    sticker_meta = await _plugin_instance._extractor.analyze_sticker(image_data_url)
    if not sticker_meta.get("description"):
        await matcher.finish("表情识别失败，这次没有记住。")

    file_path = await _save_sticker_file(image_bytes, ext)
    emotion_tags = _normalize_tags(sticker_meta.get("emotion_tags", []))
    intent_tags = _normalize_tags(sticker_meta.get("intent_tags", []))
    scene_tags = _normalize_tags(sticker_meta.get("scene_tags", []))
    aliases = _normalize_tags(sticker_meta.get("aliases", []))
    description = str(sticker_meta.get("description", "")).strip()
    ocr_text = str(sticker_meta.get("ocr_text", "")).strip()
    usage_notes = str(sticker_meta.get("usage_notes", "")).strip()

    semantic_text = _build_semantic_text(
        description=description,
        emotion_tags=emotion_tags,
        intent_tags=intent_tags,
        scene_tags=scene_tags,
        ocr_text=ocr_text,
        usage_notes=usage_notes,
        aliases=aliases,
    )
    tag_text = _build_tag_text(
        emotion_tags=emotion_tags,
        intent_tags=intent_tags,
        scene_tags=scene_tags,
        usage_notes=usage_notes,
        aliases=aliases,
    )

    semantic_vector = await _plugin_instance._extractor.generate_embedding(semantic_text)
    tag_vector = await _plugin_instance._extractor.generate_embedding(tag_text)
    if not semantic_vector or not tag_vector:
        await matcher.finish("向量化失败，这次没有记住。")

    record = await _plugin_instance._store.add(
        session_id=session_id,
        file_path=file_path,
        sha256=sha256,
        phash=phash,
        description=description,
        emotion_tags=emotion_tags,
        intent_tags=intent_tags,
        scene_tags=scene_tags,
        ocr_text=ocr_text,
        usage_notes=usage_notes,
        aliases=aliases,
        created_by=str(event.user_id),
    )
    await _plugin_instance._vector_store.add(
        sticker_id=record["id"],
        session_id=session_id,
        description=description,
        emotion_tags=emotion_tags,
        intent_tags=intent_tags,
        scene_tags=scene_tags,
        usage_notes=usage_notes,
        aliases=aliases,
        file_path=file_path,
        semantic_vector=semantic_vector,
        tag_vector=tag_vector,
    )
    logger.info(
        "simple-gpt: 已保存表情 "
        f"(session={session_id}, sticker_id={record['id']}, description={description!r})"
    )
    await matcher.finish(
        "已记住这个表情。\n"
        f"描述：{description}\n"
        f"情绪：{'、'.join(emotion_tags) or '未识别'}\n"
        f"意图：{'、'.join(intent_tags) or '未识别'}\n"
        f"场景：{'、'.join(scene_tags) or '未识别'}"
    )


def _extract_reply_message(event: MessageEvent) -> Optional[Message]:
    raw_reply = getattr(event, "reply", None)
    if not raw_reply:
        return None
    raw_message = getattr(raw_reply, "message", None)
    if not raw_message:
        return None
    try:
        return Message(raw_message)
    except Exception:
        logger.debug("simple-gpt: sticker 无法解析回复消息为 Message 对象")
        return None


async def _extract_image_data_urls_from_message(
    message: Message, *, bot: Bot, timeout: float
) -> List[str]:
    data_urls: List[str] = []
    for segment in message:
        if segment.type != "image":
            continue
        data_url = await _segment_to_data_url_with_bot(segment, bot=bot, timeout=timeout)
        if data_url:
            data_urls.append(data_url)
    return data_urls


async def _segment_to_data_url_with_bot(
    segment: OBMessageSegment, *, bot: Bot, timeout: float
) -> Optional[str]:
    data = segment.data or {}
    url = data.get("url")
    file_value = data.get("file")
    path_value = data.get("path")
    content: Optional[bytes] = None
    filename: Optional[str] = None
    content_type: Optional[str] = None

    if isinstance(file_value, str):
        filename = file_value
        if file_value.startswith("base64://"):
            content = _decode_inline_base64(file_value)
        elif _looks_like_url(file_value) and not url:
            url = file_value
        elif _is_local_path(file_value):
            content = await _load_local_file(file_value)

    if content is None and isinstance(path_value, str):
        filename = path_value
        content = await _load_local_file(path_value)

    if content is None and not url:
        file_id = data.get("file_id") or file_value
        if isinstance(file_id, str):
            try:
                response = await bot.get_image(file=file_id)
            except ActionFailed as exc:
                logger.warning(f"simple-gpt: sticker 获取图片失败: {exc.info}")
            else:
                if isinstance(response, dict):
                    url = response.get("url") or url

    if content is None and url:
        content, content_type = await _download_image(url, timeout=timeout)

    if content is None:
        return None

    mime = _resolve_mime(content, content_type=content_type, filename=filename)
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _decode_inline_base64(value: str) -> Optional[bytes]:
    payload = value.replace("base64://", "", 1)
    try:
        return base64.b64decode(payload)
    except Exception as exc:
        logger.warning(f"simple-gpt: sticker 无法解码 base64 图片：{exc}")
        return None


def _looks_like_url(text: str) -> bool:
    return text.startswith("http://") or text.startswith("https://")


def _is_local_path(text: str) -> bool:
    parsed = urlparse(text)
    if parsed.scheme == "file":
        return True
    return Path(text).exists()


async def _load_local_file(path_str: str) -> Optional[bytes]:
    path = _resolve_path(path_str)
    if path is None:
        return None
    try:
        data = await asyncio.to_thread(path.read_bytes)
    except Exception as exc:
        logger.warning(f"simple-gpt: sticker 读取本地图片失败：{exc}")
        return None
    if len(data) > MAX_IMAGE_BYTES:
        logger.warning(f"simple-gpt: sticker 本地图片过大（{len(data)} bytes），已忽略")
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


async def _download_image(
    url: str, *, timeout: float
) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        actual_timeout = max(timeout, IMAGE_FETCH_TIMEOUT)
        async with httpx.AsyncClient(timeout=actual_timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as exc:
        logger.warning(f"simple-gpt: sticker 下载图片失败 {url}：{exc}")
        return None, None

    data = response.content
    if len(data) > MAX_IMAGE_BYTES:
        logger.warning(f"simple-gpt: sticker 图片过大（{len(data)} bytes），已忽略")
        return None, None

    return data, response.headers.get("content-type")


def _resolve_mime(
    content: bytes, *, content_type: Optional[str], filename: Optional[str]
) -> str:
    if content_type:
        return content_type.split(";")[0]
    if filename:
        import mimetypes

        guessed = mimetypes.guess_type(filename)[0]
        if guessed:
            return guessed
    detected = detect_image_mime(content)
    if detected:
        return detected
    return "image/png"


async def _save_sticker_file(image_bytes: bytes, ext: str) -> str:
    file_name = f"{uuid.uuid4().hex}.{ext}"
    file_path = Path(_get_plugin_config().simple_gpt_sticker_db_path) / IMAGE_DIR / file_name
    await asyncio.to_thread(file_path.write_bytes, image_bytes)
    return str(file_path)


def _decode_data_url(data_url: str) -> Tuple[bytes, str]:
    match = data_url.split(",", 1)
    if len(match) != 2:
        return b"", "png"
    header, payload = match
    try:
        image_bytes = base64.b64decode(payload)
    except Exception:
        return b"", "png"
    mime = header.replace("data:", "").replace(";base64", "")
    ext = mime.split("/")[-1] if "/" in mime else "png"
    if ext == "jpeg":
        ext = "jpg"
    return image_bytes, ext


def _compute_perceptual_hash(image_bytes: bytes) -> str:
    if Image is None:
        return hashlib.md5(image_bytes).hexdigest()

    with Image.open(io.BytesIO(image_bytes)) as image:
        gray = image.convert("L").resize((8, 8))
        pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= avg else "0" for pixel in pixels)
    return f"{int(bits, 2):016x}"


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_tags(tags: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for tag in tags:
        stripped = str(tag).strip()
        if not stripped:
            continue
        canonical = _SYNONYM_TO_CANONICAL.get(stripped, stripped)
        if canonical in _ALLOWED_CANONICAL_TAGS and canonical not in normalized:
            normalized.append(canonical)
    return normalized[:8]


def _expand_tags(tags: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for tag in _normalize_tags(tags):
        aliases = _TAG_SYNONYMS.get(tag, [tag])
        for alias in aliases:
            canonical = _SYNONYM_TO_CANONICAL.get(alias, alias)
            if alias not in expanded:
                expanded.append(alias)
            if canonical not in expanded:
                expanded.append(canonical)
    return expanded


def _build_semantic_text(
    *,
    description: str,
    emotion_tags: Sequence[str],
    intent_tags: Sequence[str],
    scene_tags: Sequence[str],
    ocr_text: str,
    usage_notes: str,
    aliases: Sequence[str],
) -> str:
    payload = {
        "description": description,
        "emotion_tags": list(emotion_tags),
        "intent_tags": list(intent_tags),
        "scene_tags": list(scene_tags),
        "ocr_text": ocr_text,
        "usage_notes": usage_notes,
        "aliases": list(aliases),
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_tag_text(
    *,
    emotion_tags: Sequence[str],
    intent_tags: Sequence[str],
    scene_tags: Sequence[str],
    usage_notes: str,
    aliases: Sequence[str],
) -> str:
    parts = [
        " ".join(emotion_tags),
        " ".join(intent_tags),
        " ".join(scene_tags),
        usage_notes,
        " ".join(aliases),
    ]
    return " ".join(part for part in parts if part).strip()


def get_admin_handles() -> Tuple[StickerStore | None, StickerVectorStore | None]:
    _plugin_instance._ensure_initialized()
    return _plugin_instance._store, _plugin_instance._vector_store
