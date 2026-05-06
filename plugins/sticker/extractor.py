from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

from nonebot.log import logger
from openai import AsyncOpenAI

from ...models import HistoryEntry

EMOTION_TAG_ENUM = [
    "开心", "兴奋", "得意", "满足", "放松", "感动", "害羞", "喜欢", "期待", "安心",
    "平静", "疑惑", "好奇", "无语", "尴尬", "无奈", "震惊", "惊喜", "慌张", "紧张",
    "委屈", "难过", "失望", "崩溃", "生气", "不爽", "嘲讽", "嫉妒", "害怕", "警惕",
    "困倦", "疲惫", "敷衍", "冷漠", "吃瓜",
]

INTENT_TAG_ENUM = [
    "附和", "赞同", "反对", "安慰", "鼓励", "道歉", "求饶", "嘲讽", "阴阳", "吐槽",
    "拒绝", "催促", "提醒", "警告", "庆祝", "祝贺", "欢迎", "告别", "围观", "插话",
    "接梗", "结束话题", "转移话题", "卖萌", "装傻", "求助", "炫耀", "自嘲", "认错", "质问",
]

SCENE_TAG_ENUM = [
    "日常闲聊", "群聊插话", "回复别人", "被夸时", "被骂时", "被催时", "说错话时", "冷场时",
    "成功时", "失败时", "请求帮助时", "看戏时", "吵架时", "道歉时", "庆祝时", "深夜聊天",
    "工作学习", "游戏对局", "二次元话题", "发疯整活",
]

TAG_ENUMS = {
    "emotion_tags": EMOTION_TAG_ENUM,
    "intent_tags": INTENT_TAG_ENUM,
    "scene_tags": SCENE_TAG_ENUM,
}

_ENUM_BLOCK = "\n".join(
    [
        "可用标签枚举如下，只能从中选择，不允许发明新标签：",
        f"- emotion_tags: {', '.join(EMOTION_TAG_ENUM)}",
        f"- intent_tags: {', '.join(INTENT_TAG_ENUM)}",
        f"- scene_tags: {', '.join(SCENE_TAG_ENUM)}",
    ]
)

STICKER_ANALYZE_PROMPT = f"""你是一个中文聊天表情包识别助手。请分析给定表情包图片，输出 JSON：
{{
  "description": "一句话描述这张图表达的情绪和内容",
  "emotion_tags": ["开心"],
  "intent_tags": ["附和"],
  "scene_tags": ["日常闲聊"],
  "ocr_text": "图中文字，没有则为空字符串",
  "usage_notes": "适合在什么语境发送",
  "aliases": ["高兴", "乐"]
}}

{_ENUM_BLOCK}

要求：
1. 只返回 JSON，不要 markdown。
2. emotion_tags 和 intent_tags 各选 1-3 个，scene_tags 选 0-2 个。
3. 输出的标签必须严格来自上面的枚举；如果没有合适标签，对应字段返回空数组。
4. aliases 可以填写枚举外的近义表达，用于检索辅助，但标签本身必须使用枚举值。
5. description 要准确描述图片主体、表情语气和显著梗点。
6. 如果图片带有清晰文字，请尽量提取到 ocr_text；usage_notes 重点描述适合发送的语境。"""

STICKER_DECIDE_PROMPT = f"""你是一个群聊表情包选择助手。给定最近对话和助手最终回复，请判断是否适合发送一张表情包，并输出 JSON：
{{
  "should_send": true,
  "query_text": "适合检索表情包的简短查询语",
  "emotion_tags": ["开心"],
  "intent_tags": ["附和"],
  "scene_tags": ["群聊插话"],
  "negative_tags": ["严肃"]
}}

{_ENUM_BLOCK}

要求：
1. 只返回 JSON，不要 markdown。
2. 如果不适合发表情包，should_send=false，其他字段尽量留空。
3. emotion_tags 和 intent_tags 各选 0-3 个，scene_tags 选 0-2 个。
4. 输出标签必须严格来自上面的枚举；negative_tags 可以留空或填简短排斥语义。"""


def _parse_json_response(text: str) -> Dict[str, Any]:
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    raw = match.group(1).strip() if match else text.strip()
    logger.debug(
        "simple-gpt: sticker LLM JSON 解析 "
        f"(raw_len={len(raw)}, fenced={match is not None}, preview={raw[:200]!r})"
    )
    return json.loads(raw)


class StickerExtractor:
    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        embedding_api_key: str,
        embedding_api_base: str,
        embedding_model: str,
        embedding_dimensions: int,
        timeout: float,
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base
        self._model = model
        self._embedding_api_key = embedding_api_key
        self._embedding_api_base = embedding_api_base
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._timeout = timeout

    async def analyze_sticker(self, image_data_url: str) -> Dict[str, Any]:
        logger.debug(
            "simple-gpt: sticker 识别请求开始 "
            f"(model={self._model}, base_url={self._api_base}, "
            f"image_data_url_len={len(image_data_url)})"
        )
        try:
            async with AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._api_base,
                timeout=self._timeout,
            ) as client:
                completion = await client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": STICKER_ANALYZE_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "请识别这张表情包。"},
                                {"type": "image_url", "image_url": {"url": image_data_url}},
                            ],
                        },
                    ],
                )
            content = completion.choices[0].message.content or ""
            logger.debug(
                "simple-gpt: sticker 识别响应收到 "
                f"(content_len={len(content)}, preview={content[:200]!r})"
            )
            result = _parse_json_response(content)
        except Exception as exc:
            logger.warning(f"simple-gpt: sticker 识别失败: {exc}")
            return {}

        cleaned = {
            "description": str(result.get("description", "")).strip(),
            "emotion_tags": _clean_enum_list(result.get("emotion_tags", []), "emotion_tags"),
            "intent_tags": _clean_enum_list(result.get("intent_tags", []), "intent_tags"),
            "scene_tags": _clean_enum_list(result.get("scene_tags", []), "scene_tags"),
            "ocr_text": str(result.get("ocr_text", "")).strip(),
            "usage_notes": str(result.get("usage_notes", "")).strip(),
            "aliases": _clean_text_list(result.get("aliases", [])),
        }
        logger.debug(f"simple-gpt: sticker 识别清洗完成 {cleaned}")
        return cleaned

    async def decide_sticker(
        self,
        *,
        history: Sequence[HistoryEntry],
        latest_message: str,
        bot_reply: str,
    ) -> Dict[str, Any]:
        history_lines: List[str] = []
        for entry in history[-8:]:
            role = "助手" if entry.is_bot else entry.speaker
            history_lines.append(f"{role}: {entry.content}")
        history_lines.append(f"用户最新消息: {latest_message}")
        history_lines.append(f"助手最终回复: {bot_reply}")
        prompt = "\n".join(history_lines)
        logger.debug(
            "simple-gpt: sticker 发送决策请求开始 "
            f"(model={self._model}, base_url={self._api_base}, "
            f"history_entries={len(history)}, prompt_len={len(prompt)}, "
            f"latest_len={len(latest_message)}, reply_len={len(bot_reply)})"
        )

        try:
            async with AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._api_base,
                timeout=self._timeout,
            ) as client:
                completion = await client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": STICKER_DECIDE_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
            content = completion.choices[0].message.content or ""
            logger.debug(
                "simple-gpt: sticker 发送决策响应收到 "
                f"(content_len={len(content)}, preview={content[:200]!r})"
            )
            result = _parse_json_response(content)
        except Exception as exc:
            logger.warning(f"simple-gpt: sticker 决策失败: {exc}")
            return {
                "should_send": False,
                "query_text": "",
                "emotion_tags": [],
                "intent_tags": [],
                "scene_tags": [],
                "negative_tags": [],
            }

        cleaned = {
            "should_send": bool(result.get("should_send", False)),
            "query_text": str(result.get("query_text", "")).strip(),
            "emotion_tags": _clean_enum_list(result.get("emotion_tags", []), "emotion_tags"),
            "intent_tags": _clean_enum_list(result.get("intent_tags", []), "intent_tags"),
            "scene_tags": _clean_enum_list(result.get("scene_tags", []), "scene_tags"),
            "negative_tags": _clean_text_list(result.get("negative_tags", [])),
        }
        logger.debug(f"simple-gpt: sticker 发送决策清洗完成 {cleaned}")
        return cleaned

    async def generate_embedding(self, text: str) -> List[float]:
        if not text.strip():
            logger.debug("simple-gpt: sticker embedding 跳过：输入为空")
            return []

        logger.debug(
            "simple-gpt: sticker embedding 请求开始 "
            f"(model={self._embedding_model}, base_url={self._embedding_api_base}, "
            f"dimensions={self._embedding_dimensions}, text_len={len(text)}, "
            f"preview={text[:120]!r})"
        )
        try:
            async with AsyncOpenAI(
                api_key=self._embedding_api_key,
                base_url=self._embedding_api_base,
                timeout=self._timeout,
            ) as client:
                response = await client.embeddings.create(
                    model=self._embedding_model,
                    input=text,
                    dimensions=self._embedding_dimensions,
                )
                embedding = response.data[0].embedding
                logger.debug(
                    "simple-gpt: sticker embedding 响应完成 "
                    f"(dim={len(embedding)})"
                )
                return embedding
        except Exception as exc:
            logger.warning(f"simple-gpt: sticker embedding 生成失败: {exc}")
            return []


def _clean_text_list(value: Any) -> List[str]:
    if isinstance(value, str):
        values = re.split(r"[,，/\s]+", value)
    elif isinstance(value, list):
        values = [str(item) for item in value]
    else:
        values = []

    result: List[str] = []
    for item in values:
        stripped = item.strip()
        if stripped and stripped not in result:
            result.append(stripped)
    return result[:8]


def _clean_enum_list(value: Any, field_name: str) -> List[str]:
    allowed = set(TAG_ENUMS[field_name])
    result: List[str] = []
    dropped: List[str] = []
    for item in _clean_text_list(value):
        if item in allowed and item not in result:
            result.append(item)
        elif item not in allowed:
            dropped.append(item)
    if dropped:
        logger.debug(
            "simple-gpt: sticker 标签清洗丢弃非枚举值 "
            f"(field={field_name}, dropped={dropped})"
        )
    return result
