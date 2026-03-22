from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

from nonebot.log import logger
from openai import AsyncOpenAI

from ...models import HistoryEntry


# ---------- Prompts ----------

EXTRACTION_SYSTEM_PROMPT = """你是一个记忆管理助手。分析以下对话，提取三类**互斥**的信息：

## 1. profiles（用户档案）—— 仅限稳定属性
长期不变的结构化属性。**如果信息是稳定的、可用 key-value 表示的，放这里，不要放 memories。**
每条包含:
- user_id: 用户的显示名（和对话中出现的一致）
- key: 属性名，如 "nickname"、"occupation"、"birthday"、"gender"、"preference_food"、"preference_game"、"skill" 等
- value: 对应的值
- scope: "global"（跨群通用）或 "group"（仅限当前群）

scope 判断：
- global: 生日、年龄、职业、性别、固定兴趣爱好、技能等不会因群而异的属性
- group: 昵称/外号、群内职务、群内特定关系等

## 2. memories（语义记忆）—— 仅限时效性/事件性信息
具体的事件、计划、临时状态等**不适合用 key-value 表示**的信息。每条包含:
- content: 简洁完整的描述（能脱离上下文独立理解）
- category: "fact"（临时事实）| "event"（事件）| "instruction"（指令/请求）
- importance: 0.0-1.0（重要程度）
- speaker: 相关人物的显示名（可为空）
- related_user: 如果这条记忆与某个特定用户相关，填该用户的显示名；如果是群体事件则留空 ""

**互斥规则（最重要）：**
- "小明喜欢吃辣" → profiles（稳定偏好，key=preference_food, value=辣）
- "小明下周要出差" → memories（临时事件，related_user="小明"）
- "群里计划周末打游戏" → memories（群体事件，related_user=""）
- "小明是程序员" → profiles（稳定属性，key=occupation）
- "小明最近在学 Rust" → memories（临时状态，related_user="小明"）
- 绝对不要把同一条信息同时放入 profiles 和 memories

其他规则：
1. 只提取具体、有用的信息，忽略闲聊和无意义内容
2. 每条记忆应简洁完整，能脱离上下文独立理解
3. 没有值得记忆的内容则对应字段返回空数组
4. 不要重复提取已经在"已有记忆"中存在的信息

请严格以 JSON 格式返回，不要包含其他内容：
{"profiles": [{"user_id": "...", "key": "...", "value": "...", "scope": "global|group"}], "memories": [{"content": "...", "category": "fact|event|instruction", "importance": 0.0, "speaker": "...", "related_user": "..."}]}"""


def _parse_json_response(text: str) -> Dict[str, Any]:
    """从 LLM 响应中提取 JSON，兼容 markdown code block 包裹。"""
    # 尝试提取 ```json ... ``` 块
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    raw = match.group(1).strip() if match else text.strip()
    return json.loads(raw)


# ---------- 主类 ----------


class MemoryExtractor:
    """使用 LLM 提取记忆，使用 Embedding API 生成向量。"""

    def __init__(
        self,
        *,
        extract_api_key: str,
        extract_api_base: str,
        extract_model: str,
        embedding_api_key: str,
        embedding_api_base: str,
        embedding_model: str,
        embedding_dimensions: int,
    ) -> None:
        self._extract_api_key = extract_api_key
        self._extract_api_base = extract_api_base
        self._extract_model = extract_model
        self._embedding_api_key = embedding_api_key
        self._embedding_api_base = embedding_api_base
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions

    async def extract(
        self,
        history: Sequence[HistoryEntry],
        current_msg: str,
        bot_reply: str,
        sender: str,
        existing_memories: List[str] | None = None,
    ) -> Dict[str, Any]:
        """调用 LLM 提取 profiles 和 memories。

        返回:
            {"profiles": [...], "memories": [...]}
        """
        # 构造对话上下文
        context_parts: List[str] = []
        for entry in history[-10:]:
            role = "助手" if entry.is_bot else entry.speaker
            context_parts.append(f"{role}: {entry.content}")
        context_parts.append(f"{sender}: {current_msg}")
        context_parts.append(f"助手: {bot_reply}")
        conversation = "\n".join(context_parts)

        user_prompt = f"对话内容：\n{conversation}"
        if existing_memories:
            user_prompt += "\n\n已有记忆（不要重复提取）：\n" + "\n".join(
                f"- {m}" for m in existing_memories
            )

        try:
            async with AsyncOpenAI(
                api_key=self._extract_api_key,
                base_url=self._extract_api_base,
            ) as client:
                completion = await client.chat.completions.create(
                    model=self._extract_model,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                content = completion.choices[0].message.content or ""
                result = _parse_json_response(content)

                profiles = result.get("profiles", [])
                memories = result.get("memories", [])

                # 基本校验
                valid_profiles = [
                    p for p in profiles
                    if isinstance(p, dict)
                    and all(k in p for k in ("user_id", "key", "value"))
                    and p.get("scope") in ("global", "group", None)
                ]
                # 默认 scope 为 group
                for p in valid_profiles:
                    if "scope" not in p:
                        p["scope"] = "group"
                valid_memories = [
                    m for m in memories
                    if isinstance(m, dict)
                    and "content" in m
                ]

                return {"profiles": valid_profiles, "memories": valid_memories}

        except Exception as exc:
            logger.warning(f"simple-gpt: 记忆提取失败 - {exc}")
            return {"profiles": [], "memories": []}

    async def generate_embedding(self, text: str) -> List[float]:
        """调用 Embedding API 生成向量。"""
        try:
            async with AsyncOpenAI(
                api_key=self._embedding_api_key,
                base_url=self._embedding_api_base,
            ) as client:
                response = await client.embeddings.create(
                    model=self._embedding_model,
                    input=text,
                    dimensions=self._embedding_dimensions,
                )
                return response.data[0].embedding
        except Exception as exc:
            logger.warning(f"simple-gpt: embedding 生成失败 - {exc}")
            return []
