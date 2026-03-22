from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Dict, List

from nonebot.log import logger

from ...plugin_config_inject import register_plugin_config_field
from ...plugin_system import (
    LLMRequestPayload,
    LLMResponsePayload,
    SimpleGPTPlugin,
    register_simple_gpt_plugin,
)
from .extractor import MemoryExtractor
from .profile_store import ProfileStore
from .semantic_store import SemanticStore

# ---------- 注册配置字段 ----------

register_plugin_config_field(
    "simple_gpt_memory_enabled",
    bool,
    default=False,
    description="是否启用长期记忆功能",
)
register_plugin_config_field(
    "simple_gpt_memory_db_path",
    str,
    default="data/simple_gpt_memory",
    description="记忆数据存储目录（SQLite + LanceDB）",
)
register_plugin_config_field(
    "simple_gpt_memory_embedding_api_key",
    str,
    default="",
    description="Embedding API Key（留空则使用主 API Key）",
)
register_plugin_config_field(
    "simple_gpt_memory_embedding_api_base",
    str,
    default="",
    description="Embedding API Base URL（留空则使用主 API Base URL）",
)
register_plugin_config_field(
    "simple_gpt_memory_embedding_model",
    str,
    default="text-embedding-3-small",
    description="Embedding 模型名称",
)
register_plugin_config_field(
    "simple_gpt_memory_embedding_dimensions",
    int,
    default=512,
    description="Embedding 向量维度",
    ge=64,
    le=4096,
)
register_plugin_config_field(
    "simple_gpt_memory_extract_api_key",
    str,
    default="",
    description="记忆提取 LLM API Key（留空则使用主 API Key）",
)
register_plugin_config_field(
    "simple_gpt_memory_extract_api_base",
    str,
    default="",
    description="记忆提取 LLM API Base URL（留空则使用主 API Base URL）",
)
register_plugin_config_field(
    "simple_gpt_memory_extract_model",
    str,
    default="gemini-3.1-flash-lite-preview",
    description="记忆提取使用的模型",
)
register_plugin_config_field(
    "simple_gpt_memory_top_k",
    int,
    default=5,
    description="每次注入的语义记忆数量",
    ge=1,
    le=20,
)
register_plugin_config_field(
    "simple_gpt_memory_scope",
    str,
    default="group",
    description="记忆作用范围：group（按群隔离）或 global（全局共享）",
)

if TYPE_CHECKING:
    from ... import Config


def _get_plugin_config() -> "Config":
    from ... import plugin_config

    return plugin_config


class MemoryPlugin(SimpleGPTPlugin):
    """分层长期记忆插件：SQLite 用户档案 + LanceDB 语义记忆。"""

    priority = 250

    def __init__(self) -> None:
        self._config_loaded = False
        self._enabled = False
        self._scope = "group"
        self._top_k = 5
        self._profile_store: ProfileStore | None = None
        self._semantic_store: SemanticStore | None = None
        self._extractor: MemoryExtractor | None = None

    def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return

        config = _get_plugin_config()
        self._enabled = config.simple_gpt_memory_enabled
        self._scope = config.simple_gpt_memory_scope
        self._top_k = config.simple_gpt_memory_top_k

        if not self._enabled:
            self._config_loaded = True
            logger.info("simple-gpt: memory 插件未启用")
            return

        db_path = config.simple_gpt_memory_db_path
        embedding_dim = config.simple_gpt_memory_embedding_dimensions

        # 初始化存储
        self._profile_store = ProfileStore(db_path)
        self._semantic_store = SemanticStore(db_path, embedding_dim)

        # API 配置回退逻辑
        embedding_api_key = (
            config.simple_gpt_memory_embedding_api_key
            or config.simple_gpt_api_key
        )
        embedding_api_base = (
            config.simple_gpt_memory_embedding_api_base
            or config.simple_gpt_api_base
        )
        extract_api_key = (
            config.simple_gpt_memory_extract_api_key
            or config.simple_gpt_api_key
        )
        extract_api_base = (
            config.simple_gpt_memory_extract_api_base
            or config.simple_gpt_api_base
        )

        self._extractor = MemoryExtractor(
            extract_api_key=extract_api_key,
            extract_api_base=extract_api_base,
            extract_model=config.simple_gpt_memory_extract_model,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            embedding_model=config.simple_gpt_memory_embedding_model,
            embedding_dimensions=embedding_dim,
        )

        self._config_loaded = True
        logger.info(
            f"simple-gpt: memory 插件已加载 "
            f"(scope={self._scope}, top_k={self._top_k}, "
            f"embedding_model={config.simple_gpt_memory_embedding_model}, "
            f"extract_model={config.simple_gpt_memory_extract_model})"
        )

    def _resolve_session_id(self, payload_session_id: str) -> str:
        """根据 scope 配置决定实际使用的 session_id。"""
        if self._scope == "global":
            return "global"
        return payload_session_id

    async def before_llm_request(
        self, payload: LLMRequestPayload
    ) -> LLMRequestPayload:
        """检索相关记忆并注入 prompt。"""
        self._ensure_config_loaded()
        if not self._enabled:
            return payload

        session_id = self._resolve_session_id(
            payload.extra.get("session_id", "")
        )
        if not session_id:
            return payload

        assert self._profile_store is not None
        assert self._semantic_store is not None
        assert self._extractor is not None

        injection_parts: List[str] = []

        # --- 1. Profile 层：查询用户档案（global + group 合并） ---
        try:
            profiles = await self._profile_store.get_merged_profiles(session_id)
            if profiles:
                profile_lines = self._format_profiles(profiles)
                injection_parts.append("## 用户档案")
                injection_parts.extend(profile_lines)
        except Exception as exc:
            logger.warning(f"simple-gpt: 读取用户档案失败: {exc}")

        # --- 2. Semantic 层：向量搜索相关记忆 ---
        try:
            query_text = payload.latest_message
            query_vector = await self._extractor.generate_embedding(query_text)
            if query_vector:
                memories = await self._semantic_store.search(
                    query_vector, session_id, self._top_k
                )
                if memories:
                    injection_parts.append("\n## 相关记忆")
                    for mem in memories:
                        injection_parts.append(f"- {mem['content']}")
        except Exception as exc:
            logger.warning(f"simple-gpt: 语义记忆检索失败: {exc}")

        # --- 3. 注入到 prompt ---
        if injection_parts:
            memory_block = "[长期记忆]\n" + "\n".join(injection_parts) + "\n[记忆结束]\n\n"
            payload.prompt = memory_block + payload.prompt
            payload.extra["injected_memories"] = True
            logger.info(
                f"simple-gpt: 已注入记忆到 prompt "
                f"(profiles={bool(profiles)}, semantic_count={len(memories) if 'memories' in dir() else 0})"
            )

        return payload

    async def after_llm_response(
        self, payload: LLMResponsePayload
    ) -> LLMResponsePayload:
        """将记忆提取/存储作为后台任务，不阻塞回复发送。"""
        self._ensure_config_loaded()
        if not self._enabled:
            return payload

        session_id = self._resolve_session_id(
            payload.request.extra.get("session_id", "")
        )
        if not session_id:
            return payload

        # 启动后台任务，立即返回 payload 以便尽快发送回复
        asyncio.create_task(
            self._extract_and_store(payload, session_id)
        )

        return payload

    async def _extract_and_store(
        self, payload: LLMResponsePayload, session_id: str
    ) -> None:
        """后台执行：从对话中提取并存储新记忆。"""
        assert self._profile_store is not None
        assert self._semantic_store is not None
        assert self._extractor is not None

        try:
            # 构建 display_name → QQ user_id 映射
            name_to_uid = self._build_name_mapping(payload.request)

            # 获取已有语义记忆（用于去重提示）
            existing_contents: List[str] = []
            try:
                query_vector = await self._extractor.generate_embedding(
                    payload.request.latest_message
                )
                if query_vector:
                    existing = await self._semantic_store.search(
                        query_vector, session_id, 10
                    )
                    existing_contents = [m["content"] for m in existing]
            except Exception:
                pass

            # 调用 LLM 提取
            extraction = await self._extractor.extract(
                history=payload.request.history,
                current_msg=payload.request.latest_message,
                bot_reply=payload.content,
                sender=payload.request.sender,
                existing_memories=existing_contents or None,
            )

            # 存储 profiles（按 scope 分层，display_name 映射到 QQ user_id）
            profiles = extraction.get("profiles", [])
            stored_profiles = 0
            for profile in profiles:
                display_name = profile["user_id"]
                real_uid = name_to_uid.get(display_name, display_name)
                scope = profile.get("scope", "group")
                await self._profile_store.upsert(
                    user_id=real_uid,
                    session_id=session_id,
                    key=profile["key"],
                    value=profile["value"],
                    scope=scope,
                )
                stored_profiles += 1
            if stored_profiles:
                logger.info(
                    f"simple-gpt: 已存储 {stored_profiles} 条用户档案"
                )

            # 存储 semantic memories
            memories = extraction.get("memories", [])
            stored_count = 0
            for mem in memories:
                vector = await self._extractor.generate_embedding(mem["content"])
                if vector:
                    await self._semantic_store.add(
                        {
                            "session_id": session_id,
                            "content": mem["content"],
                            "speaker": mem.get("speaker", ""),
                            "category": mem.get("category", "fact"),
                            "importance": mem.get("importance", 0.5),
                        },
                        vector,
                    )
                    stored_count += 1
            if stored_count:
                logger.info(
                    f"simple-gpt: 已存储 {stored_count} 条语义记忆"
                )

        except Exception as exc:
            logger.exception(f"simple-gpt: 记忆提取/存储失败 - {exc}")

    @staticmethod
    def _build_name_mapping(request: LLMRequestPayload) -> Dict[str, str]:
        """从 payload 中构建 display_name → QQ user_id 映射。"""
        mapping: Dict[str, str] = {}
        # 当前发言者
        sender_uid = request.extra.get("sender_user_id", "")
        if sender_uid and request.sender:
            mapping[request.sender] = sender_uid
        # 历史发言者
        for entry in request.history:
            if entry.user_id and entry.speaker:
                mapping[entry.speaker] = entry.user_id
        return mapping

    @staticmethod
    def _format_profiles(
        profiles: Dict[str, Dict[str, str]],
    ) -> List[str]:
        """将用户档案格式化为可读文本。"""
        lines: List[str] = []
        for user_id, kvs in profiles.items():
            parts: List[str] = []
            nickname = kvs.pop("nickname", None)
            display = f"{user_id}（昵称: {nickname}）" if nickname else user_id
            for key, value in kvs.items():
                parts.append(f"{key}: {value}")
            if parts:
                lines.append(f"- {display}: {', '.join(parts)}")
            elif nickname:
                lines.append(f"- {display}")
        return lines


register_simple_gpt_plugin(MemoryPlugin())
