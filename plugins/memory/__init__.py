from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Dict, List

from nonebot import get_driver, on_command
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message
from nonebot.log import logger
from nonebot.matcher import Matcher
from nonebot.params import CommandArg

from ...plugin_config_inject import register_plugin_config_field
from ...plugin_system import (
    LLMRequestPayload,
    LLMResponsePayload,
    SimpleGPTPlugin,
    register_simple_gpt_plugin,
)
from .admin import register_admin_api
from .debug_store import MemoryDebugStore
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
register_plugin_config_field(
    "simple_gpt_memory_admin_token",
    str,
    default="",
    description="memory 管理 HTTP API 的 Bearer Token",
)
register_plugin_config_field(
    "simple_gpt_memory_debug_max_rows",
    int,
    default=1000,
    description="调试 SQLite 最多保留的对话记录数，0 表示不清理",
    ge=0,
    le=100000,
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
        self._admin_token = ""
        self._scope = "group"
        self._top_k = 5
        self._debug_max_rows = 1000
        self._profile_store: ProfileStore | None = None
        self._semantic_store: SemanticStore | None = None
        self._debug_store: MemoryDebugStore | None = None
        self._extractor: MemoryExtractor | None = None
        self._admin_routes_registered = False

    def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return

        config = _get_plugin_config()
        self._enabled = config.simple_gpt_memory_enabled
        self._admin_token = config.simple_gpt_memory_admin_token.strip()
        self._scope = config.simple_gpt_memory_scope
        self._top_k = config.simple_gpt_memory_top_k
        self._debug_max_rows = config.simple_gpt_memory_debug_max_rows

        if not self._enabled:
            self._config_loaded = True
            logger.info("simple-gpt: memory 插件未启用")
            return

        db_path = config.simple_gpt_memory_db_path
        embedding_dim = config.simple_gpt_memory_embedding_dimensions

        # 初始化存储
        self._profile_store = ProfileStore(db_path)
        self._semantic_store = SemanticStore(db_path, embedding_dim)
        self._debug_store = MemoryDebugStore(db_path, max_rows=self._debug_max_rows)

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
        if self._admin_token:
            try:
                self._admin_routes_registered = register_admin_api(self)
            except Exception as exc:
                self._admin_routes_registered = False
                logger.warning(f"simple-gpt: memory admin API 注册失败: {exc}")
        logger.info(
            f"simple-gpt: memory 插件已加载 "
            f"(scope={self._scope}, top_k={self._top_k}, "
            f"embedding_model={config.simple_gpt_memory_embedding_model}, "
            f"extract_model={config.simple_gpt_memory_extract_model}, "
            f"admin_enabled={bool(self._admin_token)}, debug_enabled=True)"
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
        payload.extra.setdefault("memory_conversation_id", str(uuid.uuid4()))

        assert self._profile_store is not None
        assert self._semantic_store is not None
        assert self._extractor is not None

        injection_parts: List[str] = []
        profiles: Dict[str, Dict[str, str]] = {}
        user_memories: List[dict] = []
        group_memories: List[dict] = []

        # 收集当前对话参与者的 user_id（当前发言者 + 历史记录中出现的人）
        relevant_uids: set = set()
        sender_uid = payload.extra.get("sender_user_id", "")
        if sender_uid:
            relevant_uids.add(sender_uid)
        for entry in payload.history:
            if entry.user_id:
                relevant_uids.add(entry.user_id)

        # --- 1. Profile 层：查询并过滤为当前参与者 ---
        try:
            all_profiles = await self._profile_store.get_merged_profiles(session_id)
            # 只注入本次对话参与者的档案
            profiles = (
                {uid: kvs for uid, kvs in all_profiles.items() if uid in relevant_uids}
                if relevant_uids
                else all_profiles
            )
            if profiles:
                profile_lines = self._format_profiles(profiles)
                injection_parts.append("## 用户档案")
                injection_parts.extend(profile_lines)
                logger.debug(
                    f"simple-gpt: [memory] 注入档案 - 参与者 uid={relevant_uids}, "
                    f"过滤后 {len(profiles)}/{len(all_profiles)} 人\n"
                    + "\n".join(f"  {uid}: {kvs}" for uid, kvs in profiles.items())
                )
        except Exception as exc:
            logger.warning(f"simple-gpt: 读取用户档案失败: {exc}")

        # --- 2. Semantic 层：分两路搜索（用户记忆 + 群体记忆） ---
        try:
            query_text = payload.latest_message
            query_vector = await self._extractor.generate_embedding(query_text)
            if query_vector:
                half_k = max(1, self._top_k // 2)
                rest_k = self._top_k - half_k

                # 2a. 当前发言者的个人记忆
                if sender_uid:
                    user_memories = await self._semantic_store.search(
                        query_vector, session_id, half_k,
                        related_user_id=sender_uid,
                    )
                # 2b. 群体记忆（related_user_id 为空）
                group_memories = await self._semantic_store.search(
                    query_vector, session_id, rest_k,
                    related_user_id="",
                )

                if user_memories:
                    injection_parts.append("\n## 关于当前用户的记忆")
                    for mem in user_memories:
                        injection_parts.append(f"- {mem['content']}")
                if group_memories:
                    injection_parts.append("\n## 群体记忆")
                    for mem in group_memories:
                        injection_parts.append(f"- {mem['content']}")

                all_memories = user_memories + group_memories
                if all_memories:
                    logger.debug(
                        f"simple-gpt: [memory] 注入语义记忆 "
                        f"(用户 {len(user_memories)} 条, 群体 {len(group_memories)} 条):\n"
                        + "\n".join(
                            f"  - [uid={m.get('related_user_id', '')}] {m['content']}"
                            for m in all_memories
                        )
                    )
        except Exception as exc:
            logger.warning(f"simple-gpt: 语义记忆检索失败: {exc}")

        # --- 3. 注入到 prompt ---
        if injection_parts:
            memory_block = "[长期记忆]\n" + "\n".join(injection_parts) + "\n[记忆结束]\n\n"
            payload.prompt = memory_block + payload.prompt
            payload.extra["injected_memories"] = True
            logger.info(
                f"simple-gpt: [memory] 已注入记忆 "
                f"(档案 {len(profiles)} 人, "
                f"用户记忆 {len(user_memories)} 条, "
                f"群体记忆 {len(group_memories)} 条)"
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
        conversation_id = payload.request.extra.get(
            "memory_conversation_id", str(uuid.uuid4())
        )
        extracted_profiles: List[Dict] = []
        extracted_memories: List[Dict] = []
        stored_profile_rows: List[Dict] = []
        stored_memories: List[Dict] = []
        error_text = ""

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
            extracted_profiles = list(extraction.get("profiles", []))
            extracted_memories = list(extraction.get("memories", []))

            # 存储 profiles（按 scope 分层，display_name 映射到 QQ user_id）
            profiles = extracted_profiles
            stored_profile_count = 0
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
                stored_profile_count += 1
                stored_profile_rows.append(
                    {
                        "user_id": real_uid,
                        "session_id": session_id,
                        "key": profile["key"],
                        "value": profile["value"],
                        "scope": scope,
                    }
                )
                logger.debug(
                    f"simple-gpt: [memory] 存储档案 "
                    f"uid={real_uid}({display_name}) scope={scope} "
                    f"{profile['key']}={profile['value']!r}"
                )
            if stored_profile_count:
                logger.info(
                    f"simple-gpt: [memory] 已存储 {stored_profile_count} 条用户档案"
                )

            # 存储 semantic memories（语义去重：相似度 >= 0.85 则跳过）
            DEDUP_THRESHOLD = 0.85
            memories = extracted_memories
            stored_count = 0
            skipped_count = 0
            for mem in memories:
                vector = await self._extractor.generate_embedding(mem["content"])
                if not vector:
                    continue
                # 映射 related_user display_name → QQ user_id
                related_user_display = mem.get("related_user", "")
                related_user_id = (
                    name_to_uid.get(related_user_display, "")
                    if related_user_display
                    else ""
                )
                # 检查是否已有高相似度的记忆
                existing = await self._semantic_store.search(
                    vector, session_id, top_k=1
                )
                if existing and existing[0].get("similarity", 0) >= DEDUP_THRESHOLD:
                    skipped_count += 1
                    logger.debug(
                        f"simple-gpt: [memory] 跳过重复记忆 "
                        f"(similarity={existing[0]['similarity']:.3f}): "
                        f"{mem['content']!r} ≈ {existing[0]['content']!r}"
                    )
                    continue
                await self._semantic_store.add(
                    {
                        "session_id": session_id,
                        "content": mem["content"],
                        "speaker": mem.get("speaker", ""),
                        "category": mem.get("category", "fact"),
                        "importance": mem.get("importance", 0.5),
                        "related_user_id": related_user_id,
                    },
                    vector,
                )
                stored_count += 1
                stored_memories.append(
                    {
                        "session_id": session_id,
                        "content": mem["content"],
                        "speaker": mem.get("speaker", ""),
                        "category": mem.get("category", "fact"),
                        "importance": mem.get("importance", 0.5),
                        "related_user_display": related_user_display,
                        "related_user_id": related_user_id,
                    }
                )
                logger.debug(
                    f"simple-gpt: [memory] 存储语义记忆 "
                    f"category={mem.get('category')} importance={mem.get('importance')} "
                    f"related_user={related_user_display!r}→{related_user_id!r} "
                    f"speaker={mem.get('speaker')!r}: {mem['content']!r}"
                )
            if stored_count or skipped_count:
                logger.info(
                    f"simple-gpt: [memory] 语义记忆: "
                    f"存储 {stored_count} 条, 去重跳过 {skipped_count} 条"
                )

        except Exception as exc:
            error_text = str(exc)
            logger.exception(f"simple-gpt: 记忆提取/存储失败 - {exc}")
        finally:
            if self._debug_store is not None:
                try:
                    await self._debug_store.record_dialogue(
                        conversation_id=conversation_id,
                        session_id=session_id,
                        sender=payload.request.sender,
                        sender_user_id=payload.request.extra.get(
                            "sender_user_id", ""
                        ),
                        latest_message=payload.request.latest_message,
                        final_prompt=payload.request.prompt,
                        reply_content=payload.content,
                        injected_memories=bool(
                            payload.request.extra.get("injected_memories", False)
                        ),
                        extracted_profiles=extracted_profiles,
                        extracted_memories=extracted_memories,
                        stored_profiles=stored_profile_rows,
                        stored_memories=stored_memories,
                        error_text=error_text,
                    )
                except Exception as exc:
                    logger.warning(f"simple-gpt: 写入 memory debug 失败: {exc}")

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
            nickname = kvs.get("nickname")  # 不用 pop，避免 mutate 原始 dict
            display = f"{user_id}（昵称: {nickname}）" if nickname else user_id
            parts = [f"{k}: {v}" for k, v in kvs.items() if k != "nickname"]
            if parts:
                lines.append(f"- {display}: {', '.join(parts)}")
            elif nickname:
                lines.append(f"- {display}")
        return lines


_plugin_instance = MemoryPlugin()
register_simple_gpt_plugin(_plugin_instance)
_driver = get_driver()


@_driver.on_startup
async def _startup_memory_admin() -> None:
    _plugin_instance._ensure_config_loaded()


@_driver.on_shutdown
async def _close_memory_debug_store() -> None:
    if _plugin_instance._profile_store is not None:
        _plugin_instance._profile_store.close()
    if _plugin_instance._debug_store is not None:
        _plugin_instance._debug_store.close()


# ---------- 命令公共工具 ----------

def _check_permission(event: GroupMessageEvent, bot: Bot) -> tuple[bool, bool]:
    """返回 (is_admin, is_superuser)。is_admin 包含 owner/admin。"""
    role = event.sender.role
    is_admin = role in ("owner", "admin")
    is_superuser = str(event.user_id) in bot.config.superusers
    return is_admin, is_superuser


def _resolve_target_group(
    arg_text: str, current_group_id: int, is_superuser: bool
) -> tuple[int | None, str | None]:
    """解析目标群号。返回 (group_id, error_msg)。"""
    if not arg_text:
        return current_group_id, None
    if not arg_text.isdigit():
        return None, "群号格式错误，请输入纯数字。"
    target = int(arg_text)
    if target != current_group_id and not is_superuser:
        return None, "只有超级用户才能操作其他群的记忆。"
    return target, None


def _get_stores():
    plugin = _plugin_instance
    plugin._ensure_config_loaded()
    return plugin._enabled, plugin._profile_store, plugin._semantic_store


# ---------- 清除记忆命令 ----------

_clear_memory_cmd = on_command(
    "清除记忆",
    aliases={"clear_memory", "清除群记忆"},
    priority=1,
    block=True,
)


@_clear_memory_cmd.handle()
async def _handle_clear_memory(
    matcher: Matcher,
    bot: Bot,
    event: GroupMessageEvent,
    args: Message = CommandArg(),
) -> None:
    is_admin, is_superuser = _check_permission(event, bot)
    if not is_admin and not is_superuser:
        await matcher.finish("只有群管理员或超级用户才能清除记忆。")

    enabled, profile_store, semantic_store = _get_stores()
    if not enabled:
        await matcher.finish("记忆插件未启用。")
    assert profile_store is not None and semantic_store is not None

    arg_text = args.extract_plain_text().strip()
    target_group_id, err = _resolve_target_group(
        arg_text, event.group_id, is_superuser
    )
    if err:
        await matcher.finish(err)

    session_id = f"group_{target_group_id}"
    profile_count = await profile_store.delete_session(session_id)
    memory_count = await semantic_store.delete_session(session_id)

    logger.info(
        f"simple-gpt: [memory] 用户 {event.user_id} 清除了群 {target_group_id} 的记忆 "
        f"(档案 {profile_count} 条, 语义记忆 {memory_count} 条)"
    )
    suffix = f"（群 {target_group_id}）" if target_group_id != event.group_id else "本群"
    await matcher.finish(
        f"已清除{suffix}所有记忆：\n"
        f"- 群内用户档案：{profile_count} 条\n"
        f"- 语义记忆：{memory_count} 条\n"
        f"（跨群通用档案如生日、职业等不受影响）"
    )


# ---------- 查看记忆命令 ----------

_view_memory_cmd = on_command(
    "查看记忆",
    aliases={"view_memory", "查看群记忆"},
    priority=1,
    block=True,
)


@_view_memory_cmd.handle()
async def _handle_view_memory(
    matcher: Matcher,
    bot: Bot,
    event: GroupMessageEvent,
    args: Message = CommandArg(),
) -> None:
    is_admin, is_superuser = _check_permission(event, bot)
    if not is_admin and not is_superuser:
        await matcher.finish("只有群管理员或超级用户才能查看记忆。")

    enabled, profile_store, semantic_store = _get_stores()
    if not enabled:
        await matcher.finish("记忆插件未启用。")
    assert profile_store is not None and semantic_store is not None

    arg_text = args.extract_plain_text().strip()
    target_group_id, err = _resolve_target_group(
        arg_text, event.group_id, is_superuser
    )
    if err:
        await matcher.finish(err)

    session_id = f"group_{target_group_id}"
    suffix = f"群 {target_group_id}" if target_group_id != event.group_id else "本群"

    profiles = await profile_store.get_merged_profiles(session_id)
    memories = await semantic_store.get_all_session(session_id)

    # 构建私聊消息
    lines: List[str] = [f"=== {suffix} 的记忆 ===\n"]

    lines.append(f"【用户档案】共 {sum(len(v) for v in profiles.values())} 条")
    if profiles:
        for uid, kvs in profiles.items():
            nickname = kvs.get("nickname", "")
            display = f"{uid}（{nickname}）" if nickname else uid
            attrs = ", ".join(f"{k}={v}" for k, v in kvs.items() if k != "nickname")
            lines.append(f"  {display}: {attrs}" if attrs else f"  {display}")
    else:
        lines.append("  （暂无）")

    lines.append(f"\n【语义记忆】共 {len(memories)} 条（按时间倒序）")
    if memories:
        for i, mem in enumerate(memories, 1):
            ts = mem["created_at"][:10]
            ruid = mem.get("related_user_id", "")
            scope_tag = f"用户:{ruid}" if ruid else "群体"
            lines.append(
                f"  {i}. [{mem['category']}][{scope_tag}] {mem['content']}"
                f"\n     来源: {mem['speaker'] or '未知'}  时间: {ts}"
                f"  重要度: {mem['importance']:.1f}"
            )
    else:
        lines.append("  （暂无）")

    full_msg = "\n".join(lines)

    # 分段发送（避免私聊消息过长）
    MAX_LEN = 1000
    chunks = [full_msg[i:i + MAX_LEN] for i in range(0, len(full_msg), MAX_LEN)]
    try:
        for chunk in chunks:
            await bot.send_private_msg(user_id=event.user_id, message=chunk)
    except Exception as exc:
        logger.warning(f"simple-gpt: [memory] 私聊发送失败: {exc}")
