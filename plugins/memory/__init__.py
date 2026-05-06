from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from nonebot import get_driver
from nonebot.log import logger

from ...plugin_config_inject import register_plugin_config_field
from ...plugin_system import (
    LLMRequestPayload,
    LLMResponsePayload,
    SimpleGPTPlugin,
    register_simple_gpt_plugin,
)
from .bank import BankResolution, resolve_bank
from .hindsight_client import (
    aclose_client,
    aensure_bank_mission,
    arecall,
    aretain,
    get_client,
)
from .mission import RETAIN_MISSION

# ---------- 注册配置字段 ----------

register_plugin_config_field(
    "simple_gpt_hindsight_enabled",
    bool,
    default=False,
    description="是否启用 Hindsight 长期记忆",
)
register_plugin_config_field(
    "simple_gpt_hindsight_base_url",
    str,
    default="https://nvli-hs-api.centaurea.dev",
    description="Hindsight API 地址",
)
register_plugin_config_field(
    "simple_gpt_hindsight_api_key",
    str,
    default="",
    description="Hindsight API Key（启用时必填）",
)
register_plugin_config_field(
    "simple_gpt_hindsight_bank_prefix",
    str,
    default="gukohime",
    description="Hindsight bank id 前缀，默认 gukohime",
)
register_plugin_config_field(
    "simple_gpt_hindsight_bank_scope",
    str,
    default="chat",
    description="bank 隔离粒度：chat（每群/每用户一个 bank）或 global（合并到一个 bank）",
)
register_plugin_config_field(
    "simple_gpt_hindsight_recall_max_tokens",
    int,
    default=1500,
    description="单次 recall 注入上限 token 数",
    ge=128,
    le=8192,
)
register_plugin_config_field(
    "simple_gpt_hindsight_recall_budget",
    str,
    default="mid",
    description="recall 检索预算：low / mid / high",
)
register_plugin_config_field(
    "simple_gpt_hindsight_retain_async",
    bool,
    default=True,
    description="是否让 Hindsight 异步处理 retain（即发即返）",
)
register_plugin_config_field(
    "simple_gpt_hindsight_timeout",
    float,
    default=30.0,
    description="Hindsight HTTP 调用超时（秒）",
    gt=0,
)


if TYPE_CHECKING:
    from ... import Config


def _get_plugin_config() -> "Config":
    from ... import plugin_config

    return plugin_config


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# 进程级状态：document_id 是否已经在本进程触发过首次 retain
_first_retain_seen: set[str] = set()


class HindsightMemoryPlugin(SimpleGPTPlugin):
    """基于 Hindsight 远程 API 的长期记忆插件。"""

    priority = 250

    def __init__(self) -> None:
        self._config_loaded = False
        self._enabled = False
        self._base_url = ""
        self._api_key = ""
        self._prefix = "gukohime"
        self._scope = "chat"
        self._recall_max_tokens = 1500
        self._recall_budget = "mid"
        self._retain_async = True
        self._timeout = 30.0

    def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return
        config = _get_plugin_config()
        self._enabled = bool(config.simple_gpt_hindsight_enabled)
        self._base_url = (config.simple_gpt_hindsight_base_url or "").strip()
        self._api_key = (config.simple_gpt_hindsight_api_key or "").strip()
        self._prefix = (config.simple_gpt_hindsight_bank_prefix or "gukohime").strip()
        self._scope = (config.simple_gpt_hindsight_bank_scope or "chat").strip().lower()
        self._recall_max_tokens = int(config.simple_gpt_hindsight_recall_max_tokens)
        self._recall_budget = (config.simple_gpt_hindsight_recall_budget or "mid").strip().lower()
        self._retain_async = bool(config.simple_gpt_hindsight_retain_async)
        self._timeout = float(config.simple_gpt_hindsight_timeout)

        if not self._enabled:
            self._config_loaded = True
            logger.info("simple-gpt: hindsight 插件未启用")
            return
        if not self._api_key:
            self._enabled = False
            self._config_loaded = True
            logger.warning(
                "simple-gpt: hindsight 已启用但未配置 api_key，已自动停用"
            )
            return
        try:
            get_client(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        except Exception as exc:
            self._enabled = False
            logger.warning(f"simple-gpt: hindsight 客户端初始化失败: {exc}")
        self._config_loaded = True
        if self._enabled:
            logger.info(
                f"simple-gpt: hindsight 插件已加载 "
                f"(scope={self._scope}, prefix={self._prefix}, "
                f"base={self._base_url}, recall_budget={self._recall_budget}, "
                f"recall_max_tokens={self._recall_max_tokens})"
            )

    def _resolve(self, payload_extra: dict) -> Optional[BankResolution]:
        session_id = str(payload_extra.get("session_id", "") or "")
        sender_uid = str(payload_extra.get("sender_user_id", "") or "")
        return resolve_bank(
            session_id,
            sender_uid,
            prefix=self._prefix,
            scope=self._scope,
        )

    async def before_llm_request(
        self, payload: LLMRequestPayload
    ) -> LLMRequestPayload:
        self._ensure_config_loaded()
        if not self._enabled:
            return payload
        try:
            res = self._resolve(payload.extra)
            if res is None:
                return payload

            payload.extra["hindsight_bank_id"] = res.bank_id
            payload.extra["hindsight_chat_kind"] = res.chat_kind

            query = payload.latest_message or ""
            if not query.strip():
                return payload

            hits = await arecall(
                bank_id=res.bank_id,
                query=query,
                tags=res.default_recall_tags,
                tags_match="any",
                max_tokens=self._recall_max_tokens,
                budget=self._recall_budget,
            )
            if not hits:
                return payload

            lines = ["[长期记忆]"]
            for h in hits:
                text = (h.get("text") or "").strip()
                if not text:
                    continue
                fact_type = h.get("type") or ""
                tag = f"[{fact_type}] " if fact_type else ""
                lines.append(f"- {tag}{text}")
            lines.append("[记忆结束]")
            memory_block = "\n".join(lines) + "\n\n"
            payload.prompt = memory_block + payload.prompt
            payload.extra["injected_memories"] = True
            logger.info(
                f"simple-gpt: [hindsight] 已注入 {len(hits)} 条记忆 "
                f"(bank={res.bank_id})"
            )
        except Exception as exc:
            logger.warning(f"simple-gpt: hindsight before_llm_request 异常: {exc}")
        return payload

    async def after_llm_response(
        self, payload: LLMResponsePayload
    ) -> LLMResponsePayload:
        self._ensure_config_loaded()
        if not self._enabled:
            return payload
        try:
            res = self._resolve(payload.request.extra)
            if res is None:
                return payload
            asyncio.create_task(self._retain_turn(payload, res))
        except Exception as exc:
            logger.warning(f"simple-gpt: hindsight after_llm_response 异常: {exc}")
        return payload

    async def _retain_turn(
        self, payload: LLMResponsePayload, res: BankResolution
    ) -> None:
        try:
            sender = payload.request.sender or ""
            sender_uid = str(payload.request.extra.get("sender_user_id", "") or "")
            user_msg = (payload.request.latest_message or "").strip()
            bot_reply = (payload.content or "").strip()
            if not user_msg and not bot_reply:
                return

            now_iso = _now_iso()
            speaker_label = (
                f"{sender}({sender_uid})" if sender and sender_uid else (sender or sender_uid or "用户")
            )
            content = (
                f"{speaker_label} ({now_iso}): {user_msg}\n"
                f"鸽子姬 ({now_iso}): {bot_reply}"
            )

            session_id = str(payload.request.extra.get("session_id", "") or "unknown")
            today = _today_str()
            document_id = f"{session_id}-{today}"

            first_seen = document_id not in _first_retain_seen
            update_mode = "replace" if first_seen else "append"
            _first_retain_seen.add(document_id)

            await aensure_bank_mission(
                bank_id=res.bank_id, mission=RETAIN_MISSION
            )

            if res.chat_kind == "private":
                context_label = "QQ 私聊"
            elif res.chat_kind == "group":
                context_label = "QQ 群聊"
            else:
                context_label = "QQ 聊天"

            await aretain(
                bank_id=res.bank_id,
                content=content,
                document_id=document_id,
                tags=res.default_retain_tags,
                context=context_label,
                timestamp=now_iso,
                metadata={
                    "session_id": session_id,
                    "sender_user_id": sender_uid,
                },
                update_mode=update_mode,
                retain_async=self._retain_async,
            )
        except Exception as exc:
            logger.warning(f"simple-gpt: hindsight retain_turn 异常: {exc}")


_plugin_instance = HindsightMemoryPlugin()
register_simple_gpt_plugin(_plugin_instance)


driver = get_driver()


@driver.on_startup
async def _hindsight_startup() -> None:
    _plugin_instance._ensure_config_loaded()
    if not _plugin_instance._enabled:
        return
    # 暖身 recall —— 把鉴权问题暴露在启动而不是首条用户消息
    try:
        await arecall(
            bank_id=f"{_plugin_instance._prefix}-warmup",
            query="warmup",
            max_tokens=64,
            budget="low",
        )
        logger.info("simple-gpt: hindsight warmup 完成")
    except Exception as exc:
        logger.warning(
            f"simple-gpt: hindsight warmup 失败（请检查 base_url 与 api_key 是否正确）: {exc}"
        )


@driver.on_shutdown
async def _hindsight_shutdown() -> None:
    await aclose_client()
