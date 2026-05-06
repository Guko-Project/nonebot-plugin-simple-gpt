"""Hindsight HTTP API 的薄壳封装。

依赖 `hindsight-client` 包（已在 pyproject 加入）。本模块对外暴露五个函数：
    - get_client / aclose_client    —— 单例生命周期
    - aensure_bank_mission          —— 首次见到 bank 时设一次 retain_mission
    - aretain                       —— 写入一轮对话内容
    - arecall                       —— 读取相关记忆

错误处理原则：所有公开函数遇到异常都 logger.warning 后返回降级值（retain
返回 None，recall 返回空列表），不向上抛——记忆是辅助信号，挂掉不能阻塞主链。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from nonebot.log import logger

try:
    from hindsight_client import Hindsight
except ImportError as exc:  # pragma: no cover
    Hindsight = None  # type: ignore[assignment]
    _IMPORT_ERROR: Optional[ImportError] = exc
else:
    _IMPORT_ERROR = None

_client: Optional["Hindsight"] = None
_init_lock: Optional[asyncio.Lock] = None
_missioned_banks: set[str] = set()


def _get_lock() -> asyncio.Lock:
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


def get_client(*, base_url: str, api_key: str, timeout: float) -> "Hindsight":
    """同步取得单例。Hindsight 客户端构造无 IO，可以在事件循环外调。"""
    global _client
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "未安装 hindsight-client，请 pip install hindsight-client"
        ) from _IMPORT_ERROR
    if _client is None:
        _client = Hindsight(
            base_url=base_url, api_key=api_key or None, timeout=timeout
        )
    return _client


async def aclose_client() -> None:
    """driver.on_shutdown 调用。"""
    global _client, _missioned_banks
    if _client is None:
        return
    try:
        await _client.aclose()
    except Exception as exc:
        logger.warning(f"simple-gpt: hindsight aclose 失败: {exc}")
    finally:
        _client = None
        _missioned_banks = set()


async def aensure_bank_mission(*, bank_id: str, mission: str) -> None:
    """首次见到 bank_id 时设一次 retain_mission，之后 no-op。失败 swallow。"""
    if not bank_id or not mission:
        return
    if bank_id in _missioned_banks:
        return
    async with _get_lock():
        if bank_id in _missioned_banks:
            return
        client = _client
        if client is None:
            return
        try:
            # _aupdate_bank_config 是 PATCH /v1/default/banks/{bank_id}/config
            # 公开的 update_bank_config 是同步包装，没有 async 公开形态。
            await client._aupdate_bank_config(  # type: ignore[attr-defined]
                bank_id, {"retain_mission": mission}
            )
            logger.info(f"simple-gpt: hindsight bank {bank_id} retain_mission 已设置")
        except Exception as exc:
            logger.warning(
                f"simple-gpt: hindsight 设置 bank {bank_id} retain_mission 失败: {exc}"
            )
        finally:
            # 即便失败也加入集合，避免每次 retain 都重试拖慢主链
            _missioned_banks.add(bank_id)


def _to_datetime(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # 兼容尾部 Z
        s = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


async def aretain(
    *,
    bank_id: str,
    content: str,
    document_id: str,
    tags: Sequence[str] = (),
    context: str = "",
    timestamp: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    update_mode: str = "replace",
    retain_async: bool = True,
) -> None:
    """写入一轮对话/事件。失败仅日志。"""
    client = _client
    if client is None:
        logger.warning("simple-gpt: hindsight client 未初始化，跳过 retain")
        return
    if not bank_id or not content or not document_id:
        return
    try:
        await client.aretain(
            bank_id=bank_id,
            content=content,
            document_id=document_id,
            tags=list(tags) if tags else None,
            context=context or None,
            timestamp=_to_datetime(timestamp),
            metadata=metadata or None,
            update_mode=update_mode,
            retain_async=retain_async,
        )
    except Exception as exc:
        logger.warning(
            f"simple-gpt: hindsight retain 失败 bank={bank_id} doc={document_id}: {exc}"
        )


async def arecall(
    *,
    bank_id: str,
    query: str,
    tags: Sequence[str] = (),
    tags_match: str = "any",
    types: Sequence[str] = ("world", "experience", "observation"),
    max_tokens: int = 2000,
    budget: str = "mid",
) -> List[Dict[str, Any]]:
    """读取相关记忆。返回 dict 列表（text/type/tags/context/...）。失败返回 []。"""
    client = _client
    if client is None:
        logger.warning("simple-gpt: hindsight client 未初始化，跳过 recall")
        return []
    if not bank_id or not query:
        return []
    try:
        logger.info(f"simple-gpt: hindsight recall 开始 bank={bank_id} query={query}, tags={tags}, tags_match={tags_match}, types={types}, max_tokens={max_tokens}, budget={budget}")
        resp = await client.arecall(
            bank_id=bank_id,
            query=query,
            types=list(types) if types else None,
            max_tokens=max_tokens,
            budget=budget,
            tags=list(tags) if tags else None,
            tags_match=tags_match,  # type: ignore[arg-type]
        )
    except Exception as exc:
        logger.warning(f"simple-gpt: hindsight recall 失败 bank={bank_id}: {exc}")
        return []

    out: List[Dict[str, Any]] = []
    for r in resp.results or []:
        out.append(
            {
                "id": r.id,
                "text": r.text,
                "type": r.type,
                "context": r.context,
                "tags": list(r.tags or []),
                "entities": list(r.entities or []),
                "occurred_start": r.occurred_start,
                "occurred_end": r.occurred_end,
                "mentioned_at": r.mentioned_at,
                "document_id": r.document_id,
            }
        )
    return out
