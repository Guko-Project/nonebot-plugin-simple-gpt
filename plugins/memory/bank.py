from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass(frozen=True)
class BankResolution:
    bank_id: str
    default_retain_tags: List[str] = field(default_factory=list)
    default_recall_tags: List[str] = field(default_factory=list)
    chat_kind: Literal["group", "private", "global"] = "group"


def resolve_bank(
    session_id: str,
    sender_user_id: str,
    *,
    prefix: str,
    scope: str,
) -> Optional[BankResolution]:
    """根据 session_id / scope 派生 (bank_id, retain_tags, recall_tags, chat_kind)。

    session_id 形态:
        - "group_<gid>"   → 群聊
        - "private_<uid>" → 私聊（主插件目前未启用）
        - 其他/为空        → 返回 None，钩子直接 no-op
    """
    if not session_id:
        return None

    prefix = prefix.strip() or "memory"
    scope = scope.strip().lower() or "chat"

    if scope == "global":
        retain_tags: List[str] = []
        if sender_user_id:
            retain_tags.append(f"user:{sender_user_id}")
        retain_tags.append(f"chat:{session_id}")
        recall_tags = (
            [f"user:{sender_user_id}"] if sender_user_id else []
        )
        kind: Literal["group", "private", "global"] = "global"
        return BankResolution(
            bank_id=f"{prefix}-global",
            default_retain_tags=retain_tags,
            default_recall_tags=recall_tags,
            chat_kind=kind,
        )

    if session_id.startswith("group_"):
        gid = session_id[len("group_"):]
        if not gid:
            return None
        retain = [f"user:{sender_user_id}"] if sender_user_id else []
        recall = [f"user:{sender_user_id}"] if sender_user_id else []
        return BankResolution(
            bank_id=f"{prefix}-qq-group-{gid}",
            default_retain_tags=retain,
            default_recall_tags=recall,
            chat_kind="group",
        )

    if session_id.startswith("private_"):
        uid = session_id[len("private_"):]
        if not uid:
            return None
        return BankResolution(
            bank_id=f"{prefix}-qq-user-{uid}",
            default_retain_tags=[],
            default_recall_tags=[],
            chat_kind="private",
        )

    return None
