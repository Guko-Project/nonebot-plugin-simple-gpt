from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class HistoryEntry:
    speaker: str
    content: str
    is_bot: bool = False
    images: List[str] = field(default_factory=list)
    user_id: str = ""  # 平台用户 ID（如 QQ 号），bot 消息为空
