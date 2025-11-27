from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HistoryEntry:
    speaker: str
    content: str
    is_bot: bool = False
