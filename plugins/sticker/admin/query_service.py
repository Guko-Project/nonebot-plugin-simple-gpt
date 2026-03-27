from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .. import get_admin_handles


class StickerAdminQueryService:
    def __init__(self) -> None:
        self._store = None
        self._vector_store = None

    def _ensure_handles(self) -> None:
        if self._store is not None and self._vector_store is not None:
            return
        self._store, self._vector_store = get_admin_handles()

    @staticmethod
    def _normalize_page(page: int, page_size: int) -> tuple[int, int]:
        page = max(1, page)
        page_size = min(max(1, page_size), 100)
        return page, page_size

    async def get_health(self) -> Dict[str, Any]:
        self._ensure_handles()
        return {
            "enabled": self._store is not None and self._vector_store is not None,
        }

    async def list_sessions(self, *, limit: int) -> List[str]:
        self._ensure_handles()
        if self._store is None:
            return []
        return await self._store.list_sessions(limit=limit)

    async def list_stickers(
        self,
        *,
        session_id: str,
        keyword: str,
        emotion_tag: str,
        intent_tag: str,
        scene_tag: str,
        enabled: str,
        page: int,
        page_size: int,
    ) -> Dict[str, Any]:
        self._ensure_handles()
        if self._store is None:
            return {"items": [], "total": 0, "page": 1, "page_size": 0}
        page, page_size = self._normalize_page(page, page_size)
        data = await self._store.list_stickers(
            session_id=session_id,
            keyword=keyword,
            emotion_tag=emotion_tag,
            intent_tag=intent_tag,
            scene_tag=scene_tag,
            enabled=enabled,
            page=page,
            page_size=page_size,
        )
        return {**data, "page": page, "page_size": page_size}

    async def get_sticker(self, sticker_id: str) -> Dict[str, Any] | None:
        self._ensure_handles()
        if self._store is None:
            return None
        return await self._store.get_sticker(sticker_id)

    async def get_image_path(self, sticker_id: str) -> Path | None:
        item = await self.get_sticker(sticker_id)
        if item is None:
            return None
        path = Path(item["file_path"])
        return path if path.exists() else None

    async def delete_sticker(self, sticker_id: str) -> bool:
        self._ensure_handles()
        if self._store is None or self._vector_store is None:
            return False
        deleted = await self._store.delete_sticker(sticker_id)
        if deleted is None:
            return False
        await self._vector_store.delete(sticker_id)
        file_path = Path(deleted["file_path"])
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass
        return True
