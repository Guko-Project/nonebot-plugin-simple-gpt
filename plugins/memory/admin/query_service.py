from __future__ import annotations

from typing import Any, Dict, List


class MemoryAdminQueryService:
    """Memory 管理端只读查询服务。"""

    def __init__(self, plugin: Any) -> None:
        self._plugin = plugin

    @staticmethod
    def _normalize_page(page: int, page_size: int) -> tuple[int, int]:
        page = max(1, page)
        page_size = min(max(1, page_size), 100)
        return page, page_size

    async def get_health(self) -> Dict[str, Any]:
        self._plugin._ensure_config_loaded()
        return {
            "enabled": self._plugin._enabled,
            "admin_enabled": bool(self._plugin._admin_token),
            "debug_enabled": self._plugin._debug_store is not None,
        }

    async def get_meta(self) -> Dict[str, Any]:
        self._plugin._ensure_config_loaded()
        return {
            "scope": self._plugin._scope,
            "top_k": self._plugin._top_k,
            "debug_max_rows": self._plugin._debug_max_rows,
            "routes_registered": self._plugin._admin_routes_registered,
        }

    async def list_profiles(
        self,
        *,
        session_id: str,
        user_id: str,
        key: str,
        page: int,
        page_size: int,
    ) -> Dict[str, Any]:
        self._plugin._ensure_config_loaded()
        if self._plugin._profile_store is None:
            return {"items": [], "total": 0, "page": 1, "page_size": 0}

        page, page_size = self._normalize_page(page, page_size)
        data = await self._plugin._profile_store.list_profiles(
            session_id=session_id,
            user_id=user_id,
            key=key,
            page=page,
            page_size=page_size,
        )
        return {**data, "page": page, "page_size": page_size}

    async def list_memories(
        self,
        *,
        session_id: str,
        category: str,
        related_user_id: str,
        keyword: str,
        page: int,
        page_size: int,
    ) -> Dict[str, Any]:
        self._plugin._ensure_config_loaded()
        if not session_id:
            return {"items": [], "total": 0, "page": 1, "page_size": 0}
        if self._plugin._semantic_store is None:
            return {"items": [], "total": 0, "page": 1, "page_size": 0}

        page, page_size = self._normalize_page(page, page_size)
        data = await self._plugin._semantic_store.list_session_memories(
            session_id,
            category=category,
            related_user_id=related_user_id,
            keyword=keyword,
            page=page,
            page_size=page_size,
        )
        return {**data, "page": page, "page_size": page_size}

    async def list_dialogues(
        self,
        *,
        session_id: str,
        page: int,
        page_size: int,
    ) -> Dict[str, Any]:
        self._plugin._ensure_config_loaded()
        if self._plugin._debug_store is None:
            return {"items": [], "total": 0, "page": 1, "page_size": 0}

        page, page_size = self._normalize_page(page, page_size)
        data = await self._plugin._debug_store.list_dialogues(
            session_id=session_id,
            page=page,
            page_size=page_size,
        )
        return {**data, "page": page, "page_size": page_size}

    async def get_dialogue(self, dialogue_id: int) -> Dict[str, Any] | None:
        self._plugin._ensure_config_loaded()
        if self._plugin._debug_store is None:
            return None
        return await self._plugin._debug_store.get_dialogue(dialogue_id)

    async def list_sessions(self, *, limit: int) -> List[str]:
        self._plugin._ensure_config_loaded()
        sessions: List[str] = []
        if self._plugin._profile_store is not None:
            sessions.extend(await self._plugin._profile_store.list_sessions(limit=limit))
        if self._plugin._debug_store is not None:
            sessions.extend(await self._plugin._debug_store.list_sessions(limit=limit))

        deduped: List[str] = []
        seen = set()
        for item in sessions:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped
