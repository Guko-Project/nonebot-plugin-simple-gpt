from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from nonebot import get_driver
from nonebot.log import logger

from ...sticker.admin.query_service import StickerAdminQueryService
from .auth import verify_admin_token
from .query_service import MemoryAdminQueryService

_ROUTER_REGISTERED = False
_STATIC_INDEX_PATH = Path(__file__).with_name("static").joinpath("index.html")


def _get_app() -> Any:
    try:
        from nonebot import get_app

        return get_app()
    except Exception:
        driver = get_driver()
        app = getattr(driver, "server_app", None)
        if app is None:
            raise RuntimeError("nonebot server app is unavailable")
        return app


def register_admin_api(plugin: Any) -> bool:
    global _ROUTER_REGISTERED
    if _ROUTER_REGISTERED:
        return True

    memory_service = MemoryAdminQueryService(plugin)
    sticker_service = StickerAdminQueryService()

    public_router = APIRouter(
        prefix="/simple-gpt/admin",
        tags=["simple-gpt-admin"],
    )
    protected_router = APIRouter(
        prefix="/simple-gpt/admin",
        tags=["simple-gpt-admin"],
        dependencies=[Depends(verify_admin_token)],
    )
    legacy_public_router = APIRouter(
        prefix="/simple-gpt/memory-admin",
        tags=["simple-gpt-memory-admin"],
    )
    legacy_protected_router = APIRouter(
        prefix="/simple-gpt/memory-admin",
        tags=["simple-gpt-memory-admin"],
        dependencies=[Depends(verify_admin_token)],
    )

    @public_router.get("/ui", response_class=HTMLResponse)
    @legacy_public_router.get("/ui", response_class=HTMLResponse)
    async def ui() -> str:
        return _STATIC_INDEX_PATH.read_text(encoding="utf-8")

    @protected_router.get("/health")
    async def admin_health() -> dict:
        return {
            "ok": True,
            "data": {
                "memory": await memory_service.get_health(),
                "sticker": await sticker_service.get_health(),
            },
            "error": None,
        }

    @protected_router.get("/sessions")
    async def admin_sessions(limit: int = Query(default=100, ge=1, le=500)) -> dict:
        memory_sessions = await memory_service.list_sessions(limit=limit)
        sticker_sessions = await sticker_service.list_sessions(limit=limit)
        items = []
        seen = set()
        for session_id in memory_sessions + sticker_sessions:
            if session_id in seen:
                continue
            seen.add(session_id)
            items.append(session_id)
            if len(items) >= limit:
                break
        return {"ok": True, "data": {"items": items}, "error": None}

    @protected_router.get("/memory/health")
    @legacy_protected_router.get("/health")
    async def memory_health() -> dict:
        return {"ok": True, "data": await memory_service.get_health(), "error": None}

    @protected_router.get("/memory/meta")
    @legacy_protected_router.get("/meta")
    async def memory_meta() -> dict:
        return {"ok": True, "data": await memory_service.get_meta(), "error": None}

    @protected_router.get("/memory/sessions")
    @legacy_protected_router.get("/sessions")
    async def memory_sessions(limit: int = Query(default=100, ge=1, le=500)) -> dict:
        return {
            "ok": True,
            "data": {"items": await memory_service.list_sessions(limit=limit)},
            "error": None,
        }

    @protected_router.get("/memory/profiles")
    @legacy_protected_router.get("/profiles")
    async def memory_profiles(
        session_id: str = Query(default=""),
        user_id: str = Query(default=""),
        key: str = Query(default=""),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=100),
    ) -> dict:
        data = await memory_service.list_profiles(
            session_id=session_id,
            user_id=user_id,
            key=key,
            page=page,
            page_size=page_size,
        )
        return {"ok": True, "data": data, "error": None}

    @protected_router.get("/memory/memories")
    @legacy_protected_router.get("/memories")
    async def memory_memories(
        session_id: str = Query(default=""),
        category: str = Query(default=""),
        related_user_id: str = Query(default=""),
        keyword: str = Query(default=""),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        data = await memory_service.list_memories(
            session_id=session_id,
            category=category,
            related_user_id=related_user_id,
            keyword=keyword,
            page=page,
            page_size=page_size,
        )
        return {"ok": True, "data": data, "error": None}

    @protected_router.get("/memory/dialogues")
    @legacy_protected_router.get("/dialogues")
    async def memory_dialogues(
        session_id: str = Query(default=""),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        data = await memory_service.list_dialogues(
            session_id=session_id,
            page=page,
            page_size=page_size,
        )
        return {"ok": True, "data": data, "error": None}

    @protected_router.get("/memory/dialogues/{dialogue_id}")
    @legacy_protected_router.get("/dialogues/{dialogue_id}")
    async def memory_dialogue_detail(dialogue_id: int) -> dict:
        data = await memory_service.get_dialogue(dialogue_id)
        if data is None:
            raise HTTPException(status_code=404, detail="dialogue not found")
        return {"ok": True, "data": data, "error": None}

    @protected_router.get("/sticker/health")
    async def sticker_health() -> dict:
        return {"ok": True, "data": await sticker_service.get_health(), "error": None}

    @protected_router.get("/sticker/sessions")
    async def sticker_sessions(limit: int = Query(default=100, ge=1, le=500)) -> dict:
        return {
            "ok": True,
            "data": {"items": await sticker_service.list_sessions(limit=limit)},
            "error": None,
        }

    @protected_router.get("/sticker/items")
    async def sticker_items(
        session_id: str = Query(default=""),
        keyword: str = Query(default=""),
        emotion_tag: str = Query(default=""),
        intent_tag: str = Query(default=""),
        scene_tag: str = Query(default=""),
        enabled: str = Query(default=""),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        data = await sticker_service.list_stickers(
            session_id=session_id,
            keyword=keyword,
            emotion_tag=emotion_tag,
            intent_tag=intent_tag,
            scene_tag=scene_tag,
            enabled=enabled,
            page=page,
            page_size=page_size,
        )
        return {"ok": True, "data": data, "error": None}

    @protected_router.get("/sticker/items/{sticker_id}")
    async def sticker_item_detail(sticker_id: str) -> dict:
        data = await sticker_service.get_sticker(sticker_id)
        if data is None:
            raise HTTPException(status_code=404, detail="sticker not found")
        return {"ok": True, "data": data, "error": None}

    @protected_router.get("/sticker/items/{sticker_id}/image")
    async def sticker_item_image(sticker_id: str) -> FileResponse:
        image_path = await sticker_service.get_image_path(sticker_id)
        if image_path is None:
            raise HTTPException(status_code=404, detail="sticker image not found")
        return FileResponse(image_path)

    @protected_router.delete("/sticker/items/{sticker_id}")
    async def sticker_item_delete(sticker_id: str) -> dict:
        deleted = await sticker_service.delete_sticker(sticker_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="sticker not found")
        return {"ok": True, "data": {"deleted": True}, "error": None}

    app = _get_app()
    app.include_router(public_router)
    app.include_router(protected_router)
    app.include_router(legacy_public_router)
    app.include_router(legacy_protected_router)
    _ROUTER_REGISTERED = True
    logger.info(
        "simple-gpt: admin API 已注册在 /simple-gpt/admin "
        "(legacy memory path kept at /simple-gpt/memory-admin)"
    )
    return True
