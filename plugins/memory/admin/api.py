from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from nonebot import get_driver
from nonebot.log import logger

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

    service = MemoryAdminQueryService(plugin)
    public_router = APIRouter(
        prefix="/simple-gpt/memory-admin",
        tags=["simple-gpt-memory-admin"],
    )
    protected_router = APIRouter(
        prefix="/simple-gpt/memory-admin",
        tags=["simple-gpt-memory-admin"],
        dependencies=[Depends(verify_admin_token)],
    )

    @public_router.get("/ui", response_class=HTMLResponse)
    async def ui() -> str:
        return _STATIC_INDEX_PATH.read_text(encoding="utf-8")

    @protected_router.get("/health")
    async def health() -> dict:
        return {"ok": True, "data": await service.get_health(), "error": None}

    @protected_router.get("/meta")
    async def meta() -> dict:
        return {"ok": True, "data": await service.get_meta(), "error": None}

    @protected_router.get("/sessions")
    async def sessions(limit: int = Query(default=100, ge=1, le=500)) -> dict:
        return {
            "ok": True,
            "data": {"items": await service.list_sessions(limit=limit)},
            "error": None,
        }

    @protected_router.get("/profiles")
    async def profiles(
        session_id: str = Query(default=""),
        user_id: str = Query(default=""),
        key: str = Query(default=""),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=100),
    ) -> dict:
        data = await service.list_profiles(
            session_id=session_id,
            user_id=user_id,
            key=key,
            page=page,
            page_size=page_size,
        )
        return {"ok": True, "data": data, "error": None}

    @protected_router.get("/memories")
    async def memories(
        session_id: str = Query(default=""),
        category: str = Query(default=""),
        related_user_id: str = Query(default=""),
        keyword: str = Query(default=""),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        data = await service.list_memories(
            session_id=session_id,
            category=category,
            related_user_id=related_user_id,
            keyword=keyword,
            page=page,
            page_size=page_size,
        )
        return {"ok": True, "data": data, "error": None}

    @protected_router.get("/dialogues")
    async def dialogues(
        session_id: str = Query(default=""),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        data = await service.list_dialogues(
            session_id=session_id,
            page=page,
            page_size=page_size,
        )
        return {"ok": True, "data": data, "error": None}

    @protected_router.get("/dialogues/{dialogue_id}")
    async def dialogue_detail(dialogue_id: int) -> dict:
        data = await service.get_dialogue(dialogue_id)
        if data is None:
            raise HTTPException(status_code=404, detail="dialogue not found")
        return {"ok": True, "data": data, "error": None}

    app = _get_app()
    app.include_router(public_router)
    app.include_router(protected_router)
    _ROUTER_REGISTERED = True
    logger.info("simple-gpt: memory admin API 已注册在 /simple-gpt/memory-admin")
    return True
