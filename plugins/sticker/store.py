from __future__ import annotations

import asyncio
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nonebot.log import logger


class StickerStore:
    """SQLite metadata store for saved stickers."""

    def __init__(self, db_path: str) -> None:
        self._db_path = os.path.join(db_path, "stickers.db")
        self._conn: sqlite3.Connection | None = None

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn

        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stickers (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                phash TEXT NOT NULL,
                description TEXT NOT NULL,
                emotion_tags TEXT NOT NULL,
                intent_tags TEXT NOT NULL,
                scene_tags TEXT NOT NULL,
                ocr_text TEXT NOT NULL,
                usage_notes TEXT NOT NULL,
                aliases TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_stickers_session_sha256 "
            "ON stickers(session_id, sha256)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stickers_session_enabled "
            "ON stickers(session_id, enabled)"
        )
        conn.commit()
        self._conn = conn
        logger.debug("simple-gpt: sticker store 已初始化")
        return conn

    async def find_duplicate(
        self, session_id: str, sha256: str, phash: str
    ) -> Optional[Dict[str, Any]]:
        def _do() -> Optional[Dict[str, Any]]:
            conn = self._ensure_connection()
            row = conn.execute(
                """
                SELECT id, file_path, sha256, phash, description
                FROM stickers
                WHERE session_id = ?
                  AND enabled = 1
                  AND (sha256 = ? OR phash = ?)
                LIMIT 1
                """,
                (session_id, sha256, phash),
            ).fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "file_path": row[1],
                "sha256": row[2],
                "phash": row[3],
                "description": row[4],
            }

        return await asyncio.to_thread(_do)

    async def add(
        self,
        *,
        session_id: str,
        file_path: str,
        sha256: str,
        phash: str,
        description: str,
        emotion_tags: List[str],
        intent_tags: List[str],
        scene_tags: List[str],
        ocr_text: str,
        usage_notes: str,
        aliases: List[str],
        created_by: str,
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            conn = self._ensure_connection()
            sticker_id = str(uuid.uuid4())
            created_at = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO stickers (
                    id, session_id, file_path, sha256, phash, description,
                    emotion_tags, intent_tags, scene_tags, ocr_text,
                    usage_notes, aliases, created_by, created_at, enabled
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    sticker_id,
                    session_id,
                    file_path,
                    sha256,
                    phash,
                    description,
                    ",".join(emotion_tags),
                    ",".join(intent_tags),
                    ",".join(scene_tags),
                    ocr_text,
                    usage_notes,
                    ",".join(aliases),
                    created_by,
                    created_at,
                ),
            )
            conn.commit()
            return {
                "id": sticker_id,
                "session_id": session_id,
                "file_path": file_path,
                "sha256": sha256,
                "phash": phash,
                "description": description,
                "emotion_tags": emotion_tags,
                "intent_tags": intent_tags,
                "scene_tags": scene_tags,
                "ocr_text": ocr_text,
                "usage_notes": usage_notes,
                "aliases": aliases,
                "created_by": created_by,
                "created_at": created_at,
                "enabled": True,
            }

        return await asyncio.to_thread(_do)

    async def count_enabled(self, session_id: str) -> int:
        def _do() -> int:
            conn = self._ensure_connection()
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM stickers
                WHERE session_id = ? AND enabled = 1
                """,
                (session_id,),
            ).fetchone()
            return int(row[0]) if row else 0

        return await asyncio.to_thread(_do)

    async def list_sessions(self, *, limit: int = 100) -> List[str]:
        def _do() -> List[str]:
            conn = self._ensure_connection()
            cursor = conn.execute(
                """
                SELECT session_id, MAX(created_at) AS latest_created_at
                FROM stickers
                GROUP BY session_id
                ORDER BY latest_created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [row[0] for row in cursor.fetchall()]

        return await asyncio.to_thread(_do)

    async def list_stickers(
        self,
        *,
        session_id: str = "",
        keyword: str = "",
        emotion_tag: str = "",
        intent_tag: str = "",
        scene_tag: str = "",
        enabled: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            conn = self._ensure_connection()
            where_parts: List[str] = []
            params: List[Any] = []

            if session_id:
                where_parts.append("session_id = ?")
                params.append(session_id)
            if keyword:
                like = f"%{keyword}%"
                where_parts.append(
                    "("
                    "description LIKE ? OR "
                    "usage_notes LIKE ? OR "
                    "ocr_text LIKE ? OR "
                    "aliases LIKE ?"
                    ")"
                )
                params.extend([like, like, like, like])
            if emotion_tag:
                where_parts.append("emotion_tags LIKE ?")
                params.append(f"%{emotion_tag}%")
            if intent_tag:
                where_parts.append("intent_tags LIKE ?")
                params.append(f"%{intent_tag}%")
            if scene_tag:
                where_parts.append("scene_tags LIKE ?")
                params.append(f"%{scene_tag}%")
            if enabled in {"0", "1"}:
                where_parts.append("enabled = ?")
                params.append(int(enabled))

            where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
            total = conn.execute(
                f"SELECT COUNT(*) FROM stickers {where}",
                params,
            ).fetchone()[0]
            offset = max(0, (page - 1) * page_size)
            cursor = conn.execute(
                f"""
                SELECT
                    id, session_id, file_path, description, emotion_tags,
                    intent_tags, scene_tags, ocr_text, usage_notes,
                    aliases, created_by, created_at, enabled
                FROM stickers
                {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, offset],
            )
            items = [
                {
                    "id": row[0],
                    "session_id": row[1],
                    "file_path": row[2],
                    "description": row[3],
                    "emotion_tags": row[4],
                    "intent_tags": row[5],
                    "scene_tags": row[6],
                    "ocr_text": row[7],
                    "usage_notes": row[8],
                    "aliases": row[9],
                    "created_by": row[10],
                    "created_at": row[11],
                    "enabled": bool(row[12]),
                }
                for row in cursor.fetchall()
            ]
            return {"items": items, "total": total}

        return await asyncio.to_thread(_do)

    async def get_sticker(self, sticker_id: str) -> Optional[Dict[str, Any]]:
        def _do() -> Optional[Dict[str, Any]]:
            conn = self._ensure_connection()
            row = conn.execute(
                """
                SELECT
                    id, session_id, file_path, sha256, phash, description,
                    emotion_tags, intent_tags, scene_tags, ocr_text,
                    usage_notes, aliases, created_by, created_at, enabled
                FROM stickers
                WHERE id = ?
                LIMIT 1
                """,
                (sticker_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "session_id": row[1],
                "file_path": row[2],
                "sha256": row[3],
                "phash": row[4],
                "description": row[5],
                "emotion_tags": row[6],
                "intent_tags": row[7],
                "scene_tags": row[8],
                "ocr_text": row[9],
                "usage_notes": row[10],
                "aliases": row[11],
                "created_by": row[12],
                "created_at": row[13],
                "enabled": bool(row[14]),
            }

        return await asyncio.to_thread(_do)

    async def delete_sticker(self, sticker_id: str) -> Optional[Dict[str, Any]]:
        def _do() -> Optional[Dict[str, Any]]:
            conn = self._ensure_connection()
            row = conn.execute(
                """
                SELECT id, file_path
                FROM stickers
                WHERE id = ?
                LIMIT 1
                """,
                (sticker_id,),
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM stickers WHERE id = ?", (sticker_id,))
            conn.commit()
            return {"id": row[0], "file_path": row[1]}

        return await asyncio.to_thread(_do)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
