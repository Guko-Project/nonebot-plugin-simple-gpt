from __future__ import annotations

import asyncio
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

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
        logger.debug(f"simple-gpt: sticker store 初始化开始 (db_path={self._db_path})")
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
                logger.debug(
                    "simple-gpt: sticker 查重完成：无重复 "
                    f"(session={session_id}, sha256={sha256[:12]}..., phash={phash})"
                )
                return None
            logger.debug(
                "simple-gpt: sticker 查重完成：命中重复 "
                f"(session={session_id}, sticker_id={row[0]}, sha256={row[2][:12]}..., "
                f"phash={row[3]})"
            )
            return {
                "id": row[0],
                "file_path": row[1],
                "sha256": row[2],
                "phash": row[3],
                "description": row[4],
            }

        return await asyncio.to_thread(_do)

    async def find_by_hash(
        self, session_id: str, sha256: str, phash: str
    ) -> Optional[Dict[str, Any]]:
        def _do() -> Optional[Dict[str, Any]]:
            conn = self._ensure_connection()
            row = conn.execute(
                """
                SELECT id, file_path, sha256, phash, description
                FROM stickers
                WHERE session_id = ?
                  AND (sha256 = ? OR phash = ?)
                LIMIT 1
                """,
                (session_id, sha256, phash),
            ).fetchone()
            if row is None:
                logger.debug(
                    "simple-gpt: sticker 按哈希查询未命中 "
                    f"(session={session_id}, sha256={sha256[:12]}..., phash={phash})"
                )
                return None
            logger.debug(
                "simple-gpt: sticker 按哈希查询命中 "
                f"(session={session_id}, sticker_id={row[0]}, sha256={row[2][:12]}..., "
                f"phash={row[3]})"
            )
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
            logger.debug(
                "simple-gpt: sticker 元数据写入完成 "
                f"(session={session_id}, sticker_id={sticker_id}, file_path={file_path}, "
                f"emotion_tags={emotion_tags}, intent_tags={intent_tags}, "
                f"scene_tags={scene_tags}, aliases={aliases})"
            )
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
            count = int(row[0]) if row else 0
            logger.debug(
                "simple-gpt: sticker 可用数量查询完成 "
                f"(session={session_id}, count={count})"
            )
            return count

        return await asyncio.to_thread(_do)

    async def count_enabled_for_sessions(self, session_ids: Sequence[str]) -> int:
        def _do() -> int:
            unique_session_ids = list(dict.fromkeys(session_ids))
            if not unique_session_ids:
                logger.debug("simple-gpt: sticker 多会话可用数量查询跳过：session_ids 为空")
                return 0

            conn = self._ensure_connection()
            placeholders = ",".join("?" for _ in unique_session_ids)
            row = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM stickers
                WHERE session_id IN ({placeholders}) AND enabled = 1
                """,
                unique_session_ids,
            ).fetchone()
            count = int(row[0]) if row else 0
            logger.debug(
                "simple-gpt: sticker 多会话可用数量查询完成 "
                f"(sessions={unique_session_ids}, count={count})"
            )
            return count

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
            sessions = [row[0] for row in cursor.fetchall()]
            logger.debug(
                "simple-gpt: sticker 会话列表查询完成 "
                f"(limit={limit}, count={len(sessions)})"
            )
            return sessions

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
            logger.debug(
                "simple-gpt: sticker 列表查询完成 "
                f"(session={session_id!r}, keyword={keyword!r}, emotion_tag={emotion_tag!r}, "
                f"intent_tag={intent_tag!r}, scene_tag={scene_tag!r}, enabled={enabled!r}, "
                f"page={page}, page_size={page_size}, total={total}, items={len(items)})"
            )
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
                logger.debug(f"simple-gpt: sticker 详情查询未命中 (sticker_id={sticker_id})")
                return None
            logger.debug(f"simple-gpt: sticker 详情查询命中 (sticker_id={sticker_id})")
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
                logger.debug(f"simple-gpt: sticker 删除元数据未命中 (sticker_id={sticker_id})")
                return None
            conn.execute("DELETE FROM stickers WHERE id = ?", (sticker_id,))
            conn.commit()
            logger.debug(
                "simple-gpt: sticker 删除元数据完成 "
                f"(sticker_id={row[0]}, file_path={row[1]})"
            )
            return {"id": row[0], "file_path": row[1]}

        return await asyncio.to_thread(_do)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("simple-gpt: sticker store 连接已关闭")
