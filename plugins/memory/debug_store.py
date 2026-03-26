from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nonebot.log import logger


class MemoryDebugStore:
    """SQLite 对话调试存储。"""

    def __init__(self, db_path: str, *, max_rows: int = 1000) -> None:
        self._db_path = os.path.join(db_path, "debug.db")
        self._conn: sqlite3.Connection | None = None
        self._max_rows = max_rows

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn

        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_debug_dialogues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                sender_user_id TEXT NOT NULL,
                latest_message TEXT NOT NULL,
                final_prompt TEXT NOT NULL,
                reply_content TEXT NOT NULL,
                injected_memories INTEGER NOT NULL DEFAULT 0,
                extracted_profiles_json TEXT NOT NULL,
                extracted_memories_json TEXT NOT NULL,
                stored_profiles_json TEXT NOT NULL,
                stored_memories_json TEXT NOT NULL,
                error_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_debug_session_created
            ON memory_debug_dialogues (session_id, created_at DESC)
            """
        )
        self._conn.commit()
        logger.debug("simple-gpt: memory debug_store 已初始化")
        return self._conn

    async def record_dialogue(
        self,
        *,
        conversation_id: str,
        session_id: str,
        sender: str,
        sender_user_id: str,
        latest_message: str,
        final_prompt: str,
        reply_content: str,
        injected_memories: bool,
        extracted_profiles: List[Dict[str, Any]],
        extracted_memories: List[Dict[str, Any]],
        stored_profiles: List[Dict[str, Any]],
        stored_memories: List[Dict[str, Any]],
        error_text: str = "",
    ) -> None:
        def _do() -> None:
            conn = self._ensure_connection()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO memory_debug_dialogues (
                    conversation_id,
                    session_id,
                    sender,
                    sender_user_id,
                    latest_message,
                    final_prompt,
                    reply_content,
                    injected_memories,
                    extracted_profiles_json,
                    extracted_memories_json,
                    stored_profiles_json,
                    stored_memories_json,
                    error_text,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    sender = excluded.sender,
                    sender_user_id = excluded.sender_user_id,
                    latest_message = excluded.latest_message,
                    final_prompt = excluded.final_prompt,
                    reply_content = excluded.reply_content,
                    injected_memories = excluded.injected_memories,
                    extracted_profiles_json = excluded.extracted_profiles_json,
                    extracted_memories_json = excluded.extracted_memories_json,
                    stored_profiles_json = excluded.stored_profiles_json,
                    stored_memories_json = excluded.stored_memories_json,
                    error_text = excluded.error_text,
                    created_at = excluded.created_at
                """,
                (
                    conversation_id,
                    session_id,
                    sender,
                    sender_user_id,
                    latest_message,
                    final_prompt,
                    reply_content,
                    1 if injected_memories else 0,
                    json.dumps(extracted_profiles, ensure_ascii=False),
                    json.dumps(extracted_memories, ensure_ascii=False),
                    json.dumps(stored_profiles, ensure_ascii=False),
                    json.dumps(stored_memories, ensure_ascii=False),
                    error_text,
                    now,
                ),
            )
            conn.commit()
            self._prune_locked(conn)

        await asyncio.to_thread(_do)

    def _prune_locked(self, conn: sqlite3.Connection) -> None:
        if self._max_rows <= 0:
            return
        conn.execute(
            """
            DELETE FROM memory_debug_dialogues
            WHERE id NOT IN (
                SELECT id FROM memory_debug_dialogues
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (self._max_rows,),
        )
        conn.commit()

    async def list_dialogues(
        self,
        *,
        session_id: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            conn = self._ensure_connection()
            where = ""
            params: List[Any] = []
            if session_id:
                where = "WHERE session_id = ?"
                params.append(session_id)

            total = conn.execute(
                f"SELECT COUNT(*) FROM memory_debug_dialogues {where}",
                params,
            ).fetchone()[0]
            offset = max(0, (page - 1) * page_size)
            cursor = conn.execute(
                f"""
                SELECT
                    id,
                    conversation_id,
                    session_id,
                    sender,
                    sender_user_id,
                    latest_message,
                    reply_content,
                    injected_memories,
                    error_text,
                    created_at
                FROM memory_debug_dialogues
                {where}
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, offset],
            )
            items = [
                {
                    "id": row[0],
                    "conversation_id": row[1],
                    "session_id": row[2],
                    "sender": row[3],
                    "sender_user_id": row[4],
                    "latest_message": row[5],
                    "reply_content": row[6],
                    "injected_memories": bool(row[7]),
                    "error_text": row[8],
                    "created_at": row[9],
                }
                for row in cursor.fetchall()
            ]
            return {"items": items, "total": total}

        return await asyncio.to_thread(_do)

    async def get_dialogue(self, dialogue_id: int) -> Optional[Dict[str, Any]]:
        def _do() -> Optional[Dict[str, Any]]:
            conn = self._ensure_connection()
            row = conn.execute(
                """
                SELECT
                    id,
                    conversation_id,
                    session_id,
                    sender,
                    sender_user_id,
                    latest_message,
                    final_prompt,
                    reply_content,
                    injected_memories,
                    extracted_profiles_json,
                    extracted_memories_json,
                    stored_profiles_json,
                    stored_memories_json,
                    error_text,
                    created_at
                FROM memory_debug_dialogues
                WHERE id = ?
                """,
                (dialogue_id,),
            ).fetchone()
            if row is None:
                return None

            return {
                "id": row[0],
                "conversation_id": row[1],
                "session_id": row[2],
                "sender": row[3],
                "sender_user_id": row[4],
                "latest_message": row[5],
                "final_prompt": row[6],
                "reply_content": row[7],
                "injected_memories": bool(row[8]),
                "extracted_profiles": json.loads(row[9]),
                "extracted_memories": json.loads(row[10]),
                "stored_profiles": json.loads(row[11]),
                "stored_memories": json.loads(row[12]),
                "error_text": row[13],
                "created_at": row[14],
            }

        return await asyncio.to_thread(_do)

    async def list_sessions(self, *, limit: int = 100) -> List[str]:
        def _do() -> List[str]:
            conn = self._ensure_connection()
            cursor = conn.execute(
                """
                SELECT session_id, MAX(id) AS last_id
                FROM memory_debug_dialogues
                GROUP BY session_id
                ORDER BY last_id DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [row[0] for row in cursor.fetchall()]

        return await asyncio.to_thread(_do)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
