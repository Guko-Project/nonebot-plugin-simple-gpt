from __future__ import annotations

import asyncio
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict

from nonebot.log import logger

# global 层使用的固定 session_id
GLOBAL_SESSION_ID = "__global__"


class ProfileStore:
    """SQLite 用户档案存储 - 两层架构。

    - global 层：跨群通用属性（生日、职业等），以 QQ user_id 为键
    - group 层：群内特定属性（昵称等），以 QQ user_id + session_id 为键
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = os.path.join(db_path, "profiles.db")
        self._conn: sqlite3.Connection | None = None

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id    TEXT NOT NULL,
                session_id TEXT NOT NULL,
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, session_id, key)
            )
            """
        )
        self._conn.commit()
        logger.debug("simple-gpt: memory profile_store 已初始化")
        return self._conn

    async def upsert(
        self,
        user_id: str,
        session_id: str,
        key: str,
        value: str,
        *,
        scope: str = "group",
    ) -> None:
        """插入或更新一条档案记录。

        scope="global" 时存储到 global 层（session_id 固定为 __global__）。
        scope="group" 时存储到当前 session_id。
        """
        actual_session_id = GLOBAL_SESSION_ID if scope == "global" else session_id

        def _do() -> None:
            conn = self._ensure_connection()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, session_id, key, value, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (user_id, session_id, key)
                DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (user_id, actual_session_id, key, value, now),
            )
            conn.commit()

        await asyncio.to_thread(_do)

    async def get_merged_profiles(
        self, session_id: str
    ) -> Dict[str, Dict[str, str]]:
        """获取合并后的档案：global 层 + group 层（group 层同 key 覆盖 global）。

        返回: {user_id: {key: value, ...}, ...}
        """

        def _do() -> Dict[str, Dict[str, str]]:
            conn = self._ensure_connection()
            # 一次查询两层，按 updated_at 排序让 group 层后写入覆盖 global
            cursor = conn.execute(
                """
                SELECT user_id, key, value, session_id FROM user_profiles
                WHERE session_id IN (?, ?)
                ORDER BY
                    CASE session_id WHEN ? THEN 0 ELSE 1 END,
                    updated_at
                """,
                (GLOBAL_SESSION_ID, session_id, GLOBAL_SESSION_ID),
            )
            result: Dict[str, Dict[str, str]] = {}
            for uid, key, value, sid in cursor.fetchall():
                # global 先填入，group 后覆盖同 key
                result.setdefault(uid, {})[key] = value
            return result

        return await asyncio.to_thread(_do)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
