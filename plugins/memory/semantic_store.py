from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import pyarrow as pa
from nonebot.log import logger


class SemanticStore:
    """LanceDB 语义记忆存储 - 向量相似度搜索。"""

    def __init__(self, db_path: str, embedding_dim: int) -> None:
        self._db_path = os.path.join(db_path, "semantic_lancedb")
        self._embedding_dim = embedding_dim
        self._db: Any = None
        self._table: Any = None

    def _ensure_connection(self) -> Any:
        if self._table is not None:
            return self._table

        import lancedb

        os.makedirs(self._db_path, exist_ok=True)
        self._db = lancedb.connect(self._db_path)

        schema = pa.schema(
            [
                pa.field("id", pa.utf8()),
                pa.field("session_id", pa.utf8()),
                pa.field("content", pa.utf8()),
                pa.field("speaker", pa.utf8()),
                pa.field("category", pa.utf8()),
                pa.field("importance", pa.float32()),
                pa.field("related_user_id", pa.utf8()),
                pa.field("created_at", pa.utf8()),
                pa.field(
                    "vector", pa.list_(pa.float32(), self._embedding_dim)
                ),
            ]
        )

        try:
            table = self._db.open_table("memories")
            # 检查是否有 related_user_id 列，没有则重建
            existing_names = {f.name for f in table.schema}
            if "related_user_id" not in existing_names:
                logger.info(
                    "simple-gpt: memory semantic_store schema 变更，重建表"
                )
                self._db.drop_table("memories")
                table = self._db.create_table("memories", schema=schema)
            self._table = table
        except Exception:
            self._table = self._db.create_table("memories", schema=schema)

        logger.debug("simple-gpt: memory semantic_store 已初始化")
        return self._table

    async def add(self, memory: Dict[str, Any], vector: List[float]) -> None:
        """添加一条语义记忆。"""

        def _do() -> None:
            table = self._ensure_connection()
            now = datetime.now(timezone.utc).isoformat()
            record = {
                "id": str(uuid.uuid4()),
                "session_id": memory["session_id"],
                "content": memory["content"],
                "speaker": memory.get("speaker", ""),
                "category": memory.get("category", "fact"),
                "importance": float(memory.get("importance", 0.5)),
                "related_user_id": memory.get("related_user_id", ""),
                "created_at": now,
                "vector": vector,
            }
            table.add([record])

        await asyncio.to_thread(_do)

    async def search(
        self,
        query_vector: List[float],
        session_id: str,
        top_k: int,
        *,
        related_user_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """向量相似度搜索，按 session_id 过滤。

        related_user_id:
          - None: 不过滤（返回所有记忆）
          - "": 仅返回群体记忆（related_user_id 为空）
          - 具体 uid: 仅返回该用户的记忆
        """

        def _do() -> List[Dict[str, Any]]:
            table = self._ensure_connection()
            try:
                where = f"session_id = '{session_id}'"
                if related_user_id is not None:
                    where += f" AND related_user_id = '{related_user_id}'"
                results = (
                    table.search(query_vector)
                    .where(where)
                    .metric("cosine")
                    .limit(top_k)
                    .to_pandas()
                )
            except Exception as exc:
                logger.debug(f"simple-gpt: 语义记忆搜索失败: {exc}")
                return []

            memories: List[Dict[str, Any]] = []
            for _, row in results.iterrows():
                distance = float(row.get("_distance", 1.0))
                memories.append(
                    {
                        "content": row["content"],
                        "speaker": row["speaker"],
                        "category": row["category"],
                        "importance": float(row["importance"]),
                        "related_user_id": row.get("related_user_id", ""),
                        "created_at": row["created_at"],
                        "similarity": 1.0 - distance,
                    }
                )
            return memories

        return await asyncio.to_thread(_do)

    async def delete_session(self, session_id: str) -> int:
        """删除某 session 的所有语义记忆。返回删除行数。"""

        def _do() -> int:
            table = self._ensure_connection()
            try:
                before = table.count_rows()
                table.delete(f"session_id = '{session_id}'")
                after = table.count_rows()
                return before - after
            except Exception as exc:
                logger.debug(f"simple-gpt: 语义记忆删除失败: {exc}")
                return 0

        return await asyncio.to_thread(_do)

    async def get_all_session(self, session_id: str) -> List[Dict[str, Any]]:
        """获取某 session 的所有语义记忆（不做向量搜索，按时间排序）。"""

        def _do() -> List[Dict[str, Any]]:
            try:
                return self._scan_memories(session_id=session_id)
            except Exception as exc:
                logger.debug(f"simple-gpt: 语义记忆全量读取失败: {exc}")
                return []

        return await asyncio.to_thread(_do)

    async def list_session_memories(
        self,
        session_id: str,
        *,
        category: str = "",
        related_user_id: str = "",
        keyword: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """分页获取某 session 的语义记忆。"""

        def _do() -> Dict[str, Any]:
            try:
                offset = max(0, (page - 1) * page_size)
                items = self._scan_memories(
                    session_id=session_id,
                    category=category,
                    related_user_id=related_user_id,
                    keyword=keyword,
                    limit=page_size,
                    offset=offset,
                )
                total = self._count_memories(
                    session_id=session_id,
                    category=category,
                    related_user_id=related_user_id,
                    keyword=keyword,
                )
                return {"items": items, "total": total}
            except Exception as exc:
                logger.debug(f"simple-gpt: 语义记忆分页读取失败: {exc}")
                return {"items": [], "total": 0}

        return await asyncio.to_thread(_do)

    @staticmethod
    def _escape_sql_string(value: str) -> str:
        return value.replace("'", "''")

    def _build_where_clause(
        self,
        *,
        session_id: str,
        category: str = "",
        related_user_id: str = "",
        keyword: str = "",
    ) -> str:
        parts = [
            f"session_id = '{self._escape_sql_string(session_id)}'"
        ]
        if category:
            parts.append(f"category = '{self._escape_sql_string(category)}'")
        if related_user_id:
            parts.append(
                f"related_user_id = '{self._escape_sql_string(related_user_id)}'"
            )
        if keyword:
            escaped = self._escape_sql_string(keyword)
            parts.append(
                f"(content LIKE '%{escaped}%' OR speaker LIKE '%{escaped}%')"
            )
        return " AND ".join(parts)

    def _scan_memories(
        self,
        *,
        session_id: str,
        category: str = "",
        related_user_id: str = "",
        keyword: str = "",
        limit: int | None = None,
        offset: int | None = None,
    ) -> List[Dict[str, Any]]:
        table = self._ensure_connection()
        where = self._build_where_clause(
            session_id=session_id,
            category=category,
            related_user_id=related_user_id,
            keyword=keyword,
        )
        query = (
            table.search()
            .where(where)
            .select(
                [
                    "id",
                    "session_id",
                    "content",
                    "speaker",
                    "category",
                    "importance",
                    "related_user_id",
                    "created_at",
                ]
            )
        )
        if offset:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        rows = query.to_list()
        rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "content": row["content"],
                "speaker": row.get("speaker", ""),
                "category": row.get("category", ""),
                "importance": float(row.get("importance", 0.0)),
                "related_user_id": row.get("related_user_id", ""),
                "created_at": row.get("created_at", ""),
            }
            for row in rows
        ]

    def _count_memories(
        self,
        *,
        session_id: str,
        category: str = "",
        related_user_id: str = "",
        keyword: str = "",
    ) -> int:
        table = self._ensure_connection()
        where = self._build_where_clause(
            session_id=session_id,
            category=category,
            related_user_id=related_user_id,
            keyword=keyword,
        )
        try:
            return int(table.count_rows(filter=where))
        except TypeError:
            return len(
                table.search()
                .where(where)
                .select(["id"])
                .to_list()
            )
