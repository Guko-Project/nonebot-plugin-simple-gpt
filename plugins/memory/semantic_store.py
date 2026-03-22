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
                pa.field("created_at", pa.utf8()),
                pa.field(
                    "vector", pa.list_(pa.float32(), self._embedding_dim)
                ),
            ]
        )

        try:
            self._table = self._db.open_table("memories")
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
                "created_at": now,
                "vector": vector,
            }
            table.add([record])

        await asyncio.to_thread(_do)

    async def search(
        self, query_vector: List[float], session_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """向量相似度搜索，按 session_id 过滤。返回结果包含 similarity 字段。"""

        def _do() -> List[Dict[str, Any]]:
            table = self._ensure_connection()
            try:
                results = (
                    table.search(query_vector)
                    .where(f"session_id = '{session_id}'")
                    .metric("cosine")
                    .limit(top_k)
                    .to_pandas()
                )
            except Exception as exc:
                logger.debug(f"simple-gpt: 语义记忆搜索失败: {exc}")
                return []

            memories: List[Dict[str, Any]] = []
            for _, row in results.iterrows():
                # LanceDB cosine metric 返回 _distance (0=完全相同, 2=完全相反)
                # 转换为 similarity: 1 - distance
                distance = float(row.get("_distance", 1.0))
                memories.append(
                    {
                        "content": row["content"],
                        "speaker": row["speaker"],
                        "category": row["category"],
                        "importance": float(row["importance"]),
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
            table = self._ensure_connection()
            try:
                df = table.to_pandas()
                df = df[df["session_id"] == session_id].sort_values(
                    "created_at", ascending=False
                )
            except Exception as exc:
                logger.debug(f"simple-gpt: 语义记忆全量读取失败: {exc}")
                return []

            return [
                {
                    "content": row["content"],
                    "speaker": row["speaker"],
                    "category": row["category"],
                    "importance": float(row["importance"]),
                    "created_at": row["created_at"],
                }
                for _, row in df.iterrows()
            ]

        return await asyncio.to_thread(_do)
