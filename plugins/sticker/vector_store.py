from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

import pyarrow as pa
from nonebot.log import logger

if TYPE_CHECKING:
    from lancedb.db import DBConnection
    from lancedb.table import Table


class StickerVectorStore:
    """LanceDB vector store for sticker retrieval."""

    def __init__(self, db_path: str, embedding_dim: int) -> None:
        self._db_path = os.path.join(db_path, "sticker_lancedb")
        self._embedding_dim = embedding_dim
        self._db: DBConnection | None = None
        self._table: Table | None = None

    def _ensure_connection(self) -> Table:
        table = self._table
        if table is not None:
            return table

        import lancedb

        os.makedirs(self._db_path, exist_ok=True)
        logger.debug(
            "simple-gpt: sticker vector store 初始化开始 "
            f"(db_path={self._db_path}, embedding_dim={self._embedding_dim})"
        )
        db = lancedb.connect(self._db_path)
        self._db = db
        schema = pa.schema(
            [
                pa.field("id", pa.utf8()),
                pa.field("session_id", pa.utf8()),
                pa.field("description", pa.utf8()),
                pa.field("emotion_tags", pa.utf8()),
                pa.field("intent_tags", pa.utf8()),
                pa.field("scene_tags", pa.utf8()),
                pa.field("usage_notes", pa.utf8()),
                pa.field("aliases", pa.utf8()),
                pa.field("file_path", pa.utf8()),
                pa.field("semantic_vector", pa.list_(pa.float32(), self._embedding_dim)),
                pa.field("tag_vector", pa.list_(pa.float32(), self._embedding_dim)),
            ]
        )

        try:
            table = db.open_table("stickers")
            logger.debug("simple-gpt: sticker vector table 已打开 (table=stickers)")
        except Exception:
            table = db.create_table("stickers", schema=schema)
            logger.debug("simple-gpt: sticker vector table 已创建 (table=stickers)")

        self._table = table
        logger.debug("simple-gpt: sticker vector store 已初始化")
        return table

    async def add(
        self,
        *,
        sticker_id: str,
        session_id: str,
        description: str,
        emotion_tags: List[str],
        intent_tags: List[str],
        scene_tags: List[str],
        usage_notes: str,
        aliases: List[str],
        file_path: str,
        semantic_vector: List[float],
        tag_vector: List[float],
    ) -> None:
        def _do() -> None:
            table = self._ensure_connection()
            logger.debug(
                "simple-gpt: sticker 向量写入开始 "
                f"(session={session_id}, sticker_id={sticker_id}, "
                f"semantic_dim={len(semantic_vector)}, tag_dim={len(tag_vector)})"
            )
            table.add(
                [
                    {
                        "id": sticker_id,
                        "session_id": session_id,
                        "description": description,
                        "emotion_tags": ",".join(emotion_tags),
                        "intent_tags": ",".join(intent_tags),
                        "scene_tags": ",".join(scene_tags),
                        "usage_notes": usage_notes,
                        "aliases": ",".join(aliases),
                        "file_path": file_path,
                        "semantic_vector": semantic_vector,
                        "tag_vector": tag_vector,
                    }
                ]
            )
            logger.debug(
                "simple-gpt: sticker 向量写入完成 "
                f"(session={session_id}, sticker_id={sticker_id})"
            )

        await asyncio.to_thread(_do)

    async def search(
        self,
        *,
        session_id: str,
        session_ids: Sequence[str] | None = None,
        query_vector: List[float],
        vector_column: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        def _do() -> List[Dict[str, Any]]:
            table = self._ensure_connection()
            search_session_ids = list(dict.fromkeys(session_ids or [session_id]))
            if not search_session_ids:
                logger.debug("simple-gpt: sticker 向量搜索跳过：session_ids 为空")
                return []
            session_filter = _build_session_filter(search_session_ids)
            try:
                logger.debug(
                    "simple-gpt: sticker 向量搜索开始 "
                    f"(session={session_id}, search_sessions={search_session_ids}, "
                    f"vector_column={vector_column}, "
                    f"query_dim={len(query_vector)}, top_k={top_k})"
                )
                results = (
                    table.search(query_vector, vector_column_name=vector_column)
                    .where(session_filter)
                    .metric("cosine")
                    .limit(top_k)
                    .to_pandas()
                )
            except Exception as exc:
                logger.debug(f"simple-gpt: sticker 向量搜索失败: {exc}")
                return []

            rows: List[Dict[str, Any]] = []
            for _, row in results.iterrows():
                distance = float(row.get("_distance", 1.0))
                rows.append(
                    {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "description": row["description"],
                        "emotion_tags": row["emotion_tags"],
                        "intent_tags": row["intent_tags"],
                        "scene_tags": row["scene_tags"],
                        "usage_notes": row["usage_notes"],
                        "aliases": row["aliases"],
                        "file_path": row["file_path"],
                        "similarity": 1.0 - distance,
                    }
                )
            logger.debug(
                "simple-gpt: sticker 向量搜索完成 "
                f"(session={session_id}, search_sessions={search_session_ids}, "
                f"vector_column={vector_column}, rows={len(rows)}, "
                f"ids={[row['id'] for row in rows]})"
            )
            return rows

        return await asyncio.to_thread(_do)

    async def delete(self, sticker_id: str) -> None:
        def _do() -> None:
            table = self._ensure_connection()
            try:
                logger.debug(f"simple-gpt: sticker 向量删除开始 (sticker_id={sticker_id})")
                table.delete(f"id = '{sticker_id}'")
                logger.debug(f"simple-gpt: sticker 向量删除完成 (sticker_id={sticker_id})")
            except Exception as exc:
                logger.debug(f"simple-gpt: sticker 向量删除失败: {exc}")

        await asyncio.to_thread(_do)


def _quote_sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _build_session_filter(session_ids: Sequence[str]) -> str:
    if len(session_ids) == 1:
        return f"session_id = {_quote_sql_literal(session_ids[0])}"
    return "session_id IN (" + ", ".join(_quote_sql_literal(item) for item in session_ids) + ")"
