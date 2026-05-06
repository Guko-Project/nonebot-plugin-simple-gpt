"""把旧 simple_gpt_memory 数据导出为 JSONL，便于人工编辑后导入 Hindsight。

读取：
    <db_path>/profiles.db          —— SQLite 表 user_profiles
                                       (user_id, session_id, key, value, updated_at)
    <db_path>/semantic_lancedb/    —— LanceDB 表 memories
                                       (id, session_id, content, speaker, category,
                                        importance, related_user_id, created_at, vector)

输出：每行一个 JSON 对象，结构：
    {
      "kind": "profile" | "memory",
      "bank_id":     "gukohime-qq-group-<gid>",
      "document_id": "memory-group_<gid>-<rowid>",
      "content":     "...",
      "context":     "fact" | "用户档案" | ...,
      "tags":        ["user:789", "imported:semantic", ...],
      "timestamp":   "2026-04-01T12:00:00+00:00",   // ISO 8601, 可省
      "metadata":    {"speaker": "...", "original_id": "..."},
      "_source":     { /* 原始 LanceDB / SQLite 字段, 只读, 导入时忽略 */ }
    }

人工编辑约定：
    - 自由修改 content / tags / context / metadata / timestamp / document_id。
    - bank_id 通常不动（除非你想搬到另一个 bank）。
    - "_source" 字段是只读快照, 导入时忽略, 留作改前对照。
    - 若想丢弃某条, 直接删除该行即可。

运行：
    python scripts/export_memory.py \\
        --db-path data/simple_gpt_memory \\
        --output data/memory_export.jsonl \\
        [--bank-prefix gukohime] \\
        [--bank-scope chat] \\
        [--limit 50]

依赖：lancedb 仅在导出时用到, 运行前 `pip install -e '.[migration]'`。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_PLUGIN_DIR = _HERE.parent


def _load_isolated(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_bank_mod = _load_isolated(
    "_exp_bank", _PLUGIN_DIR / "plugins" / "memory" / "bank.py"
)
resolve_bank = _bank_mod.resolve_bank


def _iso(ts: Any) -> Optional[str]:
    if ts is None or ts == "":
        return None
    if isinstance(ts, datetime):
        return ts.isoformat()
    s = str(ts).strip()
    if not s:
        return None
    # 标准化尾部 Z
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
    except Exception:
        return s  # 保留原值, 让人工决定


def _resolve_for_profile(
    session_id: str, user_id: str, *, prefix: str, scope: str
) -> Optional[Tuple[str, List[str]]]:
    if session_id == "__global__":
        bank_id = f"{prefix}-global"
        tags = (
            [f"user:{user_id}", "scope:global", "imported:profile"]
            if user_id
            else ["scope:global", "imported:profile"]
        )
        return bank_id, tags
    res = resolve_bank(session_id, user_id, prefix=prefix, scope=scope)
    if res is None:
        return None
    tags = list(res.default_retain_tags)
    if user_id and f"user:{user_id}" not in tags:
        tags.append(f"user:{user_id}")
    tags.extend(["scope:group", "imported:profile"])
    return res.bank_id, tags


def _resolve_for_semantic(
    session_id: str, related_user_id: str, *, prefix: str, scope: str
) -> Optional[Tuple[str, List[str]]]:
    res = resolve_bank(session_id, related_user_id, prefix=prefix, scope=scope)
    if res is None:
        return None
    return res.bank_id, list(res.default_retain_tags)


def _read_profiles(db_path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    sqlite_path = db_path / "profiles.db"
    if not sqlite_path.exists():
        print(f"[skip] 未找到 {sqlite_path}", file=sys.stderr)
        return []
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT user_id, session_id, key, value, updated_at FROM user_profiles"
        )
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    if limit is not None:
        rows = rows[:limit]
    return rows


def _read_semantic(db_path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    lance_root = db_path / "semantic_lancedb"
    if not lance_root.exists():
        print(f"[skip] 未找到 {lance_root}", file=sys.stderr)
        return []
    try:
        import lancedb
    except ImportError:
        print(
            "[fatal] 缺少 lancedb, 请 `pip install -e '.[migration]'` 或 `pip install lancedb pyarrow`",
            file=sys.stderr,
        )
        sys.exit(2)
    db = lancedb.connect(str(lance_root))
    try:
        tbl = db.open_table("memories")
    except Exception as exc:
        print(f"[skip] 打开 LanceDB 表 memories 失败: {exc}", file=sys.stderr)
        return []
    df = tbl.to_pandas()
    if limit is not None:
        df = df.head(limit)
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "id": str(row.get("id", "") or ""),
                "session_id": str(row.get("session_id", "") or ""),
                "content": str(row.get("content", "") or ""),
                "speaker": str(row.get("speaker", "") or ""),
                "category": str(row.get("category", "") or ""),
                "importance": float(row.get("importance", 0.0) or 0.0),
                "related_user_id": str(row.get("related_user_id", "") or ""),
                "created_at": row.get("created_at", None),
            }
        )
    return rows


def _profile_record(
    row: Dict[str, Any], *, prefix: str, scope: str
) -> Optional[Dict[str, Any]]:
    uid = str(row.get("user_id", "") or "")
    sid = str(row.get("session_id", "") or "")
    key = str(row.get("key", "") or "")
    val = str(row.get("value", "") or "")
    if not uid or not key or not val:
        return None
    resolved = _resolve_for_profile(sid, uid, prefix=prefix, scope=scope)
    if resolved is None:
        return None
    bank_id, tags = resolved
    if sid == "__global__":
        doc_id = f"profile-global-{uid}-{key}"
    elif sid.startswith("group_"):
        gid = sid[len("group_"):]
        doc_id = f"profile-group-{gid}-{uid}-{key}"
    else:
        doc_id = f"profile-{sid}-{uid}-{key}"
    return {
        "kind": "profile",
        "bank_id": bank_id,
        "document_id": doc_id,
        "content": f"用户 {uid} 的{key}是{val}",
        "context": "用户档案",
        "tags": tags,
        "timestamp": _iso(row.get("updated_at")),
        "metadata": {"original_session": sid, "key": key},
        "_source": {
            "table": "user_profiles",
            "user_id": uid,
            "session_id": sid,
            "key": key,
            "value": val,
            "updated_at": _iso(row.get("updated_at")),
        },
    }


def _memory_record(
    row: Dict[str, Any], *, prefix: str, scope: str
) -> Optional[Dict[str, Any]]:
    sid = str(row.get("session_id", "") or "")
    rid = str(row.get("id", "") or "")
    content = (row.get("content", "") or "").strip()
    if not sid or not rid or not content:
        return None
    related = str(row.get("related_user_id", "") or "")
    resolved = _resolve_for_semantic(sid, related, prefix=prefix, scope=scope)
    if resolved is None:
        return None
    bank_id, tags = resolved
    category = str(row.get("category", "") or "fact")
    importance = float(row.get("importance", 0.0) or 0.0)
    speaker = str(row.get("speaker", "") or "")
    extra_tags = [
        "imported:semantic",
        f"category:{category}",
        f"importance:{round(importance, 1)}",
    ]
    if related:
        extra_tags.append(f"user:{related}")
    all_tags = list(dict.fromkeys(tags + extra_tags))
    return {
        "kind": "memory",
        "bank_id": bank_id,
        "document_id": f"memory-{sid}-{rid}",
        "content": content,
        "context": category or "fact",
        "tags": all_tags,
        "timestamp": _iso(row.get("created_at")),
        "metadata": {"speaker": speaker, "original_id": rid},
        "_source": {
            "table": "memories",
            "id": rid,
            "session_id": sid,
            "speaker": speaker,
            "category": category,
            "importance": importance,
            "related_user_id": related,
            "created_at": _iso(row.get("created_at")),
        },
    }


def main(
    *,
    db_path: str,
    output: str,
    bank_prefix: str,
    bank_scope: str,
    limit: Optional[int],
) -> None:
    src = Path(db_path)
    if not src.exists():
        print(f"[fatal] 数据目录不存在: {src}", file=sys.stderr)
        sys.exit(2)

    profiles = _read_profiles(src, limit)
    memories = _read_semantic(src, limit)

    written = {"profile": 0, "memory": 0, "skipped": 0}
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in profiles:
            rec = _profile_record(row, prefix=bank_prefix, scope=bank_scope)
            if rec is None:
                written["skipped"] += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written["profile"] += 1
        for row in memories:
            rec = _memory_record(row, prefix=bank_prefix, scope=bank_scope)
            if rec is None:
                written["skipped"] += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written["memory"] += 1

    print(f"导出完成 → {out_path}")
    for k, v in written.items():
        print(f"  {k:8s} = {v}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--db-path", default="data/simple_gpt_memory")
    p.add_argument("--output", default="data/memory_export.jsonl")
    p.add_argument("--bank-prefix", default="gukohime")
    p.add_argument(
        "--bank-scope",
        default="chat",
        choices=["chat", "global"],
    )
    p.add_argument("--limit", type=int, default=None)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(
        db_path=args.db_path,
        output=args.output,
        bank_prefix=args.bank_prefix,
        bank_scope=args.bank_scope,
        limit=args.limit,
    )
