"""把人工编辑过的 JSONL 导入 Hindsight。

输入：scripts/export_memory.py 产出的 JSONL。每行一个对象：
    {
      "kind": "profile" | "memory",
      "bank_id":     "gukohime-qq-group-...",
      "document_id": "profile-... | memory-...",
      "content":     "...",
      "context":     "...",
      "tags":        [...],
      "timestamp":   "2026-04-01T12:00:00+00:00" | null,
      "metadata":    { ... },
      "_source":     { ... }    // 只读, 导入时忽略
    }

运行：
    python scripts/migrate_to_hindsight.py \\
        --input data/memory_export.jsonl \\
        --base-url https://nvli-hs-api.centaurea.dev \\
        --api-key sk-... \\
        [--dry-run] [--concurrency 8]

行为：
    - 每条调 aretain(update_mode='replace', retain_async=True), 用 JSONL 里的
      bank_id / document_id / tags / context / timestamp / metadata。
    - 每个新见到的 bank_id 先 PATCH retain_mission 一次。
    - document_id 稳定, 重跑只 upsert 不重复。
    - 缺字段 / bank_id 为空的行跳过并记入 err 计数。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# 只独立加载 mission.py, 不走 plugins/__init__.py（避免触发兄弟子插件的相对导入）
_mission_mod = _load_isolated(
    "_mig_mission", _PLUGIN_DIR / "plugins" / "memory" / "mission.py"
)
RETAIN_MISSION: str = _mission_mod.RETAIN_MISSION

try:
    from hindsight_client import Hindsight  # noqa: E402
except ImportError as exc:
    print(f"[fatal] 缺少 hindsight-client: {exc}", file=sys.stderr)
    print("        运行：pip install hindsight-client", file=sys.stderr)
    sys.exit(2)


def _parse_iso(ts: Any) -> Optional[datetime]:
    if ts is None or ts == "":
        return None
    if isinstance(ts, datetime):
        return ts
    s = str(ts).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[fatal] 输入文件不存在: {path}", file=sys.stderr)
        sys.exit(2)
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"[skip] 第 {ln} 行 JSON 解析失败: {exc}", file=sys.stderr)
                continue
            if not isinstance(obj, dict):
                print(f"[skip] 第 {ln} 行不是对象", file=sys.stderr)
                continue
            out.append(obj)
    return out


async def _ensure_mission(
    client: Hindsight,
    bank_id: str,
    seen: set[str],
    *,
    dry_run: bool,
) -> None:
    if bank_id in seen:
        return
    seen.add(bank_id)
    if dry_run:
        print(f"[dry-run] set retain_mission on bank={bank_id}")
        return
    try:
        await client._aupdate_bank_config(bank_id, {"retain_mission": RETAIN_MISSION})  # type: ignore[attr-defined]
        print(f"[mission] bank={bank_id} retain_mission 已设置")
    except Exception as exc:
        print(f"[warn] 设置 bank={bank_id} retain_mission 失败: {exc}")


async def _retain_one(
    client: Hindsight,
    rec: Dict[str, Any],
    *,
    dry_run: bool,
) -> bool:
    bank_id = str(rec.get("bank_id", "") or "")
    doc_id = str(rec.get("document_id", "") or "")
    content = str(rec.get("content", "") or "").strip()
    if not bank_id or not doc_id or not content:
        return False
    tags = [str(t) for t in rec.get("tags", []) or [] if str(t).strip()]
    context = str(rec.get("context", "") or "") or None
    ts = _parse_iso(rec.get("timestamp"))
    metadata_raw = rec.get("metadata") or {}
    metadata = {str(k): str(v) for k, v in metadata_raw.items()} if metadata_raw else None

    if dry_run:
        print(
            f"[dry-run] retain bank={bank_id} doc={doc_id} "
            f"tags={tags} ctx={context} ts={ts} "
            f"content[:60]={content[:60]!r}"
        )
        return True
    try:
        await client.aretain(
            bank_id=bank_id,
            content=content,
            document_id=doc_id,
            tags=tags or None,
            context=context,
            timestamp=ts,
            metadata=metadata,
            update_mode="replace",
            retain_async=True,
        )
        return True
    except Exception as exc:
        print(f"[err] retain 失败 bank={bank_id} doc={doc_id}: {exc}")
        return False


async def main(
    *,
    input_path: str,
    base_url: str,
    api_key: str,
    dry_run: bool,
    concurrency: int,
) -> None:
    records = _load_jsonl(Path(input_path))
    print(f"读取 JSONL: 共 {len(records)} 条")

    client = Hindsight(base_url=base_url, api_key=api_key or None, timeout=60.0)
    sem = asyncio.Semaphore(concurrency)
    bank_seen: set[str] = set()
    counters = {"ok": 0, "err": 0, "by_kind": {}}

    async def worker(rec: Dict[str, Any]) -> None:
        bank_id = str(rec.get("bank_id", "") or "")
        if not bank_id:
            counters["err"] += 1
            return
        kind = str(rec.get("kind", "") or "?")
        async with sem:
            await _ensure_mission(client, bank_id, bank_seen, dry_run=dry_run)
            ok = await _retain_one(client, rec, dry_run=dry_run)
        if ok:
            counters["ok"] += 1
            counters["by_kind"][kind] = counters["by_kind"].get(kind, 0) + 1
        else:
            counters["err"] += 1

    try:
        await asyncio.gather(*(worker(r) for r in records))
    finally:
        try:
            await client.aclose()
        except Exception:
            pass

    print("---")
    print("导入完成：")
    print(f"  ok  = {counters['ok']}")
    print(f"  err = {counters['err']}")
    for k, v in counters["by_kind"].items():
        print(f"  by_kind[{k}] = {v}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--input", required=True, help="JSONL 路径 (export_memory.py 的输出)")
    p.add_argument("--base-url", default="https://nvli-hs-api.centaurea.dev")
    p.add_argument("--api-key", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    asyncio.run(
        main(
            input_path=args.input,
            base_url=args.base_url,
            api_key=args.api_key,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
        )
    )
