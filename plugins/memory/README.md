# 长期记忆插件（Hindsight 后端）

这个子插件用 [Hindsight](https://nvli-hs-api.centaurea.dev) 远程记忆服务替换了原本的本地 SQLite + LanceDB 实现。所有事实抽取、向量检索、观察聚合都在 Hindsight 服务端完成；本地不再持久化任何记忆数据。

## 端到端流程

```
用户消息
  → before_llm_request
       ├─ resolve_bank(session_id, sender_uid) → bank_id
       └─ arecall(bank_id, query=latest_message) → 注入 [长期记忆] 段落
  → 主 LLM 生成回复
  → after_llm_response
       └─ asyncio.create_task → 后台
            ├─ aensure_bank_mission(bank_id, RETAIN_MISSION)（首次）
            └─ aretain(bank_id, content=本轮对话, document_id=<bank>-<日期>,
                       update_mode=replace|append, retain_async=True)
```

## Bank 隔离方案

bank_id 由 `session_id` 与 `simple_gpt_hindsight_bank_scope` 共同决定：

| session_id | scope | bank_id | retain tags | recall tags / match |
|------------|-------|---------|-------------|---------------------|
| `group_<gid>` | `chat`（默认） | `<prefix>-qq-group-<gid>` | `["user:<uid>"]` | `["user:<uid>"]`, `any` |
| `private_<uid>` | `chat` | `<prefix>-qq-user-<uid>` | `[]`（bank 已是 1:1） | `[]`, `any` |
| 任意 | `global` | `<prefix>-global` | `["user:<uid>", "chat:<session_id>"]` | `["user:<uid>"]`, `any` |

> 当前主插件（`nonebot_plugin_simple_gpt/__init__.py`）只分发 `GroupMessageEvent`，`session_id` 形如 `group_<gid>`。私聊路径已在 `bank.py` 就绪——未来主插件开放私聊只需在 message handler 里按 `PrivateMessageEvent` 派生 `session_id="private_<uid>"`，本子插件无需改动。

## 文档与增量

- `document_id = "<session_id>-<UTC YYYY-MM-DD>"`：每会话每天一份文档。
- 进程内首次写入此 document_id → `update_mode="replace"`；后续 → `update_mode="append"`，content 仅含本轮对话（用户消息 + bot 回复）。
- Hindsight 服务端自动跳过已抽过的旧 chunk，所以单轮 retain 成本与发送量**不随对话长度增长**。
- 进程重启会让当天首次 retain 触发一次 `replace`（已抽过的事实仍留在 bank，仅是文档原文重置）。

## Tag 体系

- 永远 `key:value` 格式。`user:` 永远是 QQ 号（稳定标识），不写昵称。
- 群聊不打 `group:<gid>`——bank 已按群分，重复 tag 是冗余。仅 `scope=global` 才需要 `chat:<session_id>` 回溯。
- Hindsight 不支持单条记忆改 tag。需要更换 tag 体系时，用同 `document_id` + `update_mode="replace"` 重写整文档，或按 memory_id 单删，或 `delete bank` 全清。

## retain_mission

`mission.py` 里的 `RETAIN_MISSION` 常量描述本项目想抽取的五类内容（重要事实、对他人认知、人际关系、历史事件 / 可复用知识、近期经历 / 情绪 / 活动）和明确忽略的内容（寒暄、emoji、复读等）。

应用方式：插件维护 `_missioned_banks: set[str]`，**首次** retain 一个 bank 前先 `PATCH /v1/default/banks/{bank_id}/config` 设置 `retain_mission`，之后该进程内不再重复设置。设置失败也会标记成"已尝试"，不阻塞主链。

## 配置项

所有 env 变量名为大写，对应字段如下：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `simple_gpt_hindsight_enabled` | bool | `false` | 启用开关 |
| `simple_gpt_hindsight_base_url` | str | `https://nvli-hs-api.centaurea.dev` | Hindsight API 地址 |
| `simple_gpt_hindsight_api_key` | str | `""` | 启用时必填 |
| `simple_gpt_hindsight_bank_prefix` | str | `gukohime` | bank id 前缀 |
| `simple_gpt_hindsight_bank_scope` | str | `chat` | `chat`（按对话分）或 `global`（合并） |
| `simple_gpt_hindsight_recall_max_tokens` | int | `1500` | 单次 recall token 预算 |
| `simple_gpt_hindsight_recall_budget` | str | `mid` | `low` / `mid` / `high` |
| `simple_gpt_hindsight_retain_async` | bool | `true` | Hindsight 服务端异步处理 retain |
| `simple_gpt_hindsight_timeout` | float | `30.0` | HTTP 超时（秒） |

`.env` 示例：

```
SIMPLE_GPT_HINDSIGHT_ENABLED=true
SIMPLE_GPT_HINDSIGHT_API_KEY=sk-xxxxxx
SIMPLE_GPT_HINDSIGHT_BANK_PREFIX=gukohime
```

## 旧数据迁移（导出 → 人工编辑 → 导入）

为了让你能在导入前对每条记忆做人工修订，迁移分两阶段：

### 1. 导出为 JSONL

```bash
# 一次性安装迁移所需 lancedb（运行时不再需要）
pip install -e '.[migration]'

python scripts/export_memory.py \
    --db-path data/simple_gpt_memory \
    --output data/memory_export.jsonl \
    --bank-prefix gukohime --bank-scope chat
```

每行一条记忆，包含已派生好的 `bank_id` / `document_id` / `content` / `tags` / `context` / `timestamp` / `metadata`，以及只读的 `_source`（导入时忽略，仅供改前对照）。结构示例：

```json
{
  "kind": "memory",
  "bank_id": "gukohime-qq-group-12345",
  "document_id": "memory-group_12345-abc",
  "content": "用户789在上海工作",
  "context": "fact",
  "tags": ["user:789", "imported:semantic", "category:fact", "importance:0.7"],
  "timestamp": "2026-04-01T12:00:00+00:00",
  "metadata": {"speaker": "小明", "original_id": "abc"},
  "_source": { "...": "原始 LanceDB / SQLite 字段" }
}
```

### 2. 人工编辑

直接用文本编辑器编辑 JSONL：

- 修改 `content` / `tags` / `context` / `metadata` / `timestamp` 都是合法的。
- `bank_id` 一般不动；如果你要把记忆搬到别的 bank，改这一字段即可。
- `document_id` 是上传幂等键，保持稳定即可重跑不重复；改了相当于换文档。
- 想丢弃某条记忆，**直接删掉那一行**。
- `_source` 字段会被导入脚本忽略，留作改前对照。

### 3. 导入

```bash
# 先 dry-run 看一遍要发什么
python scripts/migrate_to_hindsight.py \
    --input data/memory_export.jsonl \
    --base-url https://nvli-hs-api.centaurea.dev \
    --api-key sk-xxxxxx \
    --dry-run

# 正式导入
python scripts/migrate_to_hindsight.py \
    --input data/memory_export.jsonl \
    --base-url https://nvli-hs-api.centaurea.dev \
    --api-key sk-xxxxxx
```

每个新见到的 `bank_id` 在第一次 retain 前自动 `PATCH retain_mission`。`document_id` 稳定，重复运行只 upsert，不会产生重复记忆。

## 与旧版的差异

- 旧版自带 LLM 抽取层（`extractor.py`）+ Embedding 调用 → **删除**。Hindsight 服务端自己抽事实，每轮少一次 LLM 调用。
- 旧版本地三库（`profiles.db` / `debug.db` / `semantic_lancedb/`）→ **删除**。
- 旧版 FastAPI 管理后台（`admin/`）+ `清除记忆` / `查看记忆` 命令 → **删除**。改用 Hindsight 自带 UI / `list_memories` / `delete_bank` 端点管理。
- 不再需要 `embedding_*` / `extract_*` 等环境变量。

## 调试技巧

- 启动日志：`simple-gpt: hindsight 插件已加载 (...)` + `simple-gpt: hindsight warmup 完成`。warmup 失败说明 base_url / api_key 有误。
- 首条消息后等 5–10 s，让 Hindsight 完成 retain 与 observation 抽取，再发追问消息验证 recall 注入。
- 在日志里搜 `simple-gpt: [hindsight] 已注入 N 条记忆` 可看到每次注入命中数与目标 bank。
