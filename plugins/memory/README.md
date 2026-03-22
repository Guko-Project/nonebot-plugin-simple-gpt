# Memory Plugin

nonebot-plugin-simple-gpt 的长期记忆插件。让 bot 能在对话中自动提取并记住重要信息，并在未来的对话中将相关记忆注入 prompt。

## 功能

- **自动记忆提取**：每次 bot 回复后，在后台调用 LLM 分析对话，提取值得记忆的信息
- **分层用户档案**（SQLite）：
  - **Global 层**：跨群通用属性，如生日、职业、固定偏好，以 QQ 号为唯一标识
  - **Group 层**：群内特定属性，如群内昵称，仅在对应群可见
- **语义记忆**（LanceDB）：对话中的事件、事实、指令，通过向量相似度搜索按需召回
- **记忆注入**：每次 LLM 请求前，自动将相关档案和语义记忆注入 prompt
- **非阻塞存储**：记忆提取在后台异步执行，不影响回复速度

## 数据流

```
用户消息
    ↓
before_llm_request (priority=250)
  ├─ 从 SQLite 读取 global + group 两层用户档案（合并，group 层覆盖 global 同名 key）
  ├─ 对当前消息生成 embedding，在 LanceDB 中搜索语义记忆
  └─ 将档案 + 语义记忆注入 prompt 头部
    ↓
LLM 回复 → 发送
    ↓
after_llm_response (后台异步)
  ├─ LLM 分析对话，提取 profiles（含 scope 判断）和 memories
  ├─ profiles → SQLite（scope=global 存 __global__，scope=group 存当前 session）
  └─ memories → embedding → LanceDB
```

## 存储结构

```
data/simple_gpt_memory/       # 默认路径，可通过配置修改
├── profiles.db               # SQLite，用户档案（两层）
└── semantic_lancedb/         # LanceDB，语义记忆向量库
```

### SQLite 表结构

```sql
CREATE TABLE user_profiles (
    user_id    TEXT NOT NULL,  -- QQ 号（global 层）或 QQ 号（group 层）
    session_id TEXT NOT NULL,  -- "__global__" 或 "group_{group_id}"
    key        TEXT NOT NULL,  -- 属性名，如 nickname、birthday、occupation
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, session_id, key)
);
```

### Profile scope 规则

| scope | 示例 key | 存储位置 | 跨群可见 |
|-------|---------|---------|---------|
| `global` | birthday、occupation、preference_food、skill | `__global__` | ✅ |
| `group` | nickname、群内角色 | `group_{id}` | ❌ |

由 AI 在提取时自动判断，默认为 `group`。

## 配置

在 `.env` 或 `.env.prod` 中添加：

```bash
# 启用插件（必填）
SIMPLE_GPT_MEMORY_ENABLED=true

# 数据存储目录（默认 data/simple_gpt_memory）
SIMPLE_GPT_MEMORY_DB_PATH=data/simple_gpt_memory

# Embedding API（留空则使用主 API 配置）
SIMPLE_GPT_MEMORY_EMBEDDING_API_KEY=
SIMPLE_GPT_MEMORY_EMBEDDING_API_BASE=
SIMPLE_GPT_MEMORY_EMBEDDING_MODEL=text-embedding-3-small
SIMPLE_GPT_MEMORY_EMBEDDING_DIMENSIONS=512

# 记忆提取 LLM（留空则使用主 API 配置）
SIMPLE_GPT_MEMORY_EXTRACT_API_KEY=
SIMPLE_GPT_MEMORY_EXTRACT_API_BASE=
SIMPLE_GPT_MEMORY_EXTRACT_MODEL=gemini-2.5-flash-lite

# 检索参数
SIMPLE_GPT_MEMORY_TOP_K=5          # 每次注入的语义记忆数量（1-20）
SIMPLE_GPT_MEMORY_SCOPE=group      # group（按群隔离）或 global（全局共享语义记忆）
```

### 配置说明

**Embedding API**：用于将文本转为向量，需要支持 `/embeddings` 接口的 OpenAI 兼容 API。留空时回退到主系统的 API Key 和 Base URL。

**提取模型**：用于从对话中提取记忆，对模型能力要求不高，推荐使用轻量模型（如 gemini-2.5-flash-lite）以降低成本。留空时回退到主系统配置。

**SCOPE**：仅影响语义记忆（LanceDB）的隔离粒度。用户档案（SQLite）始终保持两层架构，不受此配置影响。

## Prompt 注入格式

当有相关记忆时，在原始 prompt 前插入：

```
[长期记忆]
## 用户档案
- 123456（昵称: 小明）: occupation: 程序员, preference_food: 辣的
- 789012: birthday: 1月1日

## 相关记忆
- 小明上周提到最近在学 Rust
- 群里计划下个月一起打游戏
[记忆结束]

{原始 prompt}
```

## 依赖

```
lancedb>=0.27.1
```

（`pyarrow` 为 lancedb 的依赖，已在父项目中声明）

## 调试

开启 prompt 调试模式可查看实际注入的记忆内容：

```bash
SIMPLE_GPT_PROMPT_DEBUG=true
```
