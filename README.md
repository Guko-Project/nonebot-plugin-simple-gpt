# [WIP] nonebot-plugin-simple-gpt

极速开发的，基于大模型的 NoneBot2 群聊对话插件，支持在群聊中通过艾特机器人或随机触发的方式获得自然语言回复。

## ✨ 功能
- 艾特机器人时，会读取该消息之前的连续 30 条群聊记录，与预设 Prompt 拼接后调用 OpenAI Chat Completions 接口生成回复。
- 支持为每个群聊维护独立的上下文，但是存在内存里比较蠢，不在群聊之间共享历史消息。
- 可配置随机回复概率，在未被艾特的情况下也能偶尔插入互动。
- 所有关键行为（Prompt 模板、模型、温度、超时、兜底回复等）都可以通过 `.env` 配置。
- 自动识别群聊消息中的图片，将其转换为 Base64 Data URL，一并作为上下文传入大模型实现“看图说话”。

## 📦 安装

项目基于 [uv](https://github.com/astral-sh/uv) 管理依赖，目前需要手动在 nonebot2 项目中添加相应缺失的依赖。

在本地开发阶段将本插件复制到 `src/plugins/` 目录即可。


## ⚙️ 配置

在 NoneBot2 项目的 `.env` 或 `.env.prod` 文件中新增以下配置项：

| 变量名 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `SIMPLE_GPT_API_KEY` | 是 | 无 | OpenAI 风格接口的 API Key |
| `SIMPLE_GPT_MODEL` | 是 | `gpt-4o-mini` | Chat Completions 模型名称 |
| `SIMPLE_GPT_API_BASE` | 是 | `https://api.openai.com/v1` | 接口基础地址，兼容自建或代理服务 |
| `SIMPLE_GPT_PROMPT_TEMPLATE` | 否 | 见下文 | Prompt 模板，支持 `{history}`、`{sender}`、`{latest_message}` 占位符 |
| `SIMPLE_GPT_HISTORY_LIMIT` | 否 | `30` | 参与上下文的历史消息条数上限（最多 50） |
| `SIMPLE_GPT_TEMPERATURE` | 否 | `0.7` | 生成温度 |
| `SIMPLE_GPT_MAX_TOKENS` | 否 | `2048` | 最大生成 Token 数 |
| `SIMPLE_GPT_TIMEOUT` | 否 | `300.0` | 请求超时时间（秒） |
| `SIMPLE_GPT_REPLY_PROBABILITY` | 否 | `0.03` | 随机触发概率，设为 `0` 表示只在艾特时回复 |
| `SIMPLE_GPT_FAILURE_REPLY` | 否 | `呜呜，暂时无法连接到大模型，请稍后再试呀。` | 调用失败时的兜底文本 |
| `SIMPLE_GPT_PROACTIVE_GROUP_WHITELIST` | 否 | 空 | 允许主动发言（随机插话）的群聊 ID，使用逗号分隔，例如 `123456,789012` |

默认 Prompt 模板：

```text
你是一个友善的中文群聊助手，需要结合最近的聊天记录进行自然对话。以下是群聊最近的消息：
{history}
请你用简体中文回复{sender}的最新发言：{latest_message}
```

## 🚀 使用

- 在群聊中艾特机器人即可触发智能回复，回复内容会基于最近 30 条群聊消息生成。
- 当 `SIMPLE_GPT_REPLY_PROBABILITY` 大于 0 时，机器人也可能在未被艾特的状态下随机插话；仅当当前群聊 ID 位于 `SIMPLE_GPT_PROACTIVE_GROUP_WHITELIST` 中时才会开启此行为。
- 如果未配置 `SIMPLE_GPT_API_KEY`，仅在被艾特时会返回兜底提示；随机回复会自动关闭。

## 🛠️ 开发提示

- 存在部分硬编码环节，请无视或者修复喵。
- 历史消息仅在内存中维护，不会写入持久化存储；重启后上下文自动清空。
- 如果需要自定义接口（如兼容 Azure/OpenAI 兼容服务），请调整 `SIMPLE_GPT_API_BASE` 与模型参数。
- 插件默认使用 OneBot v11 适配器的群聊事件，如需扩展到其他适配器，可在 `main.py` 中补充相应的事件类型判断。

欢迎在实际场景中按照需求修改 Prompt 与随机回复策略！
