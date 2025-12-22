# Simple-GPT 插件开发指南

## 概述

Simple-GPT 提供了一个灵活的插件系统，允许你在 LLM 请求前后修改 payload。每个插件都是独立的、可插拔的模块，并且**遵循 NoneBot 的标准配置加载逻辑**。

## 插件架构

### 核心组件

1. **SimpleGPTPlugin**: 插件基类
2. **LLMRequestPayload**: 请求前的数据载体
3. **LLMResponsePayload**: 响应后的数据载体
4. **PluginConfigRegistry**: 插件配置注入系统

### 配置加载流程

```
1. 插件导入 → 注册配置字段
2. 字段注入到主 Config 类
3. NoneBot 标准 get_plugin_config(Config) 加载所有配置
4. 插件从 plugin_config 访问自己的配置
```

### 数据流

```
用户消息 → before_llm_request → LLM API → after_llm_response → 发送回复
```

## 创建插件

### 1. 注册插件配置字段

在插件文件顶部注册配置字段（这会在导入时执行）：

```python
from ..plugin_config_inject import register_plugin_config_field

# 注册配置字段
register_plugin_config_field(
    "simple_gpt_my_plugin_api_key",  # 必须以 simple_gpt_ 开头
    str,                              # 字段类型
    default="",                       # 默认值
    description="我的插件 API Key",   # 描述
)

register_plugin_config_field(
    "simple_gpt_my_plugin_timeout",
    float,
    default=5.0,
    description="超时时间（秒）",
    gt=0,  # 验证规则：必须大于0
)
```

**重要说明：**
- 字段名必须以 `simple_gpt_` 开头
- 支持所有 Pydantic Field 参数：`default`, `ge`, `le`, `gt`, `lt`, `description` 等
- 配置会自动注入到主 Config 类，并通过 NoneBot 的 `get_plugin_config` 加载

### 2. 创建插件类

```python
from typing import TYPE_CHECKING, Optional
from nonebot.log import logger
from ..plugin_system import (
    LLMRequestPayload,
    LLMResponsePayload,
    SimpleGPTPlugin,
    register_simple_gpt_plugin,
)

if TYPE_CHECKING:
    from .. import Config


def _get_plugin_config() -> "Config":
    """延迟导入主配置，避免循环依赖。"""
    from .. import plugin_config
    return plugin_config


class MyPlugin(SimpleGPTPlugin):
    """我的插件描述。"""

    priority = 100  # 优先级，数字越大越先执行

    def __init__(self):
        # 初始化时不加载配置，避免循环导入
        self._config_loaded = False
        self.api_key = ""
        self.timeout = 5.0

    def _ensure_config_loaded(self) -> None:
        """延迟加载配置，在第一次使用时执行。"""
        if self._config_loaded:
            return

        config = _get_plugin_config()
        self.api_key = config.simple_gpt_my_plugin_api_key
        self.timeout = config.simple_gpt_my_plugin_timeout
        self._config_loaded = True

        logger.info(f"simple-gpt: my_plugin 已加载 (API Key: {'已配置' if self.api_key else '未配置'})")

    async def before_llm_request(
        self, payload: LLMRequestPayload
    ) -> LLMRequestPayload:
        """在调用 LLM 之前修改请求。"""

        # 确保配置已加载（第一次调用时加载）
        self._ensure_config_loaded()

        # 修改 prompt
        payload.prompt = f"[附加信息]\n\n{payload.prompt}"

        # 存储数据到 extra（供其他插件使用）
        payload.extra["my_data"] = "some_value"

        logger.debug("simple-gpt: my_plugin 已处理请求")
        return payload

    async def after_llm_response(
        self, payload: LLMResponsePayload
    ) -> LLMResponsePayload:
        """在收到 LLM 响应之后修改内容。"""

        # 修改响应内容
        payload.content = payload.content.strip()

        # 可以访问原始请求
        original_prompt = payload.request.prompt

        logger.debug("simple-gpt: my_plugin 已处理响应")
        return payload


# 注册插件
register_simple_gpt_plugin(MyPlugin())
```

### 3. 保存插件文件

将插件文件保存到 `plugins/` 目录：

```
nonebot-plugin-simple-gpt/
├── plugins/
│   ├── __init__.py          # 自动加载所有插件
│   ├── remove_think.py      # 内置插件示例
│   ├── datetime_weather.py  # 内置插件示例
│   └── my_plugin.py         # 你的插件 ← 放这里
```

插件会被自动加载，无需手动导入！

### 4. 配置插件

用户在 `.env` 或 `.env.prod` 文件中配置：

```bash
# 我的插件配置
SIMPLE_GPT_MY_PLUGIN_API_KEY=your_api_key_here
SIMPLE_GPT_MY_PLUGIN_TIMEOUT=10.0
```

配置会通过 NoneBot 的标准流程加载，与主插件配置完全一致！

## 配置注入系统详解

### 为什么需要配置注入？

- ✅ **保持插件独立性**：插件不修改核心 `Config` 类
- ✅ **遵循 NoneBot 规范**：使用标准的 `get_plugin_config` 加载
- ✅ **类型安全**：Pydantic 验证所有配置
- ✅ **易于维护**：添加/删除插件不影响核心代码

### 工作原理

1. **注册阶段**（插件导入时）：
   ```python
   # 插件文件被导入时，注册配置字段
   register_plugin_config_field("simple_gpt_xxx", ...)
   ```

2. **注入阶段**（Config 类定义后）：
   ```python
   # __init__.py 中
   inject_plugin_fields_to_config(Config)  # 将插件字段注入到 Config
   ```

3. **加载阶段**（NoneBot 标准流程）：
   ```python
   plugin_config = get_plugin_config(Config)  # NoneBot 加载所有配置
   ```

4. **使用阶段**（插件初始化）：
   ```python
   config = _get_plugin_config()  # 获取主配置
   self.api_key = config.simple_gpt_my_plugin_api_key  # 访问插件配置
   ```

### register_plugin_config_field API

```python
def register_plugin_config_field(
    field_name: str,       # 字段名（必须以 simple_gpt_ 开头）
    field_type: Type = str,  # 字段类型
    default: Any = "",     # 默认值
    description: str = "", # 描述
    **field_kwargs         # 其他 Pydantic Field 参数
) -> None:
```

**支持的 field_kwargs：**
- `ge`: 大于等于
- `le`: 小于等于
- `gt`: 大于
- `lt`: 小于
- `min_length`: 最小长度
- `max_length`: 最大长度
- `regex`: 正则表达式
- 等等（所有 Pydantic Field 参数）

## LLMRequestPayload 数据结构

```python
@dataclass
class LLMRequestPayload:
    prompt: str                      # 完整的 prompt 文本
    history: Sequence[HistoryEntry]  # 历史消息记录
    sender: str                      # 发送者名称
    latest_message: str              # 最新消息内容
    images: List[str]                # 图片 data URLs
    extra: Dict[str, Any]            # 额外数据（插件间共享）
```

## LLMResponsePayload 数据结构

```python
@dataclass
class LLMResponsePayload:
    content: str                # LLM 响应内容
    request: LLMRequestPayload  # 原始请求
    extra: Dict[str, Any]       # 额外数据
```

## 插件优先级

- 优先级范围：任意整数
- 数字越大，越先执行
- 建议范围：
  - 300+: 预处理（添加上下文信息）
  - 200: 内容增强（如 datetime_weather）
  - 100: 内容修改（如 remove_think）
  - 0-50: 后处理

## 完整示例：时间天气插件

参考 `plugins/datetime_weather.py` 查看完整实现。

**关键特性：**
1. ✅ 使用 `register_plugin_config_field` 注册 3 个配置项
2. ✅ 在 `__init__` 中从主配置读取字段
3. ✅ 异步 API 调用（高德地图天气）
4. ✅ 将数据存储到 `payload.extra`
5. ✅ 修改 prompt 添加时间天气信息

## 配置最佳实践

### ✅ 推荐：使用配置注入

```python
# 插件文件顶部
from ..plugin_config_inject import register_plugin_config_field

register_plugin_config_field(
    "simple_gpt_my_plugin_enabled",
    bool,
    default=True,
    description="是否启用插件",
)
```

### ✅ 命名规范

- 字段名：`simple_gpt_插件名_配置项`
- 环境变量：`SIMPLE_GPT_插件名_配置项`（大写）
- 示例：
  - 字段名：`simple_gpt_weather_api_key`
  - 环境变量：`SIMPLE_GPT_WEATHER_API_KEY`

### ❌ 不要做的事

- ❌ 不要直接修改 `__init__.py` 中的 `Config` 类
- ❌ 不要使用不以 `simple_gpt_` 开头的字段名
- ❌ 不要在插件中创建独立的配置类（应使用注入系统）

## 调试技巧

### 1. 启用 Prompt 调试模式

```bash
SIMPLE_GPT_PROMPT_DEBUG=true
```

查看插件修改后的完整 prompt。

### 2. 查看插件加载顺序

日志会显示：

```
simple-gpt: 插件加载顺序 -> DateTimeWeatherPlugin(priority=200), RemoveThinkTagPlugin(priority=100)
```

### 3. 检查配置注入

在 Python 中检查：

```python
from nonebot import get_plugin_config
from nonebot_plugin_simple_gpt import Config

config = get_plugin_config(Config)
print(config.simple_gpt_weather_api_key)  # 应该能访问插件配置
```

## 常见问题

### Q: 配置没有被注入？

A: 检查：
1. `register_plugin_config_field` 是否在文件顶部（导入时执行）
2. 插件文件是否在 `plugins/` 目录下
3. 字段名是否以 `simple_gpt_` 开头

### Q: 配置值没有生效？

A: 检查：
1. 环境变量名是否正确（大写，`SIMPLE_GPT_` 开头）
2. `.env` 文件是否被 NoneBot 加载
3. 配置类型是否匹配（字符串、数字、布尔值）

### Q: 为什么要延迟加载配置？

A: 避免循环导入！
- 插件在导入时就被实例化（`register_simple_gpt_plugin(MyPlugin())`）
- 但此时主配置 `plugin_config` 还未创建
- 所以必须在 `__init__` 中不加载配置，而是在第一次使用时（`_ensure_config_loaded()`）加载
- 这样可以避免 `ImportError: cannot import name 'plugin_config'` 错误

### Q: 多个插件如何共享数据？

A: 使用 `payload.extra` 字典：

```python
# 插件 A（priority=200）
payload.extra["weather"] = "晴天"

# 插件 B（priority=100）
weather = payload.extra.get("weather")  # "晴天"
```

### Q: 如何禁用某个插件？

A: 删除或重命名插件文件（如 `datetime_weather.py.disabled`）

### Q: 配置注入的执行顺序？

A:
1. 导入 `plugins` 模块（所有插件注册配置）
2. 执行 `inject_plugin_fields_to_config(Config)`
3. 执行 `get_plugin_config(Config)`（NoneBot 加载配置）
4. 插件初始化时访问配置

## 贡献你的插件

如果你创建了有用的插件，欢迎分享！

---

**Happy Coding! 🚀**
