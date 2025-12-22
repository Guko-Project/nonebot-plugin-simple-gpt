"""
插件配置注入系统。

允许插件动态地向主 Config 类注入配置字段，
同时保持与 NoneBot 配置加载逻辑的一致性。
"""

from __future__ import annotations

from typing import Any, Dict, Type

from pydantic import Field
from pydantic.fields import FieldInfo


class PluginConfigRegistry:
    """插件配置注册表，用于收集和注入插件配置项。"""

    _fields: Dict[str, tuple[Any, FieldInfo]] = {}

    @classmethod
    def register_field(
        cls,
        field_name: str,
        field_type: Type[Any],
        field_info: FieldInfo,
    ) -> None:
        """
        注册一个配置字段。

        Args:
            field_name: 字段名称（应以 simple_gpt_ 开头）
            field_type: 字段类型
            field_info: Pydantic FieldInfo 对象
        """
        if not field_name.startswith("simple_gpt_"):
            raise ValueError(
                f"插件配置字段名必须以 'simple_gpt_' 开头，收到: {field_name}"
            )

        cls._fields[field_name] = (field_type, field_info)

    @classmethod
    def get_all_fields(cls) -> Dict[str, tuple[Any, FieldInfo]]:
        """获取所有注册的配置字段。"""
        return cls._fields.copy()

    @classmethod
    def clear(cls) -> None:
        """清空所有注册的字段（主要用于测试）。"""
        cls._fields.clear()


def register_plugin_config_field(
    field_name: str,
    field_type: Type[Any] = str,
    default: Any = "",
    description: str = "",
    **field_kwargs: Any,
) -> None:
    """
    便捷函数：注册插件配置字段。

    Args:
        field_name: 字段名称（必须以 simple_gpt_ 开头）
        field_type: 字段类型，默认 str
        default: 默认值
        description: 字段描述
        **field_kwargs: 其他 Field 参数（如 ge, le, gt, lt 等）

    Example:
        ```python
        # 在插件文件顶部（导入之后）
        from ..plugin_config_inject import register_plugin_config_field

        register_plugin_config_field(
            "simple_gpt_weather_api_key",
            str,
            default="",
            description="天气 API Key（高德地图 API）",
        )
        register_plugin_config_field(
            "simple_gpt_weather_timeout",
            float,
            default=5.0,
            description="天气 API 超时时间（秒）",
            gt=0,
        )
        ```
    """
    field_info = Field(default=default, description=description, **field_kwargs)
    PluginConfigRegistry.register_field(field_name, field_type, field_info)


def inject_plugin_fields_to_config(config_class: Type[Any]) -> Type[Any]:
    """
    将注册的插件配置字段注入到 Config 类中。

    Args:
        config_class: 主 Config 类

    Returns:
        更新后的 Config 类
    """
    plugin_fields = PluginConfigRegistry.get_all_fields()

    for field_name, (field_type, field_info) in plugin_fields.items():
        # 检查字段是否已存在于模型字段中
        if field_name not in config_class.model_fields:
            # 更新 __annotations__ - 这是让 Pydantic 识别字段类型的关键
            if not hasattr(config_class, "__annotations__"):
                config_class.__annotations__ = {}
            config_class.__annotations__[field_name] = field_type

            # 设置 FieldInfo 的 annotation 属性（关键步骤）
            field_info.annotation = field_type

            # 直接将 FieldInfo 添加到 model_fields
            config_class.model_fields[field_name] = field_info

    # 重建 Pydantic 模型（确保新字段被识别）
    if plugin_fields:
        config_class.model_rebuild(force=True)

    return config_class
