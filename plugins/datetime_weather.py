from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

import httpx
from nonebot.log import logger

from ..plugin_config_inject import register_plugin_config_field
from ..plugin_system import (
    LLMRequestPayload,
    SimpleGPTPlugin,
    register_simple_gpt_plugin,
)

# 注册插件配置字段（在导入时执行，早于 Config 加载）
register_plugin_config_field(
    "simple_gpt_weather_api_key",
    str,
    default="",
    description="天气 API Key（高德地图 API），留空则不添加天气信息",
)
register_plugin_config_field(
    "simple_gpt_weather_city",
    int,
    default="",
    description="天气查询城市编码（高德地图城市编码），留空则不添加天气信息",
)
register_plugin_config_field(
    "simple_gpt_weather_timeout",
    float,
    default=5.0,
    description="天气 API 请求超时时间（秒）",
    gt=0,
)

if TYPE_CHECKING:
    from .. import Config


def _get_plugin_config() -> "Config":
    """延迟导入主配置，避免循环依赖。"""
    from .. import plugin_config
    return plugin_config


class DateTimeWeatherPlugin(SimpleGPTPlugin):
    """Add current datetime and weather information to the prompt."""

    priority = 200  # 更高优先级，确保在其他插件之前执行

    def __init__(self):
        # 延迟配置加载，避免循环导入
        self._config_loaded = False
        self.api_key = ""
        self.city = ""
        self.timeout = 5.0

        # 天气缓存
        self._weather_cache: Optional[str] = None
        self._cache_expire_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=15)  # 缓存15分钟

    def _ensure_config_loaded(self) -> None:
        """延迟加载配置，在第一次使用时执行。"""
        if self._config_loaded:
            return

        config = _get_plugin_config()
        self.api_key = config.simple_gpt_weather_api_key
        self.city = config.simple_gpt_weather_city
        self.timeout = config.simple_gpt_weather_timeout
        self._config_loaded = True

        logger.info(
            f"simple-gpt: datetime_weather 插件已加载 "
            f"(天气API: {'已配置' if self.api_key else '未配置'})"
        )

    async def before_llm_request(
        self, payload: LLMRequestPayload
    ) -> LLMRequestPayload:
        """Add datetime and weather info to the prompt."""

        # 确保配置已加载
        self._ensure_config_loaded()

        # 获取当前时间
        now = datetime.now()
        datetime_info = now.strftime("%Y年%m月%d日 %H:%M:%S %A")
        weekday_cn = self._get_chinese_weekday(now.weekday())
        datetime_str = f"{datetime_info.rsplit(' ', 1)[0]} 星期{weekday_cn}"

        # 获取天气信息
        weather_str = await self._get_weather()

        # 构造上下文信息
        context_info = f"[当前时间：{datetime_str}]"
        if weather_str:
            context_info += f" [天气：{weather_str}]"

        # 将时间和天气信息添加到 extra 中（供其他插件使用）
        payload.extra["datetime"] = datetime_str
        if weather_str:
            payload.extra["weather"] = weather_str

        # 将上下文信息添加到 prompt 的开头
        payload.prompt = f"{context_info}\n\n{payload.prompt}"

        logger.debug(f"simple-gpt: 已添加时间天气信息 - {context_info}")

        return payload

    def _get_chinese_weekday(self, weekday: int) -> str:
        """Convert weekday number to Chinese."""
        weekdays = ["一", "二", "三", "四", "五", "六", "日"]
        return weekdays[weekday]

    async def _get_weather(self) -> Optional[str]:
        """Fetch weather information from Amap API with caching."""
        if not self.api_key or not self.city:
            logger.debug("simple-gpt: 天气 API 配置未设置，跳过天气查询")
            return None

        # 检查缓存是否有效
        now = datetime.now()
        if (
            self._weather_cache is not None
            and self._cache_expire_time is not None
            and now < self._cache_expire_time
        ):
            logger.debug(
                f"simple-gpt: 使用缓存的天气数据 "
                f"(剩余 {int((self._cache_expire_time - now).total_seconds())} 秒过期)"
            )
            return self._weather_cache

        print(self.api_key, self.city)
        try:
            url = "https://restapi.amap.com/v3/weather/weatherInfo"
            params = {
                "key": self.api_key,
                "city": self.city,
                "extensions": "base",  # base=实况天气，all=预报天气
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if data.get("status") == "1" and data.get("lives"):
                    weather_data = data["lives"][0]
                    city = weather_data.get("city", "")
                    weather = weather_data.get("weather", "")
                    temperature = weather_data.get("temperature", "")
                    wind_direction = weather_data.get("winddirection", "")
                    wind_power = weather_data.get("windpower", "")
                    humidity = weather_data.get("humidity", "")

                    weather_str = f"{city} {weather} {temperature}℃"
                    if wind_direction and wind_power:
                        weather_str += f" {wind_direction}风{wind_power}级"
                    if humidity:
                        weather_str += f" 湿度{humidity}%"

                    # 更新缓存
                    self._weather_cache = weather_str
                    self._cache_expire_time = now + self._cache_duration

                    logger.info(
                        f"simple-gpt: 获取天气成功 - {weather_str} "
                        f"(缓存 {self._cache_duration.total_seconds() / 60:.0f} 分钟)"
                    )
                    return weather_str
                else:
                    logger.warning(f"simple-gpt: 天气 API 返回错误 - {data}")
                    return None

        except asyncio.TimeoutError:
            logger.warning("simple-gpt: 天气 API 请求超时")
            # 超时时如果有缓存，使用旧缓存
            if self._weather_cache is not None:
                logger.info("simple-gpt: 使用过期的缓存数据作为降级方案")
                return self._weather_cache
            return None
        except httpx.HTTPError as exc:
            logger.warning(f"simple-gpt: 天气 API 请求失败 - {exc}")
            # 请求失败时如果有缓存，使用旧缓存
            if self._weather_cache is not None:
                logger.info("simple-gpt: 使用过期的缓存数据作为降级方案")
                return self._weather_cache
            return None
        except Exception as exc:
            logger.exception(f"simple-gpt: 获取天气时发生未知错误 - {exc}")
            return None


register_simple_gpt_plugin(DateTimeWeatherPlugin())
