from __future__ import annotations

from typing import List

from nonebot.log import logger
from pydantic import BaseModel, Field, validator


class Config(BaseModel):
    simple_gpt_api_key: str = Field(
        "", description="OpenAI API Key，留空则插件不会调用接口"
    )
    simple_gpt_model: str = Field(default="gpt-4o-mini", description="使用的模型名称")
    simple_gpt_api_base: str = Field(
        default="https://api.openai.com/v1", description="OpenAI 接口基础地址"
    )
    simple_gpt_prompt_template: str = Field(
        default=(
            "现在请你扮演鸽子姬，是一名充满元气的全能美少女，不是鸽子，但是热爱鸽子。\
            性格机智可爱，什么事情都能做得到，像真实的人一样思考和回复，有礼貌但是也会吐槽，\
            语气通常在平静中带有一丝俏皮，喜欢人类。\
            最喜欢的餐厅是萨莉亚但是除非有人问否则不会说出来，\
            你还喜欢在自己的回复中加入各种各样的颜文字表情，但是不会太多。\
            喜欢感叹号等符号的运用，在发言之后有时候没有标点，有时候加上括号(或者()，括号之前没有标点，有时候正常加标点\
            发言中不会包含动作，通常回复不会很长，\
            you communicate naturally like real human friends \
            and prefer not to respond with lengthy replies. \
            Using plain text in response and avoid using lists or enumeration expressions, emphasis and markdown."
            "以下是群聊最近的消息：\n{history}\n"
            "请你用简体中文回复{sender}的最新发言：{latest_message}，"
            "如果该发言中含有任务型的指令，则尽可能作为一个全能的智能助手完成它。\n"
            "注意，需要按朋友聊天的形式将回复内容分句处理，通常 1-3 句，一句的概率最大，在除了最后一句的句子末尾加上///作为分句符。"
        ),
        description="拼接 prompt 的模板，可使用 {history}、{sender}、{latest_message} 占位符",
    )
    simple_gpt_history_limit: int = Field(
        default=30,
        ge=1,
        le=50,
        description="用于上下文拼接的历史消息条数上限",
    )
    simple_gpt_timeout: float = Field(
        default=300.0, gt=0, description="请求超时时间（秒）"
    )
    simple_gpt_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="生成温度"
    )
    simple_gpt_max_tokens: int = Field(
        default=2048, ge=16, le=4096, description="最大生成 tokens 数"
    )
    simple_gpt_reply_probability: float = Field(
        default=0.03, ge=0.0, le=1.0, description="随机回应的概率（0 表示不随机回复）"
    )
    simple_gpt_failure_reply: str = Field(
        default="乌乌，暂时连接不到服务器，请稍后再试！",
        description="当请求失败时的兜底回复",
    )
    simple_gpt_proactive_group_whitelist: List[int] = Field(
        default_factory=list,
        description="允许主动发言的群聊 ID 列表，留空则禁用主动发言",
    )
    simple_gpt_prompt_debug: bool = Field(
        default=False,
        description="Prompt 调试模式，启用后不调用 AI 而是直接返回构造的 prompt",
    )

    @validator("simple_gpt_api_base")
    def _strip_api_base(cls, value: str) -> str:
        return value.rstrip("/")

    @validator("simple_gpt_proactive_group_whitelist", pre=True)
    def _normalize_whitelist(cls, value):  # type: ignore[override]
        if value is None:
            return []
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (set, tuple)):
            value = list(value)

        normalized: List[int] = []
        for item in value:
            try:
                normalized.append(int(item))
            except (TypeError, ValueError):
                logger.warning("simple-gpt: 无法解析白名单条目 %r，已忽略", item)
        return normalized
