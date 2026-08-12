"""
core/llm/__init__.py - LLM 适配层统一入口

提供 LLM 工厂函数，根据 settings.LLM_PROVIDER 自动选择对应实现。

用法:
    from core.llm import get_llm, LLMConfig, ChatMessage

    config = LLMConfig(temperature=0.3, max_tokens=2048)
    llm = get_llm(config)
    resp = llm.chat_with_prompt("请介绍你自己")
    print(resp.text)
"""

from typing import Optional

from loguru import logger

from config.settings import settings
from .base import BaseLLM, LLMConfig, LLMResponse, ChatMessage
from .ollama_llm import OllamaLLM
from .openai_llm import OpenAILLM


def get_llm(config: Optional[LLMConfig] = None) -> BaseLLM:
    """
    LLM 工厂函数。
    根据 .env 中 LLM_PROVIDER 的值自动选择对应的 LLM 实现。

    Args:
        config: 推理参数配置，为 None 时使用默认值

    Returns:
        BaseLLM 实例（OllamaLLM 或 OpenAILLM）

    Raises:
        ValueError: 不支持的 LLM_PROVIDER

    用法:
        # 自动根据 settings 选择
        llm = get_llm()

        # 自定义参数
        llm = get_llm(LLMConfig(temperature=0.0, max_tokens=512))
    """
    provider = settings.LLM_PROVIDER

    if provider == "ollama":
        logger.info(f"工厂创建 OllamaLLM: model={settings.OLLAMA_MODEL}")
        return OllamaLLM(config)

    elif provider in ("openai", "deepseek"):
        logger.info(f"工厂创建 OpenAILLM: provider={provider}")
        return OpenAILLM(config)

    else:
        raise ValueError(
            f"不支持的 LLM_PROVIDER: '{provider}'，"
            f"可选值: ollama | openai | deepseek"
        )


# 便捷导出
__all__ = [
    "BaseLLM",
    "LLMConfig",
    "LLMResponse",
    "ChatMessage",
    "OllamaLLM",
    "OpenAILLM",
    "get_llm",
]
