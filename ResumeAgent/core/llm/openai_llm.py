"""
core/llm/openai_llm.py - OpenAI / DeepSeek 云端 API 实现

基于 langchain-openai 封装，同时兼容 OpenAI 官方 API 和 DeepSeek API
（DeepSeek 采用 OpenAI 兼容接口格式）。

支持:
- 非流式 chat()
- 流式 chat() 逐 token 输出
- 自定义 temperature/top_p/max_tokens 等推理参数
"""

from typing import Generator, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from loguru import logger

from config.settings import settings
from .base import BaseLLM, LLMConfig, LLMResponse, ChatMessage


class OpenAILLM(BaseLLM):
    """
    OpenAI / DeepSeek 云端大模型适配器。

    根据 settings.LLM_PROVIDER 自动切换 API 端点和模型名。

    使用方式:
        # OpenAI 模式
        config = LLMConfig(model="gpt-4o", temperature=0.3)
        llm = OpenAILLM(config)
        resp = llm.chat([ChatMessage.user("你好")])

        # DeepSeek 模式
        # .env 中设置 LLM_PROVIDER=deepseek，自动切换
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        # 根据 LLM_PROVIDER 自动选择 API 配置
        provider = settings.LLM_PROVIDER

        if provider == "deepseek":
            api_key = settings.DEEPSEEK_API_KEY
            api_base = settings.DEEPSEEK_API_BASE
            default_model = settings.DEEPSEEK_MODEL
        else:  # openai
            api_key = settings.OPENAI_API_KEY
            api_base = settings.OPENAI_API_BASE
            default_model = settings.OPENAI_MODEL

        if config is None:
            config = LLMConfig(
                model=default_model,
                temperature=0.7,
                top_p=0.9,
                max_tokens=2048,
            )
        super().__init__(config)

        self._provider = provider
        self._api_key = api_key
        self._api_base = api_base

        # 构建 langchain ChatOpenAI 实例
        self._client = ChatOpenAI(
            model=self.config.model,
            api_key=self._api_key,
            base_url=self._api_base,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            stop=self.config.stop,
        )
        logger.info(f"OpenAILLM 初始化: provider={provider} model={self.config.model}")

    # ----------------------------------------------------------
    # 消息格式转换
    # ----------------------------------------------------------

    @staticmethod
    def _to_langchain_messages(messages: list[ChatMessage]) -> list:
        """
        将自定义 ChatMessage 列表转换为 LangChain 消息格式。

        Args:
            messages: 自定义消息列表

        Returns:
            LangChain BaseMessage 列表
        """
        lc_messages = []
        for msg in messages:
            if msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
            else:
                lc_messages.append(HumanMessage(content=msg.content))
        return lc_messages

    # ----------------------------------------------------------
    # 核心实现
    # ----------------------------------------------------------

    @staticmethod
    def _extract_content(response) -> str:
        """
        从 LangChain AIMessage 中稳健提取文本内容。

        兼容多种 API 返回格式：
        - response.content 为 str（标准情况）
        - response.content 为 list[dict]（通义千问多模态格式）
        - response.content 为 None/空（异常回退）
        """
        raw = getattr(response, "content", None)

        if raw is None:
            # 尝试从其他字段提取
            raw = getattr(response, "text", None) or ""

        if isinstance(raw, str):
            return raw.strip() or ""

        if isinstance(raw, list):
            # content 是 [{"type":"text","text":"..."}, ...] 格式
            parts = []
            for item in raw:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts).strip() or ""

        return str(raw).strip() or ""

    def _chat_impl(self, messages: list[ChatMessage]) -> LLMResponse:
        """
        非流式调用 OpenAI / DeepSeek API。

        Args:
            messages: 对话消息列表

        Returns:
            LLMResponse 统一响应对象
        """
        try:
            lc_msgs = self._to_langchain_messages(messages)
            response = self._client.invoke(lc_msgs)

            # 稳健提取文本内容
            content = self._extract_content(response)

            # 提取 token 用量和结束原因
            usage = {}
            finish_reason = "stop"
            if hasattr(response, "response_metadata"):
                meta = response.response_metadata
                token_info = meta.get("token_usage", {})
                usage = {
                    "token_usage": token_info,
                    "model_name": meta.get("model_name", self.config.model),
                }
                finish_reason = token_info.get(
                    "finish_reason",
                    meta.get("finish_reason", "stop"),
                )

            # 诊断日志：当内容为空时输出原始响应结构
            if not content:
                logger.warning(
                    f"{self._provider.upper()} 返回空内容 | "
                    f"response.type={type(response).__name__} | "
                    f"content.type={type(getattr(response, 'content', None)).__name__} | "
                    f"finish_reason={finish_reason} | "
                    f"metadata={usage}"
                )

            return LLMResponse(
                content=content or "",
                model=self.config.model,
                usage=usage,
                finish_reason=finish_reason,
            )
        except Exception as e:
            logger.error(f"{self._provider.upper()} 非流式调用失败: {e}")
            return LLMResponse(
                content=f"[{self._provider.upper()} 调用失败] {e}",
                model=self.config.model,
                finish_reason="error",
            )

    def _chat_stream_impl(self, messages: list[ChatMessage]) -> Generator[str, None, None]:
        """
        流式调用 OpenAI / DeepSeek API，逐 chunk 返回增量文本。

        Args:
            messages: 对话消息列表

        Yields:
            str: 每次 yield 一个文本增量
        """
        try:
            lc_msgs = self._to_langchain_messages(messages)
            full_text = ""
            for chunk in self._client.stream(lc_msgs):
                delta = chunk.content if hasattr(chunk, "content") else str(chunk)
                if delta:
                    full_text += delta
                    yield delta
            logger.debug(f"{self._provider.upper()} 流式调用完成: {len(full_text)} chars")
        except Exception as e:
            logger.error(f"{self._provider.upper()} 流式调用失败: {e}")
            yield f"\n[流式调用中断: {e}]"

    # ----------------------------------------------------------
    # 配置热更新
    # ----------------------------------------------------------

    def update_config(self, **kwargs) -> None:
        """
        动态更新推理参数，同步重建底层 ChatOpenAI 客户端。

        用法:
            llm.update_config(temperature=0.1, max_tokens=512)
        """
        super().update_config(**kwargs)
        self._client = ChatOpenAI(
            model=self.config.model,
            api_key=self._api_key,
            base_url=self._api_base,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            stop=self.config.stop,
        )
        logger.debug(f"{self._provider.upper()}LLM 配置已更新: {kwargs}")
