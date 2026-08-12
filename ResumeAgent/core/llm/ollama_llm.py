"""
core/llm/ollama_llm.py - Ollama 本地模型实现

基于 langchain-ollama 封装，支持:
- 非流式 chat()
- 流式 chat() 逐 token 输出
- 自定义 temperature/top_p 等推理参数
- Function Calling（bind_tools，依赖模型原生工具支持，如 qwen2.5 / llama3.1+）

v2.0 新增:
- chat_with_tools(): 基于 ChatOllama.bind_tools() 实现真工具调用
"""

import json
from typing import Generator, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from loguru import logger

from config.settings import settings
from .base import BaseLLM, LLMConfig, LLMResponse, ChatMessage, ToolDefinition, ToolCall


class OllamaLLM(BaseLLM):
    """
    Ollama 本地大模型适配器。

    使用方式:
        config = LLMConfig(model="qwen2.5:7b", temperature=0.3)
        llm = OllamaLLM(config)
        resp = llm.chat([ChatMessage.user("你好")])
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        # 默认从 settings 填充配置
        if config is None:
            config = LLMConfig(
                model=settings.OLLAMA_MODEL,
                temperature=0.7,
                top_p=0.9,
                max_tokens=2048,
            )
        super().__init__(config)

        # 构建 langchain ChatOllama 实例
        self._client = ChatOllama(
            model=self.config.model,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            num_predict=self.config.max_tokens,
            stop=self.config.stop,
        )
        logger.info(f"OllamaLLM 初始化: model={self.config.model} base={settings.OLLAMA_BASE_URL}")

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

    def _chat_impl(self, messages: list[ChatMessage]) -> LLMResponse:
        """
        非流式调用 Ollama。

        Args:
            messages: 对话消息列表

        Returns:
            LLMResponse 统一响应对象
        """
        try:
            lc_msgs = self._to_langchain_messages(messages)
            response = self._client.invoke(lc_msgs)

            # 提取 token 用量（Ollama 在 response_metadata 中提供）
            usage = response.response_metadata or {}

            return LLMResponse(
                content=response.content,
                model=self.config.model,
                usage=usage,
                finish_reason=usage.get("done_reason", "stop"),
            )
        except Exception as e:
            logger.error(f"Ollama 非流式调用失败: {e}")
            return LLMResponse(
                content=f"[调用失败] {e}",
                model=self.config.model,
                finish_reason="error",
            )

    def _chat_with_tools_impl(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        tool_choice: Optional[str | dict],
    ) -> LLMResponse:
        """
        Function Calling 实现：基于 ChatOllama.bind_tools()。

        要求本地模型支持工具调用（qwen2.5 / llama3.1+ 均支持）。
        """
        try:
            lc_msgs = self._to_langchain_messages(messages)
            bound = self._client.bind_tools(
                [t.to_openai_tool() for t in tools],
                tool_choice=tool_choice,
            )
            response = bound.invoke(lc_msgs)

            content = getattr(response, "content", "") or ""
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                content = "".join(parts).strip()

            tool_calls: list[ToolCall] = []
            for tc in getattr(response, "tool_calls", []) or []:
                name = tc.get("name", "")
                raw_args = tc.get("args", {})
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except Exception:
                        raw_args = {}
                if not isinstance(raw_args, dict):
                    raw_args = {}
                tool_calls.append(
                    ToolCall(name=name, arguments=raw_args, id=tc.get("id", ""))
                )

            usage = response.response_metadata or {}
            finish_reason = "tool_calls" if tool_calls else usage.get("done_reason", "stop")

            return LLMResponse(
                content=content,
                model=self.config.model,
                usage=usage,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
            )
        except Exception as e:
            logger.error(f"Ollama Function Calling 失败: {e}")
            return LLMResponse(
                content=f"[Ollama 工具调用失败] {e}",
                model=self.config.model,
                finish_reason="error",
            )

    def _chat_stream_impl(self, messages: list[ChatMessage]) -> Generator[str, None, None]:
        """
        流式调用 Ollama，逐 chunk 返回增量文本。

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
            logger.debug(f"Ollama 流式调用完成: {len(full_text)} chars")
        except Exception as e:
            logger.error(f"Ollama 流式调用失败: {e}")
            yield f"\n[流式调用中断: {e}]"

    # ----------------------------------------------------------
    # 配置热更新
    # ----------------------------------------------------------

    def update_config(self, **kwargs) -> None:
        """
        动态更新推理参数，同步重建底层 ChatOllama 客户端。

        用法:
            llm.update_config(temperature=0.1, max_tokens=512)
        """
        super().update_config(**kwargs)
        # 重建客户端以应用新参数
        self._client = ChatOllama(
            model=self.config.model,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            num_predict=self.config.max_tokens,
            stop=self.config.stop,
        )
        logger.debug(f"OllamaLLM 配置已更新: {kwargs}")
