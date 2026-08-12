"""
core/llm/base.py - LLM 抽象基类

定义统一接口规范，所有 LLM 实现（Ollama/OpenAI/DeepSeek）必须继承此基类。

v2.0 新增 Function Calling 支持:
- ToolDefinition: 工具定义（name / description / parameters JSON Schema）
- ToolCall:      模型发起的工具调用（结构化参数）
- chat_with_tools(): 支持工具调用的对话接口（bind_tools）
- pydantic_model_to_tool(): 将 Pydantic 模型自动转为 ToolDefinition（结构化输出）
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Generator, Optional

from loguru import logger
from pydantic import BaseModel, Field


# ============================================================
# 配置与响应数据模型
# ============================================================

class LLMConfig(BaseModel):
    """
    LLM 推理参数配置。
    所有子类共享此配置模型，通过 settings 初始化默认值。
    """
    model: str = Field(default="", description="模型名称")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数，越高越随机")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="nucleus sampling")
    max_tokens: int = Field(default=2048, ge=1, description="最大输出 token 数")
    top_k: int = Field(default=40, ge=1, description="top-k sampling")
    repeat_penalty: float = Field(default=1.1, ge=1.0, description="重复惩罚系数")
    stop: Optional[list[str]] = Field(default=None, description="停止词列表")

    # 重试配置
    retry_times: int = Field(default=3, ge=0, description="失败后最大重试次数")
    retry_delay: float = Field(default=2.0, ge=0.5, description="重试间隔基数（秒）")
    request_timeout: float = Field(default=120.0, ge=10.0, description="单次请求超时（秒）")


class ToolDefinition(BaseModel):
    """
    工具定义（Function Calling 用）。

    用于向模型声明"可调用的函数"：
    - name:        工具名（模型在 tool_call 中引用）
    - description: 功能描述（模型据此判断何时调用）
    - parameters:  JSON Schema 格式的参数定义（含 properties）
    - required:    必填参数名列表
    """
    name: str = Field(default="", description="工具名称")
    description: str = Field(default="", description="工具功能描述")
    parameters: dict = Field(default_factory=dict, description="JSON Schema 参数定义")
    required: list[str] = Field(default_factory=list, description="必填参数名列表")

    def to_openai_tool(self) -> dict:
        """转换为 OpenAI 兼容的 tool 格式（Ollama/langchain 亦兼容）。"""
        params = dict(self.parameters)
        if "type" not in params:
            params["type"] = "object"
        if self.required:
            params["required"] = list(self.required)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


class ToolCall(BaseModel):
    """
    模型发起的工具调用。

    - name:      被调用的工具名
    - arguments: 工具参数（已解析为 dict）
    - id:        调用 ID（用于多轮工具调用追踪）
    """
    name: str = Field(default="", description="工具名称")
    arguments: dict = Field(default_factory=dict, description="工具参数 dict")
    id: str = Field(default="", description="调用 ID")

    @property
    def args(self) -> dict:
        """便捷属性：直接获取参数 dict"""
        return self.arguments


class LLMResponse(BaseModel):
    """
    LLM 返回结果统一封装。
    """
    content: str = Field(default="", description="模型回复文本")
    model: str = Field(default="", description="使用的模型名称")
    usage: dict = Field(default_factory=dict, description="token 用量信息")
    finish_reason: str = Field(default="stop", description="结束原因: stop | length | error | tool_calls")
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="模型发起的工具调用列表（Function Calling 时非空）",
    )

    @property
    def text(self) -> str:
        """便捷属性：直接获取回复文本"""
        return self.content

    @property
    def has_tool_calls(self) -> bool:
        """是否包含工具调用"""
        return len(self.tool_calls) > 0


# ============================================================
# 消息类型定义
# ============================================================

class ChatMessage(BaseModel):
    """单条对话消息"""
    role: str = Field(default="user", description="角色: system | user | assistant")
    content: str = Field(default="", description="消息内容")

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        return cls(role="assistant", content=content)


# ============================================================
# 工具定义辅助函数
# ============================================================

def pydantic_model_to_tool(
    model: type[BaseModel],
    name: str,
    description: str,
    exclude: Optional[list[str]] = None,
) -> ToolDefinition:
    """
    将 Pydantic 模型自动转换为 ToolDefinition（结构化输出神器）。

    用于"输出侧 Function Calling"：把模型应答字段定义为 Pydantic 模型，
    由 LLM 直接返回结构化 tool_call 参数，彻底告别 JSON 字符串解析。

    Args:
        model:       目标 Pydantic 模型类（如 ScoreResultSchema）
        name:        工具名
        description: 工具描述（说明要生成的结构）
        exclude:     需要排除的字段名列表

    Returns:
        ToolDefinition: 可直接传给 chat_with_tools()
    """
    schema = model.model_json_schema()
    properties = dict(schema.get("properties", {}))
    if exclude:
        for key in exclude:
            properties.pop(key, None)

    # 去掉 pydantic 自动附加的 title 字段，减小 prompt 体积
    for prop in properties.values():
        if isinstance(prop, dict):
            prop.pop("title", None)

    required = [k for k in schema.get("required", []) if k in properties]

    return ToolDefinition(
        name=name,
        description=description,
        parameters={"type": "object", "properties": properties},
        required=required,
    )


def json_safe_loads(raw: str) -> dict:
    """
    稳健解析 JSON 字符串，兼容 ```json 代码块包裹。

    Args:
        raw: 原始字符串（可能包含 ```json 围栏或额外说明文本）

    Returns:
        dict: 解析后的对象；失败返回 {}
    """
    if not raw:
        return {}
    text = raw.strip()
    # 剥离代码块围栏
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    # 截取第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ============================================================
# 抽象基类
# ============================================================

class LLMError(Exception):
    """LLM 调用异常基类。"""
    pass


class LLMConnectionError(LLMError):
    """LLM 连接异常（网络/服务不可达）。"""
    pass


class LLMTimeoutError(LLMError):
    """LLM 请求超时。"""
    pass


class LLMResponseError(LLMError):
    """LLM 返回异常（格式错误/内容异常）。"""
    pass


class BaseLLM(ABC):
    """
    LLM 抽象基类。
    所有具体 LLM 实现必须实现:
      - _chat_impl():        非流式调用
      - _chat_stream_impl(): 流式调用
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Args:
            config: LLM 推理参数，为 None 时使用默认值
        """
        self.config = config or LLMConfig()
        self._call_count = 0  # 累计调用次数统计

    # ----------------------------------------------------------
    # 子类必须实现的方法
    # ----------------------------------------------------------

    @abstractmethod
    def _chat_impl(self, messages: list[ChatMessage]) -> LLMResponse:
        """
        非流式对话实现（子类重写）。

        Args:
            messages: 对话消息列表

        Returns:
            LLMResponse: 统一响应
        """
        ...

    @abstractmethod
    def _chat_stream_impl(self, messages: list[ChatMessage]) -> Generator[str, None, None]:
        """
        流式对话实现（子类重写）。
        每个 yield 返回一个增量文本块。

        Args:
            messages: 对话消息列表

        Yields:
            str: 增量文本片段
        """
        ...

    # ----------------------------------------------------------
    # 重试逻辑
    # ----------------------------------------------------------

    def _retryable_chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """
        带重试的非流式对话。

        使用指数退避策略：延迟 = retry_delay * (2 ** (attempt - 1))
        特别注意：内容为空时允许重试（可能是 API 瞬时异常）。

        Args:
            messages: 对话消息列表

        Returns:
            LLMResponse

        Raises:
            LLMError: 所有重试均失败后抛出
        """
        max_retries = self.config.retry_times
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                self._call_count += 1
                result = self._chat_impl(messages)

                # 验证返回内容不为空（工具调用场景下 content 可能为空但 tool_calls 非空）
                if (not result.content or len(result.content.strip()) == 0) \
                        and not result.has_tool_calls:
                    raise LLMResponseError(
                        f"LLM 返回空内容 (attempt {attempt + 1}/{max_retries + 1}, "
                        f"finish_reason={result.finish_reason})"
                    )

                return result

            except LLMResponseError as e:
                # 空内容错误：允许重试，可能是 API 瞬时异常
                last_error = e
                if attempt < max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM 返回空内容(重试 {attempt + 1}/{max_retries})，"
                        f"{delay:.1f}s 后重试 | {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"LLM 返回空内容，已达最大重试次数 {max_retries}"
                    )
                    raise
            except LLMError:
                raise  # 其他自定义异常直接抛出，不重试
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM 调用失败(重试 {attempt + 1}/{max_retries})，"
                        f"{delay:.1f}s 后重试 | 错误: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"LLM 调用失败，已达最大重试次数 {max_retries} | 最终错误: {e}"
                    )

        raise LLMError(
            f"LLM 调用失败（已重试 {max_retries} 次）: {last_error}"
        ) from last_error

    # ----------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        stream: bool = False,
        with_retry: bool = True,
    ) -> LLMResponse | Generator[str, None, None]:
        """
        统一对话接口。

        Args:
            messages:   对话消息列表
            stream:     True 返回生成器（流式），False 返回 LLMResponse
            with_retry: 是否启用自动重试（仅非流式模式生效）

        Returns:
            - stream=False: LLMResponse
            - stream=True: Generator yielding str chunks
        """
        logger.debug(
            f"LLM({self.model_name}) chat | stream={stream} | "
            f"messages={len(messages)} | total_calls={self._call_count}"
        )

        if stream:
            self._call_count += 1
            return self._chat_stream_impl(messages)
        elif with_retry:
            return self._retryable_chat(messages)
        else:
            self._call_count += 1
            return self._chat_impl(messages)

    def chat_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        tool_choice: Optional[str | dict] = None,
        with_retry: bool = True,
    ) -> LLMResponse:
        """
        支持 Function Calling 的对话接口。

        Args:
            messages:     对话消息列表
            tools:        工具定义列表（模型可调用的函数）
            tool_choice:  工具选择策略：
                          - None:            由模型自主决定（auto）
                          - "required":      强制模型必须调用工具（结构化输出）
                          - {"type":"function","function":{"name":...}}:
                                            强制调用指定工具
            with_retry:   是否启用自动重试

        Returns:
            LLMResponse: 若 has_tool_calls 为 True，则 tool_calls 携带结构化参数

        注意：默认实现（子类未重写 _chat_with_tools_impl）会回退为普通 chat，
        即工具描述不会生效。OpenAILLM / OllamaLLM 已实现真正的工具调用。
        """
        if not tools:
            return self.chat(messages, with_retry=with_retry)

        if with_retry:
            return self._retryable_chat_with_tools(messages, tools, tool_choice)
        self._call_count += 1
        return self._chat_with_tools_impl(messages, tools, tool_choice)

    def _chat_with_tools_impl(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        tool_choice: Optional[str | dict],
    ) -> LLMResponse:
        """
        工具调用实现（子类重写）。

        默认实现：回退为普通 chat（工具不会真正生效，但保证接口不崩溃）。
        """
        logger.warning(
            f"{type(self).__name__} 未实现真正的 Function Calling，"
            f"chat_with_tools 回退为普通 chat（tools={[t.name for t in tools]}）"
        )
        return self._chat_impl(messages)

    def _retryable_chat_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        tool_choice: Optional[str | dict],
    ) -> LLMResponse:
        """
        带重试的工具调用对话。
        逻辑与 _retryable_chat 一致，但空内容校验兼容 tool_calls。
        """
        max_retries = self.config.retry_times
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                self._call_count += 1
                result = self._chat_with_tools_impl(messages, tools, tool_choice)

                # 空内容校验（含 tool_calls 视为有效）
                if (not result.content or len(result.content.strip()) == 0) \
                        and not result.has_tool_calls:
                    raise LLMResponseError(
                        f"LLM 返回空内容 (attempt {attempt + 1}/{max_retries + 1}, "
                        f"finish_reason={result.finish_reason})"
                    )
                return result

            except LLMResponseError as e:
                last_error = e
                if attempt < max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM 工具调用返回空内容(重试 {attempt + 1}/{max_retries})，"
                        f"{delay:.1f}s 后重试 | {e}"
                    )
                    time.sleep(delay)
                else:
                    raise
            except LLMError:
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM 工具调用失败(重试 {attempt + 1}/{max_retries})，"
                        f"{delay:.1f}s 后重试 | 错误: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"LLM 工具调用失败，已达最大重试次数 {max_retries} | 最终错误: {e}"
                    )

        raise LLMError(
            f"LLM 工具调用失败（已重试 {max_retries} 次）: {last_error}"
        ) from last_error

    def chat_with_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        with_retry: bool = True,
    ) -> LLMResponse | Generator[str, None, None]:
        """
        简化的单轮对话接口。

        Args:
            prompt:       用户输入
            system_prompt: 系统提示（可选）
            stream:       是否流式输出
            with_retry:   是否启用自动重试

        Returns:
            LLMResponse 或 Generator
        """
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage.system(system_prompt))
        messages.append(ChatMessage.user(prompt))
        return self.chat(messages, stream=stream, with_retry=with_retry)

    def update_config(self, **kwargs) -> None:
        """
        动态更新推理参数。

        用法:
            llm.update_config(temperature=0.3, max_tokens=4096)
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    @property
    def model_name(self) -> str:
        """当前使用的模型名称"""
        return self.config.model

    @property
    def total_calls(self) -> int:
        """累计 LLM 调用次数。"""
        return self._call_count
