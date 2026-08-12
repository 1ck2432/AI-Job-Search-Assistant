"""
core/agents/react_agent.py - 自研 ReAct 推理引擎（v2.0）

手写 Thought → Action → Observation → Thought 循环，不依赖任何 Agent 框架：

    System(工具清单+格式规则) → LLM 输出 Thought/Action/Action Input
    → 本地执行工具 → Observation 回填 → 再问 LLM → ... → Final Answer

特色（面试亮点）:
- 跨模型通用：纯文本协议，不要求模型原生支持 tool_calls
- 终止保障：步数上限 + 连续重复动作检测，杜绝死循环
- 可观测：完整推理轨迹（steps）暴露给上层/UI 展示
- 与 LangGraph 解耦：可独立单测、独立 Demo
"""

import json
import re
from dataclasses import field
from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from core.llm import get_llm
from core.llm.base import ChatMessage
from .tool_registry import ReActTool


# ============================================================
# 数据模型
# ============================================================

class ReActStep(BaseModel):
    """单步推理轨迹"""
    thought: str = Field(default="", description="Thought 推理内容")
    action: str = Field(default="", description="Action 工具名")
    action_input: dict = Field(default_factory=dict, description="Action Input 参数")
    observation: str = Field(default="", description="Observation 工具执行结果")
    step_index: int = Field(default=0, description="步序号")


class ReActResult(BaseModel):
    """ReAct 循环最终结果"""
    final_answer: str = Field(default="", description="最终回答")
    steps: List[ReActStep] = Field(default_factory=list, description="完整推理轨迹")
    total_steps: int = Field(default=0, description="实际步数")
    finished: bool = Field(default=False, description="是否正常完成")
    stop_reason: str = Field(
        default="",
        description="终止原因: final_answer | max_steps | repeat_detected | no_action | error",
    )


# ============================================================
# ReAct 引擎
# ============================================================

# 解析正则（兼容中英文冒号、多行内容）
_FINAL_RE = re.compile(
    r"Final\s*Answer\s*[:：]\s*(.+?)(?=Thought\s*[:：]|$)", re.S | re.I
)
_THOUGHT_RE = re.compile(
    r"Thought\s*[:：]\s*(.+?)(?=Action\s*[:：]|Final\s*Answer\s*[:：]|$)", re.S | re.I
)
_ACTION_RE = re.compile(r"Action\s*[:：]\s*(\w+)", re.I)
_ACTION_INPUT_RE = re.compile(
    r"Action\s*Input\s*[:：]\s*(\{.*\}|\[.*\]|\S+)", re.S | re.I
)

DEFAULT_SYSTEM_PROMPT = """你是 {agent_name}，一个可以自主调用工具完成任务的智能体。

你有以下工具可用：
{tool_descriptions}

严格按以下格式交替输出（每轮只输出一种）：

1. 需要调用工具时：
Thought: 分析当前情况，决定下一步
Action: 要调用的工具名（必须是上述列出的工具之一）
Action Input: 传给工具的 JSON 参数（必须是合法 JSON 对象）

2. 得到 Observation（工具执行结果）后继续上述格式，直到可以回答：

Thought: 我现在可以回答用户了
Final Answer: 最终答案

要求：
- 每个回合只输出一个 Action
- 不要编造参数，参数必须来自用户问题或 Observation
- 工具名必须精确匹配
- 最多 {max_steps} 步内完成任务"""


class ReActAgent:
    """
    手写 ReAct 推理引擎。

    Args:
        tools:        可用工具列表（ReActTool）
        llm:          LLM 实例（默认 get_llm()）
        max_steps:    最大推理步数（含工具调用），默认 8
        max_repeat:   连续重复动作次数达到该值时强制终止，默认 2
        agent_name:   提示词中的智能体名称
        verbose:      是否打印每步轨迹到控制台
    """

    def __init__(
        self,
        tools: List[ReActTool],
        llm=None,
        max_steps: int = 8,
        max_repeat: int = 2,
        agent_name: str = "ResumeAgent 自主助手",
        verbose: bool = False,
    ):
        self.tools: Dict[str, ReActTool] = {t.name: t for t in tools}
        self.llm = llm or get_llm()
        self.max_steps = max_steps
        self.max_repeat = max_repeat
        self.agent_name = agent_name
        self.verbose = verbose

    # --------------------------------------------------------
    # 公开接口
    # --------------------------------------------------------

    def run(
        self,
        task: str,
        system_prompt: Optional[str] = None,
    ) -> ReActResult:
        """
        执行一次 ReAct 推理循环。

        Args:
            task:          用户任务描述
            system_prompt: 自定义系统提示（默认内置模板）

        Returns:
            ReActResult: 最终回答 + 完整推理轨迹
        """
        steps: List[ReActStep] = []
        history: List[ChatMessage] = []
        sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        last_key: Optional[tuple] = None
        repeat_count = 0

        for step in range(self.max_steps):
            messages = [
                ChatMessage.system(self._render_system_prompt(sys_prompt)),
                ChatMessage.user(task),
                *history,
            ]

            try:
                resp = self.llm.chat(messages, with_retry=False)
            except Exception as e:
                logger.error(f"ReAct LLM 调用失败: {e}")
                return self._finish(
                    steps, final_answer=f"[LLM 调用失败] {e}", reason="error"
                )

            text = resp.text.strip()
            if not text:
                return self._finish(
                    steps, final_answer="LLM 返回空内容，任务中断", reason="no_action"
                )

            thought, action, action_input, final_answer = self._parse_response(text)
            if self.verbose:
                logger.info(f"[step {step + 1}] Thought: {thought}")

            # 命中 Final Answer
            if final_answer:
                if self.verbose:
                    logger.info(f"[step {step + 1}] Final Answer: {final_answer}")
                return self._finish(
                    steps,
                    final_answer=final_answer,
                    reason="final_answer",
                    step=step + 1,
                )

            # 未给出 Action
            if not action:
                history.append(ChatMessage.assistant(text))
                history.append(ChatMessage.user(
                    "你没有输出有效的 Action，请按格式输出："
                    "Thought: ...\nAction: 工具名\nAction Input: {...}"
                ))
                continue

            # 工具不存在
            if action not in self.tools:
                observation = (
                    f"[未知工具: {action}] 可用工具: {', '.join(self.tools.keys())}"
                )
            else:
                observation = self.tools[action].execute(**action_input)

            # 连续重复动作检测
            key = (action, json.dumps(action_input, sort_keys=True, ensure_ascii=False))
            if key == last_key:
                repeat_count += 1
                if repeat_count >= self.max_repeat:
                    logger.warning(
                        f"ReAct 检测到连续重复动作 [{action}] {self.max_repeat} 次，强制终止"
                    )
                    return self._finish(
                        steps,
                        final_answer="检测到连续重复动作，提前终止。"
                                    f"已完成 {len(steps)} 步推理。",
                        reason="repeat_detected",
                        step=step + 1,
                    )
            else:
                repeat_count = 0
                last_key = key

            step_record = ReActStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                step_index=step + 1,
            )
            steps.append(step_record)
            if self.verbose:
                logger.info(
                    f"[step {step + 1}] Action: {action} | Input: {action_input}\n"
                    f"    Observation: {observation[:150]}"
                )

            history.append(ChatMessage.assistant(text))
            history.append(ChatMessage.user(
                f"Observation: {observation}\n\n"
                "请根据 Observation 决定下一步（Thought → Action/Action Input），"
                "或直接输出 Final Answer。"
            ))

        return self._finish(
            steps,
            final_answer="已达最大推理步数，未得到最终答案。"
                        f"已完成 {len(steps)} 步推理。",
            reason="max_steps",
            step=self.max_steps,
        )

    # --------------------------------------------------------
    # 内部实现
    # --------------------------------------------------------

    def _render_system_prompt(self, template: str) -> str:
        """渲染系统提示（工具清单 + 参数）。"""
        tool_desc = "\n".join(t.describe() for t in self.tools.values())
        return template.format(
            agent_name=self.agent_name,
            tool_descriptions=tool_desc,
            max_steps=self.max_steps,
        )

    def _parse_response(
        self,
        text: str,
    ) -> tuple[str, str, dict, Optional[str]]:
        """
        解析 LLM 输出。

        Returns:
            (thought, action, action_input, final_answer)
            未命中对应字段时为空值。
        """
        thought_match = _THOUGHT_RE.search(text)
        thought = thought_match.group(1).strip() if thought_match else ""

        final_match = _FINAL_RE.search(text)
        final_answer = final_match.group(1).strip() if final_match else ""

        action_match = _ACTION_RE.search(text)
        action = action_match.group(1).strip() if action_match else ""

        action_input: dict = {}
        if action:
            input_match = _ACTION_INPUT_RE.search(text)
            if input_match:
                raw = input_match.group(1).strip()
                action_input = self._parse_action_input(raw)

        return thought, action, action_input, final_answer

    @staticmethod
    def _parse_action_input(raw: str) -> dict:
        """解析 Action Input：优先 JSON，其次兼容 'key: value' 多行格式。"""
        # 1) JSON 对象
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # 2) 兼容单引号 JSON
        try:
            data = json.loads(raw.replace("'", '"'))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # 3) key: value 多行格式
        result: Dict[str, str] = {}
        for line in raw.splitlines():
            line = line.strip().strip(",")
            if ":" in line or "：" in line:
                parts = re.split(r"[:：]", line, maxsplit=1)
                if len(parts) == 2:
                    key = parts[0].strip().strip('"').strip("'")
                    value = parts[1].strip().strip('"').strip("'")
                    if key:
                        result[key] = value
        return result

    def _finish(
        self,
        steps: List[ReActStep],
        final_answer: str,
        reason: str,
        step: int = 0,
    ) -> ReActResult:
        """构造最终结果。"""
        return ReActResult(
            final_answer=final_answer.strip(),
            steps=steps,
            total_steps=step,
            finished=(reason == "final_answer"),
            stop_reason=reason,
        )


def format_trace(result: ReActResult) -> str:
    """
    将 ReActResult 推理轨迹格式化为可读文本（用于 UI/日志展示）。

    Args:
        result: ReAct 推理结果

    Returns:
        str: 带序号与缩进的轨迹文本
    """
    lines = [f"[ReAct 轨迹] 共 {result.total_steps} 步，终止原因: {result.stop_reason}"]
    for s in result.steps:
        lines.append(f"\nStep {s.step_index}:")
        lines.append(f"  Thought    : {s.thought}")
        lines.append(f"  Action     : {s.action}")
        lines.append(f"  Action Input: {json.dumps(s.action_input, ensure_ascii=False)}")
        obs = s.observation.replace("\n", "\n    ")
        lines.append(f"  Observation: {obs}")
    lines.append(f"\nFinal Answer:\n{result.final_answer}")
    return "\n".join(lines)
