"""
core/agents/ - 自研 Agent 能力层（v2.0 新增）

包含:
1. tool_registry.ReActTool   - 工具封装（现有纯函数 → 可被模型调用的工具）
2. tool_registry.builtin_tools() - 项目内置工具注册表
3. react_agent.ReActAgent    - 手写 Thought/Action/Observation 推理循环

设计要点:
- 与 LangGraph 解耦：ReActAgent 可独立单测、独立 Demo
- 纯函数工具优先：无副作用、参数简单、可安全重试
- 终止保障：步数上限 + 连续重复动作检测，防止死循环
"""

from .tool_registry import ReActTool, builtin_tools
from .react_agent import ReActAgent, ReActResult, ReActStep

__all__ = [
    "ReActTool",
    "builtin_tools",
    "ReActAgent",
    "ReActResult",
    "ReActStep",
]
