"""
examples/react_demo.py - ReAct 自主 Agent 演示脚本（v2.0）

不启动 Web UI，直接命令行体验 ResumeAgent 的自主工具调用能力：

    python -m examples.react_demo
    python examples/react_demo.py

会依次演示：
1. 评分计算任务：技能/经验/学历 → 加权得分 + 等级
2. 匹配报告导出：三维度分数 → 完整报告并导出 txt
3. 用户自定义任务（交互式输入）
"""

import sys
from pathlib import Path

# 允许直接以脚本方式运行（python examples/react_demo.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from core.agents.react_agent import ReActAgent, format_trace
from core.agents.tool_registry import builtin_tools
from core.llm import get_llm

DEMO_TASKS = [
    "技能匹配 80 分、经验匹配 60 分、学历匹配 90 分，"
    "计算加权综合得分并给出匹配等级和描述。",
    "技能 75、经验 85、学历 60，帮我生成一份完整的三维度匹配分析报告，"
    "并把报告导出为 txt 文件。",
]


def run_task(task: str, index: int = 0) -> None:
    """执行单个 ReAct 任务并打印轨迹。"""
    print("\n" + "=" * 70)
    print(f"任务 {index}: {task}")
    print("=" * 70)

    agent = ReActAgent(
        tools=builtin_tools(),
        llm=get_llm(),
        max_steps=8,
        agent_name="ResumeAgent 自主助手",
        verbose=True,
    )
    result = agent.run(task)

    print("\n" + "-" * 70)
    print(format_trace(result))
    print("-" * 70)


def main() -> None:
    logger.remove()  # 精简控制台日志
    logger.add(sys.stderr, level="WARNING")

    print("🧠 ResumeAgent ReAct 自主 Agent 演示")
    print(f"    可用工具: {[t.name for t in builtin_tools()]}")

    for i, task in enumerate(DEMO_TASKS, start=1):
        run_task(task, i)

    # 交互模式
    print("\n" + "=" * 70)
    print("交互模式（输入 exit 退出）")
    print("=" * 70)
    while True:
        task = input("\n请输入任务 > ").strip()
        if not task or task.lower() in ("exit", "quit", "q"):
            break
        run_task(task)

    print("\n演示结束。")


if __name__ == "__main__":
    main()
