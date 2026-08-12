"""
core/graph/agentic_node.py - 自主 Agent 节点（v2.0 新增）

基于自研 ReActAgent 的第 7 个业务节点：
用户输入任意自然语言任务，节点自主决定调用哪些工具
（评分计算 / RAG 检索 / 报告导出），完成 思考→行动→观察 循环。

作为 LangGraph 工作流的新分支节点：
- 入口：score 节点后 next_action == "agentic" 时进入
- 出口：执行完 ReAct 循环后回到 END（任务完成）

函数签名：def agentic_node(state: AgentState) -> dict
"""

from typing import Dict, List

from loguru import logger

from core.agents.react_agent import ReActAgent, format_trace
from core.agents.tool_registry import builtin_tools
from core.graph.agent_state import AgentState
from core.llm import get_llm

# 自主 Agent 系统提示：结合求职助手的领域上下文
AGENTIC_SYSTEM_PROMPT = """你是 {agent_name}，一名专业的求职辅助智能体。

背景：用户正在优化简历、准备面试。你可以调用内置工具辅助完成：
- calculate_weighted_score:   计算三维度匹配加权得分与等级
- score_to_grade:             分数转等级
- format_match_report:        生成完整匹配分析报告
- search_knowledge_base:      在简历/岗位知识库中检索资料
- export_report:              将报告文本导出为 .txt 文件

严格按格式交替输出（Thought / Action / Action Input），
收到 Observation 后继续，直到给出 Final Answer。最多 {max_steps} 步。"""


def agentic_node(state: AgentState) -> dict:
    """
    自主 Agent 节点：执行 ReAct 推理循环完成用户任务。

    从 state.agentic_task 读取任务，将执行结果写入:
    - agentic_result: 最终回答
    - agentic_trace:  推理轨迹（Thought/Action/Observation 列表）
    - current_node:   标记为 "agentic"

    Args:
        state: 当前全局状态

    Returns:
        dict: 部分状态更新
    """
    task = state.agentic_task.strip()
    logger.info(f"[Agentic] 自主 Agent 节点启动 | task={task[:80]}")

    if not task:
        logger.warning("[Agentic] 未提供任务描述，直接返回")
        return {
            "agentic_result": "未提供任务描述，请先输入你想让助手完成的任务。",
            "agentic_trace": [],
            "current_node": "agentic",
        }

    try:
        llm = get_llm()
        agent = ReActAgent(
            tools=builtin_tools(),
            llm=llm,
            max_steps=8,
            agent_name="ResumeAgent 自主助手",
            verbose=True,
        )
        result = agent.run(task, system_prompt=AGENTIC_SYSTEM_PROMPT)

        # 将 ReActStep 模型序列化为 dict 存入 state
        trace: List[Dict] = [
            step.model_dump() for step in result.steps
        ]

        logger.info(
            f"[Agentic] 自主任务完成 | steps={result.total_steps} "
            f"finished={result.finished} reason={result.stop_reason}"
        )
        if result.total_steps > 0:
            logger.debug(format_trace(result))

        return {
            "agentic_result": result.final_answer,
            "agentic_trace": trace,
            "current_node": "agentic",
        }
    except Exception as e:
        logger.error(f"[Agentic] 自主任务执行失败: {e}")
        return {
            "agentic_result": f"自主任务执行失败: {e}",
            "agentic_trace": [],
            "current_node": "agentic",
        }
