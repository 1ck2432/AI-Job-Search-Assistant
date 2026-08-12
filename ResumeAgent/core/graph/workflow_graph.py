"""
core/graph/workflow_graph.py - LangGraph 多智能体工作流编排

完整流转逻辑：

    START
      │
      ▼
  parse_node ──→ retrieve_node ──→ score_node
                                      │
                 ┌────────────────────┼──────────────────────┐
                 │ next_action ==    │ next_action ==       │ next_action ==
                 │ "optimize"        │ "interview"          │ "agentic" (v2.0)
                 ▼                    ▼                      ▼
            optimize_node       interview_node         agentic_node
                 │                    │              (ReAct 自主任务)
   ┌─────────────┼─────────────┐      │                      │
   │ re_optimize │ interview   │ end  │                      │
   ▼             ▼             ▼      │                      │
optimize_node interview_node  END     │                      │
                 │                    │                      │
                 └────────────┬───────┘                      │
                              ▼                              │
                        summary_node                         │
                              │                              │
                             END ◄───────────────────────────┘

人工介入断点：
- interrupt_before=["optimize"]：优化前可手动修改简历
- interrupt_before=["interview"]：面试前可更换 JD 重新开始

v2.0 新增自主 Agent 分支：
- score 节点后 next_action == "agentic" → agentic_node
- agentic_node 基于自研 ReActAgent（Thought/Action/Observation 循环）
  自主调度评分/检索/导出工具完成用户任务，任务完成即结束

使用示例：
    graph = build_workflow()
    config = {"configurable": {"thread_id": "session-001"}}

    # 阶段1: 解析 + 检索 + 评分
    state = graph.invoke(AgentState(resume_raw="...", jd_raw="..."), config)

    # 阶段2: 优化（用户选择）
    state["next_action"] = "optimize"
    state = graph.invoke(state, config)

    # 阶段3: 面试
    state["next_action"] = "interview"
    state = graph.invoke(state, config)
"""

from typing import Literal

from loguru import logger
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config.settings import settings
from core.graph.agent_state import AgentState
from core.graph.agent_nodes import (
    parse_node,
    retrieve_node,
    score_node,
    optimize_node,
    interview_generate_question,
    summary_node,
)
from core.graph.agentic_node import agentic_node


# ============================================================
# 条件路由函数
# ============================================================

def route_after_score(state: AgentState) -> Literal["optimize", "interview", "agentic"]:
    """
    评分后的分支路由。

    逻辑：
    - next_action == "optimize" → 进入简历优化分支
    - next_action == "interview" → 直接进入面试分支（默认）
    - next_action == "agentic"  → 进入自主 Agent 分支（v2.0，ReAct 自主任务）
    """
    action = state.next_action
    logger.info(f"[路由] 评分后分支 → {action}")

    if action == "optimize":
        return "optimize"
    if action == "agentic":
        return "agentic"
    return "interview"


def route_after_optimize(state: AgentState) -> Literal["optimize", "interview", "__end__"]:
    """
    优化后的分支路由。

    逻辑：
    - next_action == "re_optimize" → 循环回 optimize（用户要求再次优化）
    - next_action == "interview"  → 进入面试
    - 其他 → 结束流程
    """
    action = state.next_action
    logger.info(f"[路由] 优化后分支 → {action}")

    if action == "re_optimize":
        return "optimize"
    elif action == "interview":
        return "interview"
    return END


# ============================================================
# 工作流构建
# ============================================================

def build_workflow(
    interrupt_before_optimize: bool = True,
    interrupt_before_interview: bool = True,
) -> StateGraph:
    """
    构建 LangGraph 多智能体工作流。

    Args:
        interrupt_before_optimize: True 时在 optimize_node 前挂起（支持人工修改简历）
        interrupt_before_interview: True 时在 interview_node 前挂起（支持更换 JD）

    Returns:
        编译后的 StateGraph（已包含 MemorySaver checkpointer）
    """
    logger.info("构建 LangGraph 工作流...")

    # 创建状态图
    workflow = StateGraph(AgentState)

    # ----------------------------------------------------------
    # 注册节点
    # ----------------------------------------------------------
    workflow.add_node("parse", parse_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("score", score_node)
    workflow.add_node("optimize", optimize_node)
    workflow.add_node("interview", interview_generate_question)
    workflow.add_node("summary", summary_node)
    workflow.add_node("agentic", agentic_node)  # v2.0: 自主 Agent 节点（ReAct）

    logger.info(
        "  节点已注册: parse, retrieve, score, optimize, interview, summary, agentic"
    )

    # ----------------------------------------------------------
    # 线性边：parse → retrieve → score
    # ----------------------------------------------------------
    workflow.set_entry_point("parse")
    workflow.add_edge("parse", "retrieve")
    workflow.add_edge("retrieve", "score")

    # ----------------------------------------------------------
    # 条件分支：评分后
    # ----------------------------------------------------------
    workflow.add_conditional_edges(
        "score",
        route_after_score,
        {
            "optimize": "optimize",
            "interview": "interview",
            "agentic": "agentic",  # v2.0: 自主 Agent 分支
        },
    )

    # ----------------------------------------------------------
    # 条件分支：优化后（支持循环）
    # ----------------------------------------------------------
    workflow.add_conditional_edges(
        "optimize",
        route_after_optimize,
        {
            "optimize": "optimize",       # 循环：重新优化
            "interview": "interview",     # 进入面试
            END: END,                     # 直接结束
        },
    )

    # ----------------------------------------------------------
    # 自主 Agent 分支：任务完成即结束（v2.0）
    # ----------------------------------------------------------
    workflow.add_edge("agentic", END)

    # ----------------------------------------------------------
    # 线性边：面试 → 复盘 → 结束
    # ----------------------------------------------------------
    workflow.add_edge("interview", "summary")
    workflow.add_edge("summary", END)

    # ----------------------------------------------------------
    # 编译（带 MemorySaver 支持断点续传）
    # ----------------------------------------------------------
    checkpointer = MemorySaver()

    # 构建断点列表
    interrupt_before: list[str] = []
    if interrupt_before_optimize:
        interrupt_before.append("optimize")
    if interrupt_before_interview:
        interrupt_before.append("interview")

    compiled = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before if interrupt_before else None,
    )

    logger.info(
        f"  工作流编译完成"
        f" (interrupt_before={interrupt_before if interrupt_before else 'None'})"
    )

    return compiled


# ============================================================
# 全局单例
# ============================================================

_workflow_instance: StateGraph | None = None


def get_workflow() -> StateGraph:
    """
    获取全局单例 LangGraph 工作流。

    Returns:
        编译后的 StateGraph 实例
    """
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = build_workflow(
            interrupt_before_optimize=True,
            interrupt_before_interview=True,
        )
    return _workflow_instance


# ============================================================
# 高级调用封装
# ============================================================

def run_analysis_pipeline(
    resume_raw: str,
    jd_raw: str,
    thread_id: str = "default",
) -> AgentState:
    """
    一键运行 解析→检索→评分 三阶段分析流水线。

    适用场景：用户在 Gradio 前端上传简历和 JD 后，获取匹配评分。

    Args:
        resume_raw: 原始简历文本
        jd_raw:     岗位 JD 文本
        thread_id:  会话 ID（用于状态持久化）

    Returns:
        AgentState: 包含切片、RAG 资料、四维评分的完整状态
    """
    logger.info(f"[Pipeline] 启动分析流水线 thread={thread_id}")

    graph = get_workflow()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = AgentState(
        resume_raw=resume_raw,
        jd_raw=jd_raw,
        next_action="interview",  # 默认评分后直接到面试
    )

    # 运行到第一个断点（optimize 或 interview 前）
    result = graph.invoke(initial_state, config)

    # graph.invoke 返回 dict，转换为 AgentState
    if isinstance(result, dict):
        result = AgentState(**result)

    logger.info(
        f"[Pipeline] 分析完成 | "
        f"chunks_resume={len(result.chunk_resume)} "
        f"chunks_jd={len(result.chunk_jd)} "
        f"rag_docs={len(result.rag_context)} "
        f"score={result.match_score.overall_score}"
    )
    return result


def run_optimize_step(
    state: AgentState,
    thread_id: str = "default",
    re_optimize: bool = False,
) -> AgentState:
    """
    运行单步简历优化。

    Args:
        state:       当前状态（含评分和 RAG 资料）
        thread_id:   会话 ID
        re_optimize: True 表示这是第 N 次优化迭代

    Returns:
        AgentState: 更新后的状态（含 optimized_resume）
    """
    logger.info(f"[Optimize] 启动优化步骤 thread={thread_id} re_optimize={re_optimize}")

    graph = get_workflow()
    config = {"configurable": {"thread_id": thread_id}}

    # 设置路由指令
    updated_state = state.model_copy()
    updated_state.next_action = "re_optimize" if re_optimize else "interview"

    result = graph.invoke(updated_state, config)

    if isinstance(result, dict):
        result = AgentState(**result)

    logger.info(f"[Optimize] 优化完成 | length={len(result.optimized_resume)}")
    return result


def run_interview_step(
    state: AgentState,
    thread_id: str = "default",
) -> AgentState:
    """
    运行面试步骤：生成第一个面试问题。

    注意：面试是交互式的，此方法只生成第一个问题。
    后续问题和回答通过 interview_generate_question / interview_evaluate_answer 逐步推进。

    Args:
        state:     当前状态（含简历、JD、评分、可选优化简历）
        thread_id: 会话 ID

    Returns:
        AgentState: 含第一轮面试问题的状态
    """
    logger.info(f"[Interview] 启动面试 thread={thread_id}")

    graph = get_workflow()
    config = {"configurable": {"thread_id": thread_id}}

    updated_state = state.model_copy()
    updated_state.next_action = "interview"

    result = graph.invoke(updated_state, config)

    if isinstance(result, dict):
        result = AgentState(**result)

    logger.info(f"[Interview] 面试开始 | rounds={result.interview_round_count}")
    return result


def run_summary_step(
    state: AgentState,
    thread_id: str = "default",
) -> AgentState:
    """
    运行面试复盘步骤。

    Args:
        state:     当前状态（含完整面试记录）
        thread_id: 会话 ID

    Returns:
        AgentState: 含 interview_report 的完成状态
    """
    logger.info(f"[Summary] 生成面试复盘报告 thread={thread_id}")

    # 直接调用 summary_node（不经过工作流图，因为它是流水线的终点）
    result = summary_node(state)
    updated = state.model_copy(update=result)
    updated.current_node = "summary"

    logger.info(f"[Summary] 报告生成完成 | length={len(updated.interview_report)}")
    return updated


def run_agentic_task(
    task: str,
    resume_raw: str = "",
    jd_raw: str = "",
    thread_id: str = "agentic-default",
    fast_path: bool = True,
) -> AgentState:
    """
    运行自主 Agent 任务（v2.0 新增）。

    ReActAgent 自主完成用户任务，支持两种路径：
    - fast_path=True（默认）: 直接调用 agentic_node，跳过解析/检索/评分，
      适合纯自主任务（无需简历/JD 上下文）
    - fast_path=False: 走完整 LangGraph 图（parse→retrieve→score→agentic），
      任务可复用简历/JD 与 RAG 上下文

    Args:
        task:         用户任务描述（如 "技能80 经验60 学历90，算综合得分并导出报告"）
        resume_raw:   简历文本（fast_path=False 时生效）
        jd_raw:       JD 文本（fast_path=False 时生效）
        thread_id:    会话 ID
        fast_path:    True 跳过前置节点直接执行自主任务

    Returns:
        AgentState: 含 agentic_result（最终回答）与 agentic_trace（推理轨迹）
    """
    logger.info(f"[Agentic] 启动自主任务 thread={thread_id} fast_path={fast_path}")

    if fast_path:
        state = AgentState(
            resume_raw=resume_raw,
            jd_raw=jd_raw,
            next_action="agentic",
            agentic_task=task,
        )
        result = agentic_node(state)
        return state.model_copy(update=result)

    # 完整图路径
    graph = get_workflow()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = AgentState(
        resume_raw=resume_raw,
        jd_raw=jd_raw,
        next_action="agentic",
        agentic_task=task,
    )
    result = graph.invoke(initial_state, config)
    if isinstance(result, dict):
        result = AgentState(**result)

    logger.info(
        f"[Agentic] 自主任务完成 | answer_len={len(result.agentic_result)} "
        f"trace_steps={len(result.agentic_trace)}"
    )
    return result
