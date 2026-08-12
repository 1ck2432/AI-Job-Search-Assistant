"""
core/agents/tool_registry.py - 工具注册表

把项目现有纯函数（评分计算、RAG 检索、文件导出）封装为
可被 ReActAgent 自主调用的工具（ReActTool）。

每个工具定义三要素：
- name:        工具名（模型在 Action 中引用）
- description: 功能描述（模型据此判断何时调用）
- parameters:  JSON Schema 参数定义

v2.0 新增，作为 Function Calling / ReAct 的功能底座。
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from core.llm.base import ToolDefinition


# ============================================================
# 工具封装
# ============================================================

@dataclass
class ReActTool:
    """
    可被模型调用的工具封装。

    Attributes:
        name:        工具名
        description: 功能描述
        func:        实际执行函数（接收 **kwargs）
        parameters:  JSON Schema 参数定义
        required:    必填参数名列表
    """
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict = field(default_factory=dict)
    required: List[str] = field(default_factory=list)

    def execute(self, **kwargs) -> str:
        """
        执行工具并返回可读文本结果（Observation）。

        Args:
            **kwargs: 工具参数（来自模型 Action Input）

        Returns:
            str: 执行结果文本；失败时返回错误描述（不抛异常）
        """
        try:
            result = self.func(**kwargs)
            return _format_result(result)
        except TypeError as e:
            return f"[工具参数错误] {e}"
        except Exception as e:
            logger.error(f"[tool:{self.name}] 执行失败: {type(e).__name__}: {e}")
            return f"[工具执行失败] {type(e).__name__}: {e}"

    def to_tool_definition(self) -> ToolDefinition:
        """转换为 LLM Function Calling 工具定义。"""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=dict(self.parameters),
            required=list(self.required),
        )

    def describe(self) -> str:
        """生成供 ReAct prompt 使用的工具说明文本。"""
        params_desc = json.dumps(
            self.parameters.get("properties", {}), ensure_ascii=False
        )
        return (
            f"- {self.name}: {self.description}\n"
            f"  参数 Schema: {params_desc}"
        )


def _format_result(result: Any) -> str:
    """将工具返回值格式化为文本。"""
    if result is None:
        return "执行成功"
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, indent=2)
    return str(result)


# ============================================================
# 内置工具实现（延迟导入，避免循环依赖 / 未初始化向量库）
# ============================================================

# 参数名别名表：模型可能输出 skill/skill_score/中文键名，统一归一化
_SCORE_KEY_ALIASES: Dict[str, List[str]] = {
    "skill_match": ["skill_match", "skill", "skill_score", "skills", "技能", "技能分"],
    "experience_match": [
        "experience_match", "experience", "experience_score",
        "exp", "project_experience", "经验", "经验分",
    ],
    "education_match": [
        "education_match", "education", "education_score",
        "edu", "edu_score", "degree", "学历", "学历分",
    ],
}


def _resolve_score_args(kwargs: dict) -> dict:
    """将模型可能输出的任意键名归一化为 skill_match/experience_match/education_match。"""
    resolved: Dict[str, Any] = {}
    for key, aliases in _SCORE_KEY_ALIASES.items():
        for alias in aliases:
            if alias in kwargs and kwargs[alias] is not None:
                resolved[key] = kwargs[alias]
                break
    if "weights" in kwargs and isinstance(kwargs["weights"], dict):
        resolved["weights"] = kwargs["weights"]
    return resolved


def _tool_calculate_weighted_score(**kwargs) -> Dict[str, Any]:
    """计算三维度加权综合得分（工具版，兼容参数名别名）。"""
    from core.tools.score_tool import (
        calculate_weighted_score,
        score_to_grade,
        DEFAULT_WEIGHTS,
    )
    from core.graph.agent_state import MatchScoreDetail

    kwargs = _resolve_score_args(kwargs)
    skill = kwargs.get("skill_match")
    exp = kwargs.get("experience_match")
    edu = kwargs.get("education_match")
    weights = kwargs.get("weights")

    if skill is None or exp is None or edu is None:
        return {
            "error": "缺少必填参数，参数名必须是 skill_match / experience_match / "
                     "education_match（可接收 skill / experience / education 等别名）",
            "received_keys": list(kwargs.keys()),
        }

    scores = MatchScoreDetail(
        skill_match=float(skill),
        experience_match=float(exp),
        education_match=float(edu),
        overall_score=0.0,
    )
    overall = calculate_weighted_score(scores, weights or DEFAULT_WEIGHTS)
    grade, desc, _ = score_to_grade(overall)
    return {
        "overall_score": overall,
        "grade": grade,
        "grade_desc": desc,
        "weights": weights or DEFAULT_WEIGHTS,
    }


def _tool_score_to_grade(**kwargs) -> Dict[str, Any]:
    """分数 → 等级映射（工具版，兼容 score/分数 等参数名）。"""
    from core.tools.score_tool import score_to_grade

    score = kwargs.get("score", kwargs.get("分数", kwargs.get("分值")))
    if score is None:
        return {"error": "缺少必填参数 score（分数 0-100）", "received_keys": list(kwargs.keys())}
    grade, desc, color = score_to_grade(float(score))
    return {"score": float(score), "grade": grade, "description": desc, "color": color}


def _tool_format_match_report(**kwargs) -> str:
    """生成人类可读的完整匹配分析报告文本（工具版，兼容参数名别名）。"""
    from core.tools.score_tool import format_match_report
    from core.graph.agent_state import MatchScoreDetail

    kwargs = _resolve_score_args(kwargs)
    skill = kwargs.get("skill_match")
    exp = kwargs.get("experience_match")
    edu = kwargs.get("education_match")
    if skill is None or exp is None or edu is None:
        return ("缺少必填参数，参数名必须是 skill_match / experience_match / "
                "education_match，收到的键: " + str(list(kwargs.keys())))

    scores = MatchScoreDetail(
        skill_match=float(skill),
        experience_match=float(exp),
        education_match=float(edu),
        overall_score=0.0,
    )
    return format_match_report(scores)


def _tool_search_knowledge_base(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    """在简历/岗位知识库中检索相关资料（工具版）。"""
    try:
        from core.rag.retriever import get_retriever

        retriever = get_retriever()
        docs = retriever.retrieve(query, top_k=int(top_k), use_rerank=False)
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content[:800],
                "source": (doc.metadata or {}).get("source", "unknown"),
            })
        return results
    except Exception as e:
        logger.warning(f"[tool:search_knowledge_base] 检索失败: {e}")
        return [{"content": f"知识库检索不可用: {e}", "source": "error"}]


def _tool_export_report(report_text: str, job_title: str = "Agent分析报告") -> str:
    """将报告文本导出为 .txt 文件（工具版）。"""
    from core.tools.file_export import export_match_report_to_txt

    path = export_match_report_to_txt(report_text, job_title=job_title)
    return f"报告已导出到: {path}"


# ============================================================
# 工具注册表
# ============================================================

def builtin_tools() -> List[ReActTool]:
    """
    返回项目内置工具列表（评分计算 / 等级映射 / 匹配报告 / RAG 检索 / 报告导出）。

    这些工具均为纯函数，可安全交由 ReActAgent 自主调度。
    """
    return [
        ReActTool(
            name="calculate_weighted_score",
            description=(
                "计算三维度（技能/经验/学历，各 0-100）加权综合得分，"
                "并给出匹配等级（S/A/B/C/D）。当需要量化评分时调用。"
                "参数名必须是 skill_match / experience_match / education_match，"
                "示例: {\"skill_match\": 80, \"experience_match\": 60, \"education_match\": 90}"
            ),
            func=_tool_calculate_weighted_score,
            parameters={
                "type": "object",
                "properties": {
                    "skill_match": {"type": "number", "description": "技能匹配分 0-100（注意参数名是 skill_match）"},
                    "experience_match": {"type": "number", "description": "经验匹配分 0-100（参数名是 experience_match）"},
                    "education_match": {"type": "number", "description": "学历匹配分 0-100（参数名是 education_match）"},
                    "weights": {"type": "object", "description": "可选自定义权重"},
                },
            },
            required=["skill_match", "experience_match", "education_match"],
        ),
        ReActTool(
            name="score_to_grade",
            description=(
                "将 0-100 分数转换为匹配等级（S/A/B/C/D）及描述。"
                "参数名必须是 score，示例: {\"score\": 80}"
            ),
            func=_tool_score_to_grade,
            parameters={
                "type": "object",
                "properties": {
                    "score": {"type": "number", "description": "分数 0-100"},
                },
            },
            required=["score"],
        ),
        ReActTool(
            name="format_match_report",
            description=(
                "根据三维度分数生成人类可读的完整匹配分析报告文本。"
                "参数名必须是 skill_match / experience_match / education_match"
            ),
            func=_tool_format_match_report,
            parameters={
                "type": "object",
                "properties": {
                    "skill_match": {"type": "number", "description": "技能匹配分 0-100"},
                    "experience_match": {"type": "number", "description": "经验匹配分 0-100"},
                    "education_match": {"type": "number", "description": "学历匹配分 0-100"},
                },
            },
            required=["skill_match", "experience_match", "education_match"],
        ),
        ReActTool(
            name="search_knowledge_base",
            description=(
                "在简历/岗位知识库中检索与 query 相关的资料片段。"
                "当需要查找简历原文、JD 要求、或补充背景信息时调用。"
            ),
            func=_tool_search_knowledge_base,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词/问题"},
                    "top_k": {"type": "integer", "description": "返回片段数，默认 5"},
                },
            },
            required=["query"],
        ),
        ReActTool(
            name="export_report",
            description="将文本内容导出为 .txt 报告文件，返回保存路径。",
            func=_tool_export_report,
            parameters={
                "type": "object",
                "properties": {
                    "report_text": {"type": "string", "description": "报告完整文本"},
                    "job_title": {"type": "string", "description": "报告标题，默认 Agent分析报告"},
                },
            },
            required=["report_text"],
        ),
    ]
