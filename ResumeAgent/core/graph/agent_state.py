"""
core/graph/agent_state.py - LangGraph 全局流转状态定义

基于 Pydantic 定义 AgentState，贯穿整个多智能体协同工作流。
所有 Agent 节点通过读写此 State 完成数据传递与状态追踪。

字段说明：
- resume_raw:       原始简历文本（用户上传/粘贴）
- jd_raw:           岗位JD原文
- chunk_resume:     简历切片列表（RAG 入库用）
- chunk_jd:         JD 切片列表（RAG 入库用）
- rag_context:      RAG 检索召回的参考文档列表
- match_score:      四维匹配分数 {skill/experience/education/overall}
- skill_gap:        候选人与 JD 之间的技能差距清单
- optimized_resume: 针对 JD 优化后的简历文本
- interview_history:AI 模拟面试问答记录
- interview_report: 面试复盘总结报告
- current_node:     当前所在节点名称（用于状态追踪与条件分支）
"""

from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.documents import Document


# ============================================================
# 四维匹配评分子结构
# ============================================================

class MatchScoreDetail(BaseModel):
    """
    四维度匹配评分明细。

    每个维度 0~100 分，用于精细化评估候选人与岗位的匹配度。
    """
    skill_match: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="技能匹配度：技术栈、工具链、编程语言等硬技能覆盖程度",
    )
    experience_match: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="经验匹配度：工作年限、项目复杂度、行业背景等经验契合度",
    )
    education_match: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="学历匹配度：学历层次、专业方向等与 JD 要求的一致性",
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="综合得分：前三项的加权综合",
    )


# ============================================================
# 面试问答记录子结构
# ============================================================

class InterviewQA(BaseModel):
    """
    单轮面试问答记录。
    """
    round: int = Field(default=0, description="轮次编号（从 1 开始）")
    question: str = Field(default="", description="面试官提问")
    answer: str = Field(default="", description="候选人回答")
    score: float = Field(default=0.0, ge=0.0, le=100.0, description="本轮得分")
    feedback: str = Field(default="", description="本轮评价/改进建议")
    category: str = Field(default="general", description="问题类别: technical|behavioral|project|general")


# ============================================================
# 全局 Agent 状态
# ============================================================

class AgentState(BaseModel):
    """
    LangGraph 多智能体协同工作流全局状态。

    贯穿解析→匹配→优化→面试→复盘全链路，
    每个 Agent 节点读取所需字段、写入产出字段。

    使用方式:
        state = AgentState(resume_raw="...", jd_raw="...")
        state.match_score = MatchScoreDetail(skill_match=85, ...)
    """

    # ----------------------------------------------------------
    # 输入层：用户原始数据
    # ----------------------------------------------------------
    resume_raw: str = Field(
        default="",
        description="原始简历文本（PDF/Word 解析后或用户粘贴）",
    )
    jd_raw: str = Field(
        default="",
        description="岗位 JD 原文（解析后或用户粘贴）",
    )

    # ----------------------------------------------------------
    # 中间层：文档切片（入库 RAG 用）
    # ----------------------------------------------------------
    chunk_resume: list[str] = Field(
        default_factory=list,
        description="简历文档切片列表",
    )
    chunk_jd: list[str] = Field(
        default_factory=list,
        description="JD 文档切片列表",
    )

    # ----------------------------------------------------------
    # RAG 层：检索增强上下文
    # ----------------------------------------------------------
    rag_context: list[Document] = Field(
        default_factory=list,
        description="RAG 混合检索召回的高质量参考文档（经 CrossEncoder 重排）",
    )

    # ----------------------------------------------------------
    # 匹配层：打分与差距
    # ----------------------------------------------------------
    match_score: MatchScoreDetail = Field(
        default_factory=MatchScoreDetail,
        description="四维度匹配评分明细",
    )
    skill_gap: list[str] = Field(
        default_factory=list,
        description="候选人缺失技能清单（JD 要求但简历中未体现的技能）",
    )

    # ----------------------------------------------------------
    # 优化层：简历改写产出
    # ----------------------------------------------------------
    optimized_resume: str = Field(
        default="",
        description="针对目标 JD 定向优化后的简历文本",
    )
    optimize_feedback: str = Field(
        default="",
        description="用户对上一版优化结果的附加反馈（多轮迭代用）",
    )
    optimize_mode: str = Field(
        default="keywords",
        description="优化模式: keywords=侧重关键词匹配 | quantify=侧重量化成果 | concise=精简冗余",
    )

    # ----------------------------------------------------------
    # 面试层：模拟面试记录
    # ----------------------------------------------------------
    interview_history: list[InterviewQA] = Field(
        default_factory=list,
        description="AI 模拟面试全部问答记录",
    )
    interview_report: str = Field(
        default="",
        description="面试复盘总结报告（含总体评分、优势、不足、改进建议）",
    )

    # ----------------------------------------------------------
    # 元信息：状态追踪与路由控制
    # ----------------------------------------------------------
    current_node: str = Field(
        default="",
        description="当前正在执行的 Agent 节点名称（用于日志与条件分支）",
    )
    next_action: str = Field(
        default="interview",
        description="路由控制指令: interview | optimize | re_optimize | end",
    )

    # ----------------------------------------------------------
    # 便捷方法
    # ----------------------------------------------------------

    @property
    def has_resume(self) -> bool:
        """是否已加载简历"""
        return bool(self.resume_raw.strip())

    @property
    def has_jd(self) -> bool:
        """是否已加载 JD"""
        return bool(self.jd_raw.strip())

    @property
    def has_rag_context(self) -> bool:
        """是否已获取 RAG 参考文档"""
        return len(self.rag_context) > 0

    @property
    def has_match_score(self) -> bool:
        """是否已完成匹配打分"""
        return self.match_score.overall_score > 0

    @property
    def has_optimized_resume(self) -> bool:
        """是否已生成优化简历"""
        return bool(self.optimized_resume.strip())

    @property
    def is_interview_complete(self) -> bool:
        """面试是否已完成（有复盘报告即为完成）"""
        return bool(self.interview_report.strip())

    @property
    def interview_round_count(self) -> int:
        """已完成面试轮次"""
        return len(self.interview_history)

    # ----------------------------------------------------------
    # Pydantic 配置
    # ----------------------------------------------------------

    class Config:
        """Pydantic 模型配置"""
        arbitrary_types_allowed = True    # 允许 langchain_core.documents.Document 等非标准类型
        validate_assignment = True        # 属性赋值时触发验证
        extra = "forbid"                  # 禁止额外字段，保证类型安全
