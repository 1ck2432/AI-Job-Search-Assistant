"""
core/utils/pydantic_models.py - Pydantic 结构化数据模型

定义系统中所有核心业务实体的数据模型，包括：
- 简历解析结果
- JD（岗位描述）解析结果
- 匹配评分结果
- 简历优化建议
- 模拟面试问答
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# 简历解析
# ============================================================

class Education(BaseModel):
    """教育经历"""
    school: str = Field(default="", description="学校名称")
    degree: str = Field(default="", description="学历/学位")
    major: str = Field(default="", description="专业")
    start_date: str = Field(default="", description="入学年份")
    end_date: str = Field(default="", description="毕业年份")


class WorkExperience(BaseModel):
    """工作/实习经历"""
    company: str = Field(default="", description="公司名称")
    position: str = Field(default="", description="职位")
    start_date: str = Field(default="", description="入职时间")
    end_date: str = Field(default="", description="离职时间")
    description: str = Field(default="", description="工作内容描述")
    achievements: list[str] = Field(default_factory=list, description="项目/成果亮点")


class Project(BaseModel):
    """项目经历"""
    name: str = Field(default="", description="项目名称")
    role: str = Field(default="", description="担任角色")
    start_date: str = Field(default="", description="开始时间")
    end_date: str = Field(default="", description="结束时间")
    description: str = Field(default="", description="项目描述")
    tech_stack: list[str] = Field(default_factory=list, description="使用的技术栈")


class ResumeData(BaseModel):
    """
    简历结构化解析结果。
    由 LLM 从原始文本中提取并填充。
    """
    name: str = Field(default="", description="姓名")
    email: str = Field(default="", description="邮箱")
    phone: str = Field(default="", description="电话")
    location: str = Field(default="", description="所在城市")
    years_of_experience: int = Field(default=0, description="总工作年限")
    summary: str = Field(default="", description="个人总结/求职意向")
    skills: list[str] = Field(default_factory=list, description="技能列表")
    education: list[Education] = Field(default_factory=list, description="教育经历")
    work_experience: list[WorkExperience] = Field(default_factory=list, description="工作经历")
    projects: list[Project] = Field(default_factory=list, description="项目经历")
    raw_text: str = Field(default="", description="简历原始文本")
    parsed_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="解析时间戳",
    )


# ============================================================
# JD 解析
# ============================================================

class JobRequirement(BaseModel):
    """岗位具体要求"""
    category: str = Field(default="", description="要求类别（学历/技能/经验等）")
    requirement: str = Field(default="", description="具体要求描述")
    is_must: bool = Field(default=True, description="是否为硬性要求")


class JDData(BaseModel):
    """
    JD（岗位描述）结构化解析结果。
    """
    job_title: str = Field(default="", description="岗位名称")
    company: str = Field(default="", description="公司名称")
    location: str = Field(default="", description="工作地点")
    salary_range: str = Field(default="", description="薪资范围")
    department: str = Field(default="", description="所属部门")
    job_summary: str = Field(default="", description="岗位概述")
    responsibilities: list[str] = Field(default_factory=list, description="岗位职责")
    requirements: list[JobRequirement] = Field(default_factory=list, description="任职要求")
    preferred_skills: list[str] = Field(default_factory=list, description="加分项/优先技能")
    tech_stack: list[str] = Field(default_factory=list, description="所需技术栈")
    raw_text: str = Field(default="", description="JD 原始文本")
    parsed_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="解析时间戳",
    )


# ============================================================
# 匹配分析
# ============================================================

class MatchDimension(BaseModel):
    """
    单一维度的匹配得分。
    """
    dimension: str = Field(default="", description="维度名称（如技能匹配、经验匹配等）")
    score: float = Field(default=0.0, ge=0.0, le=100.0, description="得分 (0-100)")
    comment: str = Field(default="", description="评分说明")


class MatchResult(BaseModel):
    """
    简历与 JD 匹配分析结果。
    """
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0, description="综合匹配得分")
    dimensions: list[MatchDimension] = Field(default_factory=list, description="各维度得分")
    strengths: list[str] = Field(default_factory=list, description="简历亮点")
    gaps: list[str] = Field(default_factory=list, description="简历不足/差距")
    summary: str = Field(default="", description="综合分析建议")
    analyzed_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="分析时间戳",
    )


# ============================================================
# 简历优化
# ============================================================

class ResumeSuggestion(BaseModel):
    """
    单条简历优化建议。
    """
    section: str = Field(default="", description="所属段落")
    original: str = Field(default="", description="原始内容")
    optimized: str = Field(default="", description="优化后内容")
    reason: str = Field(default="", description="优化理由")


class ResumeOptimization(BaseModel):
    """
    简历定向优化结果。
    """
    target_jd: str = Field(default="", description="目标岗位名称")
    suggestions: list[ResumeSuggestion] = Field(default_factory=list, description="优化建议列表")
    optimized_resume: str = Field(default="", description="优化后的完整简历文本")
    keywords_to_add: list[str] = Field(default_factory=list, description="建议补充的关键词")
    optimized_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="优化时间戳",
    )


# ============================================================
# 模拟面试
# ============================================================

class InterviewQuestion(BaseModel):
    """
    单道面试题。
    """
    question_id: int = Field(default=0, description="题目序号")
    category: str = Field(default="", description="题目类别（技术/行为/项目等）")
    question: str = Field(default="", description="面试问题")
    expected_keywords: list[str] = Field(default_factory=list, description="期望回答关键点")
    difficulty: str = Field(default="medium", description="难度: easy | medium | hard")
    reference_answer: str = Field(default="", description="参考答案提示")


class InterviewAnswer(BaseModel):
    """
    用户面试回答。
    """
    question_id: int = Field(default=0, description="对应题目序号")
    answer: str = Field(default="", description="用户回答")
    score: float = Field(default=0.0, ge=0.0, le=100.0, description="回答得分")
    feedback: str = Field(default="", description="AI 点评反馈")
    answered_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="回答时间戳",
    )


class InterviewSession(BaseModel):
    """
    一场完整的模拟面试会话。
    """
    session_id: str = Field(default="", description="会话 ID")
    job_title: str = Field(default="", description="目标岗位")
    questions: list[InterviewQuestion] = Field(default_factory=list, description="面试问题列表")
    answers: list[InterviewAnswer] = Field(default_factory=list, description="回答记录列表")
    total_score: float = Field(default=0.0, description="面试总分")
    overall_feedback: str = Field(default="", description="整体评价与建议")
    started_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="面试开始时间",
    )
    finished_at: Optional[str] = Field(default=None, description="面试结束时间")


# ============================================================
# 知识库文档
# ============================================================

class KnowledgeDocument(BaseModel):
    """
    知识库单篇文档元数据。
    """
    doc_id: str = Field(default="", description="文档唯一 ID")
    title: str = Field(default="", description="文档标题")
    category: str = Field(default="", description="分类（行业知识/面试经验/技术文章等）")
    file_path: str = Field(default="", description="原始文件路径")
    content_preview: str = Field(default="", description="内容摘要（前200字）")
    chunk_count: int = Field(default=0, description="切片数量")
    added_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="入库时间",
    )


# ============================================================
# 通用 API 响应
# ============================================================

class APIResponse(BaseModel):
    """
    统一 API 响应格式。
    """
    success: bool = Field(default=True, description="请求是否成功")
    message: str = Field(default="", description="响应消息")
    data: Optional[dict] = Field(default=None, description="响应数据体")
    error: Optional[str] = Field(default=None, description="错误信息（仅失败时）")
