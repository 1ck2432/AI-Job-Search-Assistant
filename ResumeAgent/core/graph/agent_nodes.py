"""
core/graph/agent_nodes.py - 六大 Agent 业务节点

每个节点函数签名：def xx_node(state: AgentState) -> dict
返回 dict 部分更新 AgentState，由 LangGraph 自动合并。

节点清单：
  1. parse_node       - 文档解析与清洗切片
  2. retrieve_node    - RAG 混合检索增强
  3. score_node       - 四维度匹配评估（LLM 结构化输出）
  4. optimize_node    - 简历 STAR 优化改写（支持多轮迭代）
  5. interview_node   - AI 模拟面试官（生成问题 + 评分点评）
  6. summary_node     - 面试复盘总结 + 学习资料推荐

v2.0 变更：score / interview 节点改为 Function Calling 强制结构化输出
（bind_tools 直接返回结构化参数，替换脆弱的 JSON 文本解析），
模型不支持工具调用时自动回退到 _parse_llm_json。
"""

from typing import Optional

from loguru import logger
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config.settings import settings
from core.graph.agent_state import AgentState, MatchScoreDetail, InterviewQA
from core.rag.document_loader import DocumentLoader
from core.rag.text_splitter import DocumentSplitter
from core.rag.retriever import get_retriever
from core.llm import get_llm
from core.llm.base import ChatMessage, pydantic_model_to_tool


# ============================================================
# 工具函数
# ============================================================

def _extract_keywords(text: str, max_words: int = 15) -> str:
    """从文本中提取关键词用于检索 query 构建"""
    # 简单实现：取前若干字作为关键词片段
    keywords = text.replace("\n", " ").replace("\r", " ")[:300]
    return keywords


def _build_retrieval_query(jd_text: str, resume_text: str) -> str:
    """
    根据 JD + 简历内容构建检索查询语句。
    提取 JD 中的核心要求与技术栈，结合简历形成针对性检索。
    """
    # 提取 JD 前 200 字（通常含岗位核心要求）
    jd_intro = jd_text[:400].replace("\n", " ")
    # 提取简历中的技能关键词
    resume_skills = resume_text[:200].replace("\n", " ")

    query = (
        f"岗位要求: {jd_intro}。"
        f"候选人背景: {resume_skills}。"
        f"请检索相关面试题、简历范文、技能学习资料。"
    )
    return query


def _parse_llm_json(text: str) -> dict:
    """
    从 LLM 返回文本中提取 JSON 块。
    兼容 ```json ... ``` 包裹格式。
    """
    import json
    import re

    text = text.strip()

    # 尝试提取 ```json ... ``` 代码块
    json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if json_match:
        text = json_match.group(1).strip()

    # 尝试提取 { ... } 块
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        text = brace_match.group(0)

    return json.loads(text)


# ============================================================
# v2.0 结构化输出（Function Calling 替代 JSON 解析）
# ============================================================

class ScoreResultSchema(BaseModel):
    """四维度匹配评估的结构化输出 schema"""
    skill_match: float = Field(ge=0, le=100, description="专业技能匹配度 0-100")
    experience_match: float = Field(ge=0, le=100, description="项目经验匹配度 0-100")
    education_match: float = Field(ge=0, le=100, description="学历背景匹配度 0-100")
    overall_score: float = Field(ge=0, le=100, description="综合加权得分 0-100")
    skill_gap: list[str] = Field(default_factory=list, description="候选人缺失技能清单")
    analysis: str = Field(default="", description="200 字以内简短分析")


class InterviewQuestionSchema(BaseModel):
    """面试题生成的结构化输出 schema"""
    question: str = Field(default="", description="面试问题文本")
    category: str = Field(default="general", description="问题类别")
    expected_points: list[str] = Field(default_factory=list, description="期望回答要点")
    difficulty: str = Field(default="medium", description="难度: easy/medium/hard")


class InterviewEvalSchema(BaseModel):
    """面试回答评估的结构化输出 schema"""
    score: float = Field(ge=0, le=100, description="评分 0-100")
    feedback: str = Field(default="", description="50 字以内点评")
    strengths: list[str] = Field(default_factory=list, description="回答优点")
    weaknesses: list[str] = Field(default_factory=list, description="回答不足")


def _structured_extract(
    llm,
    system_prompt: str,
    user_prompt: str,
    schema: type[BaseModel],
    tool_name: str,
    tool_desc: str,
    temperature: float = 0.1,
) -> dict:
    """
    v2.0 结构化输出核心函数。

    优先使用 Function Calling（bind_tools + 强制指定工具）让模型直接返回
    结构化 tool_call 参数，彻底告别 JSON 字符串解析；
    模型不支持工具调用时自动回退到 _parse_llm_json 文本解析。

    Args:
        llm:          LLM 实例
        system_prompt: 系统提示
        user_prompt:   用户提示
        schema:        Pydantic 结构化输出模型
        tool_name:     工具名（schema 声明）
        tool_desc:     工具描述
        temperature:   推理温度

    Returns:
        dict: 结构化字段（与 schema 字段对齐）；失败返回 {}
    """
    tool = pydantic_model_to_tool(schema, tool_name, tool_desc)
    messages = [
        ChatMessage.system(system_prompt),
        ChatMessage.user(user_prompt),
    ]

    llm.update_config(temperature=temperature)
    try:
        resp = llm.chat_with_tools(
            messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool.name}},
            with_retry=False,
        )
        if resp.has_tool_calls:
            logger.info(
                f"  Function Calling 结构化输出成功: tool={resp.tool_calls[0].name} "
                f"args_keys={list(resp.tool_calls[0].arguments.keys())}"
            )
            return dict(resp.tool_calls[0].arguments)
        logger.warning("  模型未返回 tool_call，回退 JSON 文本解析")
        return _parse_llm_json(resp.text)
    except NotImplementedError:
        logger.warning("  当前 LLM 不支持 Function Calling，回退 JSON 文本解析")
    except Exception as e:
        logger.warning(f"  Function Calling 失败，回退 JSON 文本解析: {e}")

    # 回退：普通 chat + JSON 文本解析
    try:
        resp = llm.chat(messages, with_retry=False)
        return _parse_llm_json(resp.text)
    except Exception as e:
        logger.error(f"  结构化抽取最终失败: {e}")
        return {}


# ============================================================
# 1. parse_node - 文档解析 Agent
# ============================================================

def parse_node(state: AgentState) -> dict:
    """
    文档解析节点：清洗简历/JD 文本，执行切片并写入 state。

    处理流程：
    1. 使用 DocumentLoader 清洗原始文本（去水印/乱码/多余空行）
    2. 使用 DocumentSplitter 递归切片 + 标题感知
    3. 将切片结果写入 state.chunk_resume / state.chunk_jd

    Args:
        state: 全局 Agent 状态

    Returns:
        dict: 更新后的字段
    """
    logger.info("[parse_node] 开始文档解析...")
    state.current_node = "parse"

    loader = DocumentLoader()
    splitter = DocumentSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
    )

    chunks_resume: list[str] = []
    chunks_jd: list[str] = []

    # 处理简历
    if state.resume_raw.strip():
        try:
            cleaned = loader.clean_text(state.resume_raw)
            docs = splitter.split(cleaned, metadata={"source": "resume", "type": "resume"})
            chunks_resume = [d.page_content for d in docs]
            logger.info(f"  简历切片: {len(chunks_resume)} 块")
        except Exception as e:
            logger.error(f"  简历解析失败: {e}")
    else:
        logger.warning("  简历文本为空，跳过切片")

    # 处理 JD
    if state.jd_raw.strip():
        try:
            cleaned = loader.clean_text(state.jd_raw)
            docs = splitter.split(cleaned, metadata={"source": "jd", "type": "jd"})
            chunks_jd = [d.page_content for d in docs]
            logger.info(f"  JD 切片: {len(chunks_jd)} 块")
        except Exception as e:
            logger.error(f"  JD 解析失败: {e}")
    else:
        logger.warning("  JD 文本为空，跳过切片")

    logger.info("[parse_node] 完成")
    return {
        "chunk_resume": chunks_resume,
        "chunk_jd": chunks_jd,
        "current_node": "parse",
    }


# ============================================================
# 2. retrieve_node - RAG 检索 Agent
# ============================================================

def retrieve_node(state: AgentState) -> dict:
    """
    RAG 检索节点：根据 JD + 简历构建检索 query，拉取知识库参考资料。

    检索策略：
    1. 构建多维检索 query（岗位要求 + 候选人背景 + 面试题方向）
    2. 调用 HybridRetriever（BM25 + 向量 RRF 融合 + CrossEncoder 重排）
    3. 取 Top-K 高质量文档写入 rag_context

    Args:
        state: 全局 Agent 状态

    Returns:
        dict: 更新后的字段
    """
    logger.info("[retrieve_node] 开始 RAG 检索...")
    state.current_node = "retrieve"

    rag_context: list[Document] = []

    try:
        retriever = get_retriever()

        # 构建检索 query
        query = _build_retrieval_query(state.jd_raw, state.resume_raw)
        logger.info(f"  检索 query ({len(query)} chars): {query[:100]}...")

        # 执行混合检索
        rag_context = retriever.retrieve(query, top_k=settings.RAG_TOP_K)
        logger.info(f"  检索完成: {len(rag_context)} 篇参考文档")

        # 日志输出每篇文档摘要
        for i, doc in enumerate(rag_context):
            src = doc.metadata.get("source", "unknown")
            score = doc.metadata.get("rerank_score", doc.metadata.get("rrf_score", 0))
            logger.debug(f"  [{i+1}] score={score:.4f} src={src} | {doc.page_content[:50]}...")

    except Exception as e:
        logger.error(f"  RAG 检索失败: {e}")
        logger.warning("  将在无 RAG 增强的情况下继续后续流程")

    logger.info("[retrieve_node] 完成")
    return {
        "rag_context": rag_context,
        "current_node": "retrieve",
    }


# ============================================================
# 3. score_node - 匹配评估 Agent
# ============================================================

def score_node(state: AgentState) -> dict:
    """
    匹配评估节点：LLM 结构化输出四维度匹配分数。

    评估维度：
    - skill_match:      专业技能匹配（技术栈、工具链、编程语言）
    - experience_match: 项目经验匹配（年限、项目复杂度、行业背景）
    - education_match:  学历背景匹配（学历层次、专业方向）
    - overall_score:    综合加权得分

    LLM 被强制要求输出 JSON 格式，Pydantic 做二次校验。
    同时提取 skill_gap（候选人缺失的技能清单）。

    Args:
        state: 全局 Agent 状态

    Returns:
        dict: 更新的 match_score / skill_gap
    """
    logger.info("[score_node] 开始四维度匹配评估...")
    state.current_node = "score"

    if not state.resume_raw.strip() or not state.jd_raw.strip():
        logger.warning("  简历或 JD 为空，跳过评分")
        return {"current_node": "score"}

    # 构建 RAG 参考资料文本
    rag_text = ""
    if state.rag_context:
        rag_parts = []
        for doc in state.rag_context[:3]:
            rag_parts.append(doc.page_content)
        rag_text = "\n---\n".join(rag_parts)

    # 构建评分 Prompt
    system_prompt = """你是一名资深 HR 技术面试官和职业规划顾问。
你的任务是对候选人与岗位进行四维度匹配评估，并严格输出 JSON 格式结果。

评估标准：
1. skill_match (0-100)：专业技能匹配度。考察技术栈、工具链、编程语言、框架等与 JD 的重叠程度。
2. experience_match (0-100)：项目经验匹配度。考察工作年限、项目复杂度、行业背景、管理经验的契合度。
3. education_match (0-100)：学历背景匹配度。考察学历层次、专业方向与 JD 要求的一致性。
4. overall_score (0-100)：综合加权得分。计算方法：skill_match * 0.4 + experience_match * 0.35 + education_match * 0.25。

额外输出：
- skill_gap: 候选人缺失的技能清单（JD 要求但候选人简历中不具备或较弱的技能）
- analysis: 200字以内的简短分析

输出格式（严格 JSON，不要包含任何其他文字）:
```json
{
    "skill_match": 85.0,
    "experience_match": 70.0,
    "education_match": 90.0,
    "overall_score": 82.5,
    "skill_gap": ["Kubernetes", "微服务架构"],
    "analysis": "候选人在Python和Django方面经验丰富，但缺少..."
}
```"""

    user_prompt = f"""请评估以下候选人与岗位的匹配度：

【岗位 JD】
{state.jd_raw[:2000]}

【候选人简历】
{state.resume_raw[:2000]}

【参考资料】
{rag_text if rag_text else "（无额外参考资料）"}

请严格按照 JSON 格式输出评估结果。"""

    try:
        llm = get_llm()

        # v2.0: Function Calling 强制结构化输出，失败自动回退 JSON 解析
        result = _structured_extract(
            llm,
            system_prompt,
            user_prompt,
            ScoreResultSchema,
            "evaluate_match_score",
            "对候选人与岗位进行四维度匹配评估，输出结构化评分结果",
            temperature=0.1,
        )

        # 校验并构造 MatchScoreDetail
        match_score = MatchScoreDetail(
            skill_match=float(result.get("skill_match", 0)),
            experience_match=float(result.get("experience_match", 0)),
            education_match=float(result.get("education_match", 0)),
            overall_score=float(result.get("overall_score", 0)),
        )
        skill_gap = result.get("skill_gap", [])

        logger.info(
            f"  评分完成: skill={match_score.skill_match} "
            f"exp={match_score.experience_match} "
            f"edu={match_score.education_match} "
            f"overall={match_score.overall_score}"
        )
        logger.info(f"  技能差距: {skill_gap}")

    except Exception as e:
        logger.error(f"  评分失败: {e}")
        # 返回默认值，不阻断流程
        match_score = MatchScoreDetail()
        skill_gap = []

    logger.info("[score_node] 完成")
    return {
        "match_score": match_score,
        "skill_gap": skill_gap,
        "current_node": "score",
    }


# ============================================================
# 4. optimize_node - 简历优化 Agent
# ============================================================

OPTIMIZE_SYSTEM_PROMPT = """你是一位顶级职业简历顾问，精通 ATS（Applicant Tracking System）筛选规则。
请根据以下要求优化简历：

优化原则：
1. **STAR 法则重构**：每条经历按 Situation-Task-Action-Result 结构重写
2. **关键词植入**：自然融入 JD 中的核心技术关键词，但不要强行堆砌
3. **量化成果**：用具体数字描述贡献（如"性能提升 30%""减少 50% bug"）
4. **匹配优先**：突出与目标岗位最相关的技能和经验
5. **信息诚实**：不虚构经历，但可优化措辞和呈现方式
6. **格式专业**：保持标准简历格式，分节清晰

{{MODE_INSTRUCTION}}

请直接输出优化后的完整简历文本，不要包含任何额外说明。"""

# 不同优化模式的差异化指令
MODE_INSTRUCTIONS = {
    "keywords": (
        "【关键词匹配模式】\n"
        "- 重点分析 JD 中的核心技术栈、工具链和行业术语\n"
        "- 确保简历中自然覆盖所有关键技能关键词\n"
        "- 每项技能描述应与 JD 中对口的技术要求一一呼应\n"
        "- 将匹配度低的技能描述替换为 JD 中高频出现的相关表述"
    ),
    "quantify": (
        "【量化成果模式】\n"
        "- 每段项目/工作经历必须包含至少一个可量化的成果指标\n"
        "- 使用具体数字：性能提升 X%、用户量增长 X 倍、成本降低 X%\n"
        "- 避免模糊表述如「显著提升」「大幅优化」，替换为精确数值\n"
        "- 优先使用 STAR 法则中 Result 部分的量化描述"
    ),
    "concise": (
        "【精简冗余模式】\n"
        "- 删除与目标岗位无关的经历和技能描述\n"
        "- 合并重复或相似的经历条目，保留最有代表性的\n"
        "- 每条描述控制在 1-2 句话内，去除修饰性语言\n"
        "- 确保整体简历长度紧凑，重点突出，一页内完成"
    ),
}


def optimize_node(state: AgentState, feedback: str = "") -> dict:
    """
    简历优化节点：结合 RAG 资料与匹配差距，STAR 法则重写简历。

    支持多轮迭代：
    - 首轮：基于 resume_raw 优化
    - 多轮：基于上一版 optimized_resume + 新 feedback 再次优化

    Args:
        state: 全局 Agent 状态
        feedback: 用户对上一版优化结果的反馈（多轮迭代用）

    Returns:
        dict: 更新的 optimized_resume
    """
    # 优先使用 state 中的 feedback，其次是参数传入的
    effective_feedback = state.optimize_feedback.strip() or feedback.strip()
    is_multi_round = bool(effective_feedback)

    logger.info("[optimize_node] 开始简历优化..."
                + (" (多轮迭代)" if is_multi_round else ""))

    state.current_node = "optimize"

    if not state.resume_raw.strip():
        logger.warning("  简历为空，无法优化")
        return {"current_node": "optimize"}

    # ---- 决定优化基文本：多轮时基于上一版结果迭代 ----
    if is_multi_round and state.optimized_resume.strip():
        base_resume = state.optimized_resume
        logger.info(f"  多轮迭代：基于上一版优化结果 ({len(base_resume)} 字符) 继续优化")
    else:
        base_resume = state.resume_raw

    # ---- 收集 RAG 范文参考 ----
    rag_examples = ""
    if state.rag_context:
        relevant = [d for d in state.rag_context
                    if d.metadata.get("type", "") in ("resume", "jd", "test")]
        for doc in relevant[:2]:
            rag_examples += f"\n【参考范文片段】\n{doc.page_content[:500]}\n"

    # ---- 根据模式生成差异化系统提示 ----
    mode = state.optimize_mode or "keywords"
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["keywords"])
    system_prompt = OPTIMIZE_SYSTEM_PROMPT.replace("{{MODE_INSTRUCTION}}", mode_instruction)

    # ---- 构建用户提示 ----
    gap_text = "、".join(state.skill_gap) if state.skill_gap else "无明显差距"

    user_prompt = f"""请优化以下简历，使其更匹配目标岗位：

【优化模式】{mode}

【目标岗位要求】
{state.jd_raw[:1500]}

【待优化的简历】
{base_resume[:2000]}

【当前匹配度】
- 综合得分: {state.match_score.overall_score}/100
- 技能匹配: {state.match_score.skill_match}/100
- 经验匹配: {state.match_score.experience_match}/100
- 学历匹配: {state.match_score.education_match}/100

【需要重点弥补的方向】
{', '.join(state.skill_gap) if state.skill_gap else '各项匹配良好'}

{rag_examples}

{f"【用户附加反馈 - 请务必基于以下要求调整】\n{effective_feedback}" if effective_feedback else ""}

请输出优化后的完整简历："""

    try:
        llm = get_llm()
        llm.update_config(temperature=0.3, max_tokens=4096)

        resp = llm.chat([
            ChatMessage.system(system_prompt),
            ChatMessage.user(user_prompt),
        ])

        optimized = resp.text.strip()
        logger.info(f"  优化完成: {len(optimized)} 字符"
                    + (f" (模式={mode})" if not is_multi_round else f" (多轮迭代, 模式={mode})"))

    except Exception as e:
        logger.error(f"  优化失败: {e}")
        optimized = base_resume  # 回退到基文本

    logger.info("[optimize_node] 完成")
    return {
        "optimized_resume": optimized,
        "current_node": "optimize",
    }


# ============================================================
# 5. interview_node - AI 面试官 Agent
# ============================================================

INTERVIEWER_SYSTEM_PROMPT = """你是一位专业、严谨但友善的技术面试官。

面试规则：
1. 根据岗位要求和候选人背景，提出有针对性的面试问题
2. 问题应覆盖：技术能力（40%）、项目经验（30%）、行为素质（20%）、综合（10%）
3. 每轮只提一个问题，问题要具体、有深度
4. 对候选人的回答进行评分（0-100）并给出简短点评
5. 根据回答质量动态调整后续问题难度

你需要按要求输出 JSON 格式。"""


def interview_generate_question(state: AgentState) -> dict:
    """
    面试问题生成节点：基于 JD + 简历 + 已有问答记录，生成下一道面试题。

    不直接输出问题文本，而是写入 state.interview_history（新增一轮空回答），
    由 Gradio 前端展示问题并收集用户回答后，再调用 interview_evaluate_answer。

    Args:
        state: 全局 Agent 状态

    Returns:
        dict: 更新的 interview_history
    """
    logger.info("[interview_node] 生成面试问题...")
    state.current_node = "interview"

    round_num = len(state.interview_history) + 1

    if round_num > 10:
        logger.info("  已达最大面试轮次(10)，终止提问")
        return {"current_node": "interview"}

    # 构建历史上下文
    history_text = ""
    for qa in state.interview_history:
        history_text += f"\n第{qa.round}轮 - 问题: {qa.question}\n回答: {qa.answer[:200]}\n"

    # 动态调整问题类别
    if round_num <= 3:
        category_hint = "技术基础"
    elif round_num <= 6:
        category_hint = "项目经验与架构设计"
    elif round_num <= 8:
        category_hint = "行为素质与团队协作"
    else:
        category_hint = "综合素质与职业规划"

    user_prompt = f"""请生成第 {round_num} 轮面试问题。

【目标岗位】
{state.jd_raw[:1000]}

【候选人背景】
{state.resume_raw[:1000]}

【已提问记录】
{history_text if history_text else "（尚无提问记录）"}

【本轮侧重】
{category_hint}

请以 JSON 格式输出：
```json
{{
    "question": "你的面试问题",
    "category": "{category_hint}",
    "expected_points": ["期望回答要点1", "要点2"],
    "difficulty": "medium"
}}
```"""

    try:
        llm = get_llm()

        # v2.0: Function Calling 强制结构化输出，失败自动回退 JSON 解析
        result = _structured_extract(
            llm,
            INTERVIEWER_SYSTEM_PROMPT,
            user_prompt,
            InterviewQuestionSchema,
            "generate_interview_question",
            "根据岗位与候选人背景生成下一道面试题",
            temperature=0.7,
        )
        question = result.get("question", "请简单介绍一下你自己")
        category = result.get("category", "general")

    except Exception as e:
        logger.error(f"  问题生成失败: {e}")
        question = "请简单介绍一下你的技术背景和项目经验。"
        category = "general"

    # 追加新轮次（答案待填）
    new_qa = InterviewQA(
        round=round_num,
        question=question,
        answer="",          # 等待用户输入
        score=0.0,
        feedback="",
        category=category,
    )
    updated_history = list(state.interview_history) + [new_qa]

    logger.info(f"  第{round_num}轮问题已生成 [{category}]: {question[:60]}...")
    logger.info("[interview_node] 问题生成完成")

    return {
        "interview_history": updated_history,
        "current_node": "interview",
    }


def interview_evaluate_answer(state: AgentState, round_index: int, answer: str) -> dict:
    """
    面试评估节点：对候选人某一轮的回答进行评分和点评。

    Args:
        state:   全局 Agent 状态
        round_index: 要评估的轮次索引（0-based）
        answer:  候选人的回答文本

    Returns:
        dict: 更新的 interview_history（该轮 score/feedback 已填充）
    """
    logger.info(f"[interview_node] 评估第{round_index + 1}轮回答...")

    if round_index >= len(state.interview_history):
        logger.error(f"  轮次索引越界: {round_index}")
        return {"current_node": "interview"}

    qa = state.interview_history[round_index]

    eval_prompt = f"""请评估候选人对以下面试问题的回答：

【面试问题】({qa.category})
{qa.question}

【候选人回答】
{answer}

【岗位要求参考】
{state.jd_raw[:500]}

请以 JSON 格式输出评估结果：
```json
{{
    "score": 85.0,
    "feedback": "回答准确、条理清晰...（50字以内）",
    "strengths": ["优点1", "优点2"],
    "weaknesses": ["不足1"]
}}
```"""

    try:
        llm = get_llm()

        # v2.0: Function Calling 强制结构化输出，失败自动回退 JSON 解析
        result = _structured_extract(
            llm,
            "你是一位严谨的面试官，请客观评分。",
            eval_prompt,
            InterviewEvalSchema,
            "evaluate_interview_answer",
            "评估候选人面试回答并输出评分与点评",
            temperature=0.2,
        )
        score = float(result.get("score", 70))
        feedback = result.get("feedback", "回答已记录")

    except Exception as e:
        logger.error(f"  评估失败: {e}")
        score = 70.0
        feedback = "评估异常，使用默认评分"

    # 更新该轮记录
    updated_qa = InterviewQA(
        round=qa.round,
        question=qa.question,
        answer=answer,
        score=score,
        feedback=feedback,
        category=qa.category,
    )
    updated_history = list(state.interview_history)
    updated_history[round_index] = updated_qa

    logger.info(f"  评分: {score}/100 | {feedback}")

    return {
        "interview_history": updated_history,
        "current_node": "interview",
    }


# ============================================================
# 6. summary_node - 面试复盘 Agent
# ============================================================

def summary_node(state: AgentState) -> dict:
    """
    面试复盘节点：汇总全部问答记录，生成综合评价报告。

    报告内容：
    - 总体评分与等级
    - 各维度表现分析（技术/项目/沟通/综合素质）
    - 突出优势与明显不足
    - 针对性改进建议
    - 推荐学习资料（关联 RAG 知识库）
    - 下一步行动建议

    Args:
        state: 全局 Agent 状态

    Returns:
        dict: 更新的 interview_report
    """
    logger.info("[summary_node] 开始生成面试复盘报告...")
    state.current_node = "summary"

    if not state.interview_history:
        logger.warning("  无面试记录，无法生成报告")
        return {
            "interview_report": "暂无面试记录，无法生成复盘报告。",
            "current_node": "summary",
        }

    # 构建完整面试记录文本
    history_text = ""
    total_score = 0.0
    for qa in state.interview_history:
        history_text += (
            f"### 第{qa.round}轮 [{qa.category}]\n"
            f"**问题**: {qa.question}\n"
            f"**回答**: {qa.answer[:300]}\n"
            f"**得分**: {qa.score}/100\n"
            f"**点评**: {qa.feedback}\n\n"
        )
        total_score += qa.score

    avg_score = total_score / len(state.interview_history) if state.interview_history else 0

    # 收集 RAG 推荐资料
    rag_recommend = ""
    if state.rag_context:
        rag_docs = [d for d in state.rag_context if d.metadata.get("type") != "test"]
        if rag_docs:
            rag_recommend = "【知识库参考资料】\n"
            for doc in rag_docs[:3]:
                src = doc.metadata.get("source", "?")
                rag_recommend += f"- {src}: {doc.page_content[:100]}...\n"

    report_prompt = f"""请基于以下面试记录生成一份专业的面试复盘报告。

【应聘岗位】
{state.jd_raw[:500]}

【面试记录】
{history_text}

【总体统计】
- 面试轮次: {len(state.interview_history)}
- 平均得分: {avg_score:.1f}/100

{rag_recommend}

请按以下结构输出报告（Markdown 格式）：

## 面试复盘报告

### 一、总体评价
（200字以内综合评价，含评分等级：优秀≥85 / 良好≥70 / 一般≥60 / 待提升<60）

### 二、分维度分析
| 维度 | 得分 | 评价 |
|------|------|------|
| 技术能力 | xx | ... |
| 项目经验 | xx | ... |
| 沟通表达 | xx | ... |
| 综合素质 | xx | ... |

### 三、突出优势
- 优势1
- 优势2

### 四、待改进项
- 不足1
- 不足2

### 五、改进建议
1. 具体建议1
2. 具体建议2

### 六、推荐学习资源
（基于知识库和技能差距的推荐）"""

    try:
        llm = get_llm()
        llm.update_config(temperature=0.3, max_tokens=2048)

        resp = llm.chat([
            ChatMessage.system("你是一位资深职业规划顾问和面试教练。"),
            ChatMessage.user(report_prompt),
        ])
        report = resp.text.strip()
        logger.info(f"  复盘报告生成完成: {len(report)} 字符")

    except Exception as e:
        logger.error(f"  报告生成失败: {e}")
        report = f"## 面试复盘报告\n\n面试未能正常完成。平均得分: {avg_score:.1f}/100\n\n请重新进行面试。"

    logger.info("[summary_node] 完成")
    return {
        "interview_report": report,
        "current_node": "summary",
    }
